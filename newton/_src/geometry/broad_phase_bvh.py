from __future__ import annotations

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .broad_phase_common import (
    check_aabb_overlap,
    is_pair_excluded,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _bvh_gather_aabbs_batched_kernel(
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    # Outputs
    out_lower: wp.array(dtype=wp.vec3, ndim=1),
    out_upper: wp.array(dtype=wp.vec3, ndim=1),
):
    world_id, local_id = wp.tid()

    slice_start = 0
    if world_id > 0:
        slice_start = world_slice_ends[world_id - 1]
    n = world_slice_ends[world_id] - slice_start

    if local_id >= n:
        return

    shape_id = world_index_map[slice_start + local_id]

    lower = shape_lower[shape_id]
    upper = shape_upper[shape_id]

    gap = 0.0
    if shape_gap.shape[0] > 0:
        gap = shape_gap[shape_id]

    idx = world_id * max_shapes_per_world + local_id
    out_lower[idx] = wp.vec3(lower[0] - gap, lower[1] - gap, lower[2] - gap)
    out_upper[idx] = wp.vec3(upper[0] + gap, upper[1] + gap, upper[2] + gap)


@wp.kernel
def _bvh_query_batched_kernel(
    bvh_ids: wp.array(dtype=wp.uint64, ndim=1),
    # Original shape data
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    # World mapping
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    num_regular_worlds: int,
    # Per-world expanded AABBs stored in flat layout
    flat_lower: wp.array(dtype=wp.vec3, ndim=1),
    flat_upper: wp.array(dtype=wp.vec3, ndim=1),
    # Filter pairs
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
    # Output
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    world_id, local_id = wp.tid()

    slice_start = 0
    if world_id > 0:
        slice_start = world_slice_ends[world_id - 1]
    num_shapes_in_world = world_slice_ends[world_id] - slice_start

    if local_id >= num_shapes_in_world:
        return

    shape1 = world_index_map[slice_start + local_id]
    world1 = shape_world[shape1]
    col_group1 = collision_group[shape1]

    flat_idx = world_id * max_shapes_per_world + local_id
    query_lower = flat_lower[flat_idx]
    query_upper = flat_upper[flat_idx]

    bvh_id = bvh_ids[world_id]
    query = wp.bvh_query_aabb(bvh_id, query_lower, query_upper)
    other_local_id = int(-1)

    while wp.bvh_query_next(query, other_local_id):
        if other_local_id == local_id:
            continue
        if other_local_id >= num_shapes_in_world:
            continue

        shape2 = world_index_map[slice_start + other_local_id]

        # Canonical ordering and dedup
        if shape1 >= shape2:
            continue

        world2 = shape_world[shape2]
        col_group2 = collision_group[shape2]

        # Skip -1 vs -1 pairs unless in the dedicated shared segment
        if world1 == -1 and world2 == -1:
            if world_id < num_regular_worlds:
                continue

        if not test_world_and_group_pair(world1, world2, col_group1, col_group2):
            continue

        # Verify AABB overlap on original AABBs with gaps
        gap1 = 0.0
        gap2 = 0.0
        if shape_gap.shape[0] > 0:
            gap1 = shape_gap[shape1]
            gap2 = shape_gap[shape2]

        if not check_aabb_overlap(
            shape_lower[shape1], shape_upper[shape1], gap1,
            shape_lower[shape2], shape_upper[shape2], gap2,
        ):
            continue

        if num_filter_pairs > 0 and is_pair_excluded(wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs):
            continue

        write_pair(wp.vec2i(shape1, shape2), candidate_pair, candidate_pair_count, max_candidate_pair)


class BroadPhaseBVH:

    def __init__(
        self,
        shape_world: wp.array(dtype=wp.int32, ndim=1) | np.ndarray,
        shape_flags: wp.array(dtype=wp.int32, ndim=1) | np.ndarray | None = None,
        device: Devicelike | None = None,
    ) -> None:

        # Convert to numpy
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        shape_flags_np = None
        if shape_flags is not None:
            if isinstance(shape_flags, wp.array):
                shape_flags_np = shape_flags.numpy()
            else:
                shape_flags_np = shape_flags

        # Precompute world map
        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        self.world_count = len(slice_ends_np)
        self.num_regular_worlds = max(0, self.world_count - 1)
        self.device = device

        # Store world index map and slice ends as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)

        # Compute per-world slice start/count
        self.world_slice_starts: list[int] = []
        self.world_shape_counts: list[int] = []
        start = 0
        for end in slice_ends_np:
            self.world_slice_starts.append(int(start))
            self.world_shape_counts.append(int(end - start))
            start = end

        # Max shapes per world determines the flat buffer stride
        self.max_shapes_per_world = max(self.world_shape_counts) if self.world_shape_counts else 0

        # Flat contiguous buffers for batched gather/query (all worlds packed
        # with stride = max_shapes_per_world, matching the SAP layout)
        total_flat = self.world_count * self.max_shapes_per_world
        self.flat_lower = wp.zeros(max(total_flat, 1), dtype=wp.vec3, device=device)
        self.flat_upper = wp.zeros(max(total_flat, 1), dtype=wp.vec3, device=device)

        # Per-world *views* into the flat buffers.  Each BVH is built from
        # its world's view so that refit() automatically sees data written
        # into the flat buffer by the batched gather kernel.
        self._per_world_lower_views: list[wp.array | None] = []
        self._per_world_upper_views: list[wp.array | None] = []
        for w, n in enumerate(self.world_shape_counts):
            if n == 0:
                self._per_world_lower_views.append(None)
                self._per_world_upper_views.append(None)
            else:
                s = w * self.max_shapes_per_world
                self._per_world_lower_views.append(self.flat_lower[s : s + n])
                self._per_world_upper_views.append(self.flat_upper[s : s + n])

        # BVH instances (built on first launch) and their IDs for batched query
        self.bvh_list: list[wp.Bvh | None] = [None] * self.world_count
        self.bvh_ids = wp.zeros(max(self.world_count, 1), dtype=wp.uint64, device=device)
        self._bvhs_built = False

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),
        shape_contact_margin: wp.array(dtype=float, ndim=1) | None,
        shape_collision_group: wp.array(dtype=int, ndim=1),
        shape_shape_world: wp.array(dtype=int, ndim=1),
        shape_count: int,
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        device: Devicelike | None = None,
        filter_pairs: wp.array(dtype=wp.vec2i, ndim=1) | None = None,
        num_filter_pairs: int | None = None,
    ) -> None:

        max_candidate_pair = candidate_pair.shape[0]
        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        if self.max_shapes_per_world == 0:
            return

        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # --- 1. Batched gather: single kernel launch for all worlds ----------
        wp.launch(
            kernel=_bvh_gather_aabbs_batched_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                shape_lower,
                shape_upper,
                shape_contact_margin,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
            ],
            outputs=[
                self.flat_lower,
                self.flat_upper,
            ],
            device=device,
        )

        # --- 2. Build or refit per-world BVH trees (unavoidable loop) --------
        for w in range(self.world_count):
            if self.world_shape_counts[w] == 0:
                continue
            if not self._bvhs_built:
                self.bvh_list[w] = wp.Bvh(
                    self._per_world_lower_views[w],
                    self._per_world_upper_views[w],
                )
            else:
                self.bvh_list[w].refit()

        # Collect BVH IDs after the first build so the batched query kernel
        # can look up the correct tree per world.
        if not self._bvhs_built:
            bvh_ids_np = np.zeros(self.world_count, dtype=np.uint64)
            for w in range(self.world_count):  
                if self.bvh_list[w] is not None:
                    bvh_ids_np[w] = int(self.bvh_list[w].id)
            self.bvh_ids = wp.array(bvh_ids_np, dtype=wp.uint64, device=device)

        # --- 3. Batched query: single kernel launch for all worlds -----------
        wp.launch(
            kernel=_bvh_query_batched_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                self.bvh_ids,
                shape_lower,
                shape_upper,
                shape_contact_margin,
                shape_collision_group,
                shape_shape_world,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.num_regular_worlds,
                self.flat_lower,
                self.flat_upper,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
            device=device,
        )

        self._bvhs_built = True
