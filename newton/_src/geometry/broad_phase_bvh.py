# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BVH (Bounding Volume Hierarchy) broad phase collision detection.

Provides O(N log N) construction and O(log N) per-query broad phase by building
independent BVH trees for each simulation world.  Shared shapes (world -1) are
replicated into every world's tree so that cross-world queries are unnecessary.

See Also:
    :class:`BroadPhaseAllPairs` in ``broad_phase_nxn.py`` for simpler O(N²) approach.
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for sweep-and-prune approach.
"""

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
def _bvh_gather_aabbs_kernel(
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    slice_start: int,
    # Outputs
    out_lower: wp.array(dtype=wp.vec3, ndim=1),
    out_upper: wp.array(dtype=wp.vec3, ndim=1),
):
    """Gather shape AABBs for one world into its own arrays.

    Launched with dim=num_shapes_in_world per world.
    """
    local_id = wp.tid()
    shape_id = world_index_map[slice_start + local_id]

    lower = shape_lower[shape_id]
    upper = shape_upper[shape_id]

    gap = 0.0
    if shape_gap.shape[0] > 0:
        gap = shape_gap[shape_id]

    out_lower[local_id] = wp.vec3(lower[0] - gap, lower[1] - gap, lower[2] - gap)
    out_upper[local_id] = wp.vec3(upper[0] + gap, upper[1] + gap, upper[2] + gap)


@wp.kernel
def _bvh_query_kernel(
    bvh_id: wp.uint64,
    # Original shape data
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    # World mapping
    world_index_map: wp.array(dtype=int, ndim=1),
    slice_start: int,
    num_shapes_in_world: int,
    is_dedicated_shared_segment: int,  # 1 if this is the dedicated -1 segment, 0 otherwise
    # Per-world expanded AABBs (for query bounds)
    world_lower: wp.array(dtype=wp.vec3, ndim=1),
    world_upper: wp.array(dtype=wp.vec3, ndim=1),
    # Filter pairs
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
    # Output
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    """Query one world's BVH to find overlapping shape pairs.

    Launched with dim=num_shapes_in_world per world.
    Each thread queries one shape against the BVH and emits
    overlapping pairs with canonical ordering (shape1 < shape2).
    """
    local_id = wp.tid()

    shape1 = world_index_map[slice_start + local_id]
    world1 = shape_world[shape1]
    col_group1 = collision_group[shape1]

    query_lower = world_lower[local_id]
    query_upper = world_upper[local_id]

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
        if world1 == -1 and world2 == -1 and is_dedicated_shared_segment == 0:
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
    """BVH-based broad phase collision detection with per-world BVH trees.

    Each simulation world gets its own independent BVH tree.  Shared shapes
    (world ID -1) are replicated into every world's tree.  The BVH trees are
    built on the first frame and refitted on subsequent frames.
    """

    def __init__(
        self,
        shape_world: wp.array(dtype=wp.int32, ndim=1) | np.ndarray,
        shape_flags: wp.array(dtype=wp.int32, ndim=1) | np.ndarray | None = None,
        device: Devicelike | None = None,
    ) -> None:
        """Initialize the BVH broad phase with per-world BVH trees.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
            shape_flags: Optional array of shape flags. If provided,
                only shapes with the COLLIDE_SHAPES flag will be included.
            device: Device to store arrays on.
        """
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

        # Store world index map
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)

        # Compute per-world slice start/count
        self.world_slice_starts: list[int] = []
        self.world_shape_counts: list[int] = []
        start = 0
        for end in slice_ends_np:
            self.world_slice_starts.append(int(start))
            self.world_shape_counts.append(int(end - start))
            start = end

        # Allocate per-world arrays (exact size, no padding)
        self.per_world_lower: list[wp.array] = []
        self.per_world_upper: list[wp.array] = []
        for n in self.world_shape_counts:
            self.per_world_lower.append(wp.zeros(max(n, 1), dtype=wp.vec3, device=device))
            self.per_world_upper.append(wp.zeros(max(n, 1), dtype=wp.vec3, device=device))

        # BVH instances (built on first launch)
        self.bvh_list: list[wp.Bvh | None] = [None] * self.world_count
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
        """Launch per-world BVH broad phase collision detection.

        Args:
            shape_lower: Lower bounds of each shape's AABB.
            shape_upper: Upper bounds of each shape's AABB.
            shape_contact_margin: Optional per-shape contact margins.
            shape_collision_group: Collision group ID per shape.
            shape_shape_world: World index per shape.
            shape_count: Number of active shapes (unused in per-world approach).
            candidate_pair: Output array for overlapping shape pairs.
            num_candidate_pair: Output counter for pairs found.
            device: Device to launch on.
            filter_pairs: Optional sorted excluded pairs.
            num_filter_pairs: Number of valid entries in filter_pairs.
        """
        max_candidate_pair = candidate_pair.shape[0]
        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # For each world: gather AABBs, build/refit BVH, query
        for w in range(self.world_count):
            n = self.world_shape_counts[w]
            if n == 0:
                continue

            # Gather AABBs for this world
            wp.launch(
                kernel=_bvh_gather_aabbs_kernel,
                dim=n,
                inputs=[
                    shape_lower,
                    shape_upper,
                    shape_contact_margin,
                    self.world_index_map,
                    self.world_slice_starts[w],
                ],
                outputs=[
                    self.per_world_lower[w],
                    self.per_world_upper[w],
                ],
                device=device,
            )

            # Build or refit BVH
            if not self._bvhs_built:
                self.bvh_list[w] = wp.Bvh(self.per_world_lower[w], self.per_world_upper[w])
            else:
                self.bvh_list[w].refit()

            # Query this world's BVH
            is_dedicated = 1 if w >= self.num_regular_worlds else 0
            wp.launch(
                kernel=_bvh_query_kernel,
                dim=n,
                inputs=[
                    self.bvh_list[w].id,
                    shape_lower,
                    shape_upper,
                    shape_contact_margin,
                    shape_collision_group,
                    shape_shape_world,
                    self.world_index_map,
                    self.world_slice_starts[w],
                    n,
                    is_dedicated,
                    self.per_world_lower[w],
                    self.per_world_upper[w],
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
