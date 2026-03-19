# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Spatial hash grid broad phase collision detection.

Uses a uniform spatial hash grid for expected O(N) broad phase. Each shape is
assigned to the grid cell containing its AABB center. Candidate pairs are found
by querying the 3x3x3 neighborhood of each shape's cell.

Best for scenes with roughly uniformly sized objects. For scenes with large
size disparity (e.g., ground planes alongside small objects), prefer
:class:`BroadPhaseSAP` or :class:`BroadPhaseBVH`.

See Also:
    :class:`BroadPhaseAllPairs` in ``broad_phase_nxn.py`` for O(N²) approach.
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for O(N log N) approach.
    :class:`BroadPhaseBVH` in ``broad_phase_bvh.py`` for tree-based approach.
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
from .hashtable import HASHTABLE_EMPTY_KEY, HashTable, hashtable_find, hashtable_find_or_insert

wp.set_module_options({"enable_backward": False})

# Pre-defined uint64 constants for cell key encoding.
# Defined at module scope to avoid Warp's intermediate signed-cast issue
# when converting literals inside kernels (see hashtable.py for details).
_UINT64_16 = wp.constant(wp.uint64(16))
_UINT64_32 = wp.constant(wp.uint64(32))
_UINT64_48 = wp.constant(wp.uint64(48))
_UINT64_MASK_16 = wp.constant(wp.uint64(0xFFFF))
_CELL_COORD_OFFSET = wp.constant(wp.uint64(32768))


@wp.func
def _encode_cell_key(world_id: int, cx: int, cy: int, cz: int) -> wp.uint64:
    """Encode world ID and cell coordinates into a unique uint64 key.

    Packs four values into non-overlapping 16-bit fields.  Cell coordinates
    are offset by 32768 to handle negative values via unsigned wrapping.

    Valid ranges: world_id [0, 65535], cell coords [-32768, 32767].
    """
    w = wp.uint64(world_id) & _UINT64_MASK_16
    x = (wp.uint64(cx) + _CELL_COORD_OFFSET) & _UINT64_MASK_16
    y = (wp.uint64(cy) + _CELL_COORD_OFFSET) & _UINT64_MASK_16
    z = (wp.uint64(cz) + _CELL_COORD_OFFSET) & _UINT64_MASK_16
    key = (w << _UINT64_48) | (x << _UINT64_32) | (y << _UINT64_16) | z
    if key == HASHTABLE_EMPTY_KEY:
        key = key ^ wp.uint64(1)
    return key


@wp.kernel
def _hash_insert_kernel(
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    inv_cell_size: float,
    # Hash table
    ht_keys: wp.array(dtype=wp.uint64),
    ht_active_slots: wp.array(dtype=wp.int32),
    # Cell data
    cell_shape_counts: wp.array(dtype=int, ndim=1),
    cell_shape_indices: wp.array(dtype=int, ndim=1),
    max_shapes_per_cell: int,
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

    center_x = 0.5 * (lower[0] + upper[0])
    center_y = 0.5 * (lower[1] + upper[1])
    center_z = 0.5 * (lower[2] + upper[2])

    cx = int(wp.floor(center_x * inv_cell_size))
    cy = int(wp.floor(center_y * inv_cell_size))
    cz = int(wp.floor(center_z * inv_cell_size))

    cell_key = _encode_cell_key(world_id, cx, cy, cz)
    entry_idx = hashtable_find_or_insert(cell_key, ht_keys, ht_active_slots)

    if entry_idx < 0:
        return

    slot = wp.atomic_add(cell_shape_counts, entry_idx, 1)
    if slot < max_shapes_per_cell:
        cell_shape_indices[entry_idx * max_shapes_per_cell + slot] = local_id


@wp.kernel
def _hash_query_kernel(
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    num_regular_worlds: int,
    inv_cell_size: float,
    # Hash table (read-only)
    ht_keys: wp.array(dtype=wp.uint64),
    # Cell data (read-only)
    cell_shape_counts: wp.array(dtype=int, ndim=1),
    cell_shape_indices: wp.array(dtype=int, ndim=1),
    max_shapes_per_cell: int,
    # Filter
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
    n = world_slice_ends[world_id] - slice_start

    if local_id >= n:
        return

    shape1 = world_index_map[slice_start + local_id]
    world1 = shape_world[shape1]
    col_group1 = collision_group[shape1]

    lower1 = shape_lower[shape1]
    upper1 = shape_upper[shape1]

    gap1 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]

    center_x = 0.5 * (lower1[0] + upper1[0])
    center_y = 0.5 * (lower1[1] + upper1[1])
    center_z = 0.5 * (lower1[2] + upper1[2])

    cx = int(wp.floor(center_x * inv_cell_size))
    cy = int(wp.floor(center_y * inv_cell_size))
    cz = int(wp.floor(center_z * inv_cell_size))

    # 3x3x3 neighborhood: 27 cells indexed as (dx, dy, dz) each in {-1, 0, 1}
    for n_idx in range(27):
        dz = (n_idx % 3) - 1
        dy = ((n_idx / 3) % 3) - 1
        dx = (n_idx / 9) - 1

        cell_key = _encode_cell_key(world_id, cx + dx, cy + dy, cz + dz)
        entry_idx = hashtable_find(cell_key, ht_keys)
        if entry_idx < 0:
            continue

        count = cell_shape_counts[entry_idx]
        if count > max_shapes_per_cell:
            count = max_shapes_per_cell

        for k in range(count):
            other_local_id = cell_shape_indices[entry_idx * max_shapes_per_cell + k]
            if other_local_id == local_id:
                continue

            shape2 = world_index_map[slice_start + other_local_id]

            if shape1 >= shape2:
                continue

            world2 = shape_world[shape2]
            col_group2 = collision_group[shape2]

            # Skip -1 vs -1 unless in dedicated shared segment
            if world1 == -1 and world2 == -1:
                if world_id < num_regular_worlds:
                    continue

            if not test_world_and_group_pair(world1, world2, col_group1, col_group2):
                continue

            gap2 = 0.0
            if shape_gap.shape[0] > 0:
                gap2 = shape_gap[shape2]

            if not check_aabb_overlap(
                lower1,
                upper1,
                gap1,
                shape_lower[shape2],
                shape_upper[shape2],
                gap2,
            ):
                continue

            if num_filter_pairs > 0 and is_pair_excluded(
                wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs
            ):
                continue

            write_pair(
                wp.vec2i(shape1, shape2),
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            )


class BroadPhaseHash:
    """Spatial hash grid broad phase collision detection.

    Uses a uniform spatial hash grid for expected O(N) broad phase collision
    detection. Each shape is assigned to the grid cell containing its AABB
    center, and candidate pairs are found by querying the 3x3x3 neighborhood
    of cells around each shape.

    This approach works best when objects are roughly the same size.  The cell
    size must be >= the largest AABB extent to guarantee all overlapping pairs
    are found.

    For scenes with large size disparity (e.g., a ground plane alongside small
    objects), consider :class:`BroadPhaseSAP` or :class:`BroadPhaseBVH`.
    """

    def __init__(
        self,
        shape_world: wp.array(dtype=wp.int32, ndim=1) | np.ndarray,
        shape_flags: wp.array(dtype=wp.int32, ndim=1) | np.ndarray | None = None,
        cell_size: float | None = None,
        max_shapes_per_cell: int = 128,
        device: Devicelike | None = None,
    ) -> None:
        """Initialize the spatial hash grid broad phase.

        Args:
            shape_world: Array of world IDs.  Positive/zero values are distinct
                worlds, -1 indicates shared entities that collide with all worlds.
            shape_flags: Optional shape flags for filtering non-colliding shapes.
            cell_size: Uniform grid cell size [m].  Must be >= the largest AABB
                extent for correctness.  If None, auto-computed from AABBs on
                the first :meth:`launch` call (requires one-time CPU-GPU sync).
            max_shapes_per_cell: Maximum shapes stored per grid cell.  Shapes
                beyond this limit in a single cell are silently dropped.
            device: Device for GPU arrays.
        """
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

        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        self.world_count = len(slice_ends_np)
        self.num_regular_worlds = max(0, self.world_count - 1)
        self.device = device

        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)

        self.max_shapes_per_world = 0
        start_idx = 0
        for end_idx in slice_ends_np:
            self.max_shapes_per_world = max(self.max_shapes_per_world, int(end_idx - start_idx))
            start_idx = end_idx

        self._cell_size = cell_size
        self.max_shapes_per_cell = max_shapes_per_cell

        # Hash table: capacity ~ 4x total entries for ~25% load factor
        total_shape_entries = max(1, len(index_map_np))
        hash_capacity = max(256, total_shape_entries * 4)
        self.hash_table = HashTable(hash_capacity, device=device)

        capacity = self.hash_table.capacity
        self.cell_shape_counts = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.cell_shape_indices = wp.zeros(
            capacity * max_shapes_per_cell, dtype=wp.int32, device=device
        )

    @property
    def cell_size(self) -> float | None:
        """Current cell size, or None if not yet computed."""
        return self._cell_size

    def reset_cell_size(self) -> None:
        """Force cell size recomputation on the next :meth:`launch` call."""
        self._cell_size = None

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),
        shape_gap: wp.array(dtype=float, ndim=1) | None,
        shape_collision_group: wp.array(dtype=int, ndim=1),
        shape_world: wp.array(dtype=int, ndim=1),
        shape_count: int,
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        candidate_pair_count: wp.array(dtype=int, ndim=1),
        device: Devicelike | None = None,
        filter_pairs: wp.array(dtype=wp.vec2i, ndim=1) | None = None,
        num_filter_pairs: int | None = None,
    ) -> None:
        """Launch spatial hash grid broad phase collision detection.

        Populates ``candidate_pair`` with overlapping shape pairs whose AABBs
        overlap (with optional margin expansion), whose collision groups allow
        interaction, and whose world indices are compatible.  Pairs listed in
        ``filter_pairs`` are excluded.

        Args:
            shape_lower: Lower bounds of each shape's AABB.
            shape_upper: Upper bounds of each shape's AABB.
            shape_gap: Optional per-shape effective gap.  If None or empty,
                AABBs are assumed pre-expanded.
            shape_collision_group: Collision group per shape.
            shape_world: World index per shape.
            shape_count: Number of active bounding boxes.
            candidate_pair: Output array for overlapping shape pairs.
            candidate_pair_count: Output array (size 1) for pair count.
            device: Device to launch on.
            filter_pairs: Sorted excluded pairs (empty if none).
            num_filter_pairs: Number of excluded pairs.
        """
        max_candidate_pair = candidate_pair.shape[0]
        candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        if self.max_shapes_per_world == 0:
            return

        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)

        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # Auto-compute cell size from AABBs (one-time CPU-GPU sync)
        if self._cell_size is None:
            lower_np = shape_lower.numpy()
            upper_np = shape_upper.numpy()
            index_map_np = self.world_index_map.numpy()
            unique_indices = np.unique(index_map_np)
            if len(unique_indices) > 0:
                extents = upper_np[unique_indices] - lower_np[unique_indices]
                per_shape_max = np.max(extents, axis=1)
                if shape_gap.shape[0] > 0:
                    gaps_np = shape_gap.numpy()
                    per_shape_max = per_shape_max + 2.0 * gaps_np[unique_indices]
                self._cell_size = max(float(np.max(per_shape_max)) * 1.01, 1e-6)
            else:
                self._cell_size = 1.0

        inv_cell_size = 1.0 / self._cell_size

        # Clear hash table (active entries only) and all cell counts
        self.hash_table.clear_active()
        self.cell_shape_counts.zero_()

        # Phase 1: insert each shape into its center grid cell
        wp.launch(
            _hash_insert_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
                inv_cell_size,
                self.hash_table.keys,
                self.hash_table.active_slots,
                self.cell_shape_counts,
                self.cell_shape_indices,
                self.max_shapes_per_cell,
            ],
            device=device,
        )

        # Phase 2: query 3x3x3 neighborhood and generate candidate pairs
        wp.launch(
            _hash_query_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_collision_group,
                shape_world,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.num_regular_worlds,
                inv_cell_size,
                self.hash_table.keys,
                self.cell_shape_counts,
                self.cell_shape_indices,
                self.max_shapes_per_cell,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            ],
            device=device,
        )
