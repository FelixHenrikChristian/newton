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

from __future__ import annotations

import warp as wp

from .broad_phase_common import (
    check_aabb_overlap,
    test_world_and_group_pair,
    write_pair,
)

@wp.kernel
def _bvh_expand_aabb_kernel(
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(dtype=float, ndim=1),
    # Outputs
    expanded_lower: wp.array(dtype=wp.vec3, ndim=1),
    expanded_upper: wp.array(dtype=wp.vec3, ndim=1),
):
    """Expand AABBs by contact margin for BVH construction."""
    tid = wp.tid()

    lower = shape_lower[tid]
    upper = shape_upper[tid]

    # Check if margins are provided
    margin = 0.0
    if shape_contact_margin.shape[0] > 0:
        margin = shape_contact_margin[tid]

    expanded_lower[tid] = wp.vec3(lower[0] - margin, lower[1] - margin, lower[2] - margin)
    expanded_upper[tid] = wp.vec3(upper[0] + margin, upper[1] + margin, upper[2] + margin)


@wp.kernel
def _bvh_broadphase_simple_kernel(
    bvh_id: wp.uint64,
    shape_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    shape_count: int,
    # Outputs
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    """BVH-based broad phase kernel.

    Simple implementation that iterates over all shapes directly.
    World and collision group filtering is handled by test_world_and_group_pair.
    """
    shape1 = wp.tid()

    if shape1 >= shape_count:
        return

    # Get shape1's AABB bounds
    lower1 = shape_lower[shape1]
    upper1 = shape_upper[shape1]

    # Get margin for shape1
    margin1 = 0.0
    if shape_contact_margin.shape[0] > 0:
        margin1 = shape_contact_margin[shape1]

    # Expand the query AABB by margin
    query_lower = wp.vec3(lower1[0] - margin1, lower1[1] - margin1, lower1[2] - margin1)
    query_upper = wp.vec3(upper1[0] + margin1, upper1[1] + margin1, upper1[2] + margin1)

    # Get world and collision group for shape1
    world1 = shape_world[shape1]
    col_group1 = collision_group[shape1]

    # Query BVH for overlapping shapes
    query = wp.bvh_query_aabb(bvh_id, query_lower, query_upper)
    query_index = wp.int32(-1)

    while wp.bvh_query_next(query, query_index):
        shape2 = query_index

        # Skip self-collision and ensure canonical ordering (shape1 < shape2)
        if shape1 >= shape2:
            continue

        # Skip invalid indices (BVH may contain uninitialized entries)
        if shape2 >= shape_count:
            continue

        # Get world and collision group for shape2
        world2 = shape_world[shape2]
        col_group2 = collision_group[shape2]

        # Check world and collision group compatibility
        if not test_world_and_group_pair(world1, world2, col_group1, col_group2):
            continue

        # Get margin for shape2
        margin2 = 0.0
        if shape_contact_margin.shape[0] > 0:
            margin2 = shape_contact_margin[shape2]

        # Verify AABB overlap (BVH query may have false positives due to hierarchy)
        if check_aabb_overlap(
            shape_lower[shape1],
            shape_upper[shape1],
            margin1,
            shape_lower[shape2],
            shape_upper[shape2],
            margin2,
        ):
            write_pair(
                wp.vec2i(shape1, shape2),
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            )


class BroadPhaseBVH:
    """BVH-based broad phase collision detection.

    This class implements broad phase collision detection using a Bounding Volume Hierarchy (BVH).
    BVH provides O(N log N) construction time and O(log N) query time, making it efficient for
    scenes with many objects, especially when objects are not uniformly distributed.

    The BVH is rebuilt each frame to support dynamic scenes. For each shape, the algorithm
    queries the BVH to find potentially overlapping shapes, then verifies the overlap and
    applies world/collision group filtering.
    """

    def __init__(self, shape_world, shape_flags=None, device=None):
        """Initialize the BVH broad phase with world ID information.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values (-1) represent
                shared entities that belong to all worlds.
            shape_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
        """
        # Convert to numpy if it's a warp array
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        # Get total number of shapes for BVH construction
        self.num_shapes = len(shape_world_np)
        self.device = device

        # BVH will be built during launch()
        self.bvh = None

        # Arrays for expanded AABBs will be allocated in launch() to match shape_count
        self.expanded_lower = None
        self.expanded_upper = None
        self._allocated_size = 0

    def launch(
        self,
        shape_lower: wp.array,
        shape_upper: wp.array,
        shape_contact_margin: wp.array | None,
        shape_collision_group: wp.array,
        shape_shape_world: wp.array,
        shape_count: int,
        # Outputs
        candidate_pair: wp.array,
        num_candidate_pair: wp.array,
        device=None,
    ):
        """Launch the BVH broad phase collision detection.

        This method performs collision detection using a BVH structure. It first builds
        the BVH from the shape AABBs, then queries each shape against the BVH to find
        potentially overlapping pairs.

        Args:
            shape_lower: Array of lower bounds for each shape's AABB
            shape_upper: Array of upper bounds for each shape's AABB
            shape_contact_margin: Optional array of per-shape contact margins. If None or empty array,
                assumes AABBs are pre-expanded (margins = 0). If provided, margins are added during overlap checks.
            shape_collision_group: Array of collision group IDs for each shape.
            shape_shape_world: Array of world indices for each shape. Index -1 indicates global entities
                that collide with all worlds.
            shape_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping shape pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.
        """
        max_candidate_pair = candidate_pair.shape[0]
        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        # If no margins provided, pass empty array (kernel will use 0.0 margins)
        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        # Reallocate expanded AABB arrays if needed to match shape_count
        # This ensures BVH only contains valid AABBs
        need_rebuild = False
        if self._allocated_size != shape_count:
            self.expanded_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.expanded_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self._allocated_size = shape_count
            need_rebuild = True

        # Expand AABBs by contact margin for BVH construction
        wp.launch(
            kernel=_bvh_expand_aabb_kernel,
            dim=shape_count,
            inputs=[
                shape_lower,
                shape_upper,
                shape_contact_margin,
                self.expanded_lower,
                self.expanded_upper,
            ],
            device=device,
        )

        # Build BVH on first call or when size changes, otherwise refit
        # Using refit() allows the BVH to work inside CUDA graph capture
        if self.bvh is None or need_rebuild:
            self.bvh = wp.Bvh(self.expanded_lower, self.expanded_upper)
        else:
            self.bvh.refit()

        # Launch the BVH query kernel
        # Simple implementation that iterates over all shapes directly
        wp.launch(
            kernel=_bvh_broadphase_simple_kernel,
            dim=shape_count,
            inputs=[
                self.bvh.id,
                shape_lower,
                shape_upper,
                shape_contact_margin,
                shape_collision_group,
                shape_shape_world,
                shape_count,
            ],
            outputs=[
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
            device=device,
        )
