# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np

ROVER_RELEASE_FORWARD_OFFSET = 0.75
ROVER_C_ARRIVAL_RADIUS = 0.45
ROVER_OUTER_SIDE_CLEARANCE = 0.34
STONE_COLLISION_CONTYPE = 4
STONE_COLLISION_CONAFFINITY = 7
STONE_CONTACT_MU = 1.0
STONE_CONTACT_MU_TORSIONAL = 0.015
STONE_CONTACT_MU_ROLLING = 0.015

DEFAULT_STONE_COUNT = 5
DEFAULT_STONE_SIDE = -1
DEFAULT_C_STAND_FORWARD_OFFSET = -0.35
DEFAULT_C_STAND_SIDE_OFFSET = -0.45
DEFAULT_STONE_FORWARD_OFFSET = 0.30
DEFAULT_STONE_SIDE_OFFSET = 0.28
DEFAULT_STONE_SPACING = 0.13
DEFAULT_STONE_FORWARD_SPREAD = 0.35
DEFAULT_STONE_SIDE_SPREAD = 0.20
DEFAULT_STONE_LAYOUT_SEED = 2
DEFAULT_STONE_YAW_JITTER = math.radians(35.0)
DEFAULT_STONE_DROP_HEIGHT = 0.15
DEFAULT_STONE_RELEASE_CLEARANCE = 0.48
DEFAULT_RELEASE_OFFSET_SEED = 1
DEFAULT_RELEASE_FORWARD_JITTER = 0.05
DEFAULT_RELEASE_SIDE_JITTER = 0.04
DEFAULT_ROUTE_SEGMENT_LENGTH = 3.5


def route_yaw(a_point: np.ndarray, c_point: np.ndarray) -> float:
    """Return the route yaw from A directly to C."""

    direction = c_point - a_point
    if float(np.linalg.norm(direction)) < 1.0e-6:
        return 0.0
    return float(math.atan2(direction[1], direction[0]))


def xy_axes_from_yaw(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and left-side axes for a horizontal yaw."""

    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    side_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return forward, side_axis


def rover_cargo_center(
    c_point: np.ndarray,
    yaw: float,
    *,
    release_forward_offset: float = ROVER_RELEASE_FORWARD_OFFSET,
    c_arrival_radius: float = ROVER_C_ARRIVAL_RADIUS,
) -> np.ndarray:
    """Return the cargo box center used by the rover at C."""

    forward, _ = xy_axes_from_yaw(yaw)
    return c_point + forward * max(release_forward_offset - c_arrival_radius, 0.0)


def _release_local_offset(stone_index: int, stone_count: int) -> np.ndarray:
    if stone_count == 1:
        return np.zeros(2, dtype=np.float64)

    rng = np.random.default_rng(DEFAULT_RELEASE_OFFSET_SEED)
    offsets = np.column_stack(
        (
            rng.uniform(-DEFAULT_RELEASE_FORWARD_JITTER, DEFAULT_RELEASE_FORWARD_JITTER, stone_count),
            rng.uniform(-DEFAULT_RELEASE_SIDE_JITTER, DEFAULT_RELEASE_SIDE_JITTER, stone_count),
        )
    )
    return offsets[stone_index].astype(np.float64)


def stone_release_xy(c_point: np.ndarray, yaw: float, *, stone_index: int, stone_count: int) -> np.ndarray:
    """Return the cargo release XY with deterministic offsets for natural stacking."""

    if stone_index < 0 or stone_index >= stone_count:
        raise IndexError(f"Stone index out of range: {stone_index}")
    forward, side_axis = xy_axes_from_yaw(yaw)
    local_offset = _release_local_offset(stone_index, stone_count)
    return rover_cargo_center(c_point, yaw) + local_offset[0] * forward + local_offset[1] * side_axis


def release_stone_center_z(
    terrain_z: float,
    stone_radius_z: float,
    release_clearance: float = DEFAULT_STONE_RELEASE_CLEARANCE,
) -> float:
    """Return the raised stone center height used when opening the gripper over the rover."""

    if stone_radius_z < 0.0:
        raise ValueError("Stone z radius must be non-negative.")
    if release_clearance < 0.0:
        raise ValueError("Stone release clearance must be non-negative.")
    return float(terrain_z + stone_radius_z + release_clearance)


def initial_stone_center_z(terrain_z: float, stone_radius_z: float, drop_height: float) -> float:
    """Return the initial raised stone center height before settling."""

    if stone_radius_z < 0.0:
        raise ValueError("Stone z radius must be non-negative.")
    if drop_height < 0.0:
        raise ValueError("Stone drop height must be non-negative.")
    return float(terrain_z + stone_radius_z + drop_height)


def settled_stone_center_z(terrain_z: float, stone_radius_z: float) -> float:
    """Return the non-embedded stone center height on the terrain surface."""

    if stone_radius_z < 0.0:
        raise ValueError("Stone z radius must be non-negative.")
    return float(terrain_z + stone_radius_z)


def stone_collision_masks() -> tuple[int, int]:
    """Return MuJoCo collision masks that keep stone-stone contacts enabled."""

    return STONE_COLLISION_CONTYPE, STONE_COLLISION_CONAFFINITY


def stone_contact_friction() -> tuple[float, float, float]:
    """Return stone contact friction with low torsional and rolling resistance."""

    return STONE_CONTACT_MU, STONE_CONTACT_MU_TORSIONAL, STONE_CONTACT_MU_ROLLING


def default_c_stand_point(
    a_point: np.ndarray,
    c_point: np.ndarray,
    *,
    forward_offset: float = DEFAULT_C_STAND_FORWARD_OFFSET,
    side_offset: float = DEFAULT_C_STAND_SIDE_OFFSET,
) -> np.ndarray:
    """Return the natural grasp stand point near C, outside the rover body."""

    yaw = route_yaw(a_point, c_point)
    forward, side_axis = xy_axes_from_yaw(yaw)
    return c_point + forward * forward_offset + side_axis * side_offset


def _random_local_stone_offsets(
    count: int,
    *,
    side: int,
    forward_offset: float,
    side_offset: float,
    spacing: float,
) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError("Stone spacing must be positive.")

    sample_count = max(count, DEFAULT_STONE_COUNT)
    rng = np.random.default_rng(DEFAULT_STONE_LAYOUT_SEED)
    local_offsets: list[np.ndarray] = []
    side_min = side_offset - 0.5 * DEFAULT_STONE_SIDE_SPREAD
    side_max = side_offset + 0.5 * DEFAULT_STONE_SIDE_SPREAD
    max_attempts = max(500, sample_count * 500)

    for _ in range(max_attempts):
        candidate = np.array(
            [
                rng.uniform(forward_offset, forward_offset + DEFAULT_STONE_FORWARD_SPREAD),
                float(side) * rng.uniform(side_min, side_max),
            ],
            dtype=np.float64,
        )
        if all(float(np.linalg.norm(candidate - existing)) >= spacing for existing in local_offsets):
            local_offsets.append(candidate)
            if len(local_offsets) == sample_count:
                break

    if len(local_offsets) < sample_count:
        raise ValueError(f"Could not place {count} stones with spacing {spacing:.3f} m.")

    return np.array(local_offsets[:count], dtype=np.float64)


def build_c_side_stone_yaws(
    count: int,
    *,
    base_yaw: float = 0.0,
    yaw_jitter: float = DEFAULT_STONE_YAW_JITTER,
) -> np.ndarray:
    """Build deterministic randomized stone yaw angles around the vertical axis."""

    if count < 1:
        raise ValueError("Stone count must be at least 1.")
    if yaw_jitter < 0.0:
        raise ValueError("Stone yaw jitter must be non-negative.")

    rng = np.random.default_rng(DEFAULT_STONE_LAYOUT_SEED + 1)
    return base_yaw + rng.uniform(-yaw_jitter, yaw_jitter, size=count).astype(np.float64)


def build_c_side_stone_layout(
    a_point: np.ndarray,
    c_point: np.ndarray,
    *,
    count: int = DEFAULT_STONE_COUNT,
    side: int = DEFAULT_STONE_SIDE,
    forward_offset: float = DEFAULT_STONE_FORWARD_OFFSET,
    side_offset: float = DEFAULT_STONE_SIDE_OFFSET,
    spacing: float = DEFAULT_STONE_SPACING,
) -> np.ndarray:
    """Build deterministic randomized stone XY positions at the selected C-side cluster."""

    if count < 1:
        raise ValueError("Stone count must be at least 1.")
    if side not in (-1, 1):
        raise ValueError("Stone side must be -1 or 1.")

    yaw = route_yaw(a_point, c_point)
    forward, side_axis = xy_axes_from_yaw(yaw)
    stand_point = default_c_stand_point(a_point, c_point)
    local_offsets = _random_local_stone_offsets(
        count,
        side=side,
        forward_offset=forward_offset,
        side_offset=side_offset,
        spacing=spacing,
    )

    return stand_point + local_offsets[:, :1] * forward + local_offsets[:, 1:] * side_axis


def build_route_targets(
    a_point: np.ndarray,
    c_point: np.ndarray,
    *,
    max_segment_length: float = DEFAULT_ROUTE_SEGMENT_LENGTH,
) -> np.ndarray:
    """Build internal straight-line locomotion targets from A to C, including C."""

    if max_segment_length <= 0.0:
        raise ValueError("Route segment length must be positive.")

    delta = c_point - a_point
    distance = float(np.linalg.norm(delta))
    if distance < 1.0e-6:
        return c_point.reshape(1, 2).copy()

    segment_count = max(1, int(math.ceil(distance / max_segment_length)))
    fractions = np.linspace(1.0 / segment_count, 1.0, segment_count, dtype=np.float64)
    return a_point + fractions[:, None] * delta
