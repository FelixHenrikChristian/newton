# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "robot" / "TRAINING-Aliengo"
BASE_DEMO_PATH = EXAMPLE_DIR / "spot_pick_place_demo.py"
LAYOUT_PATH = EXAMPLE_DIR / "spot_pick_place_demo_robust_layout.py"
LAYOUT_SPEC = importlib.util.spec_from_file_location("spot_pick_place_demo_robust_layout", LAYOUT_PATH)
if LAYOUT_SPEC is None or LAYOUT_SPEC.loader is None:
    raise ImportError(f"Cannot load robust layout helper from {LAYOUT_PATH}")
layout = importlib.util.module_from_spec(LAYOUT_SPEC)
LAYOUT_SPEC.loader.exec_module(layout)


def _literal_constant(module_path: Path, name: str):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "np"
                and value.func.attr == "array"
            ):
                return np.array(ast.literal_eval(value.args[0]), dtype=np.float64)
            return ast.literal_eval(value)
    raise AssertionError(f"Constant {name} not found in {module_path}")


def _add_box_suffixes(module_path: Path, function_name: str) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                str(call.args[0].value)
                for call in ast.walk(node)
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "add_box"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                )
            }
    raise AssertionError(f"Function {function_name} not found in {module_path}")


class TestSpotPickPlaceRobustLayout(unittest.TestCase):
    def test_route_yaw_uses_a_to_c_direction(self):
        a_point = np.array([5.0, -6.0], dtype=np.float64)
        c_point = np.array([13.0, -1.0], dtype=np.float64)

        self.assertAlmostEqual(layout.route_yaw(a_point, c_point), math.atan2(5.0, 8.0))

    def test_default_c_stand_point_is_rover_right_rear(self):
        a_point = np.array([5.0, -6.0], dtype=np.float64)
        c_point = np.array([13.0, -1.0], dtype=np.float64)

        stand_point = layout.default_c_stand_point(a_point, c_point)

        yaw = layout.route_yaw(a_point, c_point)
        forward, side_axis = layout.xy_axes_from_yaw(yaw)
        cargo_center = layout.rover_cargo_center(c_point, yaw)
        local = stand_point - cargo_center

        self.assertLess(local @ forward, -0.55)
        self.assertLess(local @ side_axis, -0.35)

    def test_default_stones_are_randomized_front_side_of_c_stand(self):
        a_point = np.array([5.0, -6.0], dtype=np.float64)
        c_point = np.array([13.0, -1.0], dtype=np.float64)

        stones = layout.build_c_side_stone_layout(a_point, c_point, count=layout.DEFAULT_STONE_COUNT, side=-1)
        stones_again = layout.build_c_side_stone_layout(a_point, c_point, count=layout.DEFAULT_STONE_COUNT, side=-1)
        stand_point = layout.default_c_stand_point(a_point, c_point)

        self.assertEqual(layout.DEFAULT_STONE_COUNT, 5)
        self.assertEqual(stones.shape, (layout.DEFAULT_STONE_COUNT, 2))
        np.testing.assert_allclose(stones_again, stones)

        yaw = layout.route_yaw(a_point, c_point)
        forward, side_axis = layout.xy_axes_from_yaw(yaw)
        cargo_center = layout.rover_cargo_center(c_point, yaw)
        stand_local = stones - stand_point
        stand_local_x = stand_local @ forward
        stand_local_y = stand_local @ side_axis
        cargo_local_y = (stones - cargo_center) @ side_axis
        pair_distances = np.linalg.norm(stones[:, None, :] - stones[None, :, :], axis=2)
        pair_distances += np.eye(len(stones))

        self.assertTrue(np.all(stand_local_x >= 0.30 - 1.0e-6), msg=f"stand_local_x={stand_local_x}")
        self.assertTrue(np.all(stand_local_x <= 0.65 + 1.0e-6), msg=f"stand_local_x={stand_local_x}")
        self.assertTrue(np.all(stand_local_y <= -0.17), msg=f"stand_local_y={stand_local_y}")
        self.assertTrue(np.all(stand_local_y >= -0.38), msg=f"stand_local_y={stand_local_y}")
        self.assertGreater(float(np.ptp(stand_local_x)), 0.22)
        self.assertGreater(float(np.ptp(stand_local_y)), 0.08)
        self.assertFalse(np.all(np.diff(stand_local_x) >= 0.0), msg=f"stand_local_x={stand_local_x}")
        self.assertGreater(float(np.min(pair_distances)), 0.12)
        self.assertTrue(np.all(cargo_local_y < -0.55), msg=f"cargo_y={cargo_local_y}")

    def test_default_stone_yaws_are_randomized_around_base_yaw(self):
        base_yaw = math.radians(15.0)

        yaws = layout.build_c_side_stone_yaws(layout.DEFAULT_STONE_COUNT, base_yaw=base_yaw)
        yaws_again = layout.build_c_side_stone_yaws(layout.DEFAULT_STONE_COUNT, base_yaw=base_yaw)

        self.assertEqual(yaws.shape, (layout.DEFAULT_STONE_COUNT,))
        np.testing.assert_allclose(yaws_again, yaws)
        self.assertGreater(float(np.ptp(yaws)), math.radians(20.0))
        self.assertTrue(np.all(np.abs(yaws - base_yaw) <= math.radians(35.0) + 1.0e-6), msg=f"yaws={yaws}")

    def test_release_xy_keeps_overlapping_cluster_for_natural_stack(self):
        c_point = np.array([13.0, -1.0], dtype=np.float64)
        yaw = math.atan2(5.0, 8.0)

        targets = np.array(
            [layout.stone_release_xy(c_point, yaw, stone_index=index, stone_count=5) for index in range(5)]
        )
        targets_again = np.array(
            [layout.stone_release_xy(c_point, yaw, stone_index=index, stone_count=5) for index in range(5)]
        )

        np.testing.assert_allclose(targets_again, targets)

        forward, side_axis = layout.xy_axes_from_yaw(yaw)
        cargo_center = layout.rover_cargo_center(c_point, yaw)
        local = targets - cargo_center
        local_x = local @ forward
        local_y = local @ side_axis
        pair_distances = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=2)
        pair_distances = pair_distances[~np.eye(len(local), dtype=bool)]

        self.assertGreater(float(np.ptp(local_x)), 0.04)
        self.assertGreater(float(np.ptp(local_y)), 0.03)
        self.assertLessEqual(float(np.ptp(local_x)), 0.10, msg=f"local_x={local_x}")
        self.assertLessEqual(float(np.ptp(local_y)), 0.08, msg=f"local_y={local_y}")
        self.assertLessEqual(float(np.max(pair_distances)), 0.12, msg=f"pair_distances={pair_distances}")
        self.assertGreater(float(np.min(pair_distances)), 0.015, msg=f"pair_distances={pair_distances}")

    def test_stone_heights_drop_then_settle_on_surface(self):
        terrain_z = 1.25
        stone_radius_z = 0.035
        drop_height = 0.08

        initial_z = layout.initial_stone_center_z(terrain_z, stone_radius_z, drop_height)
        settled_z = layout.settled_stone_center_z(terrain_z, stone_radius_z)
        release_z = layout.release_stone_center_z(terrain_z, stone_radius_z)

        self.assertAlmostEqual(settled_z, terrain_z + stone_radius_z)
        self.assertAlmostEqual(initial_z, settled_z + drop_height)
        self.assertGreater(release_z - settled_z, 0.43)

    def test_stone_collision_masks_allow_stone_stack(self):
        contype, conaffinity = layout.stone_collision_masks()

        self.assertNotEqual(contype & conaffinity, 0)

    def test_stone_friction_allows_rover_stack_rolloff(self):
        mu, mu_torsional, mu_rolling = layout.stone_contact_friction()

        self.assertGreaterEqual(mu, 0.8)
        self.assertLessEqual(mu_torsional, 0.03)
        self.assertLessEqual(mu_rolling, 0.03)

    def test_rover_wheels_are_centered_under_forward_chassis(self):
        wheel_offsets = np.array(_literal_constant(BASE_DEMO_PATH, "C_ROVER_WHEEL_X_OFFSETS"), dtype=np.float64)
        chassis_center = np.array(_literal_constant(BASE_DEMO_PATH, "C_ROVER_CHASSIS_CENTER_OFFSET"), dtype=np.float64)
        chassis_half_extents = np.array(
            _literal_constant(BASE_DEMO_PATH, "C_ROVER_CHASSIS_HALF_EXTENTS"), dtype=np.float64
        )

        chassis_min_x = chassis_center[0] - chassis_half_extents[0]
        chassis_max_x = chassis_center[0] + chassis_half_extents[0]
        rear_gap = float(wheel_offsets[0] - chassis_min_x)
        front_gap = float(chassis_max_x - wheel_offsets[-1])

        self.assertEqual(wheel_offsets.shape, (3,))
        self.assertTrue(np.all(np.diff(wheel_offsets) > 0.0), msg=f"wheel_offsets={wheel_offsets}")
        self.assertGreaterEqual(float(np.mean(wheel_offsets)), float(chassis_center[0] - 0.05))
        self.assertLessEqual(abs(front_gap - rear_gap), 0.08, msg=f"rear_gap={rear_gap}, front_gap={front_gap}")

    def test_rover_rear_trim_removes_lip_and_shortens_chassis(self):
        chassis_half_extents = np.array(
            _literal_constant(BASE_DEMO_PATH, "C_ROVER_CHASSIS_HALF_EXTENTS"), dtype=np.float64
        )
        box_suffixes = _add_box_suffixes(BASE_DEMO_PATH, "_add_c_rover")

        self.assertLessEqual(chassis_half_extents[0], 0.64)
        self.assertNotIn("rear_loading_lip", box_suffixes)

    def test_route_targets_keep_a_to_c_segments_short(self):
        a_point = np.array([5.0, -6.0], dtype=np.float64)
        c_point = np.array([13.0, -1.0], dtype=np.float64)

        targets = layout.build_route_targets(a_point, c_point, max_segment_length=3.5)

        self.assertGreater(len(targets), 1)
        np.testing.assert_allclose(targets[-1], c_point)
        points = np.vstack((a_point, targets))
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertTrue(np.all(segment_lengths <= 3.5 + 1.0e-6), msg=f"segments={segment_lengths}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
