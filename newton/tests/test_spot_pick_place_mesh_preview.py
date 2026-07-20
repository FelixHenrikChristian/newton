# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "robot" / "TRAINING-Aliengo"
MESH_HELPER_PATH = EXAMPLE_DIR / "spot_pick_place_mesh.py"
ROBUST_GRASP_PATH = EXAMPLE_DIR / "spot_pick_place_demo_robust_grasp.py"
ROCK_DIR = EXAMPLE_DIR / "lunar_mujoco_spot_arm_mining_scene"
ELLIPSOID_RADII = np.array([0.06, 0.045, 0.035], dtype=np.float32)


def _load_mesh_helper():
    spec = importlib.util.spec_from_file_location("spot_pick_place_mesh", MESH_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mesh helper from {MESH_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Constant not found: {name}")


class TestSpotPickPlaceMeshPreview(unittest.TestCase):
    def test_centered_rock_meshes_scale_to_ellipsoid_bounds(self):
        mesh_helper = _load_mesh_helper()

        for asset_name in ("rock1_centered_unit.obj", "rock2_centered_unit.obj"):
            vertices, indices = mesh_helper.load_obj_triangles(ROCK_DIR / asset_name)
            scale = mesh_helper.scale_to_ellipsoid_bounds(vertices, ELLIPSOID_RADII)

            self.assertEqual(vertices.dtype, np.float32)
            self.assertEqual(indices.dtype, np.int32)
            self.assertEqual(indices.ndim, 1)
            self.assertEqual(indices.size % 3, 0)
            np.testing.assert_allclose(
                (np.max(vertices, axis=0) - np.min(vertices, axis=0)) * scale,
                2.0 * ELLIPSOID_RADII,
                rtol=1.0e-6,
                atol=1.0e-6,
            )

    def test_centered_rock_meshes_return_grasp_axis_surface_points(self):
        mesh_helper = _load_mesh_helper()
        grasp_height = float(ELLIPSOID_RADII[2] * 0.45)

        for asset_name in ("rock1_centered_unit.obj", "rock2_centered_unit.obj"):
            vertices, indices = mesh_helper.load_obj_triangles(ROCK_DIR / asset_name)
            scaled_vertices = vertices * mesh_helper.scale_to_ellipsoid_bounds(vertices, ELLIPSOID_RADII)
            negative_side, positive_side = mesh_helper.grasp_axis_surface_points(scaled_vertices, indices, grasp_height)

            self.assertLess(negative_side[0], 0.0)
            self.assertGreater(positive_side[0], 0.0)
            np.testing.assert_allclose(negative_side[1:], [0.0, grasp_height], atol=1.0e-6)
            np.testing.assert_allclose(positive_side[1:], [0.0, grasp_height], atol=1.0e-6)
            self.assertGreater(positive_side[0] - negative_side[0], 0.06)

    def test_demo_assigns_first_two_stones_to_centered_rock_meshes(self):
        self.assertEqual(
            _literal_constant(ROBUST_GRASP_PATH, "MESH_STONE_ASSET_FILENAMES"),
            ("rock1_centered_unit.obj", "rock2_centered_unit.obj"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
