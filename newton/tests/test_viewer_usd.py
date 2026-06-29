# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
import warp as wp

from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerUSD

if USD_AVAILABLE:
    from pxr import UsdGeom, UsdShade


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestViewerUSD(unittest.TestCase):
    def _make_viewer(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))
        viewer = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer.close)
        self.addCleanup(lambda: setattr(viewer, "output_path", ""))
        return viewer

    def test_log_points_keeps_per_point_wp_vec3_colors_for_three_points(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        colors = wp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=wp.vec3,
        )

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_per_point", points, radii=0.01, colors=colors)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.vertex)
        np.testing.assert_allclose(display_color, colors.numpy(), atol=1e-6)

    def test_reuses_existing_layer_for_same_output_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".usda", delete=False)
        temp_file.close()
        self.addCleanup(lambda: os.path.exists(temp_file.name) and os.remove(temp_file.name))

        # Create first viewer and write some data into the stage.
        viewer1 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer1.close)
        self.addCleanup(lambda: setattr(viewer1, "output_path", ""))

        viewer1.begin_frame(0.0)
        points = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
        colors = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3)
        path = viewer1.log_points("/points_from_viewer1", points, radii=0.01, colors=colors)

        # Ensure the prim written by viewer1 is present before creating viewer2.
        prim_before = UsdGeom.Points.Get(viewer1.stage, path).GetPrim()
        self.assertTrue(prim_before.IsValid())

        # Create second viewer for the same output path; this should reuse the same
        # underlying layer and clear any previous contents.
        viewer2 = ViewerUSD(output_path=temp_file.name, num_frames=1)
        self.addCleanup(viewer2.close)
        self.addCleanup(lambda: setattr(viewer2, "output_path", ""))

        # Verify that the stage/layer reuse actually occurred.
        self.assertIsNotNone(viewer2.stage)
        self.assertIs(viewer1.stage.GetRootLayer(), viewer2.stage.GetRootLayer())

        # Verify that viewer2 cleared/overwrote viewer1's data.
        prim_after = UsdGeom.Points.Get(viewer2.stage, path).GetPrim()
        self.assertFalse(prim_after.IsValid())
        self.assertTrue(os.path.exists(temp_file.name))

    def test_log_points_treats_wp_float_triplet_as_single_constant_color(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )
        color_triplet = wp.array([0.25, 0.5, 0.75], dtype=wp.float32)

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_constant", points, radii=0.01, colors=color_triplet)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        display_color = np.asarray(points_prim.GetDisplayColorAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetDisplayColorAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.constant)
        np.testing.assert_allclose(display_color, np.array([[0.25, 0.5, 0.75]], dtype=np.float32), atol=1e-6)

    def test_log_points_defaults_radii_when_omitted(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype=wp.vec3,
        )

        viewer.begin_frame(0.0)
        path = viewer.log_points("/points_default_radii", points)

        points_prim = UsdGeom.Points.Get(viewer.stage, path)
        widths = np.asarray(points_prim.GetWidthsAttr().Get(viewer._frame_index), dtype=np.float32)
        interpolation = UsdGeom.Primvar(points_prim.GetWidthsAttr()).GetInterpolation()

        self.assertEqual(interpolation, UsdGeom.Tokens.constant)
        np.testing.assert_allclose(widths, np.array([0.2], dtype=np.float32), atol=1e-6)

    def test_log_instances_exports_uvs_and_preview_surface_material(self):
        viewer = self._make_viewer()

        points = wp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=wp.vec3,
        )
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        uvs = wp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=wp.vec2)
        texture_path = os.path.join(os.path.dirname(viewer.output_path), "albedo.png")

        viewer.begin_frame(0.0)
        viewer.log_mesh("/textured_mesh", points, indices, uvs=uvs, texture=texture_path)
        viewer.log_instances(
            "/textured_instances",
            "/textured_mesh",
            wp.array([wp.transform_identity()], dtype=wp.transform),
            wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3),
            wp.array([[0.25, 0.5, 0.75]], dtype=wp.vec3),
            wp.array([[0.8, 0.2, 0.0, 1.0]], dtype=wp.vec4),
        )
        viewer.begin_frame(1.0 / 60.0)
        viewer.log_instances(
            "/textured_instances",
            "/textured_mesh",
            wp.array([wp.transform_identity()], dtype=wp.transform),
            wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3),
            wp.array([[0.25, 0.5, 0.75]], dtype=wp.vec3),
            None,
        )

        mesh = UsdGeom.Mesh.Get(viewer.stage, "/root/textured_mesh")
        st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        self.assertTrue(st)
        self.assertEqual(st.GetInterpolation(), UsdGeom.Tokens.vertex)
        np.testing.assert_allclose(np.asarray(st.Get(), dtype=np.float32), uvs.numpy(), atol=1e-6)

        instance = viewer.stage.GetPrimAtPath("/root/textured_instances/instance_0")
        material, _ = UsdShade.MaterialBindingAPI(instance).ComputeBoundMaterial()
        self.assertTrue(material)

        preview_surfaces = [
            UsdShade.Shader(prim)
            for prim in viewer.stage.Traverse()
            if prim.IsA(UsdShade.Shader)
            and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        ]
        self.assertEqual(len(preview_surfaces), 1)
        shader = preview_surfaces[0]
        self.assertTrue(shader.GetOutput("surface"))
        self.assertAlmostEqual(shader.GetInput("roughness").Get(), 0.8)
        self.assertAlmostEqual(shader.GetInput("metallic").Get(), 0.2)

        texture_shaders = [
            UsdShade.Shader(prim)
            for prim in viewer.stage.Traverse()
            if prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetIdAttr().Get() == "UsdUVTexture"
        ]
        self.assertEqual(len(texture_shaders), 1)
        self.assertTrue(texture_shaders[0].GetOutput("rgb"))
        self.assertEqual(texture_shaders[0].GetInput("file").Get().path, "albedo.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
