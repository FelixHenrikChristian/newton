# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class VisualTerrainDeformer:
    """CPU-side visual terrain mesh with persistent footstep depressions."""

    def __init__(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        resolution: int | tuple[int, int],
        height_fn: Callable[[np.ndarray], float],
        *,
        visual_offset: float,
        texture_repeat: tuple[float, float] = (1.0, 1.0),
        max_depression: float = 0.08,
        max_ridge: float = 0.03,
    ):
        nx, ny = self._normalize_resolution(resolution)
        if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
            raise ValueError("Terrain bounds must be increasing.")

        self.x_bounds = (float(x_bounds[0]), float(x_bounds[1]))
        self.y_bounds = (float(y_bounds[0]), float(y_bounds[1]))
        self.resolution = (nx, ny)
        self.visual_offset = float(visual_offset)
        self.max_depression = float(max_depression)
        self.max_ridge = float(max_ridge)

        xs = np.linspace(self.x_bounds[0], self.x_bounds[1], nx, dtype=np.float64)
        ys = np.linspace(self.y_bounds[0], self.y_bounds[1], ny, dtype=np.float64)
        self._xx, self._yy = np.meshgrid(xs, ys)
        self._base_heights = np.empty((ny, nx), dtype=np.float64)
        for j in range(ny):
            for i in range(nx):
                self._base_heights[j, i] = float(height_fn(np.array([self._xx[j, i], self._yy[j, i]])))

        self._height_delta = np.zeros((ny, nx), dtype=np.float64)
        self._indices = self._build_indices(nx, ny)
        self._uvs = self._build_uvs(nx, ny, texture_repeat)
        self._points = np.empty((nx * ny, 3), dtype=np.float32)
        self._normals = np.empty((nx * ny, 3), dtype=np.float32)
        self._refresh_points()

    @staticmethod
    def _normalize_resolution(resolution: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = resolution
        if nx < 2 or ny < 2:
            raise ValueError("Terrain resolution must be at least 2 in each axis.")
        return int(nx), int(ny)

    @staticmethod
    def _build_indices(nx: int, ny: int) -> np.ndarray:
        indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                indices.extend((a, b, c, b, d, c))
        return np.asarray(indices, dtype=np.int32)

    @staticmethod
    def _build_uvs(nx: int, ny: int, texture_repeat: tuple[float, float]) -> np.ndarray:
        u = np.linspace(0.0, float(texture_repeat[0]), nx, dtype=np.float32)
        v = np.linspace(0.0, float(texture_repeat[1]), ny, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        return np.column_stack((uu.ravel(), vv.ravel())).astype(np.float32)

    def _refresh_points(self) -> None:
        z = self._base_heights + self._height_delta + self.visual_offset
        self._points[:, 0] = self._xx.ravel()
        self._points[:, 1] = self._yy.ravel()
        self._points[:, 2] = z.ravel()
        self._refresh_normals(z)

    def _refresh_normals(self, z: np.ndarray) -> None:
        xs = self._xx[0, :]
        ys = self._yy[:, 0]
        dz_dy, dz_dx = np.gradient(z, ys, xs, edge_order=1)
        normals = np.stack((-dz_dx, -dz_dy, np.ones_like(z)), axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1.0e-8)
        self._normals[:] = normals.reshape(-1, 3).astype(np.float32)

    def mesh_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return points, triangle indices, normals, and UVs suitable for viewer.log_mesh()."""

        return self._points.copy(), self._indices.copy(), self._normals.copy(), self._uvs.copy()

    def height_at_grid_nearest(self, xy: np.ndarray) -> float:
        """Return the visual mesh height at the closest grid vertex."""

        nx, ny = self.resolution
        x, y = np.asarray(xy, dtype=np.float64)
        i = int(round((x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) * (nx - 1)))
        j = int(round((y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) * (ny - 1)))
        i = int(np.clip(i, 0, nx - 1))
        j = int(np.clip(j, 0, ny - 1))
        return float(self._points[j * nx + i, 2])

    def stamp_contact(
        self,
        xy: np.ndarray,
        *,
        yaw: float,
        length: float,
        width: float,
        depth: float,
        ridge_height: float,
    ) -> int:
        """Press an oriented footprint into the visual terrain mesh."""

        if length <= 0.0 or width <= 0.0 or depth <= 0.0:
            return 0

        center = np.asarray(xy, dtype=np.float64)
        dx = self._xx - center[0]
        dy = self._yy - center[1]
        cos_yaw = np.cos(float(yaw))
        sin_yaw = np.sin(float(yaw))
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        half_length = 0.5 * float(length)
        half_width = 0.5 * float(width)
        radius_sq = (local_x / half_length) ** 2 + (local_y / half_width) ** 2

        before = self._height_delta.copy()
        depression_mask = radius_sq <= 1.0
        if np.any(depression_mask):
            depression = -float(depth) * (1.0 - radius_sq[depression_mask]) ** 2
            self._height_delta[depression_mask] = np.minimum(self._height_delta[depression_mask], depression)

        ridge_outer = 1.75
        radius = np.sqrt(radius_sq)
        ridge_mask = (radius > 1.0) & (radius <= ridge_outer)
        if ridge_height > 0.0 and np.any(ridge_mask):
            t = (radius[ridge_mask] - 1.0) / (ridge_outer - 1.0)
            ridge = float(ridge_height) * np.sin(np.pi * t) ** 2
            self._height_delta[ridge_mask] = np.maximum(self._height_delta[ridge_mask], ridge)

        self._height_delta = np.clip(self._height_delta, -self.max_depression, self.max_ridge)
        changed = int(np.count_nonzero(np.abs(self._height_delta - before) > 1.0e-7))
        if changed:
            self._refresh_points()
        return changed
