# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and triangulated face indices."""

    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex at {path}:{line_number} has fewer than three coordinates.")
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"OBJ face at {path}:{line_number} has fewer than three vertices.")
            face = [_obj_vertex_index(field, len(vertices), path, line_number) for field in fields[1:]]
            for vertex_index in range(1, len(face) - 1):
                indices.extend((face[0], face[vertex_index], face[vertex_index + 1]))

    if not vertices:
        raise ValueError(f"OBJ mesh has no vertices: {path}")
    if not indices:
        raise ValueError(f"OBJ mesh has no faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def scale_to_ellipsoid_bounds(vertices: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return per-axis scale that matches an ellipsoid's bounding-box dimensions."""

    extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    if np.any(extents <= 0.0):
        raise ValueError("OBJ mesh must have positive extents along every axis.")
    return (2.0 * radii / extents).astype(np.float32)


def grasp_axis_surface_points(
    vertices: np.ndarray, indices: np.ndarray, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the negative- and positive-X mesh surface points at a grasp height."""

    vertices = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must have shape [vertex_count, 3].")
    if indices.ndim != 1 or indices.size % 3:
        raise ValueError("Mesh indices must be a flat triangle-index array.")

    origin = np.array([np.min(vertices[:, 0]) - 1.0, 0.0, height], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    intersections = []
    for triangle in vertices[indices.reshape(-1, 3)]:
        edge_1 = triangle[1] - triangle[0]
        edge_2 = triangle[2] - triangle[0]
        direction_cross_edge_2 = np.cross(direction, edge_2)
        determinant = float(np.dot(edge_1, direction_cross_edge_2))
        if abs(determinant) < 1.0e-8:
            continue

        inverse_determinant = 1.0 / determinant
        origin_offset = origin - triangle[0]
        u = inverse_determinant * float(np.dot(origin_offset, direction_cross_edge_2))
        if not 0.0 <= u <= 1.0:
            continue
        origin_cross_edge_1 = np.cross(origin_offset, edge_1)
        v = inverse_determinant * float(np.dot(direction, origin_cross_edge_1))
        if not (0.0 <= v <= 1.0 and u + v <= 1.0):
            continue
        distance = inverse_determinant * float(np.dot(edge_2, origin_cross_edge_1))
        if distance >= 0.0:
            intersections.append(origin[0] + distance)

    if len(intersections) < 2:
        raise ValueError(f"Expected two mesh surface intersections at grasp height {height:.6f}.")
    return (
        np.array([min(intersections), 0.0, height], dtype=np.float32),
        np.array([max(intersections), 0.0, height], dtype=np.float32),
    )


def _obj_vertex_index(field: str, vertex_count: int, path: Path, line_number: int) -> int:
    """Convert an OBJ face vertex field to a zero-based index."""

    vertex_field = field.split("/", maxsplit=1)[0]
    if not vertex_field:
        raise ValueError(f"OBJ face at {path}:{line_number} has no vertex index.")
    raw_index = int(vertex_field)
    if raw_index == 0:
        raise ValueError(f"OBJ face at {path}:{line_number} uses invalid vertex index 0.")
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if not 0 <= index < vertex_count:
        raise ValueError(f"OBJ face at {path}:{line_number} references an unknown vertex.")
    return index
