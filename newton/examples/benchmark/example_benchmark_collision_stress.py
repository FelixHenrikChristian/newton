# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Benchmark Collision Stress
#
# Builds a dense rigid-body pile with :class:`~newton.ModelBuilder` only
# (no MJCF). Geometry uses primitives that map cleanly to MuJoCo
# (plane, box, capsule, sphere) so the same layout can be reproduced in
# MJCF for cross-engine benchmarks.
#
# Command:
#   python -m newton.examples benchmark_collision_stress --backend vbd
#   python -m newton.examples benchmark_collision_stress --backend mujoco
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


def _build_model(num_bodies: int, seed: int, *, color_bodies: bool) -> newton.Model:
    rng = np.random.default_rng(seed)

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 400.0
    builder.default_shape_cfg.mu = 0.6
    # Contact response (maps to MuJoCo geom solref via stiffness/damping; helps VBD stability).
    builder.default_shape_cfg.ke = 1.0e6
    builder.default_shape_cfg.kd = 1.0e1

    builder.add_ground_plane()

    # Packed 3D grid so bodies overlap in XY and stack in Z → many pairwise contacts.
    nx = max(2, int(np.ceil(num_bodies ** (1.0 / 3.0))))
    ny = max(2, int(np.ceil((num_bodies / nx) ** 0.5)))
    nz = max(1, (num_bodies + nx * ny - 1) // (nx * ny))
    spacing_xy = 0.55
    layer_z = 0.35
    origin_x = -0.5 * (nx - 1) * spacing_xy
    origin_y = -0.5 * (ny - 1) * spacing_xy

    for i in range(num_bodies):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        x = origin_x + ix * spacing_xy + rng.uniform(-0.02, 0.02)
        y = origin_y + iy * spacing_xy + rng.uniform(-0.02, 0.02)
        z = 0.35 + iz * layer_z + rng.uniform(0.0, 0.04)
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))

        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(float(x), float(y), float(z)), q=q),
            label=f"debris_{i}",
        )

        kind = i % 3
        if kind == 0:
            builder.add_shape_box(body, hx=0.12, hy=0.1, hz=0.09)
        elif kind == 1:
            builder.add_shape_capsule(body, radius=0.08, half_height=0.14)
        else:
            builder.add_shape_sphere(body, radius=0.1)

    if color_bodies:
        builder.color()
    return builder.finalize()


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.backend = args.backend
        self.num_bodies = int(args.num_bodies)
        self.seed = int(args.seed)

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = _build_model(self.num_bodies, self.seed, color_bodies=(self.backend == "vbd"))

        if self.backend == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model, integrator="implicitfast")
            self.contacts = None
        elif self.backend == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=int(args.iterations))
            self.contacts = self.model.contacts()
        else:
            raise ValueError("backend must be 'mujoco' or 'vbd'")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if self.backend == "vbd":
                self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is not None:
            assert not wp.isnan(self.state_0.body_q).numpy().any()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--backend",
            type=str,
            default="vbd",
            choices=["mujoco", "vbd"],
            help="Solver backend.",
        )
        parser.add_argument("--num-bodies", type=int, default=96, help="Number of free-floating debris bodies.")
        parser.add_argument("--seed", type=int, default=1, help="Random seed for initial layout jitter.")
        parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per frame.")
        parser.add_argument("--iterations", type=int, default=10, help="VBD iterations per substep.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
