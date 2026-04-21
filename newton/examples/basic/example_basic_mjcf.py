# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic MJCF
#
# Loads an MJCF scene via ModelBuilder.add_mjcf() and runs it with either
# the MuJoCo backend or the VBD backend.
#
# This example intentionally uses the stock Newton MJCF importer without
# special-casing any MuJoCo-specific scene macros (e.g. <replicate>/<attach>).
#
# Command:
#   python -m newton.examples basic_mjcf --mjcf path/to/scene.xml --backend mujoco
#   python -m newton.examples basic_mjcf --mjcf path/to/scene.xml --backend vbd
#
###########################################################################

from __future__ import annotations

import os

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.backend = args.backend
        self.mjcf_path = os.path.abspath(args.mjcf)

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_mjcf(self.mjcf_path, up_axis="Z", enable_self_collisions=True)

        if self.backend == "vbd":
            # VBD requires coloring for rigid bodies and particles.
            builder.color()

        self.model = builder.finalize()

        # IMPORTANT: VBD uses `model.body_q` as the structural rest pose.
        # Ensure the model transforms match `model.joint_q` *before* solver creation.
        if self.backend == "vbd":
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        if self.backend == "mujoco":
            if self.model.joint_count <= 0:
                raise ValueError(
                    "This MJCF import produced a Newton model with no joints. "
                    "Newton's SolverMuJoCo requires at least one joint to convert the model."
                )
            # Some mujoco_warp builds don't support the `implicit` integrator enum.
            # Use a broadly-supported default to keep this example robust.
            self.solver = newton.solvers.SolverMuJoCo(self.model, integrator="implicitfast")
            self.contacts = None
        else:
            self.solver = newton.solvers.SolverVBD(self.model, iterations=10)
            self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Keep transforms consistent for maximal-coordinate solvers.
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
                # VBD uses Newton contacts; update them each substep.
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
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--mjcf", type=str, required=True, help="Path to an MJCF XML scene file.")
        parser.add_argument(
            "--backend",
            type=str,
            default="mujoco",
            choices=["mujoco", "vbd"],
            help="Physics backend to run (MuJoCo or VBD).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

