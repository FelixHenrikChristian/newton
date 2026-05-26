# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import warp as wp

import newton
import newton.examples


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[2]
    / "mujoco"
    / "model"
    / "lunar"
    / "lunar_mujoco_spot_arm_mining_scene"
    / "lunar_scene_spot_arm_mining.xml"
)

MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000


class LunarSpotMiningDemo:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.backend = args.backend

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)
        if self.backend == "vbd":
            builder.color()

        self.model = builder.finalize()

        if self.backend == "vbd":
            # VBD uses body_q as the structural rest pose, so keep it aligned with
            # the imported MJCF joint configuration before creating the solver.
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
            self.solver = newton.solvers.SolverVBD(self.model, iterations=args.iterations)
            self.contacts = self.model.contacts()
        else:
            if self.model.joint_count <= 0:
                raise ValueError("SolverMuJoCo requires at least one joint in the imported MJCF model.")
            warnings.filterwarnings("ignore", message=r"Geom .* authored margin=.*")
            self.use_mujoco_cpu = args.mujoco_backend == "cpu"
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                integrator="implicitfast",
                use_mujoco_cpu=self.use_mujoco_cpu,
                njmax=MUJOCO_NJMAX,
                nconmax=MUJOCO_NCONMAX,
            )
            self.contacts = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        # Camera roughly faces Spot from the ore cluster side.
        self.viewer.set_camera(pos=wp.vec3(12.0, 4.5, 3.2), pitch=-24.0, yaw=130.0)

        self.graph = None
        if wp.get_device().is_cuda and not getattr(self, "use_mujoco_cpu", False):
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
        pass


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Run the lunar Spot arm mining MJCF scene in Newton."
    parser.set_defaults(
        viewer="usd",
        output_path=str(Path("lunar_spot_arm_mining_newton.usda").resolve()),
        num_frames=30,
    )
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["vbd", "mujoco"],
        help="Physics backend to run.",
    )
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per rendered frame.")
    parser.add_argument("--iterations", type=int, default=10, help="VBD solver iterations per substep.")
    parser.add_argument(
        "--mujoco-backend",
        type=str,
        default="cpu",
        choices=["cpu", "warp"],
        help="MuJoCo implementation used when --backend=mujoco.",
    )
    return parser


def main():
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotMiningDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
