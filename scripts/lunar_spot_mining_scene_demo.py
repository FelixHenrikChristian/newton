# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import warp as wp

import newton
import newton.examples

DEFAULT_LUNAR_SCENE_DIR = (
    Path(__file__).resolve().parents[1] / "newton" / "examples" / "assets" / "lunar_mujoco_spot_arm_mining_scene"
)
DEFAULT_SCENE = DEFAULT_LUNAR_SCENE_DIR / "lunar_scene_spot_arm_mining.xml"

MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000


class LunarSpotMiningSceneDemo:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)
        self.model = builder.finalize()

        if self.model.joint_count <= 0:
            raise ValueError("SolverMuJoCo requires at least one joint in the imported MJCF model.")

        warnings.filterwarnings("ignore", message=r"Geom .* authored margin=.*")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(12.0, 4.5, 3.2), pitch=-24.0, yaw=130.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
    parser.description = "Run the lunar Spot arm mining MJCF scene with Newton's MuJoCo CPU backend."
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per rendered frame.")
    return parser


def main():
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotMiningSceneDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
