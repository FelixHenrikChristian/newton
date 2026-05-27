# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
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

ORE_NAME = "ore_00_free"
SPOT_ROOT_NAME = "freejoint"

LEG_JOINTS = (
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
)
ARM_JOINTS = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x")
ACTUATED_JOINTS = LEG_JOINTS + ARM_JOINTS

LEG_HOME = np.array([0.0, 1.04, -1.8] * 4, dtype=np.float32)
ARM_HOME = np.array([0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_REACH = np.array([0.0, -1.2605, 2.4417, 0.0, -0.1032, 0.0, 0.0], dtype=np.float32)
ARM_GRASP = np.array([0.0, -1.2605, 2.4417, 0.0, -0.1032, 0.0, -1.2], dtype=np.float32)
ARM_CARRY = np.array([0.0, -2.1, 2.75, 0.0, -0.25, 0.0, -1.2], dtype=np.float32)
ARM_RELEASE = np.array([0.0, -1.3, 2.35, 0.0, -0.1, 0.0, 0.0], dtype=np.float32)

SPOT_HOME_ROOT_POS = np.array([6.7, 7.1, 0.74776551], dtype=np.float32)
SPOT_HOME_ROOT_QUAT_XYZW = np.array([0.0, 0.0, 0.35272872, 0.93572563], dtype=np.float32)
ORE_PICK_POS = np.array([8.79150, 9.10610, 0.46982], dtype=np.float32)
COLLECTION_POS = np.array([16.0, 16.0, 0.48], dtype=np.float32)


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + (b - a) * alpha


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)], dtype=np.float32)


def _joint_label_matches(label: str, name: str) -> bool:
    return label == name or label.endswith("/" + name)


class LunarSpotMiningPickPlaceDemo:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.ore_follow_offset = np.array(args.ore_follow_offset, dtype=np.float32)
        self.release_height = args.release_height

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)
        self.model = builder.finalize()

        self.joint_q_start = self.model.joint_q_start.numpy().astype(np.int32)
        self.joint_qd_start = self.model.joint_qd_start.numpy().astype(np.int32)
        self.root_joint = self._find_joint(SPOT_ROOT_NAME)
        self.ore_joint = self._find_joint(ORE_NAME)
        self.actuated_q_starts = np.array([self.joint_q_start[self._find_joint(name)] for name in ACTUATED_JOINTS])
        self.actuated_dof_starts = np.array([self.joint_qd_start[self._find_joint(name)] for name in ACTUATED_JOINTS])

        self.root_q_start = int(self.joint_q_start[self.root_joint])
        self.ore_q_start = int(self.joint_q_start[self.ore_joint])

        self.root_yaw = math.atan2(ORE_PICK_POS[1] - SPOT_HOME_ROOT_POS[1], ORE_PICK_POS[0] - SPOT_HOME_ROOT_POS[0])
        forward = np.array([math.cos(self.root_yaw), math.sin(self.root_yaw), 0.0], dtype=np.float32)
        self.pick_root_pos = np.array(
            [
                ORE_PICK_POS[0] - 0.8 * forward[0],
                ORE_PICK_POS[1] - 0.8 * forward[1],
                SPOT_HOME_ROOT_POS[2] + 0.03,
            ],
            dtype=np.float32,
        )
        self.drop_root_pos = np.array(
            [
                COLLECTION_POS[0] - 0.85 * forward[0],
                COLLECTION_POS[1] - 0.85 * forward[1],
                SPOT_HOME_ROOT_POS[2] + 0.03,
            ],
            dtype=np.float32,
        )

        self.phase_times = np.cumsum(
            np.array([1.2, 1.2, 0.45, 0.8, 2.2, 0.8, 0.45, 1.0], dtype=np.float32)
        )
        self.last_root_pos = SPOT_HOME_ROOT_POS.copy()
        self.last_root_quat = SPOT_HOME_ROOT_QUAT_XYZW.copy()
        self.last_arm_targets = ARM_HOME.copy()
        self.ore_attached = False
        self.ore_released = False

        self._apply_initial_pose()

        warnings.filterwarnings("ignore", message=r"Geom .* authored margin=.*")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            integrator="implicitfast",
            use_mujoco_cpu=True,
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
        )
        self.contacts = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._set_joint_targets(np.concatenate((LEG_HOME, ARM_HOME)))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.joint_q.assign(self.state_0.joint_q.numpy())
        self.state_1.joint_qd.assign(self.state_0.joint_qd.numpy())
        self.state_1.body_q.assign(self.state_0.body_q.numpy())
        self.state_1.body_qd.assign(self.state_0.body_qd.numpy())

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(10.8, 5.8, 3.0), pitch=-22.0, yaw=132.0)

    def _find_joint(self, name: str) -> int:
        for i, label in enumerate(self.model.joint_label):
            if _joint_label_matches(label, name):
                return i
        raise ValueError(f"Could not find joint '{name}' in imported MJCF model.")

    def _apply_initial_pose(self):
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()
        self._write_spot_root(joint_q, SPOT_HOME_ROOT_POS, SPOT_HOME_ROOT_QUAT_XYZW)
        joint_q[self.actuated_q_starts] = np.concatenate((LEG_HOME, ARM_HOME))
        joint_qd[:] = 0.0
        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

    def _write_spot_root(self, joint_q: np.ndarray, root_pos: np.ndarray, root_quat_xyzw: np.ndarray):
        joint_q[self.root_q_start : self.root_q_start + 3] = root_pos
        joint_q[self.root_q_start + 3 : self.root_q_start + 7] = root_quat_xyzw

    def _write_ore_pose(self, joint_q: np.ndarray, ore_pos: np.ndarray):
        joint_q[self.ore_q_start : self.ore_q_start + 3] = ore_pos

    def _set_joint_targets(self, targets: np.ndarray):
        joint_target_pos = self.control.joint_target_pos.numpy()
        joint_target_pos[self.actuated_dof_starts] = targets
        self.control.joint_target_pos.assign(joint_target_pos)

    def _script_targets(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        t0, t1, t2, t3, t4, t5, t6, _ = self.phase_times
        yaw_quat = _yaw_quat_xyzw(self.root_yaw)

        if t < t0:
            alpha = _smoothstep(t / t0)
            return _lerp(SPOT_HOME_ROOT_POS, self.pick_root_pos, alpha), yaw_quat, ARM_HOME, False, False
        if t < t1:
            alpha = _smoothstep((t - t0) / (t1 - t0))
            return self.pick_root_pos, yaw_quat, _lerp(ARM_HOME, ARM_REACH, alpha), False, False
        if t < t2:
            alpha = _smoothstep((t - t1) / (t2 - t1))
            return self.pick_root_pos, yaw_quat, _lerp(ARM_REACH, ARM_GRASP, alpha), True, False
        if t < t3:
            alpha = _smoothstep((t - t2) / (t3 - t2))
            return self.pick_root_pos, yaw_quat, _lerp(ARM_GRASP, ARM_CARRY, alpha), True, False
        if t < t4:
            alpha = _smoothstep((t - t3) / (t4 - t3))
            return _lerp(self.pick_root_pos, self.drop_root_pos, alpha), yaw_quat, ARM_CARRY, True, False
        if t < t5:
            alpha = _smoothstep((t - t4) / (t5 - t4))
            return self.drop_root_pos, yaw_quat, _lerp(ARM_CARRY, ARM_RELEASE, alpha), True, False
        if t < t6:
            alpha = _smoothstep((t - t5) / (t6 - t5))
            return self.drop_root_pos, yaw_quat, _lerp(ARM_RELEASE, ARM_RELEASE, alpha), False, True
        alpha = _smoothstep((t - t6) / (self.phase_times[-1] - t6))
        return self.drop_root_pos, yaw_quat, _lerp(ARM_RELEASE, ARM_HOME, alpha), False, True

    def _apply_script_to_state(self, state):
        root_pos, root_quat, arm_targets, should_attach, should_release = self._script_targets(self.sim_time)
        targets = np.concatenate((LEG_HOME, arm_targets.astype(np.float32)))
        self._set_joint_targets(targets)

        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        self._write_spot_root(joint_q, root_pos, root_quat)
        joint_q[self.actuated_q_starts] = targets
        joint_qd[self.joint_qd_start[self.root_joint] : self.joint_qd_start[self.root_joint] + 6] = 0.0
        joint_qd[self.joint_qd_start[self.ore_joint] : self.joint_qd_start[self.ore_joint] + 6] = 0.0
        joint_qd[self.actuated_dof_starts] = 0.0

        if should_attach:
            self.ore_attached = True
        if should_release:
            self.ore_attached = False
            self.ore_released = True

        if self.ore_attached:
            self._write_ore_pose(joint_q, root_pos + self.ore_follow_offset)
        elif self.ore_released:
            self._write_ore_pose(joint_q, np.array([COLLECTION_POS[0], COLLECTION_POS[1], self.release_height]))

        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

        self.last_root_pos = root_pos
        self.last_root_quat = root_quat
        self.last_arm_targets = arm_targets

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_script_to_state(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        ore_pos = self.state_0.joint_q.numpy()[self.ore_q_start : self.ore_q_start + 3]
        distance_xy = np.linalg.norm(ore_pos[:2] - COLLECTION_POS[:2])
        if distance_xy > 0.35:
            raise AssertionError(f"ore_00 was not placed in the collection zone: xy error={distance_xy:.3f} m")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Scripted lunar Spot arm mining pick-and-place demo using Newton's MuJoCo backend."
    parser.set_defaults(
        viewer="usd",
        output_path=str(Path("lunar_spot_arm_mining_pick_place.usda").resolve()),
        num_frames=600,
    )
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument("--substeps", type=int, default=4, help="Simulation substeps per rendered frame.")
    parser.add_argument(
        "--ore-follow-offset",
        type=float,
        nargs=3,
        default=(0.62, 0.58, 0.25),
        metavar=("X", "Y", "Z"),
        help="World-space offset from the scripted Spot root used while ore_00 is attached.",
    )
    parser.add_argument(
        "--release-height",
        type=float,
        default=0.62,
        help="World-space z position used when releasing ore_00 above the collection zone.",
    )
    return parser


def main():
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotMiningPickPlaceDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
