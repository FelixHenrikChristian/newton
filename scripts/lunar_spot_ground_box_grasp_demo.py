# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_LUNAR_SCENE_DIR = (
    Path(__file__).resolve().parents[1] / "newton" / "examples" / "assets" / "lunar_mujoco_spot_arm_mining_scene"
)
DEFAULT_SCENE = DEFAULT_LUNAR_SCENE_DIR / "lunar_scene_spot_arm_mining.xml"
DEFAULT_HEIGHTFIELD = DEFAULT_SCENE.parent / "lunar_heightfield_normalized.npy"

MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000

HFIELD_HALF_X = 20.0
HFIELD_HALF_Y = 20.0
HFIELD_ELEVATION_Z = 1.65
HFIELD_Z_OFFSET = -0.88

LEG_JOINT_NAMES = (
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
ARM_JOINT_NAMES = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x")

# Values from the spot_mining_home XML keyframe. MuJoCo authors free-joint
# quaternions as WXYZ; Newton stores imported free-joint quaternions as XYZW.
SPOT_ROOT_HOME = (6.7, 7.1, 0.74776551, 0.0, 0.0, 0.35272872, 0.93572563)
SPOT_LEG_HOME = (0.0, 1.04, -1.8) * 4

GRIPPER_OPEN = -1.4
GRIPPER_CLOSED = -0.36

ARM_APPROACH_OPEN = (0.0, -0.5957, 2.271782, 0.000028, 0.094714, 0.0, GRIPPER_OPEN)
ARM_GROUND_OPEN = (0.0, 0.113058, 1.563034, 0.0, -0.045296, 0.0, GRIPPER_OPEN)
ARM_GROUND_CLOSED = (*ARM_GROUND_OPEN[:-1], GRIPPER_CLOSED)
ARM_RAISE = (0.0, -0.123659, 1.738708, 0.0, -0.044253, 0.0, GRIPPER_CLOSED)
ARM_LIFT = (0.0, -0.5957, 2.071782, 0.000028, 0.094714, 0.0, GRIPPER_CLOSED)
ARM_LIFT_OPEN = (*ARM_LIFT[:-1], GRIPPER_OPEN)

# The arm targets keep the wrist orientation and xy fixed while moving in z.
# Place a smaller box on the terrain directly between Spot's stock fixed and
# moving jaws after the arm reaches the forward ground target.
# BOX_GRASP_HINT_IN_WR1 = wp.vec3(0.22, 0.0, -0.024)
BOX_GRASP_HINT_IN_WR1 = wp.vec3(0.22, 0.0, -0.008)
BOX_HX = 0.025
BOX_HY = 0.02
BOX_HZ = 0.025
BOX_MU = 25.0
BOX_YAW_DEGREES = 41.3


@dataclass(frozen=True)
class ArmStage:
    name: str
    duration: float
    target: tuple[float, ...]


BASE_ARM_STAGES = (
    ArmStage("hover above forward box", 0.8, ARM_APPROACH_OPEN),
    ArmStage("lower vertically around forward box", 2.0, ARM_GROUND_OPEN),
    ArmStage("close stock gripper", 1.0, ARM_GROUND_CLOSED),
    ArmStage("secure contact grasp", 0.8, ARM_GROUND_CLOSED),
    ArmStage("raise vertically", 2.0, ARM_RAISE),
    ArmStage("lift vertically", 2.0, ARM_LIFT),
    ArmStage("hold", 1.0, ARM_LIFT),
)
PROOF_RELEASE_STAGES = (
    ArmStage("open release proof", 1.0, ARM_LIFT_OPEN),
    ArmStage("release settle", 1.5, ARM_LIFT_OPEN),
)


class LunarSpotForwardBoxGraspDemo:
    """Spot lowers its stock gripper to lift a small box in front of it.

    The box is a free dynamic body. After model initialization its pose is
    changed only by contact dynamics. The script adds no helper pads, guides,
    scoops, welds, pose overrides, or external forces.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        if self.sim_substeps <= 0:
            raise ValueError("--substeps must be greater than zero.")
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.heightfield = self._load_heightfield(args.heightfield)
        self.proof_release = args.proof_release
        self.arm_stages = BASE_ARM_STAGES + (PROOF_RELEASE_STAGES if self.proof_release else ())

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)

        box_xform = self._compute_box_ground_xform(builder)
        self.box_initial_pos = np.asarray(wp.transform_get_translation(box_xform), dtype=np.float32)
        self.box_terrain_z = self.terrain_z(float(self.box_initial_pos[0]), float(self.box_initial_pos[1]))

        box_cfg = newton.ModelBuilder.ShapeConfig(
            density=150.0,
            mu=BOX_MU,
            mu_torsional=0.2,
            mu_rolling=0.1,
        )
        box_body = builder.add_body(xform=box_xform, label="spot_ground_box")
        builder.add_shape_box(
            box_body,
            hx=BOX_HX,
            hy=BOX_HY,
            hz=BOX_HZ,
            cfg=box_cfg,
            color=wp.vec3(0.76, 0.44, 0.18),
            label="spot_ground_box",
        )

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

        self.root_q_slice, self.root_qd_slice = self._find_joint_slices(("freejoint",), q_width=7, qd_width=6)
        self.leg_q_slice, self.leg_dof_slice = self._find_joint_slices(LEG_JOINT_NAMES)
        self.arm_q_slice, self.arm_dof_slice = self._find_joint_slices(ARM_JOINT_NAMES)
        self.box_body_index = self._find_body_index("spot_ground_box")
        self.wr1_body_index = self._find_body_index_in_labels(self.model.body_label, "/arm_link_wr1")

        self._set_initial_spot_state(self.state_0)
        self.state_1.assign(self.state_0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        self.control.joint_target_pos[self.leg_dof_slice].assign(SPOT_LEG_HOME)
        self.control.joint_target_pos[self.arm_dof_slice].assign(ARM_APPROACH_OPEN)

        self._stage_index = -1
        self._stage_start_time = 0.0
        self._stage_start_target = np.asarray(ARM_APPROACH_OPEN, dtype=np.float32)
        self.max_box_height = float(self.box_initial_pos[2])
        self.release_start_height: float | None = None
        self.release_start_local: np.ndarray | None = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, 6.4, 1.5), pitch=-22.0, yaw=132.0)

    @staticmethod
    def _load_heightfield(path: str | Path) -> np.ndarray:
        heightfield_path = Path(path).resolve()
        if not heightfield_path.exists():
            raise FileNotFoundError(f"Heightfield data file not found: {heightfield_path}")
        heightfield = np.load(heightfield_path)
        if heightfield.ndim != 2:
            raise ValueError(f"Expected a 2D heightfield, got shape {heightfield.shape}.")
        return heightfield.astype(np.float32, copy=False)

    def terrain_z(self, x: float, y: float) -> float:
        u = np.clip(
            (x + HFIELD_HALF_X) / (2.0 * HFIELD_HALF_X) * (self.heightfield.shape[1] - 1),
            0.0,
            self.heightfield.shape[1] - 1,
        )
        v = np.clip(
            (y + HFIELD_HALF_Y) / (2.0 * HFIELD_HALF_Y) * (self.heightfield.shape[0] - 1),
            0.0,
            self.heightfield.shape[0] - 1,
        )
        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1 = min(x0 + 1, self.heightfield.shape[1] - 1)
        y1 = min(y0 + 1, self.heightfield.shape[0] - 1)
        fu = float(u - x0)
        fv = float(v - y0)
        normalized_height = (
            (1.0 - fu) * (1.0 - fv) * self.heightfield[y0, x0]
            + fu * (1.0 - fv) * self.heightfield[y0, x1]
            + (1.0 - fu) * fv * self.heightfield[y1, x0]
            + fu * fv * self.heightfield[y1, x1]
        )
        return HFIELD_Z_OFFSET + HFIELD_ELEVATION_Z * float(normalized_height)

    @staticmethod
    def _find_joint_slices_in_model(
        model,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        joint_indices = []

        for name in names:
            matches = [i for i, label in enumerate(model.joint_label) if label == name or label.endswith(f"/{name}")]
            if len(matches) != 1:
                raise ValueError(f"Expected one imported joint named '{name}', found {len(matches)}.")
            joint_indices.append(matches[0])

        q_indices = [int(q_starts[index]) for index in joint_indices]
        qd_indices = [int(qd_starts[index]) for index in joint_indices]
        if q_indices != list(range(q_indices[0], q_indices[0] + len(names))) and q_width is None:
            raise ValueError(f"Expected contiguous joint coordinates for: {', '.join(names)}")
        if qd_indices != list(range(qd_indices[0], qd_indices[0] + len(names))) and qd_width is None:
            raise ValueError(f"Expected contiguous joint DoFs for: {', '.join(names)}")

        return (
            slice(q_indices[0], q_indices[0] + (q_width or len(names))),
            slice(qd_indices[0], qd_indices[0] + (qd_width or len(names))),
        )

    def _compute_box_ground_xform(self, builder) -> wp.transform:
        placement_model = builder.finalize()
        placement_state = placement_model.state()

        root_q_slice, _ = self._find_joint_slices_in_model(placement_model, ("freejoint",), q_width=7, qd_width=6)
        leg_q_slice, _ = self._find_joint_slices_in_model(placement_model, LEG_JOINT_NAMES)
        arm_q_slice, _ = self._find_joint_slices_in_model(placement_model, ARM_JOINT_NAMES)

        placement_state.joint_q[root_q_slice].assign(SPOT_ROOT_HOME)
        placement_state.joint_q[leg_q_slice].assign(SPOT_LEG_HOME)
        placement_state.joint_q[arm_q_slice].assign(ARM_GROUND_OPEN)
        newton.eval_fk(placement_model, placement_state.joint_q, placement_model.joint_qd, placement_state)

        wr1_body_index = self._find_body_index_in_labels(placement_model.body_label, "/arm_link_wr1")
        wr1_xform = wp.transform(*placement_state.body_q.numpy()[wr1_body_index])
        hinted_pos = np.asarray(wp.transform_point(wr1_xform, BOX_GRASP_HINT_IN_WR1), dtype=np.float32)
        box_pos = wp.vec3(
            float(hinted_pos[0]),
            float(hinted_pos[1]),
            self.terrain_z(float(hinted_pos[0]), float(hinted_pos[1])) + BOX_HZ,
        )
        box_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(BOX_YAW_DEGREES))
        return wp.transform(box_pos, box_rot)

    def _find_joint_slices(
        self,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        return self._find_joint_slices_in_model(self.model, names, q_width=q_width, qd_width=qd_width)

    def _find_body_index(self, label: str) -> int:
        matches = [i for i, body_label in enumerate(self.model.body_label) if body_label == label]
        if len(matches) != 1:
            raise ValueError(f"Expected one body labeled '{label}', found {len(matches)}.")
        return matches[0]

    @staticmethod
    def _find_body_index_in_labels(labels, suffix: str) -> int:
        matches = [i for i, label in enumerate(labels) if label.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one body ending in '{suffix}', found {len(matches)}.")
        return matches[0]

    def _set_initial_spot_state(self, state):
        self._lock_spot_root(state)
        state.joint_q[self.leg_q_slice].assign(SPOT_LEG_HOME)
        state.joint_q[self.arm_q_slice].assign(ARM_APPROACH_OPEN)
        state.joint_qd[self.leg_dof_slice].zero_()
        state.joint_qd[self.arm_dof_slice].zero_()

    def _lock_spot_root(self, state):
        # This isolates the arm grasp experiment from locomotion balance.
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_qd[self.root_qd_slice].zero_()

    def _set_arm_target(self):
        stage_index = 0
        stage_start_time = 0.0
        for index, stage in enumerate(self.arm_stages):
            if self.sim_time < stage_start_time + stage.duration:
                stage_index = index
                break
            stage_start_time += stage.duration
        else:
            stage_index = len(self.arm_stages) - 1
            stage_start_time -= self.arm_stages[-1].duration

        stage = self.arm_stages[stage_index]
        if stage_index != self._stage_index:
            self._stage_index = stage_index
            self._stage_start_time = stage_start_time
            self._stage_start_target = (
                np.asarray(ARM_APPROACH_OPEN, dtype=np.float32)
                if stage_index == 0
                else np.asarray(self.arm_stages[stage_index - 1].target, dtype=np.float32)
            )
            body_q = self.state_0.body_q.numpy()
            wr1_xform = wp.transform(*body_q[self.wr1_body_index])
            box_pos = wp.vec3(*body_q[self.box_body_index][:3])
            box_pos_in_wr1 = wp.transform_point(wp.transform_inverse(wr1_xform), box_pos)
            if stage.name == "open release proof":
                self.release_start_height = float(body_q[self.box_body_index][2])
                self.release_start_local = np.asarray(box_pos_in_wr1, dtype=np.float32)
            print(
                f"[INFO] Spot forward box grasp stage: {stage.name}; "
                f"box={np.round(body_q[self.box_body_index][:3], 3)}, "
                f"box in wr1={np.round(np.asarray(box_pos_in_wr1), 3)}"
            )

        alpha = min(1.0, max(0.0, (self.sim_time - self._stage_start_time) / stage.duration))
        target = (1.0 - alpha) * self._stage_start_target + alpha * np.asarray(stage.target, dtype=np.float32)
        self.control.joint_target_pos[self.arm_dof_slice].assign(target)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._lock_spot_root(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._set_arm_target()
        self.simulate()
        self.sim_time += self.frame_dt

        box_height = float(self.state_0.body_q.numpy()[self.box_body_index][2])
        self.max_box_height = max(self.max_box_height, box_height)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        box_pos = body_q[self.box_body_index][:3]
        box_height = float(box_pos[2])
        arm_q = self.state_0.joint_q.numpy()[self.arm_q_slice]
        wr1_xform = wp.transform(*body_q[self.wr1_body_index])
        box_pos_in_wr1 = wp.transform_point(wp.transform_inverse(wr1_xform), wp.vec3(*box_pos))
        lift_height = self.max_box_height - float(self.box_initial_pos[2])
        final_lift_height = box_height - float(self.box_initial_pos[2])
        box_local = np.asarray(box_pos_in_wr1)
        print(
            f"[INFO] Box terrain z={self.box_terrain_z:.3f}, initial={np.round(self.box_initial_pos, 3)}, "
            f"max z={self.max_box_height:.3f}, final={np.round(box_pos, 3)}, "
            f"final in wr1={np.round(box_local, 3)}, arm q={np.round(arm_q, 3)}"
        )
        if abs(float(self.box_initial_pos[2]) - (self.box_terrain_z + BOX_HZ)) > 1.0e-4:
            raise ValueError("The box did not start on the terrain heightfield.")
        if lift_height < 0.08:
            raise ValueError(f"Box was not lifted high enough: lift height={lift_height:.3f}m")
        in_gripper = 0.10 <= box_local[0] <= 0.30 and abs(box_local[1]) <= 0.10 and -0.14 <= box_local[2] <= 0.12
        if self.proof_release:
            if self.release_start_height is None or self.release_start_local is None:
                raise ValueError("Release proof stage did not run.")
            release_lift_height = self.release_start_height - float(self.box_initial_pos[2])
            if release_lift_height < 0.08:
                raise ValueError(f"Box was not held high enough before release: lift height={release_lift_height:.3f}m")
            dropped_after_open = self.release_start_height - box_height > 0.04
            still_near_release_local = np.linalg.norm(box_local - self.release_start_local) < 0.05
            if not dropped_after_open and still_near_release_local:
                raise ValueError("Box stayed on the gripper after opening, which looks like an attachment.")
        else:
            if final_lift_height < 0.08:
                raise ValueError(f"Box was not held high enough: final lift height={final_lift_height:.3f}m")
            if not in_gripper:
                raise ValueError(f"Box is no longer in the gripper at the end: box in wr1={box_local}")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Lower Spot's stock gripper to grasp a small box placed in front of it on the lunar heightfield."
    parser.set_defaults(num_frames=780)
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument(
        "--heightfield",
        type=str,
        default=str(DEFAULT_HEIGHTFIELD),
        help="Path to the normalized lunar heightfield .npy file used to place the box on the terrain.",
    )
    parser.add_argument("--substeps", type=int, default=30, help="Simulation substeps per rendered frame.")
    parser.add_argument(
        "--proof-release",
        action="store_true",
        help="After lifting, open the stock gripper and require the box to separate.",
    )
    return parser


def main():
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotForwardBoxGraspDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
