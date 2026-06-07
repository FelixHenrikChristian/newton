# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# 用途：月球 Spot 机械臂椭球体抓取 IK 场景，放置椭球岩石并显示 WR1 的预抓取、抓取和抬升目标位姿。
# 用法：uv run python scripts/lunar_spot_ik_grasp_scene_demo.py

from __future__ import annotations

import argparse
import math
import warnings
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
SPOT_ARM_HOME = (0.0, -3.14, 3.06, 0.0, 0.0, 0.0, GRIPPER_OPEN)
ARM_APPROACH_OPEN = (0.0, -0.5957, 2.271782, 0.000028, 0.094714, 0.0, GRIPPER_OPEN)
ARM_GROUND_OPEN = (0.0, 0.113058, 1.563034, 0.0, -0.045296, 0.0, GRIPPER_OPEN)

ARM_POSES = {
    "approach": ARM_APPROACH_OPEN,
    "ground": ARM_GROUND_OPEN,
    "home": SPOT_ARM_HOME,
}

ROCK_LABEL = "spot_ik_rock"
ROCK_GRASP_HINT_IN_WR1 = wp.vec3(0.22, 0.0, -0.008)
ROCK_DEFAULT_RADII = (0.06, 0.042, 0.032)
ROCK_DEFAULT_COLOR = (0.28, 0.28, 0.26)
ROCK_DEFAULT_YAW_DEGREES = 41.3
ROCK_DEFAULT_DENSITY = 350.0
ROCK_DEFAULT_MU = 8.0
ROCK_DEFAULT_MU_TORSIONAL = 0.4
ROCK_DEFAULT_MU_ROLLING = 0.25

DEFAULT_PREGRASP_HEIGHT = 0.18
DEFAULT_LIFT_HEIGHT = 0.30


class LunarSpotIkGraspSceneDemo:
    """Prepare a lunar Spot arm scene with a larger gray ellipsoid rock."""

    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        if self.sim_substeps <= 0:
            raise ValueError("--substeps must be greater than zero.")
        self.sim_dt = self.frame_dt / self.sim_substeps

        if args.rock_pos is not None and args.rock_xy is not None:
            raise ValueError("Use either --rock-pos or --rock-xy, not both.")

        self.heightfield = self._load_heightfield(args.heightfield)
        self.rock_radii = self._validate_positive_triplet(args.rock_radii, "--rock-radii")
        self.rock_color = self._validate_color(args.rock_color)
        self.rock_should_rest_on_terrain = args.rock_pos is None
        self.show_ik_targets = args.show_ik_targets and not args.hide_ik_targets
        self.arm_q = tuple(
            float(value) for value in (args.arm_q if args.arm_q is not None else ARM_POSES[args.arm_pose])
        )

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)

        self.rock_xform, self.grasp_wr1_tf, self.pregrasp_wr1_tf, self.lift_wr1_tf = self._compute_scene_xforms(
            args, builder
        )
        self.rock_initial_pos = np.asarray(wp.transform_get_translation(self.rock_xform), dtype=np.float32)
        self.rock_terrain_z = self.terrain_z(float(self.rock_initial_pos[0]), float(self.rock_initial_pos[1]))
        self.rock_initial_q = self._xform_to_joint_q(self.rock_xform)

        rock_cfg = newton.ModelBuilder.ShapeConfig(
            density=args.rock_density,
            mu=args.rock_mu,
            mu_torsional=args.rock_mu_torsional,
            mu_rolling=args.rock_mu_rolling,
        )
        rock_body = builder.add_body(xform=self.rock_xform, label=ROCK_LABEL)
        builder.add_shape_ellipsoid(
            rock_body,
            rx=float(self.rock_radii[0]),
            ry=float(self.rock_radii[1]),
            rz=float(self.rock_radii[2]),
            cfg=rock_cfg,
            color=wp.vec3(*self.rock_color),
            label=ROCK_LABEL,
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
        self.rock_q_slice, self.rock_qd_slice = self._find_joint_slices(
            (f"{ROCK_LABEL}_free_joint",),
            q_width=7,
            qd_width=6,
        )
        self.rock_body_index = self._find_body_index(ROCK_LABEL)
        self.wr1_body_index = self._find_body_index_in_labels(self.model.body_label, "/arm_link_wr1")

        self._set_initial_state(self.state_0)
        self._set_initial_state(self.state_1)
        self.control.joint_target_pos[self.leg_dof_slice].assign(SPOT_LEG_HOME)
        self.control.joint_target_pos[self.arm_dof_slice].assign(self.arm_q)

        self.max_rock_height = float(self.rock_initial_pos[2])
        self.max_rock_drift = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.2, 6.6, 1.55), pitch=-22.0, yaw=132.0)
        self._print_scene_info()

    @staticmethod
    def _load_heightfield(path: str | Path) -> np.ndarray:
        heightfield_path = Path(path).resolve()
        if not heightfield_path.exists():
            raise FileNotFoundError(f"Heightfield data file not found: {heightfield_path}")
        heightfield = np.load(heightfield_path)
        if heightfield.ndim != 2:
            raise ValueError(f"Expected a 2D heightfield, got shape {heightfield.shape}.")
        return heightfield.astype(np.float32, copy=False)

    @staticmethod
    def _validate_positive_triplet(values, name: str) -> tuple[float, float, float]:
        triplet = tuple(float(value) for value in values)
        if len(triplet) != 3 or any(value <= 0.0 for value in triplet):
            raise ValueError(f"{name} must contain three positive values.")
        return triplet

    @staticmethod
    def _validate_color(values) -> tuple[float, float, float]:
        color = tuple(float(value) for value in values)
        if len(color) != 3 or any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError("--rock-color must contain three values in [0, 1].")
        return color

    @staticmethod
    def _xform_to_joint_q(xform: wp.transform) -> np.ndarray:
        pos = wp.transform_get_translation(xform)
        rot = wp.transform_get_rotation(xform)
        return np.asarray((pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]), dtype=np.float32)

    @staticmethod
    def _with_world_z_offset(xform: wp.transform, z_offset: float) -> wp.transform:
        pos = wp.transform_get_translation(xform)
        rot = wp.transform_get_rotation(xform)
        return wp.transform(wp.vec3(pos[0], pos[1], pos[2] + z_offset), rot)

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

    def _compute_scene_xforms(
        self, args: argparse.Namespace, builder
    ) -> tuple[wp.transform, wp.transform, wp.transform, wp.transform]:
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
        reference_wr1_tf = wp.transform(*placement_state.body_q.numpy()[wr1_body_index])
        hinted_pos = np.asarray(wp.transform_point(reference_wr1_tf, ROCK_GRASP_HINT_IN_WR1), dtype=np.float32)

        if args.rock_pos is not None:
            rock_pos = wp.vec3(float(args.rock_pos[0]), float(args.rock_pos[1]), float(args.rock_pos[2]))
        else:
            if args.rock_xy is None:
                x = float(hinted_pos[0])
                y = float(hinted_pos[1])
            else:
                x = float(args.rock_xy[0])
                y = float(args.rock_xy[1])
            rock_pos = wp.vec3(x, y, self.terrain_z(x, y) + float(self.rock_radii[2]))

        rock_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(float(args.rock_yaw)))
        rock_xform = wp.transform(rock_pos, rock_rot)

        reference_wr1_pos = wp.transform_get_translation(reference_wr1_tf)
        reference_wr1_rot = wp.transform_get_rotation(reference_wr1_tf)
        grasp_wr1_pos = wp.vec3(
            reference_wr1_pos[0] + rock_pos[0] - float(hinted_pos[0]),
            reference_wr1_pos[1] + rock_pos[1] - float(hinted_pos[1]),
            reference_wr1_pos[2] + rock_pos[2] - float(hinted_pos[2]),
        )
        grasp_wr1_tf = wp.transform(grasp_wr1_pos, reference_wr1_rot)
        pregrasp_wr1_tf = self._with_world_z_offset(grasp_wr1_tf, float(args.pregrasp_height))
        lift_wr1_tf = self._with_world_z_offset(grasp_wr1_tf, float(args.lift_height))
        return rock_xform, grasp_wr1_tf, pregrasp_wr1_tf, lift_wr1_tf

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

    def _set_initial_state(self, state) -> None:
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_q[self.leg_q_slice].assign(SPOT_LEG_HOME)
        state.joint_q[self.arm_q_slice].assign(self.arm_q)
        state.joint_q[self.rock_q_slice].assign(self.rock_initial_q)
        state.joint_qd.zero_()
        state.body_qd.zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _lock_spot_root(self, state) -> None:
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_qd[self.root_qd_slice].zero_()

    @staticmethod
    def _rounded_translation(xform: wp.transform) -> np.ndarray:
        return np.round(np.asarray(wp.transform_get_translation(xform), dtype=np.float32), 4)

    def _print_scene_info(self) -> None:
        print(
            "[INFO] Lunar Spot IK grasp scene: "
            f"rock center={np.round(self.rock_initial_pos, 4)}, "
            f"radii={np.round(np.asarray(self.rock_radii), 4)}, "
            f"terrain z={self.rock_terrain_z:.4f}, color={np.round(np.asarray(self.rock_color), 3)}"
        )
        print(
            "[INFO] WR1 IK targets: "
            f"pregrasp={self._rounded_translation(self.pregrasp_wr1_tf)}, "
            f"grasp={self._rounded_translation(self.grasp_wr1_tf)}, "
            f"lift={self._rounded_translation(self.lift_wr1_tf)}, "
            f"ik_targets_visible={self.show_ik_targets}"
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self._lock_spot_root(self.state_0)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

        rock_pos = self.state_0.body_q.numpy()[self.rock_body_index][:3]
        self.max_rock_height = max(self.max_rock_height, float(rock_pos[2]))
        self.max_rock_drift = max(self.max_rock_drift, float(np.linalg.norm(rock_pos[:2] - self.rock_initial_pos[:2])))

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.show_ik_targets and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("target_wr1_pregrasp", self.pregrasp_wr1_tf)
            self.viewer.log_gizmo("target_wr1_grasp", self.grasp_wr1_tf)
            self.viewer.log_gizmo("target_wr1_lift", self.lift_wr1_tf)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        rock_pos = body_q[self.rock_body_index][:3]
        if not np.isfinite(rock_pos).all():
            raise ValueError(f"Rock position became non-finite: {rock_pos}")
        if self.rock_should_rest_on_terrain:
            expected_z = self.rock_terrain_z + float(self.rock_radii[2])
            if abs(float(self.rock_initial_pos[2]) - expected_z) > 1.0e-4:
                raise ValueError("The rock did not start on the lunar heightfield.")

        print(
            "[INFO] Final IK scene rock state: "
            f"pos={np.round(rock_pos, 4)}, "
            f"max_height={self.max_rock_height:.4f}, max_xy_drift={self.max_rock_drift:.4f}"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Prepare a lunar Spot arm IK grasp scene with a larger dark-gray ellipsoid rock."
    parser.set_defaults(num_frames=360, viewer="gl")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument(
        "--heightfield",
        type=str,
        default=str(DEFAULT_HEIGHTFIELD),
        help="Path to the normalized lunar heightfield .npy file used to place the rock on the terrain.",
    )
    parser.add_argument("--substeps", type=int, default=20, help="Simulation substeps per rendered frame.")
    parser.add_argument(
        "--arm-pose",
        choices=tuple(ARM_POSES),
        default="approach",
        help="Preset arm pose used before IK is added.",
    )
    parser.add_argument(
        "--arm-q",
        type=float,
        nargs=len(ARM_JOINT_NAMES),
        default=None,
        metavar=("SH0", "SH1", "EL0", "EL1", "WR0", "WR1", "F1X"),
        help="Optional explicit arm joint target [rad]. Overrides --arm-pose.",
    )
    parser.add_argument(
        "--rock-radii",
        type=float,
        nargs=3,
        default=ROCK_DEFAULT_RADII,
        metavar=("RX", "RY", "RZ"),
        help="Ellipsoid rock semi-axes [m].",
    )
    parser.add_argument(
        "--rock-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional rock center x/y [m]. The z center is placed on the heightfield.",
    )
    parser.add_argument(
        "--rock-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional full world-space rock center [m]. Overrides heightfield placement.",
    )
    parser.add_argument("--rock-yaw", type=float, default=ROCK_DEFAULT_YAW_DEGREES, help="Rock yaw angle [deg].")
    parser.add_argument(
        "--rock-color",
        type=float,
        nargs=3,
        default=ROCK_DEFAULT_COLOR,
        metavar=("R", "G", "B"),
        help="Rock display color in [0, 1].",
    )
    parser.add_argument("--rock-density", type=float, default=ROCK_DEFAULT_DENSITY, help="Rock density [kg/m^3].")
    parser.add_argument("--rock-mu", type=float, default=ROCK_DEFAULT_MU, help="Rock friction coefficient.")
    parser.add_argument(
        "--rock-mu-torsional",
        type=float,
        default=ROCK_DEFAULT_MU_TORSIONAL,
        help="Rock torsional friction coefficient.",
    )
    parser.add_argument(
        "--rock-mu-rolling",
        type=float,
        default=ROCK_DEFAULT_MU_ROLLING,
        help="Rock rolling friction coefficient.",
    )
    parser.add_argument(
        "--pregrasp-height",
        type=float,
        default=DEFAULT_PREGRASP_HEIGHT,
        help="World-z offset from grasp target to pregrasp target [m].",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=DEFAULT_LIFT_HEIGHT,
        help="World-z offset from grasp target to lift target [m].",
    )
    parser.add_argument("--show-ik-targets", action="store_true", help="Draw WR1 target gizmos.")
    parser.add_argument("--hide-ik-targets", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotIkGraspSceneDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
