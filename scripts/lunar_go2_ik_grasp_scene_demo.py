# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, ShapeFlags

DEFAULT_LUNAR_SCENE_DIR = (
    Path(__file__).resolve().parents[1] / "newton" / "examples" / "assets" / "lunar_mujoco_spot_arm_mining_scene"
)
DEFAULT_HEIGHTFIELD = DEFAULT_LUNAR_SCENE_DIR / "lunar_heightfield_normalized.npy"
DEFAULT_ALBEDO = DEFAULT_LUNAR_SCENE_DIR / "lunar_albedo_center_crater.png"
GO2_ASSET_DIR = "unitree_go2"
GO2_USD = "usd/go2.usda"
GO2_CONFIG = "rl_policies/go2.yaml"
GO2_SHELL_COLOR = (0.68, 0.70, 0.76)

MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000

HFIELD_HALF_X = 20.0
HFIELD_HALF_Y = 20.0
HFIELD_ELEVATION_Z = 1.65
HFIELD_Z_OFFSET = -0.88

GO2_DEFAULT_XY = (6.7, 7.1)
GO2_DEFAULT_ROOT_HEIGHT = 0.34

ROCK_LABEL = "go2_ik_rock"
ROCK_DEFAULT_XY = (7.1803, 7.5220)
ROCK_DEFAULT_RADII = (0.06, 0.042, 0.032)
ROCK_DEFAULT_COLOR = (0.28, 0.28, 0.26)
ROCK_DEFAULT_YAW_DEGREES = 41.3
ROCK_DEFAULT_DENSITY = 350.0
ROCK_DEFAULT_MU = 8.0
ROCK_DEFAULT_MU_TORSIONAL = 0.4
ROCK_DEFAULT_MU_ROLLING = 0.25

DEFAULT_PREGRASP_HEIGHT = 0.18
DEFAULT_LIFT_HEIGHT = 0.30


class LunarGo2IkGraspSceneDemo:
    """Prepare a lunar Go2 scene with a gray ellipsoid rock."""

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
        self.go2_xy = self._validate_pair(args.go2_xy, "--go2-xy")
        self.rock_should_rest_on_terrain = args.rock_pos is None
        self.lock_go2_root = not args.free_go2_root
        self.show_grasp_targets = args.show_grasp_targets and not args.hide_grasp_targets

        self.rock_xform, self.grasp_target_tf, self.pregrasp_target_tf, self.lift_target_tf = (
            self._compute_scene_xforms(args)
        )
        self.rock_initial_pos = np.asarray(wp.transform_get_translation(self.rock_xform), dtype=np.float32)
        self.rock_terrain_z = self.terrain_z(float(self.rock_initial_pos[0]), float(self.rock_initial_pos[1]))
        self.rock_initial_q = self._xform_to_joint_q(self.rock_xform)
        self.go2_root_home = self._compute_go2_root_home(args)

        go2_asset_path = newton.utils.download_asset(GO2_ASSET_DIR)
        go2_config = yaml.safe_load((go2_asset_path / GO2_CONFIG).read_text(encoding="utf-8"))
        self.go2_joint_names = tuple(str(name) for name in go2_config["mjw_joint_names"])
        self.go2_joint_home = tuple(float(value) for value in go2_config["mjw_joint_pos"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        go2_root_pos = wp.vec3(*self.go2_root_home[:3])
        go2_root_rot = wp.quat(*self.go2_root_home[3:])
        builder.add_usd(
            str(go2_asset_path / GO2_USD),
            xform=wp.transform(go2_root_pos, go2_root_rot),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")
        self._normalize_go2_display_colors(builder)
        builder.joint_q[:7] = self.go2_root_home
        builder.joint_q[7 : 7 + len(self.go2_joint_home)] = self.go2_joint_home
        for index, stiffness in enumerate(go2_config["mjw_joint_stiffness"]):
            dof_index = 6 + index
            builder.joint_target_ke[dof_index] = float(stiffness)
            builder.joint_target_kd[dof_index] = float(go2_config["mjw_joint_damping"][index])
            builder.joint_armature[dof_index] = float(go2_config["mjw_joint_armature"][index])
            builder.joint_target_mode[dof_index] = int(JointTargetMode.POSITION)

        self._add_lunar_heightfield(builder, args)
        self._add_rock(builder, args)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -1.62))
        if self.model.joint_count <= 0:
            raise ValueError("SolverMuJoCo requires at least one joint in the imported robot model.")

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

        self.root_q_slice, self.root_qd_slice = self._find_root_slices()
        self.go2_q_slice, self.go2_dof_slice = self._find_joint_slices(self.go2_joint_names)
        self.rock_q_slice, self.rock_qd_slice = self._find_joint_slices(
            (f"{ROCK_LABEL}_free_joint",),
            q_width=7,
            qd_width=6,
        )
        self.rock_body_index = self._find_body_index(ROCK_LABEL)
        self.go2_base_body_index = self._find_body_index_in_labels(self.model.body_label, "/go2_description/base")

        self._set_initial_state(self.state_0)
        self._set_initial_state(self.state_1)
        self.control.joint_target_pos[self.go2_dof_slice].assign(self.go2_joint_home)

        self.max_rock_height = float(self.rock_initial_pos[2])
        self.max_rock_drift = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.1, 6.4, 1.45), pitch=-22.0, yaw=132.0)
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
    def _validate_pair(values, name: str) -> tuple[float, float]:
        pair = tuple(float(value) for value in values)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain two values.")
        return pair

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

    def _compute_scene_xforms(
        self, args: argparse.Namespace
    ) -> tuple[wp.transform, wp.transform, wp.transform, wp.transform]:
        if args.rock_pos is not None:
            rock_pos = wp.vec3(float(args.rock_pos[0]), float(args.rock_pos[1]), float(args.rock_pos[2]))
        else:
            x, y = self._validate_pair(args.rock_xy or ROCK_DEFAULT_XY, "--rock-xy")
            rock_pos = wp.vec3(x, y, self.terrain_z(x, y) + float(self.rock_radii[2]))

        rock_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(float(args.rock_yaw)))
        rock_xform = wp.transform(rock_pos, rock_rot)

        grasp_target_tf = wp.transform(
            wp.vec3(rock_pos[0], rock_pos[1], rock_pos[2] + float(self.rock_radii[2])),
            rock_rot,
        )
        pregrasp_target_tf = self._with_world_z_offset(grasp_target_tf, float(args.pregrasp_height))
        lift_target_tf = self._with_world_z_offset(grasp_target_tf, float(args.lift_height))
        return rock_xform, grasp_target_tf, pregrasp_target_tf, lift_target_tf

    def _compute_go2_root_home(self, args: argparse.Namespace) -> np.ndarray:
        rock_pos = wp.transform_get_translation(self.rock_xform)
        if args.go2_yaw is None:
            yaw = math.atan2(float(rock_pos[1]) - self.go2_xy[1], float(rock_pos[0]) - self.go2_xy[0])
        else:
            yaw = math.radians(float(args.go2_yaw))
        go2_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        go2_z = self.terrain_z(self.go2_xy[0], self.go2_xy[1]) + float(args.go2_root_height)
        return np.asarray(
            (self.go2_xy[0], self.go2_xy[1], go2_z, go2_rot[0], go2_rot[1], go2_rot[2], go2_rot[3]),
            dtype=np.float32,
        )

    def _add_lunar_heightfield(self, builder, args: argparse.Namespace) -> None:
        texture_path = Path(args.albedo).resolve()
        texture = str(texture_path) if texture_path.exists() else None
        hfield = newton.Heightfield(
            data=self.heightfield,
            nrow=self.heightfield.shape[0],
            ncol=self.heightfield.shape[1],
            hx=HFIELD_HALF_X,
            hy=HFIELD_HALF_Y,
            min_z=HFIELD_Z_OFFSET,
            max_z=HFIELD_Z_OFFSET + HFIELD_ELEVATION_Z,
            color=wp.vec3(0.58, 0.56, 0.52),
            roughness=1.0,
            texture=texture,
            texture_repeat=(8.0, 8.0),
        )
        terrain_cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.35,
            mu_torsional=0.12,
            mu_rolling=0.02,
            ke=5.0e4,
            kd=5.0e2,
            kf=1.0e3,
        )
        builder.add_shape_heightfield(
            heightfield=hfield,
            cfg=terrain_cfg,
            color=wp.vec3(0.58, 0.56, 0.52),
            label="lunar_terrain",
        )

    def _add_rock(self, builder, args: argparse.Namespace) -> None:
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

    @staticmethod
    def _normalize_go2_display_colors(builder) -> None:
        for index, label in enumerate(builder.shape_label):
            if "/go2_description/" not in label:
                continue
            if "/collisions/" in label and builder.shape_flags[index] & ShapeFlags.VISIBLE:
                builder.shape_color[index] = GO2_SHELL_COLOR

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

    def _find_root_slices(self) -> tuple[slice, slice]:
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        matches = [
            i
            for i, (q_start, qd_start) in enumerate(zip(q_starts, qd_starts, strict=False))
            if q_start == 0 and qd_start == 0
        ]
        if not matches:
            raise ValueError("Could not find the Go2 root free joint.")
        return slice(0, 7), slice(0, 6)

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
        state.joint_q[self.root_q_slice].assign(self.go2_root_home)
        state.joint_q[self.go2_q_slice].assign(self.go2_joint_home)
        state.joint_q[self.rock_q_slice].assign(self.rock_initial_q)
        state.joint_qd.zero_()
        state.body_qd.zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _lock_root(self, state) -> None:
        state.joint_q[self.root_q_slice].assign(self.go2_root_home)
        state.joint_qd[self.root_qd_slice].zero_()

    @staticmethod
    def _rounded_translation(xform: wp.transform) -> np.ndarray:
        return np.round(np.asarray(wp.transform_get_translation(xform), dtype=np.float32), 4)

    def _print_scene_info(self) -> None:
        print(
            "[INFO] Lunar Go2 IK grasp scene: "
            f"go2 root={np.round(self.go2_root_home[:3], 4)}, "
            f"rock center={np.round(self.rock_initial_pos, 4)}, "
            f"radii={np.round(np.asarray(self.rock_radii), 4)}, "
            f"terrain z={self.rock_terrain_z:.4f}, color={np.round(np.asarray(self.rock_color), 3)}"
        )
        print(
            "[INFO] Rock targets: "
            f"pregrasp={self._rounded_translation(self.pregrasp_target_tf)}, "
            f"grasp={self._rounded_translation(self.grasp_target_tf)}, "
            f"lift={self._rounded_translation(self.lift_target_tf)}, "
            f"grasp_targets_visible={self.show_grasp_targets}, "
            f"go2_root_locked={self.lock_go2_root}"
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            if self.lock_go2_root:
                self._lock_root(self.state_0)
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
        if self.show_grasp_targets and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("target_rock_pregrasp", self.pregrasp_target_tf)
            self.viewer.log_gizmo("target_rock_grasp", self.grasp_target_tf)
            self.viewer.log_gizmo("target_rock_lift", self.lift_target_tf)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        rock_pos = body_q[self.rock_body_index][:3]
        go2_root_pos = body_q[self.go2_base_body_index][:3]
        if not np.isfinite(rock_pos).all():
            raise ValueError(f"Rock position became non-finite: {rock_pos}")
        if not np.isfinite(go2_root_pos).all():
            raise ValueError(f"Go2 root position became non-finite: {go2_root_pos}")
        if self.rock_should_rest_on_terrain:
            expected_z = self.rock_terrain_z + float(self.rock_radii[2])
            if abs(float(self.rock_initial_pos[2]) - expected_z) > 1.0e-4:
                raise ValueError("The rock did not start on the lunar heightfield.")

        print(
            "[INFO] Final Go2 scene state: "
            f"go2_root={np.round(go2_root_pos, 4)}, "
            f"rock_pos={np.round(rock_pos, 4)}, "
            f"max_height={self.max_rock_height:.4f}, max_xy_drift={self.max_rock_drift:.4f}"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Prepare a lunar Unitree Go2 scene with a dark-gray ellipsoid rock."
    parser.set_defaults(num_frames=360, viewer="gl")
    parser.add_argument(
        "--heightfield",
        type=str,
        default=str(DEFAULT_HEIGHTFIELD),
        help="Path to the normalized lunar heightfield .npy file used for terrain and rock placement.",
    )
    parser.add_argument(
        "--albedo",
        type=str,
        default=str(DEFAULT_ALBEDO),
        help="Path to the lunar albedo texture used for heightfield rendering.",
    )
    parser.add_argument("--substeps", type=int, default=20, help="Simulation substeps per rendered frame.")
    parser.add_argument("--go2-xy", type=float, nargs=2, default=GO2_DEFAULT_XY, metavar=("X", "Y"), help="Go2 root x/y [m].")
    parser.add_argument(
        "--go2-yaw",
        type=float,
        default=None,
        help="Optional Go2 yaw angle [deg]. Defaults to facing the rock.",
    )
    parser.add_argument(
        "--go2-root-height",
        type=float,
        default=GO2_DEFAULT_ROOT_HEIGHT,
        help="Go2 root height above the local terrain [m].",
    )
    parser.add_argument(
        "--free-go2-root",
        action="store_true",
        help="Do not lock the Go2 root during the scene preview.",
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
    parser.add_argument("--show-grasp-targets", action="store_true", help="Draw rock target gizmos.")
    parser.add_argument("--hide-grasp-targets", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarGo2IkGraspSceneDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
