from __future__ import annotations

import argparse
import io
import math
import os
import pathlib
import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import spot_pick_place_demo_robust_layout as layout
import stable_baselines3.common.base_class as sb3_base_class
import stable_baselines3.common.save_util as sb3_save_util
import torch as th
import warp as wp
from gymnasium import spaces
from PIL import Image
from spot_pick_place_mesh import grasp_axis_surface_points, load_obj_triangles, scale_to_ellipsoid_bounds
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import newton
import newton.examples
import newton.ik as ik

SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_PATH = SCRIPT_DIR / "spot_scene.xml"
MODEL_PATH = SCRIPT_DIR / "runs" / "spot_go2_style_lunar_strategy_v3_gpu1_30m" / "best_eval" / "best_model.zip"
VECNORMALIZE_PATH = MODEL_PATH.parent / "best_vecnormalize.pkl"
MESH_STONE_ASSET_DIR = SCRIPT_DIR / "lunar_mujoco_spot_arm_mining_scene"
MESH_STONE_ASSET_FILENAMES = ("rock1_centered_unit.obj", "rock2_centered_unit.obj")
MESH_STONE_ASSET_PATHS = tuple(MESH_STONE_ASSET_DIR / filename for filename in MESH_STONE_ASSET_FILENAMES)

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

OBS_DIM = 58
ACT_DIM = 12
ARM_ACTUATOR_COUNT = 7

A_POINT = np.array([5.0, -6.0], dtype=np.float64)
B_POINT = np.array([9.0, -3.5], dtype=np.float64)
C_POINT = 2.0 * B_POINT - A_POINT
A_YAW = 0.0
RESET_BASE_HEIGHT = 1.80

STONE_LABEL = "pickup_stone"
STONE_FORWARD_OFFSET = 0.55
STONE_RELEASE_FORWARD_OFFSET = 0.75
STONE_HEADING_OFFSET = 0.0
STONE_SIZE = np.array([0.06, 0.045, 0.035], dtype=np.float64)
STONE_CLEARANCE = -0.035
STONE_ALIGN_RADIUS = 1.2

ARM_BASE_OFFSET = np.array([0.292, 0.0, 0.188], dtype=np.float32)
ARM_EE_BODY = "arm_link_wr1"
GRIPPER_TRACK_OFFSET = wp.vec3(0.207, 0.0, 0.038)
WR1_DOWN_AXIS_LENGTH = 0.12
GRIPPER_OPEN = -1.4
GRIPPER_CLOSED = -0.85

ARM_PREGRASP_HEIGHT = 0.30
ARM_GRASP_TARGET_OFFSET = np.array([0.01, 0.05, 0.0], dtype=np.float32)
ARM_GRASP_CLEARANCE = 0.06
ARM_PREGRASP_SECONDS = 2.0
ARM_DESCENT_SECONDS = 2.0
GRIPPER_CLOSE_SECONDS = 1.0
ARM_GRASP_SECONDS = ARM_PREGRASP_SECONDS + ARM_DESCENT_SECONDS + GRIPPER_CLOSE_SECONDS
ARM_RETRACT_SECONDS = 2.5
ARM_PLACE_SECONDS = ARM_PREGRASP_SECONDS + ARM_DESCENT_SECONDS
ARM_RELEASE_CLEARANCE = 0.33
GRIPPER_RELEASE_SECONDS = 1.5
IK_ITERATIONS = 12
IK_STEP_SIZE = 0.8
IK_LAMBDA_INITIAL = 0.1
IK_DOWN_AXIS_WEIGHT = 0.25
IK_LIMIT_WEIGHT = 1.0
MAX_ARM_JOINT_STEP = 0.05

ARRIVAL_RADIUS = 0.15
B_HANDOFF_RADIUS = 0.28
ARRIVAL_YAW_TOLERANCE = 0.35
C_ARRIVAL_RADIUS = 0.45
MAX_SECONDS = 60.0
FORWARD_SPEED = 0.45
MIN_FORWARD_SPEED = 0.15
MAX_YAW_RATE = 0.3
YAW_GAIN = 1.4
C_ROVER_LABEL = "c_lunar_rover"
C_ROVER_CARGO_INNER_HALF_EXTENTS = np.array([0.30, 0.24], dtype=np.float64)
C_ROVER_CARGO_WALL_THICKNESS = 0.045
C_ROVER_CARGO_WALL_HEIGHT = 0.26
C_ROVER_CHASSIS_HALF_EXTENTS = np.array([0.62, 0.39, 0.06], dtype=np.float64)
C_ROVER_CHASSIS_CENTER_OFFSET = np.array([0.24, 0.0], dtype=np.float64)
C_ROVER_WHEEL_RADIUS = 0.135
C_ROVER_WHEEL_HALF_WIDTH = 0.045
C_ROVER_WHEEL_X_OFFSETS = (-0.17, 0.24, 0.65)
C_ROVER_WHEEL_Y_OFFSET = 0.43


@dataclass(frozen=True)
class StoneSpec:
    """Dynamic pickup stone configuration."""

    label: str
    pos: np.ndarray
    quat: np.ndarray
    size: np.ndarray


NOMINAL_LEG_CTRL = np.array((0.0, -0.22, 0.55) * 4, dtype=np.float32)
MANIPULATION_LEG_CTRL = np.array(
    (0.18, -0.10, 0.15, -0.18, -0.10, 0.15, 0.18, -0.10, 0.15, -0.18, -0.10, 0.15),
    dtype=np.float32,
)
ACTION_SCALE = np.array((0.18, 0.30, 0.48) * 4, dtype=np.float32)
ACTUATOR_GAIN_SCALE = 1.8
GAIT_PERIOD = 0.56
GAIT_CONTACT_SHARPNESS = 3.0
B_ALIGN_SECONDS = 1.0
B_GRASP_ROOT_Q = np.array(
    [9.076134, -3.625636, 1.593325, -0.032121, 0.035750, 0.373366, 0.926438],
    dtype=np.float32,
)
B_GRASP_LEG_Q = np.array(
    [
        0.199358,
        0.920764,
        -1.580373,
        -0.240435,
        1.055332,
        -1.414194,
        0.260501,
        1.234142,
        -1.347853,
        -0.230513,
        1.164893,
        -1.437398,
    ],
    dtype=np.float32,
)
CONTROL_DECIMATION = 10
FPS = 50
RESET_SEED = 1
MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000


def patch_sb3_zip_loader() -> None:
    """Work around PyTorch versions that cannot load tensors from ZipExtFile."""

    def load_from_zip_file(
        load_path: str | pathlib.Path | io.BufferedIOBase,
        load_data: bool = True,
        custom_objects: dict[str, Any] | None = None,
        device: th.device | str = "auto",
        verbose: int = 0,
        print_system_info: bool = False,
    ):
        file = sb3_save_util.open_path(load_path, "r", verbose=verbose, suffix="zip")
        device = get_device(device=device)

        try:
            with zipfile.ZipFile(file) as archive:
                namelist = archive.namelist()
                data = None
                pytorch_variables = None
                params = {}

                if print_system_info and "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())

                if "data" in namelist and load_data:
                    data = sb3_save_util.json_to_data(archive.read("data").decode(), custom_objects=custom_objects)

                for file_path in namelist:
                    if os.path.splitext(file_path)[1] != ".pth":
                        continue
                    th_object = th.load(io.BytesIO(archive.read(file_path)), map_location=device, weights_only=True)
                    if file_path in ("pytorch_variables.pth", "tensors.pth"):
                        pytorch_variables = th_object
                    else:
                        params[os.path.splitext(file_path)[0]] = th_object
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from exc
        finally:
            if isinstance(load_path, (str, pathlib.Path)):
                file.close()

        return data, params, pytorch_variables

    sb3_save_util.load_from_zip_file = load_from_zip_file
    sb3_base_class.load_from_zip_file = load_from_zip_file


class _VecNormalizeEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_locomotion_observation(
    *,
    base_angular: np.ndarray,
    projected_gravity: np.ndarray,
    command: np.ndarray,
    gait_observation: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    foot_contacts: np.ndarray,
    base_linear: np.ndarray,
) -> np.ndarray:
    """Build the lunar locomotion policy's 58-D observation."""
    return np.concatenate(
        [
            base_angular * 0.25,
            projected_gravity,
            command * np.array((2.0, 2.0, 0.25), dtype=np.float32),
            gait_observation,
            joint_pos,
            joint_vel * 0.05,
            last_action,
            foot_contacts,
            base_linear * 0.5,
        ]
    ).astype(np.float32)


def _smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _format_vec(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> tuple[float, float, float, float]:
    return float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])


def _yaw_to_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    q = _quat_normalize(quat)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = lhs
    bx, by, bz, bw = rhs
    return _quat_normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float32,
        )
    )


def _approach_yaw() -> float:
    direction = B_POINT - A_POINT
    if float(np.linalg.norm(direction)) < 1e-6:
        return A_YAW
    return float(math.atan2(direction[1], direction[0]))


def _stone_xy() -> np.ndarray:
    yaw = _approach_yaw() + STONE_HEADING_OFFSET
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    return B_POINT + forward * STONE_FORWARD_OFFSET


def _c_release_box_center() -> np.ndarray:
    yaw = _approach_yaw()
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    return C_POINT + forward * max(STONE_RELEASE_FORWARD_OFFSET - C_ARRIVAL_RADIUS, 0.0)


def _scene_keyframe() -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(SCENE_PATH).getroot()
    key = root.find(".//key[@name='stand']")
    if key is None:
        raise ValueError("Expected keyframe named 'stand' in spot_scene.xml.")
    return (
        np.fromstring(key.attrib["qpos"], sep=" ", dtype=np.float64),
        np.fromstring(key.attrib["ctrl"], sep=" ", dtype=np.float64),
    )


def _terrain_height_at(xy: np.ndarray) -> float:
    root = ET.parse(SCENE_PATH).getroot()
    hfield = root.find(".//hfield")
    terrain = root.find(".//geom[@name='lunar_terrain']")
    if hfield is None or terrain is None:
        return 0.0

    size = np.fromstring(hfield.attrib["size"], sep=" ", dtype=np.float64)
    terrain_pos = np.fromstring(terrain.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    image = np.asarray(Image.open(SCRIPT_DIR / hfield.attrib["file"]).convert("L"), dtype=np.float64) / 255.0

    rows, cols = image.shape
    u = np.clip((xy[0] / size[0] + 1.0) * 0.5 * (cols - 1), 0.0, cols - 1)
    v = np.clip((1.0 - (xy[1] / size[1] + 1.0) * 0.5) * (rows - 1), 0.0, rows - 1)
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    tx, ty = u - x0, v - y0
    height_value = (
        (1.0 - tx) * (1.0 - ty) * image[y0, x0]
        + tx * (1.0 - ty) * image[y0, x1]
        + (1.0 - tx) * ty * image[y1, x0]
        + tx * ty * image[y1, x1]
    )
    return float(terrain_pos[2] + size[2] * height_value)


def _stone_position() -> np.ndarray:
    xy = _stone_xy()
    z = _terrain_height_at(xy) + STONE_SIZE[2] + STONE_CLEARANCE
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def _build_spot_arm_ik_mjcf(base_tf: np.ndarray, limits: tuple[tuple[float, float], ...]) -> str:
    pos = base_tf[:3]
    quat = _quat_xyzw_to_wxyz(base_tf[3:7])
    ranges = [_format_vec(limit) for limit in limits]
    return f"""<mujoco model="spot_arm_ik_chain">
  <compiler angle="radian" autolimits="true" />
  <worldbody>
    <body name="arm_link_sh0" pos="{_format_vec(pos)}" quat="{_format_vec(quat)}">
      <joint name="arm_sh0" type="hinge" axis="0 0 1" range="{ranges[0]}" />
      <body name="arm_link_sh1">
        <joint name="arm_sh1" type="hinge" axis="0 1 0" range="{ranges[1]}" />
        <body name="arm_link_hr0">
          <body name="arm_link_el0" pos="0.3385 0 0">
            <joint name="arm_el0" type="hinge" axis="0 1 0" range="{ranges[2]}" />
            <body name="arm_link_el1" pos="0.4033 0 0.075">
              <joint name="arm_el1" type="hinge" axis="1 0 0" range="{ranges[3]}" />
              <body name="arm_link_wr0">
                <joint name="arm_wr0" type="hinge" axis="0 1 0" range="{ranges[4]}" />
                <body name="arm_link_wr1">
                  <joint name="arm_wr1" type="hinge" axis="1 0 0" range="{ranges[5]}" />
                  <body name="arm_link_fngr" pos="0.11745 0 0.01482">
                    <joint name="arm_f1x" type="hinge" axis="0 1 0" range="{ranges[6]}" />
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


class SpotPickPlaceDemo:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / CONTROL_DECIMATION
        self.sim_time = 0.0
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.gait_phase = 0.0
        self.command = np.zeros(3, dtype=np.float32)
        self.reached_b = False
        self.reached_c = False
        self.b_aligned = False
        self.b_align_time = 0.0
        self.b_align_start_root_q = None
        self.b_align_start_leg_q = None
        self.b_align_target_root_q = None
        self.contacts_ready = False

        stand_qpos, stand_ctrl = _scene_keyframe()
        self.stand_ctrl = stand_ctrl.astype(np.float32)
        self.nominal_leg_ctrl = NOMINAL_LEG_CTRL.copy()
        self.nominal_leg_qpos = (stand_qpos[7 : 7 + ACT_DIM] + self.nominal_leg_ctrl).astype(np.float32)
        self.reset_leg_noise = np.random.default_rng(RESET_SEED).uniform(-0.05, 0.05, ACT_DIM).astype(np.float32)
        self.arm_qpos = stand_qpos[7 + ACT_DIM :].astype(np.float32)
        self.arm_q_cmd = self.arm_qpos.copy()
        self.stone_specs = tuple(self._stone_specs())
        if not self.stone_specs:
            raise ValueError("Expected at least one pickup stone.")
        self.current_stone_index = 0
        self.stone_qs = [np.array((*spec.pos, *spec.quat), dtype=np.float32) for spec in self.stone_specs]
        self.initial_stone_qs = [stone_q.copy() for stone_q in self.stone_qs]
        self.stone_released_flags = [False] * len(self.stone_specs)
        self.stone_pos = self.stone_qs[0][:3].copy()
        self.stone_q = self.stone_qs[0].copy()
        self.pin_stone_until_grasp = True
        self.stone_attached = False
        self.stone_grasp_offset_pos = None
        self.stone_grasp_offset_quat = None
        self.arm_retracted = False
        self.arm_retract_start_q = None
        self.arm_release_started = False
        self.arm_release_lowered = False
        self.release_open_start_q = None
        self.stone_released = False
        self.arm_release_opened = False
        self.arm_final_retracted = False
        self.arm_final_retract_start_q = None
        self.show_ik_targets = getattr(args, "show_ik_targets", False)
        self.stone_meshes = {}
        self.stone_mesh_indices = {}
        self.stone_mesh_vertices = {}
        self.stone_mesh_scales = {}
        for stone_index, asset_path in enumerate(MESH_STONE_ASSET_PATHS):
            if stone_index >= len(self.stone_specs):
                break
            vertices, indices = load_obj_triangles(asset_path)
            scale = scale_to_ellipsoid_bounds(vertices, self.stone_specs[stone_index].size)
            self.stone_meshes[stone_index] = newton.Mesh(vertices=vertices, indices=indices)
            self.stone_mesh_indices[stone_index] = indices
            self.stone_mesh_vertices[stone_index] = vertices * scale
            self.stone_mesh_scales[stone_index] = scale

        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(SCENE_PATH.resolve()),
            up_axis="Z",
            enable_self_collisions=True,
            ctrl_direct=True,
        )
        stone_cfg = newton.ModelBuilder.ShapeConfig(density=1800.0, mu=1.5, mu_torsional=0.08, mu_rolling=0.08)
        for stone_index, spec in enumerate(self.stone_specs):
            stone_body = builder.add_body(xform=wp.transform(wp.vec3(*spec.pos), wp.quat(*spec.quat)), label=spec.label)
            mesh = self.stone_meshes.get(stone_index)
            if mesh is not None:
                builder.add_shape_mesh(
                    stone_body,
                    mesh=mesh,
                    scale=wp.vec3(*self.stone_mesh_scales[stone_index]),
                    cfg=stone_cfg,
                    color=wp.vec3(0.18, 0.17, 0.15),
                    label=spec.label,
                )
            else:
                builder.add_shape_ellipsoid(
                    stone_body,
                    rx=float(spec.size[0]),
                    ry=float(spec.size[1]),
                    rz=float(spec.size[2]),
                    cfg=stone_cfg,
                    color=wp.vec3(0.18, 0.17, 0.15),
                    label=spec.label,
                )
        self.c_rover_shape_indices = self._add_c_rover(builder)

        self.model = builder.finalize()
        self._apply_training_actuator_scale()
        warnings.filterwarnings("ignore", message=r"Geom .* authored margin=.*")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
        )
        self.mujoco, _ = newton.solvers.SolverMuJoCo.import_mujoco()
        self.terrain_geom_id = self._find_mj_geom_id("lunar_terrain")
        self.foot_geom_ids = np.array([self._find_mj_geom_id(name) for name in ("FL", "FR", "HL", "HR")])
        self.stone_geom_ids = np.array([self._find_mj_geom_id(spec.label) for spec in self.stone_specs], dtype=np.int32)
        self.stone_geom_id = int(self.stone_geom_ids[self.current_stone_index])
        self.c_rover_geom_ids = np.array(self._find_mj_geom_ids(C_ROVER_LABEL), dtype=np.int32)
        self._configure_mj_collision_filters()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.root_q_slice, self.root_qd_slice = self._find_joint_slices(("freejoint",), q_width=7, qd_width=6)
        self.leg_q_slice, self.leg_qd_slice = self._find_joint_slices(LEG_JOINTS)
        self.arm_q_slice, self.arm_qd_slice = self._find_joint_slices(ARM_JOINTS)
        self.stone_q_slices = []
        self.stone_qd_slices = []
        for spec in self.stone_specs:
            stone_q_slice, stone_qd_slice = self._find_joint_slices(
                (f"{spec.label}_free_joint",),
                q_width=7,
                qd_width=6,
            )
            self.stone_q_slices.append(stone_q_slice)
            self.stone_qd_slices.append(stone_qd_slice)
        self.stone_q_slice = self.stone_q_slices[self.current_stone_index]
        self.stone_qd_slice = self.stone_qd_slices[self.current_stone_index]
        self.spot_body_index = self._find_body_index("body")
        self.wr1_body_index = self._find_body_index(ARM_EE_BODY)
        self.stone_body_indices = np.array(
            [self._find_body_index(spec.label) for spec in self.stone_specs], dtype=np.int32
        )
        self.stone_body_index = int(self.stone_body_indices[self.current_stone_index])

        ctrl_range = self.model.mujoco.actuator_ctrlrange.numpy().astype(np.float32)
        self.leg_ctrl_low = ctrl_range[:ACT_DIM, 0]
        self.leg_ctrl_high = ctrl_range[:ACT_DIM, 1]
        self.arm_ctrl_low = ctrl_range[ACT_DIM : ACT_DIM + ARM_ACTUATOR_COUNT, 0]
        self.arm_ctrl_high = ctrl_range[ACT_DIM : ACT_DIM + ARM_ACTUATOR_COUNT, 1]
        self.arm_lower = self.model.joint_limit_lower.numpy()[self.arm_qd_slice].astype(np.float32)
        self.arm_upper = self.model.joint_limit_upper.numpy()[self.arm_qd_slice].astype(np.float32)
        self._clear_arm_ik()

        patch_sb3_zip_loader()
        model_path = args.locomotion_model.resolve()
        vecnormalize_path = args.locomotion_vecnormalize.resolve()
        self.vecnormalize = VecNormalize.load(vecnormalize_path, DummyVecEnv([_VecNormalizeEnv]))
        self.vecnormalize.training = False
        self.vecnormalize.norm_reward = False
        self.policy = PPO.load(model_path, env=self.vecnormalize)

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.reset()
        print(f"Loaded locomotion policy: {model_path}")
        print(f"Loaded VecNormalize stats: {vecnormalize_path}")
        print(f"Stone position: {self.stone_pos.round(3).tolist()}")

    def _stone_specs(self) -> tuple[StoneSpec, ...]:
        stone_pos = _stone_position()
        return (
            StoneSpec(
                label=STONE_LABEL,
                pos=stone_pos,
                quat=np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
                size=STONE_SIZE.copy(),
            ),
        )

    def _set_current_stone(self, stone_index: int) -> None:
        if not 0 <= stone_index < len(self.stone_specs):
            raise IndexError(f"Stone index out of range: {stone_index}")

        self.current_stone_index = stone_index
        self.stone_q = self.stone_qs[stone_index].copy()
        self.stone_pos = self.stone_q[:3].copy()
        self.stone_q_slice = self.stone_q_slices[stone_index]
        self.stone_qd_slice = self.stone_qd_slices[stone_index]
        self.stone_body_index = int(self.stone_body_indices[stone_index])
        self.stone_geom_id = int(self.stone_geom_ids[stone_index])
        self.stone_released = self.stone_released_flags[stone_index]

    def _find_joint_slices(
        self,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        joint_indices = []
        for name in names:
            matches = [
                i for i, label in enumerate(self.model.joint_label) if label == name or label.endswith(f"/{name}")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one imported joint named '{name}', found {len(matches)}.")
            joint_indices.append(matches[0])

        q0, qd0 = int(q_starts[joint_indices[0]]), int(qd_starts[joint_indices[0]])
        return slice(q0, q0 + (q_width or len(names))), slice(qd0, qd0 + (qd_width or len(names)))

    def _find_mj_geom_id(self, token: str) -> int:
        matches = self._find_mj_geom_ids(token)
        if len(matches) != 1:
            raise ValueError(f"Expected one MuJoCo geom containing '{token}', found {len(matches)}.")
        return matches[0]

    def _find_mj_geom_ids(self, token: str) -> list[int]:
        matches = []
        for geom_id in range(self.solver.mj_model.ngeom):
            label = self.mujoco.mj_id2name(self.solver.mj_model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if label == token or label.startswith(f"{token}_") or label.endswith(f"/{token}") or f"/{token}_" in label:
                matches.append(geom_id)
        if not matches:
            raise ValueError(f"Expected MuJoCo geoms containing '{token}', found none.")
        return matches

    def _find_body_index(self, name: str) -> int:
        return self._find_body_index_in_labels(self.model.body_label, name)

    @staticmethod
    def _find_body_index_in_labels(labels, name: str) -> int:
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named '{name}', found {len(matches)}.")
        return matches[0]

    @staticmethod
    def _add_c_rover(builder: newton.ModelBuilder) -> list[int]:
        cargo_center = _c_release_box_center()
        yaw = _approach_yaw()
        ground_z = _terrain_height_at(cargo_center)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        yaw_quat = np.array(_yaw_to_xyzw(yaw), dtype=np.float32)
        wheel_local_quat = np.array(
            [math.sin(-0.25 * math.pi), 0.0, 0.0, math.cos(-0.25 * math.pi)],
            dtype=np.float32,
        )
        wheel_quat = _quat_multiply(yaw_quat, wheel_local_quat)

        def local_xy(offset: tuple[float, float] | np.ndarray) -> np.ndarray:
            ox, oy = float(offset[0]), float(offset[1])
            return cargo_center + np.array([cos_yaw * ox - sin_yaw * oy, sin_yaw * ox + cos_yaw * oy])

        def to_wp_quat(quat: np.ndarray) -> wp.quat:
            return wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

        def make_xform(offset: tuple[float, float] | np.ndarray, z: float, quat: np.ndarray = yaw_quat):
            xy = local_xy(offset)
            return wp.transform(wp.vec3(float(xy[0]), float(xy[1]), float(z)), to_wp_quat(quat))

        rover_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=1.5,
            mu_torsional=0.03,
            mu_rolling=0.01,
        )
        rover_visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
        )
        shape_indices = []

        def add_box(
            suffix: str,
            offset: tuple[float, float] | np.ndarray,
            z: float,
            hx: float,
            hy: float,
            hz: float,
            color: wp.vec3,
            has_collision: bool = False,
        ) -> None:
            shape_indices.append(
                builder.add_shape_box(
                    body=-1,
                    xform=make_xform(offset, z),
                    hx=float(hx),
                    hy=float(hy),
                    hz=float(hz),
                    cfg=rover_cfg if has_collision else rover_visual_cfg,
                    color=color,
                    label=f"{C_ROVER_LABEL}_{suffix}",
                )
            )

        chassis_color = wp.vec3(0.17, 0.17, 0.15)
        cargo_color = wp.vec3(0.42, 0.34, 0.20)
        cabin_color = wp.vec3(0.55, 0.52, 0.44)
        wheel_color = wp.vec3(0.04, 0.04, 0.04)
        rail_color = wp.vec3(0.27, 0.27, 0.24)

        chassis_z = ground_z + C_ROVER_WHEEL_RADIUS + C_ROVER_CHASSIS_HALF_EXTENTS[2] * 0.45
        add_box(
            "chassis",
            C_ROVER_CHASSIS_CENTER_OFFSET,
            chassis_z,
            C_ROVER_CHASSIS_HALF_EXTENTS[0],
            C_ROVER_CHASSIS_HALF_EXTENTS[1],
            C_ROVER_CHASSIS_HALF_EXTENTS[2],
            chassis_color,
            has_collision=True,
        )

        chassis_top_z = chassis_z + C_ROVER_CHASSIS_HALF_EXTENTS[2]
        wall_center_z = chassis_top_z + 0.5 * C_ROVER_CARGO_WALL_HEIGHT
        inner_hx, inner_hy = C_ROVER_CARGO_INNER_HALF_EXTENTS
        thickness = C_ROVER_CARGO_WALL_THICKNESS
        half_thickness = 0.5 * thickness

        cargo_wall_specs = (
            ("cargo_front", (inner_hx + half_thickness, 0.0), half_thickness, inner_hy + thickness),
            ("cargo_rear", (-inner_hx - half_thickness, 0.0), half_thickness, inner_hy + thickness),
            ("cargo_left", (0.0, inner_hy + half_thickness), inner_hx, half_thickness),
            ("cargo_right", (0.0, -inner_hy - half_thickness), inner_hx, half_thickness),
        )
        for suffix, offset, hx, hy in cargo_wall_specs:
            add_box(
                suffix,
                offset,
                wall_center_z,
                hx,
                hy,
                0.5 * C_ROVER_CARGO_WALL_HEIGHT,
                cargo_color,
                has_collision=True,
            )

        add_box("front_equipment_box", (0.62, 0.0), chassis_top_z + 0.11, 0.20, 0.28, 0.11, cabin_color)
        add_box("front_roof", (0.62, 0.0), chassis_top_z + 0.245, 0.23, 0.31, 0.025, rail_color)
        wheel_center_z = ground_z + C_ROVER_WHEEL_RADIUS
        for axle_index, x_offset in enumerate(C_ROVER_WHEEL_X_OFFSETS):
            add_box(f"axle_{axle_index}", (x_offset, 0.0), wheel_center_z, 0.035, 0.42, 0.025, rail_color)
            for side_name, y_offset in (("left", C_ROVER_WHEEL_Y_OFFSET), ("right", -C_ROVER_WHEEL_Y_OFFSET)):
                shape_indices.append(
                    builder.add_shape_cylinder(
                        body=-1,
                        xform=make_xform((x_offset, y_offset), wheel_center_z, wheel_quat),
                        radius=C_ROVER_WHEEL_RADIUS,
                        half_height=C_ROVER_WHEEL_HALF_WIDTH,
                        cfg=rover_visual_cfg,
                        color=wheel_color,
                        label=f"{C_ROVER_LABEL}_wheel_{side_name}_{axle_index}",
                    )
                )

        mast_center_z = chassis_top_z + 0.33
        shape_indices.append(
            builder.add_shape_cylinder(
                body=-1,
                xform=make_xform((0.62, 0.26), mast_center_z),
                radius=0.012,
                half_height=0.24,
                cfg=rover_visual_cfg,
                color=rail_color,
                label=f"{C_ROVER_LABEL}_antenna_mast",
            )
        )
        add_box("antenna_panel", (0.62, 0.26), chassis_top_z + 0.58, 0.055, 0.012, 0.045, rail_color)

        return shape_indices

    def _configure_mj_collision_filters(self) -> None:
        robot_mask = np.ones(self.solver.mj_model.ngeom, dtype=bool)
        non_robot_geom_ids = np.concatenate(
            (
                np.array([self.terrain_geom_id], dtype=np.int32),
                self.stone_geom_ids,
                self.c_rover_geom_ids,
            )
        )
        robot_mask[non_robot_geom_ids] = False

        self.solver.mj_model.geom_contype[robot_mask] = 2
        self.solver.mj_model.geom_conaffinity[robot_mask] = 5
        self.solver.mj_model.geom_contype[self.terrain_geom_id] = 1
        self.solver.mj_model.geom_conaffinity[self.terrain_geom_id] = 6
        self.solver.mj_model.geom_contype[self.stone_geom_ids] = 4
        self.solver.mj_model.geom_conaffinity[self.stone_geom_ids] = 3
        self.solver.mj_model.geom_condim[self.stone_geom_ids] = 6
        self.solver.mj_model.geom_priority[self.stone_geom_ids] = 2
        self.solver.mj_model.geom_contype[self.c_rover_geom_ids] = 0
        self.solver.mj_model.geom_conaffinity[self.c_rover_geom_ids] = 4
        self.solver.mj_model.geom_condim[self.c_rover_geom_ids] = 6
        self.solver.mj_model.geom_priority[self.c_rover_geom_ids] = 2

    def _apply_training_actuator_scale(self) -> None:
        gain = self.model.mujoco.actuator_gainprm.numpy()
        bias = self.model.mujoco.actuator_biasprm.numpy()
        gain[:ACT_DIM, 0] *= ACTUATOR_GAIN_SCALE
        bias[:ACT_DIM, 0] *= ACTUATOR_GAIN_SCALE
        bias[:ACT_DIM, 1] *= ACTUATOR_GAIN_SCALE
        bias[:ACT_DIM, 2] *= ACTUATOR_GAIN_SCALE**0.5
        self.model.mujoco.actuator_gainprm.assign(gain)
        self.model.mujoco.actuator_biasprm.assign(bias)

    def reset(self) -> None:
        self._write_reset_state(self.state_0)
        self._write_reset_state(self.state_1)
        self._write_stand_ctrl()
        self.sim_time = 0.0
        self.phase_time = 0.0
        self.last_action.fill(0.0)
        self.gait_phase = 0.0
        self.reached_b = False
        self.reached_c = False
        self.b_aligned = False
        self.b_align_time = 0.0
        self.b_align_start_root_q = None
        self.b_align_start_leg_q = None
        self.b_align_target_root_q = None
        self.contacts_ready = False
        self.stone_qs = [stone_q.copy() for stone_q in self.initial_stone_qs]
        self.stone_released_flags = [False] * len(self.stone_specs)
        self._set_current_stone(0)
        self.pin_stone_until_grasp = True
        self.stone_attached = False
        self.stone_grasp_offset_pos = None
        self.stone_grasp_offset_quat = None
        self.arm_retracted = False
        self.arm_retract_start_q = None
        self.arm_release_started = False
        self.arm_release_lowered = False
        self.release_open_start_q = None
        self.stone_released = False
        self.arm_release_opened = False
        self.arm_final_retracted = False
        self.arm_final_retract_start_q = None
        self._clear_arm_ik()

    def _clear_arm_ik(self) -> None:
        self.phase_time = 0.0
        self.arm_q_cmd = self.arm_qpos.copy()
        self.ik_model = None
        self.ik_solver = None
        self.ik_joint_q = None
        self.ik_wr1_body_index = None
        self.ik_start_target = None
        self.ik_pregrasp_target = None
        self.ik_final_target = None
        self.current_ik_target = None
        self.ik_error = math.inf

    def _write_reset_state(self, state) -> None:
        state.joint_q.zero_()
        state.joint_qd.zero_()
        state.body_qd.zero_()
        state.joint_q[self.root_q_slice].assign((A_POINT[0], A_POINT[1], RESET_BASE_HEIGHT, *_yaw_to_xyzw(A_YAW)))
        state.joint_q[self.leg_q_slice].assign(self.nominal_leg_qpos + self.reset_leg_noise)
        state.joint_q[self.arm_q_slice].assign(self.arm_qpos)
        for stone_q_slice, stone_q in zip(self.stone_q_slices, self.stone_qs, strict=True):
            state.joint_q[stone_q_slice].assign(stone_q)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _pin_stone_index_to_start(self, state, stone_index: int) -> None:
        state.joint_q[self.stone_q_slices[stone_index]].assign(self.stone_qs[stone_index])
        state.joint_qd[self.stone_qd_slices[stone_index]].zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _pin_stone_to_start(self, state) -> None:
        self._pin_stone_index_to_start(state, self.current_stone_index)

    def _attach_stone_to_gripper(self, state) -> None:
        body_q = state.body_q.numpy()
        gripper_tf = body_q[self.wr1_body_index]
        stone_tf = body_q[self.stone_body_index]
        gripper_rot = _xyzw_to_matrix(gripper_tf[3:7])
        self.stone_grasp_offset_pos = (gripper_rot.T @ (stone_tf[:3] - gripper_tf[:3])).astype(np.float32)
        self.stone_grasp_offset_quat = _quat_multiply(_quat_inverse(gripper_tf[3:7]), stone_tf[3:7])
        self.pin_stone_until_grasp = False
        self.stone_attached = True
        self.phase_time = 0.0
        self._attach_stone_pose(state)
        print("Attached stone to gripper; retracting arm")

    def _attach_stone_pose(self, state, update_fk: bool = True) -> None:
        body_q = state.body_q.numpy()
        gripper_tf = body_q[self.wr1_body_index]
        gripper_rot = _xyzw_to_matrix(gripper_tf[3:7])
        stone_pos = gripper_tf[:3] + gripper_rot @ self.stone_grasp_offset_pos
        stone_quat = _quat_multiply(gripper_tf[3:7], self.stone_grasp_offset_quat)
        state.joint_q[self.stone_q_slice].assign((*stone_pos, *stone_quat))
        state.joint_qd[self.stone_qd_slice].zero_()
        if update_fk:
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _update_stone_constraint(self, state) -> None:
        if self.stone_attached:
            self._attach_stone_pose(state)
        elif self.pin_stone_until_grasp:
            self._pin_stone_to_start(state)

    def _arm_stowed_q(self) -> np.ndarray:
        arm_q = self.arm_qpos.copy()
        arm_q[-1] = GRIPPER_CLOSED
        return np.clip(arm_q, self.arm_ctrl_low, self.arm_ctrl_high).astype(np.float32)

    def _arm_initial_q(self) -> np.ndarray:
        return np.clip(self.arm_qpos, self.arm_ctrl_low, self.arm_ctrl_high).astype(np.float32)

    def _start_arm_retract(self) -> None:
        self.phase_time = 0.0
        self.current_ik_target = None
        self.arm_retract_start_q = self.arm_q_cmd.copy()

    def _arm_retract_q_at_time(self) -> np.ndarray:
        alpha = _smoothstep(self.phase_time / max(ARM_RETRACT_SECONDS, 1.0e-6))
        return ((1.0 - alpha) * self.arm_retract_start_q + alpha * self._arm_stowed_q()).astype(np.float32)

    def _release_gripper_q_at_time(self) -> np.ndarray:
        alpha = _smoothstep(self.phase_time / max(GRIPPER_RELEASE_SECONDS, 1.0e-6))
        arm_q = self.release_open_start_q.copy()
        arm_q[-1] = (1.0 - alpha) * self.release_open_start_q[-1] + alpha * GRIPPER_OPEN
        return np.clip(arm_q, self.arm_ctrl_low, self.arm_ctrl_high).astype(np.float32)

    def _start_final_arm_retract(self) -> None:
        self.phase_time = 0.0
        self.current_ik_target = None
        self.ik_solver = None
        self.arm_final_retract_start_q = self.arm_q_cmd.copy()

    def _final_arm_retract_q_at_time(self) -> np.ndarray:
        alpha = _smoothstep(self.phase_time / max(ARM_RETRACT_SECONDS, 1.0e-6))
        return ((1.0 - alpha) * self.arm_final_retract_start_q + alpha * self._arm_initial_q()).astype(np.float32)

    def _release_stone(self, state) -> None:
        state.joint_qd[self.stone_qd_slice].zero_()
        self.stone_attached = False
        self.stone_released_flags[self.current_stone_index] = True
        self.stone_released = True
        self.stone_grasp_offset_pos = None
        self.stone_grasp_offset_quat = None
        print("Released stone")

    def _write_stand_ctrl(self) -> None:
        self._write_ctrl(self.nominal_leg_ctrl, self.stand_ctrl[ACT_DIM:])

    def _write_ctrl(self, leg_ctrl: np.ndarray, arm_q: np.ndarray) -> None:
        self.control.mujoco.ctrl[:ACT_DIM].assign(leg_ctrl)
        arm_ctrl = np.clip(arm_q, self.arm_ctrl_low, self.arm_ctrl_high)
        self.control.mujoco.ctrl[ACT_DIM : ACT_DIM + ARM_ACTUATOR_COUNT].assign(arm_ctrl)

    def _write_arm_ctrl(self, arm_q: np.ndarray) -> None:
        self.last_action.fill(0.0)
        self.gait_phase = 0.0
        self._write_ctrl(MANIPULATION_LEG_CTRL, arm_q)

    def _base_xy(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.root_q_slice.start : self.root_q_slice.start + 2].copy()

    def _base_rotation(self) -> np.ndarray:
        q = self.state_0.joint_q.numpy()[self.root_q_slice.start + 3 : self.root_q_slice.start + 7]
        return _xyzw_to_matrix(q)

    def _base_yaw(self) -> float:
        rotation = self._base_rotation()
        return float(math.atan2(rotation[1, 0], rotation[0, 0]))

    def _base_velocity_body(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self._base_rotation()
        qd = self.state_0.joint_qd.numpy()
        linear = rotation.T @ qd[self.root_qd_slice.start : self.root_qd_slice.start + 3]
        angular = rotation.T @ qd[self.root_qd_slice.start + 3 : self.root_qd_slice.start + 6]
        return linear, angular

    def _gait_activity(self) -> float:
        command_speed = float(np.linalg.norm(self.command[:2]) + 0.2 * abs(self.command[2]))
        return float(np.clip((command_speed - 0.04) / 0.08, 0.0, 1.0))

    def _gait_frequency_scale(self) -> float:
        command_speed = float(np.linalg.norm(self.command[:2]) + 0.2 * abs(self.command[2]))
        speed_ratio = float(np.clip(command_speed / 0.5, 0.0, 1.5))
        return 0.7 + 0.3 * speed_ratio

    def _advance_gait_phase(self) -> None:
        activity = self._gait_activity()
        if activity <= 0.0:
            self.gait_phase = 0.0
            return
        phase_step = self.frame_dt * self._gait_frequency_scale() / GAIT_PERIOD
        self.gait_phase = float((self.gait_phase + activity * phase_step) % 1.0)

    def _desired_contacts(self) -> np.ndarray:
        activity = self._gait_activity()
        if activity <= 0.0:
            return np.ones(4, dtype=np.float32)
        diagonal_a = 0.5 + 0.5 * np.tanh(GAIT_CONTACT_SHARPNESS * np.sin(2.0 * np.pi * self.gait_phase))
        diagonal_b = 1.0 - diagonal_a
        trot_contacts = np.array([diagonal_a, diagonal_b, diagonal_b, diagonal_a], dtype=np.float32)
        return activity * trot_contacts + (1.0 - activity) * np.ones(4, dtype=np.float32)

    def _gait_observation(self) -> np.ndarray:
        angle = 2.0 * np.pi * self.gait_phase
        return np.concatenate(
            [
                np.array([np.sin(angle), np.cos(angle)], dtype=np.float32),
                self._desired_contacts(),
            ]
        )

    def _base_tilt_degrees(self) -> float:
        projected_gravity = self._base_rotation().T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return float(np.degrees(np.arccos(np.clip(-projected_gravity[2], -1.0, 1.0))))

    def _grasp_base_root_q(self) -> np.ndarray:
        return B_GRASP_ROOT_Q.copy()

    def _command_to_target(
        self,
        target_xy: np.ndarray,
        face_xy: np.ndarray | None = None,
        min_forward_speed: float = MIN_FORWARD_SPEED,
        arrival_radius: float = ARRIVAL_RADIUS,
    ) -> tuple[np.ndarray, float, float]:
        base_xy = self._base_xy()
        delta = target_xy - base_xy
        distance = float(np.linalg.norm(delta))
        near_target = distance <= arrival_radius

        if face_xy is not None and distance <= STONE_ALIGN_RADIUS:
            face_delta = face_xy - base_xy
            target_yaw = float(math.atan2(face_delta[1], face_delta[0]))
        elif distance > 1e-6:
            target_yaw = float(math.atan2(delta[1], delta[0]))
        else:
            target_yaw = self._base_yaw()

        heading_error = _wrap_angle(target_yaw - self._base_yaw())
        forward = 0.0 if near_target else float(np.clip(distance * 0.5, min_forward_speed, FORWARD_SPEED))
        if not near_target and abs(heading_error) > 0.9:
            forward = min_forward_speed

        command = np.array(
            [forward, 0.0, np.clip(YAW_GAIN * heading_error, -MAX_YAW_RATE, MAX_YAW_RATE)],
            dtype=np.float32,
        )
        return command, distance, abs(heading_error)

    def _observation(self) -> np.ndarray:
        projected_gravity = self._base_rotation().T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        base_linear, base_angular = self._base_velocity_body()
        joint_q = self.state_0.joint_q.numpy()[self.leg_q_slice]
        joint_qd = self.state_0.joint_qd.numpy()[self.leg_qd_slice]
        joint_pos = joint_q - self.nominal_leg_qpos
        obs = _build_locomotion_observation(
            base_angular=base_angular,
            projected_gravity=projected_gravity,
            command=self.command,
            gait_observation=self._gait_observation(),
            joint_pos=joint_pos,
            joint_vel=joint_qd,
            last_action=self.last_action,
            foot_contacts=self._foot_contacts(),
            base_linear=base_linear,
        )
        return self.vecnormalize.normalize_obs(obs[np.newaxis, :])

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=np.float32)
        if not self.contacts_ready:
            return contacts
        foot_to_index = {int(geom_id): idx for idx, geom_id in enumerate(self.foot_geom_ids)}
        for i in range(self.solver.mj_data.ncon):
            contact = self.solver.mj_data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.terrain_geom_id and geom2 in foot_to_index:
                contacts[foot_to_index[geom2]] = 1.0
            elif geom2 == self.terrain_geom_id and geom1 in foot_to_index:
                contacts[foot_to_index[geom1]] = 1.0
        return contacts

    def _arm_base_transform(self) -> np.ndarray:
        spot_tf = wp.transform(*self.state_0.body_q.numpy()[self.spot_body_index])
        base_pos = np.asarray(wp.transform_point(spot_tf, wp.vec3(*ARM_BASE_OFFSET)), dtype=np.float32)
        base_rot = np.asarray(wp.transform_get_rotation(spot_tf), dtype=np.float32)
        return np.concatenate((base_pos, base_rot)).astype(np.float32)

    @staticmethod
    def _link_point_position(body_q: np.ndarray, body_index: int, link_offset: wp.vec3) -> np.ndarray:
        point = wp.transform_point(wp.transform(*body_q[body_index]), link_offset)
        return np.asarray(point, dtype=np.float32)

    @staticmethod
    def _down_axis_target(target: np.ndarray) -> np.ndarray:
        return target + np.array([0.0, 0.0, -WR1_DOWN_AXIS_LENGTH], dtype=np.float32)

    def _arm_joint_limits(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(lo), float(hi)) for lo, hi in zip(self.arm_lower, self.arm_upper, strict=True))

    def _stone_approach_target(self) -> np.ndarray:
        stone_top_z = float(self.stone_pos[2] + STONE_SIZE[2])
        return (
            np.array(
                [self.stone_pos[0], self.stone_pos[1], stone_top_z + ARM_GRASP_CLEARANCE],
                dtype=np.float32,
            )
            + ARM_GRASP_TARGET_OFFSET
        )

    def _stone_release_target(self) -> np.ndarray:
        yaw = self._base_yaw()
        place_xy = (
            self._base_xy() + np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64) * STONE_RELEASE_FORWARD_OFFSET
        )
        place_z = _terrain_height_at(place_xy) + STONE_SIZE[2] + ARM_RELEASE_CLEARANCE
        return np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)

    def _start_arm_ik_motion(self, final_target: np.ndarray, label: str) -> None:
        self.phase_time = 0.0
        self.arm_q_cmd = self.state_0.joint_q.numpy()[self.arm_q_slice].astype(np.float32)

        ik_builder = newton.ModelBuilder()
        ik_builder.add_mjcf(
            _build_spot_arm_ik_mjcf(self._arm_base_transform(), self._arm_joint_limits()),
            up_axis="Z",
            enable_self_collisions=False,
        )
        self.ik_model = ik_builder.finalize()
        self.ik_wr1_body_index = self._find_body_index_in_labels(self.ik_model.body_label, ARM_EE_BODY)
        self.ik_joint_q = wp.array(self.arm_q_cmd.reshape(1, -1), dtype=wp.float32)

        body_q = self.state_0.body_q.numpy()
        self.ik_start_target = self._link_point_position(body_q, self.wr1_body_index, GRIPPER_TRACK_OFFSET)
        self.ik_final_target = final_target
        self.ik_pregrasp_target = self.ik_final_target + np.array([0.0, 0.0, ARM_PREGRASP_HEIGHT], dtype=np.float32)
        self.current_ik_target = self.ik_start_target.copy()

        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=GRIPPER_TRACK_OFFSET,
            target_positions=wp.array([wp.vec3(*self.current_ik_target)], dtype=wp.vec3),
            weight=1.0,
        )
        self.ik_down_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=GRIPPER_TRACK_OFFSET + wp.vec3(WR1_DOWN_AXIS_LENGTH, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self._down_axis_target(self.current_ik_target))], dtype=wp.vec3),
            weight=IK_DOWN_AXIS_WEIGHT,
        )
        self.ik_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=IK_LIMIT_WEIGHT,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_down_obj, self.ik_limit_obj],
            lambda_initial=IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        print(
            f"Start arm IK {label}: "
            f"pregrasp={np.round(self.ik_pregrasp_target, 3).tolist()}, "
            f"final={np.round(self.ik_final_target, 3).tolist()}"
        )

    def _start_arm_approach(self) -> None:
        self._start_arm_ik_motion(self._stone_approach_target(), "approach")

    def _start_arm_release(self) -> None:
        self.arm_release_started = True
        self.arm_release_lowered = False
        self.arm_release_opened = False
        self.release_open_start_q = None
        self._start_arm_ik_motion(self._stone_release_target(), "release")

    def _arm_target_at_time(self) -> np.ndarray:
        if self.phase_time < ARM_PREGRASP_SECONDS:
            alpha = _smoothstep(self.phase_time / max(ARM_PREGRASP_SECONDS, 1.0e-6))
            return (1.0 - alpha) * self.ik_start_target + alpha * self.ik_pregrasp_target

        descent_time = self.phase_time - ARM_PREGRASP_SECONDS
        if descent_time < ARM_DESCENT_SECONDS:
            alpha = _smoothstep(descent_time / max(ARM_DESCENT_SECONDS, 1.0e-6))
            return (1.0 - alpha) * self.ik_pregrasp_target + alpha * self.ik_final_target

        return self.ik_final_target.copy()

    def _gripper_target_at_time(self) -> float:
        if self.stone_attached:
            return GRIPPER_CLOSED

        close_time = self.phase_time - ARM_PREGRASP_SECONDS - ARM_DESCENT_SECONDS
        if close_time <= 0.0:
            return GRIPPER_OPEN

        alpha = _smoothstep(close_time / max(GRIPPER_CLOSE_SECONDS, 1.0e-6))
        return float((1.0 - alpha) * GRIPPER_OPEN + alpha * GRIPPER_CLOSED)

    def _limit_arm_joint_step(self, candidate_q: np.ndarray) -> np.ndarray:
        candidate_q = np.clip(candidate_q, self.arm_lower, self.arm_upper)
        if MAX_ARM_JOINT_STEP <= 0.0:
            return candidate_q

        delta = candidate_q - self.arm_q_cmd
        step = float(np.max(np.abs(delta)))
        if step <= MAX_ARM_JOINT_STEP:
            return candidate_q.astype(np.float32)

        limited_q = self.arm_q_cmd + delta * (MAX_ARM_JOINT_STEP / step)
        return np.clip(limited_q, self.arm_lower, self.arm_upper).astype(np.float32)

    def _solve_arm_ik(self, target: np.ndarray) -> np.ndarray:
        self.ik_pos_obj.set_target_position(0, wp.vec3(*target))
        self.ik_down_obj.set_target_position(0, wp.vec3(*self._down_axis_target(target)))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=IK_ITERATIONS,
            step_size=IK_STEP_SIZE,
        )
        candidate_q = self.ik_joint_q.numpy()[0].astype(np.float32)
        candidate_q[-1] = self._gripper_target_at_time()
        arm_q = self._limit_arm_joint_step(candidate_q)
        self.ik_joint_q.assign(arm_q.reshape(1, -1))
        return arm_q

    def _apply_arm_approach(self) -> None:
        if self.ik_solver is None:
            self._start_arm_approach()

        self.command.fill(0.0)
        self.current_ik_target = self._arm_target_at_time()
        self.arm_q_cmd = self._solve_arm_ik(self.current_ik_target)
        self._write_arm_ctrl(self.arm_q_cmd)

        body_q = self.state_0.body_q.numpy()
        actual = self._link_point_position(body_q, self.wr1_body_index, GRIPPER_TRACK_OFFSET)
        self.ik_error = float(np.linalg.norm(actual - self.current_ik_target))
        self.phase_time += self.frame_dt

    def _apply_arm_retract(self) -> None:
        if self.arm_retract_start_q is None:
            self._start_arm_retract()

        self.command.fill(0.0)
        self.arm_q_cmd = self._arm_retract_q_at_time()
        self._write_arm_ctrl(self.arm_q_cmd)
        self.phase_time += self.frame_dt

        if self.phase_time >= ARM_RETRACT_SECONDS:
            self.arm_q_cmd = self._arm_stowed_q()
            self._write_arm_ctrl(self.arm_q_cmd)
            self.arm_retracted = True
            print(f"Arm retracted; walking to C={np.round(C_POINT, 3).tolist()}")

    def _apply_arm_release(self) -> None:
        if not self.arm_release_started:
            self._start_arm_release()

        self.command.fill(0.0)
        if not self.arm_release_lowered:
            self.current_ik_target = self._arm_target_at_time()
            self.arm_q_cmd = self._solve_arm_ik(self.current_ik_target)
            self._write_arm_ctrl(self.arm_q_cmd)

            body_q = self.state_0.body_q.numpy()
            actual = self._link_point_position(body_q, self.wr1_body_index, GRIPPER_TRACK_OFFSET)
            self.ik_error = float(np.linalg.norm(actual - self.current_ik_target))
            self.phase_time += self.frame_dt
            if self.phase_time >= ARM_PLACE_SECONDS:
                self.arm_release_lowered = True
                self.release_open_start_q = self.arm_q_cmd.copy()
                self.phase_time = 0.0
                print("Stone lowered; opening gripper")
                if self.stone_attached:
                    self._attach_stone_pose(self.state_0)
                    self._release_stone(self.state_0)
            return

        self.arm_q_cmd = self._release_gripper_q_at_time()
        self._write_arm_ctrl(self.arm_q_cmd)
        self.phase_time += self.frame_dt
        if self.phase_time >= GRIPPER_RELEASE_SECONDS:
            self.arm_q_cmd[-1] = GRIPPER_OPEN
            self._write_arm_ctrl(self.arm_q_cmd)
            self.arm_release_opened = True

    def _apply_final_arm_retract(self) -> None:
        if self.arm_final_retract_start_q is None:
            self._start_final_arm_retract()

        self.command.fill(0.0)
        self.arm_q_cmd = self._final_arm_retract_q_at_time()
        self._write_arm_ctrl(self.arm_q_cmd)
        self.phase_time += self.frame_dt

        if self.phase_time >= ARM_RETRACT_SECONDS:
            self.arm_q_cmd = self._arm_initial_q()
            self._write_arm_ctrl(self.arm_q_cmd)
            self.arm_final_retracted = True
            print("Arm returned to initial pose")

    def _apply_b_alignment(self) -> bool:
        if self.b_align_start_root_q is None:
            self.b_align_start_root_q = self.state_0.joint_q.numpy()[self.root_q_slice].astype(np.float32)
            self.b_align_start_leg_q = self.state_0.joint_q.numpy()[self.leg_q_slice].astype(np.float32)
            self.b_align_target_root_q = self._grasp_base_root_q()
            print(
                f"Aligning B grasp pose: "
                f"from={self.b_align_start_root_q[:3].round(3).tolist()}, "
                f"to={self.b_align_target_root_q[:3].round(3).tolist()}"
            )

        self.command.fill(0.0)
        self._write_arm_ctrl(self.stand_ctrl[ACT_DIM:])

        self.b_align_time = min(B_ALIGN_SECONDS, self.b_align_time + self.frame_dt)
        alpha = _smoothstep(self.b_align_time / max(B_ALIGN_SECONDS, 1.0e-6))

        start_root_q = self.b_align_start_root_q
        target_root_q = self.b_align_target_root_q
        root_q = target_root_q.copy()
        root_q[:3] = ((1.0 - alpha) * start_root_q[:3] + alpha * target_root_q[:3]).astype(np.float32)

        start_quat = start_root_q[3:]
        target_quat = target_root_q[3:]
        if float(np.dot(start_quat, target_quat)) < 0.0:
            target_quat = -target_quat
        root_q[3:] = _quat_normalize((1.0 - alpha) * start_quat + alpha * target_quat)

        leg_q = ((1.0 - alpha) * self.b_align_start_leg_q + alpha * B_GRASP_LEG_Q).astype(np.float32)
        self.state_0.joint_q[self.root_q_slice].assign(root_q)
        self.state_0.joint_q[self.leg_q_slice].assign(leg_q)
        self.state_0.joint_qd[self.root_qd_slice].zero_()
        self.state_0.joint_qd[self.leg_qd_slice].zero_()
        self.state_0.body_qd.zero_()
        self._pin_stone_to_start(self.state_0)

        if self.b_align_time < B_ALIGN_SECONDS:
            return False

        self.state_0.joint_q[self.root_q_slice].assign(target_root_q)
        self.state_0.joint_q[self.leg_q_slice].assign(B_GRASP_LEG_Q)
        self.state_0.joint_qd[self.root_qd_slice].zero_()
        self.state_0.joint_qd[self.leg_qd_slice].zero_()
        self.state_0.body_qd.zero_()
        self._pin_stone_to_start(self.state_0)
        print(
            f"Aligned B grasp pose: "
            f"position={self._base_xy().round(3).tolist()}, "
            f"yaw={self._base_yaw():.3f}, tilt={self._base_tilt_degrees():.1f} deg"
        )
        return True

    def _apply_policy(
        self,
        target_xy: np.ndarray,
        target_label: str,
        arm_q: np.ndarray,
        face_xy: np.ndarray | None = None,
        min_forward_speed: float = MIN_FORWARD_SPEED,
        arrival_radius: float = ARRIVAL_RADIUS,
        require_yaw: bool = True,
    ) -> bool:
        self.command, distance, yaw_error = self._command_to_target(
            target_xy,
            face_xy,
            min_forward_speed,
            arrival_radius,
        )
        if distance <= arrival_radius and (not require_yaw or yaw_error <= ARRIVAL_YAW_TOLERANCE):
            print(f"Reached {target_label}: position={self._base_xy().round(3).tolist()}, target={target_xy.tolist()}")
            self.command.fill(0.0)
            self._write_arm_ctrl(arm_q)
            return True

        action, _ = self.policy.predict(self._observation(), deterministic=True)
        policy_action = np.clip(action[0], -1.0, 1.0).astype(np.float32)
        self.last_action = policy_action
        leg_ctrl = np.clip(
            self.nominal_leg_ctrl + policy_action * ACTION_SCALE,
            self.leg_ctrl_low,
            self.leg_ctrl_high,
        )
        self._write_ctrl(leg_ctrl, arm_q)
        self._advance_gait_phase()
        return False

    def step(self) -> None:
        if not self.reached_b and self.sim_time < MAX_SECONDS:
            self.reached_b = self._apply_policy(
                B_POINT,
                "B",
                self.stand_ctrl[ACT_DIM:],
                self.stone_pos[:2],
                arrival_radius=B_HANDOFF_RADIUS,
            )
        elif self.reached_b and not self.b_aligned:
            self.b_aligned = self._apply_b_alignment()
        elif not self.stone_attached and not self.stone_released:
            self._apply_arm_approach()
        elif not self.arm_retracted:
            self._apply_arm_retract()
        elif not self.reached_c and self.sim_time < MAX_SECONDS:
            self.reached_c = self._apply_policy(
                C_POINT,
                "C",
                self.arm_q_cmd,
                min_forward_speed=MIN_FORWARD_SPEED,
                arrival_radius=C_ARRIVAL_RADIUS,
                require_yaw=False,
            )
        elif not self.arm_release_opened:
            self._apply_arm_release()
        elif not self.arm_final_retracted:
            self._apply_final_arm_retract()
        else:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.arm_q_cmd)

        for _ in range(CONTROL_DECIMATION):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if (
                not self.stone_attached
                and not self.stone_released
                and self.ik_solver is not None
                and self.phase_time >= ARM_GRASP_SECONDS
            ):
                self._attach_stone_to_gripper(self.state_0)
            self._update_stone_constraint(self.state_0)
        self.sim_time += self.frame_dt
        self.contacts_ready = True

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.show_ik_targets and self.current_ik_target is not None and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "target_wr1_grasp_point",
                wp.transform(wp.vec3(*self.current_ik_target), wp.quat_identity()),
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise ValueError("Body transforms became non-finite.")
        if self.ik_solver is not None and self.phase_time > 0.5:
            if not (GRIPPER_OPEN - 0.1 <= self.arm_q_cmd[-1] <= GRIPPER_CLOSED + 0.1):
                raise ValueError(f"Gripper target is out of range: q={self.arm_q_cmd[-1]:.3f}")
            if self.ik_error > 0.45:
                raise ValueError(f"Arm IK target error is too high: {self.ik_error:.3f} m")


def _create_base_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Run the Spot A-to-B pickup setup through Newton with SolverMuJoCo."
    parser.set_defaults(num_frames=100000, viewer="gl", usd_fps=FPS)
    parser.add_argument("--show-ik-targets", action="store_true", help="Draw the current WR1 IK target when supported.")
    parser.add_argument(
        "--locomotion-model", type=Path, default=MODEL_PATH, help="Path to the 58-D locomotion PPO zip."
    )
    parser.add_argument(
        "--locomotion-vecnormalize",
        type=Path,
        default=VECNORMALIZE_PATH,
        help="Path to the VecNormalize statistics for the locomotion policy.",
    )
    return parser


# 原 demo 强制切到 B 抓取姿态时的机身位置和 yaw.
ORIGINAL_BASE_XY = B_GRASP_ROOT_Q[:2].astype(np.float64)
ORIGINAL_BASE_YAW = math.atan2(
    2.0 * (B_GRASP_ROOT_Q[6] * B_GRASP_ROOT_Q[5] + B_GRASP_ROOT_Q[3] * B_GRASP_ROOT_Q[4]),
    1.0 - 2.0 * (B_GRASP_ROOT_Q[4] * B_GRASP_ROOT_Q[4] + B_GRASP_ROOT_Q[5] * B_GRASP_ROOT_Q[5]),
)

# 路径只保留 A/C. B_POINT 仅用于同步 base 中仍按 A->B 计算朝向的辅助函数.
A_POINT = np.array([5.0, -6.0], dtype=np.float64)
C_POINT = np.array([13.0, -1.0], dtype=np.float64)
B_POINT = C_POINT.copy()
A_YAW = layout.route_yaw(A_POINT, C_POINT)
C_STAND_POINT = layout.default_c_stand_point(A_POINT, C_POINT)

DEFAULT_STONE_XY = layout.build_c_side_stone_layout(A_POINT, C_POINT, count=1, side=layout.DEFAULT_STONE_SIDE)[0]
DEFAULT_STONE_OFFSET_XY = DEFAULT_STONE_XY - C_POINT

# 原 demo 计算出的默认石子世界坐标, 用来保留已有参数的含义参考.
ORIGINAL_STONE_XY = _stone_xy()

# mover 指尖所在 body 名称. stator 使用 ARM_EE_BODY.
FINGER_BODY = "arm_link_fngr"

# IK 跟踪点, 用 wr1 上的 stator 代理点对准石子一侧.
STATOR_IK_OFFSET = wp.vec3(0.152, 0.0, -0.045)

# 接触判定代理点, 近似 stator 和 mover 的可视接触位置.
STATOR_CONTACT_OFFSET = wp.vec3(0.232, 0.0, -0.045)
MOVER_CONTACT_OFFSET = wp.vec3(0.112, 0.0, -0.052)

# IK 姿态权重. rotation 控制长轴对齐, down-axis 控制竖直下放.
IK_ROTATION_WEIGHT = 0.35
IK_DOWN_AXIS_WEIGHT = 1.0

# 到 B 点的到达半径, 放宽一点避免在 B 附近反复修正.
GRASP_ARRIVAL_RADIUS = 0.14

# 抓取高度占石子 z 半轴的比例.
GRASP_HEIGHT_FRACTION = 0.45

# stator 下放点相对石子长轴端点向外留出的距离 [m].
GRASP_SIDE_CLEARANCE = 0.01

# IK 目标比实际接触点略高, 给下放和碰撞留余量 [m].
GRASP_IK_VERTICAL_BIAS = 0.055

# 吸附前的视觉接触误差阈值 [m]. stator 更严格, mover 略宽松.
STATOR_ATTACH_TOLERANCE = 0.055
MOVER_ATTACH_TOLERANCE = 0.065
STATOR_VISUAL_CLOSE_TOLERANCE = 0.015
MOVER_VISUAL_CLOSE_TOLERANCE = 0.02

# 视觉闭合目标. 吸附后会记录实际夹爪 q, 搬运和下放期间保持该 q.
GRIPPER_CLOSED_VISUAL = -0.5

# 石子生成后先自由稳定的时间 [s].
STONE_SETTLE_SECONDS = 1.0

# 直走 A->C 时给 locomotion policy 的内部小段目标半径 [m].
C_ROUTE_WAYPOINT_RADIUS = 0.45
C_ROUTE_STAND_RADIUS = 0.16


def _matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """把 3x3 旋转矩阵转换为 xyzw 四元数."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float32,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                ],
                dtype=np.float32,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                ],
                dtype=np.float32,
            )
    return _quat_normalize(quat)


def _yaw_with_original_tilt(yaw: float) -> np.ndarray:
    """保留原 B 抓取姿态的机身倾斜, 只替换水平 yaw."""

    original_quat = B_GRASP_ROOT_Q[3:7].astype(np.float32)
    original_yaw_quat = np.array(_yaw_to_xyzw(ORIGINAL_BASE_YAW), dtype=np.float32)
    yaw_quat = np.array(_yaw_to_xyzw(yaw), dtype=np.float32)
    original_tilt = _quat_multiply(_quat_inverse(original_yaw_quat), original_quat)
    return _quat_multiply(yaw_quat, original_tilt)


def _xy_axes_from_yaw(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """根据 yaw 返回水平面长轴和短轴方向."""

    long_axis = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return long_axis, short_axis


class RobustSpotPickPlaceDemo(SpotPickPlaceDemo):
    def __init__(self, viewer, args: argparse.Namespace):
        """初始化 robust demo, 并在原 demo 构建前覆盖路径和石子配置."""

        self.robust_stone_base_yaw = math.radians(float(args.stone_yaw_deg))
        self.robust_stone_yaw_jitter = math.radians(float(args.stone_yaw_jitter_deg))
        self.robust_stone_size = np.array([args.stone_rx, args.stone_ry, args.stone_rz], dtype=np.float64)
        self.robust_stone_count = int(args.stone_count)
        self.robust_stone_row_side = int(args.stone_row_side)
        self.robust_stone_spacing = float(args.stone_spacing)
        self.robust_ik_rotation_weight = float(args.ik_rotation_weight)
        self.robust_grasp_arrival_radius = float(args.grasp_arrival_radius)
        self.robust_grasp_height_fraction = float(args.grasp_height_fraction)
        self.robust_stator_attach_tolerance = STATOR_ATTACH_TOLERANCE
        self.robust_mover_attach_tolerance = MOVER_ATTACH_TOLERANCE
        self.robust_gripper_closed = GRIPPER_CLOSED_VISUAL
        self.robust_stator_side = float(args.stator_side)

        self.robust_stone_xy_list = layout.build_c_side_stone_layout(
            A_POINT,
            C_POINT,
            count=self.robust_stone_count,
            side=self.robust_stone_row_side,
            spacing=self.robust_stone_spacing,
        )
        self.robust_stone_yaws = layout.build_c_side_stone_yaws(
            self.robust_stone_count,
            base_yaw=self.robust_stone_base_yaw,
            yaw_jitter=self.robust_stone_yaw_jitter,
        )
        requested_first_xy = C_POINT + np.array([args.stone_x, args.stone_y], dtype=np.float64)
        self.robust_stone_xy_list += requested_first_xy - self.robust_stone_xy_list[0]
        self.robust_stone_xy = self.robust_stone_xy_list[0].copy()
        self.robust_stone_yaw = float(self.robust_stone_yaws[0])
        self.robust_stone_offset_xy = self.robust_stone_xy - C_POINT
        self.robust_base_xy = C_STAND_POINT.copy()
        self.robust_base_yaw = A_YAW
        self.c_route_targets = layout.build_route_targets(A_POINT, C_STAND_POINT)
        self.robust_base_clearance = float(B_GRASP_ROOT_Q[2] - _terrain_height_at(ORIGINAL_BASE_XY))
        self.robust_root_q = self._compute_root_q()

        self._set_robust_stone_clearance(float(args.stone_clearance))
        super().__init__(viewer, args)

        self.fngr_body_index = self._find_body_index(FINGER_BODY)
        self.stator_mj_body_id = self._find_mj_body_id(ARM_EE_BODY)
        self.mover_mj_body_id = self._find_mj_body_id(FINGER_BODY)
        self.grasp_stator_error = math.inf
        self.grasp_mover_error = math.inf
        self.grasp_stator_contact = False
        self.grasp_mover_contact = False
        self.stone_settled = False
        self.stone_settle_time = 0.0
        self.stone_grip_slide_active = False
        self.grasp_hold_gripper_q = None
        self.next_grasp_status_time = 0.0
        grasp_height = self.robust_stone_size[2] * np.clip(self.robust_grasp_height_fraction, 0.0, 0.95)
        self.stone_mesh_grasp_points = {
            stone_index: grasp_axis_surface_points(
                self.stone_mesh_vertices[stone_index], self.stone_mesh_indices[stone_index], float(grasp_height)
            )
            for stone_index in self.stone_meshes
        }
        self.reset()
        self._warm_up_arm_ik()
        print(
            "Robust grasp setup: "
            f"stone_count={self.robust_stone_count}, "
            f"stone_xy={np.round(self.robust_stone_xy_list, 3).tolist()}, "
            f"stone_yaws={np.round(self.robust_stone_yaws, 3).tolist()}, "
            f"base_xy={np.round(self.robust_base_xy, 3).tolist()}, "
            f"base_yaw={self.robust_base_yaw:.3f}"
        )

    def _compute_root_q(self) -> np.ndarray:
        """计算 B 点抓取时给 IK 使用的机身根姿态."""

        root_q = B_GRASP_ROOT_Q.copy()
        root_q[:2] = self.robust_base_xy.astype(np.float32)
        root_q[2] = np.float32(_terrain_height_at(self.robust_base_xy) + self.robust_base_clearance)
        root_q[3:7] = _yaw_with_original_tilt(self.robust_base_yaw)
        return root_q.astype(np.float32)

    def _set_robust_stone_clearance(self, stone_clearance: float) -> None:
        """Store the initial drop height used when spawning robust stones."""

        self.robust_stone_clearance = stone_clearance

    def _configure_mj_collision_filters(self) -> None:
        """Keep robust stone-stone collisions enabled for natural rover stacking."""

        super()._configure_mj_collision_filters()
        stone_contype, stone_conaffinity = layout.stone_collision_masks()
        self.solver.mj_model.geom_contype[self.stone_geom_ids] = stone_contype
        self.solver.mj_model.geom_conaffinity[self.stone_geom_ids] = stone_conaffinity
        self.solver.mj_model.geom_friction[self.stone_geom_ids] = np.array(layout.stone_contact_friction())

    def _stone_specs(self) -> tuple[StoneSpec, ...]:
        """返回 C 点附近待逐个抓取的石子配置."""

        specs = []
        for stone_index, stone_xy in enumerate(self.robust_stone_xy_list):
            stone_quat = np.array(_yaw_to_xyzw(float(self.robust_stone_yaws[stone_index])), dtype=np.float32)
            stone_pos = np.array(
                [
                    stone_xy[0],
                    stone_xy[1],
                    layout.initial_stone_center_z(
                        _terrain_height_at(stone_xy),
                        self.robust_stone_size[2],
                        self.robust_stone_clearance,
                    ),
                ],
                dtype=np.float64,
            )
            label = STONE_LABEL if self.robust_stone_count == 1 else f"{STONE_LABEL}_{stone_index}"
            specs.append(
                StoneSpec(
                    label=label,
                    pos=stone_pos,
                    quat=stone_quat.copy(),
                    size=self.robust_stone_size.copy(),
                )
            )
        return tuple(specs)

    def _set_current_stone(self, stone_index: int) -> None:
        """切换 robust 抓取逻辑当前操作的石子."""

        super()._set_current_stone(stone_index)
        self.robust_stone_xy = self.stone_pos[:2].astype(np.float64)
        self.robust_stone_yaw = float(self.robust_stone_yaws[stone_index])
        self.robust_stone_offset_xy = self.robust_stone_xy - C_POINT

    def _clear_arm_ik(self) -> None:
        """清空原 demo 的 IK 状态和 robust 额外 IK 目标."""

        super()._clear_arm_ik()
        self.ik_down_obj = None
        self.ik_long_axis_obj = None
        self.ik_final_rotation = None
        self.ik_link_offset = None
        self.ik_axis_yaw = 0.0
        self.ik_base_tf = None

    def _grasp_base_root_q(self) -> np.ndarray:
        """返回 B 点抓取阶段使用的机身根姿态."""

        return self.robust_root_q.copy()

    def _find_mj_body_id(self, name: str) -> int:
        """按名称在 MuJoCo 模型中查找唯一 body id."""

        matches = []
        for body_id in range(self.solver.mj_model.nbody):
            label = self.mujoco.mj_id2name(self.solver.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            if label == name or label.endswith(f"_{name}") or label.endswith(f"/{name}"):
                matches.append(body_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one MuJoCo body named '{name}', found {len(matches)}.")
        return matches[0]

    def _stone_side_surface_targets(self, state=None) -> tuple[np.ndarray, np.ndarray]:
        """计算 stator 和 mover 应触碰的石子长轴两侧表面点."""

        mesh_points = self.stone_mesh_grasp_points.get(self.current_stone_index)
        if mesh_points is not None:
            stone_tf = self.stone_q if state is None else state.body_q.numpy()[self.stone_body_index]
            negative_side, positive_side = mesh_points
            stator_local, mover_local = (negative_side, positive_side)
            if self.robust_stator_side > 0.0:
                stator_local, mover_local = mover_local, stator_local
            stone_rotation = _xyzw_to_matrix(stone_tf[3:7])
            return (
                (stone_tf[:3] + stone_rotation @ stator_local).astype(np.float32),
                (stone_tf[:3] + stone_rotation @ mover_local).astype(np.float32),
            )

        stone_center = (
            self.stone_pos.astype(np.float64) if state is None else state.body_q.numpy()[self.stone_body_index, :3]
        )

        long_axis, _ = _xy_axes_from_yaw(self.robust_stone_yaw)
        height_fraction = float(np.clip(self.robust_grasp_height_fraction, 0.0, 0.95))
        contact_height = self.robust_stone_size[2] * height_fraction
        side_radius = self.robust_stone_size[0] * math.sqrt(max(0.0, 1.0 - height_fraction * height_fraction))

        stator_target = stone_center.copy()
        mover_target = stone_center.copy()
        stator_target[:2] += self.robust_stator_side * long_axis * side_radius
        mover_target[:2] -= self.robust_stator_side * long_axis * side_radius
        stator_target[2] += contact_height
        mover_target[2] += contact_height
        return stator_target.astype(np.float32), mover_target.astype(np.float32)

    def _stone_stator_approach_target(self) -> np.ndarray:
        """计算 stator 下放前的水平目标点, 略微在石子外侧."""

        stator_target, _ = self._stone_side_surface_targets()
        long_axis, _ = _xy_axes_from_yaw(self.robust_stone_yaw)
        target = stator_target.astype(np.float32)
        target[:2] += (self.robust_stator_side * long_axis * GRASP_SIDE_CLEARANCE).astype(np.float32)
        return target

    def _stone_approach_target(self) -> np.ndarray:
        """计算抓取下放的最终 IK 目标, 包含竖直安全偏置."""

        stator_target = self._stone_stator_approach_target()
        return stator_target + np.array([0.0, 0.0, GRASP_IK_VERTICAL_BIAS], dtype=np.float32)

    def _wr1_rotation_target(self, yaw: float) -> np.ndarray:
        """构造 wr1 目标姿态, 让夹爪竖直下放并跟随石子长轴."""

        long_axis_xy, short_axis_xy = _xy_axes_from_yaw(yaw)
        local_x_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        local_y_world = np.array([short_axis_xy[0], short_axis_xy[1], 0.0], dtype=np.float64)
        local_z_world = np.array([long_axis_xy[0], long_axis_xy[1], 0.0], dtype=np.float64)
        rotation = np.column_stack((local_x_world, local_y_world, local_z_world))
        return _matrix_to_xyzw(rotation)

    @staticmethod
    def _down_axis_target(target: np.ndarray) -> np.ndarray:
        """生成辅助 IK 点, 约束夹爪本地 x 轴朝世界 -Z."""

        return target + np.array([0.0, 0.0, -WR1_DOWN_AXIS_LENGTH], dtype=np.float32)

    def _long_axis_target(self, target: np.ndarray) -> np.ndarray:
        """生成辅助 IK 点, 约束夹爪长轴方向."""

        long_axis, _ = _xy_axes_from_yaw(self.ik_axis_yaw)
        return target + np.array(
            [
                long_axis[0] * WR1_DOWN_AXIS_LENGTH,
                long_axis[1] * WR1_DOWN_AXIS_LENGTH,
                0.0,
            ],
            dtype=np.float32,
        )

    def _target_in_ik_base_frame(self, target: np.ndarray) -> np.ndarray:
        """把世界目标补偿到 IK 创建时的机身坐标系."""

        if self.ik_base_tf is None:
            return target.astype(np.float32)

        current_base_tf = self._arm_base_transform()
        initial_pos = self.ik_base_tf[:3].astype(np.float64)
        current_pos = current_base_tf[:3].astype(np.float64)
        initial_rot = _xyzw_to_matrix(self.ik_base_tf[3:7])
        current_rot = _xyzw_to_matrix(current_base_tf[3:7])
        local_target = current_rot.T @ (target.astype(np.float64) - current_pos)
        return (initial_pos + initial_rot @ local_target).astype(np.float32)

    def _start_arm_ik_motion(self, final_target: np.ndarray, label: str, log: bool = True) -> None:
        """创建机械臂 IK 模型和目标, 用于抓取下放或 C 点下放."""

        self.phase_time = 0.0
        self.arm_q_cmd = self.state_0.joint_q.numpy()[self.arm_q_slice].astype(np.float32)
        self.ik_base_tf = self._arm_base_transform()

        ik_builder = newton.ModelBuilder()
        ik_builder.add_mjcf(
            _build_spot_arm_ik_mjcf(self.ik_base_tf, self._arm_joint_limits()),
            up_axis="Z",
            enable_self_collisions=False,
        )
        self.ik_model = ik_builder.finalize()
        self.ik_wr1_body_index = self._find_body_index_in_labels(self.ik_model.body_label, ARM_EE_BODY)
        self.ik_joint_q = wp.array(self.arm_q_cmd.reshape(1, -1), dtype=wp.float32)

        approach_motion = label == "approach"
        self.ik_link_offset = STATOR_IK_OFFSET if approach_motion else GRIPPER_TRACK_OFFSET

        body_q = self.state_0.body_q.numpy()
        self.ik_start_target = self._link_point_position(body_q, self.wr1_body_index, self.ik_link_offset)
        self.ik_final_target = final_target
        self.ik_pregrasp_target = self.ik_final_target + np.array([0.0, 0.0, ARM_PREGRASP_HEIGHT], dtype=np.float32)
        self.current_ik_target = self.ik_start_target.copy()
        self.ik_axis_yaw = self.robust_stone_yaw if approach_motion else self._base_yaw()
        self.ik_final_rotation = self._wr1_rotation_target(self.ik_axis_yaw)

        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset,
            target_positions=wp.array([wp.vec3(*self.current_ik_target)], dtype=wp.vec3),
            weight=1.0,
        )
        self.ik_down_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset + wp.vec3(WR1_DOWN_AXIS_LENGTH, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self._down_axis_target(self.current_ik_target))], dtype=wp.vec3),
            weight=IK_DOWN_AXIS_WEIGHT,
        )
        self.ik_long_axis_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset + wp.vec3(0.0, 0.0, WR1_DOWN_AXIS_LENGTH),
            target_positions=wp.array([wp.vec3(*self._long_axis_target(self.current_ik_target))], dtype=wp.vec3),
            weight=self.robust_ik_rotation_weight,
        )
        self.ik_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=IK_LIMIT_WEIGHT,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_down_obj, self.ik_long_axis_obj, self.ik_limit_obj],
            lambda_initial=IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        link_offset_label = "stator_track" if approach_motion else "gripper_track"

        if log:
            print(
                f"Start robust arm IK {label}: "
                f"pregrasp={np.round(self.ik_pregrasp_target, 3).tolist()}, "
                f"final={np.round(self.ik_final_target, 3).tolist()}, "
                f"wr1_rot={np.round(self.ik_final_rotation, 3).tolist()}, "
                f"link_offset={link_offset_label}"
            )

    def _warm_up_arm_ik(self) -> None:
        """在场景加载时编译 IK 内核, 避免首次抓取中途停顿."""

        self._start_arm_ik_motion(self._stone_approach_target(), "approach", log=False)
        self.arm_q_cmd = self._solve_arm_ik(self.ik_start_target)
        self.reset()
        print("Arm IK kernels warmed up during scene initialization")

    def reset(self) -> None:
        """重置 demo 状态, 并恢复 robust 抓取相关标志."""

        super().reset()
        self.robust_stone_xy_list = np.array([stone_q[:2] for stone_q in self.initial_stone_qs], dtype=np.float64)
        self.robust_stone_xy = self.stone_pos[:2].astype(np.float64)
        self.robust_stone_offset_xy = self.robust_stone_xy - C_POINT
        self.grasp_stator_error = math.inf
        self.grasp_mover_error = math.inf
        self.grasp_stator_contact = False
        self.grasp_mover_contact = False
        self.stone_settled = False
        self.stone_settle_time = 0.0
        self.stone_grip_slide_active = False
        self.pin_stone_until_grasp = False
        self.grasp_hold_gripper_q = None
        self.all_stones_done = False
        self.c_route_target_index = 0
        self.next_grasp_status_time = 0.0

    def _reset_current_stone_cycle(self) -> None:
        """重置当前石子的抓取和投放阶段标志."""

        self.grasp_stator_error = math.inf
        self.grasp_mover_error = math.inf
        self.grasp_stator_contact = False
        self.grasp_mover_contact = False
        self.stone_grip_slide_active = False
        self.pin_stone_until_grasp = True
        self.stone_attached = False
        self.stone_released = False
        self.stone_grasp_offset_pos = None
        self.stone_grasp_offset_quat = None
        self.grasp_hold_gripper_q = None
        self.arm_retracted = False
        self.arm_retract_start_q = None
        self.arm_release_started = False
        self.arm_release_lowered = False
        self.release_open_start_q = None
        self.arm_release_opened = False
        self.arm_final_retracted = False
        self.arm_final_retract_start_q = None
        self.next_grasp_status_time = 0.0
        self._clear_arm_ik()

    def _advance_to_next_stone(self) -> None:
        """当前石子投放完成后切换到下一个石子."""

        next_index = self.current_stone_index + 1
        if next_index >= len(self.stone_specs):
            self.all_stones_done = True
            print(f"All {len(self.stone_specs)} stones released into the rover")
            return

        self._set_current_stone(next_index)
        self._reset_current_stone_cycle()
        print(
            "Advancing to next stone: "
            f"index={self.current_stone_index + 1}/{len(self.stone_specs)}, "
            f"pos={np.round(self.stone_pos, 3).tolist()}"
        )

    def _apply_c_route_policy(self) -> None:
        """沿 A->C 直线内部目标推进到 C 附近."""

        route_index = min(self.c_route_target_index, len(self.c_route_targets) - 1)
        target_xy = self.c_route_targets[route_index]
        final_target = route_index == len(self.c_route_targets) - 1
        if final_target:
            stand_distance = float(np.linalg.norm(self._base_xy() - C_STAND_POINT))
            if stand_distance <= C_ROUTE_STAND_RADIUS:
                print(
                    "Reached C grasp stand: "
                    f"position={self._base_xy().round(3).tolist()}, "
                    f"target={np.round(C_STAND_POINT, 3).tolist()}, "
                    f"command_target={target_xy.tolist()}"
                )
                self.command.fill(0.0)
                self._write_arm_ctrl(self.stand_ctrl[ACT_DIM:])
                self.c_route_target_index += 1
                self.reached_c = True
                return

            self.command, _, _ = self._command_to_target(
                target_xy,
                None,
                MIN_FORWARD_SPEED,
                0.0,
            )
            action, _ = self.policy.predict(self._observation(), deterministic=True)
            policy_action = np.clip(action[0], -1.0, 1.0).astype(np.float32)
            self.last_action = policy_action
            leg_ctrl = np.clip(
                self.nominal_leg_ctrl + policy_action * ACTION_SCALE,
                self.leg_ctrl_low,
                self.leg_ctrl_high,
            )
            self._write_ctrl(leg_ctrl, self.stand_ctrl[ACT_DIM:])
            self._advance_gait_phase()
            return

        arrival_radius = C_ROUTE_WAYPOINT_RADIUS
        target_label = f"C route {route_index + 1}/{len(self.c_route_targets)}"
        reached = self._apply_policy(
            target_xy,
            target_label,
            self.stand_ctrl[ACT_DIM:],
            face_xy=self.stone_pos[:2] if final_target else None,
            arrival_radius=arrival_radius,
            require_yaw=False,
        )
        if not reached:
            return

        self.c_route_target_index += 1
        if final_target:
            self.reached_c = True

    def _grasp_hold_q(self) -> float:
        """返回吸附瞬间记录的夹爪开合量."""

        if self.grasp_hold_gripper_q is None:
            return self.robust_gripper_closed
        return self.grasp_hold_gripper_q

    def _gripper_target_at_time(self) -> float:
        """根据抓取阶段时间计算夹爪目标."""

        if self.stone_attached:
            return self._grasp_hold_q()

        close_time = self.phase_time - ARM_PREGRASP_SECONDS - ARM_DESCENT_SECONDS
        if close_time <= 0.0:
            return GRIPPER_OPEN

        alpha = _smoothstep(close_time / max(GRIPPER_CLOSE_SECONDS, 1.0e-6))
        return float((1.0 - alpha) * GRIPPER_OPEN + alpha * self.robust_gripper_closed)

    def _solve_arm_ik(self, target: np.ndarray) -> np.ndarray:
        """更新 IK 目标并求解下一帧机械臂控制量."""

        ik_target = self._target_in_ik_base_frame(target)
        ik_down_target = self._target_in_ik_base_frame(self._down_axis_target(target))
        ik_long_axis_target = self._target_in_ik_base_frame(self._long_axis_target(target))
        self.ik_pos_obj.set_target_position(0, wp.vec3(*ik_target))
        self.ik_down_obj.set_target_position(0, wp.vec3(*ik_down_target))
        self.ik_long_axis_obj.set_target_position(0, wp.vec3(*ik_long_axis_target))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=IK_ITERATIONS,
            step_size=IK_STEP_SIZE,
        )
        candidate_q = self.ik_joint_q.numpy()[0].astype(np.float32)
        candidate_q[-1] = self._gripper_target_at_time()
        arm_q = self._limit_arm_joint_step(candidate_q)
        self.ik_joint_q.assign(arm_q.reshape(1, -1))
        return arm_q

    def _close_gripper_without_ik(self) -> np.ndarray:
        """保持机械臂关节不动, 仅更新夹爪闭合目标."""

        arm_q = self.arm_q_cmd.copy()
        arm_q[-1] = self._gripper_target_at_time()
        return self._limit_arm_joint_step(arm_q)

    def _arm_stowed_q(self) -> np.ndarray:
        """返回收臂姿态, 搬运时保持吸附瞬间夹爪 q."""

        arm_q = super()._arm_stowed_q()
        if self.stone_attached:
            arm_q[-1] = self._grasp_hold_q()
        return arm_q

    def _apply_arm_retract(self) -> None:
        """抓起石子后收臂, Spot 保持在 C 点附近准备投放."""

        if self.arm_retract_start_q is None:
            self._start_arm_retract()

        self.command.fill(0.0)
        self.arm_q_cmd = self._arm_retract_q_at_time()
        self._write_arm_ctrl(self.arm_q_cmd)
        self.phase_time += self.frame_dt

        if self.phase_time >= ARM_RETRACT_SECONDS:
            self.arm_q_cmd = self._arm_stowed_q()
            self._write_arm_ctrl(self.arm_q_cmd)
            self.arm_retracted = True
            print(f"Arm retracted with stone {self.current_stone_index + 1}; ready to place into rover")

    def _stone_release_target(self) -> np.ndarray:
        """把石子下放到月球车货斗内."""

        place_xy = layout.stone_release_xy(
            C_POINT,
            A_YAW,
            stone_index=self.current_stone_index,
            stone_count=len(self.stone_specs),
        )
        place_z = layout.release_stone_center_z(_terrain_height_at(place_xy), self.robust_stone_size[2])
        return np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)

    def _attach_stone_to_gripper(self, state) -> None:
        """把石子吸附到夹爪, 并记录当前夹爪开合量."""

        self.grasp_hold_gripper_q = float(self.arm_q_cmd[-1])
        super()._attach_stone_to_gripper(state)
        print(f"Holding gripper at grasp q={self.grasp_hold_gripper_q:.3f}")

    def _release_stone(self, state) -> None:
        """释放石子并清空吸附期间保存的夹爪开合量."""

        super()._release_stone(state)
        self.grasp_hold_gripper_q = None

    def _release_stone_pin_for_grip(self) -> None:
        """夹爪开始闭合时释放水平约束, 让碰撞把石子推到双侧接触."""

        if not self.pin_stone_until_grasp:
            return
        self.pin_stone_until_grasp = False
        self.stone_grip_slide_active = True
        self.state_0.joint_qd[self.stone_qd_slice].zero_()
        self.state_1.joint_qd[self.stone_qd_slice].zero_()
        print("Stone horizontal slide released; closing gripper until both jaws touch")

    def _update_stone_constraint(self, state, update_fk: bool = True) -> None:
        """根据当前阶段更新石子约束状态."""

        if not self.stone_settled:
            for stone_index, stone_q_slice in enumerate(self.stone_q_slices):
                stone_q = state.joint_q.numpy()[stone_q_slice].copy()
                planned_q = self.stone_qs[stone_index]
                stone_q[:2] = planned_q[:2]
                stone_q[3:7] = planned_q[3:7]
                state.joint_q[stone_q_slice].assign(stone_q)
                stone_qd = state.joint_qd.numpy()[self.stone_qd_slices[stone_index]].copy()
                stone_qd[:2] = 0.0
                stone_qd[3:] = 0.0
                state.joint_qd[self.stone_qd_slices[stone_index]].assign(stone_qd)
            if update_fk:
                newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            return

        needs_fk = False
        if self.stone_attached:
            self._attach_stone_pose(state, update_fk=False)
            needs_fk = True
        elif self.pin_stone_until_grasp:
            state.joint_q[self.stone_q_slice].assign(self.stone_q)
            state.joint_qd[self.stone_qd_slice].zero_()
            needs_fk = True
        elif self.stone_grip_slide_active and not self.stone_released:
            stone_q = state.joint_q.numpy()[self.stone_q_slice].copy()
            stone_q[2] = self.stone_q[2]
            stone_q[3:7] = self.stone_q[3:7]
            state.joint_q[self.stone_q_slice].assign(stone_q)
            stone_qd = state.joint_qd.numpy()[self.stone_qd_slice].copy()
            stone_qd[2:] = 0.0
            state.joint_qd[self.stone_qd_slice].assign(stone_qd)
            needs_fk = True

        for stone_index in range(len(self.stone_specs)):
            if stone_index == self.current_stone_index or self.stone_released_flags[stone_index]:
                continue
            state.joint_q[self.stone_q_slices[stone_index]].assign(self.stone_qs[stone_index])
            state.joint_qd[self.stone_qd_slices[stone_index]].zero_()
            needs_fk = True

        if needs_fk and update_fk:
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _finish_stone_settle(self) -> None:
        """结束石子自由稳定阶段, 记录真实落点并固定到抓取前."""

        for stone_index in range(len(self.stone_qs)):
            settled_q = self.stone_qs[stone_index].copy().astype(np.float32)
            stone_xy = settled_q[:2].astype(np.float64)
            settled_q[2] = np.float32(
                layout.settled_stone_center_z(_terrain_height_at(stone_xy), self.robust_stone_size[2])
            )
            self.stone_qs[stone_index] = settled_q
            self.robust_stone_xy_list[stone_index] = stone_xy

        self._set_current_stone(0)
        self.pin_stone_until_grasp = True
        self.stone_settled = True
        for state in (self.state_0, self.state_1):
            for stone_index, stone_q in enumerate(self.stone_qs):
                state.joint_q[self.stone_q_slices[stone_index]].assign(stone_q)
                state.joint_qd[self.stone_qd_slices[stone_index]].zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        print(
            "Stones settled: "
            f"positions={np.round([q[:3] for q in self.stone_qs], 3).tolist()}, "
            f"a={np.round(A_POINT, 3).tolist()}, "
            f"c={np.round(C_POINT, 3).tolist()}, "
            f"a_yaw={A_YAW:.3f}"
        )

    def _apply_arm_approach(self) -> None:
        """执行 C 点附近当前石子的抓取动作, 包括移动到目标, 下放和闭合夹爪."""

        if self.ik_solver is None:
            self._start_arm_approach()

        self.command.fill(0.0)
        closing_gripper = self.phase_time >= ARM_PREGRASP_SECONDS + ARM_DESCENT_SECONDS
        if closing_gripper:
            self._release_stone_pin_for_grip()
        self.current_ik_target = self._arm_target_at_time()
        if closing_gripper:
            self.arm_q_cmd = self._close_gripper_without_ik()
        else:
            self.arm_q_cmd = self._solve_arm_ik(self.current_ik_target)
        self._write_arm_ctrl(self.arm_q_cmd)

        body_q = self.state_0.body_q.numpy()
        actual = self._link_point_position(body_q, self.wr1_body_index, self.ik_link_offset)
        self.ik_error = float(np.linalg.norm(actual - self.current_ik_target))

        self.phase_time += self.frame_dt

    def _grasp_contact_ready(self, state) -> bool:
        """判断 stator 和 mover 是否都已接近目标且真实碰撞."""

        close_started = self.phase_time >= ARM_PREGRASP_SECONDS + ARM_DESCENT_SECONDS
        if not close_started:
            return False

        body_q = state.body_q.numpy()
        stator_point = self._link_point_position(body_q, self.wr1_body_index, STATOR_CONTACT_OFFSET)
        mover_point = self._link_point_position(body_q, self.fngr_body_index, MOVER_CONTACT_OFFSET)
        stator_target, mover_target = self._stone_side_surface_targets(state)
        stator_delta = stator_point - stator_target
        mover_delta = mover_point - mover_target
        self.grasp_stator_error = float(np.linalg.norm(stator_point - stator_target))
        self.grasp_mover_error = float(np.linalg.norm(mover_point - mover_target))
        self.grasp_stator_contact, self.grasp_mover_contact = self._stone_jaw_contact_flags()

        if self.sim_time >= self.next_grasp_status_time:
            print(
                "Waiting for jaw contact: "
                f"stator_err={self.grasp_stator_error:.3f} m, "
                f"stator_delta={np.round(stator_delta, 3).tolist()}, "
                f"stator_contact={self.grasp_stator_contact}, "
                f"mover_err={self.grasp_mover_error:.3f} m, "
                f"mover_delta={np.round(mover_delta, 3).tolist()}, "
                f"mover_contact={self.grasp_mover_contact}"
            )
            self.next_grasp_status_time = self.sim_time + 1.0

        stator_ready = self.grasp_stator_error <= self.robust_stator_attach_tolerance and (
            self.grasp_stator_contact or self.grasp_stator_error <= STATOR_VISUAL_CLOSE_TOLERANCE
        )
        mover_ready = self.grasp_mover_error <= self.robust_mover_attach_tolerance and (
            self.grasp_mover_contact or self.grasp_mover_error <= MOVER_VISUAL_CLOSE_TOLERANCE
        )
        contact_ready = stator_ready and mover_ready
        if contact_ready:
            print(
                "Jaw contact ready: "
                f"stator_err={self.grasp_stator_error:.3f} m, "
                f"stator_contact={self.grasp_stator_contact}, "
                f"mover_err={self.grasp_mover_error:.3f} m, "
                f"mover_contact={self.grasp_mover_contact}"
            )
        return contact_ready

    def _stone_jaw_contact_flags(self) -> tuple[bool, bool]:
        """从 MuJoCo contact 中读取石子是否碰到两个夹爪侧."""

        stator_contact = False
        mover_contact = False
        geom_body_ids = self.solver.mj_model.geom_bodyid
        for i in range(self.solver.mj_data.ncon):
            contact = self.solver.mj_data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.stone_geom_id:
                other_body = int(geom_body_ids[geom2])
            elif geom2 == self.stone_geom_id:
                other_body = int(geom_body_ids[geom1])
            else:
                continue
            if other_body == self.stator_mj_body_id:
                stator_contact = True
            elif other_body == self.mover_mj_body_id:
                mover_contact = True
        return stator_contact, mover_contact

    def step(self) -> None:
        """推进一帧仿真, 并按阶段执行行走, 抓取, 搬运或放下."""

        if not self.stone_settled:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.stand_ctrl[ACT_DIM:])
        elif not self.reached_c:
            self._apply_c_route_policy()
        elif self.reached_c and not self.b_aligned:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.stand_ctrl[ACT_DIM:])
            self.b_aligned = True
            print(
                "Standing at C grasp stand without forced pose: "
                f"position={self._base_xy().round(3).tolist()}, "
                f"target={np.round(C_STAND_POINT, 3).tolist()}, "
                f"yaw={self._base_yaw():.3f}"
            )
        elif self.all_stones_done:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.arm_q_cmd)
        elif not self.stone_attached and not self.stone_released:
            self._apply_arm_approach()
        elif not self.arm_retracted:
            self._apply_arm_retract()
        elif not self.arm_release_opened:
            self._apply_arm_release()
        elif not self.arm_final_retracted:
            self._apply_final_arm_retract()
        elif not self.all_stones_done:
            self._advance_to_next_stone()
        else:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.arm_q_cmd)

        for substep in range(CONTROL_DECIMATION):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_stone_constraint(
                self.state_0,
                update_fk=substep == CONTROL_DECIMATION - 1,
            )
        if (
            self.stone_settled
            and not self.stone_attached
            and not self.stone_released
            and self._grasp_contact_ready(self.state_0)
        ):
            self._attach_stone_to_gripper(self.state_0)
        if not self.stone_settled:
            self.stone_settle_time += self.frame_dt
            if self.stone_settle_time >= STONE_SETTLE_SECONDS:
                self._finish_stone_settle()
        self.sim_time += self.frame_dt
        self.contacts_ready = True

    def test_final(self) -> None:
        """示例结束时检查数值稳定性, IK 误差和吸附时机."""

        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise ValueError("Body transforms became non-finite.")
        if self.ik_solver is not None and self.phase_time > 0.5:
            gripper_low = min(GRIPPER_OPEN, self.robust_gripper_closed) - 0.1
            gripper_high = max(GRIPPER_OPEN, self.robust_gripper_closed) + 0.1
            if not (gripper_low <= self.arm_q_cmd[-1] <= gripper_high):
                raise ValueError(f"Gripper target is out of range: q={self.arm_q_cmd[-1]:.3f}")
            if self.ik_error > 0.45:
                raise ValueError(f"Arm IK target error is too high: {self.ik_error:.3f} m")
        if self.stone_attached and (
            self.grasp_stator_error > self.robust_stator_attach_tolerance
            or self.grasp_mover_error > self.robust_mover_attach_tolerance
        ):
            raise ValueError(
                "Stone attached before both jaws reached visual contact: "
                f"stator={self.grasp_stator_error:.3f}, mover={self.grasp_mover_error:.3f}"
            )


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数, 保留原 demo 参数并追加 robust 抓取配置."""

    parser = _create_base_parser()
    parser.description = "Run the Spot pick-and-place demo with a dynamic, yaw-aware visual grasp."
    parser.add_argument(
        "--stone-x",
        type=float,
        default=float(DEFAULT_STONE_OFFSET_XY[0]),
        help="First stone world-frame x offset from C. The default cluster shifts with this stone.",
    )
    parser.add_argument(
        "--stone-y",
        type=float,
        default=float(DEFAULT_STONE_OFFSET_XY[1]),
        help="First stone world-frame y offset from C. The default cluster shifts with this stone.",
    )
    parser.add_argument("--stone-count", type=int, default=layout.DEFAULT_STONE_COUNT, help="Number of stones near C.")
    parser.add_argument(
        "--stone-row-side",
        type=int,
        choices=(-1, 1),
        default=layout.DEFAULT_STONE_SIDE,
        help="Which side of the rover row receives the C-side stones.",
    )
    parser.add_argument(
        "--stone-spacing",
        type=float,
        default=layout.DEFAULT_STONE_SPACING,
        help="Minimum center distance between randomized stones in the C-stand front-side cluster.",
    )
    parser.add_argument("--stone-yaw-deg", type=float, default=0.0, help="Base stone long-axis yaw in degrees.")
    parser.add_argument(
        "--stone-yaw-jitter-deg",
        type=float,
        default=math.degrees(layout.DEFAULT_STONE_YAW_JITTER),
        help="Deterministic per-stone yaw jitter range in degrees around --stone-yaw-deg.",
    )
    parser.add_argument("--stone-rx", type=float, default=float(STONE_SIZE[0]), help="Stone ellipsoid x radius.")
    parser.add_argument("--stone-ry", type=float, default=float(STONE_SIZE[1]), help="Stone ellipsoid y radius.")
    parser.add_argument("--stone-rz", type=float, default=float(STONE_SIZE[2]), help="Stone ellipsoid z radius.")
    parser.add_argument(
        "--stone-clearance",
        type=float,
        default=layout.DEFAULT_STONE_DROP_HEIGHT,
        help="Initial drop height added above terrain height plus stone rz before settling.",
    )
    parser.add_argument(
        "--grasp-arrival-radius",
        type=float,
        default=GRASP_ARRIVAL_RADIUS,
        help="Locomotion arrival radius for the final grasp stand point.",
    )
    parser.add_argument(
        "--grasp-height-fraction",
        type=float,
        default=GRASP_HEIGHT_FRACTION,
        help="Contact height as a fraction of stone rz above the ellipsoid center.",
    )
    parser.add_argument(
        "--ik-rotation-weight",
        type=float,
        default=IK_ROTATION_WEIGHT,
        help="Weight for matching the WR1 orientation to the stone yaw.",
    )
    parser.add_argument(
        "--stator-side",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="Which long-axis side the fixed jaw approaches first.",
    )
    return parser


def main() -> None:
    """运行 robust grasp 示例."""

    viewer, args = newton.examples.init(create_parser())
    demo = RobustSpotPickPlaceDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
