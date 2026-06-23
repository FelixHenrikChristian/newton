from __future__ import annotations

import argparse
import io
import math
import os
import pathlib
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3.common.base_class as sb3_base_class
import stable_baselines3.common.save_util as sb3_save_util
import torch as th
import warp as wp
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import newton
import newton.examples
import newton.ik as ik

SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_PATH = SCRIPT_DIR / "spot_scene.xml"
MODEL_PATH = SCRIPT_DIR / "runs" / "spot_go2_transfer_3p5m" / "best_eval" / "best_model.zip"
VECNORMALIZE_PATH = MODEL_PATH.parent / "best_vecnormalize.pkl"

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

OBS_DIM = 54
ACT_DIM = 12
ARM_ACTUATOR_COUNT = 7

# The full scene keeps Newton's imported local leg order (FL, FR, HL, HR), but
# the transfer policy was trained with the original Go2 order (FR, FL, HR, HL).
POLICY_FROM_LOCAL = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)
LOCAL_FROM_POLICY = np.argsort(POLICY_FROM_LOCAL)
POLICY_CONTACT_FROM_LOCAL = np.array([1, 0, 3, 2], dtype=np.int32)

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
IK_ITERATIONS = 32
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
C_RELEASE_BOX_LABEL = "c_release_box"
C_RELEASE_BOX_INNER_HALF_EXTENTS = np.array([0.24, 0.22], dtype=np.float64)
C_RELEASE_BOX_WALL_THICKNESS = 0.04
C_RELEASE_BOX_WALL_HEIGHT = 0.10
C_RELEASE_BOX_WALL_DEPTH = 0.04

ACTION_SCALE = np.array((0.125, 0.55, 0.55) * 4, dtype=np.float32)
GAIT_CYCLE_SECONDS = 0.5
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
        self.policy_step_count = 0
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
        self.nominal_leg_ctrl = np.array((0.0, -0.1, 0.3) * 4, dtype=np.float32)
        self.nominal_leg_qpos = (stand_qpos[7 : 7 + ACT_DIM] + self.nominal_leg_ctrl).astype(np.float32)
        self.reset_leg_noise = np.random.default_rng(RESET_SEED).uniform(-0.05, 0.05, ACT_DIM).astype(np.float32)
        self.arm_qpos = stand_qpos[7 + ACT_DIM :].astype(np.float32)
        self.arm_q_cmd = self.arm_qpos.copy()
        self.stone_pos = _stone_position()
        self.stone_q = np.array((*self.stone_pos, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
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

        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(SCENE_PATH.resolve()),
            up_axis="Z",
            enable_self_collisions=True,
            ctrl_direct=True,
        )
        stone_cfg = newton.ModelBuilder.ShapeConfig(density=1800.0, mu=1.5, mu_torsional=0.08, mu_rolling=0.08)
        stone_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.stone_pos), wp.quat_identity()), label=STONE_LABEL
        )
        builder.add_shape_ellipsoid(
            stone_body,
            rx=float(STONE_SIZE[0]),
            ry=float(STONE_SIZE[1]),
            rz=float(STONE_SIZE[2]),
            cfg=stone_cfg,
            color=wp.vec3(0.18, 0.17, 0.15),
            label=STONE_LABEL,
        )
        self.release_box_shape_indices = self._add_c_release_box(builder)

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
        self.stone_geom_id = self._find_mj_geom_id(STONE_LABEL)
        self._configure_mj_collision_filters()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.root_q_slice, self.root_qd_slice = self._find_joint_slices(("freejoint",), q_width=7, qd_width=6)
        self.leg_q_slice, self.leg_qd_slice = self._find_joint_slices(LEG_JOINTS)
        self.arm_q_slice, self.arm_qd_slice = self._find_joint_slices(ARM_JOINTS)
        self.stone_q_slice, self.stone_qd_slice = self._find_joint_slices(
            (f"{STONE_LABEL}_free_joint",),
            q_width=7,
            qd_width=6,
        )
        self.spot_body_index = self._find_body_index("body")
        self.wr1_body_index = self._find_body_index(ARM_EE_BODY)
        self.stone_body_index = self._find_body_index(STONE_LABEL)

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
        matches = []
        for geom_id in range(self.solver.mj_model.ngeom):
            label = self.mujoco.mj_id2name(self.solver.mj_model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if label == token or label.startswith(f"{token}_") or label.endswith(f"/{token}") or f"/{token}_" in label:
                matches.append(geom_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one MuJoCo geom containing '{token}', found {len(matches)}.")
        return matches[0]

    def _find_body_index(self, name: str) -> int:
        return self._find_body_index_in_labels(self.model.body_label, name)

    @staticmethod
    def _find_body_index_in_labels(labels, name: str) -> int:
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named '{name}', found {len(matches)}.")
        return matches[0]

    @staticmethod
    def _add_c_release_box(builder: newton.ModelBuilder) -> list[int]:
        center = _c_release_box_center()
        inner_hx, inner_hy = C_RELEASE_BOX_INNER_HALF_EXTENTS
        thickness = C_RELEASE_BOX_WALL_THICKNESS
        half_thickness = thickness * 0.5
        half_height = (C_RELEASE_BOX_WALL_HEIGHT + C_RELEASE_BOX_WALL_DEPTH) * 0.5
        wall_z = _terrain_height_at(center) + (C_RELEASE_BOX_WALL_HEIGHT - C_RELEASE_BOX_WALL_DEPTH) * 0.5
        wall_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=1.5,
            mu_torsional=0.03,
            mu_rolling=0.01,
        )
        wall_color = wp.vec3(0.34, 0.30, 0.22)
        wall_specs = (
            (
                "east",
                np.array([center[0] + inner_hx + half_thickness, center[1]], dtype=np.float64),
                half_thickness,
                inner_hy + thickness,
            ),
            (
                "west",
                np.array([center[0] - inner_hx - half_thickness, center[1]], dtype=np.float64),
                half_thickness,
                inner_hy + thickness,
            ),
            (
                "north",
                np.array([center[0], center[1] + inner_hy + half_thickness], dtype=np.float64),
                inner_hx,
                half_thickness,
            ),
            (
                "south",
                np.array([center[0], center[1] - inner_hy - half_thickness], dtype=np.float64),
                inner_hx,
                half_thickness,
            ),
        )

        shape_indices = []
        for suffix, xy, hx, hy in wall_specs:
            shape_indices.append(
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(float(xy[0]), float(xy[1]), float(wall_z)), wp.quat_identity()),
                    hx=float(hx),
                    hy=float(hy),
                    hz=float(half_height),
                    cfg=wall_cfg,
                    color=wall_color,
                    label=f"{C_RELEASE_BOX_LABEL}_{suffix}",
                )
            )
        return shape_indices

    def _configure_mj_collision_filters(self) -> None:
        robot_mask = np.ones(self.solver.mj_model.ngeom, dtype=bool)
        robot_mask[[self.terrain_geom_id, self.stone_geom_id]] = False

        self.solver.mj_model.geom_contype[robot_mask] = 2
        self.solver.mj_model.geom_conaffinity[robot_mask] = 5
        self.solver.mj_model.geom_contype[self.terrain_geom_id] = 1
        self.solver.mj_model.geom_conaffinity[self.terrain_geom_id] = 6
        self.solver.mj_model.geom_contype[self.stone_geom_id] = 4
        self.solver.mj_model.geom_conaffinity[self.stone_geom_id] = 3
        self.solver.mj_model.geom_condim[self.stone_geom_id] = 6
        self.solver.mj_model.geom_priority[self.stone_geom_id] = 2

    def _apply_training_actuator_scale(self) -> None:
        gain = self.model.mujoco.actuator_gainprm.numpy()
        bias = self.model.mujoco.actuator_biasprm.numpy()
        gain[:ACT_DIM, 0] *= 3.0
        bias[:ACT_DIM, 0] *= 3.0
        bias[:ACT_DIM, 1] *= 3.0
        bias[:ACT_DIM, 2] *= 3.0**0.5
        self.model.mujoco.actuator_gainprm.assign(gain)
        self.model.mujoco.actuator_biasprm.assign(bias)

    def reset(self) -> None:
        self._write_reset_state(self.state_0)
        self._write_reset_state(self.state_1)
        self._write_stand_ctrl()
        self.sim_time = 0.0
        self.phase_time = 0.0
        self.last_action.fill(0.0)
        self.policy_step_count = 0
        self.reached_b = False
        self.reached_c = False
        self.b_aligned = False
        self.b_align_time = 0.0
        self.b_align_start_root_q = None
        self.b_align_start_leg_q = None
        self.b_align_target_root_q = None
        self.contacts_ready = False
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
        state.joint_q[self.stone_q_slice].assign(self.stone_q)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _pin_stone_to_start(self, state) -> None:
        state.joint_q[self.stone_q_slice].assign(self.stone_q)
        state.joint_qd[self.stone_qd_slice].zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

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

    def _attach_stone_pose(self, state) -> None:
        body_q = state.body_q.numpy()
        gripper_tf = body_q[self.wr1_body_index]
        gripper_rot = _xyzw_to_matrix(gripper_tf[3:7])
        stone_pos = gripper_tf[:3] + gripper_rot @ self.stone_grasp_offset_pos
        stone_quat = _quat_multiply(gripper_tf[3:7], self.stone_grasp_offset_quat)
        state.joint_q[self.stone_q_slice].assign((*stone_pos, *stone_quat))
        state.joint_qd[self.stone_qd_slice].zero_()
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
        self.policy_step_count = 0
        self._write_ctrl(self.nominal_leg_ctrl, arm_q)

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

    def _gait_clock(self) -> np.ndarray:
        phase = (self.policy_step_count * self.frame_dt / GAIT_CYCLE_SECONDS) % 1.0
        angle = 2.0 * math.pi * phase
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

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
        obs = np.concatenate(
            [
                projected_gravity,
                base_linear,
                base_angular,
                self.command,
                self._gait_clock(),
                joint_pos[POLICY_FROM_LOCAL],
                joint_qd[POLICY_FROM_LOCAL],
                self.last_action,
                self._foot_contacts()[POLICY_CONTACT_FROM_LOCAL],
            ]
        ).astype(np.float32)
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
        local_action = policy_action[LOCAL_FROM_POLICY]
        leg_ctrl = np.clip(self.nominal_leg_ctrl + local_action * ACTION_SCALE, self.leg_ctrl_low, self.leg_ctrl_high)
        self._write_ctrl(leg_ctrl, arm_q)
        self.policy_step_count += 1
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


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Run the Spot A-to-B pickup setup through Newton with SolverMuJoCo."
    parser.set_defaults(num_frames=100000, viewer="gl")
    parser.add_argument("--show-ik-targets", action="store_true", help="Draw the current WR1 IK target when supported.")
    parser.add_argument(
        "--locomotion-model", type=Path, default=MODEL_PATH, help="Path to the 54-D locomotion PPO zip."
    )
    parser.add_argument(
        "--locomotion-vecnormalize",
        type=Path,
        default=VECNORMALIZE_PATH,
        help="Path to the VecNormalize statistics for the locomotion policy.",
    )
    return parser


def main() -> None:
    viewer, args = newton.examples.init(create_parser())
    demo = SpotPickPlaceDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
