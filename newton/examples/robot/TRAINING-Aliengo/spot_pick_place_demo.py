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


SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_PATH = SCRIPT_DIR / "spot_scene.xml"
MODEL_PATH = SCRIPT_DIR / "runs" / "spot_go2_style_20m" / "best_eval" / "best_model.zip"
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

OBS_DIM = 49
ACT_DIM = 12
ARM_ACTUATOR_COUNT = 7

A_POINT = np.array([5.0, -6.0], dtype=np.float64)
B_POINT = np.array([9.0, -3.5], dtype=np.float64)
A_YAW = 0.0
RESET_BASE_HEIGHT = 1.80

STONE_LABEL = "pickup_stone"
STONE_FORWARD_OFFSET = 0.55
STONE_HEADING_OFFSET = 0.0
STONE_SIZE = np.array([0.06, 0.045, 0.035], dtype=np.float64)
STONE_CLEARANCE = -0.035
STONE_ALIGN_RADIUS = 1.2

ARRIVAL_RADIUS = 0.15
ARRIVAL_YAW_TOLERANCE = 0.35
MAX_SECONDS = 60.0
FORWARD_SPEED = 0.45
MIN_FORWARD_SPEED = 0.25
MAX_YAW_RATE = 0.5
YAW_GAIN = 1.4

ACTION_SCALE = 0.25
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


def _approach_yaw() -> float:
    direction = B_POINT - A_POINT
    if float(np.linalg.norm(direction)) < 1e-6:
        return A_YAW
    return float(math.atan2(direction[1], direction[0]))


def _stone_xy() -> np.ndarray:
    yaw = _approach_yaw() + STONE_HEADING_OFFSET
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    return B_POINT + forward * STONE_FORWARD_OFFSET


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


class SpotPickPlaceDemo:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / CONTROL_DECIMATION
        self.sim_time = 0.0
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.reached_b = False
        self.contacts_ready = False

        stand_qpos, stand_ctrl = _scene_keyframe()
        self.stand_ctrl = stand_ctrl.astype(np.float32)
        self.nominal_leg_ctrl = np.array((0.0, -0.1, 0.3) * 4, dtype=np.float32)
        self.nominal_leg_qpos = (stand_qpos[7 : 7 + ACT_DIM] + self.nominal_leg_ctrl).astype(np.float32)
        self.reset_leg_noise = np.random.default_rng(RESET_SEED).uniform(-0.05, 0.05, ACT_DIM).astype(np.float32)
        self.arm_qpos = stand_qpos[7 + ACT_DIM :].astype(np.float32)
        self.stone_pos = _stone_position()
        self.stone_q = np.array((*self.stone_pos, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        self.pin_stone_until_grasp = True

        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(SCENE_PATH.resolve()),
            up_axis="Z",
            enable_self_collisions=True,
            ctrl_direct=True,
        )
        stone_cfg = newton.ModelBuilder.ShapeConfig(density=1800.0, mu=1.5, mu_torsional=0.03, mu_rolling=0.01)
        stone_body = builder.add_body(xform=wp.transform(wp.vec3(*self.stone_pos), wp.quat_identity()), label=STONE_LABEL)
        builder.add_shape_ellipsoid(
            stone_body,
            rx=float(STONE_SIZE[0]),
            ry=float(STONE_SIZE[1]),
            rz=float(STONE_SIZE[2]),
            cfg=stone_cfg,
            color=wp.vec3(0.18, 0.17, 0.15),
            label=STONE_LABEL,
        )

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
        self.arm_q_slice, self.arm_qd_slice = self._find_joint_slices((), q_start=self.leg_q_slice.stop, qd_start=self.leg_qd_slice.stop)
        self.stone_q_slice, self.stone_qd_slice = self._find_joint_slices(
            (f"{STONE_LABEL}_free_joint",),
            q_width=7,
            qd_width=6,
        )

        ctrl_range = self.model.mujoco.actuator_ctrlrange.numpy().astype(np.float32)
        self.leg_ctrl_low = ctrl_range[:ACT_DIM, 0]
        self.leg_ctrl_high = ctrl_range[:ACT_DIM, 1]

        patch_sb3_zip_loader()
        self.vecnormalize = VecNormalize.load(VECNORMALIZE_PATH, DummyVecEnv([lambda: _VecNormalizeEnv()]))
        self.vecnormalize.training = False
        self.vecnormalize.norm_reward = False
        self.policy = PPO.load(MODEL_PATH, env=self.vecnormalize)

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.reset()
        print(f"Stone position: {self.stone_pos.round(3).tolist()}")

    def _find_joint_slices(
        self,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
        q_start: int | None = None,
        qd_start: int | None = None,
    ) -> tuple[slice, slice]:
        if not names:
            q_end = self.stone_q_slice.start if hasattr(self, "stone_q_slice") else q_start + ARM_ACTUATOR_COUNT
            qd_end = self.stone_qd_slice.start if hasattr(self, "stone_qd_slice") else qd_start + ARM_ACTUATOR_COUNT
            return slice(q_start, q_end), slice(qd_start, qd_end)

        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        joint_indices = []
        for name in names:
            matches = [i for i, label in enumerate(self.model.joint_label) if label == name or label.endswith(f"/{name}")]
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

    def _configure_mj_collision_filters(self) -> None:
        robot_mask = np.ones(self.solver.mj_model.ngeom, dtype=bool)
        robot_mask[[self.terrain_geom_id, self.stone_geom_id]] = False

        self.solver.mj_model.geom_contype[robot_mask] = 2
        self.solver.mj_model.geom_conaffinity[robot_mask] = 5
        self.solver.mj_model.geom_contype[self.terrain_geom_id] = 1
        self.solver.mj_model.geom_conaffinity[self.terrain_geom_id] = 6
        self.solver.mj_model.geom_contype[self.stone_geom_id] = 4
        self.solver.mj_model.geom_conaffinity[self.stone_geom_id] = 3

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
        self.last_action.fill(0.0)
        self.reached_b = False
        self.contacts_ready = False

    def _write_reset_state(self, state) -> None:
        state.joint_q.zero_()
        state.joint_qd.zero_()
        state.body_qd.zero_()
        state.joint_q[self.root_q_slice].assign((A_POINT[0], A_POINT[1], RESET_BASE_HEIGHT, *_yaw_to_xyzw(A_YAW)))
        state.joint_q[self.leg_q_slice].assign(self.nominal_leg_qpos + self.reset_leg_noise)
        state.joint_q[self.arm_q_slice].assign(self.arm_qpos)
        state.joint_q[self.stone_q_slice].assign(self.stone_q)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _pin_stone(self, state) -> None:
        if not self.pin_stone_until_grasp:
            return
        state.joint_q[self.stone_q_slice].assign(self.stone_q)
        state.joint_qd[self.stone_qd_slice].zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _write_stand_ctrl(self) -> None:
        self.control.mujoco.ctrl[:ACT_DIM].assign(self.nominal_leg_ctrl)
        self.control.mujoco.ctrl[ACT_DIM : ACT_DIM + ARM_ACTUATOR_COUNT].assign(self.stand_ctrl[ACT_DIM:])

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

    def _command_to_target(self) -> tuple[np.ndarray, float, float]:
        base_xy = self._base_xy()
        delta = B_POINT - base_xy
        distance = float(np.linalg.norm(delta))
        near_target = distance <= ARRIVAL_RADIUS

        if distance <= STONE_ALIGN_RADIUS:
            face_delta = self.stone_pos[:2] - base_xy
            target_yaw = float(math.atan2(face_delta[1], face_delta[0]))
        elif distance > 1e-6:
            target_yaw = float(math.atan2(delta[1], delta[0]))
        else:
            target_yaw = self._base_yaw()

        heading_error = _wrap_angle(target_yaw - self._base_yaw())
        forward = 0.0 if near_target else float(np.clip(distance * 0.5, MIN_FORWARD_SPEED, FORWARD_SPEED))
        if not near_target and abs(heading_error) > 0.9:
            forward = MIN_FORWARD_SPEED

        command = np.array(
            [forward, 0.0, np.clip(YAW_GAIN * heading_error, -MAX_YAW_RATE, MAX_YAW_RATE)],
            dtype=np.float32,
        )
        return command, distance, abs(heading_error)

    def _observation(self) -> np.ndarray:
        projected_gravity = self._base_rotation().T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        _, base_angular = self._base_velocity_body()
        joint_q = self.state_0.joint_q.numpy()[self.leg_q_slice]
        joint_qd = self.state_0.joint_qd.numpy()[self.leg_qd_slice]
        obs = np.concatenate(
            [
                base_angular * 0.25,
                projected_gravity,
                self.command * np.array([2.0, 2.0, 0.25], dtype=np.float32),
                joint_q - self.nominal_leg_qpos,
                joint_qd * 0.05,
                self.last_action,
                self._foot_contacts(),
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

    def _apply_policy(self) -> None:
        self.command, distance, yaw_error = self._command_to_target()
        if distance <= ARRIVAL_RADIUS and yaw_error <= ARRIVAL_YAW_TOLERANCE:
            if not self.reached_b:
                print(f"Reached B: position={self._base_xy().round(3).tolist()}, target={B_POINT.tolist()}")
            self.reached_b = True
            self.command.fill(0.0)
            self._write_stand_ctrl()
            return

        action, _ = self.policy.predict(self._observation(), deterministic=True)
        self.last_action = np.clip(action[0], -1.0, 1.0).astype(np.float32)
        leg_ctrl = np.clip(self.nominal_leg_ctrl + self.last_action * ACTION_SCALE, self.leg_ctrl_low, self.leg_ctrl_high)
        self.control.mujoco.ctrl[:ACT_DIM].assign(leg_ctrl)
        self.control.mujoco.ctrl[ACT_DIM : ACT_DIM + ARM_ACTUATOR_COUNT].assign(self.stand_ctrl[ACT_DIM:])

    def step(self) -> None:
        if not self.reached_b and self.sim_time < MAX_SECONDS:
            self._apply_policy()
        else:
            self._write_stand_ctrl()

        for _ in range(CONTROL_DECIMATION):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._pin_stone(self.state_0)
        self.sim_time += self.frame_dt
        self.contacts_ready = True

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Run the Spot A-to-B pickup setup through Newton with SolverMuJoCo."
    parser.set_defaults(num_frames=100000, viewer="gl")
    return parser


def main() -> None:
    viewer, args = newton.examples.init(create_parser())
    demo = SpotPickPlaceDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
