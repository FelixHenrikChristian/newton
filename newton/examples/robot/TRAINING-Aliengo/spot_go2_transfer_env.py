from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# Keep the original Go2 policy order so its 54-D policy can be warm-started.
LEG_JOINTS = (
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
)
ARM_JOINTS = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x")
FOOT_GEOMS = ("FR", "FL", "HR", "HL")

OBS_DIM = 54
ACT_DIM = 12

_TERRAIN_GROUP_MASK = np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8)
_RAY_DOWN = np.array((0.0, 0.0, -1.0), dtype=np.float64)
_RAY_START_Z = 10.0


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _load_model(xml_path: Path) -> mujoco.MjModel:
    model_dir = xml_path.resolve().parent
    previous_dir = os.getcwd()
    try:
        os.chdir(model_dir)
        return mujoco.MjModel.from_xml_path(xml_path.name)
    finally:
        os.chdir(previous_dir)


class SpotGo2TransferEnv(gym.Env):
    """Spot locomotion using the original Go2 policy contract and rewards."""

    metadata: ClassVar = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "spot_scene.xml",
        frame_skip: int = 10,
        episode_seconds: float = 12.0,
        command_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.15, 0.70),
            (-0.10, 0.10),
            (-0.30, 0.30),
        ),
        gait_cycle_seconds: float = 0.5,
        gait_clock_scale: float = 1.0,
        trot_reward_weight: float = 0.35,
        same_side_contact_penalty_weight: float = 0.25,
        reset_clearance: float = 0.54,
        reset_height_offset: float = 0.0,
        target_clearance: float = 0.50,
        min_base_clearance: float = 0.24,
        nominal_leg_ctrl: tuple[float, float, float] = (0.0, -0.1, 0.3),
        action_scale: tuple[float, float, float] = (0.125, 0.55, 0.55),
        actuator_gain_scale: float = 3.0,
        spawn_center: tuple[float, float] = (9.0, -3.5),
        spawn_half_extents: tuple[float, float] = (5.0, 4.0),
        max_spawn_slope_deg: float = 12.0,
        randomize_spawn: bool = True,
        randomize_yaw: bool = False,
        spawn_yaw_range: tuple[float, float] = (-np.pi, np.pi),
        arm_pose_noise: float = 0.0,
        payload_mass: float = 0.72,
        payload_probability: float = 0.5,
        gravity_z: float | None = None,
        render_mode: str | None = None,
        render_camera: str | None = "tracking_side_view",
    ) -> None:
        self.xml_path = Path(xml_path)
        self.model = _load_model(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.command_range = command_range
        self.gait_cycle_seconds = float(gait_cycle_seconds)
        self.gait_clock_scale = float(gait_clock_scale)
        self.trot_reward_weight = float(trot_reward_weight)
        self.same_side_contact_penalty_weight = float(same_side_contact_penalty_weight)
        self.reset_clearance = float(reset_clearance)
        self.reset_height_offset = float(reset_height_offset)
        self.target_clearance = float(target_clearance)
        self.min_base_clearance = float(min_base_clearance)
        self.nominal_leg_ctrl = np.array(nominal_leg_ctrl * 4, dtype=np.float32)
        self.action_scale = np.array(action_scale * 4, dtype=np.float32)
        self.actuator_gain_scale = float(actuator_gain_scale)
        self.spawn_center = np.asarray(spawn_center, dtype=np.float64)
        self.spawn_half_extents = np.asarray(spawn_half_extents, dtype=np.float64)
        self.max_spawn_slope_deg = float(max_spawn_slope_deg)
        self.randomize_spawn = bool(randomize_spawn)
        self.randomize_yaw = bool(randomize_yaw)
        self.spawn_yaw_range = np.asarray(spawn_yaw_range, dtype=np.float64)
        self.arm_pose_noise = float(arm_pose_noise)
        self.payload_mass = float(payload_mass)
        self.payload_probability = float(payload_probability)
        self.render_mode = render_mode
        self.render_camera = render_camera
        if gravity_z is not None:
            self.model.opt.gravity[2] = float(gravity_z)

        self.root_joint_id = self._find_id(mujoco.mjtObj.mjOBJ_JOINT, "freejoint")
        self.root_body_id = self._find_id(mujoco.mjtObj.mjOBJ_BODY, "body")
        self.payload_body_id = self._find_id(mujoco.mjtObj.mjOBJ_BODY, "arm_link_wr1")
        self.root_qposadr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_dofadr = int(self.model.jnt_dofadr[self.root_joint_id])
        self.leg_joint_ids = self._find_ids(mujoco.mjtObj.mjOBJ_JOINT, LEG_JOINTS)
        self.leg_actuator_ids = self._find_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, LEG_JOINTS)
        self.arm_joint_ids = self._find_ids(mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
        self.arm_actuator_ids = self._find_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
        self.foot_geom_ids = self._find_ids(mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOMS)
        self.terrain_geom_id = self._find_id(mujoco.mjtObj.mjOBJ_GEOM, "lunar_terrain")

        self.leg_qpos_ids = self.model.jnt_qposadr[self.leg_joint_ids].astype(np.int32)
        self.leg_dof_ids = self.model.jnt_dofadr[self.leg_joint_ids].astype(np.int32)
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids].astype(np.int32)
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids].astype(np.int32)
        self.ctrl_low = self.model.actuator_ctrlrange[self.leg_actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.leg_actuator_ids, 1].astype(np.float32)

        self.stand_key_id = self._find_id(mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.stand_qpos = self.model.key_qpos[self.stand_key_id].copy()
        self.stand_leg_qpos = self.stand_qpos[self.leg_qpos_ids].astype(np.float32)
        self.nominal_leg_qpos = self.stand_leg_qpos + self.nominal_leg_ctrl
        self.stand_arm_qpos = self.stand_qpos[self.arm_qpos_ids].astype(np.float32)
        self.stand_arm_ctrl = self.model.key_ctrl[self.stand_key_id, self.arm_actuator_ids].astype(np.float32)
        self.base_payload_body_mass = float(self.model.body_mass[self.payload_body_id])

        gain = self.model.actuator_gainprm[self.leg_actuator_ids].copy()
        bias = self.model.actuator_biasprm[self.leg_actuator_ids].copy()
        gain[:, 0] *= self.actuator_gain_scale
        bias[:, 0] *= self.actuator_gain_scale
        bias[:, 1] *= self.actuator_gain_scale
        bias[:, 2] *= self.actuator_gain_scale**0.5
        self.model.actuator_gainprm[self.leg_actuator_ids] = gain
        self.model.actuator_biasprm[self.leg_actuator_ids] = bias

        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self._ground_z = 0.0
        self._terrain_missed = False
        self.current_payload_mass = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.viewer = None
        self.render_camera_id = (
            -1 if render_camera is None else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, render_camera)
        )

        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        mujoco.mj_forward(self.model, self.data)

        if options and "spawn" in options:
            spawn_x, spawn_y = map(float, options["spawn"])
        elif self.randomize_spawn:
            spawn_x, spawn_y = self._sample_spawn()
        else:
            spawn_x, spawn_y = map(float, self.spawn_center)

        ground_z = self._terrain_height(spawn_x, spawn_y)
        if ground_z is None:
            raise ValueError(f"Invalid spawn outside lunar terrain: ({spawn_x:.3f}, {spawn_y:.3f})")
        self.data.qpos[self.root_qposadr : self.root_qposadr + 3] = (
            spawn_x,
            spawn_y,
            ground_z + self.reset_clearance + self.reset_height_offset,
        )
        if options and "yaw" in options:
            yaw = float(options["yaw"])
        elif self.randomize_yaw:
            yaw = float(self.np_random.uniform(self.spawn_yaw_range[0], self.spawn_yaw_range[1]))
        else:
            yaw = 0.0
        self.data.qpos[self.root_qposadr + 3 : self.root_qposadr + 7] = (
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        )
        joint_noise = self.np_random.uniform(-0.03, 0.03, size=ACT_DIM)
        self.data.qpos[self.leg_qpos_ids] = self.nominal_leg_qpos + joint_noise

        arm_noise = self.np_random.uniform(-self.arm_pose_noise, self.arm_pose_noise, size=len(ARM_JOINTS))
        self.data.qpos[self.arm_qpos_ids] = self.stand_arm_qpos + arm_noise
        requested_payload = options.get("payload_mass") if options else None
        if requested_payload is None:
            has_payload = self.np_random.random() < self.payload_probability
            self.current_payload_mass = self.payload_mass if has_payload else 0.0
        else:
            self.current_payload_mass = float(requested_payload)
        self.model.body_mass[self.payload_body_id] = self.base_payload_body_mass + self.current_payload_mass
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        self.data.qvel[self.arm_dof_ids] = 0.0
        self.data.ctrl[self.leg_actuator_ids] = self.nominal_leg_ctrl
        self.data.ctrl[self.arm_actuator_ids] = self.stand_arm_ctrl

        if options and "command" in options:
            self.command = np.asarray(options["command"], dtype=np.float32)
        else:
            self.command = self._sample_command()

        self.last_action.fill(0.0)
        self.step_count = 0
        self._ground_z = ground_z
        self._terrain_missed = False
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = np.clip(self.nominal_leg_ctrl + action * self.action_scale, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.leg_actuator_ids] = ctrl
        self.data.ctrl[self.arm_actuator_ids] = self.stand_arm_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        reward, reward_terms = self._reward(action)
        self.step_count += 1
        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.max_steps
        self.last_action = action.copy()
        info = self._get_info()
        info.update(reward_terms)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.render_camera_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.render_camera_id
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _find_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing object in MJCF: {name}")
        return int(object_id)

    def _find_ids(self, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        return np.array([self._find_id(object_type, name) for name in names], dtype=np.int32)

    def _sample_command(self) -> np.ndarray:
        ranges = np.asarray(self.command_range, dtype=np.float32)
        return self.np_random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

    def _terrain_height(self, x: float, y: float) -> float | None:
        point = np.array((x, y, _RAY_START_Z), dtype=np.float64)
        geom_id = np.array((-1,), dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            point,
            _RAY_DOWN,
            _TERRAIN_GROUP_MASK,
            1,
            -1,
            geom_id,
        )
        if distance < 0.0 or int(geom_id[0]) != self.terrain_geom_id:
            return None
        return _RAY_START_Z - float(distance)

    def _local_slope_deg(self, x: float, y: float, sample_distance: float = 0.15) -> float:
        x_high = self._terrain_height(x + sample_distance, y)
        x_low = self._terrain_height(x - sample_distance, y)
        y_high = self._terrain_height(x, y + sample_distance)
        y_low = self._terrain_height(x, y - sample_distance)
        if x_high is None or x_low is None or y_high is None or y_low is None:
            return float("inf")
        zx = x_high - x_low
        zy = y_high - y_low
        return float(np.degrees(np.arctan(np.hypot(zx, zy) / (2.0 * sample_distance))))

    def _sample_spawn(self) -> tuple[float, float]:
        low = self.spawn_center - self.spawn_half_extents
        high = self.spawn_center + self.spawn_half_extents
        for _ in range(40):
            x, y = self.np_random.uniform(low, high)
            if self._local_slope_deg(float(x), float(y)) <= self.max_spawn_slope_deg:
                return float(x), float(y)
        return float(self.spawn_center[0]), float(self.spawn_center[1])

    def _ground_height(self) -> float:
        root = self.data.qpos[self.root_qposadr : self.root_qposadr + 2]
        ground_z = self._terrain_height(float(root[0]), float(root[1]))
        self._terrain_missed = ground_z is None
        if ground_z is not None:
            self._ground_z = ground_z
        return self._ground_z

    def _base_rotation(self) -> np.ndarray:
        quat_start = self.root_qposadr + 3
        return _quat_to_matrix(self.data.qpos[quat_start : quat_start + 4])

    def _projected_gravity(self) -> np.ndarray:
        return self._base_rotation().T @ np.array((0.0, 0.0, -1.0), dtype=np.float64)

    def _base_velocity_body(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self._base_rotation()
        dof = self.root_dofadr
        return rotation.T @ self.data.qvel[dof : dof + 3], rotation.T @ self.data.qvel[dof + 3 : dof + 6]

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=np.float32)
        foot_to_index = {int(geom_id): index for index, geom_id in enumerate(self.foot_geom_ids)}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.terrain_geom_id and geom2 in foot_to_index:
                contacts[foot_to_index[geom2]] = 1.0
            elif geom2 == self.terrain_geom_id and geom1 in foot_to_index:
                contacts[foot_to_index[geom1]] = 1.0
        return contacts

    def _nonfoot_ground_contact_penalty(self) -> float:
        foot_ids = {int(geom_id) for geom_id in self.foot_geom_ids}
        nonfoot_ids: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.terrain_geom_id and geom2 not in foot_ids:
                nonfoot_ids.add(geom2)
            elif geom2 == self.terrain_geom_id and geom1 not in foot_ids:
                nonfoot_ids.add(geom1)
        return float(len(nonfoot_ids))

    def _gait_phase(self) -> float:
        return float((self.step_count * self.dt / self.gait_cycle_seconds) % 1.0)

    def _gait_clock(self) -> np.ndarray:
        angle = 2.0 * np.pi * self._gait_phase()
        return np.array((np.sin(angle), np.cos(angle)), dtype=np.float32) * self.gait_clock_scale

    def _desired_trot_contacts(self) -> np.ndarray:
        if self._gait_phase() < 0.5:
            return np.array((1.0, 0.0, 0.0, 1.0), dtype=np.float32)
        return np.array((0.0, 1.0, 1.0, 0.0), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        base_linear, base_angular = self._base_velocity_body()
        joint_pos = self.data.qpos[self.leg_qpos_ids] - self.nominal_leg_qpos
        joint_vel = self.data.qvel[self.leg_dof_ids]
        return np.concatenate(
            (
                self._projected_gravity(),
                base_linear,
                base_angular,
                self.command,
                self._gait_clock(),
                joint_pos,
                joint_vel,
                self.last_action,
                self._foot_contacts(),
            )
        ).astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        base_linear, base_angular = self._base_velocity_body()
        velocity_error = np.array(
            (
                base_linear[0] - self.command[0],
                base_linear[1] - self.command[1],
                base_angular[2] - self.command[2],
            ),
            dtype=np.float64,
        )
        tracking = float(np.exp(-np.dot(velocity_error, velocity_error) / 0.25))
        projected_gravity = self._projected_gravity()
        upright = float(np.clip(-projected_gravity[2], 0.0, 1.0))
        orientation = float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)
        vertical_velocity = float(base_linear[2] ** 2)
        body_angular_xy = float(base_angular[0] ** 2 + base_angular[1] ** 2)
        clearance = float(self.data.qpos[self.root_qposadr + 2]) - self._ground_height()
        height_error = abs(clearance - self.target_clearance)
        height = float(np.exp(-(height_error * height_error) / 0.025))

        contacts = self._foot_contacts()
        desired_contacts = self._desired_trot_contacts()
        trot_match = float(np.exp(-np.sum(np.square(contacts - desired_contacts)) / 0.25))
        same_side = float(contacts[0] * contacts[1] + contacts[2] * contacts[3])
        all_off = 1.0 if float(np.sum(contacts)) == 0.0 else 0.0
        action_rate = float(np.sum(np.square(action - self.last_action)))
        action_size = float(np.sum(np.square(action)))
        joint_speed = float(np.sum(np.square(self.data.qvel[self.leg_dof_ids])))
        nonfoot = self._nonfoot_ground_contact_penalty()
        unhealthy = 1.0 if self._is_unhealthy() else 0.0

        reward = (
            2.0 * tracking
            + 0.60 * upright
            + 0.45 * height
            + self.trot_reward_weight * trot_match
            - self.same_side_contact_penalty_weight * same_side
            - 0.5 * all_off
            - 0.03 * action_rate
            - 0.005 * action_size
            - 0.0005 * joint_speed
            - 1.0 * orientation
            - 0.2 * vertical_velocity
            - 0.05 * body_angular_xy
            - 2.5 * nonfoot
            - 2.0 * unhealthy
        )
        return float(reward), {
            "reward_tracking": tracking,
            "reward_upright": upright,
            "reward_height": height,
            "reward_trot_match": trot_match,
            "penalty_same_side_contact": same_side,
            "penalty_all_off": all_off,
            "penalty_action_rate": action_rate,
            "penalty_action_size": action_size,
            "penalty_joint_speed": joint_speed,
            "penalty_orientation": orientation,
            "penalty_vertical_velocity": vertical_velocity,
            "penalty_body_angular_xy": body_angular_xy,
            "penalty_nonfoot_contact": nonfoot,
        }

    def _is_unhealthy(self) -> bool:
        clearance = float(self.data.qpos[self.root_qposadr + 2]) - self._ground_height()
        tipped = self._projected_gravity()[2] > -0.35
        bad_number = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        return bool(self._terrain_missed or clearance < self.min_base_clearance or tipped or bad_number)

    def _get_info(self) -> dict[str, float]:
        base_linear, base_angular = self._base_velocity_body()
        contacts = self._foot_contacts()
        projected_gravity = self._projected_gravity()
        clearance = float(self.data.qpos[self.root_qposadr + 2]) - self._ground_height()
        tilt_deg = float(np.degrees(np.arccos(np.clip(-projected_gravity[2], -1.0, 1.0))))
        return {
            "command_vx": float(self.command[0]),
            "command_vy": float(self.command[1]),
            "command_yaw": float(self.command[2]),
            "gait_phase": self._gait_phase(),
            "base_vx": float(base_linear[0]),
            "base_vy": float(base_linear[1]),
            "base_yaw_rate": float(base_angular[2]),
            "base_clearance": clearance,
            "base_clearance_error": abs(clearance - self.target_clearance),
            "base_tilt_deg": tilt_deg,
            "terrain_missed": self._terrain_missed,
            "projected_gravity_z": float(projected_gravity[2]),
            "payload_mass": self.current_payload_mass,
            "contact_fr": float(contacts[0]),
            "contact_fl": float(contacts[1]),
            "contact_hr": float(contacts[2]),
            "contact_hl": float(contacts[3]),
        }
