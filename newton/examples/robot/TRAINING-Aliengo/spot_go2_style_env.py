from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

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

ARM_JOINTS = (
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
)

FOOT_GEOMS = ("FL", "FR", "HL", "HR")

LEGACY_OBS_DIM = 55
OBS_DIM = 58
ACT_DIM = 12


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


class SpotGo2StyleEnv(gym.Env):
    """Spot locomotion environment using a Go2-style velocity-tracking setup.

    Observations follow a common quadruped locomotion layout: base angular
    velocity, projected gravity, command, relative joint positions, joint
    velocities, previous action, and foot contacts. Actions are small offsets
    around the stand pose.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "spot_scene.xml",
        control_decimation: int = 10,
        episode_seconds: float = 20.0,
        command_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.25, 0.6),
            (-0.2, 0.2),
            (-0.5, 0.5),
        ),
        reset_base_height: float = 1.80,
        target_base_height: float = 1.65,
        action_scale: float = 0.35,
        nominal_leg_ctrl: tuple[float, float, float] = (0.0, -0.22, 0.55),
        actuator_gain_scale: float = 1.8,
        gait_period: float = 0.56,
        gait_contact_sharpness: float = 3.0,
        swing_height: float = 0.10,
        randomize_domain: bool = False,
        use_curriculum: bool = True,
        observation_version: str = "v2",
        command_resample_seconds: float = 4.0,
        spawn_xy_range: float = 1.5,
        spawn_yaw_range: float = 0.35,
        reset_joint_noise: float = 0.03,
        reset_velocity_noise: float = 0.05,
        observation_noise: float = 1.0,
        max_action_delay: int = 1,
        render_mode: str | None = None,
        render_camera: str | None = "tracking_side_view",
    ) -> None:
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.control_decimation = control_decimation
        self.dt = self.model.opt.timestep * self.control_decimation
        self.max_steps = int(episode_seconds / self.dt)
        self.command_range = command_range
        self.reset_base_height = reset_base_height
        self.target_base_height = target_base_height
        # Bias action authority toward knee flexion so the policy can use the
        # distal joint instead of relying on high-frequency proximal jitter.
        self.action_scale = np.array([0.6, 1.0, 1.6] * 4, dtype=np.float32) * float(action_scale)
        self.nominal_leg_ctrl = np.array(nominal_leg_ctrl * 4, dtype=np.float32)
        self.actuator_gain_scale = actuator_gain_scale
        if gait_period <= 0.0:
            raise ValueError("gait_period must be positive")
        if gait_contact_sharpness <= 0.0:
            raise ValueError("gait_contact_sharpness must be positive")
        if swing_height < 0.0:
            raise ValueError("swing_height must be non-negative")
        self.gait_period = gait_period
        self.gait_contact_sharpness = gait_contact_sharpness
        self.swing_height = swing_height
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        if observation_version not in ("v1", "v2"):
            raise ValueError("observation_version must be 'v1' or 'v2'")
        if command_resample_seconds < 0.0:
            raise ValueError("command_resample_seconds must be non-negative")
        if max_action_delay < 0:
            raise ValueError("max_action_delay must be non-negative")
        self.observation_version = observation_version
        self.command_resample_steps = (
            max(int(round(command_resample_seconds / self.dt)), 1) if command_resample_seconds > 0.0 else 0
        )
        self.spawn_xy_range = spawn_xy_range
        self.spawn_yaw_range = spawn_yaw_range
        self.reset_joint_noise = reset_joint_noise
        self.reset_velocity_noise = reset_velocity_noise
        self.observation_noise = observation_noise
        self.max_action_delay = max_action_delay
        self.render_mode = render_mode
        self.render_camera = render_camera

        self.root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint")
        if self.root_joint_id < 0:
            raise ValueError("Expected a freejoint named 'freejoint' in spot_scene.xml")
        self.root_qposadr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_dofadr = int(self.model.jnt_dofadr[self.root_joint_id])
        self.root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")
        if self.root_body_id < 0:
            raise ValueError("Expected a body named 'body' in spot_scene.xml")

        self.leg_actuator_ids = self._find_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, LEG_JOINTS)
        self.arm_actuator_ids = self._find_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_JOINTS)
        self.leg_joint_ids = self._find_ids(mujoco.mjtObj.mjOBJ_JOINT, LEG_JOINTS)
        self.arm_joint_ids = self._find_ids(mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
        self.foot_geom_ids = self._find_ids(mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOMS)
        self.terrain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lunar_terrain")

        self.leg_qpos_ids = self.model.jnt_qposadr[self.leg_joint_ids].astype(np.int32)
        self.leg_dof_ids = self.model.jnt_dofadr[self.leg_joint_ids].astype(np.int32)
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids].astype(np.int32)
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids].astype(np.int32)

        self.ctrl_low = self.model.actuator_ctrlrange[self.leg_actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.leg_actuator_ids, 1].astype(np.float32)

        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id < 0:
            raise ValueError("Expected a keyframe named 'stand' in spot_scene.xml")
        self.stand_qpos = self.model.key_qpos[self.stand_key_id].copy()
        self.stand_leg_qpos = self.stand_qpos[self.leg_qpos_ids].astype(np.float32)
        self.nominal_leg_qpos = self.stand_leg_qpos + self.nominal_leg_ctrl
        self.stand_arm_qpos = self.stand_qpos[self.arm_qpos_ids].astype(np.float32)
        self.stand_arm_ctrl = self.model.key_ctrl[self.stand_key_id, self.arm_actuator_ids].astype(np.float32)
        self.spawn_position = self.stand_qpos[self.root_qposadr : self.root_qposadr + 2].copy()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        self.data.qpos[self.root_qposadr : self.root_qposadr + 3] = np.array(
            [self.spawn_position[0], self.spawn_position[1], self.reset_base_height],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)
        reference_ground_height = self._terrain_height_at(self.spawn_position)
        self.reset_base_clearance = float(self.reset_base_height - reference_ground_height)
        self.target_base_clearance = float(self.target_base_height - reference_ground_height)

        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._base_terrain_friction = (
            self.model.geom_friction[self.terrain_geom_id].copy() if self.terrain_geom_id >= 0 else None
        )
        self._base_leg_gainprm = self.model.actuator_gainprm[self.leg_actuator_ids].copy()
        self._base_leg_biasprm = self.model.actuator_biasprm[self.leg_actuator_ids].copy()
        self._base_leg_gainprm[:, 0] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 0] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 1] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 2] *= self.actuator_gain_scale**0.5

        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.second_last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.action_buffer = np.zeros((self.max_action_delay + 1, ACT_DIM), dtype=np.float32)
        self.action_delay_steps = 0
        self.previous_foot_positions = np.zeros((len(FOOT_GEOMS), 3), dtype=np.float64)
        self.previous_contacts = np.zeros(len(FOOT_GEOMS), dtype=np.float32)
        self.last_gait_match = 0.0
        self.last_foot_slip = 0.0
        self.command = np.zeros(3, dtype=np.float32)
        self.gait_phase = 0.0
        self.command_age_steps = 0
        self.step_count = 0
        self.curriculum_level = 0.0
        self.completed_episodes = 0
        self.last_episode_steps = 0
        self.last_episode_tracking_error = float("inf")
        self.last_episode_gait_match = 0.0
        self.last_episode_foot_slip = float("inf")
        self.episode_tracking_error_sum = 0.0
        self.episode_gait_match_sum = 0.0
        self.episode_foot_slip_sum = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        obs_dim = LEGACY_OBS_DIM if self.observation_version == "v1" else OBS_DIM
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.viewer = None
        self.render_camera_id = (
            -1
            if self.render_camera is None
            else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.render_camera)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.use_curriculum and self.completed_episodes > 0:
            success = (
                self.last_episode_steps >= 0.9 * self.max_steps
                and self.last_episode_tracking_error < 0.22
                and self.last_episode_gait_match > 0.65
                and self.last_episode_foot_slip < 0.25
            )
            delta = 0.02 if success else -0.01
            self.curriculum_level = float(np.clip(self.curriculum_level + delta, 0.0, 1.0))

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        self._apply_domain_randomization()
        mujoco.mj_forward(self.model, self.data)

        spawn_xy = self.spawn_position.copy()
        spawn_yaw = 0.0
        if options and "spawn_offset" in options:
            spawn_xy += np.asarray(options["spawn_offset"], dtype=np.float64)
        elif self.randomize_domain:
            strength = self._randomization_strength()
            spawn_xy += self.np_random.uniform(-self.spawn_xy_range, self.spawn_xy_range, size=2) * strength
        if options and "spawn_yaw" in options:
            spawn_yaw = float(options["spawn_yaw"])
        elif self.randomize_domain:
            spawn_yaw = float(
                self.np_random.uniform(-self.spawn_yaw_range, self.spawn_yaw_range) * self._randomization_strength()
            )

        ground_height = self._terrain_height_at(spawn_xy)
        self.data.qpos[self.root_qposadr : self.root_qposadr + 7] = np.array(
            [
                spawn_xy[0],
                spawn_xy[1],
                ground_height + self.reset_base_clearance,
                np.cos(0.5 * spawn_yaw),
                0.0,
                0.0,
                np.sin(0.5 * spawn_yaw),
            ],
            dtype=np.float64,
        )
        joint_noise = np.zeros(ACT_DIM, dtype=np.float64)
        velocity_noise = np.zeros(self.model.nv, dtype=np.float64)
        if self.randomize_domain:
            strength = self._randomization_strength()
            joint_noise = self.np_random.uniform(-self.reset_joint_noise, self.reset_joint_noise, ACT_DIM) * strength
            velocity_noise = (
                self.np_random.uniform(-self.reset_velocity_noise, self.reset_velocity_noise, self.model.nv) * strength
            )
        self.data.qpos[self.leg_qpos_ids] = self.nominal_leg_qpos + joint_noise
        self.data.qpos[self.arm_qpos_ids] = self.stand_arm_qpos
        self.data.qvel[:] = velocity_noise
        self.data.qvel[self.arm_dof_ids] = 0.0
        self._align_feet_to_terrain()
        self.data.ctrl[self.leg_actuator_ids] = self.nominal_leg_ctrl
        self.data.ctrl[self.arm_actuator_ids] = self.stand_arm_ctrl

        if options and "command" in options:
            self.command = np.asarray(options["command"], dtype=np.float32)
        else:
            self.command = self._sample_command()

        self.last_action.fill(0.0)
        self.second_last_action.fill(0.0)
        self.action_buffer.fill(0.0)
        if self.randomize_domain:
            available_delay = int(round(self.max_action_delay * self._randomization_strength()))
            self.action_delay_steps = int(self.np_random.integers(0, available_delay + 1))
        else:
            self.action_delay_steps = 0
        self.gait_phase = 0.0
        self.command_age_steps = 0
        self.step_count = 0
        self.episode_tracking_error_sum = 0.0
        self.episode_gait_match_sum = 0.0
        self.episode_foot_slip_sum = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.previous_foot_positions[:] = self.data.geom_xpos[self.foot_geom_ids]
        self.previous_contacts[:] = self._foot_contacts()
        self.last_gait_match = 0.0
        self.last_foot_slip = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        requested_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.max_action_delay > 0:
            self.action_buffer[1:] = self.action_buffer[:-1]
        self.action_buffer[0] = requested_action
        action = self.action_buffer[self.action_delay_steps].copy()

        leg_ctrl = np.clip(self.nominal_leg_ctrl + action * self.action_scale, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.leg_actuator_ids] = leg_ctrl
        self.data.ctrl[self.arm_actuator_ids] = self.stand_arm_ctrl

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

        reward, reward_terms = self._reward(action)
        base_linear, base_angular = self._base_velocity_body()
        tracking_error = float(
            np.linalg.norm(base_linear[:2] - self.command[:2]) + 0.25 * abs(base_angular[2] - self.command[2])
        )
        self.episode_tracking_error_sum += tracking_error
        self.episode_gait_match_sum += self.last_gait_match
        self.episode_foot_slip_sum += self.last_foot_slip
        self.second_last_action = self.last_action.copy()
        self.last_action = action.copy()
        self.previous_foot_positions[:] = self.data.geom_xpos[self.foot_geom_ids]
        self.previous_contacts[:] = self._foot_contacts()
        self.step_count += 1
        self.command_age_steps += 1
        self._advance_gait_phase()
        self.last_episode_steps = self.step_count

        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        info["reward_components"] = reward_terms

        if terminated or truncated:
            count = max(self.step_count, 1)
            self.last_episode_tracking_error = self.episode_tracking_error_sum / count
            self.last_episode_gait_match = self.episode_gait_match_sum / count
            self.last_episode_foot_slip = self.episode_foot_slip_sum / count
            self.completed_episodes += 1

        if self.command_resample_steps > 0 and self.command_age_steps >= self.command_resample_steps:
            self.command = self._sample_command()
            self.command_age_steps = 0
            if self._gait_activity() <= 0.0:
                self.gait_phase = 0.0

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
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:] = self.data.qpos[self.root_qposadr : self.root_qposadr + 3]
                self.viewer.cam.distance = 7.0
                self.viewer.cam.azimuth = 145.0
                self.viewer.cam.elevation = -18.0
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _find_ids(self, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.array([mujoco.mj_name2id(self.model, obj_type, name) for name in names], dtype=np.int32)
        if np.any(ids < 0):
            missing = [name for name, idx in zip(names, ids, strict=True) if idx < 0]
            raise ValueError(f"Missing objects in MJCF: {missing}")
        return ids

    def _terrain_height_at(self, xy: np.ndarray) -> float:
        if self.terrain_geom_id < 0:
            return 0.0

        geom_type = self.model.geom_type[self.terrain_geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            return float(self.data.geom_xpos[self.terrain_geom_id, 2])
        if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
            origin = np.array([xy[0], xy[1], 10.0], dtype=np.float64)
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            distance = mujoco.mj_rayHfield(
                self.model,
                self.data,
                self.terrain_geom_id,
                origin,
                direction,
            )
            if distance >= 0.0:
                return float(origin[2] - distance)
        return float(self.data.geom_xpos[self.terrain_geom_id, 2])

    def _terrain_heights_at(self, xy: np.ndarray) -> np.ndarray:
        return np.array([self._terrain_height_at(point) for point in xy], dtype=np.float64)

    def _base_clearance(self) -> float:
        base_position = self.data.qpos[self.root_qposadr : self.root_qposadr + 3]
        return float(base_position[2] - self._terrain_height_at(base_position[:2]))

    def _foot_clearances(self) -> np.ndarray:
        foot_positions = self.data.geom_xpos[self.foot_geom_ids]
        foot_bottom_heights = foot_positions[:, 2] - self.model.geom_size[self.foot_geom_ids, 0]
        return foot_bottom_heights - self._terrain_heights_at(foot_positions[:, :2])

    def _align_feet_to_terrain(self) -> None:
        if self.terrain_geom_id < 0:
            return

        mujoco.mj_forward(self.model, self.data)
        clearances = self._foot_clearances()
        minimum_clearance = float(np.min(clearances))
        if minimum_clearance < 0.0:
            self.data.qpos[self.root_qposadr + 2] -= minimum_clearance

    def _randomization_strength(self) -> float:
        if not self.use_curriculum:
            return 1.0
        return 0.25 + 0.75 * self.curriculum_level

    def _restore_domain_parameters(self) -> None:
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_inertia[:] = self._base_body_inertia
        if self.terrain_geom_id >= 0 and self._base_terrain_friction is not None:
            self.model.geom_friction[self.terrain_geom_id] = self._base_terrain_friction
        self.model.actuator_gainprm[self.leg_actuator_ids] = self._base_leg_gainprm
        self.model.actuator_biasprm[self.leg_actuator_ids] = self._base_leg_biasprm

    def _apply_domain_randomization(self) -> None:
        self._restore_domain_parameters()
        if not self.randomize_domain:
            return

        strength = self._randomization_strength()
        mass_scale = float(self.np_random.uniform(1.0 - 0.15 * strength, 1.0 + 0.15 * strength))
        self.model.body_mass[:] = self._base_body_mass * mass_scale
        self.model.body_inertia[:] = self._base_body_inertia * mass_scale

        if self.terrain_geom_id >= 0 and self._base_terrain_friction is not None:
            friction_scale = float(self.np_random.uniform(1.0 - 0.3 * strength, 1.0 + 0.3 * strength))
            self.model.geom_friction[self.terrain_geom_id] = self._base_terrain_friction * friction_scale

        actuator_scale = self.np_random.uniform(
            1.0 - 0.15 * strength,
            1.0 + 0.15 * strength,
            size=(len(self.leg_actuator_ids), 1),
        )
        self.model.actuator_gainprm[self.leg_actuator_ids] = self._base_leg_gainprm * actuator_scale
        self.model.actuator_biasprm[self.leg_actuator_ids] = self._base_leg_biasprm * actuator_scale

    def _sample_command(self) -> np.ndarray:
        ranges = np.asarray(self.command_range, dtype=np.float32)
        if not self.use_curriculum:
            return self.np_random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

        if self.np_random.random() < 0.1:
            return np.zeros(3, dtype=np.float32)

        level = self.curriculum_level
        vx_low, vx_high = ranges[0]
        vx_high = vx_low + (vx_high - vx_low) * max(0.25, level)
        active_vx_low = min(max(vx_low, 0.15), vx_high)
        vx = float(self.np_random.uniform(active_vx_low, vx_high))
        vy = float(self.np_random.uniform(ranges[1, 0] * level, ranges[1, 1] * level))
        yaw = float(self.np_random.uniform(ranges[2, 0] * level, ranges[2, 1] * level))
        return np.array([vx, vy, yaw], dtype=np.float32)

    def _base_rotation(self) -> np.ndarray:
        quat_start = self.root_qposadr + 3
        return _quat_to_matrix(self.data.qpos[quat_start : quat_start + 4])

    def _projected_gravity(self) -> np.ndarray:
        rotation = self._base_rotation()
        return rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _base_velocity_body(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self._base_rotation()
        dof = self.root_dofadr
        linear = rotation.T @ self.data.qvel[dof : dof + 3]
        angular = rotation.T @ self.data.qvel[dof + 3 : dof + 6]
        return linear, angular

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=np.float32)
        if self.terrain_geom_id < 0:
            return contacts

        foot_to_index = {int(geom_id): idx for idx, geom_id in enumerate(self.foot_geom_ids)}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.terrain_geom_id and geom2 in foot_to_index:
                contacts[foot_to_index[geom2]] = 1.0
            elif geom2 == self.terrain_geom_id and geom1 in foot_to_index:
                contacts[foot_to_index[geom1]] = 1.0
        return contacts

    def _foot_contact_forces(self) -> np.ndarray:
        forces = np.zeros(4, dtype=np.float64)
        if self.terrain_geom_id < 0:
            return forces

        foot_to_index = {int(geom_id): idx for idx, geom_id in enumerate(self.foot_geom_ids)}
        contact_force = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            foot_geom = geom2 if geom1 == self.terrain_geom_id else geom1 if geom2 == self.terrain_geom_id else -1
            if foot_geom not in foot_to_index:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            forces[foot_to_index[foot_geom]] += max(float(contact_force[0]), 0.0)
        return forces

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
        phase_step = self.dt * self._gait_frequency_scale() / self.gait_period
        self.gait_phase = float((self.gait_phase + activity * phase_step) % 1.0)

    def _gait_phase(self) -> float:
        return self.gait_phase

    def _desired_contacts(self) -> np.ndarray:
        activity = self._gait_activity()
        if activity <= 0.0:
            return np.ones(4, dtype=np.float32)
        phase = self._gait_phase()
        diagonal_a = 0.5 + 0.5 * np.tanh(self.gait_contact_sharpness * np.sin(2.0 * np.pi * phase))
        diagonal_b = 1.0 - diagonal_a
        trot_contacts = np.array([diagonal_a, diagonal_b, diagonal_b, diagonal_a], dtype=np.float32)
        return activity * trot_contacts + (1.0 - activity) * np.ones(4, dtype=np.float32)

    def _desired_foot_clearance(self) -> np.ndarray:
        activity = self._gait_activity()
        phase_signal = float(np.sin(2.0 * np.pi * self._gait_phase()))
        diagonal_a = self.swing_height * max(-phase_signal, 0.0)
        diagonal_b = self.swing_height * max(phase_signal, 0.0)
        return activity * np.array([diagonal_a, diagonal_b, diagonal_b, diagonal_a], dtype=np.float32)

    def _foot_velocities(self) -> np.ndarray:
        positions = self.data.geom_xpos[self.foot_geom_ids]
        return (positions - self.previous_foot_positions) / self.dt

    def _gait_observation(self) -> np.ndarray:
        phase = self._gait_phase()
        angle = 2.0 * np.pi * phase
        return np.concatenate(
            [
                np.array([np.sin(angle), np.cos(angle)], dtype=np.float32),
                self._desired_contacts(),
            ]
        )

    def _get_obs(self) -> np.ndarray:
        projected_gravity = self._projected_gravity()
        base_linear, base_angular = self._base_velocity_body()
        joint_pos = self.data.qpos[self.leg_qpos_ids] - self.nominal_leg_qpos
        joint_vel = self.data.qvel[self.leg_dof_ids]
        command = self.command * np.array([2.0, 2.0, 0.25], dtype=np.float32)

        legacy_obs = np.concatenate(
            [
                base_angular * 0.25,
                projected_gravity,
                command,
                self._gait_observation(),
                joint_pos,
                joint_vel * 0.05,
                self.last_action,
                self._foot_contacts(),
            ]
        )
        obs = legacy_obs if self.observation_version == "v1" else np.concatenate([legacy_obs, base_linear * 0.5])
        if self.randomize_domain and self.observation_noise > 0.0:
            noise_scale = np.zeros_like(obs)
            noise_scale[0:3] = 0.02
            noise_scale[3:6] = 0.01
            noise_scale[15:27] = 0.01
            noise_scale[27:39] = 0.02
            noise_scale[39:51] = 0.005
            if self.observation_version == "v2":
                noise_scale[55:58] = 0.03
            obs = obs + self.np_random.normal(0.0, noise_scale * self.observation_noise)
        return obs.astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        base_linear, base_angular = self._base_velocity_body()
        velocity_error = np.array(
            [base_linear[0] - self.command[0], base_linear[1] - self.command[1]],
            dtype=np.float64,
        )
        lin_tracking = 2.0 * float(np.exp(-np.dot(velocity_error, velocity_error) / 0.1))
        yaw_tracking = 0.5 * float(np.exp(-((base_angular[2] - self.command[2]) ** 2) / 0.25))

        projected_gravity = self._projected_gravity()
        healthy = not self._is_unhealthy()
        alive = 0.5 if healthy else 0.0
        upright = 0.5 * float(np.clip(-projected_gravity[2], 0.0, 1.0))
        vertical_velocity = -1.0 * float(base_linear[2] ** 2)
        height_error = self._base_clearance() - self.target_base_clearance
        height = 0.4 * float(np.exp(-(height_error * height_error) / 0.04))
        orientation = -1.0 * float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)
        actuator_force = self.data.actuator_force[self.leg_actuator_ids]
        joint_velocity = self.data.qvel[self.leg_dof_ids]
        torque = -1e-5 * float(np.sum(np.square(actuator_force)))
        mechanical_power = -5e-5 * float(np.sum(np.abs(actuator_force * joint_velocity)))
        smooth = -3e-3 * float(np.sum(np.square(action - self.last_action)))
        action_acceleration = action - 2.0 * self.last_action + self.second_last_action
        action_accel = -1e-3 * float(np.sum(np.square(action_acceleration)))
        action_size = -8e-3 * float(np.sum(np.square(action)))
        thigh_velocity = -2e-3 * float(np.sum(np.square(self.data.qvel[self.leg_dof_ids][1::3])))
        contacts = self._foot_contacts()
        desired_contacts = self._desired_contacts()
        contact_match = float(1.0 - np.mean(np.abs(contacts - desired_contacts)))
        gait_contact = 0.35 * (contact_match - 0.5)

        foot_velocities = self._foot_velocities()
        slip_speed_sq = np.sum(np.square(foot_velocities[:, :2]), axis=1)
        persistent_contacts = contacts * self.previous_contacts
        mean_foot_slip = float(np.mean(np.minimum(slip_speed_sq, 4.0) * persistent_contacts))
        foot_slip = -0.08 * mean_foot_slip

        foot_clearances = self._foot_clearances()
        desired_clearance = self._desired_foot_clearance()
        swing_mask = desired_clearance > 0.005
        if np.any(swing_mask):
            clearance_error = float(np.mean(np.square(foot_clearances[swing_mask] - desired_clearance[swing_mask])))
        else:
            clearance_error = 0.0
        foot_clearance = -4.0 * clearance_error

        contact_forces = self._foot_contact_forces()
        foot_impact = -1e-4 * float(np.mean(np.maximum(contact_forces - 250.0, 0.0)))

        joint_positions = self.data.qpos[self.leg_qpos_ids]
        joint_ranges = self.model.jnt_range[self.leg_joint_ids]
        lower_margin = np.maximum(0.08 - (joint_positions - joint_ranges[:, 0]), 0.0)
        upper_margin = np.maximum(0.08 - (joint_ranges[:, 1] - joint_positions), 0.0)
        joint_limit = -0.5 * float(np.sum(np.square(lower_margin) + np.square(upper_margin)))

        leg_actions = action.reshape(4, 3)
        diagonal_error = np.sum(np.square(leg_actions[0] - leg_actions[3]))
        diagonal_error += np.sum(np.square(leg_actions[1] - leg_actions[2]))
        straight_command_weight = self._gait_activity() * float(
            np.exp(-25.0 * self.command[1] ** 2 - 8.0 * self.command[2] ** 2)
        )
        diagonal_sync = -0.03 * straight_command_weight * float(diagonal_error)

        self.last_gait_match = contact_match
        self.last_foot_slip = mean_foot_slip
        unhealthy = -10.0 if not healthy else 0.0

        components = {
            "alive": alive,
            "lin": lin_tracking,
            "yaw": yaw_tracking,
            "upright": upright,
            "vertical_velocity": vertical_velocity,
            "height": height,
            "orientation": orientation,
            "torque": torque,
            "mechanical_power": mechanical_power,
            "smooth": smooth,
            "action_accel": action_accel,
            "action_size": action_size,
            "thigh_velocity": thigh_velocity,
            "gait_contact": gait_contact,
            "foot_slip": foot_slip,
            "foot_clearance": foot_clearance,
            "foot_impact": foot_impact,
            "joint_limit": joint_limit,
            "diagonal_sync": diagonal_sync,
            "unhealthy": unhealthy,
        }
        return float(sum(components.values())), components

    def _is_unhealthy(self) -> bool:
        projected_gravity = self._projected_gravity()
        base_clearance = self._base_clearance()
        too_low = base_clearance < self.target_base_clearance - 0.45
        too_high = base_clearance > self.target_base_clearance + 0.65
        tipped = projected_gravity[2] > -0.35
        bad_number = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        return bool(too_low or too_high or tipped or bad_number)

    def _get_info(self) -> dict[str, float]:
        base_linear, base_angular = self._base_velocity_body()
        return {
            "command_vx": float(self.command[0]),
            "command_vy": float(self.command[1]),
            "command_yaw": float(self.command[2]),
            "base_vx": float(base_linear[0]),
            "base_vy": float(base_linear[1]),
            "base_yaw_rate": float(base_angular[2]),
            "base_height": float(self.data.qpos[self.root_qposadr + 2]),
            "base_clearance": self._base_clearance(),
            "curriculum_level": float(self.curriculum_level),
            "gait_phase": self._gait_phase(),
            "gait_match": self.last_gait_match,
            "foot_slip": self.last_foot_slip,
            "episode_tracking_error": self.episode_tracking_error_sum / max(self.step_count, 1),
            "episode_gait_match": self.episode_gait_match_sum / max(self.step_count, 1),
            "episode_foot_slip": self.episode_foot_slip_sum / max(self.step_count, 1),
        }
