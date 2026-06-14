from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
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

OBS_DIM = 49
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

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "spot_scene.xml",
        control_decimation: int = 10,
        episode_seconds: float = 20.0,
        command_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.1, 0.6),
            (-0.2, 0.2),
            (-0.5, 0.5),
        ),
        reset_base_height: float = 1.72,
        target_base_height: float = 1.68,
        action_scale: float = 0.25,
        nominal_leg_ctrl: tuple[float, float, float] = (0.0, -0.1, 0.3),
        actuator_gain_scale: float = 3.0,
        randomize_domain: bool = True,
        use_curriculum: bool = True,
        render_mode: str | None = None,
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
        self.action_scale = np.full(ACT_DIM, action_scale, dtype=np.float32)
        self.nominal_leg_ctrl = np.array(nominal_leg_ctrl * 4, dtype=np.float32)
        self.actuator_gain_scale = actuator_gain_scale
        self.randomize_domain = randomize_domain
        self.use_curriculum = use_curriculum
        self.render_mode = render_mode

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
        self.stand_height = float(self.target_base_height)

        self._base_body_mass = float(self.model.body_mass[self.root_body_id])
        self._base_terrain_friction = (
            self.model.geom_friction[self.terrain_geom_id].copy() if self.terrain_geom_id >= 0 else None
        )
        self._base_leg_gainprm = self.model.actuator_gainprm[self.leg_actuator_ids].copy()
        self._base_leg_biasprm = self.model.actuator_biasprm[self.leg_actuator_ids].copy()
        self._base_leg_gainprm[:, 0] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 0] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 1] *= self.actuator_gain_scale
        self._base_leg_biasprm[:, 2] *= self.actuator_gain_scale ** 0.5

        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.curriculum_level = 0.0
        self.last_episode_steps = self.max_steps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.viewer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.use_curriculum:
            success = self.last_episode_steps >= 0.75 * self.max_steps
            delta = 0.005 if success else -0.002
            self.curriculum_level = float(np.clip(self.curriculum_level + delta, 0.0, 1.0))

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        self._apply_domain_randomization()

        self.data.qpos[self.root_qposadr + 2] = self.reset_base_height
        leg_noise = self.np_random.uniform(-0.05, 0.05, size=ACT_DIM)
        self.data.qpos[self.leg_qpos_ids] = self.nominal_leg_qpos + leg_noise
        self.data.qpos[self.arm_qpos_ids] = self.stand_arm_qpos
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
        self.last_episode_steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        leg_ctrl = np.clip(self.nominal_leg_ctrl + action * self.action_scale, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.leg_actuator_ids] = leg_ctrl
        self.data.ctrl[self.arm_actuator_ids] = self.stand_arm_ctrl

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

        reward, reward_terms = self._reward(action)
        self.last_action = action.copy()
        self.step_count += 1
        self.last_episode_steps = self.step_count

        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        info["reward_components"] = reward_terms

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _find_ids(self, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.array([mujoco.mj_name2id(self.model, obj_type, name) for name in names], dtype=np.int32)
        if np.any(ids < 0):
            missing = [name for name, idx in zip(names, ids) if idx < 0]
            raise ValueError(f"Missing objects in MJCF: {missing}")
        return ids

    def _restore_domain_parameters(self) -> None:
        self.model.body_mass[self.root_body_id] = self._base_body_mass
        if self.terrain_geom_id >= 0 and self._base_terrain_friction is not None:
            self.model.geom_friction[self.terrain_geom_id] = self._base_terrain_friction
        self.model.actuator_gainprm[self.leg_actuator_ids] = self._base_leg_gainprm
        self.model.actuator_biasprm[self.leg_actuator_ids] = self._base_leg_biasprm

    def _apply_domain_randomization(self) -> None:
        self._restore_domain_parameters()
        if not self.randomize_domain:
            return

        mass_scale = float(self.np_random.uniform(0.85, 1.15))
        self.model.body_mass[self.root_body_id] = self._base_body_mass * mass_scale

        if self.terrain_geom_id >= 0 and self._base_terrain_friction is not None:
            friction_scale = float(self.np_random.uniform(0.7, 1.3))
            self.model.geom_friction[self.terrain_geom_id] = self._base_terrain_friction * friction_scale

        actuator_scale = self.np_random.uniform(0.85, 1.15, size=(len(self.leg_actuator_ids), 1))
        self.model.actuator_gainprm[self.leg_actuator_ids] = self._base_leg_gainprm * actuator_scale
        self.model.actuator_biasprm[self.leg_actuator_ids] = self._base_leg_biasprm * actuator_scale

    def _sample_command(self) -> np.ndarray:
        ranges = np.asarray(self.command_range, dtype=np.float32)
        if not self.use_curriculum:
            return self.np_random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

        level = self.curriculum_level
        vx_low, vx_high = ranges[0]
        vx_high = vx_low + (vx_high - vx_low) * max(0.25, level)
        vx = float(self.np_random.uniform(vx_low, vx_high))
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

    def _get_obs(self) -> np.ndarray:
        projected_gravity = self._projected_gravity()
        _, base_angular = self._base_velocity_body()
        joint_pos = self.data.qpos[self.leg_qpos_ids] - self.nominal_leg_qpos
        joint_vel = self.data.qvel[self.leg_dof_ids]
        command = self.command * np.array([2.0, 2.0, 0.25], dtype=np.float32)

        obs = np.concatenate(
            [
                base_angular * 0.25,
                projected_gravity,
                command,
                joint_pos,
                joint_vel * 0.05,
                self.last_action,
                self._foot_contacts(),
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        base_linear, base_angular = self._base_velocity_body()
        velocity_error = np.array(
            [base_linear[0] - self.command[0], base_linear[1] - self.command[1]],
            dtype=np.float64,
        )
        lin_tracking = float(np.exp(-np.dot(velocity_error, velocity_error) / 0.25))
        yaw_tracking = 0.5 * float(np.exp(-((base_angular[2] - self.command[2]) ** 2) / 0.25))

        projected_gravity = self._projected_gravity()
        vertical_velocity = -2.0 * float(base_linear[2] ** 2)
        height = -float((self.data.qpos[self.root_qposadr + 2] - self.stand_height) ** 2)
        orientation = -0.5 * float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)
        torque = -2e-4 * float(np.sum(np.square(self.data.actuator_force[self.leg_actuator_ids])))
        smooth = -5e-3 * float(np.sum(np.square(action - self.last_action)))
        contacts = self._foot_contacts()
        contact = 0.15 * min(float(np.sum(contacts > 0.0)) / 2.0, 1.0)
        unhealthy = -2.0 if self._is_unhealthy() else 0.0

        components = {
            "lin": lin_tracking,
            "yaw": yaw_tracking,
            "vertical_velocity": vertical_velocity,
            "height": height,
            "orientation": orientation,
            "torque": torque,
            "smooth": smooth,
            "contact": contact,
            "unhealthy": unhealthy,
        }
        return float(sum(components.values())), components

    def _is_unhealthy(self) -> bool:
        projected_gravity = self._projected_gravity()
        base_height = float(self.data.qpos[self.root_qposadr + 2])
        too_low = base_height < self.stand_height - 0.45
        too_high = base_height > self.stand_height + 0.65
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
            "curriculum_level": float(self.curriculum_level),
        }
