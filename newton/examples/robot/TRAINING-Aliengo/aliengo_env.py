from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


LEG_ACTUATORS = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)

FOOT_GEOMS = (
    "FR_calf_foot_collision",
    "FL_calf_foot_collision",
    "RR_calf_foot_collision",
    "RL_calf_foot_collision",
)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


class AliengoWalkEnv(gym.Env):
    """Gymnasium environment for training Aliengo locomotion in MuJoCo.

    The MJCF actuator bias terms make ctrl=0 match the stand keyframe. Actions
    therefore command offsets around the built-in standing pose, not absolute
    joint angles.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "aliengo_scene.xml",
        frame_skip: int = 10,
        episode_seconds: float = 12.0,
        command_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.2, 0.8),
            (-0.2, 0.2),
            (-0.6, 0.6),
        ),
        render_mode: str | None = None,
    ) -> None:
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.command_range = command_range
        self.render_mode = render_mode

        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in LEG_ACTUATORS],
            dtype=np.int32,
        )
        if np.any(self.actuator_ids < 0):
            missing = [name for name, idx in zip(LEG_ACTUATORS, self.actuator_ids) if idx < 0]
            raise ValueError(f"Missing actuators in MJCF: {missing}")

        self.foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOMS],
            dtype=np.int32,
        )
        self.terrain_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "lunar_terrain"
        )

        self.action_scale = np.array(
            [0.25, 0.65, 0.65] * 4,
            dtype=np.float32,
        )
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.stand_key_id < 0:
            raise ValueError("Expected a keyframe named 'stand' in aliengo_scene.xml")

        self.stand_qpos = self.model.key_qpos[self.stand_key_id].copy()
        self.stand_joint_qpos = self.stand_qpos[7:19].astype(np.float32)
        self.stand_height = float(self.stand_qpos[2])

        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(52,), dtype=np.float32)

        self.viewer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)

        joint_noise = self.np_random.uniform(-0.03, 0.03, size=12)
        self.data.qpos[7:19] = self.stand_joint_qpos + joint_noise
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        self.data.ctrl[self.actuator_ids] = 0.0

        if options and "command" in options:
            self.command = np.asarray(options["command"], dtype=np.float32)
        else:
            self.command = self._sample_command()

        self.last_action.fill(0.0)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        ctrl = np.clip(action * self.action_scale, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[self.actuator_ids] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_terms = self._reward(action)

        self.step_count += 1
        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.max_steps
        self.last_action = action.copy()

        info = self._get_info()
        info.update(reward_terms)
        return obs, reward, terminated, truncated, info

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

    def _sample_command(self) -> np.ndarray:
        ranges = np.asarray(self.command_range, dtype=np.float32)
        return self.np_random.uniform(ranges[:, 0], ranges[:, 1]).astype(np.float32)

    def _base_rotation(self) -> np.ndarray:
        return _quat_to_matrix(self.data.qpos[3:7])

    def _projected_gravity(self) -> np.ndarray:
        rotation = self._base_rotation()
        return rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _base_velocity_body(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self._base_rotation()
        linear = rotation.T @ self.data.qvel[0:3]
        angular = rotation.T @ self.data.qvel[3:6]
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
        base_linear, base_angular = self._base_velocity_body()
        joint_pos = self.data.qpos[7:19] - self.stand_joint_qpos
        joint_vel = self.data.qvel[6:18]
        contacts = self._foot_contacts()

        obs = np.concatenate(
            [
                projected_gravity,
                base_linear,
                base_angular,
                self.command,
                joint_pos,
                joint_vel,
                self.last_action,
                contacts,
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        base_linear, base_angular = self._base_velocity_body()
        velocity_error = np.array(
            [
                base_linear[0] - self.command[0],
                base_linear[1] - self.command[1],
                base_angular[2] - self.command[2],
            ],
            dtype=np.float64,
        )
        tracking = float(np.exp(-np.dot(velocity_error, velocity_error) / 0.25))

        projected_gravity = self._projected_gravity()
        upright = float(np.clip(-projected_gravity[2], 0.0, 1.0))
        height_error = abs(float(self.data.qpos[2]) - self.stand_height)
        height = float(np.exp(-(height_error * height_error) / 0.09))

        action_rate = float(np.sum(np.square(action - self.last_action)))
        action_size = float(np.sum(np.square(action)))
        joint_speed = float(np.sum(np.square(self.data.qvel[6:18])))
        unhealthy = 1.0 if self._is_unhealthy() else 0.0

        reward = (
            2.0 * tracking
            + 0.5 * upright
            + 0.25 * height
            - 0.03 * action_rate
            - 0.005 * action_size
            - 0.0005 * joint_speed
            - 2.0 * unhealthy
        )

        return float(reward), {
            "reward_tracking": tracking,
            "reward_upright": upright,
            "reward_height": height,
            "penalty_action_rate": action_rate,
            "penalty_action_size": action_size,
            "penalty_joint_speed": joint_speed,
        }

    def _is_unhealthy(self) -> bool:
        projected_gravity = self._projected_gravity()
        too_low = float(self.data.qpos[2]) < self.stand_height - 0.45
        tipped = projected_gravity[2] > -0.35
        bad_number = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        return bool(too_low or tipped or bad_number)

    def _get_info(self) -> dict[str, float]:
        base_linear, base_angular = self._base_velocity_body()
        return {
            "command_vx": float(self.command[0]),
            "command_vy": float(self.command[1]),
            "command_yaw": float(self.command[2]),
            "base_vx": float(base_linear[0]),
            "base_vy": float(base_linear[1]),
            "base_yaw_rate": float(base_angular[2]),
            "base_height": float(self.data.qpos[2]),
        }
