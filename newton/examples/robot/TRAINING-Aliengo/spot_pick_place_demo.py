from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from play_spot_policy import patch_sb3_zip_loader
from spot_go2_style_env import SpotGo2StyleEnv


SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_PATH = SCRIPT_DIR / "spot_scene.xml"
MODEL_PATH = SCRIPT_DIR / "runs" / "spot_go2_style_20m" / "best_eval" / "best_model.zip"
VECNORMALIZE_PATH = MODEL_PATH.parent / "best_vecnormalize.pkl"

A_POINT = np.array([5.0, -6.0], dtype=np.float64)
B_POINT = np.array([11.0, -6.0], dtype=np.float64)
A_YAW = 0.0

ARRIVAL_RADIUS = 0.35
MAX_SECONDS = 45.0
FORWARD_SPEED = 0.45
MIN_FORWARD_SPEED = 0.25
MAX_YAW_RATE = 0.5
YAW_GAIN = 1.4


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64)


def _base_xy(env: SpotGo2StyleEnv) -> np.ndarray:
    return env.data.qpos[env.root_qposadr : env.root_qposadr + 2].copy()


def _base_yaw(env: SpotGo2StyleEnv) -> float:
    quat = env.data.qpos[env.root_qposadr + 3 : env.root_qposadr + 7]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat)
    mat = rotation.reshape(3, 3)
    return float(math.atan2(mat[1, 0], mat[0, 0]))


def _set_start_pose(env: SpotGo2StyleEnv) -> None:
    env.reset(seed=1, options={"command": np.zeros(3, dtype=np.float32)})
    root = env.root_qposadr
    env.data.qpos[root : root + 2] = A_POINT
    env.data.qpos[root + 2] = env.reset_base_height
    env.data.qpos[root + 3 : root + 7] = _yaw_to_quat(A_YAW)
    env.data.qvel[:] = 0.0
    env.last_action.fill(0.0)
    mujoco.mj_forward(env.model, env.data)


def _command_to_target(env: SpotGo2StyleEnv, target: np.ndarray) -> tuple[np.ndarray, float]:
    delta = target - _base_xy(env)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return np.zeros(3, dtype=np.float32), distance

    target_yaw = float(math.atan2(delta[1], delta[0]))
    heading_error = _wrap_angle(target_yaw - _base_yaw(env))
    forward = float(np.clip(distance * 0.5, MIN_FORWARD_SPEED, FORWARD_SPEED))
    if abs(heading_error) > 0.9:
        forward = MIN_FORWARD_SPEED

    command = np.array(
        [
            forward,
            0.0,
            np.clip(YAW_GAIN * heading_error, -MAX_YAW_RATE, MAX_YAW_RATE),
        ],
        dtype=np.float32,
    )
    return command, distance


def _normalized_observation(env: SpotGo2StyleEnv, vecnormalize: VecNormalize) -> np.ndarray:
    raw_obs = env._get_obs()
    return vecnormalize.normalize_obs(raw_obs[np.newaxis, :])


def _hold_stand_until_closed(env: SpotGo2StyleEnv) -> None:
    env.render()
    while env.viewer is None or env.viewer.is_running():
        env.data.ctrl[env.leg_actuator_ids] = env.nominal_leg_ctrl
        env.data.ctrl[env.arm_actuator_ids] = env.stand_arm_ctrl
        for _ in range(env.control_decimation):
            mujoco.mj_step(env.model, env.data)
        env.render()
        time.sleep(env.dt)


def main() -> None:
    patch_sb3_zip_loader()

    raw_env = SpotGo2StyleEnv(
        xml_path=SCENE_PATH,
        episode_seconds=MAX_SECONDS,
        command_range=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        randomize_domain=False,
        use_curriculum=False,
        render_mode="human",
        render_camera="tracking_side_view",
    )
    vec_env = VecNormalize.load(VECNORMALIZE_PATH, DummyVecEnv([lambda: raw_env]))
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=vec_env)
    _set_start_pose(raw_env)

    try:
        start_time = time.time()
        reached_b = False
        while time.time() - start_time < MAX_SECONDS:
            command, distance = _command_to_target(raw_env, B_POINT)
            raw_env.command = command

            if distance <= ARRIVAL_RADIUS:
                reached_b = True
                break

            obs = _normalized_observation(raw_env, vec_env)
            action, _ = model.predict(obs, deterministic=True)
            _, _, terminated, truncated, _ = raw_env.step(action[0])

            if terminated or truncated:
                break
            time.sleep(raw_env.dt)

        raw_env.command = np.zeros(3, dtype=np.float32)
        if reached_b:
            print(f"Reached B: position={_base_xy(raw_env)}, target={B_POINT}")
        else:
            print(f"Stopped before B: position={_base_xy(raw_env)}, target={B_POINT}")
        print("Holding position. Close the MuJoCo window or press Ctrl+C to exit.")
        _hold_stand_until_closed(raw_env)
    except KeyboardInterrupt:
        pass
    finally:
        raw_env.close()
        vec_env.close()


if __name__ == "__main__":
    main()
