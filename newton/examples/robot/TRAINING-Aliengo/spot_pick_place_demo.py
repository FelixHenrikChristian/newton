from __future__ import annotations

import math
import tempfile
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
STONE_FREEJOINT_NAME = "pickup_stone_free"

A_POINT = np.array([5.0, -6.0], dtype=np.float64)
B_POINT = np.array([9.0, -3.5], dtype=np.float64)
A_YAW = 0.0

STONE_FORWARD_OFFSET = 0.55
STONE_HEADING_OFFSET = 0.65
STONE_SIZE = np.array([0.06, 0.045, 0.035], dtype=np.float64)
STONE_CLEARANCE = 0.01
STONE_ALIGN_RADIUS = 1.2

ARRIVAL_RADIUS = 0.35
ARRIVAL_YAW_TOLERANCE = 0.2
MAX_SECONDS = 45.0
FORWARD_SPEED = 0.45
MIN_FORWARD_SPEED = 0.25
MAX_YAW_RATE = 0.5
YAW_GAIN = 1.4


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64)


def _approach_yaw() -> float:
    direction = B_POINT - A_POINT
    if float(np.linalg.norm(direction)) < 1e-6:
        return A_YAW
    return float(math.atan2(direction[1], direction[0]))


def _stone_xy() -> np.ndarray:
    yaw = _approach_yaw() + STONE_HEADING_OFFSET
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    return B_POINT + forward * STONE_FORWARD_OFFSET


def _terrain_height_at(xy: np.ndarray) -> float:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    origin = np.array([xy[0], xy[1], 10.0], dtype=np.float64)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction, None, True, -1, geom_id, None)
    if distance < 0.0:
        return 0.0
    return float(origin[2] - distance)


def _write_demo_scene() -> tuple[Path, np.ndarray]:
    xy = _stone_xy()
    z = _terrain_height_at(xy) + STONE_SIZE[2] + STONE_CLEARANCE
    stone_pos = np.array([xy[0], xy[1], z], dtype=np.float64)

    stone_xml = f"""
    <body name="pickup_stone" pos="{stone_pos[0]:.6f} {stone_pos[1]:.6f} {stone_pos[2]:.6f}">
      <freejoint name="pickup_stone_free" />
      <geom name="pickup_stone_geom" type="ellipsoid" size="{STONE_SIZE[0]:.6f} {STONE_SIZE[1]:.6f} {STONE_SIZE[2]:.6f}" rgba="0.18 0.17 0.15 1" density="1800" friction="1.5 0.03 0.01" condim="6" />
    </body>"""

    scene_xml = SCENE_PATH.read_text(encoding="utf-8")
    scene_xml = scene_xml.replace("\n  </worldbody>", f"{stone_xml}\n  </worldbody>", 1)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".xml",
        prefix="_spot_pick_place_scene_",
        dir=SCRIPT_DIR,
        delete=False,
    ) as scene_file:
        scene_file.write(scene_xml)
        return Path(scene_file.name), stone_pos


def _base_xy(env: SpotGo2StyleEnv) -> np.ndarray:
    return env.data.qpos[env.root_qposadr : env.root_qposadr + 2].copy()


def _base_yaw(env: SpotGo2StyleEnv) -> float:
    quat = env.data.qpos[env.root_qposadr + 3 : env.root_qposadr + 7]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat)
    mat = rotation.reshape(3, 3)
    return float(math.atan2(mat[1, 0], mat[0, 0]))


def _set_stone_pose(env: SpotGo2StyleEnv, stone_pos: np.ndarray) -> None:
    stone_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, STONE_FREEJOINT_NAME)
    if stone_joint_id < 0:
        return

    qpos_id = int(env.model.jnt_qposadr[stone_joint_id])
    qvel_id = int(env.model.jnt_dofadr[stone_joint_id])
    env.data.qpos[qpos_id : qpos_id + 3] = stone_pos
    env.data.qpos[qpos_id + 3 : qpos_id + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env.data.qvel[qvel_id : qvel_id + 6] = 0.0


def _set_start_pose(env: SpotGo2StyleEnv, stone_pos: np.ndarray | None = None) -> None:
    env.reset(seed=1, options={"command": np.zeros(3, dtype=np.float32)})
    root = env.root_qposadr
    env.data.qpos[root : root + 2] = A_POINT
    env.data.qpos[root + 2] = env.reset_base_height
    env.data.qpos[root + 3 : root + 7] = _yaw_to_quat(A_YAW)
    env.data.qvel[:] = 0.0
    env.last_action.fill(0.0)
    if stone_pos is not None:
        _set_stone_pose(env, stone_pos)
    mujoco.mj_forward(env.model, env.data)


def _command_to_target(
    env: SpotGo2StyleEnv,
    target: np.ndarray,
    face_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    delta = target - _base_xy(env)
    distance = float(np.linalg.norm(delta))
    near_target = distance <= ARRIVAL_RADIUS

    if face_xy is not None and distance <= STONE_ALIGN_RADIUS:
        face_delta = face_xy - _base_xy(env)
        target_yaw = float(math.atan2(face_delta[1], face_delta[0]))
    elif distance > 1e-6:
        target_yaw = float(math.atan2(delta[1], delta[0]))
    else:
        target_yaw = _base_yaw(env)

    heading_error = _wrap_angle(target_yaw - _base_yaw(env))
    forward = 0.0 if near_target else float(np.clip(distance * 0.5, MIN_FORWARD_SPEED, FORWARD_SPEED))
    if not near_target and abs(heading_error) > 0.9:
        forward = MIN_FORWARD_SPEED

    command = np.array(
        [
            forward,
            0.0,
            np.clip(YAW_GAIN * heading_error, -MAX_YAW_RATE, MAX_YAW_RATE),
        ],
        dtype=np.float32,
    )
    return command, distance, abs(heading_error)


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
    scene_path, stone_pos = _write_demo_scene()

    raw_env = None
    vec_env = None
    try:
        raw_env = SpotGo2StyleEnv(
            xml_path=scene_path,
            episode_seconds=MAX_SECONDS,
            command_range=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            randomize_domain=False,
            use_curriculum=False,
            render_mode="human",
            render_camera="tracking_side_view",
        )
        scene_path.unlink(missing_ok=True)
        vec_env = VecNormalize.load(VECNORMALIZE_PATH, DummyVecEnv([lambda: raw_env]))
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(MODEL_PATH, env=vec_env)
        _set_start_pose(raw_env, stone_pos)
        print(f"Stone position: {stone_pos.round(3).tolist()}")

        start_time = time.time()
        reached_b = False
        while time.time() - start_time < MAX_SECONDS:
            command, distance, yaw_error = _command_to_target(raw_env, B_POINT, stone_pos[:2])
            raw_env.command = command

            if distance <= ARRIVAL_RADIUS and yaw_error <= ARRIVAL_YAW_TOLERANCE:
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
        scene_path.unlink(missing_ok=True)
        if raw_env is not None:
            raw_env.close()
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main()
