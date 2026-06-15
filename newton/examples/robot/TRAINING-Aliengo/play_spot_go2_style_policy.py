from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from play_spot_policy import patch_sb3_zip_loader
from spot_go2_style_env import SpotGo2StyleEnv
from train_spot_go2_style_ppo import _resolve_scene_path


def _resolve_vecnormalize_path(model_path: Path, vecnormalize_path: Path | None) -> Path | None:
    if vecnormalize_path is not None:
        return vecnormalize_path

    run_vecnormalize = model_path.parent / "vecnormalize.pkl"
    if run_vecnormalize.exists():
        return run_vecnormalize

    best_vecnormalize = model_path.parent / "best_vecnormalize.pkl"
    if best_vecnormalize.exists():
        return best_vecnormalize

    match = re.match(r"(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$", model_path.name)
    if match:
        checkpoint_vecnormalize = (
            model_path.parent / f"{match.group('prefix')}_vecnormalize_{match.group('steps')}_steps.pkl"
        )
        if checkpoint_vecnormalize.exists():
            return checkpoint_vecnormalize

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Spot Go2-style PPO policy.")
    parser.add_argument("--xml", type=Path, default=Path("spot_scene.xml"))
    parser.add_argument("--model", type=Path, default=Path("runs/spot_go2_style_walk/ppo_spot_go2_style_final.zip"))
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--reset-base-height", type=float, default=1.72)
    parser.add_argument("--target-base-height", type=float, default=1.68)
    parser.add_argument("--control-decimation", type=int, default=10)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--nominal-leg-ctrl", type=float, nargs=3, default=(0.0, -0.1, 0.3), metavar=("HX", "HY", "KN"))
    parser.add_argument("--actuator-gain-scale", type=float, default=3.0)
    parser.add_argument("--command-vx", type=float, default=0.3)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--render-camera", type=str, default="tracking_side_view")
    args = parser.parse_args()

    patch_sb3_zip_loader()

    command_range = (
        (args.command_vx, args.command_vx),
        (args.command_vy, args.command_vy),
        (args.command_yaw, args.command_yaw),
    )
    raw_env = SpotGo2StyleEnv(
        xml_path=_resolve_scene_path(args.xml),
        command_range=command_range,
        reset_base_height=args.reset_base_height,
        target_base_height=args.target_base_height,
        control_decimation=args.control_decimation,
        action_scale=args.action_scale,
        nominal_leg_ctrl=tuple(args.nominal_leg_ctrl),
        actuator_gain_scale=args.actuator_gain_scale,
        randomize_domain=False,
        use_curriculum=False,
        render_mode="human",
        render_camera=args.render_camera or None,
    )
    vec_env = DummyVecEnv([lambda: raw_env])
    vecnormalize_path = _resolve_vecnormalize_path(args.model, args.vecnormalize)
    if vecnormalize_path is not None and vecnormalize_path.exists():
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"Warning: VecNormalize stats not found for {args.model}; replaying with raw observations.")

    model = PPO.load(args.model, env=vec_env)
    obs = vec_env.reset()

    end_time = time.time() + args.seconds
    while time.time() < end_time:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
        raw_env.render()
        if done[0]:
            obs = vec_env.reset()
        time.sleep(raw_env.dt)

    vec_env.close()


if __name__ == "__main__":
    main()
