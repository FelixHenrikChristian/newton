from __future__ import annotations

import argparse
import time
from pathlib import Path

from spot_go2_transfer_env import SpotGo2TransferEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = SCRIPT_DIR / "runs" / "spot_go2_transfer_3p5m"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a 54-D Go2-transfer policy on Spot.")
    parser.add_argument("--xml", type=Path, default=SCRIPT_DIR / "spot_scene.xml")
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN / "best_eval" / "best_model.zip")
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=DEFAULT_RUN / "best_eval" / "best_vecnormalize.pkl",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--command-vx", type=float, default=0.45)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--gravity-z", type=float, default=None)
    parser.add_argument("--render-camera", default="tracking_side_view")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    command = (args.command_vx, args.command_vy, args.command_yaw)
    raw_env = SpotGo2TransferEnv(
        xml_path=args.xml.resolve(),
        command_range=tuple((value, value) for value in command),
        gravity_z=args.gravity_z,
        randomize_spawn=False,
        payload_mass=args.payload_mass,
        payload_probability=1.0 if args.payload_mass > 0.0 else 0.0,
        render_mode="human",
        render_camera=args.render_camera or None,
    )
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize.resolve(), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model.resolve(), env=vec_env)
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
