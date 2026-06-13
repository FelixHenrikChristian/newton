from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spot_env import SpotWalkEnv


SCRIPT_DIR = Path(__file__).resolve().parent
VecEnvType = str


def _resolve_scene_path(xml_path: Path) -> Path:
    if xml_path.is_absolute() or xml_path.exists():
        return xml_path
    return SCRIPT_DIR / xml_path


def make_env(xml_path: Path, seed: int, rank: int, reset_base_height: float):
    def _init():
        env = SpotWalkEnv(xml_path=xml_path, reset_base_height=reset_base_height)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env(env_fns: list, vec_env_type: VecEnvType):
    if vec_env_type == "auto":
        vec_env_type = "subproc" if len(env_fns) > 1 else "dummy"
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spot walking with PPO.")
    parser.add_argument("--xml", type=Path, default=Path("spot_scene.xml"))
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="spot_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reset-base-height", type=float, default=1.68)
    parser.add_argument("--vec-env", choices=("auto", "dummy", "subproc"), default="auto")
    parser.add_argument("--n-steps", type=int, default=2048)
    args = parser.parse_args()

    xml_path = _resolve_scene_path(args.xml)
    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(xml_path, args.seed, i, args.reset_base_height) for i in range(args.num_envs)]
    env = make_vec_env(env_fns, args.vec_env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=1.0,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_spot",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(run_dir / "ppo_spot_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
