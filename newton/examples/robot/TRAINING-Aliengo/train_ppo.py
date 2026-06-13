from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from aliengo_env import AliengoWalkEnv


VecEnvType = str


def make_env(xml_path: Path, seed: int, rank: int):
    def _init():
        env = AliengoWalkEnv(xml_path=xml_path)
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
    parser = argparse.ArgumentParser(description="Train Aliengo walking with PPO.")
    parser.add_argument("--xml", type=Path, default=Path("aliengo_scene.xml"))
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="aliengo_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vec-env", choices=("auto", "dummy", "subproc"), default="auto")
    parser.add_argument("--n-steps", type=int, default=2048)
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(args.xml, args.seed, i) for i in range(args.num_envs)]
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
        name_prefix="ppo_aliengo",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(run_dir / "ppo_aliengo_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
