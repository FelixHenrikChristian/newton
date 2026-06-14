from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from spot_go2_style_env import SpotGo2StyleEnv


SCRIPT_DIR = Path(__file__).resolve().parent
VecEnvType = str


class RewardComponentCallback(BaseCallback):
    """Log averaged reward components to TensorBoard."""

    def __init__(self, log_interval: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_interval = log_interval
        self.sums: dict[str, float] = {}
        self.count = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            components = info.get("reward_components")
            if not components:
                continue
            for name, value in components.items():
                self.sums[name] = self.sums.get(name, 0.0) + float(value)
            self.count += 1

        if self.count >= self.log_interval:
            for name, value in self.sums.items():
                self.logger.record(f"reward/{name}", value / self.count)
            self.sums.clear()
            self.count = 0
        return True


def _resolve_scene_path(xml_path: Path) -> Path:
    if xml_path.is_absolute() or xml_path.exists():
        return xml_path
    return SCRIPT_DIR / xml_path


def make_env(
    xml_path: Path,
    seed: int,
    rank: int,
    reset_base_height: float,
    target_base_height: float,
    randomize_domain: bool,
    use_curriculum: bool,
    command_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    action_scale: float,
    nominal_leg_ctrl: tuple[float, float, float],
    actuator_gain_scale: float,
    control_decimation: int,
):
    def _init():
        env = SpotGo2StyleEnv(
            xml_path=xml_path,
            reset_base_height=reset_base_height,
            target_base_height=target_base_height,
            randomize_domain=randomize_domain,
            use_curriculum=use_curriculum,
            command_range=command_range,
            action_scale=action_scale,
            nominal_leg_ctrl=nominal_leg_ctrl,
            actuator_gain_scale=actuator_gain_scale,
            control_decimation=control_decimation,
        )
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
    parser = argparse.ArgumentParser(description="Train Spot locomotion with a Go2-style MuJoCo PPO setup.")
    parser.add_argument("--xml", type=Path, default=Path("spot_scene.xml"))
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="spot_go2_style_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vec-env", choices=("auto", "dummy", "subproc"), default="auto")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reset-base-height", type=float, default=1.72)
    parser.add_argument("--target-base-height", type=float, default=1.68)
    parser.add_argument("--control-decimation", type=int, default=10)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--nominal-leg-ctrl", type=float, nargs=3, default=(0.0, -0.1, 0.3), metavar=("HX", "HY", "KN"))
    parser.add_argument("--actuator-gain-scale", type=float, default=3.0)
    parser.add_argument("--command-vx", type=float, nargs=2, default=(0.1, 0.6), metavar=("MIN", "MAX"))
    parser.add_argument("--command-vy", type=float, nargs=2, default=(-0.2, 0.2), metavar=("MIN", "MAX"))
    parser.add_argument("--command-yaw", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()

    xml_path = _resolve_scene_path(args.xml)
    run_dir = Path("runs") / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    command_range = (tuple(args.command_vx), tuple(args.command_vy), tuple(args.command_yaw))
    env_fns = [
        make_env(
            xml_path,
            args.seed,
            i,
            args.reset_base_height,
            args.target_base_height,
            not args.no_domain_randomization,
            not args.no_curriculum,
            command_range,
            args.action_scale,
            tuple(args.nominal_leg_ctrl),
            args.actuator_gain_scale,
            args.control_decimation,
        )
        for i in range(args.num_envs)
    ]
    env = make_vec_env(env_fns, args.vec_env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    rollout_size = args.n_steps * args.num_envs
    batch_size = args.batch_size or min(max(args.num_envs * 128, 256), rollout_size)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=1.0,
        policy_kwargs={"net_arch": [512, 256, 128]},
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_spot_go2_style",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[RewardComponentCallback(), checkpoint_callback],
        progress_bar=True,
    )

    model.save(run_dir / "ppo_spot_go2_style_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
