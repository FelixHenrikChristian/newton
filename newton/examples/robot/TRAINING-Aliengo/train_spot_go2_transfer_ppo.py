from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spot_go2_transfer_env import SpotGo2TransferEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE = SCRIPT_DIR / "spot_scene.xml"
DEFAULT_GO2_RUN = SCRIPT_DIR.parents[4] / "Robot_on_moon_go2" / "runs" / "go2_lunar_3p5M"
DEFAULT_GO2_MODEL = DEFAULT_GO2_RUN / "ppo_go2_final.zip"
DEFAULT_GO2_VECNORMALIZE = DEFAULT_GO2_RUN / "vecnormalize.pkl"

REWARD_KEYS = (
    "reward_tracking",
    "reward_upright",
    "reward_height",
    "reward_trot_match",
    "penalty_same_side_contact",
    "penalty_all_off",
    "penalty_action_rate",
    "penalty_action_size",
    "penalty_joint_speed",
    "penalty_orientation",
    "penalty_vertical_velocity",
    "penalty_body_angular_xy",
    "penalty_nonfoot_contact",
)


class RewardComponentCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for key in REWARD_KEYS:
            values = [float(info[key]) for info in infos if key in info]
            if values:
                self.logger.record_mean(f"reward/{key}", float(np.mean(values)))
        return True


class NaturalGaitEvalCallback(BaseCallback):
    """Save the checkpoint that tracks commands with a stable diagonal trot."""

    def __init__(
        self,
        eval_env: SpotGo2TransferEnv,
        save_dir: Path,
        eval_freq: int,
        episode_seconds: float = 8.0,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.eval_freq = int(eval_freq)
        self.eval_steps = int(episode_seconds / eval_env.dt)
        self.last_eval_step = 0
        self.best_score = -np.inf
        self.cases = (
            ("forward_empty", np.array((0.45, 0.0, 0.0), dtype=np.float32), 0.0),
            ("forward_payload", np.array((0.45, 0.0, 0.0), dtype=np.float32), eval_env.payload_mass),
            ("turn_payload", np.array((0.40, 0.0, 0.25), dtype=np.float32), eval_env.payload_mass),
            ("lateral_empty", np.array((0.35, -0.10, 0.0), dtype=np.float32), 0.0),
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.training_env, VecNormalize):
            return self.training_env.normalize_obs(obs[np.newaxis, :])
        return obs[np.newaxis, :]

    def _evaluate_command(
        self,
        command: np.ndarray,
        payload_mass: float,
    ) -> dict[str, float | bool | list[float]]:
        obs, _ = self.eval_env.reset(options={"spawn": (5.0, -6.0), "command": command, "payload_mass": payload_mass})
        velocity_errors = []
        yaw_errors = []
        contacts = []
        action_saturation = []
        tilt_degrees = []
        clearance_errors = []
        terminated = False

        for _ in range(self.eval_steps):
            action, _ = self.model.predict(self._normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = self.eval_env.step(action[0])
            velocity_errors.append(abs(float(info["base_vx"]) - float(command[0])))
            yaw_errors.append(abs(float(info["base_yaw_rate"]) - float(command[2])))
            contacts.append(
                (
                    info["contact_fr"],
                    info["contact_fl"],
                    info["contact_hr"],
                    info["contact_hl"],
                )
            )
            action_saturation.append(float(np.mean(np.abs(action[0]) > 0.95)))
            tilt_degrees.append(float(info["base_tilt_deg"]))
            clearance_errors.append(float(info["base_clearance_error"]))
            if terminated or truncated:
                break

        contact_array = np.asarray(contacts, dtype=np.float32)
        diagonal_a = np.all(contact_array == (1.0, 0.0, 0.0, 1.0), axis=1)
        diagonal_b = np.all(contact_array == (0.0, 1.0, 1.0, 0.0), axis=1)
        return {
            "no_fall": not terminated,
            "steps": len(contacts),
            "vx_mae": float(np.mean(velocity_errors)),
            "yaw_mae": float(np.mean(yaw_errors)),
            "duty": np.mean(contact_array, axis=0).tolist(),
            "exact_diagonal_fraction": float(np.mean(diagonal_a | diagonal_b)),
            "all_four_fraction": float(np.mean(np.sum(contact_array, axis=1) == 4.0)),
            "flight_fraction": float(np.mean(np.sum(contact_array, axis=1) == 0.0)),
            "action_saturation": float(np.mean(action_saturation)),
            "mean_tilt_deg": float(np.mean(tilt_degrees)),
            "max_tilt_deg": float(np.max(tilt_degrees)),
            "clearance_mae": float(np.mean(clearance_errors)),
        }

    @staticmethod
    def _score(results: dict[str, dict[str, float | bool | list[float]]]) -> float:
        score = 0.0
        for result in results.values():
            if not result["no_fall"]:
                score -= 100.0
            score -= 10.0 * float(result["vx_mae"])
            score -= 3.0 * float(result["yaw_mae"])
            score += 4.0 * float(result["exact_diagonal_fraction"])
            score -= 2.0 * float(result["all_four_fraction"])
            score -= float(result["action_saturation"])
            score -= 0.25 * float(result["mean_tilt_deg"])
            score -= 0.10 * max(float(result["max_tilt_deg"]) - 12.0, 0.0)
            score -= 5.0 * float(result["clearance_mae"])
        return score

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps

        results = {name: self._evaluate_command(command, payload_mass) for name, command, payload_mass in self.cases}
        score = self._score(results)
        self.logger.record("eval/natural_gait_score", score)
        print(f"Natural gait eval at {self.num_timesteps}: score={score:.3f}")

        if score > self.best_score:
            self.best_score = score
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "best_model")
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.save_dir / "best_vecnormalize.pkl")
            metrics = {"num_timesteps": self.num_timesteps, "score": score, "commands": results}
            (self.save_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


def make_env(
    xml_path: Path,
    seed: int,
    rank: int,
    gravity_z: float | None,
    arm_pose_noise: float,
    payload_mass: float,
    payload_probability: float,
    randomize_spawn: bool,
):
    def _init():
        env = SpotGo2TransferEnv(
            xml_path=xml_path,
            gravity_z=gravity_z,
            arm_pose_noise=arm_pose_noise,
            payload_mass=payload_mass,
            payload_probability=payload_probability,
            randomize_spawn=randomize_spawn,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the original Go2 lunar PPO policy on Spot.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--total-timesteps", type=int, default=3_500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", default="spot_go2_transfer_3p5m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--gravity-z", type=float, default=None, help="Override XML gravity, e.g. -1.62 for Moon.")
    parser.add_argument("--arm-pose-noise", type=float, default=0.0)
    parser.add_argument("--payload-mass", type=float, default=0.72)
    parser.add_argument("--payload-probability", type=float, default=0.5)
    parser.add_argument("--fixed-spawn", action="store_true")
    parser.add_argument("--init-from", type=Path, default=DEFAULT_GO2_MODEL)
    parser.add_argument("--init-vecnormalize", type=Path, default=DEFAULT_GO2_VECNORMALIZE)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    xml_path = args.xml.resolve()
    run_dir = SCRIPT_DIR / "runs" / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            xml_path,
            args.seed,
            rank,
            args.gravity_z,
            args.arm_pose_noise,
            args.payload_mass,
            args.payload_probability,
            not args.fixed_spawn,
        )
        for rank in range(args.num_envs)
    ]
    if args.vec_env == "subproc" and args.num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    if args.from_scratch:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=1.0,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
        )
    else:
        if not args.init_from.exists() or not args.init_vecnormalize.exists():
            raise FileNotFoundError(
                "Go2 warm-start artifacts are missing. Pass --init-from and --init-vecnormalize, or use --from-scratch."
            )
        env = VecNormalize.load(args.init_vecnormalize, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(
            args.init_from,
            env=env,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        print(f"Warm-starting from {args.init_from}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_spot_go2_transfer",
        save_vecnormalize=True,
    )
    eval_env = SpotGo2TransferEnv(
        xml_path=xml_path,
        gravity_z=args.gravity_z,
        randomize_spawn=False,
        payload_mass=args.payload_mass,
        payload_probability=args.payload_probability,
    )
    callbacks: list[BaseCallback] = [
        RewardComponentCallback(),
        checkpoint_callback,
        NaturalGaitEvalCallback(eval_env, run_dir / "best_eval", args.eval_freq),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar,
        reset_num_timesteps=True,
    )
    model.save(run_dir / "ppo_spot_go2_transfer_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    print(f"Saved model and normalization statistics to {run_dir}")


if __name__ == "__main__":
    main()
