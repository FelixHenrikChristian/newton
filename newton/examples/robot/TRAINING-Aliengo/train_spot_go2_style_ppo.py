from __future__ import annotations

import argparse
import copy
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from spot_go2_style_env import FOOT_GEOMS, LEG_JOINTS, OBS_DIM, SpotGo2StyleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

SCRIPT_DIR = Path(__file__).resolve().parent
VecEnvType = str


@dataclass(frozen=True)
class EvalScenario:
    """Deterministic command and terrain-start combination used for evaluation."""

    name: str
    command: tuple[float, float, float]
    spawn_offset: tuple[float, float]
    spawn_yaw: float = 0.0


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


class TrainingHealthCallback(BaseCallback):
    """Stop training if the Gaussian policy variance remains unusably high."""

    def __init__(
        self,
        save_dir: Path,
        max_policy_std: float,
        check_freq: int = 100_000,
        patience: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = save_dir
        self.max_policy_std = max_policy_std
        self.check_freq = check_freq
        self.patience = patience
        self.last_check_step = 0
        self.excessive_std_checks = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check_step < self.check_freq:
            return True
        self.last_check_step = self.num_timesteps

        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is None:
            return True
        policy_std = float(log_std.detach().exp().mean().cpu().item())
        self.logger.record("health/policy_std", policy_std)
        if self.max_policy_std <= 0.0 or policy_std <= self.max_policy_std:
            self.excessive_std_checks = 0
            return True

        self.excessive_std_checks += 1
        if self.excessive_std_checks < self.patience:
            return True

        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "training_health_stop.txt").write_text(
            (
                f"num_timesteps={self.num_timesteps}\n"
                f"policy_std={policy_std:.6f}\n"
                f"max_policy_std={self.max_policy_std:.6f}\n"
            ),
            encoding="utf-8",
        )
        if self.verbose:
            print(f"Stopping training after policy std remained above {self.max_policy_std:.3f}.")
        return False


def _load_saved_score(save_dir: Path, score_name: str) -> float:
    score_path = save_dir / "best_score.txt"
    if not score_path.exists():
        return -float("inf")

    metrics: dict[str, float] = {}
    for line in score_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if not separator:
            continue
        try:
            metrics[name] = float(value)
        except ValueError:
            continue

    if score_name in metrics:
        return metrics[score_name]
    if score_name == "tracking_score":
        required = {"mean_survival", "mean_linear_rmse", "mean_yaw_rmse"}
        if required <= metrics.keys():
            return metrics["mean_survival"] - metrics["mean_linear_rmse"] - 0.25 * metrics["mean_yaw_rmse"]
    if score_name == "gait_score":
        required = {"mean_survival", "mean_gait_match", "mean_foot_slip"}
        if required <= metrics.keys():
            return metrics["mean_survival"] + metrics["mean_gait_match"] - 0.1 * metrics["mean_foot_slip"]
    return -float("inf")


class LocomotionEvalCallback(BaseCallback):
    """Evaluate fixed-command locomotion quality and save the best checkpoint."""

    def __init__(
        self,
        eval_env: VecNormalize,
        save_dir: Path,
        eval_freq: int,
        eval_repetitions: int,
        scenarios: tuple[EvalScenario, ...],
        early_stop_patience: int,
        early_stop_min_timesteps: int,
        early_stop_min_delta: float,
        restore_best_scores: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.eval_repetitions = eval_repetitions
        self.scenarios = scenarios
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_timesteps = early_stop_min_timesteps
        self.early_stop_min_delta = early_stop_min_delta
        if restore_best_scores:
            self.best_score = _load_saved_score(save_dir, "robust_score")
            self.best_tracking_score = _load_saved_score(save_dir.parent / "best_tracking", "tracking_score")
            self.best_gait_score = _load_saved_score(save_dir.parent / "best_gait", "gait_score")
        else:
            self.best_score = -float("inf")
            self.best_tracking_score = -float("inf")
            self.best_gait_score = -float("inf")
        self.last_eval_step = 0
        self.no_improvement_evals = 0

    def _sync_normalization(self) -> None:
        if isinstance(self.training_env, VecNormalize):
            self.eval_env.obs_rms = copy.deepcopy(self.training_env.obs_rms)
            self.eval_env.ret_rms = copy.deepcopy(self.training_env.ret_rms)
        self.eval_env.training = False
        self.eval_env.norm_reward = False

    def _evaluate_once(self, scenario: EvalScenario, seed: int) -> dict[str, float]:
        raw_env = self.eval_env.venv.envs[0].unwrapped
        raw_obs, _ = raw_env.reset(
            seed=seed,
            options={
                "command": scenario.command,
                "spawn_offset": scenario.spawn_offset,
                "spawn_yaw": scenario.spawn_yaw,
            },
        )
        obs = self.eval_env.normalize_obs(raw_obs[np.newaxis, :])
        score = 0.0
        steps = 0
        totals = {
            "forward_distance": 0.0,
            "linear_error_sq": 0.0,
            "yaw_error_sq": 0.0,
            "gait_match": 0.0,
            "foot_slip": 0.0,
        }
        max_steps = raw_env.max_steps
        dt = raw_env.dt
        for _ in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            raw_obs, _, terminated, truncated, info = raw_env.step(action[0])
            obs = self.eval_env.normalize_obs(raw_obs[np.newaxis, :])
            linear_error_sq = (float(info["base_vx"]) - float(info["command_vx"])) ** 2 + (
                float(info["base_vy"]) - float(info["command_vy"])
            ) ** 2
            yaw_error_sq = (float(info["base_yaw_rate"]) - float(info["command_yaw"])) ** 2
            linear_quality = float(np.exp(-linear_error_sq / 0.1))
            yaw_quality = float(np.exp(-yaw_error_sq / 0.25))
            foot_slip = float(info["foot_slip"])
            score += (
                linear_quality + 0.25 * yaw_quality + 0.25 * float(info["gait_match"]) - 0.05 * min(foot_slip, 4.0)
            ) * dt
            totals["forward_distance"] += float(info["base_vx"]) * dt
            totals["linear_error_sq"] += linear_error_sq
            totals["yaw_error_sq"] += yaw_error_sq
            totals["gait_match"] += float(info["gait_match"])
            totals["foot_slip"] += foot_slip
            steps += 1
            if terminated or truncated:
                if steps < max_steps:
                    score -= 2.0
                break
        count = max(steps, 1)
        return {
            "score": score,
            "survival": steps / max_steps,
            "forward_distance": totals["forward_distance"],
            "linear_rmse": float(np.sqrt(totals["linear_error_sq"] / count)),
            "yaw_rmse": float(np.sqrt(totals["yaw_error_sq"] / count)),
            "gait_match": totals["gait_match"] / count,
            "foot_slip": totals["foot_slip"] / count,
            "episode_length": float(steps),
        }

    def _save_checkpoint(self, save_dir: Path, summary: dict[str, float]) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(save_dir / "best_model")
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(save_dir / "best_vecnormalize.pkl")
        (save_dir / "best_score.txt").write_text(
            "".join(
                [f"num_timesteps={self.num_timesteps}\n"] + [f"{name}={value:.6f}\n" for name, value in summary.items()]
            ),
            encoding="utf-8",
        )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps
        self._sync_normalization()

        results = []
        scenario_scores: dict[str, list[float]] = {scenario.name: [] for scenario in self.scenarios}
        for scenario_index, scenario in enumerate(self.scenarios):
            for repetition in range(self.eval_repetitions):
                seed = 20_000 + 100 * scenario_index + repetition
                result = self._evaluate_once(scenario, seed)
                results.append(result)
                scenario_scores[scenario.name].append(result["score"])

        metric_names = results[0].keys()
        mean_metrics = {name: float(np.mean([result[name] for result in results])) for name in metric_names}
        worst_score = float(min(result["score"] for result in results))
        robust_score = 0.7 * mean_metrics["score"] + 0.3 * worst_score
        tracking_score = mean_metrics["survival"] - mean_metrics["linear_rmse"] - 0.25 * mean_metrics["yaw_rmse"]
        gait_score = mean_metrics["survival"] + mean_metrics["gait_match"] - 0.1 * mean_metrics["foot_slip"]
        summary = {
            "robust_score": robust_score,
            "tracking_score": tracking_score,
            "gait_score": gait_score,
            "worst_scenario_score": worst_score,
            **{f"mean_{name}": value for name, value in mean_metrics.items()},
        }
        self.logger.record("eval/robust_score", robust_score)
        self.logger.record("eval/worst_scenario_score", worst_score)
        for scenario_name, scores in scenario_scores.items():
            self.logger.record(f"eval_scenario/{scenario_name}", float(np.mean(scores)))
        for name, value in mean_metrics.items():
            self.logger.record(f"eval/{name}", value)

        previous_best_score = self.best_score
        new_best = robust_score > self.best_score
        meaningful_improvement = robust_score > self.best_score + self.early_stop_min_delta
        if new_best:
            self.best_score = robust_score
            self._save_checkpoint(self.save_dir, summary)
        if tracking_score > self.best_tracking_score:
            self.best_tracking_score = tracking_score
            self._save_checkpoint(self.save_dir.parent / "best_tracking", summary)
        if gait_score > self.best_gait_score:
            self.best_gait_score = gait_score
            self._save_checkpoint(self.save_dir.parent / "best_gait", summary)

        if meaningful_improvement or previous_best_score == -float("inf"):
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1

        self.logger.record("eval/best_robust_score", self.best_score)
        self.logger.record("eval/best_tracking_score", self.best_tracking_score)
        self.logger.record("eval/best_gait_score", self.best_gait_score)
        self.logger.record("eval/no_improvement_evals", self.no_improvement_evals)

        should_stop = (
            self.early_stop_patience > 0
            and self.num_timesteps >= self.early_stop_min_timesteps
            and self.no_improvement_evals >= self.early_stop_patience
        )
        if should_stop:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "early_stop.txt").write_text(
                (
                    f"num_timesteps={self.num_timesteps}\n"
                    f"best_robust_score={self.best_score:.6f}\n"
                    f"last_robust_score={robust_score:.6f}\n"
                    f"no_improvement_evals={self.no_improvement_evals}\n"
                ),
                encoding="utf-8",
            )
            if self.verbose:
                print(
                    "Stopping early: locomotion-quality eval did not improve for "
                    f"{self.no_improvement_evals} evaluations."
                )
            return False

        return True


def _resolve_scene_path(xml_path: Path) -> Path:
    if xml_path.is_absolute() or xml_path.exists():
        return xml_path.resolve()
    return (SCRIPT_DIR / xml_path).resolve()


def _compute_flat_terrain_height(
    source_xml_path: Path,
    reset_base_height: float,
    nominal_leg_ctrl: tuple[float, float, float],
) -> float:
    model = mujoco.MjModel.from_xml_path(str(source_xml_path))
    data = mujoco.MjData(model)

    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key_id < 0:
        raise ValueError(f"Expected a keyframe named 'stand' in {source_xml_path}")
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)

    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint")
    if root_joint_id < 0:
        raise ValueError(f"Expected a freejoint named 'freejoint' in {source_xml_path}")
    root_qposadr = int(model.jnt_qposadr[root_joint_id])
    data.qpos[root_qposadr + 2] = reset_base_height

    leg_joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LEG_JOINTS],
        dtype=np.int32,
    )
    if np.any(leg_joint_ids < 0):
        raise ValueError(f"Missing leg joints in {source_xml_path}")
    leg_qpos_ids = model.jnt_qposadr[leg_joint_ids].astype(np.int32)
    data.qpos[leg_qpos_ids] = data.qpos[leg_qpos_ids] + np.array(nominal_leg_ctrl * 4, dtype=np.float64)

    mujoco.mj_forward(model, data)
    foot_geom_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOMS],
        dtype=np.int32,
    )
    if np.any(foot_geom_ids < 0):
        raise ValueError(f"Missing foot geoms in {source_xml_path}")

    foot_bottom_heights = data.geom_xpos[foot_geom_ids, 2] - model.geom_size[foot_geom_ids, 0]
    return float(np.min(foot_bottom_heights))


def _write_flat_scene(
    source_xml_path: Path,
    flat_xml_path: Path,
    reset_base_height: float,
    nominal_leg_ctrl: tuple[float, float, float],
) -> None:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        scene_dir = source_xml_path.parent.as_posix()
        compiler.set("meshdir", scene_dir)
        compiler.set("texturedir", scene_dir)

    asset = root.find("asset")
    if asset is not None:
        for hfield in list(asset.findall("hfield")):
            if hfield.attrib.get("name") == "lunar_hfield":
                asset.remove(hfield)

    terrain_geom = next(
        (geom for geom in root.iter("geom") if geom.attrib.get("name") == "lunar_terrain"),
        None,
    )
    if terrain_geom is None:
        raise ValueError(f"Expected a geom named 'lunar_terrain' in {source_xml_path}")

    terrain_geom.attrib.pop("hfield", None)
    terrain_geom.set("type", "plane")
    terrain_height = _compute_flat_terrain_height(source_xml_path, reset_base_height, nominal_leg_ctrl)
    terrain_geom.set("pos", f"0 0 {terrain_height:.6f}")
    terrain_geom.set("size", "20.0 20.0 0.05")

    flat_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(flat_xml_path, encoding="utf-8", xml_declaration=False)


def _prepare_scene_path(
    xml_path: Path,
    terrain: str,
    run_dir: Path,
    reset_base_height: float,
    nominal_leg_ctrl: tuple[float, float, float],
) -> Path:
    resolved_xml_path = _resolve_scene_path(xml_path)
    if terrain == "heightfield":
        return resolved_xml_path

    flat_xml_path = run_dir / f"{resolved_xml_path.stem}_flat.xml"
    _write_flat_scene(resolved_xml_path, flat_xml_path, reset_base_height, nominal_leg_ctrl)
    return flat_xml_path


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
    gait_period: float,
    gait_contact_sharpness: float,
    swing_height: float,
    control_decimation: int,
    observation_version: str = "v2",
    command_resample_seconds: float = 4.0,
    spawn_xy_range: float = 1.5,
    spawn_yaw_range: float = 0.35,
    reset_joint_noise: float = 0.03,
    reset_velocity_noise: float = 0.05,
    observation_noise: float = 1.0,
    max_action_delay: int = 1,
    render_mode: str | None = None,
    render_camera: str | None = "tracking_side_view",
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
            gait_period=gait_period,
            gait_contact_sharpness=gait_contact_sharpness,
            swing_height=swing_height,
            control_decimation=control_decimation,
            observation_version=observation_version,
            command_resample_seconds=command_resample_seconds,
            spawn_xy_range=spawn_xy_range,
            spawn_yaw_range=spawn_yaw_range,
            reset_joint_noise=reset_joint_noise,
            reset_velocity_noise=reset_velocity_noise,
            observation_noise=observation_noise,
            max_action_delay=max_action_delay,
            render_mode=render_mode,
            render_camera=render_camera,
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


def linear_schedule(initial_value: float, final_value: float):
    """Create a linear learning-rate schedule."""

    def _schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return _schedule


def default_eval_scenarios(command_vx: float, lateral_speed: float, yaw_rate: float) -> tuple[EvalScenario, ...]:
    """Build a compact evaluation matrix covering normal controller use."""

    return (
        EvalScenario("stand", (0.0, 0.0, 0.0), (0.0, 0.0)),
        EvalScenario("slow", (0.25, 0.0, 0.0), (0.8, 0.6)),
        EvalScenario("forward_center", (command_vx, 0.0, 0.0), (0.0, 0.0)),
        EvalScenario("forward_patch", (command_vx, 0.0, 0.0), (-0.8, -0.6), 0.15),
        EvalScenario("lateral_left", (0.35, lateral_speed, 0.0), (0.6, -0.8)),
        EvalScenario("lateral_right", (0.35, -lateral_speed, 0.0), (-0.6, 0.8)),
        EvalScenario("turn_left", (0.35, 0.0, yaw_rate), (0.8, -0.5)),
        EvalScenario("turn_right", (0.35, 0.0, -yaw_rate), (-0.8, 0.5)),
    )


def _resolve_resume_vecnormalize(model_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    match = re.match(r"(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$", model_path.name)
    if match:
        candidate = model_path.parent / (f"{match.group('prefix')}_vecnormalize_{match.group('steps')}_steps.pkl")
        if candidate.exists():
            return candidate
    if model_path.name == "best_model.zip":
        candidate = model_path.parent / "best_vecnormalize.pkl"
        if candidate.exists():
            return candidate
    candidate = model_path.parent / "vecnormalize.pkl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not locate VecNormalize statistics for the resume checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spot locomotion with a Go2-style MuJoCo PPO setup.")
    parser.add_argument("--xml", type=Path, default=Path("spot_scene.xml"))
    parser.add_argument("--terrain", choices=("heightfield", "flat"), default="heightfield")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="spot_go2_style_walk")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vec-env", choices=("auto", "dummy", "subproc"), default="auto")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--ent-coef", type=float, default=1e-4)
    parser.add_argument("--target-kl", type=float, default=0.025)
    parser.add_argument("--log-std-init", type=float, default=-0.7)
    parser.add_argument("--max-policy-std", type=float, default=1.5)
    parser.add_argument("--reset-base-height", type=float, default=1.80)
    parser.add_argument("--target-base-height", type=float, default=1.65)
    parser.add_argument("--control-decimation", type=int, default=10)
    parser.add_argument("--action-scale", type=float, default=0.35)
    parser.add_argument(
        "--nominal-leg-ctrl", type=float, nargs=3, default=(0.0, -0.22, 0.55), metavar=("HX", "HY", "KN")
    )
    parser.add_argument("--actuator-gain-scale", type=float, default=1.8)
    parser.add_argument("--gait-period", type=float, default=0.56)
    parser.add_argument("--gait-contact-sharpness", type=float, default=3.0)
    parser.add_argument("--swing-height", type=float, default=0.10)
    parser.add_argument("--command-vx", type=float, nargs=2, default=(0.0, 0.6), metavar=("MIN", "MAX"))
    parser.add_argument("--command-vy", type=float, nargs=2, default=(-0.2, 0.2), metavar=("MIN", "MAX"))
    parser.add_argument("--command-yaw", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    parser.add_argument("--command-resample-seconds", type=float, default=4.0)
    parser.add_argument("--spawn-xy-range", type=float, default=1.5)
    parser.add_argument("--spawn-yaw-range", type=float, default=0.35)
    parser.add_argument("--reset-joint-noise", type=float, default=0.03)
    parser.add_argument("--reset-velocity-noise", type=float, default=0.05)
    parser.add_argument("--observation-noise", type=float, default=1.0)
    parser.add_argument("--max-action-delay", type=int, default=1)
    parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=2, help="Repetitions of each evaluation scenario.")
    parser.add_argument("--eval-command-vx", type=float, default=0.5)
    parser.add_argument("--eval-lateral-speed", type=float, default=0.1)
    parser.add_argument("--eval-yaw-rate", type=float, default=0.25)
    parser.add_argument(
        "--eval-domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-timesteps", type=int, default=5_000_000)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.05)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Render the first training environment while learning.")
    parser.add_argument(
        "--render-camera",
        type=str,
        default="free",
        help="Fixed camera name for --viewer, or 'free' to use the interactive viewer camera.",
    )
    args = parser.parse_args()
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")

    run_dir = Path("runs") / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_run:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a new --run-name, or pass --overwrite-run explicitly."
        )
    nominal_leg_ctrl = tuple(args.nominal_leg_ctrl)
    xml_path = _prepare_scene_path(args.xml, args.terrain, run_dir, args.reset_base_height, nominal_leg_ctrl)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update({"resolved_xml": str(xml_path), "observation_dim": OBS_DIM})
    (run_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Training scene: {xml_path}")

    command_range = (tuple(args.command_vx), tuple(args.command_vy), tuple(args.command_yaw))
    vec_env_type = args.vec_env
    if args.viewer and vec_env_type == "subproc":
        print("Viewer requested: using DummyVecEnv because MuJoCo viewer should run in the main process.")
        vec_env_type = "dummy"
    render_camera = None if args.render_camera.lower() == "free" else args.render_camera
    env_fns = [
        make_env(
            xml_path,
            args.seed,
            i,
            args.reset_base_height,
            args.target_base_height,
            args.domain_randomization,
            not args.no_curriculum,
            command_range,
            args.action_scale,
            nominal_leg_ctrl,
            args.actuator_gain_scale,
            args.gait_period,
            args.gait_contact_sharpness,
            args.swing_height,
            args.control_decimation,
            observation_version="v2",
            command_resample_seconds=args.command_resample_seconds,
            spawn_xy_range=args.spawn_xy_range,
            spawn_yaw_range=args.spawn_yaw_range,
            reset_joint_noise=args.reset_joint_noise,
            reset_velocity_noise=args.reset_velocity_noise,
            observation_noise=args.observation_noise,
            max_action_delay=args.max_action_delay,
            render_mode="human" if args.viewer and i == 0 else None,
            render_camera=render_camera,
        )
        for i in range(args.num_envs)
    ]
    base_env = make_vec_env(env_fns, vec_env_type)
    if args.resume_from is not None:
        resume_vecnormalize = _resolve_resume_vecnormalize(args.resume_from, args.resume_vecnormalize)
        env = VecNormalize.load(resume_vecnormalize, base_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    rollout_size = args.n_steps * args.num_envs
    batch_size = args.batch_size or min(max(args.num_envs * 256, 512), rollout_size)
    learning_rate = linear_schedule(args.learning_rate, args.final_learning_rate)

    if args.resume_from is not None:
        model = PPO.load(
            args.resume_from,
            env=env,
            device=args.device,
            custom_objects={
                "learning_rate": learning_rate,
                "lr_schedule": learning_rate,
                "ent_coef": args.ent_coef,
                "target_kl": args.target_kl,
                "n_epochs": args.n_epochs,
            },
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            max_grad_norm=1.0,
            policy_kwargs={"net_arch": [256, 256, 128], "log_std_init": args.log_std_init},
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

    callbacks: list[BaseCallback] = [
        RewardComponentCallback(),
        checkpoint_callback,
        TrainingHealthCallback(
            save_dir=run_dir,
            max_policy_std=args.max_policy_std,
            verbose=1,
        ),
    ]
    eval_env = None
    if not args.no_eval:
        eval_command_range = command_range
        eval_env = VecNormalize(
            DummyVecEnv(
                [
                    make_env(
                        xml_path,
                        args.seed,
                        10_000,
                        args.reset_base_height,
                        args.target_base_height,
                        args.eval_domain_randomization,
                        False,
                        eval_command_range,
                        args.action_scale,
                        nominal_leg_ctrl,
                        args.actuator_gain_scale,
                        args.gait_period,
                        args.gait_contact_sharpness,
                        args.swing_height,
                        args.control_decimation,
                        observation_version="v2",
                        command_resample_seconds=0.0,
                        observation_noise=0.25,
                        max_action_delay=args.max_action_delay,
                    )
                ]
            ),
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        eval_env.training = False
        callbacks.append(
            LocomotionEvalCallback(
                eval_env=eval_env,
                save_dir=run_dir / "best_eval",
                eval_freq=args.eval_freq,
                eval_repetitions=args.eval_episodes,
                scenarios=default_eval_scenarios(
                    args.eval_command_vx,
                    args.eval_lateral_speed,
                    args.eval_yaw_rate,
                ),
                early_stop_patience=0 if args.no_early_stop else args.early_stop_patience,
                early_stop_min_timesteps=args.early_stop_min_timesteps,
                early_stop_min_delta=args.early_stop_min_delta,
                restore_best_scores=args.resume_from is not None,
            )
        )

    remaining_timesteps = args.total_timesteps
    reset_num_timesteps = args.resume_from is None
    if args.resume_from is not None:
        remaining_timesteps = max(args.total_timesteps - model.num_timesteps, 0)
        if remaining_timesteps <= 0:
            raise ValueError(
                f"Checkpoint already has {model.num_timesteps} steps, which meets "
                f"--total-timesteps={args.total_timesteps}."
            )
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(run_dir / "ppo_spot_go2_style_final")
    env.save(run_dir / "vecnormalize.pkl")
    env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"Saved model and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
