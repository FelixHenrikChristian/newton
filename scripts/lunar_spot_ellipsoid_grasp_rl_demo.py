# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_LUNAR_SCENE_DIR = (
    Path(__file__).resolve().parents[1] / "newton" / "examples" / "assets" / "lunar_mujoco_spot_arm_mining_scene"
)
DEFAULT_SCENE = DEFAULT_LUNAR_SCENE_DIR / "lunar_scene_spot_arm_mining.xml"

MUJOCO_NJMAX = 20000
MUJOCO_NCONMAX = 10000

LEG_JOINT_NAMES = (
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
)
ARM_JOINT_NAMES = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x")

# Values from the spot_mining_home XML keyframe. MuJoCo authors free-joint
# quaternions as WXYZ; Newton stores imported free-joint quaternions as XYZW.
SPOT_ROOT_HOME = (6.7, 7.1, 0.74776551, 0.0, 0.0, 0.35272872, 0.93572563)
SPOT_LEG_HOME = (0.0, 1.04, -1.8) * 4

GRIPPER_OPEN = -1.4

# Fixed world-space pose copied from the existing ground ellipsoid grasp demo.
# The new RL task intentionally does not compute this from an arm pose.
ELLIPSOID_INITIAL_POS = (7.180263042449951, 7.522050380706787, 0.26244837045669556)
ELLIPSOID_INITIAL_YAW_DEGREES = 41.3
ELLIPSOID_RX = 0.03
ELLIPSOID_RY = 0.024
ELLIPSOID_RZ = 0.016
ELLIPSOID_MU = 25.0

# Local point used only as the reward/observation proxy for the pinch center.
PINCH_CENTER_IN_WR1 = wp.vec3(0.22, 0.0, -0.008)
ARM_RESET_Q = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN), dtype=np.float32)
ARM_ACTION_SCALE = np.asarray((0.07, 0.09, 0.09, 0.08, 0.07, 0.08, 0.12), dtype=np.float32)


class LunarSpotEllipsoidGraspTask:
    """Fixed-position Spot arm grasp task with normalized continuous actions."""

    observation_size = 32
    action_size = len(ARM_JOINT_NAMES)

    def __init__(self, args: argparse.Namespace, viewer=None):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        if self.sim_substeps <= 0:
            raise ValueError("--substeps must be greater than zero.")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.action_repeat = args.action_repeat
        if self.action_repeat <= 0:
            raise ValueError("--action-repeat must be greater than zero.")

        self.max_episode_steps = args.max_episode_steps
        self.success_lift = args.success_lift
        self.max_object_drift = args.max_object_drift
        self.ellipsoid_initial_pos = np.asarray(args.ellipsoid_pos, dtype=np.float32)
        self.ellipsoid_initial_rot = wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0),
            float(np.deg2rad(args.ellipsoid_yaw)),
        )
        self.ellipsoid_initial_q = np.asarray(
            (
                self.ellipsoid_initial_pos[0],
                self.ellipsoid_initial_pos[1],
                self.ellipsoid_initial_pos[2],
                self.ellipsoid_initial_rot[0],
                self.ellipsoid_initial_rot[1],
                self.ellipsoid_initial_rot[2],
                self.ellipsoid_initial_rot[3],
            ),
            dtype=np.float32,
        )

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)

        ellipsoid_cfg = newton.ModelBuilder.ShapeConfig(
            density=150.0,
            mu=ELLIPSOID_MU,
            mu_torsional=0.2,
            mu_rolling=0.1,
        )
        ellipsoid_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.ellipsoid_initial_pos), self.ellipsoid_initial_rot),
            label="spot_rl_ellipsoid",
        )
        builder.add_shape_ellipsoid(
            ellipsoid_body,
            rx=ELLIPSOID_RX,
            ry=ELLIPSOID_RY,
            rz=ELLIPSOID_RZ,
            cfg=ellipsoid_cfg,
            color=wp.vec3(0.46, 0.46, 0.44),
            label="spot_rl_ellipsoid",
        )

        self.model = builder.finalize()
        if self.model.joint_count <= 0:
            raise ValueError("SolverMuJoCo requires at least one joint in the imported MJCF model.")

        warnings.filterwarnings("ignore", message=r"Geom .* authored margin=.*")
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.root_q_slice, self.root_qd_slice = self._find_joint_slices(("freejoint",), q_width=7, qd_width=6)
        self.leg_q_slice, self.leg_dof_slice = self._find_joint_slices(LEG_JOINT_NAMES)
        self.arm_q_slice, self.arm_dof_slice = self._find_joint_slices(ARM_JOINT_NAMES)
        self.ellipsoid_q_slice, self.ellipsoid_qd_slice = self._find_joint_slices(
            ("spot_rl_ellipsoid_free_joint",),
            q_width=7,
            qd_width=6,
        )

        self.ellipsoid_body_index = self._find_body_index("spot_rl_ellipsoid")
        self.wr1_body_index = self._find_body_index_in_labels(self.model.body_label, "/arm_link_wr1")

        self.arm_lower = self.model.joint_limit_lower.numpy()[self.arm_dof_slice].astype(np.float32)
        self.arm_upper = self.model.joint_limit_upper.numpy()[self.arm_dof_slice].astype(np.float32)
        self.arm_target = np.clip(ARM_RESET_Q.copy(), self.arm_lower, self.arm_upper)

        self.sim_time = 0.0
        self.episode_step = 0
        self.last_reward = 0.0
        self.last_info: dict[str, float | bool] = {}

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(8.0, 6.4, 1.5), pitch=-22.0, yaw=132.0)

        self.reset()

    @staticmethod
    def _find_joint_slices_in_model(
        model,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        joint_indices = []

        for name in names:
            matches = [i for i, label in enumerate(model.joint_label) if label == name or label.endswith(f"/{name}")]
            if len(matches) != 1:
                raise ValueError(f"Expected one imported joint named '{name}', found {len(matches)}.")
            joint_indices.append(matches[0])

        q_indices = [int(q_starts[index]) for index in joint_indices]
        qd_indices = [int(qd_starts[index]) for index in joint_indices]
        if q_indices != list(range(q_indices[0], q_indices[0] + len(names))) and q_width is None:
            raise ValueError(f"Expected contiguous joint coordinates for: {', '.join(names)}")
        if qd_indices != list(range(qd_indices[0], qd_indices[0] + len(names))) and qd_width is None:
            raise ValueError(f"Expected contiguous joint DoFs for: {', '.join(names)}")

        return (
            slice(q_indices[0], q_indices[0] + (q_width or len(names))),
            slice(qd_indices[0], qd_indices[0] + (qd_width or len(names))),
        )

    def _find_joint_slices(
        self,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        return self._find_joint_slices_in_model(self.model, names, q_width=q_width, qd_width=qd_width)

    def _find_body_index(self, label: str) -> int:
        matches = [i for i, body_label in enumerate(self.model.body_label) if body_label == label]
        if len(matches) != 1:
            raise ValueError(f"Expected one body labeled '{label}', found {len(matches)}.")
        return matches[0]

    @staticmethod
    def _find_body_index_in_labels(labels, suffix: str) -> int:
        matches = [i for i, label in enumerate(labels) if label.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one body ending in '{suffix}', found {len(matches)}.")
        return matches[0]

    def _write_reset_state(self, state) -> None:
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_q[self.leg_q_slice].assign(SPOT_LEG_HOME)
        state.joint_q[self.arm_q_slice].assign(self.arm_target)
        state.joint_q[self.ellipsoid_q_slice].assign(self.ellipsoid_initial_q)
        state.joint_qd.zero_()
        state.body_qd.zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _lock_spot_root(self, state) -> None:
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_qd[self.root_qd_slice].zero_()

    def reset(self, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.sim_time = 0.0
        self.episode_step = 0
        self.last_reward = 0.0
        self.last_info = {}
        self.arm_target = np.clip(ARM_RESET_Q.copy(), self.arm_lower, self.arm_upper)
        self._write_reset_state(self.state_0)
        self._write_reset_state(self.state_1)
        self.control.joint_target_pos[self.leg_dof_slice].assign(SPOT_LEG_HOME)
        self.control.joint_target_pos[self.arm_dof_slice].assign(self.arm_target)
        return self._get_obs(), self._get_info()

    def _body_transform(self, body_index: int) -> wp.transform:
        return wp.transform(*self.state_0.body_q.numpy()[body_index])

    def _pinch_center_world(self, body_q: np.ndarray | None = None) -> np.ndarray:
        if body_q is None:
            wr1_xform = self._body_transform(self.wr1_body_index)
        else:
            wr1_xform = wp.transform(*body_q[self.wr1_body_index])
        return np.asarray(wp.transform_point(wr1_xform, PINCH_CENTER_IN_WR1), dtype=np.float32)

    def _ellipsoid_local_to_wr1(self, body_q: np.ndarray | None = None) -> np.ndarray:
        if body_q is None:
            body_q = self.state_0.body_q.numpy()
        wr1_xform = wp.transform(*body_q[self.wr1_body_index])
        ellipsoid_pos = wp.vec3(*body_q[self.ellipsoid_body_index][:3])
        return np.asarray(wp.transform_point(wp.transform_inverse(wr1_xform), ellipsoid_pos), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        arm_q = joint_q[self.arm_q_slice].astype(np.float32)
        arm_qd = joint_qd[self.arm_dof_slice].astype(np.float32)
        ellipsoid_pos = body_q[self.ellipsoid_body_index][:3].astype(np.float32)
        ellipsoid_vel = body_qd[self.ellipsoid_body_index][:3].astype(np.float32)
        pinch_pos = self._pinch_center_world(body_q)
        ellipsoid_local = self._ellipsoid_local_to_wr1(body_q)
        arm_range = np.maximum(self.arm_upper - self.arm_lower, 1.0e-5)

        obs = np.concatenate(
            (
                2.0 * (arm_q - self.arm_lower) / arm_range - 1.0,
                np.clip(arm_qd * 0.1, -5.0, 5.0),
                2.0 * (self.arm_target - self.arm_lower) / arm_range - 1.0,
                ellipsoid_pos - pinch_pos,
                ellipsoid_local,
                np.clip(ellipsoid_vel * 0.2, -5.0, 5.0),
                np.asarray(
                    (
                        ellipsoid_pos[2] - self.ellipsoid_initial_pos[2],
                        self.episode_step / max(1, self.max_episode_steps),
                    ),
                    dtype=np.float32,
                ),
            )
        )
        return obs.astype(np.float32)

    def _get_info(self) -> dict[str, float | bool]:
        body_q = self.state_0.body_q.numpy()
        ellipsoid_pos = body_q[self.ellipsoid_body_index][:3].astype(np.float32)
        pinch_pos = self._pinch_center_world(body_q)
        local = self._ellipsoid_local_to_wr1(body_q)
        lift = float(ellipsoid_pos[2] - self.ellipsoid_initial_pos[2])
        distance = float(np.linalg.norm(ellipsoid_pos - pinch_pos))
        in_gripper = bool(0.10 <= local[0] <= 0.32 and abs(local[1]) <= 0.10 and -0.14 <= local[2] <= 0.14)
        success = bool(lift >= self.success_lift and in_gripper)
        drift = float(np.linalg.norm(ellipsoid_pos[:2] - self.ellipsoid_initial_pos[:2]))
        return {
            "distance": distance,
            "lift": lift,
            "drift": drift,
            "in_gripper": in_gripper,
            "is_success": success,
        }

    def _compute_reward(self, action: np.ndarray, previous_info: dict[str, float | bool]) -> tuple[float, bool, bool, dict]:
        info = self._get_info()
        distance = float(info["distance"])
        lift = float(info["lift"])
        drift = float(info["drift"])
        previous_distance = float(previous_info["distance"])

        reward = 4.0 * (previous_distance - distance)
        reward += 1.5 * np.exp(-12.0 * distance)
        reward += 12.0 * max(0.0, lift)
        reward -= 0.015 * float(np.dot(action, action))
        reward -= 0.5 * drift

        if info["in_gripper"]:
            reward += 2.0 + 6.0 * max(0.0, lift)
        terminated = bool(info["is_success"])
        if terminated:
            reward += 50.0

        lost_object = drift > self.max_object_drift or lift < -0.08
        if lost_object:
            reward -= 10.0
            terminated = True

        truncated = self.episode_step >= self.max_episode_steps
        return float(reward), terminated, truncated, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_size,):
            raise ValueError(f"Expected action shape {(self.action_size,)}, got {action.shape}.")
        action = np.clip(action, -1.0, 1.0)
        previous_info = self._get_info()

        self.arm_target = np.clip(self.arm_target + action * ARM_ACTION_SCALE, self.arm_lower, self.arm_upper)
        self.control.joint_target_pos[self.leg_dof_slice].assign(SPOT_LEG_HOME)
        self.control.joint_target_pos[self.arm_dof_slice].assign(self.arm_target)

        for _ in range(self.action_repeat):
            for _ in range(self.sim_substeps):
                self._lock_spot_root(self.state_0)
                self.state_0.clear_forces()
                if self.viewer is not None:
                    self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.frame_dt

        self.episode_step += 1
        reward, terminated, truncated, info = self._compute_reward(action, previous_info)
        self.last_reward = reward
        self.last_info = info
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


class LunarSpotEllipsoidGraspRlDemo:
    """Newton viewer wrapper for the fixed-position RL grasp task."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.task = LunarSpotEllipsoidGraspTask(args, viewer=viewer)
        self.obs, _ = self.task.reset()
        self.policy = None
        if args.policy:
            self.policy = _load_sb3_policy(args.policy, args.torch_device)
        self.state_0 = self.task.state_0

    def step(self) -> None:
        if self.policy is None:
            action = np.zeros(self.task.action_size, dtype=np.float32)
        else:
            action, _ = self.policy.predict(self.obs, deterministic=self.args.deterministic)
        self.obs, reward, terminated, truncated, info = self.task.step(action)
        self.state_0 = self.task.state_0
        if terminated or truncated:
            print(
                "[INFO] Episode finished: "
                f"reward={reward:.3f}, lift={info['lift']:.3f}, "
                f"distance={info['distance']:.3f}, success={info['is_success']}"
            )
            self.obs, _ = self.task.reset()
            self.state_0 = self.task.state_0

    def render(self) -> None:
        self.task.render()

    def test_final(self) -> None:
        info = self.task._get_info()
        print(
            "[INFO] Final fixed-position RL grasp state: "
            f"ellipsoid_pos={np.round(self.task.state_0.body_q.numpy()[self.task.ellipsoid_body_index][:3], 4)}, "
            f"lift={info['lift']:.4f}, distance={info['distance']:.4f}, success={info['is_success']}"
        )


def _load_sb3_policy(policy_path: str, torch_device: str):
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Loading a policy requires stable-baselines3. Run with "
            "`uv run --with gymnasium --with stable-baselines3 python "
            "scripts/lunar_spot_ellipsoid_grasp_rl_demo.py --policy PATH`."
        ) from exc
    return PPO.load(policy_path, device=torch_device)


def _make_gym_env(args: argparse.Namespace):
    try:
        import gymnasium as gym  # noqa: PLC0415
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Training requires gymnasium and stable-baselines3. Run with "
            "`uv run --with gymnasium --with stable-baselines3 python "
            "scripts/lunar_spot_ellipsoid_grasp_rl_demo.py --train --viewer null`."
        ) from exc

    class LunarSpotEllipsoidGraspGymEnv(gym.Env):
        metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.task = LunarSpotEllipsoidGraspTask(args, viewer=None)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task.observation_size,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.task.action_size,),
                dtype=np.float32,
            )

        def reset(self, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            return self.task.reset(seed=seed, options=options)

        def step(self, action):
            return self.task.step(action)

    return LunarSpotEllipsoidGraspGymEnv()


def train(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415
        from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: PLC0415
        from stable_baselines3.common.monitor import Monitor  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Training requires stable-baselines3. Run with "
            "`uv run --with gymnasium --with stable-baselines3 python "
            "scripts/lunar_spot_ellipsoid_grasp_rl_demo.py --train --viewer null`."
        ) from exc

    env = Monitor(_make_gym_env(args))
    policy_out = Path(args.policy_out).resolve()
    if args.resume_policy:
        model = PPO.load(Path(args.resume_policy).resolve(), env=env, device=args.torch_device)
        reset_num_timesteps = False
        print(f"[INFO] Resuming training from {Path(args.resume_policy).resolve()}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            gamma=args.gamma,
            verbose=1,
            device=args.torch_device,
        )
        reset_num_timesteps = True

    callbacks = []
    if args.checkpoint_freq > 0:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="lunar_spot_ellipsoid_grasp_ppo",
            )
        )
        print(f"[INFO] Saving checkpoints every {args.checkpoint_freq} steps to {checkpoint_dir}")

    try:
        model.learn(
            total_timesteps=args.train_timesteps,
            callback=callbacks or None,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("[INFO] Training interrupted; saving current policy before exit.")

    policy_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(policy_out)
    print(f"[INFO] Saved policy to {policy_out}")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Train or preview a fixed-position lunar Spot arm ellipsoid grasp task."
    parser.set_defaults(num_frames=600, viewer="gl")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per rendered frame.")
    parser.add_argument("--action-repeat", type=int, default=4, help="Rendered frames advanced per RL action.")
    parser.add_argument("--max-episode-steps", type=int, default=240, help="Maximum RL steps per episode.")
    parser.add_argument(
        "--ellipsoid-pos",
        type=float,
        nargs=3,
        default=ELLIPSOID_INITIAL_POS,
        metavar=("X", "Y", "Z"),
        help="Fixed world-space ellipsoid center. It is not randomized during reset.",
    )
    parser.add_argument(
        "--ellipsoid-yaw",
        type=float,
        default=ELLIPSOID_INITIAL_YAW_DEGREES,
        help="Fixed ellipsoid yaw angle in degrees.",
    )
    parser.add_argument("--success-lift", type=float, default=0.08, help="Lift height required for success.")
    parser.add_argument("--max-object-drift", type=float, default=0.7, help="Terminate if the ellipsoid drifts this far.")
    parser.add_argument("--policy", type=str, default="", help="Optional Stable-Baselines3 PPO policy zip to preview.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train", action="store_true", help="Train a PPO policy instead of opening the viewer demo.")
    parser.add_argument("--train-timesteps", type=int, default=20000, help="PPO training timesteps.")
    parser.add_argument(
        "--policy-out",
        type=str,
        default="outputs/lunar_spot_ellipsoid_grasp_ppo.zip",
        help="Path for the trained PPO policy.",
    )
    parser.add_argument(
        "--resume-policy",
        type=str,
        default="",
        help="Optional PPO policy zip or checkpoint to continue training from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/lunar_spot_ellipsoid_grasp_checkpoints",
        help="Directory for periodic PPO training checkpoints.",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="Checkpoint interval in training steps.")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--torch-device", type=str, default="auto", help="Stable-Baselines3 torch device.")
    return parser


def _apply_basic_warp_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.quiet:
        wp.config.quiet = True
    for entry in args.warp_config:
        if "=" not in entry:
            parser.error(f"invalid --warp-config format '{entry}': expected KEY=VALUE")
        key, value_str = entry.split("=", 1)
        if not hasattr(wp.config, key):
            parser.error(f"unknown warp.config option '{key}'")
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        setattr(wp.config, key, value)
    if args.device:
        wp.set_device(args.device)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.train:
        _apply_basic_warp_args(parser, args)
        train(args)
        return

    viewer, args = newton.examples.init(parser)
    demo = LunarSpotEllipsoidGraspRlDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
