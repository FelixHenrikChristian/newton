# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

DEFAULT_SCENE = (
    Path(__file__).resolve().parents[1]
    / "newton"
    / "examples"
    / "assets"
    / "aliengo_z1_mujoco_scene"
    / "aliengoz1_scene_lunar.xml"
)

LEG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
Z1_JOINT_NAMES = ("z1_joint1", "z1_joint2", "z1_joint3", "z1_joint4", "z1_joint5", "z1_joint6")

ROCK_BODY_NAME = "aliengo_ik_rock"
ROCK_GEOM_NAME = "aliengo_ik_rock_geom"
ALIENGO_ROOT_JOINT_NAME = "floating_base"
Z1_BASE_BODY_NAME = "z1_link00"
Z1_EE_BODY_NAME = "z1_gripper_stator"
Z1_GRIPPER_JOINT_NAME = "z1_gripper_joint"

TCP_OFFSET_IN_GRIPPER = wp.vec3(0.145, 0.0, 0.0)
DOWN_AXIS_LENGTH = 0.08


def _quat_xyzw_to_mjcf_wxyz(q: np.ndarray) -> tuple[float, float, float, float]:
    return float(q[3]), float(q[0]), float(q[1]), float(q[2])


def _format_vec(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _format_float(value: float) -> str:
    return f"{float(value):.9g}"


def _smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _parse_joint_refs(mjcf: Path, names: tuple[str, ...]) -> tuple[float, ...]:
    root = ET.parse(mjcf).getroot()
    refs = []
    for name in names:
        matches = [elem for elem in root.iter("joint") if elem.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one MJCF joint named '{name}', found {len(matches)}.")
        refs.append(float(matches[0].get("ref", "0.0")))
    return tuple(refs)


def _build_z1_ik_mjcf(base_tf: np.ndarray, limits: tuple[tuple[float, float], ...]) -> str:
    pos = base_tf[:3]
    quat = _quat_xyzw_to_mjcf_wxyz(base_tf[3:7])
    ranges = [_format_vec(limit) for limit in limits]
    return f"""<mujoco model="aliengo_z1_ik_chain">
  <compiler angle="radian" autolimits="true" />
  <worldbody>
    <body name="z1_link00" pos="{_format_vec(pos)}" quat="{_format_vec(quat)}">
      <body name="z1_link01" pos="0 0 0.0585">
        <joint name="z1_joint1" type="hinge" axis="0 0 1" range="{ranges[0]}" />
        <body name="z1_link02" pos="0 0 0.045">
          <joint name="z1_joint2" type="hinge" axis="0 1 0" range="{ranges[1]}" />
          <body name="z1_link03" pos="-0.35 0 0">
            <joint name="z1_joint3" type="hinge" axis="0 1 0" range="{ranges[2]}" />
            <body name="z1_link04" pos="0.218 0 0.057">
              <joint name="z1_joint4" type="hinge" axis="0 1 0" range="{ranges[3]}" />
              <body name="z1_link05" pos="0.07 0 0">
                <joint name="z1_joint5" type="hinge" axis="0 0 1" range="{ranges[4]}" />
                <body name="z1_link06" pos="0.0492 0 0">
                  <joint name="z1_joint6" type="hinge" axis="1 0 0" range="{ranges[5]}" />
                  <body name="z1_gripper_stator" pos="0.051 0 0" />
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


class LunarAliengoZ1IkApproachDemo:
    """Track and descend toward the lunar ellipsoid with the Z1 arm."""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.args = args
        self.mjcf = Path(args.mjcf).resolve()

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(self.mjcf), up_axis="Z", enable_self_collisions=True)
        self.model = builder.finalize()
        if self.model.joint_count <= 0:
            raise ValueError("Imported MJCF model has no joints.")

        self.state = self.model.state()
        self.root_q_slice, self.root_qd_slice = self._find_joint_slices(
            (ALIENGO_ROOT_JOINT_NAME,), q_width=7, qd_width=6
        )
        self.leg_q_slice, self.leg_dof_slice = self._find_joint_slices(LEG_JOINT_NAMES)
        self.z1_q_slice, self.z1_dof_slice = self._find_joint_slices(Z1_JOINT_NAMES)
        self.gripper_q_slice, self.gripper_dof_slice = self._find_joint_slices((Z1_GRIPPER_JOINT_NAME,))
        self.rock_body_index = self._find_body_index(ROCK_BODY_NAME)
        self.rock_shape_index = self._find_shape_index(ROCK_GEOM_NAME)
        self.z1_base_body_index = self._find_body_index(Z1_BASE_BODY_NAME)
        self.z1_ee_body_index = self._find_body_index(Z1_EE_BODY_NAME)

        self.root_q = self.model.joint_q.numpy()[self.root_q_slice].astype(np.float32)
        self.leg_q = np.asarray(_parse_joint_refs(self.mjcf, LEG_JOINT_NAMES), dtype=np.float32)
        self.z1_q = np.zeros(len(Z1_JOINT_NAMES), dtype=np.float32)
        self.gripper_q = np.zeros(1, dtype=np.float32)
        self.full_joint_q = self.model.joint_q.numpy().astype(np.float32)

        self._set_scene_state(self.z1_q)
        body_q = self.state.body_q.numpy()
        z1_base_tf = body_q[self.z1_base_body_index].astype(np.float32)

        z1_limits = self._z1_limits()
        ik_builder = newton.ModelBuilder()
        ik_builder.add_mjcf(_build_z1_ik_mjcf(z1_base_tf, z1_limits), up_axis="Z", enable_self_collisions=False)
        self.ik_model = ik_builder.finalize()
        self.ik_state = self.ik_model.state()
        self.ik_ee_body_index = self._find_body_index_in_labels(self.ik_model.body_label, Z1_EE_BODY_NAME)
        self.ik_joint_q = wp.array(self.z1_q.reshape(1, -1), dtype=wp.float32)

        self.rock_pos = body_q[self.rock_body_index][:3].astype(np.float32)
        self.rock_radii = self.model.shape_scale.numpy()[self.rock_shape_index].astype(np.float32)
        self.rock_top_z = float(self.rock_pos[2] + self.rock_radii[2])
        self.final_target = np.array(
            [
                self.rock_pos[0],
                self.rock_pos[1],
                self.rock_top_z + float(args.final_clearance),
            ],
            dtype=np.float32,
        )
        self.pregrasp_target = self.final_target + np.array([0.0, 0.0, float(args.pregrasp_height)], dtype=np.float32)
        self.start_target = self._tcp_position(self.state.body_q.numpy(), self.z1_ee_body_index)
        self.current_target = self.start_target.copy()
        self.current_tcp = self.start_target.copy()
        self.final_error = math.inf
        self.max_joint_step_observed = 0.0

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_body_index,
            link_offset=TCP_OFFSET_IN_GRIPPER,
            target_positions=wp.array([wp.vec3(*self.current_target)], dtype=wp.vec3),
            weight=1.0,
        )
        self.down_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_body_index,
            link_offset=TCP_OFFSET_IN_GRIPPER + wp.vec3(DOWN_AXIS_LENGTH, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self._down_axis_target(self.current_target))], dtype=wp.vec3),
            weight=float(args.down_axis_weight),
        )
        self.base_down_axis_weight = float(args.down_axis_weight)
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=float(args.limit_weight),
        )
        self.z1_lower = self.model.joint_limit_lower.numpy()[self.z1_dof_slice].astype(np.float32)
        self.z1_upper = self.model.joint_limit_upper.numpy()[self.z1_dof_slice].astype(np.float32)
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.down_obj, self.limit_obj],
            lambda_initial=float(args.lambda_initial),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            sampler=ik.IKSampler.ROBERTS if int(args.ik_seeds) > 1 else ik.IKSampler.NONE,
            n_seeds=int(args.ik_seeds),
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(7.3, 6.5, 1.6), pitch=-18.0, yaw=128.0)
        self._print_scene_info()

    def _find_joint_slices(
        self,
        names: tuple[str, ...],
        *,
        q_width: int | None = None,
        qd_width: int | None = None,
    ) -> tuple[slice, slice]:
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        joint_indices = []
        for name in names:
            matches = [
                i for i, label in enumerate(self.model.joint_label) if label == name or label.endswith(f"/{name}")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one imported joint named '{name}', found {len(matches)}.")
            joint_indices.append(matches[0])

        q_indices = [int(q_starts[index]) for index in joint_indices]
        qd_indices = [int(qd_starts[index]) for index in joint_indices]
        if q_width is None and q_indices != list(range(q_indices[0], q_indices[0] + len(names))):
            raise ValueError(f"Expected contiguous joint coordinates for: {', '.join(names)}")
        if qd_width is None and qd_indices != list(range(qd_indices[0], qd_indices[0] + len(names))):
            raise ValueError(f"Expected contiguous joint DoFs for: {', '.join(names)}")
        return (
            slice(q_indices[0], q_indices[0] + (q_width or len(names))),
            slice(qd_indices[0], qd_indices[0] + (qd_width or len(names))),
        )

    def _find_body_index(self, name: str) -> int:
        return self._find_body_index_in_labels(self.model.body_label, name)

    @staticmethod
    def _find_body_index_in_labels(labels, name: str) -> int:
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named '{name}', found {len(matches)}.")
        return matches[0]

    def _find_shape_index(self, name: str) -> int:
        matches = [i for i, label in enumerate(self.model.shape_label) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one shape named '{name}', found {len(matches)}.")
        return matches[0]

    def _z1_limits(self) -> tuple[tuple[float, float], ...]:
        lower = self.model.joint_limit_lower.numpy()[self.z1_dof_slice]
        upper = self.model.joint_limit_upper.numpy()[self.z1_dof_slice]
        return tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True))

    def _set_scene_state(self, z1_q: np.ndarray) -> None:
        self.full_joint_q[self.root_q_slice] = self.root_q
        self.full_joint_q[self.leg_q_slice] = self.leg_q
        self.full_joint_q[self.z1_q_slice] = z1_q
        self.full_joint_q[self.gripper_q_slice] = self.gripper_q
        self.state.joint_q.assign(self.full_joint_q)
        self.state.joint_qd.zero_()
        self.state.body_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    @staticmethod
    def _tcp_position(body_q: np.ndarray, body_index: int) -> np.ndarray:
        tcp = wp.transform_point(wp.transform(*body_q[body_index]), TCP_OFFSET_IN_GRIPPER)
        return np.asarray(tcp, dtype=np.float32)

    @staticmethod
    def _down_axis_target(tcp_target: np.ndarray) -> np.ndarray:
        return tcp_target + np.array([0.0, 0.0, -DOWN_AXIS_LENGTH], dtype=np.float32)

    def _target_at_time(self) -> np.ndarray:
        unfold_duration = float(self.args.unfold_duration)
        descent_duration = float(self.args.descent_duration)

        if self.sim_time < unfold_duration:
            alpha = _smoothstep(self.sim_time / max(unfold_duration, 1.0e-6))
            return (1.0 - alpha) * self.start_target + alpha * self.pregrasp_target

        descent_t = self.sim_time - unfold_duration
        if descent_t < descent_duration:
            alpha = _smoothstep(descent_t / max(descent_duration, 1.0e-6))
            return (1.0 - alpha) * self.pregrasp_target + alpha * self.final_target

        return self.final_target.copy()

    def _solve_ik_target(self, target: np.ndarray) -> np.ndarray:
        self.pos_obj.set_target_position(0, wp.vec3(*target))
        self.down_obj.set_target_position(0, wp.vec3(*self._down_axis_target(target)))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=int(self.args.ik_iters),
            step_size=float(self.args.ik_step_size),
        )
        z1_q = self.ik_joint_q.numpy()[0].astype(np.float32)
        if self.args.lock_wrist_roll:
            z1_q[5] = 0.0
        z1_q = self._limit_joint_step(z1_q)
        self.ik_joint_q.assign(z1_q.reshape(1, -1))
        return z1_q

    def _limit_joint_step(self, candidate_q: np.ndarray) -> np.ndarray:
        candidate_q = np.clip(candidate_q, self.z1_lower, self.z1_upper)
        max_step = float(self.args.max_joint_step)
        if max_step <= 0.0:
            return candidate_q

        delta = candidate_q - self.z1_q
        step = float(np.max(np.abs(delta)))
        if step <= max_step:
            self.max_joint_step_observed = max(self.max_joint_step_observed, step)
            return candidate_q

        limited_q = self.z1_q + delta * (max_step / step)
        limited_q = np.clip(limited_q, self.z1_lower, self.z1_upper).astype(np.float32)
        actual_step = float(np.max(np.abs(limited_q - self.z1_q)))
        self.max_joint_step_observed = max(self.max_joint_step_observed, actual_step)
        return limited_q

    def _update_down_axis_weight(self) -> None:
        ramp_duration = float(self.args.down_axis_ramp_duration)
        if ramp_duration <= 0.0:
            self.down_obj.weight = self.base_down_axis_weight
            return

        self.down_obj.weight = self.base_down_axis_weight * _smoothstep(self.sim_time / ramp_duration)

    def _update_gripper(self) -> None:
        open_start = float(self.args.gripper_open_start)
        open_duration = max(float(self.args.gripper_open_duration), 1.0e-6)
        alpha = _smoothstep((self.sim_time - open_start) / open_duration)
        self.gripper_q[0] = float(self.args.gripper_open_angle) * alpha

    def _print_scene_info(self) -> None:
        print(
            "[INFO] Lunar Aliengo Z1 IK approach: "
            f"rock center={np.round(self.rock_pos, 4)}, "
            f"radii={np.round(self.rock_radii, 4)}, "
            f"pregrasp={np.round(self.pregrasp_target, 4)}, "
            f"final={np.round(self.final_target, 4)}"
        )
        print(
            "[INFO] Fixed Aliengo pose: "
            f"root_q={np.round(self.root_q, 4)}, "
            f"leg_q={np.round(self.leg_q, 4)}, "
            f"z1_limits={tuple((round(lo, 3), round(hi, 3)) for lo, hi in self._z1_limits())}"
        )

    def step(self) -> None:
        self._update_down_axis_weight()
        self._update_gripper()
        self.current_target = self._target_at_time()
        self.z1_q = self._solve_ik_target(self.current_target)
        self._set_scene_state(self.z1_q)
        self.current_tcp = self._tcp_position(self.state.body_q.numpy(), self.z1_ee_body_index)
        self.final_error = float(np.linalg.norm(self.current_tcp - self.current_target))
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.args.show_ik_targets and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("target_z1_tcp", wp.transform(wp.vec3(*self.current_target), wp.quat_identity()))
            self.viewer.log_gizmo("target_rock_top", wp.transform(wp.vec3(*self.final_target), wp.quat_identity()))
        self.viewer.end_frame()

    def test_final(self) -> None:
        root_q = self.state.joint_q.numpy()[self.root_q_slice]
        leg_q = self.state.joint_q.numpy()[self.leg_q_slice]
        if not np.allclose(root_q, self.root_q, atol=1.0e-6):
            raise ValueError(f"Aliengo root moved: expected {self.root_q}, got {root_q}")
        if not np.allclose(leg_q, self.leg_q, atol=1.0e-6):
            raise ValueError(f"Aliengo leg joints moved: expected {self.leg_q}, got {leg_q}")

        tcp_to_rock_xy = float(np.linalg.norm(self.current_tcp[:2] - self.rock_pos[:2]))
        tcp_clearance = float(self.current_tcp[2] - self.rock_top_z)
        print(
            "[INFO] Final Aliengo Z1 IK approach: "
            f"target={np.round(self.current_target, 4)}, "
            f"tcp={np.round(self.current_tcp, 4)}, "
            f"target_error={self.final_error:.4f}, "
            f"xy_error={tcp_to_rock_xy:.4f}, "
            f"clearance={tcp_clearance:.4f}, "
            f"max_joint_step={self.max_joint_step_observed:.4f}, "
            f"gripper_q={self.gripper_q[0]:.4f}, "
            f"z1_q={np.round(self.z1_q, 4)}"
        )
        if self.final_error > float(self.args.max_final_error):
            raise ValueError(f"Z1 TCP did not converge to the ellipsoid approach target: error={self.final_error:.4f}m")
        if tcp_to_rock_xy > float(self.args.max_xy_error):
            raise ValueError(f"Z1 TCP did not locate the ellipsoid in x/y: error={tcp_to_rock_xy:.4f}m")
        if tcp_clearance < 0.0:
            raise ValueError(f"Z1 TCP penetrated below the ellipsoid top: clearance={tcp_clearance:.4f}m")
        if 0.0 < float(self.args.max_joint_step) + 1.0e-6 < self.max_joint_step_observed:
            raise ValueError(
                f"Z1 joint step exceeded limit: {self.max_joint_step_observed:.4f}rad > "
                f"{float(self.args.max_joint_step):.4f}rad"
            )
        if self.gripper_q[0] < float(self.args.gripper_open_angle) - 1.0e-4:
            raise ValueError(
                f"Z1 gripper did not finish opening: {self.gripper_q[0]:.4f}rad < "
                f"{float(self.args.gripper_open_angle):.4f}rad"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Use Newton IK to unfold Aliengo's Z1 arm and descend from above toward the lunar ellipsoid."
    parser.set_defaults(num_frames=360, viewer="gl")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the Aliengo+Z1 lunar MJCF file.")
    parser.add_argument(
        "--unfold-duration", type=float, default=2.0, help="Time to move from folded pose to pregrasp [s]."
    )
    parser.add_argument(
        "--descent-duration", type=float, default=2.0, help="Time to descend from pregrasp to target [s]."
    )
    parser.add_argument(
        "--pregrasp-height", type=float, default=0.30, help="Height above final target before descent [m]."
    )
    parser.add_argument("--final-clearance", type=float, default=0.04, help="TCP clearance above ellipsoid top [m].")
    parser.add_argument("--ik-iters", type=int, default=32, help="IK iterations per frame.")
    parser.add_argument("--ik-seeds", type=int, default=1, help="Candidate IK seeds per frame.")
    parser.add_argument("--ik-step-size", type=float, default=0.8, help="IK LM step size.")
    parser.add_argument("--lambda-initial", type=float, default=0.1, help="Initial IK LM damping.")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.05,
        help="Maximum Z1 joint-coordinate change applied per frame [rad]. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--down-axis-ramp-duration",
        type=float,
        default=2.0,
        help="Time to ramp in the downward gripper-axis objective [s]. Set <= 0 to apply immediately.",
    )
    parser.add_argument(
        "--lock-wrist-roll",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep z1_joint6 fixed because TCP position and down-axis objectives do not constrain wrist roll.",
    )
    parser.add_argument(
        "--gripper-open-angle", type=float, default=0.75, help="Open angle for the Z1 gripper mover [rad]."
    )
    parser.add_argument("--gripper-open-start", type=float, default=0.4, help="Time to start opening the gripper [s].")
    parser.add_argument(
        "--gripper-open-duration", type=float, default=0.8, help="Time over which the gripper opens [s]."
    )
    parser.add_argument(
        "--down-axis-weight", type=float, default=0.7, help="Weight for aligning the gripper +X axis downward."
    )
    parser.add_argument("--limit-weight", type=float, default=1.0, help="Weight for Z1 joint-limit residuals.")
    parser.add_argument("--max-final-error", type=float, default=0.08, help="Maximum allowed TCP target error [m].")
    parser.add_argument(
        "--max-xy-error", type=float, default=0.06, help="Maximum allowed TCP x/y error from ellipsoid center [m]."
    )
    parser.add_argument(
        "--show-ik-targets", action="store_true", help="Draw target gizmos when the viewer supports it."
    )
    return parser


def main() -> None:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarAliengoZ1IkApproachDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
