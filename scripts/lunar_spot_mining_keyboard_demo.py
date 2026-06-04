# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[2]
    / "mujoco"
    / "model"
    / "lunar"
    / "lunar_mujoco_spot_arm_mining_scene"
    / "lunar_scene_spot_arm_mining.xml"
)

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
SPOT_ARM_HOME = (0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0)

GAIT_FREQUENCY = 0.6
SWING_FRACTION = 0.18
HIP_PITCH_AMPLITUDE = 0.2
HIP_ROLL_AMPLITUDE = 0.1
KNEE_LIFT_AMPLITUDE = 0.1
STANCE_PRESS_AMPLITUDE = 0.05
LATERAL_STRIDE_SCALE = 2.0
TURN_PITCH_SCALE = 0.55
TURN_ROLL_SCALE = 0.55
DEFAULT_LEG_PHASE_OFFSETS = (0.0, 0.25, 0.5, 0.75)
LEFT_LEG_PHASE_OFFSETS = (0.0, 0.75, 0.25, 0.5)
RIGHT_LEG_PHASE_OFFSETS = (0.0, 0.75, 0.5, 0.25)

BALANCE_ENVELOPE_CENTER = 0.09
BALANCE_ENVELOPE_WIDTH = 0.46
BALANCE_ROLL_AMPLITUDE = 0.08
BALANCE_PITCH_AMPLITUDE = 0.1

ROLL_HIP_KP = 0.35
ROLL_HIP_KD = 0.035
ROLL_HIP_LIMIT = 0.22
ROLL_KNEE_KP = 0.12
ROLL_KNEE_KD = 0.01
ROLL_KNEE_LIMIT = 0.1
PITCH_HIP_KP = 0.2
PITCH_HIP_KD = 0.012
PITCH_HIP_LIMIT = 0.1
PITCH_KNEE_KP = 0.08
PITCH_KNEE_KD = 0.004
PITCH_KNEE_LIMIT = 0.06


class LunarSpotMiningKeyboardDemo:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        if self.sim_substeps <= 0:
            raise ValueError("--substeps must be greater than zero.")
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.gait_phase = 0.0
        self._gait_profile = "default"
        self._reset_key_prev = False
        self._stop_key_prev = False

        builder = newton.ModelBuilder()
        builder.add_mjcf(str(Path(args.mjcf).resolve()), up_axis="Z", enable_self_collisions=True)
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
        self.leg_target_lower = self.model.joint_limit_lower.numpy()[self.leg_dof_slice]
        self.leg_target_upper = self.model.joint_limit_upper.numpy()[self.leg_dof_slice]

        self._set_spot_home_state(self.state_0)
        self.state_1.assign(self.state_0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        self._set_standing_targets()
        self._last_roll, self._last_pitch = self._get_spot_tilt()

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(11.0, 3.2, 3.2), pitch=-20.0, yaw=135.0)
        self._register_ui()
        self._print_controls()

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
            matches = [i for i, label in enumerate(self.model.joint_label) if label.endswith(f"/{name}")]
            if len(matches) != 1:
                raise ValueError(f"Expected one imported joint named '{name}', found {len(matches)}.")
            joint_indices.append(matches[0])

        q_indices = [int(q_starts[index]) for index in joint_indices]
        qd_indices = [int(qd_starts[index]) for index in joint_indices]
        if q_indices != list(range(q_indices[0], q_indices[0] + len(names))):
            if q_width is None:
                raise ValueError(f"Expected contiguous joint coordinates for: {', '.join(names)}")
        if qd_indices != list(range(qd_indices[0], qd_indices[0] + len(names))):
            if qd_width is None:
                raise ValueError(f"Expected contiguous joint DoFs for: {', '.join(names)}")

        return (
            slice(q_indices[0], q_indices[0] + (q_width or len(names))),
            slice(qd_indices[0], qd_indices[0] + (qd_width or len(names))),
        )

    def _set_spot_home_state(self, state):
        state.joint_q[self.root_q_slice].assign(SPOT_ROOT_HOME)
        state.joint_q[self.leg_q_slice].assign(SPOT_LEG_HOME)
        state.joint_q[self.arm_q_slice].assign(SPOT_ARM_HOME)
        state.joint_qd[self.root_qd_slice].zero_()
        state.joint_qd[self.leg_dof_slice].zero_()
        state.joint_qd[self.arm_dof_slice].zero_()

    def _set_standing_targets(self):
        self.control.joint_target_pos[self.leg_dof_slice].assign(SPOT_LEG_HOME)
        self.control.joint_target_pos[self.arm_dof_slice].assign(SPOT_ARM_HOME)

    def _get_spot_tilt(self) -> tuple[float, float]:
        qx, qy, qz, qw = [float(value) for value in self.state_0.joint_q.numpy()[self.root_q_slice][3:7]]
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch_sine = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, pitch_sine)))
        return roll, pitch

    def _balance_envelope(self, cycle: float) -> float:
        distance = abs((cycle - BALANCE_ENVELOPE_CENTER + 0.5) % 1.0 - 0.5)
        half_width = BALANCE_ENVELOPE_WIDTH * 0.5
        if distance >= half_width:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * distance / half_width))

    @staticmethod
    def _clamp_symmetric(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _select_leg_phase_offsets(lateral: float) -> tuple[str, tuple[float, ...]]:
        if lateral > 0.0:
            return "left", LEFT_LEG_PHASE_OFFSETS
        if lateral < 0.0:
            return "right", RIGHT_LEG_PHASE_OFFSETS
        return "default", DEFAULT_LEG_PHASE_OFFSETS

    def _register_ui(self):
        if not isinstance(self.viewer, newton.viewer.ViewerGL):
            return

        def render_ui(imgui):
            imgui.text("Lunar Spot Arm keyboard demo")
            imgui.text("I/K forward/back, J/L lateral, U/O turn")
            imgui.text("M stand, P reset scene")

        self.viewer.register_ui_callback(render_ui, position="side")

    def _print_controls(self):
        print("[INFO] Keys: I/K forward/back, J/L lateral, U/O turn, M stand, P reset scene")

    def _is_key_down(self, key: str) -> bool:
        return bool(self.viewer.is_key_down(key))

    def _read_command(self) -> tuple[float, float, float]:
        forward = float(self._is_key_down("i")) - float(self._is_key_down("k"))
        lateral = float(self._is_key_down("j")) - float(self._is_key_down("l"))
        turn = float(self._is_key_down("u")) - float(self._is_key_down("o"))

        reset_down = self._is_key_down("p")
        if reset_down and not self._reset_key_prev:
            self.reset()
        self._reset_key_prev = reset_down

        stop_down = self._is_key_down("m")
        if stop_down and not self._stop_key_prev:
            self.gait_phase = 0.0
        self._stop_key_prev = stop_down
        if stop_down:
            return 0.0, 0.0, 0.0

        magnitude = math.sqrt(forward * forward + lateral * lateral + turn * turn)
        if magnitude > 1.0:
            forward /= magnitude
            lateral /= magnitude
            turn /= magnitude
        return forward, lateral, turn

    def _set_gait_targets(self, forward: float, lateral: float, turn: float):
        activity = min(1.0, math.sqrt(forward * forward + lateral * lateral + turn * turn))
        roll, pitch = self._get_spot_tilt()
        roll_rate = (roll - self._last_roll) / self.frame_dt
        pitch_rate = (pitch - self._last_pitch) / self.frame_dt
        self._last_roll = roll
        self._last_pitch = pitch

        if activity == 0.0:
            self.gait_phase = 0.0
            self._gait_profile = "default"
            self._set_standing_targets()
            return

        gait_profile, phase_offsets = self._select_leg_phase_offsets(lateral)
        if gait_profile != self._gait_profile:
            self.gait_phase = 0.0
            self._gait_profile = gait_profile

        self.gait_phase = (self.gait_phase + math.tau * GAIT_FREQUENCY * self.frame_dt) % math.tau
        targets = np.asarray(SPOT_LEG_HOME, dtype=np.float32).copy()
        leg_phases = []
        body_roll = 0.0
        body_pitch = 0.0

        for leg_index in range(4):
            side = 1.0 if leg_index in (0, 2) else -1.0
            front = 1.0 if leg_index in (0, 1) else -1.0
            cycle = (self.gait_phase / math.tau + phase_offsets[leg_index]) % 1.0
            if cycle < SWING_FRACTION:
                swing_progress = cycle / SWING_FRACTION
                stride_wave = -1.0 + 2.0 * swing_progress
                swing_lift = math.sin(math.pi * swing_progress)
            else:
                stance_progress = (cycle - SWING_FRACTION) / (1.0 - SWING_FRACTION)
                stride_wave = 1.0 - 2.0 * stance_progress
                swing_lift = 0.0

            balance = self._balance_envelope(cycle)
            body_roll += BALANCE_ROLL_AMPLITUDE * side * balance
            body_pitch += BALANCE_PITCH_AMPLITUDE * front * balance
            leg_phases.append((side, front, stride_wave, swing_lift))

        roll_hip = self._clamp_symmetric(-(ROLL_HIP_KP * roll + ROLL_HIP_KD * roll_rate), ROLL_HIP_LIMIT)
        roll_knee = self._clamp_symmetric(ROLL_KNEE_KP * roll + ROLL_KNEE_KD * roll_rate, ROLL_KNEE_LIMIT)
        pitch_hip = self._clamp_symmetric(
            -(PITCH_HIP_KP * pitch + PITCH_HIP_KD * pitch_rate),
            PITCH_HIP_LIMIT,
        )
        pitch_knee = self._clamp_symmetric(
            PITCH_KNEE_KP * pitch + PITCH_KNEE_KD * pitch_rate,
            PITCH_KNEE_LIMIT,
        )

        for leg_index, (side, front, stride_wave, swing_lift) in enumerate(leg_phases):
            leg_offset = leg_index * 3
            leg_stride = forward - side * TURN_PITCH_SCALE * turn
            lateral_stride = LATERAL_STRIDE_SCALE * lateral + front * TURN_ROLL_SCALE * turn
            targets[leg_offset] += roll_hip + body_roll + HIP_ROLL_AMPLITUDE * lateral_stride * stride_wave
            targets[leg_offset + 1] += pitch_hip - body_pitch - HIP_PITCH_AMPLITUDE * leg_stride * stride_wave
            targets[leg_offset + 2] += (
                STANCE_PRESS_AMPLITUDE * (1.0 - swing_lift)
                - KNEE_LIFT_AMPLITUDE * activity * swing_lift
                - side * roll_knee
                + front * pitch_knee
            )

        np.clip(targets, self.leg_target_lower, self.leg_target_upper, out=targets)
        self.control.joint_target_pos[self.leg_dof_slice].assign(targets)
        self.control.joint_target_pos[self.arm_dof_slice].assign(SPOT_ARM_HOME)

    def reset(self):
        print("[INFO] Resetting lunar Spot Arm scene")
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        self.gait_phase = 0.0
        self._gait_profile = "default"
        self._set_standing_targets()
        self._last_roll, self._last_pitch = self._get_spot_tilt()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._set_gait_targets(*self._read_command())
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _check_finite(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise FloatingPointError("joint_q contains a non-finite value.")
        if not np.isfinite(self.state_0.joint_qd.numpy()).all():
            raise FloatingPointError("joint_qd contains a non-finite value.")
        if not np.isfinite(self.control.joint_target_pos.numpy()).all():
            raise FloatingPointError("joint_target_pos contains a non-finite value.")

    def test_post_step(self):
        self._check_finite()

    def test_final(self):
        self._check_finite()


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = "Control the lunar Spot arm scene with a simple actuator-driven keyboard gait."
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the lunar mining MJCF file.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per rendered frame.")
    return parser


def main():
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarSpotMiningKeyboardDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
