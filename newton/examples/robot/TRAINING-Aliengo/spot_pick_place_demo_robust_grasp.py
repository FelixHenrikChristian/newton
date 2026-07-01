from __future__ import annotations

import argparse
import math

import numpy as np
import spot_pick_place_demo as base
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# 原 demo 计算出的默认石子世界坐标, 用来得到默认的相对 B 点偏移.
ORIGINAL_STONE_XY = base._stone_xy()

# 原 demo 强制切到 B 抓取姿态时的机身位置和 yaw.
# robust 版本不强制切姿态, 但保留这个高度和倾斜作为 IK 根姿态参考.
ORIGINAL_BASE_XY = base.B_GRASP_ROOT_Q[:2].astype(np.float64)
ORIGINAL_BASE_YAW = math.atan2(
    2.0 * (base.B_GRASP_ROOT_Q[6] * base.B_GRASP_ROOT_Q[5] + base.B_GRASP_ROOT_Q[3] * base.B_GRASP_ROOT_Q[4]),
    1.0 - 2.0 * (base.B_GRASP_ROOT_Q[4] * base.B_GRASP_ROOT_Q[4] + base.B_GRASP_ROOT_Q[5] * base.B_GRASP_ROOT_Q[5]),
)

# 路径点. A/B/C 直接写死, A_YAW 由 A->B 方向计算.
A_POINT = np.array([5.0, -6.0], dtype=np.float64)
B_POINT = np.array([9.0, -3.5], dtype=np.float64)
C_POINT = np.array([13.0, -1.0], dtype=np.float64)
A_YAW = float(math.atan2(B_POINT[1] - A_POINT[1], B_POINT[0] - A_POINT[0]))

# --stone-x 和 --stone-y 是相对 B 点的世界坐标偏移.
# 默认 offset 来自原 demo 的默认石子位置, 所以不传参数时行为保持一致.
ORIGINAL_STONE_OFFSET_XY = ORIGINAL_STONE_XY - base.B_POINT

# mover 指尖所在 body 名称. stator 使用 base.ARM_EE_BODY.
FINGER_BODY = "arm_link_fngr"

# IK 跟踪点, 用 wr1 上的 stator 代理点对准石子一侧.
STATOR_IK_OFFSET = wp.vec3(0.152, 0.0, -0.045)

# 接触判定代理点, 近似 stator 和 mover 的可视接触位置.
STATOR_CONTACT_OFFSET = wp.vec3(0.232, 0.0, -0.045)
MOVER_CONTACT_OFFSET = wp.vec3(0.112, 0.0, -0.052)

# IK 姿态权重. rotation 控制长轴对齐, down-axis 控制竖直下放.
IK_ROTATION_WEIGHT = 0.35
IK_DOWN_AXIS_WEIGHT = 1.0

# 到 B 点的到达半径, 放宽一点避免在 B 附近反复修正.
GRASP_ARRIVAL_RADIUS = 0.14

# 抓取高度占石子 z 半轴的比例.
GRASP_HEIGHT_FRACTION = 0.45

# stator 下放点相对石子长轴端点向外留出的距离 [m].
GRASP_SIDE_CLEARANCE = 0.02

# IK 目标比实际接触点略高, 给下放和碰撞留余量 [m].
GRASP_IK_VERTICAL_BIAS = 0.078

# 吸附前的视觉接触误差阈值 [m]. stator 更严格, mover 略宽松.
STATOR_ATTACH_TOLERANCE = 0.018
MOVER_ATTACH_TOLERANCE = 0.04

# 视觉闭合目标. 吸附后会记录实际夹爪 q, 搬运和下放期间保持该 q.
GRIPPER_CLOSED_VISUAL = -0.5

# 石子生成后先自由稳定的时间 [s].
STONE_SETTLE_SECONDS = 1.0


def _sync_route_globals() -> None:
    """把本文件的 A/B/C 路径点同步到原 demo 模块."""

    base.A_POINT = A_POINT.copy()
    base.B_POINT = B_POINT.copy()
    base.C_POINT = C_POINT.copy()
    base.A_YAW = A_YAW


def _matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """把 3x3 旋转矩阵转换为 xyzw 四元数."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float32,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                ],
                dtype=np.float32,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                ],
                dtype=np.float32,
            )
    return base._quat_normalize(quat)


def _yaw_with_original_tilt(yaw: float) -> np.ndarray:
    """保留原 B 抓取姿态的机身倾斜, 只替换水平 yaw."""

    original_quat = base.B_GRASP_ROOT_Q[3:7].astype(np.float32)
    original_yaw_quat = np.array(base._yaw_to_xyzw(ORIGINAL_BASE_YAW), dtype=np.float32)
    yaw_quat = np.array(base._yaw_to_xyzw(yaw), dtype=np.float32)
    original_tilt = base._quat_multiply(base._quat_inverse(original_yaw_quat), original_quat)
    return base._quat_multiply(yaw_quat, original_tilt)


def _xy_axes_from_yaw(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """根据 yaw 返回水平面长轴和短轴方向."""

    long_axis = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return long_axis, short_axis


class RobustSpotPickPlaceDemo(base.SpotPickPlaceDemo):
    def __init__(self, viewer, args: argparse.Namespace):
        """初始化 robust demo, 并在原 demo 构建前覆盖路径和石子配置."""

        self.robust_stone_yaw = math.radians(float(args.stone_yaw_deg))
        self.robust_stone_size = np.array([args.stone_rx, args.stone_ry, args.stone_rz], dtype=np.float64)
        self.robust_ik_rotation_weight = float(args.ik_rotation_weight)
        self.robust_grasp_arrival_radius = float(args.grasp_arrival_radius)
        self.robust_grasp_height_fraction = float(args.grasp_height_fraction)
        self.robust_stator_attach_tolerance = STATOR_ATTACH_TOLERANCE
        self.robust_mover_attach_tolerance = MOVER_ATTACH_TOLERANCE
        self.robust_gripper_closed = GRIPPER_CLOSED_VISUAL
        self.robust_stator_side = float(args.stator_side)

        self.robust_stone_offset_xy = np.array([args.stone_x, args.stone_y], dtype=np.float64)
        self.robust_stone_xy = B_POINT + self.robust_stone_offset_xy
        self.robust_base_xy = B_POINT.copy()
        self.robust_base_yaw = A_YAW
        self.robust_base_clearance = float(base.B_GRASP_ROOT_Q[2] - base._terrain_height_at(ORIGINAL_BASE_XY))
        self.robust_root_q = self._compute_root_q()

        self._patch_base_demo_globals(float(args.stone_clearance))
        super().__init__(viewer, args)

        self.stone_q = np.array((*self.stone_pos, *base._yaw_to_xyzw(self.robust_stone_yaw)), dtype=np.float32)
        self.fngr_body_index = self._find_body_index(FINGER_BODY)
        self.stator_mj_body_id = self._find_mj_body_id(base.ARM_EE_BODY)
        self.mover_mj_body_id = self._find_mj_body_id(FINGER_BODY)
        self.grasp_stator_error = math.inf
        self.grasp_mover_error = math.inf
        self.grasp_stator_contact = False
        self.grasp_mover_contact = False
        self.stone_settled = False
        self.stone_settle_time = 0.0
        self.stone_grip_slide_active = False
        self.grasp_hold_gripper_q = None
        self.next_grasp_status_time = 0.0
        self.reset()
        print(
            "Robust grasp setup: "
            f"stone_offset_xy={np.round(self.robust_stone_offset_xy, 3).tolist()}, "
            f"stone_xy={np.round(self.robust_stone_xy, 3).tolist()}, "
            f"stone_yaw={self.robust_stone_yaw:.3f}, "
            f"base_xy={np.round(self.robust_base_xy, 3).tolist()}, "
            f"base_yaw={self.robust_base_yaw:.3f}"
        )

    def _compute_root_q(self) -> np.ndarray:
        """计算 B 点抓取时给 IK 使用的机身根姿态."""

        root_q = base.B_GRASP_ROOT_Q.copy()
        root_q[:2] = self.robust_base_xy.astype(np.float32)
        root_q[2] = np.float32(base._terrain_height_at(self.robust_base_xy) + self.robust_base_clearance)
        root_q[3:7] = _yaw_with_original_tilt(self.robust_base_yaw)
        return root_q.astype(np.float32)

    def _patch_base_demo_globals(self, stone_clearance: float) -> None:
        """覆盖原 demo 的全局路径点和石子生成函数."""

        base.STONE_SIZE = self.robust_stone_size.copy()
        _sync_route_globals()

        stone_pos = np.array(
            [
                self.robust_stone_xy[0],
                self.robust_stone_xy[1],
                base._terrain_height_at(self.robust_stone_xy) + self.robust_stone_size[2] + stone_clearance,
            ],
            dtype=np.float64,
        )

        def _robust_stone_position() -> np.ndarray:
            """返回 robust 版本计算出的石子初始坐标."""

            return stone_pos.copy()

        base._stone_position = _robust_stone_position

    def _clear_arm_ik(self) -> None:
        """清空原 demo 的 IK 状态和 robust 额外 IK 目标."""

        super()._clear_arm_ik()
        self.ik_down_obj = None
        self.ik_long_axis_obj = None
        self.ik_final_rotation = None
        self.ik_link_offset = None
        self.ik_axis_yaw = 0.0
        self.ik_base_tf = None

    def _grasp_base_root_q(self) -> np.ndarray:
        """返回 B 点抓取阶段使用的机身根姿态."""

        return self.robust_root_q.copy()

    def _find_mj_body_id(self, name: str) -> int:
        """按名称在 MuJoCo 模型中查找唯一 body id."""

        matches = []
        for body_id in range(self.solver.mj_model.nbody):
            label = self.mujoco.mj_id2name(self.solver.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            if label == name or label.endswith(f"_{name}") or label.endswith(f"/{name}"):
                matches.append(body_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one MuJoCo body named '{name}', found {len(matches)}.")
        return matches[0]

    def _stone_side_surface_targets(self, state=None) -> tuple[np.ndarray, np.ndarray]:
        """计算 stator 和 mover 应触碰的石子长轴两侧表面点."""

        if state is None:
            stone_center = self.stone_pos.astype(np.float64)
        else:
            stone_center = state.body_q.numpy()[self.stone_body_index, :3].astype(np.float64)

        long_axis, _ = _xy_axes_from_yaw(self.robust_stone_yaw)
        height_fraction = float(np.clip(self.robust_grasp_height_fraction, 0.0, 0.95))
        contact_height = self.robust_stone_size[2] * height_fraction
        side_radius = self.robust_stone_size[0] * math.sqrt(max(0.0, 1.0 - height_fraction * height_fraction))

        stator_target = stone_center.copy()
        mover_target = stone_center.copy()
        stator_target[:2] += self.robust_stator_side * long_axis * side_radius
        mover_target[:2] -= self.robust_stator_side * long_axis * side_radius
        stator_target[2] += contact_height
        mover_target[2] += contact_height
        return stator_target.astype(np.float32), mover_target.astype(np.float32)

    def _stone_stator_approach_target(self) -> np.ndarray:
        """计算 stator 下放前的水平目标点, 略微在石子外侧."""

        stator_target, _ = self._stone_side_surface_targets()
        long_axis, _ = _xy_axes_from_yaw(self.robust_stone_yaw)
        target = stator_target.astype(np.float32)
        target[:2] += (self.robust_stator_side * long_axis * GRASP_SIDE_CLEARANCE).astype(np.float32)
        return target

    def _stone_approach_target(self) -> np.ndarray:
        """计算抓取下放的最终 IK 目标, 包含竖直安全偏置."""

        stator_target = self._stone_stator_approach_target()
        return stator_target + np.array([0.0, 0.0, GRASP_IK_VERTICAL_BIAS], dtype=np.float32)

    def _wr1_rotation_target(self, yaw: float) -> np.ndarray:
        """构造 wr1 目标姿态, 让夹爪竖直下放并跟随石子长轴."""

        long_axis_xy, short_axis_xy = _xy_axes_from_yaw(yaw)
        local_x_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        local_y_world = np.array([short_axis_xy[0], short_axis_xy[1], 0.0], dtype=np.float64)
        local_z_world = np.array([long_axis_xy[0], long_axis_xy[1], 0.0], dtype=np.float64)
        rotation = np.column_stack((local_x_world, local_y_world, local_z_world))
        return _matrix_to_xyzw(rotation)

    @staticmethod
    def _down_axis_target(target: np.ndarray) -> np.ndarray:
        """生成辅助 IK 点, 约束夹爪本地 x 轴朝世界 -Z."""

        return target + np.array([0.0, 0.0, -base.WR1_DOWN_AXIS_LENGTH], dtype=np.float32)

    def _long_axis_target(self, target: np.ndarray) -> np.ndarray:
        """生成辅助 IK 点, 约束夹爪长轴方向."""

        long_axis, _ = _xy_axes_from_yaw(self.ik_axis_yaw)
        return target + np.array(
            [
                long_axis[0] * base.WR1_DOWN_AXIS_LENGTH,
                long_axis[1] * base.WR1_DOWN_AXIS_LENGTH,
                0.0,
            ],
            dtype=np.float32,
        )

    def _target_in_ik_base_frame(self, target: np.ndarray) -> np.ndarray:
        """把世界目标补偿到 IK 创建时的机身坐标系."""

        if self.ik_base_tf is None:
            return target.astype(np.float32)

        current_base_tf = self._arm_base_transform()
        initial_pos = self.ik_base_tf[:3].astype(np.float64)
        current_pos = current_base_tf[:3].astype(np.float64)
        initial_rot = base._xyzw_to_matrix(self.ik_base_tf[3:7])
        current_rot = base._xyzw_to_matrix(current_base_tf[3:7])
        local_target = current_rot.T @ (target.astype(np.float64) - current_pos)
        return (initial_pos + initial_rot @ local_target).astype(np.float32)

    def _start_arm_ik_motion(self, final_target: np.ndarray, label: str) -> None:
        """创建机械臂 IK 模型和目标, 用于抓取下放或 C 点下放."""

        self.phase_time = 0.0
        self.arm_q_cmd = self.state_0.joint_q.numpy()[self.arm_q_slice].astype(np.float32)
        self.ik_base_tf = self._arm_base_transform()

        ik_builder = newton.ModelBuilder()
        ik_builder.add_mjcf(
            base._build_spot_arm_ik_mjcf(self.ik_base_tf, self._arm_joint_limits()),
            up_axis="Z",
            enable_self_collisions=False,
        )
        self.ik_model = ik_builder.finalize()
        self.ik_wr1_body_index = self._find_body_index_in_labels(self.ik_model.body_label, base.ARM_EE_BODY)
        self.ik_joint_q = wp.array(self.arm_q_cmd.reshape(1, -1), dtype=wp.float32)

        approach_motion = label == "approach"
        self.ik_link_offset = STATOR_IK_OFFSET if approach_motion else base.GRIPPER_TRACK_OFFSET

        body_q = self.state_0.body_q.numpy()
        self.ik_start_target = self._link_point_position(body_q, self.wr1_body_index, self.ik_link_offset)
        self.ik_final_target = final_target
        self.ik_pregrasp_target = self.ik_final_target + np.array([0.0, 0.0, base.ARM_PREGRASP_HEIGHT], dtype=np.float32)
        self.current_ik_target = self.ik_start_target.copy()
        self.ik_axis_yaw = self.robust_stone_yaw if approach_motion else self._base_yaw()
        self.ik_final_rotation = self._wr1_rotation_target(
            self.ik_axis_yaw
        )

        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset,
            target_positions=wp.array([wp.vec3(*self.current_ik_target)], dtype=wp.vec3),
            weight=1.0,
        )
        self.ik_down_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset + wp.vec3(base.WR1_DOWN_AXIS_LENGTH, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self._down_axis_target(self.current_ik_target))], dtype=wp.vec3),
            weight=IK_DOWN_AXIS_WEIGHT,
        )
        self.ik_long_axis_obj = ik.IKObjectivePosition(
            link_index=self.ik_wr1_body_index,
            link_offset=self.ik_link_offset + wp.vec3(0.0, 0.0, base.WR1_DOWN_AXIS_LENGTH),
            target_positions=wp.array([wp.vec3(*self._long_axis_target(self.current_ik_target))], dtype=wp.vec3),
            weight=self.robust_ik_rotation_weight,
        )
        self.ik_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=base.IK_LIMIT_WEIGHT,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_down_obj, self.ik_long_axis_obj, self.ik_limit_obj],
            lambda_initial=base.IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        link_offset_label = "stator_track" if approach_motion else "gripper_track"

        print(
            f"Start robust arm IK {label}: "
            f"pregrasp={np.round(self.ik_pregrasp_target, 3).tolist()}, "
            f"final={np.round(self.ik_final_target, 3).tolist()}, "
            f"wr1_rot={np.round(self.ik_final_rotation, 3).tolist()}, "
            f"link_offset={link_offset_label}"
        )

    def reset(self) -> None:
        """重置 demo 状态, 并恢复 robust 抓取相关标志."""

        super().reset()
        self.grasp_stator_error = math.inf
        self.grasp_mover_error = math.inf
        self.grasp_stator_contact = False
        self.grasp_mover_contact = False
        self.stone_settled = False
        self.stone_settle_time = 0.0
        self.stone_grip_slide_active = False
        self.pin_stone_until_grasp = False
        self.grasp_hold_gripper_q = None
        self.next_grasp_status_time = 0.0

    def _grasp_hold_q(self) -> float:
        """返回吸附瞬间记录的夹爪开合量."""

        if self.grasp_hold_gripper_q is None:
            return self.robust_gripper_closed
        return self.grasp_hold_gripper_q

    def _gripper_target_at_time(self) -> float:
        """根据抓取阶段时间计算夹爪目标."""

        if self.stone_attached:
            return self._grasp_hold_q()

        close_time = self.phase_time - base.ARM_PREGRASP_SECONDS - base.ARM_DESCENT_SECONDS
        if close_time <= 0.0:
            return base.GRIPPER_OPEN

        alpha = base._smoothstep(close_time / max(base.GRIPPER_CLOSE_SECONDS, 1.0e-6))
        return float((1.0 - alpha) * base.GRIPPER_OPEN + alpha * self.robust_gripper_closed)

    def _solve_arm_ik(self, target: np.ndarray) -> np.ndarray:
        """更新 IK 目标并求解下一帧机械臂控制量."""

        ik_target = self._target_in_ik_base_frame(target)
        ik_down_target = self._target_in_ik_base_frame(self._down_axis_target(target))
        ik_long_axis_target = self._target_in_ik_base_frame(self._long_axis_target(target))
        self.ik_pos_obj.set_target_position(0, wp.vec3(*ik_target))
        self.ik_down_obj.set_target_position(0, wp.vec3(*ik_down_target))
        self.ik_long_axis_obj.set_target_position(0, wp.vec3(*ik_long_axis_target))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=base.IK_ITERATIONS,
            step_size=base.IK_STEP_SIZE,
        )
        candidate_q = self.ik_joint_q.numpy()[0].astype(np.float32)
        candidate_q[-1] = self._gripper_target_at_time()
        arm_q = self._limit_arm_joint_step(candidate_q)
        self.ik_joint_q.assign(arm_q.reshape(1, -1))
        return arm_q

    def _arm_stowed_q(self) -> np.ndarray:
        """返回收臂姿态, 搬运时保持吸附瞬间夹爪 q."""

        arm_q = super()._arm_stowed_q()
        if self.stone_attached:
            arm_q[-1] = self._grasp_hold_q()
        return arm_q

    def _attach_stone_to_gripper(self, state) -> None:
        """把石子吸附到夹爪, 并记录当前夹爪开合量."""

        self.grasp_hold_gripper_q = float(self.arm_q_cmd[-1])
        super()._attach_stone_to_gripper(state)
        print(f"Holding gripper at grasp q={self.grasp_hold_gripper_q:.3f}")

    def _release_stone(self, state) -> None:
        """释放石子并清空吸附期间保存的夹爪开合量."""

        super()._release_stone(state)
        self.grasp_hold_gripper_q = None

    def _release_stone_pin_for_grip(self) -> None:
        """夹爪开始闭合时释放水平约束, 让碰撞把石子推到双侧接触."""

        if not self.pin_stone_until_grasp:
            return
        self.pin_stone_until_grasp = False
        self.stone_grip_slide_active = True
        self.state_0.joint_qd[self.stone_qd_slice].zero_()
        self.state_1.joint_qd[self.stone_qd_slice].zero_()
        print("Stone horizontal slide released; closing gripper until both jaws touch")

    def _update_stone_constraint(self, state) -> None:
        """根据当前阶段更新石子约束状态."""

        if self.stone_attached:
            self._attach_stone_pose(state)
        elif self.pin_stone_until_grasp:
            self._pin_stone_to_start(state)
        elif self.stone_grip_slide_active and not self.stone_released:
            stone_q = state.joint_q.numpy()[self.stone_q_slice].copy()
            stone_q[2] = self.stone_q[2]
            stone_q[3:7] = self.stone_q[3:7]
            state.joint_q[self.stone_q_slice].assign(stone_q)
            stone_qd = state.joint_qd.numpy()[self.stone_qd_slice].copy()
            stone_qd[2:] = 0.0
            state.joint_qd[self.stone_qd_slice].assign(stone_qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        elif not self.stone_settled:
            stone_q = state.joint_q.numpy()[self.stone_q_slice].copy()
            stone_q[3:7] = self.stone_q[3:7]
            state.joint_q[self.stone_q_slice].assign(stone_q)
            stone_qd = state.joint_qd.numpy()[self.stone_qd_slice].copy()
            stone_qd[3:] = 0.0
            state.joint_qd[self.stone_qd_slice].assign(stone_qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _finish_stone_settle(self) -> None:
        """结束石子自由稳定阶段, 记录真实落点并固定到抓取前."""

        settled_q = self.state_0.joint_q.numpy()[self.stone_q_slice].copy().astype(np.float32)
        settled_q[3:7] = self.stone_q[3:7]
        self.stone_q = settled_q
        self.stone_pos = self.state_0.body_q.numpy()[self.stone_body_index, :3].copy().astype(np.float32)
        self.robust_stone_xy = self.stone_pos[:2].astype(np.float64)
        self.robust_stone_offset_xy = self.robust_stone_xy - B_POINT
        _sync_route_globals()
        self.pin_stone_until_grasp = True
        self.stone_settled = True
        for state in (self.state_0, self.state_1):
            state.joint_q[self.stone_q_slice].assign(self.stone_q)
            state.joint_qd[self.stone_qd_slice].zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        print(
            "Stone settled: "
            f"pos={np.round(self.stone_pos, 3).tolist()}, "
            f"stone_offset_xy={np.round(self.robust_stone_offset_xy, 3).tolist()}, "
            f"a={np.round(A_POINT, 3).tolist()}, "
            f"b={np.round(B_POINT, 3).tolist()}, "
            f"c={np.round(C_POINT, 3).tolist()}, "
            f"a_yaw={A_YAW:.3f}"
        )

    def _apply_arm_approach(self) -> None:
        """执行 B 点抓取动作, 包括移动到目标, 下放和闭合夹爪."""

        if self.ik_solver is None:
            self._start_arm_approach()

        self.command.fill(0.0)
        if self.phase_time >= base.ARM_PREGRASP_SECONDS + base.ARM_DESCENT_SECONDS:
            self._release_stone_pin_for_grip()
        self.current_ik_target = self._arm_target_at_time()
        self.arm_q_cmd = self._solve_arm_ik(self.current_ik_target)
        self._write_arm_ctrl(self.arm_q_cmd)

        body_q = self.state_0.body_q.numpy()
        actual = self._link_point_position(body_q, self.wr1_body_index, self.ik_link_offset)
        self.ik_error = float(np.linalg.norm(actual - self.current_ik_target))

        self.phase_time += self.frame_dt

    def _grasp_contact_ready(self, state) -> bool:
        """判断 stator 和 mover 是否都已接近目标且真实碰撞."""

        close_started = self.phase_time >= base.ARM_PREGRASP_SECONDS + base.ARM_DESCENT_SECONDS
        if not close_started:
            return False

        body_q = state.body_q.numpy()
        stator_point = self._link_point_position(body_q, self.wr1_body_index, STATOR_CONTACT_OFFSET)
        mover_point = self._link_point_position(body_q, self.fngr_body_index, MOVER_CONTACT_OFFSET)
        stator_target, mover_target = self._stone_side_surface_targets(state)
        stator_delta = stator_point - stator_target
        mover_delta = mover_point - mover_target
        self.grasp_stator_error = float(np.linalg.norm(stator_point - stator_target))
        self.grasp_mover_error = float(np.linalg.norm(mover_point - mover_target))
        self.grasp_stator_contact, self.grasp_mover_contact = self._stone_jaw_contact_flags()

        if self.sim_time >= self.next_grasp_status_time:
            print(
                "Waiting for jaw contact: "
                f"stator_err={self.grasp_stator_error:.3f} m, "
                f"stator_delta={np.round(stator_delta, 3).tolist()}, "
                f"stator_contact={self.grasp_stator_contact}, "
                f"mover_err={self.grasp_mover_error:.3f} m, "
                f"mover_delta={np.round(mover_delta, 3).tolist()}, "
                f"mover_contact={self.grasp_mover_contact}"
            )
            self.next_grasp_status_time = self.sim_time + 1.0

        contact_ready = (
            self.grasp_stator_error <= self.robust_stator_attach_tolerance
            and self.grasp_mover_error <= self.robust_mover_attach_tolerance
            and self.grasp_stator_contact
            and self.grasp_mover_contact
        )
        if contact_ready:
            print(
                "Jaw contact ready: "
                f"stator_err={self.grasp_stator_error:.3f} m, "
                f"stator_contact={self.grasp_stator_contact}, "
                f"mover_err={self.grasp_mover_error:.3f} m, "
                f"mover_contact={self.grasp_mover_contact}"
            )
        return contact_ready

    def _stone_jaw_contact_flags(self) -> tuple[bool, bool]:
        """从 MuJoCo contact 中读取石子是否碰到两个夹爪侧."""

        stator_contact = False
        mover_contact = False
        geom_body_ids = self.solver.mj_model.geom_bodyid
        for i in range(self.solver.mj_data.ncon):
            contact = self.solver.mj_data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.stone_geom_id:
                other_body = int(geom_body_ids[geom2])
            elif geom2 == self.stone_geom_id:
                other_body = int(geom_body_ids[geom1])
            else:
                continue
            if other_body == self.stator_mj_body_id:
                stator_contact = True
            elif other_body == self.mover_mj_body_id:
                mover_contact = True
        return stator_contact, mover_contact

    def step(self) -> None:
        """推进一帧仿真, 并按阶段执行行走, 抓取, 搬运或放下."""

        if not self.stone_settled:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.stand_ctrl[base.ACT_DIM:])
        elif not self.reached_b and self.sim_time < base.MAX_SECONDS:
            self.reached_b = self._apply_policy(
                B_POINT,
                "B grasp stand",
                self.stand_ctrl[base.ACT_DIM:],
                face_xy=None,
                arrival_radius=self.robust_grasp_arrival_radius,
                require_yaw=False,
            )
        elif self.reached_b and not self.b_aligned:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.stand_ctrl[base.ACT_DIM:])
            self.b_aligned = True
            print(
                "Standing at B without forced pose: "
                f"position={self._base_xy().round(3).tolist()}, yaw={self._base_yaw():.3f}"
            )
        elif not self.stone_attached and not self.stone_released:
            self._apply_arm_approach()
        elif not self.arm_retracted:
            self._apply_arm_retract()
        elif not self.reached_c and self.sim_time < base.MAX_SECONDS:
            self.reached_c = self._apply_policy(
                C_POINT,
                "C",
                self.arm_q_cmd,
                min_forward_speed=base.MIN_FORWARD_SPEED,
                arrival_radius=base.C_ARRIVAL_RADIUS,
                require_yaw=False,
            )
        elif not self.arm_release_opened:
            self._apply_arm_release()
        elif not self.arm_final_retracted:
            self._apply_final_arm_retract()
        else:
            self.command.fill(0.0)
            self._write_arm_ctrl(self.arm_q_cmd)

        for _ in range(base.CONTROL_DECIMATION):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_stone_constraint(self.state_0)
            if (
                self.stone_settled
                and not self.stone_attached
                and not self.stone_released
                and self._grasp_contact_ready(self.state_0)
            ):
                self._attach_stone_to_gripper(self.state_0)
        if not self.stone_settled:
            self.stone_settle_time += self.frame_dt
            if self.stone_settle_time >= STONE_SETTLE_SECONDS:
                self._finish_stone_settle()
        self.sim_time += self.frame_dt
        self.contacts_ready = True

    def test_final(self) -> None:
        """示例结束时检查数值稳定性, IK 误差和吸附时机."""

        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise ValueError("Body transforms became non-finite.")
        if self.ik_solver is not None and self.phase_time > 0.5:
            gripper_low = min(base.GRIPPER_OPEN, self.robust_gripper_closed) - 0.1
            gripper_high = max(base.GRIPPER_OPEN, self.robust_gripper_closed) + 0.1
            if not (gripper_low <= self.arm_q_cmd[-1] <= gripper_high):
                raise ValueError(f"Gripper target is out of range: q={self.arm_q_cmd[-1]:.3f}")
            if self.ik_error > 0.45:
                raise ValueError(f"Arm IK target error is too high: {self.ik_error:.3f} m")
        if self.stone_attached and (
            self.grasp_stator_error > self.robust_stator_attach_tolerance
            or self.grasp_mover_error > self.robust_mover_attach_tolerance
        ):
            raise ValueError(
                "Stone attached before both jaws reached visual contact: "
                f"stator={self.grasp_stator_error:.3f}, mover={self.grasp_mover_error:.3f}"
            )


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数, 保留原 demo 参数并追加 robust 抓取配置."""

    parser = base.create_parser()
    parser.description = "Run the Spot pick-and-place demo with a dynamic, yaw-aware visual grasp."
    parser.add_argument(
        "--stone-x",
        type=float,
        default=float(ORIGINAL_STONE_OFFSET_XY[0]),
        help="Stone world-frame x offset from B.",
    )
    parser.add_argument(
        "--stone-y",
        type=float,
        default=float(ORIGINAL_STONE_OFFSET_XY[1]),
        help="Stone world-frame y offset from B.",
    )
    parser.add_argument("--stone-yaw-deg", type=float, default=0.0, help="Stone long-axis yaw in degrees.")
    parser.add_argument("--stone-rx", type=float, default=float(base.STONE_SIZE[0]), help="Stone ellipsoid x radius.")
    parser.add_argument("--stone-ry", type=float, default=float(base.STONE_SIZE[1]), help="Stone ellipsoid y radius.")
    parser.add_argument("--stone-rz", type=float, default=float(base.STONE_SIZE[2]), help="Stone ellipsoid z radius.")
    parser.add_argument(
        "--stone-clearance",
        type=float,
        default=0.025,
        help="Initial vertical offset added to terrain height plus stone rz before settling.",
    )
    parser.add_argument(
        "--grasp-arrival-radius",
        type=float,
        default=GRASP_ARRIVAL_RADIUS,
        help="Locomotion arrival radius for the final grasp stand point.",
    )
    parser.add_argument(
        "--grasp-height-fraction",
        type=float,
        default=GRASP_HEIGHT_FRACTION,
        help="Contact height as a fraction of stone rz above the ellipsoid center.",
    )
    parser.add_argument(
        "--ik-rotation-weight",
        type=float,
        default=IK_ROTATION_WEIGHT,
        help="Weight for matching the WR1 orientation to the stone yaw.",
    )
    parser.add_argument(
        "--stator-side",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="Which long-axis side the fixed jaw approaches first.",
    )
    return parser


def main() -> None:
    """运行 robust grasp 示例."""

    viewer, args = newton.examples.init(create_parser())
    demo = RobustSpotPickPlaceDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
