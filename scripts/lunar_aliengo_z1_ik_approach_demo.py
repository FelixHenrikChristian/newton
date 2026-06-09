# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# region Configuration

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

# 这些名字来自 MJCF。脚本通过名字定位需要固定的腿关节、Z1 关节、椭球和夹爪几何。
ROCK_BODY_NAME = "aliengo_ik_rock"
ROCK_GEOM_NAME = "aliengo_ik_rock_geom"
ALIENGO_ROOT_JOINT_NAME = "floating_base"
Z1_BASE_BODY_NAME = "z1_link00"
Z1_EE_BODY_NAME = "z1_gripper_stator"
Z1_GRIPPER_JOINT_NAME = "z1_gripper_joint"
Z1_GRIPPER_MOVER_BODY_NAME = "z1_gripper_mover"
Z1_GRIPPER_STATOR_GEOM_NAME = "z1_gripper_stator_collision"
Z1_GRIPPER_MOVER_GEOM_NAME = "z1_gripper_mover_collision"

# 用第二个 IK 位置目标把夹爪 +X 方向拉向世界坐标的向下方向。
DOWN_AXIS_LENGTH = 0.08  # 第二个 IK 位置目标相对夹爪尖中点沿局部 +X 的距离 [m]。

UNFOLD_DURATION = 2.0  # 从初始折叠姿态移动到预抓取点的时间 [s]。
DESCENT_DURATION = 2.0  # 从预抓取点垂直下探到最终目标点的时间 [s]。
PREGRASP_HEIGHT = 0.30  # 预抓取点高于最终目标点的高度 [m]。
FINAL_CLEARANCE = 0.0  # 最终夹爪尖中点高于椭球顶部的间隙 [m]。
GRASP_TARGET_OFFSET = (0.0, 0.0, 0.0)  # 在椭球顶部目标点基础上叠加的世界坐标偏移 [m]。

IK_ITERS = 32  # 每帧 IK 求解迭代次数。
IK_SEEDS = 1  # IK 候选种子数量；大于 1 时启用 Roberts 采样。
IK_STEP_SIZE = 0.8  # Levenberg-Marquardt IK 单步更新尺度。
IK_LAMBDA_INITIAL = 0.1  # IK 初始阻尼系数。
MAX_JOINT_STEP = 0.05  # 每帧允许写回完整场景的最大 Z1 关节角变化 [rad]。
DOWN_AXIS_RAMP_DURATION = 2.0  # 夹爪朝下约束从 0 平滑增加到目标权重的时间 [s]。
DOWN_AXIS_WEIGHT = 0.7  # 夹爪局部 +X 轴朝世界 -Z 方向的约束权重。
LIMIT_WEIGHT = 1.0  # IK 关节限位残差权重。
LOCK_WRIST_ROLL = True  # 是否固定 z1_joint6，避免仅靠位置目标时腕部绕夹爪轴自由旋转。

GRIPPER_OPEN_ANGLE = 0.75  # z1_gripper_joint 最终打开角度 [rad]。
GRIPPER_OPEN_START = 0.4  # 开始打开夹爪的时间 [s]。
GRIPPER_OPEN_DURATION = 0.8  # 夹爪从闭合到目标打开角度的过渡时间 [s]。

# endregion

# region Small helpers: formatting and interpolation


def _quat_xyzw_to_mjcf_wxyz(q: np.ndarray) -> tuple[float, float, float, float]:
    """把 Newton/Warp 常用的 xyzw 四元数顺序转成 MJCF 需要的 wxyz。"""
    return float(q[3]), float(q[0]), float(q[1]), float(q[2])


def _format_vec(values) -> str:
    """把数值向量格式化成 MJCF 属性字符串。"""
    return " ".join(f"{float(value):.9g}" for value in values)


def _format_float(value: float) -> str:
    """把单个浮点数格式化成 MJCF 属性字符串。"""
    return f"{float(value):.9g}"


def _smoothstep(alpha: float) -> float:
    """平滑插值曲线, 用来避免目标轨迹在阶段切换时速度突变。"""
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _wp_vec3(values: np.ndarray) -> wp.vec3:
    """把 NumPy 三维向量转成 Warp 的 vec3 类型。"""
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))

# endregion

# region MJCF parsing and gripper geometry helpers


def _parse_vec_attr(element: ET.Element, attr: str, default: tuple[float, float, float] | None = None) -> np.ndarray:
    """读取 MJCF 元素上的三维向量属性, 例如 pos、size 或 axis。"""
    text = element.get(attr)
    if text is None:
        if default is None:
            raise ValueError(f"Expected MJCF element '{element.get('name')}' to define '{attr}'.")
        return np.asarray(default, dtype=np.float32)

    values = np.fromstring(text, sep=" ", dtype=np.float32)
    if values.shape != (3,):
        raise ValueError(f"Expected MJCF '{attr}' on '{element.get('name')}' to have three values, got: {text}")
    return values


def _find_mjcf_element(root: ET.Element, tag: str, name: str) -> ET.Element:
    """在 MJCF 树里按 tag 和 name 找唯一元素。"""
    matches = [elem for elem in root.iter(tag) if elem.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one MJCF {tag} named '{name}', found {len(matches)}.")
    return matches[0]


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """根据局部转轴和角度生成 3x3 旋转矩阵。"""
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        raise ValueError("Cannot build a rotation matrix from a zero-length axis.")

    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _resolve_mesh_asset_path(mjcf: Path, root: ET.Element, mesh_asset: ET.Element) -> Path:
    file_attr = mesh_asset.get("file")
    if file_attr is None:
        raise ValueError(f"Expected MJCF mesh asset '{mesh_asset.get('name')}' to define 'file'.")

    path = Path(file_attr)
    if path.is_absolute():
        return path

    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    return (mjcf.parent / meshdir / path).resolve()


def _stl_mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []

    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            offset = 84
            for _ in range(triangle_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack_from("<fff", data, offset))
                    offset += 12
                offset += 2
            values = np.asarray(vertices, dtype=np.float32)
            return values.min(axis=0), values.max(axis=0)

    for line in data.decode(errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex "):
            vertices.append(tuple(float(value) for value in stripped.split()[1:4]))

    if not vertices:
        raise ValueError(f"Could not read STL vertices from {path}.")

    values = np.asarray(vertices, dtype=np.float32)
    return values.min(axis=0), values.max(axis=0)


def _obj_mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))

    if not vertices:
        raise ValueError(f"Could not read OBJ vertices from {path}.")

    values = np.asarray(vertices, dtype=np.float32)
    return values.min(axis=0), values.max(axis=0)


def _mesh_asset_bounds(mjcf: Path, root: ET.Element, mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh_asset = _find_mjcf_element(root, "mesh", mesh_name)
    mesh_path = _resolve_mesh_asset_path(mjcf, root, mesh_asset)
    suffix = mesh_path.suffix.lower()

    if suffix == ".stl":
        mins, maxs = _stl_mesh_bounds(mesh_path)
    elif suffix == ".obj":
        mins, maxs = _obj_mesh_bounds(mesh_path)
    else:
        raise ValueError(f"Unsupported gripper mesh format for IK target bounds: {mesh_path}")

    scale = _parse_vec_attr(mesh_asset, "scale", (1.0, 1.0, 1.0))
    scaled_mins = np.minimum(mins * scale, maxs * scale)
    scaled_maxs = np.maximum(mins * scale, maxs * scale)
    return scaled_mins, scaled_maxs


def _gripper_tip_offset_from_geom(mjcf: Path, root: ET.Element, geom: ET.Element) -> np.ndarray:
    geom_pos = _parse_vec_attr(geom, "pos", (0.0, 0.0, 0.0))
    if geom.get("type", "sphere") == "box":
        return geom_pos + np.array([_parse_vec_attr(geom, "size")[0], 0.0, 0.0], dtype=np.float32)

    mesh_name = geom.get("mesh")
    if mesh_name is None:
        raise ValueError(f"Expected gripper geom '{geom.get('name')}' to define 'mesh' or box 'size'.")

    mins, maxs = _mesh_asset_bounds(mjcf, root, mesh_name)
    center = 0.5 * (mins + maxs)
    return geom_pos + np.array([maxs[0], center[1], center[2]], dtype=np.float32)


def _gripper_tip_center_offset(mjcf: Path, open_angle: float) -> np.ndarray:
    """计算张开后两爪尖中点在 z1_gripper_stator 局部坐标系里的位置。"""
    root = ET.parse(mjcf).getroot()
    stator_geom = _find_mjcf_element(root, "geom", Z1_GRIPPER_STATOR_GEOM_NAME)
    mover_body = _find_mjcf_element(root, "body", Z1_GRIPPER_MOVER_BODY_NAME)
    mover_geom = _find_mjcf_element(root, "geom", Z1_GRIPPER_MOVER_GEOM_NAME)
    gripper_joint = _find_mjcf_element(root, "joint", Z1_GRIPPER_JOINT_NAME)

    # MJCF 中 body/geom 的 pos 是局部坐标。这里算出来的 link_offset 也保持为局部坐标,
    # 所以后续夹爪怎么运动, 这个点都会跟着 z1_gripper_stator 一起变换到世界坐标。
    stator_tip = _gripper_tip_offset_from_geom(mjcf, root, stator_geom)
    mover_tip_closed = _gripper_tip_offset_from_geom(mjcf, root, mover_geom)
    mover_rotation = _axis_angle_matrix(_parse_vec_attr(gripper_joint, "axis", (0.0, 0.0, 1.0)), float(open_angle))
    mover_tip = _parse_vec_attr(mover_body, "pos", (0.0, 0.0, 0.0)) + mover_rotation @ mover_tip_closed
    return 0.5 * (stator_tip + mover_tip)


def _parse_joint_refs(mjcf: Path, names: tuple[str, ...]) -> tuple[float, ...]:
    """从 MJCF 读取关节 ref, 作为固定腿部姿态的初始角度。"""
    root = ET.parse(mjcf).getroot()
    refs = []
    for name in names:
        matches = [elem for elem in root.iter("joint") if elem.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one MJCF joint named '{name}', found {len(matches)}.")
        refs.append(float(matches[0].get("ref", "0.0")))
    return tuple(refs)

# endregion

# region IK-only MJCF builder


def _build_z1_ik_mjcf(base_tf: np.ndarray, limits: tuple[tuple[float, float], ...]) -> str:
    """构造只包含 Z1 机械臂的临时 MJCF, 专门给 IK 求解器使用。"""
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

# endregion


class LunarAliengoZ1IkApproachDemo:
    """固定 Aliengo 身体和腿部, 只用 Z1 机械臂向椭球上方展开并下降。"""

    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.args = args
        self.mjcf = Path(args.mjcf).resolve()

        # 导入完整场景, 用来显示 Aliengo、Z1、椭球和月面。
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

        # 缓存需要保持固定的根节点和腿部关节; 每帧都会重新写回这些值。
        self.root_q = self.model.joint_q.numpy()[self.root_q_slice].astype(np.float32)
        self.leg_q = np.asarray(_parse_joint_refs(self.mjcf, LEG_JOINT_NAMES), dtype=np.float32)
        self.z1_q = np.zeros(len(Z1_JOINT_NAMES), dtype=np.float32)
        self.gripper_q = np.zeros(1, dtype=np.float32)
        self.full_joint_q = self.model.joint_q.numpy().astype(np.float32)

        self._set_scene_state(self.z1_q)
        body_q = self.state.body_q.numpy()
        z1_base_tf = body_q[self.z1_base_body_index].astype(np.float32)

        # IK 只需要 Z1 链条本身。单独构造一个小模型能避免把整条狗和场景都放进 IK。
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

        # track_offset 是“夹爪上的哪个局部点要去追踪目标”。这里选择张开后两爪尖中点。
        self.track_offset_np = _gripper_tip_center_offset(self.mjcf, GRIPPER_OPEN_ANGLE).astype(np.float32)
        self.track_offset = _wp_vec3(self.track_offset_np)
        grasp_target_offset = np.asarray(GRASP_TARGET_OFFSET, dtype=np.float32)

        # final_target 是“世界坐标里的目标点”。默认是椭球顶部, 可用命令行 offset 微调。
        self.final_target = np.array(
            [self.rock_pos[0], self.rock_pos[1], self.rock_top_z + FINAL_CLEARANCE], dtype=np.float32
        ) + grasp_target_offset
        self.pregrasp_target = self.final_target + np.array([0.0, 0.0, PREGRASP_HEIGHT], dtype=np.float32)
        self.start_target = self._link_point_position(self.state.body_q.numpy(), self.z1_ee_body_index, self.track_offset)
        self.current_target = self.start_target.copy()

        # 位置目标: 让夹爪尖中点走到 current_target。
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_body_index,
            link_offset=self.track_offset,
            target_positions=wp.array([wp.vec3(*self.current_target)], dtype=wp.vec3),
            weight=1.0,
        )
        # 方向目标: 让夹爪的 +X 轴大致朝下, 形成从上往下接近椭球的姿态。
        self.down_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_body_index,
            link_offset=self.track_offset + wp.vec3(DOWN_AXIS_LENGTH, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*self._down_axis_target(self.current_target))], dtype=wp.vec3),
            weight=DOWN_AXIS_WEIGHT,
        )
        self.base_down_axis_weight = DOWN_AXIS_WEIGHT
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=LIMIT_WEIGHT,
        )
        self.z1_lower = self.model.joint_limit_lower.numpy()[self.z1_dof_slice].astype(np.float32)
        self.z1_upper = self.model.joint_limit_upper.numpy()[self.z1_dof_slice].astype(np.float32)
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.down_obj, self.limit_obj],
            lambda_initial=IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            sampler=ik.IKSampler.ROBERTS if IK_SEEDS > 1 else ik.IKSampler.NONE,
            n_seeds=IK_SEEDS,
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
        """根据关节名找到 joint_q 和 joint_qd 里对应的连续切片。"""
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
        """在完整场景模型中查找 body 下标。"""
        return self._find_body_index_in_labels(self.model.body_label, name)

    @staticmethod
    def _find_body_index_in_labels(labels, name: str) -> int:
        """在 Newton 导入后的 label 列表中按原始 MJCF 名字查找 body。"""
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named '{name}', found {len(matches)}.")
        return matches[0]

    def _find_shape_index(self, name: str) -> int:
        """在完整场景模型中查找 geom/shape 下标。"""
        matches = [i for i, label in enumerate(self.model.shape_label) if label == name or label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one shape named '{name}', found {len(matches)}.")
        return matches[0]

    def _z1_limits(self) -> tuple[tuple[float, float], ...]:
        """读取 Z1 六个关节的上下限, 供临时 IK 模型复用。"""
        lower = self.model.joint_limit_lower.numpy()[self.z1_dof_slice]
        upper = self.model.joint_limit_upper.numpy()[self.z1_dof_slice]
        return tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True))

    def _set_scene_state(self, z1_q: np.ndarray) -> None:
        """把当前 Z1、夹爪和固定的 Aliengo 姿态写回完整场景并执行 FK。"""
        self.full_joint_q[self.root_q_slice] = self.root_q
        self.full_joint_q[self.leg_q_slice] = self.leg_q
        self.full_joint_q[self.z1_q_slice] = z1_q
        self.full_joint_q[self.gripper_q_slice] = self.gripper_q
        self.state.joint_q.assign(self.full_joint_q)
        self.state.joint_qd.zero_()
        self.state.body_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    @staticmethod
    def _link_point_position(body_q: np.ndarray, body_index: int, link_offset: wp.vec3) -> np.ndarray:
        """把 link 局部点 link_offset 变换成世界坐标位置。"""
        point = wp.transform_point(wp.transform(*body_q[body_index]), link_offset)
        return np.asarray(point, dtype=np.float32)

    @staticmethod
    def _down_axis_target(tcp_target: np.ndarray) -> np.ndarray:
        """给方向约束生成第二个目标点: 位于主目标正下方。"""
        return tcp_target + np.array([0.0, 0.0, -DOWN_AXIS_LENGTH], dtype=np.float32)

    def _target_at_time(self) -> np.ndarray:
        """根据当前时间生成平滑目标轨迹: 先展开到预抓取点, 再下降到最终点。"""
        if self.sim_time < UNFOLD_DURATION:
            alpha = _smoothstep(self.sim_time / max(UNFOLD_DURATION, 1.0e-6))
            return (1.0 - alpha) * self.start_target + alpha * self.pregrasp_target

        descent_t = self.sim_time - UNFOLD_DURATION
        if descent_t < DESCENT_DURATION:
            alpha = _smoothstep(descent_t / max(DESCENT_DURATION, 1.0e-6))
            return (1.0 - alpha) * self.pregrasp_target + alpha * self.final_target

        return self.final_target.copy()

    def _solve_ik_target(self, target: np.ndarray) -> np.ndarray:
        """针对当前目标点执行一次 IK, 并返回限幅后的 Z1 关节角。"""
        self.pos_obj.set_target_position(0, wp.vec3(*target))
        self.down_obj.set_target_position(0, wp.vec3(*self._down_axis_target(target)))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=IK_ITERS,
            step_size=IK_STEP_SIZE,
        )
        z1_q = self.ik_joint_q.numpy()[0].astype(np.float32)
        if LOCK_WRIST_ROLL:
            z1_q[5] = 0.0
        z1_q = self._limit_joint_step(z1_q)
        self.ik_joint_q.assign(z1_q.reshape(1, -1))
        return z1_q

    def _limit_joint_step(self, candidate_q: np.ndarray) -> np.ndarray:
        """限制每帧关节角变化, 避免 IK 在相邻帧之间跳变太大。"""
        candidate_q = np.clip(candidate_q, self.z1_lower, self.z1_upper)
        max_step = MAX_JOINT_STEP
        if max_step <= 0.0:
            return candidate_q

        delta = candidate_q - self.z1_q
        step = float(np.max(np.abs(delta)))
        if step <= max_step:
            return candidate_q

        limited_q = self.z1_q + delta * (max_step / step)
        return np.clip(limited_q, self.z1_lower, self.z1_upper).astype(np.float32)

    def _update_down_axis_weight(self) -> None:
        """逐渐引入夹爪朝下约束, 减少展开初期的姿态突变。"""
        if DOWN_AXIS_RAMP_DURATION <= 0.0:
            self.down_obj.weight = self.base_down_axis_weight
            return

        self.down_obj.weight = self.base_down_axis_weight * _smoothstep(self.sim_time / DOWN_AXIS_RAMP_DURATION)

    def _update_gripper(self) -> None:
        """按时间平滑打开夹爪, 只展开不执行闭合抓取。"""
        open_duration = max(GRIPPER_OPEN_DURATION, 1.0e-6)
        alpha = _smoothstep((self.sim_time - GRIPPER_OPEN_START) / open_duration)
        self.gripper_q[0] = GRIPPER_OPEN_ANGLE * alpha

    def _print_scene_info(self) -> None:
        """打印场景、目标和固定姿态信息, 方便检查当前配置是否合理。"""
        print(
            "[INFO] Lunar Aliengo Z1 IK approach: "
            f"rock center={np.round(self.rock_pos, 4)}, "
            f"radii={np.round(self.rock_radii, 4)}, "
            f"jaw_tip_center_offset={np.round(self.track_offset_np, 4)}, "
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
        """推进一帧: 更新目标、求 IK、写回完整场景并记录误差。"""
        self._update_down_axis_weight()
        self._update_gripper()
        self.current_target = self._target_at_time()
        self.z1_q = self._solve_ik_target(self.current_target)
        self._set_scene_state(self.z1_q)
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """把当前完整场景状态提交给 viewer; 可选显示 IK 目标点。"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.args.show_ik_targets and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "target_z1_jaw_tip_center", wp.transform(wp.vec3(*self.current_target), wp.quat_identity())
            )
            self.viewer.log_gizmo(
                "target_rock_top", wp.transform(wp.vec3(*self.final_target), wp.quat_identity())
            )
        self.viewer.end_frame()

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数, 控制轨迹时长、IK 求解、夹爪展开和目标偏移。"""
    parser = newton.examples.create_parser()
    parser.description = "Use Newton IK to unfold Aliengo's Z1 arm and descend from above toward the lunar ellipsoid."
    parser.set_defaults(num_frames=360, viewer="gl")
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_SCENE), help="Path to the Aliengo+Z1 lunar MJCF file.")
    parser.add_argument(
        "--show-ik-targets", action="store_true", help="Draw target gizmos when the viewer supports it."
    )
    return parser


def main() -> None:
    """脚本入口: 初始化 viewer、创建 demo, 并交给 Newton 示例循环运行。"""
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    demo = LunarAliengoZ1IkApproachDemo(viewer, args)
    newton.examples.run(demo, args)


if __name__ == "__main__":
    main()
