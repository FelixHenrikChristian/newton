# Spot 关节与控制说明

本文档说明 `spot_scene.xml` 和 `SpotGo2StyleEnv` 中使用的 Spot 腿部关节、Go2 对应关节、动作顺序和控制方向。

## 坐标约定

在 `stand` keyframe 中，Spot 根节点姿态为单位四元数。这里的坐标只用于描述机身方向，不直接表示某个关节 `ctrl` 的正负方向：

- `+x`：机器人前方，朝前腿方向。
- `+y`：机器人左侧。
- `+z`：向上。

本文中的左、右、前、后均从机器人自身视角描述。每个关节的实际正负控制效果以“关节对应表”中的描述为准。

## 命名规则

Spot 关节名使用短命名：

- `f`：front，前腿。
- `h`：hind，后腿。
- `l`：left，左腿。
- `r`：right，右腿。
- `hx`：hip roll / abduction，髋关节横滚，主要带动整条腿左右摆。
- `hy`：hip pitch，髋关节俯仰，主要带动大腿前后摆。
- `kn`：knee pitch，膝关节俯仰，主要带动小腿抬起或伸展。

Go2 关节名使用 Unitree 风格命名：

- `FL`、`FR`、`RL`、`RR`：前左、前右、后左、后右。
- `hip`：对应 Spot 的 `hx`。
- `thigh`：对应 Spot 的 `hy`。
- `calf`：对应 Spot 的 `kn`。

## 关节对应表

训练环境中的 12 维动作顺序与 `spot_scene.xml` 中腿部 actuator 顺序一致。

| 动作索引 | Spot 关节 | Go2 关节 | 部位 | 控制方向 |
| --- | --- | --- | --- | --- |
| 0 | `fl_hx` | `FL_hip_joint` | 前左髋横滚 / 左大腿外摆 | 正：向左；负：向右。 |
| 1 | `fl_hy` | `FL_thigh_joint` | 前左髋俯仰 / 左大腿前后摆 | 正：向后；负：向前。 |
| 2 | `fl_kn` | `FL_calf_joint` | 前左膝 / 左小腿 | 负：向前抬；正：向后或向下伸展。 |
| 3 | `fr_hx` | `FR_hip_joint` | 前右髋横滚 / 右大腿外摆 | 正：向左；负：向右。 |
| 4 | `fr_hy` | `FR_thigh_joint` | 前右髋俯仰 / 右大腿前后摆 | 正：向后；负：向前。 |
| 5 | `fr_kn` | `FR_calf_joint` | 前右膝 / 右小腿 | 负：向前抬；正：向后或向下伸展。 |
| 6 | `hl_hx` | `RL_hip_joint` | 后左髋横滚 / 左大腿外摆 | 正：向左；负：向右。 |
| 7 | `hl_hy` | `RL_thigh_joint` | 后左髋俯仰 / 左大腿前后摆 | 正：向后；负：向前。 |
| 8 | `hl_kn` | `RL_calf_joint` | 后左膝 / 左小腿 | 负：向前抬；正：向后或向下伸展。 |
| 9 | `hr_hx` | `RR_hip_joint` | 后右髋横滚 / 右大腿外摆 | 正：向左；负：向右。 |
| 10 | `hr_hy` | `RR_thigh_joint` | 后右髋俯仰 / 右大腿前后摆 | 正：向后；负：向前。 |
| 11 | `hr_kn` | `RR_calf_joint` | 后右膝 / 右小腿 | 负：向前抬；正：向后或向下伸展。 |

注意：你在 viewer 里看到的 `kn` “只能负，往前抬”，通常是因为当前站姿附近继续给正方向控制会让小腿往伸展方向走，容易接近地面或受关节范围限制。MuJoCo 的 `ctrlrange` 仍允许 `kn` 有正负控制值，但训练动作一般应避免让膝关节过度伸直。

## 控制语义

`spot_scene.xml` 的腿部 actuator 使用 affine PD 控制。MuJoCo 里的 `ctrl` 不是物理关节绝对角度，而是相对站姿的控制偏移。

```text
ctrl = nominal_leg_ctrl + action * action_scale
```

其中 `action` 来自 PPO policy，范围是 `[-1, 1]`。Go2-style 训练环境默认：

```text
nominal_leg_ctrl    = [0.0, -0.1, 0.3] repeated for each leg
action_scale        = 0.25
actuator_gain_scale = 3.0
reset height        = 1.72
target height       = 1.68
```

`reset height` 略高于 `target height`，是为了避免 reset 时脚部 geom 已经插进 heightfield。reward 仍按 `1.68` 作为目标身体高度。

## 范围说明

Spot 近似物理关节范围：

```text
hx: [-0.7854,  0.7854]
hy: [-0.8988,  2.24 to 2.30]
kn: [-2.7929, -0.25]
```

Spot 近似相对控制范围：

```text
hx: [-0.7854,  0.7854]
hy: [-1.9388,  1.20 to 1.26]
kn: [-0.9929,  1.54 to 1.55]
```

`kn` 的物理关节角仍然是负数范围，但 MuJoCo `ctrl` 可以是正数或负数，因为它是相对站姿 `-1.8` 的偏移。
