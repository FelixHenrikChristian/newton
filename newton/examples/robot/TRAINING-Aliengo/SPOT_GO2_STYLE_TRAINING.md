# Spot Go2-Style 行走训练说明

本文档记录如何用 MuJoCo + Stable-Baselines3 PPO 训练 Spot+机械臂行走策略，以及如何查看训练过程和播放训练结果。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `spot_scene.xml` | Spot+机械臂 MuJoCo 场景。 |
| `spot_go2_style_env.py` | 当前推荐的 Spot 行走 Gymnasium 环境。 |
| `train_spot_go2_style_ppo.py` | PPO 训练脚本，支持并行环境、VecNormalize 和 checkpoint。 |
| `debug_spot_go2_style_training_viewer.py` | 训练时打开 MuJoCo viewer，用于观察 rollout。 |
| `play_spot_go2_style_policy.py` | 播放训练好的 PPO policy。 |
| `SPOT_JOINTS.md` | Spot 和 Go2 关节对应关系、动作顺序和控制方向说明。 |

训练可视化和播放默认使用 `spot_scene.xml` 中的 `tracking_side_view` 相机。它挂在 Spot 根 body 下，会跟随机械狗移动。

## 环境准备

```powershell
cd D:\Learn\VisualStudioCode\source\repos\Code\newton\newton\examples\robot\TRAINING-Aliengo
conda activate newton
```

在训练目录运行脚本，避免 XML、模型和日志路径解析到其他位置。

## Smoke Test

```powershell
python train_spot_go2_style_ppo.py --total-timesteps 256 --num-envs 1 --n-steps 256 --vec-env dummy --run-name spot_go2_style_smoke --device cpu --no-domain-randomization --no-curriculum
```

这个命令只检查脚本、环境和依赖能否正常跑通。输出目录是 `runs\spot_go2_style_smoke\`，测试后可以删除。

主要参数：

- `--total-timesteps`：总训练步数。Smoke test 用很小的值即可。
- `--num-envs`：并行环境数量。调试时用 `1`。
- `--n-steps`：每次 PPO 更新前每个环境采样的步数。
- `--vec-env dummy`：使用单进程 vector env，最适合短测试。
- `--no-domain-randomization`、`--no-curriculum`：关闭随机化和课程学习，减少调试变量。

## 本地训练

```powershell
python train_spot_go2_style_ppo.py --total-timesteps 20000000 --num-envs 8 --vec-env subproc --device cpu --run-name spot_go2_style_20m
```

训练输出目录是 `runs\spot_go2_style_20m\`。最终模型保存为 `ppo_spot_go2_style_final.zip`，VecNormalize 统计保存为 `vecnormalize.pkl`，中间模型保存在 `checkpoints\`。

训练脚本还会定期用固定 `0.5 m/s` 前进命令做评估，并把评估分数最高的模型保存到：

```text
runs\spot_go2_style_20m\best_eval\best_model.zip
runs\spot_go2_style_20m\best_eval\best_vecnormalize.pkl
runs\spot_go2_style_20m\best_eval\best_score.txt
```

长时间训练时 final 不一定最好。如果后期策略退化，优先播放 `best_eval` 里的模型，或者播放中间 checkpoint。默认启用 early stop：达到最小训练步数后，如果连续多次固定速度评估没有刷新 best，就会提前停止，避免继续浪费时间。

主要参数：

- `--total-timesteps`：总训练步数。正式训练示例使用 `20000000`。
- `--num-envs`：并行环境数量。CPU 不够时可以降到 `4`。
- `--vec-env subproc`：多进程采样，正式训练通常比 `dummy` 快。
- `--device cpu`：在本地 CPU 上训练。
- `--run-name`：输出目录名。
- `--eval-freq`：每隔多少 env steps 做一次固定速度评估，默认 `500000`。
- `--early-stop-patience`：连续多少次评估没有明显提升就停止，默认 `8`。
- `--early-stop-min-timesteps`：至少训练多少步后才允许 early stop，默认 `5000000`。
- `--early-stop-min-delta`：评估分数至少提升多少才算刷新 best，默认 `0.05`。
- `--no-early-stop`：保留 best eval，但不自动停止训练。
- `--no-eval`：关闭 best eval 保存。

## TensorBoard

```powershell
tensorboard --logdir runs\spot_go2_style_20m\tensorboard
```

启动后在浏览器打开 TensorBoard 输出的本地地址，通常是 `http://localhost:6006`。

重点看这些曲线：

- `rollout/ep_rew_mean`：episode 平均奖励。
- `rollout/ep_len_mean`：episode 平均长度，越接近最大长度越不容易摔。
- `reward/lin`：前进速度跟踪奖励。
- `reward/orientation`、`reward/height`：姿态和高度是否稳定。
- `reward/torque`、`reward/smooth`：动作是否过猛。

## 训练可视化

```powershell
python debug_spot_go2_style_training_viewer.py --total-timesteps 20000 --n-steps 256 --batch-size 256 --device cpu --render-every 5 --sleep-scale 0
```

这个脚本会边训练边打开 MuJoCo viewer，适合观察关节是否在动、脚是否插地、身体是否趴下。它会显著拖慢训练，不建议用于正式长时间训练。

主要参数：

- `--render-every`：每隔多少个环境 step 渲染一次。值越小越流畅，也越慢。
- `--sleep-scale`：渲染时的等待倍率。`0` 表示尽量快跑。
- `--render-camera ""`：禁用默认跟随相机，使用 MuJoCo 自由相机。

## 播放训练结果

```powershell
python play_spot_go2_style_policy.py --model runs\spot_go2_style_20m\ppo_spot_go2_style_final.zip --seconds 60 --command-vx 0.3
```

播放脚本默认从模型所在目录加载 `vecnormalize.pkl`。如果统计文件不在同一目录，可以用 `--vecnormalize` 手动指定。

播放 best eval 模型：

```powershell
python play_spot_go2_style_policy.py --model runs\spot_go2_style_20m\best_eval\best_model.zip --seconds 60 --command-vx 0.5
```

播放某个中间 checkpoint（如果训练输出目录还没有清理 checkpoints）：

```powershell
python play_spot_go2_style_policy.py --model runs\spot_go2_style_20m\checkpoints\ppo_spot_go2_style_7000000_steps.zip --seconds 60 --command-vx 0.5
```

如果 checkpoint 目录里存在对应的 `ppo_spot_go2_style_vecnormalize_7000000_steps.pkl`，播放脚本会自动加载它。

主要参数：

- `--model`：要播放的 PPO 模型。
- `--seconds`：播放时长。
- `--command-vx`：前进目标速度。
- `--command-yaw`：偏航角速度目标，非零时会尝试转向。
- `--render-camera ""`：禁用默认跟随相机。

## 默认训练参数

`SpotGo2StyleEnv` 默认值：

```text
reset_base_height   = 1.72
target_base_height  = 1.68
nominal_leg_ctrl    = [0.0, -0.1, 0.3] repeated for each leg
action_scale        = 0.25
actuator_gain_scale = 3.0
control_decimation  = 10
command_vx          = 0.25 to 0.6
```

PPO 默认值：

```text
learning_rate = 3e-4
n_steps       = 2048
n_epochs      = 10
gamma         = 0.99
gae_lambda    = 0.95
ent_coef      = 0.005
net_arch      = [512, 256, 128]
```
