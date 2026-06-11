# Aliengo MuJoCo Locomotion Training

This folder now contains a minimal PPO training setup for the Aliengo + Z1
MuJoCo scene.

## Files

- `aliengo_env.py`: Gymnasium environment wrapping `aliengo_scene.xml`.
- `train_ppo.py`: PPO training entry point.
- `play_policy.py`: Load and replay a trained policy in the MuJoCo viewer.
- `smoke_test.py`: Short random-action check for model loading and stepping.
- `requirements.txt`: Python dependencies.

## Install

```powershell
python -m pip install -r requirements.txt
```

If you have a CUDA PyTorch install preference, install PyTorch first from the
official PyTorch command for your machine, then run the requirements command.

## Quick Check

```powershell
python smoke_test.py
```

This should print an observation shape of `(52,)`, a total reward, and the final
base velocity/height info.

## Train

```powershell
python train_ppo.py --total-timesteps 5000000 --num-envs 4
```

Outputs are written to:

```text
runs/aliengo_walk/
```

The script saves both the PPO model and `vecnormalize.pkl`. Keep both files;
the replay script needs the normalization statistics.

## Replay

```powershell
python play_policy.py --model runs/aliengo_walk/ppo_aliengo_final.zip --vecnormalize runs/aliengo_walk/vecnormalize.pkl
```

## Practical Training Notes

Start with short runs such as `--total-timesteps 100000` to confirm the loop is
healthy, then increase to several million steps. A useful curriculum is:

1. Train with smaller command ranges in `AliengoWalkEnv`.
2. Train mostly forward walking first.
3. Add sideways and yaw commands.
4. Increase terrain/randomization difficulty.

The current XML uses actuator bias terms so `ctrl=0` corresponds to the stand
keyframe. The policy action is therefore an offset around the stand posture,
clipped to each actuator's MJCF `ctrlrange`.
