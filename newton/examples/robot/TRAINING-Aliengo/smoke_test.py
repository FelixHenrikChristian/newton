from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aliengo_env import AliengoWalkEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short random-action environment check.")
    parser.add_argument("--xml", type=Path, default=Path("aliengo_scene.xml"))
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env = AliengoWalkEnv(xml_path=args.xml)
    obs, info = env.reset(seed=0, options={"command": np.array([0.4, 0.0, 0.0], dtype=np.float32)})
    print(f"reset obs_shape={obs.shape} info={info}")

    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step={step} terminated={terminated} truncated={truncated}")
            break

    print(f"final obs_shape={obs.shape}")
    print(f"total_reward={total_reward:.3f}")
    print(f"last_info={info}")
    env.close()


if __name__ == "__main__":
    main()
