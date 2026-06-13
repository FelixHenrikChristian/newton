from __future__ import annotations

import argparse
import io
import os
import pathlib
import time
import zipfile
from pathlib import Path
from typing import Any

import stable_baselines3.common.base_class as sb3_base_class
import stable_baselines3.common.save_util as sb3_save_util
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from spot_env import SpotWalkEnv


def patch_sb3_zip_loader() -> None:
    """Work around PyTorch 2.6 failing to load .pth files from ZipExtFile."""

    def load_from_zip_file(
        load_path: str | pathlib.Path | io.BufferedIOBase,
        load_data: bool = True,
        custom_objects: dict[str, Any] | None = None,
        device: th.device | str = "auto",
        verbose: int = 0,
        print_system_info: bool = False,
    ):
        file = sb3_save_util.open_path(load_path, "r", verbose=verbose, suffix="zip")
        device = get_device(device=device)

        try:
            with zipfile.ZipFile(file) as archive:
                namelist = archive.namelist()
                data = None
                pytorch_variables = None
                params = {}

                if print_system_info and "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())

                if "data" in namelist and load_data:
                    json_data = archive.read("data").decode()
                    data = sb3_save_util.json_to_data(json_data, custom_objects=custom_objects)

                pth_files = [
                    file_name
                    for file_name in namelist
                    if os.path.splitext(file_name)[1] == ".pth"
                ]
                for file_path in pth_files:
                    param_bytes = io.BytesIO(archive.read(file_path))
                    th_object = th.load(param_bytes, map_location=device, weights_only=True)
                    if file_path in ("pytorch_variables.pth", "tensors.pth"):
                        pytorch_variables = th_object
                    else:
                        params[os.path.splitext(file_path)[0]] = th_object
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from exc
        finally:
            if isinstance(load_path, (str, pathlib.Path)):
                file.close()

        return data, params, pytorch_variables

    sb3_save_util.load_from_zip_file = load_from_zip_file
    sb3_base_class.load_from_zip_file = load_from_zip_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a trained Spot PPO policy.")
    parser.add_argument("--xml", type=Path, default=Path("spot_scene.xml"))
    parser.add_argument("--model", type=Path, default=Path("runs/spot_walk/ppo_spot_final.zip"))
    parser.add_argument("--vecnormalize", type=Path, default=Path("runs/spot_walk/vecnormalize.pkl"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--reset-base-height", type=float, default=1.68)
    parser.add_argument("--command-vx", type=float, default=0.3)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    args = parser.parse_args()

    patch_sb3_zip_loader()

    command_range = (
        (args.command_vx, args.command_vx),
        (args.command_vy, args.command_vy),
        (args.command_yaw, args.command_yaw),
    )
    raw_env = SpotWalkEnv(
        xml_path=args.xml,
        command_range=command_range,
        reset_base_height=args.reset_base_height,
        render_mode="human",
    )
    vec_env = DummyVecEnv([lambda: raw_env])
    if args.vecnormalize.exists():
        vec_env = VecNormalize.load(args.vecnormalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)
    obs = vec_env.reset()

    end_time = time.time() + args.seconds
    while time.time() < end_time:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
        raw_env.render()
        if done[0]:
            obs = vec_env.reset()
        time.sleep(raw_env.dt)

    vec_env.close()


if __name__ == "__main__":
    main()
