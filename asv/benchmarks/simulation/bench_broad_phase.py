# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark broad phase algorithms (NXN, SAP, BVH, Hash) with multi-world support.

Measures only the broad phase kernel(s) after AABBs have been filled.
Supports configurable per-world shape count, world count, and optional visualization.

Run from repo root:
  uv run python asv/benchmarks/simulation/bench_broad_phase.py
  uv run python asv/benchmarks/simulation/bench_broad_phase.py --num-shapes 2048 --num-worlds 4
  uv run python asv/benchmarks/simulation/bench_broad_phase.py --num-shapes 512 --num-worlds 2 --visualize
  uv run python asv/benchmarks/simulation/bench_broad_phase.py --num-shapes 512 --num-worlds 2 --visualize --viewer gl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import warp as wp

wp.config.quiet = True

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import newton


def _build_single_world(
    num_shapes: int,
    rng: np.random.Generator,
    radius: float = 0.25,
    spacing: float = 0.8,
) -> newton.ModelBuilder:
    """Build a sub-builder containing one world's worth of spheres on a 3-D grid."""
    sub = newton.ModelBuilder(gravity=0.0)
    grid_size = max(2, int(round(num_shapes ** (1.0 / 3.0))) + 1)

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if count >= num_shapes:
                    break
                pos = wp.vec3(
                    float((i - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2)),
                    float((j - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2)),
                    float((k - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2)),
                )
                body = sub.add_body(xform=wp.transform(pos))
                sub.add_shape_sphere(body, radius=radius)
                count += 1
            if count >= num_shapes:
                break
        if count >= num_shapes:
            break
    return sub


def build_model(
    num_shapes: int,
    num_worlds: int,
    device,
    seed: int = 42,
) -> newton.Model:
    """Build a model with *num_shapes* spheres per world across *num_worlds* worlds.

    Each world gets an independent cluster of spheres arranged in a 3-D grid
    with small random jitter.  Worlds are laid out in a 2-D grid via
    ``ModelBuilder.replicate()`` so they appear as a neat matrix in the viewer.

    Args:
        num_shapes: Number of sphere shapes per world.
        num_worlds: Number of simulation worlds.
        device: Warp device.
        seed: Random seed for reproducibility.
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    sub = _build_single_world(num_shapes, rng)

    grid_size = max(2, int(round(num_shapes ** (1.0 / 3.0))) + 1)
    cluster_span = grid_size * 0.8 + 2.0
    world_spacing = (cluster_span, cluster_span, 0.0)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(sub, num_worlds, spacing=world_spacing)

    return builder.finalize(device=device)


def _fill_aabbs(pipeline: newton.CollisionPipeline, state: newton.State) -> None:
    """Run one full collide to populate the AABB buffers in the pipeline."""
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)


def _launch_broad_phase(pipeline: newton.CollisionPipeline, model: newton.Model) -> None:
    """Execute only the broad phase stage (AABBs must already be filled)."""
    pipeline.broad_phase_pair_count.zero_()
    pipeline.broad_phase.launch(
        pipeline.narrow_phase.shape_aabb_lower,
        pipeline.narrow_phase.shape_aabb_upper,
        None,
        model.shape_collision_group,
        model.shape_world,
        model.shape_count,
        pipeline.broad_phase_shape_pairs,
        pipeline.broad_phase_pair_count,
        device=pipeline.device,
        filter_pairs=pipeline.shape_pairs_excluded,
        num_filter_pairs=pipeline.shape_pairs_excluded_count,
    )


def benchmark_broad_phase(
    num_shapes: int,
    num_worlds: int,
    warmup: int = 10,
    repeat: int = 50,
    number: int = 100,
    seed: int = 42,
) -> dict:
    """Run broad phase benchmarks for NXN, SAP, BVH, and Hash.

    Args:
        num_shapes: Number of shapes per world.
        num_worlds: Number of simulation worlds.
        warmup: Warmup iterations (discarded).
        repeat: Number of timing samples.
        number: Invocations per sample.
        seed: Random seed for scene layout.

    Returns:
        Tuple of (results dict, device string).
    """
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    model = build_model(num_shapes, num_worlds, device, seed=seed)
    state = model.state()
    device_info = str(device)

    pipelines: dict[str, newton.CollisionPipeline] = {}
    for mode in ("nxn", "sap", "bvh", "hash"):
        p = newton.CollisionPipeline(model, broad_phase=mode)
        _fill_aabbs(p, state)
        pipelines[mode] = p

    wp.synchronize_device()

    results: dict[str, dict] = {}
    for name, pipeline in pipelines.items():
        for _ in range(warmup):
            _launch_broad_phase(pipeline, model)
        wp.synchronize_device()

        samples = []
        for _ in range(repeat):
            wp.synchronize_device()
            t0 = time.perf_counter()
            for _ in range(number):
                _launch_broad_phase(pipeline, model)
            wp.synchronize_device()
            elapsed = (time.perf_counter() - t0) / number
            samples.append(elapsed)

        _launch_broad_phase(pipeline, model)
        wp.synchronize_device()
        pair_count = int(pipeline.broad_phase_pair_count.numpy()[0])

        results[name.upper()] = {
            "avg_ms": np.mean(samples) * 1000.0,
            "min_ms": np.min(samples) * 1000.0,
            "max_ms": np.max(samples) * 1000.0,
            "median_ms": np.median(samples) * 1000.0,
            "std_ms": np.std(samples) * 1000.0,
            "pairs": pair_count,
        }

    return results, device_info


def print_results(results: dict, num_shapes: int, num_worlds: int, device: str = "") -> None:
    """Pretty-print benchmark results."""
    total_shapes = num_shapes * num_worlds
    print(f"\n{'=' * 78}")
    print(f"  Broad Phase Benchmark — {num_shapes} shapes/world, {num_worlds} world(s) ({total_shapes} total)")
    if device:
        print(f"  Device: {device}")
    print(f"{'=' * 78}")
    header = f"{'Algorithm':<8} {'Avg (ms)':>10} {'Min (ms)':>10} {'Median':>10} {'Std':>10} {'Pairs':>8}"
    print(header)
    print("-" * 78)

    for name in ("NXN", "SAP", "BVH", "HASH"):
        r = results[name]
        print(
            f"{name:<8} {r['avg_ms']:>10.4f} {r['min_ms']:>10.4f} "
            f"{r['median_ms']:>10.4f} {r['std_ms']:>10.4f} {r['pairs']:>8}"
        )

    nxn_avg = results["NXN"]["avg_ms"]
    print(f"\n  Speedup vs NXN (by avg):")
    for name in ("SAP", "BVH", "HASH"):
        other_avg = results[name]["avg_ms"]
        speedup = nxn_avg / other_avg if other_avg > 0 else float("inf")
        print(f"    {name}: {speedup:.2f}x")
    print()


def run_visualized(num_shapes: int, num_worlds: int, viewer_type: str = "gl", seed: int = 42) -> None:
    """Run a visualized scene and benchmark all four broad phase algorithms.

    A single viewer window stays open.  The four algorithms (NXN, SAP, BVH, Hash)
    are each run for ``frames_per_algo`` frames in sequence; when all four are
    done the timing summary is printed and the window remains open until the
    user closes it.
    """
    import newton.viewer  # noqa: PLC0415

    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    model = build_model(num_shapes, num_worlds, device, seed=seed)
    state = model.state()

    if viewer_type == "gl":
        viewer = newton.viewer.ViewerGL()
    elif viewer_type == "rerun":
        viewer = newton.viewer.ViewerRerun()
    elif viewer_type == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        viewer = newton.viewer.ViewerNull(num_frames=900)

    viewer.set_model(model)

    modes = [("NXN", "nxn"), ("SAP", "sap"), ("BVH", "bvh"), ("HASH", "hash")]
    frames_per_algo = 200
    warmup_frames = 20

    pipelines: dict[str, newton.CollisionPipeline] = {}
    contacts_map: dict[str, newton.Contacts] = {}
    for _, mode in modes:
        p = newton.CollisionPipeline(model, broad_phase=mode)
        c = p.contacts()
        p.collide(state, c)
        pipelines[mode] = p
        contacts_map[mode] = c
    wp.synchronize_device()

    algo_idx = 0
    frame_in_algo = 0
    frame_global = 0
    algo_samples: dict[str, list[float]] = {m: [] for _, m in modes}

    label, mode = modes[algo_idx]
    total_shapes = num_shapes * num_worlds
    print(f"\n[Visualize] {num_shapes} shapes/world, {num_worlds} world(s) ({total_shapes} total)")
    print(f"  Now running: {label}  (frame 0/{frames_per_algo})")

    while viewer.is_running():
        pipeline = pipelines[mode]
        contacts = contacts_map[mode]

        wp.synchronize_device()
        t0 = time.perf_counter()
        _launch_broad_phase(pipeline, model)
        wp.synchronize_device()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if frame_in_algo >= warmup_frames:
            algo_samples[mode].append(elapsed_ms)

        viewer.begin_frame(frame_global / 60.0)
        viewer.log_state(state)
        viewer.log_contacts(contacts, state)
        viewer.end_frame()

        frame_in_algo += 1
        frame_global += 1

        if frame_in_algo >= frames_per_algo:
            pair_count = int(pipeline.broad_phase_pair_count.numpy()[0])
            samples = algo_samples[mode]
            if samples:
                print(
                    f"  {label}: avg={np.mean(samples):.4f} ms, "
                    f"median={np.median(samples):.4f} ms, "
                    f"min={np.min(samples):.4f} ms, pairs={pair_count}"
                )

            algo_idx += 1
            if algo_idx < len(modes):
                frame_in_algo = 0
                label, mode = modes[algo_idx]
                print(f"  Now running: {label}  (frame 0/{frames_per_algo})")
            else:
                print("\n  All algorithms done. Close the window to exit.")
                _print_visual_summary(algo_samples, pipelines, model, modes)

                while viewer.is_running():
                    viewer.begin_frame(frame_global / 60.0)
                    viewer.log_state(state)
                    viewer.end_frame()
                    frame_global += 1
                break

    viewer.close()


def _print_visual_summary(
    algo_samples: dict[str, list[float]],
    pipelines: dict[str, newton.CollisionPipeline],
    model: newton.Model,
    modes: list[tuple[str, str]],
) -> None:
    """Print a summary table after all visual benchmark phases complete."""
    print(f"\n{'=' * 60}")
    print(f"  {'Algorithm':<8} {'Avg (ms)':>10} {'Median':>10} {'Min (ms)':>10} {'Pairs':>8}")
    print(f"  {'-' * 54}")

    nxn_avg = 0.0
    for label, mode in modes:
        s = algo_samples[mode]
        if not s:
            continue
        avg = np.mean(s)
        if mode == "nxn":
            nxn_avg = avg
        _launch_broad_phase(pipelines[mode], model)
        wp.synchronize_device()
        pairs = int(pipelines[mode].broad_phase_pair_count.numpy()[0])
        print(f"  {label:<8} {avg:>10.4f} {np.median(s):>10.4f} {np.min(s):>10.4f} {pairs:>8}")

    if nxn_avg > 0:
        print()
        for label, mode in modes:
            if mode == "nxn":
                continue
            s = algo_samples[mode]
            if s:
                speedup = nxn_avg / np.mean(s) if np.mean(s) > 0 else float("inf")
                print(f"  Speedup {label} vs NXN: {speedup:.2f}x")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark broad phase algorithms (NXN, SAP, BVH, Hash).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-shapes", type=int, default=1024, help="Number of shapes per world.")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of simulation worlds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scene layout.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (discarded).")
    parser.add_argument("--repeat", type=int, default=50, help="Number of timing samples.")
    parser.add_argument("--number", type=int, default=100, help="Broad phase invocations per sample.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Open a viewer to visualize the scene (cycles through NXN, SAP, BVH, Hash).",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "rerun", "viser", "null"],
        help="Viewer type when --visualize is used.",
    )
    args = parser.parse_args()

    wp.init()

    if args.visualize:
        run_visualized(args.num_shapes, args.num_worlds, args.viewer, seed=args.seed)
    else:
        results, device_info = benchmark_broad_phase(
            args.num_shapes,
            args.num_worlds,
            warmup=args.warmup,
            repeat=args.repeat,
            number=args.number,
            seed=args.seed,
        )
        print_results(results, args.num_shapes, args.num_worlds, device_info)


if __name__ == "__main__":
    main()
