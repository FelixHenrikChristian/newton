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

"""Benchmark broad phase algorithms (NXN, SAP, BVH) and full collision pipeline.

Measures:
  - Broad phase only: time for the broad phase kernel(s) after AABBs are filled.
  - Overall: time for full model.collide() (AABB + broad phase + narrow phase).

Each benchmark method is first called once as warmup (result discarded), then called
``repeat`` times.  The **minimum** sample is reported (standard practice – least
affected by OS/GPU scheduling noise).  Each sample internally runs the routine
``number`` times then synchronises the GPU.  Divide by ``number`` for per-invocation
time.

Run from repo root:
  uv run python asv/benchmarks/simulation/bench_broad_phase.py
  uv run python asv/benchmarks/simulation/bench_broad_phase.py -b BenchBroadPhase
  uv run python asv/benchmarks/simulation/bench_broad_phase.py -b BenchBroadPhase --num-shapes 1024
"""

import os
import sys

import numpy as np
import warp as wp

wp.config.quiet = True

# Allow importing newton from repo root when running this file directly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from asv_runner.benchmarks.mark import skip_benchmark_if

import newton
from newton import BroadPhaseMode


def build_model_many_shapes(num_shapes: int, device, seed: int = 42):
    """Build a model with num_shapes bodies, each with one sphere (for broad phase benchmarking)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    builder = newton.ModelBuilder(gravity=0.0)
    # Spread spheres in a box to get a mix of overlapping and non-overlapping pairs
    grid_size = max(2, int(round(num_shapes ** (1.0 / 3.0))))
    spacing = 1.5
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if count >= num_shapes:
                    break
                # Slight random offset to avoid perfect grid (worse case for BVH)
                pos = np.array(
                    [
                        (i - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2),
                        (j - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2),
                        (k - grid_size / 2) * spacing + rng.uniform(-0.2, 0.2),
                    ],
                    dtype=np.float32,
                )
                body = builder.add_body(xform=wp.transform(wp.vec3(*pos)))
                builder.add_shape_sphere(body, radius=0.25)
                count += 1
            if count >= num_shapes:
                break
        if count >= num_shapes:
            break
    return builder.finalize(device=device)


class BenchBroadPhase:
    """Benchmark NXN / SAP / BVH: broad phase only and full collide."""

    params = ([256, 512, 1024, 2048, 4096],)
    param_names = ["num_shapes"]
    repeat = 50
    number = 1000

    def setup(self, num_shapes):
        wp.init()
        self.device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
        self.model = build_model_many_shapes(num_shapes, self.device)
        self.state = self.model.state()
        self.pipeline_nxn = newton.CollisionPipeline.from_model(
            self.model, broad_phase_mode=BroadPhaseMode.NXN
        )
        self.pipeline_sap = newton.CollisionPipeline.from_model(
            self.model, broad_phase_mode=BroadPhaseMode.SAP
        )
        self.pipeline_bvh = newton.CollisionPipeline.from_model(
            self.model, broad_phase_mode=BroadPhaseMode.BVH
        )
        # One full collide per pipeline to fill AABBs and warm up
        self.model.collide(self.state, collision_pipeline=self.pipeline_nxn)
        self.model.collide(self.state, collision_pipeline=self.pipeline_sap)
        self.model.collide(self.state, collision_pipeline=self.pipeline_bvh)
        wp.synchronize_device()

    def teardown(self, num_shapes):
        del self.pipeline_bvh
        del self.pipeline_sap
        del self.pipeline_nxn
        del self.state
        del self.model

    def _launch_broad_phase_only(self, pipeline):
        """Run only the broad phase launch (AABBs already filled in setup)."""
        pipeline.broad_phase_pair_count.zero_()
        if pipeline.nxn_broadphase is not None:
            pipeline.nxn_broadphase.launch(
                pipeline.shape_aabb_lower,
                pipeline.shape_aabb_upper,
                None,
                self.model.shape_collision_group,
                self.model.shape_world,
                self.model.shape_count,
                pipeline.broad_phase_shape_pairs,
                pipeline.broad_phase_pair_count,
                device=pipeline.device,
            )
        elif pipeline.sap_broadphase is not None:
            pipeline.sap_broadphase.launch(
                pipeline.shape_aabb_lower,
                pipeline.shape_aabb_upper,
                None,
                self.model.shape_collision_group,
                self.model.shape_world,
                self.model.shape_count,
                pipeline.broad_phase_shape_pairs,
                pipeline.broad_phase_pair_count,
                device=pipeline.device,
            )
        else:
            pipeline.bvh_broadphase.launch(
                pipeline.shape_aabb_lower,
                pipeline.shape_aabb_upper,
                None,
                self.model.shape_collision_group,
                self.model.shape_world,
                self.model.shape_count,
                pipeline.broad_phase_shape_pairs,
                pipeline.broad_phase_pair_count,
                device=pipeline.device,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_broad_phase_nxn(self, num_shapes):
        for _ in range(self.number):
            self._launch_broad_phase_only(self.pipeline_nxn)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_broad_phase_sap(self, num_shapes):
        for _ in range(self.number):
            self._launch_broad_phase_only(self.pipeline_sap)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_broad_phase_bvh(self, num_shapes):
        for _ in range(self.number):
            self._launch_broad_phase_only(self.pipeline_bvh)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide_nxn(self, num_shapes):
        for _ in range(self.number):
            self.model.collide(self.state, collision_pipeline=self.pipeline_nxn)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide_sap(self, num_shapes):
        for _ in range(self.number):
            self.model.collide(self.state, collision_pipeline=self.pipeline_sap)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide_bvh(self, num_shapes):
        for _ in range(self.number):
            self.model.collide(self.state, collision_pipeline=self.pipeline_bvh)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "BenchBroadPhase": BenchBroadPhase,
    }

    parser = argparse.ArgumentParser(
        description="Benchmark broad phase (NXN, SAP, BVH): broad phase only and full collide.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=list(benchmark_list.keys()),
        help="Run a single benchmark class.",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=None,
        help="Override param: run only with this many shapes (single run, no param sweep).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = list(benchmark_list.keys())
    else:
        benchmarks = args.bench

    if args.num_shapes is not None:
        # Single shape count: override params for this run
        for name in benchmarks:
            cls = benchmark_list[name]
            cls.params = ([args.num_shapes],)
            cls.param_names = ["num_shapes"]
        print(f"Running with num_shapes={args.num_shapes} only (param sweep disabled).")

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
