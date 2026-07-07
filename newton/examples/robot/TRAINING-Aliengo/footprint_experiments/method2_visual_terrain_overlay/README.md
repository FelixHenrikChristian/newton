# Method 2: Visual Terrain Overlay Footprints

This directory contains method 2 of the lunar footprint experiments: a
visual-only terrain overlay for the Spot pick-and-place demo.

The experiment keeps the original demo files unchanged:

- `spot_pick_place_demo_robust_grasp_visual_terrain.py` adds the visual terrain
  overlay, footprint stamping, and renderer shadow tweaks.
- `visual_terrain.py` builds and updates the USD/GL-friendly visual terrain
  mesh.
- `spot_scene_visual_terrain.xml` keeps the experiment-specific lighting while
  loading mesh and texture assets from the base demo directory.

Run this variant from the repository root with:

```bash
uv run python newton/examples/robot/TRAINING-Aliengo/footprint_experiments/method2_visual_terrain_overlay/spot_pick_place_demo_robust_grasp_visual_terrain.py
```

It requires the same RL dependencies as the original demo, including
`gymnasium`, `stable-baselines3`, and `torch`.
