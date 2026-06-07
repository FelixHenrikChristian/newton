# Lunar MuJoCo Spot Arm Mining Scene

This package merges the uploaded Boston Dynamics Spot-with-arm MJCF model into the lunar mining scene.

## Main file

```bash
python -m mujoco.viewer --mjcf lunar_scene_spot_arm_mining.xml
```

## What is included

- High-resolution lunar heightfield terrain.
- Clear central crater.
- Dynamic free spherical gravel on the surface.
- Textured OBJ collectible ore cluster near `(10, 10)`.
- Spot robot with arm from `spot_arm.xml`.
- Spot mesh assets copied into `spot_assets/`.
- Spot position actuators copied into the scene.
- Keyframe `spot_mining_home` adjusted to this terrain.

## Robot placement

Spot is placed near the ore cluster but not on top of it:

- Spot body root initial `(x, y)`: (6.70, 7.10)
- Ore cluster center: `(10, 10)`
- Robot yaw: 41.31 degrees, facing the ore cluster
- Dynamic gravel removed around the robot start radius: 1.45 m
- Removed gravel bodies: 4

## Useful names

- Robot root body: `body`
- Robot freejoint: `freejoint`
- Keyframe: `spot_mining_home`
- Ore pickup sites: `ore_XX_pickup_site`
- Ore cluster marker: `ore_grasp_target_area`
- Camera: `spot_to_ore_view`
- Camera: `spot_arm_close_view`

## Notes

The scene uses lunar gravity `0 0 -1.62`. The Spot model was designed for Earth gravity, so stable locomotion and grasping may require controller tuning, actuator gains, and possibly a pre-stabilized pose before attempting ore pickup.

There are many dynamic gravel spheres. If simulation is slow, reduce the number of `gravel_free_*` bodies or pre-settle them and save a state.
