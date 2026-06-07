# Aliengo Z1 MuJoCo Scene

This folder contains a MuJoCo scene for staging a Unitree Aliengo quadruped
before adding a Unitree Z1 arm.

## Files

- `aliengo_scene.xml`: MuJoCo scene entry point.
- `aliengo_assets/*.obj`: Aliengo visual meshes converted from Unitree's
  official `aliengo_description` DAE meshes.
- `LICENSE.unitree_ros`: Upstream Unitree ROS license copied with the converted
  assets.

## Source

The Aliengo body, joint, inertial, collision, and mesh data are derived from:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/aliengo_description

MuJoCo does not decode the upstream DAE meshes directly in this environment, so
the DAE visual meshes were converted to OBJ while preserving the official URDF
collision primitives for contact.

## Usage

Load:

```bash
simulate aliengo_scene.xml
```

The model includes a `stand` keyframe with a bent-leg inspection pose and
matching position actuator controls. In Python:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("aliengo_scene.xml")
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, key)
data.ctrl[:] = model.key_ctrl[key]
```
