# Aliengo Z1 MuJoCo Scene

This folder contains a MuJoCo scene for staging a Unitree Aliengo quadruped
with a folded Unitree Z1 arm fixed to the robot trunk.

## Files

- `aliengo_scene.xml`: MuJoCo scene entry point.
- `aliengo_assets/*.obj`: Aliengo visual meshes converted from Unitree's
  official `aliengo_description` DAE meshes.
- `z1_assets/*.obj`: Z1 visual meshes converted from Unitree's official
  `z1_description` DAE meshes.
- `LICENSE.unitree_ros`: Upstream Unitree ROS license copied with the converted
  assets.

## Source

The Aliengo body, joint, inertial, collision, and mesh data are derived from:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/aliengo_description

The Z1 arm link, inertial, and mesh data are derived from:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/z1_description

The Z1 mounting offset follows Unitree's combined Aliengo-Z1 Xacro:

https://github.com/unitreerobotics/unitree_ros/tree/master/robots/aliengoZ1_description

MuJoCo does not decode the upstream DAE meshes directly in this environment, so
the DAE visual meshes were converted to OBJ while preserving the official URDF
collision primitives for contact.

The flat inspection scenes keep the Z1 modeled as a fixed folded payload
attached to Aliengo's `trunk`. The lunar IK scene exposes six Z1 hinge joints
and position actuators so scripts can solve and preview arm motion without
embedding trajectories in the XML. The Z1 has simplified collision primitives
that collide with the ground only; they do not collide with Aliengo or with
other Z1 links. This keeps the Aliengo locomotion action space unchanged while
accounting for the mounted arm mass and preventing the arm from passing through
the floor when the robot falls.

The Z1 gripper collision meshes are low-face-count simplifications of Unitree's
official gripper collision STL files so the rendered collision shape preserves
the jaw opening instead of using a filled axis-aligned box.

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
