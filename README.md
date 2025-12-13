# UR5 Interactive Kinematics Simulator

**Robot Mechanics and Control Course Project**

A 6-DOF UR5 robot kinematics simulator with analytical forward kinematics, Newton's method inverse kinematics, trajectory generation, and real-time 3D visualization.

## Video Demonstration

[![UR5 Kinematics Simulator Demo](https://img.youtube.com/vi/M5NVXSnEazY/0.jpg)](https://youtu.be/M5NVXSnEazY)

**Watch the video:** [https://youtu.be/M5NVXSnEazY](https://youtu.be/M5NVXSnEazY)

The video shows real-time execution of square and helical trajectories followed by interactive keyboard-driven manipulation.

---

## Project Overview

This project implements a complete kinematics simulator for the Universal Robots UR5 manipulator, meeting all course requirements:

- ✅ **6-DOF Robot:** UR5 collaborative robot
- ✅ **Analytical Forward Kinematics:** DH parameter-based transformations
- ✅ **Newton's Method IK:** Damped least squares solver
- ✅ **Trajectory Generation:** Square and helical paths with 1mm increments
- ✅ **3D Visualization:** Matplotlib stick figure with RGB coordinate frame
- ✅ **Interactive Control:** Keyboard-driven task-space movement
- ✅ **Single File Implementation:** No robotics packages used

---

## Running the Simulator

```bash
# Activate virtual environment
source venv/bin/activate

# Run simulator
python ur5_kinematics.py
```

---

## Keyboard Controls

**Translation (Task Space):**
- `↑` / `↓` - Move ±X direction
- `←` / `→` - Move ±Y direction  
- `+` / `-` - Move ±Z direction

**Rotation (Task Space):**
- `I` / `K` - Rotate ±Roll (X-axis)
- `J` / `L` - Rotate ±Pitch (Y-axis)
- `U` / `O` - Rotate ±Yaw (Z-axis)

**Trajectory Execution:**
- `1` - Generate square trajectory from current pose
- `2` - Generate helical trajectory from current pose
- `Enter` - Execute next waypoint

**Path Visualization:**
- `P` - Toggle path recording ON/OFF
- `C` - Clear path trail

**Other:**
- `Home` / `H` - Reset to home position
- `?` - Show help

---

## UR5 Robot Specifications

**Universal Robots UR5**
- 6-DOF collaborative robot (all revolute joints)
- Reach: 850mm
- Payload: 5kg

**DH Parameters:**

| Joint | a (m)      | d (m)     | α (rad)  |
|-------|------------|-----------|----------|
| 1     | 0          | 0.089159  | π/2      |
| 2     | -0.425     | 0         | 0        |
| 3     | -0.39225   | 0         | 0        |
| 4     | 0          | 0.10985   | π/2      |
| 5     | 0          | 0.09465   | -π/2     |
| 6     | 0          | 0.0823    | 0        |

*Source: Universal Robots official documentation*

---

## Implementation Details

### Forward Kinematics
- **Method:** Standard DH transformation matrices
- **Formula:** `T = Rot_z(θ) × Trans_z(d) × Trans_x(a) × Rot_x(α)`
- Computes all joint positions and end-effector SE(3) pose

### Inverse Kinematics
- **Algorithm:** Newton-Raphson with damped least squares
- **Jacobian:** Numerical differentiation
- **Convergence:** < 1e-4 tolerance, max 100 iterations
- **Damping factor:** λ = 0.01 for singularity avoidance

### Trajectories
- **Square:** 200mm × 200mm in YZ plane, 800 waypoints (1mm spacing)
- **Helix:** 150mm radius, 300mm height, 2 turns, ~942 waypoints (1mm spacing)
- Both trajectories start from current robot pose with smooth orientation interpolation

### Visualization
- 3D stick figure with gradient-colored links
- RGB coordinate frame (80mm arrows) showing end-effector SE(3) pose
- Real-time path trail visualization
- Control mode display (Keyboard/Square/Helix)

---

## Code Structure

All code in a single file `ur5_kinematics.py`:

**5 Modular Classes:**
1. `UR5Robot` - Robot state and forward kinematics
2. `InverseKinematicsSolver` - IK solving algorithms
3. `TrajectoryGenerator` - Path generation
4. `RobotVisualizer` - 3D rendering
5. `InteractiveController` - User interaction

---

## Dependencies

- Python 3.11+
- NumPy 2.3.5 (matrix operations)
- Matplotlib 3.10.8 (3D visualization)
