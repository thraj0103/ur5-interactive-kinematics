#!/usr/bin/env python3
"""
UR5 Interactive Kinematics Simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Dict
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# UR5 DH Parameters (Standard DH convention: [a, d, alpha])
DH_PARAMS = np.array([
    [0.0,      0.089159,  np.pi/2],   # Joint 1
    [-0.425,   0.0,       0.0],        # Joint 2
    [-0.39225, 0.0,       0.0],        # Joint 3
    [0.0,      0.10985,   np.pi/2],   # Joint 4
    [0.0,      0.09465,  -np.pi/2],   # Joint 5
    [0.0,      0.0823,    0.0]         # Joint 6
])

# Joint limits (radians)
JOINT_LIMITS = np.array([[-2*np.pi, 2*np.pi]] * 6)

# Home position
HOME_POSITION = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])

# Control parameters
INCREMENT_STEP = 0.01      # 10mm for translation
ROTATION_STEP = 0.1        # 0.1 rad for rotation

# Trajectory parameters
SQUARE_SIDE_LENGTH = 0.2   # 200mm
HELIX_RADIUS = 0.15        # 150mm
HELIX_HEIGHT = 0.3         # 300mm
HELIX_TURNS = 2
STEP_SIZE = 0.001          # 1mm

# Visualization settings
RGB_FRAME_SCALE = 0.08     # 80mm arrows
ROBOT_COLORS = ['#0e7490', '#0891b2', '#06b6d4', '#22d3ee', '#67e8f9', '#a5f3fc']  # Teal/cyan gradient
JOINT_COLOR = '#f59e0b'    # Amber/gold
PATH_COLOR = '#fbbf24'     # Yellow
GRID_ALPHA = 0.1

def pose_to_vector(pose: np.ndarray) -> np.ndarray:
    """Convert SE(3) pose to 6D vector [x, y, z, rx, ry, rz]."""
    position = pose[:3, 3]
    R = pose[:3, :3]
    
    # Rotation matrix to axis-angle
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    
    if theta < 1e-6:
        rotation = np.zeros(3)
    elif abs(theta - np.pi) < 1e-6:
        # 180 degree rotation
        diag = np.diag(R)
        k = np.argmax(diag)
        axis = np.sqrt((R[k, k] + 1) / 2)
        rotation = np.zeros(3)
        rotation[k] = axis * theta
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        rotation = axis * theta
    
    return np.concatenate([position, rotation])


def vector_to_pose(vector: np.ndarray) -> np.ndarray:
    """Convert 6D vector [x, y, z, rx, ry, rz] to SE(3) pose."""
    position = vector[:3]
    rotation_vec = vector[3:]
    theta = np.linalg.norm(rotation_vec)
    
    if theta < 1e-6:
        R = np.eye(3)
    else:
        axis = rotation_vec / theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    return pose


class MetricsCollector:
    """Collects and analyzes IK performance and singularity metrics."""
    
    def __init__(self):
        """Initialize metrics storage."""
        self.ik_metrics = []  # List of dicts with solve metrics
        self.singularity_metrics = []  # List of singularity analysis results
    
    def record_ik_solve(self, solve_time: float, iterations: int, 
                       success: bool, position_error: float, 
                       orientation_error: float, manipulability: float,
                       condition_number: float):
        """Record metrics from a single IK solve."""
        self.ik_metrics.append({
            'solve_time': solve_time,
            'iterations': iterations,
            'success': success,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'manipulability': manipulability,
            'condition_number': condition_number
        })
    
    def get_ik_statistics(self) -> Dict:
        """Compute statistics from collected IK metrics."""
        if not self.ik_metrics:
            return {}
        
        successful = [m for m in self.ik_metrics if m['success']]
        
        stats = {
            'total_solves': len(self.ik_metrics),
            'successful_solves': len(successful),
            'success_rate': len(successful) / len(self.ik_metrics) * 100,
        }
        
        if successful:
            solve_times = [m['solve_time'] for m in successful]
            iterations = [m['iterations'] for m in successful]
            pos_errors = [m['position_error'] for m in successful]
            ori_errors = [m['orientation_error'] for m in successful]
            manipulabilities = [m['manipulability'] for m in successful]
            cond_numbers = [m['condition_number'] for m in successful]
            
            stats.update({
                'avg_solve_time_ms': np.mean(solve_times) * 1000,
                'std_solve_time_ms': np.std(solve_times) * 1000,
                'avg_iterations': np.mean(iterations),
                'std_iterations': np.std(iterations),
                'avg_position_error_mm': np.mean(pos_errors) * 1000,
                'max_position_error_mm': np.max(pos_errors) * 1000,
                'avg_orientation_error_deg': np.degrees(np.mean(ori_errors)),
                'max_orientation_error_deg': np.degrees(np.max(ori_errors)),
                'avg_manipulability': np.mean(manipulabilities),
                'min_manipulability': np.min(manipulabilities),
                'avg_condition_number': np.mean(cond_numbers),
                'max_condition_number': np.max(cond_numbers),
            })
        
        return stats
    
    def print_report(self):
        """Print formatted metrics report."""
        stats = self.get_ik_statistics()
        
        if not stats:
            logger.warning("No metrics collected yet.")
            return
        
        logger.info("\n" + "="*60)
        logger.info("IK PERFORMANCE & SINGULARITY ANALYSIS REPORT")
        logger.info("="*60)
        
        logger.info(f"\n{'OVERALL PERFORMANCE':<30}")
        logger.info(f"{'─'*30}")
        logger.info(f"  Total IK Solves:              {stats['total_solves']}")
        logger.info(f"  Successful Solves:            {stats['successful_solves']}")
        logger.info(f"  Success Rate:                 {stats['success_rate']:.2f}%")
        
        if stats['successful_solves'] > 0:
            logger.info(f"\n{'SOLVE TIME':<30}")
            logger.info(f"{'─'*30}")
            logger.info(f"  Average:                      {stats['avg_solve_time_ms']:.2f} ms")
            logger.info(f"  Std Dev:                      {stats['std_solve_time_ms']:.2f} ms")
            
            logger.info(f"\n{'ITERATIONS':<30}")
            logger.info(f"{'─'*30}")
            logger.info(f"  Average:                      {stats['avg_iterations']:.1f}")
            logger.info(f"  Std Dev:                      {stats['std_iterations']:.1f}")
            
            logger.info(f"\n{'ACCURACY':<30}")
            logger.info(f"{'─'*30}")
            logger.info(f"  Avg Position Error:           {stats['avg_position_error_mm']:.4f} mm")
            logger.info(f"  Max Position Error:           {stats['max_position_error_mm']:.4f} mm")
            logger.info(f"  Avg Orientation Error:        {stats['avg_orientation_error_deg']:.4f}°")
            logger.info(f"  Max Orientation Error:        {stats['max_orientation_error_deg']:.4f}°")
            
            logger.info(f"\n{'SINGULARITY ANALYSIS':<30}")
            logger.info(f"{'─'*30}")
            logger.info(f"  Avg Manipulability Index:     {stats['avg_manipulability']:.6f}")
            logger.info(f"  Min Manipulability Index:     {stats['min_manipulability']:.6f}")
            logger.info(f"  Avg Condition Number:         {stats['avg_condition_number']:.2f}")
            logger.info(f"  Max Condition Number:         {stats['max_condition_number']:.2f}")
        
        logger.info("\n" + "="*60 + "\n")
    
    def export_to_file(self, filename: str = "metrics_report.txt"):
        """Export metrics report to text file."""
        with open(filename, 'w') as f:
            stats = self.get_ik_statistics()
            
            f.write("="*60 + "\n")
            f.write("UR5 IK PERFORMANCE & SINGULARITY ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"OVERALL PERFORMANCE\n")
            f.write(f"{'─'*30}\n")
            f.write(f"Total IK Solves:              {stats['total_solves']}\n")
            f.write(f"Successful Solves:            {stats['successful_solves']}\n")
            f.write(f"Success Rate:                 {stats['success_rate']:.2f}%\n\n")
            
            if stats['successful_solves'] > 0:
                f.write(f"SOLVE TIME\n")
                f.write(f"{'─'*30}\n")
                f.write(f"Average:                      {stats['avg_solve_time_ms']:.2f} ms\n")
                f.write(f"Std Dev:                      {stats['std_solve_time_ms']:.2f} ms\n\n")
                
                f.write(f"ITERATIONS\n")
                f.write(f"{'─'*30}\n")
                f.write(f"Average:                      {stats['avg_iterations']:.1f}\n")
                f.write(f"Std Dev:                      {stats['std_iterations']:.1f}\n\n")
                
                f.write(f"ACCURACY\n")
                f.write(f"{'─'*30}\n")
                f.write(f"Avg Position Error:           {stats['avg_position_error_mm']:.4f} mm\n")
                f.write(f"Max Position Error:           {stats['max_position_error_mm']:.4f} mm\n")
                f.write(f"Avg Orientation Error:        {stats['avg_orientation_error_deg']:.4f}°\n")
                f.write(f"Max Orientation Error:        {stats['max_orientation_error_deg']:.4f}°\n\n")
                
                f.write(f"SINGULARITY ANALYSIS\n")
                f.write(f"{'─'*30}\n")
                f.write(f"Avg Manipulability Index:     {stats['avg_manipulability']:.6f}\n")
                f.write(f"Min Manipulability Index:     {stats['min_manipulability']:.6f}\n")
                f.write(f"Avg Condition Number:         {stats['avg_condition_number']:.2f}\n")
                f.write(f"Max Condition Number:         {stats['max_condition_number']:.2f}\n\n")
            
            f.write("="*60 + "\n")
        
        logger.info("Metrics report exported to {filename}")

METRICS = MetricsCollector()

class UR5Robot:
    """Manages UR5 robot state and forward kinematics."""
    
    def __init__(self, initial_joints: np.ndarray = HOME_POSITION):
        """Initialize robot with joint configuration."""
        self.joint_angles = initial_joints.copy()
        self.dh_params = DH_PARAMS
        self.joint_limits = JOINT_LIMITS
    
    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set joint angles with limit enforcement."""
        self.joint_angles = np.clip(angles, 
                                    self.joint_limits[:, 0], 
                                    self.joint_limits[:, 1])
    
    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles."""
        return self.joint_angles.copy()
    
    def dh_transform(self, a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,   st*sa,   a*ct],
            [st,     ct*ca,  -ct*sa,   a*st],
            [0,      sa,      ca,      d],
            [0,      0,       0,       1]
        ])
    
    def forward_kinematics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics.
        
        Returns:
            joint_positions: (7, 3) array of joint positions
            end_effector_pose: 4x4 SE(3) transformation
        """
        joint_positions = np.zeros((7, 3))
        T = np.eye(4)
        
        for i in range(6):
            a, d, alpha = self.dh_params[i]
            theta = self.joint_angles[i]
            T = T @ self.dh_transform(a, d, alpha, theta)
            joint_positions[i+1] = T[:3, 3]
        
        return joint_positions, T
    
    def reset_to_home(self) -> None:
        """Reset robot to home position."""
        self.joint_angles = HOME_POSITION.copy()

class InverseKinematicsSolver:
    """Solves inverse kinematics using Newton-Raphson method."""
    
    def __init__(self, robot: UR5Robot):
        """Initialize solver with robot reference."""
        self.robot = robot
        self.epsilon = 1e-6  # For numerical differentiation
        self.damping = 0.01  # Damped least squares factor
    
    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute numerical Jacobian matrix."""
        # Save current state
        original_angles = self.robot.get_joint_angles()
        
        # Set to query configuration
        self.robot.set_joint_angles(joint_angles)
        _, current_pose = self.robot.forward_kinematics()
        current_vec = pose_to_vector(current_pose)
        
        J = np.zeros((6, 6))
        
        # Numerical differentiation for each joint
        for i in range(6):
            perturbed = joint_angles.copy()
            perturbed[i] += self.epsilon
            
            self.robot.set_joint_angles(perturbed)
            _, perturbed_pose = self.robot.forward_kinematics()
            perturbed_vec = pose_to_vector(perturbed_pose)
            
            J[:, i] = (perturbed_vec - current_vec) / self.epsilon
        
        # Restore original state
        self.robot.set_joint_angles(original_angles)
        
        return J
    
    def compute_manipulability(self, J: np.ndarray) -> float:
        """Compute manipulability index (Yoshikawa measure)."""
        return np.sqrt(np.linalg.det(J @ J.T))
    
    def compute_condition_number(self, J: np.ndarray) -> float:
        """Compute condition number of Jacobian."""
        singular_values = np.linalg.svd(J, compute_uv=False)
        return singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-10 else np.inf
    
    def solve(self, target_pose: np.ndarray, 
              initial_guess: np.ndarray,
              max_iter: int = 100,
              tolerance: float = 1e-4,
              collect_metrics: bool = True) -> Tuple[np.ndarray, bool]:
        """
        Solve IK using Newton-Raphson method.
        
        Args:
            target_pose: Desired 4x4 SE(3) pose
            initial_guess: Starting joint configuration
            max_iter: Maximum iterations
            tolerance: Convergence threshold
            collect_metrics: Whether to collect performance metrics
        
        Returns:
            joint_angles: Solution configuration
            success: True if converged
        """
        start_time = time.time()
        target_vec = pose_to_vector(target_pose)
        joint_angles = initial_guess.copy()
        
        # Save robot state
        original_angles = self.robot.get_joint_angles()
        
        iterations = 0
        for iterations in range(max_iter):
            # Compute current pose and error
            self.robot.set_joint_angles(joint_angles)
            _, current_pose = self.robot.forward_kinematics()
            current_vec = pose_to_vector(current_pose)
            error = target_vec - current_vec
            
            # Check convergence
            if np.linalg.norm(error) < tolerance:
                solve_time = time.time() - start_time
                
                # Collect metrics if requested
                if collect_metrics:
                    J = self.compute_jacobian(joint_angles)
                    manipulability = self.compute_manipulability(J)
                    condition_number = self.compute_condition_number(J)
                    
                    position_error = np.linalg.norm(error[:3])
                    orientation_error = np.linalg.norm(error[3:])
                    
                    METRICS.record_ik_solve(
                        solve_time=solve_time,
                        iterations=iterations + 1,
                        success=True,
                        position_error=position_error,
                        orientation_error=orientation_error,
                        manipulability=manipulability,
                        condition_number=condition_number
                    )
                
                self.robot.set_joint_angles(original_angles)
                return joint_angles, True
            
            # Compute Jacobian
            J = self.compute_jacobian(joint_angles)
            
            # Damped least squares update
            delta = np.linalg.solve(J.T @ J + self.damping * np.eye(6), J.T @ error)
            joint_angles += delta
            
            # Apply limits
            joint_angles = np.clip(joint_angles, 
                                  self.robot.joint_limits[:, 0],
                                  self.robot.joint_limits[:, 1])
        
        # Did not converge
        solve_time = time.time() - start_time
        
        if collect_metrics:
            METRICS.record_ik_solve(
                solve_time=solve_time,
                iterations=max_iter,
                success=False,
                position_error=np.linalg.norm(error[:3]),
                orientation_error=np.linalg.norm(error[3:]),
                manipulability=0.0,
                condition_number=np.inf
            )
        
        # Restore robot state
        self.robot.set_joint_angles(original_angles)
        return joint_angles, False

class TrajectoryGenerator:
    """Generates trajectories from current robot pose."""
    
    def __init__(self, robot: UR5Robot):
        """Initialize generator with robot reference."""
        self.robot = robot
    
    def generate_square(self, side_length: float = SQUARE_SIDE_LENGTH,
                       step_size: float = STEP_SIZE) -> List[np.ndarray]:
        """
        Generate square trajectory starting from current pose.
        
        Args:
            side_length: Side length in meters
            step_size: Distance between waypoints
        
        Returns:
            List of SE(3) poses
        """
        # Get current end-effector pose
        _, current_pose = self.robot.forward_kinematics()
        start_position = current_pose[:3, 3]
        orientation = current_pose[:3, :3]
        
        # Define square corners starting from current position (bottom-left)
        # Square goes: current -> right -> up -> left -> down (back to start)
        corners = [
            start_position,                                    # Start at current position
            start_position + np.array([0,  side_length, 0]),   # Move right
            start_position + np.array([0,  side_length, side_length]),  # Move up
            start_position + np.array([0, 0, side_length]),    # Move left
        ]
        
        poses = []
        num_points = int(side_length / step_size)
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            for j in range(num_points):
                t = j / num_points
                position = start + t * (end - start)
                
                pose = np.eye(4)
                pose[:3, :3] = orientation
                pose[:3, 3] = position
                poses.append(pose)
        
        return poses
    
    def generate_helix(self, radius: float = HELIX_RADIUS,
                      height: float = HELIX_HEIGHT,
                      turns: float = HELIX_TURNS,
                      step_size: float = STEP_SIZE) -> List[np.ndarray]:
        """
        Generate helical trajectory starting from current pose.
        
        Args:
            radius: Helix radius
            height: Total height
            turns: Number of rotations
            step_size: Arc length between waypoints
        
        Returns:
            List of SE(3) poses
        """
        # Get current pose
        _, current_pose = self.robot.forward_kinematics()
        start_position = current_pose[:3, 3]
        start_orientation = current_pose[:3, :3]
        
        # Calculate number of points
        pitch = height / turns
        arc_per_turn = 2 * np.pi * radius
        arc_length_per_turn = np.sqrt(arc_per_turn**2 + pitch**2)
        total_arc = arc_length_per_turn * turns
        num_points = int(total_arc / step_size)
        
        poses = []
        
        for i in range(num_points):
            t = i / num_points
            theta = 2 * np.pi * turns * t
            z_offset = height * t
            
            # Position relative to start
            x = start_position[0]
            y = start_position[1] + radius * (np.cos(theta) - 1)
            z = start_position[2] + z_offset
            position = np.array([x, y, z])
            
            # Compute target orientation (tangent to helix)
            tangent = np.array([0, -radius * np.sin(theta), height / (2 * np.pi * turns)])
            tangent = tangent / np.linalg.norm(tangent)
            
            z_axis = tangent
            x_axis = np.array([0, np.cos(theta), np.sin(theta)])
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            target_orientation = np.column_stack([x_axis, y_axis, z_axis])
            
            # Interpolate orientation from start to target (SLERP approximation)
            # For first 10% of trajectory, blend from current to tangent orientation
            if t < 0.1:
                blend = t / 0.1  # 0 to 1 over first 10%
                # Simple linear interpolation of rotation matrices (approximation)
                orientation = (1 - blend) * start_orientation + blend * target_orientation
                # Re-orthogonalize using SVD
                U, _, Vt = np.linalg.svd(orientation)
                orientation = U @ Vt
            else:
                orientation = target_orientation
            
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            poses.append(pose)
        
        return poses


class RobotVisualizer:
    """Handles 3D visualization and rendering."""
    
    def __init__(self):
        """Initialize visualizer with plot setup."""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.path_trail = []  # Unlimited path storage
        self.path_recording = False
        self._setup_plot()
    
    def _setup_plot(self) -> None:
        """Configure 3D plot aesthetics."""
        self.ax.set_xlim([-0.2, 0.8])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.8])
        
        self.ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
        self.ax.set_title('UR5 Interactive Kinematics Simulator', 
                         fontsize=15, fontweight='bold', pad=20)
        
        # Grid on floor
        self.ax.grid(True, alpha=GRID_ALPHA)
        self.ax.set_facecolor('#f8fafc')
    
    def draw_robot(self, joint_positions: np.ndarray) -> None:
        """Draw robot stick figure with gradient colors."""
        # Draw links with gradient
        for i in range(6):
            self.ax.plot([joint_positions[i, 0], joint_positions[i+1, 0]],
                        [joint_positions[i, 1], joint_positions[i+1, 1]],
                        [joint_positions[i, 2], joint_positions[i+1, 2]],
                        color=ROBOT_COLORS[i], linewidth=5, 
                        solid_capstyle='round')
        
        # Draw joints as spheres
        self.ax.scatter(joint_positions[1:, 0], 
                       joint_positions[1:, 1],
                       joint_positions[1:, 2],
                       color=JOINT_COLOR, s=100, edgecolors='black',
                       linewidths=1.5, zorder=5)
        
        # Highlight base
        self.ax.scatter([joint_positions[0, 0]], 
                       [joint_positions[0, 1]],
                       [joint_positions[0, 2]],
                       color='black', s=150, marker='s', 
                       edgecolors='white', linewidths=2, zorder=5)
        
        # Highlight end-effector
        self.ax.scatter([joint_positions[-1, 0]], 
                       [joint_positions[-1, 1]],
                       [joint_positions[-1, 2]],
                       color='#ec4899', s=120, marker='D',
                       edgecolors='black', linewidths=1.5, zorder=5)
    
    def draw_rgb_frame(self, pose: np.ndarray, scale: float = RGB_FRAME_SCALE) -> None:
        """Draw RGB coordinate frame at pose."""
        origin = pose[:3, 3]
        R = pose[:3, :3]
        
        # X-axis (Red)
        self.ax.quiver(origin[0], origin[1], origin[2],
                      R[0, 0], R[1, 0], R[2, 0],
                      color='#dc2626', length=scale, 
                      arrow_length_ratio=0.3, linewidth=3)
        
        # Y-axis (Green)
        self.ax.quiver(origin[0], origin[1], origin[2],
                      R[0, 1], R[1, 1], R[2, 1],
                      color='#16a34a', length=scale,
                      arrow_length_ratio=0.3, linewidth=3)
        
        # Z-axis (Blue)
        self.ax.quiver(origin[0], origin[1], origin[2],
                      R[0, 2], R[1, 2], R[2, 2],
                      color='#2563eb', length=scale,
                      arrow_length_ratio=0.3, linewidth=3)
    
    def draw_path_trail(self) -> None:
        """Draw accumulated path trail."""
        if len(self.path_trail) > 1:
            trail = np.array(self.path_trail)
            self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                        color=PATH_COLOR, linewidth=2.5, 
                        alpha=0.7, linestyle='-')
    
    def add_to_path(self, position: np.ndarray) -> None:
        """Add position to path trail if recording."""
        if self.path_recording:
            self.path_trail.append(position.copy())
    
    def clear_path(self) -> None:
        """Clear path trail."""
        self.path_trail = []
    
    def toggle_path_recording(self) -> bool:
        """Toggle path recording on/off."""
        self.path_recording = not self.path_recording
        return self.path_recording
    
    def update(self, joint_positions: np.ndarray, 
               end_effector_pose: np.ndarray,
               joint_angles: np.ndarray,
               control_mode: str = "Keyboard") -> None:
        """Update complete visualization."""
        self.ax.clear()
        self._setup_plot()
        
        # Draw components
        self.draw_robot(joint_positions)
        self.draw_rgb_frame(end_effector_pose)
        self.draw_path_trail()
        
        # Add info text
        pos = end_effector_pose[:3, 3]
        info_text = f'Mode: {control_mode}\n'
        info_text += f'Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\n'
        info_text += f'Path Recording: {"ON" if self.path_recording else "OFF"} | '
        info_text += f'Trail Points: {len(self.path_trail)}'
        
        self.ax.text2D(0.02, 0.98, info_text,
                      transform=self.ax.transAxes, fontsize=9,
                      verticalalignment='top', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.draw()


class InteractiveController:
    """Manages user interaction and control flow."""
    
    def __init__(self):
        """Initialize controller with all components."""
        self.robot = UR5Robot()
        self.ik_solver = InverseKinematicsSolver(self.robot)
        self.trajectory_gen = TrajectoryGenerator(self.robot)
        self.visualizer = RobotVisualizer()
        
        self.trajectory_mode = False
        self.trajectory_poses = []
        self.trajectory_index = 0
        self.control_mode = "Keyboard"  # Track current control mode
        
        # Connect keyboard events
        self.visualizer.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial visualization
        self._update_display()
    
    def _update_display(self) -> None:
        """Update visualization with current robot state."""
        joint_positions, ee_pose = self.robot.forward_kinematics()
        self.visualizer.add_to_path(ee_pose[:3, 3])
        self.visualizer.update(joint_positions, ee_pose, self.robot.get_joint_angles(), self.control_mode)
    
    def _move_incremental(self, delta_pose: np.ndarray) -> None:
        """Move robot incrementally in task space."""
        _, current_pose = self.robot.forward_kinematics()
        current_vec = pose_to_vector(current_pose)
        
        target_vec = current_vec + delta_pose
        target_pose = vector_to_pose(target_vec)
        
        new_joints, success = self.ik_solver.solve(target_pose, self.robot.get_joint_angles())
        
        if success:
            self.robot.set_joint_angles(new_joints)
            self._update_display()
            logger.info("✓ Moved successfully")
        else:
            logger.warning("✗ IK failed to converge")
    
    def _on_key_press(self, event) -> None:
        """Handle keyboard events."""
        delta = np.zeros(6)
        
        # Translation (Arrow keys)
        if event.key == 'up':
            delta[0] = INCREMENT_STEP  # +X
            logger.debug("Moving +X")
        elif event.key == 'down':
            delta[0] = -INCREMENT_STEP  # -X
            logger.debug("Moving -X")
        elif event.key == 'right':
            delta[1] = INCREMENT_STEP  # +Y
            logger.debug("Moving +Y")
        elif event.key == 'left':
            delta[1] = -INCREMENT_STEP  # -Y
            logger.debug("Moving -Y")
        elif event.key == '=' or event.key == '+':  # Mac: = key (no shift needed)
            delta[2] = INCREMENT_STEP  # +Z
            logger.debug("Moving +Z")
        elif event.key == '-' or event.key == '_':
            delta[2] = -INCREMENT_STEP  # -Z
            logger.debug("Moving -Z")
        
        # Rotation
        elif event.key == 'i':
            delta[3] = ROTATION_STEP
            logger.debug("Rotating +Roll")
        elif event.key == 'k':
            delta[3] = -ROTATION_STEP
            logger.debug("Rotating -Roll")
        elif event.key == 'j':
            delta[4] = ROTATION_STEP
            logger.debug("Rotating +Pitch")
        elif event.key == 'l':
            delta[4] = -ROTATION_STEP
            logger.debug("Rotating -Pitch")
        elif event.key == 'u':
            delta[5] = ROTATION_STEP
            logger.debug("Rotating +Yaw")
        elif event.key == 'o':
            delta[5] = -ROTATION_STEP
            logger.debug("Rotating -Yaw")
        
        # Trajectories
        elif event.key == '1':
            logger.info("Generating square trajectory from current pose...")
            self.trajectory_poses = self.trajectory_gen.generate_square()
            self.trajectory_index = 0
            self.trajectory_mode = True
            self.control_mode = "Square Trajectory"
            logger.info("Generated {len(self.trajectory_poses)} waypoints")
            self._update_display()
            return
        
        elif event.key == '2':
            logger.info("Generating helical trajectory from current pose...")
            self.trajectory_poses = self.trajectory_gen.generate_helix()
            self.trajectory_index = 0
            self.trajectory_mode = True
            self.control_mode = "Helix Trajectory"
            logger.info("Generated {len(self.trajectory_poses)} waypoints")
            self._update_display()
            return
        
        elif event.key == 'enter':
            if self.trajectory_mode and self.trajectory_index < len(self.trajectory_poses):
                target_pose = self.trajectory_poses[self.trajectory_index]
                new_joints, success = self.ik_solver.solve(target_pose, 
                                                           self.robot.get_joint_angles())
                
                if success:
                    self.robot.set_joint_angles(new_joints)
                    self.trajectory_index += 1
                    self._update_display()
                    logger.info("Waypoint {self.trajectory_index}/{len(self.trajectory_poses)}")
                    
                    if self.trajectory_index >= len(self.trajectory_poses):
                        print("✓ Trajectory complete!")
                        self.trajectory_mode = False
                        self.control_mode = "Keyboard"
                else:
                    logger.warning("✗ IK failed at waypoint {self.trajectory_index}")
            else:
                logger.warning("No trajectory loaded. Press '1' for square or '2' for helix")
            return
        
        # Path controls
        elif event.key == 'p':
            state = self.visualizer.toggle_path_recording()
            print(f"Path recording: {'ON' if state else 'OFF'}")
            return
        
        elif event.key == 'c':
            self.visualizer.clear_path()
            self._update_display()
            logger.info("Path cleared")
            return
        
        # Reset
        elif event.key == 'home' or event.key == 'h':
            logger.info("Resetting to home position...")
            self.robot.reset_to_home()
            self.control_mode = "Keyboard"
            self._update_display()
            return
        
        # Metrics
        elif event.key == 'm':
            print("\nGenerating metrics report...")
            METRICS.print_report()
            METRICS.export_to_file("ur5_metrics_report.txt")
            return
        
        elif event.key == '?':
            self._print_help()
            return
        
        else:
            return
        
        # Execute incremental movement
        if np.any(delta != 0):
            self._move_incremental(delta)
    
    def _print_help(self) -> None:
        """Print keyboard controls."""
        help_text = """
        ==========================================
        UR5 INTERACTIVE KINEMATICS - CONTROLS
        ==========================================
        
        TRANSLATION:
          ↑/↓      - Move ±X direction
          ←/→      - Move ±Y direction
          +/-      - Move ±Z direction
        
        ROTATION:
          I/K      - Rotate ±Roll (X-axis)
          J/L      - Rotate ±Pitch (Y-axis)
          U/O      - Rotate ±Yaw (Z-axis)
        
        TRAJECTORIES:
          1        - Generate square from current pose
          2        - Generate helix from current pose
          Enter    - Execute next waypoint
        
        PATH VISUALIZATION:
          P        - Toggle path recording ON/OFF
          C        - Clear path trail
        
        METRICS:
          M        - Print metrics report & export to file
        
        OTHER:
          Home/H   - Reset to home position
          ?        - Show this help
        
        ==========================================
        """
        logger.info(help_text)
    
    def run(self) -> None:
        """Start interactive visualization."""
        self._print_help()
        plt.show()


def main():
    """Initialize and run simulator."""
    logger.info("=" * 60)
    logger.info("UR5 INTERACTIVE KINEMATICS SIMULATOR")
    logger.info("Robot Mechanics and Control Course Project")
    logger.info("=" * 60)
    logger.info("")
    
    # Quick FK/IK test
    logger.info("Testing Forward Kinematics...")
    robot = UR5Robot()
    _, ee_pose = robot.forward_kinematics()
    logger.info(f"Home position end-effector: {ee_pose[:3, 3]}")
    logger.info("")
    
    logger.info("Testing Inverse Kinematics...")
    ik_solver = InverseKinematicsSolver(robot)
    test_joints, success = ik_solver.solve(ee_pose, np.zeros(6), collect_metrics=False)
    if success:
        logger.info("✓ IK converged successfully")
    else:
        logger.warning("✗ IK failed")
    logger.info("")
    
    # Start interactive controller
    logger.info("Starting interactive visualization...")
    logger.info("Press '?' for keyboard controls")
    logger.info("Press 'M' anytime to generate metrics report")
    logger.info("")
    
    controller = InteractiveController()
    controller.run()


if __name__ == "__main__":
    main()
