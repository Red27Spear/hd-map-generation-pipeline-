"""
Stage 3: Trajectory Interpolation
=================================
Interpolates 100Hz GNSS/INS trajectory to image timestamps using:
- Cubic spline for position (smooth, continuous velocity)
- SLERP (Spherical Linear Interpolation) for attitude quaternions

Output: Scanner pose (X, Y, Z, Roll, Pitch, Yaw) at each image timestamp.

Usage:
    python stage3_trajectory_interp.py --trajectory moro.track.txt --synced_times synced_times.json --output poses_at_images.json
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial.transform import Rotation as R
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class TrajectoryPoint:
    """Single trajectory epoch from GNSS/INS."""
    time: float          # GPS seconds-of-week
    x: float             # UTM Easting (m)
    y: float             # UTM Northing (m)  
    z: float             # Ellipsoidal height (m)
    roll: float          # Roll (degrees)
    pitch: float         # Pitch (degrees)
    heading: float       # Heading/yaw (degrees)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def attitude_rad(self) -> np.ndarray:
        """Attitude in radians [roll, pitch, heading]."""
        return np.radians([self.roll, self.pitch, self.heading])


@dataclass  
class InterpolatedPose:
    """Interpolated scanner pose at image timestamp."""
    time: float
    x: float
    y: float
    z: float
    roll: float      # degrees
    pitch: float     # degrees
    heading: float   # degrees

    # Full transform
    T_scanner_to_enu: Optional[np.ndarray] = None  # 4×4

    def to_dict(self) -> dict:
        result = {
            "time": self.time,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "attitude_deg": {"roll": self.roll, "pitch": self.pitch, "heading": self.heading},
            "attitude_rad": {"roll": np.radians(self.roll), 
                             "pitch": np.radians(self.pitch), 
                             "heading": np.radians(self.heading)}
        }
        if self.T_scanner_to_enu is not None:
            result["T_scanner_to_enu"] = self.T_scanner_to_enu.tolist()
        return result


class TrajectoryInterpolator:
    """
    Interpolates GNSS/INS trajectory using cubic splines and SLERP.

    The trajectory file (moro.track.txt) format:
        Time(sow)  Easting  Northing  Height  Roll(deg)  Pitch(deg)  Heading(deg)

    For each image timestamp, we need:
        T_scanner_to_enu = [R_scanner_to_enu | t_scanner_in_enu]

    where:
        t_scanner_in_enu = [Easting, Northing, Height]ᵀ
        R_scanner_to_enu = Rz(heading) @ Ry(pitch) @ Rx(roll)

    Convention: ZYX Euler angles (heading-yaw, pitch, roll)
        Roll: rotation about X (positive = right wing down)
        Pitch: rotation about Y (positive = nose up)
        Heading: rotation about Z (positive = clockwise from North)
    """

    def __init__(self, trajectory_points: List[TrajectoryPoint]):
        """
        Args:
            trajectory_points: List of trajectory epochs (should be sorted by time)
        """
        self.points = sorted(trajectory_points, key=lambda p: p.time)
        self.times = np.array([p.time for p in self.points])

        if len(self.times) < 4:
            raise ValueError(f"Need at least 4 trajectory points for cubic spline, got {len(self.times)}")

        # Position arrays
        self.positions = np.array([p.position for p in self.points])

        # Attitude (roll, pitch, heading) in radians
        self.attitudes_rad = np.array([p.attitude_rad for p in self.points])

        # Build interpolators
        self._build_position_interpolator()
        self._build_attitude_interpolator()

    def _build_position_interpolator(self):
        """Build cubic spline interpolator for 3D position."""
        # Cubic spline for each coordinate independently
        # This ensures C2 continuity (smooth acceleration)
        self.cs_x = CubicSpline(self.times, self.positions[:, 0])
        self.cs_y = CubicSpline(self.times, self.positions[:, 1])
        self.cs_z = CubicSpline(self.times, self.positions[:, 2])

    def _build_attitude_interpolator(self):
        """Build SLERP interpolator for attitude quaternions."""
        # Convert roll-pitch-heading → rotation matrices → quaternions
        # Convention: ZYX (heading = Z, pitch = Y, roll = X)
        quaternions = []
        for att in self.attitudes_rad:
            roll, pitch, heading = att
            # R = Rz(heading) @ Ry(pitch) @ Rx(roll)
            R_mat = R.from_euler("ZYX", [heading, pitch, roll], degrees=False)
            quaternions.append(R_mat.as_quat())  # [x, y, z, w]

        self.quaternions = np.array(quaternions)
        self.rotations = R.from_quat(self.quaternions)

        # Create SLERP interpolator
        # SLERP ensures constant angular velocity between keyframes
        self.slerp = Slerp(self.times, self.rotations)

    def interpolate_position(self, t: float) -> np.ndarray:
        """Interpolate position at time t using cubic spline."""
        return np.array([
            self.cs_x(t),
            self.cs_y(t),
            self.cs_z(t)
        ])

    def interpolate_attitude(self, t: float) -> R:
        """Interpolate attitude at time t using SLERP."""
        return self.slerp([t])[0]

    def interpolate(self, t: float) -> InterpolatedPose:
        """
        Full pose interpolation at time t.

        Returns scanner pose in ENU frame:
            Position: [Easting, Northing, Height]
            Attitude: [Roll, Pitch, Heading] in degrees
        """
        # Clamp to valid range with warning
        t_min, t_max = self.times[0], self.times[-1]
        if t < t_min or t > t_max:
            # Extrapolation warning - clamp for safety
            t = np.clip(t, t_min, t_max)

        # Position
        pos = self.interpolate_position(t)

        # Attitude
        rot = self.interpolate_attitude(t)

        # Convert back to Euler angles (ZYX convention)
        # Returns [heading, pitch, roll] for ZYX
        euler = rot.as_euler("ZYX", degrees=True)
        heading, pitch, roll = euler

        # Normalize heading to [0, 360)
        heading = heading % 360.0

        # Build 4×4 transform: T_scanner_to_enu
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = pos

        return InterpolatedPose(
            time=t,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            roll=roll,
            pitch=pitch,
            heading=heading,
            T_scanner_to_enu=T
        )

    def interpolate_batch(self, times: np.ndarray) -> List[InterpolatedPose]:
        """Interpolate poses for a batch of timestamps."""
        return [self.interpolate(t) for t in times]

    def get_velocity(self, t: float) -> np.ndarray:
        """Get instantaneous velocity at time t (m/s)."""
        return np.array([
            self.cs_x(t, 1),
            self.cs_y(t, 1),
            self.cs_z(t, 1)
        ])

    def get_acceleration(self, t: float) -> np.ndarray:
        """Get instantaneous acceleration at time t (m/s²)."""
        return np.array([
            self.cs_x(t, 2),
            self.cs_y(t, 2),
            self.cs_z(t, 2)
        ])


def load_trajectory(trajectory_path: str) -> List[TrajectoryPoint]:
    """
    Load trajectory from moro.track.txt format.

    Expected columns (space-separated):
        Time(sow)  Easting  Northing  Height  Roll(deg)  Pitch(deg)  Heading(deg)
    """
    points = []

    with open(trajectory_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                point = TrajectoryPoint(
                    time=float(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    z=float(parts[3]),
                    roll=float(parts[4]),
                    pitch=float(parts[5]),
                    heading=float(parts[6])
                )
                points.append(point)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num}: {line}")
                continue

    print(f"Loaded {len(points)} trajectory points")
    if points:
        duration = points[-1].time - points[0].time
        print(f"Time span: {duration:.3f} s ({duration/60:.2f} min)")
        print(f"Average rate: {len(points)/duration:.1f} Hz")

    return points


def load_synced_times(synced_json_path: str) -> List[float]:
    """Load image GPS seconds-of-week from Stage 2 output."""
    with open(synced_json_path, "r") as f:
        data = json.load(f)

    times = []
    for img in data.get("images", []):
        sow = img.get("gps_seconds_of_week")
        if sow is not None:
            times.append(sow)

    return sorted(times)


def main():
    parser = argparse.ArgumentParser(description="Trajectory Interpolation for Z+F 9020")
    parser.add_argument("--trajectory", required=True, help="Trajectory file (moro.track.txt)")
    parser.add_argument("--synced_times", required=True, help="Stage 2 output JSON")
    parser.add_argument("--output", default="./poses_at_images.json", help="Output JSON")
    parser.add_argument("--validate", action="store_true", help="Run validation plots")
    args = parser.parse_args()

    # Load data
    print("Loading trajectory...")
    traj_points = load_trajectory(args.trajectory)

    print("\nLoading image timestamps...")
    image_times = load_synced_times(args.synced_times)
    print(f"Images to interpolate: {len(image_times)}")

    # Build interpolator
    print("\nBuilding interpolator...")
    interp = TrajectoryInterpolator(traj_points)

    # Interpolate
    print("Interpolating poses...")
    poses = interp.interpolate_batch(np.array(image_times))

    # Save
    output_data = {
        "trajectory_file": args.trajectory,
        "synced_times_file": args.synced_times,
        "total_poses": len(poses),
        "interpolation_method": "cubic_spline_position_slerp_attitude",
        "attitude_convention": "ZYX_Euler_heading_pitch_roll",
        "poses": [p.to_dict() for p in poses]
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(poses)} interpolated poses to: {args.output}")

    # Validation
    if args.validate:
        print("\n=== Validation ===")

        # Check position continuity
        positions = np.array([[p.x, p.y, p.z] for p in poses])
        diffs = np.diff(positions, axis=0)
        step_sizes = np.linalg.norm(diffs, axis=1)
        print(f"Position step sizes: min={step_sizes.min():.4f}m, max={step_sizes.max():.4f}m, mean={step_sizes.mean():.4f}m")

        # Check velocity at trajectory points
        sample_times = np.linspace(traj_points[0].time, traj_points[-1].time, 100)
        velocities = np.array([interp.get_velocity(t) for t in sample_times])
        speeds = np.linalg.norm(velocities, axis=1)
        print(f"Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
        print(f"Mean speed: {speeds.mean():.2f} m/s ({speeds.mean()*3.6:.1f} km/h)")

        # Check time coverage
        traj_start = traj_points[0].time
        traj_end = traj_points[-1].time
        img_start = min(image_times)
        img_end = max(image_times)

        print(f"\nTrajectory coverage: {traj_start:.3f} - {traj_end:.3f} s")
        print(f"Image time range:    {img_start:.3f} - {img_end:.3f} s")

        if img_start < traj_start:
            print(f"WARNING: {traj_start - img_start:.3f}s of images before trajectory start")
        if img_end > traj_end:
            print(f"WARNING: {img_end - traj_end:.3f}s of images after trajectory end")


if __name__ == "__main__":
    main()
