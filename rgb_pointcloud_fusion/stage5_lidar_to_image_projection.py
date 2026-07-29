#!/usr/bin/env python3
"""
Stage 5: LiDAR-to-Image Projection
Projects 3D world/UTM points into fisheye camera images using OpenCV fisheye model.

Coordinate chain:
    World/UTM Point ──► Scanner Frame ──► Camera Frame ──► Fisheye Image (u,v)

Usage:
    python stage5_lidar_to_image_projection.py \
        --registered_las registered.las \
        --poses poses_at_images.json \
        --calibration /data/Project/hd_map_p06/new_config/9020C_0140_toScanner_final.xml \
        --camera_id 0 \
        --image MoRo_Bonn{...}.jpg \
        --output projected.png
"""

import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional
import argparse
from pathlib import Path


@dataclass
class FisheyeCamera:
    """Fisheye camera intrinsics and extrinsics."""
    serial: str
    camera_id: int
    K: np.ndarray          # 3x3 intrinsic matrix
    D: np.ndarray          # 1x4 distortion coeffs [k1, k2, k3, k4]
    T_scanner_to_cam: np.ndarray  # 4x4 transform: scanner frame -> camera frame
    width: int = 2448
    height: int = 2048


def parse_calibration_xml(xml_path: str) -> List[FisheyeCamera]:
    """Parse Z+F 9020C calibration XML into FisheyeCamera objects.

    The real 9020C XML uses lowercase element names (fx, fy, cx, cy, k1-k4)
    and stores the camera name as the 'name' attribute ("left", "down", "right", "up").
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    name_to_id = {"left": 0, "down": 1, "right": 2, "up": 3}

    cameras = []
    for cam_elem in root.findall('Camera'):
        serial = cam_elem.get('serialno', cam_elem.get('SerialNr', ''))
        cam_name = cam_elem.get('name', '')
        cam_id = name_to_id.get(cam_name, len(cameras))

        # Intrinsics — real XML uses lowercase tags
        fx = float(cam_elem.findtext('fx', cam_elem.findtext('Fx', '1305.0')))
        fy = float(cam_elem.findtext('fy', cam_elem.findtext('Fy', '1305.0')))
        cx = float(cam_elem.findtext('cx', cam_elem.findtext('Cx', '1224.0')))
        cy = float(cam_elem.findtext('cy', cam_elem.findtext('Cy', '1024.0')))

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)

        # Distortion: Kannala-Brandt k1-k4 — real XML uses lowercase tags
        k1 = float(cam_elem.findtext('k1', cam_elem.findtext('K1', '0.0')))
        k2 = float(cam_elem.findtext('k2', cam_elem.findtext('K2', '0.0')))
        k3 = float(cam_elem.findtext('k3', cam_elem.findtext('K3', '0.0')))
        k4 = float(cam_elem.findtext('k4', cam_elem.findtext('K4', '0.0')))
        D = np.array([[k1, k2, k3, k4]], dtype=np.float64)

        # Extrinsics: scanner -> camera
        tx = float(cam_elem.findtext('R0_tx', '0.0'))
        ty = float(cam_elem.findtext('R0_ty', '0.0'))
        tz = float(cam_elem.findtext('R0_tz', '0.0'))
        roll = float(cam_elem.findtext('R0_roll', '0.0'))
        pitch = float(cam_elem.findtext('R0_pitch', '0.0'))
        yaw = float(cam_elem.findtext('R0_yaw', '0.0'))

        # Build T_scanner_to_cam from Euler angles (ZYX convention: yaw-pitch-roll)
        T = euler_to_matrix_ZYX(yaw, pitch, roll)
        T[:3, 3] = [tx, ty, tz]

        cameras.append(FisheyeCamera(
            serial=serial, camera_id=cam_id,
            K=K, D=D, T_scanner_to_cam=T
        ))

    return cameras


def euler_to_matrix_ZYX(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert ZYX Euler angles (radians) to 4x4 rotation matrix.

    Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Standard mobile mapping convention for ENU frame.
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx, 0],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx, 0],
        [-sy,   cy*sx,            cy*cx,            0],
        [0,     0,                0,                1]
    ], dtype=np.float64)

    return R


def load_poses(pose_json: str) -> dict:
    """Load scanner poses from Stage 3 output (poses_at_images.json).

    Stage 3 format:
      { "poses": [ { "time": float, "T_scanner_to_enu": [[...4x4...]] }, ... ] }

    Returns dict mapping GPS-SOW → 4x4 T_scanner_to_enu matrix.
    project_points_fisheye() receives this as T_world_to_scanner and immediately
    inverts it to obtain T_enu_to_scanner, which is the correct world→camera chain.
    """
    with open(pose_json, 'r') as f:
        data = json.load(f)

    poses = {}
    for entry in data.get("poses", []):
        time = float(entry['time'])
        T = np.array(entry['T_scanner_to_enu'], dtype=np.float64)
        poses[time] = T

    return poses


def project_points_fisheye(
    points_world: np.ndarray,
    T_world_to_scanner: np.ndarray,
    camera: FisheyeCamera
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D world points into fisheye image.

    Args:
        points_world: Nx3 array of points in UTM/ENU world frame
        T_world_to_scanner: 4x4 scanner pose in world frame
        camera: FisheyeCamera with intrinsics/extrinsics

    Returns:
        uv: Nx2 projected pixel coordinates (may include out-of-bounds)
        valid: N bool array, True if point is in front of camera and in image bounds
        depths: N array of Z-depths in camera frame (positive = in front)
    """
    N = points_world.shape[0]

    # World -> Scanner
    pts_world_h = np.hstack([points_world, np.ones((N, 1))])
    T_scanner_to_world = np.linalg.inv(T_world_to_scanner)
    pts_scanner_h = (T_scanner_to_world @ pts_world_h.T).T
    pts_scanner = pts_scanner_h[:, :3]

    # Scanner -> Camera
    T_cam_to_scanner = np.linalg.inv(camera.T_scanner_to_cam)
    pts_cam_h = (T_cam_to_scanner @ np.hstack([pts_scanner, np.ones((N, 1))]).T).T
    pts_cam = pts_cam_h[:, :3]

    # Check: Z > 0 (point in front of camera)
    depths = pts_cam[:, 2]
    in_front = depths > 0.01  # small epsilon

    # Fisheye projection using OpenCV
    # cv2.fisheye.projectPoints expects points in camera frame, outputs (u,v)
    pts_cam_reshaped = pts_cam.reshape(-1, 1, 3).astype(np.float64)

    # Dummy rotation/translation since points are already in camera frame
    rvec = np.zeros(3)
    tvec = np.zeros(3)

    uv, _ = cv2.fisheye.projectPoints(
        pts_cam_reshaped, rvec, tvec,
        camera.K, camera.D
    )
    uv = uv.reshape(-1, 2)

    # Check image bounds
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    )

    valid = in_front & in_bounds

    return uv, valid, depths


def project_las_to_image(
    las_path: str,
    pose_json: str,
    calib_xml: str,
    camera_id: int,
    image_time: float,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Main entry: project a registered LAS point cloud into a camera image.

    Returns Nx6 array: [u, v, X, Y, Z, intensity] for valid projected points.
    """
    # Load data
    try:
        import laspy
        las = laspy.read(las_path)
        points_world = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        intensity = las.intensity if hasattr(las, 'intensity') else np.zeros(len(points_world))
    except ImportError:
        # Fallback: read from numpy array or simple binary
        points_world = np.loadtxt(las_path.replace('.las', '.xyz'))
        intensity = np.zeros(len(points_world))

    cameras = parse_calibration_xml(calib_xml)
    camera = next(c for c in cameras if c.camera_id == camera_id)
    poses = load_poses(pose_json)

    # Get scanner pose at image time (interpolate if needed)
    times = sorted(poses.keys())
    if image_time in poses:
        T_world_to_scanner = poses[image_time]
    else:
        # Linear interpolation between nearest poses
        T_world_to_scanner = interpolate_pose(poses, times, image_time)

    # Project
    uv, valid, depths = project_points_fisheye(points_world, T_world_to_scanner, camera)

    # Collect valid projections
    result = np.column_stack([
        uv[valid],
        points_world[valid],
        intensity[valid],
        depths[valid]
    ])

    if output_path:
        np.save(output_path, result)
        print(f"Saved {len(result)} projected points to {output_path}")

    return result


def interpolate_pose(poses: dict, times: List[float], target_time: float) -> np.ndarray:
    """Linear interpolation between two poses for target time."""
    # Find bracketing times
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return poses[times[0]]
    if idx >= len(times):
        return poses[times[-1]]

    t0, t1 = times[idx-1], times[idx]
    alpha = (target_time - t0) / (t1 - t0)

    T0, T1 = poses[t0], poses[t1]

    # Interpolate translation linearly
    t_interp = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    # Interpolate rotation via SLERP
    R0, R1 = T0[:3, :3], T1[:3, :3]
    q0 = rotation_matrix_to_quaternion(R0)
    q1 = rotation_matrix_to_quaternion(R1)
    q_interp = slerp(q0, q1, alpha)
    R_interp = quaternion_to_rotation_matrix(q_interp)

    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp

    return T_interp


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def render_projection(
    image_path: str,
    projected_points: np.ndarray,
    output_path: str,
    point_size: int = 2,
    colormap: str = 'jet'
):
    """Render projected LiDAR points overlaid on camera image.

    projected_points: Nx6 [u, v, X, Y, Z, intensity, depth]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Color by depth (camera Z)
    depths = projected_points[:, 6]
    depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)

    # Apply colormap
    if colormap == 'jet':
        colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    elif colormap == 'intensity':
        intens = projected_points[:, 5]
        intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-6)
        colors = cv2.applyColorMap((intens_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    else:
        colors = np.full((len(projected_points), 3), (0, 255, 0), dtype=np.uint8)

    for i, (u, v) in enumerate(projected_points[:, :2].astype(int)):
        cv2.circle(img, (u, v), point_size, tuple(int(c) for c in colors[i][0]), -1)

    cv2.imwrite(output_path, img)
    print(f"Rendered projection saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 5: LiDAR-to-Image Projection')
    parser.add_argument('--registered_las', required=True, help='Registered point cloud (Stage 4 output)')
    parser.add_argument('--poses', required=True, help='Poses at image times (Stage 3 output)')
    parser.add_argument('--calibration', required=True, help='Calibration XML (Stage 1 output)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID (0=left, 1=down, 2=right, 3=up)')
    parser.add_argument('--image', required=True, help='Camera image file path')
    parser.add_argument('--image_time', type=float, help='GPS time of image (auto-detected from filename if omitted)')
    parser.add_argument('--output', default='projected_overlay.png', help='Output image path')
    parser.add_argument('--point_size', type=int, default=2)
    parser.add_argument('--colormap', default='jet', choices=['jet', 'intensity', 'green'])
    args = parser.parse_args()

    # Auto-detect image time from filename if not provided
    if args.image_time is None:
        args.image_time = extract_time_from_filename(args.image)

    # Project
    projected = project_las_to_image(
        args.registered_las, args.poses, args.calibration,
        args.camera_id, args.image_time
    )

    # Save raw Nx7 array as .npy so Stage 6 can consume it
    npy_path = args.output.replace('.png', '').replace('.jpg', '') + '.npy'
    import numpy as _np
    _np.save(npy_path, projected)
    print(f"Saved projected points to {npy_path}")

    # Render visual overlay
    render_projection(args.image, projected, args.output, args.point_size, args.colormap)

    print(f"\nProjection complete: {len(projected)} valid points")
    print(f"Depth range: [{projected[:, 6].min():.2f}, {projected[:, 6].max():.2f}] m")


def extract_time_from_filename(filename: str) -> float:
    """Extract GPS seconds-of-week from Z+F MoRo_Bonn filename.

    Delegates to Stage 2's TimeSynchronizer for accurate UTC→GPS SOW conversion.
    Fallback: return 0.0 (caller should supply --image_time explicitly).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from stage2_time_sync import TimeSynchronizer
        sync = TimeSynchronizer()
        utc_dt = sync.filename_to_utc(os.path.basename(filename))
        if utc_dt is None:
            return 0.0
        _, gps_sow, _ = sync.utc_to_gps(utc_dt)
        return gps_sow
    except Exception:
        return 0.0


if __name__ == '__main__':
    main()
