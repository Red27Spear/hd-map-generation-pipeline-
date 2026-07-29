#!/usr/bin/env python3
"""
Stage 8: Multi-Camera Selection & Blending
Selects the best camera for each point from 4 fisheye cameras and blends
colors from multiple views for seamless texturing.

Selection strategies:
1. Closest to image center (minimizes radial distortion)
2. Maximum viewing angle (most frontal view)
3. Minimum distance to camera (highest resolution)
4. Best color consistency (lowest variance across views)

Usage:
    python stage8_multi_camera_blend.py \
        --camera_outputs cam0_colored.npy cam1_colored.npy cam2_colored.npy cam3_colored.npy \
        --world_points registered.las \
        --strategy center \
        --output final_colored.las
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import argparse
from dataclasses import dataclass
from enum import Enum
from scipy.spatial import cKDTree
import json


class SelectionStrategy(Enum):
    CENTER = "center"           # Closest to image center (default, best for fisheye)
    FRONTAL = "frontal"         # Most frontal viewing angle
    CLOSEST = "closest"         # Minimum distance to camera
    CONSISTENCY = "consistency" # Best multi-view color consistency
    BLEND = "blend"             # Weighted blend of all views


@dataclass
class CameraView:
    """Data for one camera's view of the point cloud."""
    camera_id: int
    serial: str
    colored_points: np.ndarray  # Nx10 [u, v, X, Y, Z, intensity, depth, B, G, R]
    camera_center: np.ndarray   # 3D camera position in world frame
    camera_normal: np.ndarray   # Camera look direction in world frame


@dataclass
class BlendConfig:
    """Configuration for multi-camera blending."""
    strategy: SelectionStrategy = SelectionStrategy.CENTER

    # Center strategy: penalize distance from image center
    center_sigma: float = 500.0  # pixels: controls falloff from center

    # Frontal strategy: penalize angle from surface normal
    frontal_threshold: float = np.radians(75)  # max angle to consider

    # Closest strategy: prefer nearby cameras
    closest_sigma: float = 10.0  # meters

    # Consistency strategy: outlier rejection
    consistency_zscore: float = 2.0

    # Blend strategy: weights
    blend_center_weight: float = 0.4
    blend_frontal_weight: float = 0.3
    blend_distance_weight: float = 0.3

    # Seam smoothing
    seam_blend_radius: int = 5  # pixels: blend across camera boundaries


def compute_image_center_distance(
    colored_points: np.ndarray,
    image_width: int = 2448,
    image_height: int = 2048
) -> np.ndarray:
    """Compute normalized distance from image center for each point.

    Returns: N array of distances (0 = center, 1 = corner)
    """
    cx, cy = image_width / 2, image_height / 2
    u = colored_points[:, 0]
    v = colored_points[:, 1]

    dx = (u - cx) / cx
    dy = (v - cy) / cy

    # Euclidean distance normalized to [0, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return np.clip(dist, 0, 1)


def compute_viewing_angle(
    point_world: np.ndarray,
    camera_center: np.ndarray,
    camera_normal: np.ndarray
) -> float:
    """Compute angle between camera view direction and point direction.

    Returns: angle in radians (0 = most frontal, pi/2 = edge-on)
    """
    view_dir = point_world - camera_center
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)

    # Angle between camera normal and view direction
    cos_angle = np.dot(camera_normal, view_dir)
    cos_angle = np.clip(cos_angle, -1, 1)

    angle = np.arccos(cos_angle)
    return angle


def select_best_camera_per_point(
    views: List[CameraView],
    config: BlendConfig,
    image_width: int = 2448,
    image_height: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """Select best camera for each 3D point.

    Args:
        views: List of CameraView objects (one per camera)
        config: BlendConfig

    Returns:
        best_colors: Mx10 array of final colored points
        camera_labels: M array of selected camera_id per point
    """
    # Build KD-tree from first view's 3D points for spatial lookup
    # All views should cover the same points (just from different angles)

    # Strategy-specific selection
    if config.strategy == SelectionStrategy.CENTER:
        return _select_by_center(views, config, image_width, image_height)
    elif config.strategy == SelectionStrategy.FRONTAL:
        return _select_by_frontal(views, config)
    elif config.strategy == SelectionStrategy.CLOSEST:
        return _select_by_closest(views, config)
    elif config.strategy == SelectionStrategy.CONSISTENCY:
        return _select_by_consistency(views, config)
    elif config.strategy == SelectionStrategy.BLEND:
        return _blend_all_views(views, config, image_width, image_height)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


def _select_by_center(
    views: List[CameraView],
    config: BlendConfig,
    image_width: int,
    image_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Select camera where point is closest to image center.

    This is the default and recommended strategy for fisheye cameras
    because radial distortion is minimized at the center.
    """
    all_points = []
    all_scores = []
    all_camera_ids = []

    for view in views:
        pts = view.colored_points
        center_dist = compute_image_center_distance(pts, image_width, image_height)

        # Score: higher = better (1 at center, 0 at edge)
        score = np.exp(-center_dist**2 / (2 * (config.center_sigma / image_width)**2))

        all_points.append(pts)
        all_scores.append(score)
        all_camera_ids.append(np.full(len(pts), view.camera_id))

    # For each unique 3D point, select view with highest score
    # Build spatial index
    xyz_all = np.vstack([v.colored_points[:, 2:5] for v in views])
    tree = cKDTree(xyz_all)

    # Use first view as reference for unique points
    ref_pts = views[0].colored_points
    ref_xyz = ref_pts[:, 2:5]

    best_colors = []
    best_cameras = []

    for i, xyz in enumerate(ref_xyz):
        # Find this point in all views
        scores_for_point = []
        colors_for_point = []

        for j, view in enumerate(views):
            # Find nearest point in this view
            dist, idx = tree.query(xyz, k=1, distance_upper_bound=0.01)
            if dist < 0.01:
                # Point found in this view
                offset = sum(len(v.colored_points) for v in views[:j])
                actual_idx = idx - offset if j > 0 else idx
                if 0 <= actual_idx < len(view.colored_points):
                    scores_for_point.append(all_scores[j][actual_idx])
                    colors_for_point.append(view.colored_points[actual_idx])
                else:
                    scores_for_point.append(-1)
                    colors_for_point.append(None)
            else:
                scores_for_point.append(-1)
                colors_for_point.append(None)

        # Select best
        best_idx = np.argmax(scores_for_point)
        if scores_for_point[best_idx] > 0:
            best_colors.append(colors_for_point[best_idx])
            best_cameras.append(views[best_idx].camera_id)

    return np.array(best_colors), np.array(best_cameras)


def _select_by_frontal(
    views: List[CameraView],
    config: BlendConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Select camera with most frontal view of each point.

    Requires surface normals. If not available, falls back to center strategy.
    """
    # Simplified: use viewing angle from camera to point
    # A point is "frontal" when the camera looks directly at it

    all_points = []
    all_scores = []

    for view in views:
        pts = view.colored_points
        xyz = pts[:, 2:5]

        # Compute viewing angle for each point
        angles = np.array([
            compute_viewing_angle(xyz[i], view.camera_center, view.camera_normal)
            for i in range(len(xyz))
        ])

        # Score: 1 at 0 degrees, 0 at threshold
        score = np.maximum(0, 1 - angles / config.frontal_threshold)

        all_points.append(pts)
        all_scores.append(score)

    # Similar selection logic as center strategy
    xyz_all = np.vstack([v.colored_points[:, 2:5] for v in views])
    tree = cKDTree(xyz_all)

    ref_pts = views[0].colored_points
    best_colors = []
    best_cameras = []

    for i, xyz in enumerate(ref_pts[:, 2:5]):
        scores_for_point = []
        colors_for_point = []

        for j, view in enumerate(views):
            dist, idx = tree.query(xyz, k=1, distance_upper_bound=0.01)
            if dist < 0.01:
                offset = sum(len(v.colored_points) for v in views[:j])
                actual_idx = idx - offset if j > 0 else idx
                if 0 <= actual_idx < len(view.colored_points):
                    scores_for_point.append(all_scores[j][actual_idx])
                    colors_for_point.append(view.colored_points[actual_idx])
                else:
                    scores_for_point.append(-1)
                    colors_for_point.append(None)
            else:
                scores_for_point.append(-1)
                colors_for_point.append(None)

        best_idx = np.argmax(scores_for_point)
        if scores_for_point[best_idx] > 0:
            best_colors.append(colors_for_point[best_idx])
            best_cameras.append(views[best_idx].camera_id)

    return np.array(best_colors), np.array(best_cameras)


def _select_by_closest(
    views: List[CameraView],
    config: BlendConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Select closest camera to each point."""
    all_points = []
    all_scores = []

    for view in views:
        pts = view.colored_points
        xyz = pts[:, 2:5]
        distances = np.linalg.norm(xyz - view.camera_center, axis=1)

        # Score: higher = closer
        score = np.exp(-distances / config.closest_sigma)

        all_points.append(pts)
        all_scores.append(score)

    xyz_all = np.vstack([v.colored_points[:, 2:5] for v in views])
    tree = cKDTree(xyz_all)

    ref_pts = views[0].colored_points
    best_colors = []
    best_cameras = []

    for i, xyz in enumerate(ref_pts[:, 2:5]):
        scores_for_point = []
        colors_for_point = []

        for j, view in enumerate(views):
            dist, idx = tree.query(xyz, k=1, distance_upper_bound=0.01)
            if dist < 0.01:
                offset = sum(len(v.colored_points) for v in views[:j])
                actual_idx = idx - offset if j > 0 else idx
                if 0 <= actual_idx < len(view.colored_points):
                    scores_for_point.append(all_scores[j][actual_idx])
                    colors_for_point.append(view.colored_points[actual_idx])
                else:
                    scores_for_point.append(-1)
                    colors_for_point.append(None)
            else:
                scores_for_point.append(-1)
                colors_for_point.append(None)

        best_idx = np.argmax(scores_for_point)
        if scores_for_point[best_idx] > 0:
            best_colors.append(colors_for_point[best_idx])
            best_cameras.append(views[best_idx].camera_id)

    return np.array(best_colors), np.array(best_cameras)


def _select_by_consistency(
    views: List[CameraView],
    config: BlendConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Select camera with best color consistency across views.

    For each point, compute color variance across all views that see it.
    Select the view whose color is closest to the median.
    """
    xyz_all = np.vstack([v.colored_points[:, 2:5] for v in views])
    colors_all = np.vstack([v.colored_points[:, 7:10] for v in views])
    tree = cKDTree(xyz_all)

    ref_pts = views[0].colored_points
    best_colors = []
    best_cameras = []

    for i, xyz in enumerate(ref_pts[:, 2:5]):
        # Find all views of this point
        colors_for_point = []
        camera_ids_for_point = []

        for j, view in enumerate(views):
            dist, idx = tree.query(xyz, k=1, distance_upper_bound=0.01)
            if dist < 0.01:
                offset = sum(len(v.colored_points) for v in views[:j])
                actual_idx = idx - offset if j > 0 else idx
                if 0 <= actual_idx < len(view.colored_points):
                    colors_for_point.append(view.colored_points[actual_idx, 7:10])
                    camera_ids_for_point.append(view.camera_id)

        if len(colors_for_point) < 2:
            # Only one view, use it
            if colors_for_point:
                best_colors.append(views[camera_ids_for_point[0]].colored_points[
                    np.where(views[camera_ids_for_point[0]].colored_points[:, 2:5] == xyz)[0][0]
                ])
                best_cameras.append(camera_ids_for_point[0])
            continue

        colors_for_point = np.array(colors_for_point)
        median_color = np.median(colors_for_point, axis=0)

        # Compute distance to median for each view
        distances = np.linalg.norm(colors_for_point - median_color, axis=1)
        best_view_idx = np.argmin(distances)

        # Find the full point record
        best_cam_id = camera_ids_for_point[best_view_idx]
        view = next(v for v in views if v.camera_id == best_cam_id)

        # Find index in that view
        dist, idx = cKDTree(view.colored_points[:, 2:5]).query(xyz, k=1)
        if dist < 0.01:
            best_colors.append(view.colored_points[idx])
            best_cameras.append(best_cam_id)

    return np.array(best_colors), np.array(best_cameras)


def _blend_all_views(
    views: List[CameraView],
    config: BlendConfig,
    image_width: int,
    image_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted blend of colors from all cameras.

    Combines center proximity, frontal angle, and distance weights.
    """
    xyz_all = np.vstack([v.colored_points[:, 2:5] for v in views])
    tree = cKDTree(xyz_all)

    ref_pts = views[0].colored_points
    blended_colors = []
    best_cameras = []

    for i, xyz in enumerate(ref_pts[:, 2:5]):
        colors_for_point = []
        weights = []
        camera_ids_for_point = []

        for j, view in enumerate(views):
            dist, idx = tree.query(xyz, k=1, distance_upper_bound=0.01)
            if dist < 0.01:
                offset = sum(len(v.colored_points) for v in views[:j])
                actual_idx = idx - offset if j > 0 else idx
                if 0 <= actual_idx < len(view.colored_points):
                    pt = view.colored_points[actual_idx]
                    colors_for_point.append(pt[7:10])
                    camera_ids_for_point.append(view.camera_id)

                    # Compute weights
                    center_dist = compute_image_center_distance(
                        pt.reshape(1, -1), image_width, image_height
                    )[0]
                    w_center = np.exp(-center_dist**2 / (2 * (config.center_sigma / image_width)**2))

                    angle = compute_viewing_angle(xyz, view.camera_center, view.camera_normal)
                    w_frontal = np.maximum(0, 1 - angle / config.frontal_threshold)

                    distance = np.linalg.norm(xyz - view.camera_center)
                    w_dist = np.exp(-distance / config.closest_sigma)

                    w = (config.blend_center_weight * w_center +
                         config.blend_frontal_weight * w_frontal +
                         config.blend_distance_weight * w_dist)
                    weights.append(w)

        if not colors_for_point:
            continue

        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)
        colors_for_point = np.array(colors_for_point)

        # Weighted blend
        blended_color = np.sum(colors_for_point * weights[:, None], axis=0)
        blended_color = np.clip(blended_color, 0, 255)

        # Use the point record from the highest-weighted camera
        best_cam = camera_ids_for_point[np.argmax(weights)]
        view = next(v for v in views if v.camera_id == best_cam)
        dist, idx = cKDTree(view.colored_points[:, 2:5]).query(xyz, k=1)
        base_record = view.colored_points[idx].copy()
        base_record[7:10] = blended_color

        blended_colors.append(base_record)
        best_cameras.append(best_cam)

    return np.array(blended_colors), np.array(best_cameras)


def write_colored_las(
    output_path: str,
    colored_points: np.ndarray,
    camera_labels: Optional[np.ndarray] = None
):
    """Write colored points to LAS/LAZ file.

    colored_points: Nx10 [u, v, X, Y, Z, intensity, depth, B, G, R]
    """
    try:
        import laspy

        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales = np.array([0.001, 0.001, 0.001])
        # Set offsets so UTM coordinates fit in int32 at 1mm scale
        header.offsets = np.array([
            float(int(colored_points[:, 2].min())),
            float(int(colored_points[:, 3].min())),
            float(int(colored_points[:, 4].min())),
        ])

        las = laspy.LasData(header)
        las.x = colored_points[:, 2]
        las.y = colored_points[:, 3]
        las.z = colored_points[:, 4]
        las.intensity = colored_points[:, 5].astype(np.uint16)

        # LAS stores RGB as 16-bit
        las.red = (colored_points[:, 9] * 257).astype(np.uint16)   # R
        las.green = (colored_points[:, 8] * 257).astype(np.uint16) # G
        las.blue = (colored_points[:, 7] * 257).astype(np.uint16)  # B

        if camera_labels is not None:
            # Store camera ID in user_data or classification
            las.classification = camera_labels.astype(np.uint8)

        las.write(output_path)
        print(f"Written colored LAS to {output_path}")

    except ImportError:
        # Fallback: write PLY
        write_colored_ply(output_path.replace('.las', '.ply'), colored_points, camera_labels)


def write_colored_ply(
    output_path: str,
    colored_points: np.ndarray,
    camera_labels: Optional[np.ndarray] = None
):
    """Write colored points to PLY format (ASCII)."""
    N = len(colored_points)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if camera_labels is not None:
            f.write("property uchar camera_id\n")
        f.write("end_header\n")

        for i in range(N):
            x, y, z = colored_points[i, 2:5]
            b, g, r = colored_points[i, 7:10]
            if camera_labels is not None:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {int(camera_labels[i])}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"Written colored PLY to {output_path}")


def generate_camera_selection_report(
    camera_labels: np.ndarray,
    views: List[CameraView],
    output_path: str
):
    """Generate report of camera selection statistics."""
    report = {
        'total_points': len(camera_labels),
        'strategy': 'multi-camera selection',
        'camera_distribution': {}
    }

    for view in views:
        count = np.sum(camera_labels == view.camera_id)
        pct = 100 * count / len(camera_labels)
        report['camera_distribution'][f'camera_{view.camera_id}'] = {
            'serial': view.serial,
            'points': int(count),
            'percentage': round(pct, 2)
        }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Selection report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Stage 8: Multi-Camera Selection & Blending')
    parser.add_argument('--camera_outputs', nargs='+', required=True,
                        help='Stage 7 outputs for each camera: cam0.npy cam1.npy cam2.npy cam3.npy')
    parser.add_argument('--strategy', default='center',
                        choices=['center', 'frontal', 'closest', 'consistency', 'blend'])
    parser.add_argument('--camera_centers', nargs='+', type=float,
                        help='Camera centers in world frame: x0 y0 z0 x1 y1 z1 ...')
    parser.add_argument('--camera_normals', nargs='+', type=float,
                        help='Camera normals in world frame: nx0 ny0 nz0 ...')
    parser.add_argument('--output', default='final_colored.las')
    parser.add_argument('--report', default='camera_selection_report.json')
    args = parser.parse_args()

    # Load all camera views
    views = []
    for i, path in enumerate(args.camera_outputs):
        points = np.load(path)
        print(f"Camera {i}: loaded {len(points)} points from {path}")

        # Default camera center/normal (would come from calibration in practice)
        center = np.array(args.camera_centers[i*3:(i+1)*3]) if args.camera_centers else np.zeros(3)
        normal = np.array(args.camera_normals[i*3:(i+1)*3]) if args.camera_normals else np.array([0, 0, 1])

        views.append(CameraView(
            camera_id=i,
            serial=f"cam_{i}",
            colored_points=points,
            camera_center=center,
            camera_normal=normal / (np.linalg.norm(normal) + 1e-10)
        ))

    # Configure
    config = BlendConfig(strategy=SelectionStrategy(args.strategy))

    # Select/blend
    print(f"\nApplying {args.strategy} strategy...")
    best_colors, camera_labels = select_best_camera_per_point(views, config)

    print(f"Final colored points: {len(best_colors)}")

    # Write output
    write_colored_las(args.output, best_colors, camera_labels)

    # Generate report
    report = generate_camera_selection_report(camera_labels, views, args.report)

    print(f"\nMulti-camera blending complete:")
    for cam_id, stats in report['camera_distribution'].items():
        print(f"  {cam_id}: {stats['points']} points ({stats['percentage']}%)")


if __name__ == '__main__':
    main()
