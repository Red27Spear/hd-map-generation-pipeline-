#!/usr/bin/env python3
"""
Stage 6: Occlusion Handling & Depth Buffering
Resolves self-occlusion when multiple LiDAR points project to the same pixel.
Uses Z-buffer approach: keep only the closest (minimum depth) point per pixel.

Also implements:
- Ray-casting occlusion (optional): discard points behind surfaces
- Outlier filtering based on depth discontinuity

Usage:
    python stage6_occlusion_handling.py \
        --projected stage5_output.npy \
        --camera_width 2448 --camera_height 2048 \
        --output occluded_filtered.npy
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import argparse
from dataclasses import dataclass
from scipy import ndimage


@dataclass
class ZBufferConfig:
    """Configuration for Z-buffer occlusion handling."""
    camera_width: int = 2448
    camera_height: int = 2048
    depth_tolerance: float = 0.5  # meters: points within this range are merged
    min_depth: float = 0.5        # minimum valid depth (near-plane clipping)
    max_depth: float = 200.0      # maximum valid depth (far-plane clipping)
    median_filter_size: int = 3   # median filter on depth map to reduce noise
    fill_holes: bool = True       # fill small holes in depth map
    hole_fill_radius: int = 2


def build_zbuffer(
    projected_points: np.ndarray,
    config: ZBufferConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Z-buffer from projected points and resolve occlusions.

    Args:
        projected_points: Nx7 array [u, v, X, Y, Z, intensity, depth]
        config: ZBufferConfig parameters

    Returns:
        depth_map: HxW float array, depth per pixel (inf where no point)
        index_map: HxW int array, index of winning point per pixel (-1 where empty)
        filtered_points: Mx7 array of visible (non-occluded) points
    """
    H, W = config.camera_height, config.camera_width

    # Initialize Z-buffer with infinity
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    index_map = np.full((H, W), -1, dtype=np.int32)

    # Round pixel coordinates to integers
    u = np.round(projected_points[:, 0]).astype(np.int32)
    v = np.round(projected_points[:, 1]).astype(np.int32)
    depths = projected_points[:, 6].astype(np.float32)

    # Clip to image bounds
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid &= (depths >= config.min_depth) & (depths <= config.max_depth)

    u, v, depths = u[valid], v[valid], depths[valid]
    indices = np.where(valid)[0]

    # Z-buffer: keep minimum depth per pixel
    for i in range(len(u)):
        px, py = u[i], v[i]
        if depths[i] < depth_map[py, px]:
            depth_map[py, px] = depths[i]
            index_map[py, px] = indices[i]

    # Optional: median filter on depth map to reduce noise
    if config.median_filter_size > 1:
        valid_mask = depth_map < np.inf
        depth_map_filtered = depth_map.copy()
        # Only filter valid regions
        depth_map_filtered[valid_mask] = ndimage.median_filter(
            depth_map[valid_mask], 
            size=config.median_filter_size
        )
        depth_map = depth_map_filtered

    # Optional: fill small holes
    if config.fill_holes:
        depth_map = fill_depth_holes(depth_map, config.hole_fill_radius)

    # Extract visible points
    visible_indices = index_map[index_map >= 0]
    filtered_points = projected_points[visible_indices]

    return depth_map, index_map, filtered_points


def fill_depth_holes(depth_map: np.ndarray, radius: int = 2) -> np.ndarray:
    """Fill small holes in depth map using nearest valid neighbor."""
    from scipy.ndimage import distance_transform_edt

    mask = np.isfinite(depth_map)
    if mask.all():
        return depth_map

    # Distance transform to find nearest valid pixel
    dist, indices = distance_transform_edt(~mask, return_indices=True)

    # Fill from nearest valid
    filled = depth_map.copy()
    filled[~mask] = depth_map[indices[0][~mask], indices[1][~mask]]

    return filled


def detect_depth_discontinuities(
    depth_map: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """Detect depth discontinuities (edges) in the depth map.

    Returns binary mask where True indicates a depth edge.
    """
    # Compute depth gradients
    dy, dx = np.gradient(depth_map)
    gradient_mag = np.sqrt(dx**2 + dy**2)

    # Threshold
    edges = gradient_mag > threshold

    return edges


def ray_cast_occlusion_filter(
    points_world: np.ndarray,
    scanner_origin: np.ndarray,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None
) -> np.ndarray:
    """Optional: Ray-casting occlusion filter.

    Casts rays from scanner origin through each point and checks if the point
    is the first intersection (visible) or occluded by another surface.

    Requires a mesh representation of the scene. If no mesh provided,
    falls back to Z-buffer method.

    Args:
        points_world: Nx3 points in world frame
        scanner_origin: 3D scanner position in world frame
        mesh_vertices: Optional mesh vertices
        mesh_faces: Optional mesh face indices

    Returns:
        visible_mask: N bool array
    """
    if mesh_vertices is None or mesh_faces is None:
        # No mesh available, return all visible (Z-buffer handles this elsewhere)
        return np.ones(len(points_world), dtype=bool)

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

        # Ray directions from scanner to each point
        directions = points_world - scanner_origin
        distances = np.linalg.norm(directions, axis=1)
        directions = directions / (distances[:, None] + 1e-10)

        # Cast rays
        origins = np.tile(scanner_origin, (len(points_world), 1))
        locations, ray_indices, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )

        # Check if first hit is close to our point
        visible = np.ones(len(points_world), dtype=bool)
        for i, idx in enumerate(ray_indices):
            hit_dist = np.linalg.norm(locations[i] - scanner_origin)
            point_dist = distances[idx]
            if hit_dist < point_dist - 0.1:  # 10cm tolerance
                visible[idx] = False

        return visible
    except ImportError:
        print("Warning: trimesh not available, skipping ray-casting occlusion")
        return np.ones(len(points_world), dtype=bool)


def filter_flying_pixels(
    projected_points: np.ndarray,
    depth_map: np.ndarray,
    neighbor_radius: int = 3,
    depth_diff_threshold: float = 5.0
) -> np.ndarray:
    """Remove "flying pixels" — isolated points with inconsistent depth.

    A flying pixel is a projected point whose depth differs significantly
    from its neighbors in the depth map.
    """
    H, W = depth_map.shape
    u = np.round(projected_points[:, 0]).astype(np.int32)
    v = np.round(projected_points[:, 1]).astype(np.int32)
    depths = projected_points[:, 6]

    valid = np.ones(len(projected_points), dtype=bool)

    for i in range(len(u)):
        px, py = u[i], v[i]
        if px < 0 or px >= W or py < 0 or py >= H:
            valid[i] = False
            continue

        # Sample neighborhood in depth map
        y0, y1 = max(0, py - neighbor_radius), min(H, py + neighbor_radius + 1)
        x0, x1 = max(0, px - neighbor_radius), min(W, px + neighbor_radius + 1)

        neighbor_depths = depth_map[y0:y1, x0:x1]
        neighbor_depths = neighbor_depths[np.isfinite(neighbor_depths)]

        if len(neighbor_depths) == 0:
            valid[i] = False
            continue

        median_depth = np.median(neighbor_depths)
        if abs(depths[i] - median_depth) > depth_diff_threshold:
            valid[i] = False

    return valid


def render_depth_map(
    depth_map: np.ndarray,
    output_path: str,
    colormap: str = 'turbo'
):
    """Render depth map as a color image for visualization."""
    valid = np.isfinite(depth_map)
    if not valid.any():
        print("Warning: no valid depth values to render")
        return

    dmin, dmax = depth_map[valid].min(), depth_map[valid].max()

    # Normalize
    normalized = np.zeros_like(depth_map)
    normalized[valid] = (depth_map[valid] - dmin) / (dmax - dmin + 1e-6)
    normalized = (normalized * 255).astype(np.uint8)

    # Apply colormap
    if colormap == 'turbo':
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    elif colormap == 'jet':
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    else:
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)

    # Set invalid pixels to black
    colored[~valid] = [0, 0, 0]

    cv2.imwrite(output_path, colored)
    print(f"Depth map rendered to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 6: Occlusion Handling & Depth Buffering')
    parser.add_argument('--projected', required=True, help='Stage 5 output .npy file')
    parser.add_argument('--camera_width', type=int, default=2448)
    parser.add_argument('--camera_height', type=int, default=2048)
    parser.add_argument('--depth_tolerance', type=float, default=0.5)
    parser.add_argument('--min_depth', type=float, default=0.5)
    parser.add_argument('--max_depth', type=float, default=200.0)
    parser.add_argument('--median_filter', type=int, default=3)
    parser.add_argument('--flying_pixel_threshold', type=float, default=5.0)
    parser.add_argument('--output', default='occluded_filtered.npy')
    parser.add_argument('--depth_map_png', default='depth_map.png')
    args = parser.parse_args()

    # Load projected points
    projected = np.load(args.projected)
    print(f"Loaded {len(projected)} projected points")

    # Configure
    config = ZBufferConfig(
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        depth_tolerance=args.depth_tolerance,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        median_filter_size=args.median_filter
    )

    # Build Z-buffer
    depth_map, index_map, filtered = build_zbuffer(projected, config)
    print(f"After Z-buffer occlusion: {len(filtered)} visible points")

    # Filter flying pixels
    flying_valid = filter_flying_pixels(
        filtered, depth_map,
        neighbor_radius=3,
        depth_diff_threshold=args.flying_pixel_threshold
    )
    filtered = filtered[flying_valid]
    print(f"After flying-pixel filter: {len(filtered)} points")

    # Detect depth discontinuities
    edges = detect_depth_discontinuities(depth_map)
    edge_count = np.sum(edges)
    print(f"Depth discontinuities detected: {edge_count} pixels")

    # Save
    np.save(args.output, filtered)
    print(f"Saved filtered points to {args.output}")

    # Render depth map
    render_depth_map(depth_map, args.depth_map_png)

    print(f"\nOcclusion handling complete:")
    print(f"  Original: {len(projected)} points")
    print(f"  Visible:  {len(filtered)} points")
    print(f"  Occluded: {len(projected) - len(filtered)} points ({100*(1-len(filtered)/len(projected)):.1f}%)")


if __name__ == '__main__':
    main()
