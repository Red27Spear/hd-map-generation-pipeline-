#!/usr/bin/env python3
"""
Phase 2 v8 -- Vertical Objects ONLY (NO road surface)
======================================================
Copied from scripts/proj/phase2_vertical_only_v8.py. Two runs of the
original on this project's merged part24-into-part48 cloud (48.8M points,
16.97M non-ground) both got OOM-killed at the same point: open3d's
cluster_dbscan() "precompute neighbors" step, even after the original
script's own 2,000,000-point random subsample cap. Random subsampling
preserves relative density, so a packed region (a building facade with
millions of points close together) is still over-represented in the
2M-point sample, and DBSCAN's neighbour search cost is dominated by local
density, not point count -- hence the crash surviving the existing cap.

Fix: replace the random subsample with a voxel-grid downsample (one point
per small 3D cell) before DBSCAN. This directly bounds worst-case local
density instead of just total count, which is what the neighbour-precompute
step actually blows up on. Also lowered the point budget as a second,
independent safety margin, since this project's LAZ tiles are unusually
dense to begin with (see the semester report's point-density finding).

Usage:
    python phase2_vertical_only_v8.py <laz_file>

Output (in /home/rohit/Documents/Output/phase2/):
    vertical_only_<part>.laz
"""

import sys
import os
import time
import re

import numpy as np
import laspy

OUTPUT_DIR = "/home/rohit/Documents/Output/phase2"

RANSAC_DIST_THRESH  = 0.20
RANSAC_N_POINTS     = 3
RANSAC_ITERATIONS   = 3000
DBSCAN_EPS          = 0.50
DBSCAN_MIN_PTS      = 8
MAX_DBSCAN_POINTS   = 800000
VOXEL_DOWNSAMPLE_M  = 0.08   # bounds worst-case local density before DBSCAN

POLE_MAX_WIDTH_M        = 2.00
POLE_MIN_HEIGHT_M       = 1.00
POLE_MIN_HW_RATIO       = 1.0
POLE_MIN_POINTS         = 10
POLE_MAX_POINTS         = 50000
POLE_MIN_VERTICALITY    = 0.40

SIGN_MAX_WIDTH_M        = 3.00
SIGN_MIN_HW_RATIO       = 0.3
SIGN_MIN_HEIGHT_M       = 0.5
SIGN_MAX_BASE_HEIGHT_M  = 8.0

LIGHT_MAX_WIDTH_M       = 2.00
LIGHT_MIN_HEIGHT_M      = 0.3
LIGHT_MAX_HEIGHT_M      = 4.0
LIGHT_MIN_BASE_HEIGHT_M = 2.0
LIGHT_MAX_BASE_HEIGHT_M = 12.0

GANTRY_MIN_HEIGHT_M     = 4.0
GANTRY_MAX_WIDTH_M      = 3.0
GANTRY_MIN_POINTS       = 50

KEEP_NOISE_POINTS       = True
NOISE_MIN_HEIGHT_M      = 1.5
NOISE_MAX_HEIGHT_M      = 15.0


def load_laz(laz_path):
    print(f"  Loading {os.path.basename(laz_path)}...", flush=True)
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    print(f"    {len(xyz):,} points | Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]", flush=True)
    return las, xyz


def ransac_ground(xyz):
    import open3d as o3d
    print("\n  [2/4] RANSAC ground plane...", flush=True)

    n_sub = min(500000, len(xyz))
    sub_idx = np.random.choice(len(xyz), n_sub, replace=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[sub_idx])

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH,
        ransac_n=RANSAC_N_POINTS,
        num_iterations=RANSAC_ITERATIONS,
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    verticality = abs(normal[2])

    if verticality < 0.8:
        print("    WARNING: Not horizontal. Using percentile fallback.", flush=True)
        ground_z = np.percentile(xyz[:, 2], 5)
        ground_mask = xyz[:, 2] < (ground_z + RANSAC_DIST_THRESH)
        return ground_mask, ~ground_mask

    dist_to_plane = np.abs(a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d)
    dist_to_plane /= np.sqrt(a**2 + b**2 + c**2)

    ground_mask = dist_to_plane < RANSAC_DIST_THRESH
    non_ground_mask = ~ground_mask

    print(f"    Ground: {ground_mask.sum():,} | Non-ground: {non_ground_mask.sum():,}", flush=True)
    return ground_mask, non_ground_mask


def voxel_downsample_indices(xyz, voxel):
    """One representative index per occupied voxel -- bounds worst-case
    local density (what DBSCAN's neighbour search actually costs), unlike
    plain random subsampling which preserves relative density."""
    origin = xyz.min(axis=0)
    cells = np.floor((xyz - origin) / voxel).astype(np.int64)
    keys = (cells[:, 0].astype(np.int64) << 42) | (cells[:, 1].astype(np.int64) << 21) | cells[:, 2].astype(np.int64)
    _, first_idx = np.unique(keys, return_index=True)
    return first_idx


def cluster_geometry(cpts, ground_z):
    z_min = float(cpts[:, 2].min())
    z_max = float(cpts[:, 2].max())
    height = z_max - z_min
    x_range = float(cpts[:, 0].max() - cpts[:, 0].min())
    y_range = float(cpts[:, 1].max() - cpts[:, 1].min())
    max_width = max(x_range, y_range)

    verticality = 0.0
    elongation = 1.0
    if len(cpts) >= 3:
        centered = cpts - cpts.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.shape == (3, 3):
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                verticality = abs(eigvecs[:, 0][2])
                elongation = eigvals[0] / (eigvals[2] + 1e-10) if eigvals[2] > 0 else 999
            except Exception:
                pass

    return {
        "height": height, "max_width": max_width, "verticality": verticality,
        "elongation": elongation, "base_height": z_min - ground_z,
        "n_points": len(cpts),
    }


def classify_by_shape(geo):
    h, mw, n, vert, elong, base_h = geo["height"], geo["max_width"], geo["n_points"], geo["verticality"], geo["elongation"], geo["base_height"]

    if n < POLE_MIN_POINTS:
        return "remove", f"too few ({n})"
    if n > POLE_MAX_POINTS and mw > 5.0:
        return "remove", f"too big ({n} pts)"
    if h < POLE_MIN_HEIGHT_M:
        return "remove", f"too short ({h:.1f}m)"

    if h >= POLE_MIN_HEIGHT_M and mw <= POLE_MAX_WIDTH_M and (h/max(mw,0.01)) >= POLE_MIN_HW_RATIO and vert >= POLE_MIN_VERTICALITY:
        return "keep", f"POLE h={h:.1f}m w={mw:.2f}m"
    if h >= SIGN_MIN_HEIGHT_M and mw <= SIGN_MAX_WIDTH_M and (h/max(mw,0.01)) >= SIGN_MIN_HW_RATIO and base_h <= SIGN_MAX_BASE_HEIGHT_M:
        return "keep", f"SIGN h={h:.1f}m w={mw:.2f}m"
    if mw <= LIGHT_MAX_WIDTH_M and LIGHT_MIN_BASE_HEIGHT_M <= base_h <= LIGHT_MAX_BASE_HEIGHT_M and LIGHT_MIN_HEIGHT_M <= h <= LIGHT_MAX_HEIGHT_M:
        return "keep", f"LIGHT h={h:.1f}m w={mw:.2f}m"
    if h >= GANTRY_MIN_HEIGHT_M and mw <= GANTRY_MAX_WIDTH_M and n >= GANTRY_MIN_POINTS and elong >= 5.0:
        return "keep", f"GANTRY h={h:.1f}m w={mw:.2f}m"
    if h >= 2.0 and elong >= 8.0 and mw <= 2.5:
        return "keep", f"ELONGATED h={h:.1f}m"

    return "remove", f"not vertical h={h:.1f}m w={mw:.2f}m"


def cluster_and_classify(xyz, non_ground_mask, ground_z):
    import open3d as o3d
    non_ground_xyz = xyz[non_ground_mask]
    non_ground_idx = np.where(non_ground_mask)[0]
    n_non_ground = len(non_ground_xyz)

    print(f"\n  [3/4] Clustering {n_non_ground:,} non-ground points...", flush=True)

    print(f"    Voxel-downsampling at {VOXEL_DOWNSAMPLE_M}m to bound local density before DBSCAN...", flush=True)
    vox_idx = voxel_downsample_indices(non_ground_xyz, VOXEL_DOWNSAMPLE_M)
    print(f"    {len(vox_idx):,} points after voxel downsample "
          f"(from {n_non_ground:,}, {100*len(vox_idx)/n_non_ground:.1f}%)", flush=True)

    if len(vox_idx) > MAX_DBSCAN_POINTS:
        sub_idx = np.random.choice(vox_idx, MAX_DBSCAN_POINTS, replace=False)
    else:
        sub_idx = vox_idx
    sub_xyz = non_ground_xyz[sub_idx]
    sub_original_idx = non_ground_idx[sub_idx]
    print(f"    DBSCAN input: {len(sub_xyz):,} points", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_xyz)
    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS, print_progress=True))

    n_clusters = int(labels.max()) + 1
    print(f"    Clusters: {n_clusters} | Noise: {(labels == -1).sum():,}", flush=True)

    keep_indices = []
    counts = {"keep": 0, "keep_pts": 0, "remove": 0, "remove_pts": 0}
    decisions_log = []

    for lbl in range(n_clusters):
        cmask = labels == lbl
        cpts = sub_xyz[cmask]
        cidx = sub_original_idx[cmask]
        geo = cluster_geometry(cpts, ground_z)
        decision, reason = classify_by_shape(geo)
        counts[decision] += 1
        counts[f"{decision}_pts"] += len(cpts)
        if decision == "keep":
            keep_indices.extend(cidx.tolist())
        if geo["n_points"] > 20:
            decisions_log.append(f"  [{decision:6s}] pts={geo['n_points']:5d} h={geo['height']:4.1f}m w={geo['max_width']:4.2f}m -> {reason}")

    if KEEP_NOISE_POINTS and (labels == -1).sum() > 0:
        noise_mask = labels == -1
        noise_pts = sub_xyz[noise_mask]
        noise_idx = sub_original_idx[noise_mask]
        valid_noise = (noise_pts[:, 2] > ground_z + NOISE_MIN_HEIGHT_M) & (noise_pts[:, 2] < ground_z + NOISE_MAX_HEIGHT_M)
        if valid_noise.sum() > 0:
            keep_indices.extend(noise_idx[valid_noise].tolist())
            counts["keep"] += 1
            counts["keep_pts"] += valid_noise.sum()

    return np.array(keep_indices, dtype=int), counts, decisions_log


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_vertical_only_v8.py <laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    basename = os.path.basename(laz_path)
    part_match = re.search(r'(part\d+)', basename)
    part_name = part_match.group(1) if part_match else basename.replace('.laz', '')

    print(f"\n{'='*70}")
    print(f"  PHASE 2 v8 -- VERTICAL OBJECTS ONLY (no road, no ground)")
    print(f"  {basename}")
    print(f"{'='*70}\n", flush=True)

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Loading...", flush=True)
    las, xyz = load_laz(laz_path)
    n_total = len(xyz)

    ground_mask, non_ground_mask = ransac_ground(xyz)
    ground_z = np.median(xyz[ground_mask, 2])
    print(f"    Ground Z = {ground_z:.2f}m", flush=True)

    if non_ground_mask.sum() == 0:
        print("\n  WARNING: No non-ground points!")
        keep_idx = np.array([], dtype=int)
        counts = {"keep": 0, "keep_pts": 0, "remove": 0, "remove_pts": 0}
        decisions_log = []
    else:
        keep_idx, counts, decisions_log = cluster_and_classify(xyz, non_ground_mask, ground_z)

    print(f"\n  [4/4] Saving vertical-only output...", flush=True)
    vertical_mask = np.zeros(n_total, dtype=bool)
    vertical_mask[keep_idx] = True

    n_vert = int(vertical_mask.sum())
    n_removed = n_total - n_vert
    print(f"    Vertical: {n_vert:,} ({100*n_vert/n_total:.1f}%)", flush=True)
    print(f"    Removed:  {n_removed:,} ({100*n_removed/n_total:.1f}%)", flush=True)

    out_path = os.path.join(OUTPUT_DIR, f"vertical_only_{part_name}.laz")

    hdr = laspy.LasHeader(point_format=las.point_format.id, version=las.header.version)
    hdr.offsets = las.header.offsets
    hdr.scales = las.header.scales

    for name in las.point_format.extra_dimension_names:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name=name, type=las[name].dtype))

    new_las = laspy.LasData(hdr)

    for dim in las.point_format.dimension_names:
        arr = np.array(getattr(las, dim))
        setattr(new_las, dim, arr[vertical_mask])

    for name in las.point_format.extra_dimension_names:
        new_las[name] = np.array(las[name])[vertical_mask]

    new_las.write(out_path)
    print(f"    Saved -> {out_path}", flush=True)
    print(f"    Final count: {len(new_las.x):,}", flush=True)

    elapsed = time.time() - t_start
    report = f"""Phase 2 v8 -- Vertical Only Report
===================================
Input: {n_total:,}
Vertical kept: {n_vert:,} ({100*n_vert/n_total:.1f}%)
Removed (road+ground+buildings+trees+noise): {n_removed:,} ({100*n_removed/n_total:.1f}%)
Output: {out_path}
Output points: {len(new_las.x):,}
Elapsed: {elapsed:.1f}s
"""
    print(report, flush=True)
    with open(os.path.join(OUTPUT_DIR, f"phase2_{part_name}_report.txt"), "w") as f:
        f.write(report)
    print("Done.\n", flush=True)


if __name__ == "__main__":
    main()
