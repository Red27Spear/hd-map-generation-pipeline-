#!/usr/bin/env python3
"""
Phase 2 (clean) — Road Surface + Vertical Infrastructure Only
================================================================
Consolidates the earlier phase2 experiments (v5 ground-stripper, v7b
ransac-merged, v8 vertical-only, v2 voxel-guided patch interpolation) into
one script that keeps exactly what an HD map needs and removes the rest:

  1 = road_surface   (RANSAC ground plane inliers)
  2 = vertical       (poles / signs / lights / gantries, shape-classified)

Everything else (buildings, trees, parked cars, unclassified noise) is
removed.

FIX vs v7b (phase2_ransac_merged_v7b.py): v7b had a
`KEEP_NOISE_POINTS` bypass that dumped ANY unclustered DBSCAN "noise"
point between 1.5-15m above ground straight into the kept set, with no
shape check at all. On part15 this let through wide, diffuse haze of
tree/building-edge fragments scattered on both sides of the road --
visually confirmed: the "vertical" class was NOT tight poles/signs but a
loose scatter spread far off the road corridor. This version removes
that bypass entirely -- a point only survives as "vertical" if its
DBSCAN cluster passes the same pole/sign/light/gantry shape checks as
everything else. Nothing gets a free pass on height range alone.

Usage:
    python clean_and_classify.py <laz_file> [--output-dir ./output]
"""

import argparse
import os
import time
import re

import numpy as np
import laspy

RANSAC_DIST_THRESH  = 0.20
RANSAC_N_POINTS     = 3
RANSAC_ITERATIONS   = 3000
DBSCAN_EPS          = 0.50
DBSCAN_MIN_PTS      = 8
MAX_DBSCAN_POINTS   = 2000000

POLE_MAX_WIDTH_M        = 2.00
POLE_MIN_HEIGHT_M       = 1.00
POLE_MIN_HW_RATIO       = 1.0
POLE_MIN_POINTS         = 20
POLE_MAX_POINTS         = 50000
# Genuine poles are near-perfect vertical cylinders (PCA verticality
# close to 1.0). 0.40 was loose enough that roadside bush/branch clumps
# (irregular but coincidentally taller-than-wide) passed as "poles" --
# visually confirmed on part15 as a wide scatter of small red clusters
# hugging the verge, not discrete poles. 0.75 requires the cluster's
# dominant axis to actually be vertical, not just its bounding box.
POLE_MIN_VERTICALITY    = 0.75

SIGN_MAX_WIDTH_M        = 3.00
SIGN_MIN_HW_RATIO       = 0.3
SIGN_MIN_HEIGHT_M       = 0.5
SIGN_MAX_BASE_HEIGHT_M  = 8.0
SIGN_MIN_VERTICALITY    = 0.45

LIGHT_MAX_WIDTH_M       = 2.00
LIGHT_MIN_HEIGHT_M      = 0.3
LIGHT_MAX_HEIGHT_M      = 4.0
LIGHT_MIN_BASE_HEIGHT_M = 2.0
LIGHT_MAX_BASE_HEIGHT_M = 12.0

GANTRY_MIN_HEIGHT_M     = 4.0
GANTRY_MAX_WIDTH_M      = 3.0
GANTRY_MIN_POINTS       = 50

# Real infrastructure (poles/signs/lights/gantries) sits near the road;
# tree-branch fragments and building-edge scraps can coincidentally pass
# the shape thresholds above but tend to sit much farther away. Median
# distance-to-road for genuine infra clusters on part15 was ~0.14m, with
# 90% under ~7m -- 12m gives headroom for wide verges/medians while still
# rejecting the long tail (some fragments were 20-60m from any road point).
MAX_ROAD_DIST_M         = 12.0

CLS_ROAD_SURFACE = 1
CLS_VERTICAL     = 2


def load_laz(laz_path):
    print(f"  Loading {os.path.basename(laz_path)}...")
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    print(f"    {len(xyz):,} points | Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    return las, xyz


def ransac_ground(xyz):
    import open3d as o3d
    print("\n  [2/5] RANSAC ground plane...")

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
        print("    WARNING: Not horizontal. Using percentile fallback.")
        ground_z = np.percentile(xyz[:, 2], 5)
        ground_mask = xyz[:, 2] < (ground_z + RANSAC_DIST_THRESH)
        return ground_mask, ~ground_mask, None

    dist_to_plane = np.abs(a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d)
    dist_to_plane /= np.sqrt(a**2 + b**2 + c**2)

    ground_mask = dist_to_plane < RANSAC_DIST_THRESH
    non_ground_mask = ~ground_mask

    print(f"    Ground: {ground_mask.sum():,} | Non-ground: {non_ground_mask.sum():,}")
    return ground_mask, non_ground_mask, plane_model


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
    if h >= SIGN_MIN_HEIGHT_M and mw <= SIGN_MAX_WIDTH_M and (h/max(mw,0.01)) >= SIGN_MIN_HW_RATIO and base_h <= SIGN_MAX_BASE_HEIGHT_M and vert >= SIGN_MIN_VERTICALITY:
        return "keep", f"SIGN h={h:.1f}m w={mw:.2f}m"
    if mw <= LIGHT_MAX_WIDTH_M and LIGHT_MIN_BASE_HEIGHT_M <= base_h <= LIGHT_MAX_BASE_HEIGHT_M and LIGHT_MIN_HEIGHT_M <= h <= LIGHT_MAX_HEIGHT_M:
        return "keep", f"LIGHT h={h:.1f}m w={mw:.2f}m"
    if h >= GANTRY_MIN_HEIGHT_M and mw <= GANTRY_MAX_WIDTH_M and n >= GANTRY_MIN_POINTS and elong >= 5.0:
        return "keep", f"GANTRY h={h:.1f}m w={mw:.2f}m"
    if h >= 2.0 and elong >= 8.0 and mw <= 2.5:
        return "keep", f"ELONGATED h={h:.1f}m"

    return "remove", f"not infra h={h:.1f}m w={mw:.2f}m base={base_h:.1f}m vert={vert:.2f}"


def cluster_and_classify(xyz, non_ground_mask, ground_mask, ground_z):
    import open3d as o3d
    from scipy.spatial import cKDTree

    road_xy = xyz[ground_mask][:, :2]
    if len(road_xy) > 500000:
        road_xy = road_xy[np.random.choice(len(road_xy), 500000, replace=False)]
    road_tree = cKDTree(road_xy)

    non_ground_xyz = xyz[non_ground_mask]
    non_ground_idx = np.where(non_ground_mask)[0]
    n_non_ground = len(non_ground_xyz)

    print(f"\n  [3/5] Clustering {n_non_ground:,} non-ground points...")

    if n_non_ground > MAX_DBSCAN_POINTS:
        sub_idx = np.random.choice(n_non_ground, MAX_DBSCAN_POINTS, replace=False)
        sub_xyz = non_ground_xyz[sub_idx]
        sub_original_idx = non_ground_idx[sub_idx]
    else:
        sub_xyz = non_ground_xyz
        sub_original_idx = non_ground_idx

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub_xyz)
    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS, print_progress=True))

    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"    Clusters: {n_clusters} | Noise (dropped, unclustered): {n_noise:,}")

    keep_indices = []
    counts = {"keep": 0, "keep_pts": 0, "remove": 0, "remove_pts": 0}
    decisions_log = []

    n_far_rejected = 0
    for lbl in range(n_clusters):
        cmask = labels == lbl
        cpts = sub_xyz[cmask]
        cidx = sub_original_idx[cmask]
        geo = cluster_geometry(cpts, ground_z)
        decision, reason = classify_by_shape(geo)

        if decision == "keep":
            centroid_xy = cpts[:, :2].mean(axis=0)
            road_dist, _ = road_tree.query(centroid_xy)
            if road_dist > MAX_ROAD_DIST_M:
                decision, reason = "remove", f"{reason} but {road_dist:.1f}m from road (>{MAX_ROAD_DIST_M}m)"
                n_far_rejected += 1

        counts[decision] += 1
        counts[f"{decision}_pts"] += len(cpts)
        if decision == "keep":
            keep_indices.extend(cidx.tolist())
        if geo["n_points"] > 20:
            decisions_log.append(f"  [{decision:6s}] pts={geo['n_points']:5d} h={geo['height']:4.1f}m w={geo['max_width']:4.2f}m → {reason}")

    print(f"    Rejected as too far from road (>{MAX_ROAD_DIST_M}m): {n_far_rejected} clusters")

    # NOTE (fix vs v7b): unclustered DBSCAN noise (labels == -1) is NOT
    # kept here, regardless of height. A point with no cluster has no
    # shape to classify -- it cannot be verified as a pole/sign/light/
    # gantry, so per "remove all unnecessary elements" it is dropped
    # rather than given a height-range free pass.

    return np.array(keep_indices, dtype=int), counts, decisions_log, n_noise


def main():
    parser = argparse.ArgumentParser(description="Road surface + vertical infrastructure extraction")
    parser.add_argument("laz_file", help="input LAZ tile")
    parser.add_argument("--output-dir", default="./output", help="where to write the cleaned LAZ + report")
    args = parser.parse_args()

    laz_path = args.laz_file
    OUTPUT_DIR = args.output_dir
    basename = os.path.basename(laz_path)
    part_match = re.search(r'(part\d+)', basename)
    part_name = part_match.group(1) if part_match else basename.replace('.laz', '')

    print(f"\n{'='*70}")
    print(f"  PHASE 2 (clean) — Road + Vertical ONLY, shape-verified")
    print(f"  {basename}")
    print(f"{'='*70}\n")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/5] Loading...")
    las, xyz = load_laz(laz_path)
    n_total = len(xyz)

    ground_mask, non_ground_mask, plane_model = ransac_ground(xyz)
    ground_z = np.median(xyz[ground_mask, 2])
    print(f"    Ground Z = {ground_z:.2f}m")

    if non_ground_mask.sum() == 0:
        keep_idx = np.array([], dtype=int)
        counts = {"keep": 0, "keep_pts": 0, "remove": 0, "remove_pts": 0}
        decisions_log = []
        n_noise = 0
    else:
        keep_idx, counts, decisions_log, n_noise = cluster_and_classify(xyz, non_ground_mask, ground_mask, ground_z)

    print("\n  [4/5] Building combined mask...")
    combined_mask = np.zeros(n_total, dtype=bool)
    combined_mask[ground_mask] = True      # road surface
    combined_mask[keep_idx] = True          # vertical objects

    n_road = int(ground_mask.sum())
    n_vert = int(len(keep_idx))
    n_kept = int(combined_mask.sum())
    n_removed = n_total - n_kept

    print(f"    Road: {n_road:,} | Vertical: {n_vert:,}")
    print(f"    Kept: {n_kept:,} ({100*n_kept/n_total:.1f}%) | Removed: {n_removed:,} ({100*n_removed/n_total:.1f}%)")

    classification = np.full(n_total, 0, dtype=np.uint8)
    classification[ground_mask] = CLS_ROAD_SURFACE
    classification[keep_idx] = CLS_VERTICAL

    print("\n  [5/5] Saving filtered output...")
    out_path = os.path.join(OUTPUT_DIR, f"clean_{part_name}.laz")

    hdr = laspy.LasHeader(
        point_format=las.point_format.id,
        version=las.header.version
    )
    hdr.offsets = las.header.offsets
    hdr.scales = las.header.scales

    for name in las.point_format.extra_dimension_names:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name=name, type=las[name].dtype))

    new_las = laspy.LasData(hdr)

    for dim in las.point_format.dimension_names:
        arr = np.array(getattr(las, dim))
        setattr(new_las, dim, arr[combined_mask])

    for name in las.point_format.extra_dimension_names:
        new_las[name] = np.array(las[name])[combined_mask]

    cls_filtered = classification[combined_mask]
    new_las.classification = cls_filtered.astype(np.uint8)

    new_las.write(out_path)
    print(f"    Saved → {out_path}")
    print(f"    Final count: {len(new_las.x):,}")

    elapsed = time.time() - t_start
    report = f"""Phase 2 (clean) Report
======================
Input: {n_total:,}
Road (cls 1): {n_road:,} ({100*n_road/n_total:.1f}%)
Vertical (cls 2, shape-verified): {n_vert:,} ({100*n_vert/n_total:.1f}%)
Unclustered noise dropped: {n_noise:,}
Removed (buildings/trees/noise): {n_removed:,} ({100*n_removed/n_total:.1f}%)
Final output: {len(new_las.x):,} points
Output: {out_path}
Elapsed: {elapsed:.1f}s

Cluster decisions (>20 pts):
""" + "\n".join(decisions_log)
    print(report[:2000])
    with open(os.path.join(OUTPUT_DIR, f"phase2_clean_{part_name}_report.txt"), "w") as f:
        f.write(report)
    print("Done.\n")


if __name__ == "__main__":
    main()
