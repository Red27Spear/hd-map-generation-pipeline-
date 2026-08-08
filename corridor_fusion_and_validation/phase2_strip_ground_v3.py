#!/usr/bin/env python3
"""
Phase 2 v3 -- Ground-Only Stripper + Vertical Object Keeper
===========================================================
Copied from scripts/proj/phase2b.py. More thorough than v8: strips ground
using BOTH a global RANSAC plane AND an adaptive local-ground pass (nearest
RANSAC-inlier ground seed per point), which catches footpaths/curbs sitting
at a different height than the main road plane -- v8's single global plane
only catches one ground level, which is why building-base debris could
survive into its output. Elevated points are then DBSCAN'd and classified
with a strict whitelist (POLE_MAX_WIDTH_M=1.20m, POLE_MIN_HW_RATIO=2.0 --
"must be clearly vertical"), rejecting anything that isn't clearly a pole,
sign, or traffic-light-sized box.

Two fixes vs. the original:
  1. CAMEXTR_PATH pointed at a path that doesn't exist on this machine, but
     was dead code (never actually read) -- removed rather than fixed.
  2. The original calls open3d's cluster_dbscan() directly on every elevated
     point with no subsampling cap at all -- worse than v8's crash risk.
     Added the same voxel-downsample-before-DBSCAN fix used for v8, since
     that fix is what got v8's DBSCAN step running reliably on this
     project's point clouds.

Usage:
    python phase2_strip_ground_v3.py <laz_file>

Outputs (in /home/rohit/Documents/Output/phase2/):
    vertical_only_<part>.laz
    phase2_<part>_strip_report.txt
"""

import sys
import os
import time
import re

import numpy as np
import laspy

OUTPUT_DIR = "/home/rohit/Documents/Output/phase2"

RANSAC_DIST_THRESH  = 0.15
RANSAC_N_POINTS     = 3
RANSAC_ITERATIONS   = 2000

GROUND_STRIP_HEIGHT_M = 0.25
ABSOLUTE_MIN_Z_OFFSET = -0.10

ADAPTIVE_STRIP_RADIUS_M = 2.0
ADAPTIVE_STRIP_HEIGHT_M = 0.30

ELEVATED_DBSCAN_EPS     = 0.50
ELEVATED_DBSCAN_MIN_PTS = 4
MAX_DBSCAN_POINTS       = 800000
VOXEL_DOWNSAMPLE_M      = 0.08

POLE_MAX_WIDTH_M        = 1.20
POLE_MIN_HEIGHT_M       = 1.80
POLE_MIN_HW_RATIO       = 2.0
POLE_MAX_GROUND_GAP_M   = 6.0
POLE_MIN_POINTS         = 8
POLE_MAX_POINTS         = 5000
POLE_MIN_VERTICALITY    = 0.75

SIGN_MAX_WIDTH_M        = 1.50
SIGN_MIN_HW_RATIO       = 0.8

KEEP_NOISE_POINTS       = True
NOISE_MAX_HEIGHT_M      = 8.0


def log(msg):
    print(msg, flush=True)


def load_laz(laz_path):
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    log(f"    {len(xyz):,} points loaded")
    return las, xyz


def save_laz(las_original, keep_mask, out_path):
    hdr = laspy.LasHeader(point_format=las_original.point_format, version=las_original.header.version)
    hdr.offsets = las_original.header.offsets
    hdr.scales = las_original.header.scales
    new_las = laspy.LasData(hdr)
    for dim in las_original.point_format.dimension_names:
        setattr(new_las, dim, np.array(getattr(las_original, dim))[keep_mask])
    new_las.write(out_path)
    log(f"    Saved {keep_mask.sum():,} points -> {out_path}")


def voxel_downsample_indices(xyz, voxel):
    origin = xyz.min(axis=0)
    cells = np.floor((xyz - origin) / voxel).astype(np.int64)
    keys = (cells[:, 0].astype(np.int64) << 42) | (cells[:, 1].astype(np.int64) << 21) | cells[:, 2].astype(np.int64)
    _, first_idx = np.unique(keys, return_index=True)
    return first_idx


def estimate_ground_and_strip(xyz):
    import open3d as o3d
    from scipy.spatial import cKDTree

    # Fix for a real degeneracy: running RANSAC on the WHOLE cloud (as the
    # original phase2b.py does) lets it converge on a large near-vertical
    # building-facade plane instead of the ground whenever a tile has dense
    # facade points -- confirmed on part24 (RANSAC picked a wall, giving
    # ground_z=-25717m and 0 kept points). Bias the candidate set toward
    # true ground first with a coarse percentile estimate, then verify the
    # winning plane is actually near-horizontal before trusting it.
    log("    Estimating coarse ground level (5th percentile Z)...")
    coarse_z = float(np.percentile(xyz[:, 2], 5))
    log(f"    Coarse ground Z (5th pct): {coarse_z:.2f}m")

    band_mask = (xyz[:, 2] >= coarse_z - 1.0) & (xyz[:, 2] <= coarse_z + 3.0)
    band_xyz = xyz[band_mask]
    log(f"    RANSAC candidate band [{coarse_z-1.0:.2f}, {coarse_z+3.0:.2f}]: "
        f"{len(band_xyz):,} points ({100*len(band_xyz)/len(xyz):.1f}%)")

    log("    Running RANSAC ground plane (band-restricted)...")
    n_sub = min(500000, len(band_xyz))
    sub_idx = np.random.choice(len(band_xyz), n_sub, replace=False)
    band_sub = band_xyz[sub_idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(band_sub)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH, ransac_n=RANSAC_N_POINTS, num_iterations=RANSAC_ITERATIONS)

    a, b, c, d = plane_model
    norm = float(np.sqrt(a * a + b * b + c * c))
    is_horizontal = (abs(c) / norm) > 0.9

    if is_horizontal:
        # NOT "-d/c": x,y here are absolute UTM coordinates (hundreds of
        # thousands to millions of metres), so even a tiny non-zero a/b
        # tilt gets amplified into a huge, meaningless offset by that -d/c
        # simplification (which is only valid near the coordinate origin).
        # Use the actual median Z of the inlier ground points instead --
        # robust regardless of any residual plane tilt or coordinate scale.
        ground_seed_xyz = band_sub[inliers]
        ground_z_global = float(np.median(ground_seed_xyz[:, 2]))
        log(f"    RANSAC plane is horizontal (|c|/|n|={abs(c)/norm:.2f}), "
            f"ground Z (median of inliers): {ground_z_global:.2f}m ({len(inliers):,} inliers)")
        dist_to_plane = np.abs(a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / norm
        near_global_ground = dist_to_plane < GROUND_STRIP_HEIGHT_M
    else:
        ground_z_global = coarse_z
        ground_seed_xyz = band_sub
        log(f"    RANSAC plane was NOT horizontal (|c|/|n|={abs(c)/norm:.2f}, likely a facade) "
            f"-- falling back to percentile ground Z: {ground_z_global:.2f}m")
        near_global_ground = np.abs(xyz[:, 2] - ground_z_global) < GROUND_STRIP_HEIGHT_M

    too_low = xyz[:, 2] < (ground_z_global + ABSOLUTE_MIN_Z_OFFSET)

    log("    Computing adaptive local ground...")
    ground_tree = cKDTree(ground_seed_xyz[:, :2])
    _, nearest_ground_idx = ground_tree.query(xyz[:, :2], k=1)
    local_ground_z = ground_seed_xyz[nearest_ground_idx, 2]
    near_local_ground = np.abs(xyz[:, 2] - local_ground_z) < ADAPTIVE_STRIP_HEIGHT_M

    ground_mask = near_global_ground | near_local_ground | too_low
    log(f"    Ground points removed: {ground_mask.sum():,} ({100*ground_mask.sum()/len(xyz):.1f}%)")
    elevated_mask = ~ground_mask
    log(f"    Elevated points remaining: {elevated_mask.sum():,}")
    return ground_mask, elevated_mask, ground_z_global


def cluster_geometry_fast(cpts):
    z_min = float(cpts[:, 2].min())
    z_max = float(cpts[:, 2].max())
    height = z_max - z_min
    width_e = float(cpts[:, 0].max() - cpts[:, 0].min())
    width_n = float(cpts[:, 1].max() - cpts[:, 1].min())
    max_width = max(width_e, width_n)
    min_width = min(width_e, width_n)
    verticality = 0.0
    if len(cpts) >= 3:
        centered = cpts - cpts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dominant = eigvecs[:, np.argmax(eigvals)]
        verticality = abs(dominant[2])
    return {"z_min": z_min, "z_max": z_max, "height": height, "max_width": max_width,
            "min_width": min_width, "verticality": verticality, "n_points": len(cpts)}


def classify_elevated_cluster(geo, ground_z):
    h = geo["height"]
    mw = geo["max_width"]
    n = geo["n_points"]
    vert = geo["verticality"]
    z_min = geo["z_min"]
    base_hag = z_min - ground_z

    if n < POLE_MIN_POINTS:
        return "remove", f"too few points ({n})"
    if n > POLE_MAX_POINTS:
        return "remove", f"too many points ({n}) -- merged structure"
    if h < POLE_MIN_HEIGHT_M:
        return "remove", f"too short h={h:.2f}m"

    is_pole = (mw <= POLE_MAX_WIDTH_M and base_hag <= POLE_MAX_GROUND_GAP_M and
               (h / max(mw, 0.01)) >= POLE_MIN_HW_RATIO and vert >= POLE_MIN_VERTICALITY)
    if is_pole:
        return "keep", f"POLE h={h:.1f}m w={mw:.2f}m base={base_hag:.1f}m vert={vert:.2f}"

    is_sign = (mw <= SIGN_MAX_WIDTH_M and base_hag <= POLE_MAX_GROUND_GAP_M and
               (h / max(mw, 0.01)) >= SIGN_MIN_HW_RATIO and h >= 1.0)
    if is_sign:
        return "keep", f"SIGN h={h:.1f}m w={mw:.2f}m base={base_hag:.1f}m"

    is_light = (mw <= 0.8 and 2.0 <= base_hag <= 8.0 and 0.5 <= h <= 2.0)
    if is_light:
        return "keep", f"LIGHT h={h:.1f}m w={mw:.2f}m base={base_hag:.1f}m"

    return "remove", f"not infrastructure h={h:.1f}m w={mw:.2f}m base={base_hag:.1f}m vert={vert:.2f}"


def process_elevated_points(elevated_xyz, elevated_indices, ground_z):
    import open3d as o3d

    log(f"    Voxel-downsampling {len(elevated_xyz):,} elevated points at {VOXEL_DOWNSAMPLE_M}m before DBSCAN...")
    vox_idx = voxel_downsample_indices(elevated_xyz, VOXEL_DOWNSAMPLE_M)
    log(f"    {len(vox_idx):,} points after voxel downsample")
    if len(vox_idx) > MAX_DBSCAN_POINTS:
        sub_idx = np.random.choice(vox_idx, MAX_DBSCAN_POINTS, replace=False)
    else:
        sub_idx = vox_idx
    dbscan_xyz = elevated_xyz[sub_idx]
    dbscan_orig_idx = elevated_indices[sub_idx]
    log(f"    DBSCAN input: {len(dbscan_xyz):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dbscan_xyz)
    labels = np.array(pcd.cluster_dbscan(eps=ELEVATED_DBSCAN_EPS, min_points=ELEVATED_DBSCAN_MIN_PTS, print_progress=True))

    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    log(f"    Elevated clusters: {n_clusters}  Noise: {n_noise:,}")

    keep_indices = []
    counts = {"keep": 0, "keep_pts": 0, "remove": 0, "remove_pts": 0}
    decisions_log = []

    for lbl in range(n_clusters):
        cmask = labels == lbl
        cpts = dbscan_xyz[cmask]
        cidx = dbscan_orig_idx[cmask]
        geo = cluster_geometry_fast(cpts)
        decision, reason = classify_elevated_cluster(geo, ground_z)
        counts[decision] += 1
        counts[f"{decision}_pts"] += len(cpts)
        if decision == "keep":
            keep_indices.extend(cidx.tolist())
        if geo["n_points"] > 50:
            decisions_log.append(f"  [{decision:6s}] pts={geo['n_points']:5d} h={geo['height']:4.1f}m "
                                  f"w={geo['max_width']:4.2f}m -> {reason}")

    if KEEP_NOISE_POINTS and n_noise > 0:
        noise_mask = labels == -1
        noise_pts = dbscan_xyz[noise_mask]
        noise_idx = dbscan_orig_idx[noise_mask]
        valid_noise = noise_pts[:, 2] < (ground_z + NOISE_MAX_HEIGHT_M)
        keep_indices.extend(noise_idx[valid_noise].tolist())
        counts["keep"] += 1
        counts["keep_pts"] += valid_noise.sum()
        decisions_log.append(f"  [keep  ] NOISE pts={valid_noise.sum():,} (isolated elevated points)")

    return np.array(keep_indices, dtype=int), counts, decisions_log


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_strip_ground_v3.py <path_to_laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    basename = os.path.basename(laz_path)
    part_match = re.search(r'(part\d+)', basename)
    part_name = part_match.group(1) if part_match else basename.replace('.laz', '')

    log(f"\n{'='*70}")
    log(f"  PHASE 2 v3 -- Ground Stripper + Vertical Object Keeper")
    log(f"  LAZ tile : {basename}")
    log(f"  Part     : {part_name}")
    log(f"{'='*70}\n")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("[1/4] Loading point cloud...")
    las_original, xyz = load_laz(laz_path)

    log("\n[2/4] Stripping ground surface...")
    ground_mask, elevated_mask, ground_z = estimate_ground_and_strip(xyz)

    if elevated_mask.sum() == 0:
        log("    WARNING: No elevated points found!")
        out_path = os.path.join(OUTPUT_DIR, f"vertical_only_{part_name}.laz")
        save_laz(las_original, np.zeros(len(xyz), dtype=bool), out_path)
        return

    log("\n[3/4] Clustering elevated points...")
    elevated_xyz = xyz[elevated_mask]
    elevated_indices = np.where(elevated_mask)[0]
    keep_elevated_idx, counts, decisions_log = process_elevated_points(elevated_xyz, elevated_indices, ground_z)

    keep_mask = np.zeros(len(xyz), dtype=bool)
    keep_mask[keep_elevated_idx] = True

    log("\n[4/4] Saving vertical-only point cloud...")
    out_path = os.path.join(OUTPUT_DIR, f"vertical_only_{part_name}.laz")
    save_laz(las_original, keep_mask, out_path)

    elapsed = time.time() - t_start
    n_in = len(xyz)
    n_out = int(keep_mask.sum())

    report = f"""
Phase 2 v3 -- Ground Stripper Report
====================================
LAZ tile: {basename}
Part: {part_name}

Input points:        {n_in:,}
Ground stripped:     {ground_mask.sum():,}  ({100*ground_mask.sum()/n_in:.1f}%)
Elevated remaining:  {elevated_mask.sum():,}  ({100*elevated_mask.sum()/n_in:.1f}%)
Final kept:          {n_out:,}  ({100*n_out/n_in:.1f}%)

Cluster decisions:
  Kept clusters:     {counts['keep']}  ({counts['keep_pts']:,} pts)
  Removed clusters:  {counts['remove']}  ({counts['remove_pts']:,} pts)

Output: {out_path}
Elapsed: {elapsed:.1f}s
"""
    log(report)
    report_path = os.path.join(OUTPUT_DIR, f"phase2_{part_name}_strip_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log("Done.\n")


if __name__ == "__main__":
    main()
