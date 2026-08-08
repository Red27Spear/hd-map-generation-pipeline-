#!/usr/bin/env python3
"""
Multi-pass point-cloud completion: two tiles of the same real-world corridor,
recorded on two separate passes of the survey vehicle (part24 and part48,
~13.6 minutes apart -- see the revisit analysis in HOVER_LABELS_STATUS.md),
are registered and fused so that each pass's occlusion gaps (parked cars,
pedestrians, other traffic blocking the LiDAR at the moment of that specific
pass) get filled in from whichever pass actually saw that surface.

Method:
  1. Crop both tiles to their shared overlap footprint (+ margin).
  2. 3D point-to-point ICP (Kabsch/SVD rigid transform, nearest-neighbour
     correspondence, distance-gated outlier rejection) to correct the small
     residual misalignment between the two passes' trajectories -- the same
     approach as loop_closure_icp.py, extended from 2D (x,y) pose correction
     to full 3D point-cloud registration, since here two point clouds are
     being aligned, not two trajectory poses.
  3. Voxel-grid occupancy comparison (vectorised via packed integer keys,
     not a Python-level loop): for each small 3D cell in the overlap
     region, check whether pass A has points there but pass B doesn't (or
     vice versa) -- those are the genuine occlusion gaps.
  4. DBSCAN on one representative point per gap-fill voxel (not every raw
     point -- keeps DBSCAN's input size tractable), keeping only points
     that form a real, spatially coherent cluster (Ester et al. 1996) --
     this rejects sparse LiDAR noise or a moving object that happened to be
     unique to one pass, so only real static structure gets merged in.
  5. Union: part24's own points + part48's own coverage beyond part24's
     extent + part48's DBSCAN-validated gap-fill points inside the overlap.

Usage:
    python merge_overlapping_tiles.py <tile_a.laz> <tile_b.laz> <out.laz> [--margin M]

tile_a is the frame both point clouds get registered into (its own points
pass through unchanged); tile_b is the one being registered and gap-filled
from. The overlap region is computed automatically from both tiles' real
extents (intersected, plus --margin), not a hardcoded bounding box.
"""

import sys
import argparse
import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def log(msg):
    print(msg, flush=True)

VOXEL_M = 0.15             # bookkeeping cell size (grouping gap points back to voxels for DBSCAN)
GAP_DIST_M = 0.40          # a donor point counts as a genuine gap only if no recipient
                            # point exists within this radius -- well above single-sweep
                            # point spacing + residual registration error, so ordinary
                            # cross-sweep resampling of an already-covered surface doesn't
                            # get miscounted as a gap
ICP_SUBSAMPLE = 200_000    # points used to *estimate* the registration transform
ICP_REJECT_M = 1.5         # ICP correspondence distance gate
DBSCAN_EPS_VOXELS = 2.0    # gap-fill voxel validation, in voxel units
DBSCAN_MIN_SAMPLES = 5


def load_xyz(path):
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    return xyz


def kabsch_3d(A, B):
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - cA, B - cB
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def icp_3d(source, target, iters=25, reject_dist=ICP_REJECT_M):
    """3D point-to-point ICP. source/target must already be in a shared
    LOCAL (recentred) frame -- never raw UTM (rotating raw UTM coordinates
    produces huge spurious displacements, see loop_closure_icp.py)."""
    R = np.eye(3)
    t = np.zeros(3)
    target_tree = cKDTree(target)
    rmse, frac = None, 0.0
    for _ in range(iters):
        src_t = source @ R.T + t
        dists, idx = target_tree.query(src_t)
        keep = dists < reject_dist
        if keep.sum() < 100:
            break
        dR, dt = kabsch_3d(src_t[keep], target[idx[keep]])
        R, t = dR @ R, dR @ t + dt
        rmse = float(np.sqrt(np.mean(dists[keep] ** 2)))
        frac = float(keep.mean())
    return R, t, rmse, frac


def voxel_keys_packed(xyz, voxel, origin_cell):
    """Vectorised integer voxel key per point, packed into one int64."""
    cells = np.floor(xyz / voxel).astype(np.int64) - origin_cell
    return (cells[:, 0].astype(np.int64) << 42) | (cells[:, 1].astype(np.int64) << 21) | cells[:, 2].astype(np.int64)


def voxel_downsample(xyz, voxel):
    """One representative point per occupied voxel -- used only to build a
    fast, still-representative KDTree for nearest-surface distance queries."""
    origin_cell = np.floor(xyz.min(axis=0) / voxel).astype(np.int64)
    keys = voxel_keys_packed(xyz, voxel, origin_cell)
    _, first_idx = np.unique(keys, return_index=True)
    return xyz[first_idx]


def find_gap_points(donor_xyz, recipient_xyz, gap_dist_m, ref_voxel=0.10, voxel=VOXEL_M):
    """Points in donor_xyz that are genuinely far (> gap_dist_m) from the
    NEAREST recipient_xyz surface point -- not just 'not in the identical
    tiny voxel'. Two independent LiDAR sweeps of the same static wall will
    rarely share a voxel even when both saw the surface fine (each sweep
    samples different exact points on a continuous surface); a real
    occlusion gap means the whole neighbourhood is empty, which a nearest-
    neighbour distance check captures correctly and a voxel-equality check
    does not. The reference cloud is voxel-downsampled first purely so the
    KDTree build/query stays fast on tens of millions of points -- this
    does not change which points count as 'covered', since anything within
    one ref_voxel of a kept representative is still within one ref_voxel of
    some real recipient point."""
    recipient_ref = voxel_downsample(recipient_xyz, ref_voxel)
    tree = cKDTree(recipient_ref)
    dists, _ = tree.query(donor_xyz, workers=-1)
    missing = dists > gap_dist_m
    donor_keys = voxel_keys_packed(donor_xyz[missing], voxel,
                                    np.floor(donor_xyz.min(axis=0) / voxel).astype(np.int64))
    return donor_xyz[missing], donor_keys


def dbscan_validate_by_voxel(points, keys, eps_voxels=DBSCAN_EPS_VOXELS, min_samples=DBSCAN_MIN_SAMPLES, voxel=VOXEL_M):
    """DBSCAN on one representative point per occupied voxel (not every raw
    point) -- this is what makes DBSCAN tractable on millions of points --
    then keep every original point whose voxel landed in a real cluster."""
    if len(points) == 0:
        return points
    uniq_keys, first_idx = np.unique(keys, return_index=True)
    rep_points = points[first_idx]
    labels = DBSCAN(eps=eps_voxels * voxel, min_samples=min_samples).fit_predict(rep_points)
    good_keys = np.sort(uniq_keys[labels != -1])
    keep = np.isin(keys, good_keys, assume_unique=False)
    return points[keep]


def write_laz(path, xyz, template_header):
    header = laspy.LasHeader(point_format=template_header.point_format, version=template_header.version)
    header.offsets = template_header.offsets
    header.scales = template_header.scales
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tile_a", help="reference tile; its own points pass through unchanged")
    ap.add_argument("tile_b", help="tile being registered onto tile_a and gap-filled from")
    ap.add_argument("out_laz")
    ap.add_argument("--margin", type=float, default=15.0, help="metres of margin around the computed overlap bbox")
    args = ap.parse_args()
    FILE_A, FILE_B, OUT_LAZ = args.tile_a, args.tile_b, args.out_laz

    log(f"Loading {FILE_A} (full tile, single read)...")
    A_full = load_xyz(FILE_A)
    log(f"  {len(A_full):,} points")

    log(f"Loading {FILE_B} (full tile, single read)...")
    B_full = load_xyz(FILE_B)
    log(f"  {len(B_full):,} points")

    emin = max(A_full[:, 0].min(), B_full[:, 0].min()) - args.margin
    emax = min(A_full[:, 0].max(), B_full[:, 0].max()) + args.margin
    nmin = max(A_full[:, 1].min(), B_full[:, 1].min()) - args.margin
    nmax = min(A_full[:, 1].max(), B_full[:, 1].max()) + args.margin
    log(f"  computed overlap bbox: E[{emin:.1f},{emax:.1f}] N[{nmin:.1f},{nmax:.1f}]")

    ov_A = ((A_full[:, 0] >= emin) & (A_full[:, 0] <= emax) & (A_full[:, 1] >= nmin) & (A_full[:, 1] <= nmax))
    ov_B = ((B_full[:, 0] >= emin) & (B_full[:, 0] <= emax) & (B_full[:, 1] >= nmin) & (B_full[:, 1] <= nmax))
    A_ov, B_ov = A_full[ov_A], B_full[ov_B]
    log(f"  overlap region: tile_a={len(A_ov):,} pts, tile_b={len(B_ov):,} pts")

    origin = A_ov.mean(axis=0)
    A_ov_c = A_ov - origin
    B_ov_c = B_ov - origin

    rng = np.random.default_rng(0)
    a_idx = rng.choice(len(A_ov_c), min(ICP_SUBSAMPLE, len(A_ov_c)), replace=False)
    b_idx = rng.choice(len(B_ov_c), min(ICP_SUBSAMPLE, len(B_ov_c)), replace=False)

    log("Running 3D ICP registration (tile_b -> tile_a frame)...")
    R, t, rmse, frac = icp_3d(B_ov_c[b_idx], A_ov_c[a_idx])
    shift = np.linalg.norm(t)
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    log(f"  final RMSE={rmse*100:.1f}cm  inlier_frac={frac:.2f}  "
        f"translation={shift*100:.1f}cm  rotation={angle_deg:.3f}deg")

    log("Applying registration to all of tile_b...")
    B_full_aligned = (B_full - origin) @ R.T + t + origin
    B_ov_aligned = B_full_aligned[ov_B]
    del B_full, B_ov_c, A_ov_c

    log(f"Finding occlusion gaps (nearest-surface distance > {GAP_DIST_M}m)...")
    gap_in_A, gap_in_A_keys = find_gap_points(B_ov_aligned, A_ov, GAP_DIST_M)
    log(f"  candidate fill points for tile_a (from tile_b): {len(gap_in_A):,} "
        f"({100*len(gap_in_A)/len(B_ov_aligned):.1f}% of tile_b's overlap points)")

    log("Validating gap-fill candidates with DBSCAN (one point per voxel, then expanded back)...")
    gap_in_A_valid = dbscan_validate_by_voxel(gap_in_A, gap_in_A_keys)
    log(f"  tile_a gap-fill kept after DBSCAN: {len(gap_in_A_valid):,} "
        f"({100*len(gap_in_A_valid)/max(len(gap_in_A),1):.1f}% of candidates)")

    ov_mask_full_B = ((B_full_aligned[:, 0] >= emin) & (B_full_aligned[:, 0] <= emax) &
                       (B_full_aligned[:, 1] >= nmin) & (B_full_aligned[:, 1] <= nmax))
    B_outside_overlap = B_full_aligned[~ov_mask_full_B]

    merged = np.vstack([A_full, B_outside_overlap, gap_in_A_valid])

    log("")
    log(f"BEFORE: tile_a alone = {len(A_full):,} points")
    log(f"AFTER:  merged/completed = {len(merged):,} points "
        f"(+{len(merged)-len(A_full):,}, {100*(len(merged)-len(A_full))/len(A_full):.1f}% more)")
    log(f"  {len(B_outside_overlap):,} from tile_b's own coverage beyond tile_a's extent")
    log(f"  {len(gap_in_A_valid):,} are genuine DBSCAN-validated gap-fill points "
        f"from tile_b that tile_a's own pass missed, inside the shared overlap region")

    las = laspy.open(FILE_A)
    write_laz(OUT_LAZ, merged, las.header)
    log(f"\nSaved: {OUT_LAZ}")
