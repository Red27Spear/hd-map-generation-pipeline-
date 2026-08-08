#!/usr/bin/env python3
"""
Generalized version of gapfill_part48_from_part24.py -- same validated
method (3D ICP registration, nearest-surface-distance gap detection,
DBSCAN-validation of gap-fill candidates), parametrized to work on any
donor/recipient tile pair instead of being hardcoded to part24/part48.

The recipient should already be colorized (its own points are kept as-is
and are the base); the donor is raw geometry -- the gap-fill points this
script outputs are NOT yet colorized (colorize_pointcloud.py handles that
separately, restricted to whichever pass images are appropriate).

Usage:
    python gapfill_recipient_from_donor.py <donor_raw_laz> <recipient_colorized_laz> <out_gapfill_laz>
"""

import sys
import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def log(msg):
    print(msg, flush=True)


ICP_SUBSAMPLE = 200_000
ICP_REJECT_M = 1.5
GAP_DIST_M = 0.40
REF_VOXEL_M = 0.10
DBSCAN_VOXEL_M = 0.15
DBSCAN_EPS_VOXELS = 2.0
DBSCAN_MIN_SAMPLES = 5


def load_xyz(path):
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    return las, xyz


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
    cells = np.floor(xyz / voxel).astype(np.int64) - origin_cell
    return (cells[:, 0].astype(np.int64) << 42) | (cells[:, 1].astype(np.int64) << 21) | cells[:, 2].astype(np.int64)


def voxel_downsample(xyz, voxel):
    origin_cell = np.floor(xyz.min(axis=0) / voxel).astype(np.int64)
    keys = voxel_keys_packed(xyz, voxel, origin_cell)
    _, first_idx = np.unique(keys, return_index=True)
    return xyz[first_idx]


def find_gap_points(donor_xyz, recipient_xyz, gap_dist_m, ref_voxel=REF_VOXEL_M, voxel=DBSCAN_VOXEL_M):
    recipient_ref = voxel_downsample(recipient_xyz, ref_voxel)
    tree = cKDTree(recipient_ref)
    dists, _ = tree.query(donor_xyz, workers=-1)
    missing = dists > gap_dist_m
    donor_missing = donor_xyz[missing]
    donor_keys = voxel_keys_packed(donor_missing, voxel, np.floor(donor_missing.min(axis=0) / voxel).astype(np.int64)) \
        if len(donor_missing) else np.array([], dtype=np.int64)
    return donor_missing, donor_keys, missing


def dbscan_validate_by_voxel(points, keys, eps_voxels=DBSCAN_EPS_VOXELS, min_samples=DBSCAN_MIN_SAMPLES, voxel=DBSCAN_VOXEL_M):
    if len(points) == 0:
        return points, np.array([], dtype=bool)
    uniq_keys, first_idx = np.unique(keys, return_index=True)
    rep_points = points[first_idx]
    labels = DBSCAN(eps=eps_voxels * voxel, min_samples=min_samples).fit_predict(rep_points)
    good_keys = np.sort(uniq_keys[labels != -1])
    keep = np.isin(keys, good_keys, assume_unique=False)
    return points[keep], keep


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python gapfill_recipient_from_donor.py <donor_raw_laz> <recipient_colorized_laz> <out_gapfill_laz> [--transform <transform.json>]")
        sys.exit(1)
    FILE_DONOR, FILE_RECIPIENT, OUT_GAPFILL_LAZ = sys.argv[1], sys.argv[2], sys.argv[3]
    PRECOMPUTED_TRANSFORM = None
    if "--transform" in sys.argv:
        import json
        with open(sys.argv[sys.argv.index("--transform") + 1]) as f:
            PRECOMPUTED_TRANSFORM = json.load(f)

    log(f"Loading recipient (colorized, defines the output box + is the base): {FILE_RECIPIENT}")
    las_recip, recip_full = load_xyz(FILE_RECIPIENT)
    log(f"  recipient: {len(recip_full):,} points")

    log(f"Loading donor (raw, dense): {FILE_DONOR}")
    las_donor, donor_full = load_xyz(FILE_DONOR)
    log(f"  donor: {len(donor_full):,} points")

    RECIP_EMIN, RECIP_EMAX = recip_full[:, 0].min(), recip_full[:, 0].max()
    RECIP_NMIN, RECIP_NMAX = recip_full[:, 1].min(), recip_full[:, 1].max()
    RECIP_ZMIN, RECIP_ZMAX = recip_full[:, 2].min(), recip_full[:, 2].max()
    log(f"  recipient exact bbox: E[{RECIP_EMIN:.3f},{RECIP_EMAX:.3f}] N[{RECIP_NMIN:.3f},{RECIP_NMAX:.3f}] Z[{RECIP_ZMIN:.3f},{RECIP_ZMAX:.3f}]")

    OV_EMIN = max(RECIP_EMIN, donor_full[:, 0].min())
    OV_EMAX = min(RECIP_EMAX, donor_full[:, 0].max())
    OV_NMIN = max(RECIP_NMIN, donor_full[:, 1].min())
    OV_NMAX = min(RECIP_NMAX, donor_full[:, 1].max())
    OV_ZMIN = max(RECIP_ZMIN, donor_full[:, 2].min())
    OV_ZMAX = min(RECIP_ZMAX, donor_full[:, 2].max())

    ov_recip = ((recip_full[:, 0] >= OV_EMIN) & (recip_full[:, 0] <= OV_EMAX) &
                (recip_full[:, 1] >= OV_NMIN) & (recip_full[:, 1] <= OV_NMAX) &
                (recip_full[:, 2] >= OV_ZMIN) & (recip_full[:, 2] <= OV_ZMAX))
    ov_donor = ((donor_full[:, 0] >= OV_EMIN) & (donor_full[:, 0] <= OV_EMAX) &
                (donor_full[:, 1] >= OV_NMIN) & (donor_full[:, 1] <= OV_NMAX) &
                (donor_full[:, 2] >= OV_ZMIN) & (donor_full[:, 2] <= OV_ZMAX))
    recip_ov, donor_ov = recip_full[ov_recip], donor_full[ov_donor]
    log(f"  overlap region used for registration: recipient={len(recip_ov):,} pts, donor={len(donor_ov):,} pts")

    if PRECOMPUTED_TRANSFORM is not None:
        log(f"Using precomputed vertical-structure-based transform "
            f"(inlier_frac={PRECOMPUTED_TRANSFORM['inlier_frac']:.2f}, "
            f"translation={PRECOMPUTED_TRANSFORM['translation_cm']:.1f}cm, "
            f"rotation={PRECOMPUTED_TRANSFORM['rotation_deg']:.3f}deg) instead of dense-cloud ICP")
        origin = np.array(PRECOMPUTED_TRANSFORM["origin"])
        R = np.array(PRECOMPUTED_TRANSFORM["R"])
        t = np.array(PRECOMPUTED_TRANSFORM["t"])
        del donor_ov, recip_ov
    else:
        origin = recip_ov.mean(axis=0)
        recip_ov_c = recip_ov - origin
        donor_ov_c = donor_ov - origin

        rng = np.random.default_rng(0)
        r_idx = rng.choice(len(recip_ov_c), min(ICP_SUBSAMPLE, len(recip_ov_c)), replace=False)
        d_idx = rng.choice(len(donor_ov_c), min(ICP_SUBSAMPLE, len(donor_ov_c)), replace=False)

        log("Running 3D ICP registration (donor -> recipient frame)...")
        R, t, rmse, frac = icp_3d(donor_ov_c[d_idx], recip_ov_c[r_idx])
        shift = np.linalg.norm(t)
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        log(f"  final RMSE={rmse*100:.1f}cm  inlier_frac={frac:.2f}  "
            f"translation={shift*100:.1f}cm  rotation={angle_deg:.3f}deg")
        del donor_ov_c, recip_ov_c, donor_ov

    log("Registering all of donor and hard-cropping to recipient's exact box...")
    donor_full_aligned = (donor_full - origin) @ R.T + t + origin
    in_box = ((donor_full_aligned[:, 0] >= RECIP_EMIN) & (donor_full_aligned[:, 0] <= RECIP_EMAX) &
              (donor_full_aligned[:, 1] >= RECIP_NMIN) & (donor_full_aligned[:, 1] <= RECIP_NMAX) &
              (donor_full_aligned[:, 2] >= RECIP_ZMIN) & (donor_full_aligned[:, 2] <= RECIP_ZMAX))
    donor_in_box = donor_full_aligned[in_box]
    log(f"  donor total: {len(donor_full):,}  in recipient's box: {len(donor_in_box):,}")
    del donor_full, donor_full_aligned

    log(f"Finding occlusion gaps in recipient (nearest-surface distance > {GAP_DIST_M}m)...")
    gap_candidates, gap_keys, missing_mask = find_gap_points(donor_in_box, recip_full, GAP_DIST_M)
    log(f"  candidate gap-fill points from donor: {len(gap_candidates):,} "
        f"({100*len(gap_candidates)/max(len(donor_in_box),1):.1f}% of donor-in-box)")

    log("Validating gap-fill candidates with DBSCAN (one point per voxel, then expanded back)...")
    gap_valid, keep_mask = dbscan_validate_by_voxel(gap_candidates, gap_keys)
    log(f"  gap-fill kept after DBSCAN: {len(gap_valid):,} "
        f"({100*len(gap_valid)/max(len(gap_candidates),1):.1f}% of candidates)")

    log(f"\nrecipient (colorized) own points: {len(recip_full):,}")
    log(f"genuine gap-fill points added from donor: {len(gap_valid):,} "
        f"(+{100*len(gap_valid)/len(recip_full):.2f}%)")

    hdr = laspy.LasHeader(point_format=las_donor.point_format, version=las_donor.header.version)
    hdr.offsets = las_donor.header.offsets
    hdr.scales = las_donor.header.scales
    out_las = laspy.LasData(hdr)
    out_las.x, out_las.y, out_las.z = gap_valid[:, 0], gap_valid[:, 1], gap_valid[:, 2]
    out_las.write(OUT_GAPFILL_LAZ)
    log(f"\nSaved (gap-fill points only, not yet colorized): {OUT_GAPFILL_LAZ}")
