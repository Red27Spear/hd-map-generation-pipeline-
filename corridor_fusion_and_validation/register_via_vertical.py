#!/usr/bin/env python3
"""
Computes a donor->recipient rigid transform using ONLY vertical/pole-like
structures (phase2_strip_ground_v3.py output) instead of the full dense
point clouds -- established this session as a much better ICP correspondence
source (0.72 inlier fraction vs 0.23 for part24/48's dense-cloud
registration), since poles are compact, well-localized, unambiguous
features, unlike flat/repetitive road and building-facade surfaces that can
trap ICP in a bad local minimum.

Saves the resulting transform (R, t, origin) to JSON so a downstream script
(gapfill_recipient_from_donor.py --transform) can apply it directly to the
FULL raw donor cloud without re-running its own (weaker) dense-cloud ICP.

Usage:
    python register_via_vertical.py <donor_vertical_laz> <recipient_vertical_laz> <out_transform_json>
"""

import sys
import json
import numpy as np
import laspy
from scipy.spatial import cKDTree


def log(msg):
    print(msg, flush=True)


ICP_REJECT_M = 1.5


def load_xyz(path):
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


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


def icp_3d(source, target, iters=50, reject_dist=ICP_REJECT_M):
    R = np.eye(3)
    t = np.zeros(3)
    target_tree = cKDTree(target)
    rmse, frac = None, 0.0
    for _ in range(iters):
        src_t = source @ R.T + t
        dists, idx = target_tree.query(src_t)
        keep = dists < reject_dist
        if keep.sum() < 10:
            break
        dR, dt = kabsch_3d(src_t[keep], target[idx[keep]])
        R, t = dR @ R, dR @ t + dt
        rmse = float(np.sqrt(np.mean(dists[keep] ** 2)))
        frac = float(keep.mean())
    return R, t, rmse, frac


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python register_via_vertical.py <donor_vertical_laz> <recipient_vertical_laz> <out_transform_json>")
        sys.exit(1)
    DONOR_PATH, RECIP_PATH, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]

    log(f"Loading donor vertical: {DONOR_PATH}")
    donor = load_xyz(DONOR_PATH)
    log(f"  {len(donor):,} points")

    log(f"Loading recipient vertical: {RECIP_PATH}")
    recip = load_xyz(RECIP_PATH)
    log(f"  {len(recip):,} points")

    origin = recip.mean(axis=0)
    donor_c = donor - origin
    recip_c = recip - origin

    log("Running 3D ICP registration on vertical structures (donor -> recipient frame)...")
    R, t, rmse, frac = icp_3d(donor_c, recip_c)
    shift = np.linalg.norm(t)
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    log(f"  final RMSE={rmse*100:.1f}cm  inlier_frac={frac:.2f}  "
        f"translation={shift*100:.1f}cm  rotation={angle_deg:.3f}deg")

    out = {
        "origin": origin.tolist(),
        "R": R.tolist(),
        "t": t.tolist(),
        "rmse_cm": rmse * 100 if rmse else None,
        "inlier_frac": frac,
        "translation_cm": shift * 100,
        "rotation_deg": angle_deg,
        "n_donor_pts": len(donor),
        "n_recip_pts": len(recip),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved transform: {OUT_JSON}")
