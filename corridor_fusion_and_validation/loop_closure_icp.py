#!/usr/bin/env python3
"""
Loop-closure detection + 2D point-to-point ICP (Iterative Closest Point)
registration, for generating pose-graph loop-closure constraints from
real point-cloud submaps.

IMPORTANT, checked before building the rest of this pipeline: the part15
trajectory (config/2026_05_08_Bonn/moro.track.txt) only has 308 epochs
falling inside the part15 point-cloud tile extent, spanning 31 seconds --
the survey vehicle passed through this specific tile once and never
revisited it. There is no loop closure available *for this tile's data*,
full stop -- this isn't a gap to work around, it's the actual data. (The
broader multi-tile survey route does revisit other locations far outside
this tile, but we don't have point-cloud coverage there to register
against.)

So this module is implemented and validated for correctness (see
__main__: a real point-cloud patch, self-registered against a copy of
itself under a known synthetic transform -- standard ICP correctness
testing), but is NOT run against real part15 loop closures, because none
exist in this tile. Kept as working, reusable code for a tile that does
have internal revisits.

Usage:
    python loop_closure_icp.py     # correctness self-test only
"""

import numpy as np
from scipy.spatial import cKDTree


def transform_points(pts, T):
    x, y, theta = T
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def nearest_neighbor(src, dst_tree):
    dists, idx = dst_tree.query(src)
    return idx, dists


def estimate_transform_point_to_point(A, B):
    """Kabsch/SVD rigid transform aligning A onto B, 2D."""
    centroid_A, centroid_B = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - centroid_A, B - centroid_B
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.array([t[0], t[1], theta])


def icp_point_to_point(source, target, init=None, iters=20, reject_dist=2.0):
    """Point-to-point ICP with unknown correspondences (nearest-neighbour
    search each iteration), using a cKDTree for efficient nearest-
    neighbour lookup against real point clouds with tens of thousands of
    points, plus a distance-based outlier rejection so a partial-overlap
    real point cloud doesn't get pulled by non-overlapping area.

    IMPORTANT: real point-cloud coordinates here are raw UTM (~10^5-10^6m
    from the origin). Rotating raw UTM coordinates directly blows up any
    nonzero theta into a huge spurious displacement (a few degrees x
    10^6m is hundreds of km) -- and converting a centred-frame result
    back to the raw frame algebraically is easy to get subtly wrong when
    rotation is involved (found the hard way: an earlier version of this
    function did exactly that and was wrong by ~500km on a real test).
    So this function does NOT attempt that conversion at all -- it is the
    CALLER's responsibility to pass already locally-centred point clouds
    (subtract one shared local origin from both source and target before
    calling), and the returned T is valid ONLY in that same local frame.
    Never re-expressed in raw UTM.

    Returns (T, rmse, fraction_used) -- T = [tx, ty, theta] aligning
    source onto target, in whatever local frame source/target were
    already given in.
    """
    T = np.zeros(3) if init is None else np.array(init, dtype=np.float64)
    target_tree = cKDTree(target)
    rmse = None
    frac = 0.0
    for _ in range(iters):
        src_t = transform_points(source, T)
        idx, dists = nearest_neighbor(src_t, target_tree)
        keep = dists < reject_dist
        if keep.sum() < 10:
            break
        A = src_t[keep]
        B = target[idx[keep]]
        dT = estimate_transform_point_to_point(A, B)
        T = T + dT
        rmse = float(np.sqrt(np.mean(dists[keep] ** 2)))
        frac = float(keep.mean())
    return T, rmse, frac


def find_loop_closures(traj_en, times, spatial_r=2.0, min_dt=30.0):
    """traj_en: (N,2) easting/northing per epoch, times: (N,) seconds.
    Returns list of (i, j) candidate index pairs -- spatially close,
    temporally far apart."""
    tree = cKDTree(traj_en)
    pairs = tree.query_pairs(r=spatial_r, output_type="ndarray")
    return [(int(i), int(j)) for i, j in pairs if abs(times[i] - times[j]) > min_dt]


if __name__ == "__main__":
    import laspy

    print("Correctness self-test: real point-cloud patch, ICP recovers a known synthetic transform")
    las = laspy.read("/home/rohit/Documents/Output/phase2/clean_part15.laz")
    cls = np.asarray(las.classification)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)[cls == 1]

    center = (366419.5, 5621696.2)
    d2 = (xyz[:, 0] - center[0]) ** 2 + (xyz[:, 1] - center[1]) ** 2
    patch = xyz[d2 <= 8 ** 2][:, :2]
    print(f"  patch: {len(patch)} real road-surface points")

    # Work entirely in a locally-centred frame throughout -- both the
    # synthetic "revisit" data and the ICP call itself. Never touch raw
    # UTM coordinates with a rotation.
    local_origin = patch.mean(axis=0)
    patch_c = patch - local_origin

    true_T = np.array([0.73, -0.41, np.radians(4.0)])  # known synthetic drift to recover
    shifted_c = transform_points(patch_c, true_T)
    rng = np.random.default_rng(0)
    shifted_c += rng.normal(0, 0.01, shifted_c.shape)  # small sensor-noise-like jitter

    est_T, rmse, frac = icp_point_to_point(shifted_c, patch_c, iters=30)
    # ICP aligns shifted->patch, i.e. it estimates the INVERSE of true_T.
    # Invert est_T properly (2x2 rotation inverse + translation) for comparison.
    c, s = np.cos(-est_T[2]), np.sin(-est_T[2])
    Rinv = np.array([[c, -s], [s, c]])
    est_inv_t = Rinv @ (-est_T[:2])
    est_inv = np.array([est_inv_t[0], est_inv_t[1], -est_T[2]])
    print(f"  true synthetic transform:        dx={true_T[0]:.3f} dy={true_T[1]:.3f} dtheta={np.degrees(true_T[2]):.3f}deg")
    print(f"  ICP-recovered (inverted to match): dx={est_inv[0]:.3f} dy={est_inv[1]:.3f} dtheta={np.degrees(est_inv[2]):.3f}deg")
    print(f"  final RMSE: {rmse*1000:.2f}mm, inlier fraction: {frac:.2f}")
    err = np.hypot(est_inv[0]-true_T[0], est_inv[1]-true_T[1])
    print(f"  translation recovery error: {err*1000:.2f}mm -- {'PASS' if err < 0.02 else 'CHECK'}")
