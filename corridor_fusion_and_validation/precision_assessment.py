#!/usr/bin/env python3
"""
Point-cloud internal precision assessment via Gauss-Helmert plane fitting.

This gives *precision* (internal consistency / noise level), not *trueness*
(absolute accuracy) -- there's no ground-control-point in this project to
measure trueness against. It's a real, standard geodetic method though,
and a genuine addition to what we've had so far: an indirect ~1-2m
absolute-accuracy estimate from comparing our own GNSS trajectory to OSM
road centrelines (see HOVER_LABELS_STATUS.md), with no direct measurement
of the point cloud's own internal noise level at all.

Model: implicit plane X.n = 1 (n = unit normal / d, d = plane's distance
from origin along n) -- avoids a separate intercept parameter. Gauss-
Helmert (errors-in-observations) least squares: both the plane parameters
AND the input points themselves are adjusted, iterated to convergence.
sigma_o is the assumed isotropic per-point range noise (the same role a
manufacturer-quoted scanner-noise spec would play, e.g. a few mm for a
terrestrial laser scanner) -- we don't have a manufacturer spec for this
survey's LiDAR, so this is treated as an assumption to state, not a known
fact; the point-to-plane residual std this produces is the actual output
that matters, not sigma_o itself.

Usage:
    python precision_assessment.py
"""

import numpy as np
import laspy


def fit_plane_gauss_helmert(xyz, sigma_o=0.01, max_iter=50, tol=1e-8):
    """Iterative Gauss-Helmert plane fit (X.n = 1 parameterization).

    Parameters
    ----------
    xyz : (n, 3) array of points assumed to lie on one flat surface.
    sigma_o : assumed isotropic per-point noise (metres).

    Returns dict with the fitted normal, point-to-plane residual
    mean/std, and parameter standard deviations.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    w = 1.0 / (sigma_o ** 2)

    Xdash = xyz.mean(axis=0)
    Xc = xyz - Xdash
    C = (Xc.T @ Xc) / n
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    n_o = evecs[:, order][:, 2]
    n_o = n_o / np.linalg.norm(n_o)

    XYZ = xyz.copy()
    x_o = None
    N = None
    v = np.zeros(3 * n)
    for it in range(1, max_iter + 1):
        I = XYZ
        d = np.dot(n_o, Xdash)
        x_o = n_o / d

        A = I  # (n,3) -- d(residual)/d(plane params)
        W = -(I @ x_o - 1.0)  # discrepancy vector
        scale = w / np.dot(x_o, x_o)
        N = scale * (A.T @ A)
        U = scale * (A.T @ W)
        x = np.linalg.solve(N, U)

        # B: (n, 3n), row i = x_o repeated at columns [3i:3i+3]
        nx, ny, nz = x_o
        rows = np.repeat(np.arange(n), 3)
        cols = np.arange(3 * n)
        data = np.tile([nx, ny, nz], n)
        import scipy.sparse as sp
        B = sp.csr_matrix((data, (rows, cols)), shape=(n, 3 * n))
        v = (1.0 / w) * (B.T @ (scale * (W - A @ x)))
        v = np.asarray(v).ravel()

        x_o_updated = x_o + x
        d_updated = 1.0 / np.linalg.norm(x_o_updated)
        n_o_updated = x_o_updated * d_updated
        XYZ = XYZ + v.reshape(n, 3)
        Xdash = XYZ.mean(axis=0)

        converged = np.max(np.abs(x)) < tol
        n_o, XYZ_final_normal = n_o_updated, n_o_updated
        n_o = n_o_updated
        if converged or it == max_iter:
            break

    d = np.dot(n_o, Xdash)
    dist = xyz @ n_o - d  # point-to-plane distance of the ORIGINAL (unadjusted) points
    Q_xx = np.linalg.inv(N)
    sigma_params = np.sqrt(np.diag(Q_xx)) * d

    return {
        "n_points": n,
        "iterations": it,
        "normal": n_o.tolist(),
        "plane_offset_d": float(d),
        "point_to_plane_mean_m": float(np.mean(dist)),
        "point_to_plane_std_m": float(np.std(dist)),
        "point_to_plane_rms_m": float(np.sqrt(np.mean(dist ** 2))),
        "param_std": sigma_params.tolist(),
        "assumed_sigma_o_m": sigma_o,
    }


def extract_flat_patch(laz_path, center_en, radius_m=1.5, classification=None):
    """Pull a local, small patch of real points around (e,n) -- small
    enough that a genuinely flat real-world surface (road/facade) should
    look locally planar."""
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    if classification is not None:
        cls = np.asarray(las.classification)
        xyz = xyz[cls == classification]
    d2 = (xyz[:, 0] - center_en[0]) ** 2 + (xyz[:, 1] - center_en[1]) ** 2
    return xyz[d2 <= radius_m ** 2]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laz", default="./output/phase2/road_and_vertical.laz",
                     help="road-classified LAZ, e.g. from phase2_road_and_vertical.py or phase2_strip_ground_v3.py "
                          "run on data/part15/raw/*.laz -- not bundled in this repo directly")
    ap.add_argument("--road-class", type=int, default=1, help="LAS classification code for road-surface points")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--center", action="append", type=float, nargs=2, metavar=("EASTING", "NORTHING"), dest="centers",
                     help="one or more UTM patch centres to test; repeatable. Defaults to four mid-block points "
                          "found on this project's own part15 tile -- edit for your own data")
    args = ap.parse_args()

    centers = args.centers or [
        (366410.4, 5621748.9), (366415.8, 5621734.1),
        (366421.3, 5621719.4), (366470.0, 5621629.0),
    ]

    print(f"{'center':<22} {'n_pts':>7} {'mean(mm)':>10} {'std(mm)':>9} {'rms(mm)':>9}")
    for c in centers:
        patch = extract_flat_patch(args.laz, c, radius_m=args.radius, classification=args.road_class)
        if len(patch) < 20:
            print(f"{str(c):<22} too few points ({len(patch)})")
            continue
        result = fit_plane_gauss_helmert(patch, sigma_o=0.01)
        print(f"{str(c):<22} {result['n_points']:7d} "
              f"{result['point_to_plane_mean_m']*1000:10.2f} "
              f"{result['point_to_plane_std_m']*1000:9.2f} "
              f"{result['point_to_plane_rms_m']*1000:9.2f}")
