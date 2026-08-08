#!/usr/bin/env python3
"""
Iterative reprojection-error refinement for a triangulated 3D point --
Gauss-Newton bundle-adjustment-style optimization (Jacobian of pixel
coordinates w.r.t. the 3D point, d(u,v)/dX) for our real camera model
(full 3D rotation + fx/fy/cx/cy), with cameras FIXED (only the 3D point
is refined -- our camera poses are already known from CamExtr.json, so
there's no need to refine camera positions the way full bundle
adjustment would).

localize_multiview_dbscan.py's triangulate() minimizes 3D ray-distance
(closest point to multiple lines) in one linear solve. This module
instead does Gauss-Newton minimizing actual 2D pixel reprojection error
-- the metric that's actually being measured (a detection's bbox centre
in pixels), which is what real bundle adjustment / structure-from-motion
does. Optionally applies a Huber robust kernel per observation so one
bad detection (misdetected bbox) doesn't pull the point off nearly as
much as an unweighted least-squares solve would let it.

Usage:
    python refine_triangulation.py     # runs a self-test against
                                        # Option 3's already-triangulated
                                        # lights, reports before/after
                                        # reprojection RMS
"""

import numpy as np


def project(X, cam_xyz, R, K):
    """World point X (3,) -> pixel (u,v), given camera position, rotation
    (world_dir = R @ cam_dir), and intrinsics K=[[fx,0,cx],[0,fy,cy],[0,0,1]]."""
    Xc = R.T @ (X - cam_xyz)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * Xc[0] / Xc[2] + cx
    v = fy * Xc[1] / Xc[2] + cy
    return np.array([u, v]), Xc


def reprojection_jacobian(X, cam_xyz, R, K):
    """d(u,v)/dX (2,3), full 3D-rotation + fx/fy/cx/cy camera model."""
    Xc = R.T @ (X - cam_xyz)
    fx, fy = K[0, 0], K[1, 1]
    Xc0, Xc1, Xc2 = Xc
    d_uv_d_Xc = np.array([
        [fx / Xc2, 0.0, -fx * Xc0 / Xc2 ** 2],
        [0.0, fy / Xc2, -fy * Xc1 / Xc2 ** 2],
    ])
    return d_uv_d_Xc @ R.T  # chain rule: d(Xc)/dX = R.T


def huber_weight(chi2, delta):
    """Huber robust-kernel weight: full weight inside the threshold,
    down-weighted (1/sqrt(chi2)) beyond it."""
    if abs(chi2) < delta ** 2:
        return 1.0
    return delta / np.sqrt(abs(chi2))


def refine_point(X_init, observations, iters=8, pixel_sigma=15.0, huber_delta_px=80.0):
    """observations: list of dicts with keys cam_xyz(3,), R(3,3), K(3,3),
    pixel(2,) = observed [u,v] (bbox centre). Returns refined X (3,) and
    diagnostics (reprojection RMS before/after, per-obs final weight)."""
    X = np.array(X_init, dtype=np.float64)
    W = np.eye(2) / (pixel_sigma ** 2)

    def rms(Xc):
        errs = []
        for o in observations:
            pred, Xcam = project(Xc, o["cam_xyz"], o["R"], o["K"])
            if Xcam[2] <= 0:
                continue
            errs.append(np.linalg.norm(pred - o["pixel"]))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

    rms_before = rms(X)
    weights_final = [1.0] * len(observations)

    for _ in range(iters):
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for i, o in enumerate(observations):
            pred, Xcam = project(X, o["cam_xyz"], o["R"], o["K"])
            if Xcam[2] <= 0:  # behind the camera -- not a valid observation at this X
                weights_final[i] = 0.0
                continue
            r = pred - o["pixel"]
            chi2 = r @ W @ r
            w = huber_weight(chi2, huber_delta_px)
            weights_final[i] = w
            J = reprojection_jacobian(X, o["cam_xyz"], o["R"], o["K"])
            Weff = w * W
            H += J.T @ Weff @ J
            g += J.T @ Weff @ r
        try:
            dX = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        X = X + dX
        if np.linalg.norm(dX) < 1e-5:
            break

    rms_after = rms(X)
    return X, {"rms_before_px": rms_before, "rms_after_px": rms_after, "final_weights": weights_final}


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.dirname(__file__))
    from colorize_pointcloud import load_pinhole_K, CAMEXTR_PATH

    def rotMat(roll, pitch, yaw):
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}
    pinhole_k = load_pinhole_K()

    detections_path = "/home/rohit/Downloads/Project/hd_map_p06/output/colorization/detections_part15_stvo_bucketed.jsonl"
    dets = []
    with open(detections_path) as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == "light":
                dets.append(d)

    tags = json.load(open("/home/rohit/Downloads/Project/hd_map_p06/output/colorization/part15_option3_tags.json"))
    lights = [t for t in tags if t["kind"] == "light"]

    print(f"{'seed pos':<22} {'n_obs':>6} {'rms_before(px)':>16} {'rms_after(px)':>15} {'shift(m)':>10}")
    for l in lights:
        seed = np.array([l["utm_easting"], l["utm_northing"], l["utm_z"]])
        # find raw detections plausibly belonging to this cluster: within 5m of seed's own camera-ray geometry
        # simplest robust approach: any raw light detection whose camera is within 40m and whose bbox-centre ray
        # passes within a few metres of the seed position
        obs = []
        for d in dets:
            e = by_image.get(d["image"])
            if e is None:
                continue
            cam_xyz = np.array(e["Xyz"][:3])
            if np.linalg.norm(cam_xyz[:2] - seed[:2]) > 40:
                continue
            sn = str(e["SerialNr"])
            K = pinhole_k.get(sn)
            if K is None:
                continue
            R = rotMat(*np.radians(e["Hrp"]))
            x1, y1, x2, y2 = d["bbox_px"]
            pixel = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            pred, Xcam = project(seed, cam_xyz, R, K)
            if Xcam[2] <= 0:
                continue
            if np.linalg.norm(pred - pixel) < 150:  # px gate: plausibly the same object
                obs.append({"cam_xyz": cam_xyz, "R": R, "K": K, "pixel": pixel})
        if len(obs) < 2:
            continue
        refined, diag = refine_point(seed, obs)
        shift = np.linalg.norm(refined[:2] - seed[:2])
        print(f"({seed[0]:.1f},{seed[1]:.1f})   {len(obs):6d} {diag['rms_before_px']:16.2f} "
              f"{diag['rms_after_px']:15.2f} {shift:10.3f}")
