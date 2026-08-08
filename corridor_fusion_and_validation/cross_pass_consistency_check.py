#!/usr/bin/env python3
"""
Cross-pass triangulation consistency check -- a validation axis that does
NOT touch OSM or the LiDAR point cloud at all, unlike every other
comparison run this session.

For each dual-pass pole, its pass1 and pass2 observations come from two
genuinely independent vehicle positions (~13.5 minutes apart, a separate
drive-by with a different lane position/viewing angle). Triangulating each
pass's rays SEPARATELY (the same Gauss-Newton reprojection-error refinement
used earlier this session, refine_triangulation.py) gives two independent
3D position estimates for what should be the same physical pole -- purely
from camera geometry, no LiDAR involved on either side.

How far apart the two estimates land is a PRECISION / self-consistency
signal, not an absolute-accuracy one: a shared systematic bias (e.g. a
calibration error common to all frames) would still pass this check even
though both estimates would be precisely wrong in the same way. It
complements the OSM comparison rather than replacing it.

Usage:
    python cross_pass_consistency_check.py <poles_json>
"""

import sys
import os
import re
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from refine_triangulation import refine_point
from colorize_pointcloud import load_pinhole_K, CAMEXTR_PATH
from localize_part48_area import rotMat

PASS_SPLIT_T = 31100
MIN_OBS_PER_PASS = 2

# Huber's own delta (huber_delta_px=80.0 in refine_triangulation.py) marks
# where residuals start being down-weighted as likely non-Gaussian/outlier;
# a converged RMS well above that means the pass's own rays never actually
# found a consistent 3D point (confirmed empirically: every cross-pass
# outlier >10m this session coincided with an RMS of 300-600px+ or NaN in
# at least one pass, while every RMS<100px pair agreed to within ~1.7m).
RMS_TRUST_THRESHOLD_PX = 100.0


def t_of(image):
    m = re.search(r"utc-\d{8}_(\d{2})-(\d{2})-(\d{2})-\d+", image)
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mi * 60 + s


def build_observations(detections, by_image, pinhole_k):
    obs = []
    for d in detections:
        e = by_image.get(d["image"])
        if e is None:
            continue
        sn = str(e["SerialNr"])
        K = pinhole_k.get(sn)
        if K is None:
            continue
        cam_xyz = np.array(e["Xyz"][:3])
        R = rotMat(*np.radians(e["Hrp"]))
        x1, y1, x2, y2 = d["bbox_px"]
        pixel = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        obs.append({"cam_xyz": cam_xyz, "R": R, "K": K, "pixel": pixel})
    return obs


if __name__ == "__main__":
    poles_json = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else poles_json.replace(".json", "_cross_pass_check.json")

    poles = json.load(open(poles_json))
    dual_poles = [p for p in poles if p.get("dual_pass")]
    print(f"{len(dual_poles)} dual-pass poles in input")

    if not dual_poles or "member_detections" not in dual_poles[0]:
        print("ERROR: poles JSON lacks 'member_detections' (image + bbox_px per observation) -- "
              "rerun extract_and_snap_poles.py first (updated to store this).")
        sys.exit(1)

    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}
    pinhole_k = load_pinhole_K()

    results = []
    n_skipped_insufficient = 0
    for p in dual_poles:
        dets = p["member_detections"]
        pass1_dets = [d for d in dets if t_of(d["image"]) < PASS_SPLIT_T]
        pass2_dets = [d for d in dets if t_of(d["image"]) >= PASS_SPLIT_T]

        if len(pass1_dets) < MIN_OBS_PER_PASS or len(pass2_dets) < MIN_OBS_PER_PASS:
            n_skipped_insufficient += 1
            continue

        obs1 = build_observations(pass1_dets, by_image, pinhole_k)
        obs2 = build_observations(pass2_dets, by_image, pinhole_k)
        if len(obs1) < MIN_OBS_PER_PASS or len(obs2) < MIN_OBS_PER_PASS:
            n_skipped_insufficient += 1
            continue

        X_init = np.array([p["utm_easting"], p["utm_northing"], 110.0])  # rough Z guess, refined below
        X1, diag1 = refine_point(X_init, obs1)
        X2, diag2 = refine_point(X_init, obs2)

        dist_horiz = float(np.hypot(X1[0] - X2[0], X1[1] - X2[1]))
        dist_3d = float(np.linalg.norm(X1 - X2))

        rms1, rms2 = diag1["rms_after_px"], diag2["rms_after_px"]
        both_converged = (rms1 is not None and rms2 is not None and
                           not np.isnan(rms1) and not np.isnan(rms2) and
                           rms1 < RMS_TRUST_THRESHOLD_PX and rms2 < RMS_TRUST_THRESHOLD_PX)

        results.append({
            "lidar_utm_easting": p["utm_easting"], "lidar_utm_northing": p["utm_northing"],
            "n_pass1_obs": len(obs1), "n_pass2_obs": len(obs2),
            "pass1_triangulated": X1.tolist(), "pass2_triangulated": X2.tolist(),
            "pass1_rms_after_px": rms1, "pass2_rms_after_px": rms2,
            "cross_pass_horiz_dist_m": round(dist_horiz, 3),
            "cross_pass_3d_dist_m": round(dist_3d, 3),
            "cross_pass_trusted": both_converged,
        })

    print(f"{len(results)} poles had >={MIN_OBS_PER_PASS} observations in BOTH passes "
          f"({n_skipped_insufficient} dual-pass poles skipped, insufficient per-pass observations)")

    if results:
        def stats(rs):
            ds = sorted(r["cross_pass_horiz_dist_m"] for r in rs)
            return {"n": len(ds), "mean": round(sum(ds) / len(ds), 3), "median": round(ds[len(ds) // 2], 3),
                    "min": round(min(ds), 3), "max": round(max(ds), 3)} if ds else {"n": 0}

        trusted = [r for r in results if r["cross_pass_trusted"]]
        summary = {
            "method": "Gauss-Newton reprojection-error triangulation run SEPARATELY on each pass's "
                      "observations (no LiDAR, no OSM) -- distance between the two independent "
                      "per-pass position estimates measures triangulation PRECISION/self-consistency, "
                      "not absolute accuracy. cross_pass_trusted requires BOTH passes' own "
                      f"reprojection RMS < {RMS_TRUST_THRESHOLD_PX}px after refinement -- an "
                      "untrusted pass's triangulation never actually converged, so comparing it "
                      "to the other pass is comparing a real estimate to noise.",
            "n_poles_checked": len(results),
            "n_trusted": len(trusted),
            "cross_pass_horiz_dist_stats_m_all": stats(results),
            "cross_pass_horiz_dist_stats_m_trusted_only": stats(trusted),
        }
        print(json.dumps(summary, indent=2))
        out = {"summary": summary, "poles": results}
    else:
        out = {"summary": {"n_poles_checked": 0}, "poles": []}

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_json}")
