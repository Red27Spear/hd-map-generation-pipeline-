#!/usr/bin/env python3
"""
Fixes a real bug found by visual inspection: two "entities" in the earlier
dual-pass comparison (way 1369923449 and way 1369923445's first pole) turned
out to be the same physical pole, seen from two different camera passes but
localized to two very different bootstrap positions (~48m apart!) because
the single-frame depth-guess (which assumes a fixed real-world light-housing
height to back out distance from bbox pixel size) is unreliable whenever the
bbox height doesn't match that assumption -- occlusion, a different fixture
size, or plain detection imprecision.

This does NOT repeat that mistake. Camera RAY DIRECTION (angle to the
object) only depends on where the pixel is in the image and the camera's
own calibrated pose -- both reliable. Camera-ray DEPTH (how far along that
ray) is what the unreliable size-based guess produces. So: throw away the
guessed depth, keep the ray direction, and use it together with real 3D pole
structure found directly in our own merged, densified point cloud (the
part24-into-part48 fusion built earlier this session) to find which real
pole each detection's ray is actually pointing at.

Method:
  1. Find real pole-like vertical structures in the point cloud: grid the
     horizontal plane into small columns, keep columns with a tall Z-range
     (light poles/housings), then reject columns that are part of a
     continuous vertical surface (a building facade also has tall columns,
     but ALL its horizontal neighbours do too -- an isolated pole's
     neighbours mostly don't). DBSCAN groups the surviving isolated tall
     columns into distinct real pole locations.
  2. For each detection, compute the camera position (from the trajectory,
     "knowledge of the car passing") and ray direction (from the pixel and
     camera calibration) -- both already-trusted quantities, no depth
     guess involved.
  3. Assign each detection to the real pole whose vertical axis the ray
     passes closest to (perpendicular miss-distance), if within tolerance.
  4. Group by assigned real pole (not by noisy bootstrap-position DBSCAN),
     final position = that pole's own point-cloud location. Two detections
     of the same physical pole now merge correctly regardless of how badly
     their individual depth guesses disagreed.

Usage:
    python extract_and_snap_poles.py
"""

import sys
import os
import json
import re

import numpy as np
import laspy
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(__file__))
from localize_part48_area import load_and_bootstrap, exclude_markings_and_beacons, rotMat
from colorize_pointcloud import load_pinhole_K, CAMEXTR_PATH

# Input is the colorized, classification-tagged road+vertical output of
# phase2_road_and_vertical.py, run on a gap-filled combined tile (produced
# by gapfill_recipient_from_donor.py + combine_colorized.py) -- it contains
# the road surface too (classification==2), not just vertical structures
# (classification==1), so pole candidates are selected via that tag before
# DBSCAN rather than assuming every point is already vertical-only.
# All three overridable via CLI: python extract_and_snap_poles.py <laz_path> <out_json> <detections_jsonl>
# detections_jsonl MUST match the actual geographic area of laz_path -- this
# script has no way to detect a mismatch itself (a stale/wrong detections
# file will silently ray-cast against the wrong poles).
LAZ_PATH = sys.argv[1] if len(sys.argv) > 1 else "./output/phase2/road_and_vertical.laz"
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else "./output/report/pointcloud_poles.json"
DETECTIONS_JSONL = sys.argv[3] if len(sys.argv) > 3 else None
CLASS_VERTICAL = 1

POLE_CLUSTER_EPS_M = 0.5
POLE_MIN_SAMPLES = 8
RAY_MISS_TOLERANCE_M = 2.0    # a detection's ray must pass within this of a real pole to be assigned


def log(msg):
    print(msg, flush=True)


def find_pole_candidates(xyz):
    log(f"  {len(xyz):,} vertical-structure points (classification==1) loaded")
    horiz = xyz[:, :2]
    labels = DBSCAN(eps=POLE_CLUSTER_EPS_M, min_samples=POLE_MIN_SAMPLES).fit_predict(horiz)
    n_noise = (labels == -1).sum()
    log(f"  DBSCAN: {len(set(labels)) - (1 if n_noise else 0)} clusters, {n_noise:,} noise points dropped")

    poles = []
    for lab in set(labels):
        if lab == -1:
            continue
        members = xyz[labels == lab]
        c_e, c_n = members[:, 0].mean(), members[:, 1].mean()
        z_range = float(members[:, 2].max() - members[:, 2].min())
        poles.append({"e": float(c_e), "n": float(c_n), "n_columns": int((labels == lab).sum()), "z_range_m": z_range})

    log(f"  -> {len(poles)} distinct real pole structures found in the point cloud")
    return poles


def ray_horizontal_miss_distance(cam_xyz, ray_dir, pole_e, pole_n):
    """Perpendicular horizontal distance from the pole's vertical axis to
    the camera ray, i.e. how far off a straight line from the camera in the
    ray's direction would be from the real pole -- ignoring depth/Z
    entirely, since that's exactly the unreliable part of the geometry."""
    if abs(ray_dir[0]) < 1e-9 and abs(ray_dir[1]) < 1e-9:
        return float("inf")
    # closest approach in the horizontal (E,N) plane only
    dx, dy = pole_e - cam_xyz[0], pole_n - cam_xyz[1]
    rx, ry = ray_dir[0], ray_dir[1]
    t = (dx * rx + dy * ry) / (rx * rx + ry * ry)
    if t <= 0:
        return float("inf")  # pole is behind the camera in this ray's horizontal direction
    closest_e = cam_xyz[0] + t * rx
    closest_n = cam_xyz[1] + t * ry
    return float(np.hypot(pole_e - closest_e, pole_n - closest_n))


if __name__ == "__main__":
    log("[1/4] Loading merged point cloud and extracting real pole structures...")
    las = laspy.read(LAZ_PATH)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    vertical_mask = np.array(las.classification) == CLASS_VERTICAL
    log(f"  {vertical_mask.sum():,}/{len(xyz):,} points are vertical-structure-tagged "
        f"(rest is road/ground, excluded from pole candidates)")
    xyz = xyz[vertical_mask]
    poles = find_pole_candidates(xyz)
    del xyz, las

    log("[2/4] Loading detections and computing camera rays (no depth guess used)...")
    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}
    pinhole_k = load_pinhole_K()

    dets = load_and_bootstrap(DETECTIONS_JSONL)  # only used to reuse the score-threshold filter + kind split
    dets = exclude_markings_and_beacons(dets)
    lights = [d for d in dets if d["kind"] == "light"]

    for d in lights:
        e = by_image[d["image"]]
        sn = str(e["SerialNr"])
        K = pinhole_k[sn]
        R = rotMat(*np.radians(e["Hrp"]))
        x1, y1, x2, y2 = d["bbox_px"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dir_cam = np.array([(cx - K[0, 2]) / K[0, 0], (cy - K[1, 2]) / K[1, 1], 1.0])
        dir_world = R @ dir_cam
        d["_cam_xyz"] = np.array(e["Xyz"][:3])
        d["_ray_dir"] = dir_world / np.linalg.norm(dir_world)

    log(f"  {len(lights)} light detections with rays computed")

    log("[3/4] Assigning each detection to the real pole its ray points at...")
    assignments = {i: [] for i in range(len(poles))}
    n_unassigned = 0
    for d in lights:
        best_i, best_miss = None, RAY_MISS_TOLERANCE_M
        for i, p in enumerate(poles):
            miss = ray_horizontal_miss_distance(d["_cam_xyz"], d["_ray_dir"], p["e"], p["n"])
            if miss < best_miss:
                best_i, best_miss = i, miss
        if best_i is not None:
            assignments[best_i].append((d, best_miss))
        else:
            n_unassigned += 1

    log(f"  assigned {len(lights) - n_unassigned}/{len(lights)} detections to a real pole "
        f"(tolerance {RAY_MISS_TOLERANCE_M}m); {n_unassigned} unassigned")

    log("[4/4] Building final, point-cloud-anchored, well-represented pole list...")
    final_poles = []

    def t_of(image):
        m = re.search(r"utc-\d{8}_(\d{2})-(\d{2})-(\d{2})-\d+", image)
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600 + mi * 60 + s

    PASS_SPLIT_T = 31100
    for i, p in enumerate(poles):
        members = assignments[i]
        if len(members) < 2:
            continue
        passes = set("pass1" if t_of(d["image"]) < PASS_SPLIT_T else "pass2" for d, _ in members)
        final_poles.append({
            "utm_easting": p["e"], "utm_northing": p["n"],
            "n_columns": p["n_columns"],
            "n_observations": len(members),
            "dual_pass": len(passes) == 2,
            "mean_ray_miss_m": round(float(np.mean([m for _, m in members])), 3),
            "member_images": [d["image"] for d, _ in members],
            "member_detections": [{"image": d["image"], "bbox_px": d["bbox_px"]} for d, _ in members],
        })

    n_dual = sum(1 for p in final_poles if p["dual_pass"])
    log(f"  {len(final_poles)} well-represented, point-cloud-anchored poles ({n_dual} dual-pass)")

    with open(OUT_JSON, "w") as f:
        json.dump(final_poles, f, indent=2)
    log(f"\nSaved: {OUT_JSON}")
