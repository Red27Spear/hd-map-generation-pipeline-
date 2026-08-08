#!/usr/bin/env python3
"""
Tests whether triangulation quality (mean_ray_miss_m, already computed per
pole) correlates with the multi-view geometry actually available for that
pole -- specifically the intersection angle between the two most-separated
observing cameras. A mapping vehicle driving along a street sees objects
ahead/behind it at a narrow, near-degenerate intersection angle; objects
abeam the vehicle get a much wider, better-conditioned angle. Poor
intersection angle is a known SfM failure mode independent of detection
quality, so this checks it directly rather than assuming poor solves are
"bad detections."

For each pole: look up every member detection's source image in CamExtr to
get the real camera position, take the two observing camera positions
furthest apart (the widest available baseline), and compute the angle each
makes to the pole position -- the classic intersection/parallax angle.

Usage:
    python geometric_degeneracy_analysis.py
"""

import json
import math
import numpy as np

CAMEXTR_PATH = "/home/rohit/Downloads/Project/hd_map_p06/config/2026_05_08_Bonn/CamExtr.json"
POLE_FILES = {
    "part24/48": "/home/rohit/Downloads/Project/hd_map_p06/output/report/part48_area_pointcloud_poles.json",
    "part23/49": "/home/rohit/Downloads/Project/hd_map_p06/output/report/part49_area_pointcloud_poles.json",
    "part25/47": "/home/rohit/Downloads/Project/hd_map_p06/output/report/part47_area_pointcloud_poles.json",
    "part22/50": "/home/rohit/Downloads/Project/hd_map_p06/output/report/part50_area_pointcloud_poles.json",
}
OUT_JSON = "/home/rohit/Downloads/Project/hd_map_p06/output/report/geometric_degeneracy.json"


def load_camextr_positions():
    entries = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    return {e["Image"]: np.array(e["Xyz"][:3]) for e in entries}


def intersection_angle_deg(cam_a, cam_b, pole_xyz):
    ray_a = cam_a - pole_xyz
    ray_b = cam_b - pole_xyz
    cos_a = np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


if __name__ == "__main__":
    cam_pos = load_camextr_positions()
    print(f"CamExtr: {len(cam_pos)} image positions loaded")

    rows = []
    for pair_name, path in POLE_FILES.items():
        poles = json.load(open(path))
        for p in poles:
            pole_xyz = np.array([p["utm_easting"], p["utm_northing"], 0.0])
            cams = []
            for d in p["member_detections"]:
                c = cam_pos.get(d["image"])
                if c is not None:
                    cams.append(np.array([c[0], c[1], 0.0]))
            if len(cams) < 2:
                continue
            # widest available baseline among this pole's observing cameras
            best_pair, best_dist = None, -1.0
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    dist = np.linalg.norm(cams[i] - cams[j])
                    if dist > best_dist:
                        best_dist, best_pair = dist, (cams[i], cams[j])
            baseline_m = best_dist
            angle_deg = intersection_angle_deg(best_pair[0], best_pair[1], pole_xyz)
            mean_depth = np.mean([np.linalg.norm(c - pole_xyz) for c in cams])
            rows.append({
                "pair": pair_name,
                "n_observations": p["n_observations"],
                "n_cams_matched": len(cams),
                "baseline_m": round(float(baseline_m), 2),
                "mean_depth_m": round(float(mean_depth), 2),
                "baseline_to_depth_ratio": round(float(baseline_m / mean_depth), 4),
                "intersection_angle_deg": round(float(angle_deg), 2),
                "mean_ray_miss_m": p["mean_ray_miss_m"],
                "dual_pass": p.get("dual_pass", False),
            })

    print(f"\n{len(rows)} poles with >=2 matched camera positions")

    angles = np.array([r["intersection_angle_deg"] for r in rows])
    ray_miss = np.array([r["mean_ray_miss_m"] for r in rows])
    corr = np.corrcoef(angles, ray_miss)[0, 1]
    print(f"Pearson correlation (intersection angle vs mean_ray_miss_m): {corr:.3f}")

    # split into narrow-angle (likely near-degenerate, <15 deg) vs wide-angle
    narrow = ray_miss[angles < 15]
    wide = ray_miss[angles >= 15]
    print(f"\nNarrow angle (<15deg), n={len(narrow)}: mean ray_miss={narrow.mean():.3f}m, median={np.median(narrow):.3f}m")
    print(f"Wide angle (>=15deg), n={len(wide)}: mean ray_miss={wide.mean():.3f}m, median={np.median(wide):.3f}m")

    json.dump({"rows": rows, "pearson_r": round(float(corr), 3),
               "narrow_lt15deg": {"n": int(len(narrow)), "mean_ray_miss_m": round(float(narrow.mean()), 3), "median_ray_miss_m": round(float(np.median(narrow)), 3)},
               "wide_ge15deg": {"n": int(len(wide)), "mean_ray_miss_m": round(float(wide.mean()), 3), "median_ray_miss_m": round(float(np.median(wide)), 3)}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nSaved: {OUT_JSON}")
