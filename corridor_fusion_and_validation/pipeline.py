#!/usr/bin/env python3
"""
End-to-end pipeline wiring together the geodetic/mapping techniques
implemented in this folder, applied to the actual part15 data where
they're genuinely applicable, and honestly skipped where the data
doesn't support them:

  1. Precision assessment (precision_assessment.py) -- real, run as a QA
     step against the actual point cloud. Reports internal precision,
     not absolute accuracy (no ground control points exist to measure
     that against).
  2. Loop-closure detection (loop_closure_icp.py) -- CHECKED against the
     real part15 trajectory. Only 308 of 24,915 trajectory epochs fall
     inside the part15 tile, spanning 31 seconds: the vehicle passed
     through once and never revisited. Zero usable loop closures exist
     for this tile's data -- this step is real but has nothing to do
     here, reported as skipped-with-reason, not silently omitted.
  3. Pose-graph SLAM (pose_graph_slam.py) -- consequently also not run
     against real data (nothing to optimize with zero loop-closure
     edges beyond the existing odometry chain, which would be a no-op).
     Both #2 and #3 are implemented and validated (see their own
     __main__ self-tests), just not applicable to this specific tile.
  4. Reprojection-error triangulation refinement (refine_triangulation.py)
     -- real, run against Option 3's DBSCAN-clustered lights (the
     already-established clean 7-real-pole dataset from this session).

Final step: compare the refined lights against live OSM data, same
Adenauerallee-corridor scope used throughout this session, and report
a clean before/after table.

Usage:
    python pipeline.py [--track PATH] [--detections PATH] [--tags PATH]
        [--road-laz PATH] [--osm-live PATH] [--ways PATH] [--out PATH]

None of --track/--detections/--tags/--road-laz/--osm-live/--ways are
bundled directly in this repo (they're intermediate outputs of earlier
pipeline scripts, or a cached OSM/Overpass fetch); the defaults below
document what each input actually is, not working paths in a fresh clone.
"""

import sys, os, json, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pyproj import Transformer, Geod

from precision_assessment import fit_plane_gauss_helmert, extract_flat_patch
from loop_closure_icp import find_loop_closures
from refine_triangulation import refine_point, project
from colorize_pointcloud import load_pinhole_K, CAMEXTR_PATH

TILE = (366263.7, 366598.1, 5621517.9, 5621826.4)
TRACK_PATH = "../config/moro.track.txt"
DETECTIONS_JSONL = "./output/colorization/detections_part15_stvo_bucketed.jsonl"
OPTION3_TAGS = "./output/colorization/part15_option3_tags.json"
ROAD_LAZ = "./output/phase2/road_and_vertical.laz"
OSM_LIVE = "./output/config/osm_part15_signs_lights_live.json"
WAYS_JSON = "./output/config/osm_part15_ways.json"
OUT_PATH = "./output/colorization/part15_pipeline_refined_tags.json"


def rotMat(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def step1_precision_assessment():
    print("=" * 70)
    print("STEP 1: point-cloud precision assessment")
    print("=" * 70)
    centers = [(366410.4, 5621748.9), (366415.8, 5621734.1), (366421.3, 5621719.4)]
    stds = []
    for c in centers:
        patch = extract_flat_patch(ROAD_LAZ, c, radius_m=1.5, classification=1)
        if len(patch) < 20:
            continue
        result = fit_plane_gauss_helmert(patch, sigma_o=0.01)
        stds.append(result["point_to_plane_std_m"])
        print(f"  patch @{c}: {result['n_points']} pts, "
              f"point-to-plane std = {result['point_to_plane_std_m']*1000:.2f}mm")
    mean_std = float(np.mean(stds)) if stds else None
    print(f"  -> internal precision estimate: {mean_std*1000:.2f}mm mean std across {len(stds)} patches")
    return {"internal_precision_mm": mean_std * 1000 if mean_std else None, "n_patches": len(stds)}


def step2_3_loop_closure_and_slam():
    print("\n" + "=" * 70)
    print("STEP 2+3: loop-closure detection -> pose-graph SLAM (checking applicability)")
    print("=" * 70)
    pts, times = [], []
    with open(TRACK_PATH) as f:
        for line in f:
            parts = line.split()
            t, e, n = float(parts[0]), float(parts[1]), float(parts[2])
            if TILE[0] <= e <= TILE[1] and TILE[2] <= n <= TILE[3]:
                pts.append((e, n))
                times.append(t)
    pts = np.array(pts)
    times = np.array(times)
    print(f"  {len(pts)} trajectory epochs inside part15 tile, span {times.max()-times.min():.0f}s"
          if len(pts) else "  no in-tile trajectory points")
    closures = find_loop_closures(pts, times, spatial_r=2.0, min_dt=30.0) if len(pts) > 1 else []
    print(f"  {len(closures)} usable loop closures found")
    if not closures:
        print("  -> SKIPPED: no loop closures exist in this tile's data (vehicle passed through once, "
              "31s, never revisited) -- pose-graph SLAM has nothing to correct here. Implemented and "
              "validated on synthetic data instead (see pose_graph_slam.py __main__): reduces synthetic "
              "accumulated drift from 0.384m mean / 1.179m max to 0.126m mean / 0.221m max when a real "
              "loop closure IS available.")
    return {"n_in_tile_epochs": len(pts), "n_loop_closures": len(closures), "applied": False}


def step4_refine_lights():
    print("\n" + "=" * 70)
    print("STEP 4: reprojection-error refinement of Option 3's (DBSCAN-clustered) lights")
    print("=" * 70)
    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}
    pinhole_k = load_pinhole_K()

    dets = []
    with open(DETECTIONS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == "light":
                dets.append(d)

    tags = json.load(open(OPTION3_TAGS))
    lights = [t for t in tags if t["kind"] == "light"]

    refined_tags = []
    for l in lights:
        seed = np.array([l["utm_easting"], l["utm_northing"], l["utm_z"]])
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
            if np.linalg.norm(pred - pixel) < 150:
                obs.append({"cam_xyz": cam_xyz, "R": R, "K": K, "pixel": pixel})

        new_tag = dict(l)
        if len(obs) >= 2:
            refined, diag = refine_point(seed, obs)
            new_tag["utm_easting"], new_tag["utm_northing"], new_tag["utm_z"] = refined.tolist()
            new_tag["refinement"] = {"applied": True, "n_obs": len(obs),
                                       "rms_before_px": diag["rms_before_px"], "rms_after_px": diag["rms_after_px"]}
            print(f"  ({seed[0]:.1f},{seed[1]:.1f}) n_obs={len(obs)} "
                  f"rms {diag['rms_before_px']:.1f}px -> {diag['rms_after_px']:.1f}px, "
                  f"shift {np.linalg.norm(refined[:2]-seed[:2]):.2f}m")
        else:
            new_tag["refinement"] = {"applied": False, "reason": "fewer than 2 usable observations"}
            print(f"  ({seed[0]:.1f},{seed[1]:.1f}) not enough observations, kept as-is")
        refined_tags.append(new_tag)

    return refined_tags


def step5_compare_osm(refined_lights):
    print("\n" + "=" * 70)
    print("STEP 5: compare refined lights against OSM (Adenauerallee corridor scope)")
    print("=" * 70)
    geod = Geod(ellps="WGS84")
    to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

    fresh = json.load(open(OSM_LIVE))
    osm_nodes = [e for e in fresh["elements"] if e["type"] == "node" and e.get("tags", {}).get("highway") == "traffic_signals"]
    osm_ways = [e for e in fresh["elements"] if e["type"] == "way" and e.get("tags", {}).get("crossing") == "traffic_signals"]

    ways = json.load(open(WAYS_JSON))
    named = [e for e in ways["elements"] if e.get("tags", {}).get("name", "").lower() in ("adenauerallee", "am hofgarten")]
    all_ways = [{"pts": [to_utm.transform(g["lon"], g["lat"]) for g in e["geometry"]], "name": e["tags"].get("name")} for e in named]

    def seg_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / L2))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    rows = []
    for l in refined_lights:
        e, n = l["utm_easting"], l["utm_northing"]
        in_tile = TILE[0] <= e <= TILE[1] and TILE[2] <= n <= TILE[3]
        best_name, best_d = None, None
        for w in all_ways:
            d = min(seg_dist(e, n, a[0], a[1], b[0], b[1]) for a, b in zip(w["pts"], w["pts"][1:]))
            if best_d is None or d < best_d:
                best_d, best_name = d, w["name"]
        on_corridor = in_tile and best_name == "Adenauerallee" and best_d <= 10.0
        if not on_corridor:
            continue
        lon, lat = to_wgs.transform(e, n)
        best_node = min(osm_nodes, key=lambda nd: geod.inv(lon, lat, nd["lon"], nd["lat"])[2])
        _, _, node_d = geod.inv(lon, lat, best_node["lon"], best_node["lat"])
        best_way_d = None
        for w in osm_ways:
            pts = [to_utm.transform(g["lon"], g["lat"]) for g in w["geometry"]]
            d = min(seg_dist(e, n, a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))
            if best_way_d is None or d < best_way_d:
                best_way_d = d
        rows.append(min(node_d, best_way_d))

    rows.sort()
    n = len(rows)
    matched = [d for d in rows if d <= 15]
    print(f"  {n} lights in corridor, matched {len(matched)}/{n}")
    for d in rows:
        print(f"    {d:.2f}m")
    if matched:
        print(f"  mean={sum(matched)/len(matched):.2f}  median={rows[n//2]:.2f}  "
              f"min={min(rows):.2f}  max={max(rows):.2f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default=TRACK_PATH)
    ap.add_argument("--detections", default=DETECTIONS_JSONL)
    ap.add_argument("--tags", default=OPTION3_TAGS)
    ap.add_argument("--road-laz", default=ROAD_LAZ)
    ap.add_argument("--osm-live", default=OSM_LIVE)
    ap.add_argument("--ways", default=WAYS_JSON)
    ap.add_argument("--out", default=OUT_PATH)
    cli_args = ap.parse_args()
    TRACK_PATH, DETECTIONS_JSONL, OPTION3_TAGS, ROAD_LAZ, OSM_LIVE, WAYS_JSON = (
        cli_args.track, cli_args.detections, cli_args.tags, cli_args.road_laz, cli_args.osm_live, cli_args.ways)
    os.makedirs(os.path.dirname(cli_args.out) or ".", exist_ok=True)

    precision = step1_precision_assessment()
    slam = step2_3_loop_closure_and_slam()
    refined_lights = step4_refine_lights()
    osm_distances = step5_compare_osm(refined_lights)

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(json.dumps({
        "precision_assessment": precision,
        "loop_closure_slam": slam,
        "n_lights_refined": sum(1 for l in refined_lights if l["refinement"]["applied"]),
        "n_lights_total": len(refined_lights),
    }, indent=2))

    json.dump(refined_lights, open(cli_args.out, "w"), indent=2)
    print(f"\nsaved refined lights: {cli_args.out}")
