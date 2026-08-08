#!/usr/bin/env python3
"""
OSM comparison for the point-cloud-anchored, ray-snapped pole list produced
by extract_and_snap_poles.py (v3 filter run: h/w>=2.0, width<=1.20m ground
stripping, then real-pole DBSCAN + ray-to-pole assignment -- no depth guess).

Same entity-grouping methodology as compare_part48_area_osm.py (group by
nearest OSM traffic_signals node/way, not the reverse), applied to BOTH the
full well-represented pole set and the dual-pass-only subset (established
earlier this session as the more trustworthy subset, since same-pass-only
observations come from a weak, near-parallel triangulation baseline).

Also reports, per final entity, the min/max camera-to-pole 3D distance
across its member detections (i.e. how close/far the car was when each
detection was made), computed directly from the trajectory + point-cloud
pole position -- no pixel-size depth guess involved.

Usage:
    python compare_pointcloud_poles_osm.py
"""

import sys
import os
import json
import re
import math

import numpy as np
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from colorize_pointcloud import CAMEXTR_PATH

# All four overridable via CLI: python compare_pointcloud_poles_osm.py <poles_json> <osm_json> <out_json> <out_geojson>
POLES_JSON = sys.argv[1] if len(sys.argv) > 1 else "./output/report/pointcloud_poles.json"
OSM_JSON = sys.argv[2] if len(sys.argv) > 2 else "./output/report/osm_signals.json"
OUT_JSON = sys.argv[3] if len(sys.argv) > 3 else "./output/report/pointcloud_poles_osm_comparison.json"
OUT_GEOJSON = sys.argv[4] if len(sys.argv) > 4 else "./output/report/pointcloud_poles_osm_comparison.geojson"

DEDUP_DIST_M = 1.5

to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_osm_entities():
    """Only real traffic-signal features. The `>;` recursion in the Overpass
    query used to fetch OSM_JSON also pulls in every way-member node
    (curbs, path vertices, etc. with no traffic_signals tag at all) --
    counting those as entities would silently corrupt the nearest-entity
    match, so tags are checked explicitly rather than trusting element type
    alone."""
    d = json.load(open(OSM_JSON))
    entities = []
    for e in d["elements"]:
        tags = e.get("tags", {})
        is_signal = tags.get("highway") == "traffic_signals" or tags.get("crossing") == "traffic_signals"
        if not is_signal:
            continue
        if e["type"] == "node":
            entities.append({"osm_type": "node", "osm_id": e["id"], "lon": e["lon"], "lat": e["lat"]})
        elif e["type"] == "way" and "geometry" in e:
            lons = [n["lon"] for n in e["geometry"]]
            lats = [n["lat"] for n in e["geometry"]]
            entities.append({"osm_type": "way", "osm_id": e["id"],
                              "lon": sum(lons) / len(lons), "lat": sum(lats) / len(lats)})
    return entities


def dedup(poles):
    kept = []
    for p in sorted(poles, key=lambda x: -x["n_observations"]):
        if any(math.hypot(p["utm_easting"] - k["utm_easting"], p["utm_northing"] - k["utm_northing"]) <= DEDUP_DIST_M
               for k in kept):
            continue
        kept.append(p)
    return kept


def compare(poles, osm_entities, label):
    groups = {}
    for p in poles:
        lon, lat = to_wgs.transform(p["utm_easting"], p["utm_northing"])
        best = min(osm_entities, key=lambda o: haversine_m(lon, lat, o["lon"], o["lat"]))
        key = (best["osm_type"], best["osm_id"])
        groups.setdefault(key, {"osm": best, "poles": []})
        groups[key]["poles"].append({"pole": p, "lon": lon, "lat": lat})

    summary_entities = []
    features = []
    for (osm_type, osm_id), g in groups.items():
        members = g["poles"]
        c_e = sum(m["pole"]["utm_easting"] for m in members) / len(members)
        c_n = sum(m["pole"]["utm_northing"] for m in members) / len(members)
        c_lon, c_lat = to_wgs.transform(c_e, c_n)
        dist = haversine_m(c_lon, c_lat, g["osm"]["lon"], g["osm"]["lat"])
        cam_dists = [d for m in members for d in m["pole"].get("cam_distances_m", [])]
        summary_entities.append({
            "osm_type": osm_type, "osm_id": osm_id, "n_poles": len(members),
            "dist_to_osm_m": round(dist, 2),
            "any_dual_pass": any(m["pole"].get("dual_pass") for m in members),
            "camera_distance_range_m": [round(min(cam_dists), 2), round(max(cam_dists), 2)] if cam_dists else None,
        })
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c_lon, c_lat]},
                          "properties": {"source": "entity_centroid", "osm_type": osm_type, "osm_id": osm_id,
                                         "n_poles": len(members), "dist_to_osm_m": round(dist, 2)}})
        features.append({"type": "Feature", "geometry": {"type": "Point",
                          "coordinates": [g["osm"]["lon"], g["osm"]["lat"]]},
                          "properties": {"source": "osm_entity", "osm_type": osm_type, "osm_id": osm_id}})
        for m in members:
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [m["lon"], m["lat"]]},
                              "properties": {"source": "our_pole_pointcloud", "n_observations": m["pole"]["n_observations"],
                                             "dual_pass": m["pole"].get("dual_pass"),
                                             "mean_ray_miss_m": m["pole"].get("mean_ray_miss_m")}})

    dists = sorted(e["dist_to_osm_m"] for e in summary_entities)
    summary = {
        "label": label,
        "n_distinct_poles": len(poles),
        "n_osm_entities_matched": len(summary_entities),
        "entities": summary_entities,
        "distance_stats_m": {
            "mean": round(sum(dists) / len(dists), 2) if dists else None,
            "median": round(dists[len(dists) // 2], 2) if dists else None,
            "min": round(min(dists), 2) if dists else None,
            "max": round(max(dists), 2) if dists else None,
        },
    }
    return summary, features


if __name__ == "__main__":
    print("[1/3] Loading point-cloud-anchored poles and computing camera distances...")
    poles = json.load(open(POLES_JSON))
    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}

    for p in poles:
        dists = []
        pole_en = np.array([p["utm_easting"], p["utm_northing"]])
        for img in p["member_images"]:
            e = by_image.get(img)
            if e is None:
                continue
            cam_en = np.array(e["Xyz"][:2])
            dists.append(float(np.linalg.norm(cam_en - pole_en)))
        p["cam_distances_m"] = dists

    print(f"  {len(poles)} well-represented point-cloud-anchored poles loaded")

    print("[2/3] Deduping and fetching OSM entities...")
    poles_dedup = dedup(poles)
    print(f"  {len(poles_dedup)} distinct poles after dedup")
    osm_entities = load_osm_entities()
    print(f"  {len(osm_entities)} OSM entities in area")

    print("[3/3] Comparing (all well-represented), (dual-pass only), and (cross-pass-trusted only)...")
    summary_all, feats_all = compare(poles_dedup, osm_entities, "all well-represented point-cloud-anchored poles")

    dual_only = [p for p in poles_dedup if p.get("dual_pass")]
    summary_dual, feats_dual = compare(dual_only, osm_entities, "dual-pass-only point-cloud-anchored poles")

    out = {"all_well_represented": summary_all, "dual_pass_only": summary_dual}

    trusted_only = [p for p in poles_dedup if p.get("cross_pass_status") == "trusted"]
    if trusted_only:
        summary_trusted, _ = compare(trusted_only, osm_entities,
                                      "cross-pass-trusted only (independently-triangulated RMS<100px both passes)")
        out["cross_pass_trusted_only"] = summary_trusted
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    geo = {"type": "FeatureCollection", "summary": summary_all, "features": feats_all}
    with open(OUT_GEOJSON, "w") as f:
        json.dump(geo, f, indent=2)

    print()
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_GEOJSON}")
