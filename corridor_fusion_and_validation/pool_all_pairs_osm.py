#!/usr/bin/env python3
"""
Pools raw pole positions from all successfully-processed revisit pairs
(part24/48, part23/49, part25/47, part22/50 -- part21/51 and part24/47 were
abandoned, weak/untrustworthy registration) and re-runs the entity-level
OSM comparison ONCE over the combined set.

This is not a naive average of each pair's own summary stats -- 6 of the
corridor's OSM entities are matched by more than one pair (adjacent tiles
share geography), so averaging per-pair distances would double-count them.
Instead, every pole from every pair is pooled first, then grouped by
nearest OSM entity a single time, so a shared entity's distance reflects
the true centroid of ALL poles (from every pair) assigned to it.

Usage:
    python pool_all_pairs_osm.py --osm-json cached_overpass_response.json \
        --poles part24/48:path/to/part48_pointcloud_poles.json \
        --poles part23/49:path/to/part49_pointcloud_poles.json \
        [...] --out pooled_corridor_osm_comparison.json

--osm-json is a cached response from the corridor's own Overpass query
(see this stage's README, "Fetching OSM reference data", for the exact
query and bug it works around). No demo multi-pair corridor data is
bundled in this repo, so at least one --poles and --osm-json must be
supplied.
"""

import argparse
import os
import json
import math
from pyproj import Transformer

DEDUP_DIST_M = 1.5

to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_osm_entities(osm_json_path):
    d = json.load(open(osm_json_path))
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
        pairs_here = sorted(set(m["pole"]["_pair"] for m in members))
        summary_entities.append({
            "osm_type": osm_type, "osm_id": osm_id, "n_poles": len(members),
            "dist_to_osm_m": round(dist, 2), "source_pairs": pairs_here,
            "n_source_pairs": len(pairs_here),
        })
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c_lon, c_lat]},
                          "properties": {"source": "entity_centroid", "osm_type": osm_type, "osm_id": osm_id,
                                         "n_poles": len(members), "dist_to_osm_m": round(dist, 2)}})
        features.append({"type": "Feature", "geometry": {"type": "Point",
                          "coordinates": [g["osm"]["lon"], g["osm"]["lat"]]},
                          "properties": {"source": "osm_entity", "osm_type": osm_type, "osm_id": osm_id}})

    dists = sorted(e["dist_to_osm_m"] for e in summary_entities)
    n_shared = sum(1 for e in summary_entities if e["n_source_pairs"] > 1)
    summary = {
        "label": label,
        "n_distinct_poles": len(poles),
        "n_osm_entities_matched": len(summary_entities),
        "n_entities_matched_by_multiple_pairs": n_shared,
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poles", action="append", required=True, dest="pole_files",
                     help='one per revisit pair: "pair_name:path/to/pointcloud_poles.json"')
    ap.add_argument("--osm-json", required=True, help="cached Overpass API response for the corridor bbox")
    ap.add_argument("--out", default="./output/report/pooled_corridor_osm_comparison.json")
    ap.add_argument("--out-geojson", default=None, help="defaults to --out with a .geojson extension")
    ap.add_argument("--abandoned-pairs", nargs="*", default=[],
                     help="pair names attempted but excluded for weak registration, recorded for the record only")
    args = ap.parse_args()
    pole_files = dict(spec.split(":", 1) for spec in args.pole_files)
    out_geojson = args.out_geojson or os.path.splitext(args.out)[0] + ".geojson"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[1/3] Loading and pooling poles from {len(pole_files)} pairs...")
    all_poles = []
    for pair_name, path in pole_files.items():
        poles = json.load(open(path))
        for p in poles:
            p["_pair"] = pair_name
        all_poles.extend(poles)
        print(f"  {pair_name}: {len(poles)} poles ({sum(1 for p in poles if p.get('dual_pass'))} dual-pass)")
    print(f"  Total pooled: {len(all_poles)} poles")

    print("[2/3] Deduping and fetching OSM entities...")
    poles_dedup = dedup(all_poles)
    print(f"  {len(poles_dedup)} distinct poles after dedup")
    osm_entities = load_osm_entities(args.osm_json)
    print(f"  {len(osm_entities)} OSM entities in pooled corridor area")

    print("[3/3] Comparing (all well-represented) and (dual-pass only)...")
    summary_all, feats_all = compare(poles_dedup, osm_entities, f"pooled corridor: all well-represented poles, {len(pole_files)} pairs")

    dual_only = [p for p in poles_dedup if p.get("dual_pass")]
    summary_dual, feats_dual = compare(dual_only, osm_entities, f"pooled corridor: dual-pass-only poles, {len(pole_files)} pairs")

    out = {"all_well_represented": summary_all, "dual_pass_only": summary_dual,
           "source_pairs": list(pole_files.keys()), "abandoned_pairs": args.abandoned_pairs}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    geo = {"type": "FeatureCollection", "summary": summary_all, "features": feats_all}
    with open(out_geojson, "w") as f:
        json.dump(geo, f, indent=2)

    print()
    print(json.dumps({"all_well_represented": summary_all["distance_stats_m"] | {"n_entities": summary_all["n_osm_entities_matched"], "n_poles": summary_all["n_distinct_poles"]},
                       "dual_pass_only": summary_dual["distance_stats_m"] | {"n_entities": summary_dual["n_osm_entities_matched"], "n_poles": summary_dual["n_distinct_poles"]}}, indent=2))
    print(f"\nSaved: {args.out}")
    print(f"Saved: {out_geojson}")
