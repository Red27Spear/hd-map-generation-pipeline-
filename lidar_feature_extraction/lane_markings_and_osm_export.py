#!/usr/bin/env python3
"""
Phase 5 — Lane Marking Extraction + Lanelet2/OSM-XML Assembly
==============================================================
Two sub-steps in one script:

Step A — Lane Marking Extraction (from LiDAR intensity)
  - RANSAC road segmentation
  - Intensity threshold on road points (top 7% = white paint)
  - DBSCAN clustering of high-intensity road points
  - Geometry classification: dashed line / solid line / stop line / symbol
  - PCA polyline fitting per cluster
  - Output: lane_markings_<part>.geojson

Step B — Lanelet2/OSM-XML Assembly
  - Coordinate conversion EPSG:25832 → EPSG:4326 (WGS84)
  - Lane marking polylines → OSM ways
  - Signs → OSM nodes with German StVO tags (DE:xxx)
  - Traffic lights → OSM nodes (highway=traffic_signals)
  - Poles → OSM nodes (man_made=pole)
  - UQ fields (lidar_confidence, position_uncertainty_m) → hd: tags
  - Output: hd_map_<part>.osm  (valid Lanelet2/OSM-XML)

Usage:
    python lane_markings_and_osm_export.py <laz_file> [--phase3-dir ../camera_feature_extraction/output] [--output-dir ./output]

Outputs (in --output-dir):
    lane_markings_<part>.geojson
    hd_map_<part>.osm
    phase5_<part>_report.txt

Dependencies:
    pip install pyproj scikit-learn requests laspy
"""

import argparse
import os
import json
import requests
import time
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

import numpy as np
import laspy

# RANSAC (same parameters as Phase 3)
RANSAC_DIST_THRESH = 0.15
RANSAC_N_POINTS    = 3
RANSAC_ITERATIONS  = 2000

# Lane marking extraction
MARKING_INTENSITY_PERCENTILE = 93   # top 7% of road points = white paint
MARKING_DBSCAN_EPS           = 0.20 # metres — tight, lane lines are narrow
MARKING_DBSCAN_MIN_PTS       = 8
MARKING_MIN_POINTS           = 15   # minimum points to consider a cluster
MARKING_MAX_WIDTH_M          = 1.20 # wider than this = zebra/stop, not lane line
MARKING_MIN_LENGTH_M         = 0.30 # shorter than this = noise/symbol
SOLID_MIN_LENGTH_M           = 3.0  # solid lines are long continuous stripes
DASHED_MAX_LENGTH_M          = 3.5  # dashed segments are short
STOP_LINE_MIN_WIDTH_M        = 0.25 # stop lines are wide
ZEBRA_MIN_WIDTH_M            = 0.35 # zebra crossings are wide

# Polyline simplification
POLYLINE_SIMPLIFY_TOL = 0.10        # metres — Douglas-Peucker tolerance

# Coordinate system
EPSG_UTM   = "EPSG:25832"
EPSG_WGS84 = "EPSG:4326"

# OSM ID counter (negative = new elements not from OSM)
_osm_id_counter = -1

def next_osm_id():
    global _osm_id_counter
    _osm_id_counter -= 1
    return _osm_id_counter

# ── German StVO sign code mapping (76-class → DE:xxx) ─────────────────────────
# Source: Straßenverkehrs-Ordnung (StVO) Anlage 1-4
STVO_CODES = {
    # Warning signs (Gefahrzeichen) — triangular, red border
    "warning--roadworks":                        "DE:123",
    "warning--roundabout":                       "DE:215",   # Kreisverkehr ahead
    "warning--road-bump":                        "DE:138",
    "warning--uneven-road":                      "DE:112",
    "warning--slippery-road-surface":            "DE:114",
    "warning--curve-right":                      "DE:105",
    "warning--curve-left":                       "DE:105",
    "warning--double-curve-first-right":         "DE:107",
    "warning--double-curve-first-left":          "DE:107",
    "warning--pedestrians-crossing":             "DE:101",   # Fußgänger
    "warning--children":                         "DE:136",
    "warning--crossroads":                       "DE:102",
    "warning--junction-with-a-side-road-perpendicular-right": "DE:102",
    "warning--junction-with-a-side-road-perpendicular-left":  "DE:102",
    "warning--road-narrows":                     "DE:120",
    "warning--road-narrows-right":               "DE:121",
    "warning--road-narrows-left":                "DE:122",
    "warning--other-danger":                     "DE:101",
    "warning--traffic-signals":                  "DE:131",
    "warning--railroad-crossing-with-barriers":  "DE:150",
    "warning--railroad-crossing-without-barriers":"DE:151",
    "warning--falling-rocks-or-debris-right":    "DE:125",
    "warning--falling-rocks-or-debris-left":     "DE:125",
    "warning--wild-animals":                     "DE:142",
    "warning--domestic-animals":                 "DE:145",
    "warning--traffic-merges-right":             "DE:123",
    "warning--traffic-merges-left":              "DE:123",
    "warning--road-slope-right":                 "DE:108",
    "warning--road-slope-left":                  "DE:108",
    "warning--road-dip":                         "DE:112",
    "warning--bicycles-crossing":                "DE:138",
    "warning--delineator":                       "DE:625",
    "warning--delineator--right":                "DE:625-10",
    "warning--delineator--left":                 "DE:625-20",
    # Regulatory — prohibition (Verbote) — circular, red border
    "regulatory--no-entry":                      "DE:267",
    "regulatory--no-parking":                    "DE:286",
    "regulatory--no-right-turn":                 "DE:267",   # no-right-turn
    "regulatory--no-left-turn":                  "DE:267",
    "regulatory--no-overtaking":                 "DE:276",
    "regulatory--no-u-turn":                     "DE:272",
    "regulatory--no-bicycles":                   "DE:254",
    "regulatory--no-pedestrians":                "DE:259",
    "regulatory--no-motor-vehicles":             "DE:251",
    "regulatory--no-buses":                      "DE:257",
    "regulatory--no-goods-vehicles":             "DE:253",
    "regulatory--no-goods-vehicles-exceeding-limit": "DE:253",
    "regulatory--no-heavy-goods-vehicles":       "DE:253",
    "regulatory--maximum-speed-limit":           "DE:274",
    "regulatory--height-limit":                  "DE:265",
    "regulatory--weight-limit":                  "DE:263",
    "regulatory--axel-mass-limit":               "DE:263",
    # Regulatory — obligation (Gebote) — circular, blue
    "regulatory--keep-right":                    "DE:222",
    "regulatory--keep-left":                     "DE:222",
    "regulatory--go-straight":                   "DE:209",
    "regulatory--turn-right":                    "DE:211",
    "regulatory--turn-left":                     "DE:211",
    "regulatory--go-straight-or-turn-right":     "DE:214",
    "regulatory--go-straight-or-turn-left":      "DE:214",
    "regulatory--turn-left-or-right":            "DE:214",
    "regulatory--pass-on-either-side":           "DE:222",
    "regulatory--bicycles-only":                 "DE:237",
    "regulatory--pedestrians-only":              "DE:239",
    "regulatory--shared-path-pedestrians-and-bicycles": "DE:240",
    "regulatory--roundabout":                    "DE:215",
    # Priority signs
    "regulatory--yield":                         "DE:205",
    "regulatory--yield-to-oncoming-traffic":     "DE:208",
    "regulatory--stop":                          "DE:206",
    # Information signs (blue rectangle)
    "information--parking":                      "DE:314",
    "information--hospital":                     "DE:358",
    "information--gas-station":                  "DE:365",
    "information--motorway":                     "DE:330",
    "information--disabled-persons":             "DE:314",
    "information--tram-bus-stop":                "DE:224",
    # Complementary
    "complementary--chevron-left":               "DE:625-20",
    "complementary--chevron-right":              "DE:625-10",
    "complementary--distance":                   "DE:1004",
}

def get_stvo_code(specific_class):
    return STVO_CODES.get(specific_class, "DE:999")  # 999 = unknown


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
def make_transformer():
    from pyproj import Transformer
    return Transformer.from_crs(EPSG_UTM, EPSG_WGS84, always_xy=True)

def utm_to_wgs84(transformer, easting, northing):
    """Convert UTM EPSG:25832 to WGS84 lon/lat."""
    lon, lat = transformer.transform(easting, northing)
    return lat, lon  # OSM uses lat/lon order


# ═══════════════════════════════════════════════════════════════════════════════
# STEP A — LANE MARKING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def load_laz(laz_path):
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    intensity = np.array(las.intensity, dtype=np.float32)
    print(f"    {len(xyz):,} points loaded")
    return xyz, intensity


def ransac_road_segmentation(xyz):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH,
        ransac_n=RANSAC_N_POINTS,
        num_iterations=RANSAC_ITERATIONS,
    )
    road_mask = np.zeros(len(xyz), dtype=bool)
    road_mask[inliers] = True
    print(f"    Road points: {road_mask.sum():,} / {len(xyz):,} ({100*road_mask.sum()/len(xyz):.1f}%)")
    return road_mask


def douglas_peucker(points, tolerance):
    """Simplify a polyline using Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return np.array([start, end])
    line_unit = line_vec / line_len
    # Distance from each point to the line
    vecs = points - start
    proj = np.dot(vecs, line_unit)
    perp = vecs - np.outer(proj, line_unit)
    dists = np.linalg.norm(perp, axis=1)
    max_idx = np.argmax(dists)
    if dists[max_idx] > tolerance:
        left  = douglas_peucker(points[:max_idx+1], tolerance)
        right = douglas_peucker(points[max_idx:], tolerance)
        return np.vstack([left[:-1], right])
    return np.array([start, end])


def fit_polyline(pts_2d):
    """
    Fit an ordered polyline to a cluster of 2D points.
    Uses PCA to find the principal axis, then sorts points along it.
    """
    centroid = pts_2d.mean(axis=0)
    centred  = pts_2d - centroid
    cov = np.cov(centred.T)
    if cov.ndim < 2:
        return pts_2d
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]  # dominant direction
    proj = centred @ principal
    order = np.argsort(proj)
    ordered = pts_2d[order]
    # Simplify
    simplified = douglas_peucker(ordered, POLYLINE_SIMPLIFY_TOL)
    return simplified


def classify_marking(pts_2d, centroid):
    """
    Classify a lane marking cluster by its geometry.
    Returns: marking_type, length_m, width_m
    """
    centred  = pts_2d - centroid
    cov      = np.cov(centred.T)
    if cov.ndim < 2:
        return "unknown", 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    minor_axis = eigvecs[:, np.argmin(eigvals)]

    proj_major = centred @ major_axis
    proj_minor = centred @ minor_axis
    length_m = float(proj_major.max() - proj_major.min())
    width_m  = float(proj_minor.max() - proj_minor.min())

    if length_m < MARKING_MIN_LENGTH_M:
        return "symbol", length_m, width_m

    if width_m >= ZEBRA_MIN_WIDTH_M and length_m < 5.0:
        return "stop_line", length_m, width_m

    if width_m >= STOP_LINE_MIN_WIDTH_M:
        return "stop_line", length_m, width_m

    if length_m >= SOLID_MIN_LENGTH_M:
        return "solid", length_m, width_m

    if length_m <= DASHED_MAX_LENGTH_M:
        return "dashed", length_m, width_m

    return "solid", length_m, width_m


def extract_lane_markings(xyz, intensity, road_mask):
    """
    Extract lane marking polylines from high-intensity road points.

    Returns list of dicts:
      {type, length_m, width_m, n_points, confidence, polyline: [[e,n,z],...]}
    """
    from sklearn.cluster import DBSCAN

    road_xyz = xyz[road_mask]
    road_int = intensity[road_mask]

    # Intensity threshold — top 7% of road points = white paint
    threshold = np.percentile(road_int, MARKING_INTENSITY_PERCENTILE)
    marking_mask = road_int >= threshold
    marking_xyz  = road_xyz[marking_mask]
    print(f"    Road marking candidates: {marking_mask.sum():,} pts "
          f"(intensity >= {threshold:.0f})")

    if len(marking_xyz) < MARKING_MIN_POINTS:
        print("    WARNING: Too few marking candidates — check intensity threshold")
        return []

    # DBSCAN in 2D (project to road plane, ignore Z variation)
    marking_2d = marking_xyz[:, :2]
    db = DBSCAN(eps=MARKING_DBSCAN_EPS, min_samples=MARKING_DBSCAN_MIN_PTS, n_jobs=-1)
    labels = db.fit_predict(marking_2d)

    n_clusters = labels.max() + 1
    print(f"    DBSCAN found {n_clusters} marking clusters")

    markings = []
    n_symbol = 0
    n_noise  = 0

    for lbl in range(n_clusters):
        cmask = labels == lbl
        if cmask.sum() < MARKING_MIN_POINTS:
            n_noise += 1
            continue

        cpts_2d = marking_2d[cmask]
        cpts_3d = marking_xyz[cmask]
        centroid = cpts_2d.mean(axis=0)

        mtype, length_m, width_m = classify_marking(cpts_2d, centroid)

        if mtype == "symbol":
            n_symbol += 1
            continue  # skip arrows, cycle symbols etc — not useful for lanelets

        polyline_2d = fit_polyline(cpts_2d)
        z_mean = float(cpts_3d[:, 2].mean())

        # Attach mean Z to all polyline points
        polyline_3d = [[float(p[0]), float(p[1]), z_mean] for p in polyline_2d]

        # Confidence: based on point density relative to length
        expected_pts = length_m / 0.05  # ~20 pts/m for dense MoRo scan
        confidence = round(min(1.0, cmask.sum() / max(expected_pts, 1)), 3)

        markings.append({
            "marking_type": mtype,
            "length_m":     round(length_m, 2),
            "width_m":      round(width_m, 3),
            "n_points":     int(cmask.sum()),
            "confidence":   confidence,
            "polyline":     polyline_3d,
        })

    print(f"    Markings extracted: {len(markings)} "
          f"({sum(1 for m in markings if m['marking_type']=='solid')} solid, "
          f"{sum(1 for m in markings if m['marking_type']=='dashed')} dashed, "
          f"{sum(1 for m in markings if m['marking_type']=='stop_line')} stop_lines)")
    print(f"    Symbols skipped: {n_symbol}  Noise: {n_noise}")
    return markings


def write_markings_geojson(markings, out_path):
    features = []
    for m in markings:
        coords = m["polyline"]
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "marking_type": m["marking_type"],
                "length_m":     m["length_m"],
                "width_m":      m["width_m"],
                "n_points":     m["n_points"],
                "confidence":   m["confidence"],
            },
        })

    fc = {
        "type": "FeatureCollection",
        "name": "lane_markings",
        "crs": {"type": "name", "properties": {"name": EPSG_UTM}},
        "features": features,
    }
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"    Wrote {len(features)} lane markings → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP B — LANELET2 / OSM-XML ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════
def load_geojson(path):
    if not os.path.isfile(path):
        print(f"    WARNING: {path} not found — skipping")
        return []
    with open(path) as f:
        gj = json.load(f)
    return gj.get("features", [])
def download_osm_features(transformer, xyz):
    xmin = xyz[:, 0].min()
    xmax = xyz[:, 0].max()
    ymin = xyz[:, 1].min()
    ymax = xyz[:, 1].max()

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    query = f"""
    [out:json][timeout:25];
    (
      node["highway"="traffic_signals"]({lat_min},{lon_min},{lat_max},{lon_max});
      node["traffic_sign"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "hd_map_p06/1.0"}
        )
        response.raise_for_status()
        osm = response.json()
        print(f"    Downloaded {len(osm.get('elements', []))} OSM objects")
        return osm

    except Exception as e:
        print(f"    WARNING: OSM download failed: {e}")
        return {"elements": []}

def build_osm_xml(markings, signs, lights, poles, osm_features, transformer, part_name):
    osm_elements = osm_features.get("elements", []) if osm_features else []
    """
    Build a Lanelet2-compatible OSM-XML ElementTree.

    Structure:
      <osm version="0.6">
        <node id="-N" lat="..." lon="..."> ... </node>   ← marking vertices
        <node id="-N" lat="..." lon="..."> ... </node>   ← signs / lights / poles
        <way id="-N">                                    ← lane marking polylines
          <nd ref="-N"/>
          ...
          <tag k="..." v="..."/>
        </way>
      </osm>
    """
    root = ET.Element("osm", version="0.6", generator="hd_map_p06_phase5")

    # ── Meta note ──
    note = ET.SubElement(root, "note")
    note.text = (f"HD map tile {part_name} — generated by Phase 5 pipeline. "
                 f"CRS: {EPSG_UTM} projected to {EPSG_WGS84}. "
                 f"Uncertainty fields prefixed hd: are non-standard Lanelet2 extensions.")

    def add_tag(parent, k, v):
        ET.SubElement(parent, "tag", k=str(k), v=str(v))

    def make_node(lat, lon, tags=None):
        nid = next_osm_id()
        node = ET.SubElement(root, "node", id=str(nid),
                             lat=f"{lat:.8f}", lon=f"{lon:.8f}",
                             version="1", action="modify")
        if tags:
            for k, v in tags.items():
                add_tag(node, k, v)
        return nid

    # ── Lane marking ways ──
    n_ways = 0
    for m in markings:
        coords = m["polyline"]
        if len(coords) < 2:
            continue

        node_refs = []
        for pt in coords:
            lat, lon = utm_to_wgs84(transformer, pt[0], pt[1])
            nid = make_node(lat, lon)
            node_refs.append(nid)

        way_id = next_osm_id()
        way = ET.SubElement(root, "way", id=str(way_id), version="1", action="modify")
        for ref in node_refs:
            ET.SubElement(way, "nd", ref=str(ref))

        add_tag(way, "lane_marking",  "yes")
        add_tag(way, "marking_type",  m["marking_type"])
        add_tag(way, "length_m",      str(m["length_m"]))
        add_tag(way, "width_m",       str(m["width_m"]))
        add_tag(way, "confidence",    str(m["confidence"]))
        add_tag(way, "source",        "lidar_intensity_phase5")

        # Lanelet2 type attribute
        if m["marking_type"] in ("solid", "dashed"):
            add_tag(way, "type",    "line_thin")
            add_tag(way, "subtype", m["marking_type"])
        elif m["marking_type"] == "stop_line":
            add_tag(way, "type",    "stop_line")
            add_tag(way, "subtype", "solid")

        n_ways += 1

    print(f"    Lane marking ways written: {n_ways}")

    # ── Sign nodes ──
    n_signs = 0
    for feat in signs:
        coords = feat["geometry"]["coordinates"]
        props  = feat["properties"]
        lat, lon = utm_to_wgs84(transformer, coords[0], coords[1])

        specific_class = props.get("specific_class", "unknown")
        stvo_code      = get_stvo_code(specific_class)
        label          = props.get("specific_label", "UNKNOWN")
        

        tags = {
            "type":                     "traffic_sign",
            "traffic_sign":             stvo_code,
            "traffic_sign:class":       specific_class,
            "traffic_sign:label":       label,
            "height_above_ground_m":    str(round(props.get("height_above_ground", 0), 2)),
            "source":                   "lidar_phase3_v5",
            "hd:lidar_confidence":      str(props.get("lidar_confidence", 0)),
            "hd:position_uncertainty_m": str(props.get("position_uncertainty_m", 0)),
            "hd:classification_uncertainty": str(props.get("classification_uncertainty", 0)),
            "hd:n_view_baseline_m":     str(props.get("n_view_baseline_m", 0)),
            "hd:classifier_conf":       str(round(props.get("classifier_conf", 0), 4)),
            "hd:n_detections":          str(props.get("n_detections", 0)),
            "hd:source_method": props.get("source_method", "lidar_verified"),
            
        }
        make_node(lat, lon, tags)
        n_signs += 1

    print(f"    Sign nodes written: {n_signs}")

    # ── Traffic light nodes ──
    n_lights = 0
    for feat in lights:
        coords = feat["geometry"]["coordinates"]
        props  = feat["properties"]
        lat, lon = utm_to_wgs84(transformer, coords[0], coords[1])

        tags = {
            "highway":                  "traffic_signals",
            "traffic_signals":          "signal",
            "height_above_ground_m":    str(round(props.get("height_above_ground", 0), 2)),
            "source":                   "lidar_phase3_v5",
            "hd:lidar_confidence":      str(props.get("lidar_confidence", 0)),
            "hd:position_uncertainty_m": str(props.get("position_uncertainty_m", 0)),
            "hd:n_view_baseline_m":     str(props.get("n_view_baseline_m", 0)),
            "hd:n_detections":          str(props.get("n_detections", 0)),
            "hd:source_method": props.get("source_method", "lidar_verified"),
        }
        make_node(lat, lon, tags)
        n_lights += 1

    print(f"    Traffic light nodes written: {n_lights}")

    # ── Pole nodes ──
    n_poles = 0
    for feat in poles:
        coords = feat["geometry"]["coordinates"]
        props  = feat["properties"]
        lat, lon = utm_to_wgs84(transformer, coords[0], coords[1])

        tags = {
            "man_made":              "pole",
            "height_above_ground_m": str(round(props.get("height_above_ground", 0), 2)),
            "source":                "lidar_phase3_v5",
        }
        make_node(lat, lon, tags)
        n_poles += 1

    print(f"    Pole nodes written: {n_poles}")

    return root


def pretty_xml(root):
    raw = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(raw)
    return dom.toprettyxml(indent="  ", encoding=None)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Lane marking extraction + Lanelet2/OSM-XML assembly")
    parser.add_argument("laz_file", help="input LAZ tile (ideally Phase-2-cleaned)")
    parser.add_argument("--phase3-dir", default="../camera_feature_extraction/output",
                         help="directory with signs_3d_<part>.geojson etc. from the feature-extraction step")
    parser.add_argument("--output-dir", default="./output", help="where to write the OSM + geojson output")
    args = parser.parse_args()

    laz_path = args.laz_file
    PHASE3_DIR = args.phase3_dir
    OUTPUT_DIR = args.output_dir
    if not os.path.isfile(laz_path):
        print(f"ERROR: LAZ file not found: {laz_path}")
        return

    basename = os.path.basename(laz_path)
    part_match = re.search(r'(part\d+)', basename)
    part_name = part_match.group(1) if part_match else basename.replace('.laz', '')

    print(f"\n{'='*70}")
    print(f"  PHASE 5 — Lane Markings + Lanelet2/OSM-XML Assembly")
    print(f"  LAZ tile : {basename}")
    print(f"  Part     : {part_name}")
    print(f"{'='*70}\n")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ───────── STEP A ─────────
    print("[A/2] Lane marking extraction...")
    print("  [A.1] Loading point cloud...")
    xyz, intensity = load_laz(laz_path)

    print("  [A.2] RANSAC road segmentation...")
    road_mask = ransac_road_segmentation(xyz)

    print("  [A.3] Extracting lane markings...")
    markings = extract_lane_markings(xyz, intensity, road_mask)

    markings_path = os.path.join(OUTPUT_DIR, f"lane_markings_{part_name}.geojson")
    write_markings_geojson(markings, markings_path)

    # ───────── STEP B ─────────
    print("\n[B/2] OSM-XML assembly...")

    print("  [B.1] Loading Phase 3 GeoJSON outputs...")
    signs  = load_geojson(os.path.join(PHASE3_DIR, f"signs_3d_{part_name}.geojson"))
    lights = load_geojson(os.path.join(PHASE3_DIR, f"traffic_lights_3d_{part_name}.geojson"))
    poles  = load_geojson(os.path.join(PHASE3_DIR, f"poles_3d_{part_name}.geojson"))

    print(f"    {len(signs)} signs, {len(lights)} lights, {len(poles)} poles")

    print("  [B.2] Coordinate transformer init...")
    transformer = make_transformer()

    print("  [B.2.1] Downloading OSM features...")
    osm_features = download_osm_features(transformer, xyz)

    print(f"    OSM objects: {len(osm_features.get('elements', []))}")

    print("  [B.3] Building OSM-XML...")

    root = build_osm_xml(
        markings,
        signs,
        lights,
        poles,
        osm_features,
        transformer,
        part_name
    )

    osm_path = os.path.join(OUTPUT_DIR, f"hd_map_{part_name}.osm")
    xml_str = pretty_xml(root)

    with open(osm_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"    Wrote OSM → {osm_path}")

    # ───────── REPORT ─────────
    elapsed = time.time() - t_start

    report_text = f"""
Phase 5 Report
=====================
LAZ tile: {basename}
Part: {part_name}

Lane markings: {len(markings)}
Signs: {len(signs)}
Lights: {len(lights)}
Poles: {len(poles)}
OSM objects: {len(osm_features.get('elements', []))}

Output: {osm_path}
Elapsed: {elapsed:.2f}s
"""

    print(report_text)

    report_path = os.path.join(OUTPUT_DIR, f"phase5_{part_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report → {report_path}")
    print("Done.")

if __name__ == "__main__":
    main()