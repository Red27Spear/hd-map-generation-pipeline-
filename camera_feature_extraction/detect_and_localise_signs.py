#!/usr/bin/env python3
"""
Phase 3 v6 — Image-First Detection + Type-Aware LiDAR Localisation
====================================================================
Rewritten to fix issues found in two draft variants (v5.1 / phase4c-adjacent
scripts) that had drifted from what was actually validated this project:

  FIXED — heading convention:
    Previous drafts used `heading_rad = radians(heading)` directly, and one
    additionally built a full 3x3 world<->camera rotation from R0/Hrp for the
    bearing ray. Neither is correct. The ONLY validated fix is:
        bearing = -radians(heading) + angle_offset
    (confirmed: trajectory travel bearing ~+24 deg vs camera Hrp[0] ~-25.5 deg
    for the same segment — equal & opposite). The full LiDAR<->camera PIXEL
    projection (needed for true cross-verification) remains UNRESOLVED — this
    script does not attempt it. Ray projection only needs the heading, not
    the full rotation, so it is unaffected by that open problem.

  FIXED — input format:
    Previous drafts assumed a `merged_<part>.laz` with a `classification`
    field (1=road, 2=vertical). That file/format was never built in this
    project. This script takes a plain LAZ (raw or Phase-2-cleaned) and does
    its own RANSAC road segmentation, same as the validated v5 pipeline.

  KEPT / IMPROVED:
    - Per-camera pinhole K lookup (better than a single global default)
    - StVO-consistent 76-class handling, class-specific real sign widths
    - Hybrid crop-vs-template signal (crop_quality via Laplacian variance)
    - Uncertainty Quantification fields (position/classification/baseline/
      lidar confidence)
    - "camera_only_unverified" fallback for detections with no LiDAR cluster
      match, instead of silently relaxing thresholds until noise passes
    - Type-aware localisation (signs=pole-like, lights=arm-mounted OK)

  NEW (flagged as heuristic / not yet empirically validated):
    - Post-localisation same-type dedup (merges duplicate detections that
      localise to the same LiDAR cluster within 1.0m)
    - Co-location Z-offset when a sign and light share one pole cluster
    - Back-face flag for signs likely facing away from the camera
    - Bare-pole `probable_sign` tagging (wide top vs shaft = possible
      unclassified sign head)

Usage:
    python detect_and_localise_signs.py <laz_file>

Requires camera_preprocessing/undistort_and_correct.py to have already been run
(this script reads its corrected frames + pinhole_K.json from PHASE0_DIR below),
and a trained sign detector + 76-class classifier (not bundled here — see the
section README for how to point CLASSIFIER_MODEL / YOLO_SIGN_MODEL at your own).

Outputs (in OUTPUT_DIR below):
    signs_3d_<part>.geojson
    traffic_lights_3d_<part>.geojson
    poles_3d_<part>.geojson
    phase3_<part>_objects.laz
    phase3_<part>_report.txt
"""

import sys, os, json, time, re, math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import laspy

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — override any of these with environment variables of the same name
# ═══════════════════════════════════════════════════════════════════════════════
PHASE0_DIR       = os.environ.get("PHASE0_DIR", "../camera_preprocessing/output")
CAMEXTR_PATH     = os.environ.get("CAMEXTR_PATH", "../config/CamExtr.json")
OUTPUT_DIR       = os.environ.get("PHASE3_OUTPUT_DIR", "./output")
CROP_OUTPUT_DIR  = os.environ.get("CROP_OUTPUT_DIR", "./output/detection_crops")

# Trained weights — not bundled in this repo (see README). Point these at your
# own sign detector / 76-class classifier, trained e.g. on GTSDB + Mapillary.
CLASSIFIER_MODEL    = os.environ.get("CLASSIFIER_MODEL", "./models/sign_classifier_76cls/best.pt")
YOLO_SIGN_MODEL     = os.environ.get("YOLO_SIGN_MODEL", "./models/sign_detector/best.pt")
YOLO_LIGHT_MODEL    = "yolov8s.pt"
YOLO_LIGHT_CLASS_ID = 9

YOLO_CONF_THRESH = 0.25
YOLO_IMGSZ       = 1280
CLASSIFIER_IMGSZ = 160   # matches sign_classifier_76cls_m_noflip's training imgsz (was 128 for the old s-model)

REAL_LIGHT_HEIGHT_M = 1.3

CROP_CONF_THRESHOLD  = 0.95
CROP_SHARP_THRESHOLD = 100.0

DEDUP_RADIUS_M = 3.0
DEDUP_PARAMS = {
    "sign":  {"min_det": 2, "max_spread": 3.0},
    "light": {"min_det": 2, "max_spread": 5.0},
}

RANSAC_DIST_THRESH = 0.15
RANSAC_N_POINTS    = 3
RANSAC_ITERATIONS  = 2000

LOCAL_DBSCAN_EPS        = 0.35
LOCAL_DBSCAN_MIN_POINTS = 5

SIGN_SEARCH_RADIUS   = 8.0
SIGN_MIN_HW_RATIO    = 1.2
SIGN_MAX_WIDTH       = 1.2
SIGN_MIN_HEIGHT      = 0.8
SIGN_MAX_GROUND_OFF  = 1.5
SIGN_MIN_CLUSTER_PTS = 12

LIGHT_SEARCH_RADIUS   = 15.0
LIGHT_MIN_HEIGHT      = 2.5
LIGHT_MAX_WIDTH       = 6.0
LIGHT_MAX_GROUND_OFF  = 8.0
LIGHT_MIN_CLUSTER_PTS = 5
LIGHT_DBSCAN_EPS      = 0.80
LIGHT_DBSCAN_MIN_PTS  = 5

POLE_MIN_HEIGHT        = 3.5
POLE_MAX_WIDTH         = 0.8
POLE_MAX_GROUND_OFFSET = 1.5

SAME_TYPE_DEDUP_DIST_M   = 1.0
CO_LOCATION_DIST_M       = 2.0
CO_LOCATION_Z_OFFSET_M   = 1.5
BACK_FACE_DOT_THRESHOLD  = -0.3
POLE_ADJACENT_RADIUS_M   = 5.0

BBOX_MARGIN_M = 20.0
ALLOW_CAMERA_ONLY_FALLBACK = True

SIGN_WIDTH_BY_CLASS = {
    "regulatory--no-entry": 0.60, "regulatory--no-parking": 0.60,
    "regulatory--no-right-turn": 0.60, "regulatory--no-left-turn": 0.60,
    "regulatory--no-overtaking": 0.60, "regulatory--no-u-turn": 0.60,
    "regulatory--no-bicycles": 0.60, "regulatory--no-pedestrians": 0.60,
    "regulatory--no-motor-vehicles": 0.60, "regulatory--no-buses": 0.60,
    "regulatory--no-goods-vehicles": 0.60,
    "regulatory--no-goods-vehicles-exceeding-limit": 0.60,
    "regulatory--no-heavy-goods-vehicles": 0.60,
    "regulatory--maximum-speed-limit": 0.60,
    "regulatory--keep-right": 0.60, "regulatory--keep-left": 0.60,
    "regulatory--go-straight": 0.60, "regulatory--turn-right": 0.60,
    "regulatory--turn-left": 0.60,
    "regulatory--go-straight-or-turn-right": 0.60,
    "regulatory--go-straight-or-turn-left": 0.60,
    "regulatory--turn-left-or-right": 0.60,
    "regulatory--pass-on-either-side": 0.60,
    "regulatory--bicycles-only": 0.60, "regulatory--pedestrians-only": 0.60,
    "regulatory--shared-path-pedestrians-and-bicycles": 0.60,
    "regulatory--roundabout": 0.60,
    "regulatory--yield": 0.90, "regulatory--yield-to-oncoming-traffic": 0.90,
    "regulatory--stop": 0.90,
    "regulatory--height-limit": 0.70, "regulatory--weight-limit": 0.70,
    "regulatory--axel-mass-limit": 0.70,
    "warning--roadworks": 0.70, "warning--roundabout": 0.70,
    "warning--road-bump": 0.70, "warning--uneven-road": 0.70,
    "warning--slippery-road-surface": 0.70, "warning--curve-right": 0.70,
    "warning--curve-left": 0.70, "warning--double-curve-first-right": 0.70,
    "warning--double-curve-first-left": 0.70,
    "warning--pedestrians-crossing": 0.70, "warning--children": 0.70,
    "warning--crossroads": 0.70,
    "warning--junction-with-a-side-road-perpendicular-right": 0.70,
    "warning--junction-with-a-side-road-perpendicular-left": 0.70,
    "warning--road-narrows": 0.70, "warning--road-narrows-right": 0.70,
    "warning--road-narrows-left": 0.70, "warning--other-danger": 0.70,
    "warning--traffic-signals": 0.70,
    "warning--railroad-crossing-with-barriers": 0.70,
    "warning--railroad-crossing-without-barriers": 0.70,
    "warning--falling-rocks-or-debris-right": 0.70,
    "warning--falling-rocks-or-debris-left": 0.70,
    "warning--wild-animals": 0.70, "warning--domestic-animals": 0.70,
    "warning--traffic-merges-right": 0.70, "warning--traffic-merges-left": 0.70,
    "warning--road-slope-right": 0.70, "warning--road-slope-left": 0.70,
    "warning--road-dip": 0.70, "warning--bicycles-crossing": 0.70,
    "warning--delineator": 0.40, "warning--delineator--right": 0.40,
    "warning--delineator--left": 0.40,
    "information--parking": 0.60, "information--hospital": 0.60,
    "information--gas-station": 0.60, "information--motorway": 0.60,
    "information--disabled-persons": 0.60, "information--tram-bus-stop": 0.60,
    "complementary--chevron-left": 0.40, "complementary--chevron-right": 0.40,
    "complementary--distance": 0.40,
}
SIGN_WIDTH_DEFAULT = 0.60

def get_sign_real_width(specific_class):
    return SIGN_WIDTH_BY_CLASS.get(specific_class, SIGN_WIDTH_DEFAULT)


CLASS_76_NAMES = [
    "regulatory--keep-right","regulatory--height-limit",
    "warning--railroad-crossing-with-barriers","warning--falling-rocks-or-debris-right",
    "regulatory--yield","warning--curve-right","regulatory--pedestrians-only",
    "warning--pedestrians-crossing","regulatory--no-entry","warning--slippery-road-surface",
    "warning--curve-left","information--parking","information--tram-bus-stop",
    "warning--crossroads","regulatory--stop","regulatory--maximum-speed-limit",
    "regulatory--turn-right","warning--roundabout","warning--road-bump",
    "warning--uneven-road","warning--railroad-crossing-without-barriers",
    "regulatory--bicycles-only","regulatory--yield-to-oncoming-traffic",
    "regulatory--shared-path-pedestrians-and-bicycles","regulatory--no-bicycles",
    "regulatory--no-pedestrians","regulatory--no-overtaking","regulatory--keep-left",
    "regulatory--go-straight","regulatory--no-parking","regulatory--no-right-turn",
    "regulatory--no-left-turn","complementary--chevron-left",
    "regulatory--no-heavy-goods-vehicles","regulatory--weight-limit",
    "regulatory--no-u-turn","warning--other-danger",
    "warning--junction-with-a-side-road-perpendicular-right",
    "warning--double-curve-first-right","regulatory--turn-left","warning--roadworks",
    "warning--children","warning--traffic-merges-right","warning--road-narrows-right",
    "information--motorway","regulatory--pass-on-either-side",
    "warning--bicycles-crossing","complementary--distance",
    "regulatory--go-straight-or-turn-right","regulatory--go-straight-or-turn-left",
    "regulatory--no-motor-vehicles","complementary--chevron-right",
    "information--disabled-persons","regulatory--no-goods-vehicles",
    "regulatory--roundabout","regulatory--no-goods-vehicles-exceeding-limit",
    "warning--traffic-signals","warning--road-narrows","warning--road-narrows-left",
    "information--gas-station","regulatory--axel-mass-limit","warning--delineator",
    "warning--wild-animals","warning--domestic-animals","regulatory--turn-left-or-right",
    "warning--road-slope-left","warning--double-curve-first-left",
    "regulatory--no-buses","warning--falling-rocks-or-debris-left",
    "warning--delineator--right","warning--delineator--left",
    "warning--junction-with-a-side-road-perpendicular-left","warning--road-dip",
    "warning--road-slope-right","warning--traffic-merges-left","information--hospital",
]

def class_to_label(cls_name):
    parts = cls_name.split("--", 1)
    return parts[1].replace("-", " ").upper() if len(parts) > 1 else cls_name.upper()


# Vienna-convention shape families used for the triangle-referenced tilt fix
# below: warning signs are apex-up equilateral triangles (a natural "spirit
# level" — their base is known to be horizontal in the real world), most
# regulatory signs are circles (rotation-invariant outline, but any internal
# glyph/number/arrow still tilts with camera roll). Stop (octagon) and yield
# (inverted triangle) are excluded from the circular family.
NON_CIRCULAR_REGULATORY = {
    "regulatory--stop", "regulatory--yield", "regulatory--yield-to-oncoming-traffic",
}

def is_triangular_class(cls_name):
    return cls_name.startswith("warning--")

def is_circular_class(cls_name):
    return cls_name.startswith("regulatory--") and cls_name not in NON_CIRCULAR_REGULATORY


def estimate_triangle_tilt_deg(crop_bgr):
    """Find the sign's triangular border in a crop and return the in-plane
    rotation (degrees, +ve = clockwise-on-screen) needed to make its base
    horizontal, or None if no stable 3-vertex contour is found. Assumes an
    apex-up triangle (standard warning-sign orientation): the base is the
    edge opposite the topmost vertex.
    """
    import cv2
    h, w = crop_bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    crop_area = float(h * w)

    grey = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    edges = cv2.Canny(grey, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.20 * crop_area or area > 0.97 * crop_area:
            continue
        peri = cv2.arcLength(cnt, True)
        tri = None
        for eps_frac in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
            if len(approx) == 3:
                tri = approx.reshape(3, 2).astype(np.float64)
                break
        if tri is None:
            continue

        apex_idx = int(np.argmin(tri[:, 1]))
        base_pts = np.delete(tri, apex_idx, axis=0)
        (bx1, by1), (bx2, by2) = base_pts
        tilt_deg = math.degrees(math.atan2(by2 - by1, bx2 - bx1))
        if tilt_deg > 90:
            tilt_deg -= 180
        elif tilt_deg < -90:
            tilt_deg += 180
        if abs(tilt_deg) > 30:
            continue
        return tilt_deg
    return None


def rotate_image_region(img, x1, y1, x2, y2, iw, ih, tilt_deg, out_w, out_h):
    """Rotate a generously-padded region of the full image around the box
    centre by tilt_deg (undoing the camera-roll tilt measured from a
    triangular sign in the same frame), then centre-crop back down to
    (out_w, out_h) so it's directly comparable to the unrotated crop."""
    import cv2
    bw, bh = x2 - x1, y2 - y1
    pad2 = int(max(bw, bh) * 0.6) + 4
    rx0, ry0 = max(0, x1 - pad2), max(0, y1 - pad2)
    rx1, ry1 = min(iw, x2 + pad2), min(ih, y2 + pad2)
    region = img[ry0:ry1, rx0:rx1]
    if region.size == 0 or min(region.shape[:2]) < 10:
        return None
    rh, rw = region.shape[:2]
    centre = (rw / 2.0, rh / 2.0)
    M = cv2.getRotationMatrix2D(centre, tilt_deg, 1.0)
    rotated = cv2.warpAffine(region, M, (rw, rh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    cx0 = max(0, int(round(rw / 2.0 - out_w / 2.0)))
    cy0 = max(0, int(round(rh / 2.0 - out_h / 2.0)))
    out = rotated[cy0:cy0 + out_h, cx0:cx0 + out_w]
    if out.size == 0 or min(out.shape[:2]) < 10:
        return None
    return out


def compute_uq(cluster_dets, best_cluster):
    est_positions = np.array([[d["est_utm"][0], d["est_utm"][1]] for d in cluster_dets])
    pos_std = float(np.std(est_positions, axis=0).mean())

    best_clf_conf = max(d.get("classifier_conf", 0.0) for d in cluster_dets)
    clf_uncertainty = round(1.0 - best_clf_conf, 4)

    cam_positions = np.array([[d["utm"][0], d["utm"][1]] for d in cluster_dets])
    if len(cam_positions) > 1:
        baseline = max(np.linalg.norm(a - b) for a, b in combinations(cam_positions, 2))
    else:
        baseline = 0.0

    if best_cluster is not None:
        n_pts = best_cluster.get("n_points", 0)
        dist  = best_cluster.get("dist_to_approx", 99.0)
        lidar_conf = round(min(1.0, (n_pts / 1000.0) * (1.0 / (1.0 + dist / 5.0))), 4)
    else:
        lidar_conf = 0.0

    return {
        "position_uncertainty_m":     round(pos_std, 3),
        "classification_uncertainty": clf_uncertainty,
        "n_view_baseline_m":          round(float(baseline), 2),
        "lidar_confidence":           lidar_conf,
    }


def load_laz(laz_path):
    print(f"  Loading {os.path.basename(laz_path)}...")
    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    intensity = np.array(las.intensity, dtype=np.float32)
    print(f"    {len(xyz):,} points")
    return xyz, intensity


def get_laz_bbox(laz_path):
    with laspy.open(laz_path) as f:
        hdr = f.header
        return {
            "e_min": hdr.mins[0] - BBOX_MARGIN_M, "e_max": hdr.maxs[0] + BBOX_MARGIN_M,
            "n_min": hdr.mins[1] - BBOX_MARGIN_M, "n_max": hdr.maxs[1] + BBOX_MARGIN_M,
        }


def ransac_road_segmentation(xyz):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH, ransac_n=RANSAC_N_POINTS,
        num_iterations=RANSAC_ITERATIONS,
    )
    road_mask = np.zeros(len(xyz), dtype=bool)
    road_mask[inliers] = True
    print(f"    Road inliers: {len(inliers):,} ({100*len(inliers)/len(xyz):.1f}%)")
    return road_mask


def build_road_height_lookup(xyz, road_mask, grid_size=2.0):
    road_pts = xyz[road_mask]
    e_min, n_min = road_pts[:, 0].min(), road_pts[:, 1].min()
    e_idx = ((road_pts[:, 0] - e_min) / grid_size).astype(int)
    n_idx = ((road_pts[:, 1] - n_min) / grid_size).astype(int)
    grid = defaultdict(list)
    for i in range(len(road_pts)):
        grid[(e_idx[i], n_idx[i])].append(road_pts[i, 2])
    grid_median = {k: np.median(v) for k, v in grid.items()}
    global_median = float(np.median(road_pts[:, 2]))

    def get_ground_z(easting, northing):
        ei = int((easting - e_min) / grid_size)
        ni = int((northing - n_min) / grid_size)
        for radius in range(5):
            for de in range(-radius, radius + 1):
                for dn in range(-radius, radius + 1):
                    z = grid_median.get((ei + de, ni + dn))
                    if z is not None:
                        return z
        return global_median

    print(f"    Ground lookup grid: {len(grid_median)} cells")
    return get_ground_z


# --- Road no-touch zone: keep placements off the actual drivable/median ---
# corridor, using the white edge-lines and the green median strip as the
# real boundary evidence (rather than trusting the coarse RANSAC road plane
# footprint, which doesn't distinguish shoulder from travel lane). Needs a
# colorized point cloud (RGB channels index-aligned with xyz) — currently
# only built for part15, so this is a no-op elsewhere until that's rolled
# out further.
COLORIZED_LAZ_CANDIDATES = {
    "part15": [
        "../data/part15/results/colorized_part15.laz",
    ],
}

def load_colorized_rgb(part_name, xyz):
    """Load red/green/blue arrays from a colorized LAZ for this part, aligned
    point-for-point with xyz. Returns None if unavailable or misaligned."""
    for path in COLORIZED_LAZ_CANDIDATES.get(part_name, []):
        if not os.path.isfile(path):
            continue
        try:
            las = laspy.read(path)
        except Exception:
            continue
        if las.header.point_count != len(xyz):
            continue
        col_xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        check_idx = np.random.default_rng(0).choice(len(xyz), min(200, len(xyz)), replace=False)
        if not np.allclose(xyz[check_idx], col_xyz[check_idx], atol=0.05):
            continue
        rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)
        print(f"    Colorized point cloud loaded for road-marking detection: {os.path.basename(path)}")
        return rgb
    print(f"    No aligned colorized point cloud found for {part_name} "
          f"— road no-touch zone will be skipped.")
    return None


def build_car_path(cam_entries):
    pts = [e["Xyz"][:2] for e in cam_entries]
    path = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - path[-1][0], p[1] - path[-1][1]) > 0.5:
            path.append(p)
    path = np.array(path, dtype=np.float64)
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seglen)])
    return path, cum_s


def project_to_path_frame(points_xy, path, cum_s, path_tree):
    """Along-track (s) / cross-track (t) coordinates of each point relative
    to the nearest path segment's local tangent frame."""
    _, idx = path_tree.query(points_xy)
    idx = np.clip(idx, 1, len(path) - 2)
    tangent = path[idx + 1] - path[idx - 1]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    delta = points_xy - path[idx]
    t = np.einsum('ij,ij->i', delta, normal)
    s = cum_s[idx] + np.einsum('ij,ij->i', delta, tangent)
    return s, t


def build_road_notouch_zone(xyz, rgb, intensity, road_mask, cam_entries, get_ground_z,
                             s_bin=2.0, fallback_halfwidth=4.5, near_path_dist=8.0):
    """
    Returns a lookup zone_at(e, n) -> (t_left, t_right) giving the cross-
    track extent (metres either side of the car path) of the true no-touch
    corridor at that along-track position: the paved travel lanes, bounded
    by the white edge-lines.

    The white lines are the only evidence used to set t_left/t_right — a
    road-side lawn or hedge is also "green" and, unlike the true central
    median, sits *outside* those lines, so letting green points widen the
    zone directly pulled in roadside vegetation the full length of the
    street. Green points are only used to confirm a median exists *inside*
    the white-line envelope that's already been measured, for reporting.
    """
    from scipy.spatial import cKDTree

    path, cum_s = build_car_path(cam_entries)
    path_tree = cKDTree(path)

    xy = xyz[:, :2]
    z = xyz[:, 2]
    near_path_mask = path_tree.query(xy, distance_upper_bound=near_path_dist)[0] < near_path_dist

    brightness = rgb.mean(axis=1)
    rgb_max = rgb.max(axis=1)
    rgb_min = rgb.min(axis=1)
    is_white = (rgb_min / (rgb_max + 1e-6) > 0.80) & (brightness > np.percentile(brightness, 80))
    white_mask = road_mask & near_path_mask & is_white & (intensity > np.percentile(intensity[road_mask], 70))

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    ground_z_at = np.array([get_ground_z(p[0], p[1]) for p in xy[near_path_mask]])
    ground_z_full = np.full(len(xyz), np.nan)
    ground_z_full[near_path_mask] = ground_z_at
    height_ag = z - ground_z_full
    is_green = (g > r * 1.12) & (g > b * 1.12) & (brightness > np.percentile(brightness, 15))
    green_mask = near_path_mask & is_green & (height_ag > -0.3) & (height_ag < 2.5)

    n_white, n_green = int(white_mask.sum()), int(green_mask.sum())
    print(f"    Road-marking evidence: {n_white:,} white edge-line pts, "
          f"{n_green:,} roadside-green pts (informational only)")

    edge_pts = xy[white_mask]
    if len(edge_pts) < 20:
        print("    Not enough white-line evidence — falling back to a fixed "
              f"+/-{fallback_halfwidth}m corridor around the car path.")
        def zone_at(e, n):
            return -fallback_halfwidth, fallback_halfwidth
        return zone_at

    s_edge, t_edge = project_to_path_frame(edge_pts, path, cum_s, path_tree)
    n_bins = int(cum_s[-1] // s_bin) + 2
    bin_idx = np.clip((s_edge // s_bin).astype(int), 0, n_bins - 1)

    t_left_bin = np.full(n_bins, np.nan)
    t_right_bin = np.full(n_bins, np.nan)
    for bi in np.unique(bin_idx):
        t_here = t_edge[bin_idx == bi]
        left_side = t_here[t_here < 0]
        right_side = t_here[t_here > 0]
        if len(left_side) >= 3:
            t_left_bin[bi] = np.percentile(left_side, 2)
        if len(right_side) >= 3:
            t_right_bin[bi] = np.percentile(right_side, 98)

    valid = ~np.isnan(t_left_bin)
    if valid.sum() >= 2:
        t_left_bin = np.interp(np.arange(n_bins), np.where(valid)[0], t_left_bin[valid])
    else:
        t_left_bin[:] = -fallback_halfwidth
    valid = ~np.isnan(t_right_bin)
    if valid.sum() >= 2:
        t_right_bin = np.interp(np.arange(n_bins), np.where(valid)[0], t_right_bin[valid])
    else:
        t_right_bin[:] = fallback_halfwidth

    if n_green >= 20:
        green_xy = xy[green_mask]
        s_g, t_g = project_to_path_frame(green_xy, path, cum_s, path_tree)
        bin_g = np.clip((s_g // s_bin).astype(int), 0, n_bins - 1)
        interior = (t_g > t_left_bin[bin_g]) & (t_g < t_right_bin[bin_g])
        n_bins_with_median = len(np.unique(bin_g[interior]))
        print(f"    Median-strip confirmation: green points found inside the "
              f"white-line envelope in {n_bins_with_median}/{len(np.unique(bin_idx))} "
              f"road bins (roadside vegetation outside the lines doesn't widen the zone).")

    def zone_at(e, n):
        pt = np.array([[e, n]])
        s, t = project_to_path_frame(pt, path, cum_s, path_tree)
        bi = int(np.clip(s[0] // s_bin, 0, n_bins - 1))
        return float(t_left_bin[bi]), float(t_right_bin[bi])

    return zone_at


def apply_road_notouch_zone(localised, zone_fn, cam_entries, margin=0.5):
    path, cum_s = build_car_path(cam_entries)
    from scipy.spatial import cKDTree
    path_tree = cKDTree(path)
    n_pushed = 0
    for obj in localised:
        e, n, zc = obj["centroid"]
        t_left, t_right = zone_fn(e, n)
        s, t = project_to_path_frame(np.array([[e, n]]), path, cum_s, path_tree)
        s, t = float(s[0]), float(t[0])
        obj["pushed_off_road"] = False
        if t_left - margin <= t <= t_right + margin:
            new_t = (t_left - margin) if abs(t - t_left) < abs(t - t_right) else (t_right + margin)
            idx = np.clip(int(np.searchsorted(cum_s, s)), 1, len(path) - 2)
            tangent = path[idx + 1] - path[idx - 1]
            tangent /= np.linalg.norm(tangent) + 1e-9
            normal = np.array([-tangent[1], tangent[0]])
            base = path[idx] + tangent * (s - cum_s[idx])
            new_e, new_n = base + normal * new_t
            obj["push_dist_m"] = round(abs(new_t - t), 2)
            obj["centroid"] = [float(new_e), float(new_n), zc]
            if "approx_utm" in obj:
                obj["approx_utm"] = [float(new_e), float(new_n), obj["approx_utm"][2]]
            obj["pushed_off_road"] = True
            n_pushed += 1
    print(f"    Pushed off road/median no-touch zone: {n_pushed}/{len(localised)}")
    return localised


def load_camextr_for_bbox(bbox):
    with open(CAMEXTR_PATH) as f:
        data = json.load(f)
    entries = data.get("Profiler_0", [])
    if not entries and isinstance(data, list):
        entries = data
    filtered = [e for e in entries
                if bbox["e_min"] <= e["Xyz"][0] <= bbox["e_max"]
                and bbox["n_min"] <= e["Xyz"][1] <= bbox["n_max"]]
    print(f"    CamExtr: {len(filtered)} images in bbox")
    return filtered


def load_pinhole_K():
    path = os.path.join(PHASE0_DIR, "phase0_pinhole_K.json")
    with open(path) as f:
        raw = json.load(f)
    K = {}
    for sn, v in raw.items():
        K[str(sn)] = {"fx": float(v["fx"]), "fy": float(v["fy"]),
                      "cx": float(v["cx"]), "cy": float(v["cy"])}
    print(f"    Loaded pinhole K for {len(K)} cameras")
    return K


def build_pointcloud_tree(xyz, road_mask, get_ground_z, min_height_ag=0.5):
    """
    2D (X,Y) KDTree over ELEVATED non-road points, for ray-density placement.

    Must filter to points genuinely above ground (min_height_ag) — a plain
    non-road mask still includes curbs, low vegetation, and other ground-
    adjacent clutter, which caused ray-marching to snap sign/light Z to
    near-ground height (height_above_ground≈0) when that clutter happened
    to sit closer along the bearing ray than the real elevated structure.
    """
    from scipy.spatial import cKDTree
    nonroad_idx = np.where(~road_mask)[0]
    nonroad_xyz = xyz[nonroad_idx]

    # Vectorized ground-height lookup per point (grid-based, cheap)
    ground_z = np.array([get_ground_z(p[0], p[1]) for p in nonroad_xyz])
    elevated_mask = (nonroad_xyz[:, 2] - ground_z) >= min_height_ag
    elevated_xyz = nonroad_xyz[elevated_mask]

    tree = cKDTree(elevated_xyz[:, :2])
    return tree, elevated_xyz


def ray_march_to_density(car_utm, bearing_rad, tree, nonroad_xyz,
                          dist_min=1.5, dist_max=25.0, step=0.25,
                          radius=0.6, min_pts=4):
    """
    Walk along the image-derived bearing ray and find the distance where
    real LiDAR point density peaks, instead of trusting the monocular
    size-based distance guess blindly.

    This is what actually fixes signs placed outside the point cloud,
    duplicate detections of the same physical sign (per-frame monocular
    distance is noisy; ray-density converges to the same real structure
    regardless), and "hovering" placement (Z comes from real matched
    points, not an assumed offset).

    Returns (distance_m, real_z, n_points) for the best-supported distance
    along the ray, or (None, None, 0) if no real structure is found
    anywhere along the ray within min_pts.
    """
    best_dist, best_count, best_idx = None, 0, None
    d = dist_min
    while d <= dist_max:
        px = car_utm[0] + d * math.sin(bearing_rad)
        py = car_utm[1] + d * math.cos(bearing_rad)
        idx = tree.query_ball_point([px, py], radius)
        if len(idx) > best_count:
            best_count, best_dist, best_idx = len(idx), d, idx
        d += step
    if best_count >= min_pts:
        matched = nonroad_xyz[best_idx]
        return best_dist, float(np.median(matched[:, 2])), best_count
    return None, None, 0


def batch_yolo_detection(cam_entries, pinhole_K, pc_tree, pc_nonroad_xyz):
    from ultralytics import YOLO
    import torch
    import cv2

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    print("    Loading sign detector...")
    sign_model = YOLO(YOLO_SIGN_MODEL).to(device)
    print("    Loading traffic light detector...")
    light_model = YOLO(YOLO_LIGHT_MODEL).to(device)
    print("    Loading 76-class sign classifier...")
    classifier = YOLO(CLASSIFIER_MODEL).to(device)

    os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

    all_detections = []
    n_images = len(cam_entries)
    n_with_signs = n_with_lights = crop_counter = 0
    n_source_rectified = 0
    default_fxcx = {"fx": 963.0, "fy": 963.0, "cx": 1230.0, "cy": 1050.0}

    for i, entry in enumerate(cam_entries):
        img_path = os.path.join(PHASE0_DIR, entry["Image"])
        if not os.path.isfile(img_path):
            continue

        serial = str(entry.get("SerialNr", ""))
        car_utm = entry["Xyz"][:3]
        heading = entry["Hrp"][0]
        cam_k = pinhole_K.get(serial, default_fxcx)
        fx, fy, cx, cy = cam_k["fx"], cam_k["fy"], cam_k["cx"], cam_k["cy"]

        img = None

        results = sign_model.predict(img_path, conf=YOLO_CONF_THRESH, imgsz=YOLO_IMGSZ,
                                     device=device, verbose=False)
        found_sign = False

        # --- Pass 1: gather candidate boxes for this image + first-pass classify ---
        candidates = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy.cpu().numpy().flatten().tolist()]
                bw, bh = x2 - x1, y2 - y1
                if bw < 18 or bh < 18:
                    continue

                if img is None:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                ih, iw = img.shape[:2]
                if (y1 + y2) / 2.0 > ih * 0.62:
                    continue
                aspect = bw / max(bh, 1)
                if aspect > 2.5 or aspect < 0.3:
                    continue

                specific_class, specific_label, classifier_conf = "unknown", "UNKNOWN", 0.0
                crop_quality = 0.0
                pad = int(max(bw, bh) * 0.15)
                cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
                cx1_, cy1_ = min(iw, x2 + pad), min(ih, y2 + pad)
                crop = img[cy0:cy1_, cx0:cx1_]

                if crop.size > 0 and min(crop.shape[:2]) >= 10:
                    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_quality = float(cv2.Laplacian(grey, cv2.CV_64F).var())
                    cls_results = classifier.predict(crop, imgsz=CLASSIFIER_IMGSZ, verbose=False)
                    if cls_results and cls_results[0].probs is not None:
                        probs = cls_results[0].probs
                        top1 = int(probs.top1)
                        classifier_conf = float(probs.top1conf)
                        if top1 < len(CLASS_76_NAMES):
                            specific_class = CLASS_76_NAMES[top1]
                            specific_label = class_to_label(specific_class)

                candidates.append({
                    "cls_id": cls_id, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bw": bw, "bh": bh, "cx0": cx0, "cy0": cy0, "cx1_": cx1_, "cy1_": cy1_,
                    "crop": crop, "specific_class": specific_class,
                    "specific_label": specific_label, "classifier_conf": classifier_conf,
                    "crop_quality": crop_quality,
                    "pass1_class": specific_class, "pass1_conf": classifier_conf,
                })

        # --- Derive a per-image camera-roll tilt from triangular (warning) signs ---
        # Triangles are apex-up by convention, so their base gives a direct
        # "spirit level" reading of the frame's roll. Circular signs are
        # rotation-invariant in outline but still carry tilted internal
        # glyphs/arrows/numbers, so the same correction is applied to them.
        tilt_angles = []
        for cand in candidates:
            if is_triangular_class(cand["specific_class"]) and cand["classifier_conf"] >= 0.5:
                t = estimate_triangle_tilt_deg(cand["crop"])
                if t is not None:
                    tilt_angles.append(t)

        image_tilt_deg = float(np.median(tilt_angles)) if tilt_angles else None
        apply_tilt = image_tilt_deg is not None and abs(image_tilt_deg) >= 0.3

        if apply_tilt:
            for cand in candidates:
                out_h, out_w = cand["crop"].shape[:2]
                rotated = rotate_image_region(img, cand["x1"], cand["y1"], cand["x2"], cand["y2"],
                                               iw, ih, image_tilt_deg, out_w, out_h)
                if rotated is None:
                    continue
                cls_results = classifier.predict(rotated, imgsz=CLASSIFIER_IMGSZ, verbose=False)
                if not (cls_results and cls_results[0].probs is not None):
                    continue
                probs = cls_results[0].probs
                top1 = int(probs.top1)
                pass2_conf = float(probs.top1conf)
                if top1 >= len(CLASS_76_NAMES):
                    continue
                pass2_class = CLASS_76_NAMES[top1]
                if pass2_conf > cand["classifier_conf"]:
                    grey2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                    cand["crop"] = rotated
                    cand["specific_class"] = pass2_class
                    cand["specific_label"] = class_to_label(pass2_class)
                    cand["classifier_conf"] = pass2_conf
                    cand["crop_quality"] = float(cv2.Laplacian(grey2, cv2.CV_64F).var())
                    cand["tilt_status"] = "corrected"
                else:
                    cand["tilt_status"] = "attempted_no_improvement"
                cand["pass2_class"] = pass2_class
                cand["pass2_conf"] = pass2_conf

            # At least one sign in this frame was verifiably classified
            # better after derotation -> the whole-frame roll estimate is
            # real, not noise. Commit the correction to the source image
            # itself (not just this run's crops) so every future pass over
            # this dataset starts from an already-level frame. Storage is
            # tight, so this replaces the file in place rather than
            # versioning it.
            if any(cand.get("tilt_status") == "corrected" for cand in candidates):
                ih2, iw2 = img.shape[:2]
                M_full = cv2.getRotationMatrix2D((iw2 / 2.0, ih2 / 2.0), image_tilt_deg, 1.0)
                img_rectified = cv2.warpAffine(img, M_full, (iw2, ih2), flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                cv2.imwrite(img_path, img_rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_source_rectified += 1

        for cand in candidates:
            cls_id = cand["cls_id"]; conf = cand["conf"]
            x1, y1, x2, y2 = cand["x1"], cand["y1"], cand["x2"], cand["y2"]
            bw = cand["bw"]
            specific_class = cand["specific_class"]
            specific_label = cand["specific_label"]
            classifier_conf = cand["classifier_conf"]
            crop_quality = cand["crop_quality"]
            crop = cand["crop"]

            real_width_m = get_sign_real_width(specific_class)
            approx_dist_m = np.clip((fx * real_width_m) / max(bw, 1), 2.0, 25.0)
            bbox_cx = (x1 + x2) / 2.0
            angle_offset_rad = math.atan2(bbox_cx - cx, fx)
            total_bearing_rad = -math.radians(heading) + angle_offset_rad

            # Correct the monocular size-based distance guess with real
            # point-cloud density along this bearing — this is what the
            # sign actually points at, not just an assumed sign-width.
            pc_dist, pc_z, pc_n = ray_march_to_density(
                car_utm, total_bearing_rad, pc_tree, pc_nonroad_xyz,
                dist_min=1.5, dist_max=25.0)
            ray_matched = pc_dist is not None
            dist_used = pc_dist if ray_matched else approx_dist_m
            z_used = pc_z if ray_matched else car_utm[2]

            est_utm = [
                car_utm[0] + dist_used * math.sin(total_bearing_rad),
                car_utm[1] + dist_used * math.cos(total_bearing_rad),
                z_used,
            ]

            heading_vec = np.array([math.sin(-math.radians(heading)),
                                    math.cos(-math.radians(heading))])
            cam_to_sign = np.array(est_utm[:2]) - np.array(car_utm[:2])
            norm = np.linalg.norm(cam_to_sign)
            likely_back_face = False
            if norm > 1e-6:
                dotp = np.dot(heading_vec, cam_to_sign / norm)
                likely_back_face = dotp < BACK_FACE_DOT_THRESHOLD

            crop_counter += 1
            crop_filename = f"crop_{crop_counter:05d}_{specific_label.replace(' ', '_')}.jpg"
            crop_path = os.path.join(CROP_OUTPUT_DIR, crop_filename)
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            use_crop_recommended = (
                classifier_conf >= CROP_CONF_THRESHOLD and
                crop_quality    >= CROP_SHARP_THRESHOLD and
                specific_class  != "unknown"
            )

            all_detections.append({
                "image": entry["Image"], "utm": car_utm, "est_utm": est_utm,
                "heading": heading, "bearing_rad": total_bearing_rad,
                "type": "sign", "class_id": cls_id, "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "specific_class": specific_class, "specific_label": specific_label,
                "classifier_conf": classifier_conf, "crop_path": crop_path,
                "crop_quality": round(crop_quality, 1),
                "use_crop_recommended": use_crop_recommended,
                "real_width_used_m": real_width_m,
                "likely_back_face": bool(likely_back_face),
                "ray_matched": ray_matched,
                "ray_matched_pts": pc_n,
                "tilt_deg_measured": round(image_tilt_deg, 2) if image_tilt_deg is not None else 0.0,
                "tilt_deg_used": round(image_tilt_deg, 2) if cand.get("tilt_status") == "corrected" else 0.0,
                "tilt_source": cand.get("tilt_status", "measured_negligible" if tilt_angles else "none"),
                "tilt_n_triangle_refs": len(tilt_angles),
                "pass1_class": cand["pass1_class"], "pass1_conf": round(cand["pass1_conf"], 3),
                "pass2_class": cand.get("pass2_class"),
                "pass2_conf": round(cand["pass2_conf"], 3) if cand.get("pass2_conf") is not None else None,
            })
            found_sign = True
        if found_sign:
            n_with_signs += 1

        results = light_model.predict(img_path, conf=YOLO_CONF_THRESH, imgsz=YOLO_IMGSZ,
                                      classes=[YOLO_LIGHT_CLASS_ID], device=device, verbose=False)
        found_light = False
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy.cpu().numpy().flatten().tolist()]
                bw, bh = x2 - x1, y2 - y1

                approx_dist_m = np.clip((fx * REAL_LIGHT_HEIGHT_M) / max(bh, 1), 3.0, 30.0)
                bbox_cx = (x1 + x2) / 2.0
                angle_offset_rad = math.atan2(bbox_cx - cx, fx)
                total_bearing_rad = -math.radians(heading) + angle_offset_rad

                pc_dist, pc_z, pc_n = ray_march_to_density(
                    car_utm, total_bearing_rad, pc_tree, pc_nonroad_xyz,
                    dist_min=2.5, dist_max=30.0)
                ray_matched = pc_dist is not None
                dist_used = pc_dist if ray_matched else approx_dist_m
                z_used = pc_z if ray_matched else (car_utm[2] + 3.0)

                est_utm = [
                    car_utm[0] + dist_used * math.sin(total_bearing_rad),
                    car_utm[1] + dist_used * math.cos(total_bearing_rad),
                    z_used,
                ]

                all_detections.append({
                    "image": entry["Image"], "utm": car_utm, "est_utm": est_utm,
                    "heading": heading, "bearing_rad": total_bearing_rad,
                    "type": "light", "class_id": YOLO_LIGHT_CLASS_ID,
                    "conf": conf, "bbox": [x1, y1, x2, y2],
                    "specific_class": "traffic-light", "specific_label": "TRAFFIC LIGHT",
                    "classifier_conf": 1.0, "crop_path": "", "crop_quality": 0.0,
                    "use_crop_recommended": False, "real_width_used_m": REAL_LIGHT_HEIGHT_M,
                    "likely_back_face": False,
                    "ray_matched": ray_matched,
                    "ray_matched_pts": pc_n,
                })
                found_light = True
        if found_light:
            n_with_lights += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_images:
            print(f"      [{i+1}/{n_images}] detections={len(all_detections)} "
                  f"(signs_in={n_with_signs} lights_in={n_with_lights} crops={crop_counter})")

    print(f"    Total raw detections: {len(all_detections)}")
    print(f"    Source images rectified in place (verified tilt correction): "
          f"{n_source_rectified}/{n_images}")
    return all_detections


def triangulate_rays(car_utms, bearings):
    """
    Multi-view least-squares ray intersection (classical multi-view
    geometry — the standard fix for monocular distance estimates, which
    depend on knowing the object's real-world size and are only as
    accurate as the classifier that guessed it). Each observation is a
    ray: origin car_utms[i] (2D), direction (sin(bearing_i), cos(bearing_i)).
    Finds the 2D point minimizing summed squared perpendicular distance
    to all rays — i.e. the best-fit intersection point, using every
    observation's DIRECTION only, never an assumed object width. Needs
    >=2 observations from angularly-distinct viewpoints (near-parallel
    rays, e.g. the same lane pass, don't meaningfully constrain a
    solution) to be numerically stable.

    Returns (point, ok) — ok=False if the rays are too close to parallel
    to trust (falls back to the caller's existing monocular estimate).
    """
    pts = np.array(car_utms, dtype=np.float64)[:, :2]
    dirs = np.array([[np.sin(b), np.cos(b)] for b in bearings], dtype=np.float64)

    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in zip(pts, dirs):
        d = d / np.linalg.norm(d)
        P = np.eye(2) - np.outer(d, d)  # projector onto the line's perpendicular subspace
        A += P
        b += P @ p

    # Degeneracy check: if all rays are nearly parallel, A is close to
    # singular (or rank-deficient) and the "intersection" is meaningless.
    eigvals = np.linalg.eigvalsh(A)
    if eigvals.min() < 0.15:  # empirically: well-separated viewpoints give eigvals well above 1
        return None, False

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, False
    return point, True


def deduplicate_detections(detections):
    by_type = defaultdict(list)
    for det in detections:
        by_type[det["type"]].append(det)

    unique_objects = []
    for obj_type, dets in by_type.items():
        if not dets:
            continue
        thresh = DEDUP_PARAMS.get(obj_type, {"min_det": 2, "max_spread": 3.0})
        positions = np.array([[d["est_utm"][0], d["est_utm"][1]] for d in dets])
        assigned = np.zeros(len(dets), dtype=bool)
        clusters = []
        for i in range(len(dets)):
            if assigned[i]:
                continue
            dists = np.linalg.norm(positions - positions[i], axis=1)
            nearby = (~assigned) & (dists <= DEDUP_RADIUS_M)
            idx = np.where(nearby)[0]
            assigned[idx] = True
            clusters.append(idx)

        n_dropped = 0
        for cluster_idx in clusters:
            cluster_dets = [dets[j] for j in cluster_idx]
            if len(cluster_dets) < thresh["min_det"]:
                n_dropped += 1
                continue
            pos_cluster = np.array([[d["est_utm"][0], d["est_utm"][1]] for d in cluster_dets])
            spread = np.std(pos_cluster, axis=0).max()
            if spread > thresh["max_spread"]:
                n_dropped += 1
                continue

            if obj_type == "sign":
                bboxes = [d["bbox"] for d in cluster_dets if d.get("bbox")]
                if bboxes:
                    aspects = []
                    for bx1, by1, bx2, by2 in bboxes:
                        w2, h2 = bx2 - bx1, by2 - by1
                        aspects.append(max(w2, h2) / max(min(w2, h2), 1))
                    avg_aspect = np.mean(aspects)
                    best_clf = max(d.get("classifier_conf", 0) for d in cluster_dets)
                    if avg_aspect < 1.25 and best_clf < 0.85:
                        print(f"    REJECTED tire-like: aspect={avg_aspect:.2f} clf={best_clf:.2f}")
                        n_dropped += 1
                        continue

            mean_est_utm = np.mean(
                [[d["est_utm"][0], d["est_utm"][1], d["est_utm"][2]] for d in cluster_dets],
                axis=0).tolist()

            # Multi-view triangulation: replaces the monocular-distance-based
            # XY estimate with a real ray-intersection when this object was
            # seen from >=2 angularly-distinct viewpoints. Doesn't need to
            # know the object's real-world size at all (unlike the monocular
            # estimate, which is only as accurate as the classifier's guess
            # of what sign it is) — this is what actually fixes the 4-8m
            # gap measured between image-estimated and true sign positions,
            # for any detection with enough view diversity to support it.
            triangulated = False
            car_utms_c = [d["utm"] for d in cluster_dets if d.get("bearing_rad") is not None]
            bearings_c = [d["bearing_rad"] for d in cluster_dets if d.get("bearing_rad") is not None]
            if len(car_utms_c) >= 2:
                tri_point, tri_ok = triangulate_rays(car_utms_c, bearings_c)
                if tri_ok:
                    mean_est_utm = [float(tri_point[0]), float(tri_point[1]), mean_est_utm[2]]
                    triangulated = True

            class_votes = Counter(d["class_id"] for d in cluster_dets)
            best_class = class_votes.most_common(1)[0][0]
            max_conf = max(d["conf"] for d in cluster_dets)

            best_crop_det = max(cluster_dets,
                key=lambda d: d.get("classifier_conf", 0.0) * (1 + d.get("crop_quality", 0) / 1000.0))
            specific_class  = best_crop_det.get("specific_class", "unknown")
            specific_label  = best_crop_det.get("specific_label", "UNKNOWN")
            classifier_conf = best_crop_det.get("classifier_conf", 0.0)
            crop_path       = best_crop_det.get("crop_path", "")
            crop_quality    = best_crop_det.get("crop_quality", 0.0)
            use_crop_recommended = best_crop_det.get("use_crop_recommended", False)
            heading         = best_crop_det.get("heading", 0.0)
            bearing_rad     = best_crop_det.get("bearing_rad", 0.0)
            likely_back_face = any(d.get("likely_back_face", False) for d in cluster_dets)
            tilt_deg_used   = best_crop_det.get("tilt_deg_used", 0.0)
            tilt_deg_measured = best_crop_det.get("tilt_deg_measured", 0.0)
            tilt_source     = best_crop_det.get("tilt_source", "none")
            tilt_n_triangle_refs = best_crop_det.get("tilt_n_triangle_refs", 0)
            pass1_class     = best_crop_det.get("pass1_class")
            pass1_conf      = best_crop_det.get("pass1_conf")
            pass2_class     = best_crop_det.get("pass2_class")
            pass2_conf      = best_crop_det.get("pass2_conf")

            cls76_votes = Counter(d.get("specific_class", "unknown")
                for d in cluster_dets if d.get("classifier_conf", 0) > 0.70)
            if cls76_votes:
                specific_class = cls76_votes.most_common(1)[0][0]
                specific_label = class_to_label(specific_class)
                # FIX: crop_path/crop_quality/classifier_conf/heading above were
                # picked from best_crop_det independently of this vote — if the
                # winning class differs from best_crop_det's own class (e.g. one
                # frame classified confidently-but-wrongly, while the majority of
                # OTHER frames in the same spatial cluster agreed on the real
                # class), the crop shown for this object would visually contradict
                # its own label. Re-pick the crop from detections that actually
                # voted for the winning class.
                voters = [d for d in cluster_dets if d.get("specific_class") == specific_class]
                if voters:
                    best_voter = max(voters,
                        key=lambda d: d.get("classifier_conf", 0.0) * (1 + d.get("crop_quality", 0) / 1000.0))
                    classifier_conf = best_voter.get("classifier_conf", 0.0)
                    crop_path       = best_voter.get("crop_path", "")
                    crop_quality    = best_voter.get("crop_quality", 0.0)
                    use_crop_recommended = best_voter.get("use_crop_recommended", False)
                    heading         = best_voter.get("heading", heading)
                    bearing_rad     = best_voter.get("bearing_rad", bearing_rad)
                    tilt_deg_used   = best_voter.get("tilt_deg_used", tilt_deg_used)
                    tilt_deg_measured = best_voter.get("tilt_deg_measured", tilt_deg_measured)
                    tilt_source     = best_voter.get("tilt_source", tilt_source)
                    tilt_n_triangle_refs = best_voter.get("tilt_n_triangle_refs", tilt_n_triangle_refs)
                    pass1_class     = best_voter.get("pass1_class", pass1_class)
                    pass1_conf      = best_voter.get("pass1_conf", pass1_conf)
                    pass2_class     = best_voter.get("pass2_class", pass2_class)
                    pass2_conf      = best_voter.get("pass2_conf", pass2_conf)

            uq = compute_uq(cluster_dets, None)

            unique_objects.append({
                "type": obj_type, "class_id": best_class, "confidence": max_conf,
                "n_detections": len(cluster_dets),
                "n_images": len(set(d["image"] for d in cluster_dets)),
                "approx_utm": mean_est_utm,
                "specific_class": specific_class, "specific_label": specific_label,
                "classifier_conf": classifier_conf, "crop_path": crop_path,
                "crop_quality": round(crop_quality, 1),
                "use_crop_recommended": use_crop_recommended,
                "heading": heading, "bearing_rad": float(bearing_rad),
                "triangulated": triangulated, "likely_back_face": bool(likely_back_face),
                "tilt_deg_used": tilt_deg_used, "tilt_deg_measured": tilt_deg_measured,
                "tilt_source": tilt_source, "tilt_n_triangle_refs": tilt_n_triangle_refs,
                "pass1_class": pass1_class, "pass1_conf": pass1_conf,
                "pass2_class": pass2_class, "pass2_conf": pass2_conf,
                "position_uncertainty_m":     uq["position_uncertainty_m"],
                "classification_uncertainty": uq["classification_uncertainty"],
                "n_view_baseline_m":          uq["n_view_baseline_m"],
                "_cluster_dets": cluster_dets,
            })

        print(f"    {obj_type}s: {len(clusters)} raw clusters -> "
              f"{len([o for o in unique_objects if o['type']==obj_type])} kept, {n_dropped} dropped")

    signs  = [o for o in unique_objects if o["type"] == "sign"]
    lights = [o for o in unique_objects if o["type"] == "light"]
    print(f"    Unique objects: {len(signs)} signs, {len(lights)} lights")
    return unique_objects


def localise_in_pointcloud(unique_objects, xyz, road_mask, get_ground_z):
    import open3d as o3d

    nonroad_idx = np.where(~road_mask)[0]
    nonroad_xyz = xyz[nonroad_idx]

    localised = []
    n_verified = n_camera_only = 0

    for obj in unique_objects:
        approx_e, approx_n = obj["approx_utm"][0], obj["approx_utm"][1]
        obj_type = obj["type"]
        label = obj.get("specific_label", obj_type)
        cluster_dets = obj.pop("_cluster_dets", [])
        obj["ray_matched"] = any(d.get("ray_matched") for d in cluster_dets)

        if obj_type == "light":
            search_r, dbscan_eps, dbscan_min, min_pts = (
                LIGHT_SEARCH_RADIUS, LIGHT_DBSCAN_EPS, LIGHT_DBSCAN_MIN_PTS, LIGHT_MIN_CLUSTER_PTS)
        else:
            search_r, dbscan_eps, dbscan_min, min_pts = (
                SIGN_SEARCH_RADIUS, LOCAL_DBSCAN_EPS, LOCAL_DBSCAN_MIN_POINTS, SIGN_MIN_CLUSTER_PTS)

        dists_2d = np.sqrt((nonroad_xyz[:, 0]-approx_e)**2 + (nonroad_xyz[:, 1]-approx_n)**2)
        local_mask = dists_2d <= search_r
        local_count = local_mask.sum()

        best_cluster = None
        if local_count >= dbscan_min:
            local_pts = nonroad_xyz[local_mask]
            local_original_idx = nonroad_idx[local_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(local_pts)
            labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min, print_progress=False))

            if labels.max() >= 0:
                ground_z = get_ground_z(approx_e, approx_n)
                best_score = -1
                for lbl in range(labels.max() + 1):
                    cmask = labels == lbl
                    cpts, cidx = local_pts[cmask], local_original_idx[cmask]
                    if len(cpts) < min_pts:
                        continue
                    z_min, z_max = cpts[:, 2].min(), cpts[:, 2].max()
                    height_ag, ground_gap = z_max - ground_z, z_min - ground_z
                    max_width = max(cpts[:,0].max()-cpts[:,0].min(), cpts[:,1].max()-cpts[:,1].min())
                    centroid = cpts.mean(axis=0)
                    dist_to_approx = np.sqrt((centroid[0]-approx_e)**2 + (centroid[1]-approx_n)**2)

                    if obj_type == "light":
                        if height_ag < LIGHT_MIN_HEIGHT or max_width > LIGHT_MAX_WIDTH or ground_gap > LIGHT_MAX_GROUND_OFF:
                            continue
                        score = height_ag / (dist_to_approx + 1.0)
                    else:
                        if (ground_gap > SIGN_MAX_GROUND_OFF or height_ag < SIGN_MIN_HEIGHT or
                                max_width > SIGN_MAX_WIDTH or
                                (max_width > 0.1 and height_ag/max_width < SIGN_MIN_HW_RATIO)):
                            continue
                        score = height_ag / (max_width + 0.1) / (dist_to_approx + 1.0)

                    if score > best_score:
                        best_score = score
                        best_cluster = {
                            "centroid": centroid.tolist(), "indices": cidx,
                            "height_above_ground": float(height_ag), "max_width": float(max_width),
                            "n_points": int(len(cpts)), "dist_to_approx": float(dist_to_approx),
                            "ground_z": float(ground_z),
                        }

        if best_cluster is not None:
            uq_lidar = compute_uq(cluster_dets, best_cluster)
            obj["lidar_confidence"] = uq_lidar["lidar_confidence"]
            obj["source_method"] = "lidar_verified"
            localised.append({**obj, **best_cluster})
            n_verified += 1
        elif ALLOW_CAMERA_ONLY_FALLBACK:
            obj["lidar_confidence"] = 0.0
            obj["source_method"] = "camera_only_unverified"
            localised.append({
                **obj,
                "centroid": [approx_e, approx_n, obj["approx_utm"][2]],
                "indices": np.array([], dtype=int),
                "height_above_ground": 0.0, "max_width": 0.0, "n_points": 0,
                "dist_to_approx": 0.0, "ground_z": get_ground_z(approx_e, approx_n),
            })
            n_camera_only += 1

    print(f"    Localised: {len(localised)}  (lidar_verified={n_verified}, "
          f"camera_only_unverified={n_camera_only})")
    return localised


def detect_bare_poles(xyz, road_mask, get_ground_z, localised_objects):
    import open3d as o3d
    nonroad_idx = np.where(~road_mask)[0]
    nonroad_xyz = xyz[nonroad_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nonroad_xyz)
    labels = np.array(pcd.cluster_dbscan(eps=0.40, min_points=8, print_progress=False))

    claimed = [obj["centroid"][:2] for obj in localised_objects
               if "centroid" in obj and obj.get("n_points", 0) > 0]

    poles = []
    for lbl in range(max(0, labels.max() + 1)):
        cmask = labels == lbl
        cpts, cidx = nonroad_xyz[cmask], nonroad_idx[cmask]
        if len(cpts) < 15 or len(cpts) > 1500:
            continue
        centroid = cpts.mean(axis=0)
        ground_z = get_ground_z(centroid[0], centroid[1])
        z_min, z_max = cpts[:, 2].min(), cpts[:, 2].max()
        height_ag, ground_gap = z_max - ground_z, z_min - ground_z
        if ground_gap > POLE_MAX_GROUND_OFFSET or height_ag < POLE_MIN_HEIGHT:
            continue
        max_width = max(cpts[:,0].max()-cpts[:,0].min(), cpts[:,1].max()-cpts[:,1].min())
        if max_width > POLE_MAX_WIDTH:
            continue
        if any(np.hypot(centroid[0]-cp[0], centroid[1]-cp[1]) < 3.0 for cp in claimed):
            continue
        poles.append({
            "type": "pole", "centroid": centroid.tolist(), "indices": cidx,
            "height_above_ground": float(height_ag), "max_width": float(max_width),
            "n_points": int(len(cpts)), "ground_z": float(ground_z),
        })

    print(f"    Bare poles found: {len(poles)}")
    return poles


def post_localisation_refinement(localised, poles, xyz, road_mask, get_ground_z):
    deduped, used = [], set()
    for i, obj in enumerate(localised):
        if i in used:
            continue
        best = obj
        for j in range(i + 1, len(localised)):
            if j in used or localised[j]["type"] != obj["type"]:
                continue
            if obj.get("n_points", 0) == 0 or localised[j].get("n_points", 0) == 0:
                continue
            ci = np.array(obj["centroid"][:2])
            cj = np.array(localised[j]["centroid"][:2])
            if np.linalg.norm(ci - cj) < SAME_TYPE_DEDUP_DIST_M:
                used.add(j)
                if localised[j].get("confidence", 0) > best.get("confidence", 0):
                    best = localised[j]
        deduped.append(best)
    if len(localised) != len(deduped):
        print(f"    Post-dedup: {len(localised)} -> {len(deduped)} "
              f"(merged {len(localised)-len(deduped)} same-cluster duplicates)")

    signs  = [o for o in deduped if o["type"] == "sign"]
    lights = [o for o in deduped if o["type"] == "light"]
    for s in signs:
        sc = np.array(s["centroid"][:2])
        for l in lights:
            lc = np.array(l["centroid"][:2])
            if np.linalg.norm(sc - lc) < CO_LOCATION_DIST_M:
                light_top_z = l["ground_z"] + l["height_above_ground"]
                s["centroid"][2] = light_top_z - CO_LOCATION_Z_OFFSET_M
                s["co_located_with"] = "traffic_light"
                print(f"    CO-LOCATED: {s.get('specific_label','?')} offset below "
                      f"{l.get('specific_label','?')}")

    detection_positions = [np.array(o["centroid"][:2]) for o in deduped if o.get("n_points", 0) > 0]
    for pole in poles:
        pc = np.array(pole["centroid"][:2])
        pole["verification_status"] = "unclassified_pole"
        if detection_positions:
            min_dist = min(np.linalg.norm(pc - dp) for dp in detection_positions)
            if min_dist < POLE_ADJACENT_RADIUS_M:
                pole["verification_status"] = "adjacent_to_detection"
                pole["nearest_detection_m"] = round(float(min_dist), 2)

        if "indices" in pole and len(pole["indices"]) > 0:
            pole_pts = xyz[pole["indices"]]
            z_range = pole_pts[:, 2].max() - pole_pts[:, 2].min()
            if z_range > 1.0:
                z_thr = pole_pts[:, 2].min() + z_range * 0.75
                top = pole_pts[pole_pts[:, 2] > z_thr]
                bot = pole_pts[pole_pts[:, 2] <= z_thr]
                if len(top) > 5 and len(bot) > 5:
                    top_w = max(top[:,0].max()-top[:,0].min(), top[:,1].max()-top[:,1].min())
                    bot_w = max(bot[:,0].max()-bot[:,0].min(), bot[:,1].max()-bot[:,1].min())
                    if top_w > bot_w * 1.5 and top_w > 0.2:
                        pole["verification_status"] = "probable_sign"
                        pole["head_width_m"] = round(float(top_w), 3)
                        pole["shaft_width_m"] = round(float(bot_w), 3)

    n_adj = sum(1 for p in poles if p.get("verification_status") == "adjacent_to_detection")
    n_prob = sum(1 for p in poles if p.get("verification_status") == "probable_sign")
    print(f"    Pole tags: {n_adj} adjacent_to_detection, {n_prob} probable_sign")

    return deduped, poles


def write_geojson(objects, out_path, kind):
    STRIP = {"centroid", "indices", "_cluster_dets"}
    features = []
    for obj in objects:
        c = obj["centroid"]
        props = {k: v for k, v in obj.items() if k not in STRIP}
        for k, v in list(props.items()):
            if isinstance(v, np.ndarray):
                props[k] = v.tolist()
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(c[0]), float(c[1]), float(c[2])]},
            "properties": props,
        })
    fc = {"type": "FeatureCollection", "name": kind,
          "crs": {"type": "name", "properties": {"name": "EPSG:25832"}}, "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"    Wrote {len(features)} {kind} -> {out_path}")


def write_proof_gallery(localised, poles, out_path, crop_dir, output_dir):
    """
    One row per final detection: the actual source crop image used to
    classify it, plus class/confidence/placement method — so each placement
    can be visually checked against what the camera really saw, instead of
    trusting the geometry blindly.
    """
    rel_crop_dir = os.path.relpath(crop_dir, output_dir)

    def row(obj, idx):
        crop_path = obj.get("crop_path", "")
        has_img = bool(crop_path) and os.path.isfile(crop_path)
        img_html = (
            f'<img src="{os.path.join(rel_crop_dir, os.path.basename(crop_path))}" '
            f'style="max-width:220px;max-height:160px;border:1px solid #444;">'
            if has_img else
            '<div style="width:220px;height:160px;display:flex;align-items:center;'
            'justify-content:center;background:#222;color:#888;border:1px solid #444;">'
            'no crop (LiDAR-only)</div>'
        )
        src = obj.get("source_method", "?")
        src_color = {"lidar_verified": "#2ecc71", "camera_only_unverified": "#e67e22"}.get(src, "#999")
        ray_ok = obj.get("ray_matched")
        ray_html = ("✓ point-cloud matched" if ray_ok else "✗ no point-cloud match (geometry-only)") if ray_ok is not None else "n/a"
        c = obj.get("centroid", [0, 0, 0])
        return f"""
        <tr>
          <td>{idx}</td>
          <td>{img_html}</td>
          <td>{obj.get('type','?')}<br><b>{obj.get('specific_label', obj.get('type','?'))}</b></td>
          <td>det_conf={obj.get('confidence', 0):.2f}<br>cls_conf={obj.get('classifier_conf', 0):.2f}</td>
          <td style="color:{src_color}">{src}</td>
          <td>{ray_html}</td>
          <td>n_imgs={obj.get('n_images', obj.get('n_detections', '?'))}</td>
          <td>{c[0]:.1f}, {c[1]:.1f}, {c[2]:.2f}</td>
        </tr>"""

    rows = []
    for i, obj in enumerate(localised, 1):
        rows.append(row(obj, i))
    for i, obj in enumerate(poles, 1):
        rows.append(row(obj, f"pole-{i}"))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 3 Detection Proof Gallery</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;padding:20px;}}
table{{border-collapse:collapse;width:100%;}}
td,th{{border:1px solid #444;padding:8px;text-align:left;vertical-align:top;font-size:13px;}}
th{{background:#222;position:sticky;top:0;}}
</style></head><body>
<h2>Phase 3 Detection Proof — {len(localised)} signs/lights, {len(poles)} poles</h2>
<p>Green = LiDAR-verified position. Orange = camera-only (no matching LiDAR cluster found).
"✓ point-cloud matched" = the image-derived bearing ray was corrected using real point-cloud
density instead of the raw monocular distance guess.</p>
<table>
<tr><th>#</th><th>Crop (proof)</th><th>Class</th><th>Confidence</th><th>Source</th>
<th>Ray/density check</th><th>Views</th><th>Position (E, N, Z)</th></tr>
{"".join(rows)}
</table>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"    Wrote proof gallery ({len(localised)+len(poles)} rows) -> {out_path}")


def save_classified_laz(xyz, intensity, road_mask, localised, poles, out_path):
    colors = np.zeros((len(xyz), 3), dtype=np.uint8)
    if road_mask.sum() > 0:
        road_int = intensity[road_mask]
        lo = np.percentile(road_int, 2)
        hi = max(np.percentile(road_int, 98), lo + 1)
        normed = np.clip((road_int - lo) / (hi - lo), 0, 1)
        grey = (50 + normed * 205).astype(np.uint8)
        colors[road_mask] = np.stack([grey, grey, grey], axis=1)
    colors[~road_mask] = [40, 40, 60]

    for obj in localised:
        if "indices" not in obj or len(obj["indices"]) == 0:
            continue
        idx = obj["indices"]
        if obj["type"] == "sign":
            if obj.get("likely_back_face"):
                colors[idx] = [160, 32, 240]
            elif obj.get("source_method") == "camera_only_unverified":
                colors[idx] = [255, 140, 0]
            else:
                colors[idx] = [220, 30, 30]
        elif obj["type"] == "light":
            if obj.get("source_method") == "camera_only_unverified":
                colors[idx] = [255, 140, 0]
            else:
                colors[idx] = [240, 220, 40]

    for obj in poles:
        if "indices" not in obj or len(obj["indices"]) == 0:
            continue
        idx = obj["indices"]
        status = obj.get("verification_status", "unclassified_pole")
        if status == "probable_sign":
            colors[idx] = [255, 100, 200]
        elif status == "adjacent_to_detection":
            colors[idx] = [255, 200, 60]
        else:
            colors[idx] = [60, 200, 220]

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = intensity.astype(np.uint16)
    las.red   = colors[:, 0].astype(np.uint16) * 256
    las.green = colors[:, 1].astype(np.uint16) * 256
    las.blue  = colors[:, 2].astype(np.uint16) * 256
    las.write(out_path)
    print(f"    Saved visualisation LAZ: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase3_v6.py <laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    if not os.path.isfile(laz_path):
        print(f"ERROR: LAZ file not found: {laz_path}")
        sys.exit(1)

    basename = os.path.basename(laz_path)
    m = re.search(r'(part\d+)', basename)
    part_name = m.group(1) if m else basename.replace('.laz', '')

    print(f"\n{'='*70}\n  PHASE 3 v6 - Validated Heading + Camera-Only Fallback\n"
          f"  LAZ tile : {basename}\n  Part     : {part_name}\n{'='*70}\n")

    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/8] Loading point cloud...")
    xyz, intensity = load_laz(laz_path)

    print("\n[2/8] RANSAC road segmentation...")
    road_mask = ransac_road_segmentation(xyz)

    print("\n[2.5/8] Building road height lookup...")
    get_ground_z = build_road_height_lookup(xyz, road_mask)

    print("\n[3/8] Loading camera calibration (per-camera pinhole K)...")
    pinhole_K = load_pinhole_K()

    print("\n[4/8] Loading camera entries...")
    bbox = get_laz_bbox(laz_path)
    cam_entries = load_camextr_for_bbox(bbox)
    if not cam_entries:
        print("  ERROR: No images in tile bbox.")
        sys.exit(1)

    print("\n[4.5/8] Building point-cloud density tree for ray-corrected placement...")
    pc_tree, pc_nonroad_xyz = build_pointcloud_tree(xyz, road_mask, get_ground_z)

    print("\n[5/8] Running YOLO on all images...")
    raw_detections = batch_yolo_detection(cam_entries, pinhole_K, pc_tree, pc_nonroad_xyz)

    print("\n[6/8] Spatial deduplication...")
    unique_objects = deduplicate_detections(raw_detections)

    print("\n[7/8] Localising objects (LiDAR-verified + camera-only fallback)...")
    localised = localise_in_pointcloud(unique_objects, xyz, road_mask, get_ground_z)

    print("\n[7.5/8] Detecting bare poles...")
    poles = detect_bare_poles(xyz, road_mask, get_ground_z, localised)

    print("\n[7.8/8] Post-localisation refinement (heuristic)...")
    localised, poles = post_localisation_refinement(localised, poles, xyz, road_mask, get_ground_z)

    print("\n[7.9/8] Road/median no-touch zone (keeps placements off the "
          "drivable corridor)...")
    rgb = load_colorized_rgb(part_name, xyz)
    if rgb is not None:
        zone_fn = build_road_notouch_zone(xyz, rgb, intensity, road_mask, cam_entries, get_ground_z)
        localised = apply_road_notouch_zone(localised, zone_fn, cam_entries)
        poles = apply_road_notouch_zone(poles, zone_fn, cam_entries)

    print("\n[8/8] Writing outputs...")
    signs  = [o for o in localised if o["type"] == "sign"]
    lights = [o for o in localised if o["type"] == "light"]

    signs_path  = os.path.join(OUTPUT_DIR, f"signs_3d_{part_name}.geojson")
    lights_path = os.path.join(OUTPUT_DIR, f"traffic_lights_3d_{part_name}.geojson")
    poles_path  = os.path.join(OUTPUT_DIR, f"poles_3d_{part_name}.geojson")
    out_laz     = os.path.join(OUTPUT_DIR, f"phase3_{part_name}_objects.laz")
    report_path = os.path.join(OUTPUT_DIR, f"phase3_{part_name}_report.txt")

    write_geojson(signs, signs_path, "signs")
    write_geojson(lights, lights_path, "traffic_lights")
    write_geojson(poles, poles_path, "poles")
    save_classified_laz(xyz, intensity, road_mask, localised, poles, out_laz)

    gallery_path = os.path.join(OUTPUT_DIR, f"detection_proof_{part_name}.html")
    write_proof_gallery(localised, poles, gallery_path, CROP_OUTPUT_DIR, OUTPUT_DIR)

    elapsed = time.time() - t0
    n_verified_s = sum(1 for o in signs if o.get("source_method")=="lidar_verified")
    n_verified_l = sum(1 for o in lights if o.get("source_method")=="lidar_verified")
    n_ray_matched = sum(1 for o in (signs + lights) if o.get("ray_matched"))

    summary = [
        "Phase 3 v6 Report", "="*50,
        f"LAZ tile: {basename}", f"Total points: {len(xyz):,}",
        f"Road points: {road_mask.sum():,}", f"Non-road points: {(~road_mask).sum():,}",
        "", f"Images scanned: {len(cam_entries)}", f"Raw detections: {len(raw_detections)}",
        f"Unique objects: {len(unique_objects)}", "",
        f"Signs:  {len(signs)}  (lidar_verified={n_verified_s}, "
        f"camera_only={len(signs)-n_verified_s})",
        f"Lights: {len(lights)}  (lidar_verified={n_verified_l}, "
        f"camera_only={len(lights)-n_verified_l})",
        f"Bare poles: {len(poles)}",
        f"Ray/point-cloud-density matched: {n_ray_matched}/{len(signs)+len(lights)} "
        f"(position corrected against real LiDAR structure, not just monocular distance guess)",
        "",
        "Color coding (visualisation LAZ):",
        "  Grey        = Road",
        "  Red         = Verified sign      Yellow = Verified light",
        "  Orange      = Camera-only unverified (either type)",
        "  Purple      = Sign likely back-facing camera",
        "  Cyan        = Plain pole   Amber = Pole adjacent to a detection",
        "  Pink        = Pole with probable_sign geometry (never camera-confirmed)",
        "", f"Elapsed: {elapsed:.1f}s", "",
        "Outputs:", f"  {signs_path}", f"  {lights_path}", f"  {poles_path}", f"  {out_laz}",
        f"  {gallery_path}  (visual proof gallery — one crop image per detection)",
    ]
    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")
    with open(report_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nReport -> {report_path}\nDone.\n")


if __name__ == "__main__":
    main()
