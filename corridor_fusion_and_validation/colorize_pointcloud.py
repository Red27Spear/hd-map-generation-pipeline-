#!/usr/bin/env python3
"""
Occlusion-aware LiDAR point cloud colorization.

Fixes two gaps found in the project's existing colorization scripts:
  1. No occlusion handling -- points behind a closer surface along the same
     camera ray could get colored with the closer surface's pixel. Fixed
     with a per-image z-buffer (nearest point per pixel wins for that view).
  2. Nearest-neighbour pixel sampling -- replaced with bilinear sampling.

Calibration: uses the validated convention (rotMat(*Hrp) direct order,
pinhole K from phase0_pinhole_K.json) -- confirmed pixel-accurate against
real LiDAR/image data, not the "xyz" euler convention used elsewhere in
this project's history that produced 700-1150px reprojection error.

Across multiple camera views seeing the same point, the closest-range
observation wins (least foreshortening, sharpest sample, and a natural
tie-breaker when a point is visible -- unoccluded -- in more than one shot).

Usage:
    python colorize_pointcloud.py <laz_file>

Output:
    <OUTPUT_DIR>/colorized_<part>.laz
"""

import sys, os, json, re, time, math
from collections import defaultdict
import numpy as np
import cv2
import laspy
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformationTools import rotMat, transformMatrix

CAMEXTR_PATH = "/home/rohit/Downloads/Project/hd_map_p06/config/2026_05_08_Bonn/CamExtr.json"
PINHOLE_K_PATH = "/home/rohit/Documents/Output/data_fix/phase0_pinhole_K.json"
DATA_FIX_DIR = "/home/rohit/Documents/Output/data_fix"
OUTPUT_DIR = "/home/rohit/Downloads/Project/hd_map_p06/output/colorization"

BBOX_MARGIN_M = 20.0
POINT_SEARCH_RADIUS_M = 30.0
MIN_FORWARD_DEPTH_M = 1.0
UNCOLORED_FALLBACK = (60, 60, 60)

# The survey vehicle's own roof/roof-rack is visible in the lower part of
# some cameras' frames (confirmed by inspection of raw images at 2448x2048).
# Real-world ground points that happen to project into that pixel region
# would otherwise get colored with the vehicle's own body instead of their
# true ground texture -- every frame contributes a fresh mis-colored patch
# as the vehicle moves, producing a repeating "car-shaped" pattern down the
# road. Excluded per camera serial as (row_cutoff, col_cutoff): a point is
# rejected if v >= row_cutoff and u >= col_cutoff. col_cutoff=0 masks the
# full row width (down camera, roof spans nearly the whole frame); a higher
# col_cutoff restricts the mask to one corner (right camera).
VEHICLE_OCCLUSION_MASK = {
    "4108736907": (1150, 0),      # down: roof/rack spans full frame width
    "4108736902": (1100, 1150),   # right: roof visible in lower-right corner only
}

# The overcast sky in these images is bright and desaturated (near-white to
# pale pink/lavender). No real 3D point should ever be colored from a sky
# pixel, but tall points (building facades reaching 15-20m up) are best
# boresight-aligned to the "up" camera -- which is ~90% sky -- so points
# right at the building/sky edge can sample sky instead of facade texture,
# producing a pale blue haze on the upper portion of tall structures.
SKY_V_THRESH = 200   # HSV value: sky is bright
SKY_S_THRESH = 30    # HSV saturation: sky is desaturated


def compute_sky_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] > SKY_V_THRESH) & (hsv[..., 1] < SKY_S_THRESH)


# Even after excluding pure sky, fine structure right at a sky boundary --
# tree canopy edges, roof/parapet lines -- carries JPEG-compression and
# bilinear-sampling blur that blends real material with sky in the source
# image itself; no per-pixel classifier can cleanly separate that after the
# fact. The "up" camera is ~90% sky (confirmed by inspection), with genuine
# non-sky content confined to small slivers near its frame edges/corners, so
# rather than perfect sky detection, heavily discourage it from winning
# unless no other camera has any view at all.
CAMERA_SCORE_PENALTY = {
    "4108690914": 4.0,   # up
}


def get_laz_bbox(laz_path):
    with laspy.open(laz_path) as f:
        mins, maxs = f.header.mins, f.header.maxs
    return {
        "e_min": mins[0] - BBOX_MARGIN_M, "e_max": maxs[0] + BBOX_MARGIN_M,
        "n_min": mins[1] - BBOX_MARGIN_M, "n_max": maxs[1] + BBOX_MARGIN_M,
    }


# The part24/part48 corridor was driven twice ~13.5min apart (established
# elsewhere this session via moro.track.txt timestamps): pass1 (outbound,
# part24) ends ~30802s-of-day, pass2 (return, part48) starts ~31471s.
PASS_SPLIT_T = 31100


def image_time_of_day(image_name):
    m = re.search(r"utc-\d{8}_(\d{2})-(\d{2})-(\d{2})-\d+", image_name)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mi * 60 + s


def load_camextr(bbox, pass_filter=None):
    """pass_filter: None (all images), "pass1", or "pass2" -- restricts which
    camera images are eligible for colorization, e.g. to keep a merged tile's
    colors visually consistent (same lighting/exposure/vehicle position)
    instead of a patchwork of whichever pass's camera happened to win each
    point."""
    entries = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    filtered = [e for e in entries
                if bbox["e_min"] <= e["Xyz"][0] <= bbox["e_max"]
                and bbox["n_min"] <= e["Xyz"][1] <= bbox["n_max"]]
    if pass_filter is not None:
        want_pass2 = pass_filter == "pass2"
        filtered = [e for e in filtered
                    if (image_time_of_day(e["Image"]) or 0) >= PASS_SPLIT_T] if want_pass2 else \
                   [e for e in filtered
                    if (image_time_of_day(e["Image"]) or 0) < PASS_SPLIT_T]
    filtered.sort(key=lambda e: e["Time"])
    return filtered


def load_pinhole_K():
    raw = json.load(open(PINHOLE_K_PATH))
    K = {}
    for sn, v in raw.items():
        K[sn] = np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]])
    return K


def bilinear_sample(img, u, v):
    """img: HxWx3 uint8 (BGR). u,v: 1D float arrays, already bounds-checked
    to [0, w-1.001] / [0, h-1.001]. Returns Nx3 float64 BGR."""
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, w - 1); v1 = np.minimum(v0 + 1, h - 1)
    fu = (u - u0)[:, None]; fv = (v - v0)[:, None]

    Ia = img[v0, u0].astype(np.float64)
    Ib = img[v0, u1].astype(np.float64)
    Ic = img[v1, u0].astype(np.float64)
    Id = img[v1, u1].astype(np.float64)

    top = Ia * (1 - fu) + Ib * fu
    bot = Ic * (1 - fu) + Id * fu
    return top * (1 - fv) + bot * fv


# ══════════════════════════════════════════════════════════════════════════
# HOVER LABELS -- floating sign/light detection tags baked into the cloud
# ══════════════════════════════════════════════════════════════════════════
# LAZ point clouds have no mouse-hover events, so a "hovering tag" here means
# what phase4c_stvo_sprites.py already does for the phase3 pipeline: render
# the tag text to a small image, then turn every non-background pixel into a
# colored 3D point floating just above the real detection -- readable in any
# standard point-cloud viewer (Open3D, CloudCompare), no extra viewer needed.
#
# Detections come in as bare 2D (image, bbox_px, kind, sam3_score, [label,
# clf_conf]) rows -- e.g. detect_and_label.py's detections.jsonl sidecar --
# with no 3D position yet. Localized here by unprojecting through each
# camera's full 3D rotation (see localize_detections below) at an assumed
# real-world sign/light size, using this project's own post-undistortion
# pinhole_K (not the raw calibration XML -- data_fix images are already
# corrected to that model).
LABEL_PLATE_W_M = 1.1
LABEL_PLATE_H_M = 0.22

LABEL_REAL_SIGN_WIDTH_M = 0.60
LABEL_REAL_LIGHT_HEIGHT_M = 1.30
LABEL_SIGN_DIST_RANGE_M = (2.0, 25.0)
LABEL_LIGHT_DIST_RANGE_M = (3.0, 30.0)

# Same three rules validated against real survey footage in apply_filters.py:
# a SAM3-score floor, sign/light overlap dedup (SAM3 double-detecting one
# physical traffic light as both), and confidence-gated specific-vs-generic
# sign labels (classifier confidence alone isn't reliable enough on this
# footage to trust unflagged).
LABEL_SCORE_THRESHOLD = 0.6
LABEL_IOU_THRESHOLD = 0.3
LABEL_CLF_CONF_THRESHOLD = 0.5


def localize_detections(detections_jsonl, camextr_path, pinhole_k, bbox, margin=30.0):
    """2D bbox detections -> 3D UTM points, restricted to cameras that fall
    inside `bbox` (+margin) -- callers only care about one point-cloud tile
    at a time, and most surveys cover far more ground than one tile.

    Unprojects the bbox-centre pixel through the camera's full 3D rotation
    (rotMat(*Hrp), same convention colorize() already uses/validates for
    LiDAR reprojection) rather than a heading-only 2D bearing. This rig's 4
    cameras (up/down/left/right) carry large roll/pitch in their per-image
    Hrp -- e.g. one 'right'-camera frame here has Hrp=[-87.0, 65.4, -122.3]
    -- so a heading-only formula (fine for a single roughly-forward camera,
    as in detect_signs_lights_sam3.py) throws away most of the real
    orientation and scatters the result across all 4 cameras.
    """
    camextr = json.load(open(camextr_path))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}

    e_min, e_max = bbox["e_min"] - margin, bbox["e_max"] + margin
    n_min, n_max = bbox["n_min"] - margin, bbox["n_max"] + margin

    out = []
    with open(detections_jsonl) as f:
        for line in f:
            d = json.loads(line)
            e = by_image.get(d["image"])
            if e is None:
                continue
            car_e, car_n, car_z = e["Xyz"][:3]
            if not (e_min <= car_e <= e_max and n_min <= car_n <= n_max):
                continue
            sn = str(e["SerialNr"])
            K = pinhole_k.get(sn)
            if K is None:
                continue
            fx = (K[0, 0] + K[1, 1]) / 2.0
            x1, y1, x2, y2 = d["bbox_px"]
            bw, bh = x2 - x1, y2 - y1
            bbox_cx, bbox_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if d["kind"] == "sign":
                depth = min(max(fx * LABEL_REAL_SIGN_WIDTH_M / max(bw, 1.0), LABEL_SIGN_DIST_RANGE_M[0]), LABEL_SIGN_DIST_RANGE_M[1])
            else:
                depth = min(max(fx * LABEL_REAL_LIGHT_HEIGHT_M / max(bh, 1.0), LABEL_LIGHT_DIST_RANGE_M[0]), LABEL_LIGHT_DIST_RANGE_M[1])

            # camera-frame ray scaled so its Z-component equals the assumed
            # depth -- same semantics as the original heading-only formula,
            # just carried through the real 3D rotation instead of yaw-only.
            p_cam = depth * np.array([(bbox_cx - K[0, 2]) / K[0, 0],
                                       (bbox_cy - K[1, 2]) / K[1, 1], 1.0])
            R_cam_to_world = rotMat(*np.radians(e["Hrp"]))
            p_world = np.array([car_e, car_n, car_z]) + R_cam_to_world @ p_cam

            dir_world = R_cam_to_world @ np.array([0.0, 0.0, 1.0])
            bearing = math.atan2(dir_world[0], dir_world[1])  # for label-plate facing only

            out.append({
                **d,
                "utm_easting": p_world[0],
                "utm_northing": p_world[1],
                "utm_z": p_world[2],
                "bearing_rad": bearing,
            })
    return out


LIDAR_BBOX_SHRINK = 0.7    # match against the centre 70% of each bbox -- avoids
                           # background bleeding in near the box's own edges
LIDAR_MIN_MATCHES = 3      # below this, treat the ray-cast as unreliable
LIDAR_DEPTH_PERCENTILE = 15  # low percentile, not pure min -- robust to a
                              # single stray close outlier (e.g. a foreground
                              # branch), still favours the near surface over
                              # background that also falls inside the box
MAX_LIDAR_MONO_RATIO = 2.5   # a LiDAR match more than this multiple of the
                              # independent monocular depth estimate is
                              # treated as "saw through to the background",
                              # not a real object hit -- see localize_detections_via_lidar
MAX_LIDAR_CLUSTER_SPAN_M = 2.0  # a real sign/light housing is well under 1.5m
                              # across; a "nearest-depth-percentile" point set
                              # spanning more than this in any axis isn't one
                              # compact object -- it's the percentile trick
                              # averaging across unrelated background/foreground
                              # that happened to share a similar depth. Confirmed
                              # failure mode (2026-08-06): large/close-up bboxes
                              # (light seen filling much of the frame) can pull
                              # in 15m+ of scene, landing the "position" on
                              # whatever's most common nearby -- typically the
                              # road surface. See _largest_spatial_cluster.
LIDAR_CLUSTER_EPS_M = 1.0    # neighbour radius for the spatial-coherence pass


def _largest_spatial_cluster_mask(pts_xyz, eps=LIDAR_CLUSTER_EPS_M, min_size=3):
    """Among a noisy set of candidate 3D points, returns a boolean mask
    selecting the single largest tightly-connected group (any two points
    within `eps` of each other are in the same group), or None if nothing
    groups up. Cheap connected-components clustering -- no sklearn
    dependency needed."""
    if len(pts_xyz) < min_size:
        return None
    tree = cKDTree(pts_xyz)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if len(pairs) == 0:
        return None
    n = len(pts_xyz)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    best = sizes.argmax()
    if sizes[best] < min_size:
        return None
    return labels == best


def localize_detections_via_lidar(detections_jsonl, camextr_path, pinhole_k, xyz, bbox, margin=30.0):
    """Same purpose as localize_detections() -- 2D bbox detections -> 3D UTM
    points -- but sources depth from the tile's actual LiDAR points instead
    of guessing it from an assumed real-world sign width / light height.

    That assumption is the root cause of a documented placement problem:
    HOVER_LABELS_STATUS.md already found only 62% of tags land within 5m of
    any real LiDAR point (mean 15m, max 66m for the rest) -- a real sign
    correctly found in the photo, tagged at a wrong distance along the
    correct bearing, which reads as "the tag is hovering over some random
    object" in a viewer even though the 2D detection itself was fine.

    For each unique image among the kept detections: project the *whole*
    local point cloud into that camera (identical convention to colorize()
    -- T_world_cam = inv(transformMatrix(*Hrp, *Xyz)), same K), then for
    each detection in that image, take the real 3D point(s) whose projected
    pixel falls inside a shrunk version of its bbox_px and use their actual
    position (not a depth guess) as the tag's 3D location.

    Falls back to the old assumed-size depth estimate (localize_detections's
    approach) when too few real points land inside a detection's box --
    sparse LiDAR coverage for that patch, not a reason to drop the tag
    entirely. Every returned dict carries `lidar_matched: bool` so callers
    (and HOVER_LABELS_STATUS.md's spot checks) can tell which path was used.
    """
    camextr = json.load(open(camextr_path))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}

    e_min, e_max = bbox["e_min"] - margin, bbox["e_max"] + margin
    n_min, n_max = bbox["n_min"] - margin, bbox["n_max"] + margin

    dets_by_image = defaultdict(list)
    n_out_of_bbox = 0
    with open(detections_jsonl) as f:
        for line in f:
            d = json.loads(line)
            e = by_image.get(d["image"])
            if e is None:
                continue
            car_e, car_n, _ = e["Xyz"][:3]
            if not (e_min <= car_e <= e_max and n_min <= car_n <= n_max):
                n_out_of_bbox += 1
                continue
            dets_by_image[d["image"]].append(d)

    tree = cKDTree(xyz[:, :2])

    out = []
    n_lidar = n_fallback = 0
    for img_name, dets in dets_by_image.items():
        e = by_image[img_name]
        sn = str(e["SerialNr"])
        K = pinhole_k.get(sn)
        if K is None:
            continue

        cam_xyz = np.array(e["Xyz"][:3])
        R_cam_to_world = rotMat(*np.radians(e["Hrp"]))
        dir_world = R_cam_to_world @ np.array([0.0, 0.0, 1.0])
        bearing = math.atan2(dir_world[0], dir_world[1])

        cand_idx = np.array(tree.query_ball_point(cam_xyz[:2], POINT_SEARCH_RADIUS_M))
        u = v = depth = None
        if len(cand_idx) > 0:
            T_world_cam = np.linalg.inv(transformMatrix(*np.radians(e["Hrp"]), *cam_xyz))
            pts_h = np.hstack([xyz[cand_idx], np.ones((len(cand_idx), 1))])
            pts_cam = (T_world_cam @ pts_h.T).T[:, :3]
            depth_all = pts_cam[:, 2]
            front = depth_all > MIN_FORWARD_DEPTH_M
            if front.any():
                cand_idx, pts_cam, depth_all = cand_idx[front], pts_cam[front], depth_all[front]
                proj = (K @ pts_cam.T).T
                u = proj[:, 0] / proj[:, 2]
                v = proj[:, 1] / proj[:, 2]
                depth = depth_all

        for d in dets:
            x1, y1, x2, y2 = d["bbox_px"]
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            sx1, sx2 = cx - bw * LIDAR_BBOX_SHRINK / 2.0, cx + bw * LIDAR_BBOX_SHRINK / 2.0
            sy1, sy2 = cy - bh * LIDAR_BBOX_SHRINK / 2.0, cy + bh * LIDAR_BBOX_SHRINK / 2.0

            # Independent monocular sanity bound -- computed unconditionally,
            # not just as a fallback. A thin/small object (a median-mounted
            # sign, say) can have near-zero real LiDAR return on its own
            # surface; the "nearest matched points in this pixel region"
            # heuristic then has nothing genuine to find and instead locks
            # onto real background (e.g. a building across the road) that
            # happens to project into the same box -- confirmed failure
            # mode (2026-08-06): median/divider objects were being placed on
            # the far side of the road. A LiDAR match wildly farther than
            # the monocular guess is exactly that signature; reject it
            # rather than trust a precise-looking but wrong position.
            fx = (K[0, 0] + K[1, 1]) / 2.0
            if d["kind"] == "sign":
                fallback_depth = min(max(fx * LABEL_REAL_SIGN_WIDTH_M / max(bw, 1.0), LABEL_SIGN_DIST_RANGE_M[0]), LABEL_SIGN_DIST_RANGE_M[1])
            else:
                fallback_depth = min(max(fx * LABEL_REAL_LIGHT_HEIGHT_M / max(bh, 1.0), LABEL_LIGHT_DIST_RANGE_M[0]), LABEL_LIGHT_DIST_RANGE_M[1])

            p_world, lidar_matched = None, False
            if u is not None:
                inb = (u >= sx1) & (u <= sx2) & (v >= sy1) & (v <= sy2)
                if inb.sum() >= LIDAR_MIN_MATCHES:
                    match_idx = cand_idx[inb]
                    match_depth = depth[inb]
                    thresh = np.percentile(match_depth, LIDAR_DEPTH_PERCENTILE)
                    near = match_depth <= max(thresh, match_depth.min())
                    near_xyz = xyz[match_idx[near]]
                    near_depth = match_depth[near]
                    # A large/close-up bbox (light filling much of the frame)
                    # can have its "nearest 15%" span many metres -- not one
                    # compact object, several unrelated things at similar
                    # depth. Isolate the single tight group before trusting
                    # any of it (see MAX_LIDAR_CLUSTER_SPAN_M).
                    cluster_mask = _largest_spatial_cluster_mask(near_xyz)
                    if cluster_mask is not None:
                        cluster_xyz = near_xyz[cluster_mask]
                        span = cluster_xyz.max(axis=0) - cluster_xyz.min(axis=0)
                        if span.max() <= MAX_LIDAR_CLUSTER_SPAN_M:
                            matched_depth_repr = float(near_depth[cluster_mask].mean())
                            if matched_depth_repr <= fallback_depth * MAX_LIDAR_MONO_RATIO:
                                p_world = cluster_xyz.mean(axis=0)
                                lidar_matched = True
                    # else: no compact real object found inside the box (too
                    # spread out, or LiDAR match implausibly far vs. the
                    # monocular estimate -- almost certainly saw through to
                    # background); fall through to the monocular position.

            if p_world is None:
                p_cam = fallback_depth * np.array([(cx - K[0, 2]) / K[0, 0], (cy - K[1, 2]) / K[1, 1], 1.0])
                p_world = cam_xyz + R_cam_to_world @ p_cam

            n_lidar += lidar_matched
            n_fallback += not lidar_matched
            out.append({
                **d,
                "utm_easting": p_world[0],
                "utm_northing": p_world[1],
                "utm_z": p_world[2],
                "bearing_rad": bearing,
                "lidar_matched": lidar_matched,
            })

    print(f"  localize_detections_via_lidar: {n_lidar} LiDAR-matched, {n_fallback} fell back to size-estimate "
          f"({n_out_of_bbox} out-of-tile detections skipped)")
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def filter_detections_for_labels(dets):
    lights_by_image = defaultdict(list)
    for d in dets:
        if d["kind"] == "light" and d["sam3_score"] >= LABEL_SCORE_THRESHOLD:
            lights_by_image[d["image"]].append(d)

    kept = []
    for d in dets:
        if d["sam3_score"] < LABEL_SCORE_THRESHOLD:
            continue
        if d["kind"] == "sign":
            overlaps_light = any(
                _iou(d["bbox_px"], l["bbox_px"]) > LABEL_IOU_THRESHOLD
                for l in lights_by_image.get(d["image"], []))
            if overlaps_light:
                continue
            if "stvo_code" in d:
                # bucket_signs_sam3.py already applied its own confidence
                # floor + margin before writing this field -- a different,
                # separately-validated confidence story than clf_conf below,
                # so no "(REVIEW)" hedge here.
                text = d.get("label", "Sign")
            else:
                label, conf = d.get("label", "Sign"), d.get("clf_conf", 0.0)
                text = f"{label} (REVIEW)" if conf >= LABEL_CLF_CONF_THRESHOLD else "Sign"
        else:
            text = "Traffic Light"
        kept.append({**d, "tag_text": text})
    return kept


def make_label_plate(text, w=320, h=64):
    """High-contrast floating label with a dark background plate -- same
    rendering as phase4c_stvo_sprites.make_label_tex."""
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (180, 180, 180), 2)
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), (45, 45, 45), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = w, h
    for sc in [0.75, 0.65, 0.55, 0.45, 0.38]:
        (tw, th), _ = cv2.getTextSize(text, font, sc, 2)
        if tw <= w - 20:
            break
    x = max(10, (w - tw) // 2)
    y = max(th + 8, (h + th) // 2)
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
        cv2.putText(img, text, (x + dx, y + dy), font, sc, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, sc, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def place_label_plate(centre, tex, bearing_rad, w_m, h_m):
    """Turn a rendered text image into a small billboard of colored 3D
    points facing back along the camera->sign bearing -- same technique as
    phase4c_stvo_sprites.place_billboard, adapted to this file's compass
    bearing convention (x=E=sin(bearing), y=N=cos(bearing))."""
    face_dir = math.pi / 2.0 - bearing_rad  # compass bearing -> math angle
    right = np.array([math.cos(face_dir + math.pi / 2), math.sin(face_dir + math.pi / 2), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    th, tw = tex.shape[:2]
    nc = 96
    nr = max(int(nc * th / tw), 8)
    resized = cv2.resize(tex, (nc, nr), interpolation=cv2.INTER_AREA)

    pxyz, prgb = [], []
    for row in range(nr):
        for col in range(nc):
            bgr = resized[row, col]
            if int(bgr.max()) < 10:
                continue
            un = (col / (nc - 1)) - 0.5
            vn = 0.5 - (row / (nr - 1))
            pxyz.append(centre + right * (un * w_m) + up * (vn * h_m))
            prgb.append([bgr[2], bgr[1], bgr[0]])

    if not pxyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.array(pxyz), np.array(prgb, dtype=np.uint8)


def bake_hover_labels(las_path_in, detections_jsonl, out_path,
                       camextr_path=CAMEXTR_PATH, pinhole_k_path=PINHOLE_K_PATH):
    """Add floating sign/light hover tags to a (colorized or not) LAZ."""
    print(f"Baking hover labels onto {las_path_in}")
    las = laspy.read(las_path_in)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    bbox = {"e_min": xyz[:, 0].min(), "e_max": xyz[:, 0].max(),
            "n_min": xyz[:, 1].min(), "n_max": xyz[:, 1].max()}
    print(f"  cloud bbox: E[{bbox['e_min']:.1f},{bbox['e_max']:.1f}] "
          f"N[{bbox['n_min']:.1f},{bbox['n_max']:.1f}], {len(xyz):,} pts")

    pinhole_k = load_pinhole_K()

    dets = localize_detections_via_lidar(detections_jsonl, camextr_path, pinhole_k, xyz, bbox)
    print(f"  {len(dets)} raw detections localized inside this tile")
    # Detections localize_detections_via_lidar couldn't ray-cast against a
    # real point (sparse LiDAR coverage for that patch) fall back to the old
    # assumed-size depth guess -- empirically a mean ~40m placement error on
    # this tile's unmatched set (vs ~0.3m for the matched set), which reads
    # as "the tag is hovering over some unrelated object" in a viewer. A
    # wrong tag is worse than no tag here, so only bake LiDAR-matched ones.
    n_before = len(dets)
    dets = [d for d in dets if d["lidar_matched"]]
    print(f"  {len(dets)}/{n_before} have a real LiDAR-matched position -- "
          f"dropping the rest rather than showing a badly-placed guess")
    kept = filter_detections_for_labels(dets)
    print(f"  {len(kept)} survive filtering (score>={LABEL_SCORE_THRESHOLD}, overlap-dedup, ...)")

    # Small pad above the ray-estimated sign/light position itself (utm_z),
    # not a nearest-ground-point + fixed-height guess -- utm_z already
    # reflects the camera's real pitch/roll via the full 3D unprojection.
    LABEL_Z_PAD_M = 0.35
    new_xyz, new_rgb = [], []
    for d in kept:
        centre = np.array([d["utm_easting"], d["utm_northing"], d["utm_z"] + LABEL_Z_PAD_M])
        tex = make_label_plate(d["tag_text"])
        pxyz, prgb = place_label_plate(centre, tex, d["bearing_rad"], LABEL_PLATE_W_M, LABEL_PLATE_H_M)
        if len(pxyz):
            new_xyz.append(pxyz)
            new_rgb.append(prgb)

    new_xyz = np.vstack(new_xyz) if new_xyz else np.empty((0, 3))
    new_rgb = np.vstack(new_rgb) if new_rgb else np.empty((0, 3), dtype=np.uint8)
    print(f"  {len(new_xyz):,} label points generated for {len(kept)} tags")

    if "red" not in las.point_format.dimension_names:
        las = laspy.convert(las, point_format_id=3)
        for ch, val in zip(("red", "green", "blue"), UNCOLORED_FALLBACK):
            setattr(las, ch, np.full(len(las.x), val, dtype=np.uint16) * 257)

    out_las = laspy.LasData(header=las.header)
    out_las.x = np.concatenate([las.x, new_xyz[:, 0]])
    out_las.y = np.concatenate([las.y, new_xyz[:, 1]])
    out_las.z = np.concatenate([las.z, new_xyz[:, 2]])
    out_las.red = np.concatenate([np.asarray(las.red), new_rgb[:, 0].astype(np.uint16) * 257])
    out_las.green = np.concatenate([np.asarray(las.green), new_rgb[:, 1].astype(np.uint16) * 257])
    out_las.blue = np.concatenate([np.asarray(las.blue), new_rgb[:, 2].astype(np.uint16) * 257])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_las.write(out_path)
    print(f"  saved: {out_path}")
    return out_path, kept


def colorize(laz_path, pass_filter=None, out_name=None):
    part_match = re.search(r"part(\d+)", os.path.basename(laz_path))
    part_name = f"part{part_match.group(1)}" if part_match else os.path.basename(laz_path).replace(".laz", "")
    print(f"Colorizing {part_name} from {laz_path}" + (f" ({pass_filter}-only images)" if pass_filter else ""))

    bbox = get_laz_bbox(laz_path)
    print(f"  bbox: E[{bbox['e_min']:.1f},{bbox['e_max']:.1f}] N[{bbox['n_min']:.1f},{bbox['n_max']:.1f}]")

    camextr = load_camextr(bbox, pass_filter=pass_filter)
    print(f"  CamExtr entries in bbox: {len(camextr)}")
    Kmats = load_pinhole_K()

    las = laspy.read(laz_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    n = len(xyz)
    print(f"  points: {n:,}")

    print("  building spatial index...")
    t0 = time.time()
    tree = cKDTree(xyz[:, :2])
    print(f"    done in {time.time()-t0:.1f}s")

    best_depth = np.full(n, np.inf, dtype=np.float64)
    colors_bgr = np.zeros((n, 3), dtype=np.float64)
    colored_mask = np.zeros(n, dtype=bool)

    n_images_used = 0
    for i, e in enumerate(camextr):
        img_path = os.path.join(DATA_FIX_DIR, e["Image"])
        if not os.path.exists(img_path):
            continue
        sn = str(e["SerialNr"])
        K = Kmats.get(sn)
        if K is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        cam_xy = np.array(e["Xyz"][:2])
        cand_idx = np.array(tree.query_ball_point(cam_xy, POINT_SEARCH_RADIUS_M))
        if len(cand_idx) == 0:
            continue

        Xyz_cam = np.array(e["Xyz"])
        T_world_cam = np.linalg.inv(transformMatrix(*np.radians(e["Hrp"]), *Xyz_cam))
        pts_h = np.hstack([xyz[cand_idx], np.ones((len(cand_idx), 1))])
        pts_cam = (T_world_cam @ pts_h.T).T[:, :3]

        depth = pts_cam[:, 2]
        front = depth > MIN_FORWARD_DEPTH_M
        if not front.any():
            continue
        cand_idx_f, pts_cam_f, depth_f = cand_idx[front], pts_cam[front], depth[front]

        proj = (K @ pts_cam_f.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        mask = VEHICLE_OCCLUSION_MASK.get(sn)
        if mask is not None:
            row_cutoff, col_cutoff = mask
            inb &= ~((v >= row_cutoff) & (u >= col_cutoff))
        if not inb.any():
            continue
        sky = compute_sky_mask(img)
        u_idx = np.clip(np.floor(u[inb]).astype(np.int64), 0, w - 1)
        v_idx = np.clip(np.floor(v[inb]).astype(np.int64), 0, h - 1)
        inb[inb] &= ~sky[v_idx, u_idx]
        if not inb.any():
            continue
        cand_idx_f, u, v, depth_f = cand_idx_f[inb], u[inb], v[inb], depth_f[inb]

        # Raw "closest wins" prefers close-but-grazing shots (e.g. down/up
        # cameras) over slightly farther but boresight-aligned ones, which
        # hurts quality at edges. Penalize samples far from the principal
        # point so boresight-aligned, less-peripheral views are preferred.
        r_pix = np.sqrt((u - K[0, 2]) ** 2 + (v - K[1, 2]) ** 2)
        r_max = np.hypot(w, h) / 2.0
        score = depth_f * (1.0 + 2.0 * (r_pix / r_max) ** 2) * CAMERA_SCORE_PENALTY.get(sn, 1.0)

        # --- per-image z-buffer: best-score point wins each pixel bin ---
        bin_id = np.floor(v).astype(np.int64) * w + np.floor(u).astype(np.int64)
        order = np.lexsort((score, bin_id))             # sort by bin, then score ascending
        bin_sorted = bin_id[order]
        first_in_bin = np.empty(len(order), dtype=bool)
        first_in_bin[0] = True
        first_in_bin[1:] = bin_sorted[1:] != bin_sorted[:-1]
        winners = order[first_in_bin]                   # best-score index per occupied pixel

        w_idx, w_u, w_v, w_score = cand_idx_f[winners], u[winners], v[winners], score[winners]

        # --- global best-observation-wins across all images ---
        better = w_score < best_depth[w_idx]
        if not better.any():
            continue
        w_idx, w_u, w_v, w_score = w_idx[better], w_u[better], w_v[better], w_score[better]

        sampled = bilinear_sample(img, w_u, w_v)
        colors_bgr[w_idx] = sampled
        best_depth[w_idx] = w_score
        colored_mask[w_idx] = True
        n_images_used += 1

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(camextr)}] images processed, {colored_mask.sum():,}/{n:,} points colored so far")

    print(f"  images actually used: {n_images_used}/{len(camextr)}")
    print(f"  points colored: {colored_mask.sum():,}/{n:,} ({100*colored_mask.mean():.1f}%)")

    colors_rgb = colors_bgr[:, ::-1]
    colors_rgb[~colored_mask] = UNCOLORED_FALLBACK
    colors_16 = np.clip(colors_rgb, 0, 255).astype(np.uint16) * 257  # LAS RGB is 16-bit

    if "red" not in las.point_format.dimension_names:
        las = laspy.convert(las, point_format_id=3)
    las.red = colors_16[:, 0]
    las.green = colors_16[:, 1]
    las.blue = colors_16[:, 2]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name or f"colorized_{part_name}.laz")
    las.write(out_path)
    print(f"  saved: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colorize_pointcloud.py <laz_file> [--pass1|--pass2] [--out <name.laz>]")
        print("       python colorize_pointcloud.py <laz_file> --labels <detections.jsonl> [--labels-out <out.laz>]")
        sys.exit(1)
    if "--labels" in sys.argv:
        laz_in = sys.argv[1]
        det_path = sys.argv[sys.argv.index("--labels") + 1]
        if "--labels-out" in sys.argv:
            out_path = sys.argv[sys.argv.index("--labels-out") + 1]
        else:
            base = os.path.basename(laz_in).replace(".laz", "")
            out_path = os.path.join(OUTPUT_DIR, f"{base}_labeled.laz")
        bake_hover_labels(laz_in, det_path, out_path)
    else:
        pf = "pass1" if "--pass1" in sys.argv else ("pass2" if "--pass2" in sys.argv else None)
        out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
        colorize(sys.argv[1], pass_filter=pf, out_name=out_name)
