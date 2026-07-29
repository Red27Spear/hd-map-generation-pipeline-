#!/usr/bin/env python3
"""
Phase 4d — Semantic Sprites: StVO Templates + Hybrid Crop, Consuming Phase 3 v6
=================================================================================
Rewritten from a draft ("Phase 4c") that had two problems relative to what
this project actually validated:

  FIXED — input format:
    The draft required a `merged_<part>.laz` with a `classification` field
    (1=road, 2=vertical). That format doesn't exist in this pipeline. This
    script takes the SAME raw/cleaned LAZ Phase 3 uses and runs its own
    RANSAC road segmentation, exactly like the original validated Phase 4b.

  KEPT / IMPROVED from the draft:
    - StVO-compliant template generator (colors, shapes, per-class symbols)
    - Hybrid crop-vs-template decision, but now driven by the
      `use_crop_recommended` flag Phase 3 v6 already computed (classifier
      conf >= 0.95 AND Laplacian sharpness >= 100) rather than a separate,
      looser distance+confidence check — one hybrid decision, made once,
      consumed consistently everywhere downstream.

  NEW, to match Phase 3 v6's output fields:
    - `source_method` (lidar_verified / camera_only_unverified) -> tint +
      "(UNVERIFIED)" label suffix, dashed-looking marker
    - `likely_back_face` -> purple tint + "(?)" suffix, since we can't
      confirm the sign's true face without the (unresolved) full projection
    - `co_located_with` -> sign billboard is placed at the Z Phase 3 already
      offset; this script does not re-derive that position
    - Bare-pole `verification_status` (unclassified_pole / adjacent_to_
      detection / probable_sign) -> three distinct tint colors

Usage:
    python semantic_classification.py <laz_file>

Output:
    <output-dir>/semantic_<part>.laz
"""

import sys, os, json, time, re, math
import numpy as np
import cv2
import laspy

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — override with environment variables of the same name
# ═══════════════════════════════════════════════════════════════════════════════
PHASE0_DIR   = os.environ.get("PHASE0_DIR", "../camera_preprocessing/output")
PHASE3_DIR   = os.environ.get("PHASE3_DIR", "../camera_feature_extraction/output")
CAMEXTR_PATH = os.environ.get("CAMEXTR_PATH", "../config/CamExtr.json")
OUTPUT_DIR   = os.environ.get("SEMANTIC_OUTPUT_DIR", "./output")
CROP_DIR     = os.environ.get("CROP_OUTPUT_DIR", "../camera_feature_extraction/output/detection_crops")

SIGN_SIZE_DEFAULT = 0.60
SIGN_SIZE_WARNING = 0.70
SIGN_SIZE_LARGE   = 0.90
LIGHT_W, LIGHT_H  = 0.40, 1.10
POLE_THICKNESS_M  = 0.08

MIN_SIGN_CONFIDENCE = 0.55   # YOLO detection confidence floor (catches tires etc.)
MIN_CLUSTER_PTS  = 10
MIN_WIDTH        = 0.03
MAX_WIDTH        = 3.0
MIN_HEIGHT       = 0.3
MAX_DIST_APPROX  = 15.0
DEDUP_DIST       = 1.5      # extra safety net; Phase 3 v6 already dedupes

BILLBOARD_RES  = 128
LABEL_OFFSET_Z = 0.45
LABEL_H_M      = 0.35
LABEL_W_M      = 2.00
LABEL_TEX_H    = 40
LABEL_TEX_W    = 260

ROAD_INT_MIN = 50
ROAD_INT_MAX = 255
BASE_COLOR   = np.array([20, 20, 25], dtype=np.uint8)

CLUSTER_R_SCALE = 2.5
CLUSTER_R_MIN   = 0.3
CLUSTER_R_MAX   = 1.2

RANSAC_DIST = 0.15
RANSAC_N    = 3
RANSAC_ITER = 2000

BBOX_MARGIN = 30.0

STVO_RED, STVO_BLUE = (0, 0, 200), (200, 80, 0)
STVO_WHITE, STVO_BLACK = (255, 255, 255), (0, 0, 0)

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

def class_to_shape(cls_name):
    if cls_name.startswith("regulatory--no-") or cls_name.startswith("regulatory--maximum-speed"):
        return "prohibitory"
    if cls_name.startswith("regulatory--yield"):
        return "yield"
    if cls_name.startswith("regulatory--stop"):
        return "stop"
    if cls_name.startswith("regulatory--"):
        return "mandatory"
    if cls_name.startswith("warning--"):
        return "danger"
    if cls_name.startswith("information--"):
        return "info"
    if cls_name.startswith("complementary--"):
        return "complementary"
    return "other"

def sign_real_size(cls_name):
    shape = class_to_shape(cls_name)
    if shape in ("stop", "yield"):
        return SIGN_SIZE_LARGE
    if shape == "danger":
        return SIGN_SIZE_WARNING
    if shape == "info":
        return SIGN_SIZE_LARGE if ("motorway" in cls_name or "hospital" in cls_name) else SIGN_SIZE_DEFAULT
    if shape == "prohibitory":
        if any(k in cls_name for k in ("height", "weight", "axel")):
            return SIGN_SIZE_WARNING
        return SIGN_SIZE_DEFAULT
    return SIGN_SIZE_DEFAULT

def sign_stvo_number(cls_name):
    mapping = {
        "regulatory--no-entry": "267", "regulatory--stop": "206",
        "regulatory--yield": "205", "regulatory--maximum-speed-limit": "274",
        "regulatory--no-u-turn": "272", "regulatory--no-overtaking": "276",
        "regulatory--no-parking": "286", "regulatory--height-limit": "265",
        "regulatory--weight-limit": "263", "regulatory--go-straight": "209",
        "regulatory--keep-right": "222", "regulatory--keep-left": "222",
        "warning--crossroads": "102", "warning--curve-right": "103",
        "warning--curve-left": "103", "warning--roundabout": "215",
        "warning--roadworks": "123", "warning--children": "136",
        "warning--pedestrians-crossing": "133", "warning--slippery-road-surface": "114",
        "warning--traffic-signals": "131",
        "warning--railroad-crossing-with-barriers": "151",
        "warning--railroad-crossing-without-barriers": "150",
        "information--parking": "314", "information--hospital": "358",
        "information--motorway": "330", "information--gas-station": "361",
    }
    return mapping.get(cls_name, "???")


# ═══════════════════════════════════════════════════════════════════════════════
# STVO TEMPLATE GENERATOR (kept from the draft — genuinely good work)
# ═══════════════════════════════════════════════════════════════════════════════
def make_template_stvo(cls_name, size=256):
    shape = class_to_shape(cls_name)
    stvo_num = sign_stvo_number(cls_name)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 2 - 14

    def _txt(text, scale=1.0, thick=2, color=STVO_BLACK):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text, (cx - tw//2, cy + th//2), font, scale, color, thick, cv2.LINE_AA)

    if shape == "prohibitory":
        cv2.circle(img, (cx, cy), r, STVO_RED, -1)
        cv2.circle(img, (cx, cy), r-10, STVO_WHITE, -1)
        if "no-entry" in cls_name:
            cv2.rectangle(img, (cx-r+28, cy-14), (cx+r-28, cy+14), STVO_WHITE, -1)
        elif "speed" in cls_name or "limit" in cls_name:
            cv2.circle(img, (cx, cy), r-22, STVO_RED, 5)
            m = re.search(r'(\d+)', cls_name)
            _txt(m.group(1) if m else "50", 1.3, 3)
        elif "height" in cls_name:
            _txt("h", 1.2, 3)
        elif "weight" in cls_name or "axel" in cls_name:
            _txt("t", 1.2, 3)
        else:
            cv2.line(img, (cx-r+25, cy-r+25), (cx+r-25, cy+r-25), STVO_RED, 10)

    elif shape == "yield":
        pts = np.array([[cx, cy+r-8], [cx-r+15, cy-int(r*.55)], [cx+r-15, cy-int(r*.55)]], np.int32)
        cv2.fillPoly(img, [pts], STVO_WHITE)
        cv2.polylines(img, [pts], True, STVO_RED, 10)
        _txt("YIELD", 0.7, 2)

    elif shape == "stop":
        angles = np.linspace(0, 2*np.pi, 9)[:-1] + np.pi/8
        pts = np.array([(cx+int((r-8)*np.cos(a)), cy+int((r-8)*np.sin(a))) for a in angles], np.int32)
        cv2.fillPoly(img, [pts], STVO_RED)
        _txt("STOP", 0.75, 2, STVO_WHITE)

    elif shape == "mandatory":
        cv2.circle(img, (cx, cy), r, STVO_BLUE, -1)
        cv2.circle(img, (cx, cy), r, STVO_WHITE, 4)
        if "straight" in cls_name:
            _txt("^", 1.3, 3, STVO_WHITE)
        elif "right" in cls_name:
            _txt(">", 1.3, 3, STVO_WHITE)
        elif "left" in cls_name:
            _txt("<", 1.3, 3, STVO_WHITE)
        else:
            _txt("^", 1.3, 3, STVO_WHITE)

    elif shape == "danger":
        margin = size // 8
        pts = np.array([[cx, margin+8], [margin+8, size-margin-8], [size-margin-8, size-margin-8]], np.int32)
        cv2.fillPoly(img, [pts], STVO_WHITE)
        cv2.polylines(img, [pts], True, STVO_RED, 10)
        if "roadworks" in cls_name:
            _txt("A", 1.4, 3)
        elif "children" in cls_name:
            _txt("KIDS", 0.6, 2)
        elif "pedestrians" in cls_name or "crossing" in cls_name:
            _txt("PED", 0.7, 2)
        elif "slippery" in cls_name:
            _txt("SLIP", 0.6, 2)
        elif "curve" in cls_name:
            _txt("CURVE", 0.5, 2)
        elif "roundabout" in cls_name:
            _txt("O", 1.2, 3)
        else:
            cv2.rectangle(img, (cx-7, cy-22), (cx+7, cy+2), STVO_BLACK, -1)
            cv2.circle(img, (cx, cy+22), 9, STVO_BLACK, -1)

    elif shape == "info":
        cv2.rectangle(img, (12, 12), (size-12, size-12), STVO_BLUE, -1)
        cv2.rectangle(img, (12, 12), (size-12, size-12), STVO_WHITE, 5)
        if "parking" in cls_name:
            _txt("P", 1.5, 3, STVO_WHITE)
        elif "hospital" in cls_name:
            _txt("H", 1.5, 3, STVO_WHITE)
        elif "motorway" in cls_name:
            _txt("A", 1.5, 3, STVO_WHITE)
        else:
            _txt("i", 1.5, 3, STVO_WHITE)

    else:
        cv2.rectangle(img, (12, 12), (size-12, size-12), (240, 240, 240), -1)
        cv2.rectangle(img, (12, 12), (size-12, size-12), STVO_BLACK, 4)
        _txt("...", 0.8, 2)

    _txt_corner = f"StVO {stvo_num}"
    cv2.putText(img, _txt_corner, (size-90, size-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (100, 100, 100), 1, cv2.LINE_AA)
    return img


def make_light_texture(w=128, h=320, verified=True):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    border_color = (60, 60, 60) if verified else (0, 100, 255)
    cv2.rectangle(img, (8, 8), (w-8, h-8), border_color, 4 if verified else 8)
    rd, cx = w // 3, w // 2
    for cy_l, col in zip([h//6, h//2, 5*h//6], [(0,0,220), (0,180,255), (0,220,0)]):
        cv2.circle(img, (cx, cy_l), rd+3, (80,80,80), -1)
        cv2.circle(img, (cx, cy_l), rd, col, -1)
        cv2.circle(img, (cx-rd//3, cy_l-rd//3), rd//4, (255,255,255), -1)
    return img


def make_label_tex(text, w=LABEL_TEX_W, h=LABEL_TEX_H):
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    cv2.rectangle(img, (0,0), (w-1,h-1), (180,180,180), 2)
    cv2.rectangle(img, (3,3), (w-4,h-4), (45,45,45), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for sc in [0.55, 0.48, 0.42, 0.36, 0.30]:
        (tw, th), _ = cv2.getTextSize(text, font, sc, 2)
        if tw <= w - 20:
            break
    x = max(10, (w-tw)//2); y = max(th+8, (h+th)//2)
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
        cv2.putText(img, text, (x+dx, y+dy), font, sc, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, sc, (255,255,255), 2, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# LAZ / ROAD
# ═══════════════════════════════════════════════════════════════════════════════
def load_laz(p):
    las = laspy.read(p)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    intensity = np.array(las.intensity, dtype=np.float32)
    print(f"    {len(xyz):,} pts")
    return xyz, intensity

def save_laz(xyz, colors, intensity, out):
    hdr = laspy.LasHeader(point_format=2, version="1.2")
    hdr.offsets = np.min(xyz, axis=0)
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    las.intensity = intensity.astype(np.uint16)
    las.red   = colors[:,0].astype(np.uint16) * 256
    las.green = colors[:,1].astype(np.uint16) * 256
    las.blue  = colors[:,2].astype(np.uint16) * 256
    las.write(out)

def ransac_road(xyz):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inl = pcd.segment_plane(RANSAC_DIST, RANSAC_N, RANSAC_ITER)
    m = np.zeros(len(xyz), dtype=bool)
    m[inl] = True
    print(f"    Road: {m.sum():,}/{len(xyz):,}")
    return m

def color_road(intensity, mask):
    c = np.tile(BASE_COLOR, (len(intensity), 1))
    if mask.sum() == 0:
        return c
    ri = intensity[mask]
    lo, hi = np.percentile(ri, 2), np.percentile(ri, 98)
    if hi - lo < 1:
        hi = lo + 1
    g = (ROAD_INT_MIN + np.clip((ri-lo)/(hi-lo), 0, 1) * (ROAD_INT_MAX-ROAD_INT_MIN)).astype(np.uint8)
    c[mask] = np.stack([g, g, g], axis=1)
    return c


def filter_dets(raw):
    kept = []
    for d in raw:
        tag = f"{d['type']:6s} h={d['height_above_ground']:.1f}m w={d['max_width']:.2f}m pts={d['n_points']}"
        is_camera_only = d.get("source_method") == "camera_only_unverified"

        if not is_camera_only:
            if d["n_points"] < MIN_CLUSTER_PTS:
                print(f"    DROPPED: {tag} -> n_points < {MIN_CLUSTER_PTS}")
                continue
            if d["max_width"] < MIN_WIDTH or d["max_width"] > MAX_WIDTH:
                print(f"    DROPPED: {tag} -> width outside [{MIN_WIDTH},{MAX_WIDTH}]")
                continue
            if d["height_above_ground"] < MIN_HEIGHT:
                print(f"    DROPPED: {tag} -> height < {MIN_HEIGHT}")
                continue
            dta = d.get("dist_to_approx", 0)
            if dta > MAX_DIST_APPROX and dta > 0:
                print(f"    DROPPED: {tag} -> dist_to_approx > {MAX_DIST_APPROX}")
                continue

        if d["type"] == "sign" and d.get("confidence", 0) < MIN_SIGN_CONFIDENCE:
            print(f"    DROPPED: sign conf={d.get('confidence',0):.2f} < {MIN_SIGN_CONFIDENCE} "
                  f"({d.get('specific_class','?')})")
            continue

        if is_camera_only:
            print(f"    KEPT (camera-only, unverified): {tag}")
        kept.append(d)

    # Extra safety-net dedup (Phase 3 v6 already deduped; this just catches
    # anything that slipped through, e.g. across a tile boundary re-run)
    deduped, used = [], set()
    for i, d in enumerate(kept):
        if i in used:
            continue
        best = d
        for j in range(i+1, len(kept)):
            if j in used:
                continue
            dist = np.linalg.norm(np.array(kept[j]["xyz"]) - np.array(d["xyz"]))
            if dist < DEDUP_DIST:
                used.add(j)
                if kept[j]["confidence"] > best["confidence"]:
                    best = kept[j]
        deduped.append(best)
    print(f"    Raw={len(raw)} -> Filtered={len(deduped)}")
    return deduped


def place_billboard(centre, tex, face_dir_rad, w_m, h_m):
    right = np.array([np.cos(face_dir_rad+np.pi/2), np.sin(face_dir_rad+np.pi/2), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    th, tw = tex.shape[:2]
    asp = tw / max(th, 1)
    if asp >= 1:
        nc = BILLBOARD_RES; nr = max(int(BILLBOARD_RES/asp), 8)
    else:
        nr = BILLBOARD_RES; nc = max(int(BILLBOARD_RES*asp), 8)
    resized = cv2.resize(tex, (nc, nr), interpolation=cv2.INTER_AREA)
    pxyz, prgb = [], []
    for row in range(nr):
        for col in range(nc):
            un = (col/(nc-1)) - 0.5
            vn = 0.5 - (row/(nr-1))
            pt = centre + right*(un*w_m) + up*(vn*h_m)
            bgr = resized[row, col]
            rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)
            if rgb.max() < 10:
                continue
            pxyz.append(pt); prgb.append(rgb)
    if not pxyz:
        return np.empty((0,3)), np.empty((0,3), dtype=np.uint8)
    return np.array(pxyz), np.array(prgb, dtype=np.uint8)


def make_sprite(det, cls76_name, cam_entry, use_crop=False, crop_img=None):
    dtype = det["type"]
    ground_z  = det.get("ground_z", det["xyz"][2] - det.get("height_above_ground", 2.0))
    height_ag = det.get("height_above_ground", 2.0)
    is_verified = det.get("source_method", "lidar_verified") == "lidar_verified"
    is_back_face = det.get("likely_back_face", False)

    if dtype == "light":
        centre = np.array([det["xyz"][0], det["xyz"][1], ground_z + height_ag])
    else:
        centre = det["xyz"].copy()   # already Z-offset by Phase 3 if co-located

    if dtype == "sign":
        sz = sign_real_size(cls76_name)
        w_m, h_m = sz, sz
    elif dtype == "light":
        w_m, h_m = LIGHT_W, LIGHT_H
    else:
        w_m, h_m = 0.20, 0.20

    if cam_entry is not None:
        cam_xy = np.array(cam_entry["Xyz"][:2])
        bearing = np.arctan2(centre[1]-cam_xy[1], centre[0]-cam_xy[0])
        heading_rad = bearing + np.pi
    else:
        heading_rad = 0.0

    all_xyz, all_rgb = [], []

    if dtype == "sign":
        if use_crop and crop_img is not None:
            tex = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            status = "CROP"
        else:
            tex = make_template_stvo(cls76_name)
            status = f"StVO-{sign_stvo_number(cls76_name)}"
        sx, sr = place_billboard(centre, tex, heading_rad, w_m, h_m)
        if len(sx): all_xyz.append(sx); all_rgb.append(sr)
    elif dtype == "light":
        tex = make_light_texture(verified=is_verified)
        sx, sr = place_billboard(centre, tex, heading_rad, w_m, h_m)
        if len(sx): all_xyz.append(sx); all_rgb.append(sr)
        status = "LIGHT" if is_verified else "LIGHT?"
    else:
        status = "NONE"

    # Label
    if dtype == "light":
        label_text = "TRAFFIC LIGHT" if is_verified else "TRAFFIC LIGHT (UNVERIFIED)"
    else:
        label_text = class_to_label(cls76_name) + f" [StVO {sign_stvo_number(cls76_name)}]"
        if not is_verified:
            label_text += " (UNVERIFIED)"
        if is_back_face:
            label_text += " (?)"

    lt = make_label_tex(label_text)
    lc = centre.copy()
    lc[2] += h_m/2 + LABEL_OFFSET_Z
    lx, lr = place_billboard(lc, lt, heading_rad, LABEL_W_M, LABEL_H_M)
    if len(lx): all_xyz.append(lx); all_rgb.append(lr)

    # Pole stem — only when we actually have a ground-connected cluster
    if dtype in ("sign", "light") and det.get("n_points", 0) > 0:
        pole_top = centre[2] - h_m/2
        if pole_top > ground_z + 0.1:
            n_pole = 60
            thickness = POLE_THICKNESS_M / 2
            pole_pts, pole_rgb = [], []
            for z in np.linspace(ground_z, pole_top, n_pole):
                for dx, dy in [(0,0),(thickness,0),(-thickness,0),(0,thickness),(0,-thickness)]:
                    pole_pts.append([centre[0]+dx, centre[1]+dy, z])
                    pole_rgb.append([180,180,180])
            all_xyz.append(np.array(pole_pts))
            all_rgb.append(np.array(pole_rgb, dtype=np.uint8))

    if not all_xyz:
        return np.empty((0,3)), np.empty((0,3), dtype=np.uint8), status
    return np.vstack(all_xyz), np.vstack(all_rgb), status


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage: python semantic_classification.py <laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    bn = os.path.basename(laz_path)
    m = re.search(r'(part\d+)', bn)
    pn = m.group(1) if m else bn.replace('.laz', '')

    print(f"\n{'='*70}\n  PHASE 4d - StVO Templates + Hybrid Crop (Phase 3 v6 input)\n"
          f"  Tile: {bn}  |  Part: {pn}\n{'='*70}")

    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    print("\n[1/6] Loading detections from Phase 3 v6 GeoJSON...")
    raw_dets = []
    for kind in ["signs_3d", "traffic_lights_3d", "poles_3d"]:
        path = os.path.join(PHASE3_DIR, f"{kind}_{pn}.geojson")
        if not os.path.isfile(path):
            print(f"    WARNING: {path} not found")
            continue
        with open(path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            c = feat["geometry"]["coordinates"]
            p = feat["properties"]
            raw_dets.append({
                "xyz": np.array(c, dtype=np.float64),
                "type": p.get("type", "unknown"),
                "class_id": p.get("class_id", -1),
                "confidence": p.get("confidence", 0.0),
                "max_width": p.get("max_width", 0.0),
                "height_above_ground": p.get("height_above_ground", 0.0),
                "ground_z": p.get("ground_z", 0.0),
                "n_points": p.get("n_points", 0),
                "dist_to_approx": p.get("dist_to_approx", 0.0),
                "specific_class": p.get("specific_class", "unknown"),
                "specific_label": p.get("specific_label", "UNKNOWN"),
                "classifier_conf": p.get("classifier_conf", 0.0),
                "crop_path": p.get("crop_path", ""),
                "use_crop_recommended": p.get("use_crop_recommended", False),
                "heading": p.get("heading", 0.0),
                "source_method": p.get("source_method", "lidar_verified"),
                "likely_back_face": p.get("likely_back_face", False),
                "co_located_with": p.get("co_located_with", None),
                "verification_status": p.get("verification_status", None),
            })
        print(f"    {len(gj.get('features',[]))} {kind}")

    dets = filter_dets(raw_dets)

    print("\n[2/6] Loading point cloud...")
    xyz, intensity = load_laz(laz_path)

    print("\n[3/6] Road segmentation...")
    road_mask = ransac_road(xyz)
    colors = color_road(intensity, road_mask)

    print("\n[4/6] Loading camera entries...")
    with open(CAMEXTR_PATH) as f:
        ce = json.load(f)
    all_e = ce.get("Profiler_0", [])
    xn, xx = xyz[:,0].min()-BBOX_MARGIN, xyz[:,0].max()+BBOX_MARGIN
    yn, yx = xyz[:,1].min()-BBOX_MARGIN, xyz[:,1].max()+BBOX_MARGIN
    cam_entries = [e for e in all_e if xn<=e["Xyz"][0]<=xx and yn<=e["Xyz"][1]<=yx]
    cam_positions = np.array([e["Xyz"] for e in cam_entries]) if cam_entries else np.empty((0,3))
    print(f"    {len(cam_entries)} images")

    print(f"\n[5/6] Processing {len(dets)} detections...\n")
    all_sx, all_sr, all_si = [], [], []
    n_spr = n_cl = n_used_crop = n_used_template = 0
    report = []

    for i, det in enumerate(dets):
        dtype, mw, hag = det["type"], det["max_width"], det["height_above_ground"]

        if det.get("n_points", 0) > 0:
            r = np.clip(mw*CLUSTER_R_SCALE, CLUSTER_R_MIN, CLUSTER_R_MAX)
            dm = np.linalg.norm(xyz - det["xyz"], axis=1)
            cm = (dm < r) & (~road_mask)
            nc = cm.sum()
            if dtype == "sign":
                tint = [160,32,240] if det.get("likely_back_face") else \
                       ([255,140,0] if det.get("source_method")=="camera_only_unverified" else [220,30,30])
            elif dtype == "light":
                tint = [255,140,0] if det.get("source_method")=="camera_only_unverified" else [240,220,40]
            elif dtype == "pole":
                status = det.get("verification_status", "unclassified_pole")
                tint = {"probable_sign": [255,100,200],
                        "adjacent_to_detection": [255,200,60]}.get(status, [60,200,220])
            else:
                tint = [255,140,0]
            if nc > 0:
                colors[cm] = tint
                n_cl += nc
        else:
            nc = 0

        if dtype == "pole":
            report.append(f"det{i+1:02d}  pole  TINT pts={nc}  status={det.get('verification_status')}")
            continue

        specific_class = det.get("specific_class", "unknown")
        crop_path = det.get("crop_path", "")
        use_crop = False
        crop_img = None

        if dtype == "sign" and det.get("use_crop_recommended") and crop_path and os.path.isfile(crop_path):
            crop_img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if crop_img is not None and crop_img.size > 0:
                use_crop = True
                n_used_crop += 1

        if dtype == "sign":
            cls76 = specific_class if specific_class not in ("unknown", "", None) else "warning--other-danger"
            label = class_to_label(cls76)
            if not use_crop:
                n_used_template += 1
        elif dtype == "light":
            cls76, label = "traffic-light", "TRAFFIC LIGHT"
        else:
            cls76, label = "unknown", "UNKNOWN"

        print(f"  [{i+1}/{len(dets)}] {dtype:5s} {cls76:40s} conf={det['confidence']:.2f} "
              f"h={hag:.1f}m w={mw:.2f}m src={det.get('source_method')}")

        if len(cam_positions) > 0:
            d2c = np.linalg.norm(cam_positions[:, :2] - det["xyz"][:2], axis=1)
            nearest_cam = cam_entries[int(np.argmin(d2c))]
        else:
            nearest_cam = None

        sx, sr, status = make_sprite(det, cls76, nearest_cam, use_crop=use_crop, crop_img=crop_img)
        if len(sx) > 0:
            all_sx.append(sx); all_sr.append(sr)
            all_si.append(np.full(len(sx), 220, dtype=np.float32))
            n_spr += len(sx)

        print(f"    -> sprite={len(sx):,}pts label='{label}' tex={status}")
        report.append(f"det{i+1:02d}  {dtype:6s}  {label:30s}  {status}")

    print("\n[6/6] Saving...")
    if all_sx:
        fxyz = np.vstack([xyz, np.vstack(all_sx)])
        fcol = np.vstack([colors, np.vstack(all_sr)])
        fint = np.concatenate([intensity, np.concatenate(all_si)])
    else:
        fxyz, fcol, fint = xyz, colors, intensity

    out = os.path.join(OUTPUT_DIR, f"semantic_{pn}.laz")
    save_laz(fxyz, fcol, fint, out)

    print(f"  Used crops:     {n_used_crop}")
    print(f"  Used templates: {n_used_template}")
    print(f"  Sprite pts:     {n_spr:,}")
    print(f"  Cluster tint:   {n_cl:,}")
    print(f"  -> {out} ({len(fxyz):,} pts)")
    print(f"  Done in {time.time()-t0:.1f}s\n")

    rpt = os.path.join(CROP_DIR, f"detection_report_{pn}.txt")
    with open(rpt, "w") as f:
        f.write("\n".join([
            "Phase 4d Detection Report", "="*50,
            f"Tile: {bn}", f"Used crops: {n_used_crop}", f"Used templates: {n_used_template}",
            f"Sprite pts: {n_spr:,}", "", *report,
        ]) + "\n")
    print(f"  Report -> {rpt}")


if __name__ == "__main__":
    main()
