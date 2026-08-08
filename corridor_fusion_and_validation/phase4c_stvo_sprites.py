#!/usr/bin/env python3
"""
Phase 4c — Semantic Sprites with Actual Crops + StVO Templates
==============================================================
Reads merged .laz with classification field (1=road, 2=vertical).

For each detected object:
  - If crop image exists and camera was close (< 15m): use ACTUAL crop as billboard texture
  - Else: use accurate StVO-compliant template matching the 76-class label

Color coding:
  - Grey = road (cls 1)
  - Dark blue = unclassified vertical (cls 2, no object)
  - Red = verified sign panels
  - Yellow = verified traffic lights
  - Orange = unverified detections
  - Purple = back-facing signs
  - Cyan = bare poles
  - Dark green = pole stems (in multi-attachment poles)

Output: semantic_<part>.laz with billboards, labels, and poles.
"""

import sys, os, json, time, re, math
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import laspy

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# These module-level paths are only used by this script's OWN standalone
# __main__ mode below (a from-scratch semantic-sprite pipeline run) -- not
# by make_light_texture_stvo(), the pure function other scripts in this
# stage import from this file. None of phase0/phase2/phase3's intermediate
# outputs are bundled in this repo; edit these or run with your own tile's
# equivalent directory structure before using __main__ mode directly.
PHASE0_DIR    = "./output/data_fix"           # from phase0_image_fix.py
PHASE2_DIR    = "./output/phase2"             # from phase2_*.py
PHASE3_DIR    = "./output/phase3"             # detection geojson outputs, not covered by this stage
CAMEXTR_PATH  = "../config/CamExtr.json"
INTRINSIC_XML = "../config/9020C_0140_toScanner_final.xml"
OUTPUT_DIR    = "./output/lidar_color"
CROP_DIR      = "./output/lidar_color/detection_crops"

# Realistic German sign dimensions (metres) — StVO compliant
SIGN_SIZE_DEFAULT = 0.60      # Standard prohibitory/mandatory (Sign 250-299)
SIGN_SIZE_WARNING = 0.70      # Warning triangles (Sign 100-199)
SIGN_SIZE_LARGE   = 0.90      # Stop, yield, large info
SIGN_SIZE_SMALL   = 0.42      # Small prohibitory (Sign 253-258)
LIGHT_W, LIGHT_H  = 0.30, 0.85
POLE_THICKNESS_M  = 0.08

# Crop usage threshold
MAX_CROP_DISTANCE_M = 15.0    # Only use actual crop if camera was within 15m
MIN_CROP_CONFIDENCE = 0.50    # Minimum classifier confidence to use crop

# Filtering
MIN_CLUSTER_PTS  = 10
MIN_WIDTH        = 0.03
MAX_WIDTH        = 3.0
MIN_HEIGHT       = 0.3
MAX_DIST_APPROX  = 15.0
DEDUP_DIST       = 1.5
MIN_SIGN_CONFIDENCE = 0.55

# Camera / projection
BBOX_MARGIN  = 30.0
MIN_DEPTH    = 0.3
MAX_DEPTH    = 50.0

# Billboard
BILLBOARD_RES = 128
LABEL_OFFSET_Z = 0.45
LABEL_H_M      = 0.35
LABEL_W_M      = 2.00
LABEL_TEX_H    = 40
LABEL_TEX_W    = 220

# Road coloring
ROAD_INT_MIN = 50
ROAD_INT_MAX = 255
BASE_COLOR   = np.array([20, 20, 25], dtype=np.uint8)

# Cluster tint radius
CLUSTER_R_SCALE = 2.5
CLUSTER_R_MIN   = 0.3
CLUSTER_R_MAX   = 1.2

# Object type colors (for LiDAR point tinting)
TINT_SIGN        = np.array([220, 30, 30], dtype=np.uint8)     # Red
TINT_LIGHT       = np.array([240, 220, 40], dtype=np.uint8)  # Yellow
TINT_POLE        = np.array([60, 200, 220], dtype=np.uint8)  # Cyan
TINT_UNVERIFIED  = np.array([255, 140, 0], dtype=np.uint8)    # Orange
TINT_BACKFACE    = np.array([160, 32, 240], dtype=np.uint8)   # Purple
TINT_POLE_STEM   = np.array([34, 139, 34], dtype=np.uint8)   # Dark green

# StVO official colors (BGR for OpenCV)
STVO_RED    = (0, 0, 200)      # Prohibitory border, warning border
STVO_BLUE   = (200, 80, 0)     # Mandatory background
STVO_WHITE  = (255, 255, 255)  # Background
STVO_BLACK  = (0, 0, 0)        # Symbols
STVO_YELLOW = (0, 200, 255)    # Priority road


# 76 class names (must match Phase 3)
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
    if len(parts) > 1:
        return parts[1].replace("-", " ").upper()
    return cls_name.upper()


def class_to_shape(cls_name):
    if cls_name.startswith("regulatory--no-") or cls_name.startswith("regulatory--maximum-speed"):
        return "prohibitory"
    elif cls_name.startswith("regulatory--yield"):
        return "yield"
    elif cls_name.startswith("regulatory--stop"):
        return "stop"
    elif cls_name.startswith("regulatory--"):
        return "mandatory"
    elif cls_name.startswith("warning--"):
        return "danger"
    elif cls_name.startswith("information--"):
        return "info"
    elif cls_name.startswith("complementary--"):
        return "complementary"
    return "other"


def sign_real_size(cls_name):
    """Return realistic world size (metres) for German sign per StVO."""
    shape = class_to_shape(cls_name)
    if shape == "stop":
        return SIGN_SIZE_LARGE
    elif shape == "yield":
        return SIGN_SIZE_LARGE
    elif shape == "danger":
        return SIGN_SIZE_WARNING
    elif shape == "info":
        if "motorway" in cls_name or "hospital" in cls_name:
            return SIGN_SIZE_LARGE
        return SIGN_SIZE_DEFAULT
    elif shape == "prohibitory":
        if "height" in cls_name or "weight" in cls_name or "axel" in cls_name:
            return SIGN_SIZE_WARNING
        return SIGN_SIZE_DEFAULT
    elif shape == "mandatory":
        return SIGN_SIZE_DEFAULT
    return SIGN_SIZE_DEFAULT


def sign_stvo_number(cls_name):
    """Map class name to approximate StVO sign number for reference."""
    mapping = {
        "regulatory--no-entry": "267",
        "regulatory--stop": "206",
        "regulatory--yield": "205",
        "regulatory--maximum-speed-limit": "274",
        "regulatory--no-u-turn": "272",
        "regulatory--no-right-turn": "267",
        "regulatory--no-left-turn": "267",
        "regulatory--no-overtaking": "276",
        "regulatory--no-parking": "286",
        "regulatory--height-limit": "265",
        "regulatory--weight-limit": "263",
        "regulatory--go-straight": "209",
        "regulatory--turn-right": "209",
        "regulatory--turn-left": "209",
        "regulatory--keep-right": "222",
        "regulatory--keep-left": "222",
        "warning--crossroads": "102",
        "warning--curve-right": "103",
        "warning--curve-left": "103",
        "warning--roundabout": "215",
        "warning--roadworks": "123",
        "warning--children": "136",
        "warning--pedestrians-crossing": "133",
        "warning--slippery-road-surface": "114",
        "warning--traffic-signals": "131",
        "warning--railroad-crossing-with-barriers": "151",
        "warning--railroad-crossing-without-barriers": "150",
        "information--parking": "314",
        "information--hospital": "358",
        "information--motorway": "330",
        "information--gas-station": "361",
    }
    return mapping.get(cls_name, "???")


# ═══════════════════════════════════════════════════════════════════════════════
# STVO-COMPLIANT TEMPLATE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_template_stvo(cls_name, size=256):
    """
    Generate StVO-compliant German traffic sign template.
    Uses official colors, shapes, and proportions per Straßenverkehrs-Ordnung.
    """
    shape = class_to_shape(cls_name)
    label = class_to_label(cls_name)
    stvo_num = sign_stvo_number(cls_name)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 2 - 14

    def _txt(text, scale=1.0, thick=2, color=STVO_BLACK, font=cv2.FONT_HERSHEY_SIMPLEX):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text, (cx - tw//2, cy + th//2), font, scale, color, thick, cv2.LINE_AA)

    def _txt_at(text, x, y, scale=1.0, thick=2, color=STVO_BLACK, font=cv2.FONT_HERSHEY_SIMPLEX):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text, (x - tw//2, y + th//2), font, scale, color, thick, cv2.LINE_AA)

    if shape == "prohibitory":
        # StVO: Red circle, white background, red diagonal bar or black symbol
        cv2.circle(img, (cx, cy), r, STVO_RED, -1)
        cv2.circle(img, (cx, cy), r - 10, STVO_WHITE, -1)

        if "no-entry" in cls_name:
            # Sign 267: White horizontal bar
            bar_h = max(int(r * 0.18), 14)
            cv2.rectangle(img, (cx - r + 28, cy - bar_h//2), 
                         (cx + r - 28, cy + bar_h//2), STVO_WHITE, -1)
        elif "speed" in cls_name or "limit" in cls_name:
            # Sign 274: Speed limit in black
            cv2.circle(img, (cx, cy), r - 22, STVO_RED, 5)
            m = re.search(r'(\d+)', cls_name)
            speed = m.group(1) if m else "50"
            _txt(speed, 1.4, 4, STVO_BLACK)
            # Add "km/h" below
            _txt_at("km/h", cx, cy + int(r*0.45), 0.5, 1, STVO_BLACK)
        elif "height" in cls_name:
            # Sign 265: Height limit
            _txt("h", 1.3, 3, STVO_BLACK)
            m = re.search(r'(\d+\.?\d*)', cls_name)
            h_val = m.group(1) if m else "3.8"
            _txt_at(h_val + "m", cx, cy + int(r*0.45), 0.6, 2, STVO_BLACK)
        elif "weight" in cls_name or "axel" in cls_name:
            # Sign 263: Weight limit
            _txt("t", 1.3, 3, STVO_BLACK)
            m = re.search(r'(\d+\.?\d*)', cls_name)
            w_val = m.group(1) if m else "7.5"
            _txt_at(w_val + "t", cx, cy + int(r*0.45), 0.6, 2, STVO_BLACK)
        elif "no-u-turn" in cls_name:
            # Sign 272: U-turn arrow with red bar
            arr_r = r // 3
            # Draw U-turn arrow
            cv2.circle(img, (cx, cy - 5), arr_r, STVO_BLACK, 4)
            cv2.arrowedLine(img, (cx + arr_r, cy - 5), (cx + arr_r//2, cy - arr_r), 
                           STVO_BLACK, 4, tipLength=0.3)
            # Red diagonal bar
            cv2.line(img, (cx - r + 22, cy - r + 22), (cx + r - 22, cy + r - 22), STVO_RED, 10)
        elif "no-right-turn" in cls_name:
            # Right turn arrow with red bar
            pts = np.array([(cx, cy - r//2), (cx + r//2, cy), (cx, cy + r//2)], np.int32)
            cv2.fillPoly(img, [pts], STVO_BLACK)
            cv2.line(img, (cx - r + 22, cy - r + 22), (cx + r - 22, cy + r - 22), STVO_RED, 10)
        elif "no-left-turn" in cls_name:
            pts = np.array([(cx, cy - r//2), (cx - r//2, cy), (cx, cy + r//2)], np.int32)
            cv2.fillPoly(img, [pts], STVO_BLACK)
            cv2.line(img, (cx - r + 22, cy - r + 22), (cx + r - 22, cy + r - 22), STVO_RED, 10)
        elif "no-overtaking" in cls_name:
            # Two cars, one overtaking, with red bar
            _txt("NO", 0.9, 2, STVO_BLACK)
            _txt_at("PASSING", cx, cy + int(r*0.35), 0.5, 1, STVO_BLACK)
            cv2.line(img, (cx - r + 22, cy - r + 22), (cx + r - 22, cy + r - 22), STVO_RED, 10)
        elif "no-parking" in cls_name:
            # Sign 286: Blue circle with red X
            cv2.circle(img, (cx, cy), r - 15, (200, 100, 0), -1)
            cv2.line(img, (cx - r//2, cy - r//2), (cx + r//2, cy + r//2), STVO_RED, 8)
            cv2.line(img, (cx + r//2, cy - r//2), (cx - r//2, cy + r//2), STVO_RED, 8)
        else:
            # Generic prohibitory: red diagonal bar
            cv2.line(img, (cx - r + 25, cy - r + 25), (cx + r - 25, cy + r - 25), STVO_RED, 12)
            # Class label
            short_label = label.replace("REGULATORY ", "").replace("NO ", "")[:12]
            _txt(short_label, 0.6, 1, STVO_BLACK)

    elif shape == "yield":
        # StVO Sign 205: Inverted triangle, red border, white fill
        pts = np.array([[cx, cy + r - 8], 
                       [cx - r + 15, cy - int(r*0.55)], 
                       [cx + r - 15, cy - int(r*0.55)]], np.int32)
        cv2.fillPoly(img, [pts], STVO_WHITE)
        cv2.polylines(img, [pts], True, STVO_RED, 10)
        _txt("VORFAHRT", 0.55, 1, STVO_BLACK)
        _txt_at("GEWÄHREN", cx, cy + int(r*0.25), 0.45, 1, STVO_BLACK)

    elif shape == "stop":
        # StVO Sign 206: Octagon, red fill, white STOP
        angles = np.linspace(0, 2*np.pi, 9)[:-1] + np.pi/8
        pts = np.array([(cx + int((r-8)*np.cos(a)), cy + int((r-8)*np.sin(a))) 
                       for a in angles], np.int32)
        cv2.fillPoly(img, [pts], STVO_RED)
        cv2.polylines(img, [pts], True, STVO_WHITE, 4)
        _txt("STOP", 0.8, 2, STVO_WHITE)

    elif shape == "mandatory":
        # StVO: Blue circle, white arrow/symbol
        cv2.circle(img, (cx, cy), r, STVO_BLUE, -1)
        cv2.circle(img, (cx, cy), r, STVO_WHITE, 4)

        if "straight" in cls_name and "right" in cls_name:
            _txt("↑→", 0.9, 2, STVO_WHITE)
        elif "straight" in cls_name and "left" in cls_name:
            _txt("←↑", 0.9, 2, STVO_WHITE)
        elif "turn-left-or-right" in cls_name:
            _txt("←→", 0.9, 2, STVO_WHITE)
        elif "straight" in cls_name:
            _txt("↑", 1.3, 3, STVO_WHITE)
        elif "right" in cls_name:
            _txt("→", 1.3, 3, STVO_WHITE)
        elif "left" in cls_name:
            _txt("←", 1.3, 3, STVO_WHITE)
        elif "keep-right" in cls_name:
            # Arrow curving right
            _txt("↱", 1.3, 3, STVO_WHITE)
        elif "keep-left" in cls_name:
            _txt("↰", 1.3, 3, STVO_WHITE)
        elif "pass-on-either-side" in cls_name:
            _txt("↕", 1.3, 3, STVO_WHITE)
        elif "roundabout" in cls_name:
            _txt("↻", 1.3, 3, STVO_WHITE)
        else:
            _txt("↑", 1.3, 3, STVO_WHITE)

    elif shape == "danger":
        # StVO: Equilateral triangle, red border, white background, black symbol
        margin = size // 8
        pts = np.array([[cx, margin + 8], 
                       [margin + 8, size - margin - 8], 
                       [size - margin - 8, size - margin - 8]], np.int32)
        cv2.fillPoly(img, [pts], STVO_WHITE)
        cv2.polylines(img, [pts], True, STVO_RED, 10)

        if "roundabout" in cls_name:
            # Circular arrow
            cv2.circle(img, (cx, cy - 5), r//3, STVO_BLACK, 4)
            cv2.arrowedLine(img, (cx + r//3, cy - 5), (cx + r//4, cy - r//4), 
                           STVO_BLACK, 4, tipLength=0.4)
        elif "roadworks" in cls_name:
            _txt("A", 1.4, 3)
        elif "children" in cls_name:
            _txt("KINDER", 0.55, 2)
        elif "pedestrians" in cls_name or "crossing" in cls_name:
            _txt("FUSS", 0.6, 2)
            _txt_at("GÄNGER", cx, cy + int(r*0.25), 0.5, 1)
        elif "slippery" in cls_name:
            _txt("SCHLEUDER", 0.45, 1)
            _txt_at("GEFAHR", cx, cy + int(r*0.25), 0.5, 1)
        elif "curve" in cls_name:
            if "right" in cls_name:
                _txt("↱", 1.4, 3)
            elif "left" in cls_name:
                _txt("↰", 1.4, 3)
            else:
                _txt("KURVE", 0.55, 2)
        elif "bump" in cls_name or "uneven" in cls_name:
            _txt("UNEBEN", 0.5, 1)
        elif "falling" in cls_name:
            _txt("STEINSCHLAG", 0.4, 1)
        elif "railroad" in cls_name:
            _txt("BAHNÜBERGANG", 0.38, 1)
        elif "traffic-signals" in cls_name:
            _txt("AMPEL", 0.6, 2)
        elif "road-narrows" in cls_name:
            _txt("ENGSTELLE", 0.45, 1)
        elif "wild-animals" in cls_name:
            _txt("WILD", 0.6, 2)
        elif "domestic-animals" in cls_name:
            _txt("VIEH", 0.6, 2)
        elif "double-curve" in cls_name:
            _txt("S", 1.3, 3)
        elif "crossroads" in cls_name:
            _txt("X", 1.3, 3)
        elif "junction" in cls_name:
            _txt("Y", 1.3, 3)
        elif "traffic-merges" in cls_name:
            _txt("EINMÜNDUNG", 0.4, 1)
        elif "road-slope" in cls_name:
            _txt("GEFÄLLE", 0.5, 1)
        elif "road-dip" in cls_name:
            _txt("SENKE", 0.55, 2)
        elif "other-danger" in cls_name:
            # Exclamation mark
            cv2.rectangle(img, (cx - 7, cy - 22), (cx + 7, cy + 2), STVO_BLACK, -1)
            cv2.circle(img, (cx, cy + 22), 9, STVO_BLACK, -1)
        else:
            _txt("!", 1.4, 3)

    elif shape == "info":
        # StVO: Rectangle, blue background, white symbol
        cv2.rectangle(img, (12, 12), (size - 12, size - 12), (180, 80, 0), -1)
        cv2.rectangle(img, (12, 12), (size - 12, size - 12), STVO_WHITE, 5)

        if "parking" in cls_name:
            _txt("P", 1.5, 3, STVO_WHITE)
            if "no-parking" not in cls_name:
                _txt_at("PARKEN", cx, cy + int(r*0.45), 0.45, 1, STVO_WHITE)
        elif "hospital" in cls_name:
            _txt("H", 1.5, 3, STVO_WHITE)
            _txt_at("KRANKENHAUS", cx, cy + int(r*0.45), 0.4, 1, STVO_WHITE)
        elif "gas" in cls_name:
            _txt("TANKEN", 0.6, 2, STVO_WHITE)
        elif "motorway" in cls_name:
            _txt("A", 1.5, 3, STVO_WHITE)
            _txt_at("AUTOBAHN", cx, cy + int(r*0.45), 0.45, 1, STVO_WHITE)
        elif "disabled" in cls_name:
            _txt("♿", 1.2, 2, STVO_WHITE)
        elif "tram" in cls_name:
            _txt("T", 1.5, 3, STVO_WHITE)
            _txt_at("STRASSENBAHN", cx, cy + int(r*0.45), 0.35, 1, STVO_WHITE)
        else:
            _txt("i", 1.5, 3, STVO_WHITE)

    else:  # complementary / other
        cv2.rectangle(img, (12, 12), (size - 12, size - 12), (240, 240, 240), -1)
        cv2.rectangle(img, (12, 12), (size - 12, size - 12), STVO_BLACK, 4)
        if "chevron" in cls_name:
            _txt(">>", 0.9, 2) if "right" in cls_name else _txt("<<", 0.9, 2)
        elif "distance" in cls_name:
            _txt("m", 1.3, 3)
        else:
            _txt("...", 0.8, 2)

    # Add StVO number in corner for reference
    _txt_at("StVO " + stvo_num, size - 45, size - 15, 0.3, 1, (100, 100, 100))

    return img


def make_light_texture_stvo(w=128, h=320, verified=True):
    """StVO-compliant traffic light texture."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    border_color = (60, 60, 60) if verified else (0, 100, 255)
    border_thick = 4 if verified else 8
    cv2.rectangle(img, (8, 8), (w-8, h-8), border_color, border_thick)

    rd, cx = w//3, w//2
    # StVO light order: red (top), yellow/amber (middle), green (bottom)
    for cy_l, col, label in zip(
        [h//6, h//2, 5*h//6], 
        [(0, 0, 220), (0, 180, 255), (0, 220, 0)],
        ["ROT", "GELB", "GRÜN"]
    ):
        cv2.circle(img, (cx, cy_l), rd+3, (80, 80, 80), -1)
        cv2.circle(img, (cx, cy_l), rd, col, -1)
        # Highlight
        cv2.circle(img, (cx-rd//3, cy_l-rd//3), rd//4, (255, 255, 255), -1)

    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "AMPEL" if verified else "AMPEL ?"
    (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
    cv2.putText(img, text, (cx - tw//2, h - 12), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def make_label_tex(text, w=320, h=60):
    """High-contrast floating label with dark background plate."""
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w-1, h-1), (180, 180, 180), 2)
    cv2.rectangle(img, (3, 3), (w-4, h-4), (45, 45, 45), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for sc in [0.70, 0.60, 0.50, 0.45, 0.40]:
        (tw, th), _ = cv2.getTextSize(text, font, sc, 2)
        if tw <= w - 20:
            break

    x = max(10, (w - tw) // 2)
    y = max(th + 8, (h + th) // 2)

    # Black outline
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
        cv2.putText(img, text, (x+dx, y+dy), font, sc, (0,0,0), 2, cv2.LINE_AA)
    # White text
    cv2.putText(img, text, (x, y), font, sc, (255,255,255), 2, cv2.LINE_AA)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# MATH + CALIBRATION + LAZ
# ═══════════════════════════════════════════════════════════════════════════════
def _Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=np.float64)
def _Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float64)
def _Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)
def hrp_to_R_w2s(h,r,p): return (_Rz(np.radians(h))@_Ry(np.radians(r))@_Rx(np.radians(p))).T

def project(p_world, cam_xyz, R_w2s, R0, t_vec, K):
    ps = R_w2s @ (p_world - cam_xyz)
    pc = R0.T @ (ps - t_vec)
    d = pc[2]
    if d < MIN_DEPTH or d > MAX_DEPTH:
        return None
    return K[0,0]*pc[0]/d + K[0,2], K[1,1]*pc[1]/d + K[1,2], d, pc

def load_calibration():
    with open(os.path.join(PHASE0_DIR, "phase0_pinhole_K.json")) as f:
        raw = json.load(f)
    pK = {}
    for sn, v in raw.items():
        pK[sn] = np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]], dtype=np.float64)
    tree = ET.parse(INTRINSIC_XML)
    ext, names = {}, {}
    for c in tree.getroot().findall("Camera"):
        sn = c.get("serialno")
        names[sn] = c.get("name")
        R0 = _Rx(float(c.find("R0_roll").text)) @ _Ry(float(c.find("R0_pitch").text)) @ _Rz(float(c.find("R0_yaw").text))
        t = np.array([float(c.find("R0_tx").text), float(c.find("R0_ty").text), float(c.find("R0_tz").text)], dtype=np.float64)
        ext[sn] = {"R0": R0, "t_vec": t}
    return pK, ext, names

def load_laz(p):
    las = laspy.read(p)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    intensity = np.array(las.intensity, dtype=np.float32)

    # Read classification if present
    if hasattr(las, 'classification'):
        classification = np.array(las.classification, dtype=np.uint8)
        road_mask = classification == 1
        vert_mask = classification == 2
        print(f"    {len(xyz):,} pts (road={road_mask.sum():,}, vertical={vert_mask.sum():,})")
    else:
        road_mask = None
        vert_mask = None
        print(f"    {len(xyz):,} pts (no classification)")

    return xyz, intensity, road_mask, vert_mask


def ransac_road(xyz):
    """Fallback road segmentation when classification is missing."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inl = pcd.segment_plane(0.15, 3, 2000)
    m = np.zeros(len(xyz), dtype=bool)
    m[inl] = True
    print(f"    Road (RANSAC): {m.sum():,}/{len(xyz):,}")
    return m

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

def color_road(intensity, mask):
    c = np.tile(BASE_COLOR, (len(intensity), 1))
    if mask.sum() == 0:
        return c
    ri = intensity[mask]
    if len(ri) == 0:
        return c
    lo, hi = np.percentile(ri, 2), np.percentile(ri, 98)
    if hi - lo < 1:
        hi = lo + 1
    g = (ROAD_INT_MIN + np.clip((ri - lo)/(hi - lo), 0, 1)*(ROAD_INT_MAX - ROAD_INT_MIN)).astype(np.uint8)
    c[mask] = np.stack([g, g, g], axis=1)
    return c

def color_vertical_unclassified(intensity, mask):
    """Dark blue tint for unclassified vertical points."""
    c = np.zeros((len(intensity), 3), dtype=np.uint8)
    c[mask, 0] = 40
    c[mask, 1] = 40
    c[mask, 2] = 100
    return c


def filter_dets(raw):
    kept = []
    for d in raw:
        tag = f"{d['type']:6s} h={d['height_above_ground']:.1f}m w={d['max_width']:.2f}m pts={d['n_points']}"
        is_camera_only = (d.get("source_method") == "camera_only_unverified" or 
                          d.get("source") == "camera_only" or
                          d.get("source_method") == "camera_only")
        if d["n_points"] < MIN_CLUSTER_PTS and not is_camera_only:
            print(f"    DROPPED: {tag}  → n_points={d['n_points']} < {MIN_CLUSTER_PTS}")
            continue
        if (d["max_width"] < MIN_WIDTH or d["max_width"] > MAX_WIDTH) and not is_camera_only and d["n_points"] > 0:
            print(f"    DROPPED: {tag}  → width={d['max_width']:.2f} outside [{MIN_WIDTH},{MAX_WIDTH}]")
            continue
        if d["height_above_ground"] < MIN_HEIGHT:
            print(f"    DROPPED: {tag}  → height={d['height_above_ground']:.2f}m < {MIN_HEIGHT}")
            continue
        if is_camera_only:
            print(f"    KEPT (camera-only, unverified): {tag}")
        dta = d.get("dist_to_approx", 0)
        if dta > MAX_DIST_APPROX and dta > 0:
            print(f"    DROPPED: {tag}  → dist_to_approx={dta:.1f}m > {MAX_DIST_APPROX}")
            continue
        if d["type"] == "sign" and d.get("confidence", 0) < MIN_SIGN_CONFIDENCE:
            cls_hint = d.get("specific_class", "?")
            print(f"    DROPPED: {tag}  → sign conf={d['confidence']:.2f} < {MIN_SIGN_CONFIDENCE} ({cls_hint})")
            continue
        kept.append(d)

    # Deduplicate
    deduped = []
    used = set()
    for i, d in enumerate(kept):
        if i in used:
            continue
        best = d
        for j in range(i+1, len(kept)):
            if j in used:
                continue
            dist = np.linalg.norm(kept[j]["xyz"] - d["xyz"])
            if dist < DEDUP_DIST:
                loser = kept[j] if kept[j]["confidence"] <= best["confidence"] else d
                print(f"    DEDUPED: {loser['type']:6s} ({loser.get('specific_class','?')}) "
                      f"within {dist:.1f}m of {best.get('specific_class', best['type'])}")
                used.add(j)
                if kept[j]["confidence"] > best["confidence"]:
                    best = kept[j]
        deduped.append(best)
    print(f"    Raw={len(raw)} → Filtered={len(deduped)}")
    return deduped


def place_billboard(centre, tex, face_dir_rad, w_m, h_m):
    """Place billboard facing camera direction."""
    right = np.array([np.cos(face_dir_rad + np.pi/2), np.sin(face_dir_rad + np.pi/2), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    th, tw = tex.shape[:2]
    asp = tw / max(th, 1)
    if asp >= 1:
        nc = BILLBOARD_RES
        nr = max(int(BILLBOARD_RES / asp), 8)
    else:
        nr = BILLBOARD_RES
        nc = max(int(BILLBOARD_RES * asp), 8)

    resized = cv2.resize(tex, (nc, nr), interpolation=cv2.INTER_AREA)
    pxyz, prgb = [], []

    for row in range(nr):
        for col in range(nc):
            un = (col / (nc - 1)) - 0.5
            vn = 0.5 - (row / (nr - 1))
            pt = centre + right * (un * w_m) + up * (vn * h_m)
            bgr = resized[row, col]
            rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)
            if rgb.max() < 10:
                continue
            pxyz.append(pt)
            prgb.append(rgb)

    if not pxyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.array(pxyz), np.array(prgb, dtype=np.uint8)


def load_crop_image(crop_path):
    """Load crop image if it exists."""
    if not crop_path or not os.path.isfile(crop_path):
        return None
    img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def make_sprite(det, cls76_name, cam_entry, real_top_z=None, use_crop=False, crop_img=None):
    """
    Generate sign/light billboard + floating label + optional pole.

    Priority:
      1. If crop available AND close camera AND high confidence: use ACTUAL crop
      2. Else: use StVO-compliant template

    real_top_z: max Z of the actual LiDAR points tinted as this detection's
    pole/cluster (see caller). When present, the billboard is clamped to sit
    AT that height, never above it — otherwise the icon renders floating
    over empty space while the real, visibly-tinted pole ends lower down.
    """
    dtype = det["type"]
    ground_z = det.get("ground_z", det["xyz"][2] - det.get("height_above_ground", 2.0))
    height_ag = det.get("height_above_ground", 2.0)

    # Realistic world sizes (computed first — needed to clamp centre height)
    if dtype == "sign":
        sz = sign_real_size(cls76_name)
        w_m, h_m = sz, sz
    elif dtype == "light":
        w_m, h_m = 0.40, 1.10
    else:
        w_m, h_m = 0.20, 0.20

    grounded = real_top_z is not None

    # Determine billboard center, clamped to the real detected extent
    if dtype == "light":
        z_head = ground_z + height_ag
        if grounded:
            z_head = min(z_head, real_top_z + 0.05)   # never above real data
        z_head = max(z_head, ground_z + h_m / 2 + 0.05)
        centre = np.array([det["xyz"][0], det["xyz"][1], z_head])
    else:
        centre = det["xyz"].copy()
        if grounded:
            centre[2] = min(centre[2], real_top_z - h_m * 0.1)
        centre[2] = max(centre[2], ground_z + h_m / 2 + 0.05)

    # Billboard orientation ("tilt"): face the direction a driver actually
    # approaches from. Prefer bearing_rad (the FULL bearing phase3 used for
    # this detection's own position — camera heading combined with where
    # in the frame the object actually sits) over a heading-only
    # approximation, which implicitly assumes the object is dead-center in
    # frame. A sign faces back the way it was photographed from, so we
    # flip the camera->sign bearing by pi. Falls back to the older
    # heading-only approximation for geojson written before this field
    # existed, and to nearest-camera-position as a last resort.
    det_bearing = det.get("bearing_rad")
    det_heading = det.get("heading")
    if det_bearing is not None:
        heading_rad = det_bearing + np.pi
    elif det_heading is not None:
        heading_rad = -math.radians(det_heading) + np.pi
    elif cam_entry is not None:
        cam_xy = np.array(cam_entry["Xyz"][:2])
        bearing = np.arctan2(centre[1] - cam_xy[1], centre[0] - cam_xy[0])
        heading_rad = bearing + np.pi
    else:
        heading_rad = 0.0

    all_xyz, all_rgb = [], []

    # ── Main billboard: crop or template ──
    if dtype == "sign":
        if use_crop and crop_img is not None:
            # Use actual crop image
            tex = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            status = "CROP"
        else:
            # Use StVO template
            tex = make_template_stvo(cls76_name)
            status = f"StVO-{sign_stvo_number(cls76_name)}"

        sx, sr = place_billboard(centre, tex, heading_rad, w_m, h_m)
        if len(sx) > 0:
            all_xyz.append(sx); all_rgb.append(sr)

    elif dtype == "light":
        is_verified = det.get("source_method", "lidar_verified") == "lidar_verified"
        tex = make_light_texture_stvo(verified=is_verified)
        sx, sr = place_billboard(centre, tex, heading_rad, w_m, h_m)
        if len(sx) > 0:
            all_xyz.append(sx); all_rgb.append(sr)
        status = "LIGHT" if is_verified else "LIGHT?"

    # ── Floating label above billboard ──
    if dtype == "light":
        is_verified = det.get("source_method", "lidar_verified") == "lidar_verified"
        label_text = "TRAFFIC LIGHT" if is_verified else "TRAFFIC LIGHT (UNVERIFIED)"
    else:
        label_text = class_to_label(cls76_name)
        # Add StVO number to label
        label_text += f" [StVO {sign_stvo_number(cls76_name)}]"

    lt = make_label_tex(label_text, w=LABEL_TEX_W, h=LABEL_TEX_H)
    lc = centre.copy()
    lc[2] += h_m / 2 + LABEL_OFFSET_Z
    lx, lr = place_billboard(lc, lt, heading_rad, LABEL_W_M, LABEL_H_M)
    if len(lx) > 0:
        all_xyz.append(lx); all_rgb.append(lr)

    # ── Vertical pole stem — always connects ground to billboard, drawn as
    # a dense filled-ring cylinder (not a thin cross) so it reads as solid
    # and the sprite never appears disconnected from the ground. ──
    if dtype in ("sign", "light"):
        pole_top = centre[2] - h_m / 2
        if pole_top > ground_z + 0.05:
            thickness = POLE_THICKNESS_M / 2
            n_pole = max(20, int((pole_top - ground_z) / 0.05))
            ring = [(thickness*np.cos(a), thickness*np.sin(a))
                    for a in np.linspace(0, 2*np.pi, 8, endpoint=False)] + [(0.0, 0.0)]
            pole_pts, pole_rgb = [], []
            for z in np.linspace(ground_z, pole_top, n_pole):
                for dx, dy in ring:
                    pole_pts.append([centre[0]+dx, centre[1]+dy, z])
                    pole_rgb.append([150, 150, 155])
            all_xyz.append(np.array(pole_pts))
            all_rgb.append(np.array(pole_rgb, dtype=np.uint8))

    if not all_xyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), "NONE"
    return np.vstack(all_xyz), np.vstack(all_rgb), status


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase4c_stvo_sprites.py <laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    bn = os.path.basename(laz_path)
    
    # Extract part name: handle "merged_part15.laz", "phase3_part15_objects.laz", etc.
    m = re.search(r'(part\\d+)', bn)
    if m:
        pn = m.group(1)
    else:
        pn = bn.replace('.laz', '').replace('phase3_', '').replace('merged_', '').replace('_objects', '')
    
    # For point cloud: use original merged file if available (has classification)
    original_laz = laz_path
    candidate = os.path.join(PHASE2_DIR, f"merged_{pn}.laz")
    if os.path.isfile(candidate):
        print(f"    Using original merged file for point cloud: {candidate}")
        original_laz = candidate
    else:
        # Try other patterns
        for pattern in [f"{pn}.laz", f"merged_{pn}.laz", f"phase2_{pn}.laz"]:
            candidate2 = os.path.join(PHASE2_DIR, pattern)
            if os.path.isfile(candidate2):
                print(f"    Using original file for point cloud: {candidate2}")
                original_laz = candidate2
                break
    
    print(f"\n{'='*70}")
    print(f"  PHASE 4c — StVO Sprites with Actual Crops + Templates")
    print(f"  Tile: {bn}  |  Part: {pn}")
    print(f"{'='*70}")

    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    print("\n[1/6] Loading detections from Phase 3 GeoJSON...")
    raw_dets = []
    for kind in ["signs_3d", "traffic_lights_3d", "poles_3d"]:
        # Try multiple filename patterns
        paths_to_try = [
            os.path.join(PHASE3_DIR, f"{kind}_{pn}_cleaned.geojson"),
            os.path.join(PHASE3_DIR, f"{kind}_{pn}.geojson"),
            os.path.join(PHASE3_DIR, f"{kind}_{pn}_objects.geojson"),
        ]
        if kind == "signs_3d":
            paths_to_try.append(os.path.join(PHASE3_DIR, f"poles_with_attachments_{pn}.geojson"))

        path = None
        for p in paths_to_try:
            if os.path.isfile(p):
                path = p
                break

        if path is None:
            print(f"    WARNING: No {kind} GeoJSON found for {pn}")
            continue

        with open(path) as f:
            gj = json.load(f)

        src = "CLEANED" if "_cleaned" in path else ("V6" if "poles_with" in path else "ORIGINAL")
        print(f"    {len(gj.get('features', []))} {kind} [{src}] from {os.path.basename(path)}")

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
                "approx_utm": p.get("approx_utm"),
                "specific_class": p.get("specific_class", "unknown"),
                "specific_label": p.get("specific_label", "UNKNOWN"),
                "classifier_conf": p.get("classifier_conf", 0.0),
                "crop_path": p.get("crop_path", ""),
                "heading": p.get("heading", 0.0),
                "source_method": p.get("source_method", "lidar_verified"),
                "likely_back_face": p.get("likely_back_face", False),
                "pole_id": p.get("pole_id", -1),
            })

        if "_cleaned" in path:
            src = "CLEANED"
        elif "poles_with" in path:
            src = "V6"
        else:
            src = "ORIGINAL"
        print(f"    {len(gj.get('features', []))} {kind} [{src}]")

    print(f"\n    Total raw detections loaded: {len(raw_dets)}")

    # Camera-only (unverified) detections carry height_above_ground=0.0 from
    # Phase 3 regardless of their actual elevation, which made every one of
    # them fail the MIN_HEIGHT filter below (silently dropped, not floating).
    # Recompute the real value from the triangulated Z vs. ground so they're
    # judged on their actual height instead.
    for d in raw_dets:
        if d["n_points"] == 0 and d.get("source_method") == "camera_only_unverified":
            d["height_above_ground"] = float(d["xyz"][2] - d["ground_z"])

    dets = filter_dets(raw_dets)

    print(f"\n[2/6] Loading point cloud...")
    xyz, intensity, road_mask, vert_mask = load_laz(original_laz)

    print(f"\n[3/6] Coloring point cloud...")
    if road_mask is not None and vert_mask is not None:
        # Use classification from merged .laz
        colors = np.zeros((len(xyz), 3), dtype=np.uint8)
        colors[road_mask] = color_road(intensity, road_mask)[road_mask]
        colors[vert_mask] = color_vertical_unclassified(intensity, vert_mask)[vert_mask]
        print(f"    Using classification field: road={road_mask.sum():,}, vertical={vert_mask.sum():,}")
    else:
        # Fallback: RANSAC road segmentation
        print("    No classification field. Falling back to RANSAC...")
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        _, inl = pcd.segment_plane(0.15, 3, 2000)
        road_mask = np.zeros(len(xyz), dtype=bool)
        road_mask[inl] = True
        colors = color_road(intensity, road_mask)
        vert_mask = ~road_mask

    print(f"\n[4/6] Loading camera entries...")
    with open(CAMEXTR_PATH) as f:
        ce = json.load(f)

    all_e = ce.get("Profiler_0", [])
    xn, xx = xyz[:,0].min()-BBOX_MARGIN, xyz[:,0].max()+BBOX_MARGIN
    yn, yx = xyz[:,1].min()-BBOX_MARGIN, xyz[:,1].max()+BBOX_MARGIN

    cam_entries = [
        e for e in all_e
        if xn <= e["Xyz"][0] <= xx and yn <= e["Xyz"][1] <= yx
    ]

    print(f"    {len(cam_entries)} images")

    cam_positions = (
        np.array([e["Xyz"] for e in cam_entries], dtype=np.float64)
        if cam_entries else np.empty((0, 3))
    )

    print(f"\n[5/6] Processing {len(dets)} detections...\n")

    all_sx, all_sr, all_si = [], [], []
    n_spr, n_cl = 0, 0
    n_used_crop, n_used_template, n_fallback = 0, 0, 0
    report = []

    for i, det in enumerate(dets):
        dtype = det["type"]
        mw = det["max_width"]
        hag = det["height_above_ground"]
        cls_id = det.get("class_id", 0)

        # Tint LiDAR cluster — use the true vertical-infra classification
        # (vert_mask) rather than just "non-road" for a tighter match.
        r = np.clip(mw * CLUSTER_R_SCALE, CLUSTER_R_MIN, CLUSTER_R_MAX)
        dm = np.linalg.norm(xyz - det["xyz"], axis=1)
        cm = (dm < r) & vert_mask
        nc = cm.sum()
        # Real detected height at this location — the same points that get
        # tinted below. Sprite placement is clamped to this so the icon
        # never floats above the pole the viewer can actually see.
        real_top_z = float(xyz[cm, 2].max()) if nc > 0 else None

        # Choose tint color based on type and verification
        if dtype == "sign":
            if det.get("likely_back_face", False):
                tint = TINT_BACKFACE
            elif det.get("source_method", "lidar_verified") == "lidar_verified":
                tint = TINT_SIGN
            else:
                tint = TINT_UNVERIFIED
        elif dtype == "light":
            if det.get("source_method", "lidar_verified") == "lidar_verified":
                tint = TINT_LIGHT
            else:
                tint = TINT_UNVERIFIED
        elif dtype == "pole":
            tint = TINT_POLE
        else:
            tint = TINT_UNVERIFIED

        if nc > 0:
            colors[cm] = tint
            n_cl += nc

        if dtype == "pole":
            report.append(f"det{i+1:02d}  pole        TINT       pts={nc}")
            continue

        specific_class = det.get("specific_class", "unknown")
        classifier_conf = det.get("classifier_conf", 0.0)
        crop_path = det.get("crop_path", "")

        # Decide whether to use crop or template
        use_crop = False
        crop_img = None

        if dtype == "sign" and crop_path and classifier_conf >= MIN_CROP_CONFIDENCE:
            # Check if any camera was close enough
            if len(cam_positions) > 0:
                d2c = np.linalg.norm(cam_positions[:, :2] - det["xyz"][:2], axis=1)
                min_cam_dist = d2c.min()
                if min_cam_dist < MAX_CROP_DISTANCE_M:
                    crop_img = load_crop_image(crop_path)
                    if crop_img is not None:
                        use_crop = True
                        n_used_crop += 1

        if dtype == "sign":
            if specific_class not in ("unknown", None, ""):
                cls76 = specific_class
                label = class_to_label(cls76)
                if use_crop:
                    status = f"CROP({label})"
                else:
                    status = f"StVO-{sign_stvo_number(cls76)}({label})"
                    n_used_template += 1
            else:
                n_fallback += 1
                fallback_map = {
                    0: "regulatory--no-entry",
                    1: "warning--other-danger",
                    2: "regulatory--go-straight",
                    3: "complementary--chevron-right"
                }
                cls76 = fallback_map.get(cls_id, "warning--other-danger")
                label = class_to_label(cls76)
                status = f"FALLBACK({label})"

        elif dtype == "light":
            cls76 = "traffic-light"
            is_verified = det.get("source_method", "lidar_verified") == "lidar_verified"
            label = "TRAFFIC LIGHT" if is_verified else "TRAFFIC LIGHT (UNVERIFIED)"
            status = "LIGHT" if is_verified else "LIGHT?"

        else:
            cls76 = "unknown"
            label = "UNKNOWN"
            status = "UNKNOWN"

        print(
            f"  [{i+1}/{len(dets)}] {dtype:5s}  cls={cls76:45s} "
            f"conf={det['confidence']:.2f}  h={hag:.1f}m  w={mw:.2f}m  [{status}]"
        )

        # Find nearest camera for billboard orientation
        if len(cam_positions) > 0:
            d2c = np.linalg.norm(cam_positions[:, :2] - det["xyz"][:2], axis=1)
            nearest_cam = cam_entries[int(np.argmin(d2c))]
        else:
            nearest_cam = None

        sx, sr, tex_status = make_sprite(det, cls76, nearest_cam, real_top_z=real_top_z, use_crop=use_crop, crop_img=crop_img)

        if len(sx) > 0:
            all_sx.append(sx)
            all_sr.append(sr)
            all_si.append(np.full(len(sx), 220, dtype=np.float32))
            n_spr += len(sx)

        ground_tag = "GROUNDED" if real_top_z is not None else "EST-HEIGHT"
        print(f"    → sprite={len(sx):,}pts  label='{label}'  tex={tex_status}  [{ground_tag}]")
        report.append(f"det{i+1:02d}  {dtype:6s}  {label:30s}  {tex_status}  [{ground_tag}]")

    print(f"\n[6/6] Saving...")

    if all_sx:
        fxyz = np.vstack([xyz, np.vstack(all_sx)])
        fcol = np.vstack([colors, np.vstack(all_sr)])
        fint = np.concatenate([intensity, np.concatenate(all_si)])
    else:
        fxyz, fcol, fint = xyz, colors, intensity

    out = os.path.join(OUTPUT_DIR, f"semantic_{pn}.laz")
    save_laz(fxyz, fcol, fint, out)

    print(f"  → {out} ({len(fxyz):,} pts)")

    rpt_path = os.path.join(CROP_DIR, f"detection_report_{pn}.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join([
            "Phase 4c — StVO Sprites with Actual Crops + Templates",
            "=" * 60,
            f"Tile: {bn}",
            f"Total points: {len(xyz):,}",
            f"Road points: {road_mask.sum():,}" if road_mask is not None else "Road: RANSAC",
            f"Vertical points: {vert_mask.sum():,}" if vert_mask is not None else "",
            f"Detections: {len(dets)}",
            f"Sprites: {n_spr:,}",
            f"Used actual crops: {n_used_crop}",
            f"Used StVO templates: {n_used_template}",
            f"Fallback templates: {n_fallback}",
            "",
            *report
        ]) + "\n")

    print(f"  Report → {rpt_path}")
    print(f"  Done in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
