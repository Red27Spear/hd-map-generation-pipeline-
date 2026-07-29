#!/usr/bin/env python3
"""
Camera pre-processing — undistortion + radiometric correction
================================================================
Fixes three compounding problems in the raw fisheye camera frames:
  1. Barrel distortion  -> Kannala-Brandt undistortion to pinhole
  2. Underexposure       -> CLAHE on the luminance channel
  3. Olive / green cast  -> gray-world AWB on clean pixels + olive desaturation

Usage:
    python undistort_and_correct.py --laz ../data/part15/raw/*.laz \
        --image-dir ../data/part15/images \
        --camextr ../config/CamExtr.json \
        --intrinsics ../config/9020C_0140_toScanner_final.xml \
        --output-dir ./output

Inputs:
    --laz         a LAZ tile (only its header is read, for the tile's UTM bounding box)
    --image-dir   directory of raw camera frames named like the "Image" field in CamExtr.json
    --camextr     CamExtr.json — per-frame pose + serial number records
    --intrinsics  toScanner XML — per-camera fisheye intrinsics (fx, fy, cx, cy, k1..k4)

Outputs:
    <output-dir>/<original_filename>.jpg   corrected images
    <output-dir>/pinhole_K.json            new pinhole K per camera (consumed by later stages)
    <output-dir>/report.txt                processing summary

Known limitation: the AWB + olive-desaturation pass can overcorrect bright sky regions
into a magenta/pink cast on some frames. Worth revisiting the olive hue band and the
gray-world assumption before relying on this for anything beyond a visual demo.
"""

import argparse
import os
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
import cv2

BBOX_MARGIN_M = 20.0         # metres to expand the LAZ bounding box for image search
UNDIST_BALANCE = 0.3         # 0.0 = crop hard (no black), 1.0 = keep all (black borders)

# Radiometric correction parameters
AWB_BLOWN_THRESH = 240       # pixels above this in any channel = blown -> exclude from AWB
AWB_BLACK_THRESH = 15        # pixels below this in all channels = black -> exclude from AWB
OLIVE_HUE_LO = 25            # HSV hue range for olive/green cast (degrees, OpenCV 0-179)
OLIVE_HUE_HI = 95
OLIVE_SAT_THRESH = 30        # only desaturate if saturation > this
OLIVE_DESAT_FACTOR = 0.6     # multiply saturation by this in the olive band (1.0 = no change)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def get_laz_bbox(laz_path):
    """Read the UTM bounding box from a LAZ file header without loading all points."""
    import laspy
    with laspy.open(laz_path) as f:
        hdr = f.header
        mins = hdr.mins  # [x_min, y_min, z_min] -- UTM Easting, Northing, Z
        maxs = hdr.maxs
    bbox = {
        "e_min": mins[0] - BBOX_MARGIN_M,
        "e_max": maxs[0] + BBOX_MARGIN_M,
        "n_min": mins[1] - BBOX_MARGIN_M,
        "n_max": maxs[1] + BBOX_MARGIN_M,
    }
    print(f"  LAZ bounding box (with {BBOX_MARGIN_M}m margin):")
    print(f"    Easting  : {bbox['e_min']:.1f} - {bbox['e_max']:.1f}")
    print(f"    Northing : {bbox['n_min']:.1f} - {bbox['n_max']:.1f}")
    return bbox


def load_and_filter_camextr(camextr_path, bbox):
    """Return the CamExtr entries whose UTM position falls inside bbox."""
    with open(camextr_path, "r") as f:
        data = json.load(f)

    entries = data.get("Profiler_0", [])
    if not entries:
        if isinstance(data, list):
            entries = data
        else:
            raise ValueError(f"Cannot find image entries in {camextr_path}")

    filtered = []
    for entry in entries:
        xyz = entry["Xyz"]  # [Easting, Northing, Z]
        e, n = xyz[0], xyz[1]
        if bbox["e_min"] <= e <= bbox["e_max"] and bbox["n_min"] <= n <= bbox["n_max"]:
            filtered.append(entry)

    print(f"  CamExtr: {len(entries)} total entries, {len(filtered)} within tile bbox")
    return filtered


def load_intrinsics(intrinsics_path):
    """Parse the calibration XML. Returns a dict keyed by camera serial number."""
    tree = ET.parse(intrinsics_path)
    root = tree.getroot()
    cams = {}
    for cam_el in root.findall("Camera"):
        sn = cam_el.get("serialno")
        name = cam_el.get("name")

        fx = float(cam_el.findtext("fx"))
        fy = float(cam_el.findtext("fy"))
        cx = float(cam_el.findtext("cx"))
        cy = float(cam_el.findtext("cy"))

        k1 = float(cam_el.findtext("k1"))
        k2 = float(cam_el.findtext("k2"))
        k3 = float(cam_el.findtext("k3"))
        k4 = float(cam_el.findtext("k4"))

        img_w = int(cam_el.findtext("image_w"))
        img_h = int(cam_el.findtext("image_h"))

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(4, 1)

        cams[sn] = {"name": name, "K": K, "D": D, "img_w": img_w, "img_h": img_h}
        print(f"    Camera '{name}' (sn={sn}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    return cams


def compute_pinhole_Ks(intrinsics):
    """For each camera, compute the undistorted pinhole K and the remap tables."""
    pinhole_data = {}
    for sn, cam in intrinsics.items():
        K, D = cam["K"], cam["D"]
        img_size = (cam["img_w"], cam["img_h"])

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, img_size, R=np.eye(3), balance=UNDIST_BALANCE,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R=np.eye(3), P=new_K, size=img_size, m1type=cv2.CV_16SC2,
        )
        pinhole_data[sn] = {"new_K": new_K, "map1": map1, "map2": map2, "name": cam["name"]}
        print(f"    Pinhole K for '{cam['name']}': fx={new_K[0,0]:.1f} fy={new_K[1,1]:.1f} "
              f"cx={new_K[0,2]:.1f} cy={new_K[1,2]:.1f}")

    return pinhole_data


def fix_radiometrics(img):
    """Apply AWB + olive desaturation + CLAHE. Input/output: BGR uint8."""
    img_f = img.astype(np.float32)

    max_ch = img.max(axis=2)
    min_ch = img.min(axis=2)
    clean_mask = (max_ch < AWB_BLOWN_THRESH) & (min_ch > AWB_BLACK_THRESH)

    if clean_mask.sum() > 1000:
        for c in range(3):
            ch = img_f[:, :, c]
            mean_c = ch[clean_mask].mean()
            if mean_c > 1.0:
                overall_mean = np.mean([img_f[:, :, i][clean_mask].mean() for i in range(3)])
                img_f[:, :, c] = ch * (overall_mean / mean_c)

    img_f = np.clip(img_f, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    olive_mask = (h >= OLIVE_HUE_LO) & (h <= OLIVE_HUE_HI) & (s > OLIVE_SAT_THRESH)
    s[olive_mask] *= OLIVE_DESAT_FACTOR
    s = np.clip(s, 0, 255)
    hsv[:, :, 1] = s
    img_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_image(img_path, serial_nr, pinhole_data):
    """Load a raw fisheye frame, undistort, fix colours. Returns a corrected BGR image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"    WARNING: could not read {img_path}")
        return None

    cam = pinhole_data.get(serial_nr)
    if cam is None:
        print(f"    WARNING: no intrinsics for serial {serial_nr}, skipping")
        return None

    undistorted = cv2.remap(img, cam["map1"], cam["map2"],
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return fix_radiometrics(undistorted)


def save_pinhole_K_json(pinhole_data, output_dir):
    out = {}
    for sn, cam in pinhole_data.items():
        K = cam["new_K"]
        out[sn] = {
            "camera_name": cam["name"],
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        }
    path = os.path.join(output_dir, "pinhole_K.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Pinhole K saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laz", required=True, help="LAZ tile (only the header is read, for its bounding box)")
    p.add_argument("--image-dir", required=True, help="directory of raw camera frames")
    p.add_argument("--camextr", required=True, help="CamExtr.json with per-frame pose + serial number")
    p.add_argument("--intrinsics", required=True, help="calibration XML with per-camera intrinsics")
    p.add_argument("--output-dir", default="./output", help="where to write corrected frames + report")
    return p.parse_args()


def main():
    args = parse_args()
    part_name = os.path.basename(args.laz)
    print(f"\n{'='*70}\n  Camera pre-processing\n  LAZ tile: {part_name}\n{'='*70}\n")

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] Reading LAZ bounding box...")
    bbox = get_laz_bbox(args.laz)

    print("\n[2/6] Filtering CamExtr.json to tile extent...")
    cam_entries = load_and_filter_camextr(args.camextr, bbox)
    if not cam_entries:
        print("  ERROR: no images found within the tile bounding box. Check --image-dir / --camextr.")
        return

    cam_counts = {}
    for entry in cam_entries:
        sn = entry["SerialNr"]
        cam_counts[sn] = cam_counts.get(sn, 0) + 1
    print("  Per-camera image counts:")
    for sn, count in sorted(cam_counts.items()):
        print(f"    sn={sn}: {count} images")

    print("\n[3/6] Loading camera intrinsics from XML...")
    intrinsics = load_intrinsics(args.intrinsics)

    print("\n[4/6] Computing pinhole K and undistortion maps...")
    pinhole_data = compute_pinhole_Ks(intrinsics)

    print("\n[5/6] Saving pinhole K data...")
    save_pinhole_K_json(pinhole_data, args.output_dir)

    print(f"\n[6/6] Processing {len(cam_entries)} images...")
    processed = skipped = missing = 0

    for i, entry in enumerate(cam_entries):
        filename = entry["Image"]
        serial = entry["SerialNr"]
        img_path = os.path.join(args.image_dir, filename)

        if not os.path.isfile(img_path):
            missing += 1
            if missing <= 5:
                print(f"    MISSING: {filename}")
            elif missing == 6:
                print("    ... suppressing further missing-file warnings")
            continue

        corrected = process_image(img_path, serial, pinhole_data)
        if corrected is None:
            skipped += 1
            continue

        out_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(out_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(cam_entries):
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    [{i+1}/{len(cam_entries)}] processed={processed} "
                  f"skipped={skipped} missing={missing} ({rate:.1f} img/s)")

    elapsed = time.time() - t_start
    summary = [
        "Camera pre-processing report", "=" * 40,
        f"LAZ tile      : {part_name}",
        f"Total entries : {len(cam_entries)} (within bbox)",
        f"Processed     : {processed}",
        f"Skipped       : {skipped}",
        f"Missing files : {missing}",
        f"Elapsed       : {elapsed:.1f}s",
        f"Rate          : {processed/elapsed:.1f} img/s" if elapsed > 0 else "Rate: N/A",
        f"Output dir    : {args.output_dir}",
        "", "Per-camera breakdown:",
    ]
    for sn, count in sorted(cam_counts.items()):
        cam_name = intrinsics.get(sn, {}).get("name", "unknown")
        summary.append(f"  {cam_name} (sn={sn}): {count} images")

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")
    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write(summary_text + "\n")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
