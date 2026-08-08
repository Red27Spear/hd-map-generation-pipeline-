#!/usr/bin/env python3
"""
Bakes this pipeline's refined light positions onto the existing colorized
point cloud as visual stickers -- same mechanism scripts/proj/
place_light_stickers.py already uses (real StVO light texture via
phase4c_stvo_sprites.make_light_texture_stvo, billboard placement via
colorize_pointcloud.place_label_plate), adapted here because our tags
don't carry the bearing_rad field finalize_hover_tags.py's tags have
(that pipeline never ran on this data) -- computed fresh per light from
its representative detection's camera pose instead.

Colour-codes each sticker by what actually happened to it in the
pipeline, so the visualization itself communicates the honest picture
from HOVER_LABELS_STATUS.md rather than hiding it:
  - refined (Gauss-Newton reprojection refinement applied): normal sticker
  - not refined (fewer than 2 usable observations, kept as-is): sticker
    tinted blue, so it's visually distinguishable in CloudCompare/QGIS

Usage:
    python visualize_pipeline_result.py [--refined-tags PATH] [--base-laz PATH] [--out PATH]

--refined-tags is pipeline.py's own output (part15_pipeline_refined_tags.json
by default); --base-laz is any already-colorized LAZ for the same tile, e.g.
data/part15/results/colorized_part15.laz in this repo's bundled demo data.
"""

import sys, os, json, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import laspy

from colorize_pointcloud import place_label_plate, LABEL_PLATE_H_M, load_pinhole_K, CAMEXTR_PATH
from phase4c_stvo_sprites import make_light_texture_stvo

REFINED_TAGS = "./output/colorization/part15_pipeline_refined_tags.json"  # from pipeline.py
BASE_LAZ = "../data/part15/results/colorized_part15.laz"
OUTPUT_LAZ = "./output/colorization/part15_pipeline_refined_lights_visual.laz"

STICKER_W_M, STICKER_H_M = 0.40, 1.10
STICKER_GAP_M = 0.15


def rotMat(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def bearing_from_rep_image(image_name, by_image):
    e = by_image.get(image_name)
    if e is None:
        return 0.0
    R = rotMat(*np.radians(e["Hrp"]))
    dir_world = R @ np.array([0.0, 0.0, 1.0])
    return math.atan2(dir_world[0], dir_world[1])


def main():
    global REFINED_TAGS, BASE_LAZ, OUTPUT_LAZ

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refined-tags", default=REFINED_TAGS)
    ap.add_argument("--base-laz", default=BASE_LAZ)
    ap.add_argument("--out", default=OUTPUT_LAZ)
    args = ap.parse_args()
    REFINED_TAGS, BASE_LAZ, OUTPUT_LAZ = args.refined_tags, args.base_laz, args.out
    os.makedirs(os.path.dirname(OUTPUT_LAZ) or ".", exist_ok=True)

    print("[1/3] Loading pipeline output + camera poses...")
    tags = json.load(open(REFINED_TAGS))
    camextr = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    by_image = {e["Image"]: e for e in camextr}
    print(f"  {len(tags)} lights")

    print("[2/3] Building sticker billboards (blue tint = not refined)...")
    tex_refined = make_light_texture_stvo(verified=True)
    tex_unrefined = tex_refined.copy()
    tex_unrefined[..., 2] = np.minimum(255, tex_unrefined[..., 2].astype(int) + 80).astype(np.uint8)  # boost blue channel

    new_xyz, new_rgb = [], []
    for t in tags:
        bearing = bearing_from_rep_image(t["rep_image"], by_image)
        sticker_z = t["utm_z"] - LABEL_PLATE_H_M / 2.0 - STICKER_GAP_M - STICKER_H_M / 2.0
        centre = np.array([t["utm_easting"], t["utm_northing"], sticker_z])
        tex = tex_refined if t["refinement"]["applied"] else tex_unrefined
        pxyz, prgb = place_label_plate(centre, tex, bearing, STICKER_W_M, STICKER_H_M)
        if len(pxyz):
            new_xyz.append(pxyz)
            new_rgb.append(prgb)
        status = "refined" if t["refinement"]["applied"] else "NOT refined (kept as-is)"
        print(f"  ({t['utm_easting']:.1f},{t['utm_northing']:.1f}) bearing={math.degrees(bearing):.0f}deg -- {status}")

    new_xyz = np.vstack(new_xyz) if new_xyz else np.empty((0, 3))
    new_rgb = np.vstack(new_rgb) if new_rgb else np.empty((0, 3), dtype=np.uint8)
    print(f"  {len(new_xyz):,} sticker points for {len(tags)} lights")

    print("[3/3] Merging into the point cloud...")
    las = laspy.read(BASE_LAZ)
    out_las = laspy.LasData(header=las.header)
    out_las.x = np.concatenate([las.x, new_xyz[:, 0]])
    out_las.y = np.concatenate([las.y, new_xyz[:, 1]])
    out_las.z = np.concatenate([las.z, new_xyz[:, 2]])
    out_las.red = np.concatenate([np.asarray(las.red), new_rgb[:, 0].astype(np.uint16) * 257])
    out_las.green = np.concatenate([np.asarray(las.green), new_rgb[:, 1].astype(np.uint16) * 257])
    out_las.blue = np.concatenate([np.asarray(las.blue), new_rgb[:, 2].astype(np.uint16) * 257])
    out_las.write(OUTPUT_LAZ)
    print(f"Saved: {OUTPUT_LAZ}")


if __name__ == "__main__":
    main()
