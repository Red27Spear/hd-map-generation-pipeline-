#!/usr/bin/env python3
"""
Variant of phase2_strip_ground_v3.py that keeps the road/ground surface
INSTEAD OF discarding it -- the previous vertical-only pipeline threw the
road away entirely, but the road surface is needed for context/visual
validation. Reuses the exact same RANSAC+adaptive-local ground estimation
and pole/sign/light classification logic (same "keep 1.20m-wide, h/w>=2.0
vertical structures" behaviour), just changes what gets written out:

    final kept = ground_mask (the road/ground surface) OR vertical_mask
                 (pole/sign/light clusters)

Run on an ALREADY-COLORIZED input tile (see colorize_pointcloud.py), so RGB
carries straight through -- laspy's save_laz already copies every point
dimension, RGB included, no separate re-colorize pass needed afterward.

Points are tagged via the standard LAS `classification` field so downstream
pole-extraction code can still isolate just the vertical structures without
re-running the ground/vertical split:
    classification == 2  ->  road/ground
    classification == 1  ->  vertical structure (pole/sign/light)

Usage:
    python phase2_road_and_vertical.py <colorized_laz_file>
"""

import sys
import os
import re
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from phase2_strip_ground_v3 import (
    estimate_ground_and_strip, process_elevated_points, log,
)

OUTPUT_DIR = "/home/rohit/Documents/Output/phase2"

CLASS_ROAD = 2
CLASS_VERTICAL = 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_road_and_vertical.py <colorized_laz_file>")
        sys.exit(1)

    laz_path = sys.argv[1]
    basename = os.path.basename(laz_path)
    part_match = re.search(r'(part\d+)', basename)
    part_name = part_match.group(1) if part_match else basename.replace('.laz', '')

    log(f"\n{'='*70}")
    log(f"  PHASE 2 -- Road + Vertical Keeper (colorized input)")
    log(f"  LAZ tile : {basename}")
    log(f"  Part     : {part_name}")
    log(f"{'='*70}\n")

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import laspy
    log("[1/4] Loading colorized point cloud...")
    las_original = laspy.read(laz_path)
    xyz = np.vstack([las_original.x, las_original.y, las_original.z]).T.astype(np.float64)
    has_rgb = "red" in las_original.point_format.dimension_names
    log(f"    {len(xyz):,} points loaded (RGB present: {has_rgb})")

    log("\n[2/4] Estimating ground/road surface...")
    ground_mask, elevated_mask, ground_z = estimate_ground_and_strip(xyz)

    log("\n[3/4] Clustering elevated points to find pole/sign/light structures...")
    elevated_xyz = xyz[elevated_mask]
    elevated_indices = np.where(elevated_mask)[0]
    vertical_mask = np.zeros(len(xyz), dtype=bool)
    if elevated_mask.sum() > 0:
        keep_elevated_idx, counts, decisions_log = process_elevated_points(elevated_xyz, elevated_indices, ground_z)
        vertical_mask[keep_elevated_idx] = True
        log(f"    Vertical structures kept: {vertical_mask.sum():,} points "
            f"({counts['keep']} clusters)")
    else:
        log("    WARNING: no elevated points found, no vertical structures possible")

    final_keep_mask = ground_mask | vertical_mask
    log(f"\n    Road/ground points: {ground_mask.sum():,}")
    log(f"    Vertical points:    {vertical_mask.sum():,}")
    log(f"    Combined kept:      {final_keep_mask.sum():,} / {len(xyz):,} "
        f"({100*final_keep_mask.sum()/len(xyz):.1f}%)")

    log("\n[4/4] Saving road+vertical point cloud (classification-tagged)...")
    classification = np.zeros(len(xyz), dtype=np.uint8)
    classification[ground_mask] = CLASS_ROAD
    classification[vertical_mask] = CLASS_VERTICAL

    hdr = laspy.LasHeader(point_format=las_original.point_format, version=las_original.header.version)
    hdr.offsets = las_original.header.offsets
    hdr.scales = las_original.header.scales
    new_las = laspy.LasData(hdr)
    for dim in las_original.point_format.dimension_names:
        setattr(new_las, dim, np.array(getattr(las_original, dim))[final_keep_mask])
    new_las.classification = classification[final_keep_mask]

    out_path = os.path.join(OUTPUT_DIR, f"road_and_vertical_{part_name}.laz")
    new_las.write(out_path)
    log(f"    Saved {final_keep_mask.sum():,} points -> {out_path}")

    elapsed = time.time() - t_start
    log(f"\nElapsed: {elapsed:.1f}s")
    log("Done.\n")


if __name__ == "__main__":
    main()
