#!/usr/bin/env python3
"""
Builds the part47-48-49-50 corridor asset for the drive-through video:
  1. Merge the 4 tiles' colorized, gap-filled LAZ files into one continuous
     cloud (~119.6M points spanning ~1.2km, established this session via a
     bbox-adjacency check).
  2. Concatenate their 4 detections.jsonl sidecars into one file.
  3. Bake STVO hover-label stickers onto the merged cloud in one pass, using
     colorize_pointcloud.bake_hover_labels (LiDAR-matched detections only --
     see that function's own docstring for why unmatched ones are dropped).

Usage:
    python build_corridor_driveThrough.py
"""

import sys, os, time
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import bake_hover_labels

COLOR_DIR = "/home/rohit/Downloads/Project/hd_map_p06/output/colorization"
TILES = [
    ("part47", f"{COLOR_DIR}/part47_completed_from_part25_colorized.laz",
               f"{COLOR_DIR}/part47_completed_detections_sam3.jsonl"),
    ("part48", f"{COLOR_DIR}/part48_completed_from_part24_colorized.laz",
               f"{COLOR_DIR}/part48_completed_detections_sam3.jsonl"),
    ("part49", f"{COLOR_DIR}/part49_completed_from_part23_colorized.laz",
               f"{COLOR_DIR}/part49_completed_detections_sam3.jsonl"),
    ("part50", f"{COLOR_DIR}/part50_completed_from_part22_colorized.laz",
               f"{COLOR_DIR}/part50_completed_detections_sam3.jsonl"),
]

OUT_DIR = "/home/rohit/Documents/Output/phase2"
MERGED_LAZ = f"{OUT_DIR}/corridor_47_48_49_50_merged.laz"
MERGED_DETECTIONS = f"{OUT_DIR}/corridor_47_48_49_50_detections.jsonl"
LABELED_LAZ = f"{OUT_DIR}/corridor_47_48_49_50_labeled.laz"


def log(msg):
    print(msg, flush=True)


def merge_laz(tiles, out_path):
    log(f"[1/3] Merging {len(tiles)} colorized tiles into one corridor cloud...")
    xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
    ref_header = None
    total = 0
    for name, laz_path, _ in tiles:
        t0 = time.time()
        las = laspy.read(laz_path)
        n = len(las.x)
        total += n
        log(f"  {name}: {n:,} points ({time.time()-t0:.1f}s)")
        if ref_header is None:
            ref_header = laspy.LasHeader(point_format=las.point_format, version=las.header.version)
            ref_header.offsets = las.header.offsets
            ref_header.scales = las.header.scales
        xs.append(np.asarray(las.x)); ys.append(np.asarray(las.y)); zs.append(np.asarray(las.z))
        rs.append(np.asarray(las.red)); gs.append(np.asarray(las.green)); bs.append(np.asarray(las.blue))

    out_las = laspy.LasData(ref_header)
    out_las.x = np.concatenate(xs)
    out_las.y = np.concatenate(ys)
    out_las.z = np.concatenate(zs)
    out_las.red = np.concatenate(rs)
    out_las.green = np.concatenate(gs)
    out_las.blue = np.concatenate(bs)
    log(f"  total merged: {total:,} points")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_las.write(out_path)
    log(f"  saved: {out_path}")
    return total


def merge_detections(tiles, out_path):
    log(f"[2/3] Concatenating {len(tiles)} detections.jsonl sidecars...")
    n_total = 0
    with open(out_path, "w") as fout:
        for name, _, det_path in tiles:
            with open(det_path) as fin:
                lines = fin.readlines()
            for line in lines:
                fout.write(line if line.endswith("\n") else line + "\n")
            n_total += len(lines)
            log(f"  {name}: {len(lines)} detection rows")
    log(f"  total: {n_total} rows -> {out_path}")


if __name__ == "__main__":
    merge_laz(TILES, MERGED_LAZ)
    merge_detections(TILES, MERGED_DETECTIONS)
    log("[3/3] Baking STVO hover-label stickers onto the merged corridor...")
    bake_hover_labels(MERGED_LAZ, MERGED_DETECTIONS, LABELED_LAZ)
    log(f"\nDone. Labeled corridor cloud: {LABELED_LAZ}")
