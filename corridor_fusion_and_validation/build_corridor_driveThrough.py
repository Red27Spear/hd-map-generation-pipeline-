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
    python build_corridor_driveThrough.py --out-dir ./output \
        --tile part47:completed_from_part25_colorized.laz:part47_detections.jsonl \
        --tile part48:completed_from_part24_colorized.laz:part48_detections.jsonl \
        [...]

Each --tile is "name:colorized_laz_path:detections_jsonl_path" (colon-
separated) for one already gap-filled, colorized recipient tile -- the
output of merge_overlapping_tiles.py + gapfill_recipient_from_donor.py +
combine_colorized.py run on that tile. No demo corridor data is bundled in
this repo (only the single part15 tile), so --tile must be supplied.
"""

import sys, os, time, argparse
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_pointcloud import bake_hover_labels


def parse_tile_arg(spec):
    name, laz, jsonl = spec.split(":", 2)
    return name, laz, jsonl


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", action="append", required=True, dest="tiles",
                     help='one per recipient tile: "name:colorized_laz_path:detections_jsonl_path"')
    ap.add_argument("--out-dir", default="./output")
    args = ap.parse_args()

    tiles = [parse_tile_arg(t) for t in args.tiles]
    os.makedirs(args.out_dir, exist_ok=True)
    merged_laz = os.path.join(args.out_dir, "corridor_merged.laz")
    merged_detections = os.path.join(args.out_dir, "corridor_detections.jsonl")
    labeled_laz = os.path.join(args.out_dir, "corridor_labeled.laz")

    merge_laz(tiles, merged_laz)
    merge_detections(tiles, merged_detections)
    log("[3/3] Baking STVO hover-label stickers onto the merged corridor...")
    bake_hover_labels(merged_laz, merged_detections, labeled_laz)
    log(f"\nDone. Labeled corridor cloud: {labeled_laz}")
