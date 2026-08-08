#!/usr/bin/env python3
"""
Generalized version of combine_part48_gapfill.py -- unions two colorized
LAZ files (recipient's own points + colorized gap-fill points) into one
combined tile, taking file paths via CLI instead of hardcoded part24/part48
paths.

Usage:
    python combine_colorized.py <recipient_colorized_laz> <gapfill_colorized_laz> <out_laz>
"""

import sys
import numpy as np
import laspy


def log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python combine_colorized.py <recipient_colorized_laz> <gapfill_colorized_laz> <out_laz>")
        sys.exit(1)
    FILE_A, FILE_B, OUT_LAZ = sys.argv[1], sys.argv[2], sys.argv[3]

    log(f"Loading recipient (own, colorized): {FILE_A}")
    las_a = laspy.read(FILE_A)
    log(f"  {len(las_a.x):,} points")

    log(f"Loading gap-fill (colorized): {FILE_B}")
    las_b = laspy.read(FILE_B)
    log(f"  {len(las_b.x):,} points")

    hdr = laspy.LasHeader(point_format=las_a.point_format, version=las_a.header.version)
    hdr.offsets = las_a.header.offsets
    hdr.scales = las_a.header.scales
    out_las = laspy.LasData(hdr)

    out_las.x = np.concatenate([np.asarray(las_a.x), np.asarray(las_b.x)])
    out_las.y = np.concatenate([np.asarray(las_a.y), np.asarray(las_b.y)])
    out_las.z = np.concatenate([np.asarray(las_a.z), np.asarray(las_b.z)])
    out_las.red = np.concatenate([np.asarray(las_a.red), np.asarray(las_b.red)])
    out_las.green = np.concatenate([np.asarray(las_a.green), np.asarray(las_b.green)])
    out_las.blue = np.concatenate([np.asarray(las_a.blue), np.asarray(las_b.blue)])

    n_total = len(out_las.x)
    log(f"\nCombined: {n_total:,} points ({len(las_a.x):,} recipient own + {len(las_b.x):,} gap-fill)")

    out_las.write(OUT_LAZ)
    log(f"\nSaved: {OUT_LAZ}")
