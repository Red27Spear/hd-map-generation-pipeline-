#!/usr/bin/env python3
"""
Merges cross_pass_consistency_check.py's per-pole trust flag back into the
main pole-list JSON (extract_and_snap_poles.py's output), matched by LiDAR
UTM position. Every pole gets a "cross_pass_status" field:
    "trusted"     -- checked, both passes' triangulation converged (RMS<100px),
                     agreed to within a few metres
    "untrusted"   -- checked, but at least one pass's triangulation never
                     converged -- the cross-pass comparison for this pole is
                     not meaningful
    "not_checked" -- not dual-pass, or fewer than 2 observations in one pass,
                     so this check couldn't run at all (most poles)

Usage:
    python merge_trust_flag.py <poles_json> <cross_pass_check_json> <out_json>
"""

import sys
import json

MATCH_TOL_M = 0.01  # positions come from the same source computation, should match exactly


if __name__ == "__main__":
    poles_path, check_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    poles = json.load(open(poles_path))
    check = json.load(open(check_path))

    checked_by_pos = {}
    for r in check["poles"]:
        key = (round(r["lidar_utm_easting"], 2), round(r["lidar_utm_northing"], 2))
        checked_by_pos[key] = r

    n_trusted = n_untrusted = n_not_checked = 0
    for p in poles:
        key = (round(p["utm_easting"], 2), round(p["utm_northing"], 2))
        r = checked_by_pos.get(key)
        if r is None:
            p["cross_pass_status"] = "not_checked"
            n_not_checked += 1
        elif r["cross_pass_trusted"]:
            p["cross_pass_status"] = "trusted"
            p["cross_pass_horiz_dist_m"] = r["cross_pass_horiz_dist_m"]
            n_trusted += 1
        else:
            p["cross_pass_status"] = "untrusted"
            p["cross_pass_horiz_dist_m"] = r["cross_pass_horiz_dist_m"]
            n_untrusted += 1

    print(f"trusted={n_trusted}  untrusted={n_untrusted}  not_checked={n_not_checked}  "
          f"(total {len(poles)})")

    with open(out_path, "w") as f:
        json.dump(poles, f, indent=2)
    print(f"\nSaved: {out_path}")
