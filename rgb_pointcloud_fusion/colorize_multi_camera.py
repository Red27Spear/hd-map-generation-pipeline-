#!/usr/bin/env python3
"""
Colorize a LAZ point cloud using multiple fisheye cameras, generalizing the
validated approach from colorize_with_left_camera.py:
  - Camera pose = CamExtr.json's per-shot Hrp/Xyz used DIRECTLY as the
    camera's world pose (xyz euler order [roll,pitch,heading], no separate
    R0 composition), plus the refined per-camera correction from
    joint_calibration.py's correspondence-based optimizer:
        R_actual = R_nominal @ dR_correction
        t_actual = t_nominal + R_nominal @ dt_correction
  - Points matched to camera shots by GPS TIME (physically correct: color a
    point using the image captured at the same instant it was scanned).

KNOWN LIMITATION (see conversation record): reprojection error against the
real picked landmarks in correspondences.json remains large even after
correction (roughly 700-1150 px mean on a 2448x2048 image, cameras
4108690955/4108736902/4108690914). This is a large improvement over the
uncorrected nominal pose (~1300-1570 px mean) and over yesterday's
completely broken stage5 chain (near-zero valid points), but is NOT
pixel-accurate. Expect visibly smeared/offset texturing on fine features
(signs, poles); broad regional coloring (road, vegetation, building
facades) is qualitatively plausible (visually confirmed on the left
camera: colored points form a continuous strip following the road
corridor, not scattered garbage).

DOWN/UP POSE SWAP BUG (found and fixed): CamExtr.json's Hrp/Xyz entries for
serial 4108736907 ("down") and 4108690914 ("up") are cross-swapped -- the
Image filenames are correctly labeled per serial, but the pose attached to
each serial's record actually belongs to the OTHER physical camera.
Diagnostic: computing each camera's world-frame forward direction
(R_nominal @ [0,0,1]) from its own Hrp gives "down" a forward pointing
~21.9 deg off world +Z (i.e. looking up) and "up" a forward pointing
~21.9 deg off world -Z (looking down) -- backwards from both the camera
names AND their actual photo content (down's photos show the road, up's
show the sky). Using the OTHER serial's Hrp/Xyz at the matching Time fixes
this (see POSE_SOURCE_SERIAL below); confirmed visually on 2 independent
down-camera images -- projected points trace the real lane markings and
road surface with a correct near/far depth gradient, instead of forming a
nonsensical vertical band. Left/right cameras were checked and do NOT show
this symptom (their forward directions are near-horizontal and consistent
with their names/mirrored mounting).

Camera 4108736907 ("down", after the pose-swap fix) has ZERO observations
in correspondences.json -- there is no fitted dR/dt correction for it, only
the swap-corrected nominal pose. Camera 4108690914 ("up") is EXCLUDED by
default per user direction (its own fitted correction was likely partly
compensating for this same swap bug and is not trustworthy either).

Usage:
    python3 colorize_multi_camera.py \
        --laz /data/Project/hd_map_p06/data/Test/LAZ/<tile>.laz \
        --camextr /data/Project/hd_map_p06/new_config/CamExtr.json \
        --intrinsics-xml /data/Project/hd_map_p06/new_config/9020C_0140_toScanner_final.xml \
        --refined-calibration refined_calibration.json \
        --image-dir /data/Project/hd_map_p06/data/Test/color/color \
        --output /path/to/output.laz \
        --cameras 4108690955 4108736902 4108736907
"""
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import laspy
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERAS = ["4108690955", "4108736902", "4108736907"]  # left, right, down
# 4108690914 (up) excluded per user direction.

# Down/up pose-swap fix: when loading POSE data (Hrp/Xyz) for a camera in
# this map, pull it from the other serial's CamExtr entries (matched by
# nearest Time) instead of the camera's own. Intrinsics (K/D) and image
# filenames still come from the camera's own serial -- only the Hrp/Xyz
# source is swapped.
POSE_SOURCE_SERIAL = {
    "4108736907": "4108690914",  # down's pose comes from up's entries
    "4108690914": "4108736907",  # up's pose comes from down's entries (unused by default)
}


def load_intrinsics(xml_path, serial):
    tree = ET.parse(xml_path)
    for cam in tree.getroot().findall("Camera"):
        if cam.get("serialno") == serial:
            fx = float(cam.findtext("fx")); fy = float(cam.findtext("fy"))
            cx = float(cam.findtext("cx")); cy = float(cam.findtext("cy"))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            D = np.array([float(cam.findtext(f"k{i}")) for i in range(1, 5)]).reshape(4, 1)
            img_w = int(cam.findtext("image_w"))
            img_h = int(cam.findtext("image_h"))
            name = cam.get("name", serial)
            return K, D, img_w, img_h, name
    raise ValueError(f"Serial {serial} not found in {xml_path}")


def load_camera_poses(camextr_path, refined_calibration_path, serial):
    with open(camextr_path, "r") as f:
        data = json.load(f)
    all_entries = data["Profiler_0"]
    entries = [e for e in all_entries if str(e["SerialNr"]).strip() == serial]
    entries.sort(key=lambda e: e["Time"])

    pose_serial = POSE_SOURCE_SERIAL.get(serial, serial)
    if pose_serial != serial:
        print(f"  [{serial}] pose-swap fix active: using {pose_serial}'s Hrp/Xyz instead of its own")
        pose_entries = [e for e in all_entries if str(e["SerialNr"]).strip() == pose_serial]
        pose_times_arr = np.array([e["Time"] for e in pose_entries])
        sort_idx = np.argsort(pose_times_arr)
        pose_entries = [pose_entries[i] for i in sort_idx]
        pose_times_arr = pose_times_arr[sort_idx]

        def pose_source(e):
            idx = int(np.argmin(np.abs(pose_times_arr - e["Time"])))
            return pose_entries[idx]
    else:
        def pose_source(e):
            return e

    dR, dt = np.eye(3), np.zeros(3)
    if refined_calibration_path:
        with open(refined_calibration_path, "r") as f:
            refined = json.load(f)
        cam_corr = refined["cameras"].get(serial)
        if cam_corr is not None:
            dR = np.array(cam_corr["dR_correction"])
            dt = np.array(cam_corr["dt_correction_m"])
            print(f"  [{serial}] correction: {cam_corr['correction_rotation_magnitude_deg']:.2f} deg, "
                  f"{cam_corr['correction_translation_magnitude_m']*1000:.1f} mm")
        else:
            print(f"  [{serial}] WARNING: no refined correction found -- using identity")

    times = np.zeros(len(entries))
    poses = []
    for i, e in enumerate(entries):
        pe = pose_source(e)
        heading, roll, pitch = pe["Hrp"]
        R_nominal = R.from_euler("xyz", [roll, pitch, heading], degrees=True).as_matrix()
        cam_xyz = np.array(pe["Xyz"], dtype=np.float64)
        R_actual = R_nominal @ dR
        t_actual = cam_xyz + R_nominal @ dt
        times[i] = e["Time"]
        poses.append((R_actual, t_actual, e["Image"]))
    return times, poses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", required=True)
    ap.add_argument("--camextr", required=True)
    ap.add_argument("--intrinsics-xml", required=True)
    ap.add_argument("--refined-calibration", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    ap.add_argument("--max-time-diff", type=float, default=1.0)
    ap.add_argument("--max-camera-dist", type=float, default=30.0)
    ap.add_argument("--fallback-color", nargs=3, type=float, default=[0.4, 0.4, 0.4])
    args = ap.parse_args()

    print(f"Cameras: {args.cameras}")
    cams = {}
    for serial in args.cameras:
        K, D, w, h, name = load_intrinsics(args.intrinsics_xml, serial)
        print(f"Loading poses for {serial} ({name})...")
        times, poses = load_camera_poses(args.camextr, args.refined_calibration, serial)
        cams[serial] = dict(K=K, D=D, w=w, h=h, name=name, times=times, poses=poses)
        print(f"  {len(poses)} shots")

    print(f"\nLoading point cloud: {args.laz}")
    las = laspy.read(args.laz)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    n_points = len(pts)
    print(f"  {n_points:,} points loaded")

    has_gps_time = "gps_time" in las.point_format.dimension_names
    if not has_gps_time:
        raise RuntimeError("LAZ has no gps_time field -- time-based matching requires it")
    pt_times = np.asarray(las.gps_time)

    best_color = np.tile(np.array(args.fallback_color), (n_points, 1))
    best_center_dist = np.full(n_points, np.inf)
    colored_by_cam = {s: 0 for s in args.cameras}
    image_cache = {}

    for serial, cam in cams.items():
        print(f"\n=== Camera {serial} ({cam['name']}) ===")
        times, poses = cam["times"], cam["poses"]
        K, D, W, H = cam["K"], cam["D"], cam["w"], cam["h"]

        time_idx = np.searchsorted(times, pt_times)
        time_idx = np.clip(time_idx, 1, len(times) - 1)
        left_diff = np.abs(pt_times - times[time_idx - 1])
        right_diff = np.abs(pt_times - times[time_idx])
        cam_idx = np.where(left_diff < right_diff, time_idx - 1, time_idx)
        time_diffs = np.minimum(left_diff, right_diff)

        cam_positions = np.array([p[1][:2] for p in poses])
        spatial_dists = np.linalg.norm(cam_positions[cam_idx] - pts[:, :2], axis=1)
        valid_camera = (time_diffs <= args.max_time_diff) & (spatial_dists <= args.max_camera_dist)
        print(f"  {valid_camera.sum():,}/{n_points:,} points within {args.max_time_diff}s and {args.max_camera_dist}m")

        unique_cam_indices = np.unique(cam_idx[valid_camera])
        n_colored_this_cam = 0
        for ci in unique_cam_indices:
            mask = valid_camera & (cam_idx == ci)
            if not mask.any():
                continue
            R_actual, t_actual, image_name = poses[ci]
            pts_group = pts[mask]
            p_cam = (R_actual.T @ (pts_group - t_actual).T).T

            front_mask = p_cam[:, 2] > 1e-6
            if not front_mask.any():
                continue

            obj = p_cam[front_mask].reshape(-1, 1, 3).astype(np.float64)
            pix, _ = cv2.fisheye.projectPoints(obj, np.zeros((3, 1)), np.zeros((3, 1)), K, D)
            pix = pix.reshape(-1, 2)

            in_frame = ((pix[:, 0] >= 0) & (pix[:, 0] < W) &
                        (pix[:, 1] >= 0) & (pix[:, 1] < H))
            if not in_frame.any():
                continue

            cache_key = (serial, image_name)
            if cache_key not in image_cache:
                img_path = f"{args.image_dir}/{image_name}"
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image_cache[cache_key] = img
                if len(image_cache) > 80:
                    image_cache.pop(next(iter(image_cache)))
            img = image_cache[cache_key]
            if img is None:
                continue

            group_indices = np.where(mask)[0]
            front_indices = group_indices[front_mask]
            valid_final = front_indices[in_frame]
            valid_pix = pix[in_frame]

            u = np.clip(valid_pix[:, 0].astype(int), 0, W - 1)
            v = np.clip(valid_pix[:, 1].astype(int), 0, H - 1)
            sampled_bgr = img[v, u]

            dx = (valid_pix[:, 0] - W / 2.0) / (W / 2.0)
            dy = (valid_pix[:, 1] - H / 2.0) / (H / 2.0)
            center_dist = np.sqrt(dx**2 + dy**2)

            update = center_dist < best_center_dist[valid_final]
            upd_idx = valid_final[update]
            best_color[upd_idx] = sampled_bgr[update][:, ::-1] / 255.0
            best_center_dist[upd_idx] = center_dist[update]
            n_colored_this_cam += update.sum()

        colored_by_cam[serial] = n_colored_this_cam
        print(f"  contributed {n_colored_this_cam:,} winning points")

    n_colored = int((best_center_dist < np.inf).sum())
    print(f"\n=== Final ===")
    print(f"Colored: {n_colored:,}/{n_points:,} ({100*n_colored/n_points:.1f}%)")
    for s, c in colored_by_cam.items():
        print(f"  {s} ({cams[s]['name']}): {c:,} winning points")

    rgb_capable_formats = {2, 3, 5, 7, 8, 10}
    src_format_id = las.header.point_format.id
    out_format_id = src_format_id if src_format_id in rgb_capable_formats else 3
    out_header = laspy.LasHeader(point_format=out_format_id, version=las.header.version)
    out_header.offsets = las.header.offsets
    out_header.scales = las.header.scales
    out_las = laspy.LasData(out_header)
    out_las.x, out_las.y, out_las.z = las.x, las.y, las.z

    src_dims = set(las.point_format.dimension_names)
    out_dims = set(out_las.point_format.dimension_names)
    for dim in src_dims & out_dims:
        if dim in ("X", "Y", "Z"):
            continue
        try:
            setattr(out_las, dim, np.array(getattr(las, dim)))
        except Exception:
            pass

    rgb_16bit = np.clip(best_color * 65535, 0, 65535).astype(np.uint16)
    out_las.red, out_las.green, out_las.blue = rgb_16bit[:, 0], rgb_16bit[:, 1], rgb_16bit[:, 2]
    out_las.write(args.output)
    print(f"Saved colorized LAZ to {args.output}")


if __name__ == "__main__":
    main()
