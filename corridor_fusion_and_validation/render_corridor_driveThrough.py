#!/usr/bin/env python3
"""
Renders a first-person drive-through video along the real vehicle trajectory
through a labeled point cloud tile/corridor (STVO hover-label stickers
already baked in via colorize_pointcloud.bake_hover_labels).

Camera path: real vehicle positions (CamExtr "down" camera entries, sorted
by time -- one position per capture cycle, not per individual camera) inside
the cloud's bbox, resampled to even 1.5m spacing and lightly smoothed. A
virtual chase camera is placed above and looks a fixed distance ahead along
the smoothed path -- not any single real camera's own (heavily rolled/
pitched) pose, since none of the 4 survey cameras face straight ahead.

Rendering: Open3D offscreen (headless) rendering, one frame per path sample,
assembled into an MP4 via OpenCV's VideoWriter.

Usage:
    python render_corridor_driveThrough.py [--laz <path>] [--frames-dir <dir>]
        [--out <mp4>] [--pass {1,2}]

Defaults to the part47-48-49-50 corridor build if no args given. --pass 1/2
restricts the trajectory to before/after the whole survey day's outbound/
return-leg time split (PASS_SPLIT_T); omit it to use every in-bbox capture
(the right choice for a single-pass tile like part15).
"""

import sys, os, json, glob
import numpy as np

os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.setdefault("DISPLAY", ":1")

import cv2
import laspy
import open3d as o3d
import open3d.visualization.rendering as rendering

CAMEXTR_PATH = "../config/CamExtr.json"


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


LABELED_LAZ = _arg("--laz", "../data/part15/results/colorized_part15.laz")  # --laz is effectively required: no multi-tile corridor demo data is bundled
FRAMES_DIR = _arg("--frames-dir", "./output/driveThrough_frames")
OUT_MP4 = _arg("--out", "./output/corridor_driveThrough.mp4")
PASS_FILTER = _arg("--pass", None)  # None, "1", or "2"
RENDERER = _arg("--renderer", "legacy")  # "legacy" (fast, small/blocky) or "hq" (Filament, no downsample)

# The whole day's survey has one outbound leg (pass1) and one return leg
# (pass2), split by time-of-day -- same global constant already established
# and used to colorize these exact 4 recipient tiles (colorize_pointcloud.py).
PASS_SPLIT_T = 31100
DOWN_CAM_SERIAL = "4108736907"

RESAMPLE_SPACING_M = 1.5
CAM_HEIGHT_M = 2.6          # above the real trajectory point, roughly windshield height
LOOKAHEAD_M = 9.0
SMOOTH_WINDOW = 7           # odd, samples

VOXEL_SIZE_M = 0.0 if RENDERER == "hq" else float(_arg("--voxel", "0.06"))  # 0 = no downsampling
FRAME_W, FRAME_H = (int(_arg("--width", "1280")), int(_arg("--height", "720"))) if RENDERER == "hq" \
    else (int(_arg("--width", "960")), int(_arg("--height", "540")))
FPS = 20
HQ_POINT_SIZE = float(_arg("--point-size", "4.0"))
HQ_VERTICAL_FOV_DEG = 55.0
LABEL_SPLIT_N = _arg("--label-split", None)  # point index where STVO label points begin (trailing block from bake_hover_labels); render them oversized so they don't visually blend into nearby real structure
LABEL_SPLIT_N = int(LABEL_SPLIT_N) if LABEL_SPLIT_N is not None else None
LABEL_POINT_SIZE = float(_arg("--label-point-size", "16.0"))


def log(msg):
    print(msg, flush=True)


def image_time_of_day(image_name):
    import re
    m = re.search(r"utc-\d{8}_(\d{2})-(\d{2})-(\d{2})-\d+", image_name)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mi * 60 + s


def load_trajectory(bbox, margin=15.0):
    entries = json.load(open(CAMEXTR_PATH))["Profiler_0"]
    e_min, e_max = bbox["e_min"] - margin, bbox["e_max"] + margin
    n_min, n_max = bbox["n_min"] - margin, bbox["n_max"] + margin
    pts = []
    for e in entries:
        if e["SerialNr"] != DOWN_CAM_SERIAL:
            continue
        t = image_time_of_day(e["Image"]) or 0
        if PASS_FILTER == "2" and t < PASS_SPLIT_T:
            continue
        if PASS_FILTER == "1" and t >= PASS_SPLIT_T:
            continue
        x, y, z = e["Xyz"][:3]
        if not (e_min <= x <= e_max and n_min <= y <= n_max):
            continue
        pts.append((e["Time"], x, y, z))
    pts.sort(key=lambda p: p[0])
    xyz = np.array([[p[1], p[2], p[3]] for p in pts])
    log(f"  raw trajectory samples (down cam, pass filter={PASS_FILTER}, in-bbox): {len(xyz)}")
    return xyz


def smooth_path(xyz, window):
    if window <= 1 or len(xyz) < window:
        return xyz
    k = np.ones(window) / window
    pad = window // 2
    padded = np.pad(xyz, ((pad, pad), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(3)], axis=1)
    return out


def resample_evenly(xyz, spacing):
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(total // spacing), 2)
    targets = np.linspace(0, total, n)
    out = np.empty((n, 3))
    for i in range(3):
        out[:, i] = np.interp(targets, cum, xyz[:, i])
    return out, total


def build_frame_poses(path_xyz):
    n = len(path_xyz)
    poses = []
    for i in range(n):
        pos = path_xyz[i].copy()
        pos[2] += CAM_HEIGHT_M
        look_idx = min(i + int(LOOKAHEAD_M / RESAMPLE_SPACING_M), n - 1)
        target = path_xyz[look_idx].copy()
        target[2] += CAM_HEIGHT_M * 0.6
        if np.allclose(target, pos):
            continue
        poses.append((pos, target))
    return poses


def _load_pcd(labeled_laz, split_labels=False):
    log(f"  loading {labeled_laz} ...")
    las = laspy.read(labeled_laz)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / 65535.0
    log(f"  {len(xyz):,} points loaded")

    label_pcd = None
    if split_labels and LABEL_SPLIT_N is not None:
        label_xyz, label_rgb = xyz[LABEL_SPLIT_N:], rgb[LABEL_SPLIT_N:]
        xyz, rgb = xyz[:LABEL_SPLIT_N], rgb[:LABEL_SPLIT_N]
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_xyz)
        label_pcd.colors = o3d.utility.Vector3dVector(label_rgb)
        log(f"  split off {len(label_xyz):,} STVO label points to render oversized")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if VOXEL_SIZE_M > 0:
        log(f"  voxel-downsampling at {VOXEL_SIZE_M}m ...")
        pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
        log(f"  downsampled to {len(pcd.points):,} points")
    else:
        log("  no downsampling (full native density)")

    if split_labels:
        return pcd, label_pcd
    return pcd


def render_frames_hq(labeled_laz, poses, frames_dir):
    """Filament-based OffscreenRenderer path -- anti-aliased point shader,
    full native point density (no voxel downsample), used for --renderer hq.
    Uses camera.look_at()-driven setup_camera() (matched to the technique
    validated during STVO-tag close-up verification) instead of the legacy
    Visualizer's ViewControl.convert_from_pinhole_camera_parameters(), which
    was found to mis-render close-range geometry.

    STVO label points (last LABEL_SPLIT_N: of the file, when given) render
    as a separate geometry with a much larger point_size -- at normal
    driving-camera distance a 1.1x0.22m plate hovering just 0.35m above the
    real sign/light it tags projects to a small patch that lands right on
    top of similarly-toned real material (confirmed: the plate itself IS
    there and correctly positioned, it just visually blends into the real
    object next to it otherwise)."""
    pcd, label_pcd = _load_pcd(labeled_laz, split_labels=True)

    os.makedirs(frames_dir, exist_ok=True)
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = HQ_POINT_SIZE

    label_mat = rendering.MaterialRecord()
    label_mat.shader = "defaultUnlit"
    label_mat.point_size = LABEL_POINT_SIZE

    n_written = 0
    for i, (pos, target) in enumerate(poses):
        renderer = rendering.OffscreenRenderer(FRAME_W, FRAME_H)
        renderer.scene.set_background([0.53, 0.62, 0.68, 1.0])
        renderer.scene.view.set_antialiasing(True)
        renderer.scene.add_geometry("cloud", pcd, mat)
        if label_pcd is not None:
            renderer.scene.add_geometry("labels", label_pcd, label_mat)
        renderer.setup_camera(HQ_VERTICAL_FOV_DEG, target, pos, [0, 0, 1])
        img = renderer.render_to_image()
        out_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        o3d.io.write_image(out_path, img)
        del renderer
        n_written += 1
        if n_written % 25 == 0:
            log(f"    rendered {n_written}/{len(poses)} frames")

    log(f"  {n_written} frames written to {frames_dir}")
    return n_written


def render_frames(labeled_laz, poses, frames_dir):
    pcd = _load_pcd(labeled_laz)

    os.makedirs(frames_dir, exist_ok=True)
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(width=FRAME_W, height=FRAME_H, visible=False)
    if not ok:
        raise RuntimeError("Open3D offscreen window creation failed")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.53, 0.62, 0.68])

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    intr = cam_params.intrinsic
    fx = fy = 0.9 * FRAME_W
    intr.set_intrinsics(FRAME_W, FRAME_H, fx, fy, FRAME_W / 2 - 0.5, FRAME_H / 2 - 0.5)

    up = np.array([0.0, 0.0, 1.0])
    n_written = 0
    for i, (pos, target) in enumerate(poses):
        forward = target - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.stack([right, -true_up, forward], axis=0)
        t = -R @ pos
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intr
        params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        out_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        vis.capture_screen_image(out_path, do_render=True)
        n_written += 1
        if n_written % 100 == 0:
            log(f"    rendered {n_written}/{len(poses)} frames")

    vis.destroy_window()
    log(f"  {n_written} frames written to {frames_dir}")
    return n_written


BLANK_STD_THRESHOLD = 3.0   # a frame that's essentially flat background color
BLANK_TAIL_RUN = 5          # this many consecutive near-blank frames -> real tail, trim from there


def _trim_trailing_blank(frame_files):
    """The resampled trajectory can run slightly past the real point cloud's
    covered extent (CamExtr's own margin isn't the same as the tile's actual
    data footprint) -- confirmed twice now: 61/752 corridor frames and 51/237
    part15 frames faded to solid background color once the virtual camera
    left real coverage. Detect and drop that trailing run instead of shipping
    a blank tail.

    Must measure std PER CHANNEL, spatially -- img.std() across all 3 RGB
    channels together has a nonzero floor (~15-16 here) even for a perfectly
    flat frame, because the background color's R/G/B values differ from each
    other; that floor sat above the original 3.0 threshold and silently
    defeated this check on its first real (part15) run. The max of each
    channel's own spatial std is ~0 for a truly flat frame regardless of
    what color that flat frame happens to be."""
    first_blank = None
    run = 0
    for i, fp in enumerate(frame_files):
        img = cv2.imread(fp)
        spatial_std = img.reshape(-1, img.shape[2]).std(axis=0).max()
        if spatial_std < BLANK_STD_THRESHOLD:
            run += 1
            if run >= BLANK_TAIL_RUN and first_blank is None:
                first_blank = i - run + 1
        else:
            run = 0
            first_blank = None
    if first_blank is not None:
        log(f"  trimming trailing blank run: keeping {first_blank}/{len(frame_files)} frames "
            f"(camera path ran past the cloud's real coverage after that point)")
        return frame_files[:first_blank]
    return frame_files


def assemble_video(frames_dir, out_mp4, fps):
    log("  assembling MP4 with OpenCV VideoWriter (no system ffmpeg needed)...")
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        raise RuntimeError("no frames found to assemble")
    frame_files = _trim_trailing_blank(frame_files)
    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open for writing")
    for fp in frame_files:
        img = cv2.imread(fp)
        vw.write(img)
    vw.release()
    log(f"  {len(frame_files)} frames encoded -> saved: {out_mp4}")


if __name__ == "__main__":
    log("[1/4] Loading labeled corridor cloud bbox...")
    with laspy.open(LABELED_LAZ) as f:
        mins, maxs = f.header.mins, f.header.maxs
    bbox = {"e_min": mins[0], "e_max": maxs[0], "n_min": mins[1], "n_max": maxs[1]}
    log(f"  corridor bbox: E[{bbox['e_min']:.1f},{bbox['e_max']:.1f}] N[{bbox['n_min']:.1f},{bbox['n_max']:.1f}]")

    log("[2/4] Extracting + smoothing + resampling real vehicle trajectory...")
    raw_xyz = load_trajectory(bbox)
    smoothed = smooth_path(raw_xyz, SMOOTH_WINDOW)
    path_xyz, total_len = resample_evenly(smoothed, RESAMPLE_SPACING_M)
    log(f"  path length: {total_len:.1f}m, {len(path_xyz)} resampled points @ {RESAMPLE_SPACING_M}m spacing")

    poses = build_frame_poses(path_xyz)
    log(f"  {len(poses)} camera poses -> ~{len(poses)/FPS:.1f}s of video @ {FPS}fps")

    log(f"[3/4] Rendering offscreen frames (Open3D, renderer={RENDERER})...")
    n_frames = render_frames_hq(LABELED_LAZ, poses, FRAMES_DIR) if RENDERER == "hq" \
        else render_frames(LABELED_LAZ, poses, FRAMES_DIR)

    log("[4/4] Encoding video...")
    assemble_video(FRAMES_DIR, OUT_MP4, FPS)

    log(f"\nDone. Drive-through video: {OUT_MP4}")
