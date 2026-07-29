# RGB and point cloud fusion

Colorizes a LiDAR point cloud by projecting it into the fisheye camera frames and sampling RGB. Two parts: an 8-stage calibration-to-projection chain, and a colorization script that consumes stage 1's refined calibration to actually produce a colored `.laz`.

## The 8-stage chain

| Stage | Script | Does |
|---|---|---|
| 1 | `stage1_charuco_fisheye_calib.py` | ChArUco-board fisheye intrinsic/extrinsic calibration |
| 2 | `stage2_time_sync.py` | aligns camera frame timestamps with LiDAR GPS time |
| 3 | `stage3_trajectory_interp.py` | interpolates the vehicle trajectory to per-frame poses |
| 4 | `stage4_pointcloud_registration.py` | registers LiDAR scans into a common frame |
| 5 | `stage5_lidar_to_image_projection.py` | projects LiDAR points into each camera image |
| 6 | `stage6_occlusion_handling.py` | Z-buffered occlusion handling so occluded points don't get the wrong color |
| 7 | `stage7_rgb_extraction.py` | samples RGB at each point's projected pixel |
| 8 | `stage8_multi_camera_blend.py` | blends RGB across multiple cameras that see the same point |

Each stage is a standalone CLI script (`--help` for its exact arguments) and can be run independently; run `--help` on each in order for the full argument list.

## `colorize_multi_camera.py`

The script that was actually used to produce this repo's demo output. It colorizes a LAZ tile using three cameras (left, right, down) by default, applying a fitted per-camera pose correction on top of the raw `CamExtr.json` pose.

The up-facing camera (serial `4108690914`) is excluded by default: `CamExtr.json` has a pose-swap bug between the down and up camera entries (worked around in code, see the script's docstring), and the up camera's own fitted correction was likely compensating for that same bug rather than reflecting a real calibration, so it isn't trustworthy.

### Input

| Input | Format | Where the demo data is |
|---|---|---|
| LiDAR tile | `.laz` | `../data/part15/raw/*.laz` |
| Camera poses | `CamExtr.json` | `../config/CamExtr.json` |
| Camera intrinsics | XML | `../config/9020C_0140_toScanner_final.xml` |
| Fitted per-camera pose correction | `refined_calibration.json` | bundled in this directory |
| Raw camera frames | `.jpg` | `../data/part15/images/` (demo subset; only 10 frames, so a full re-colorization from this repo alone will be sparse) |

### Output

A single colorized `.laz`, written to `--output`.

### Usage

```bash
python colorize_multi_camera.py \
    --laz ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz \
    --camextr ../config/CamExtr.json \
    --intrinsics-xml ../config/9020C_0140_toScanner_final.xml \
    --refined-calibration ./refined_calibration.json \
    --image-dir ../data/part15/images \
    --output ./output/colorized_part15.laz \
    --cameras 4108690955 4108736902 4108736907
```

`../data/part15/results/colorized_part15.laz` is this script's output run against the full frame set for the tile (not just the 10-image demo subset bundled here), so re-running against the bundled 10 images alone will produce a much sparser result.

## Known limitation

Reprojection error against manually-picked ground-truth correspondences is still large: roughly 700-1150 px mean error on a 2448x2048 image (measured on cameras `4108690955`/`4108736902`/`4108690914`), down from about 1300-1570 px on the uncorrected nominal pose, but far from pixel-accurate. In practice this means fine features like signs and poles get visibly smeared or offset color, while broad regions (road surface, vegetation, building facades) come out qualitatively plausible: colored points form a continuous strip following the road corridor rather than scattered noise, but individual edges won't line up precisely.
