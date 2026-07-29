# Camera pre-processing

Fixes three problems in the raw fisheye camera frames before anything downstream touches them: barrel distortion, underexposure, and an olive/green color cast.

## What it does

1. **Undistortion**: Kannala-Brandt fisheye model to pinhole, using per-camera intrinsics from the calibration XML.
2. **Gray-world auto white balance**: computed on "clean" pixels only (excludes blown-out and near-black pixels, so bright sky and shadow don't skew the color estimate).
3. **Olive/green desaturation**: the raw frames carry a green-yellow cast in a specific hue band; this pass desaturates just that band.
4. **CLAHE**: contrast-limited adaptive histogram equalization on the luminance channel, to lift underexposed frames.

## Input

| Input | Format | Where the demo data is |
|---|---|---|
| Raw fisheye frames | `.jpg`, named `<survey>{utc-...;sn-<serial>;...}.jpg` | `../data/part15/images/` (10 representative frames; the full raw set for this tile is not bundled) |
| LiDAR tile | `.laz` | `../data/part15/raw/*.laz` (only the header is read, for the tile's UTM bounding box) |
| Camera poses | `CamExtr.json`: one entry per frame, each with `Image` (filename), `Xyz` (UTM easting/northing/Z), `Hrp` (heading/roll/pitch), `SerialNr` | `../config/CamExtr.json` |
| Camera intrinsics | XML, one `<Camera serialno=".." name="..">` block per unit with `fx, fy, cx, cy, k1..k4, image_w, image_h` | `../config/9020C_0140_toScanner_final.xml` |

## Output

Written to `--output-dir` (default `./output`):

- `<original_filename>.jpg`: undistorted, color-corrected frame, same filename as the input.
- `pinhole_K.json`: the new pinhole camera matrix per camera serial, keyed by serial number. Consumed by `camera_feature_extraction` for bearing-angle calculations.
- `report.txt`: per-camera frame counts and processing stats.

## Usage

```bash
python undistort_and_correct.py \
    --laz ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz \
    --image-dir ../data/part15/images \
    --camextr ../config/CamExtr.json \
    --intrinsics ../config/9020C_0140_toScanner_final.xml \
    --output-dir ./output
```

## Known limitation

The white-balance and olive-desaturation steps assume the color cast is roughly uniform across the frame. On frames with a lot of bright sky, this pushes the sky toward magenta/pink instead of neutral white. It's visible on the sky region in several corrected frames from this tile. The fix likely needs excluding sky pixels from the gray-world estimate (e.g. via a brightness+position heuristic) rather than relying on the blown-highlight threshold alone, which doesn't catch a bright-but-not-blown sky.
