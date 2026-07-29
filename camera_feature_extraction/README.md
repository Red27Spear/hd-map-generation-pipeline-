# Camera feature extraction

Detects road signs and traffic lights in the corrected camera frames, classifies signs into 76 StVO (German traffic sign) classes, and localises each detection in 3D by combining the monocular bearing with real LiDAR point density along that bearing.

## What it does

1. Runs a YOLO sign detector and a YOLO traffic-light detector (COCO class 9) over every frame in the tile.
2. Classifies each sign crop into one of 76 StVO classes with a dedicated classifier, including a second pass that de-rotates the crop using the camera roll measured from nearby triangular (warning) signs.
3. Converts each detection's image position into a bearing ray from the camera's UTM position, using the frame's heading.
4. Walks that ray through the LiDAR point cloud and places the object at the distance where real point density peaks, instead of trusting a monocular size-based distance guess alone.
5. Deduplicates detections seen from multiple frames, and triangulates a better 2D position from multiple angularly-distinct viewpoints when available.
6. Optionally keeps detections off the drivable road/median corridor, using white-line and median-strip evidence from a colorized point cloud if one is available.

## Input

| Input | Format | Where the demo data is |
|---|---|---|
| Corrected camera frames + `pinhole_K.json` | output of `camera_preprocessing` | `../camera_preprocessing/output/` (run that stage first) |
| LiDAR tile | `.laz` | `../data/part15/raw/*.laz` |
| Camera poses | `CamExtr.json` | `../config/CamExtr.json` |
| Colorized point cloud (optional, for the road no-touch zone) | `.laz` | `../data/part15/results/colorized_part15.laz` |
| Sign detector weights | YOLO `.pt` | not bundled, see below |
| 76-class sign classifier weights | YOLO-classification `.pt` | not bundled, see below |

### Trained weights

This repo doesn't include the trained sign detector or the 76-class classifier: they're several hundred MB combined and specific to one training run. `YOLO_LIGHT_MODEL` uses the stock `yolov8s.pt` COCO weights, which `ultralytics` downloads automatically. For the sign detector and classifier, either train your own (the detector was trained on GTSDB + Mapillary Traffic Sign Dataset; the classifier on the same 76-class label set defined in this script's `CLASS_76_NAMES`) or point `CLASSIFIER_MODEL` / `YOLO_SIGN_MODEL` at any compatible YOLO detection/classification pair via environment variables:

```bash
export YOLO_SIGN_MODEL=/path/to/your/sign_detector.pt
export CLASSIFIER_MODEL=/path/to/your/76class_classifier.pt
```

## Output

Written to `--output-dir` equivalent (`PHASE3_OUTPUT_DIR` env var, default `./output`):

- `signs_3d_<part>.geojson`, `traffic_lights_3d_<part>.geojson`, `poles_3d_<part>.geojson`: one GeoJSON `Point` feature per localised object, in UTM, with confidence/uncertainty fields (`lidar_confidence`, `position_uncertainty_m`, `classifier_conf`, `source_method` = `lidar_verified` or `camera_only_unverified`).
- `phase3_<part>_objects.laz`: the input point cloud with a visualisation classification field (road / verified sign / verified light / camera-only / back-facing / pole variants), color-coded.
- `detection_proof_<part>.html`: a gallery with one crop image per detection, for visually auditing results.
- `phase3_<part>_report.txt`: detection counts and processing stats.
- `output/detection_crops/`: the individual cropped detection images referenced by the gallery.

## Usage

```bash
export PHASE0_DIR=../camera_preprocessing/output
export CAMEXTR_PATH=../config/CamExtr.json
python detect_and_localise_signs.py ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz
```

## Known limitation

The full LiDAR-to-camera pixel projection (needed to cross-verify a detection's exact 3D position against the point cloud pixel-for-pixel) is unresolved in this project. Localisation instead uses only the frame's heading to build a bearing ray, then finds where real LiDAR density peaks along it. That's enough to place objects close to their true position and reject noise, but it's not a full projective solve.
