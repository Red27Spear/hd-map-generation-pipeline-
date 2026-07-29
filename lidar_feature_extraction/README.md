# LiDAR feature extraction

Two scripts: one extracts lane markings from LiDAR intensity and assembles a Lanelet2/OSM-XML map (folding in the signs, lights, and poles from `camera_feature_extraction`); the other renders a semantic, StVO-styled visualisation of the same detections directly into a point cloud.

## `lane_markings_and_osm_export.py`

1. RANSAC road segmentation (same approach as `lidar_preprocessing`).
2. Takes the top 7% of road points by LiDAR intensity as candidate paint markings, clusters them with DBSCAN, and classifies each cluster as a dashed line, solid line, stop line, or symbol (arrows/cycle icons are dropped) by its length and width.
3. Fits an ordered, simplified (Douglas-Peucker) polyline through each marking cluster.
4. Converts everything to WGS84 and assembles a Lanelet2-compatible OSM-XML file: lane markings as ways, signs/lights/poles as nodes with German StVO tags (`DE:xxx`) and non-standard `hd:` extension tags carrying the uncertainty fields from `camera_feature_extraction`.
5. Optionally queries the public Overpass API for existing OSM traffic signals/signs in the tile's bounding box, for comparison (network access required; falls back gracefully if the request fails).

### Input

| Input | Format | Where the demo data is |
|---|---|---|
| LiDAR tile (ideally pre-cleaned) | `.laz` | `../data/part15/raw/*.laz`, or the `lidar_preprocessing` output |
| Sign/light/pole detections | GeoJSON, output of `camera_feature_extraction` | `--phase3-dir ../camera_feature_extraction/output` |

### Output

Written to `--output-dir` (default `./output`):

- `lane_markings_<part>.geojson`: one `LineString` feature per marking, with `marking_type`, `length_m`, `width_m`, `confidence`.
- `hd_map_<part>.osm`: the assembled Lanelet2/OSM-XML map.
- `phase5_<part>_report.txt`: counts and elapsed time.

### Usage

```bash
python lane_markings_and_osm_export.py ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz \
    --phase3-dir ../camera_feature_extraction/output \
    --output-dir ./output
```

## `semantic_classification.py`

Renders every localised sign/light/pole as a StVO-styled billboard sprite directly into the point cloud, color-coded by verification status (LiDAR-verified vs. camera-only-unverified, likely back-facing, bare pole vs. pole adjacent to a detection vs. probable unclassified sign).

### Input

Same LiDAR tile, plus the corrected camera frames (`camera_preprocessing` output), the detections (`camera_feature_extraction` output), and `CamExtr.json`. Paths are read from environment variables (`PHASE0_DIR`, `PHASE3_DIR`, `CAMEXTR_PATH`), each with a relative default pointing at the sibling section directories.

### Output

`<output-dir>/semantic_<part>.laz`: the point cloud with StVO-colored billboard geometry inserted at each detection's 3D position.

### Usage

```bash
python semantic_classification.py ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz
```
