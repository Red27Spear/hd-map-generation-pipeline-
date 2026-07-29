# LiDAR pre-processing

Reduces a raw LiDAR tile down to exactly what an HD map needs: the road surface and vertical infrastructure (poles, signs, lights, gantries). Buildings, trees, parked cars, and unclustered noise are all dropped.

## What it does

1. RANSAC ground-plane segmentation to separate road surface from everything else.
2. DBSCAN clustering of the non-ground points.
3. Shape classification of each cluster by height, width, verticality (PCA-based), and elongation, against separate thresholds for poles, signs, lights, and gantries.
4. Distance-to-road filtering: a cluster that passes the shape check but sits far from the road (tree-branch fragments and building edges can coincidentally pass shape thresholds) is dropped.

A cluster only survives if its shape passes classification. Earlier iterations of this step let any unclustered point in a plausible height range through, which pulled in a diffuse haze of tree and building-edge noise on both sides of the road. That bypass is removed here.

## Input

| Input | Format |
|---|---|
| LiDAR tile | `.laz` |

Demo input: `../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz`

## Output

Written to `--output-dir` (default `./output`):

- `clean_<part>.laz`: the filtered point cloud, with a `classification` field: `1` = road surface, `2` = vertical infrastructure. Points that are neither are removed entirely.
- `phase2_clean_<part>_report.txt`: point counts (input, road, vertical, dropped) and the shape-classification decision for every cluster with more than 20 points.

## Usage

```bash
python clean_and_classify.py ../data/part15/raw/9020C-0140_08.05.2026_08.10.22_MoRo_Bonn.all.part15.laz --output-dir ./output
```

## Note on scope

A separate, more advanced ground segmentation using the Patchwork++ algorithm (rather than a single global RANSAC plane) was also developed for this project, refined further with a KML road-corridor + DBSCAN pole-candidate filter. It isn't included here: it was developed and only ever run against a collaborator's local dataset copy, so there's no output from it on this machine to demonstrate, and it isn't runnable as-is without that data. The RANSAC-based approach here is simpler but fully self-contained and runnable against the bundled `part15` tile.
