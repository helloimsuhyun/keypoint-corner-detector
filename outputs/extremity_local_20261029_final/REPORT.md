# Local extremity experiment results

CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; BLAS threads: 1; seeds: 20.
OpenMP library thread counts (which may differ from BLAS) are recorded in metadata.json.

CED settings and 10 mm voxel preprocessing use the existing DetectorConfig defaults unchanged.
Voting uses original noisy XYZ, not voxel centroids. All times below exclude generation, I/O and visualization.
GT-near queries isolate local behavior; CED queries measure actual region coverage. These are separate experiments.

| Scene (base) | CED recall | CED ms | C local recall (GT-near ROI) | C end-to-end recall | C extra ms | D end-to-end recall |
|---|---:|---:|---:|---:|---:|---:|
| one_face | 1.000 | 7.99 | 1.000 | 1.000 | 2.32 | 1.000 |
| two_faces | 0.667 | 15.26 | 1.000 | 0.667 | 2.29 | 0.667 |
| three_faces | 0.571 | 23.03 | 1.000 | 0.571 | 2.82 | 0.571 |
| same_object_two_corners | 0.571 | 11.69 | 1.000 | 1.000 | 2.78 | 1.000 |
| separated_objects | 0.214 | 19.89 | 1.000 | 0.214 | 2.06 | 0.214 |
| unequal_sizes | 0.214 | 16.11 | 1.000 | 0.500 | 2.20 | 0.500 |
| gap_5 | 0.214 | 20.01 | 1.000 | 0.214 | 1.88 | 0.214 |
| gap_10 | 0.214 | 19.90 | 1.000 | 0.214 | 1.94 | 0.214 |
| gap_20 | 0.214 | 19.71 | 1.000 | 0.214 | 1.89 | 0.214 |
| touching_objects | 0.273 | 18.15 | 0.000 | 0.273 | 2.08 | 0.273 |
| plane_interior | 0.463 | 31.45 | nan | 1.000 | 2.54 | 1.000 |
| straight_boundary | 0.762 | 14.98 | nan | 1.000 | 7.55 | 1.000 |
| concave_notch | 0.000 | 17.09 | 0.000 | 0.417 | 2.04 | 0.417 |
| box_ground | 0.186 | 33.80 | 0.000 | 0.364 | 2.60 | 0.364 |
| hidden_small_object | 0.067 | 24.01 | 1.000 | 0.067 | 1.39 | 0.067 |
| partial_rgbd | 0.257 | 22.95 | 1.000 | 0.429 | 2.04 | 0.429 |

## A-D ablation: actual CED queries

Macro averages across scene/seed observations. Errors average matched GT only; they do not penalize misses.

| Variant | Recall | FP / scene | Matched error (mm) | Extra time, mean / median / P95 (ms) |
|---|---:|---:|---:|---|
| A | 0.5349 | 16.631 | 0.598 | 1.270 / 1.258 / 1.689 |
| B | 0.4911 | 0.013 | 0.311 | 1.243 / 1.203 / 1.632 |
| C | 0.5090 | 0.019 | 0.311 | 2.527 / 2.166 / 2.836 |
| D | 0.5090 | 0.019 | 0.311 | 2.373 / 2.331 / 3.142 |
| C_oracle | 0.5090 | 0.013 | 0.311 | 1.285 / 1.277 / 1.689 |
| D_oracle | 0.5090 | 0.013 | 0.311 | 1.534 / 1.540 / 2.059 |

## GT-near queries: isolate local behavior

These fixed query targets are defined before detection; they are not added to the real CED pipeline.

| Scene | B recall in ROI | C recall in ROI | C candidates | C FP | C error (mm) | C oracle recall |
|---|---:|---:|---:|---:|---:|---:|
| one_face | 1.000 | 1.000 | 1.00 | 0.00 | 0.206 | 1.000 |
| two_faces | 1.000 | 1.000 | 1.00 | 0.00 | 0.206 | 1.000 |
| three_faces | 1.000 | 1.000 | 1.00 | 0.00 | 0.206 | 1.000 |
| same_object_two_corners | 1.000 | 1.000 | 2.00 | 0.00 | 0.190 | 1.000 |
| separated_objects | 0.000 | 1.000 | 2.00 | 0.00 | 0.203 | 1.000 |
| unequal_sizes | 0.333 | 1.000 | 3.00 | 0.00 | 0.203 | 1.000 |
| gap_5 | 0.000 | 1.000 | 2.00 | 0.00 | 0.203 | 1.000 |
| gap_10 | 0.000 | 1.000 | 2.00 | 0.00 | 0.203 | 1.000 |
| gap_20 | 0.000 | 1.000 | 2.00 | 0.00 | 0.203 | 1.000 |
| touching_objects | 0.000 | 0.000 | 0.00 | 0.00 | nan | 1.000 |
| plane_interior | nan | nan | 0.00 | 0.00 | nan | nan |
| straight_boundary | nan | nan | 0.00 | 0.00 | nan | nan |
| concave_notch | 0.000 | 0.000 | 0.00 | 0.00 | nan | 0.000 |
| box_ground | 0.000 | 0.000 | 0.00 | 0.00 | nan | 1.000 |
| hidden_small_object | 0.000 | 1.000 | 7.00 | 0.00 | 0.204 | 1.000 |
| partial_rgbd | 1.000 | 1.000 | 1.10 | 0.10 | 1.251 | 1.000 |

## Noise and density stress

| Condition | C CED recall | C FP / scene | Oracle C FP / scene | C extra mean / median / P95 (ms) |
|---|---:|---:|---:|---|
| base | 0.5090 | 0.019 | 0.013 | 2.527 / 2.166 / 2.836 |
| noisy | 0.5039 | 0.122 | 0.087 | 2.253 / 2.225 / 2.990 |
| sparse | 0.5385 | 115.141 | 0.206 | 5.559 / 5.693 / 8.755 |

## CED comparison and coverage

Base CED time, mean / median / P95: 19.751 / 19.684 / 33.560 ms.
Base CED matched error: 2.951 mm. C matched error: 0.311 mm.
The matched sets differ. Use paired columns for a fair shared-GT localization comparison:
paired CED error = 2.951 mm; paired refined error = 0.310 mm (macro means over rows with shared matches).
Base CED query coverage recall = 0.5043; C conditional recall within covered GT = 0.9911.
Coverage tests literal GT coordinates inside the core. The 5 mm matching tolerance can credit a nearby GT just outside it, so this coverage number is not a strict upper bound on tolerance-based recall.
CED radius=80 mm, PCA radius=120 mm and NMS=50 mm are the unmodified project defaults; several synthetic objects are smaller than these kernels.
Thus this benchmark is not an optimized CED comparison or evidence of universal gains on sensor data.

## Direct answers and failure analysis

1. One, two and three observed faces use exactly the same 3-D voting code. Read their GT-near rows for local success and CED rows for upstream coverage limits.
2. The same-object-two-corners scene returns both 18 mm-separated vertices from one query. No Top-1 restriction or coordinate averaging is applied.
3. Component separation helps disconnected facing corners and the observed small object inside a larger shell. Compare B, C and oracle C in the GT-near table.
4. Padding suppresses artificial ball-cut extrema. Plane-interior and straight-boundary focus ROIs have no GT; their local FP counts, not scene recall at the distant finite patch corners, measure this failure mode.
5. Touching-object and box-ground components merge, leaving inner vertices hidden from supporting directions. Oracle separation is a diagnostic upper bound, not an automatic fix.
6. Concave notch origin lies inside the local convex hull and has no stable maximizing direction. Noise may create nearby outliers, not evidence for reliable concave detection.
7. Fixed 4.5 mm connectivity severely fragments the 6 mm sampling condition. Singleton fragments vote for themselves; higher recall with very large FP counts is not a success.
8. D does not improve base end-to-end recall over C on these scenes. Component separation already restores the tested disconnected small objects; multi-scale adds work and can dilute a small-scale-only peak.
9. An actual CED query may miss the small-object region even when its GT-near query succeeds. A local stage cannot recover GT outside all supplied balls.
10. The partial-RGB-D case uses one pinhole ray-cast depth view. GT is restricted to visible vertices with samples within 5 mm; the shell case is explicitly a multi-view/observed-interior stress test.

## Interpretation and limits

This stage adds runtime to CED; it cannot be described as a faster replacement detector.
Missing CED coverage cannot be repaired outside the supplied query balls. See coverage_recall and conditional_recall.
Each component receives its own direction sweep, including components not containing q. This can recover separate small objects.
Padding discards support-function maxima on the artificial core-ball boundary. It cannot remove real sampling/noise extrema.
NMS retains multiple peaks and never averages their positions. Equal-score neighbors use smaller raw indices.
Concave vertices inside a local convex hull have no supporting direction and receive no stable votes.
Touching objects and the ground can connect; sparse sampling fragments objects into small components, producing confident false positives.
A shell can hide interior observed vertices from support directions; small radii and oracle/component separation can help only when there is observed support.
Unobserved vertices cannot be recovered. XYZ alone also cannot distinguish convex/concave solid orientation when the observed surfaces coincide.
Oracle labels are an ideal separation reference, not a deployable segmenter. Connected components do not guarantee object instances.
Multi-scale votes are divided by directions times eligible scale count; small-scale peaks may be diluted by large-scale nonselection.
The raw IDs are deduplicated exactly for scene evaluation. Nearby hypotheses from different queries remain separate and can count as false positives.

Per-scene means, medians and P95 are in summary.csv; per-query times in query_summary.csv; baseline in ced_summary.csv.
A repeatability score is spatial agreement with seed 0, not proof of semantic correctness. Misses and matched-only errors must be read together.
No STL GUI, RealSense, or Kinect hardware run is claimed. Partial RGB-D is a synthetic pinhole view.
