# Local extremity experiment results

CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; BLAS threads: 1; seeds: 20.

CED settings and 10 mm voxel preprocessing use the existing DetectorConfig defaults unchanged.
Voting uses original noisy XYZ, not voxel centroids. All times below exclude generation, I/O and visualization.
GT-near queries isolate local behavior; CED queries measure actual region coverage. These are separate experiments.

| Scene (base) | CED recall | CED ms | C local recall (GT-near ROI) | C end-to-end recall | C extra ms | D end-to-end recall |
|---|---:|---:|---:|---:|---:|---:|
| one_face | 1.000 | 9.06 | 1.000 | 1.000 | 2.55 | 1.000 |
| two_faces | 0.667 | 15.56 | 1.000 | 0.667 | 2.38 | 0.667 |
| three_faces | 0.571 | 22.96 | 1.000 | 0.571 | 2.77 | 0.571 |
| same_object_two_corners | 0.571 | 12.16 | 1.000 | 1.000 | 2.84 | 1.000 |
| separated_objects | 0.214 | 19.89 | 1.000 | 0.214 | 2.03 | 0.214 |
| unequal_sizes | 0.214 | 16.01 | 1.000 | 0.500 | 2.20 | 0.500 |
| gap_5 | 0.214 | 20.12 | 1.000 | 0.214 | 1.98 | 0.214 |
| gap_10 | 0.214 | 19.81 | 1.000 | 0.214 | 2.05 | 0.214 |
| gap_20 | 0.214 | 19.78 | 1.000 | 0.214 | 1.93 | 0.214 |
| touching_objects | 0.273 | 18.05 | 0.000 | 0.273 | 2.10 | 0.273 |
| plane_interior | 0.463 | 31.27 | nan | 1.000 | 2.59 | 1.000 |
| straight_boundary | 0.762 | 14.93 | nan | 1.000 | 2.28 | 1.000 |
| concave_notch | 0.000 | 17.13 | 0.000 | 0.417 | 2.02 | 0.417 |
| box_ground | 0.186 | 33.84 | 0.000 | 0.364 | 2.66 | 0.364 |
| hidden_small_object | 0.067 | 21.87 | 1.000 | 0.067 | 1.23 | 0.067 |
| partial_rgbd | 0.257 | 22.56 | 1.000 | 0.429 | 1.93 | 0.429 |

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
