# Quantile soft voting: measured comparison

AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; 30 seeds per scene/noise; 7 scenes; 6 noise levels.
Full CPU/thread/config provenance: metadata.json. Timings exclude data generation, matching, I/O and rendering.
CED/PCA/NMS defaults are unchanged. Local study fixes r=30, padding=10, cluster=4.5, NMS=5 mm, tau=0.25 mm, 256 directions and score gate=0.
The user's live ROI settings are not overwritten. Every variant sees identical raw XYZ and unchanged CED queries; GT-near queries are a separate diagnostic.

## Aggregate metrics

Macro means over scenes/noise. Error values condition on successful 1:1 matches within 5 mm; read recall and FP alongside errors.

| Source | Variant | Recall | FP/frame | Matched median mm | Matched P95 mm | Mean query ms | Mean frame ms |
|---|---|---:|---:|---:|---:|---:|---:|
| gt_near | A_hard_f64 | 0.870 | 3.22 | 0.505 | 0.936 | 1.264 | 1.519 |
| gt_near | B_dense_f64 | 0.883 | 17.28 | 0.672 | 0.999 | 5.285 | 5.543 |
| gt_near | C_partition_f64 | 0.883 | 17.28 | 0.672 | 0.999 | 4.920 | 5.178 |
| gt_near | D_sparse4_f64 | 0.883 | 13.55 | 0.672 | 0.999 | 3.132 | 3.388 |
| gt_near | D_sparse6_f64 | 0.883 | 15.74 | 0.672 | 0.999 | 3.743 | 4.000 |
| gt_near | D_sparse8_f64 | 0.883 | 16.88 | 0.672 | 0.999 | 4.175 | 4.432 |
| gt_near | A_hard_f32 | 0.870 | 3.22 | 0.505 | 0.936 | 0.999 | 1.253 |
| gt_near | B_dense_f32 | 0.883 | 17.28 | 0.672 | 0.999 | 5.127 | 5.385 |
| gt_near | C_partition_f32 | 0.883 | 17.28 | 0.672 | 0.999 | 4.781 | 5.038 |
| gt_near | D_sparse4_f32 | 0.883 | 13.55 | 0.672 | 0.999 | 3.077 | 3.333 |
| gt_near | D_sparse6_f32 | 0.883 | 15.74 | 0.672 | 0.999 | 3.668 | 3.924 |
| gt_near | D_sparse8_f32 | 0.883 | 16.88 | 0.672 | 0.999 | 4.105 | 4.361 |
| ced | A_hard_f64 | 0.631 | 7.11 | 0.303 | 0.855 | 0.666 | 2.748 |
| ced | B_dense_f64 | 0.638 | 26.73 | 0.306 | 0.971 | 2.418 | 9.225 |
| ced | C_partition_f64 | 0.638 | 26.73 | 0.306 | 0.971 | 2.206 | 8.441 |
| ced | D_sparse4_f64 | 0.638 | 22.85 | 0.306 | 0.971 | 1.551 | 6.036 |
| ced | D_sparse6_f64 | 0.638 | 25.52 | 0.306 | 0.971 | 1.830 | 7.068 |
| ced | D_sparse8_f64 | 0.638 | 26.87 | 0.306 | 0.971 | 2.043 | 7.841 |
| ced | A_hard_f32 | 0.631 | 7.11 | 0.303 | 0.855 | 0.596 | 2.489 |
| ced | B_dense_f32 | 0.638 | 26.73 | 0.306 | 0.971 | 2.352 | 8.983 |
| ced | C_partition_f32 | 0.638 | 26.73 | 0.306 | 0.971 | 2.147 | 8.222 |
| ced | D_sparse4_f32 | 0.638 | 22.85 | 0.306 | 0.971 | 1.525 | 5.940 |
| ced | D_sparse6_f32 | 0.638 | 25.52 | 0.306 | 0.971 | 1.803 | 6.967 |
| ced | D_sparse8_f32 | 0.638 | 26.87 | 0.306 | 0.971 | 1.994 | 7.667 |

## Stage timing (actual CED queries)

| Variant | Projection | Quantile | Sigmoid + support | Normalize | Accumulate | NMS | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_hard_f64 | 0.034 | 0.000 | 0.000 | 0.000 | 0.015 | 0.102 | 0.666 |
| B_dense_f64 | 0.059 | 0.419 | 0.254 | 0.044 | 0.031 | 1.170 | 2.418 |
| C_partition_f64 | 0.056 | 0.219 | 0.247 | 0.042 | 0.030 | 1.168 | 2.206 |
| D_sparse4_f64 | 0.056 | 0.218 | 0.135 | 0.017 | 0.024 | 0.668 | 1.551 |
| D_sparse6_f64 | 0.056 | 0.218 | 0.141 | 0.018 | 0.024 | 0.936 | 1.830 |
| D_sparse8_f64 | 0.057 | 0.218 | 0.148 | 0.020 | 0.025 | 1.114 | 2.043 |
| A_hard_f32 | 0.028 | 0.000 | 0.000 | 0.000 | 0.015 | 0.101 | 0.596 |
| B_dense_f32 | 0.049 | 0.415 | 0.219 | 0.038 | 0.028 | 1.167 | 2.352 |
| C_partition_f32 | 0.045 | 0.214 | 0.214 | 0.036 | 0.027 | 1.167 | 2.147 |
| D_sparse4_f32 | 0.046 | 0.213 | 0.126 | 0.019 | 0.025 | 0.667 | 1.525 |
| D_sparse6_f32 | 0.046 | 0.213 | 0.131 | 0.021 | 0.025 | 0.935 | 1.803 |
| D_sparse8_f32 | 0.046 | 0.213 | 0.137 | 0.023 | 0.026 | 1.111 | 1.994 |

All table times are ms. ROI, clustering and shared spatial-index preparation are separate CSV columns.

## Exact quantile and sparse approximation

Maximum checked threshold difference: 0 mm (seed 0, every scene/noise/dtype).
The optimized quantile partitions two ranks and retains NumPy's stable linear interpolation, rather than substituting a single order statistic.
Installed NumPy 1.21.5 already implements np.quantile with partition. A speed gain here is wrapper/layout overhead reduction, not a change from full sort to linear selection.

| Variant (GT-near) | Vote L1 vs dense | Candidate Jaccard vs dense | Sigmoid support fraction |
|---|---:|---:|---:|
| A_hard_f64 | 68.5681 | 0.1894 | nan |
| B_dense_f64 | 0 | 1.0000 | 1.0000 |
| C_partition_f64 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f64 | 0.488904 | 0.7599 | 0.0256 |
| D_sparse6_f64 | 0.0832957 | 0.8892 | 0.0326 |
| D_sparse8_f64 | 0.0142147 | 0.9699 | 0.0403 |
| A_hard_f32 | 68.5681 | 0.1894 | nan |
| B_dense_f32 | 0 | 1.0000 | 1.0000 |
| C_partition_f32 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f32 | 0.488904 | 0.7599 | 0.0256 |
| D_sparse6_f32 | 0.0832958 | 0.8892 | 0.0326 |
| D_sparse8_f32 | 0.0142149 | 0.9699 | 0.0403 |

## Research question: measured answer

GT-near matched median error (macro): hard 0.5051 mm; soft 0.6716 mm.
GT-near recall: hard 0.8698; soft 0.8833.
Thus soft voting is not assumed to be more robust merely because weights vary continuously. The error/recall curves and paired shared-GT comparison determine the measured benefit.
Mean actual-query dense/partition speed ratio = 1.096x; quantile-stage ratio = 1.914x.
Partition/sparse-alpha8 speed ratio = 1.080x.
The partition optimization is exact relative to the dense reference at the chosen precision. Sparse is kept only as an explicit comparison option, never silently substituted for dense.

## Paired localization / float precision

Only shared successful GT matches contribute. Misses/false positives remain in summary.csv; these rows cannot establish equal recall.

| Comparison | Variant (GT-near) | Method median mm | Reference median mm | Mean within-condition P95 position delta mm | Max delta mm | Changed raw index fraction |
|---|---|---:|---:|---:|---:|---:|
| hard | B_dense_f64 | 0.4986 | 0.5051 | 0.383492 | 6.3618 | 0.0186 |
| hard | C_partition_f64 | 0.4986 | 0.5051 | 0.383492 | 6.3618 | 0.0186 |
| dense | A_hard_f64 | 0.5051 | 0.4986 | 0.383492 | 6.3618 | 0.0186 |
| dense | B_dense_f64 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | C_partition_f64 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse4_f64 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse6_f64 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse8_f64 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | A_hard_f32 | 0.5051 | 0.4986 | 0.383492 | 6.3618 | 0.0186 |
| dense | B_dense_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | C_partition_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse4_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse6_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| dense | D_sparse8_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| float64 | A_hard_f32 | 0.5051 | 0.5051 | 0 | 0 | 0.0000 |
| float64 | B_dense_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| float64 | C_partition_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| float64 | D_sparse4_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| float64 | D_sparse6_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |
| float64 | D_sparse8_f32 | 0.6716 | 0.6716 | 0 | 0 | 0.0000 |

## Whole candidate pipeline latency

Includes voxel preprocessing + unchanged CED/PCA/NMS + local stage for every CED query, excludes capture/I/O/rendering.

| Variant | Mean ms | Median ms | P95 ms | 1000 / median ms |
|---|---:|---:|---:|---:|
| A_hard_f64 | 21.843 | 23.265 | 30.575 | 43.0 |
| B_dense_f64 | 28.321 | 28.111 | 37.170 | 35.6 |
| C_partition_f64 | 27.537 | 27.553 | 36.095 | 36.3 |
| D_sparse4_f64 | 25.131 | 25.609 | 33.980 | 39.0 |
| D_sparse6_f64 | 26.163 | 26.319 | 34.781 | 38.0 |
| D_sparse8_f64 | 26.937 | 26.861 | 35.489 | 37.2 |
| A_hard_f32 | 21.584 | 23.015 | 30.304 | 43.5 |
| B_dense_f32 | 28.079 | 27.930 | 37.080 | 35.8 |
| C_partition_f32 | 27.318 | 27.353 | 35.823 | 36.6 |
| D_sparse4_f32 | 25.036 | 25.545 | 33.689 | 39.1 |
| D_sparse6_f32 | 26.062 | 26.266 | 34.658 | 38.1 |
| D_sparse8_f32 | 26.762 | 26.784 | 35.286 | 37.3 |

## Interpretation / limits

Hard is retained as the live default; none of these measurements authorizes silently changing the user's ROI, NMS or threshold settings.
All variants use the same score gate 0. At 0.04, diffuse soft mass may be rejected; gate-zero runs isolate peak behavior and must also report added false positives.
Sigmoid normalization includes padding first; weights outside the candidate core are then discarded without renormalization. Sparse alpha cuts support before normalization and is genuinely approximate.
Higher alpha can reduce omitted mass but cannot guarantee identical peak identities near ties. Float32 is also checked separately; exact-zero-noise ties can change under rounding.
No Gaussian smoothing, spatial weighted mean, new geometric refinement or query sampling is used. Each output coordinate is an actual observation.
Missing observed vertices limit localization; soft votes may prefer near-boundary interior samples, so lower variance alone is not proof of lower bias.
Euclidean components are not outlier rejection. Isolated outliers can become singleton components receiving 256 votes in every method, causing stable/confident false positives.
Spatial sparse voting still computes every projection and support mask; gather/scatter overhead may outweigh saved sigmoid evaluations.
Default hard heatmaps were approximately 98% zero on the inspected synthetic face cases. Zero is now gray and positive raw mass is log-colored with numeric range; fractional soft weights are never cast to int.
Position RMS graphs use only successful repeated matches and show observation counts in stability.csv. They are not smoothed detector outputs.
This is a small-cloud CPU benchmark, not a guarantee of sensor-frame real time. Frame results include all CED queries; CED cost is reported separately.
No STL GUI, RealSense or Kinect hardware validation was performed.

Graphs: noise_localization.png, noise_detection.png, repeatability.png, stage_times.png, tradeoff.png.
3-D figures: one_face_voting.png, two_faces_voting.png, three_faces_voting.png (sigma=0.2, identical GT-near query).
