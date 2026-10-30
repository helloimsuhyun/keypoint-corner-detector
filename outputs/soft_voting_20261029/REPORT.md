# Quantile soft voting: measured comparison

AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; 100 seeds per scene/noise; 7 scenes; 6 noise levels.
Full CPU/thread/config provenance: metadata.json. Timings exclude data generation, matching, I/O and rendering.
CED/PCA/NMS defaults are unchanged. Local study fixes r=30, padding=10, cluster=4.5, NMS=5 mm, tau=0.25 mm, 256 directions and score gate=0.04.
The user's live ROI settings are not overwritten. Every variant sees identical raw XYZ and unchanged CED queries; GT-near queries are a separate diagnostic.

## Aggregate metrics

Macro means over scenes/noise. Error values condition on successful 1:1 matches within 5 mm; read recall and FP alongside errors.

| Source | Variant | Recall | FP/frame | Matched median mm | Matched P95 mm | Mean query ms | Mean frame ms |
|---|---|---:|---:|---:|---:|---:|---:|
| gt_near | A_hard_f64 | 0.857 | 0.74 | 0.419 | 0.887 | 1.214 | 1.472 |
| gt_near | B_dense_f64 | 0.143 | 0.65 | 0.570 | 0.821 | 2.320 | 2.577 |
| gt_near | C_partition_f64 | 0.143 | 0.65 | 0.570 | 0.821 | 1.934 | 2.192 |
| gt_near | D_sparse4_f64 | 0.143 | 0.65 | 0.570 | 0.821 | 1.573 | 1.831 |
| gt_near | D_sparse6_f64 | 0.143 | 0.65 | 0.570 | 0.821 | 1.573 | 1.831 |
| gt_near | D_sparse8_f64 | 0.143 | 0.65 | 0.570 | 0.821 | 1.588 | 1.846 |
| gt_near | A_hard_f32 | 0.857 | 0.74 | 0.419 | 0.887 | 0.924 | 1.182 |
| gt_near | B_dense_f32 | 0.143 | 0.65 | 0.570 | 0.821 | 2.157 | 2.414 |
| gt_near | C_partition_f32 | 0.143 | 0.65 | 0.570 | 0.821 | 1.792 | 2.050 |
| gt_near | D_sparse4_f32 | 0.143 | 0.65 | 0.570 | 0.821 | 1.501 | 1.759 |
| gt_near | D_sparse6_f32 | 0.143 | 0.65 | 0.570 | 0.821 | 1.504 | 1.762 |
| gt_near | D_sparse8_f32 | 0.143 | 0.65 | 0.570 | 0.821 | 1.517 | 1.775 |
| ced | A_hard_f64 | 0.627 | 0.41 | 0.310 | 0.583 | 0.619 | 2.582 |
| ced | B_dense_f64 | 0.529 | 0.40 | 0.311 | 0.563 | 1.267 | 4.973 |
| ced | C_partition_f64 | 0.529 | 0.40 | 0.311 | 0.563 | 1.064 | 4.225 |
| ced | D_sparse4_f64 | 0.530 | 0.40 | 0.311 | 0.563 | 0.917 | 3.684 |
| ced | D_sparse6_f64 | 0.529 | 0.40 | 0.311 | 0.563 | 0.920 | 3.692 |
| ced | D_sparse8_f64 | 0.529 | 0.40 | 0.311 | 0.563 | 0.928 | 3.720 |
| ced | A_hard_f32 | 0.627 | 0.41 | 0.310 | 0.583 | 0.545 | 2.305 |
| ced | B_dense_f32 | 0.529 | 0.40 | 0.311 | 0.563 | 1.210 | 4.762 |
| ced | C_partition_f32 | 0.529 | 0.40 | 0.311 | 0.563 | 1.004 | 4.004 |
| ced | D_sparse4_f32 | 0.530 | 0.40 | 0.311 | 0.563 | 0.893 | 3.593 |
| ced | D_sparse6_f32 | 0.529 | 0.40 | 0.311 | 0.563 | 0.896 | 3.606 |
| ced | D_sparse8_f32 | 0.529 | 0.40 | 0.311 | 0.563 | 0.906 | 3.641 |

## Stage timing (actual CED queries)

| Variant | Projection | Quantile | Sigmoid + support | Normalize | Accumulate | NMS | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_hard_f64 | 0.036 | 0.000 | 0.000 | 0.000 | 0.016 | 0.070 | 0.619 |
| B_dense_f64 | 0.059 | 0.414 | 0.254 | 0.043 | 0.030 | 0.088 | 1.267 |
| C_partition_f64 | 0.055 | 0.217 | 0.249 | 0.042 | 0.029 | 0.090 | 1.064 |
| D_sparse4_f64 | 0.056 | 0.216 | 0.137 | 0.017 | 0.024 | 0.089 | 0.917 |
| D_sparse6_f64 | 0.056 | 0.216 | 0.142 | 0.018 | 0.024 | 0.088 | 0.920 |
| D_sparse8_f64 | 0.056 | 0.216 | 0.148 | 0.020 | 0.025 | 0.088 | 0.928 |
| A_hard_f32 | 0.028 | 0.000 | 0.000 | 0.000 | 0.015 | 0.069 | 0.545 |
| B_dense_f32 | 0.048 | 0.412 | 0.218 | 0.038 | 0.028 | 0.088 | 1.210 |
| C_partition_f32 | 0.045 | 0.212 | 0.214 | 0.036 | 0.027 | 0.089 | 1.004 |
| D_sparse4_f32 | 0.045 | 0.211 | 0.128 | 0.019 | 0.024 | 0.088 | 0.893 |
| D_sparse6_f32 | 0.045 | 0.211 | 0.132 | 0.020 | 0.025 | 0.088 | 0.896 |
| D_sparse8_f32 | 0.045 | 0.211 | 0.138 | 0.022 | 0.026 | 0.088 | 0.906 |

All table times are ms. ROI, clustering and shared spatial-index preparation are separate CSV columns.

## Exact quantile and sparse approximation

Maximum checked threshold difference: 0 mm (seed 0, every scene/noise/dtype).
The optimized quantile partitions two ranks and retains NumPy's stable linear interpolation, rather than substituting a single order statistic.
Installed NumPy 1.21.5 already implements np.quantile with partition. A speed gain here is wrapper/layout overhead reduction, not a change from full sort to linear selection.

| Variant (GT-near) | Vote L1 vs dense | Candidate Jaccard vs dense | Sigmoid support fraction |
|---|---:|---:|---:|
| A_hard_f64 | 68.3616 | 0.3203 | nan |
| B_dense_f64 | 0 | 1.0000 | 1.0000 |
| C_partition_f64 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f64 | 0.488775 | 0.9995 | 0.0256 |
| D_sparse6_f64 | 0.0833102 | 0.9995 | 0.0326 |
| D_sparse8_f64 | 0.0141994 | 1.0000 | 0.0404 |
| A_hard_f32 | 68.3616 | 0.3203 | nan |
| B_dense_f32 | 0 | 1.0000 | 1.0000 |
| C_partition_f32 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f32 | 0.488775 | 0.9995 | 0.0256 |
| D_sparse6_f32 | 0.0833098 | 0.9995 | 0.0326 |
| D_sparse8_f32 | 0.0141995 | 1.0000 | 0.0404 |

## Research question: measured answer

GT-near matched median error (macro): hard 0.4187 mm; soft 0.5701 mm.
GT-near recall: hard 0.8570; soft 0.1430.
Thus soft voting is not assumed to be more robust merely because weights vary continuously. The error/recall curves and paired shared-GT comparison determine the measured benefit.
Mean actual-query dense/partition speed ratio = 1.191x; quantile-stage ratio = 1.911x.
Partition/sparse-alpha8 speed ratio = 1.147x.
The partition optimization is exact relative to the dense reference at the chosen precision. Sparse is kept only as an explicit comparison option, never silently substituted for dense.

## Paired localization / float precision

Only shared successful GT matches contribute. Misses/false positives remain in summary.csv; these rows cannot establish equal recall.

| Comparison | Variant (GT-near) | Method median mm | Reference median mm | Mean within-condition P95 position delta mm | Max delta mm | Changed raw index fraction |
|---|---|---:|---:|---:|---:|---:|
| hard | B_dense_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| hard | C_partition_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | A_hard_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | B_dense_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | C_partition_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse4_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse6_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse8_f64 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | A_hard_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | B_dense_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | C_partition_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse4_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse6_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| dense | D_sparse8_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| float64 | A_hard_f32 | 0.4187 | 0.4187 | 0 | 0 | 0.0000 |
| float64 | B_dense_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| float64 | C_partition_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| float64 | D_sparse4_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| float64 | D_sparse6_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |
| float64 | D_sparse8_f32 | 0.5701 | 0.5701 | 0 | 0 | 0.0000 |

## Whole candidate pipeline latency

Includes voxel preprocessing + unchanged CED/PCA/NMS + local stage for every CED query, excludes capture/I/O/rendering.

| Variant | Mean ms | Median ms | P95 ms | 1000 / median ms |
|---|---:|---:|---:|---:|
| A_hard_f64 | 21.732 | 22.660 | 31.161 | 44.1 |
| B_dense_f64 | 24.123 | 24.677 | 34.275 | 40.5 |
| C_partition_f64 | 23.375 | 24.093 | 32.995 | 41.5 |
| D_sparse4_f64 | 22.834 | 23.626 | 32.659 | 42.3 |
| D_sparse6_f64 | 22.842 | 23.630 | 32.672 | 42.3 |
| D_sparse8_f64 | 22.870 | 23.656 | 32.558 | 42.3 |
| A_hard_f32 | 21.455 | 22.450 | 30.794 | 44.5 |
| B_dense_f32 | 23.912 | 24.481 | 34.035 | 40.8 |
| C_partition_f32 | 23.154 | 23.927 | 32.797 | 41.8 |
| D_sparse4_f32 | 22.743 | 23.558 | 32.549 | 42.4 |
| D_sparse6_f32 | 22.756 | 23.556 | 32.431 | 42.5 |
| D_sparse8_f32 | 22.791 | 23.599 | 32.503 | 42.4 |

## Interpretation / limits

Hard is retained as the live default; none of these measurements authorizes silently changing the user's ROI, NMS or threshold settings.
All variants use the same score gate 0.04. At 0.04, diffuse soft mass may be rejected; gate-zero runs isolate peak behavior and must also report added false positives.
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
