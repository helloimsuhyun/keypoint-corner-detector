# Quantile soft voting: measured comparison

AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; 1 seeds per scene/noise; 7 scenes; 6 noise levels.
Full CPU/thread/config provenance: metadata.json. Timings exclude data generation, matching, I/O and rendering.
CED/PCA/NMS defaults are unchanged. Local study fixes r=30, padding=10, cluster=4.5, NMS=5 mm, tau=0.25 mm, 256 directions and score gate=0.04.
The user's live ROI settings are not overwritten. Every variant sees identical raw XYZ and unchanged CED queries; GT-near queries are a separate diagnostic.

## Aggregate metrics

Macro means over scenes/noise. Error values condition on successful 1:1 matches within 5 mm; read recall and FP alongside errors.

| Source | Variant | Recall | FP/frame | Matched median mm | Matched P95 mm | Mean query ms | Mean frame ms |
|---|---|---:|---:|---:|---:|---:|---:|
| gt_near | A_hard_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 1.315 | 1.474 |
| gt_near | B_dense_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 1.782 | 1.933 |
| gt_near | C_partition_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 1.226 | 1.372 |
| gt_near | D_sparse4_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 0.943 | 1.092 |
| gt_near | D_sparse6_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 0.909 | 1.054 |
| gt_near | D_sparse8_f64 | 1.000 | 0.00 | 0.067 | 0.067 | 0.956 | 1.103 |
| gt_near | A_hard_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 0.551 | 0.695 |
| gt_near | B_dense_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 1.273 | 1.418 |
| gt_near | C_partition_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 1.089 | 1.235 |
| gt_near | D_sparse4_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 0.922 | 1.067 |
| gt_near | D_sparse6_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 0.913 | 1.058 |
| gt_near | D_sparse8_f32 | 1.000 | 0.00 | 0.067 | 0.067 | 0.921 | 1.066 |
| ced | A_hard_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 0.760 | 3.247 |
| ced | B_dense_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 1.127 | 4.700 |
| ced | C_partition_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 0.975 | 4.094 |
| ced | D_sparse4_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 0.825 | 3.490 |
| ced | D_sparse6_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 0.814 | 3.444 |
| ced | D_sparse8_f64 | 1.000 | 0.00 | 0.200 | 0.228 | 0.796 | 3.378 |
| ced | A_hard_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 0.458 | 2.015 |
| ced | B_dense_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 1.098 | 4.582 |
| ced | C_partition_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 0.885 | 3.730 |
| ced | D_sparse4_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 0.813 | 3.445 |
| ced | D_sparse6_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 0.785 | 3.326 |
| ced | D_sparse8_f32 | 1.000 | 0.00 | 0.200 | 0.228 | 0.779 | 3.304 |

## Stage timing (actual CED queries)

| Variant | Projection | Quantile | Sigmoid + support | Normalize | Accumulate | NMS | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_hard_f64 | 0.076 | 0.000 | 0.000 | 0.000 | 0.015 | 0.066 | 0.760 |
| B_dense_f64 | 0.053 | 0.366 | 0.222 | 0.039 | 0.026 | 0.086 | 1.127 |
| C_partition_f64 | 0.049 | 0.190 | 0.226 | 0.038 | 0.026 | 0.092 | 0.975 |
| D_sparse4_f64 | 0.048 | 0.185 | 0.123 | 0.015 | 0.020 | 0.088 | 0.825 |
| D_sparse6_f64 | 0.049 | 0.184 | 0.122 | 0.016 | 0.021 | 0.087 | 0.814 |
| D_sparse8_f64 | 0.048 | 0.180 | 0.127 | 0.017 | 0.020 | 0.081 | 0.796 |
| A_hard_f32 | 0.023 | 0.000 | 0.000 | 0.000 | 0.013 | 0.057 | 0.458 |
| B_dense_f32 | 0.045 | 0.368 | 0.195 | 0.035 | 0.026 | 0.087 | 1.098 |
| C_partition_f32 | 0.040 | 0.182 | 0.185 | 0.032 | 0.025 | 0.085 | 0.885 |
| D_sparse4_f32 | 0.042 | 0.182 | 0.116 | 0.016 | 0.023 | 0.089 | 0.813 |
| D_sparse6_f32 | 0.038 | 0.177 | 0.113 | 0.016 | 0.020 | 0.082 | 0.785 |
| D_sparse8_f32 | 0.038 | 0.177 | 0.118 | 0.018 | 0.021 | 0.082 | 0.779 |

All table times are ms. ROI, clustering and shared spatial-index preparation are separate CSV columns.

## Exact quantile and sparse approximation

Maximum checked threshold difference: 0 mm (seed 0, every scene/noise/dtype).
The optimized quantile partitions two ranks and retains NumPy's stable linear interpolation, rather than substituting a single order statistic.
Installed NumPy 1.21.5 already implements np.quantile with partition. A speed gain here is wrapper/layout overhead reduction, not a change from full sort to linear selection.

| Variant (GT-near) | Vote L1 vs dense | Candidate Jaccard vs dense | Sigmoid support fraction |
|---|---:|---:|---:|
| A_hard_f64 | 91.0404 | 1.0000 | nan |
| B_dense_f64 | 0 | 1.0000 | 1.0000 |
| C_partition_f64 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f64 | 0.591382 | 1.0000 | 0.0280 |
| D_sparse6_f64 | 0.0954182 | 1.0000 | 0.0369 |
| D_sparse8_f64 | 0.0150406 | 1.0000 | 0.0469 |
| A_hard_f32 | 91.0404 | 1.0000 | nan |
| B_dense_f32 | 0 | 1.0000 | 1.0000 |
| C_partition_f32 | 0 | 1.0000 | 1.0000 |
| D_sparse4_f32 | 0.591383 | 1.0000 | 0.0280 |
| D_sparse6_f32 | 0.0954188 | 1.0000 | 0.0369 |
| D_sparse8_f32 | 0.0150424 | 1.0000 | 0.0469 |

## Interpretation / limits

Hard is retained as the live default; none of these measurements authorizes silently changing the user's ROI, NMS or threshold settings.
A diffuse soft vote can fall below the same 0.04 per-point gate even when its total corner-region mass is high. We did not lower the gate to manufacture a soft-voting win.
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
