surface_aware_harris3d.py
=========================

python surface_aware_harris3d.py \
    --input /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/laser_pose_cross.stl \
    --voxel 1.0 \
    --normal-radius 3.0 \
    --spatial-sigma-ratio 0.5 \
    --sigma-e-deg 15 \
    --normal-kernel cauchy \
    --sigma-r 0.25 \
    --normal-max-iterations 10 \
    --normal-update-mode synchronous \
    --noise-std 0.05 \
    --outlier-ratio 0.10 \
    --outlier-offset-min 0.5 \
    --outlier-offset-max 1.5 \
    --debug-view all

Minimal reusable module containing only:

1) Surface-aware iterative normal estimation
2) Gaussian-window Harris3D
3) Optional PCL-style corner refinement

Dependencies
------------
numpy
open3d
pclpybridge

All geometric parameters use the SAME length unit as the input point cloud.
If the point cloud is in millimeters, radius/sigma values are also in mm.

Typical usage
-------------
import surface_aware_harris3d as sah

normals = sah.estimate_surface_aware_normals(
    points,
    radius=3.0,
    sigma_e_deg=15.0,
    sigma_r=0.25,
    max_iterations=5,
    update_mode="fixed",          # A: original fixed-neighbor mode
)

normals_B = sah.estimate_surface_aware_normals(
    points,
    radius=3.0,
    sigma_e_deg=15.0,
    sigma_r=0.25,
    max_iterations=5,
    update_mode="synchronous",    # B: full-field synchronous mode
)

corners, response = sah.gaussian_harris3d(
    points,
    normals,
    radius=4.0,
    sigma=1.5,
    threshold=1e-6,
    nonmax=True,
    refine=True,
    refine_max_iterations=5,
)

Weight-return usage (both fixed and synchronous modes)
----------------------------------------------------
normals, weights = sah.estimate_surface_aware_normals(
    points, radius=3.0, sigma_e_deg=15.0, sigma_r=0.25,
    update_mode="synchronous", return_weights=True,
)
query_index = 0
w = weights[query_index]
neighbor_points = points[w["indices"]]
surface_weight = w["surface_weight"]   # w_e * w_r: no distance Gaussian
full_weight = w["full_weight"]         # w_d * w_e * w_r
# Also available: normal_field_weight (w_e), residual_weight (w_r),
# spatial_weight (w_d), pca_weight, valid, iteration.
# Normal estimation itself ALWAYS uses all three factors, as before.
# No sum-to-one normalization is applied to these returned weights.
# Query-only fixed-mode alternative:
# debug = sah.inspect_surface_aware_point(points, query_index, radius=3.0)
# w = debug["weights"]

Or run the full pipeline:

result = sah.surface_aware_harris3d(
    points,
    normal_radius=3.0,
    harris_radius=4.0,
    gaussian_sigma=1.5,
    refine=True,
)
"""