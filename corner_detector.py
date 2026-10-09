#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

RealSense Harris3D -> tangent-frame -> normal-weighted projection inspector.



Scene point-cloud corner detector built from the tangent-frame Harris pipeline.

Main automatic path:

  1) P: freeze the current aligned RGB-D frame
  2) convert depth to a scene point cloud
  3) voxel-downsample with a user parameter
  4) estimate surface-aware normals once on the downsampled scene
  5) select query points with farthest-point sampling (FPS)
  6) show the full downsampled cloud + FPS query markers
  7) run the existing tangent normal-weight -> raster -> masked Harris2D corner
     calculation independently around every query point
  8) reject edge-like peaks by Harris eigenvalue balance, then reject weak
     responses and radius-cluster the remaining 3-D corners,
     reject spatial outliers inside each cluster, then use the robust centroid
  9) show the full point cloud + final robust-centroid 3-D corner markers
 10) project the final 3-D corners back to the frozen RGB image and overlay markers

The previous mouse-click inspector is retained only as an optional diagnostic path.

There is NO FREE/UNKNOWN voxel reasoning or contour extraction.



Main controls

-------------

  P       : freeze current aligned RGB-D frame, compute normals + Harris, open viewer

  R       : return to live mode

  Click   : while frozen, inspect local tangent frame and weighted projection

  [ / ]   : decrease / increase local inspection radius

  1       : RGB

  2       : depth

  3       : RGB + depth

  H       : reopen Harris/normal viewer for the frozen frame
  V       : reopen raw pre-clustering corner hypotheses
  C       : reopen final clustered corners

  Q / ESC : quit



Harris/normal Open3D viewer

---------------------------

  1 : Harris response heatmap only

  2 : gray point cloud + estimated normals

  3 : Harris heatmap + estimated normals

  R : reset camera

  Q / ESC : close viewer



Units

-----

All point-cloud distances are millimeters.

"""



from __future__ import annotations



import importlib.util

import time

from pathlib import Path



import cv2

import numpy as np

import open3d as o3d

import pyrealsense2 as rs

from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt





# ============================================================

# User Surface-Aware Harris3D module

# ============================================================



SAH_FILE = Path(

    "/home/choisuhyun/keypoint-corner-detector/"

    "surface_aware_harris3d/surface_aware_harris3d.py"

)



if not SAH_FILE.exists():

    raise FileNotFoundError(f"Surface-Aware Harris3D module not found:\n{SAH_FILE}")



_spec = importlib.util.spec_from_file_location("surface_aware_harris3d_impl", str(SAH_FILE))

if _spec is None or _spec.loader is None:

    raise ImportError(f"Could not create import spec for: {SAH_FILE}")



sah = importlib.util.module_from_spec(_spec)

_spec.loader.exec_module(sah)



for required_name in ("estimate_surface_aware_normals", "gaussian_harris_response"):

    if not hasattr(sah, required_name):

        raise AttributeError(

            f"Loaded {SAH_FILE.name}, but {required_name}() was not found."

        )



print(f"[module] loaded: {SAH_FILE}")





# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
#
# Parameters are grouped by PURPOSE / processing stage.
# No algorithmic behavior is changed here; this is only a reorganization
# of the existing global variables.
# ============================================================


# ------------------------------------------------------------
# 1) RealSense input
# ------------------------------------------------------------

WIDTH = 640
HEIGHT = 480
FPS = 30

MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 3.0


# ------------------------------------------------------------
# 2) Point-cloud sampling
# ------------------------------------------------------------

PIXEL_STRIDE = 1

# Automatic scene-detector sampling.
# Depth points are first voxel-downsampled, then FPS chooses spatially spread
# query points directly from the downsampled cloud.
SCENE_VOXEL_SIZE_MM = 2.0
SCENE_FPS_NUM_QUERIES = 15000

# Scene-level candidate filtering + radius clustering in camera-frame 3-D.
# Multiple nearby FPS queries can converge to the same physical corner.
#
# Pipeline:
#   raw positive Harris candidates that already pass eigenvalue-balance gating
#   -> absolute response floor (optional)
#   -> global response percentile cut
#   -> radius-connected clustering
#   -> reject small clusters
#   -> robust spatial outlier rejection inside each cluster
#   -> arithmetic centroid of the remaining inlier detections
#
# The final corner is therefore supported by repeated FPS-query detections rather
# than chosen from one max-response query.  Harris response is used for the
# pre-clustering rejection, not as a coordinate weight.
SCENE_CORNER_MIN_RESPONSE = 0.0
SCENE_CORNER_RESPONSE_PERCENTILE = 50.0

# Harris structure-tensor shape constraint.
# Let lambda_max >= lambda_min >= 0 be the two eigenvalues of the local
# 2-D structure tensor. A line/edge has lambda_min << lambda_max, while a
# true 2-D corner should have substantial energy in both directions.
#
#   rho = lambda_min / (lambda_max + eps)
#
# Candidates are allowed into local-peak selection only when rho >= threshold.
# This is applied BEFORE scene clustering, and the click diagnostic uses the
# same threshold so its displayed peak matches the automatic detector.
SCENE_CORNER_MIN_EIGEN_RATIO = 0.75

SCENE_CORNER_CLUSTER_RADIUS_MM = 10.0
SCENE_CORNER_CLUSTER_MIN_SIZE = 4

# Robust in-cluster trimming.  Distances are measured from the coordinate-wise
# median.  Inlier threshold = median_distance + K * 1.4826 * MAD(distance),
# with a small physical floor and never larger than the clustering radius.
SCENE_CORNER_CLUSTER_OUTLIER_MAD_SCALE = 2.5
SCENE_CORNER_CLUSTER_INLIER_FLOOR_MM = 2.0
SCENE_MAX_FINAL_CORNERS = 300

# Stage visualization.
SCENE_SHOW_QUERY_VIEW_3D = True
SCENE_SHOW_RAW_CANDIDATES_3D = True
SCENE_SHOW_CORNER_VIEW_3D = True
SCENE_SHOW_RESULT_2D = True
SCENE_RESULT_2D_WINDOW = "Scene corner detector | projected final corners"

RAW_CANDIDATE_MARKER_RADIUS_MM = 3.5
RAW_CANDIDATE_COLOR = [1.0, 0.25, 0.95]        # magenta
RAW_CANDIDATE_QUERY_COLOR = [1.0, 0.80, 0.10]  # yellow-orange


# ------------------------------------------------------------
# 3) Surface-aware normal estimation
# ------------------------------------------------------------

# Global scene normal radius used by the automatic detector and Harris3D diagnostic.
NORMAL_RADIUS_MM = 8.0

# Click-stage local normal radius:
#
#   r_n(Z) = NORMAL_RADIUS_REFERENCE_MM
#            * Z / NORMAL_RADIUS_REFERENCE_Z_MM
#
NORMAL_RADIUS_REFERENCE_MM = 8.0
NORMAL_RADIUS_REFERENCE_Z_MM = 500.0

# Clamp ratios relative to NORMAL_RADIUS_REFERENCE_MM.
NORMAL_RADIUS_MIN_RATIO = 0.50
NORMAL_RADIUS_MAX_RATIO = 2.00

NORMAL_RADIUS_MIN_MM = (
    NORMAL_RADIUS_REFERENCE_MM * NORMAL_RADIUS_MIN_RATIO
)
NORMAL_RADIUS_MAX_MM = (
    NORMAL_RADIUS_REFERENCE_MM * NORMAL_RADIUS_MAX_RATIO
)

# Surface-aware normal refinement parameters.
NORMAL_SIGMA_E_DEG = 15.0
NORMAL_SIGMA_R = 1.0
NORMAL_MAX_ITERATIONS = 5
NORMAL_UPDATE_MODE = "synchronous"


# ------------------------------------------------------------
# 4) Local tangent inspection ROI
# ------------------------------------------------------------

# Inspection radius is expressed relative to the reference normal radius.
INSPECT_TO_NORMAL_RATIO = 3.75
INSPECT_STEP_TO_NORMAL_RATIO = 0.625
INSPECT_MIN_TO_NORMAL_RATIO = 1.25
INSPECT_MAX_TO_NORMAL_RATIO = 10.0

# Base inspection radius at the reference depth.
INSPECT_RADIUS_MM = (
    NORMAL_RADIUS_REFERENCE_MM * INSPECT_TO_NORMAL_RATIO
)
INSPECT_RADIUS_REFERENCE_Z_MM = NORMAL_RADIUS_REFERENCE_Z_MM

# [ / ] changes the BASE radius.
INSPECT_RADIUS_STEP_MM = (
    NORMAL_RADIUS_REFERENCE_MM * INSPECT_STEP_TO_NORMAL_RATIO
)
INSPECT_RADIUS_MIN_MM = (
    NORMAL_RADIUS_REFERENCE_MM * INSPECT_MIN_TO_NORMAL_RATIO
)
INSPECT_RADIUS_MAX_MM = (
    NORMAL_RADIUS_REFERENCE_MM * INSPECT_MAX_TO_NORMAL_RATIO
)

# Clamp after depth adaptation.
INSPECT_RADIUS_EFFECTIVE_MIN_MM = 10.0
INSPECT_RADIUS_EFFECTIVE_MAX_MM = 120.0

MIN_LOCAL_POINTS = 10


# LOCAL corenr detector -----------------------------------------------

# normal 차이 weight sigma 기본 20. 널널 30
# Click-stage normal-similarity weight:
#
#   theta_i = acos(|n_i^T n_q|)
#   w_i = exp(-0.5 * (theta_i / sigma_n)^2)
#
TANGENT_NORMAL_WEIGHT_SIGMA_DEG = 30.0


# ------------------------------------------------------------
# 6) Tangent-plane spacing / rasterization / hole reconstruction
# ------------------------------------------------------------

# Characteristic tangent-plane sampling spacing s.
# s 파라미터 구하는 것 
# K-NN으로 주위 점에 대해서 distance를 구하고 , median >> 대표 점과 점사이 거리 구함 
LOCAL_SPACING_K = 6
TANGENT_SPACING_FALLBACK_MM = 0.6 # fallback

# Adaptive raster:
#   cell_mm = GRID_CELL_SCALE * s

GRID_CELL_SCALE = 0.40
GRID_SIZE_MIN = 96
GRID_SIZE_MAX = 800

# mean fill --- 안쪽 영역 mean으로 채우기
#   r_mean = MEAN_MEAN_FILL_RADIUS_SCALE * s
MEAN_MEAN_FILL_RADIUS_SCALE = 3.00
MEAN_FILL_MIN_VALID_NEIGHBORS = 4


# ------------------------------------------------------------
# 7) Pixel-domain denoising + live GUI parameters
# ------------------------------------------------------------

# Bilateral range sigma in W-value units.
BILATERAL_SIGMA_COLOR = 0.12

# Spatial image-processing scales after adaptive rasterization.
# These two values are changed by the live OpenCV GUI.
PIXEL_TUNING_DEFAULTS = {
    "bilateral_sigma_space_px": 5.00,
    "harris_sigma_px": 6.00,
}
PIXEL_TUNING = dict(PIXEL_TUNING_DEFAULTS)


# ------------------------------------------------------------
# 8) Harris2D detector
# ------------------------------------------------------------

# Harris:
#
#   R = det(M) - k * trace(M)^2

HARRIS2D_K = 0.04
HARRIS2D_SOBEL_KSIZE = 3
# vis
HARRIS2D_RESPONSE_PERCENTILE = 95.0
HARRIS2D_EPS = 1e-12
HARRIS2D_SHOW_WINDOW = True
HARRIS2D_SHOW_ONE_SIGMA = True
HARRIS2D_EFFECTIVE_SUPPORT_SIGMA = 3.0

# Query-local corner extraction:
# local maxima are detected within a (2*r+1)x(2*r+1) neighborhood,
# then only the strongest one is kept for each clicked query.
HARRIS2D_LOCAL_MAX_RADIUS_PX = 3

# Accept a detected peak only inside the central fraction of the tangent ROI:
#
#   ||[u*, v*]|| <= HARRIS2D_PEAK_ROI_RATIO * r_ROI
#
# With 0.80, the outer 20% radial band is excluded from peak selection.
# Harris response is still computed/visualized over the full valid ROI.
HARRIS2D_PEAK_ROI_RATIO = 0.80


# ------------------------------------------------------------
# Harris2D ROI boundary handling
# ------------------------------------------------------------
# No value is synthesized outside the circular ROI.
# Sobel gradients are trusted only where the full Sobel kernel lies inside ROI,
# and the Harris structure tensor uses mask-normalized Gaussian integration.
#
# Keep only the requested INNER boundary smoothing band:
#   r_inner_band = HARRIS_PAD_INNER_SMOOTH_BAND_RATIO * sigma_H
HARRIS_PAD_INNER_SMOOTH_BAND_RATIO = 1.50

# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# 10) Freeze-time Gaussian Harris3D diagnostic
# ------------------------------------------------------------

HARRIS_RADIUS_MM = 15.0
GAUSSIAN_SIGMA_MM = 6.0

# Optional manual heatmap range. None = percentile-based automatic range.
HEATMAP_MIN = None
HEATMAP_MAX = None


# ------------------------------------------------------------
# 11) Visualization / UI
# ------------------------------------------------------------

# Open3D point / normal visualization.
POINT_SIZE = 4.0
CAMERA_FRAME_SIZE_MM = 100.0

NORMAL_VIS_STRIDE = 80
NORMAL_VIS_LENGTH_MM = 18.0
NORMAL_VIS_COLOR = [0.15, 0.95, 0.20]

O3D_BACKGROUND = np.array(
    [0.025, 0.025, 0.035],
    dtype=np.float64,
)

# Full-scene corner visualization.
# Cyan scene points on a dark background make the red corner markers easy to see.
GLOBAL_CLOUD_COLOR = [0.18, 0.72, 0.88]
GLOBAL_CLOUD_POINT_SIZE = 4.0
GLOBAL_CORNER_MARKER_RADIUS_MM = 5.0
GLOBAL_QUERY_MARKER_RADIUS_MM = 3.0
GLOBAL_CORNER_COLOR = [1.0, 0.08, 0.08]
GLOBAL_QUERY_COLOR = [1.0, 0.75, 0.10]

# Tangent projection panel.
TANGENT_VIEW_SIZE = 760
TANGENT_VIEW_MARGIN = 65

# OpenCV window names.
WINDOW_NAME = "RealSense Harris + Tangent Inspector"
WEIGHTED_PROJECTION_WINDOW = (
    "Tangent normal-weight field + direct mean-fill + bilateral + masked Harris2D"
)
FROZEN_RGBD_WINDOW = "Frozen RGB + Depth"
PIXEL_TUNER_WINDOW = "Pixel-space tuning | live update"



# ============================================================

# Basic helpers

# ============================================================



def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    v = np.asarray(v, dtype=np.float64).reshape(3)

    n = float(np.linalg.norm(v))

    if n < eps:

        raise ValueError("Cannot normalize near-zero vector.")

    return v / n





def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:

    out = np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    valid = (

        np.isfinite(depth_m)

        & (depth_m >= MIN_DEPTH_M)

        & (depth_m <= MAX_DEPTH_M)

    )

    if not np.any(valid):

        return out



    zmin = float(np.percentile(depth_m[valid], 2.0))

    zmax = float(np.percentile(depth_m[valid], 98.0))

    if zmax <= zmin:

        zmax = zmin + 1e-3



    norm = np.clip((depth_m - zmin) / (zmax - zmin), 0.0, 1.0)

    gray = ((1.0 - norm) * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    out[valid] = colored[valid]

    return out





def depth_to_xyz_mm(depth_m: np.ndarray, intr, stride: int = 1):

    h, w = depth_m.shape

    vv, uu = np.mgrid[0:h:stride, 0:w:stride]

    z = depth_m[0:h:stride, 0:w:stride]



    valid = (

        np.isfinite(z)

        & (z >= MIN_DEPTH_M)

        & (z <= MAX_DEPTH_M)

    )

    if not np.any(valid):

        return (

            np.empty((0, 3), dtype=np.float32),

            np.empty((0, 2), dtype=np.int32),

        )



    u = uu[valid].astype(np.float64)

    v = vv[valid].astype(np.float64)

    z_m = z[valid].astype(np.float64)



    x_m = (u - intr.ppx) / intr.fx * z_m

    y_m = (v - intr.ppy) / intr.fy * z_m



    points_mm = np.column_stack((x_m, y_m, z_m)) * 1000.0

    uv = np.column_stack((u, v)).astype(np.int32)

    return points_mm.astype(np.float32), uv





def pixel_to_point_mm(u: int, v: int, depth_m: np.ndarray, intr):

    h, w = depth_m.shape

    if not (0 <= u < w and 0 <= v < h):

        return None



    z = float(depth_m[v, u])

    if not np.isfinite(z) or z < MIN_DEPTH_M or z > MAX_DEPTH_M:

        return None



    x = (u - intr.ppx) / intr.fx * z

    y = (v - intr.ppy) / intr.fy * z

    return np.array([x, y, z], dtype=np.float64) * 1000.0


def point_mm_to_pixel(point_mm: np.ndarray, intr):
    """Project a 3-D camera-frame point [mm] to the aligned RGB image."""
    p = np.asarray(point_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(p).all() or p[2] <= 1e-9:
        return None

    u = intr.fx * (p[0] / p[2]) + intr.ppx
    v = intr.fy * (p[1] / p[2]) + intr.ppy
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def _grid_index_to_tangent_uv_mm(
    x_idx: int,
    y_idx: int,
    shape,
    cell_mm: float,
):
    """Convert processing-grid index (x,y) to tangent coordinates (u,v) [mm]."""
    h, w = shape
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    u_mm = (float(x_idx) - cx) * float(cell_mm)
    v_mm = (cy - float(y_idx)) * float(cell_mm)
    return np.array([u_mm, v_mm], dtype=np.float64)


def _find_single_harris_local_maximum(
    response_pos: np.ndarray,
    valid_mask: np.ndarray,
    cell_mm: float,
):
    """Return exactly one strongest positive local maximum for a query ROI.

    A pixel is a local maximum when it equals the dilation maximum inside a
    configurable neighborhood. Among all such positive maxima, the strongest
    response is kept.
    """
    response = np.asarray(response_pos, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    if response.shape != valid.shape or response.size == 0 or not np.any(valid):
        return None

    work = response.copy()
    work[~valid] = 0.0
    work[~np.isfinite(work)] = 0.0

    if float(np.max(work)) <= 0.0:
        return None

    r = max(1, int(HARRIS2D_LOCAL_MAX_RADIUS_PX))
    k = 2 * r + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    local_max_image = cv2.dilate(work, kernel)

    # Small relative tolerance avoids missing a plateau due to float roundoff.
    tol = max(float(np.max(work)) * 1e-7, float(HARRIS2D_EPS))
    local_max_mask = valid & (work > 0.0) & (work >= (local_max_image - tol))

    ys, xs = np.nonzero(local_max_mask)
    if len(xs) == 0:
        # Positive global maximum is necessarily a valid fallback.
        flat_idx = int(np.argmax(work))
        y_peak, x_peak = np.unravel_index(flat_idx, work.shape)
    else:
        values = work[ys, xs]
        best = int(np.argmax(values))
        y_peak = int(ys[best])
        x_peak = int(xs[best])

    uv_mm = _grid_index_to_tangent_uv_mm(
        x_peak,
        y_peak,
        work.shape,
        cell_mm,
    )

    return {
        "grid_xy": (int(x_peak), int(y_peak)),
        "uv_mm": uv_mm,
        "response": float(work[y_peak, x_peak]),
    }


def _draw_corner_marker_2d(
    image: np.ndarray,
    pixel_xy,
    label: str = "C",
    color=(0, 0, 255),
):
    """Draw a high-visibility corner marker on an OpenCV BGR image."""
    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    cv2.circle(image, (x, y), 10, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(image, (x, y), 9, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        image,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=19,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if label:
        tx = min(max(0, x + 11), max(0, w - 60))
        ty = min(max(16, y - 11), max(16, h - 4))
        cv2.putText(
            image,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )


def show_detected_corners_global(
    points_mm: np.ndarray,
    corner_points_mm,
    query_points_mm=None,
):
    """Show full frozen point cloud with accumulated detected corners.

    Display convention:
      - cyan   : full scene point cloud
      - red    : detected 3-D corners
      - yellow : clicked query points
    """
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    corners = np.asarray(corner_points_mm, dtype=np.float64).reshape(-1, 3)

    if query_points_mm is None:
        queries = np.empty((0, 3), dtype=np.float64)
    else:
        queries = np.asarray(query_points_mm, dtype=np.float64).reshape(-1, 3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.asarray(GLOBAL_CLOUD_COLOR, dtype=np.float64)[None, :],
            (len(pts), 1),
        )
    )

    geoms = [cloud]
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(CAMERA_FRAME_SIZE_MM),
            origin=[0.0, 0.0, 0.0],
        )
    )

    for p in queries:
        if not np.isfinite(p).all():
            continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=float(GLOBAL_QUERY_MARKER_RADIUS_MM)
        )
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(GLOBAL_QUERY_COLOR)
        sphere.translate(p)
        geoms.append(sphere)

    for p in corners:
        if not np.isfinite(p).all():
            continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=float(GLOBAL_CORNER_MARKER_RADIUS_MM)
        )
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(GLOBAL_CORNER_COLOR)
        sphere.translate(p)
        geoms.append(sphere)

    print(
        f"[global corner view] corners={len(corners)} | "
        "red=corner, yellow=query, cyan=scene"
    )

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="Full point cloud | corners=red | queries=yellow",
        width=1280,
        height=900,
    )
    if not ok:
        print("[global corner view] failed to create Open3D window")
        return

    for i, geom in enumerate(geoms):
        vis.add_geometry(geom, reset_bounding_box=(i == 0))

    option = vis.get_render_option()
    if option is not None:
        option.background_color = np.asarray(O3D_BACKGROUND, dtype=np.float64)
        option.point_size = float(GLOBAL_CLOUD_POINT_SIZE)
        option.show_coordinate_frame = False

    vis.run()
    vis.destroy_window()





def show_raw_corner_candidates_global(
    points_mm: np.ndarray,
    raw_corner_points_mm,
    query_points_mm=None,
):
    """Show full scene cloud with raw pre-clustering 3-D corner hypotheses."""
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    raw = np.asarray(raw_corner_points_mm, dtype=np.float64).reshape(-1, 3)

    if query_points_mm is None:
        queries = np.empty((0, 3), dtype=np.float64)
    else:
        queries = np.asarray(query_points_mm, dtype=np.float64).reshape(-1, 3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.asarray(GLOBAL_CLOUD_COLOR, dtype=np.float64)[None, :],
            (len(pts), 1),
        )
    )

    geoms = [cloud]
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(CAMERA_FRAME_SIZE_MM),
            origin=[0.0, 0.0, 0.0],
        )
    )

    for p in queries:
        if not np.isfinite(p).all():
            continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=float(GLOBAL_QUERY_MARKER_RADIUS_MM)
        )
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(RAW_CANDIDATE_QUERY_COLOR)
        sphere.translate(p)
        geoms.append(sphere)

    for p in raw:
        if not np.isfinite(p).all():
            continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=float(RAW_CANDIDATE_MARKER_RADIUS_MM)
        )
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(RAW_CANDIDATE_COLOR)
        sphere.translate(p)
        geoms.append(sphere)

    print(
        f"[raw candidate view] raw corner hypotheses={len(raw)} | "
        "magenta=raw, yellow=queries, cyan=scene"
    )

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="Full point cloud | raw corner hypotheses=magenta | queries=yellow",
        width=1280,
        height=900,
    )
    if not ok:
        print("[raw candidate view] failed to create Open3D window")
        return

    for i, geom in enumerate(geoms):
        vis.add_geometry(geom, reset_bounding_box=(i == 0))

    option = vis.get_render_option()
    if option is not None:
        option.background_color = np.asarray(O3D_BACKGROUND, dtype=np.float64)
        option.point_size = float(GLOBAL_CLOUD_POINT_SIZE)
        option.show_coordinate_frame = False

    vis.run()
    vis.destroy_window()

# ============================================================
# Scene sampling helpers: voxel downsampling + deterministic FPS
# ============================================================


def voxel_downsample_points_mm(points_mm: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """Voxel-downsample a camera-frame point cloud. Units stay in millimeters."""
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    voxel = float(voxel_size_mm)
    if voxel <= 0.0:
        return pts.astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    down = pcd.voxel_down_sample(voxel_size=voxel)
    out = np.asarray(down.points, dtype=np.float64)
    return out.astype(np.float32)


def farthest_point_sampling_indices(points_mm: np.ndarray, num_samples: int) -> np.ndarray:
    """Deterministic Euclidean FPS over an Nx3 point array.

    The first point is chosen as the point farthest from the cloud centroid.
    Each following point maximizes distance to the already-selected set.

    Complexity: O(N*K), where K=min(num_samples, N).
    """
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(np.clip(int(num_samples), 1, n))

    centroid = np.mean(pts, axis=0)
    first = int(np.argmax(np.sum((pts - centroid[None, :]) ** 2, axis=1)))

    selected = np.empty((k,), dtype=np.int64)
    selected[0] = first

    min_d2 = np.sum((pts - pts[first][None, :]) ** 2, axis=1)
    min_d2[first] = 0.0

    for i in range(1, k):
        idx = int(np.argmax(min_d2))
        selected[i] = idx

        d2 = np.sum((pts - pts[idx][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[: i + 1]] = 0.0

    return selected


def _filter_corner_records_by_response(
    records,
    min_response: float = 0.0,
    percentile: float | None = None,
):
    """Drop weak scene-level corner candidates before spatial clustering.

    The percentile is computed over the finite candidate responses in the
    current frozen scene.  For example, percentile=70 keeps approximately
    the top 30% of responses, with ties retained.
    """
    if not records:
        return [], None

    finite_records = []
    responses = []
    absolute_floor = float(min_response)

    for rec in records:
        response = float(rec.get("corner_response", np.nan))
        point = np.asarray(rec.get("corner_point_mm"), dtype=np.float64).reshape(3)
        if not np.isfinite(response) or not np.isfinite(point).all():
            continue
        if response <= absolute_floor:
            continue
        finite_records.append(rec)
        responses.append(response)

    if not finite_records:
        return [], None

    responses = np.asarray(responses, dtype=np.float64)
    if percentile is None:
        percentile_threshold = -np.inf
    else:
        pct = float(np.clip(float(percentile), 0.0, 100.0))
        percentile_threshold = float(np.percentile(responses, pct))

    threshold = max(absolute_floor, percentile_threshold)
    filtered = [
        rec for rec in finite_records
        if float(rec.get("corner_response", -np.inf)) >= threshold
    ]
    return filtered, float(threshold)


def _radius_cluster_corner_records(
    records,
    radius_mm: float,
    min_cluster_size: int = 1,
    outlier_mad_scale: float = 2.5,
    inlier_floor_mm: float = 0.0,
    max_keep: int | None = None,
):
    """Radius-cluster 3-D candidates and return robust cluster centroids.

    Cluster formation is radius-connected (DBSCAN-style connectivity with
    ``eps=radius_mm`` and ``min_samples=1``), implemented with cKDTree so no
    sklearn dependency is required.

    Per cluster:
      1) reject clusters smaller than ``min_cluster_size``;
      2) compute the coordinate-wise median of member 3-D positions;
      3) robustly reject spatial outliers from their radial distance to that
         median using median + K * 1.4826 * MAD;
      4) enforce the same minimum size again after trimming;
      5) use the arithmetic centroid of the retained inliers as the final
         physical corner location.

    Harris response is intentionally NOT used to weight the centroid because a
    larger response is not guaranteed to mean a smaller localization error.
    """
    if not records:
        return [], []

    pts = np.asarray(
        [rec["corner_point_mm"] for rec in records],
        dtype=np.float64,
    ).reshape(-1, 3)
    responses = np.asarray(
        [float(rec.get("corner_response", -np.inf)) for rec in records],
        dtype=np.float64,
    )

    valid = np.isfinite(pts).all(axis=1) & np.isfinite(responses)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return [], []

    pts_valid = pts[valid_indices]
    n = len(valid_indices)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    r = max(float(radius_mm), 0.0)
    if r > 0.0 and n > 1:
        tree = cKDTree(pts_valid)
        for a, b in tree.query_pairs(r=r, output_type="set"):
            union(a, b)

    components = {}
    for local_i in range(n):
        root = find(local_i)
        components.setdefault(root, []).append(local_i)

    min_size = max(1, int(min_cluster_size))
    mad_scale = max(0.0, float(outlier_mad_scale))
    inlier_floor = max(0.0, float(inlier_floor_mm))

    cluster_infos = []
    representatives = []

    for local_members in components.values():
        if len(local_members) < min_size:
            continue

        original_members = valid_indices[np.asarray(local_members, dtype=np.int64)]
        member_pts = pts[original_members]
        member_responses = responses[original_members]

        # Robust center used ONLY to decide which detections are spatial outliers.
        median_center = np.median(member_pts, axis=0)
        radial_dist = np.linalg.norm(member_pts - median_center[None, :], axis=1)
        median_dist = float(np.median(radial_dist))
        mad_dist = float(np.median(np.abs(radial_dist - median_dist)))
        robust_sigma = 1.4826 * mad_dist

        inlier_threshold = median_dist + mad_scale * robust_sigma
        inlier_threshold = max(inlier_threshold, inlier_floor)
        if r > 0.0:
            inlier_threshold = min(inlier_threshold, r)

        inlier_local_mask = radial_dist <= (inlier_threshold + 1e-12)
        inlier_members = original_members[inlier_local_mask]
        if len(inlier_members) < min_size:
            continue

        inlier_pts = pts[inlier_members]
        inlier_responses = responses[inlier_members]
        centroid = np.mean(inlier_pts, axis=0)

        centroid_dist = np.linalg.norm(inlier_pts - centroid[None, :], axis=1)
        spread_rms = float(np.sqrt(np.mean(centroid_dist ** 2)))
        spread_max = float(np.max(centroid_dist)) if len(centroid_dist) else 0.0

        # Preserve one real member record only as metadata container.  The final
        # 3-D position itself is replaced by the robust cluster centroid.
        best_inlier_idx = int(inlier_members[int(np.argmax(inlier_responses))])
        rep = dict(records[best_inlier_idx])
        rep["representative_member_index"] = int(best_inlier_idx)
        rep["representative_member_point_mm"] = np.asarray(
            records[best_inlier_idx]["corner_point_mm"], dtype=np.float64
        ).copy()
        rep["corner_point_mm"] = centroid.astype(np.float64)
        rep["corner_point_source"] = "robust_cluster_centroid"
        rep["corner_response"] = float(np.max(inlier_responses))
        rep["cluster_size"] = int(len(original_members))
        rep["cluster_inlier_count"] = int(len(inlier_members))
        rep["cluster_outlier_count"] = int(len(original_members) - len(inlier_members))
        rep["cluster_radius_mm"] = float(r)
        rep["cluster_inlier_threshold_mm"] = float(inlier_threshold)
        rep["cluster_median_center_mm"] = median_center.astype(np.float64)
        rep["cluster_centroid_mm"] = centroid.astype(np.float64)
        rep["cluster_spread_rms_mm"] = float(spread_rms)
        rep["cluster_spread_max_mm"] = float(spread_max)
        rep["cluster_response_max"] = float(np.max(inlier_responses))
        rep["cluster_response_median"] = float(np.median(inlier_responses))
        rep["cluster_member_indices"] = [int(i) for i in original_members]
        rep["cluster_inlier_indices"] = [int(i) for i in inlier_members]

        cluster_infos.append({
            "member_indices": [int(i) for i in original_members],
            "inlier_indices": [int(i) for i in inlier_members],
            "size": int(len(original_members)),
            "inlier_count": int(len(inlier_members)),
            "outlier_count": int(len(original_members) - len(inlier_members)),
            "representative_member_index": int(best_inlier_idx),
            "median_center_mm": median_center.astype(np.float64),
            "centroid_mm": centroid.astype(np.float64),
            "inlier_threshold_mm": float(inlier_threshold),
            "spread_rms_mm": float(spread_rms),
            "spread_max_mm": float(spread_max),
            "response_max": float(np.max(inlier_responses)),
            "response_median": float(np.median(inlier_responses)),
        })
        representatives.append(rep)

    # Ranking affects only display / max_keep truncation, not centroid position.
    # Prefer clusters supported by more repeated detections, then stronger response.
    order = sorted(
        range(len(representatives)),
        key=lambda i: (
            -int(representatives[i].get("cluster_inlier_count", 0)),
            -float(representatives[i].get("cluster_response_max", -np.inf)),
        ),
    )
    representatives = [representatives[int(i)] for i in order]
    cluster_infos = [cluster_infos[int(i)] for i in order]

    if max_keep is not None:
        k = max(0, int(max_keep))
        representatives = representatives[:k]
        cluster_infos = cluster_infos[:k]

    return representatives, cluster_infos


# ============================================================

# Surface-aware normals + Gaussian Harris3D

# ============================================================



def estimate_surface_aware_normals_mm(
    points_mm: np.ndarray,
    radius_mm: float | None = None,
) -> np.ndarray:

    pts = np.asarray(points_mm, dtype=np.float32)

    if len(pts) == 0:

        return np.empty((0, 3), dtype=np.float32)



    if radius_mm is None:
        radius_mm = float(NORMAL_RADIUS_MM)
    radius_mm = float(radius_mm)

    kwargs = dict(
        radius=radius_mm,

        sigma_e_deg=NORMAL_SIGMA_E_DEG,

        sigma_r=NORMAL_SIGMA_R,

        max_iterations=NORMAL_MAX_ITERATIONS,

    )



    try:

        normals = sah.estimate_surface_aware_normals(

            pts,

            update_mode=NORMAL_UPDATE_MODE,

            **kwargs,

        )

    except TypeError:

        normals = sah.estimate_surface_aware_normals(pts, **kwargs)



    normals = np.asarray(normals, dtype=np.float64)

    if normals.shape != pts.shape:

        raise RuntimeError(

            "estimate_surface_aware_normals returned unexpected shape: "

            f"{normals.shape}, expected {pts.shape}"

        )



    lengths = np.linalg.norm(normals, axis=1)

    good = np.isfinite(normals).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)



    out = np.full_like(normals, np.nan)

    out[good] = normals[good] / lengths[good, None]

    return out.astype(np.float32)





def compute_harris_response(points_mm: np.ndarray, normals: np.ndarray) -> np.ndarray:

    response = sah.gaussian_harris_response(

        np.asarray(points_mm, dtype=np.float64),

        np.asarray(normals, dtype=np.float64),

        radius=HARRIS_RADIUS_MM,

        sigma=GAUSSIAN_SIGMA_MM,

    )

    response = np.asarray(response, dtype=np.float64).reshape(-1)

    if len(response) != len(points_mm):

        raise RuntimeError(

            f"gaussian_harris_response returned {len(response)} values for "

            f"{len(points_mm)} points"

        )

    return response





# ============================================================

# Harris / normal Open3D viewer (same behavior as second code)

# ============================================================



def turbo_colors(values, vmin, vmax):

    values = np.asarray(values, dtype=np.float64)

    scale = max(float(vmax) - float(vmin), 1e-15)

    x = np.clip((values - float(vmin)) / scale, 0.0, 1.0)



    kr = np.array([

        0.13572138, 4.61539260, -42.66032258,

        132.13108234, -152.94239396, 59.28637943,

    ])

    kg = np.array([

        0.09140261, 2.19418839, 4.84296658,

        -14.18503333, 4.27729857, 2.82956604,

    ])

    kb = np.array([

        0.10667330, 12.64194608, -60.58204836,

        110.36276771, -89.90310912, 27.34824973,

    ])



    X = np.stack([

        np.ones_like(x), x, x**2, x**3, x**4, x**5,

    ], axis=1)

    rgb = np.c_[X @ kr, X @ kg, X @ kb]

    return np.clip(rgb, 0.0, 1.0)





def choose_heatmap_range(response):

    finite = response[np.isfinite(response)]

    if len(finite) == 0:

        return 0.0, 1.0



    vmin = float(HEATMAP_MIN) if HEATMAP_MIN is not None else float(np.percentile(finite, 1.0))

    vmax = float(HEATMAP_MAX) if HEATMAP_MAX is not None else float(np.percentile(finite, 99.5))

    if vmax <= vmin:

        vmax = vmin + 1e-12

    return vmin, vmax





def make_normal_lines(

    points_mm,

    normals,

    stride=NORMAL_VIS_STRIDE,

    length_mm=NORMAL_VIS_LENGTH_MM,

):

    points_mm = np.asarray(points_mm, dtype=np.float64)

    normals = np.asarray(normals, dtype=np.float64)



    valid = np.all(np.isfinite(points_mm), axis=1) & np.all(np.isfinite(normals), axis=1)

    idx = np.flatnonzero(valid)[::max(1, int(stride))]

    if len(idx) == 0:

        return o3d.geometry.LineSet()



    p0 = points_mm[idx]

    n = normals[idx]

    norm = np.linalg.norm(n, axis=1, keepdims=True)

    good = np.squeeze(norm > 1e-12)

    p0 = p0[good]

    n = n[good] / norm[good]

    p1 = p0 + float(length_mm) * n



    vertices = np.empty((2 * len(p0), 3), dtype=np.float64)

    vertices[0::2] = p0

    vertices[1::2] = p1



    lines = np.column_stack((

        np.arange(0, 2 * len(p0), 2, dtype=np.int32),

        np.arange(1, 2 * len(p0), 2, dtype=np.int32),

    ))

    colors = np.tile(

        np.asarray(NORMAL_VIS_COLOR, dtype=np.float64)[None, :],

        (len(lines), 1),

    )



    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(vertices)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set





def show_harris_heatmap(points_mm, response, normals):

    vmin, vmax = choose_heatmap_range(response)

    heat_colors = turbo_colors(response, vmin=vmin, vmax=vmax)

    gray_colors = np.tile(

        np.array([[0.55, 0.55, 0.55]], dtype=np.float64),

        (len(points_mm), 1),

    )



    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_mm, dtype=np.float64))

    cloud.colors = o3d.utility.Vector3dVector(heat_colors)



    valid_normals = np.asarray(normals, dtype=np.float64).copy()

    valid_normals[~np.isfinite(valid_normals)] = 0.0

    cloud.normals = o3d.utility.Vector3dVector(valid_normals)



    normal_lines = make_normal_lines(points_mm, normals)



    finite = response[np.isfinite(response)]

    print("\n" + "=" * 80)

    print("GAUSSIAN HARRIS3D + ESTIMATED NORMAL DIAGNOSTIC")

    print("=" * 80)

    if len(finite):

        print(f"min       : {np.min(finite):.10g}")

        print(f"p01       : {np.percentile(finite, 1):.10g}")

        print(f"median    : {np.median(finite):.10g}")

        print(f"mean      : {np.mean(finite):.10g}")

        print(f"p95       : {np.percentile(finite, 95):.10g}")

        print(f"p99       : {np.percentile(finite, 99):.10g}")

        print(f"p99.5     : {np.percentile(finite, 99.5):.10g}")

        print(f"max       : {np.max(finite):.10g}")

    print(f"heatmap   : [{vmin:.10g}, {vmax:.10g}]")

    print(f"normal vis: every {NORMAL_VIS_STRIDE}-th valid point")

    print(f"normal len: {NORMAL_VIS_LENGTH_MM:.3f} mm")

    print("Open3D controls: 1=Harris, 2=Normals, 3=Both, R=reset")



    vis = o3d.visualization.VisualizerWithKeyCallback()

    ok = vis.create_window(

        window_name=(

            "Harris3D / Surface-aware normals | "

            "1:Harris  2:Normals  3:Both  R:reset"

        ),

        width=1280,

        height=900,

    )

    if not ok:

        print("[Harris viewer] failed to create window")

        return



    vis.add_geometry(cloud)

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(

        size=float(CAMERA_FRAME_SIZE_MM),

        origin=[0.0, 0.0, 0.0],

    )

    vis.add_geometry(camera_frame)



    state = {"normals_added": False}



    def add_normals():

        if not state["normals_added"]:

            vis.add_geometry(normal_lines, reset_bounding_box=False)

            state["normals_added"] = True



    def remove_normals():

        if state["normals_added"]:

            vis.remove_geometry(normal_lines, reset_bounding_box=False)

            state["normals_added"] = False



    def set_harris(v):

        cloud.colors = o3d.utility.Vector3dVector(heat_colors)

        v.update_geometry(cloud)

        remove_normals()

        print("[view] Harris response")

        return False



    def set_normals(v):

        cloud.colors = o3d.utility.Vector3dVector(gray_colors)

        v.update_geometry(cloud)

        add_normals()

        print("[view] Estimated surface-aware normals")

        return False



    def set_both(v):

        cloud.colors = o3d.utility.Vector3dVector(heat_colors)

        v.update_geometry(cloud)

        add_normals()

        print("[view] Harris response + normals")

        return False



    def reset_view(v):

        v.reset_view_point(True)

        return False



    vis.register_key_callback(ord("1"), set_harris)

    vis.register_key_callback(ord("2"), set_normals)

    vis.register_key_callback(ord("3"), set_both)

    vis.register_key_callback(ord("R"), reset_view)



    option = vis.get_render_option()

    if option is not None:

        option.background_color = np.array([0.04, 0.04, 0.04])

        option.point_size = float(POINT_SIZE)

        option.show_coordinate_frame = False



    print(

        "camera frame : origin=(0,0,0) mm | "

        "+X right (red), +Y down (green), +Z forward (blue)"

    )



    vis.run()

    vis.destroy_window()





# ============================================================

# Tangent frame

# ============================================================



def make_tangent_basis(normal: np.ndarray):

    n = normalize(normal)

    if abs(n[2]) < 0.9:

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    else:

        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)



    u = normalize(np.cross(ref, n))

    v = normalize(np.cross(n, u))

    return u, v, n





def project_to_tangent_coordinates(

    points_mm: np.ndarray,

    query_point_mm: np.ndarray,

    query_normal: np.ndarray,

):

    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)

    q = np.asarray(query_point_mm, dtype=np.float64).reshape(3)

    u, v, n = make_tangent_basis(query_normal)



    delta = pts - q[None, :]

    uu = delta @ u

    vv = delta @ v

    hh = delta @ n

    uvh = np.column_stack((uu, vv, hh))

    return uvh, u, v, n





def normal_similarity_weights(normals: np.ndarray, query_normal: np.ndarray):

    ns = np.asarray(normals, dtype=np.float64).reshape(-1, 3)

    qn = normalize(query_normal)



    lengths = np.linalg.norm(ns, axis=1)

    valid = (

        np.isfinite(ns).all(axis=1)

        & np.isfinite(lengths)

        & (lengths > 1e-12)

    )



    unit = np.zeros_like(ns)

    unit[valid] = ns[valid] / lengths[valid, None]



    cosang = np.zeros((len(ns),), dtype=np.float64)

    cosang[valid] = np.clip(np.abs(unit[valid] @ qn), 0.0, 1.0)



    theta_deg = np.full((len(ns),), np.nan, dtype=np.float64)

    theta_deg[valid] = np.degrees(np.arccos(cosang[valid]))



    sigma = max(float(TANGENT_NORMAL_WEIGHT_SIGMA_DEG), 1e-9)

    weights = np.zeros((len(ns),), dtype=np.float64)

    weights[valid] = np.exp(-0.5 * (theta_deg[valid] / sigma) ** 2)

    return weights, theta_deg, valid





# ============================================================

# 3-D local tangent-coordinate visualization

# ============================================================



def make_tangent_plane_grid_local(radius_mm: float, spacing_mm: float | None = None):

    radius_mm = float(radius_mm)

    if spacing_mm is None:

        spacing_mm = max(5.0, 0.2 * radius_mm)



    coords = np.arange(-radius_mm, radius_mm + 0.5 * spacing_mm, spacing_mm)

    points = []

    lines = []

    colors = []



    grid_color = [0.78, 0.72, 0.18]



    def add_line(a, b, color):

        idx = len(points)

        points.extend([a, b])

        lines.append([idx, idx + 1])

        colors.append(color)



    for s in coords:

        add_line([-radius_mm, s, 0.0], [radius_mm, s, 0.0], grid_color)

        add_line([s, -radius_mm, 0.0], [s, radius_mm, 0.0], grid_color)



    # Explicit axes in tangent coordinates:

    # u = red, v = green, h = blue.

    add_line([-radius_mm, 0.0, 0.0], [radius_mm, 0.0, 0.0], [1.0, 0.2, 0.2])

    add_line([0.0, -radius_mm, 0.0], [0.0, radius_mm, 0.0], [0.2, 1.0, 0.2])

    add_line([0.0, 0.0, -0.65 * radius_mm], [0.0, 0.0, 0.65 * radius_mm], [0.2, 0.45, 1.0])



    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))

    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))

    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

    return line_set





def show_local_tangent_3d(

    uvh: np.ndarray,

    weights: np.ndarray,

    radius_mm: float,

):

    uvh = np.asarray(uvh, dtype=np.float64).reshape(-1, 3)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)



    finite = np.isfinite(uvh).all(axis=1) & np.isfinite(weights)

    uvh = uvh[finite]

    weights = weights[finite]



    if len(uvh) == 0:

        print("[tangent 3D] no valid local points")

        return



    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(uvh)

    pcd.colors = o3d.utility.Vector3dVector(turbo_colors(weights, 0.0, 1.0))



    plane_grid = make_tangent_plane_grid_local(radius_mm)



    query = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.8, 0.025 * radius_mm))

    query.translate([0.0, 0.0, 0.0])

    query.paint_uniform_color([1.0, 1.0, 1.0])

    query.compute_vertex_normals()



    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(

        size=max(8.0, 0.35 * radius_mm),

        origin=[0.0, 0.0, 0.0],

    )



    vis = o3d.visualization.Visualizer()

    ok = vis.create_window(

        window_name="Local tangent coordinates | X=u, Y=v, Z=h | color=normal weight",

        width=1280,

        height=850,

    )

    if not ok:

        print("[tangent 3D] failed to create window")

        return



    vis.add_geometry(pcd)

    vis.add_geometry(plane_grid, reset_bounding_box=False)

    vis.add_geometry(query, reset_bounding_box=False)

    vis.add_geometry(frame, reset_bounding_box=False)



    option = vis.get_render_option()

    if option is not None:

        option.background_color = O3D_BACKGROUND

        option.point_size = 4.0

        option.show_coordinate_frame = False



    ctr = vis.get_view_control()

    if ctr is not None:

        ctr.set_lookat([0.0, 0.0, 0.0])

        ctr.set_front([0.8, -0.8, -0.6])

        ctr.set_up([0.0, 0.0, 1.0])

        ctr.set_zoom(0.72)



    print("\n[Local tangent 3D]")

    print("  X axis = u")

    print("  Y axis = v")

    print("  Z axis = h = signed distance from tangent plane")

    print("  plane  = h = 0")

    print("  point color = normal-similarity weight [0,1]")



    vis.run()

    vis.destroy_window()





# ============================================================

# 2-D normal-weighted projection + direct local-mean fill + bilateral smoothing

# ============================================================



def _adaptive_normal_radius_mm(query_point_mm: np.ndarray):
    """Depth-adaptive click-stage radius for surface-aware normal estimation.

    Keeps the radius-based formulation and compensates for the approximate
    linear growth of physical RGB-D sample spacing with camera depth:

        r_n(Z) = r_ref * Z / Z_ref
    """
    q = np.asarray(query_point_mm, dtype=np.float64).reshape(3)
    z_mm = (
        float(abs(q[2]))
        if np.isfinite(q[2])
        else float(NORMAL_RADIUS_REFERENCE_Z_MM)
    )
    z_ref = max(float(NORMAL_RADIUS_REFERENCE_Z_MM), 1e-9)

    scale = z_mm / z_ref
    radius = float(NORMAL_RADIUS_REFERENCE_MM) * scale
    radius = float(np.clip(
        radius,
        float(NORMAL_RADIUS_MIN_MM),
        float(NORMAL_RADIUS_MAX_MM),
    ))
    return radius, float(scale)


def _adaptive_inspect_radius_mm(
    query_point_mm: np.ndarray,
    base_radius_mm: float,
):
    """Depth-adaptive local inspection radius.

    RGB-D cameras have approximately fixed angular/pixel sampling, so physical
    point spacing grows roughly in proportion to camera depth Z. Keep a similar
    image/angular ROI using

        r_eff = r_ref * Z / Z_ref

    then clamp to a safe physical range.
    """
    q = np.asarray(query_point_mm, dtype=np.float64).reshape(3)
    z_mm = float(abs(q[2])) if np.isfinite(q[2]) else float(INSPECT_RADIUS_REFERENCE_Z_MM)
    z_ref = max(float(INSPECT_RADIUS_REFERENCE_Z_MM), 1e-9)

    scale = z_mm / z_ref
    radius = float(base_radius_mm) * scale
    radius = float(np.clip(
        radius,
        float(INSPECT_RADIUS_EFFECTIVE_MIN_MM),
        float(INSPECT_RADIUS_EFFECTIVE_MAX_MM),
    ))
    return radius, float(scale)


def _tangent_uv_to_pixel(uv: np.ndarray, radius_mm: float):

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    drawable = TANGENT_VIEW_SIZE - 2 * TANGENT_VIEW_MARGIN

    scale = drawable / max(2.0 * float(radius_mm), 1e-9)



    px = TANGENT_VIEW_MARGIN + (uv[:, 0] + radius_mm) * scale

    py = TANGENT_VIEW_SIZE - TANGENT_VIEW_MARGIN - (uv[:, 1] + radius_mm) * scale

    return np.column_stack((px, py)).round().astype(np.int32)





def _estimate_tangent_sampling_spacing_mm(uv: np.ndarray, k: int = LOCAL_SPACING_K) -> float:
    """Robust characteristic projected sampling spacing in the tangent plane.

    Uses spatial kNN only.  For each point, take the median distance to its k
    nearest non-self neighbors, then take the median over all points.  This is
    more representative than 1-NN for scanline-like RGB-D sampling.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = len(uv)
    if n < 3:
        return float(TANGENT_SPACING_FALLBACK_MM)

    kk = int(max(2, min(int(k), n - 1)))
    try:
        tree = cKDTree(uv)
        dists, _ = tree.query(uv, k=kk + 1)
        local = np.asarray(dists[:, 1:kk + 1], dtype=np.float64)
        local[~np.isfinite(local)] = np.nan
        local[local <= 1e-9] = np.nan
        per_point = np.nanmedian(local, axis=1)
        per_point = per_point[np.isfinite(per_point) & (per_point > 1e-9)]
        if len(per_point) == 0:
            return float(TANGENT_SPACING_FALLBACK_MM)
        return float(np.median(per_point))
    except Exception:
        return float(TANGENT_SPACING_FALLBACK_MM)


def _make_processing_roi_mask(shape, radius_mm: float, cell_mm: float):
    h, w = shape
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    yy, xx = np.mgrid[0:h, 0:w]
    du = (xx - cx) * float(cell_mm)
    dv = (cy - yy) * float(cell_mm)
    return (du * du + dv * dv) <= float(radius_mm) ** 2


def _make_weighted_support_map(uv: np.ndarray, weights: np.ndarray, radius_mm: float):
    """Rasterize the tangent normal-weight samples into a sampling-adaptive field.

    Active path:
        projected (u,v,w)
        -> estimate characteristic spacing s
        -> rasterize with cell ~= GRID_CELL_SCALE * s
        -> direct local-mean fill of empty cells with sufficient ORIGINAL support
        -> bilateral denoising

    No morphology closing and no Gaussian point splatting are used.
    """

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if len(uv) == 0:
        grid_n = int(GRID_SIZE_MIN)
        support = np.zeros((grid_n, grid_n), dtype=np.float32)
        occupied = np.zeros_like(support, dtype=bool)
        meta = {
            'spacing_mm': float(TANGENT_SPACING_FALLBACK_MM),
            'cell_mm': 2.0 * float(radius_mm) / max(grid_n - 1, 1),
            'grid_size': grid_n,
            'initial_valid_frac': 0.0,
            'final_valid_frac': 0.0,
            'filled_frac': 0.0,
            'num_filled_px': 0,
            'mean_radius_mm': 0.0,
            'mean_radius_px': 0,
            'bilateral_sigma_space_mm': 0.0,
            'bilateral_sigma_space_px': 0.0,
        }
        return support, occupied, meta

    # 1) Estimate the current tangent-plane sampling spacing directly from (u,v).
    spacing_mm = _estimate_tangent_sampling_spacing_mm(uv, LOCAL_SPACING_K)
    if not np.isfinite(spacing_mm) or spacing_mm <= 1e-9:
        spacing_mm = float(TANGENT_SPACING_FALLBACK_MM)

    # 2) Adapt the internal raster resolution to that spacing.
    requested_cell_mm = max(
        1e-6,
        float(GRID_CELL_SCALE) * float(spacing_mm),
    )
    grid_n = int(np.ceil((2.0 * float(radius_mm)) / max(requested_cell_mm, 1e-9))) + 1
    grid_n = int(np.clip(grid_n, int(GRID_SIZE_MIN), int(GRID_SIZE_MAX)))
    # Use the exact physical cell size implied by the final grid size.
    cell_mm = (2.0 * float(radius_mm)) / max(grid_n - 1, 1)
    pixels_per_mm = 1.0 / max(cell_mm, 1e-9)

    support = np.zeros((grid_n, grid_n), dtype=np.float32)
    occupied = np.zeros((grid_n, grid_n), dtype=np.uint8)

    # 3) Rasterize measured weights. Multiple points in one cell are averaged.
    acc = np.zeros((grid_n, grid_n), dtype=np.float32)
    cnt = np.zeros((grid_n, grid_n), dtype=np.float32)

    x = (uv[:, 0] + float(radius_mm)) * pixels_per_mm
    y = (grid_n - 1) - (uv[:, 1] + float(radius_mm)) * pixels_per_mm
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)

    inside = (
        (xi >= 0) & (xi < grid_n)
        & (yi >= 0) & (yi < grid_n)
        & np.isfinite(weights)
    )
    for xx, yy, ww in zip(xi[inside], yi[inside], weights[inside]):
        acc[yy, xx] += float(ww)
        cnt[yy, xx] += 1.0

    original_mask = cnt > 0
    support[original_mask] = acc[original_mask] / np.maximum(cnt[original_mask], 1e-12)
    occupied[original_mask] = 1

    # 4) Direct local-mean fill geometry.
    #
    #   r_mean = MEAN_MEAN_FILL_RADIUS_SCALE * s
    #
    # No morphology pre-selection: every empty cell is considered directly.
    # Only ORIGINAL valid raster cells contribute to the mean.
    mean_radius_mm = float(MEAN_MEAN_FILL_RADIUS_SCALE) * float(spacing_mm)
    mean_radius_px = max(
        1,
        int(round(mean_radius_mm / max(cell_mm, 1e-9))),
    )

    # Pixel-domain bilateral scale.
    sigma_space_px = max(
        0.10,
        float(PIXEL_TUNING["bilateral_sigma_space_px"]),
    )
    bilateral_sigma_space_mm = float(sigma_space_px) * float(cell_mm)

    # Computational safety clamp.
    mean_radius_px = min(mean_radius_px, max(1, grid_n // 8))

    # 5) Direct local mean over ORIGINAL valid raster samples.
    # Circular kernel: the configured value is a true radius in pixel space.
    r = int(mean_radius_px)
    yy_k, xx_k = np.ogrid[-r:r + 1, -r:r + 1]
    mean_kernel = (
        (xx_k * xx_k + yy_k * yy_k) <= (r * r)
    ).astype(np.float32)

    original_float = original_mask.astype(np.float32)

    weighted_sum = cv2.filter2D(
        support * original_float,
        ddepth=-1,
        kernel=mean_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    valid_count = cv2.filter2D(
        original_float,
        ddepth=-1,
        kernel=mean_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    local_mean = weighted_sum / np.maximum(valid_count, 1e-12)

    # Every empty cell is eligible in principle.
    # Fill only when enough ORIGINAL valid cells exist in its kernel.
    fillable = (
        (~original_mask)
        & (valid_count >= float(MEAN_FILL_MIN_VALID_NEIGHBORS))
    )

    filled = support.copy()
    filled[fillable] = local_mean[fillable]
    final_mask = original_mask | fillable

    # 6) Edge-preserving denoise.  d=0 lets OpenCV derive the support size from
    # sigmaSpace, which is now expressed in local-spacing units rather than px.
    filtered = cv2.bilateralFilter(
        filled.astype(np.float32),
        d=0,
        sigmaColor=float(BILATERAL_SIGMA_COLOR),
        sigmaSpace=float(sigma_space_px),
        borderType=cv2.BORDER_REPLICATE,
    )
    filtered[~final_mask] = 0.0

    meta = {
        'spacing_mm': float(spacing_mm),
        'cell_mm': float(cell_mm),
        'grid_size': int(grid_n),
        'initial_valid_frac': float(np.mean(original_mask)),
        'final_valid_frac': float(np.mean(final_mask)),
        'filled_frac': float(np.mean(fillable)),
        'num_filled_px': int(np.count_nonzero(fillable)),
        'mean_radius_mm': float(mean_radius_mm),
        'mean_radius_px': int(mean_radius_px),
        'bilateral_sigma_space_mm': float(bilateral_sigma_space_mm),
        'bilateral_sigma_space_px': float(sigma_space_px),
    }
    return filtered.astype(np.float32), final_mask.astype(bool), meta



def _smooth_harris_inner_boundary(
    intensity: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Smooth only a narrow band INSIDE the circular ROI.

    Outside-ROI values are never synthesized. The smoothing itself is a
    mask-normalized Gaussian, so only ROI-inside intensity contributes.

    The retained control is:
        inner_band = HARRIS_PAD_INNER_SMOOTH_BAND_RATIO * sigma_H
    """
    I = np.asarray(intensity, dtype=np.float32)
    roi = np.asarray(roi_mask, dtype=bool)

    if I.shape != roi.shape:
        raise ValueError(
            f"inner smoothing shape mismatch: {I.shape} vs {roi.shape}"
        )
    if not np.any(roi):
        return np.zeros_like(I, dtype=np.float32)

    sigma_h_px = max(0.10, float(PIXEL_TUNING["harris_sigma_px"]))
    inner_band_px = max(
        1.0,
        float(HARRIS_PAD_INNER_SMOOTH_BAND_RATIO) * sigma_h_px,
    )

    roi_f = roi.astype(np.float32)

    # ROI-normalized smoothing: outside ROI contributes zero WEIGHT, not zero value.
    numerator = cv2.GaussianBlur(
        I * roi_f,
        (0, 0),
        sigmaX=sigma_h_px,
        sigmaY=sigma_h_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.GaussianBlur(
        roi_f,
        (0, 0),
        sigmaX=sigma_h_px,
        sigmaY=sigma_h_px,
        borderType=cv2.BORDER_CONSTANT,
    )
    smooth_inside = numerator / np.maximum(
        denominator,
        float(HARRIS2D_EPS),
    )

    inside_dist = distance_transform_edt(roi).astype(np.float32)
    d_from_boundary = np.maximum(inside_dist - 1.0, 0.0)
    inner_band = roi & (d_from_boundary <= float(inner_band_px))

    # Strongest blend at the ROI boundary; smoothly fades inward.
    alpha = np.exp(
        -0.5 * (d_from_boundary / max(sigma_h_px, 1e-6)) ** 2
    ).astype(np.float32)
    alpha[~inner_band] = 0.0

    result = np.zeros_like(I, dtype=np.float32)
    result[roi] = I[roi]
    result[inner_band] = (
        (1.0 - alpha[inner_band]) * I[inner_band]
        + alpha[inner_band] * smooth_inside[inner_band]
    )
    return result.astype(np.float32)


def _compute_masked_harris2d_response(
    intensity_norm: np.ndarray,
    circular_roi_mask: np.ndarray,
    cell_mm: float,
    spacing_mm: float,
):
    """Harris2D with Sobel-valid masking + normalized Gaussian integration.

    Boundary handling
    -----------------
    1) NO outer padding, reflection, or synthetic continuation.
    2) Keep a narrow smoothed band INSIDE the circular ROI.
    3) Trust Sobel gradients only where the full Sobel kernel lies inside ROI.
    4) Integrate the structure tensor using a mask-normalized Gaussian:

           Ixx = G(m * Ix^2) / (G(m) + eps)
           Iyy = G(m * Iy^2) / (G(m) + eps)
           Ixy = G(m * IxIy) / (G(m) + eps)

    Internal unreconstructed holes remain intensity 0, preserving the previous
    semantics inside the ROI.

    Returns
    -------
    response_pos : (H,W) float32
        Positive Harris corner response.
    valid_mask : (H,W) bool
        ROI positions with non-negligible Sobel-valid Gaussian support.
    meta : dict
        Physical/pixel diagnostics.
    work : (H,W) float32
        Exact ROI-only intensity image used for the Sobel stage.
    eigen_ratio : (H,W) float32
        Structure-tensor balance rho=lambda_min/(lambda_max+eps), in [0,1].
        Edge-like pixels approach 0; isotropic corner-like pixels approach 1.
    """
    I = np.asarray(intensity_norm, dtype=np.float32)
    roi = np.asarray(circular_roi_mask, dtype=bool)

    if I.shape != roi.shape:
        raise ValueError(
            f"Harris intensity/mask shape mismatch: {I.shape} vs {roi.shape}"
        )

    # Requested inner boundary smoothing; still uses ROI data only.
    work = _smooth_harris_inner_boundary(I, roi)

    cell_mm = max(float(cell_mm), 1e-9)
    spacing_mm = max(float(spacing_mm), cell_mm)

    sobel_ksize = int(HARRIS2D_SOBEL_KSIZE)
    if sobel_ksize < 3 or sobel_ksize % 2 == 0:
        raise ValueError("HARRIS2D_SOBEL_KSIZE must be an odd integer >= 3")

    # A gradient is valid only if its complete Sobel footprint is inside ROI.
    sobel_kernel = np.ones((sobel_ksize, sobel_ksize), dtype=np.uint8)
    gradient_valid = cv2.erode(
        roi.astype(np.uint8),
        sobel_kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    # Keep the existing physical-gradient scale for the configured 3x3 Sobel.
    if sobel_ksize == 3:
        sobel_scale = 1.0 / (8.0 * cell_mm)
    else:
        sobel_scale = 1.0 / cell_mm

    Ix = cv2.Sobel(
        work,
        cv2.CV_32F,
        1,
        0,
        ksize=sobel_ksize,
        scale=sobel_scale,
        borderType=cv2.BORDER_CONSTANT,
    )
    Iy = cv2.Sobel(
        work,
        cv2.CV_32F,
        0,
        1,
        ksize=sobel_ksize,
        scale=sobel_scale,
        borderType=cv2.BORDER_CONSTANT,
    )

    sigma_px = max(0.10, float(PIXEL_TUNING["harris_sigma_px"]))
    sigma_mm = float(sigma_px) * float(cell_mm)

    m = gradient_valid.astype(np.float32)

    # G(m): amount of the Gaussian window supported by valid gradients.
    gaussian_support = cv2.GaussianBlur(
        m,
        (0, 0),
        sigmaX=sigma_px,
        sigmaY=sigma_px,
        borderType=cv2.BORDER_CONSTANT,
    )

    def normalized_gaussian(field: np.ndarray) -> np.ndarray:
        numerator = cv2.GaussianBlur(
            np.asarray(field, dtype=np.float32) * m,
            (0, 0),
            sigmaX=sigma_px,
            sigmaY=sigma_px,
            borderType=cv2.BORDER_CONSTANT,
        )
        return numerator / np.maximum(
            gaussian_support,
            float(HARRIS2D_EPS),
        )

    Ixx = normalized_gaussian(Ix * Ix)
    Iyy = normalized_gaussian(Iy * Iy)
    Ixy = normalized_gaussian(Ix * Iy)

    det_m = Ixx * Iyy - Ixy * Ixy
    trace_m = Ixx + Iyy
    response = det_m - float(HARRIS2D_K) * trace_m * trace_m

    # Structure-tensor eigenvalue balance. For the symmetric 2x2 tensor
    # [[Ixx, Ixy], [Ixy, Iyy]], the eigenvalues are available analytically.
    # This rejects long edge/ridge responses that can still have positive Harris R.
    discr = np.sqrt(np.maximum((Ixx - Iyy) ** 2 + 4.0 * Ixy * Ixy, 0.0))
    lambda_max = 0.5 * (trace_m + discr)
    lambda_min = 0.5 * (trace_m - discr)
    lambda_max = np.maximum(lambda_max, 0.0)
    lambda_min = np.maximum(lambda_min, 0.0)

    eigen_ratio = np.zeros_like(response, dtype=np.float32)
    energetic = lambda_max > float(HARRIS2D_EPS)
    eigen_ratio[energetic] = (
        lambda_min[energetic]
        / np.maximum(lambda_max[energetic], float(HARRIS2D_EPS))
    ).astype(np.float32)
    eigen_ratio = np.clip(eigen_ratio, 0.0, 1.0)
    eigen_ratio[~np.isfinite(eigen_ratio)] = 0.0

    # The normalized Gaussian allows evaluation near the ROI boundary without
    # inventing outside values. Reject only numerically unsupported locations.
    valid_mask = roi & (gaussian_support > 1e-6)

    response_pos = np.maximum(response, 0.0)
    response_pos[~valid_mask] = 0.0
    response_pos[~np.isfinite(response_pos)] = 0.0

    effective_support_sigma = float(HARRIS2D_EFFECTIVE_SUPPORT_SIGMA)
    effective_radius_mm = effective_support_sigma * float(sigma_mm)
    effective_radius_px = effective_support_sigma * float(sigma_px)
    inner_band_px = (
        float(HARRIS_PAD_INNER_SMOOTH_BAND_RATIO) * float(sigma_px)
    )

    support_vals = gaussian_support[roi]
    support_min = float(np.min(support_vals)) if support_vals.size else 0.0
    support_mean = float(np.mean(support_vals)) if support_vals.size else 0.0

    meta = {
        "cell_mm": float(cell_mm),
        "spacing_mm": float(spacing_mm),
        "sigma_mm": float(sigma_mm),
        "sigma_px": float(sigma_px),
        "effective_support_sigma": float(effective_support_sigma),
        "effective_radius_mm": float(effective_radius_mm),
        "effective_radius_px": float(effective_radius_px),
        "effective_diameter_mm": float(2.0 * effective_radius_mm),
        "effective_diameter_px": float(2.0 * effective_radius_px),
        "sobel_valid_fraction": float(np.mean(gradient_valid)),
        "gaussian_support_min_inside": support_min,
        "gaussian_support_mean_inside": support_mean,
        "inner_smooth_band_px": float(inner_band_px),
        "method": "sobel-valid + normalized Gaussian Harris",
        "positive_max": float(np.max(response_pos)) if response_pos.size else 0.0,
        "eigen_ratio_threshold": float(SCENE_CORNER_MIN_EIGEN_RATIO),
        "eigen_ratio_max": float(np.max(eigen_ratio[valid_mask])) if np.any(valid_mask) else 0.0,
    }
    return (
        response_pos.astype(np.float32),
        valid_mask,
        meta,
        work.astype(np.float32),
        eigen_ratio.astype(np.float32),
    )


def _compute_single_corner_peak_from_tangent(
    uvh: np.ndarray,
    weights: np.ndarray,
    radius_mm: float,
):
    """Pure computational version of the existing tangent Harris2D path.

    It intentionally mirrors draw_normal_weighted_projection(), but does not
    allocate or display the four diagnostic panels. This makes scene-wide
    batch processing practical.
    """
    uvh = np.asarray(uvh, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    valid = np.isfinite(uvh).all(axis=1) & np.isfinite(weights)
    if np.count_nonzero(valid) < MIN_LOCAL_POINTS:
        return None

    uv = uvh[valid, :2]
    w = weights[valid]

    weight_field, support_valid_mask, support_meta = _make_weighted_support_map(
        uv,
        w,
        float(radius_mm),
    )
    if weight_field.size == 0:
        return None

    proc_roi_mask = _make_processing_roi_mask(
        weight_field.shape,
        float(radius_mm),
        support_meta["cell_mm"],
    )

    norm_img = np.zeros_like(weight_field, dtype=np.float32)
    reconstructed_inside = (
        proc_roi_mask
        & support_valid_mask
        & np.isfinite(weight_field)
    )
    finite = weight_field[reconstructed_inside]
    if finite.size == 0 or float(np.max(finite)) <= 0.0:
        return None

    denom = max(float(np.percentile(finite, 99.0)), 1e-12)
    norm_img[reconstructed_inside] = np.clip(
        weight_field[reconstructed_inside] / denom,
        0.0,
        1.0,
    ).astype(np.float32)

    harris_response, harris_safe, harris_meta, _, eigen_ratio_map = (
        _compute_masked_harris2d_response(
            norm_img,
            proc_roi_mask,
            support_meta["cell_mm"],
            support_meta["spacing_mm"],
        )
    )

    peak_roi_ratio = float(np.clip(HARRIS2D_PEAK_ROI_RATIO, 0.0, 1.0))
    peak_radius_mm = peak_roi_ratio * float(radius_mm)
    peak_search_mask = (
        harris_safe
        & _make_processing_roi_mask(
            harris_response.shape,
            peak_radius_mm,
            support_meta["cell_mm"],
        )
        & (eigen_ratio_map >= float(SCENE_CORNER_MIN_EIGEN_RATIO))
    )

    peak = _find_single_harris_local_maximum(
        harris_response,
        peak_search_mask,
        support_meta["cell_mm"],
    )
    if peak is None:
        return None

    response = float(peak["response"])
    if response <= float(SCENE_CORNER_MIN_RESPONSE):
        return None

    gx, gy = int(peak["grid_xy"][0]), int(peak["grid_xy"][1])
    peak_eigen_ratio = float(eigen_ratio_map[gy, gx])

    return {
        "uv_mm": np.asarray(peak["uv_mm"], dtype=np.float64).reshape(2),
        "grid_xy": tuple(peak["grid_xy"]),
        "response": response,
        "eigen_ratio": peak_eigen_ratio,
        "cell_mm": float(support_meta["cell_mm"]),
        "spacing_mm": float(support_meta["spacing_mm"]),
        "effective_radius_mm": float(radius_mm),
        "harris_meta": harris_meta,
    }


def detect_corner_at_query_index(
    query_idx: int,
    frozen: dict,
    base_radius_mm: float,
):
    """Run one automatic query through the existing tangent-Harris detector."""
    points = np.asarray(frozen["points_mm"], dtype=np.float64)
    normals = np.asarray(frozen["normals"], dtype=np.float64)
    intr = frozen["intr"]

    query_idx = int(query_idx)
    if query_idx < 0 or query_idx >= len(points):
        return None

    q = points[query_idx].copy()
    qn = normals[query_idx].copy()
    if (
        not np.isfinite(q).all()
        or not np.isfinite(qn).all()
        or np.linalg.norm(qn) <= 1e-12
    ):
        return None

    qn = normalize(qn)
    if np.dot(qn, -q) < 0.0:
        qn *= -1.0

    effective_radius_mm, radius_scale = _adaptive_inspect_radius_mm(
        q,
        float(base_radius_mm),
    )

    tree = frozen.get("point_tree")
    if tree is None:
        tree = cKDTree(points)

    local_idx = np.asarray(
        tree.query_ball_point(q, r=float(effective_radius_mm)),
        dtype=np.int64,
    )
    if len(local_idx) < MIN_LOCAL_POINTS:
        return None

    local_points = points[local_idx]
    local_normals = normals[local_idx]

    uvh, u_axis, v_axis, _ = project_to_tangent_coordinates(
        local_points,
        q,
        qn,
    )
    weights, theta_deg, normal_valid = normal_similarity_weights(
        local_normals,
        qn,
    )

    valid = (
        normal_valid
        & np.isfinite(uvh).all(axis=1)
        & np.isfinite(weights)
        & np.isfinite(theta_deg)
    )
    if np.count_nonzero(valid) < MIN_LOCAL_POINTS:
        return None

    peak = _compute_single_corner_peak_from_tangent(
        uvh[valid],
        weights[valid],
        effective_radius_mm,
    )
    if peak is None:
        return None

    corner_uv_mm = np.asarray(peak["uv_mm"], dtype=np.float64).reshape(2)
    corner_point_mm = (
        q
        + corner_uv_mm[0] * u_axis
        + corner_uv_mm[1] * v_axis
    ).astype(np.float64)

    corner_pixel = point_mm_to_pixel(corner_point_mm, intr)

    local_points_valid = local_points[valid]
    nearest_surface_dist_mm = None
    if len(local_points_valid):
        d3 = np.linalg.norm(
            local_points_valid - corner_point_mm[None, :],
            axis=1,
        )
        if len(d3) and np.any(np.isfinite(d3)):
            nearest_surface_dist_mm = float(np.nanmin(d3))

    return {
        "query_index": int(query_idx),
        "query_point_mm": q,
        "query_normal": qn,
        "corner_uv_mm": corner_uv_mm,
        "corner_point_mm": corner_point_mm,
        "corner_pixel": corner_pixel,
        "corner_response": float(peak["response"]),
        "corner_eigen_ratio": float(peak.get("eigen_ratio", np.nan)),
        "effective_radius_mm": float(effective_radius_mm),
        "radius_depth_scale": float(radius_scale),
        "local_point_count": int(np.count_nonzero(valid)),
        "nearest_surface_dist_mm": nearest_surface_dist_mm,
    }


def _make_scene_corner_overlay(rgb: np.ndarray, records):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    for i, rec in enumerate(records, start=1):
        px = rec.get("corner_pixel")
        if px is None:
            continue
        _draw_corner_marker_2d(
            out,
            (int(px[0]), int(px[1])),
            label=f"C{i}",
            color=(0, 0, 255),
        )

    cv2.putText(
        out,
        f"final corners: {len(records)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"final corners: {len(records)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def run_scene_corner_detector(frozen: dict, base_radius_mm: float):
    """Execute the complete automatic scene detector after P is pressed."""
    points = np.asarray(frozen["points_mm"], dtype=np.float64)
    normals = np.asarray(frozen["normals"], dtype=np.float64)

    valid_query_pool = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (np.linalg.norm(np.nan_to_num(normals, nan=0.0), axis=1) > 1e-12)
    )
    pool_indices = np.flatnonzero(valid_query_pool)
    if len(pool_indices) == 0:
        print("[scene] no points with valid normals")
        frozen["scene_query_indices"] = np.empty((0,), dtype=np.int64)
        frozen["corner_records_raw"] = []
        frozen["corner_records_filtered"] = []
        frozen["corner_clusters"] = []
        frozen["corner_records"] = []
        return frozen

    t0 = time.perf_counter()
    fps_local = farthest_point_sampling_indices(
        points[pool_indices],
        int(SCENE_FPS_NUM_QUERIES),
    )
    query_indices = pool_indices[fps_local]
    t_fps = time.perf_counter() - t0

    frozen["scene_query_indices"] = query_indices

    print("\n" + "=" * 80)
    print("SCENE FPS QUERY SAMPLING")
    print("=" * 80)
    print(f"valid query pool : {len(pool_indices):,}")
    print(f"FPS queries      : {len(query_indices):,}")
    print(f"FPS time         : {t_fps:.3f} s")

    if SCENE_SHOW_QUERY_VIEW_3D:
        show_detected_corners_global(
            points,
            corner_points_mm=[],
            query_points_mm=points[query_indices],
        )

    raw_records = []
    t0 = time.perf_counter()

    total = len(query_indices)
    progress_step = max(1, total // 10)
    for j, query_idx in enumerate(query_indices, start=1):
        rec = detect_corner_at_query_index(
            int(query_idx),
            frozen,
            float(base_radius_mm),
        )
        if rec is not None:
            raw_records.append(rec)

        if j == 1 or j == total or (j % progress_step == 0):
            print(
                f"[scene corner] {j:4d}/{total:4d} queries | "
                f"raw candidates={len(raw_records):4d}"
            )

    t_corner = time.perf_counter() - t0

    if SCENE_SHOW_RAW_CANDIDATES_3D:
        show_raw_corner_candidates_global(
            points,
            [r["corner_point_mm"] for r in raw_records],
            query_points_mm=points[query_indices],
        )

    # Scene-level filtering FIRST: discard weak Harris responses before any
    # spatial grouping.  This prevents weak candidates from creating bridges
    # between otherwise separate radius-connected clusters.
    filtered_records, response_cutoff = _filter_corner_records_by_response(
        raw_records,
        min_response=float(SCENE_CORNER_MIN_RESPONSE),
        percentile=float(SCENE_CORNER_RESPONSE_PERCENTILE),
    )

    # Radius clustering -> minimum support -> robust in-cluster trimming -> centroid.
    final_records, cluster_infos = _radius_cluster_corner_records(
        filtered_records,
        float(SCENE_CORNER_CLUSTER_RADIUS_MM),
        min_cluster_size=int(SCENE_CORNER_CLUSTER_MIN_SIZE),
        outlier_mad_scale=float(SCENE_CORNER_CLUSTER_OUTLIER_MAD_SCALE),
        inlier_floor_mm=float(SCENE_CORNER_CLUSTER_INLIER_FLOOR_MM),
        max_keep=int(SCENE_MAX_FINAL_CORNERS),
    )

    # The centroid is a new 3-D position, so its RGB projection must be recomputed
    # instead of inheriting the projection of the max-response member record.
    for rec in final_records:
        rec["corner_pixel"] = point_mm_to_pixel(rec["corner_point_mm"], frozen["intr"])

    frozen["corner_records_raw"] = raw_records
    frozen["corner_records_filtered"] = filtered_records
    frozen["corner_clusters"] = cluster_infos
    frozen["corner_records"] = final_records

    print("\n" + "=" * 80)
    print("SCENE CORNER RESULT")
    print("=" * 80)
    print(f"raw candidates     : {len(raw_records):,}")
    print(
        f"response cutoff     : p{SCENE_CORNER_RESPONSE_PERCENTILE:.1f} "
        f"=> {response_cutoff if response_cutoff is not None else float('nan'):.6e}"
    )
    print(f"after response cut  : {len(filtered_records):,}")
    print(f"min eigen ratio     : {SCENE_CORNER_MIN_EIGEN_RATIO:.3f}")
    if raw_records:
        ratios = np.asarray(
            [float(r.get("corner_eigen_ratio", np.nan)) for r in raw_records],
            dtype=np.float64,
        )
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            print(
                f"accepted eigen ratio: min={np.min(ratios):.3f} | "
                f"median={np.median(ratios):.3f} | max={np.max(ratios):.3f}"
            )
    print(f"cluster radius      : {SCENE_CORNER_CLUSTER_RADIUS_MM:.3f} mm")
    print(f"cluster min size    : {SCENE_CORNER_CLUSTER_MIN_SIZE}")
    print(
        f"outlier trimming    : median + {SCENE_CORNER_CLUSTER_OUTLIER_MAD_SCALE:.2f} "
        f"* 1.4826*MAD | floor={SCENE_CORNER_CLUSTER_INLIER_FLOOR_MM:.2f} mm"
    )
    print(f"final clusters      : {len(final_records):,}")
    if final_records:
        sizes = [int(r.get("cluster_size", 1)) for r in final_records]
        inliers = [int(r.get("cluster_inlier_count", 1)) for r in final_records]
        spreads = [float(r.get("cluster_spread_rms_mm", np.nan)) for r in final_records]
        print(
            f"cluster size stats  : min={min(sizes)} | "
            f"median={float(np.median(sizes)):.1f} | max={max(sizes)}"
        )
        print(
            f"inlier count stats  : min={min(inliers)} | "
            f"median={float(np.median(inliers)):.1f} | max={max(inliers)}"
        )
        finite_spreads = np.asarray(spreads, dtype=np.float64)
        finite_spreads = finite_spreads[np.isfinite(finite_spreads)]
        if finite_spreads.size:
            print(
                f"cluster spread RMS  : median={np.median(finite_spreads):.3f} mm | "
                f"max={np.max(finite_spreads):.3f} mm"
            )
    print(f"corner batch time  : {t_corner:.3f} s")

    if final_records and SCENE_SHOW_CORNER_VIEW_3D:
        show_detected_corners_global(
            points,
            [r["corner_point_mm"] for r in final_records],
            query_points_mm=None,
        )

    overlay = _make_scene_corner_overlay(
        frozen["rgb"],
        final_records,
    )
    frozen["scene_result_2d"] = overlay

    if SCENE_SHOW_RESULT_2D:
        cv2.namedWindow(SCENE_RESULT_2D_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(SCENE_RESULT_2D_WINDOW, overlay)

    return frozen


def draw_normal_weighted_projection(

    uvh: np.ndarray,

    weights: np.ndarray,

    theta_deg: np.ndarray,

    radius_mm: float,

):

    uvh = np.asarray(uvh, dtype=np.float64).reshape(-1, 3)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    theta_deg = np.asarray(theta_deg, dtype=np.float64).reshape(-1)



    valid = (

        np.isfinite(uvh).all(axis=1)

        & np.isfinite(weights)

        & np.isfinite(theta_deg)

    )

    uv = uvh[valid, :2]

    w = weights[valid]

    th = theta_deg[valid]



    panel_w = TANGENT_VIEW_SIZE + 90

    gap = 26

    full_w = panel_w * 4 + gap * 3

    canvas = np.full((TANGENT_VIEW_SIZE, full_w, 3), 245, dtype=np.uint8)

    # Filled after Harris2D is evaluated. Exactly one strongest local maximum
    # is retained for the current query.
    peak_info = {
        "found": False,
        "uv_mm": None,
        "grid_xy": None,
        "response": 0.0,
        "eigen_ratio": None,
        "cell_mm": None,
    }



    drawable = TANGENT_VIEW_SIZE - 2 * TANGENT_VIEW_MARGIN

    yy, xx = np.mgrid[0:drawable, 0:drawable]

    cx = 0.5 * (drawable - 1)

    cy = 0.5 * (drawable - 1)

    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    roi_radius_px = 0.5 * drawable

    roi_mask = rr <= roi_radius_px



    def draw_panel_frame(x_offset: int, title: str, subtitle: str):

        x0 = x_offset + TANGENT_VIEW_MARGIN

        y0 = TANGENT_VIEW_MARGIN

        x1 = x_offset + TANGENT_VIEW_SIZE - TANGENT_VIEW_MARGIN

        y1 = TANGENT_VIEW_SIZE - TANGENT_VIEW_MARGIN

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), 1)

        for frac in (0.25, 0.50, 0.75):

            xxg = int(round(x0 + frac * (x1 - x0)))

            yyg = int(round(y0 + frac * (y1 - y0)))

            cv2.line(canvas, (xxg, y0), (xxg, y1), (215, 215, 215), 1)

            cv2.line(canvas, (x0, yyg), (x1, yyg), (215, 215, 215), 1)

        local_origin = _tangent_uv_to_pixel(np.array([[0.0, 0.0]]), radius_mm)[0]

        ox = int(local_origin[0] + x_offset)

        oy = int(local_origin[1])

        cv2.line(canvas, (ox, y0), (ox, y1), (130, 130, 130), 1)

        cv2.line(canvas, (x0, oy), (x1, oy), (130, 130, 130), 1)

        # Circular tangent ROI overlay.

        cv2.circle(canvas, (ox, oy), int(round(roi_radius_px)), (105, 105, 105), 1, cv2.LINE_AA)

        cv2.drawMarker(canvas, (ox, oy), (0,0,0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        cv2.putText(canvas, title, (x_offset + 18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (25,25,25), 2, cv2.LINE_AA)

        cv2.putText(canvas, subtitle, (x_offset + 18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (55,55,55), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'u [mm]', (x1 - 45, TANGENT_VIEW_SIZE - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'v [mm]', (x_offset + 12, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1, cv2.LINE_AA)

        return x0, y0, x1, y1, ox, oy



    left_xoff = 0

    mid_xoff = panel_w + gap

    harris_input_xoff = (panel_w + gap) * 2

    right_xoff = (panel_w + gap) * 3



    # Panel 1: raw points

    lx0, ly0, lx1, ly1, lox, loy = draw_panel_frame(

        left_xoff,

        'raw normal-weighted tangent projection',

        f'w = exp(-0.5*(theta/{TANGENT_NORMAL_WEIGHT_SIGMA_DEG:.1f}deg)^2)'

    )

    if len(uv):

        pix = _tangent_uv_to_pixel(uv, radius_mm)

        pix[:, 0] += left_xoff

        colors_rgb = turbo_colors(w, 0.0, 1.0)

        colors_bgr = np.clip(np.rint(colors_rgb[:, ::-1] * 255.0), 0, 255).astype(np.uint8)

        inside = ((pix[:, 0] >= lx0) & (pix[:, 0] <= lx1) & (pix[:, 1] >= ly0) & (pix[:, 1] <= ly1))

        order = np.argsort(w)

        for idx in order:

            if not inside[idx]:

                continue

            p = (int(pix[idx, 0]), int(pix[idx, 1]))

            color = tuple(int(c) for c in colors_bgr[idx])

            cv2.circle(canvas, p, 2, color, -1, cv2.LINE_AA)



    # Panel 2: sampling-adaptive raster + direct local-mean fill + bilateral weight field

    weight_field, support_valid_mask, support_meta = _make_weighted_support_map(uv, w, radius_mm)
    proc_roi_mask = _make_processing_roi_mask(
        weight_field.shape,
        radius_mm,
        support_meta['cell_mm'],
    )
    valid_proc_mask = proc_roi_mask & support_valid_mask

    mx0, my0, mx1, my1, mox, moy = draw_panel_frame(
        mid_xoff,
        'direct local-mean fill + pixel-space bilateral',
        f's={support_meta["spacing_mm"]:.3f} mm | cell={support_meta["cell_mm"]:.3f} mm | grid={support_meta["grid_size"]}x{support_meta["grid_size"]}'
    )

    if weight_field.size:
        finite = weight_field[np.isfinite(weight_field) & valid_proc_mask]
        if finite.size and float(np.max(finite)) > 0.0:
            denom = max(float(np.percentile(finite, 99.0)), 1e-12)
            dn = np.clip(weight_field / denom, 0.0, 1.0)
        else:
            dn = np.zeros_like(weight_field)

        heat_proc = cv2.applyColorMap(
            np.uint8(np.rint(dn * 255.0)),
            cv2.COLORMAP_TURBO,
        )
        heat_proc[~valid_proc_mask] = np.array([150, 150, 150], dtype=np.uint8)
        heat = cv2.resize(
            heat_proc,
            (drawable, drawable),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas[my0:my0+drawable, mx0:mx0+drawable] = heat

    # Panel 3: exact ROI-only intensity field used by Harris.
    # Outside the circular ROI is never synthesized and is displayed as gray.
    jx0, jy0, jx1, jy1, jox, joy = draw_panel_frame(
        harris_input_xoff,
        'Harris input intensity W(u,v)',
        'ROI-only inner smoothing | outside unused by Harris'
    )

    harris_input_vis = np.full((drawable, drawable, 3), 150, dtype=np.uint8)

    # Panel 4: Harris2D response on the reconstructed W(u,v) intensity field.
    rx0, ry0, rx1, ry1, rox, roy = draw_panel_frame(
        right_xoff,
        'masked Harris2D on W(u,v) intensity',
        'normalized Gaussian tensor | outside ROI unused'
    )

    response_vis = np.full((drawable, drawable, 3), 150, dtype=np.uint8)
    harris_meta = {
        'sigma_mm': 0.0,
        'sigma_px': 0.0,
        'effective_support_sigma': float(HARRIS2D_EFFECTIVE_SUPPORT_SIGMA),
        'effective_radius_mm': 0.0,
        'effective_radius_px': 0.0,
        'effective_diameter_mm': 0.0,
        'effective_diameter_px': 0.0,
        'sobel_valid_fraction': 0.0,
        'gaussian_support_min_inside': 0.0,
        'gaussian_support_mean_inside': 0.0,
        'inner_smooth_band_px': 0.0,
        'method': 'sobel-valid + normalized Gaussian Harris',
    }

    if weight_field.size:
        # Normalize W using only genuinely reconstructed support samples.
        # This keeps zero-valued internal holes exactly at intensity 0.
        norm_img = np.zeros_like(weight_field, dtype=np.float32)
        reconstructed_inside = proc_roi_mask & support_valid_mask & np.isfinite(weight_field)
        finite = weight_field[reconstructed_inside]

        if finite.size and float(np.max(finite)) > 0.0:
            denom = max(float(np.percentile(finite, 99.0)), 1e-12)
            norm_img[reconstructed_inside] = np.clip(
                weight_field[reconstructed_inside] / denom,
                0.0,
                1.0,
            ).astype(np.float32)

        # Harris processing:
        #   ROI-only inner smoothing
        #   -> Sobel-valid mask
        #   -> mask-normalized Gaussian structure tensor.
        # No outside-ROI padding remains.
        harris_response, harris_safe, harris_meta, harris_input_img, eigen_ratio_map = (
            _compute_masked_harris2d_response(
                norm_img,
                proc_roi_mask,
                support_meta['cell_mm'],
                support_meta['spacing_mm'],
            )
        )

        input_proc = cv2.applyColorMap(
            np.uint8(np.rint(np.clip(harris_input_img, 0.0, 1.0) * 255.0)),
            cv2.COLORMAP_TURBO,
        )
        input_proc[~proc_roi_mask] = np.array([150, 150, 150], dtype=np.uint8)
        harris_input_vis = cv2.resize(
            input_proc,
            (drawable, drawable),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas[jy0:jy0+drawable, jx0:jx0+drawable] = harris_input_vis


        finite_resp = harris_response[
            harris_safe & np.isfinite(harris_response) & (harris_response > 0.0)
        ]
        if finite_resp.size:
            vmax = max(
                float(np.percentile(
                    finite_resp,
                    float(HARRIS2D_RESPONSE_PERCENTILE),
                )),
                float(HARRIS2D_EPS),
            )
            rn = np.clip(harris_response / vmax, 0.0, 1.0)
        else:
            rn = np.zeros_like(harris_response, dtype=np.float32)

        response_proc = cv2.applyColorMap(
            np.uint8(np.rint(rn * 255.0)),
            cv2.COLORMAP_TURBO,
        )

        # Gray means outside the circular ROI. Internal holes remain zero intensity
        # and still participate inside the ROI.
        response_proc[~harris_safe] = np.array([150, 150, 150], dtype=np.uint8)

        response_vis = cv2.resize(
            response_proc,
            (drawable, drawable),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas[ry0:ry0+drawable, rx0:rx0+drawable] = response_vis

        # Visualize the allowed peak-search region (central alpha * ROI).
        peak_search_display_radius = int(round(
            float(np.clip(HARRIS2D_PEAK_ROI_RATIO, 0.0, 1.0)) * roi_radius_px
        ))
        cv2.circle(
            canvas,
            (int(rox), int(roy)),
            max(1, peak_search_display_radius),
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f'peak search <= {HARRIS2D_PEAK_ROI_RATIO:.2f} x ROI',
            (rx0 + 10, ry0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 110, 220),
            1,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------
        # Query -> one Harris2D corner: strongest positive local maximum.
        #
        # IMPORTANT:
        # Harris itself is evaluated over the full valid ROI, but final peak
        # selection is restricted to the central HARRIS2D_PEAK_ROI_RATIO
        # fraction of the physical tangent ROI. This rejects peaks in the
        # outer boundary band:
        #
        #   ||[u*, v*]|| <= alpha * r_ROI,
        #   alpha = HARRIS2D_PEAK_ROI_RATIO.
        # ------------------------------------------------------------
        peak_roi_ratio = float(np.clip(HARRIS2D_PEAK_ROI_RATIO, 0.0, 1.0))
        peak_radius_mm = peak_roi_ratio * float(radius_mm)
        peak_search_mask = (
            harris_safe
            & _make_processing_roi_mask(
                harris_response.shape,
                peak_radius_mm,
                support_meta["cell_mm"],
            )
            & (eigen_ratio_map >= float(SCENE_CORNER_MIN_EIGEN_RATIO))
        )

        peak = _find_single_harris_local_maximum(
            harris_response,
            peak_search_mask,
            support_meta["cell_mm"],
        )

        if peak is not None:
            gx_peak, gy_peak = int(peak["grid_xy"][0]), int(peak["grid_xy"][1])
            peak_eigen_ratio = float(eigen_ratio_map[gy_peak, gx_peak])
            peak_info = {
                "found": True,
                "uv_mm": np.asarray(peak["uv_mm"], dtype=np.float64),
                "grid_xy": tuple(peak["grid_xy"]),
                "response": float(peak["response"]),
                "eigen_ratio": peak_eigen_ratio,
                "cell_mm": float(support_meta["cell_mm"]),
                "peak_roi_ratio": float(peak_roi_ratio),
                "peak_radius_mm": float(peak_radius_mm),
            }

            peak_uv = peak_info["uv_mm"]

            # Mark the same detected point on the raw tangent projection.
            raw_peak_px = _tangent_uv_to_pixel(peak_uv.reshape(1, 2), radius_mm)[0]
            _draw_corner_marker_2d(
                canvas,
                (int(raw_peak_px[0] + left_xoff), int(raw_peak_px[1])),
                label="peak",
            )

            # Mark it on the Harris response panel as well.
            proc_h, proc_w = harris_response.shape
            gx, gy = peak_info["grid_xy"]
            disp_x = int(round(gx * (drawable - 1) / max(proc_w - 1, 1)))
            disp_y = int(round(gy * (drawable - 1) / max(proc_h - 1, 1)))
            _draw_corner_marker_2d(
                canvas,
                (int(rx0 + disp_x), int(ry0 + disp_y)),
                label=f"peak rho={peak_info['eigen_ratio']:.2f}",
            )

        # ------------------------------------------------------------
        # Harris Gaussian-window scale visualization.
        #
        # Harris has a local Gaussian integration window around EVERY pixel.
        # To make its physical size visible, draw the same-size window at the
        # tangent-frame origin as a scale legend on the Harris-response panel:
        #   thin circle  = 1 sigma
        #   thick circle = configured effective support
        # ------------------------------------------------------------
        if HARRIS2D_SHOW_WINDOW and harris_response.size:
            proc_h, proc_w = harris_response.shape
            sx = float(drawable) / max(float(proc_w), 1.0)
            sy = float(drawable) / max(float(proc_h), 1.0)
            display_scale = 0.5 * (sx + sy)

            sigma_disp_px = max(
                1,
                int(round(float(harris_meta["sigma_px"]) * display_scale)),
            )
            support_disp_px = max(
                sigma_disp_px,
                int(round(float(harris_meta["effective_radius_px"]) * display_scale)),
            )

            if HARRIS2D_SHOW_ONE_SIGMA:
                cv2.circle(
                    canvas,
                    (int(rox), int(roy)),
                    sigma_disp_px,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

            cv2.circle(
                canvas,
                (int(rox), int(roy)),
                support_disp_px,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )

            label_y = min(
                ry1 - 10,
                int(roy + support_disp_px + 22),
            )
            cv2.putText(
                canvas,
                (
                    f'Harris window: 1sigma={harris_meta["sigma_mm"]:.2f}mm | '
                    f'{harris_meta["effective_support_sigma"]:.1f}sigma '
                    f'diameter={harris_meta["effective_diameter_px"]:.1f}px '
                    f'({harris_meta["effective_diameter_mm"]:.2f}mm)'
                ),
                (rx0 + 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )

    # global text

    cv2.putText(canvas,

                f'points={len(uv):,} | u,v range +/-{radius_mm:.1f} mm',

                (18, TANGENT_VIEW_SIZE - 34),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    if len(w):

        cv2.putText(canvas,

                    f'theta median={np.median(th):.1f} deg | weight median={np.median(w):.3f}',

                    (18, TANGENT_VIEW_SIZE - 14),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    cv2.putText(canvas,

                f'ROI outside circle = INVALID on panel 2 and panel 4',

                (mid_xoff + 18, TANGENT_VIEW_SIZE - 34),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    cv2.putText(canvas,

                f's={support_meta["spacing_mm"]:.3f}mm | cell={support_meta["cell_mm"]:.3f}mm | mean radius={MEAN_MEAN_FILL_RADIUS_SCALE:.2f}s | min valid={MEAN_FILL_MIN_VALID_NEIGHBORS} | filled={support_meta["num_filled_px"]} cells',

                (mid_xoff + 18, TANGENT_VIEW_SIZE - 14),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    cv2.putText(canvas,

                f'Panel 3 = ROI-only Harris input | holes=0 | outside unused | inner band={HARRIS_PAD_INNER_SMOOTH_BAND_RATIO:.2f}x sigma_H',

                (harris_input_xoff + 18, TANGENT_VIEW_SIZE - 34),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    cv2.putText(canvas,

                f'pixel stage: bilateral={PIXEL_TUNING["bilateral_sigma_space_px"]:.2f}px | Harris sigma={PIXEL_TUNING["harris_sigma_px"]:.2f}px',

                (harris_input_xoff + 18, TANGENT_VIEW_SIZE - 14),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)

    cv2.putText(canvas,

                f'Harris2D: k={HARRIS2D_K:.3f} | sigma={harris_meta["sigma_mm"]:.2f}mm ({harris_meta["sigma_px"]:.1f}px) | eig ratio>={SCENE_CORNER_MIN_EIGEN_RATIO:.2f} | peak ROI<={HARRIS2D_PEAK_ROI_RATIO:.2f}r | display p{HARRIS2D_RESPONSE_PERCENTILE:.1f}',

                (right_xoff + 18, TANGENT_VIEW_SIZE - 14),

                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55,55,55), 1, cv2.LINE_AA)



    # Colorbars

    def draw_colorbar(x, y0, y1, label):

        nbar = max(y1 - y0 + 1, 2)

        vals = np.linspace(1.0, 0.0, nbar)

        colors = turbo_colors(vals, 0.0, 1.0)

        colors_bgr = np.clip(np.rint(colors[:, ::-1] * 255.0), 0, 255).astype(np.uint8)

        for j in range(nbar):

            cv2.line(canvas, (x, y0+j), (x+22, y0+j), tuple(int(c) for c in colors_bgr[j]), 1)

        cv2.rectangle(canvas, (x, y0), (x+22, y1), (50,50,50), 1)

        cv2.putText(canvas, label, (x-2, y0-14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40,40,40), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'hi', (x+30, y0+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40,40,40), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'lo', (x+30, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40,40,40), 1, cv2.LINE_AA)



    draw_colorbar(left_xoff + TANGENT_VIEW_SIZE + 30, 95, TANGENT_VIEW_SIZE - 80, 'w')

    draw_colorbar(mid_xoff + TANGENT_VIEW_SIZE + 30, 95, TANGENT_VIEW_SIZE - 80, 'D')

    draw_colorbar(harris_input_xoff + TANGENT_VIEW_SIZE + 30, 95, TANGENT_VIEW_SIZE - 80, 'I')

    draw_colorbar(right_xoff + TANGENT_VIEW_SIZE + 30, 95, TANGENT_VIEW_SIZE - 80, 'R_H')



    return canvas, peak_info





# ============================================================


# ============================================================
# Pixel-space live tuning GUI
# ============================================================

_PIXEL_TUNER_CONTEXT = None
_PIXEL_TUNER_READY = False
_PIXEL_TUNER_BUSY = False


def _draw_pixel_tuner_status():
    canvas = np.full((285, 780, 3), 245, dtype=np.uint8)

    current_harris_sigma = float(PIXEL_TUNING["harris_sigma_px"])
    inner_band_px = (
        float(HARRIS_PAD_INNER_SMOOTH_BAND_RATIO) * current_harris_sigma
    )

    lines = [
        "PIXEL-SPACE LIVE TUNING",
        "",
        f'Bilateral sigmaSpace : {PIXEL_TUNING["bilateral_sigma_space_px"]:.2f} px',
        f'Harris Gaussian sigma: {current_harris_sigma:.2f} px',
        f'Inner smooth band     : {HARRIS_PAD_INNER_SMOOTH_BAND_RATIO:.2f} x sigma_H = {inner_band_px:.2f} px',
        f'Harris support display: fixed {HARRIS2D_EFFECTIVE_SUPPORT_SIGMA:.1f} sigma',
        f'Peak search radius      : {HARRIS2D_PEAK_ROI_RATIO:.2f} x tangent ROI radius',
        f'Min eigenvalue ratio    : {SCENE_CORNER_MIN_EIGEN_RATIO:.2f}',
        "Boundary method       : Sobel-valid mask + normalized Gaussian Harris",
        "Outer ROI             : no padding / no mirror / no synthesized values",
        "",
        "Changing Harris sigma updates Harris integration and inner smoothing together.",
        "T in the main window -> reset live tuning defaults.",
    ]

    y = 28
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 if i == 0 else 0.46,
            (20, 20, 20),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )
        y += 31 if i == 0 else 24
    return canvas


def _render_pixel_tuner_context():
    global _PIXEL_TUNER_BUSY

    if _PIXEL_TUNER_BUSY or _PIXEL_TUNER_CONTEXT is None:
        return

    _PIXEL_TUNER_BUSY = True
    try:
        ctx = _PIXEL_TUNER_CONTEXT
        image, peak_info = draw_normal_weighted_projection(
            ctx["uvh"],
            ctx["weights"],
            ctx["theta_deg"],
            ctx["radius_mm"],
        )
        ctx["projection_image"] = image
        ctx["peak_info"] = peak_info
        cv2.imshow(WEIGHTED_PROJECTION_WINDOW, image)
        cv2.imshow(PIXEL_TUNER_WINDOW, _draw_pixel_tuner_status())
    finally:
        _PIXEL_TUNER_BUSY = False


def _read_pixel_tuner_trackbars():
    if not _PIXEL_TUNER_READY:
        return

    PIXEL_TUNING["bilateral_sigma_space_px"] = max(
        0.10,
        cv2.getTrackbarPos(
            "Bilateral sigma x100",
            PIXEL_TUNER_WINDOW,
        ) / 100.0,
    )
    PIXEL_TUNING["harris_sigma_px"] = max(
        0.10,
        cv2.getTrackbarPos(
            "Harris sigma x100",
            PIXEL_TUNER_WINDOW,
        ) / 100.0,
    )


def _pixel_tuner_callback(_=None):
    if not _PIXEL_TUNER_READY:
        return
    _read_pixel_tuner_trackbars()
    _render_pixel_tuner_context()


def _ensure_pixel_tuner():
    global _PIXEL_TUNER_READY

    if _PIXEL_TUNER_READY:
        return

    cv2.namedWindow(PIXEL_TUNER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PIXEL_TUNER_WINDOW, 820, 430)

    cv2.createTrackbar(
        "Bilateral sigma x100",
        PIXEL_TUNER_WINDOW,
        int(round(100.0 * PIXEL_TUNING["bilateral_sigma_space_px"])),
        1000,
        _pixel_tuner_callback,
    )
    cv2.createTrackbar(
        "Harris sigma x100",
        PIXEL_TUNER_WINDOW,
        int(round(100.0 * PIXEL_TUNING["harris_sigma_px"])),
        1000,
        _pixel_tuner_callback,
    )
    _PIXEL_TUNER_READY = True
    cv2.imshow(PIXEL_TUNER_WINDOW, _draw_pixel_tuner_status())


def _set_pixel_tuner_context(
    uvh: np.ndarray,
    weights: np.ndarray,
    theta_deg: np.ndarray,
    radius_mm: float,
):
    global _PIXEL_TUNER_CONTEXT

    _PIXEL_TUNER_CONTEXT = {
        "uvh": np.asarray(uvh, dtype=np.float64).copy(),
        "weights": np.asarray(weights, dtype=np.float64).copy(),
        "theta_deg": np.asarray(theta_deg, dtype=np.float64).copy(),
        "radius_mm": float(radius_mm),
        "projection_image": None,
        "peak_info": None,
    }

    _ensure_pixel_tuner()
    _render_pixel_tuner_context()


def _clear_pixel_tuner_context():
    global _PIXEL_TUNER_CONTEXT
    _PIXEL_TUNER_CONTEXT = None


def _reset_pixel_tuning():
    PIXEL_TUNING.clear()
    PIXEL_TUNING.update(PIXEL_TUNING_DEFAULTS)

    if _PIXEL_TUNER_READY:
        cv2.setTrackbarPos(
            "Bilateral sigma x100",
            PIXEL_TUNER_WINDOW,
            int(round(100.0 * PIXEL_TUNING["bilateral_sigma_space_px"])),
        )
        cv2.setTrackbarPos(
            "Harris sigma x100",
            PIXEL_TUNER_WINDOW,
            int(round(100.0 * PIXEL_TUNING["harris_sigma_px"])),
        )

    _render_pixel_tuner_context()



# Query inspection

# ============================================================



def inspect_query(query_u: int, query_v: int, frozen: dict, radius_mm: float):

    depth_m = frozen["depth_m"]

    intr = frozen["intr"]



    clicked_point = pixel_to_point_mm(query_u, query_v, depth_m, intr)

    if clicked_point is None:

        print(f"[inspect] invalid depth at pixel ({query_u}, {query_v})")

        return None



    valid_tree = frozen.get("valid_normal_tree")

    valid_indices = frozen.get("valid_normal_indices")

    if valid_tree is None or valid_indices is None or len(valid_indices) == 0:

        print("[inspect] no valid normals in frozen cloud")

        return None



    nn_dist, nn_local_idx = valid_tree.query(clicked_point, k=1)

    query_idx = int(valid_indices[int(nn_local_idx)])



    points = np.asarray(frozen["points_mm"], dtype=np.float64)

    normals = np.asarray(frozen["normals"], dtype=np.float64)



    # Use the actual measured cloud sample as q.
    q = points[query_idx].copy()

    # Depth-adapt both the outer tangent ROI and the local normal-estimation radius.
    effective_radius_mm, radius_scale = _adaptive_inspect_radius_mm(q, radius_mm)
    adaptive_normal_radius_mm, normal_radius_scale = _adaptive_normal_radius_mm(q)

    # Give the normal estimator one extra normal radius of context so points near
    # the tangent-ROI boundary are not biased by an artificial crop.
    context_radius_mm = effective_radius_mm + adaptive_normal_radius_mm

    dist = np.linalg.norm(points - q[None, :], axis=1)
    context_mask = np.isfinite(dist) & (dist <= float(context_radius_mm))
    context_idx = np.flatnonzero(context_mask)

    local_mask = np.isfinite(dist) & (dist <= float(effective_radius_mm))
    local_idx = np.flatnonzero(local_mask)

    if len(context_idx) < MIN_LOCAL_POINTS:
        print(
            f"[inspect] too few context points for local normal estimation: "
            f"{len(context_idx)} < {MIN_LOCAL_POINTS}"
        )
        return None

    if len(local_idx) < MIN_LOCAL_POINTS:
        print(
            f"[inspect] too few local points: {len(local_idx)} < {MIN_LOCAL_POINTS}"
        )
        return None

    # Re-estimate the normal field only for this click-stage context.
    context_points = points[context_idx]
    context_normals = estimate_surface_aware_normals_mm(
        context_points,
        radius_mm=adaptive_normal_radius_mm,
    )

    # Map the requested tangent ROI back into the larger normal context.
    local_context_pos = np.searchsorted(context_idx, local_idx)
    if (
        np.any(local_context_pos >= len(context_idx))
        or np.any(context_idx[local_context_pos] != local_idx)
    ):
        raise RuntimeError("Failed to map local ROI into local-normal context.")

    local_points = points[local_idx]
    local_normals = context_normals[local_context_pos]

    # Use the newly estimated query normal. Fall back to the frozen normal only
    # if this single local estimate is invalid.
    q_context_pos = int(np.searchsorted(context_idx, query_idx))
    local_qn_valid = (
        0 <= q_context_pos < len(context_idx)
        and context_idx[q_context_pos] == query_idx
        and np.isfinite(context_normals[q_context_pos]).all()
        and np.linalg.norm(context_normals[q_context_pos]) > 1e-12
    )

    if local_qn_valid:
        qn = normalize(context_normals[q_context_pos])
        qn_source = "local depth-adaptive re-estimation"
    else:
        qn = normalize(normals[query_idx])
        qn_source = "freeze-time fallback"
        print("[inspect] WARNING: local query normal invalid; using frozen normal")

    # Stable display convention: +h roughly faces the camera.
    if np.dot(qn, -q) < 0.0:
        qn *= -1.0



    uvh, u, v, n = project_to_tangent_coordinates(local_points, q, qn)

    weights, theta_deg, normal_valid = normal_similarity_weights(local_normals, qn)



    valid = normal_valid & np.isfinite(uvh).all(axis=1)

    uvh_valid = uvh[valid]

    w_valid = weights[valid]

    theta_valid = theta_deg[valid]

    local_points_valid = local_points[valid]



    print("\n" + "=" * 80)

    print("LOCAL TANGENT INSPECTION")

    print("=" * 80)

    print(f"clicked pixel            : ({query_u}, {query_v})")

    print(f"nearest measured q [mm]  : {np.round(q, 3).tolist()}")

    print(f"click -> q distance [mm] : {float(nn_dist):.3f}")

    print(f"query normal n_q         : {np.round(qn, 6).tolist()}")
    print(f"query normal source      : {qn_source}")
    print(
        f"adaptive normal radius   : {adaptive_normal_radius_mm:.2f} mm "
        f"(x{normal_radius_scale:.3f} vs {NORMAL_RADIUS_REFERENCE_Z_MM:.0f}mm ref)"
    )
    print(f"normal context radius    : {context_radius_mm:.2f} mm")
    print(f"normal context points    : {len(context_idx):,}")

    print(f"u                        : {np.round(u, 6).tolist()}")

    print(f"v                        : {np.round(v, 6).tolist()}")

    print(f"h-axis / n               : {np.round(n, 6).tolist()}")

    print(f"query camera Z           : {abs(float(q[2])):.2f} mm")
    print(f"radius base @ {INSPECT_RADIUS_REFERENCE_Z_MM:.0f}mm : {radius_mm:.2f} mm")
    print(f"radius depth scale       : x{radius_scale:.3f}")
    print(f"effective local radius   : {effective_radius_mm:.2f} mm")
    print(f"local points             : {len(local_idx):,}")

    print(f"valid weighted points    : {len(uvh_valid):,}")

    if len(theta_valid):

        print(f"theta median             : {np.median(theta_valid):.3f} deg")

        print(f"weight median            : {np.median(w_valid):.6f}")



    # Stage 1: 3-D local tangent coordinates.

    show_local_tangent_3d(uvh_valid, w_valid, effective_radius_mm)



    # Stage 2: 2-D projection + reconstruction + Harris.
    # Pixel-domain spatial parameters are updated live from the tuner GUI.
    _set_pixel_tuner_context(
        uvh_valid,
        w_valid,
        theta_valid,
        effective_radius_mm,
    )
    image = _PIXEL_TUNER_CONTEXT["projection_image"]
    peak_info = _PIXEL_TUNER_CONTEXT.get("peak_info")

    # ------------------------------------------------------------
    # Harris2D peak (u,v) -> tangent-plane 3-D corner -> RGB pixel.
    #
    # q is the tangent origin, and (u, v, n) are orthonormal axes, so
    # the inverse tangent projection at h=0 is simply
    #     p_corner = q + u_peak * u_axis + v_peak * v_axis.
    # ------------------------------------------------------------
    corner_uv_mm = None
    corner_point_mm = None
    corner_pixel = None
    corner_response = None
    corner_eigen_ratio = None
    corner_nearest_surface_point_mm = None
    corner_nearest_surface_dist_mm = None

    if peak_info is not None and peak_info.get("found", False):
        corner_uv_mm = np.asarray(peak_info["uv_mm"], dtype=np.float64).reshape(2)
        corner_response = float(peak_info["response"])
        corner_eigen_ratio = float(peak_info.get("eigen_ratio", np.nan))

        corner_point_mm = (
            q
            + corner_uv_mm[0] * u
            + corner_uv_mm[1] * v
        ).astype(np.float64)

        corner_pixel = point_mm_to_pixel(corner_point_mm, intr)

        # Diagnostic only: nearest actually measured local surface point.
        # The displayed/detected 3-D corner remains the direct h=0 back-projection.
        if len(local_points_valid):
            d3 = np.linalg.norm(
                local_points_valid - corner_point_mm[None, :],
                axis=1,
            )
            if len(d3) and np.any(np.isfinite(d3)):
                nn3 = int(np.nanargmin(d3))
                corner_nearest_surface_point_mm = np.asarray(
                    local_points_valid[nn3], dtype=np.float64
                ).copy()
                corner_nearest_surface_dist_mm = float(d3[nn3])

        print("\n[Harris2D single local maximum]")
        print(f"  tangent uv [mm]       : {np.round(corner_uv_mm, 3).tolist()}")
        print(f"  response              : {corner_response:.6e}")
        print(f"  eigen ratio rho       : {corner_eigen_ratio:.4f} (min {SCENE_CORNER_MIN_EIGEN_RATIO:.2f})")
        print(f"  corner 3D [mm]        : {np.round(corner_point_mm, 3).tolist()}")
        print(f"  RGB overlay pixel     : {corner_pixel}")
        if corner_nearest_surface_dist_mm is not None:
            print(
                f"  nearest measured dist : {corner_nearest_surface_dist_mm:.3f} mm"
            )
    else:
        print("[Harris2D] no positive local maximum for this query")



    return {

        "query_index": query_idx,

        "query_point_mm": q,

        "query_normal": qn,

        "local_indices": local_idx,

        "uvh": uvh_valid,

        "weights": w_valid,

        "theta_deg": theta_valid,
        "base_radius_mm": float(radius_mm),
        "effective_radius_mm": float(effective_radius_mm),
        "radius_depth_scale": float(radius_scale),
        "adaptive_normal_radius_mm": float(adaptive_normal_radius_mm),
        "normal_radius_depth_scale": float(normal_radius_scale),
        "normal_context_radius_mm": float(context_radius_mm),
        "normal_source": qn_source,

        "corner_uv_mm": corner_uv_mm,
        "corner_response": corner_response,
        "corner_eigen_ratio": corner_eigen_ratio,
        "corner_point_mm": corner_point_mm,
        "corner_pixel": corner_pixel,
        "corner_nearest_surface_point_mm": corner_nearest_surface_point_mm,
        "corner_nearest_surface_dist_mm": corner_nearest_surface_dist_mm,
        "peak_info": peak_info,

        "projection_image": image,

    }





# ============================================================

# Freeze processing

# ============================================================



def build_frozen_state(latest: dict):
    rgb = latest["rgb"].copy()
    depth_raw = latest["depth_raw"].copy()
    depth_m = latest["depth_m"].copy()
    intr = latest["intr"]

    raw_points_mm, raw_uv = depth_to_xyz_mm(
        depth_m,
        intr,
        stride=PIXEL_STRIDE,
    )
    if len(raw_points_mm) < MIN_LOCAL_POINTS:
        print("[P] not enough valid depth points")
        return None

    t0 = time.perf_counter()
    points_mm = voxel_downsample_points_mm(
        raw_points_mm,
        float(SCENE_VOXEL_SIZE_MM),
    )
    t_voxel = time.perf_counter() - t0

    if len(points_mm) < MIN_LOCAL_POINTS:
        print("[P] not enough points after voxel downsampling")
        return None

    print("\n" + "=" * 80)
    print("FREEZE + SCENE PREPROCESS")
    print("=" * 80)
    print(f"raw depth points     : {len(raw_points_mm):,}")
    print(f"pixel stride         : {PIXEL_STRIDE}")
    print(f"voxel size           : {SCENE_VOXEL_SIZE_MM:.3f} mm")
    print(f"voxel points         : {len(points_mm):,}")
    print(f"voxel time           : {t_voxel:.3f} s")
    print(f"global normal radius : {NORMAL_RADIUS_MM:.3f} mm")
    print(f"normal sigma_e       : {NORMAL_SIGMA_E_DEG:.3f} deg")
    print(f"normal sigma_r       : {NORMAL_SIGMA_R:.3f}")
    print(f"normal iterations    : {NORMAL_MAX_ITERATIONS}")

    t0 = time.perf_counter()
    normals = estimate_surface_aware_normals_mm(points_mm)
    t_normal = time.perf_counter() - t0

    valid_normals = (
        np.isfinite(normals).all(axis=1)
        & (np.linalg.norm(np.nan_to_num(normals, nan=0.0), axis=1) > 1e-12)
    )
    valid_indices = np.flatnonzero(valid_normals)

    valid_tree = None
    if len(valid_indices):
        valid_tree = cKDTree(
            np.asarray(points_mm[valid_indices], dtype=np.float64)
        )

    point_tree = cKDTree(
        np.asarray(points_mm, dtype=np.float64)
    )

    print(f"surface-aware normals : {t_normal:.3f} s")
    print(f"valid normals         : {len(valid_indices):,}/{len(normals):,}")
    print(
        f"preprocess total      : {t_voxel + t_normal:.3f} s "
        "(Harris3D diagnostic is now lazy; H computes it on demand)"
    )

    return {
        "rgb": rgb,
        "depth_raw": depth_raw,
        "depth_m": depth_m,
        "intr": intr,
        "raw_points_mm": raw_points_mm,
        "raw_uv": raw_uv,
        "points_mm": points_mm,
        "uv": None,
        "normals": normals,
        "response": None,
        "valid_normal_indices": valid_indices,
        "valid_normal_tree": valid_tree,
        "point_tree": point_tree,
        "scene_query_indices": np.empty((0,), dtype=np.int64),
        "corner_records_raw": [],
        "corner_records_filtered": [],
        "corner_clusters": [],
        "corner_records": [],
        "scene_result_2d": None,
    }



# ============================================================

# Main

# ============================================================



def main():

    global INSPECT_RADIUS_MM



    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)



    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)



    depth_sensor = profile.get_device().first_depth_sensor()

    depth_scale = float(depth_sensor.get_depth_scale())

    print(f"[RealSense] depth scale = {depth_scale:.9f} m/unit")



    print("\nMain controls")

    print("  P       : freeze -> voxel -> normals -> FPS -> auto corner detection")

    print("  R       : return to live")

    print("  Click   : optional single-query tangent diagnostic on frozen frame")

    print("  [ / ]   : change local inspection radius")

    print("  1/2/3   : RGB / depth / RGB+depth")

    print("  H       : compute/reopen optional Harris3D + normal diagnostic")
    print("  C       : show full cloud + automatic final corners")
    print("  T       : reset pixel-space tuning")

    print("  Q / ESC : quit\n")



    state = {

        "mode": "live",

        "display_mode": 3,

        "latest": None,

        "frozen": None,

        "last_inspection": None,

    }



    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)



    def mouse_callback(event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:

            return

        if state["mode"] != "frozen" or state["frozen"] is None:

            print("[mouse] press P first")

            return



        result = inspect_query(

            int(x),

            int(y),

            state["frozen"],

            INSPECT_RADIUS_MM,

        )

        state["last_inspection"] = result

        if result is None:
            return

        corner_point_mm = result.get("corner_point_mm")
        corner_pixel = result.get("corner_pixel")
        if corner_point_mm is None:
            return

        record = {
            "query_pixel": (int(x), int(y)),
            "query_point_mm": np.asarray(
                result["query_point_mm"], dtype=np.float64
            ).copy(),
            "corner_pixel": (
                None
                if corner_pixel is None
                else (int(corner_pixel[0]), int(corner_pixel[1]))
            ),
            "corner_point_mm": np.asarray(
                corner_point_mm, dtype=np.float64
            ).copy(),
            "corner_uv_mm": np.asarray(
                result["corner_uv_mm"], dtype=np.float64
            ).copy(),
            "corner_response": float(result["corner_response"]),
            "corner_eigen_ratio": float(result.get("corner_eigen_ratio", np.nan)),
        }
        state["frozen"].setdefault("corner_records", []).append(record)

        # Immediately show the detected 3-D corner(s) on the full frozen cloud.
        records = state["frozen"]["corner_records"]
        show_detected_corners_global(
            state["frozen"]["points_mm"],
            [r["corner_point_mm"] for r in records],
            query_points_mm=[r["query_point_mm"] for r in records],
        )



    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)



    try:

        while True:

            if state["mode"] == "live":

                frames = pipeline.wait_for_frames()

                frames = align.process(frames)



                depth_frame = frames.get_depth_frame()

                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:

                    continue



                depth_raw = np.asanyarray(depth_frame.get_data()).copy()

                rgb = np.asanyarray(color_frame.get_data()).copy()



                depth_m = depth_raw.astype(np.float32) * depth_scale

                depth_m[

                    (~np.isfinite(depth_m))

                    | (depth_m < MIN_DEPTH_M)

                    | (depth_m > MAX_DEPTH_M)

                ] = 0.0



                intr = (

                    depth_frame.profile

                    .as_video_stream_profile()

                    .get_intrinsics()

                )



                state["latest"] = {

                    "rgb": rgb,

                    "depth_raw": depth_raw,

                    "depth_m": depth_m,

                    "intr": intr,

                }



            current = state["latest"] if state["mode"] == "live" else state["frozen"]

            if current is None:

                continue



            rgb = current["rgb"]

            depth_vis = make_depth_vis(current["depth_m"])

            display_mode = state["display_mode"]



            if display_mode == 1:

                show = rgb.copy()

            elif display_mode == 2:

                show = depth_vis.copy()

            else:

                show = np.hstack((rgb, depth_vis))



            # Keep the click window RGB-only while frozen so x,y map to depth directly.

            if state["mode"] == "frozen" and display_mode == 3:

                cv2.imshow(FROZEN_RGBD_WINDOW, show)

                show = rgb.copy()



            status = "LIVE" if state["mode"] == "live" else "FROZEN"

            status_text = (

                f"{status} | P auto scene detect | V raw3D | C final3D | R live | click debug | "

                f"base radius@{INSPECT_RADIUS_REFERENCE_Z_MM:.0f}mm={INSPECT_RADIUS_MM:.1f}mm"

            )

            cv2.putText(

                show, status_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,

                0.52, (0, 0, 0), 3, cv2.LINE_AA,

            )

            cv2.putText(

                show, status_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,

                0.52, (0, 255, 0), 1, cv2.LINE_AA,

            )

            # Overlay accumulated query/corner positions on the frozen RGB image.
            # Q = clicked query (blue), C = Harris2D corner back-projection (red).
            if state["mode"] == "frozen" and state["frozen"] is not None:
                records = state["frozen"].get("corner_records", [])
                for i, rec in enumerate(records, start=1):
                    qpx = rec.get("query_pixel")
                    cpx = rec.get("corner_pixel")

                    if qpx is not None:
                        qx, qy = int(qpx[0]), int(qpx[1])
                        if 0 <= qx < show.shape[1] and 0 <= qy < show.shape[0]:
                            cv2.circle(show, (qx, qy), 4, (255, 120, 0), -1, cv2.LINE_AA)
                            cv2.putText(
                                show, f"Q{i}", (qx + 6, qy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 120, 0), 1, cv2.LINE_AA,
                            )

                    if cpx is not None:
                        _draw_corner_marker_2d(
                            show,
                            cpx,
                            label=f"C{i}",
                            color=(0, 0, 255),
                        )

            cv2.imshow(WINDOW_NAME, show)



            key = cv2.waitKey(1) & 0xFF



            if key in (27, ord("q"), ord("Q")):

                break

            elif key == ord("1"):

                state["display_mode"] = 1

            elif key == ord("2"):

                state["display_mode"] = 2

            elif key == ord("3"):

                state["display_mode"] = 3



            elif key in (ord("p"), ord("P")):

                if state["latest"] is None:

                    continue



                print("\n[P] freezing current aligned RGB-D frame...")

                frozen = build_frozen_state(state["latest"])

                if frozen is None:

                    continue



                state["frozen"] = frozen

                state["mode"] = "frozen"

                state["last_inspection"] = None



                # Automatic scene pipeline:
                # voxel -> normals -> FPS query visualization -> all-query corner
                # detection -> response cut -> radius clustering -> final 3-D view
                # -> 2-D RGB projection.
                run_scene_corner_detector(
                    frozen,
                    INSPECT_RADIUS_MM,
                )

                print(
                    "[P] automatic scene detection complete. "
                    "V=reopen raw pre-clustering candidates, "
                    "C=reopen final 3-D corners, H=optional Harris3D diagnostic, "
                    "click=single-query debug."
                )



            elif key in (ord("h"), ord("H")):

                frozen = state.get("frozen")

                if frozen is None:

                    print("[H] no frozen frame; press P first")

                else:

                    if frozen.get("response") is None:
                        print("[H] computing optional Gaussian Harris3D diagnostic...")
                        t0 = time.perf_counter()
                        frozen["response"] = compute_harris_response(
                            frozen["points_mm"],
                            frozen["normals"],
                        )
                        print(
                            f"[H] Gaussian Harris3D time: "
                            f"{time.perf_counter() - t0:.3f} s"
                        )

                    show_harris_heatmap(
                        frozen["points_mm"],
                        frozen["response"],
                        frozen["normals"],
                    )



            elif key in (ord("v"), ord("V")):
                frozen = state.get("frozen")
                if frozen is None:
                    print("[V] no frozen frame; press P first")
                else:
                    raw_records = frozen.get("corner_records_raw", [])
                    query_indices = np.asarray(
                        frozen.get("scene_query_indices", np.empty((0,), dtype=np.int64)),
                        dtype=np.int64,
                    )
                    if not raw_records:
                        print("[V] no raw pre-clustering corner hypotheses were detected")
                    else:
                        query_points = None
                        if len(query_indices) > 0:
                            pts_all = np.asarray(frozen["points_mm"], dtype=np.float64)
                            q_valid = query_indices[(query_indices >= 0) & (query_indices < len(pts_all))]
                            query_points = pts_all[q_valid] if len(q_valid) > 0 else None
                        show_raw_corner_candidates_global(
                            frozen["points_mm"],
                            [r["corner_point_mm"] for r in raw_records],
                            query_points_mm=query_points,
                        )

            elif key in (ord("c"), ord("C")):
                frozen = state.get("frozen")
                if frozen is None:
                    print("[C] no frozen frame; press P first")
                else:
                    records = frozen.get("corner_records", [])
                    if not records:
                        print("[C] no final automatic corners were detected")
                    else:
                        show_detected_corners_global(
                            frozen["points_mm"],
                            [r["corner_point_mm"] for r in records],
                            query_points_mm=None,
                        )

            elif key in (ord("t"), ord("T")):
                _reset_pixel_tuning()
                print(
                    "[pixel tuning reset] "
                    f'bilateral={PIXEL_TUNING["bilateral_sigma_space_px"]:.2f}px | '
                    f'harris sigma={PIXEL_TUNING["harris_sigma_px"]:.2f}px | '
                    f'inner smooth band={HARRIS_PAD_INNER_SMOOTH_BAND_RATIO:.2f}x sigma_H | '
                    'boundary=sobel-valid + normalized Gaussian'
                )

            elif key in (ord("r"), ord("R")):

                state["mode"] = "live"

                state["frozen"] = None

                state["last_inspection"] = None
                _clear_pixel_tuner_context()

                for name in (WEIGHTED_PROJECTION_WINDOW, FROZEN_RGBD_WINDOW, SCENE_RESULT_2D_WINDOW):

                    try:

                        cv2.destroyWindow(name)

                    except Exception:

                        pass

                print("[R] live mode")



            elif key in (ord("["), ord("{")):

                INSPECT_RADIUS_MM = max(

                    INSPECT_RADIUS_MIN_MM,

                    INSPECT_RADIUS_MM - INSPECT_RADIUS_STEP_MM,

                )

                print(f"[base radius @ {INSPECT_RADIUS_REFERENCE_Z_MM:.0f}mm] {INSPECT_RADIUS_MM:.1f} mm")



            elif key in (ord("]"), ord("}")):

                INSPECT_RADIUS_MM = min(

                    INSPECT_RADIUS_MAX_MM,

                    INSPECT_RADIUS_MM + INSPECT_RADIUS_STEP_MM,

                )

                print(f"[base radius @ {INSPECT_RADIUS_REFERENCE_Z_MM:.0f}mm] {INSPECT_RADIUS_MM:.1f} mm")



    finally:

        pipeline.stop()

        cv2.destroyAllWindows()





if __name__ == "__main__":

    main()