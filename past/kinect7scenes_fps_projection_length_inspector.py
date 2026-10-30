"""Kinect 7-Scenes / RealSense global FPS + Robust Projection Length inspector.

Peak voting debug: #6 original Hough support and voters, #9 independent
normalized direction diversity, #7 original lambda_min detector/ranking.
Optional absolute Hough lambda_min threshold is enforced BEFORE the 3-D verifier.
The score heatmaps remain unfiltered for diagnostic comparison.

Kinect input examples (NO per-frame extraction needed):
    --capture ~/datasets/7scenes/office.zip --frame-index 0
    --capture ~/datasets/7scenes/office/seq-01.zip --frame-index 250
    --capture ~/datasets/7scenes/office/seq-01 --frame-index 0

The Kinect 7-Scenes RGB and depth are NOT pre-registered. Detector and mouse
picking operate on depth-camera pixels, with optional raw RGB reference only.

Default EXT verifier: 1024-direction projection-length upper-tail score.
Scale-space diagnostic grows the EXT support and candidate ROI with Gaussian sigma:
    r(sigma) = sqrt(r0^2 + (2*sigma)^2)  [--ext-scale-radius-gain 2].
For each direction, robust projection bounds (5th/95th percentile) determine
where the candidate lies relative to the upper endpoint in physical length.
Legacy Max and Soft Rank scores remain diagnostic / optional A/B methods.
No additional SAH calls or refinement are introduced before the gate.

conda activate laser_handeye
cd ~/keypoint-corner-detector

conda activate laser_handeye
cd ~/keypoint-corner-detector

python kinect7scenes_fps_projection_length_inspector.py \
  --capture ~/datasets/7scenes/office.zip \
  --kinect-seq seq-01 \
  --frame-index 120 \
  --voxel-3d-mm 15 \
  --voxel-2d-mm 15 \
  --fps-count 500 \
  --ext-method projection-length \
  --ext-radius-x 4 \
  --ext-scale-best1 \
  --ext-scale-best1-sigmas-x 0,1,2,4 \
  --ext-scale-radius-gain 2 \
  --ext-scale-best1-directions 128 \
  --ext-scale-best1-threshold 0.15 \
  --hough-min-score 0.020 \
  --no-iteration \
  --no-refine

python kinect7scenes_fps_projection_length_inspector.py \
  --capture ~/datasets/7scenes/office.zip \
  --kinect-seq seq-01 \
  --frame-index 120 \
  --voxel-3d-mm 15 \
  --voxel-2d-mm 15 \
  --normal-radius-x 4 \
  --normal-sigma-r-x 0.75 \
  --query-radius-x 10 \
  --boundary-radius-x 4 \
  --boundary-epsilon-x 1 \
  --tensor-radius-x 4 \
  --tensor-sigma-d-x 1.5 \
  --vote-line-sigma-x 1 \
  --corner-search-radius-x 6 \
  --surface-weight-min 0.1 \
  --fps-count 500 \
  --corner-nms-radius-x 3 \
  --ext-radius-x 4 \
  --ext-method projection-length \
  --ext-top-fraction 0.20 \
  --ext-trim-fraction 0.05 \
  --ext-min-span-mm 1.0 \
  --ext-directions 1024 \
  --hough-min-score 0.000 \
  --ext-min-score 0.08 \
  --no-iteration \
  --no-refine

New 3-D EXT Scale-Space diagnostic (only after selecting a Hough candidate):
Optional Best-1 FINAL gate: --ext-scale-best1 --ext-scale-best1-threshold 0.14835
  (Threshold is a synthetic pilot value, not universally calibrated!)
  S or toolbar button: run a spatial+scale extrema test on N_r(c).
  [ / ] in new Open3D window: change sigma; rotate using mouse.
  Saves 3D score-panel PNG, per-point scores NPZ, JSON metadata.
  Original detector and 3D PASS/FAIL are unchanged.
"""

from __future__ import annotations



import argparse
import io
import re
import shutil
import zipfile

import importlib.util

import inspect

import sys

import time

from pathlib import Path

from dataclasses import dataclass



import numpy as np

from scipy.spatial import cKDTree
from scipy.special import expit

import matplotlib

# Direct Agg figures do not use pyplot or select a Qt window backend.

matplotlib.use("Agg", force=True)

from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg

from matplotlib.patches import Circle

from PIL import Image, ImageDraw



try:

    import tkinter as tk

    from PIL import ImageTk

except ImportError:

    tk = None

    ImageTk = None



try:

    import open3d as o3d

except ImportError:

    o3d = None



WIDTH, HEIGHT = 640, 480

MIN_DEPTH_M, MAX_DEPTH_M = 0.01, 5.0

VOXEL_3D_MM, VOXEL_2D_MM = 2.0, 2.0

WINDOW_FROZEN = "7-Scenes Kinect / Saved RealSense | FPS + click"

WINDOW_BOUNDARY = "Tangent projection | second-order boundary estimate"

BUILD = "Kinect7Scenes-FPS-Hough-3D-Extremality-ScaleSpace-v2-VariableRadius"
HOUGH_TOP_K = 4





@dataclass

class BoundaryConfig:

    # Base spatial scales. 3-D quantities scale with voxel_3d_mm;
    # tangent/UV quantities scale with voxel_2d_mm.
    voxel_3d_mm: float = 2.0

    voxel_2d_mm: float = 2.0

    # Resolved metric parameters (computed from voxel multipliers in configure()).
    query_radius_mm: float = 40.0

    normal_radius_mm: float = 8.0

    boundary_radius_mm: float = 16.0

    boundary_epsilon_mm: float = 2.0

    tensor_radius_mm: float = 16.0

    tensor_sigma_d_mm: float = 3.0

    # Visualization-only Hough-style vote width, measured perpendicular to lines.
    # Every valid UV observation can cast a vote onto every observed UV candidate.
    vote_line_sigma_mm: float = 2.0
    # Absolute Hough lambda_min gate, BEFORE Top-4 NMS and 3-D extremality.
    # 0 disables; default 0.015 is a trial value, NOT a calibrated threshold.
    hough_min_score: float = 0.015

    # 3-D Hough-peak verifier (original 3-D voxel cloud BEFORE surface filtering).
    # A sample is locally extreme along direction u when no neighbor extends
    # farther than tolerance epsilon along that direction.
    ext_radius_mm: float = 8.0
    ext_epsilon_mm: float = 0.5  # Used ONLY for the old hard-score comparison.
    ext_sigma_mm: float = 0.5    # Width of the new continuous directional score.
    ext_directions: int = 1024
    ext_min_score: float = 0.08   # Experimental: recalibrate for the active score.
    ext_method: str = "projection-length"  # Active, with max and soft-rank for A/B.
    ext_top_fraction: float = 0.20  # Fraction of ROBUST projected span near the upper end.
    ext_trim_fraction: float = 0.05  # Use 5th to 95th percentiles to trim outliers.
    ext_min_span_mm: float = 1.0    # Ignore directions with nearly zero observed span.
    ext_rank_tau_mm: float = 2.0   # Soft Rank comparison only (diagnostic).
    ext_scale_sigmas_x: tuple = (0, 0.5, 1, 2)
    ext_scale_mode: str = "full"
    ext_scale_radius_gain: float = 2.0  # r(sigma) = sqrt(r0**2 + (gain*sigma)**2)
    ext_scale_directions: int = 0  # 0 means same 1024 rays as the existing verifier
    ext_scale_output: str = "ext_scale_diagnostics"
    # OPTIONAL candidate-level Best-1 scale-persistence verifier (opt-in).
    # The legacy projection-length verifier is still calculated and saved.
    ext_scale_best1: bool = False
    ext_scale_best1_threshold: float = float("nan")  # MUST be provided when enabled
    ext_scale_best1_directions: int = 128
    ext_scale_best1_sigmas_x: tuple = (0, 1, 2, 4)



    # Iterative corner localizer. These all live in tangent/UV space, so their
    # metric values are resolved from 2-D voxel multipliers in configure().
    corner_search_radius_mm: float = 12.0
    corner_step_max_mm: float = 12.0

    corner_refine_radius_mm: float = 4.0

    corner_converge_mm: float = 2.0

    corner_peak_ratio: float = 0.70

    corner_max_iterations: int = 10

    # Final false-positive rejection, evaluated only after localization stops.
    # S = lambda_min, C = lambda_min / (lambda_max + lambda_min + eps).
    corner_lambda_min_threshold: float = 0.03

    corner_balance_threshold: float = 0.15

    sigma_e_deg: float = 15.0

    sigma_r_mm: float = 1.0

    normal_kernel: str = "cauchy"

    normal_iterations: int = 5

    normal_update_mode: str = "synchronous"

    surface_weight_min: float = 0.1

    min_neighbors: int = 6

    show_3d: bool = True
    enable_iteration: bool = False  # OFF by default
    enable_refine: bool = True      # ON by default



    @property

    def support_radius_mm(self):

        # Entirely a tangent/UV support quantity, so it scales with 2-D voxel size.
        return (self.query_radius_mm + 2.5 * self.boundary_radius_mm

                + 2.0 * np.sqrt(2.0) * self.voxel_2d_mm)





@dataclass
class SceneCache:
    """Frozen-scene acceleration cache shared across FPS seeds and iterations."""
    points: np.ndarray
    tree: cKDTree
    normals: np.ndarray
    normal_valid: np.ndarray
    normal_iterations: np.ndarray
    normal_backend: str
    normal_elapsed_s: float
    directions: np.ndarray


def precompute_scene_cache(points_mm, cfg, estimator):
    """Build the 3-D KD-tree and the SAH normal field once per frozen frame."""
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    tree = cKDTree(points)

    t0 = time.perf_counter()
    normals, info = estimator(
        points, radius=cfg.normal_radius_mm,
        sigma_e_deg=cfg.sigma_e_deg, sigma_r=cfg.sigma_r_mm,
        normal_kernel=cfg.normal_kernel, max_iterations=cfg.normal_iterations,
        update_mode=cfg.normal_update_mode, min_neighbors=cfg.min_neighbors,
        return_info=True, return_weights=False,
    )
    elapsed = time.perf_counter() - t0

    normals = np.asarray(normals, dtype=np.float64)
    iterations = np.asarray(info.get("iterations", np.zeros(len(points))), dtype=np.int32)
    finite = (
        np.isfinite(normals).all(axis=1)
        & (np.linalg.norm(np.nan_to_num(normals, nan=0.0), axis=1) > 1e-12)
    )
    valid = finite & (iterations > 0)

    normals = orient_normals_to_camera(points, normals)
    normals[~valid] = np.nan

    backend = str(info.get("backend", "unknown"))
    valid_count = int(np.count_nonzero(valid))
    print(
        f"[SAH precompute] valid={valid_count:,}/{len(points):,} "
        f"({100.0 * valid_count / max(len(points),1):.1f}%) | "
        f"invalid={len(points)-valid_count:,} | time={elapsed:.2f}s | backend={backend}"
    )

    return SceneCache(
        points=points, tree=tree, normals=normals, normal_valid=valid,
        normal_iterations=iterations, normal_backend=backend,
        normal_elapsed_s=float(elapsed),
        directions=fibonacci_sphere_directions(cfg.ext_directions),
    )


def build_tangent_context_cached(scene, query_index, cfg):
    """Fast tangent context using the frozen-frame SAH normal cache.

    No normal-dependency halo is needed because all normals already exist on
    the full cloud. Only the geometric support required by the current tangent
    projection is fetched from the scene KD-tree.
    """
    points = scene.points
    query_index = int(query_index)
    query = points[query_index].copy()

    if not bool(scene.normal_valid[query_index]):
        neighbor_count = len(scene.tree.query_ball_point(query, cfg.normal_radius_mm))
        raise ValueError(
            "Cached SAH query invalid"
            f" | r={cfg.normal_radius_mm:g} mm"
            f" | neighbors={neighbor_count}"
            f" | iteration={int(scene.normal_iterations[query_index])}"
        )

    support_r = float(cfg.support_radius_mm)
    if cfg.surface_weight_min > 0:
        height_bound = cfg.sigma_r_mm * np.sqrt(1.0 / cfg.surface_weight_min - 1.0)
        context_r = float(np.hypot(support_r, height_bound))
    else:
        context_r = support_r + 2.0 * cfg.normal_radius_mm

    ids = np.asarray(scene.tree.query_ball_point(query, context_r), dtype=np.int64)
    if len(ids) < cfg.min_neighbors:
        raise ValueError("Too few points in cached tangent context.")

    local = points[ids]
    normals = scene.normals[ids]
    query_pos = np.flatnonzero(ids == query_index)
    if len(query_pos) != 1:
        raise ValueError("Cached query index missing from local context.")
    query_local = int(query_pos[0])

    u, v, normal = make_tangent_basis(normals[query_local])
    delta = local - query
    uv = np.column_stack((delta @ u, delta @ v))
    weights, we, wr, height = projection_surface_weights(
        local, normals, query, normal, cfg
    )

    crop = np.linalg.norm(uv, axis=1) <= support_r
    uv_raw = uv[crop]
    ids_raw = ids[crop]
    w_raw = weights[crop]
    keep = (
        np.ones(len(uv_raw), dtype=bool)
        if cfg.surface_weight_min == 0
        else w_raw >= cfg.surface_weight_min
    )

    selected_ids = ids_raw[keep]
    selected_uv = uv_raw[keep]
    if len(selected_uv) < cfg.min_neighbors:
        raise ValueError("Too few points after surface filtering; lower --surface-weight-min.")

    query_selected = np.flatnonzero(selected_ids == query_index)
    if len(query_selected) != 1:
        raise ValueError("Query was excluded by projection filtering.")

    representatives, query_uv_id = voxel_representatives_uv(
        selected_uv, cfg.voxel_2d_mm, int(query_selected[0])
    )
    support_uv = selected_uv[representatives]
    support_ids = selected_ids[representatives]
    if len(support_uv) < cfg.min_neighbors:
        raise ValueError("Too few 2-D cells; reduce --voxel-2d-mm.")

    normal_neighbor_ids = np.asarray(
        scene.tree.query_ball_point(query, cfg.normal_radius_mm), dtype=np.int64
    )
    query_fit = {
        "indices": normal_neighbor_ids,
        "valid": True,
        "iteration": int(scene.normal_iterations[query_index]),
    }

    return {
        "query_world": query, "query_index": query_index,
        "normal": normal, "u": u, "v": v, "config": cfg,
        "support_uv_raw": uv_raw, "raw_world": points[ids_raw],
        "projection_surface_weight": w_raw, "projection_keep": keep,
        "projection_height": height[crop],
        "support_uv": support_uv, "support_indices": support_ids,
        "support_world": points[support_ids], "query_uv_index": query_uv_id,
        "normal_fit_weights": query_fit, "normal_backend": scene.normal_backend,
    }


def load_normal_estimator(module_path=None):

    if module_path:

        path = Path(module_path).expanduser().resolve()

        if not path.is_file():

            raise FileNotFoundError(path)

        spec = importlib.util.spec_from_file_location("_sah_boundary_module", path)

        module = importlib.util.module_from_spec(spec)

        sys.modules[spec.name] = module

        spec.loader.exec_module(module)

        estimator = module.estimate_surface_aware_normals

    else:

        try:

            from surface_aware_harris3d import estimate_surface_aware_normals

        except ImportError as exc:

            raise ImportError(

                "Put the updated surface_aware_harris3d.py beside this script, "

                "or pass --normal-module /path/to/surface_aware_harris3d.py. "

                "Open3D and pclpybridge are also required."

            ) from exc

        estimator = estimate_surface_aware_normals

    if "return_weights" not in inspect.signature(estimator).parameters:

        raise RuntimeError("The imported normal module is old: use the version with return_weights=True.")

    return estimator





def projection_surface_weights(points, normals, query, query_normal, cfg):

    """Evaluate surface compatibility over the extended projection support."""

    delta = np.asarray(points, dtype=float) - query

    normals = np.asarray(normals, dtype=float)

    valid = np.isfinite(normals).all(axis=1)

    theta = np.full(len(points), np.inf)

    theta[valid] = np.arccos(np.clip(np.abs(normals[valid] @ query_normal), 0, 1))

    angle_scaled = theta / np.deg2rad(cfg.sigma_e_deg)

    if cfg.normal_kernel == "gaussian":

        we = np.exp(-0.5 * angle_scaled ** 2)

    else:

        we = 1.0 / (1.0 + angle_scaled ** 2)

    height = delta @ query_normal

    wr = 1.0 / (1.0 + (height / cfg.sigma_r_mm) ** 2)

    surface = we * wr

    surface[~valid] = 0.0

    return surface, we, wr, height





def voxel_representatives_uv(uv, voxel_mm, query_index):

    """One ORIGINAL observation per cell, nearest its centroid.



    Keep the actual query as its cell's representative. Consequently the

    paper is evaluated at a sampled point with UV=(0,0), not a new off-cloud

    location. This also retains exact 3-D source IDs through downsampling.

    """

    uv = np.asarray(uv, dtype=float)

    if len(uv) == 0:

        return np.empty(0, dtype=np.int64), -1

    cells = np.floor(uv / voxel_mm + 0.5).astype(np.int64)

    _, inverse = np.unique(cells, axis=0, return_inverse=True)

    count = np.bincount(inverse)

    center = np.column_stack([

        np.bincount(inverse, weights=uv[:, k]) / count for k in range(2)

    ])

    d2 = np.sum((uv - center[inverse]) ** 2, axis=1)

    order = np.lexsort((np.arange(len(uv)), d2, inverse))

    first = np.r_[True, inverse[order[1:]] != inverse[order[:-1]]]

    selected = order[first]

    query_cell = int(inverse[query_index])

    selected[query_cell] = int(query_index)

    return selected.astype(np.int64), query_cell





def build_tangent_context(points_mm, query_index, cfg, estimator, scene_cache=None):
    if scene_cache is not None:
        return build_tangent_context_cached(scene_cache, query_index, cfg)
    return build_tangent_context_local(points_mm, query_index, cfg, estimator)


def build_tangent_context_local(points_mm, query_index, cfg, estimator):

    points = np.asarray(points_mm, dtype=float)

    query = points[int(query_index)].copy()

    support_r = cfg.support_radius_mm



    # Surface threshold implies |height| <= sigma_r*sqrt(1/tau-1).

    # Include that height plus the local normal dependency halo in 3-D.

    if cfg.surface_weight_min > 0:

        height_bound = cfg.sigma_r_mm * np.sqrt(1.0 / cfg.surface_weight_min - 1.0)

        normal_halo = (cfg.normal_iterations + 1) * cfg.normal_radius_mm

        context_r = np.hypot(support_r, height_bound) + normal_halo

        ids = np.asarray(cKDTree(points).query_ball_point(query, context_r), dtype=np.int64)

    else:

        ids = np.arange(len(points), dtype=np.int64)

    local = points[ids]

    if len(local) < cfg.min_neighbors:

        raise ValueError("Too few points for surface-aware normal estimation.")

    query_local = int(np.flatnonzero(ids == query_index)[0])



    normals, normal_info, fit_weights = estimator(

        local, radius=cfg.normal_radius_mm,

        sigma_e_deg=cfg.sigma_e_deg, sigma_r=cfg.sigma_r_mm,

        normal_kernel=cfg.normal_kernel, max_iterations=cfg.normal_iterations,

        update_mode=cfg.normal_update_mode, min_neighbors=cfg.min_neighbors,

        return_info=True, return_weights=True,

    )

    query_fit = fit_weights[query_local]

    if not query_fit["valid"] or not np.isfinite(normals[query_local]).all():

        raise ValueError("Query has no successful surface-aware normal update; select a better-supported point.")

    # Retain only this query's stored weight record, with scene-global IDs.

    query_fit = dict(query_fit)

    query_fit["indices"] = ids[query_fit["indices"]]

    del fit_weights

    normals = orient_normals_to_camera(local, normals)

    u, v, normal = make_tangent_basis(normals[query_local])

    delta = local - query

    uv = np.column_stack((delta @ u, delta @ v))

    weights, we, wr, height = projection_surface_weights(local, normals, query, normal, cfg)

    crop = np.linalg.norm(uv, axis=1) <= support_r

    uv_raw = uv[crop]

    ids_raw = ids[crop]

    w_raw = weights[crop]

    keep = np.ones(len(uv_raw), dtype=bool) if cfg.surface_weight_min == 0 else w_raw >= cfg.surface_weight_min

    selected_ids = ids_raw[keep]

    selected_uv = uv_raw[keep]

    if len(selected_uv) < cfg.min_neighbors:

        raise ValueError("Too few points after surface filtering; lower --surface-weight-min.")

    query_selected = np.flatnonzero(selected_ids == query_index)

    if len(query_selected) != 1:

        raise ValueError("Query was excluded by projection filtering.")

    representatives, query_uv_id = voxel_representatives_uv(

        selected_uv, cfg.voxel_2d_mm, int(query_selected[0]),

    )

    support_uv = selected_uv[representatives]

    support_ids = selected_ids[representatives]

    if len(support_uv) < cfg.min_neighbors:

        raise ValueError("Too few 2-D cells; reduce --uv-voxel-mm.")

    return {

        "query_world": query, "query_index": int(query_index),

        "normal": normal, "u": u, "v": v, "config": cfg,

        "support_uv_raw": uv_raw, "raw_world": points[ids_raw],

        "projection_surface_weight": w_raw, "projection_keep": keep,

        "projection_height": height[crop],

        "support_uv": support_uv, "support_indices": support_ids,

        "support_world": points[support_ids], "query_uv_index": query_uv_id,

        "normal_fit_weights": query_fit, "normal_backend": normal_info.get("backend", "unknown"),

    }





def estimate_boundary_uv(points_uv, radius, epsilon, *, min_neighbors=6,

                         evaluation_radius=None, crop_radius=None,

                         voxel_mm=0.0):

    """Paper's density-corrected SECOND-order Euclidean statistic.



    Only the final classification is restricted to the evaluation ROI.

    Directions and density counts use the entire padded support.

    Invalid/numerically symmetric neighborhoods yield NaN distance instead

    of being silently classified as zero-distance boundary points.

    """

    p = np.asarray(points_uv, dtype=float)

    if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0 or not np.isfinite(p).all():

        raise ValueError("points_uv must be finite with shape (N,2), N>0")

    if radius <= 0 or epsilon <= 0 or min_neighbors < 3:

        raise ValueError("Require radius>0, epsilon>0, min_neighbors>=3")

    tree = cKDTree(p)

    neighborhoods = tree.query_ball_point(p, radius)

    half = tree.query_ball_point(p, radius * 0.5)

    density_count = np.asarray([len(a) for a in half], dtype=np.int64)

    inverse_density = 1.0 / np.maximum(density_count, 1)

    count = np.asarray([len(a) for a in neighborhoods], dtype=np.int64)

    normals = np.zeros_like(p)

    valid = np.zeros(len(p), dtype=bool)

    imbalance = np.zeros(len(p))

    for i, neighbors in enumerate(neighborhoods):

        j = np.asarray(neighbors, dtype=np.int64)

        w = inverse_density[j]

        vector = np.sum((p[j] - p[i]) * w[:, None], axis=0)

        norm = np.linalg.norm(vector)

        scale = radius * np.sum(w)

        imbalance[i] = norm / max(scale, np.finfo(float).tiny)

        if len(j) >= min_neighbors and norm > 1e-8 * scale:

            normals[i] = vector / norm

            valid[i] = True



    radial_distance = np.linalg.norm(p, axis=1)

    evaluate = np.ones(len(p), dtype=bool)

    if evaluation_radius is not None:

        evaluate &= radial_distance <= evaluation_radius

    if crop_radius is not None:

        # x -> distance neighbor (r) -> direction neighbor (r) -> density (r/2).

        evaluate &= radial_distance + 2.5 * radius + np.sqrt(2.0) * voxel_mm <= crop_radius

    distance = np.full(len(p), np.nan)

    best_id = np.full(len(p), -1, dtype=np.int64)

    # Do not normalize the averaged pair direction: its length matters.

    for i in np.flatnonzero(evaluate & valid):

        j = np.asarray(neighborhoods[i], dtype=np.int64)

        pair = np.broadcast_to(normals[i], (len(j), 2)).copy()

        same = valid[j] & ((normals[j] @ normals[i]) > 0.0)

        pair[same] = 0.5 * (normals[i] + normals[j[same]])

        scores = np.sum((p[i] - p[j]) * pair, axis=1)

        argmax = int(np.argmax(scores))

        distance[i] = max(float(scores[argmax]), 0.0)  # self score is zero

        best_id[i] = int(j[argmax])

    threshold = 1.5 * epsilon

    boundary = evaluate & valid & np.isfinite(distance) & (distance < threshold)

    return {

        "distance_mm": distance, "inward_normal_uv": normals,

        "boundary_mask": boundary, "valid_direction": valid,

        "evaluation_mask": evaluate, "best_support_index": best_id,

        "density_count": density_count, "neighbor_count": count,

        "direction_imbalance": imbalance, "threshold_mm": threshold,

    }







def estimate_corner_eigenvalues_uv(ctx, result):
    """Harris-like 2-D structure tensor from estimated boundary directions.

    Each valid UV sample j already has
        - boundary-distance estimate d_hat_j
        - inward boundary direction n_j.

    At each evaluated sample x_i, accumulate

        M_i = sum_j w_d(j) * n_j n_j^T / N_i

    where
        w_d = exp(-d_hat_j^2 / (2*sigma_d^2)).

    Spatial Gaussian weighting is intentionally disabled in this debug build:
    every valid contributor inside tensor_radius has equal spatial weight 1.
    Only boundary-distance confidence w_d remains. Therefore:
        plane/interior : lambda_max ~ 0, lambda_min ~ 0
        straight edge  : lambda_max > 0, lambda_min ~ 0
        corner         : lambda_max > 0, lambda_min > 0

    No interpolation is performed; eigenvalues are computed only at UV samples.
    """
    cfg = ctx["config"]
    uv = np.asarray(ctx["support_uv"], dtype=float)
    directions = np.asarray(result["inward_normal_uv"], dtype=float)
    distance = np.asarray(result["distance_mm"], dtype=float)
    evaluate = np.asarray(result["evaluation_mask"], dtype=bool)

    # Tensor neighborhood is unchanged, but spatial Gaussian weighting is OFF.
    tensor_radius = float(cfg.tensor_radius_mm)            # 8 x 2-D voxel = 16 mm by default
    sigma_s = np.inf  # diagnostic metadata: w_s == 1 for every contributor
    sigma_d = max(float(cfg.tensor_sigma_d_mm), np.finfo(float).eps)  # 1.5 x 2-D voxel = 3 mm by default

    source = result["valid_direction"] & np.isfinite(distance)
    target = evaluate & result["valid_direction"] & np.isfinite(distance)

    lambda_max = np.full(len(uv), np.nan, dtype=float)
    lambda_min = np.full(len(uv), np.nan, dtype=float)
    tensor_weight = np.zeros(len(uv), dtype=float)
    contributor_count = np.zeros(len(uv), dtype=np.int64)

    source_ids = np.flatnonzero(source)
    if len(source_ids) == 0:
        return {
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
            "weight_sum": tensor_weight,
            "contributor_count": contributor_count,
            "radius_mm": tensor_radius,
            "sigma_spatial_mm": sigma_s,
            "sigma_distance_mm": sigma_d,
        }

    src_uv = uv[source_ids]
    tree = cKDTree(src_uv)

    for i in np.flatnonzero(target):
        local = np.asarray(tree.query_ball_point(uv[i], tensor_radius), dtype=np.int64)
        if len(local) < cfg.min_neighbors:
            continue

        j = source_ids[local]

        # Spatial Gaussian weighting OFF: every contributor inside the
        # tensor radius has equal spatial weight 1.
        w_d = np.exp(-0.5 * (distance[j] / sigma_d) ** 2)
        w = w_d

        spatial_norm = float(len(j))
        if spatial_norm <= 0:
            continue

        n = directions[j]
        # M = mean_j( w_d(j) * n_j n_j^T )
        M = np.einsum("i,ij,ik->jk", w, n, n) / spatial_norm
        eig = np.linalg.eigvalsh(M)  # ascending

        lambda_min[i] = max(float(eig[0]), 0.0)
        lambda_max[i] = max(float(eig[1]), 0.0)
        tensor_weight[i] = float(np.sum(w)) / spatial_norm
        contributor_count[i] = len(j)

    return {
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "weight_sum": tensor_weight,
        "contributor_count": contributor_count,
        "radius_mm": tensor_radius,
        "sigma_spatial_mm": sigma_s,
        "sigma_distance_mm": sigma_d,
    }





def estimate_observed_point_hough_uv(ctx, result):
    """Observed-candidate soft Hough votes (active FPS detector; click diagnostic).

    One valid voter p_i supplies only (d_i, inward n_i). It predicts the line
      b_i = p_i - d_i*n_i,   n_i.dot(c-b_i) = 0.
    All valid voters in the evaluated ROI vote onto OBSERVED UV positions c:
      v_i(c) = exp(-d_i**2 / (2*sigma_d**2))
               * exp(-(n_i.dot(c-b_i))**2 / (2*sigma_line**2)).

    At each candidate accumulate a direction tensor of those line votes:
      M(c) = sum_i v_i(c)*n_i*n_i.T / N_voters.
    min_eigenvalue(M) rewards two independent supported directions; a straight
    edge can get many votes but should have nearly zero lambda_min. A separate
    mean-vote map shows the support strength before directional testing.

    No hard boundary distance threshold, no intersection-pair enumeration, no
    candidate boundary weight, no spatial distance along the estimated line,
    no tensor radius. Only valid finite direction/distance values are required.
    """
    cfg = ctx["config"]
    uv = np.asarray(ctx["support_uv"], dtype=float)
    normals = np.asarray(result["inward_normal_uv"], dtype=float)
    distances = np.asarray(result["distance_mm"], dtype=float)
    evaluate = np.asarray(result["evaluation_mask"], dtype=bool)
    valid = (np.asarray(result["valid_direction"], dtype=bool)
             & evaluate & np.isfinite(distances)
             & np.isfinite(normals).all(axis=1))
    voter_ids = np.flatnonzero(valid)
    candidate_ids = np.flatnonzero(evaluate & np.isfinite(uv).all(axis=1))
    support = np.full(len(uv), np.nan, dtype=float)
    response = np.full(len(uv), np.nan, dtype=float)
    diversity = np.full(len(uv), np.nan, dtype=float)
    if len(candidate_ids):
        support[candidate_ids] = 0.0
        response[candidate_ids] = 0.0
        diversity[candidate_ids] = 0.0
    output = {
        "source_indices": voter_ids, "candidate_indices": candidate_ids,
        "support": support, "response": response, "diversity": diversity,
        "peaks": [], "vote_max": 0.0, "support_max": 0.0,
        "absolute_threshold": float(cfg.hough_min_score),
        "eligible_candidate_count": 0,
        "sigma_distance_mm": float(cfg.tensor_sigma_d_mm),
        "sigma_line_mm": float(cfg.vote_line_sigma_mm),
    }
    if len(voter_ids) < 2 or not len(candidate_ids):
        return output

    n = normals[voter_ids]
    p = uv[voter_ids]
    d = distances[voter_ids]
    sigma_d = max(cfg.tensor_sigma_d_mm, 1e-12)
    sigma_line = max(cfg.vote_line_sigma_mm, 1e-12)
    boundary_confidence = np.exp(-0.5*(d/sigma_d)**2)
    # Signed normal-form line intercept: n_i dot x = n_i dot (p_i-d_i*n_i).
    b = p - d[:, None]*n
    intercept = np.einsum("ij,ij->i", n, b)
    n0, n1 = n[:, 0], n[:, 1]
    denom = float(len(voter_ids))   # fixed normalization, not sum of vote weights

    # Chunked vectorization avoids a huge [candidate, source] allocation.
    for offset in range(0, len(candidate_ids), 256):
        ids = candidate_ids[offset:offset + 256]
        c = uv[ids]
        signed_dist = c @ n.T - intercept[None, :]
        w = (np.exp(-0.5*(signed_dist/sigma_line)**2)
             * boundary_confidence[None, :])
        total = np.sum(w, axis=1) / denom
        xx = (w @ (n0*n0)) / denom
        xy = (w @ (n0*n1)) / denom
        yy = (w @ (n1*n1)) / denom
        eig_min = 0.5*(xx + yy - np.sqrt(np.maximum((xx-yy)**2 + 4*xy**2, 0.0)))
        support[ids] = total
        response[ids] = np.maximum(eig_min, 0.0)
        # D(c) = 2*lambda_min(M(c)) / trace(M(c)) = 2*H(c)/S(c).
        # Magnitude-independent diagnostic: D=0 where S is numerically zero.
        # This is NOT used for peak ranking, NMS, or the 3D verifier.
        diversity[ids] = np.clip(np.divide(
            2.0 * response[ids], total,
            out=np.zeros_like(total), where=total > 1e-12), 0.0, 1.0)

    finite_response = np.maximum(np.nan_to_num(response[candidate_ids], nan=0.0), 0.0)
    output["vote_max"] = float(np.max(finite_response))
    output["support_max"] = float(np.nanmax(support[candidate_ids]))
    if output["vote_max"] <= 1e-12:
        return output

    # Apply BOTH the preexisting per-query relative cutoff and the new
    # absolute lambda_min cutoff before Top-4 NMS and before 3-D verification.
    # Unlike a per-query relative cutoff, an absolute gate can reject an
    # entire straight-edge ROI even if its weak response is the local maximum.
    # Keep all response/support/diversity maps UNCHANGED for diagnostics.
    cutoff = max(0.20 * output["vote_max"], float(cfg.hough_min_score))
    pick = candidate_ids[finite_response >= cutoff]
    output["eligible_candidate_count"] = int(len(pick))
    pick = pick[np.argsort(response[pick])[::-1]]
    separation = max(2.0*cfg.voxel_2d_mm, 2.0*cfg.vote_line_sigma_mm)
    peaks = []
    for idx in pick:
        pos = uv[idx]
        if all(np.linalg.norm(pos - peak["uv"]) >= separation for peak in peaks):
            peaks.append({"index": int(idx), "uv": pos.copy(),
                          "value": float(response[idx]), "inside_search": True})
        if len(peaks) >= HOUGH_TOP_K:
            break
    output["peaks"] = peaks
    return output


def hough_peak_voter_diagnostics(ctx, result, peak_rank):
    """Exactly reconstruct the selected Hough peak's ORIGINAL per-voter weights.

    Returns unnormalized vote weights w_i(c), the per-voter contribution to
    lambda_min, and geometric distances ALONG the inferred infinite lines.
    This is diagnostic only; Hough scores, rankings and 3D gates are unchanged.
    """
    vote = result["hough_vote"]
    peaks = vote.get("peaks", [])
    if not (1 <= int(peak_rank) <= len(peaks)):
        return None
    cached = vote.setdefault("_voter_debug_cache", {})
    if int(peak_rank) in cached:
        return cached[int(peak_rank)]

    uv = np.asarray(ctx["support_uv"], dtype=np.float64)
    ids = np.asarray(vote["source_indices"], dtype=np.int64)
    if len(ids) < 2:
        return None
    p = uv[ids]
    n = np.asarray(result["inward_normal_uv"], dtype=np.float64)[ids]
    d = np.asarray(result["distance_mm"], dtype=np.float64)[ids]
    c = np.asarray(peaks[int(peak_rank) - 1]["uv"], dtype=np.float64)
    sigma_d = max(float(vote["sigma_distance_mm"]), 1e-12)
    sigma_l = max(float(vote["sigma_line_mm"]), 1e-12)
    b = p - d[:, None] * n
    signed_perp = np.einsum("ij,ij->i", n, c[None, :] - b)
    weight = np.exp(-0.5 * (d / sigma_d)**2) * np.exp(-0.5 * (signed_perp / sigma_l)**2)

    # Orthogonal line tangent t_i=(-n_iy,n_ix). Along-line extrapolation is
    # evaluated from the estimated boundary anchor b_i, not from the voter p_i.
    line_tangent = np.column_stack((-n[:, 1], n[:, 0]))
    signed_along = np.einsum("ij,ij->i", line_tangent, c[None, :] - b)
    along = np.abs(signed_along)
    euclidean = np.linalg.norm(p - c[None, :], axis=1)

    # Recompute M(c) from SAME weights as in estimate_observed_point_hough_uv.
    M = np.einsum("i,ij,ik->jk", weight, n, n) / len(ids)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    min_axis = eigenvectors[:, 0]
    # By Rayleigh's quotient: lambda_min = sum_i contribution_i exactly.
    minor = weight * (n @ min_axis)**2 / len(ids)
    cfg = ctx["config"]
    reference_radius = float(cfg.boundary_radius_mm)  # Diagnostic only; NOT a voting cutoff
    far = along > reference_radius
    total_w = max(float(np.sum(weight)), np.finfo(float).tiny)
    total_minor = max(float(np.sum(minor)), np.finfo(float).tiny)
    strength = weight >= 0.20 * float(np.max(weight)) if np.max(weight) > 1e-12 else np.zeros(len(weight), bool)

    out = {
        "voter_ids": ids, "point_uv": p, "boundary_uv": b,
        "candidate_uv": c, "normal_uv": n, "weight": weight,
        "signed_perpendicular_mm": signed_perp,
        "along_mm": along, "signed_along_mm": signed_along,
        "euclidean_mm": euclidean, "minor_contribution": minor,
        "eigen_min_axis": min_axis, "eigenvalues": eigenvalues,
        "reference_radius_mm": reference_radius,
        "far_mask": far, "strong_mask": strength,
        "strong_count": int(np.count_nonzero(strength)),
        "far_vote_fraction": float(np.sum(weight[far]) / total_w),
        "far_minor_fraction": float(np.sum(minor[far]) / total_minor),
        "support_reconstructed": float(np.sum(weight) / len(ids)),
        "lambda_min_reconstructed": float(np.sum(minor)),
        "weighted_mean_along_mm": float(np.sum(weight * along) / total_w),
    }
    cached[int(peak_rank)] = out
    return out


def fibonacci_sphere_directions(count):
    """Approximately equal-area antipodally paired directions on S^2."""
    count = int(count)
    if count < 16 or count % 2:
        raise ValueError("--ext-directions must be even and >= 16")
    half = count // 2
    k = np.arange(half, dtype=np.float64)
    z = 1.0 - 2.0*(k + 0.5)/half
    phi = k * (np.pi*(3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    directions = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), z))
    return np.vstack((directions, -directions))


def directional_soft_rank(delta, directions, tau_mm, top_fraction):
    """Continuous directional top-rank evidence, excluding the candidate itself.

    For each outward direction u, t_j = u dot (p_j-c).  The fraction of
    neighbors *ahead* of c is f(u) = mean_j sigmoid(t_j/tau).

    sigmoid(0)=0.5 explicitly treats tied projections as half a competitor;
    this matters on planes: a plane is NOT an extreme point simply because
    every point ties along the plane normal. The candidate is excluded from
    the denominator, so its own zero projection does not dilute the rank.

    A rank alone averaged over antipodal directions is always 0.5:
        f(u) + f(-u) = 1.
    Therefore use upper-tail evidence s(u)=clip(1-f(u)/q, 0, 1) BEFORE
    averaging.  q = top_fraction, e.g. 0.10 means a soft top-10% region.
    """
    delta = np.asarray(delta, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 3:
        raise ValueError("delta must be shaped (N,3)")
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("directions must be shaped (K,3)")
    if len(delta) == 0:
        return np.full(len(dirs), np.nan), np.full(len(dirs), np.nan)
    if not 0 < top_fraction < 0.5 or not tau_mm > 0:
        raise ValueError("top_fraction must be (0, 0.5), tau_mm > 0")
    projections = delta @ dirs.T
    ahead_fraction = expit(projections / tau_mm).mean(axis=0)
    direction_score = np.clip(1.0 - ahead_fraction / top_fraction, 0.0, 1.0)
    return ahead_fraction, direction_score


def directional_projection_length(projections, top_fraction, trim_fraction, min_span_mm):
    """Compute continuous length-based local extremality from projected differences.

    Input projections[j,k] = dot(P_j-C, u_k), with C excluded from neighbors.
    Candidate C is at projection zero. For each u_k, take robust endpoints:
        lo = Q_trim(z_j), hi = Q_(1-trim)(z_j), L = hi - lo.
    Candidate's *remaining* upper-tail distance (fraction of local length):
        d = max(hi, 0) / L
    (If hi < 0, the candidate already lies beyond the robust upper end.)
    Continuous vote:
        s = clip(1 - d/top_fraction, 0, 1).
    If robust span < min_span_mm, score is deliberately 0 (degenerate axis).
    Quantile bounds, not original Max, are what define robust physical length.
    This is a soft geometric rank, NOT exact convex-hull normal-cone angle.
    """
    z = np.asarray(projections, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or not np.isfinite(z).all():
        raise ValueError("projections must be finite (N,K), N>=1")
    if not (0 < top_fraction < 0.5 and 0 <= trim_fraction < 0.25
            and np.isfinite(min_span_mm) and min_span_mm > 0):
        raise ValueError("invalid length score parameters")

    lower, upper = np.quantile(z, [trim_fraction, 1.0 - trim_fraction], axis=0)
    span = np.maximum(0.0, upper - lower)
    tail = np.full(z.shape[1], np.inf, dtype=np.float64)
    good = span >= min_span_mm
    tail[good] = np.maximum(upper[good], 0.0) / span[good]
    score = np.clip(1.0 - tail / top_fraction, 0.0, 1.0)
    return tail, score, span, lower, upper


def verify_hough_peaks_extremality_3d(ctx, result, points3d, point_tree=None, sphere_directions=None):
    """Compare projection-length, Soft Rank, and legacy Max on SAME neighbors.

    Only cfg.ext_method controls the PASS/FAIL gate and the 3D direction colors.
    Hough, cached normals, and candidate coordinates are not modified.
    """
    cfg = ctx["config"]
    scene = np.asarray(points3d, dtype=np.float64)
    tree = point_tree if point_tree is not None else cKDTree(scene)
    directions = (sphere_directions if sphere_directions is not None
                  else fibonacci_sphere_directions(cfg.ext_directions))
    records = []
    for rank, peak in enumerate(result["hough_vote"]["peaks"], 1):
        uv_idx = int(peak["index"])
        scene_id = int(ctx["support_indices"][uv_idx])
        center = scene[scene_id]
        neighbor_ids = np.asarray(tree.query_ball_point(center, cfg.ext_radius_mm), dtype=np.int64)
        # Exclude C so it does not bias quantile bounds / rank statistics.
        comparison_delta = scene[neighbor_ids[neighbor_ids != scene_id]] - center
        enough = len(comparison_delta) >= cfg.min_neighbors

        if enough:
            projections = comparison_delta @ directions.T  # only one matrix product per peak
            # Equiv. to the original max with self (0) included.
            margins = np.maximum(np.max(projections, axis=0), 0.0)
            max_direction_score = np.exp(-0.5 * (margins / cfg.ext_sigma_mm)**2)
            max_score = float(np.mean(max_direction_score))
            hard = margins <= cfg.ext_epsilon_mm
            hard_score = float(np.mean(hard))

            ahead_fraction = expit(projections / cfg.ext_rank_tau_mm).mean(axis=0)
            rank_direction_score = np.clip(1.0 - ahead_fraction / cfg.ext_top_fraction, 0.0, 1.0)
            rank_score = float(np.mean(rank_direction_score))
            rank_sensitivity = {
                pct: float(np.mean(np.clip(1.0 - ahead_fraction/pct, 0.0, 1.0)))
                for pct in (0.05, 0.10, 0.15)
            }

            tail_fraction, length_direction_score, span, lower, upper = directional_projection_length(
                projections, cfg.ext_top_fraction, cfg.ext_trim_fraction, cfg.ext_min_span_mm)
            length_score = float(np.mean(length_direction_score))
            # Uses SAME quantile endpoints, only width threshold changes.
            length_sensitivity = {
                pct: float(np.mean(np.clip(1.0 - tail_fraction/pct, 0.0, 1.0)))
                for pct in (0.10, 0.20, 0.30)
            }
            coverage = float(np.mean(tail_fraction < cfg.ext_top_fraction))
            mean_span = float(np.mean(span))
        else:
            margins = np.full(len(directions), np.nan)
            max_direction_score = rank_direction_score = length_direction_score = np.full(len(directions), np.nan)
            ahead_fraction = tail_fraction = span = lower = upper = np.full(len(directions), np.nan)
            hard = np.zeros(len(directions), dtype=bool)
            max_score = hard_score = rank_score = length_score = float("nan")
            rank_sensitivity = {p: float("nan") for p in (0.05, 0.10, 0.15)}
            length_sensitivity = {p: float("nan") for p in (0.10, 0.20, 0.30)}
            coverage = mean_span = float("nan")

        method_scores = {"projection-length": length_score, "soft-rank": rank_score, "max": max_score}
        method_dirs = {"projection-length": length_direction_score,
                       "soft-rank": rank_direction_score, "max": max_direction_score}
        active_score = method_scores[cfg.ext_method]
        record = {
            "rank": rank, "uv_index": uv_idx, "scene_index": scene_id,
            "center": center.copy(), "hough_score": float(peak["value"]),
            "neighbor_indices": neighbor_ids, "neighbor_count": int(len(neighbor_ids)),
            "comparison_count": int(len(comparison_delta)),
            "directions": directions, "max_projection_mm": margins,
            "direction_scores": method_dirs[cfg.ext_method], "extreme_mask": hard,
            "score": active_score, "hard_score": hard_score,
            "length_score": length_score, "rank_score": rank_score, "max_score": max_score,
            "length_direction_scores": length_direction_score,
            "rank_direction_scores": rank_direction_score,
            "max_direction_scores": max_direction_score,
            "rank_ahead_fraction": ahead_fraction,
            "rank_sensitivity": rank_sensitivity,
            "length_sensitivity": length_sensitivity,
            "length_tail_fraction": tail_fraction,
            "length_span_mm": span,
            "length_robust_lower_mm": lower,
            "length_robust_upper_mm": upper,
            "top_direction_coverage": coverage, "mean_projection_span_mm": mean_span,
            "method": cfg.ext_method, "rank_tau_mm": float(cfg.ext_rank_tau_mm),
            "top_fraction": float(cfg.ext_top_fraction),
            "trim_fraction": float(cfg.ext_trim_fraction),
            "min_span_mm": float(cfg.ext_min_span_mm),
            "valid": enough,
            "pass": bool(enough and active_score >= cfg.ext_min_score),
            "radius_mm": float(cfg.ext_radius_mm),
            "epsilon_mm": float(cfg.ext_epsilon_mm),
            "sigma_mm": float(cfg.ext_sigma_mm),
            "threshold": float(cfg.ext_min_score),
        }
        if cfg.ext_scale_best1:
            best1 = verify_extremality_best1_multiscale(
                scene, scene_id, cfg, point_tree=tree)
            record["legacy_pass"] = record["pass"]
            record["best1"] = best1
            # Best-1 REPLACES the legacy scalar threshold. AND would block
            # rounded corners which were rejected by the single-scale gate.
            record["pass"] = bool(best1["pass"])
            print(f"[EXT Best1] Hough#{rank} score_min={best1['score']:.5f} "
                  f"sigmas={best1['sigma_mm']} radii={best1['radius_mm']} "
                  f"legacy={record['legacy_pass']} best1={record['pass']}", flush=True)
        records.append(record)
        peak["extremality_3d"] = record
    result["extremality_cloud_3d"] = scene
    return records



def estimate_context_boundary(ctx, include_tensor=True):

    cfg = ctx["config"]

    result = estimate_boundary_uv(

        ctx["support_uv"], cfg.boundary_radius_mm, cfg.boundary_epsilon_mm,

        min_neighbors=cfg.min_neighbors, evaluation_radius=cfg.query_radius_mm,

        crop_radius=cfg.support_radius_mm, voxel_mm=cfg.voxel_2d_mm,

    )

    qi = ctx["query_uv_index"]

    distance = result["distance_mm"][qi]

    near = bool(np.isfinite(distance) and distance < cfg.boundary_radius_mm * 0.5)

    result["query_near_boundary"] = near

    result["query_boundary_uv"] = (

        ctx["support_uv"][qi] - distance * result["inward_normal_uv"][qi]

        if near else None

    )

    # Harris-like corner evidence from the already-estimated boundary directions.
    result["corner_eigen"] = estimate_corner_eigenvalues_uv(ctx, result) if include_tensor else None
    # Diagnostic only: observed-point Hough does NOT move the original localizer.
    result["hough_vote"] = estimate_observed_point_hough_uv(ctx, result)

    return result







def estimate_corner_target_uv(ctx, result):
    """Estimate one corner target in the CURRENT tangent plane.

    The corner score is lambda_min of the boundary-direction tensor.  Search is
    deliberately restricted to a central radius so every candidate has a full
    tensor neighborhood inside the evaluated query ROI.  A small response
    plateau around the strongest sample is averaged to reduce voxel jitter.

    This function only proposes a target in UV.  The iterative localizer then
    moves by at most corner_step_max_mm and snaps that target back to an actual
    observed 3-D support sample before recomputing the tangent frame.
    """
    cfg = ctx["config"]
    uv = np.asarray(ctx["support_uv"], dtype=float)
    evaluate = np.asarray(result["evaluation_mask"], dtype=bool)
    corner = result["corner_eigen"]
    lam_min = np.asarray(corner["lambda_min"], dtype=float)
    lam_max = np.asarray(corner["lambda_max"], dtype=float)

    radial = np.linalg.norm(uv, axis=1)
    candidate = evaluate & np.isfinite(lam_min) & np.isfinite(lam_max)
    candidate &= radial <= cfg.corner_search_radius_mm

    ids = np.flatnonzero(candidate)
    if len(ids) == 0:
        return None

    peak_id = int(ids[np.argmax(lam_min[ids])])
    peak_score = float(lam_min[peak_id])
    if not np.isfinite(peak_score) or peak_score <= np.finfo(float).eps:
        return None

    peak_uv = uv[peak_id].copy()
    refine = candidate.copy()
    refine &= np.linalg.norm(uv - peak_uv, axis=1) <= cfg.corner_refine_radius_mm
    refine &= lam_min >= cfg.corner_peak_ratio * peak_score
    refine_ids = np.flatnonzero(refine)
    if len(refine_ids) == 0:
        refine_ids = np.array([peak_id], dtype=np.int64)

    # lambda_min itself is already strength-weighted by boundary proximity via
    # w_d, so use it directly rather than introducing another spatial kernel.
    weights = np.maximum(lam_min[refine_ids], 0.0)
    if float(np.sum(weights)) <= np.finfo(float).tiny:
        target_uv = peak_uv.copy()
    else:
        target_uv = np.average(uv[refine_ids], axis=0, weights=weights)

    peak_lmax = float(lam_max[peak_id])
    ratio = peak_score / max(peak_lmax, np.finfo(float).eps)
    return {
        "target_uv": np.asarray(target_uv, dtype=float),
        "peak_uv": peak_uv,
        "peak_index": peak_id,
        "peak_score": peak_score,
        "peak_lambda_max": peak_lmax,
        "peak_ratio": float(ratio),
        "refine_indices": refine_ids,
        "search_radius_mm": float(cfg.corner_search_radius_mm),
        "refine_radius_mm": float(cfg.corner_refine_radius_mm),
    }




def refine_corner_3d_subpix(points_mm, coarse_ctx, coarse_result, cfg, estimator, scene_cache=None):
    """Continuous 3-D corner refinement inspired by cv2.cornerSubPix().

    The coarse detector already decided that the final query is corner-like.
    We reuse *exactly the samples that contributed to that query's tensor*.

    For each contributor P_i:
      1) rebuild its own SAH tangent frame (u_i, v_i, s_i),
      2) run the same 2-D boundary estimator in that local frame,
      3) lift the estimated 2-D boundary-normal direction g_i to 3-D,
      4) form a 3-D boundary point B_i = P_i - d_i g_i,
      5) form the local 3-D boundary line with tangent t_i = s_i x g_i.

    The refined corner is the weighted least-squares point closest to all
    those 3-D lines:

        c* = argmin_c sum_i w_i ||(I - t_i t_i^T)(c - B_i)||^2

    with the same boundary-distance confidence used by the tensor,

        w_i = exp(-d_i^2 / (2 sigma_d^2)).

    No extra spatial window, angle threshold, distance cutoff, or ad-hoc
    refinement radius is introduced.  If the 3-D normal equations are not
    full-rank, refinement is reported as unavailable and the coarse point is
    kept by the caller.
    """
    points = np.asarray(points_mm, dtype=float)
    uv = np.asarray(coarse_ctx["support_uv"], dtype=float)
    support_indices = np.asarray(coarse_ctx["support_indices"], dtype=np.int64)
    qi = int(coarse_ctx["query_uv_index"])

    distance0 = np.asarray(coarse_result["distance_mm"], dtype=float)
    valid0 = np.asarray(coarse_result["valid_direction"], dtype=bool)
    tensor_radius = float(coarse_result["corner_eigen"]["radius_mm"])
    sigma_d = max(float(coarse_result["corner_eigen"]["sigma_distance_mm"]),
                  np.finfo(float).eps)

    # Reconstruct the exact source set used by the final query tensor.
    query_delta = uv - uv[qi]
    tensor_contributor = (
        valid0
        & np.isfinite(distance0)
        & (np.sum(query_delta * query_delta, axis=1) <= tensor_radius * tensor_radius)
    )
    local_ids = np.flatnonzero(tensor_contributor)
    scene_ids = support_indices[local_ids]

    # The 2-D UV representatives already map to original 3-D observations, but
    # keep the first occurrence deterministically if any index is repeated.
    if len(scene_ids):
        _, first = np.unique(scene_ids, return_index=True)
        scene_ids = scene_ids[np.sort(first)]

    observations = []
    A = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    I3 = np.eye(3, dtype=float)

    for scene_id in scene_ids:
        try:
            # Crucial point: every contributor gets its OWN tangent frame.
            # The final corner-query normal is therefore not shared by all
            # refinement constraints.
            ctx_i = build_tangent_context(points, int(scene_id), cfg, estimator, scene_cache)
            boundary_i = estimate_boundary_uv(
                ctx_i["support_uv"],
                cfg.boundary_radius_mm,
                cfg.boundary_epsilon_mm,
                min_neighbors=cfg.min_neighbors,
                evaluation_radius=cfg.query_radius_mm,
                crop_radius=cfg.support_radius_mm,
                voxel_mm=cfg.voxel_2d_mm,
            )

            q_i = int(ctx_i["query_uv_index"])
            d_i = float(boundary_i["distance_mm"][q_i])
            n2 = np.asarray(boundary_i["inward_normal_uv"][q_i], dtype=float)
            if (not bool(boundary_i["valid_direction"][q_i])
                    or not np.isfinite(d_i)
                    or not np.isfinite(n2).all()):
                continue

            n2_norm = float(np.linalg.norm(n2))
            if n2_norm <= np.finfo(float).eps:
                continue
            n2 = n2 / n2_norm

            # 2-D boundary normal -> 3-D tangent-space boundary normal.
            g3 = n2[0] * np.asarray(ctx_i["u"], dtype=float) + n2[1] * np.asarray(ctx_i["v"], dtype=float)
            g3_norm = float(np.linalg.norm(g3))
            if g3_norm <= np.finfo(float).eps:
                continue
            g3 = g3 / g3_norm

            s3 = np.asarray(ctx_i["normal"], dtype=float)
            s3_norm = float(np.linalg.norm(s3))
            if s3_norm <= np.finfo(float).eps:
                continue
            s3 = s3 / s3_norm

            # Boundary line direction in 3-D.  This combines two orthogonal
            # constraints: one from the boundary normal g3 and one from the
            # local surface normal s3.
            t3 = np.cross(s3, g3)
            t3_norm = float(np.linalg.norm(t3))
            if t3_norm <= np.finfo(float).eps:
                continue
            t3 = t3 / t3_norm

            p3 = np.asarray(ctx_i["query_world"], dtype=float)
            b3 = p3 - d_i * g3
            w_i = float(np.exp(-0.5 * (d_i / sigma_d) ** 2))

            # Orthogonal projector onto the plane perpendicular to the line.
            # ||P_i(c-B_i)|| is the Euclidean point-to-line distance.
            P_i = I3 - np.outer(t3, t3)
            A += w_i * P_i
            rhs += w_i * (P_i @ b3)

            observations.append({
                "scene_index": int(scene_id),
                "point": p3,
                "boundary_point": b3,
                "surface_normal": s3,
                "boundary_normal": g3,
                "boundary_tangent": t3,
                "distance_mm": d_i,
                "weight": w_i,
            })
        except Exception:
            # A contributor whose own local tangent/boundary estimate is not
            # defined simply contributes no valid 3-D measurement.
            continue

    coarse = np.asarray(coarse_ctx["query_world"], dtype=float).copy()
    if len(observations) == 0:
        return {
            "success": False,
            "reason": "no_valid_3d_constraints",
            "position": coarse,
            "coarse_position": coarse,
            "tensor_contributor_count": int(len(local_ids)),
            "observation_count": 0,
            "rank": 0,
            "condition_number": np.inf,
            "rmse_mm": np.nan,
            "shift_mm": 0.0,
            "observations": observations,
        }

    # Solve the 3x3 normal equations with an SVD-based least-squares solve.
    c_star, _, rank, singular = np.linalg.lstsq(A, rhs, rcond=None)
    rank = int(rank)
    singular = np.asarray(singular, dtype=float)
    cond = (
        float(singular[0] / singular[-1])
        if len(singular) == 3 and singular[-1] > np.finfo(float).eps
        else np.inf
    )

    if rank < 3 or not np.isfinite(c_star).all():
        return {
            "success": False,
            "reason": "rank_deficient_3d_constraints",
            "position": coarse,
            "coarse_position": coarse,
            "tensor_contributor_count": int(len(local_ids)),
            "observation_count": int(len(observations)),
            "rank": rank,
            "condition_number": cond,
            "rmse_mm": np.nan,
            "shift_mm": 0.0,
            "observations": observations,
        }

    weighted_sq = 0.0
    weight_sum = 0.0
    for obs in observations:
        t3 = np.asarray(obs["boundary_tangent"], dtype=float)
        b3 = np.asarray(obs["boundary_point"], dtype=float)
        P_i = I3 - np.outer(t3, t3)
        r = P_i @ (c_star - b3)
        w_i = float(obs["weight"])
        weighted_sq += w_i * float(r @ r)
        weight_sum += w_i

    rmse = float(np.sqrt(weighted_sq / max(weight_sum, np.finfo(float).tiny)))
    shift = float(np.linalg.norm(c_star - coarse))
    return {
        "success": True,
        "reason": "ok",
        "position": np.asarray(c_star, dtype=float),
        "coarse_position": coarse,
        "tensor_contributor_count": int(len(local_ids)),
        "observation_count": int(len(observations)),
        "rank": rank,
        "condition_number": cond,
        "rmse_mm": rmse,
        "shift_mm": shift,
        "observations": observations,
        "normal_matrix": A,
    }



def verify_corner_candidate(ctx, result, status):
    """Final corner verification using only the already-computed tensor eigenvalues.

    Localization and verification are deliberately separated:
      - lambda_min response moves the query,
      - only the FINAL location is accepted/rejected.

    Accept iff localization is eligible (converged if iteration is ON,
    single-pass if OFF) and both
        S = lambda_min >= tau_abs
        C = lambda_min / (lambda_max + lambda_min + eps) >= tau_balance
    hold at the final query sample.
    """
    cfg = ctx["config"]
    qi = int(ctx["query_uv_index"])
    corner = result["corner_eigen"]
    lmax = float(corner["lambda_max"][qi])
    lmin = float(corner["lambda_min"][qi])

    finite = bool(np.isfinite(lmax) and np.isfinite(lmin))
    if finite:
        balance = lmin / max(lmax + lmin, np.finfo(float).eps)
    else:
        balance = np.nan

    converged = status in ("converged_target", "converged_snap")
    localization_ok = converged or (not cfg.enable_iteration and status == "single_pass")
    pass_strength = finite and lmin >= cfg.corner_lambda_min_threshold
    pass_balance = finite and balance >= cfg.corner_balance_threshold
    accepted = bool(localization_ok and pass_strength and pass_balance)

    if not localization_ok:
        reason = f"localizer_not_converged:{status}"
    elif not finite:
        reason = "invalid_eigenvalues"
    elif not pass_strength:
        reason = "lambda_min_below_threshold"
    elif not pass_balance:
        reason = "eigen_balance_below_threshold"
    else:
        reason = "accepted"

    # Ranking score is diagnostic only; acceptance uses the two explicit gates above.
    score = float(lmin * balance) if finite else 0.0
    return {
        "accepted": accepted,
        "reason": reason,
        "converged": bool(converged),
        "localization_required": bool(cfg.enable_iteration),
        "localization_gate_satisfied": bool(localization_ok),
        "lambda_max": lmax,
        "lambda_min": lmin,
        "balance": float(balance) if np.isfinite(balance) else np.nan,
        "lambda_min_threshold": float(cfg.corner_lambda_min_threshold),
        "balance_threshold": float(cfg.corner_balance_threshold),
        "score": score,
    }


def localize_corner_iterative(points_mm, initial_query_index, cfg, estimator, scene_cache=None):
    """Iteratively move the query toward a local lambda_min corner maximum.

    Loop:
        query sample
          -> SAH normal / tangent frame
          -> 2-D boundary + direction tensor
          -> local corner target from lambda_min
          -> capped tangent-plane move
          -> snap to an observed support sample
          -> recompute tangent frame

    Termination is geometric: the proposed UV target is within
    corner_converge_mm, the snap returns the same 3-D sample, a sample is
    revisited (cycle), or corner_max_iterations is reached.
    """
    points = np.asarray(points_mm, dtype=float)
    current = int(initial_query_index)
    visited = {current}
    history = []
    # Store every original tangent context and its response maps.  These are the
    # computed arrays (not recomputations with potentially changed inputs).
    iteration_frames = []
    status = "max_iterations"
    final_ctx = None
    final_result = None
    # The cloud is the voxelized 3-D scene, before tangent filtering.
    voxel_tree = scene_cache.tree if scene_cache is not None else cKDTree(points)

    max_steps = int(cfg.corner_max_iterations) if cfg.enable_iteration else 1
    for iteration in range(max_steps):
        ctx = build_tangent_context(points, current, cfg, estimator, scene_cache)
        result = estimate_context_boundary(ctx)
        result["extremality_3d"] = verify_hough_peaks_extremality_3d(
            ctx, result, points, voxel_tree,
            sphere_directions=scene_cache.directions if scene_cache is not None else None)
        # First verified candidate in descending original Hough score order.
        # Stored for diagnostics; it does NOT replace active tensor localization.
        result["hough_verified_peak"] = next(
            (entry for entry in result["extremality_3d"] if entry["pass"]), None)
        final_ctx, final_result = ctx, result

        record = {
            "iteration": int(iteration),
            "query_index": int(current),
            "query_world": np.asarray(ctx["query_world"], dtype=float).copy(),
            "normal": np.asarray(ctx["normal"], dtype=float).copy(),
            "status": "running",
        }
        iteration_frames.append({"ctx": ctx, "result": result, "record": record})

        if not cfg.enable_iteration:
            record.update({"status": "single_pass", "step_uv": np.zeros(2), "move_mm": 0.0})
            history.append(record)
            status = "single_pass"
            break

        estimate = estimate_corner_target_uv(ctx, result)
        if estimate is None:
            record["status"] = "no_corner_response"
            history.append(record)
            status = "no_corner_response"
            break

        result["corner_estimate"] = estimate
        target_uv = np.asarray(estimate["target_uv"], dtype=float)
        offset_mm = float(np.linalg.norm(target_uv))
        record.update({
            "corner_target_uv": target_uv.copy(),
            "corner_peak_uv": np.asarray(estimate["peak_uv"], dtype=float).copy(),
            "corner_peak_score": float(estimate["peak_score"]),
            "corner_peak_ratio": float(estimate["peak_ratio"]),
            "target_offset_mm": offset_mm,
        })

        # If the current query is already within one 2-D voxel-scale tolerance
        # of the estimated corner center, keep this tangent frame as final.
        if offset_mm <= cfg.corner_converge_mm:
            record["step_uv"] = np.zeros(2, dtype=float)
            record["move_mm"] = 0.0
            record["status"] = "converged_target"
            history.append(record)
            status = "converged_target"
            break

        # Capped move prevents a single noisy response map from jumping across
        # the whole query ROI.  Direction comes only from the current tangent.
        step_scale = min(1.0, cfg.corner_step_max_mm / max(offset_mm, np.finfo(float).eps))
        step_uv = target_uv * step_scale
        proposed_world = (ctx["query_world"]
                          + step_uv[0] * ctx["u"]
                          + step_uv[1] * ctx["v"])

        # Snap only to CURRENT surface-compatible UV support, not to an arbitrary
        # 3-D point on another nearby face.  This keeps the motion on the
        # observed surface while still allowing the tangent to be recomputed.
        support_world = np.asarray(ctx["support_world"], dtype=float)
        support_indices = np.asarray(ctx["support_indices"], dtype=np.int64)
        snap_distance, local_id = cKDTree(support_world).query(proposed_world)
        next_index = int(support_indices[int(local_id)])
        next_world = points[next_index]
        move_mm = float(np.linalg.norm(next_world - ctx["query_world"]))

        record.update({
            "step_uv": np.asarray(step_uv, dtype=float).copy(),
            "proposed_world": np.asarray(proposed_world, dtype=float).copy(),
            "next_query_index": next_index,
            "snap_distance_mm": float(snap_distance),
            "move_mm": move_mm,
        })

        if next_index == current:
            record["status"] = "converged_snap"
            history.append(record)
            status = "converged_snap"
            break

        if next_index in visited:
            record["status"] = "cycle_detected"
            history.append(record)
            status = "cycle_detected"
            break

        if iteration == int(cfg.corner_max_iterations) - 1:
            record["status"] = "max_iterations"
            history.append(record)
            status = "max_iterations"
            break

        record["status"] = "move"
        history.append(record)
        visited.add(next_index)
        current = next_index

    if final_ctx is None or final_result is None:
        raise RuntimeError("Corner localizer produced no valid tangent context.")

    path_world = np.asarray([h["query_world"] for h in history], dtype=float)
    if path_world.ndim != 2:
        path_world = np.asarray([final_ctx["query_world"]], dtype=float)

    localization = {
        "status": status,
        "history": history,
        "iteration_frames": iteration_frames,
        "path_world": path_world,
        "initial_query_index": int(initial_query_index),
        "final_query_index": int(history[-1]["query_index"] if history else initial_query_index),
        "iterations": len(history),
    }
    localization["verification"] = verify_corner_candidate(final_ctx, final_result, status)
    return final_ctx, final_result, localization




def response_peak_in_query_uv(ctx, result, eig_key):
    """Peak location in exactly the SAME query ROI as the active localizer.

    This comparison is diagnostic. Only the original direction-based target
    affects the corner-localization trajectory.
    """
    cfg = ctx["config"]
    uv = np.asarray(ctx["support_uv"], dtype=float)
    eig = result[eig_key]
    lam_min = np.asarray(eig["lambda_min"], dtype=float)
    lam_max = np.asarray(eig["lambda_max"], dtype=float)
    mask = (np.asarray(result["evaluation_mask"], dtype=bool)
            & np.isfinite(lam_min) & np.isfinite(lam_max)
            & (np.linalg.norm(uv, axis=1) <= cfg.query_radius_mm))
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return None
    peak_id = int(candidates[np.argmax(lam_min[candidates])])
    peak_score = float(lam_min[peak_id])
    # A zero response on a straight edge is NOT a corner peak. Mirror the
    # active localizer's positive-response condition for fair visualization.
    if not np.isfinite(peak_score) or peak_score <= np.finfo(float).eps:
        return None
    return {"index": peak_id, "uv": uv[peak_id].copy(),
            "value": peak_score}


def make_boundary_figure(ctx, result, localization=None, eig_vmax=None,
                         vote_vmax=None, support_vmax=None, selected_rank=1):
    """Nine diagnostic plots; last row is Hough Support / Diversity / Lambda.

    Hough numeric scoring and peak selection are UNCHANGED. The 3x3 layout
    makes the three components comparable at the same observed UV positions.
    """
    cfg = ctx["config"]
    fig = Figure(figsize=(23.4, 17.0))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 3, left=0.050, right=0.955,
                          bottom=0.055, top=0.915, hspace=0.40, wspace=0.55)
    ax_surface, ax_dist, ax_dir = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_query_tensor, ax_dir_min, ax_ext = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_support, ax_div, ax_vote = [fig.add_subplot(gs[2, i]) for i in range(3)]

    raw = np.asarray(ctx["support_uv_raw"], dtype=float)
    keep = np.asarray(ctx["projection_keep"], dtype=bool)
    surface_weight = np.asarray(ctx["projection_surface_weight"], dtype=float)
    uv = np.asarray(ctx["support_uv"], dtype=float)
    evaluate = np.asarray(result["evaluation_mask"], dtype=bool)
    distance = np.asarray(result["distance_mm"], dtype=float)
    directions = np.asarray(result["inward_normal_uv"], dtype=float)
    valid = evaluate & np.asarray(result["valid_direction"], dtype=bool) & np.isfinite(distance)
    lim = cfg.query_radius_mm * 1.08
    qi = int(ctx["query_uv_index"])

    # 1. 3D surface compatibility projected into the current UV frame.
    visible = np.linalg.norm(raw, axis=1) <= lim * 1.42
    ax_surface.scatter(raw[visible & ~keep, 0], raw[visible & ~keep, 1],
                       s=5, c="#d0d0d0", label="Excluded points")
    use = visible & keep
    sc = ax_surface.scatter(raw[use, 0], raw[use, 1], c=surface_weight[use],
                            vmin=0, vmax=1, cmap="viridis", s=11)
    fig.colorbar(sc, ax=ax_surface, shrink=.78, label="Surface weight w_e * w_r")
    ax_surface.set_title("1. Surface selection / tangent projection")

    # 2. Boundary-distance heatmap (unchanged).
    outside = ~evaluate
    if np.any(outside):
        ax_dist.scatter(uv[outside, 0], uv[outside, 1],
                        s=7, c="#d7d7d7", alpha=.45)
    invalid = evaluate & ~np.asarray(result["valid_direction"], dtype=bool)
    if np.any(invalid):
        ax_dist.scatter(uv[invalid, 0], uv[invalid, 1],
                        s=18, c="#9a9a9a", marker="x")
    if np.any(valid):
        sc_d = ax_dist.scatter(
            uv[valid, 0], uv[valid, 1], c=distance[valid],
            s=26, cmap="turbo_r", vmin=0.0, vmax=cfg.boundary_radius_mm,
            edgecolors="none")
        fig.colorbar(sc_d, ax=ax_dist, shrink=.78,
                     label="Estimated boundary distance d_hat [mm]")
    d_q = distance[qi]
    d_status = (f"query d_hat={d_q:.3f} mm" if np.isfinite(d_q)
                else "query direction undefined")
    ax_dist.set_title("2. Boundary-distance heatmap\n" + d_status)

    # Display-only spatial thinning; this does not modify any tensor.
    def thin_arrow_ids(mask, spacing_mm):
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            return ids
        cells = np.floor(uv[ids] / spacing_mm + .5).astype(np.int64)
        center = cells.astype(float) * spacing_mm
        d2 = np.sum((uv[ids] - center) ** 2, axis=1)
        order = np.lexsort((d2, cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        first = np.r_[True, np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)]
        return ids[order[first]]

    # 3. Boundary direction field (unchanged vectors).
    arrow_spacing = max(3.0 * cfg.voxel_2d_mm, cfg.voxel_2d_mm)
    arrow_len = max(2.0 * cfg.voxel_2d_mm, 4.0)
    arrow_ids = thin_arrow_ids(valid, arrow_spacing)
    ax_dir.scatter(uv[valid, 0], uv[valid, 1], s=9, c="#cfcfcf", alpha=.75)
    if len(arrow_ids):
        outward = -directions[arrow_ids]
        ax_dir.quiver(uv[arrow_ids, 0], uv[arrow_ids, 1],
                      arrow_len * outward[:, 0], arrow_len * outward[:, 1],
                      color="#1f77b4", angles="xy", scale_units="xy", scale=1,
                      width=.0042, headwidth=3.6, headlength=4.6,
                      headaxislength=4.0)
    if valid[qi]:
        q_vec = -distance[qi] * directions[qi]
        ax_dir.quiver([uv[qi, 0]], [uv[qi, 1]], [q_vec[0]], [q_vec[1]],
                      color="#d97706", angles="xy", scale_units="xy", scale=1,
                      width=.008, headwidth=4.2, headlength=5.0, zorder=12)
        q_boundary = uv[qi] + q_vec
        ax_dir.scatter([q_boundary[0]], [q_boundary[1]], marker="D", s=52,
                       c="#d97706", zorder=13, label="Query boundary estimate")
    corner = result["corner_eigen"]
    ax_dir.add_patch(Circle((0, 0), corner["radius_mm"], fill=False,
                            ls="-", color="#d97706", lw=1.3,
                            label=f"Tensor radius {corner['radius_mm']:g} mm"))
    ax_dir.set_title("3. Boundary direction field (-n_i)\n"
                     f"arrow length={arrow_len:g} mm")
    ax_dir.legend(loc="upper right", fontsize=8)

    # 4. True query-tensor contributors: identical samples + weights as before.
    tensor_radius = float(corner["radius_mm"])
    sigma_d = float(corner["sigma_distance_mm"])
    source = np.asarray(result["valid_direction"], dtype=bool) & np.isfinite(distance)
    query_delta = uv - uv[qi]
    query_r2 = np.sum(query_delta * query_delta, axis=1)
    query_ids = np.flatnonzero(source & (query_r2 <= tensor_radius**2))
    ax_query_tensor.scatter(uv[evaluate, 0], uv[evaluate, 1],
                            s=7, c="#e1e1e1", alpha=.55)
    if len(query_ids):
        w = np.exp(-.5 * (distance[query_ids] / sigma_d) ** 2)
        w_max = float(np.max(w))
        if not np.isfinite(w_max) or w_max <= np.finfo(float).tiny:
            w_max = 1.0
        w_norm = np.clip(w / w_max, 0.0, 1.0)
        sc_w = ax_query_tensor.scatter(uv[query_ids, 0], uv[query_ids, 1],
                     c=w, s=22 + 45 * w_norm, cmap="plasma",
                     vmin=0.0, vmax=w_max, edgecolors="none", zorder=3)
        fig.colorbar(sc_w, ax=ax_query_tensor, shrink=.78,
                     label="Query tensor contribution w_d")
        ax_query_tensor.quiver(uv[query_ids, 0], uv[query_ids, 1],
                     -arrow_len * directions[query_ids, 0],
                     -arrow_len * directions[query_ids, 1],
                     color=matplotlib.colormaps["plasma"](w_norm),
                     angles="xy", scale_units="xy", scale=1,
                     width=.005, headwidth=3.5, headlength=4.4,
                     headaxislength=3.8, zorder=4)
    ax_query_tensor.add_patch(Circle((0, 0), tensor_radius, fill=False,
                          ls="-", color="#202020", lw=1.4))
    ax_query_tensor.set_title("4. QUERY tensor contributors\n"
                              f"r={tensor_radius:g} mm, sigma_d={sigma_d:g} mm")

    # 5. Active direction-tensor minimum-eigenvalue heatmap.
    eig_direction = result["corner_eigen"]
    d_min = np.asarray(eig_direction["lambda_min"], dtype=float)

    def plot_minimum(ax, eigen, eig_values, manual_vmax, title, colorbar_label):
        eig_max = np.asarray(eigen["lambda_max"], dtype=float)
        mask = evaluate & np.isfinite(eig_values) & np.isfinite(eig_max)
        if np.any(mask):
            local_vmax = (float(manual_vmax) if manual_vmax is not None
                          else float(np.percentile(eig_values[mask], 99)))
            if not np.isfinite(local_vmax) or local_vmax <= 1e-12:
                local_vmax = 1.0
            scatter = ax.scatter(uv[mask, 0], uv[mask, 1],
                                 c=eig_values[mask], s=31, cmap="magma",
                                 vmin=0, vmax=local_vmax, edgecolors="none")
            fig.colorbar(scatter, ax=ax, shrink=.78, label=colorbar_label)
        else:
            ax.text(.5, .5, "No valid tensor samples", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title)

    plot_minimum(ax_dir_min, eig_direction, d_min, eig_vmax,
                 "5. Direction tensor: lambda_min (DIAGNOSTIC)\n"
                 "independent color range; spatial weight OFF",
                 "Direction lambda_min")

    # 6. Soft line-vote strength, at OBSERVED candidates only.
    vote = result["hough_vote"]
    candidates = vote["candidate_indices"]
    sup = np.asarray(vote["support"], dtype=float)
    resp = np.asarray(vote["response"], dtype=float)
    div = np.asarray(vote["diversity"], dtype=float)
    ax_support.scatter(uv[evaluate, 0], uv[evaluate, 1],
                       s=7, c="#d7d7d7", alpha=.35)
    if len(candidates):
        sup_ceiling = float(support_vmax) if support_vmax is not None else float(vote["support_max"])
        sup_ceiling = sup_ceiling if sup_ceiling > 1e-12 else 1.0
        sc_s = ax_support.scatter(uv[candidates, 0], uv[candidates, 1],
                                  c=sup[candidates], s=31, cmap="magma", vmin=0,
                                  vmax=sup_ceiling, edgecolors="none")
        fig.colorbar(sc_s, ax=ax_support, shrink=.78,
                     label="Mean soft line vote")
    # SHOW THE ORIGINAL VOTERS for the selected Hough peak (no score changes).
    # Cyan: near along the inferred infinite line; orange: long extrapolation.
    # Hollow magenta diamonds: largest contributors to the minority eigen-axis.
    voter_debug = hough_peak_voter_diagnostics(ctx, result, selected_rank)
    if voter_debug is not None:
        src = voter_debug["point_uv"]
        cand = voter_debug["candidate_uv"]
        weight = voter_debug["weight"]
        strong = voter_debug["strong_mask"]
        far = voter_debug["far_mask"]
        sizes = 24 + 105 * weight / max(float(np.max(weight)), 1e-12)
        # Colored source circles mark ALL strong (>=20% max) voter locations.
        for mask, color, name in (
            (strong & ~far, "#06cfc3", "Near strong voter"),
            (strong & far, "#ff8a27", "Far strong voter"),
        ):
            if np.any(mask):
                ax_support.scatter(src[mask, 0], src[mask, 1], s=sizes[mask],
                                   facecolors="none", edgecolors=color, linewidths=2.0,
                                   zorder=25, label=name)
        # A handful of greatest lambda_min contributors show which directions
        # actually sustain the smaller eigenvalue; often NOT the top raw voters.
        minor = voter_debug["minor_contribution"]
        top_minor = np.argsort(minor)[::-1][:min(8, len(minor))]
        top_minor = top_minor[minor[top_minor] > 1e-12]
        if len(top_minor):
            ax_support.scatter(src[top_minor, 0], src[top_minor, 1],
                               marker="D", s=62, facecolors="none",
                               edgecolors="#e400d6", linewidths=1.8,
                               zorder=28, label="Top lambda_min contributors")
            # Boundary-normal arrows display source n_i, not Hough lines.
            normals_dbg = voter_debug["normal_uv"][top_minor]
            arrow_len_dbg = 1.15 * cfg.voxel_2d_mm
            ax_support.quiver(src[top_minor, 0], src[top_minor, 1],
                              -arrow_len_dbg * normals_dbg[:, 0],
                              -arrow_len_dbg * normals_dbg[:, 1],
                              color="#d000b9", angles="xy", scale_units="xy", scale=1,
                              width=.0045, headwidth=3.5, headlength=4.5, zorder=29)
            # For the four largest *far* minority-axis voters, show the
            # extrapolated line segment from their boundary anchor to the peak.
            far_minor_order = np.argsort(np.where(far, minor, 0.0))[::-1]
            far_minor_order = [int(j) for j in far_minor_order[:4] if far[j] and minor[j] > 1e-12]
            anchors = voter_debug["boundary_uv"]
            for j in far_minor_order:
                ax_support.plot([anchors[j, 0], cand[0]],
                                [anchors[j, 1], cand[1]], ":",
                                color="#ff8a27", linewidth=1.4, alpha=.85, zorder=23)
        ax_support.scatter([cand[0]], [cand[1]], marker="X", s=180,
                           c="#00ebff", edgecolors="#101820", linewidths=1,
                           zorder=35, label=f"Selected Hough #{selected_rank}")
        ax_support.add_patch(Circle(tuple(cand), voter_debug["reference_radius_mm"],
                                    fill=False, edgecolor="#06cfc3", lw=1,
                                    linestyle="--", alpha=.75))
        ax_support.legend(loc="lower right", fontsize=6.2, framealpha=.9)
        ax_support.set_title(
            f"6. Hough vote support + selected #{selected_rank} CONTRIBUTORS\n"
            f"far line > {voter_debug['reference_radius_mm']:g} mm: "
            f"votes {100*voter_debug['far_vote_fraction']:.0f}% / "
            f"lambda_min {100*voter_debug['far_minor_fraction']:.0f}%")
    else:
        ax_support.set_title(
            f"6. Hough line vote support (DEBUG)\n"
            f"{len(vote['source_indices'])} voters -> {len(candidates)} observed candidates")

    # 9. Direction diversity is dimensionless: D=2*lambda_min/support.
    # 0: single normal axis, 1: two equally supported orthogonal axes.
    # Same observed candidate positions and ORIGINAL votes as panels 6/7.
    ax_div.scatter(uv[evaluate, 0], uv[evaluate, 1],
                   s=7, c="#d7d7d7", alpha=.35)
    if len(candidates):
        sc_div = ax_div.scatter(uv[candidates, 0], uv[candidates, 1],
                                c=div[candidates], s=31, cmap="viridis",
                                vmin=0.0, vmax=1.0, edgecolors="none")
        fig.colorbar(sc_div, ax=ax_div, shrink=.78,
                     label="Direction diversity D=2*lambda_min/S [0,1]")
    for k, peak in enumerate(vote["peaks"], 1):
        pt = peak["uv"]
        selected = k == selected_rank
        ax_div.scatter([pt[0]], [pt[1]], marker="o",
                       s=240 if selected else 125, facecolors="none",
                       edgecolors="#00e5ff" if selected else "#f97316",
                       linewidths=3.0 if selected else 1.8, zorder=21)
        ax_div.text(pt[0] + .7*cfg.voxel_2d_mm,
                    pt[1] + .7*cfg.voxel_2d_mm, str(k),
                    color="#008a9e" if selected else "#e3700d",
                    fontsize=11 if selected else 10, weight="bold", zorder=23)
    if 1 <= selected_rank <= len(vote["peaks"]):
        peak_id = int(vote["peaks"][selected_rank-1]["index"])
        ax_div.set_title(
            "9. Direction DIVERSITY (DIAGNOSTIC ONLY)\n"
            f"selected #{selected_rank}: S={sup[peak_id]:.4f}, "
            f"D={div[peak_id]:.3f}, H={resp[peak_id]:.5f}")
    else:
        ax_div.set_title("9. Direction DIVERSITY (DIAGNOSTIC ONLY)\n"
                         "D=2*lambda_min/S | 0=one axis, 1=balanced")

    # 7. Hough-style direction tensor's minimum eigenvalue at those SAME
    # observed candidate positions (no interpolation and no point-pair crossing).
    ax_vote.scatter(uv[evaluate, 0], uv[evaluate, 1],
                    s=7, c="#d7d7d7", alpha=.35)
    if len(candidates):
        ceiling = float(vote_vmax) if vote_vmax is not None else float(vote["vote_max"])
        ceiling = ceiling if ceiling > 1e-12 else 1.0
        sc_v = ax_vote.scatter(uv[candidates, 0], uv[candidates, 1],
                                c=resp[candidates], s=31, cmap="magma", vmin=0,
                                vmax=ceiling, edgecolors="none")
        fig.colorbar(sc_v, ax=ax_vote, shrink=.78,
                     label="Hough direction tensor lambda_min")
    for k, peak in enumerate(vote["peaks"], 1):
        pt = peak["uv"]
        ext = peak.get("extremality_3d")
        mark_color = (matplotlib.colormaps["viridis"](float(np.clip(ext["score"], 0.0, 1.0)))
                      if ext and ext["valid"] else "#a0a0a0")
        selected = k == selected_rank
        ax_vote.scatter([pt[0]], [pt[1]], marker="o", s=240 if selected else 125,
                        facecolors="none", edgecolors="#00e5ff" if selected else mark_color,
                        linewidths=3.0 if selected else 2.0, zorder=21 if selected else 18)
        ax_vote.text(pt[0] + .7*cfg.voxel_2d_mm,
                     pt[1] + .7*cfg.voxel_2d_mm, str(k),
                     c="#00bdd6" if selected else mark_color, fontsize=11 if selected else 10,
                     weight="bold", zorder=23 if selected else 19)
    ax_vote.set_title(
        f"7. Combined HOUGH: H=lambda_min=S*D/2 (ACTIVE FPS)\n"
        f"abs gate H>={cfg.hough_min_score:g} | "
        f"line sigma={cfg.vote_line_sigma_mm:g} mm | {len(vote['peaks'])} peaks"
        f" | max={vote['vote_max']:.4g}")

    # 8. 3-D extremality per Hough peak; these results use the ORIGINAL prefilter
    # 3-D voxel cloud. Direction rays + local 3-D neighbors appear in a separate
    # rotatable Open3D diagnostic window (keys 1-4 select another peak).
    ext_records = result.get("extremality_3d", [])
    if ext_records:
        ranks = np.arange(1, len(ext_records) + 1)
        values = [r["score"] if r["valid"] else 0.0 for r in ext_records]
        colors = [(matplotlib.colormaps["viridis"](float(np.clip(r["score"], 0.0, 1.0)))
                   if r["valid"] else "#a0a0a0") for r in ext_records]
        bars = ax_ext.bar(ranks, values, width=0.72, color=colors, label=f"ACTIVE {cfg.ext_method}")
        # Two alternative methods as small markers; active method is the bar.
        diagnostics = (("max_score", "legacy Max", "D", "#1f2937"),
                       ("rank_score", "Soft Rank", "^", "#7c3aed"),
                       ("length_score", "Projection Length", "s", "#0891b2"))
        plotted = list(values)
        for key, name, marker, color in diagnostics:
            if ((key == "max_score" and cfg.ext_method == "max") or
                (key == "rank_score" and cfg.ext_method == "soft-rank") or
                (key == "length_score" and cfg.ext_method == "projection-length")):
                continue
            baseline = np.array([r[key] for r in ext_records])
            ax_ext.scatter(ranks, baseline, s=42, marker=marker, color=color,
                           zorder=20, label=name)
            plotted.extend(baseline.tolist())
        if 1 <= selected_rank <= len(ext_records):
            bars[selected_rank - 1].set_edgecolor("#00e5ff")
            bars[selected_rank - 1].set_linewidth(3.0)
        ax_ext.axhline(cfg.ext_min_score, ls="--", lw=1.6, color="#282828",
                       label=f"test threshold={cfg.ext_min_score:g}")
        for bar, entry in zip(bars, ext_records):
            val = entry["score"]
            label = f"{val:.3f}" if np.isfinite(val) else "N/A"
            ax_ext.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.012, label, ha="center", fontsize=9)
        finite_plotted = [v for v in plotted if np.isfinite(v)]
        ax_ext.set(xlim=(0.3, max(4.7, len(ext_records)+0.7)),
                       ylim=(0, max(.20, max(finite_plotted, default=0)*1.30)),
                       xticks=ranks, xlabel="Hough peak rank", ylabel="Mean directional score")
        ax_ext.legend(loc="upper right", fontsize=8)
    else:
        ax_ext.text(.5, .5,
                    f"No Hough peaks passing H >= {cfg.hough_min_score:g}\n"
                    f"(local max={vote['vote_max']:.5f})",
                    ha="center", va="center", transform=ax_ext.transAxes)
    ax_ext.set_title(f"8. 3D {cfg.ext_method.upper()} verifier | selected #{selected_rank}\n"
                     f"r={cfg.ext_radius_mm:g} mm, length top={100*cfg.ext_top_fraction:g}%, "
                     f"trim={100*cfg.ext_trim_fraction:g}%, K={cfg.ext_directions}")

    # Common current query, trajectory and the ORIGINAL direction target.
    path_uv = None
    if localization is not None and len(localization.get("path_world", [])):
        pw = np.asarray(localization["path_world"], dtype=float)
        dq = pw - ctx["query_world"]
        path_uv = np.column_stack((dq @ ctx["u"], dq @ ctx["v"]))
    all_axes = (ax_surface, ax_dist, ax_dir, ax_query_tensor,
                ax_dir_min, ax_support, ax_div, ax_vote)  # ax_ext uses rank axis.
    for ax in all_axes:
        query_color = "#d02531" if ax is ax_surface else "#202020"
        ax.scatter([0], [0], marker="+", s=170, linewidths=2,
                   color=query_color, zorder=15)
        if path_uv is not None and len(path_uv):
            ax.plot(path_uv[:, 0], path_uv[:, 1], "-o",
                    color="#2f6f9f", lw=1.5, ms=4, alpha=.9, zorder=14)
            ax.scatter([path_uv[0, 0]], [path_uv[0, 1]],
                       marker="s", s=42, c="#2f6f9f", zorder=15)
        ax.add_patch(Circle((0, 0), cfg.query_radius_mm, fill=False,
                            ls="--", color="#a9b1b5", lw=.8))
        ax.axhline(0, color="#e8e8e8", lw=.6, zorder=0)
        ax.axvline(0, color="#e8e8e8", lw=.6, zorder=0)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
               xlabel="u [mm]", ylabel="v [mm]", aspect="equal")

    # Diagnostic peak of the active direction tensor; full query ROI.
    direction_peak = response_peak_in_query_uv(ctx, result, "corner_eigen")
    for ax in (ax_dir_min, ax_vote, ax_support):
        if direction_peak is not None:
            a = direction_peak["uv"]
            ax.scatter([a[0]], [a[1]], marker="o", s=175,
                       facecolors="none", edgecolors="#00d4ff", linewidths=2.2,
                       zorder=22, label="Direction peak")
    for ax in (ax_dir_min, ax_vote):
        if direction_peak is not None or (ax is ax_vote and vote["peaks"]):
            ax.legend(loc="upper right", fontsize=8)

    # Red star remains the corner target chosen by the EXISTING direction tensor.
    est = result.get("corner_estimate")
    if est is not None:
        target = np.asarray(est["target_uv"], dtype=float)
        for ax in all_axes:
            ax.scatter([target[0]], [target[1]], marker="*", s=125,
                       c="#d02531", edgecolors="white", linewidths=.6, zorder=24)

    if np.isfinite(d_min[qi]):
        ax_dir_min.text(.02, .02, f"query D lambda_min={d_min[qi]:.5f}",
                        transform=ax_dir_min.transAxes, fontsize=9, va="bottom")

    verdict_text = ""
    if localization is not None:
        verification = localization.get("verification")
        verdict = ("ACCEPT" if verification.get("accepted", False) else "REJECT") \
                  if verification is not None else "INTERMEDIATE"
        verdict_text = (f" | iteration {localization.get('frame_number', '?')}/"
                        f"{localization.get('total_frames', '?')}"
                        f" | step={localization.get('status', '?')} | {verdict}")
    peak_text = ("Direction peak shown" if direction_peak is not None
                 else "No positive direction lambda_min peak")
    fig.suptitle(
        f"boundary r={cfg.boundary_radius_mm:g} mm | tensor r={tensor_radius:g} mm"
        f" | query/search r={cfg.query_radius_mm:g} mm{verdict_text}\n"
        f"{peak_text} (current tangent UV) | cyan circle=Direction peak; "
        "red star=tensor-only target | Hough+EXT=FPS candidate gate",
        fontsize=10,
    )
    return fig


def figure_to_rgb(fig):

    """Render into an owned RGB image, independent of the figure lifetime."""

    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())

    return rgba[:, :, :3].copy()





def _min_color_limit(localization, key):
    """Independent lambda_min ceiling for one method; fixed over iterations.

    This uses the same 99th percentile rule as the original renderer, but on
    lambda_min rather than lambda_max so the actual corner-response contrast
    is visible. Saturation above the ceiling is intentional and only visual.
    """
    p99 = []
    for frame in localization["iteration_frames"]:
        vals = np.asarray(frame["result"][key]["lambda_min"], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            p99.append(float(np.percentile(vals, 99)))
    value = max(p99, default=1.0)
    return value if np.isfinite(value) and value > 1e-12 else 1.0


def iteration_heatmap_limit(localization):
    return _min_color_limit(localization, "corner_eigen")


def iteration_vote_heatmap_limit(localization, key="vote_max"):
    """Fixed Hough comparison scales across all iterations of a query."""
    vmax = max((float(frame["result"]["hough_vote"][key])
                for frame in localization["iteration_frames"]), default=0.0)
    return vmax if np.isfinite(vmax) and vmax > 1e-12 else 1.0


def iteration_localization_view(localization, frame_index):
    """History prefix in the current tangent frame; final verdict only at end."""
    frames = localization["iteration_frames"]
    idx = int(frame_index)
    info = {
        "status": frames[idx]["record"]["status"],
        "frame_number": idx + 1,
        "total_frames": len(frames),
        "iterations": idx + 1,
        "path_world": localization["path_world"][:idx + 1],
    }
    # A fixed viewing orientation makes actual tangent rotations visible in 3-D
    # instead of having the camera rotate together with each new normal.
    reference = frames[0]["ctx"]
    info["reference_view_front"] = normalize(
        reference["normal"] + .4*reference["u"] + .2*reference["v"]
    )
    info["reference_view_up"] = np.asarray(reference["v"], dtype=float)
    if idx == len(frames) - 1:
        info["verification"] = localization["verification"]
    return info


def render_iteration_view(frozen, frame_index):
    """Update 2-D maps and matching Open3D tangent plane without re-estimating."""
    loc = frozen["corner_localization"]
    frames = loc["iteration_frames"]
    idx = int(np.clip(frame_index, 0, len(frames) - 1))
    frame = frames[idx]
    ctx, result = frame["ctx"], frame["result"]
    view_loc = iteration_localization_view(loc, idx)

    # Cache rendered 2-D figures on demand; the underlying numerical arrays
    # remain identical to the ones used by localization at that iteration.
    # Hough peak ranks vary by iteration. A missing peak is never silently
    # replaced by a synthetic candidate; clamp navigation to available peaks.
    peak_count = len(result.get("extremality_3d", []))
    rank = min(max(1, int(frozen.get("extremality_rank", 1))), max(1, peak_count))
    frozen["extremality_rank"] = rank

    cache = frozen.setdefault("iteration_image_cache", {})
    cache_key = (idx, rank)
    if cache_key not in cache:
        fig = make_boundary_figure(
            ctx, result, view_loc,
            eig_vmax=frozen["iteration_eig_vmax"],
            vote_vmax=frozen["iteration_vote_vmax"],
            support_vmax=frozen["iteration_support_vmax"],
            selected_rank=rank,
        )
        try:
            cache[cache_key] = figure_to_rgb(fig)
        finally:
            fig.clear()
    frozen["boundary_image"] = cache[cache_key]
    frozen["iteration_index"] = idx

    if ctx["config"].show_3d:
        previous = frozen.get("tangent_view")
        refine = frozen.get("corner_refinement_3d") if idx == len(frames) - 1 else None
        frozen["tangent_view"] = open_tangent_view(
            ctx, result, previous, view_loc, refine
        )
        old_ext = frozen.get("extremality_view")
        frozen["extremality_view"] = open_extremality_3d_view(
            ctx, result, old_ext, rank,
            iteration_number=idx + 1, total_iterations=len(frames),
            navigation=frozen.get("extremality_controls"),
        )




# -----------------------------------------------------------------------------
# Optional spatial + Gaussian-scale diagnostic for the ORIGINAL 3-D EXT verifier.
# This does not affect Hough peak selection, 3-D PASS/FAIL, or full-scene FPS.
# -----------------------------------------------------------------------------
def _ext_scale_parse_sigmas(text, voxel_mm):
    factors = np.array([float(q.strip()) for q in str(text).split(",")], dtype=float)
    if (len(factors) < 3 or not np.isfinite(factors).all()
            or np.any(factors < 0) or np.any(np.diff(factors) <= 0)
            or factors[0] != 0):
        raise ValueError("--ext-scale-sigmas-x: use >=3 increasing values starting at 0, e.g. 0,0.5,1,2")
    return factors * float(voxel_mm)


def _ext_scale_smooth(local_xyz, source_xyz, source_tree, sigma_mm, mode="full"):
    """Gaussian-weighted 3-D sample-position smoothing (NOT volumetric convolution).

    Each target keeps its point identity. All source points within 3*sigma
    are used. Normal-only mode projects the coordinate displacement onto a
    local PCA normal. No new points are invented in unobserved space.
    """
    if sigma_mm <= 0:
        return local_xyz.copy()
    out = local_xyz.copy()
    radius = 3.0 * sigma_mm
    for i, p in enumerate(local_xyz):
        js = source_tree.query_ball_point(p, radius)
        if len(js) < 3:
            continue
        neighbors = source_xyz[js]
        d2 = np.einsum("ij,ij->i", neighbors-p, neighbors-p)
        weights = np.exp(-0.5*d2/(sigma_mm*sigma_mm))
        weights /= max(float(weights.sum()), 1e-12)
        mean = weights @ neighbors
        offset = mean - p
        if mode == "normal":
            delta = neighbors - mean
            covariance = (delta * weights[:, None]).T @ delta
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            n = eigenvectors[:, 0]
            offset = n * float(n @ offset)
        out[i] = p + offset
    return out


def _ext_scale_score(delta, directions, cfg):
    """Exactly the existing selected EXT measure, evaluated at arbitrary 3-D points."""
    if len(delta) < cfg.min_neighbors:
        return float("nan")
    projections = delta @ directions.T
    if cfg.ext_method == "projection-length":
        _, direction_scores, _, _, _ = directional_projection_length(
            projections, cfg.ext_top_fraction, cfg.ext_trim_fraction,
            cfg.ext_min_span_mm)
    elif cfg.ext_method == "soft-rank":
        ahead = expit(projections / cfg.ext_rank_tau_mm).mean(axis=0)
        direction_scores = np.clip(1.0-ahead/cfg.ext_top_fraction, 0.0, 1.0)
    else:
        margins = np.maximum(np.max(projections, axis=0), 0.0)
        direction_scores = np.exp(-0.5*(margins/cfg.ext_sigma_mm)**2)
    return float(direction_scores.mean())


def verify_extremality_best1_multiscale(points_mm, candidate_index, cfg, *,
                                         point_tree=None, directions=None):
    """Candidate-only multi-scale verifier selected by held-out synthetic tests.

    Q(c) = min_{sigma > 0} E(c_sigma, P_sigma, r(sigma));
    r(sigma) = hypot(r0, gain*sigma).

    E is the exact original selected 3-D projection-length score. The input
    candidate is not re-detected: point IDs are preserved when smoothing.
    Gaussian weighted XYZ neighbor averaging is NOT volumetric convolution.
    No spatial/scale maximum is required (strict max was empirically worse).
    """
    P = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    ci = int(candidate_index)
    if not 0 <= ci < len(P):
        raise IndexError("Candidate outside point cloud")
    if cfg.ext_method != "projection-length":
        raise ValueError("Best-1 was calibrated only for --ext-method projection-length")
    factors = np.asarray(cfg.ext_scale_best1_sigmas_x, dtype=float)
    sigmas = factors * float(cfg.voxel_3d_mm)
    if len(sigmas) < 2 or sigmas[0] != 0:
        raise ValueError("Best-1 requires sigmas starting at 0")
    gain = float(cfg.ext_scale_radius_gain)
    r0 = float(cfg.ext_radius_mm)
    radii = np.hypot(r0, gain*sigmas)
    dirs = (fibonacci_sphere_directions(cfg.ext_scale_best1_directions)
            if directions is None else directions)
    tree = point_tree if point_tree is not None else cKDTree(P)
    c0 = P[ci]
    # Build a local dependency halo: each scoring neighbor and the center
    # can move by <=3*sigma, because the smoothing kernel is truncated there.
    sigmax = float(sigmas[-1]); rmax = float(radii[-1])
    target_ids = np.sort(np.asarray(tree.query_ball_point(c0, rmax + 3*sigmax), dtype=int))
    source_ids = np.sort(np.asarray(tree.query_ball_point(c0, rmax + 6*sigmax), dtype=int))
    if len(target_ids) < cfg.min_neighbors + 1:
        return {'valid': False, 'score': float('nan'), 'pass': False,
                'sigma_mm':sigmas.tolist(), 'radius_mm':radii.tolist(), 'scores':[],
                'reason':'insufficient_local_cloud'}
    target = P[target_ids]
    source = P[source_ids]
    row = int(np.searchsorted(target_ids, ci))
    if row >= len(target_ids) or target_ids[row] != ci:
        raise RuntimeError("Best-1 candidate index not found")
    source_tree = cKDTree(source)
    values = []
    for sigma, radius in zip(sigmas[1:],radii[1:]):
        sm = _ext_scale_smooth(target, source, source_tree, float(sigma),
                               "full")
        local_tree = cKDTree(sm)
        nbr = np.asarray(local_tree.query_ball_point(sm[row],float(radius)),dtype=int)
        nbr = nbr[nbr != row]
        values.append(_ext_scale_score(sm[nbr]-sm[row], dirs, cfg))
    good = bool(len(values) == len(sigmas)-1 and np.isfinite(values).all())
    score = float(min(values)) if good else float('nan')
    return {'valid':good, 'score':score,
            'pass':bool(good and score >= cfg.ext_scale_best1_threshold),
            'sigma_mm':sigmas[1:].tolist(), 'radius_mm':radii[1:].tolist(),
            'scores':[float(v) for v in values],
            'reason':'ok' if good else 'invalid_scale_support',
            'candidate_index':ci,
            'smoothing':'full', 'radius_gain':gain}


def compute_extremality_scale_space(points_mm, candidate_index, cfg, *,
                                    sigma_factors=(0, 0.5, 1, 2),
                                    smooth_mode="full", directions_count=0,
                                    radius_gain=None, progress=None):
    """Original 3-D verifier at Gaussian scale AND a scale-dependent radius.

    r(sigma) = hypot(r0, radius_gain*sigma).
      - Center candidates live in the ORIGINAL radius-r(sigma) 3-D ball.
      - Each candidate gets its own radius-r(sigma) support neighborhood in
        the smoothed 3-D point set.
      - Point IDs remain fixed across scales, with NaN outside each level's ROI.
      - Global maximum over each per-scale ROI is reported. Its strict
        space+scale comparison is diagnostic, not a SIFT/DoG detector.
      - The original existing verifier itself and Hough gates are unchanged.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if not (0 <= candidate_index < len(points)):
        raise IndexError("Candidate index out of range")
    r0 = float(cfg.ext_radius_mm)
    sigmas = np.asarray(sigma_factors, dtype=float) * float(cfg.voxel_3d_mm)
    if len(sigmas) < 3 or sigmas[0] != 0 or np.any(np.diff(sigmas) <= 0):
        raise ValueError("sigmas must start at 0 and strictly increase")
    if smooth_mode not in ("normal", "full"):
        raise ValueError("smooth_mode must be normal or full")
    gain = float(cfg.ext_scale_radius_gain if radius_gain is None else radius_gain)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("radius_gain must be finite and >= 0")
    radii = np.hypot(r0, gain*sigmas)
    rmax = float(radii[-1])
    n_dirs = int(directions_count or cfg.ext_directions)
    directions = fibonacci_sphere_directions(n_dirs)
    center = points[int(candidate_index)]
    full_tree = cKDTree(points)
    # One master ID set avoids index drift when the observation ROI grows.
    candidate_ids = np.sort(np.asarray(full_tree.query_ball_point(center, rmax), dtype=int))
    if len(candidate_ids) < 2:
        raise ValueError("Insufficient points within maximum EXT scale radius")
    candidate_distances = np.linalg.norm(points[candidate_ids] - center, axis=1)
    active = candidate_distances[None, :] <= radii[:, None] + 1e-9
    # Each scored point may query up to rmax from its SMOOTHED location;
    # both it and the support points can move. Gaussian smoothing uses at most
    # a 3*sigma neighborhood, bounding each displacement by <= 3*sigma.
    # Include that geometric movement halo around the maximum possible support.
    sigma_max = float(sigmas[-1])
    drift_halo = 3.0 * sigma_max
    target_radius = 2*rmax + 2*drift_halo
    source_radius = target_radius + drift_halo
    target_ids = np.sort(np.asarray(full_tree.query_ball_point(center, target_radius), dtype=int))
    source_ids = np.sort(np.asarray(full_tree.query_ball_point(center, source_radius), dtype=int))
    targets = points[target_ids]
    sources = points[source_ids]
    src_tree = cKDTree(sources)
    query_rows = np.searchsorted(target_ids, candidate_ids)
    center_row = int(np.searchsorted(candidate_ids, int(candidate_index)))
    if candidate_ids[center_row] != candidate_index:
        raise RuntimeError("Center missing from its own ROI")
    scores = np.full((len(sigmas), len(candidate_ids)), np.nan, dtype=np.float64)
    xyz = np.empty((len(sigmas), len(candidate_ids), 3), dtype=np.float64)
    maxima = np.full(len(sigmas), -1, dtype=int)
    for level, (sigma, radius) in enumerate(zip(sigmas, radii)):
        ids_here = np.flatnonzero(active[level])
        if progress is not None:
            progress(f"EXT scale {level+1}/{len(sigmas)}: sigma={sigma:.2f}mm "
                     f"| r={radius:.2f}mm | {len(ids_here)} queries")
        smoothed = _ext_scale_smooth(targets, sources, src_tree, float(sigma), smooth_mode)
        scale_tree = cKDTree(smoothed)
        xyz[level] = smoothed[query_rows]
        for i in ids_here:
            row = int(query_rows[i])
            local_ids = np.asarray(scale_tree.query_ball_point(smoothed[row], radius), dtype=int)
            local_ids = local_ids[local_ids != row]
            scores[level, i] = _ext_scale_score(smoothed[local_ids]-smoothed[row], directions, cfg)
        good = ids_here[np.isfinite(scores[level, ids_here])]
        if len(good):
            maxima[level] = int(good[np.argmax(scores[level, good])])
    center_scores = scores[:, center_row].copy()
    spatial_winner_is_center = maxima == center_row
    four_d_peaks = []
    for level in range(1, len(sigmas)-1):
        if maxima[level] < 0:
            continue
        i = int(maxima[level])
        score = float(scores[level, i])
        others = np.concatenate((scores[level-1], scores[level+1],
                                 np.delete(scores[level], i)))
        others = others[np.isfinite(others)]
        if len(others) and np.isfinite(score) and score > float(others.max()) + 1e-12:
            four_d_peaks.append({
                "scale_index": level, "point_index": i,
                "scene_index": int(candidate_ids[i]),
                "score": score, "sigma_mm": float(sigmas[level]),
                "radius_mm": float(radii[level]),
                "xyz_mm": xyz[level, i].tolist(),
                "is_original_candidate": bool(i == center_row),
            })
    orig_neighbors = np.asarray(full_tree.query_ball_point(center, r0), dtype=int)
    orig_neighbors = orig_neighbors[orig_neighbors != candidate_index]
    original_check = _ext_scale_score(points[orig_neighbors]-center, directions, cfg)
    return {
        "original_center": center, "candidate_index": int(candidate_index),
        "candidate_ids": candidate_ids, "candidate_row": center_row,
        "sigma_mm": sigmas, "radii_mm": radii,
        "active_roi": active, "score": scores, "xyz_mm": xyz,
        "center_scores": center_scores, "spatial_max_rows": maxima,
        "center_is_spatial_max": spatial_winner_is_center,
        "four_d_peaks": four_d_peaks, "reference_original_score": original_check,
        "radius_mm": r0, "radius_gain": gain, "method": cfg.ext_method,
        "smooth_mode": smooth_mode, "directions_count": n_dirs,
        "support_target_count": len(targets), "support_source_count": len(sources),
    }


def make_extremality_scale_figure(scan, selected_rank=1):
    """3-D scale-by-scale score cloud + response and spatial maxima displacement."""
    from matplotlib.colors import Normalize
    from matplotlib import colormaps
    sigmas = scan["sigma_mm"]
    xyz = scan["xyz_mm"] - scan["original_center"]
    score = scan["score"]
    maxi = scan["spatial_max_rows"]
    ci = scan["candidate_row"]
    r = scan["radius_mm"]
    rmax = float(scan["radii_mm"][-1])
    fig = Figure(figsize=(17, 9.5), dpi=110, facecolor="white")
    grid = fig.add_gridspec(2, len(sigmas), height_ratios=(1.65, 1.0),
                            left=.04, right=.92, top=.91, bottom=.08,
                            wspace=.16, hspace=.28)
    cmap = colormaps["turbo"]
    norm = Normalize(vmin=0, vmax=max(.10, float(np.nanmax(np.where(np.isfinite(score), score, 0.0)))))
    for level, sigma in enumerate(sigmas):
        ax = fig.add_subplot(grid[0, level], projection="3d")
        valid = np.isfinite(score[level])
        pp = xyz[level, valid]
        ax.scatter(pp[:,0], pp[:,1], pp[:,2], c=score[level,valid],
                   s=14, alpha=.90, cmap=cmap, norm=norm, depthshade=False)
        cp = xyz[level, ci]
        ax.scatter([cp[0]], [cp[1]], [cp[2]], s=165, marker="*",
                   color="magenta", edgecolors="black", linewidths=.65,
                   depthshade=False, label="Hough candidate")
        if maxi[level] >= 0:
            mp = xyz[level, maxi[level]]
            is_4d = any(rec["scale_index"] == level for rec in scan["four_d_peaks"])
            ax.scatter([mp[0]], [mp[1]], [mp[2]], s=145 if is_4d else 105,
                       marker="D" if is_4d else "o",
                       color="gold" if is_4d else "lime", edgecolors="black",
                       linewidths=1.0, depthshade=False,
                       label="ROI 3D+scale maximum" if is_4d else "ROI 3D spatial maximum")
        cscore = score[level,ci]
        shift = np.linalg.norm(xyz[level,maxi[level]] - xyz[level,ci]) if maxi[level]>=0 else np.nan
        ax.set_title(f"sigma={sigma:.2f} mm | EXT r={scan['radii_mm'][level]:.1f}mm\nC={cscore:.3f}  max shift={shift:.1f}mm", fontsize=10)
        for k in ("x", "y", "z"):
            getattr(ax,f"set_{k}lim")(-rmax, rmax)
            getattr(ax,f"set_{k}label")(k.upper()+" [mm]", fontsize=8)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=22, azim=-57)
        ax.tick_params(labelsize=7, pad=-1)
    import matplotlib.cm as cm
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes[:len(sigmas)],
                 fraction=.014, pad=.014, label="3D extremality S(p, sigma)")
    a = fig.add_subplot(grid[1, :max(1,len(sigmas)//2)])
    a.plot(sigmas, scan["center_scores"], "-o", linewidth=2.0,
           label="Fixed Hough candidate")
    best = np.array([score[j,k] if k>=0 else np.nan for j,k in enumerate(maxi)])
    a.plot(sigmas, best, "--s", linewidth=1.3, label="Max over 3D ROI")
    for rec in scan["four_d_peaks"]:
        a.axvline(rec["sigma_mm"], color="darkorchid", linestyle=":", alpha=.7)
    a.set_title("Extremality vs Gaussian scale", fontsize=11)
    a.set_xlabel("Gaussian sigma [mm]")
    a.set_ylabel("S (0 to 1)")
    a.grid(alpha=.25)
    a.legend(fontsize=9)
    b = fig.add_subplot(grid[1, max(1,len(sigmas)//2):])
    shifts = [np.linalg.norm(xyz[j,maxi[j]]-xyz[j,ci]) if maxi[j]>=0 else np.nan
              for j in range(len(sigmas))]
    b.plot(sigmas, shifts, "-o", color="darkorange", label="3D max to Hough candidate")
    b.set_title("Spatial maximum drift", fontsize=11)
    b.set_xlabel("Gaussian sigma [mm]")
    b.set_ylabel("Distance [mm]")
    b.grid(alpha=.25)
    peaks = scan["four_d_peaks"]
    verdict = ("ROI 3D+scale maxima at " + ", ".join(f"{v['sigma_mm']:g}mm" for v in peaks)) if peaks else "No strict interior ROI 3D+scale maximum"
    fig.suptitle(f"Hough #{selected_rank} | {scan['method']} 3D verifier scale-space | "
                 f"r(sigma)=hypot({r:g}, {scan['radius_gain']:g}*sigma) mm | "
                 f"smooth={scan['smooth_mode']}\n"
                 f"Magenta star=Hough candidate, Green dot=ROI spatial max, Gold diamond=ROI 3D+scale max | {verdict}",
                 fontsize=13, y=.99)
    return fig


def save_extremality_scale_diagnostic(scan, selected_rank, out_dir):
    import json
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    key = f"ext4d_scene{scan['candidate_index']}_peak{selected_rank}_{int(time.time()*1000)}"
    png = out / (key+".png")
    npz = out / (key+".npz")
    info_file = out / (key+".json")
    fig = make_extremality_scale_figure(scan, selected_rank)
    try:
        fig.savefig(png, dpi=115, facecolor="white")
    finally:
        fig.clear()
    np.savez_compressed(npz, scene_ids=scan["candidate_ids"],
                        scales_mm=scan["sigma_mm"], score=scan["score"],
                        xyz_mm=scan["xyz_mm"], center_score=scan["center_scores"],
                        spatial_max_rows=scan["spatial_max_rows"],
                        radii_mm=scan["radii_mm"], active_roi=scan["active_roi"])
    meta = {"scene_index": scan["candidate_index"], "hough_rank": int(selected_rank),
            "method": scan["method"], "sigma_mm": scan["sigma_mm"].tolist(),
            "r0_mm": scan["radius_mm"], "r_mm_by_scale": scan["radii_mm"].tolist(),
            "r_mm": scan["radius_mm"], "radius_gain": scan["radius_gain"],
            "radius_rule": "hypot(r0, radius_gain*sigma)",
            "smoothing": scan["smooth_mode"],
            "directions": scan["directions_count"], "candidate_count": len(scan["candidate_ids"]),
            "candidate_score": scan["center_scores"].tolist(),
            "original_verifier_score": float(scan["reference_original_score"]),
            "candidate_is_spatial_max": scan["center_is_spatial_max"].tolist(),
            "four_d_peaks": scan["four_d_peaks"],
            "interpretation": "Exploratory: no new gate or 2D Hough change. Point-sample Gaussian is not volumetric convolution."}
    info_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return png, npz, info_file


def open_extremality_scale_view(scan, previous=None):
    """Interactive 3-D Open3D heatmap. '[' and ']' switch Gaussian scales."""
    if o3d is None:
        print("[EXT scale] Open3D not installed; saved 3D-multipanel PNG is available")
        return None
    if previous is not None:
        try:
            previous["vis"].destroy_window()
        except Exception:
            pass
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name="3D EXT Scale-Space | [ ]: sigma | magenta=Hough, green=max",
                             width=1040, height=790):
        return None
    cloud = o3d.geometry.PointCloud()
    # Open3D often refuses to add empty geometry. Give it the initial scale first.
    cloud.points = o3d.utility.Vector3dVector(scan["xyz_mm"][0])
    cloud.paint_uniform_color([.5,.5,.5])
    r = scan["radius_mm"]
    rmax = float(scan["radii_mm"][-1])
    candidate_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(.04*r, .4))
    candidate_sphere.compute_vertex_normals()
    peak_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(.055*r, .6))
    peak_sphere.compute_vertex_normals()
    # 3 orthogonal great circles visualize the growing verification radius.
    theta = np.linspace(0.0, 2*np.pi, 97)
    unit_rings = []
    ring_edges = []
    for axis in range(3):
        base = len(unit_rings)
        for t in theta[:-1]:
            q = np.array([np.cos(t), np.sin(t), 0.0])
            if axis == 1:
                q = q[[2, 0, 1]]
            elif axis == 2:
                q = q[[0, 2, 1]]
            unit_rings.append(q)
        ring_edges.extend([[base+j, base+(j+1)%96] for j in range(96)])
    unit_rings = np.asarray(unit_rings)
    ring_lines = o3d.geometry.LineSet()
    ring_lines.lines = o3d.utility.Vector2iVector(np.asarray(ring_edges, dtype=np.int32))
    ring_lines.points = o3d.utility.Vector3dVector(
        scan["original_center"] + unit_rings*float(scan["radii_mm"][0]))
    ring_lines.paint_uniform_color([.32, .68, .96])
    vis.add_geometry(cloud)
    vis.add_geometry(candidate_sphere)
    vis.add_geometry(peak_sphere)
    vis.add_geometry(ring_lines)
    state = {"vis": vis, "level": 0, "scan": scan}
    center_pos = np.zeros(3)
    peak_pos = np.zeros(3)
    from matplotlib import colormaps
    cmap = colormaps["turbo"]
    vmax = max(.1, float(np.nanmax(np.nan_to_num(scan["score"], nan=0.0))))

    def show_scale(level):
        nonlocal center_pos, peak_pos
        state["level"] = int(np.clip(level, 0, len(scan["sigma_mm"])-1))
        j = state["level"]
        samples = scan["xyz_mm"][j]
        data = scan["score"][j]
        active_ids = np.flatnonzero(scan["active_roi"][j])
        cloud.points = o3d.utility.Vector3dVector(samples[active_ids])
        colors = np.asarray(cmap(np.clip(np.nan_to_num(data, nan=0.0)/vmax, 0, 1)))[:, :3]
        colors[~np.isfinite(data)] = [.25,.27,.3]
        cloud.colors = o3d.utility.Vector3dVector(colors[active_ids])
        cp = samples[scan["candidate_row"]]
        candidate_sphere.translate(cp-center_pos)
        center_pos = cp.copy()
        candidate_sphere.paint_uniform_color([.95,.08,.76])
        max_idx = int(scan["spatial_max_rows"][j])
        if max_idx >= 0:
            np_ = samples[max_idx]
            peak_sphere.translate(np_-peak_pos)
            peak_pos = np_.copy()
            is_4d = any(x["scale_index"] == j for x in scan["four_d_peaks"])
            peak_sphere.paint_uniform_color([1.0,.75,.02] if is_4d else [.1,1.0,.15])
        else:
            peak_sphere.paint_uniform_color([.25,.25,.25])
        ring_lines.points = o3d.utility.Vector3dVector(
            scan["original_center"] + unit_rings*float(scan["radii_mm"][j]))
        for geom in (cloud, candidate_sphere, peak_sphere, ring_lines):
            vis.update_geometry(geom)
        flag = any(x["scale_index"]==j for x in scan["four_d_peaks"])
        cscore = float(scan["center_scores"][j])
        print(f"[3D EXT SCALE] sigma={scan['sigma_mm'][j]:.2f}mm "
              f"| EXT radius={scan['radii_mm'][j]:.2f}mm "
              f"| ROI points={len(active_ids)} | "
              f"candidate score={cscore:.4f} | candidate spatial-max={bool(scan['center_is_spatial_max'][j])} "
              f"| 4D maximum at scale={flag} | left/right [ ] change scale", flush=True)
        return False

    vis.register_key_callback(ord("["), lambda _vis: show_scale(state["level"]-1))
    vis.register_key_callback(ord("]"), lambda _vis: show_scale(state["level"]+1))
    vis.register_key_callback(263, lambda _vis: show_scale(state["level"]-1))
    vis.register_key_callback(262, lambda _vis: show_scale(state["level"]+1))
    show_scale(0)
    opt = vis.get_render_option()
    opt.background_color = np.array([.04,.045,.07])
    opt.point_size = 7.0
    ctrl = vis.get_view_control()
    ctrl.set_lookat(scan["original_center"])
    ctrl.set_front([.6,-.7,-.38])
    ctrl.set_up([0,0,1])
    ctrl.set_zoom(.73)
    return state


def open_extremality_3d_view(ctx, result, previous=None, peak_rank=1,
                             iteration_number=1, total_iterations=1, navigation=None):
    """Interactive Open3D debug: unfiltered 3-D support + extreme/blocked rays.

    Candidate switching: Tk toolbar buttons / 1..4 / A,D in the 3-D window.
    Iteration switching: Tk buttons / J,L in the 3-D window.
    Rays are colored continuously RED (score=0), YELLOW (~0.5),
    GREEN (score=1). All 3-D voxel neighbors contribute to each ray score,
    even if they belong to a face excluded in tangent-plane filtering.
    """
    # Preserve the user's 3-D camera orientation when moving between peaks or
    # iterations; only the look-at center follows the selected real 3-D point.
    camera_state = None
    if previous is not None:
        try:
            old_ctrl = previous.get_view_control()
            camera_state = (np.asarray(old_ctrl.get_front(), dtype=float),
                            np.asarray(old_ctrl.get_up(), dtype=float),
                            float(old_ctrl.get_zoom()))
        except (AttributeError, RuntimeError):
            camera_state = None
        previous.destroy_window()
    candidates = result.get("extremality_3d", [])
    if not candidates:
        return None
    rank = int(np.clip(peak_rank, 1, len(candidates)))
    entry = candidates[rank-1]
    center = entry["center"]
    r = float(entry["radius_mm"])
    score = entry["score"]
    verdict = "PASS" if entry["pass"] else ("FAIL" if entry["valid"] else "INSUFFICIENT")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    best1_flag = (f" Best1={entry['best1']['score']:.3f}" if "best1" in entry else "")
    title = (f"3D {entry['method'].upper()} | Iter {iteration_number}/{total_iterations} | "
             f"Hough #{rank}/{len(candidates)} | E0={score:.3f}{best1_flag} ({verdict}) "
             f"| Len={entry['length_score']:.3f} Max={entry['max_score']:.3f} "
             f"Rank={entry['rank_score']:.3f} "
             f"len-dir={100*entry['top_direction_coverage']:.1f}% | "
             "A/D: peak | J/L: iteration | 1-4: direct peak")
    if not vis.create_window(window_name=title, width=900, height=750):
        print("[3D Extremality] Could not open viewer")
        return None

    # Make the entire prefilter voxel neighborhood visible, including other faces.
    voxel_points = np.asarray(result["extremality_cloud_3d"], dtype=np.float64)
    neighbors = voxel_points[entry["neighbor_indices"]]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(neighbors)
    cloud.paint_uniform_color([0.62, 0.67, 0.75])
    vis.add_geometry(cloud)

    # The chosen Hough candidate is always a real observed point, never a
    # synthetic intersection. Use a sphere so it remains visible when rotated.
    candidate = o3d.geometry.TriangleMesh.create_sphere(radius=max(.42, .30*ctx["config"].voxel_3d_mm))
    candidate.translate(center)
    candidate.paint_uniform_color([1.0, .80, .12])
    vis.add_geometry(candidate)

    # Every ray encodes the ACTIVE directional score (Length, Max or Soft Rank).
    # Show up to 96 rays for clarity; all sampled rays still enter E(C).
    directions = entry["directions"]
    direction_scores = entry["direction_scores"]
    ids = np.linspace(0, len(directions)-1, min(96, len(directions)), dtype=int)
    line_points = [center.copy()]
    lines, colors = [], []
    for sample in ids:
        start_index = 0
        line_points.append(center + (0.80*r)*directions[sample])
        lines.append([start_index, len(line_points)-1])
        t = float(np.clip(direction_scores[sample], 0., 1.)) if np.isfinite(direction_scores[sample]) else 0.
        # Red -> yellow -> green; 0=low evidence, 1=high evidence.
        colors.append([1.0, 2*t, .10] if t <= 0.5 else [2*(1-t), 1.0, .10])
    rays = o3d.geometry.LineSet()
    rays.points = o3d.utility.Vector3dVector(np.asarray(line_points))
    rays.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    rays.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    vis.add_geometry(rays)

    # Physical-scale neighborhood: 3 wire rings at radius r, drawn faintly.
    ring_xyz, ring_edges = [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        t = np.linspace(0.0, 2*np.pi, 72, endpoint=False)
        ring = np.repeat(center[None, :], len(t), axis=0)
        ring[:, a] += r*np.cos(t)
        ring[:, b] += r*np.sin(t)
        offset = len(ring_xyz)
        ring_xyz.extend(ring)
        ring_edges.extend([[offset+i, offset+(i+1)%len(t)] for i in range(len(t))])
    rings = o3d.geometry.LineSet()
    rings.points = o3d.utility.Vector3dVector(np.asarray(ring_xyz))
    rings.lines = o3d.utility.Vector2iVector(np.asarray(ring_edges, dtype=np.int32))
    rings.paint_uniform_color([.32, .35, .43])
    vis.add_geometry(rings)

    opt = vis.get_render_option()
    opt.background_color = np.array([.035, .04, .055])
    opt.point_size = 5.5
    ctrl = vis.get_view_control()
    ctrl.set_lookat(center)
    if camera_state is not None:
        ctrl.set_front(camera_state[0])
        ctrl.set_up(camera_state[1])
        ctrl.set_zoom(camera_state[2])
    else:
        ctrl.set_front([.65, -.7, -.33])
        ctrl.set_up([0., 0., 1.])
        ctrl.set_zoom(.72)

    # Open3D receives keys when its own window has focus. Queue navigation;
    # rebuilding Open3D windows inside a native key callback is unsafe.
    if navigation:
        for k in range(1, HOUGH_TOP_K+1):
            vis.register_key_callback(ord(str(k)),
                                      lambda _vis, number=k: (navigation["select_peak"](number), False)[1])
        vis.register_key_callback(ord("A"),
                                  lambda _vis: (navigation["step_peak"](-1), False)[1])
        vis.register_key_callback(ord("D"),
                                  lambda _vis: (navigation["step_peak"](+1), False)[1])
        vis.register_key_callback(ord("J"),
                                  lambda _vis: (navigation["step_iteration"](-1), False)[1])
        vis.register_key_callback(ord("L"),
                                  lambda _vis: (navigation["step_iteration"](+1), False)[1])
    return vis


def open_tangent_view(ctx, result, previous=None, localization=None, refinement=None):

    if previous is not None:
        previous.destroy_window()

    vis = o3d.visualization.Visualizer()
    frame_caption = ""
    if localization is not None:
        frame_caption = (f" | iteration {localization.get('frame_number', '?')}/"
                         f"{localization.get('total_frames', '?')}")
    if not vis.create_window(window_name="3D: tangent plane / query / boundary" + frame_caption,
                             width=950, height=750):

        print("[3D] Could not open tangent view; 2-D boundary plot remains available.")

        return None

    cfg = ctx["config"]

    q, u, v, n = (ctx[key] for key in ["query_world", "u", "v", "normal"])

    raw = ctx["raw_world"]

    visible = np.linalg.norm(ctx["support_uv_raw"], axis=1) <= cfg.query_radius_mm * 1.2

    colors = matplotlib.colormaps["viridis"](ctx["projection_surface_weight"])[:, :3]

    colors[~ctx["projection_keep"]] = [0.55, 0.55, 0.55]

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(raw[visible])

    cloud.colors = o3d.utility.Vector3dVector(colors[visible])

    vis.add_geometry(cloud)

    # Tangent plane shown as an actual metric UV grid rather than only an ROI
    # circle.  The frame is regenerated from each iteration's SAH normal.
    r = float(cfg.query_radius_mm)
    grid_step = max(4.0 * cfg.voxel_2d_mm, 5.0)
    grid_xyz, grid_edges, grid_colors = [], [], []

    def add_grid_segment(a, b, color):
        base = len(grid_xyz)
        grid_xyz.extend([a, b])
        grid_edges.append([base, base + 1])
        grid_colors.append(color)

    offsets = np.arange(-r + grid_step, r, grid_step)
    for offset in offsets:
        if abs(offset) >= r:
            continue
        half = float(np.sqrt(max(r*r - offset*offset, 0.0)))
        add_grid_segment(q + offset*u - half*v, q + offset*u + half*v,
                         [0.34, 0.38, 0.45])
        add_grid_segment(q + offset*v - half*u, q + offset*v + half*u,
                         [0.34, 0.38, 0.45])
    add_grid_segment(q - r*u, q + r*u, [1.0, .3, .25])      # local u
    add_grid_segment(q - r*v, q + r*v, [.2, .9, .35])      # local v
    add_grid_segment(q, q + .65*r*n, [.2, .65, 1.0])       # SAH normal
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(grid_xyz))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(grid_edges, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.asarray(grid_colors))
    vis.add_geometry(grid)

    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)

    ring = q + cfg.query_radius_mm * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)

    vertices = np.vstack((ring, q, q + 0.4 * cfg.query_radius_mm * n))

    edges = [[i, (i + 1) % len(ring)] for i in range(len(ring))] + [[100, 101]]

    line = o3d.geometry.LineSet()

    line.points = o3d.utility.Vector3dVector(vertices)

    line.lines = o3d.utility.Vector2iVector(edges)

    line.colors = o3d.utility.Vector3dVector([[.85, .65, .2]] * 100 + [[.1, .65, 1.]])

    vis.add_geometry(line)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(.4, cfg.voxel_3d_mm * .6))

    sphere.translate(q)

    sphere.paint_uniform_color([1, .15, .1])

    vis.add_geometry(sphere)

    boundary = result["boundary_mask"]

    if np.any(boundary):

        # Original 3-D samples, not lifted synthetic points.

        boundary_cloud = o3d.geometry.PointCloud()

        boundary_cloud.points = o3d.utility.Vector3dVector(ctx["support_world"][boundary])

        boundary_cloud.paint_uniform_color([1., .18, .12])

        vis.add_geometry(boundary_cloud)

    if localization is not None and len(localization.get("path_world", [])) >= 2:
        path = np.asarray(localization["path_world"], dtype=float)
        path_ls = o3d.geometry.LineSet()
        path_ls.points = o3d.utility.Vector3dVector(path)
        path_ls.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(path) - 1)]
        )
        path_ls.colors = o3d.utility.Vector3dVector(
            [[0.2, 0.55, 0.95]] * (len(path) - 1)
        )
        vis.add_geometry(path_ls)

        for k, p in enumerate(path):
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.35, cfg.voxel_3d_mm * .45))
            marker.translate(p)
            marker.paint_uniform_color([0.2, 0.55, 0.95] if k < len(path)-1 else [1.0, .15, .1])
            vis.add_geometry(marker)

    # The UV target is on this iteration's tangent plane (not yet snapped to
    # the real surface).  Orange shows the direction of the next step.
    estimate = result.get("corner_estimate")
    if estimate is not None:
        target_uv = np.asarray(estimate["target_uv"], dtype=float)
        target_world = q + target_uv[0] * u + target_uv[1] * v
        target_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(.5, cfg.voxel_3d_mm * .6)
        )
        target_marker.translate(target_world)
        target_marker.paint_uniform_color([1., .55, .08])
        vis.add_geometry(target_marker)
        move_line = o3d.geometry.LineSet()
        move_line.points = o3d.utility.Vector3dVector(np.vstack((q, target_world)))
        move_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        move_line.colors = o3d.utility.Vector3dVector([[1., .55, .08]])
        vis.add_geometry(move_line)

    # Optional continuous 3-D corner position after subpixel refinement.
    if refinement is not None and refinement.get("success", False):
        coarse = np.asarray(ctx["query_world"], dtype=float)
        refined = np.asarray(refinement["position"], dtype=float)
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(.6, .8 * cfg.voxel_3d_mm))
        marker.translate(refined)
        marker.paint_uniform_color([1.0, 0.65, 0.1])
        vis.add_geometry(marker)
        if np.linalg.norm(refined - coarse) > 1e-12:
            displacement = o3d.geometry.LineSet()
            displacement.points = o3d.utility.Vector3dVector(np.vstack((coarse, refined)))
            displacement.lines = o3d.utility.Vector2iVector([[0, 1]])
            displacement.colors = o3d.utility.Vector3dVector([[0.1, 0.9, 0.9]])
            vis.add_geometry(displacement)

    opt = vis.get_render_option()

    opt.background_color = np.array([.045, .05, .065])

    opt.point_size = 4.0

    ctrl = vis.get_view_control()

    ctrl.set_lookat(q)

    # Keep camera axes fixed across steps; only the actual tangent frame moves.
    if localization is not None and "reference_view_front" in localization:
        ctrl.set_front(localization["reference_view_front"])
        ctrl.set_up(localization["reference_view_up"])
    else:
        ctrl.set_front(normalize(n + .4 * u + .2 * v))
        ctrl.set_up(v)

    ctrl.set_zoom(.8)

    return vis







# ============================================================
# Full-scene FPS detection, linked to the same click-debug pipeline
# ============================================================

def farthest_point_sampling_count(points_mm, count):
    """Original global FPS policy: centroid-nearest start, then greedy farthest."""
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.inf
    count = min(n, max(1, int(count)))
    first = int(np.argmin(np.sum((points - points.mean(axis=0))**2, axis=1)))
    ids = np.empty(count, dtype=np.int64)
    ids[0] = first
    distances = np.sum((points - points[first])**2, axis=1)
    for k in range(1, count):
        distances[ids[:k]] = -1.0
        i = int(np.argmax(distances))
        ids[k] = i
        distances = np.minimum(distances, np.sum((points - points[i])**2, axis=1))
    distances[ids] = 0.0
    return ids, float(np.sqrt(max(0.0, np.max(distances))))


def nms_corner_candidates(candidates, radius_mm):
    """Original global greedy NMS; Hough strength ranks already-verified peaks."""
    result = []
    for c in sorted(candidates, key=lambda x: x["hough_score"], reverse=True):
        p = np.asarray(c["position"], dtype=np.float64)
        if all(np.linalg.norm(p - np.asarray(v["position"])) > radius_mm for v in result):
            result.append(c)
    return result


def detect_corners_from_fps_scene(frozen, cfg, estimator, count, nms_radius_mm,
                                  progress_callback=None):
    """Global count FPS -> identical Hough Top4 + length gate -> 3-D NMS -> optional LS.

    Scan pass deliberately skips the diagnostic Harris tensor when iteration is OFF.
    Detailed figures are computed only for the selected detection on click.
    """
    points = np.asarray(frozen["points_voxel_mm"], dtype=np.float64)
    cache = frozen.get("scene_cache")
    if cache is None:
        # Full-scene FPS must have globally comparable cached SAH normals.
        cache = precompute_scene_cache(points, cfg, estimator)
        frozen["scene_cache"] = cache
    valid_ids = np.flatnonzero(cache.normal_valid)
    if len(valid_ids) == 0:
        raise RuntimeError("FPS cannot run: zero valid global SAH normals")
    local_indices, coverage = farthest_point_sampling_count(points[valid_ids], count)
    seed_ids = valid_ids[local_indices]
    accepted, records = [], []
    t0 = time.perf_counter()
    for num, seed in enumerate(seed_ids, 1):
        try:
            if cfg.enable_iteration:
                ctx, result, loc = localize_corner_iterative(
                    points, int(seed), cfg, estimator, cache)
                query_id = int(loc["final_query_index"])
                status = loc["status"]
                num_iter = int(loc["iterations"])
            else:
                ctx = build_tangent_context(points, int(seed), cfg, estimator, cache)
                result = estimate_context_boundary(ctx, include_tensor=False)
                result["extremality_3d"] = verify_hough_peaks_extremality_3d(
                    ctx, result, points, cache.tree, sphere_directions=cache.directions)
                query_id = int(seed)
                status, num_iter = "single_pass", 1

            entries = result.get("extremality_3d", [])
            assert len(entries) == len(result["hough_vote"]["peaks"])
            passing = 0
            for e in entries:
                if not e["pass"]:
                    continue
                scene_id = int(e["scene_index"])
                pos = points[scene_id].copy()
                accepted.append({
                    "seed_index": int(seed),
                    "seed_position": points[int(seed)].copy(),
                    "query_index": query_id,
                    "scene_index": scene_id,
                    "hough_rank": int(e["rank"]),
                    "hough_score": float(e["hough_score"]),
                    "extremality_score": float(e["score"]),
                    "length_score": float(e["length_score"]),
                    "max_score": float(e["max_score"]),
                    "rank_score": float(e["rank_score"]),
                    "method": str(cfg.ext_method),
                    "score": float(e["hough_score"]),
                    "coarse_position": pos.copy(),
                    "position": pos.copy(),
                    "iterations": num_iter,
                    "status": status,
                })
                passing += 1
            records.append({
                "seed_index": int(seed), "query_index": query_id,
                "hough_count": len(entries), "passed_count": passing,
                "status": status,
                "peak_scores": [(e["rank"], e["hough_score"], e["score"], e["pass"])
                                for e in entries],
            })
        except Exception as exc:
            records.append({"seed_index": int(seed), "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
            if num <= 4:
                print(f"[FPS {num}/{len(seed_ids)}] {type(exc).__name__}: {exc}", flush=True)
        if num % 8 == 0 or num == len(seed_ids):
            msg = (f"[FPS] {num}/{len(seed_ids)} seeds | accepted={len(accepted)} "
                   f"| elapsed={time.perf_counter()-t0:.1f}s")
            print(msg, flush=True)
            if progress_callback is not None:
                progress_callback(num, len(seed_ids), len(accepted))

    corners = nms_corner_candidates(accepted, nms_radius_mm)
    print(f"[FPS NMS] {len(accepted)} passes -> {len(corners)} corners "
          f"(r={nms_radius_mm:g} mm)", flush=True)
    if cfg.enable_refine:
        # Crucially, ONLY post-NMS positions are refined; the verifier remains cheap.
        for c in corners:
            try:
                peak_ctx = build_tangent_context(points, int(c["scene_index"]), cfg, estimator, cache)
                peak_result = estimate_context_boundary(peak_ctx)
                refinement = refine_corner_3d_subpix(
                    points, peak_ctx, peak_result, cfg, estimator, cache)
                c["refinement_3d"] = refinement
                if refinement["success"]:
                    c["position"] = np.asarray(refinement["position"], dtype=float)
            except Exception as exc:
                c["refinement_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "seed_indices": seed_ids, "cover_radius_mm": coverage,
        "accepted_before_nms": accepted, "corners": corners,
        "records": records,
        "elapsed_s": float(time.perf_counter()-t0),
        "method": cfg.ext_method,
    }


def point_mm_to_pixel(p, intr):
    x, y, z = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.isfinite([x,y,z]).all() or z <= 0:
        return None
    return (int(round(intr.fx*x/z + intr.ppx)),
            int(round(intr.fy*y/z + intr.ppy)))


def pick_scene_detection(pixel, frozen, max_px=17.0):
    """RGB click on red NMS marker -> its original FPS seed, not a new query."""
    d = frozen.get("global_detection")
    if not d:
        return None
    x, y = pixel
    categories = [(d["corners"], "corner", max_px)]
    if frozen.get("show_accepted_candidates", False):
        categories.append((d["accepted_before_nms"], "accepted", 12.0))
    if frozen.get("show_fps_seeds", False):
        categories.append(([
            {"seed_index": int(i), "scene_index": None,
             "position": frozen["points_voxel_mm"][int(i)]}
            for i in d["seed_indices"]], "seed", 9.0))
    for items, kind, radius in categories:
        winner, dist = None, float(radius)
        for item in items:
            pix = point_mm_to_pixel(item["position"], frozen["intr"])
            if pix is None:
                continue
            gap = float(np.hypot(pix[0]-x, pix[1]-y))
            if gap < dist:
                winner, dist = item, gap
        if winner is not None:
            return {"kind": kind, "candidate": winner, "distance_px": dist}
    return None


def save_fps_detection_summary(frozen, cfg, detection, filename="detections_projection_length_fps.json"):
    """Non-destructive: do not overwrite the capture's original detections.json."""
    import json
    def position(a):
        return np.asarray(a, dtype=np.float64).tolist()
    def entry(c):
        return {"position_mm": position(c["position"]),
                "coarse_position_mm": position(c["coarse_position"]),
                "seed_index": c["seed_index"], "scene_index": c["scene_index"],
                "hough_rank": c["hough_rank"],
                "hough_score": c["hough_score"],
                "extremality_score": c["extremality_score"],
                "length_score": c["length_score"],
                "max_score": c["max_score"], "rank_score": c["rank_score"]}
    payload = {
        "source_capture": str(frozen["capture_path"]),
        "source_meta": frozen.get("source_meta", {}),
        "detector": BUILD,
        "ext_method": cfg.ext_method,
        "ext_min_score": cfg.ext_min_score,
        "hough_min_score": cfg.hough_min_score,
        "ext_top_fraction": cfg.ext_top_fraction,
        "ext_trim_fraction": cfg.ext_trim_fraction,
        "ext_radius_mm": cfg.ext_radius_mm,
        "fps_count_actual": len(detection["seed_indices"]),
        "fps_cover_radius_mm": detection["cover_radius_mm"],
        "nms_radius_mm": float(frozen["nms_radius_mm"]),
        "seed_indices": detection["seed_indices"].astype(int).tolist(),
        "accepted_before_nms": [entry(c) for c in detection["accepted_before_nms"]],
        "corners": [entry(c) for c in detection["corners"]],
    }
    path = Path(frozen["capture_path"]) / filename
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def open_global_fps_view(frozen, cfg, previous=None):
    """Non-blocking Open3D overview: seed cloud (blue), passes (orange), NMS (red)."""
    if previous is not None:
        previous.destroy_window()
    detection = frozen.get("global_detection")
    if detection is None or o3d is None:
        return None
    vis = o3d.visualization.Visualizer()
    if not vis.create_window("FPS scene | blue=seeds orange=passes red=NMS corners",
                             width=1040, height=750):
        return None
    pts = np.asarray(frozen["points_voxel_mm"], dtype=float)
    def cloud(xyz, rgb, point_size=None):
        if len(xyz):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=float))
            pcd.paint_uniform_color(rgb)
            vis.add_geometry(pcd)
    cloud(pts, [0.62,0.67,0.72])
    cloud(pts[np.asarray(detection["seed_indices"], dtype=int)], [0.16,0.51,0.98])
    cloud([c["coarse_position"] for c in detection["accepted_before_nms"]], [1.,.61,.13])
    cloud([c["position"] for c in detection["corners"]], [.98,.18,.16])
    # Larger spheres for final results to make them visible over the entire cloud.
    for c in detection["corners"]:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=max(1., cfg.voxel_3d_mm*.65), resolution=6)
        m.translate(np.asarray(c["position"], dtype=float))
        m.paint_uniform_color([1.,.15,.1])
        vis.add_geometry(m)
    opt = vis.get_render_option()
    opt.background_color = np.array([.035,.04,.05])
    opt.point_size = 4.0
    if len(pts):
        ctrl = vis.get_view_control()
        ctrl.set_lookat(np.mean(pts, axis=0))
        ctrl.set_front([.65,-.7,-.33])
        ctrl.set_up([0,0,1])
        ctrl.set_zoom(.72)
    return vis



def inspect_query(pixel_x, pixel_y, frozen, cfg, estimator, *,
                  forced_query_index=None, focus_scene_index=None):

    clicked = pixel_to_point_mm(pixel_x, pixel_y, frozen["depth_m"], frozen["intr"])
    # Clicking a detected marker uses its original FPS seed, even when the RGB
    # pixel itself has no valid depth. This reproduces the same Hough Top-4.
    if clicked is None and forced_query_index is None:
        print(f"[query] Invalid depth at ({pixel_x}, {pixel_y})")
        return

    # Downsample once per frozen frame; clicking only changes the query.

    if "points_voxel_mm" not in frozen:

        frozen["points_voxel_mm"] = voxel_downsample_3d(frozen["points_raw_mm"], cfg.voxel_3d_mm)

    points = frozen["points_voxel_mm"]

    if len(points) < cfg.min_neighbors:

        print("[query] Too few downsampled 3-D points.")

        return

    scene_cache = frozen["scene_cache"]
    if forced_query_index is not None:
        qi = int(forced_query_index)
        if not 0 <= qi < len(points):
            raise IndexError(f"Invalid FPS seed index: {qi}")
        click_distance = 0.0 if clicked is None else float(np.linalg.norm(points[qi]-clicked))
        print(f"[FPS inspect] seed #{qi} | focus scene index={focus_scene_index}")
    elif scene_cache is not None:
        valid = np.flatnonzero(scene_cache.normal_valid)
        if len(valid) == 0:
            print("[query] No valid SAH points in this capture")
            return
        click_distance, local_id = cKDTree(points[valid]).query(clicked)
        qi = int(valid[int(local_id)])
    else:
        click_distance, qi = cKDTree(points).query(clicked)

    if cfg.enable_iteration:
        print("[query] ITER ON: query -> tangent -> tensor target -> move -> repeat")
    else:
        print("[query] ITER OFF: single tangent/Hough/3D extremality pass, query stays fixed")

    try:

        ctx, result, localization = localize_corner_iterative(points, int(qi), cfg, estimator, scene_cache)

    except Exception as exc:

        print(f"[query] {type(exc).__name__}: {exc}")

        return

    frozen["selected_pixel"] = (pixel_x, pixel_y)

    frozen["context"], frozen["boundary"] = ctx, result
    frozen["corner_localization"] = localization
    # FPS-compatible: Hough+EXT, not the old query-tensor gate, is decisive.
    localization["tensor_verification"] = dict(localization["verification"])
    accepted_peaks = [p for p in result.get("extremality_3d", []) if p["pass"]]
    focus_entry = next((p for p in result.get("extremality_3d", [])
                        if focus_scene_index is not None and
                        int(p["scene_index"]) == int(focus_scene_index)), None)
    # On a detected marker, inspect THAT peak, not whichever of the Top-4
    # happens to be accepted first. The separate overall verdict remains global.
    best = (focus_entry if focus_entry is not None and focus_entry["pass"]
            else (accepted_peaks[0] if accepted_peaks else None))
    if focus_scene_index is not None and focus_entry is None:
        print("[FPS inspect] WARNING: stored peak not found in recomputed seed")
    localization["verification"] = {
        "accepted": bool(accepted_peaks),
        "reason": "hough_and_soft_3d_extremality" if accepted_peaks else "no_extremality_passed_hough_peak",
        "lambda_min": float(best["hough_score"]) if best else np.nan,
        "lambda_min_threshold": float("nan"),
        "balance": np.nan, "balance_threshold": float("nan"),
    }
    # The Hough peak itself (an observed 3D voxel), not the clicked query,
    # is the FPS detector's unrefined corner candidate.
    refinement = None
    if best is not None:
        coarse_position = points[int(best["scene_index"])].copy()
    else:
        coarse_position = np.asarray(ctx["query_world"], dtype=float).copy()
    frozen["corner_position_3d"] = coarse_position
    frozen["corner_refinement_3d"] = None
    if cfg.enable_refine and best is not None:
        try:
            # FPS implementation refines at Hough center, not at original query.
            peak_ctx = build_tangent_context(points, int(best["scene_index"]),
                                             cfg, estimator, scene_cache)
            peak_boundary = estimate_context_boundary(peak_ctx)
            refinement = refine_corner_3d_subpix(
                points, peak_ctx, peak_boundary, cfg, estimator, scene_cache)
            if refinement["success"]:
                frozen["corner_position_3d"] = np.asarray(refinement["position"], dtype=float).copy()
        except Exception as exc:
            print(f"  [refine3d] failed: {type(exc).__name__}: {exc}; keeping coarse Hough point")
    frozen["corner_refinement_3d"] = refinement
    if refinement is not None:
        print(f"  [refine3d] {'OK' if refinement['success'] else 'FALLBACK'} "
              f"reason={refinement['reason']} "
              f"shift={refinement['shift_mm']:.3f} mm "
              f"rmse={refinement['rmse_mm']:.3f} mm")
    print(f"  [FPS detector verdict: {cfg.ext_method}] {'PASS' if best is not None else 'FAIL'} "
          f"| accepted Hough peaks={len(accepted_peaks)} "
          f"| final candidate XYZ={np.round(frozen['corner_position_3d'],3)}")

    # Start at frame 1 (the only frame when iteration is disabled).
    # The final solution and verification still come from the last frame.
    frozen["extremality_rank"] = int(focus_entry["rank"]) if focus_entry else 1
    frozen.pop("pending_ext_rank", None)
    frozen["iteration_image_cache"] = {}
    frozen["iteration_eig_vmax"] = iteration_heatmap_limit(localization)
    frozen["iteration_vote_vmax"] = iteration_vote_heatmap_limit(localization)
    frozen["iteration_support_vmax"] = iteration_vote_heatmap_limit(localization, "support_max")
    render_iteration_view(frozen, 0)

    ext_list = result.get("extremality_3d", [])
    chosen = result.get("hough_verified_peak")
    print("  [Hough + 3D EXT] verified Hough choice: " +
          (f"peak #{chosen['rank']} (E={chosen['score']:.3f})" if chosen else "NONE"))
    for record in ext_list:
        vtext = f"E={record['score']:.3f}" if record["valid"] else "E=N/A"
        length_pct = record["length_sensitivity"]
        print(f"  [3D {cfg.ext_method.upper()} Hough #{record['rank']}] {vtext} "
              f"| length={record['length_score']:.3f} "
              f"| old_max={record['max_score']:.3f} "
              f"| rank={record['rank_score']:.3f} "
              f"| LENGTH top10/20/30={length_pct[0.10]:.3f}/{length_pct[0.20]:.3f}/{length_pct[0.30]:.3f} "
              f"| length-dir={100*record['top_direction_coverage']:.1f}% "
              f"| hard={record['hard_score']:.3f} | "
              f"{'PASS' if record['pass'] else 'FAIL'} "
              f"(thr={record['threshold']:.3f}) | "
              f"neighbors={record['neighbor_count']} | "
              f"XYZ={np.round(record['center'], 2)}")

    qi_uv = ctx["query_uv_index"]

    d = result["distance_mm"][qi_uv]

    print(f"  raw / 3D voxel: {len(frozen['points_raw_mm']):,} / {len(points):,}")

    print(f"  clicked -> initial query: {click_distance:.3f} mm; final SAH normal: {np.round(ctx['normal'], 6)}")
    print(f"  localizer: {localization['status']} after {localization['iterations']} iteration(s)")
    ver = localization["verification"]
    verdict = "ACCEPT CORNER" if ver["accepted"] else "REJECT"
    print("  [tensor-only diagnostic] "
          f"accepted={localization['tensor_verification']['accepted']} "
          f"reason={localization['tensor_verification']['reason']}")
    print(f"  [Hough+EXT final gate] {verdict} | "
          f"accepted_peaks={len(accepted_peaks)} | reason={ver['reason']}")
    for h in localization["history"]:
        if "corner_target_uv" in h:
            print(
                f"    iter {h['iteration']:02d}: offset={h['target_offset_mm']:.3f} mm, "
                f"peak_lambda_min={h['corner_peak_score']:.5f}, ratio={h['corner_peak_ratio']:.3f}, "
                f"move={h.get('move_mm', 0.0):.3f} mm -> {h['status']}"
            )
        else:
            print(f"    iter {h['iteration']:02d}: {h['status']}")

    print(f"  projected / selected / UV cells: {len(ctx['support_uv_raw'])} / {np.sum(ctx['projection_keep'])} / {len(ctx['support_uv'])}")

    print(f"  query normal support: {len(ctx['normal_fit_weights']['indices'])}; backend: {ctx['normal_backend']}")

    print(f"  query density count(r/2): {result['density_count'][qi_uv]}; neighbors(r): {result['neighbor_count'][qi_uv]}")

    print(f"  query d_hat: {d:.4f} mm; boundary-strip result: {bool(result['boundary_mask'][qi_uv])}")
    corner = result["corner_eigen"]
    l1_q = corner["lambda_max"][qi_uv]
    l2_q = corner["lambda_min"][qi_uv]
    print(f"  query eigen lambda_max/min: {l1_q:.6f} / {l2_q:.6f}; "
          f"tensor r={corner['radius_mm']:g} mm, spatial weight OFF, sigma_d={corner['sigma_distance_mm']:g} mm")

    print(f"  boundary points: {np.sum(result['boundary_mask'])}; display/support radii: {cfg.query_radius_mm:g}/{cfg.support_radius_mm:g} mm")

    v = result["hough_vote"]
    vote_peaks = ", ".join(
        f"{np.round(peak['uv'], 1).tolist()}:{peak['value']:.4g}"
        for peak in v["peaks"][:4]) or "none"
    print(f"  [Hough DEBUG] voters={len(v['source_indices'])} "
          f"candidates={len(v['candidate_indices'])} "
          f"lambda_min_max={v['vote_max']:.5g}; "
          f"abs_gate={cfg.hough_min_score:g}; "
          f"eligible={v['eligible_candidate_count']}; peaks UV={vote_peaks}")
    for rank, pk in enumerate(v["peaks"], 1):
        ix = int(pk["index"])
        ss, dd, hh = (float(v[key][ix]) for key in ("support", "diversity", "response"))
        print(f"    Hough #{rank}: Support={ss:.6f} | Diversity={dd:.6f} "
              f"| lambda_min={hh:.6f} | S*D/2={ss*dd/2:.6f}")

    if result["query_boundary_uv"] is not None:

        b = result["query_boundary_uv"]

        b3 = ctx["query_world"] + b[0] * ctx["u"] + b[1] * ctx["v"]

        print(f"  local boundary estimate UV: {np.round(b, 4)}; tangent-lifted XYZ: {np.round(b3, 4)}")

    else:

        print("  No local boundary location displayed (direction undefined or d_hat >= r/2).")







def close_query_views(frozen):
    scale_view = frozen.pop("ext_scale_view", None)
    if scale_view is not None:
        scale_view["vis"].destroy_window()
    scale_window = frozen.pop("ext_scale_window", None)
    if scale_window is not None:
        try:
            scale_window.destroy()
        except (RuntimeError, tk.TclError):
            pass

    vis = frozen.pop("tangent_view", None)
    if vis is not None:
        vis.destroy_window()
    ext_vis = frozen.pop("extremality_view", None)
    if ext_vis is not None:
        ext_vis.destroy_window()

    frozen.pop("boundary_image", None)
    frozen.pop("iteration_image_cache", None)
    frozen.pop("iteration_eig_vmax", None)
    frozen.pop("iteration_vote_vmax", None)
    frozen.pop("iteration_support_vmax", None)
    frozen.pop("corner_localization", None)
    frozen.pop("pending_iteration", None)
    frozen.pop("pending_query", None)
    frozen.pop("pending_ext_rank", None)
    frozen.pop("extremality_rank", None)
    frozen.pop("extremality_controls", None)





def run_frozen_click_stage(frozen, cfg, estimator):
    if tk is None:
        raise RuntimeError("Tk is unavailable. In this conda environment, run: conda install tk pillow")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Cannot open the Tk display. Run in a desktop session with Tk installed "
            "(conda install tk pillow). Original error: " + str(exc)
        ) from exc

    action = None
    boundary_window = None
    boundary_view = None
    prev_button = None
    next_button = None
    peak_prev_button = None
    peak_next_button = None
    peak_buttons = []
    iteration_text = tk.StringVar(master=root, value="")
    peak_text = tk.StringVar(master=root, value="No Hough candidates")

    root.title(f"{WINDOW_FROZEN} | {BUILD}")
    iter_instructions = "Left/Right or J/L: iteration | " if cfg.enable_iteration else ""
    instructions = ("G: run/refresh global FPS | click red C#: inspect detected peak | "
                    "click elsewhere: query | " + iter_instructions +
                    "A/D: Hough peak | 1-4: select peak | S: 3D Scale-Space | F: FPS seeds "
                    "| T: pre-NMS | V: 3D overview | C: Kinect RGB/Depth | R: reset | Q/Esc: quit")
    status = tk.StringVar(master=root, value=instructions)
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x")
    controls = tk.Frame(root)
    controls.pack(fill="x")
    tk.Button(controls, text="Run / Refresh FPS [G]",
              command=lambda: frozen.__setitem__("pending_detect", True)).pack(side="left", padx=4)
    tk.Button(controls, text="Show FPS seeds [F]",
              command=lambda: frozen.__setitem__("pending_toggle_seeds", True)).pack(side="left", padx=4)
    tk.Button(controls, text="Show pre-NMS [T]",
              command=lambda: frozen.__setitem__("pending_toggle_accepted", True)).pack(side="left", padx=4)
    tk.Button(controls, text="3D overview [V]",
              command=lambda: frozen.__setitem__("pending_toggle_scene", True)).pack(side="left", padx=4)
    if frozen.get("kinect_depth_preview") is not None:
        tk.Button(controls, text="Depth / raw RGB [C]",
                  command=lambda: frozen.__setitem__("pending_toggle_kinect_view", True)).pack(side="left", padx=4)
    frozen["ext_scale_sigmas_x"] = tuple(cfg.ext_scale_sigmas_x)
    frozen["ext_scale_mode"] = cfg.ext_scale_mode
    frozen["ext_scale_directions"] = cfg.ext_scale_directions
    frozen["ext_scale_output"] = cfg.ext_scale_output
    frame_view = TkImagePanel(root, draw_frozen(frozen))
    set_image_window_size(root, frozen["rgb"], extra_height=75)

    def request_action(value):
        nonlocal action
        action = value

    def navigate(delta):
        loc = frozen.get("corner_localization")
        if not loc or not loc.get("iteration_frames"):
            return
        # Pending navigation is applied outside Tcl callbacks in the GUI loop.
        current = frozen.get("pending_iteration", frozen.get("iteration_index", 0))
        target = int(np.clip(current + delta, 0, len(loc["iteration_frames"]) - 1))
        if target != current:
            frozen["pending_iteration"] = target

    def select_peak(rank):
        loc = frozen.get("corner_localization")
        if loc is None:
            return
        frame_idx = int(frozen.get("pending_iteration", frozen.get("iteration_index", 0)))
        frame = loc["iteration_frames"][frame_idx]
        count = len(frame["result"].get("extremality_3d", []))
        if 1 <= rank <= count:
            frozen["pending_ext_rank"] = int(rank)

    def step_peak(delta):
        current = int(frozen.get("pending_ext_rank", frozen.get("extremality_rank", 1)))
        select_peak(current + delta)

    frozen["extremality_controls"] = {
        "select_peak": select_peak,
        "step_peak": step_peak,
        "step_iteration": navigate,
    }

    def key_cb(event):
        key = event.keysym.lower()
        if key in ("q", "escape", "r"):
            request_action("recapture" if key == "r" else "quit")
        elif key == "g":
            frozen["pending_detect"] = True
        elif key == "f":
            frozen["pending_toggle_seeds"] = True
        elif key == "t":
            frozen["pending_toggle_accepted"] = True
        elif key == "v":
            frozen["pending_toggle_scene"] = True
        elif key == "s":
            frozen["pending_ext_scale"] = True
        elif key == "c" and frozen.get("kinect_depth_preview") is not None:
            frozen["pending_toggle_kinect_view"] = True
        elif key in ("1", "2", "3", "4"):
            select_peak(int(key))
        elif key == "a":
            step_peak(-1)
        elif key == "d":
            step_peak(+1)
        elif key in ("j", "bracketleft"):
            navigate(-1)
        elif key in ("l", "bracketright"):
            navigate(+1)
        elif key == "left":
            navigate(-1)
        elif key == "right":
            navigate(+1)
        elif key == "home":
            loc = frozen.get("corner_localization")
            if loc:
                frozen["pending_iteration"] = 0
        elif key == "end":
            loc = frozen.get("corner_localization")
            if loc:
                frozen["pending_iteration"] = len(loc["iteration_frames"]) - 1

    def mouse_cb(event):
        point = frame_view.image_pixel(event.x, event.y)
        if point is not None:
            hit = pick_scene_detection(point, frozen)
            if hit is None:
                frozen["pending_query"] = (point, None)
            else:
                frozen["pending_query"] = (point, hit)
                candidate = hit["candidate"]
                print(f"[RGB click] {hit['kind']} {hit['distance_px']:.1f}px "
                      f"-> original FPS seed {candidate['seed_index']}")

    def close_boundary():
        nonlocal boundary_window, boundary_view, prev_button, next_button
        nonlocal peak_prev_button, peak_next_button, peak_buttons
        if boundary_window is not None:
            boundary_window.destroy()
        boundary_window = None
        boundary_view = None
        prev_button = None
        next_button = None
        peak_prev_button = None
        peak_next_button = None
        peak_buttons = []

    def refresh_diagnostics_window():
        nonlocal boundary_window, boundary_view, prev_button, next_button
        nonlocal peak_prev_button, peak_next_button, peak_buttons
        rgb = frozen.get("boundary_image")
        loc = frozen.get("corner_localization")
        if rgb is None or not loc:
            return
        idx = int(frozen.get("iteration_index", 0))
        frames = loc["iteration_frames"]
        record = frames[idx]["record"]
        normal = np.asarray(record["normal"])
        xyz = np.asarray(record["query_world"])
        iteration_text.set(
            f"Iteration {idx+1}/{len(frames)} | status: {record['status']} | "
            f"query XYZ [mm]: {np.round(xyz, 2)} | normal: {np.round(normal, 3)}"
        )
        entries = frames[idx]["result"].get("extremality_3d", [])
        rank = int(frozen.get("extremality_rank", 1))
        if entries:
            entry = entries[rank-1]
            verdict = "PASS" if entry["pass"] else ("FAIL" if entry["valid"] else "INSUFFICIENT")
            dbg = hough_peak_voter_diagnostics(
                frames[idx]["ctx"], frames[idx]["result"], rank)
            vote = frames[idx]["result"]["hough_vote"]
            peak_id = int(vote["peaks"][rank-1]["index"])
            s_peak = float(vote["support"][peak_id])
            d_peak = float(vote["diversity"][peak_id])
            h_peak = float(vote["response"][peak_id])
            vote_suffix = (
                f" | line>{dbg['reference_radius_mm']:g}mm: "
                f"votes={100*dbg['far_vote_fraction']:.0f}% "
                f"lambda_min={100*dbg['far_minor_fraction']:.0f}%"
                if dbg is not None else "")
            best1_suffix = (f" | Best1 Qmin={entry['best1']['score']:.3f} "
                            f"(threshold={cfg.ext_scale_best1_threshold:g})"
                            if 'best1' in entry else "")
            peak_text.set(f"Hough {rank}/{len(entries)} | {entry['method']}="
                           f"{entry['score']:.3f} ({verdict}) | "
                           f"Length={entry['length_score']:.3f} / Max={entry['max_score']:.3f} "
                           f"/ Rank={entry['rank_score']:.3f} "
                           f"| length-dir={100*entry['top_direction_coverage']:.1f}% "
                           f"| 3D neighbors={entry['neighbor_count']}"
                           f" | Hough S={s_peak:.4f} D={d_peak:.3f} H={h_peak:.5f}"
                           + vote_suffix + best1_suffix)
            print(f"[HOUGH S/D/H #{rank}] support={s_peak:.8f} "
                  f"diversity={d_peak:.8f} lambda_min={h_peak:.8f} "
                  f"reconstructed={0.5*s_peak*d_peak:.8f}")
            if dbg is not None:
                print(f"[HOUGH VOTERS #{rank}] strong(>=20% max)={dbg['strong_count']}/"
                      f"{len(dbg['voter_ids'])} | along line > "
                      f"{dbg['reference_radius_mm']:g}mm: "
                      f"vote-weight={100*dbg['far_vote_fraction']:.1f}% "
                      f"lambda_min-weight={100*dbg['far_minor_fraction']:.1f}% "
                      f"| mean along(weighted)={dbg['weighted_mean_along_mm']:.1f}mm "
                      f"| support={dbg['support_reconstructed']:.6f} "
                      f"lambda_min={dbg['lambda_min_reconstructed']:.6f}")
        else:
            peak_text.set(f"Iteration {idx+1}: no Hough peaks")
        if boundary_window is None:
            boundary_window = tk.Toplevel(root)
            boundary_window.title(WINDOW_BOUNDARY + " | per-iteration")
            boundary_window.protocol("WM_DELETE_WINDOW", close_boundary)

            toolbar = tk.Frame(boundary_window)
            toolbar.pack(side="top", fill="x")
            prev_button = tk.Button(toolbar, text="◀ Previous", command=lambda: navigate(-1))
            prev_button.pack(side="left", padx=5, pady=4)
            next_button = tk.Button(toolbar, text="Next ▶", command=lambda: navigate(+1))
            next_button.pack(side="left", padx=5, pady=4)
            tk.Label(toolbar, textvariable=iteration_text, anchor="w").pack(
                side="left", padx=12
            )
            peak_toolbar = tk.Frame(boundary_window)
            peak_toolbar.pack(side="top", fill="x", pady=3)
            peak_prev_button = tk.Button(peak_toolbar, text="◀ Hough Peak", command=lambda: step_peak(-1))
            peak_prev_button.pack(side="left", padx=4)
            peak_next_button = tk.Button(peak_toolbar, text="Hough Peak ▶", command=lambda: step_peak(+1))
            peak_next_button.pack(side="left", padx=4)
            for k in range(1, HOUGH_TOP_K+1):
                button = tk.Button(peak_toolbar, text=f"#{k}", width=3, command=lambda k=k: select_peak(k))
                button.pack(side="left", padx=2)
                peak_buttons.append(button)
            tk.Button(peak_toolbar, text="3D Scale-Space [S]",
                      command=lambda: frozen.__setitem__("pending_ext_scale", True)).pack(side="left", padx=6)
            tk.Label(peak_toolbar, textvariable=peak_text, anchor="w").pack(
                side="left", padx=12
            )
            boundary_view = TkImagePanel(boundary_window, rgb)
            set_image_window_size(boundary_window, rgb, extra_height=45)
        else:
            boundary_view.set_rgb(rgb)

        prev_button.configure(state="normal" if idx > 0 else "disabled")
        next_button.configure(state="normal" if idx + 1 < len(frames) else "disabled")
        peak_count = len(entries)
        peak_prev_button.configure(state="normal" if peak_count and rank > 1 else "disabled")
        peak_next_button.configure(state="normal" if rank < peak_count else "disabled")
        for k, button in enumerate(peak_buttons, 1):
            button.configure(state="normal" if k <= peak_count else "disabled",
                             relief="sunken" if k == rank and k <= peak_count else "raised")
        boundary_window.deiconify()
        boundary_window.lift()

    root.protocol("WM_DELETE_WINDOW", lambda: request_action("quit"))
    root.bind_all("<KeyPress>", key_cb)
    frame_view.canvas.bind("<Button-1>", mouse_cb)
    print(f"Frozen frame ({BUILD}): {instructions}", flush=True)
    try:
        while action is None:
            root.update_idletasks()
            root.update()
            if action is not None:
                break

            if frozen.pop("pending_detect", False):
                status.set(f"Global FPS scanning {frozen['fps_count']} seeds... terminal shows progress")
                root.update_idletasks()
                try:
                    detection = detect_corners_from_fps_scene(
                        frozen, cfg, estimator, frozen["fps_count"], frozen["nms_radius_mm"],
                        progress_callback=lambda done,total,accepted: status.set(
                            f"FPS {done}/{total} | accepted peaks {accepted}"))
                    # Publish only completed scans. Do not keep stale click diagnostics.
                    close_query_views(frozen)
                    close_boundary()
                    # close_query_views clears callbacks; restore those belonging
                    # to THIS Tk instance for Open3D A/D/1-4 navigation.
                    frozen["extremality_controls"] = {
                        "select_peak": select_peak,
                        "step_peak": step_peak,
                        "step_iteration": navigate,
                    }
                    frozen["global_detection"] = detection
                    frozen["selected_pixel"] = None
                    if frozen["save_global_results"]:
                        try:
                            path = save_fps_detection_summary(frozen, cfg, detection)
                            print(f"[FPS saved] {path}", flush=True)
                        except OSError as exc:
                            print(f"[FPS save] ERROR: {exc}; results still shown in GUI", flush=True)
                    if frozen["enable_scene_3d"]:
                        try:
                            frozen["global_view"] = open_global_fps_view(
                                frozen, cfg, frozen.get("global_view"))
                        except Exception as exc:
                            print(f"[FPS 3D] {type(exc).__name__}: {exc}")
                            frozen["global_view"] = None
                    frame_view.set_rgb(draw_frozen(frozen))
                    status.set(f"FPS {len(detection['seed_indices'])} | passing "
                               f"{len(detection['accepted_before_nms'])} | "
                               f"NMS corners {len(detection['corners'])} | click red C#")
                except Exception as exc:
                    print(f"[FPS scan] {type(exc).__name__}: {exc}", flush=True)
                    status.set(f"FPS ERROR: {type(exc).__name__}: {exc}")
                root.update_idletasks()

            if frozen.pop("pending_toggle_seeds", False):
                frozen["show_fps_seeds"] = not frozen.get("show_fps_seeds", False)
                frame_view.set_rgb(draw_frozen(frozen))
            if frozen.pop("pending_toggle_accepted", False):
                frozen["show_accepted_candidates"] = not frozen.get("show_accepted_candidates", False)
                frame_view.set_rgb(draw_frozen(frozen))
            if frozen.pop("pending_toggle_scene", False):
                if frozen.get("global_view") is not None:
                    frozen["global_view"].destroy_window()
                    frozen["global_view"] = None
                elif frozen.get("global_detection") is not None:
                    try:
                        frozen["global_view"] = open_global_fps_view(frozen, cfg)
                    except Exception as exc:
                        print(f"[FPS 3D] {type(exc).__name__}: {exc}")

            if frozen.pop("pending_toggle_kinect_view", False):
                old = frozen.get("kinect_view", "depth")
                frozen["kinect_view"] = "rgb" if old == "depth" else "depth"
                frozen["rgb"] = (frozen["kinect_original_rgb"] if frozen["kinect_view"] == "rgb"
                                 else frozen["kinect_depth_preview"])
                print(f"[KINECT UI] {frozen['kinect_view']} | "
                      "RGB is unregistered; marker coordinates always use the depth pixel grid")
                frame_view.set_rgb(draw_frozen(frozen))

            pending = frozen.pop("pending_query", None)
            if pending is not None:
                (pix_x, pix_y), hit = pending
                status.set("Inspecting FPS seed or arbitrary click: computing 9 diagnostics...")
                root.update_idletasks()
                previous_image = frozen.get("boundary_image")
                forced_seed = hit["candidate"]["seed_index"] if hit else None
                focus_id = hit["candidate"].get("scene_index") if hit else None
                inspect_query(pix_x, pix_y, frozen, cfg, estimator,
                              forced_query_index=forced_seed, focus_scene_index=focus_id)
                frame_view.set_rgb(draw_frozen(frozen))
                result_image = frozen.get("boundary_image")
                if result_image is not None and result_image is not previous_image:
                    refresh_diagnostics_window()
                status.set(instructions)

            pending_iter = frozen.pop("pending_iteration", None)
            if pending_iter is not None and frozen.get("corner_localization"):
                try:
                    render_iteration_view(frozen, pending_iter)
                    refresh_diagnostics_window()
                except Exception as exc:
                    print(f"[iteration view] {type(exc).__name__}: {exc}")
                    status.set(f"Iteration display error: {exc}")

            selected_rank = frozen.pop("pending_ext_rank", None)
            if selected_rank is not None and frozen.get("corner_localization"):
                loc = frozen["corner_localization"]
                idx = frozen.get("iteration_index", 0)
                frame = loc["iteration_frames"][idx]
                count = len(frame["result"].get("extremality_3d", []))
                if 1 <= selected_rank <= count:
                    frozen["extremality_rank"] = selected_rank
                    # Regenerate only the selected (iteration, peak) 2D panel;
                    # do not reopen/recompute the original 3D tangent viewer.
                    view_loc = iteration_localization_view(loc, idx)
                    cache = frozen.setdefault("iteration_image_cache", {})
                    cache_key = (idx, selected_rank)
                    if cache_key not in cache:
                        fig = make_boundary_figure(
                            frame["ctx"], frame["result"], view_loc,
                            eig_vmax=frozen["iteration_eig_vmax"],
                            vote_vmax=frozen["iteration_vote_vmax"],
                            support_vmax=frozen["iteration_support_vmax"],
                            selected_rank=selected_rank,
                        )
                        try:
                            cache[cache_key] = figure_to_rgb(fig)
                        finally:
                            fig.clear()
                    frozen["boundary_image"] = cache[cache_key]
                    if cfg.show_3d:
                        frozen["extremality_view"] = open_extremality_3d_view(
                            frame["ctx"], frame["result"], frozen.get("extremality_view"),
                            selected_rank, iteration_number=idx+1, total_iterations=len(loc["iteration_frames"]),
                            navigation=frozen["extremality_controls"])
                    refresh_diagnostics_window()

            if frozen.pop("pending_ext_scale", False):
                loc = frozen.get("corner_localization")
                if loc is None:
                    status.set("Select a Hough corner first, then press S")
                else:
                    idx = int(frozen.get("iteration_index", 0))
                    frame = loc["iteration_frames"][idx]
                    entries = frame["result"].get("extremality_3d", [])
                    rank = int(frozen.get("extremality_rank", 1))
                    if rank > len(entries) or rank < 1:
                        status.set("No Hough peak at this selection")
                    else:
                        try:
                            selected_entry = entries[rank-1]
                            if frozen.get("ext_scale_view") is not None:
                                frozen["ext_scale_view"]["vis"].destroy_window()
                                frozen["ext_scale_view"] = None
                            def _progress(message):
                                status.set(message)
                                root.update_idletasks()
                                print("[EXT scale] " + message, flush=True)
                            scan = compute_extremality_scale_space(
                                frozen["points_voxel_mm"], int(selected_entry["scene_index"]),
                                cfg, sigma_factors=frozen["ext_scale_sigmas_x"],
                                smooth_mode=frozen["ext_scale_mode"],
                                radius_gain=cfg.ext_scale_radius_gain,
                                directions_count=frozen["ext_scale_directions"],
                                progress=_progress)
                            png, npz, meta = save_extremality_scale_diagnostic(
                                scan, rank, frozen["ext_scale_output"])
                            print(f"[EXT scale] saved: {png} | {npz} | {meta}")
                            original_difference = abs(float(selected_entry["score"])-float(scan["reference_original_score"]))
                            print(f"[EXT scale] sigma=0 agreement with existing verifier: "
                                  f"absolute difference={original_difference:.3e}")
                            if frozen.get("ext_scale_window") is not None:
                                frozen["ext_scale_window"].destroy()
                            win = tk.Toplevel(root)
                            win.title("3D Extremality Scale-Space: multi-view + response")
                            preview = np.asarray(Image.open(png).convert("RGB"))
                            panel = TkImagePanel(win, preview)
                            win._ext_scale_panel = panel  # keep Tk image alive
                            set_image_window_size(win, preview, extra_height=25)
                            frozen["ext_scale_window"] = win
                            frozen["ext_scale_view"] = open_extremality_scale_view(scan)
                            status.set(f"3D EXT Scale-Space saved: {png} | [ ] keys in Open3D window")
                        except Exception as exc:
                            print(f"[EXT scale] ERROR: {type(exc).__name__}: {exc}", flush=True)
                            status.set(f"3D Scale-Space error: {exc}")

            for view_name in ("tangent_view", "extremality_view", "global_view"):
                vis = frozen.get(view_name)
                if vis is not None:
                    if vis.poll_events():
                        vis.update_renderer()
                    else:
                        vis.destroy_window()
                        frozen[view_name] = None
            scale_view = frozen.get("ext_scale_view")
            if scale_view is not None:
                try:
                    if scale_view["vis"].poll_events():
                        scale_view["vis"].update_renderer()
                    else:
                        scale_view["vis"].destroy_window()
                        frozen["ext_scale_view"] = None
                except Exception:
                    frozen["ext_scale_view"] = None
            time.sleep(.01)
    finally:
        close_query_views(frozen)
        global_view = frozen.pop("global_view", None)
        if global_view is not None:
            global_view.destroy_window()
        root.destroy()
    return action or "quit"


class TkImagePanel:

    """Resizable RGB canvas with click coordinates mapped to source pixels."""



    def __init__(self, parent, rgb):

        self.canvas = tk.Canvas(parent, highlightthickness=0, background="#20242a")

        self.canvas.pack(fill="both", expand=True)

        self.item = self.canvas.create_image(0, 0, anchor="nw")

        self.photo = None

        self.layout = None

        self.source = Image.fromarray(np.asarray(rgb, dtype=np.uint8))

        self.canvas.bind("<Configure>", self.redraw)



    def set_rgb(self, rgb):

        self.source = Image.fromarray(np.asarray(rgb, dtype=np.uint8))

        self.redraw()



    def redraw(self, event=None):

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()

        if min(cw, ch) <= 1:

            return

        iw, ih = self.source.size

        scale = min(cw / iw, ch / ih)

        dw, dh = max(1, round(iw * scale)), max(1, round(ih * scale))

        left, top = (cw - dw) // 2, (ch - dh) // 2

        resized = self.source.resize((dw, dh), Image.Resampling.BILINEAR)

        self.photo = ImageTk.PhotoImage(resized, master=self.canvas)

        self.canvas.coords(self.item, left, top)

        self.canvas.itemconfigure(self.item, image=self.photo)

        self.layout = (left, top, dw, dh)



    def image_pixel(self, x, y):

        if self.layout is None:

            return None

        left, top, dw, dh = self.layout

        if not (left <= x < left + dw and top <= y < top + dh):

            return None

        iw, ih = self.source.size

        return (min(iw - 1, int((x - left) * iw / dw)),

                min(ih - 1, int((y - top) * ih / dh)))





def set_image_window_size(window, rgb, extra_height=0):

    height, width = rgb.shape[:2]

    scale = min(1., .9 * window.winfo_screenwidth() / width,

                max(1, .8 * window.winfo_screenheight() - extra_height) / height)

    window.geometry(f"{max(1, round(width * scale))}x{max(1, round(height * scale)) + extra_height}")





def parse_args(argv=None):
    p = argparse.ArgumentParser(description=(
        "Offline FPS + click inspection for Kinect 7-Scenes or saved RealSense RGB-D: "
        "Hough Top-4 + soft 3D extremality 1024 with cached global SAH"))
    p.add_argument("--capture", "--input", "-i", required=True,
                   help="RealSense capture folder OR Kinect 7-Scenes seq-XX.zip / seq-XX folder / scene folder")
    p.add_argument("--frame-index", "--kinect-frame", type=int, default=0,
                   help="Kinect frame NUMBER, e.g. 0 = frame-000000 (default 0)")
    p.add_argument("--kinect-seq", type=str, default=None,
                   help="When --capture is a 7-Scenes scene folder, choose sequence (e.g. seq-01)")
    p.add_argument("--kinect-view", choices=("depth", "rgb"), default="depth",
                   help="Kinect UI: depth-aligned geometry view (default) or raw UNREGISTERED RGB")
    p.add_argument("--kinect-fx", type=float, default=585.0,
                   help="Approximate Kinect DEPTH camera focal length x for 640x480")
    p.add_argument("--kinect-fy", type=float, default=585.0,
                   help="Approximate Kinect DEPTH camera focal length y for 640x480")
    p.add_argument("--kinect-cx", type=float, default=320.0,
                   help="Approximate Kinect DEPTH principal point x for 640x480")
    p.add_argument("--kinect-cy", type=float, default=240.0,
                   help="Approximate Kinect DEPTH principal point y for 640x480")
    p.add_argument("--normal-module", help="Path to surface_aware_harris3d.py")
    p.add_argument("--version", action="version", version=BUILD)
    it = p.add_mutually_exclusive_group()
    it.add_argument("--iteration", "--iter", dest="enable_iteration", action="store_true")
    it.add_argument("--no-iteration", dest="enable_iteration", action="store_false")
    p.set_defaults(enable_iteration=False)
    ref = p.add_mutually_exclusive_group()
    ref.add_argument("--refine", dest="enable_refine", action="store_true")
    ref.add_argument("--no-refine", dest="enable_refine", action="store_false")
    p.set_defaults(enable_refine=False)
    p.add_argument("--local-normal-recompute", action="store_true",
                   help="Debug comparison: recompute SAH per query. Default: full-cloud SAH only once.")
    p.add_argument("--voxel-3d-mm", "--voxel-mm", dest="voxel_3d_mm", type=float, default=2.0)
    p.add_argument("--voxel-2d-mm", "--uv-voxel-mm", dest="voxel_2d_mm", type=float, default=2.0)
    p.add_argument("--normal-radius-x", type=float, default=4.0)
    p.add_argument("--normal-sigma-r-x", type=float, default=0.5)
    p.add_argument("--query-radius-x", type=float, default=10.0)
    p.add_argument("--boundary-radius-x", type=float, default=4.0)
    p.add_argument("--boundary-epsilon-x", type=float, default=1.0)
    p.add_argument("--tensor-radius-x", type=float, default=4.0)
    p.add_argument("--tensor-sigma-d-x", type=float, default=1.5)
    p.add_argument("--vote-line-sigma-x", "--vote-sigma-x", type=float, default=1.0)
    p.add_argument("--hough-min-score", type=float, default=0.015,
                   help="Absolute Hough lambda_min minimum before Top-4 / 3D verifier "
                        "(default 0.015, experimental; use 0 to disable)")
    p.add_argument("--corner-search-radius-x", type=float, default=6.0)
    p.add_argument("--corner-step-max-x", type=float, default=4.0)
    p.add_argument("--corner-refine-radius-x", type=float, default=2.0)
    p.add_argument("--corner-converge-x", type=float, default=1.0)
    p.add_argument("--corner-peak-ratio", type=float, default=0.7)
    p.add_argument("--corner-max-iterations", type=int, default=10)
    p.add_argument("--corner-lambda-min-threshold", type=float, default=0.03)
    p.add_argument("--corner-balance-threshold", type=float, default=0.15)
    p.add_argument("--surface-weight-min", type=float, default=0.1)
    p.add_argument("--ext-radius-x", type=float, default=4.0)
    p.add_argument("--ext-epsilon-mm", type=float, default=0.5)
    p.add_argument("--ext-sigma-mm", type=float, default=0.5)
    p.add_argument("--ext-directions", type=int, default=1024)
    p.add_argument("--ext-min-score", type=float, default=0.08,
                   help="Experimental active-method score threshold [0,1], requires calibration")
    p.add_argument("--ext-method", choices=("projection-length", "soft-rank", "max"),
                   default="projection-length",
                   help="PASS/FAIL metric; all three methods are logged for comparison")
    p.add_argument("--ext-top-fraction", type=float, default=0.20,
                   help="Robust projection-length upper-end fraction (default 0.20)")
    p.add_argument("--ext-trim-fraction", type=float, default=0.05,
                   help="Trim fraction at BOTH projection ends (default 0.05: Q05..Q95)")
    p.add_argument("--ext-min-span-mm", type=float, default=1.0,
                   help="Min robust directional projection span; smaller spans score zero")
    p.add_argument("--ext-rank-tau-mm", type=float, default=2.0,
                   help="Old Soft Rank diagnostic comparison tolerance (mm)")
    p.add_argument("--ext-scale-sigmas-x", default="0,0.5,1,2",
                   help="Gaussian sigma / 3D voxel mm; strictly increasing, starting at 0")
    p.add_argument("--ext-scale-mode", choices=("full", "normal"), default="full",
                   help="Gaussian-weighted 3D point smoothing; not volumetric convolution")
    p.add_argument("--ext-scale-radius-gain", type=float, default=2.0,
                   help="Scale-dependent EXT radius r(sigma)=hypot(r0,gain*sigma); 0 keeps fixed radius")
    p.add_argument("--ext-scale-directions", type=int, default=0,
                   help="3D directions for Scale-Space test (0=use same as original verifier)")
    p.add_argument("--ext-scale-best1", action="store_true",
                   help="Enable trained candidate-only min-post-scale EXT verifier (replaces legacy 3-D gate)")
    p.add_argument("--ext-scale-best1-threshold", type=float, default=None,
                   help="Required with --ext-scale-best1; must be calibrated on sensor-matched labeled data")
    p.add_argument("--ext-scale-best1-sigmas-x", default="0,1,2,4",
                   help="Scale factors for Best-1; separate from 3D visualization scale list")
    p.add_argument("--ext-scale-best1-directions", type=int, default=128,
                   help="Verifier sphere rays, synthetic selection used 128 (even, >=16)")
    p.add_argument("--ext-scale-output", default="ext_scale_diagnostics",
                   help="Folder for 3D scale-space figures and numeric results")
    p.add_argument("--normal-sigma-e-deg", type=float, default=15.0)
    p.add_argument("--normal-kernel", choices=("cauchy", "gaussian"), default="cauchy")
    p.add_argument("--normal-max-iterations", type=int, default=5)
    p.add_argument("--normal-update-mode", choices=("fixed", "synchronous"), default="synchronous")
    p.add_argument("--min-neighbors", type=int, default=6)
    p.add_argument("--min-depth-m", type=float, default=None,
                   help="Override saved valid_depth_range_m minimum [m]")
    p.add_argument("--max-depth-m", type=float, default=None,
                   help="Override saved valid_depth_range_m maximum [m]")
    p.add_argument("--fps-count", type=int, default=64,
                   help="Number of full-scene FPS seeds; default 64")
    p.add_argument("--corner-nms-radius-x", type=float, default=3.0,
                   help="Global 3D NMS radius in 3D voxel units, default 3")
    auto = p.add_mutually_exclusive_group()
    auto.add_argument("--auto-detect", dest="auto_detect", action="store_true")
    auto.add_argument("--no-auto-detect", dest="auto_detect", action="store_false")
    p.set_defaults(auto_detect=True)
    saves = p.add_mutually_exclusive_group()
    saves.add_argument("--save-global-results", dest="save_global_results", action="store_true")
    saves.add_argument("--no-save-global-results", dest="save_global_results", action="store_false")
    p.set_defaults(save_global_results=True)
    p.add_argument("--no-3d-scene", action="store_true",
                   help="Disable separate full-scene FPS Open3D overview")
    p.add_argument("--no-3d-view", action="store_true")
    p.add_argument("--no-detection-overlay", action="store_true",
                   help="Hide markers from saved detections.json on the RGB image")
    return p.parse_args(argv)



def configure(args):
    global WIDTH, HEIGHT, MIN_DEPTH_M, MAX_DEPTH_M, VOXEL_3D_MM, VOXEL_2D_MM
    positive = [args.voxel_3d_mm, args.voxel_2d_mm, args.normal_radius_x,
                args.normal_sigma_r_x, args.query_radius_x, args.boundary_radius_x,
                args.boundary_epsilon_x, args.tensor_radius_x, args.tensor_sigma_d_x,
                args.vote_line_sigma_x, args.corner_search_radius_x,
                args.corner_step_max_x, args.corner_refine_radius_x,
                args.corner_converge_x, args.ext_radius_x, args.ext_sigma_mm,
                 args.ext_rank_tau_mm, args.ext_min_span_mm,
                args.normal_sigma_e_deg]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("All geometric scales and sigmas must be finite and positive")
    if not 0 <= args.surface_weight_min <= 1:
        raise ValueError("surface-weight-min must be in [0,1]")
    if not np.isfinite(args.hough_min_score) or not 0 <= args.hough_min_score <= 1:
        raise ValueError("hough-min-score must be finite and in [0,1]")
    if not 0 <= args.ext_min_score <= 1:
        raise ValueError("ext-min-score must be in [0,1]")
    if not 0 < args.ext_top_fraction < 0.5:
        raise ValueError("ext-top-fraction must be in (0,0.5)")
    if not np.isfinite(args.ext_trim_fraction) or not 0 <= args.ext_trim_fraction < 0.25:
        raise ValueError("ext-trim-fraction must be in [0,0.25)")
    if args.ext_directions < 16 or args.ext_directions % 2:
        raise ValueError("ext-directions must be even and >= 16")
    if args.min_neighbors < 3 or args.normal_max_iterations < 1 or args.corner_max_iterations < 1:
        raise ValueError("min-neighbors >=3, iterations >=1 required")
    if args.corner_search_radius_x + args.tensor_radius_x > args.query_radius_x + 1e-12:
        raise ValueError("corner-search-radius-x + tensor-radius-x must be <= query-radius-x")
    if 1.5 * args.boundary_epsilon_x >= .5 * args.boundary_radius_x:
        raise ValueError("Require boundary-radius-x > 3*boundary-epsilon-x")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("Require 0 < min-depth-m < max-depth-m")
    if not 0 < args.corner_peak_ratio <= 1:
        raise ValueError("corner-peak-ratio must be in (0,1]")
    if args.fps_count < 1 or not np.isfinite(args.corner_nms_radius_x) or args.corner_nms_radius_x <= 0:
        raise ValueError("Require fps-count >= 1 and corner-nms-radius-x > 0")
    _ext_scale_parse_sigmas(args.ext_scale_sigmas_x, args.voxel_3d_mm)
    if not np.isfinite(args.ext_scale_radius_gain) or args.ext_scale_radius_gain < 0:
        raise ValueError("--ext-scale-radius-gain must be finite and nonnegative")
    best1_levels = _ext_scale_parse_sigmas(args.ext_scale_best1_sigmas_x, args.voxel_3d_mm)
    if args.ext_scale_best1 and len(best1_levels) < 3:
        raise ValueError("Best-1 requires >=3 sigma levels")
    if args.ext_scale_best1:
        if args.ext_method != "projection-length":
            raise ValueError("--ext-scale-best1 currently needs --ext-method projection-length")
        if args.ext_scale_best1_threshold is None:
            raise ValueError("--ext-scale-best1 requires --ext-scale-best1-threshold (e.g. 0.14835 is SYNTHETIC ONLY)")
    if (args.ext_scale_best1_threshold is not None and
        (not np.isfinite(args.ext_scale_best1_threshold) or
         not 0 <= args.ext_scale_best1_threshold <= 1)):
        raise ValueError("--ext-scale-best1-threshold must be in [0,1]")
    if args.ext_scale_best1_directions < 16 or args.ext_scale_best1_directions % 2:
        raise ValueError("--ext-scale-best1-directions must be even and >=16")
    if args.ext_scale_directions != 0 and (args.ext_scale_directions < 16 or args.ext_scale_directions % 2):
        raise ValueError("ext-scale-directions must be 0 or even >=16")
    v3, v2 = float(args.voxel_3d_mm), float(args.voxel_2d_mm)
    VOXEL_3D_MM, VOXEL_2D_MM = v3, v2
    MIN_DEPTH_M, MAX_DEPTH_M = float(args.min_depth_m),float(args.max_depth_m)
    cfg = BoundaryConfig(
        voxel_3d_mm=v3, voxel_2d_mm=v2,
        normal_radius_mm=args.normal_radius_x*v3,
        sigma_r_mm=args.normal_sigma_r_x*v3,
        query_radius_mm=args.query_radius_x*v2,
        boundary_radius_mm=args.boundary_radius_x*v2,
        boundary_epsilon_mm=args.boundary_epsilon_x*v2,
        tensor_radius_mm=args.tensor_radius_x*v2,
        tensor_sigma_d_mm=args.tensor_sigma_d_x*v2,
        vote_line_sigma_mm=args.vote_line_sigma_x*v2,
        hough_min_score=args.hough_min_score,
        corner_search_radius_mm=args.corner_search_radius_x*v2,
        corner_step_max_mm=args.corner_step_max_x*v2,
        corner_refine_radius_mm=args.corner_refine_radius_x*v2,
        corner_converge_mm=args.corner_converge_x*v2,
        corner_peak_ratio=args.corner_peak_ratio,
        corner_max_iterations=args.corner_max_iterations,
        corner_lambda_min_threshold=args.corner_lambda_min_threshold,
        corner_balance_threshold=args.corner_balance_threshold,
        ext_radius_mm=args.ext_radius_x*v3,
        ext_epsilon_mm=args.ext_epsilon_mm,
        ext_sigma_mm=args.ext_sigma_mm,
        ext_directions=args.ext_directions,
        ext_min_score=args.ext_min_score,
        ext_method=args.ext_method,
        ext_top_fraction=args.ext_top_fraction,
        ext_trim_fraction=args.ext_trim_fraction,
        ext_min_span_mm=args.ext_min_span_mm,
        ext_rank_tau_mm=args.ext_rank_tau_mm,
        ext_scale_sigmas_x=tuple(float(x) for x in args.ext_scale_sigmas_x.split(",")),
        ext_scale_mode=args.ext_scale_mode,
        ext_scale_radius_gain=args.ext_scale_radius_gain,
        ext_scale_directions=args.ext_scale_directions,
        ext_scale_output=args.ext_scale_output,
        ext_scale_best1=args.ext_scale_best1,
        ext_scale_best1_threshold=(float(args.ext_scale_best1_threshold)
                                   if args.ext_scale_best1_threshold is not None else float("nan")),
        ext_scale_best1_directions=args.ext_scale_best1_directions,
        ext_scale_best1_sigmas_x=tuple(float(s) for s in args.ext_scale_best1_sigmas_x.split(",")),

        sigma_e_deg=args.normal_sigma_e_deg,
        normal_kernel=args.normal_kernel,
        normal_iterations=args.normal_max_iterations,
        normal_update_mode=args.normal_update_mode,
        surface_weight_min=args.surface_weight_min,
        min_neighbors=args.min_neighbors,
        show_3d=not args.no_3d_view,
        enable_iteration=args.enable_iteration,
        enable_refine=args.enable_refine,
    )
    return cfg



def kinect_depth_preview(depth_mm):
    """Geometry-registered display. Every display pixel uses the same depth pixel."""
    dep = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(dep) & (dep > 0) & (dep != 65535)
    image = np.empty((*dep.shape, 3), dtype=np.uint8)
    image[:] = (18, 20, 28)
    if np.any(valid):
        z = dep[valid]
        low = float(np.percentile(z, 1))
        high = float(np.percentile(z, 99))
        if high - low < 1.0:
            high = low + 1.0
        scale = np.clip((dep[valid] - low) / (high - low), 0, 1)
        image[valid] = (matplotlib.colormaps["turbo_r"](scale)[:, :3] * 255).astype(np.uint8)
    return image


def resolve_kinect_sequence(input_path, sequence_name=None):
    """Accept extracted sequence, sequence ZIP, or parent containing sequences."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip" or not zipfile.is_zipfile(source):
            raise ValueError(f"7-Scenes input must be a valid ZIP or directory: {source}")
        # Handle BOTH seq-01.zip and the outer office.zip without requiring
        # extraction of thousands of frames or loading a huge nested ZIP into RAM.
        with zipfile.ZipFile(source) as outer:
            members = [info for info in outer.infolist() if not info.is_dir()]
            if any(info.filename.endswith(".depth.png") for info in members):
                return source
            nested = [info for info in members
                      if re.search(r"(?:^|/)seq-\d+\.zip$", info.filename, re.I)]
            if sequence_name:
                name = sequence_name.removesuffix(".zip")
                nested = [info for info in nested if Path(info.filename).stem == name]
            if not nested:
                raise ValueError(f"{source} does not contain a 7-Scenes sequence or "
                                 f"a seq-XX.zip member")
            nested.sort(key=lambda x: (Path(x.filename).stem != "seq-01", x.filename))
            info = nested[0]
            dest_dir = source.parent / ".7scenes_sequence_cache" / source.stem
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / Path(info.filename).name
            reuse = (target.is_file() and target.stat().st_size == info.file_size
                     and zipfile.is_zipfile(target))
            if not reuse:
                print(f"[KINECT] Outer ZIP found: extracting ONLY {info.filename} "
                      f"({info.file_size / 1024**3:.2f} GiB) to {target}...", flush=True)
                temp = target.with_name(target.name + ".partial")
                try:
                    with outer.open(info) as f, temp.open("wb") as dest:
                        shutil.copyfileobj(f, dest, length=8 * 1024 * 1024)
                    if not zipfile.is_zipfile(temp):
                        raise ValueError("Extracted sequence ZIP is corrupt")
                    temp.replace(target)
                finally:
                    temp.unlink(missing_ok=True)
            else:
                print(f"[KINECT] Reusing extracted sequence: {target}", flush=True)
            return target
    if list(source.glob("frame-*.depth.png")):
        return source
    candidates = sorted([q for q in source.iterdir()
                         if ((q.is_dir() and q.name.startswith("seq-") and
                              any(q.glob("frame-*.depth.png"))) or
                             (q.is_file() and q.name.startswith("seq-") and q.suffix.lower()==".zip"))],
                        key=lambda q: (q.stem, 0 if q.is_dir() else 1))
    if sequence_name:
        name = sequence_name.removesuffix(".zip")
        candidates = [q for q in candidates if q.stem == name or q.name == name]
    if not candidates:
        raise FileNotFoundError(f"No unpacked seq-XX or seq-XX.zip inside {source}; "
                                "unzip the OUTER office.zip once, then pass office/seq-01.zip")
    chosen = candidates[0]
    print(f"[KINECT] sequence selected: {chosen}", flush=True)
    return chosen


def read_kinect7scenes_frame(source, frame_index):
    """Read depth/color pair directly from 7-Scenes sequence ZIP or extracted folder."""
    if frame_index < 0:
        raise ValueError("--frame-index must be >= 0")
    depth_name = f"frame-{frame_index:06d}.depth.png"
    color_name = f"frame-{frame_index:06d}.color.png"
    if source.is_dir():
        depth_path = source / depth_name
        color_path = source / color_name
        if not depth_path.is_file() or not color_path.is_file():
            example = next(iter(sorted(source.glob("frame-*.depth.png"))), None)
            raise FileNotFoundError(f"Missing {depth_name} / {color_name} in {source}. "
                                    f"Available sample: {example.name if example else 'none'}")
        with Image.open(depth_path) as im:
            depth = np.asarray(im).copy()
        with Image.open(color_path) as im:
            color = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    else:
        if not zipfile.is_zipfile(source):
            raise ValueError(f"Not a valid 7-Scenes sequence ZIP: {source}")
        with zipfile.ZipFile(source) as zf:
            names = [q for q in zf.namelist() if not q.endswith("/")]
            matches = {"depth": [q for q in names if q.endswith("/" + depth_name) or q == depth_name],
                       "color": [q for q in names if q.endswith("/" + color_name) or q == color_name]}
            if not matches["depth"] or not matches["color"]:
                samples = [q for q in names if q.endswith(".depth.png")][:3]
                outer = any(q.lower().endswith(".zip") for q in names)
                if outer and not samples:
                    raise ValueError(f"{source.name} is likely the OUTER scene ZIP. "
                                     "Unzip it ONCE and pass its seq-01.zip to --capture.")
                raise FileNotFoundError(f"Missing frame {frame_index} in {source}; "
                                        f"first depth entries: {samples}")
            # Duplicate paths are unusual; select shortest matching member deterministically.
            dn = min(matches["depth"], key=len)
            cn = min(matches["color"], key=len)
            with zf.open(dn) as f, Image.open(io.BytesIO(f.read())) as im:
                depth = np.asarray(im).copy()
            with zf.open(cn) as f, Image.open(io.BytesIO(f.read())) as im:
                color = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    if depth.ndim != 2 or depth.dtype.kind not in "ui":
        raise ValueError(f"Expected unmodified 16-bit depth PNG, got {depth.shape} / {depth.dtype}")
    if depth.dtype.itemsize < 2:
        raise ValueError(f"Depth PNG is 8-bit; load 16-bit original: {source}")
    if color.shape[:2] != depth.shape:
        raise ValueError(f"7-Scenes depth/color dimensions do not match: {depth.shape}, {color.shape}")
    return depth, color


def load_kinect7scenes_capture(capture_path, args):
    """Load 7-Scenes as depth-camera XYZ. Raw RGB is deliberately NOT registered."""
    source = resolve_kinect_sequence(capture_path, args.kinect_seq)
    depth_mm, original_rgb = read_kinect7scenes_frame(source, args.frame_index)
    h, w = depth_mm.shape
    valid = (depth_mm != 65535) & (depth_mm > 0)
    invalid_n = int(np.size(valid) - np.count_nonzero(valid))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[~valid] = np.nan
    sx, sy = w / 640.0, h / 480.0
    intr = SimIntrinsics(w, h, args.kinect_fx*sx, args.kinect_fy*sy,
                         args.kinect_cx*sx, args.kinect_cy*sy)
    if not all(np.isfinite([intr.fx, intr.fy, intr.ppx, intr.ppy])) or min(intr.fx, intr.fy) <= 0:
        raise ValueError("Kinect depth camera intrinsics must be finite; focal lengths positive")
    depth_preview = kinect_depth_preview(depth_mm)
    display = depth_preview if args.kinect_view == "depth" else original_rgb
    # This is its own output directory: do not modify/extract the source sequence.
    output_root = source.parent / "kinect_fps_results" / f"{source.stem}_frame-{args.frame_index:06d}"
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "Microsoft 7-Scenes Kinect", "source_sequence": str(source),
        "source_frame": int(args.frame_index), "valid_depth_range_m": [0.3, 5.0],
        "depth_unit": "mm", "depth_invalid_value": 65535,
        "depth_camera_intrinsics_assumed": {"fx": intr.fx, "fy": intr.fy,
                                           "ppx": intr.ppx, "ppy": intr.ppy},
        "rgb_depth_aligned": False,
    }
    print(f"[KINECT] frame={args.frame_index:06d} | depth={depth_mm.dtype} {w}x{h} "
          f"| invalid={invalid_n:,}/{w*h:,} | source={source}", flush=True)
    print(f"[KINECT] depth camera intrinsics fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"cx={intr.ppx:.2f} cy={intr.ppy:.2f}", flush=True)
    print("[KINECT] WARNING: original RGB and depth are NOT calibrated/registered. "
          "Depth preview is the geometrically correct pixel grid. 'C' toggles raw RGB (approximate overlay).", flush=True)
    return output_root, meta, {
        "rgb": display, "depth_m": depth_m, "intr": intr,
        "kinect_depth_preview": depth_preview,
        "kinect_original_rgb": original_rgb, "kinect_view": args.kinect_view,
    }


def load_input_capture(args):
    """Keep RealSense compatibility, auto-detect Kinect 7-Scenes input."""
    path = Path(args.capture).expanduser().resolve()
    if path.is_dir():
        required = ("rgb.png", "depth_m.npy", "camera_info.json")
        if all((path/name).is_file() for name in required) or any(
            q.is_dir() and all((q/name).is_file() for name in required)
            for q in path.iterdir()):
            return load_saved_realsense_capture(path)
    return load_kinect7scenes_capture(path, args)


def load_saved_realsense_capture(capture_dir):
    """Reconstruct EXACT aligned depth->XYZ input of the RealSense detector."""
    import json
    folder = Path(capture_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a capture directory: {folder}")
    required = ("rgb.png", "depth_m.npy", "camera_info.json")
    # --capture may point to the root capture directory; select the newest
    # complete timestamped subfolder without requiring manual path lookup.
    if not all((folder/name).is_file() for name in required):
        captures = [p for p in folder.iterdir() if p.is_dir() and
                    all((p/name).is_file() for name in required)]
        if captures:
            folder = max(captures, key=lambda p: (p.name, p.stat().st_mtime))
            print(f"[CAPTURE] selected latest: {folder}")
    missing = [name for name in required if not (folder/name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {folder}")
    meta = json.loads((folder/"camera_info.json").read_text(encoding="utf-8"))
    intr_data = meta.get("aligned_intrinsics_used_by_detector")
    if not intr_data:
        raise ValueError("camera_info.json is missing aligned_intrinsics_used_by_detector")
    intr = SimIntrinsics(**{key: intr_data[key] for key in
                            ("width", "height", "fx", "fy", "ppx", "ppy")})
    depth_m = np.load(folder/"depth_m.npy", allow_pickle=False)
    if depth_m.ndim != 2:
        raise ValueError("depth_m.npy must be an (H,W) array [meters]")
    rgb = np.asarray(Image.open(folder/"rgb.png").convert("RGB"),dtype=np.uint8)
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(f"RGB/depth dimensions differ: {rgb.shape[:2]} vs {depth_m.shape}")
    if depth_m.shape != (intr.height,intr.width):
        raise ValueError("Saved intrinsics don't match aligned depth dimensions")
    return folder, meta, {"rgb":rgb, "depth_m":np.asarray(depth_m,dtype=np.float32),
                          "intr":intr}


def main():
    args = parse_args()
    folder, meta, capture = load_input_capture(args)
    depth_range=meta.get("valid_depth_range_m", [0.15,3.0])
    if args.min_depth_m is None:
        args.min_depth_m=float(depth_range[0])
    if args.max_depth_m is None:
        args.max_depth_m=float(depth_range[1])
    cfg=configure(args)
    if o3d is None:
        raise RuntimeError("open3d is required for voxel downsampling and 3D debug")
    if tk is None:
        raise RuntimeError("Tk unavailable; install tkinter, pillow")
    estimator=load_normal_estimator(args.normal_module)
    saved_corners = []
    saved_detection_path = folder / "detections.json"
    if saved_detection_path.is_file() and not args.no_detection_overlay:
        import json
        try:
            payload=json.loads(saved_detection_path.read_text(encoding="utf-8"))
            for entry in payload.get("corners", []):
                pos=np.asarray(entry.get("position_mm", []),dtype=float)
                if pos.shape==(3,) and np.isfinite(pos).all() and pos[2]>0:
                    saved_corners.append(pos)
            print(f"[OVERLAY] {len(saved_corners)} saved FPS corners from detections.json")
        except (ValueError, OSError) as exc:
            print(f"[OVERLAY] cannot load detections.json: {exc}")
    frozen={
        "rgb":capture["rgb"], "depth_m":capture["depth_m"],
        "kinect_depth_preview":capture.get("kinect_depth_preview"),
        "kinect_original_rgb":capture.get("kinect_original_rgb"),
        "kinect_view":capture.get("kinect_view"),
        "intr":capture["intr"], "selected_pixel":None,
        "capture_path":folder,
        "source_meta":meta,
        "saved_fps_corners":saved_corners,
        "points_raw_mm":depth_to_xyz_mm(capture["depth_m"],capture["intr"]),
        "local_normal_recompute":bool(args.local_normal_recompute),
        "fps_count": int(args.fps_count),
        "nms_radius_mm": float(args.corner_nms_radius_x*cfg.voxel_3d_mm),
        "pending_detect": bool(args.auto_detect),
        "save_global_results": bool(args.save_global_results),
        "enable_scene_3d": not bool(args.no_3d_scene or args.no_3d_view),
        "show_fps_seeds": False, "show_accepted_candidates": False,
        "global_detection": None, "global_view": None,
    }
    if len(frozen["points_raw_mm"])<cfg.min_neighbors:
        raise RuntimeError("Too few valid depth samples")
    print(f"[BUILD] {BUILD}")
    print(f"[CAPTURE] {folder}")
    print(f"[INPUT] {meta.get('dataset', 'aligned RealSense RGB+depth')} "
          f"{capture['depth_m'].shape} | "
          f"depth range={MIN_DEPTH_M:g}..{MAX_DEPTH_M:g} m "
          f"| raw points={len(frozen['points_raw_mm']):,}")
    print(f"[SWITCHES] iteration={cfg.enable_iteration} refine={cfg.enable_refine} "
          f"Hough Top-{HOUGH_TOP_K} EXT-directions={cfg.ext_directions}")
    print(f"[HOUGH] min lambda_min={cfg.hough_min_score:g} (absolute, BEFORE Top-4/3D; "
          f"0 disables) | relative cutoff=20% of query max")
    print(f"[EXT] ACTIVE={cfg.ext_method}, min_score={cfg.ext_min_score:g} "
          f"| Length top={100*cfg.ext_top_fraction:g}%, "
          f"trim={100*cfg.ext_trim_fraction:g}%, min_span={cfg.ext_min_span_mm:g} mm "
          f"| legacy Max sigma={cfg.ext_sigma_mm:g} mm")
    print("[EXT] Projection Length / Max / Soft Rank all logged; threshold must be recalibrated")
    print("[DETECTOR] FINAL Hough+selected EXT PASS/FAIL; old tensor verification DIAGNOSTIC only")
    t0=time.perf_counter()
    frozen["points_voxel_mm"]=voxel_downsample_3d(frozen["points_raw_mm"], cfg.voxel_3d_mm)
    print(f"[VOXEL] {len(frozen['points_voxel_mm']):,} points in {time.perf_counter()-t0:.2f}s")
    if not args.local_normal_recompute:
        frozen["scene_cache"]=precompute_scene_cache(
            frozen["points_voxel_mm"], cfg, estimator)
    else:
        frozen["scene_cache"]=None
    print(f"[GUI] global FPS seeds={args.fps_count} auto-detect={args.auto_detect} "
          f"| red C# click=inspect original FPS seed | G=rescan, F/T=overlays", flush=True)
    try:
        while True:
            if run_frozen_click_stage(frozen,cfg,estimator)=="quit":
                break
            # R resets GUI/query while reusing the same stored depth and SAH cache.
            frozen["selected_pixel"]=None
    finally:
        close_query_views(frozen)




# Original camera/depth/noise helpers are appended below without changing

# their computation. The script entry point follows those definitions.





def normalize(v: np.ndarray, eps: float=1e-12) -> np.ndarray:

    v = np.asarray(v, dtype=np.float64).reshape(3)

    n = float(np.linalg.norm(v))

    if not np.isfinite(n) or n < eps:

        raise ValueError('Cannot normalize near-zero vector')

    return v / n



def make_tangent_basis(normal: np.ndarray):

    n = normalize(normal)

    if abs(float(n[2])) < 0.9:

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    else:

        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = normalize(np.cross(ref, n))

    v = normalize(np.cross(n, u))

    return (u, v, n)



def depth_to_xyz_mm(depth_m: np.ndarray, intr) -> np.ndarray:

    h, w = depth_m.shape

    vv, uu = np.mgrid[0:h, 0:w]

    z = np.asarray(depth_m, dtype=np.float64)

    valid = np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)

    if not np.any(valid):

        return np.empty((0, 3), dtype=np.float32)

    u = uu[valid].astype(np.float64)

    v = vv[valid].astype(np.float64)

    zm = z[valid]

    x = (u - intr.ppx) / intr.fx * zm

    y = (v - intr.ppy) / intr.fy * zm

    return (np.column_stack((x, y, zm)) * 1000.0).astype(np.float32)



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



def voxel_downsample_3d(points_mm: np.ndarray, voxel_mm: float) -> np.ndarray:

    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)

    if len(pts) == 0:

        return np.empty((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pts)

    down = pcd.voxel_down_sample(float(voxel_mm))

    return np.asarray(down.points, dtype=np.float64)



def orient_normals_to_camera(points_mm: np.ndarray, normals: np.ndarray) -> np.ndarray:

    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)

    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3).copy()

    lengths = np.linalg.norm(nrm, axis=1)

    valid = np.isfinite(nrm).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)

    if not np.any(valid):

        return nrm

    ids = np.flatnonzero(valid)

    view = -pts[ids]

    flip = np.sum(nrm[ids] * view, axis=1) < 0.0

    nrm[ids[flip]] *= -1.0

    nrm[ids] /= np.linalg.norm(nrm[ids], axis=1)[:, None]

    return nrm



@dataclass

class SimIntrinsics:

    width: int

    height: int

    fx: float

    fy: float

    ppx: float

    ppy: float



@dataclass

class NoiseConfig:

    gaussian_sigma_mm: float = 0.0

    outlier_rate: float = 0.0

    outlier_mm: float = 0.0

    dropout_rate: float = 0.0

    dropout_patches: int = 0

    dropout_patch_radius_px: int = 0

    quantization_mm: float = 0.0

    seed: int = 0



def validate_probability(name: str, value: float):

    if not 0.0 <= value <= 1.0:

        raise ValueError(f'{name} must be in [0,1]')



def load_stl_as_mm(path: str, unit_mm: float):

    mesh = o3d.io.read_triangle_mesh(path)

    if mesh.is_empty() or len(mesh.triangles) == 0:

        raise RuntimeError(f'Failed to load STL: {path}')

    if unit_mm <= 0.0:

        raise ValueError('--stl-unit-mm must be > 0')

    if abs(unit_mm - 1.0) > 1e-12:

        mesh.scale(float(unit_mm), center=(0.0, 0.0, 0.0))

    mesh.compute_vertex_normals()

    mesh.paint_uniform_color([0.72, 0.74, 0.78])

    return mesh



def make_raycast_scene(mesh):

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(mesh_t)

    return scene



def pinhole_to_intrinsics(params) -> SimIntrinsics:

    K = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)

    return SimIntrinsics(width=int(params.intrinsic.width), height=int(params.intrinsic.height), fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2]))



def make_camera_rays(params):

    intr = pinhole_to_intrinsics(params)

    K = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)

    E_w2c = np.asarray(params.extrinsic, dtype=np.float64)

    T_c2w = np.linalg.inv(E_w2c)

    vv, uu = np.mgrid[0:intr.height, 0:intr.width]

    x = (uu.astype(np.float64) - K[0, 2]) / K[0, 0]

    y = (vv.astype(np.float64) - K[1, 2]) / K[1, 1]

    dirs_c = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3)

    R_c2w = T_c2w[:3, :3]

    origin_w = T_c2w[:3, 3]

    dirs_w = dirs_c @ R_c2w.T

    origins_w = np.repeat(origin_w[None, :], len(dirs_w), axis=0)

    rays = np.concatenate((origins_w, dirs_w), axis=1).astype(np.float32)

    return (rays, intr)



def raycast_depth_mm(scene, params):

    rays_np, intr = make_camera_rays(params)

    result = scene.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))

    depth_mm = result['t_hit'].numpy().reshape(intr.height, intr.width).astype(np.float64)

    valid = np.isfinite(depth_mm)

    valid &= depth_mm >= MIN_DEPTH_M * 1000.0

    valid &= depth_mm <= MAX_DEPTH_M * 1000.0

    depth_mm[~valid] = np.nan

    return (depth_mm, intr)



def corrupt_depth(depth_mm: np.ndarray, cfg: NoiseConfig) -> np.ndarray:

    rng = np.random.default_rng(cfg.seed)

    out = np.asarray(depth_mm, dtype=np.float64).copy()

    valid = np.isfinite(out)

    if cfg.gaussian_sigma_mm > 0.0 and np.any(valid):

        out[valid] += rng.normal(0.0, cfg.gaussian_sigma_mm, np.count_nonzero(valid))

    valid = np.isfinite(out)

    if cfg.outlier_rate > 0.0 and cfg.outlier_mm > 0.0 and np.any(valid):

        flat = np.flatnonzero(valid.ravel())

        n = min(len(flat), int(round(cfg.outlier_rate * len(flat))))

        if n > 0:

            ids = rng.choice(flat, n, replace=False)

            mag = rng.uniform(0.5 * cfg.outlier_mm, cfg.outlier_mm, n)

            sign = rng.choice(np.array([-1.0, 1.0]), n)

            out.ravel()[ids] += sign * mag

    if cfg.quantization_mm > 0.0:

        valid = np.isfinite(out)

        out[valid] = np.round(out[valid] / cfg.quantization_mm) * cfg.quantization_mm

    valid = np.isfinite(out)

    bad = valid & ((out < MIN_DEPTH_M * 1000.0) | (out > MAX_DEPTH_M * 1000.0))

    out[bad] = np.nan

    if cfg.dropout_rate > 0.0:

        valid = np.isfinite(out)

        drop = valid & (rng.random(out.shape) < cfg.dropout_rate)

        out[drop] = np.nan

    if cfg.dropout_patches > 0 and cfg.dropout_patch_radius_px > 0:

        yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]

        for _ in range(cfg.dropout_patches):

            ys, xs = np.nonzero(np.isfinite(out))

            if len(xs) == 0:

                break

            j = int(rng.integers(0, len(xs)))

            cx = int(xs[j])

            cy = int(ys[j])

            r = cfg.dropout_patch_radius_px

            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

            out[mask] = np.nan

    return out.astype(np.float32)



def initialize_view(vis, mesh, fov_deg: float):

    ctr = vis.get_view_control()

    bbox = mesh.get_axis_aligned_bounding_box()

    ctr.set_lookat(bbox.get_center())

    ctr.set_front([0.6, -0.7, -0.35])

    ctr.set_up([0.0, 0.0, 1.0])

    ctr.set_zoom(0.72)

    try:

        ctr.change_field_of_view(float(fov_deg) - float(ctr.get_field_of_view()))

    except Exception:

        pass



def capture_render_rgb(vis, width: int, height: int):

    screen = np.asarray(vis.capture_screen_float_buffer(do_render=True))

    if screen.ndim != 3 or screen.shape[2] < 3:

        return np.full((height, width, 3), 32, dtype=np.uint8)

    rgb = np.clip(screen[..., :3] * 255.0, 0, 255).astype(np.uint8)

    if rgb.shape[:2] != (height, width):

        rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))

    return rgb.copy()



def run_camera_selector(mesh, scene, noise_cfg, width, height, fov_deg):

    state = {'capture': None, 'cancel': False}

    vis = o3d.visualization.VisualizerWithKeyCallback()

    ok = vis.create_window(window_name='Synthetic STL camera | mouse: orbit/pan/zoom | P: capture | Q: quit', width=int(width), height=int(height))

    if not ok:

        raise RuntimeError('Failed to create Open3D window')

    vis.add_geometry(mesh)

    opt = vis.get_render_option()

    if opt is not None:

        opt.background_color = np.array([0.025, 0.025, 0.025])

        opt.mesh_show_back_face = True

    initialize_view(vis, mesh, fov_deg)



    def capture_cb(v):

        params = v.get_view_control().convert_to_pinhole_camera_parameters()

        intr = pinhole_to_intrinsics(params)

        rgb = capture_render_rgb(v, intr.width, intr.height)

        clean_depth_mm, intr = raycast_depth_mm(scene, params)

        noisy_depth_mm = corrupt_depth(clean_depth_mm, noise_cfg)

        state['capture'] = {'rgb': rgb, 'depth_mm': noisy_depth_mm, 'intr': intr}

        try:

            v.close()

        except Exception:

            pass

        return False



    def quit_cb(v):

        state['cancel'] = True

        try:

            v.close()

        except Exception:

            pass

        return False

    vis.register_key_callback(ord('P'), capture_cb)

    vis.register_key_callback(ord('Q'), quit_cb)

    vis.run()

    try:

        vis.destroy_window()

    except Exception:

        pass

    if state['cancel']:

        return None

    return state['capture']



def prepare_frozen(capture: dict) -> dict:

    depth_m = np.asarray(capture['depth_mm'], dtype=np.float32) / 1000.0

    points_raw_mm = depth_to_xyz_mm(depth_m, capture['intr'])

    if len(points_raw_mm) == 0:

        raise RuntimeError('No valid depth points')

    return {'rgb': capture['rgb'].copy(), 'depth_m': depth_m, 'intr': capture['intr'], 'points_raw_mm': points_raw_mm, 'selected_pixel': None}



def draw_frozen(frozen: dict) -> np.ndarray:
    """Draw up-to-date FPS detections on RGB, not just old detections.json."""
    img = Image.fromarray(frozen["rgb"])
    draw = ImageDraw.Draw(img)
    intr = frozen["intr"]
    det = frozen.get("global_detection")
    if det is not None:
        if frozen.get("show_fps_seeds", False):
            points = frozen["points_voxel_mm"]
            for sid in det["seed_indices"]:
                q = point_mm_to_pixel(points[int(sid)], intr)
                if q:
                    u,v=q
                    draw.ellipse((u-2,v-2,u+2,v+2), fill=(60,155,250))
        if frozen.get("show_accepted_candidates", False):
            for c in det["accepted_before_nms"]:
                q = point_mm_to_pixel(c["coarse_position"], intr)
                if q:
                    u,v=q
                    draw.ellipse((u-3,v-3,u+3,v+3), outline=(255,174,40), width=2)
        for i,c in enumerate(det["corners"], 1):
            q = point_mm_to_pixel(c["position"], intr)
            if q:
                u,v=q
                draw.ellipse((u-9,v-9,u+9,v+9), outline=(245,45,45), width=3)
                draw.line((u-12,v,u+12,v), fill=(245,45,45), width=2)
                draw.line((u,v-12,u,v+12), fill=(245,45,45), width=2)
                draw.text((u+10,v-16), f"C{i}", fill=(255,60,55), stroke_width=1,
                          stroke_fill=(10,10,10))
        draw.rectangle((0,0,320,25), fill=(10,17,28))
        draw.text((7,5),
                  f"FPS={len(det['seed_indices'])}  PASS={len(det['accepted_before_nms'])}  NMS={len(det['corners'])}",
                  fill=(242,244,248))
    elif frozen.get("saved_fps_corners"):
        for k,pos in enumerate(frozen["saved_fps_corners"],1):
            q = point_mm_to_pixel(pos, intr)
            if q:
                u,v=q
                draw.ellipse((u-7,v-7,u+7,v+7), outline=(255,210,0), width=2)
                draw.text((u+8,v-12),f"old C{k}",fill=(255,230,0))
    if frozen.get("kinect_depth_preview") is not None:
        h = frozen["rgb"].shape[0]
        mode = frozen.get("kinect_view", "depth")
        draw.rectangle((0,h-23,430,h), fill=(10,17,28))
        title = ("KINECT: DEPTH PIXELS (GEOMETRY ALIGNED)" if mode == "depth"
                 else "KINECT: RAW RGB NOT REGISTERED (APPROX OVERLAY)")
        draw.text((7,h-18), title + "  |  C: toggle", fill=(243,245,250))
    if frozen.get("selected_pixel") is not None:
        x,y=frozen["selected_pixel"]
        draw.line((x-12,y,x+12,y), fill=(0,255,240), width=3)
        draw.line((x,y-12,x,y+12), fill=(0,255,240), width=3)
    return np.asarray(img).copy()


if __name__ == "__main__":

    main()