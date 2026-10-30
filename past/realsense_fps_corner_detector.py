"""RealSense Hough Top-4 + 1024-direction Extremality FPS corner detector.

python realsense_fps_corner_detector.py \
  --voxel-3d-mm 10 \
  --voxel-2d-mm 10 \
  --normal-radius-x 4 \
  --normal-sigma-r-x 0.5 \
  --query-radius-x 10 \
  --boundary-radius-x 4 \
  --boundary-epsilon-x 1 \
  --tensor-radius-x 4 \
  --tensor-sigma-d-x 1.5 \
  --vote-line-sigma-x 1 \
  --fps-count 1500 \
  --corner-nms-radius-x 3 \
  --surface-weight-min 0.1 \
  --ext-radius-x 4 \
  --ext-sigma-mm 0.5 \
  --ext-directions 1024 \
  --ext-min-score 0.08 \
  --no-iteration \
  --no-refine

"""

from __future__ import annotations



import argparse

import json

from datetime import datetime

import importlib.util

import inspect

import sys

import time

from pathlib import Path

from dataclasses import dataclass



import numpy as np
import cv2
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from scipy.spatial import cKDTree

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

WINDOW_FROZEN = "Synthetic frozen frame | click query"

WINDOW_BOUNDARY = "Tangent projection | second-order boundary estimate"

BUILD = "realsense-fps-hough-top4-ext1024-global-sah"
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

    vote_line_sigma_mm: float = 2.0
    ext_radius_mm: float = 8.0
    ext_epsilon_mm: float = 0.5
    ext_sigma_mm: float = 0.5
    ext_directions: int = 1024
    ext_min_score: float = 0.08

    enable_iteration: bool = False
    enable_refine: bool = True
    corner_nms_radius_mm: float = 6.0

    # Iterative corner localizer. These all live in tangent/UV space, so their
    # metric values are resolved from 2-D voxel multipliers in configure().
    corner_search_radius_mm: float = 24.0

    corner_step_max_mm: float = 12.0

    corner_refine_radius_mm: float = 4.0

    corner_converge_mm: float = 2.0

    corner_peak_ratio: float = 0.70

    corner_max_iterations: int = 10

    # Final false-positive rejection, evaluated only after localization stops.
    # S = lambda_min, C = lambda_min / (lambda_max + lambda_min + eps).
    corner_lambda_min_threshold: float = 0.03

    corner_balance_threshold: float = 0.15

    # Directional-structure verification at the final query.
    # A physical 2-D corner should contain TWO coherent boundary-direction modes,
    # not merely an isotropic/random set of directions with large lambda_min.
    two_mode_min_coherence: float = 0.80
    two_mode_min_fraction: float = 0.20
    two_mode_min_angle_deg: float = 35.0
    two_mode_max_angle_deg: float = 150.0

    # Scene-level detector: FPS query count is explicit.
    # Exactly min(fps_count, number_of_valid_SAH_points) seeds are generated.
    fps_count: int = 64

    # Multi-query convergence consensus. Candidates that pass the local verifier
    # must form a 3-D radius cluster supported by >= min_support FPS seeds.
    corner_consensus_radius_mm: float = 6.0
    corner_consensus_min_support: int = 2

    sigma_e_deg: float = 15.0

    sigma_r_mm: float = 1.0

    normal_kernel: str = "cauchy"

    normal_iterations: int = 5

    normal_update_mode: str = "synchronous"

    surface_weight_min: float = 0.1

    min_neighbors: int = 6

    show_3d: bool = True



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





def describe_failed_sah_query(query_fit, normal_info, query_local, cfg):
    """Return a compact diagnosis when the query SAH update never succeeds.

    query_fit["valid"] means that at least one weighted-PCA SAH update
    succeeded.  This helper separates the two cheap cases we can diagnose
    directly from the returned data:

      1) too few radius neighbors,
      2) too few finite rough/PCA normals,

    and labels everything else as a weighted-update failure.
    """
    fit_ids = np.asarray(query_fit.get("indices", []), dtype=np.int64)
    raw_neighbors = int(len(fit_ids))
    iteration = int(query_fit.get("iteration", 0))

    rough_valid_neighbors = -1
    rough = normal_info.get("rough_normals", None)
    if rough is not None:
        rough = np.asarray(rough, dtype=float)
        if rough.ndim == 2 and rough.shape[1] == 3 and len(fit_ids):
            good_ids = fit_ids[(fit_ids >= 0) & (fit_ids < len(rough))]
            if len(good_ids):
                rr = rough[good_ids]
                rr_norm = np.linalg.norm(np.nan_to_num(rr, nan=0.0), axis=1)
                rough_valid_neighbors = int(np.count_nonzero(
                    np.isfinite(rr).all(axis=1) & (rr_norm > 1e-12)
                ))
            else:
                rough_valid_neighbors = 0

    if raw_neighbors < int(cfg.min_neighbors):
        reason = "insufficient_radius_neighbors"
    elif 0 <= rough_valid_neighbors < 3:
        reason = "insufficient_valid_rough_normals"
    else:
        reason = "weighted_SAH_update_failed"

    return {
        "reason": reason,
        "raw_neighbors": raw_neighbors,
        "rough_valid_neighbors": rough_valid_neighbors,
        "iteration": iteration,
    }


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
        diag = describe_failed_sah_query(query_fit, normal_info, query_local, cfg)
        rough_txt = (
            str(diag["rough_valid_neighbors"])
            if diag["rough_valid_neighbors"] >= 0 else "unknown"
        )
        raise ValueError(
            "SAH query update failed"
            f" | reason={diag['reason']}"
            f" | r={cfg.normal_radius_mm:g} mm"
            f" | neighbors={diag['raw_neighbors']}"
            f" | valid_rough={rough_txt}"
            f" | min_neighbors={cfg.min_neighbors}"
            f" | iteration={diag['iteration']}"
        )

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



def analyze_two_direction_coherence(ctx, result):
    """Test whether QUERY tensor support is explained by two coherent directions.

    IMPORTANT: this uses exactly the same boundary-distance weight w_d as the
    tensor. Spatial Gaussian weighting remains OFF.

    The test is intentionally different from the eigenvalue test. A random or
    highly scattered direction field can make lambda_min large and balance near
    0.5. Here we require two compact directional modes with sufficient support
    and a meaningful angular separation.
    """
    cfg = ctx["config"]
    qi = int(ctx["query_uv_index"])
    uv = np.asarray(ctx["support_uv"], dtype=float)
    directions = np.asarray(result["inward_normal_uv"], dtype=float)
    distance = np.asarray(result["distance_mm"], dtype=float)

    source = np.asarray(result["valid_direction"], dtype=bool) & np.isfinite(distance)
    delta = uv - uv[qi]
    support = source & (np.sum(delta * delta, axis=1) <= float(cfg.tensor_radius_mm) ** 2)
    ids = np.flatnonzero(support)

    fail = {
        "passed": False,
        "reason": "",
        "contributor_count": int(len(ids)),
        "coherence_1": np.nan,
        "coherence_2": np.nan,
        "min_coherence": np.nan,
        "weight_fraction_1": np.nan,
        "weight_fraction_2": np.nan,
        "min_weight_fraction": np.nan,
        "separation_deg": np.nan,
    }
    if len(ids) < max(4, int(cfg.min_neighbors)):
        fail["reason"] = "too_few_direction_contributors"
        return fail

    n = directions[ids].copy()
    norm = np.linalg.norm(n, axis=1)
    finite = np.isfinite(n).all(axis=1) & np.isfinite(norm) & (norm > 1e-12)
    if np.count_nonzero(finite) < max(4, int(cfg.min_neighbors)):
        fail["reason"] = "too_few_finite_directions"
        return fail
    ids = ids[finite]
    n = n[finite] / norm[finite, None]

    sigma_d = max(float(cfg.tensor_sigma_d_mm), np.finfo(float).eps)
    w = np.exp(-0.5 * (distance[ids] / sigma_d) ** 2)
    valid_w = np.isfinite(w) & (w > 1e-12)
    n = n[valid_w]
    w = w[valid_w]
    fail["contributor_count"] = int(len(n))
    if len(n) < 4 or float(np.sum(w)) <= np.finfo(float).tiny:
        fail["reason"] = "insufficient_weighted_directions"
        return fail

    # Deterministic weighted 2-means on unit direction vectors.
    # Seed 1: strongest contributor. Seed 2: most angularly separated
    # contributor among non-negligible contributors, preventing a tiny-weight
    # outlier from defining the second mode.
    first = int(np.argmax(w))
    c1 = n[first].copy()
    significant = w >= 0.10 * float(np.max(w))
    sig_ids = np.flatnonzero(significant)
    if len(sig_ids) < 2:
        fail["reason"] = "no_second_direction_seed"
        return fail
    dot1 = np.clip(n[sig_ids] @ c1, -1.0, 1.0)
    second = int(sig_ids[np.argmin(dot1)])
    c2 = n[second].copy()

    centers = np.vstack((c1, c2))
    labels = np.zeros(len(n), dtype=np.int32)
    for _ in range(20):
        similarity = n @ centers.T
        new_labels = np.argmax(similarity, axis=1).astype(np.int32)
        if np.all(new_labels == 0) or np.all(new_labels == 1):
            fail["reason"] = "collapsed_two_mode_fit"
            return fail

        new_centers = np.zeros((2, 2), dtype=float)
        ok = True
        for k in (0, 1):
            m = new_labels == k
            v = np.sum(n[m] * w[m, None], axis=0)
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv <= 1e-12:
                ok = False
                break
            new_centers[k] = v / nv
        if not ok:
            fail["reason"] = "degenerate_two_mode_center"
            return fail

        converged = np.array_equal(new_labels, labels)
        labels = new_labels
        centers = new_centers
        if converged:
            break

    total_w = float(np.sum(w))
    cluster_w = []
    coherence = []
    for k in (0, 1):
        m = labels == k
        wk = float(np.sum(w[m]))
        cluster_w.append(wk)
        resultant = np.sum(n[m] * w[m, None], axis=0)
        coherence.append(float(np.linalg.norm(resultant) / max(wk, np.finfo(float).tiny)))

    fractions = np.asarray(cluster_w, dtype=float) / max(total_w, np.finfo(float).tiny)
    separation_deg = float(np.degrees(np.arccos(np.clip(float(centers[0] @ centers[1]), -1.0, 1.0))))
    min_coherence = float(min(coherence))
    min_fraction = float(np.min(fractions))

    pass_coherence = min_coherence >= float(cfg.two_mode_min_coherence)
    pass_fraction = min_fraction >= float(cfg.two_mode_min_fraction)
    pass_angle = (
        float(cfg.two_mode_min_angle_deg) <= separation_deg
        <= float(cfg.two_mode_max_angle_deg)
    )
    passed = bool(pass_coherence and pass_fraction and pass_angle)

    if not pass_coherence:
        reason = "direction_modes_not_coherent"
    elif not pass_fraction:
        reason = "second_direction_support_too_small"
    elif separation_deg < float(cfg.two_mode_min_angle_deg):
        reason = "direction_modes_too_parallel"
    elif separation_deg > float(cfg.two_mode_max_angle_deg):
        reason = "direction_modes_too_opposed"
    else:
        reason = "accepted"

    return {
        "passed": passed,
        "reason": reason,
        "contributor_count": int(len(n)),
        "coherence_1": float(coherence[0]),
        "coherence_2": float(coherence[1]),
        "min_coherence": min_coherence,
        "weight_fraction_1": float(fractions[0]),
        "weight_fraction_2": float(fractions[1]),
        "min_weight_fraction": min_fraction,
        "separation_deg": separation_deg,
        "center_1": centers[0].copy(),
        "center_2": centers[1].copy(),
    }

def verify_corner_candidate(ctx, result, status):
    """Final corner verification using only the already-computed tensor eigenvalues.

    Localization and verification are deliberately separated:
      - lambda_min response moves the query,
      - only the FINAL location is accepted/rejected (click mode).

    Accept iff the localizer geometrically converged and both
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


def make_boundary_figure(ctx, result, localization=None):
    """Diagnostic 2-D view for the boundary estimator and corner tensor.

    This version does NOT change the estimator or tensor computation.  It only
    adds arrow diagnostics so we can inspect the boundary-direction field that
    feeds the Harris-like tensor.
    """
    cfg = ctx["config"]

    # 2x3 diagnostic view:
    #   1) surface compatibility
    #   2) point-wise d_hat
    #   3) outward boundary-direction field (-n_i)
    #   4) the directions/weights that contribute to the QUERY tensor
    #   5) lambda_max
    #   6) lambda_min
    fig = Figure(figsize=(18, 11), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    ax_surface, ax_dist, ax_dir = axes[0]
    ax_query_tensor, ax_l1, ax_l2 = axes[1]

    raw = ctx["support_uv_raw"]
    keep = ctx["projection_keep"]
    surface_weight = ctx["projection_surface_weight"]
    uv = np.asarray(ctx["support_uv"], dtype=float)
    evaluate = np.asarray(result["evaluation_mask"], dtype=bool)
    distance = np.asarray(result["distance_mm"], dtype=float)
    directions = np.asarray(result["inward_normal_uv"], dtype=float)
    valid = evaluate & result["valid_direction"] & np.isfinite(distance)
    lim = cfg.query_radius_mm * 1.08
    qi = ctx["query_uv_index"]

    # ------------------------------------------------------------------
    # 1. Surface-selection / tangent-projection view
    # ------------------------------------------------------------------
    visible = np.linalg.norm(raw, axis=1) <= lim * 1.42
    ax_surface.scatter(raw[visible & ~keep, 0], raw[visible & ~keep, 1],
                       s=5, c="#d0d0d0", label="Excluded before projection sampling")
    use = visible & keep
    sc = ax_surface.scatter(raw[use, 0], raw[use, 1], c=surface_weight[use],
                            vmin=0, vmax=1, cmap="viridis", s=11)
    fig.colorbar(sc, ax=ax_surface, shrink=.78, label="Surface weight w_e * w_r")
    ax_surface.set_title("1. Surface selection and tangent projection")
    ax_surface.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # 2. Point-wise boundary-distance heatmap
    # ------------------------------------------------------------------
    context = ~evaluate
    if np.any(context):
        ax_dist.scatter(uv[context, 0], uv[context, 1],
                        s=7, c="#d7d7d7", alpha=.45, label="Outside evaluation ROI")

    invalid = evaluate & ~result["valid_direction"]
    if np.any(invalid):
        ax_dist.scatter(uv[invalid, 0], uv[invalid, 1],
                        s=18, c="#9a9a9a", marker="x", label="Undefined direction")

    if np.any(valid):
        sc_d = ax_dist.scatter(
            uv[valid, 0], uv[valid, 1],
            c=distance[valid], s=26, cmap="turbo_r",
            vmin=0.0, vmax=cfg.boundary_radius_mm,
            edgecolors="none", label="UV samples",
        )
        fig.colorbar(sc_d, ax=ax_dist, shrink=.78,
                     label="Estimated boundary distance d_hat [mm]")

    d_q = distance[qi]
    status = f"query d_hat={d_q:.3f} mm" if np.isfinite(d_q) else "query direction undefined"
    ax_dist.set_title("2. Point-wise boundary-distance heatmap\n" + status, fontsize=11)
    ax_dist.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Helper: spatially thin arrows only for readability.
    # This does NOT affect the tensor computation.
    # ------------------------------------------------------------------
    def thin_arrow_ids(mask, spacing_mm):
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            return ids
        spacing_mm = max(float(spacing_mm), np.finfo(float).eps)
        cells = np.floor(uv[ids] / spacing_mm + 0.5).astype(np.int64)
        center = cells.astype(float) * spacing_mm
        d2 = np.sum((uv[ids] - center) ** 2, axis=1)
        order = np.lexsort((d2, cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        first = np.r_[True, np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)]
        return ids[order[first]]

    # ------------------------------------------------------------------
    # 3. Boundary-direction field.
    # normals point INWARD; -normal therefore points OUTWARD / toward the
    # estimated local boundary.  Arrow length is fixed so this panel shows
    # direction quality, not d_hat magnitude.
    # ------------------------------------------------------------------
    arrow_spacing = max(3.0 * cfg.voxel_2d_mm, cfg.voxel_2d_mm)  # 6 mm by default
    arrow_len = max(2.0 * cfg.voxel_2d_mm, 4.0)                  # 4 mm by default
    arrow_ids = thin_arrow_ids(valid, arrow_spacing)

    ax_dir.scatter(uv[valid, 0], uv[valid, 1], s=9, c="#cfcfcf", alpha=.75)
    if len(arrow_ids):
        outward = -directions[arrow_ids]
        ax_dir.quiver(
            uv[arrow_ids, 0], uv[arrow_ids, 1],
            arrow_len * outward[:, 0], arrow_len * outward[:, 1],
            color="#1f77b4", angles="xy", scale_units="xy", scale=1,
            width=.0042, headwidth=3.6, headlength=4.6, headaxislength=4.0,
        )

    # Draw the query's ACTUAL estimated boundary vector p_q -> b_q.
    if valid[qi]:
        q_vec = -distance[qi] * directions[qi]
        ax_dir.quiver(
            [uv[qi, 0]], [uv[qi, 1]], [q_vec[0]], [q_vec[1]],
            color="#d97706", angles="xy", scale_units="xy", scale=1,
            width=.008, headwidth=4.2, headlength=5.0, zorder=12,
        )
        q_boundary = uv[qi] + q_vec
        ax_dir.scatter([q_boundary[0]], [q_boundary[1]], marker="D", s=52,
                       c="#d97706", zorder=13, label="Query boundary estimate")

    corner = result["corner_eigen"]
    ax_dir.add_patch(Circle((0, 0), corner["radius_mm"], fill=False,
                            ls="-", color="#d97706", lw=1.3,
                            label=f"Tensor radius {corner['radius_mm']:g} mm"))
    ax_dir.set_title(
        f"3. Boundary-direction field: outward = -n_i\n"
        f"fixed arrow={arrow_len:g} mm, display spacing~{arrow_spacing:g} mm"
    )
    ax_dir.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # 4. EXACT contributors to the QUERY tensor.
    # This panel is the key debug view: if the query lambda_min is large, we
    # should be able to see at least two independent arrow orientations here.
    # ------------------------------------------------------------------
    tensor_radius = float(corner["radius_mm"])
    sigma_s = float(corner["sigma_spatial_mm"])
    sigma_d = float(corner["sigma_distance_mm"])

    source = result["valid_direction"] & np.isfinite(distance)
    query_delta = uv - uv[qi]
    query_r2 = np.sum(query_delta * query_delta, axis=1)
    query_support = source & (query_r2 <= tensor_radius * tensor_radius)
    query_ids = np.flatnonzero(query_support)

    ax_query_tensor.scatter(uv[evaluate, 0], uv[evaluate, 1],
                            s=7, c="#e1e1e1", alpha=.55)

    if len(query_ids):
        # Match the tensor computation exactly: spatial weight is OFF.
        w_d = np.exp(-0.5 * (distance[query_ids] / sigma_d) ** 2)
        w = w_d

        w_max = float(np.max(w)) if len(w) else 1.0
        if not np.isfinite(w_max) or w_max <= np.finfo(float).tiny:
            w_max = 1.0
        w_norm = np.clip(w / w_max, 0.0, 1.0)

        sc_w = ax_query_tensor.scatter(
            uv[query_ids, 0], uv[query_ids, 1],
            c=w, s=22 + 45 * w_norm, cmap="plasma",
            vmin=0.0, vmax=w_max, edgecolors="none", zorder=3,
        )
        fig.colorbar(sc_w, ax=ax_query_tensor, shrink=.78,
                     label="Query tensor contribution w_d (spatial weight OFF)")

        # All true contributors are shown here (no thinning): radius is only 16 mm.
        outward_q = -directions[query_ids]
        arrow_colors = matplotlib.colormaps["plasma"](w_norm)
        ax_query_tensor.quiver(
            uv[query_ids, 0], uv[query_ids, 1],
            arrow_len * outward_q[:, 0], arrow_len * outward_q[:, 1],
            color=arrow_colors, angles="xy", scale_units="xy", scale=1,
            width=.005, headwidth=3.5, headlength=4.4, headaxislength=3.8,
            zorder=4,
        )

    ax_query_tensor.add_patch(Circle((0, 0), tensor_radius, fill=False,
                                     ls="-", color="#202020", lw=1.4))
    ax_query_tensor.set_title(
        "4. Directions actually used by QUERY tensor\n"
        f"r={tensor_radius:g} mm, spatial weight OFF, sigma_d={sigma_d:g} mm"
    )

    # ------------------------------------------------------------------
    # 5-6. Harris-like tensor eigenvalues (unchanged algorithm)
    # ------------------------------------------------------------------
    lam_max = corner["lambda_max"]
    lam_min = corner["lambda_min"]
    eig_valid = evaluate & np.isfinite(lam_max) & np.isfinite(lam_min)

    if np.any(eig_valid):
        finite_max = lam_max[eig_valid]
        vmax_eig = float(np.percentile(finite_max, 99.0))
        if not np.isfinite(vmax_eig) or vmax_eig <= 1e-12:
            vmax_eig = 1.0

        sc_l1 = ax_l1.scatter(
            uv[eig_valid, 0], uv[eig_valid, 1],
            c=lam_max[eig_valid], s=25, cmap="magma",
            vmin=0.0, vmax=vmax_eig, edgecolors="none",
        )
        fig.colorbar(sc_l1, ax=ax_l1, shrink=.78,
                     label="lambda_max (dominant direction)")

        sc_l2 = ax_l2.scatter(
            uv[eig_valid, 0], uv[eig_valid, 1],
            c=lam_min[eig_valid], s=25, cmap="magma",
            vmin=0.0, vmax=vmax_eig, edgecolors="none",
        )
        fig.colorbar(sc_l2, ax=ax_l2, shrink=.78,
                     label="lambda_min (second-direction evidence)")
    else:
        ax_l1.text(0.5, 0.5, "No valid tensor samples", ha="center", va="center",
                   transform=ax_l1.transAxes)
        ax_l2.text(0.5, 0.5, "No valid tensor samples", ha="center", va="center",
                   transform=ax_l2.transAxes)

    ax_l1.set_title(
        f"5. Boundary-direction tensor: lambda_max\n"
        f"radius={corner['radius_mm']:g} mm, spatial weight OFF"
    )
    ax_l2.set_title(
        f"6. Boundary-direction tensor: lambda_min\n"
        f"sigma_d={corner['sigma_distance_mm']:g} mm"
    )

    # Shared geometric annotations.

    # Re-express the full 3-D iteration trajectory in the FINAL tangent frame.
    path_uv_final = None
    if localization is not None and len(localization.get("path_world", [])):
        pw = np.asarray(localization["path_world"], dtype=float)
        dq = pw - ctx["query_world"]
        path_uv_final = np.column_stack((dq @ ctx["u"], dq @ ctx["v"]))

    for ax in (ax_surface, ax_dist, ax_dir, ax_query_tensor, ax_l1, ax_l2):

        query_color = "#d02531" if ax is ax_surface else "#202020"
        ax.scatter([0], [0], marker="+", s=170, linewidths=2,
                   color=query_color, zorder=15)
        if path_uv_final is not None and len(path_uv_final):
            ax.plot(path_uv_final[:, 0], path_uv_final[:, 1], "-o",
                    color="#2f6f9f", lw=1.5, ms=4, alpha=.9, zorder=14)
            ax.scatter([path_uv_final[0, 0]], [path_uv_final[0, 1]],
                       marker="s", s=42, c="#2f6f9f", zorder=15)
        ax.add_patch(Circle((0, 0), cfg.query_radius_mm, fill=False,
                            ls="--", color="#a9b1b5", lw=.8))
        ax.axhline(0, color="#e8e8e8", lw=.6, zorder=0)
        ax.axvline(0, color="#e8e8e8", lw=.6, zorder=0)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
               xlabel="u [mm]", ylabel="v [mm]", aspect="equal")

    # Current tangent-frame corner target (if a valid response exists).
    est = result.get("corner_estimate")
    if est is not None:
        t = np.asarray(est["target_uv"], dtype=float)
        for ax in (ax_surface, ax_dist, ax_dir, ax_query_tensor, ax_l1, ax_l2):
            ax.scatter([t[0]], [t[1]], marker="*", s=125, c="#d02531",
                       edgecolors="white", linewidths=.6, zorder=17)

    l1_q = lam_max[qi]
    l2_q = lam_min[qi]
    eig_status = (
        f"query lambda_max/min={l1_q:.4f}/{l2_q:.4f}"
        if np.isfinite(l1_q) and np.isfinite(l2_q)
        else "query tensor undefined"
    )
    ax_l2.text(0.02, 0.02, eig_status, transform=ax_l2.transAxes,
               fontsize=9, va="bottom", ha="left")

    loc_text = ""
    if localization is not None:
        ver = localization.get("verification", {})
        verdict = "ACCEPT" if ver.get("accepted", False) else "REJECT"
        loc_text = (f" | localizer={localization.get('status', '?')} "
                    f"({localization.get('iterations', 0)} iter) | {verdict}")
    fig.suptitle(
        f"r_boundary={cfg.boundary_radius_mm:g} mm | epsilon={cfg.boundary_epsilon_mm:g} mm | "
        f"threshold={result['threshold_mm']:g} mm | tensor r={corner['radius_mm']:g} mm{loc_text}\n"
        "Blue path = iterative query trajectory | red star = current corner target",
        fontsize=11,
    )

    return fig

def figure_to_rgb(fig):

    """Render into an owned RGB image, independent of the figure lifetime."""

    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())

    return rgba[:, :, :3].copy()





def open_tangent_view(ctx, result, previous=None, localization=None):

    if previous is not None:

        previous.destroy_window()

    vis = o3d.visualization.Visualizer()

    if not vis.create_window(window_name="3D: surface weights / query / tangent / boundary", width=950, height=750):

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

    opt = vis.get_render_option()

    opt.background_color = np.array([.045, .05, .065])

    opt.point_size = 4.0

    ctrl = vis.get_view_control()

    ctrl.set_lookat(q)

    ctrl.set_front(normalize(n + .4 * u + .2 * v))

    ctrl.set_up(v)

    ctrl.set_zoom(.8)

    return vis





def inspect_query(pixel_x, pixel_y, frozen, cfg, estimator):

    clicked = pixel_to_point_mm(pixel_x, pixel_y, frozen["depth_m"], frozen["intr"])

    if clicked is None:

        print(f"[query] Invalid depth at ({pixel_x}, {pixel_y})")

        return

    # Downsample once per frozen frame; clicking only changes the query.

    if "points_voxel_mm" not in frozen:

        frozen["points_voxel_mm"] = voxel_downsample_3d(frozen["points_raw_mm"], cfg.voxel_3d_mm)

    points = frozen["points_voxel_mm"]

    if len(points) < cfg.min_neighbors:

        print("[query] Too few downsampled 3-D points.")

        return

    click_distance, qi = cKDTree(points).query(clicked)

    print("[query] Iterative corner localization: query -> tangent -> corner -> move -> repeat...")

    try:

        ctx, result, localization = localize_corner_iterative(points, int(qi), cfg, estimator)

    except Exception as exc:

        print(f"[query] {type(exc).__name__}: {exc}")

        return

    frozen["selected_pixel"] = (pixel_x, pixel_y)

    frozen["context"], frozen["boundary"] = ctx, result
    frozen["corner_localization"] = localization

    fig = make_boundary_figure(ctx, result, localization)

    try:

        frozen["boundary_image"] = figure_to_rgb(fig)

    finally:

        fig.clear()

    if cfg.show_3d:

        frozen["tangent_view"] = open_tangent_view(ctx, result, frozen.get("tangent_view"), localization)

    qi_uv = ctx["query_uv_index"]

    d = result["distance_mm"][qi_uv]

    print(f"  raw / 3D voxel: {len(frozen['points_raw_mm']):,} / {len(points):,}")

    print(f"  clicked -> initial query: {click_distance:.3f} mm; final SAH normal: {np.round(ctx['normal'], 6)}")
    print(f"  localizer: {localization['status']} after {localization['iterations']} iteration(s)")
    ver = localization["verification"]
    verdict = "ACCEPT CORNER" if ver["accepted"] else "REJECT"
    print(
        f"  verifier: {verdict} | lambda_min={ver['lambda_min']:.6f} "
        f"(thr={ver['lambda_min_threshold']:.6f}) | "
        f"balance={ver['balance']:.3f} (thr={ver['balance_threshold']:.3f}) | "
        f"reason={ver['reason']}"
    )
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

    if result["query_boundary_uv"] is not None:

        b = result["query_boundary_uv"]

        b3 = ctx["query_world"] + b[0] * ctx["u"] + b[1] * ctx["v"]

        print(f"  local boundary estimate UV: {np.round(b, 4)}; tangent-lifted XYZ: {np.round(b3, 4)}")

    else:

        print("  No local boundary location displayed (direction undefined or d_hat >= r/2).")





def close_query_views(frozen):

    vis = frozen.pop("tangent_view", None)

    if vis is not None:

        vis.destroy_window()

    frozen.pop("boundary_image", None)

    frozen.pop("pending_query", None)





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

    root.title(f"{WINDOW_FROZEN} | {BUILD}")

    instructions = "Left click: query | R: recapture | Q/Esc: quit"

    status = tk.StringVar(master=root, value=instructions)

    tk.Label(root, textvariable=status, anchor="w").pack(fill="x")

    frame_view = TkImagePanel(root, draw_frozen(frozen))

    set_image_window_size(root, frozen["rgb"], extra_height=35)



    def request_action(value):

        nonlocal action

        action = value



    def key_cb(event):

        key = event.keysym.lower()

        if key in ("q", "escape", "r"):

            request_action("recapture" if key == "r" else "quit")



    def mouse_cb(event):

        point = frame_view.image_pixel(event.x, event.y)

        if point is not None:

            # Compute after root.update() returns, outside the Tcl callback.

            frozen["pending_query"] = point



    def close_boundary():

        nonlocal boundary_window

        if boundary_window is not None:

            boundary_window.destroy()

            boundary_window = None



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

            pending = frozen.pop("pending_query", None)

            if pending is not None:

                status.set("Iterative corner localization: tangent / response / move / retangent...")

                root.update_idletasks()

                previous_image = frozen.get("boundary_image")

                inspect_query(pending[0], pending[1], frozen, cfg, estimator)

                frame_view.set_rgb(draw_frozen(frozen))

                result_image = frozen.get("boundary_image")

                if result_image is not None and result_image is not previous_image:

                    if boundary_window is None:

                        boundary_window = tk.Toplevel(root)

                        boundary_window.title(WINDOW_BOUNDARY)

                        boundary_window.protocol("WM_DELETE_WINDOW", close_boundary)

                        boundary_view = TkImagePanel(boundary_window, result_image)

                        set_image_window_size(boundary_window, result_image)

                    else:

                        boundary_view.set_rgb(result_image)

                    boundary_window.deiconify()

                    boundary_window.lift()

                status.set(instructions)

            vis = frozen.get("tangent_view")

            if vis is not None:

                if vis.poll_events():

                    vis.update_renderer()

                else:

                    vis.destroy_window()

                    frozen["tangent_view"] = None

            time.sleep(.01)

    finally:

        close_query_views(frozen)

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
    parser = argparse.ArgumentParser(
        description=(
            "RealSense -> global SAH -> fixed-count FPS -> Hough Top4 -> soft 3D EXT 1024 -> NMS -> optional LS"
        )
    )
    parser.add_argument("--version", action="version", version=f"{BUILD} | {Path(__file__).resolve()}")
    parser.add_argument("--iteration", "--iter", action=argparse.BooleanOptionalAction,
                        default=False, help="Tensor query relocation, default OFF")
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction,
                        default=True, help="3D least-squares refinement after NMS, default ON")
    parser.add_argument("--normal-module", help="Path to updated surface_aware_harris3d.py")

    # RealSense stream / physical depth range.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)

    # Base scales. 3-D quantities use voxel_3d_mm; tangent/UV quantities use voxel_2d_mm.
    parser.add_argument("--voxel-3d-mm", "--voxel-mm", dest="voxel_3d_mm",
                        type=float, default=2.0)
    parser.add_argument("--voxel-2d-mm", "--uv-voxel-mm", dest="voxel_2d_mm",
                        type=float, default=2.0)

    # 3-D scales.
    parser.add_argument("--normal-radius-x", "--normal-radius-vox", dest="normal_radius_x",
                        type=float, default=4.0)
    parser.add_argument("--normal-sigma-r-x", type=float, default=0.5)

    # 2-D tangent/UV scales. Defaults match the compact setup used in the latest tests.
    parser.add_argument("--query-radius-x", "--query-radius-vox", dest="query_radius_x",
                        type=float, default=10.0)
    parser.add_argument("--boundary-radius-x", "--boundary-radius-vox", dest="boundary_radius_x",
                        type=float, default=4.0)
    parser.add_argument("--boundary-epsilon-x", "--boundary-epsilon-vox", dest="boundary_epsilon_x",
                        type=float, default=1.0)
    parser.add_argument("--tensor-radius-x", type=float, default=4.0)
    parser.add_argument("--tensor-sigma-d-x", type=float, default=1.5)
    parser.add_argument("--vote-line-sigma-x", "--vote-sigma-x", dest="vote_line_sigma_x",
                        type=float, default=1.0)
    parser.add_argument("--ext-radius-x", type=float, default=4.0)
    parser.add_argument("--ext-epsilon-mm", type=float, default=0.5)
    parser.add_argument("--ext-sigma-mm", type=float, default=0.5)
    parser.add_argument("--ext-directions", type=int, default=1024)
    parser.add_argument("--ext-min-score", type=float, default=0.08)

    # Iterative corner localizer, 2-D voxel multiples.
    parser.add_argument("--corner-search-radius-x", type=float, default=6.0)
    parser.add_argument("--corner-step-max-x", type=float, default=4.0)
    parser.add_argument("--corner-refine-radius-x", type=float, default=2.0)
    parser.add_argument("--corner-converge-x", type=float, default=1.0)
    parser.add_argument("--corner-peak-ratio", type=float, default=0.70)
    parser.add_argument("--corner-max-iterations", type=int, default=10)

    # Final FP rejection at the converged location.
    parser.add_argument("--corner-lambda-min-threshold", type=float, default=0.03)
    parser.add_argument("--corner-balance-threshold", type=float, default=0.15)

    # Two coherent boundary-direction modes. Spatial Gaussian remains OFF.
    parser.add_argument("--two-mode-min-coherence", type=float, default=0.80)
    parser.add_argument("--two-mode-min-fraction", type=float, default=0.20)
    parser.add_argument("--two-mode-min-angle-deg", type=float, default=35.0)
    parser.add_argument("--two-mode-max-angle-deg", type=float, default=150.0)

    # Scene-level FPS / verified Hough peak NMS.
    # FPS is now controlled ONLY by the requested number of query points.
    parser.add_argument(
        "--fps-count", "--fps-max-queries", dest="fps_count",
        type=int, default=64,
        help="Exact number of FPS query points. Uses min(count, cloud size). Default: 64",
    )
    # Backward-compatibility only: accepted but ignored.
    parser.add_argument(
        "--fps-spacing-x", type=float, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--corner-nms-radius-x", "--corner-consensus-radius-x",
        dest="corner_nms_radius_x", type=float, default=3.0,
        help="3-D NMS radius in 3D voxels. Default 3",
    )
    parser.add_argument(
        "--corner-consensus-min-support", type=int, default=2,
        help="Legacy option; ignored because the new detector uses 3D NMS.",
    )
    parser.add_argument(
        "--capture-dir", type=str, default="realsense_captures",
        help="Directory where P-key RGB/depth/camera-info captures are stored.",
    )

    # Dimensionless/angular parameters.
    parser.add_argument("--normal-sigma-e-deg", type=float, default=15.0)
    parser.add_argument("--normal-kernel", choices=("cauchy", "gaussian"), default="cauchy")
    parser.add_argument("--normal-max-iterations", type=int, default=5)
    parser.add_argument("--normal-update-mode", choices=("fixed", "synchronous"), default="synchronous")
    parser.add_argument("--surface-weight-min", type=float, default=0.1)
    parser.add_argument("--min-neighbors", type=int, default=6)
    parser.add_argument("--no-3d-view", action="store_true")
    return parser.parse_args(argv)


def configure(args):
    global WIDTH, HEIGHT, MIN_DEPTH_M, MAX_DEPTH_M, VOXEL_3D_MM, VOXEL_2D_MM

    numeric = [v for v in vars(args).values() if isinstance(v, (int, float))]
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric arguments must be finite.")
    if args.width < 64 or args.height < 64 or args.fps < 1:
        raise ValueError("Require width/height >=64 and fps >=1.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("Require 0 < min-depth < max-depth.")

    positive = [
        args.voxel_3d_mm, args.voxel_2d_mm,
        args.normal_radius_x, args.normal_sigma_r_x,
        args.query_radius_x, args.boundary_radius_x, args.boundary_epsilon_x,
        args.tensor_radius_x, args.tensor_sigma_d_x,
        args.vote_line_sigma_x, args.ext_radius_x, args.ext_sigma_mm,
        args.corner_search_radius_x, args.corner_step_max_x,
        args.corner_refine_radius_x, args.corner_converge_x,
        args.corner_nms_radius_x,
        args.normal_sigma_e_deg,
    ]
    if min(positive) <= 0:
        raise ValueError("Voxel sizes, spatial multipliers, sigma, and epsilon must be positive.")
    if args.normal_max_iterations < 1 or args.corner_max_iterations < 1:
        raise ValueError("normal/corner iterations must be >=1.")
    if args.min_neighbors < 3:
        raise ValueError("min-neighbors must be >=3.")
    if args.ext_directions < 16 or args.ext_directions % 2:
        raise ValueError("--ext-directions must be even and >=16")
    if args.ext_epsilon_mm < 0 or not 0 <= args.ext_min_score <= 1:
        raise ValueError("Require ext-epsilon-mm >=0 and ext-min-score in [0,1]")
    if args.fps_count < 1:
        raise ValueError("fps-count must be >=1.")
    if args.corner_consensus_min_support < 1:
        raise ValueError("corner-consensus-min-support must be >=1.")
    if not 0.0 < args.corner_peak_ratio <= 1.0:
        raise ValueError("corner-peak-ratio must be in (0,1].")
    if args.corner_lambda_min_threshold < 0.0:
        raise ValueError("corner-lambda-min-threshold must be >=0.")
    if not 0.0 <= args.corner_balance_threshold <= 0.5:
        raise ValueError("corner-balance-threshold must be in [0,0.5].")
    if not 0.0 <= args.two_mode_min_coherence <= 1.0:
        raise ValueError("two-mode-min-coherence must be in [0,1].")
    if not 0.0 < args.two_mode_min_fraction <= 0.5:
        raise ValueError("two-mode-min-fraction must be in (0,0.5].")
    if not 0.0 <= args.two_mode_min_angle_deg < args.two_mode_max_angle_deg <= 180.0:
        raise ValueError("Require 0 <= two-mode-min-angle-deg < two-mode-max-angle-deg <= 180.")
    if not 0.0 <= args.surface_weight_min <= 1.0:
        raise ValueError("surface-weight-min must be in [0,1].")
    if args.iteration and args.corner_search_radius_x + args.tensor_radius_x > args.query_radius_x + 1e-12:
        raise ValueError(
            "Require corner-search-radius-x + tensor-radius-x <= query-radius-x "
            "to keep full tensor support inside the query ROI."
        )
    if 1.5 * args.boundary_epsilon_x >= 0.5 * args.boundary_radius_x:
        raise ValueError("Use boundary-radius-x > 3*boundary-epsilon-x.")

    WIDTH, HEIGHT = int(args.width), int(args.height)
    MIN_DEPTH_M, MAX_DEPTH_M = float(args.min_depth_m), float(args.max_depth_m)
    VOXEL_3D_MM, VOXEL_2D_MM = float(args.voxel_3d_mm), float(args.voxel_2d_mm)
    v3, v2 = VOXEL_3D_MM, VOXEL_2D_MM

    cfg = BoundaryConfig(
        voxel_3d_mm=v3,
        voxel_2d_mm=v2,
        normal_radius_mm=args.normal_radius_x * v3,
        sigma_r_mm=args.normal_sigma_r_x * v3,
        query_radius_mm=args.query_radius_x * v2,
        boundary_radius_mm=args.boundary_radius_x * v2,
        boundary_epsilon_mm=args.boundary_epsilon_x * v2,
        tensor_radius_mm=args.tensor_radius_x * v2,
        tensor_sigma_d_mm=args.tensor_sigma_d_x * v2,
        vote_line_sigma_mm=args.vote_line_sigma_x * v2,
        ext_radius_mm=args.ext_radius_x * v3,
        ext_epsilon_mm=args.ext_epsilon_mm,
        ext_sigma_mm=args.ext_sigma_mm,
        ext_directions=args.ext_directions,
        ext_min_score=args.ext_min_score,
        enable_iteration=bool(args.iteration),
        enable_refine=bool(args.refine),
        corner_nms_radius_mm=args.corner_nms_radius_x * v3,
        corner_search_radius_mm=args.corner_search_radius_x * v2,
        corner_step_max_mm=args.corner_step_max_x * v2,
        corner_refine_radius_mm=args.corner_refine_radius_x * v2,
        corner_converge_mm=args.corner_converge_x * v2,
        corner_peak_ratio=args.corner_peak_ratio,
        corner_max_iterations=args.corner_max_iterations,
        corner_lambda_min_threshold=args.corner_lambda_min_threshold,
        corner_balance_threshold=args.corner_balance_threshold,
        two_mode_min_coherence=args.two_mode_min_coherence,
        two_mode_min_fraction=args.two_mode_min_fraction,
        two_mode_min_angle_deg=args.two_mode_min_angle_deg,
        two_mode_max_angle_deg=args.two_mode_max_angle_deg,
        fps_count=int(args.fps_count),
        corner_consensus_radius_mm=args.corner_nms_radius_x * v3,
        corner_consensus_min_support=int(args.corner_consensus_min_support),
        sigma_e_deg=args.normal_sigma_e_deg,
        normal_kernel=args.normal_kernel,
        normal_iterations=args.normal_max_iterations,
        normal_update_mode=args.normal_update_mode,
        surface_weight_min=args.surface_weight_min,
        min_neighbors=args.min_neighbors,
        show_3d=not args.no_3d_view,
    )
    return cfg




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
    if len(candidate_ids):
        support[candidate_ids] = 0.0
        response[candidate_ids] = 0.0
    output = {
        "source_indices": voter_ids, "candidate_indices": candidate_ids,
        "support": support, "response": response,
        "peaks": [], "vote_max": 0.0, "support_max": 0.0,
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

    finite_response = np.maximum(np.nan_to_num(response[candidate_ids], nan=0.0), 0.0)
    output["vote_max"] = float(np.max(finite_response))
    output["support_max"] = float(np.nanmax(support[candidate_ids]))
    if output["vote_max"] <= 1e-12:
        return output

    # Peak NMS over observed positions. Peak MUST be an observed sample.
    # Relative 20% filter picks active FPS peaks, not a vote-construction filter.
    pick = candidate_ids[finite_response >= 0.20*output["vote_max"]]
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


def verify_hough_peaks_extremality_3d(ctx, result, points3d, point_tree=None, sphere_directions=None):
    """Soft 3-D extremality at observed Hough peaks (all neighbors projected).

    For each candidate C and spherical direction u, ALL pre-filter 3-D voxel
    neighbors P_j within 3-D radius r are projected onto u:
        m(u) = max_j [u dot (P_j-C)]
        s(u) = exp(-0.5 * (max(0,m(u))/sigma)**2)
        E(C) = mean_u s(u).

    The original hard fraction using epsilon is retained for DEBUG comparison
    only. The active direction-tensor localizer and its verifier are unchanged.
    """
    cfg = ctx["config"]
    scene = np.asarray(points3d, dtype=np.float64)
    tree = point_tree if point_tree is not None else cKDTree(scene)
    directions = (sphere_directions if sphere_directions is not None
                  else fibonacci_sphere_directions(cfg.ext_directions))
    peaks = result["hough_vote"]["peaks"]
    records = []
    for rank, peak in enumerate(peaks, 1):
        uv_idx = int(peak["index"])
        scene_id = int(ctx["support_indices"][uv_idx])
        center = scene[scene_id]
        neighbor_ids = np.asarray(tree.query_ball_point(center, cfg.ext_radius_mm), dtype=np.int64)
        delta = scene[neighbor_ids] - center
        enough = len(neighbor_ids) >= cfg.min_neighbors
        # Orthogonal projection onto EVERY tested direction, not a narrow tube
        # around the line. A 3D neighbor is counted irrespective of its face.
        margins = np.max(delta @ directions.T, axis=0) if len(delta) else np.full(len(directions), np.nan)
        if enough:
            soft = np.exp(-0.5 * (np.maximum(0.0, margins) / cfg.ext_sigma_mm)**2)
            hard = margins <= cfg.ext_epsilon_mm  # Compare against old binary statistic.
            score = float(np.mean(soft))
            hard_score = float(np.mean(hard))
        else:
            soft = np.full(len(directions), np.nan)
            hard = np.zeros(len(directions), dtype=bool)
            score = float("nan")
            hard_score = float("nan")
        record = {
            "rank": rank, "uv_index": uv_idx, "scene_index": scene_id,
            "center": center.copy(), "hough_score": float(peak["value"]),
            "neighbor_indices": neighbor_ids, "neighbor_count": int(len(neighbor_ids)),
            "directions": directions, "max_projection_mm": margins,
            "direction_scores": soft, "extreme_mask": hard,
            "score": score, "hard_score": hard_score,
            "valid": enough, "pass": bool(enough and score >= cfg.ext_min_score),
            "radius_mm": float(cfg.ext_radius_mm),
            "epsilon_mm": float(cfg.ext_epsilon_mm),
            "sigma_mm": float(cfg.ext_sigma_mm),
            "threshold": float(cfg.ext_min_score),
        }
        records.append(record)
        peak["extremality_3d"] = record
    result["extremality_cloud_3d"] = scene
    return records


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


def nms_corner_candidates(candidates, radius_mm):
    """Greedy 3-D NMS; retain the strongest verified localization per corner."""
    if not candidates:
        return []
    radius_mm = float(radius_mm)
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    for cand in ordered:
        p = np.asarray(cand["position"], dtype=float)
        if all(np.linalg.norm(p - np.asarray(k["position"], dtype=float)) > radius_mm
               for k in kept):
            kept.append(cand)
    return kept


def farthest_point_sampling_count(points_mm, count):
    """Deterministic FPS with an explicit requested number of seeds.

    Exactly K=min(count, N) observed cloud points are returned.
    The first seed is the observed point closest to the cloud centroid.
    Every following seed is the point farthest from its nearest selected seed.

    Returns
    -------
    seed_indices : (K,) int64
    cover_radius_mm : float
        Maximum distance of any cloud point to its nearest selected seed.
        This is diagnostic only; it does NOT stop FPS early.
    """
    pts = np.asarray(points_mm, dtype=float)
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.inf

    k_target = min(max(1, int(count)), n)

    centroid = np.mean(pts, axis=0)
    first = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))

    selected = np.empty(k_target, dtype=np.int64)
    selected[0] = first

    min_d2 = np.sum((pts - pts[first]) ** 2, axis=1)
    min_d2[first] = 0.0

    for k in range(1, k_target):
        farthest = int(np.argmax(min_d2))
        selected[k] = farthest
        d2 = np.sum((pts - pts[farthest]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[:k + 1]] = 0.0

    final_cover = float(np.sqrt(np.max(min_d2))) if n else np.inf
    return selected, final_cover


def cluster_corner_consensus(candidates, radius_mm, min_support):
    """Radius-connect verifier-passing localizations and require multi-seed support.

    Each retained cluster represents an attractor reached by multiple distinct
    FPS queries. The reported location is a score-weighted centroid. No NMS is
    applied afterward; the cluster itself performs duplicate consolidation.
    """
    if not candidates:
        return [], []

    radius_mm = float(radius_mm)
    min_support = max(1, int(min_support))
    pts = np.asarray([c["position"] for c in candidates], dtype=float)
    n = len(pts)

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    tree = cKDTree(pts)
    for a, b in tree.query_pairs(radius_mm):
        union(int(a), int(b))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    all_clusters = []
    retained = []
    for member_ids in groups.values():
        members = [candidates[i] for i in member_ids]
        pos = pts[np.asarray(member_ids, dtype=np.int64)]
        raw_score = np.asarray([max(float(m.get("score", 0.0)), 0.0) for m in members], dtype=float)
        if float(np.sum(raw_score)) <= np.finfo(float).tiny:
            weights = np.ones(len(members), dtype=float)
        else:
            weights = raw_score
        centroid = np.average(pos, axis=0, weights=weights)
        distances = np.linalg.norm(pos - centroid[None, :], axis=1)
        best_local = int(np.argmax(raw_score)) if len(raw_score) else 0
        representative = members[best_local]

        cluster = dict(representative)
        cluster.update({
            "position": np.asarray(centroid, dtype=float),
            "support_count": int(len(members)),
            "member_seed_indices": [int(m["seed_index"]) for m in members],
            "member_final_query_indices": [int(m["final_query_index"]) for m in members],
            "cluster_mean_radius_mm": float(np.mean(distances)) if len(distances) else 0.0,
            "cluster_max_radius_mm": float(np.max(distances)) if len(distances) else 0.0,
            "cluster_score": float(np.sum(raw_score)),
            "members": members,
            "consensus_passed": bool(len(members) >= min_support),
        })
        all_clusters.append(cluster)
        if cluster["consensus_passed"]:
            retained.append(cluster)

    retained.sort(key=lambda c: (c["support_count"], c["cluster_score"]), reverse=True)
    all_clusters.sort(key=lambda c: (c["support_count"], c["cluster_score"]), reverse=True)
    return retained, all_clusters


def detect_corners_from_fps(points_mm, cfg, estimator, scene_cache=None):
    """FPS -> tangent/UV -> Hough peaks -> 3-D soft extremality -> NMS -> optional LS.

    Iteration OFF evaluates each FPS seed once (no move, no retangent).
    Iteration ON uses the existing tensor-based retangent localizer FIRST,
    then checks Hough peaks in its final frame. Multiple Hough peaks can be
    accepted per seed, and 3-D NMS deduplicates them across the entire scene.
    """
    points = np.asarray(points_mm, dtype=float)
    if scene_cache is None:
        scene_cache = build_scene_normal_cache(points, cfg, estimator)
    # Fixed-number deterministic FPS, sampled only from valid global SAH points.
    valid_ids = np.flatnonzero(scene_cache.normal_valid)
    if len(valid_ids) == 0:
        raise RuntimeError("No valid cached SAH normals for FPS")
    local_seeds, cover_mm = farthest_point_sampling_count(points[valid_ids], cfg.fps_count)
    seed_ids = valid_ids[local_seeds]
    tree = scene_cache.tree
    accepted = []
    records = []

    print(f"[FPS] seeds={len(seed_ids)} / requested={cfg.fps_count} | "
          f"cover={cover_mm:.3f} mm | iteration={'ON' if cfg.enable_iteration else 'OFF'} "
          f"| refine={'ON' if cfg.enable_refine else 'OFF'}", flush=True)

    for number, seed_id in enumerate(seed_ids, 1):
        try:
            if cfg.enable_iteration:
                # Old tensor localizer is ONLY active when explicitly enabled.
                ctx, result, loc = localize_corner_iterative(
                    points, int(seed_id), cfg, estimator, scene_cache)
                # Use the same computed final Hough peaks; no re-evaluation.
                query_id = int(loc['final_query_index'])
                iterations = int(loc['iterations'])
                status = loc['status']
            else:
                # Exactly ONE tangent context per FPS seed. Query never moves.
                ctx = build_tangent_context(points, int(seed_id), cfg, estimator, scene_cache)
                result = estimate_context_boundary(ctx, include_tensor=False)
                result['extremality_3d'] = verify_hough_peaks_extremality_3d(
                    ctx, result, points, tree, sphere_directions=scene_cache.directions)
                query_id = int(seed_id)
                iterations = 1
                status = 'single_pass'

            peaks = result['hough_vote']['peaks']
            ext = result.get('extremality_3d', [])
            if len(peaks) != len(ext):
                raise RuntimeError('Hough peak / extremality alignment mismatch')
            accepted_here = 0
            for entry in ext:
                if not entry['pass']:
                    continue
                rank = int(entry['rank'])
                # Both the Hough peak and the verifier center correspond to the
                # SAME original 3-D observation. Do not substitute FPS query.
                scene_id = int(entry['scene_index'])
                position = points[scene_id].copy()
                candidate = {
                    'seed_index': int(seed_id),
                    'seed_position': points[int(seed_id)].copy(),
                    'query_index': query_id,
                    'final_query_index': scene_id,
                    'hough_rank': rank,
                    'status': status,
                    'iterations': iterations,
                    'coarse_position': position.copy(),
                    'position': position.copy(),
                    'hough_score': float(entry['hough_score']),
                    'extremality_score': float(entry['score']),
                    # Rank by original Hough response. E is an independent gate;
                    # do not silently multiply scores with different meaning.
                    'score': float(entry['hough_score']),
                    'verification': {
                        'accepted': True, 'reason': 'hough_and_soft_3d_extremality',
                        'lambda_min': float(entry['hough_score']),
                        'balance': float('nan'),
                        'extremality': float(entry['score']),
                    },
                }
                accepted.append(candidate)
                accepted_here += 1

            records.append({
                'seed_index': int(seed_id),
                'seed_position': points[int(seed_id)].copy(),
                'final_query_index': query_id,
                'iterations': iterations, 'status': status,
                'hough_peaks': len(peaks), 'accepted_peaks': accepted_here,
                'peak_scores': [(int(e['rank']), float(e['hough_score']),
                                 float(e['score']), bool(e['pass'])) for e in ext],
            })
            print(f"  [{number:03d}/{len(seed_ids):03d}] "
                  f"Hough={len(peaks):d} accepted={accepted_here:d} "
                  f"iter={iterations:d} status={status}", flush=True)
        except Exception as exc:
            records.append({
                'seed_index': int(seed_id),
                'seed_position': points[int(seed_id)].copy(),
                'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
            })
            print(f"  [{number:03d}/{len(seed_ids):03d}] ERROR: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Use full-scene 3-D duplicate suppression for ALL passing peaks,
    # not just the best peak from each FPS seed.
    corners = nms_corner_candidates(accepted, cfg.corner_nms_radius_mm)
    print(f"[NMS] accepted Hough peaks={len(accepted)} -> unique corners={len(corners)}", flush=True)

    if cfg.enable_refine:
        # Only survivors are refined. Crucial: rebuild tangent AT THE HOUGH
        # PEAK, not at the original FPS query; otherwise LS shifts the wrong center.
        for cand in corners:
            try:
                scene_id = int(cand['final_query_index'])
                ctx_c = build_tangent_context(points, scene_id, cfg, estimator, scene_cache)
                boundary_c = estimate_context_boundary(ctx_c)
                refinement = refine_corner_3d_subpix(
                    points, ctx_c, boundary_c, cfg, estimator, scene_cache)
                cand['refinement_3d'] = refinement
                if refinement['success']:
                    cand['position'] = np.asarray(refinement['position'], dtype=float).copy()
                    print(f"  [REFINE] FPS={cand['seed_index']} "
                          f"peak={cand['hough_rank']} "
                          f"shift={refinement['shift_mm']:.3f} mm "
                          f"RMSE={refinement['rmse_mm']:.3f} mm", flush=True)
                else:
                    print(f"  [REFINE] FPS={cand['seed_index']} "
                          f"fallback: {refinement['reason']}", flush=True)
            except Exception as exc:
                print(f"  [REFINE] FPS={cand['seed_index']} "
                      f"error: {type(exc).__name__}: {exc}; keep coarse", flush=True)
    return {
        'seed_indices': seed_ids,
        'cover_radius_mm': cover_mm,
        'records': records,
        'accepted_before_nms': accepted,
        'accepted_before_consensus': accepted,  # compatible with existing RealSense overlay
        'consensus_clusters': [],  # legacy output field; no consensus gate
        'corners': corners,
    }


def open_fps_detection_view(points_mm, detection, cfg):
    """3-D scene view: cloud + FPS seeds + verified candidates + NMS corners."""
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name="FPS corner detector | close window to recapture", width=1100, height=800):
        print("[3D] Could not open FPS result viewer.")
        return

    pts = np.asarray(points_mm, dtype=float)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.paint_uniform_color([0.63, 0.65, 0.68])
    vis.add_geometry(cloud)

    seed_ids = np.asarray(detection["seed_indices"], dtype=np.int64)
    if len(seed_ids):
        seeds = o3d.geometry.PointCloud()
        seeds.points = o3d.utility.Vector3dVector(pts[seed_ids])
        seeds.paint_uniform_color([0.15, 0.55, 0.95])
        vis.add_geometry(seeds)

    # All Hough+EXT passing candidates before 3-D NMS.
    for cand in detection["accepted_before_nms"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.45, 0.45 * cfg.voxel_3d_mm))
        marker.translate(cand["position"])
        marker.paint_uniform_color([1.0, 0.65, 0.1])
        vis.add_geometry(marker)

    # Final detections after global 3-D NMS.
    for cand in detection["corners"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.8, 0.85 * cfg.voxel_3d_mm))
        marker.translate(cand["position"])
        marker.paint_uniform_color([1.0, 0.12, 0.12])
        vis.add_geometry(marker)

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([0.035, 0.04, 0.05])
        opt.point_size = 3.0
    ctrl = vis.get_view_control()
    if len(pts):
        ctrl.set_lookat(np.mean(pts, axis=0))
        ctrl.set_front([0.6, -0.7, -0.35])
        ctrl.set_up([0.0, 0.0, 1.0])
        ctrl.set_zoom(0.72)
    vis.run()
    vis.destroy_window()



def _intrinsics_dict(intr):
    coeffs = list(getattr(intr, "coeffs", []))
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(getattr(intr, "model", "unknown")),
        "coeffs": [float(x) for x in coeffs],
    }


def build_camera_metadata(profile, depth_scale, width, height, fps):
    """Static RealSense calibration/stream information saved with each P capture."""
    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = color_vsp.get_intrinsics()
    depth_intr = depth_vsp.get_intrinsics()
    extr = depth_vsp.get_extrinsics_to(color_vsp)

    device = profile.get_device()
    device_info = {}
    for label, key in (
        ("name", rs.camera_info.name),
        ("serial_number", rs.camera_info.serial_number),
        ("firmware_version", rs.camera_info.firmware_version),
        ("product_line", rs.camera_info.product_line),
    ):
        try:
            if device.supports(key):
                device_info[label] = device.get_info(key)
        except Exception:
            pass

    return {
        "device": device_info,
        "stream": {"width": int(width), "height": int(height), "fps": int(fps)},
        "depth_scale_m_per_unit": float(depth_scale),
        "color_intrinsics": _intrinsics_dict(color_intr),
        "depth_intrinsics_before_alignment": _intrinsics_dict(depth_intr),
        "depth_to_color_extrinsics": {
            "rotation_row_major": [float(x) for x in extr.rotation],
            "translation_m": [float(x) for x in extr.translation],
        },
        "saved_depth_is_aligned_to": "color",
    }


def save_rgbd_capture(latest, base_dir, camera_metadata):
    """Save the exact aligned RGB-D frame used when P is pressed."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder = Path(base_dir).expanduser() / stamp
    folder.mkdir(parents=True, exist_ok=False)

    rgb = np.asarray(latest["rgb"], dtype=np.uint8)
    depth_m = np.asarray(latest["depth_m"], dtype=np.float32)

    # rgb is BGR because the RealSense color stream is configured as bgr8.
    if not cv2.imwrite(str(folder / "rgb.png"), rgb):
        raise RuntimeError("Failed to save rgb.png")

    # Exact aligned Z16 depth frame, when available. Combined with
    # depth_scale_m_per_unit in camera_info.json this preserves original units.
    depth_raw = latest.get("depth_raw")
    if depth_raw is not None:
        depth_raw = np.asarray(depth_raw, dtype=np.uint16)
        cv2.imwrite(str(folder / "depth_raw_z16.png"), depth_raw)

    # Exact floating aligned depth used by the detector.
    np.save(folder / "depth_m.npy", depth_m)

    # Convenient 16-bit millimeter depth image. Invalid/out-of-range values are 0.
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm_float = np.rint(depth_m[valid] * 1000.0)
    depth_mm[valid] = np.clip(depth_mm_float, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(str(folder / "depth_mm.png"), depth_mm):
        raise RuntimeError("Failed to save depth_mm.png")

    depth_vis = make_realsense_depth_vis(depth_m)
    cv2.imwrite(str(folder / "depth_vis.png"), depth_vis)

    metadata = dict(camera_metadata)
    metadata.update({
        "capture_local_time": datetime.now().isoformat(timespec="milliseconds"),
        "depth_storage": {
            "depth_raw_z16.png": "aligned uint16 RealSense depth units; use depth_scale_m_per_unit",
            "depth_m.npy": "float32 meters; exact aligned detector input",
            "depth_mm.png": "uint16 millimeters; invalid=0",
            "depth_vis.png": "display-only colorized depth",
        },
        "rgb_storage": {"rgb.png": "BGR8 PNG from aligned color frame"},
        "aligned_intrinsics_used_by_detector": _intrinsics_dict(latest["intr"]),
        "valid_depth_range_m": [float(MIN_DEPTH_M), float(MAX_DEPTH_M)],
    })
    (folder / "camera_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[CAPTURE] saved RGB-D + camera info -> {folder}")
    return folder


def save_detection_summary(folder, detection, cfg):
    """Persist selected Hough/Extremality candidates and refined positions."""
    if folder is None:
        return
    folder = Path(folder)
    payload = {
        "build": BUILD,
        "fps_count_requested": int(cfg.fps_count),
        "fps_count_actual": int(len(detection.get("seed_indices", []))),
        "hough_top_k": HOUGH_TOP_K,
        "ext_directions": int(cfg.ext_directions),
        "ext_radius_mm": float(cfg.ext_radius_mm),
        "ext_sigma_mm": float(cfg.ext_sigma_mm),
        "ext_min_score": float(cfg.ext_min_score),
        "iteration_enabled": bool(cfg.enable_iteration),
        "refine_enabled": bool(cfg.enable_refine),
        "nms_radius_mm": float(cfg.corner_nms_radius_mm),
        "accepted_peak_count": int(len(detection.get("accepted_before_nms", []))),
        "corners": [],
    }
    for c in detection.get("corners", []):
        rf = c.get("refinement_3d", {})
        payload["corners"].append({
            "position_mm": np.asarray(c["position"], dtype=float).tolist(),
            "coarse_position_mm": np.asarray(c["coarse_position"], dtype=float).tolist(),
            "seed_index": int(c["seed_index"]),
            "hough_rank": int(c["hough_rank"]),
            "hough_score": float(c["hough_score"]),
            "extremality_score": float(c["extremality_score"]),
            "refine_success": bool(rf.get("success", False)),
            "refine_shift_mm": float(rf.get("shift_mm", 0.0)),
        })
    (folder / "detections.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

def main():
    args = parse_args()
    cfg = configure(args)

    if args.fps_spacing_x is not None:
        print("[FPS] NOTE: --fps-spacing-x is deprecated and ignored; use --fps-count.")
    if args.corner_consensus_min_support != 2:
        print("[NMS] NOTE: --corner-consensus-min-support is ignored by Hough+EXT detector.")

    print(f"[BUILD {BUILD}] {Path(__file__).resolve()}", flush=True)
    print("RealSense -> global SAH -> count FPS -> Hough Top4 -> EXT -> 3D NMS -> LS", flush=True)
    print(
        f"[3-D] voxel={cfg.voxel_3d_mm:g} mm | normal r={cfg.normal_radius_mm:g} mm | "
        f"FPS count={cfg.fps_count} | NMS r={cfg.corner_nms_radius_mm:g} mm"
    )
    print(
        f"[2-D] voxel={cfg.voxel_2d_mm:g} mm | query={cfg.query_radius_mm:g} mm | "
        f"boundary={cfg.boundary_radius_mm:g} mm | tensor={cfg.tensor_radius_mm:g} mm | "
        f"search={cfg.corner_search_radius_mm:g} mm"
    )
    print(f"[Hough/EXT] Top-{HOUGH_TOP_K} | directions={cfg.ext_directions} | "
          f"E>={cfg.ext_min_score:g} | iteration={cfg.enable_iteration} | refine={cfg.enable_refine}")
    print(
        "[optimization] KD-tree + SAH normals are cached once per frozen frame; "
        "F reuses the frozen normal field."
    )
    print("Controls: P=SAVE RGB-D+camera info + freeze+detect, F=rerun frozen, R=live, C=3D, D=reopen depth, Q/Esc=quit")

    if rs is None:
        raise RuntimeError("pyrealsense2 is required. Install Intel RealSense SDK Python bindings in this environment.")
    if o3d is None:
        raise RuntimeError("open3d is required for voxel downsampling and 3-D visualization.")

    estimator = load_normal_estimator(args.normal_module)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    rs_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[RealSense] depth scale={depth_scale:.9f} m/unit")
    camera_metadata = build_camera_metadata(profile, depth_scale, args.width, args.height, args.fps)
    print(f"[capture] P saves rgb.png + depth_m.npy + depth_mm.png + camera_info.json under {args.capture_dir}")

    window = "RealSense FPS corner detector | P freeze+detect | R live"
    depth_window = "RealSense depth | LIVE / FROZEN"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)

    state = {
        "latest": None,
        "frozen": None,
        "result_vis": None,
        "result_geometries": None,
    }

    def close_result_viewer():
        vis = state.get("result_vis")
        if vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass
        state["result_vis"] = None
        state["result_geometries"] = None

    def open_result_viewer():
        if state["frozen"] is None:
            return
        close_result_viewer()
        vis, geoms = create_realsense_fps_viewer(
            state["frozen"]["points_mm"],
            state["frozen"]["detection"],
            cfg,
        )
        state["result_vis"] = vis
        state["result_geometries"] = geoms

    def run_detection_on_latest(latest, reuse_frozen=None):
        print("\n" + "=" * 78)
        print("FREEZE -> 3-D VOXEL -> GLOBAL SAH -> COUNT FPS -> HOUGH -> 3D EXT -> NMS")
        print("=" * 78)

        if reuse_frozen is not None and "points_mm" in reuse_frozen:
            points_mm = np.asarray(reuse_frozen["points_mm"], dtype=np.float64)
            raw_count = int(reuse_frozen.get("raw_point_count", len(points_mm)))
            print("[cache] reusing frozen voxel cloud")
        else:
            points_raw_mm = depth_to_xyz_mm(latest["depth_m"], latest["intr"])
            if len(points_raw_mm) == 0:
                print("[P] no valid depth points")
                return None
            points_mm = voxel_downsample_3d(points_raw_mm, cfg.voxel_3d_mm)
            if len(points_mm) < cfg.min_neighbors:
                print("[P] too few points after voxel downsampling")
                return None
            raw_count = len(points_raw_mm)

        print(f"raw / voxel points : {raw_count:,} / {len(points_mm):,}")
        print(f"depth range        : {MIN_DEPTH_M:g} - {MAX_DEPTH_M:g} m")

        normal_time = 0.0
        if reuse_frozen is not None and reuse_frozen.get("scene_cache") is not None:
            scene_cache = reuse_frozen["scene_cache"]
            print("[cache] reusing frozen KD-tree + SAH normal field")
        else:
            scene_cache = precompute_scene_cache(points_mm, cfg, estimator)
            normal_time = scene_cache.normal_elapsed_s

        t0 = time.perf_counter()
        detection = detect_corners_from_fps(
            points_mm, cfg, estimator, scene_cache=scene_cache
        )
        detect_time = time.perf_counter() - t0
        elapsed = normal_time + detect_time
        print(
            f"[RESULT] seeds={len(detection['seed_indices'])} | "
            f"EXT-passed={len(detection['accepted_before_nms'])} | "
            f"final={len(detection['corners'])} | total={elapsed:.2f} s "
            f"(SAH={normal_time:.2f}s, FPS+localize={detect_time:.2f}s)"
        )
        for i, cand in enumerate(detection["corners"], start=1):
            print(
                f"  corner {i:02d}: XYZ={np.round(cand['position'], 3).tolist()} mm | "
                f"H={cand['hough_score']:.5f} | E={cand['extremality_score']:.4f} | "
                f"refined={bool(cand.get('refinement_3d', {}).get('success', False))}"
            )

        capture_dir = latest.get("capture_dir")
        if capture_dir is not None:
            save_detection_summary(capture_dir, detection, cfg)

        return {
            "rgb": latest["rgb"].copy(),
            "depth_m": latest["depth_m"].copy(),
            "depth_raw": None if latest.get("depth_raw") is None else latest["depth_raw"].copy(),
            "intr": latest["intr"],
            "capture_dir": capture_dir,
            "depth_vis": make_realsense_depth_vis(latest["depth_m"]),
            "points_mm": np.asarray(points_mm, dtype=np.float64),
            "detection": detection,
            "scene_cache": scene_cache,
            "raw_point_count": raw_count,
            "elapsed_s": elapsed,
        }

    try:
        while True:
            # Keep the Open3D result viewer responsive without blocking the RealSense UI.
            if state.get("result_vis") is not None:
                try:
                    alive = state["result_vis"].poll_events()
                    state["result_vis"].update_renderer()
                    if alive is False:
                        close_result_viewer()
                except Exception:
                    close_result_viewer()

            if state["frozen"] is None:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_raw = np.asanyarray(depth_frame.get_data())
                rgb = np.asanyarray(color_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * depth_scale
                intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                state["latest"] = {
                    "rgb": rgb.copy(),
                    "depth_raw": depth_raw.copy(),
                    "depth_m": depth_m.copy(),
                    "intr": intr,
                }

                # Keep the depth view live together with RGB.  P freezes the
                # same aligned RGB-D frame so detector input and visualization
                # stay synchronized.
                depth_show = make_realsense_depth_vis(depth_m)
                cv2.putText(
                    depth_show, "LIVE", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imshow(depth_window, depth_show)

                show = rgb.copy()
                status = "LIVE RGB-D | P: freeze + FPS detection"
            else:
                show = draw_realsense_detection_overlay(state["frozen"])
                d = state["frozen"]["detection"]
                status = (
                    f"FROZEN | seeds={len(d['seed_indices'])} | "
                    f"EXT={len(d['accepted_before_nms'])} | NMS={len(d['corners'])} | "
                    f"{state['frozen']['elapsed_s']:.1f}s"
                )
                frozen_depth = state["frozen"]["depth_vis"].copy()
                cv2.putText(
                    frozen_depth, "FROZEN", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imshow(depth_window, frozen_depth)

            cv2.putText(show, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(show, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('p'), ord('P')):
                if state["latest"] is None:
                    continue
                latest_for_detection = dict(state["latest"])
                try:
                    capture_dir = save_rgbd_capture(
                        latest_for_detection, args.capture_dir, camera_metadata
                    )
                    latest_for_detection["capture_dir"] = str(capture_dir)
                except Exception as exc:
                    print(f"[CAPTURE] save failed: {type(exc).__name__}: {exc}")
                    latest_for_detection["capture_dir"] = None
                frozen = run_detection_on_latest(latest_for_detection)
                if frozen is not None:
                    state["frozen"] = frozen
                    frozen_depth = frozen["depth_vis"].copy()
                    cv2.putText(
                        frozen_depth, "FROZEN", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow(depth_window, frozen_depth)
                    if cfg.show_3d:
                        open_result_viewer()
            elif key in (ord('f'), ord('F')):
                if state["frozen"] is not None:
                    latest = {
                        "rgb": state["frozen"]["rgb"],
                        "depth_raw": state["frozen"].get("depth_raw"),
                        "depth_m": state["frozen"]["depth_m"],
                        "intr": state["frozen"]["intr"],
                        "capture_dir": state["frozen"].get("capture_dir"),
                    }
                    frozen = run_detection_on_latest(latest, reuse_frozen=state["frozen"])
                    if frozen is not None:
                        state["frozen"] = frozen
                        if cfg.show_3d:
                            open_result_viewer()
            elif key in (ord('r'), ord('R')):
                close_result_viewer()
                state["frozen"] = None
                print("[R] live RGB-D mode")
            elif key in (ord('c'), ord('C')):
                if state["frozen"] is not None and state.get("result_vis") is None:
                    open_result_viewer()
            elif key in (ord('d'), ord('D')):
                # Re-create the depth window if the user closed it manually.
                cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
                if state["frozen"] is not None:
                    depth_show = state["frozen"]["depth_vis"].copy()
                    cv2.putText(
                        depth_show, "FROZEN", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow(depth_window, depth_show)
                elif state["latest"] is not None:
                    depth_show = make_realsense_depth_vis(state["latest"]["depth_m"])
                    cv2.putText(
                        depth_show, "LIVE", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow(depth_window, depth_show)
    finally:
        close_result_viewer()
        pipeline.stop()
        cv2.destroyAllWindows()


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

    img = Image.fromarray(frozen['rgb'])

    p = frozen.get('selected_pixel')

    if p is not None:

        x, y = p

        draw = ImageDraw.Draw(img)

        draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=3)

        draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=3)

    return np.asarray(img).copy()



# ============================================================
# RealSense visualization helpers
# ============================================================
def point_mm_to_pixel(point_mm, intr):
    p = np.asarray(point_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(p).all() or p[2] <= 1e-9:
        return None
    x, y, z = p
    u = intr.fx * (x / z) + intr.ppx
    v = intr.fy * (y / z) + intr.ppy
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def make_realsense_depth_vis(depth_m):
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    zmin = float(np.percentile(depth[valid], 2.0))
    zmax = float(np.percentile(depth[valid], 98.0))
    if zmax <= zmin:
        zmax = zmin + 1e-6
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - zmin) / (zmax - zmin), 0.0, 1.0)
    gray = np.uint8(np.rint((1.0 - norm) * 255.0))
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[valid] = color[valid]
    cv2.putText(out, f"depth {zmin:.2f}-{zmax:.2f} m", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_realsense_detection_overlay(frozen):
    image = np.asarray(frozen["rgb"], dtype=np.uint8).copy()
    intr = frozen["intr"]
    detection = frozen["detection"]
    h, w = image.shape[:2]

    # Verifier-passing candidates before NMS: small orange marks.
    for cand in detection["accepted_before_nms"]:
        px = point_mm_to_pixel(cand["position"], intr)
        if px is None:
            continue
        x, y = px
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4, (0, 165, 255), 1, cv2.LINE_AA)

    # Final Hough+EXT 3-D NMS corners: red target + index.
    for i, cand in enumerate(detection["corners"], start=1):
        px = point_mm_to_pixel(cand["position"], intr)
        if px is None:
            continue
        x, y = px
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 10, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS,
                           20, 2, cv2.LINE_AA)
            cv2.putText(image, f"C{i}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def create_realsense_fps_viewer(points_mm, detection, cfg):
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="RealSense FPS | blue=FPS orange=EXT-passed red=Hough+EXT NMS",
        width=1100, height=800,
    )
    if not ok:
        print("[3D] failed to create result viewer")
        return None, None

    geometries = []
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if len(pts):
        z = pts[:, 2]
        z0, z1 = np.percentile(z, [2.0, 98.0])
        den = max(float(z1 - z0), 1e-9)
        t = np.clip((z - z0) / den, 0.0, 1.0)
        colors = np.column_stack((0.35 + 0.30 * t, 0.65 - 0.20 * t, 0.90 - 0.45 * t))
        cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(cloud)
    geometries.append(cloud)

    seed_ids = np.asarray(detection["seed_indices"], dtype=np.int64)
    if len(seed_ids):
        seeds = o3d.geometry.PointCloud()
        seeds.points = o3d.utility.Vector3dVector(pts[seed_ids])
        seeds.paint_uniform_color([0.15, 0.55, 0.95])
        vis.add_geometry(seeds)
        geometries.append(seeds)

    for cand in detection["accepted_before_nms"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(0.45, 0.45 * cfg.voxel_3d_mm)
        )
        marker.translate(np.asarray(cand["position"], dtype=float))
        marker.paint_uniform_color([1.0, 0.65, 0.1])
        marker.compute_vertex_normals()
        vis.add_geometry(marker)
        geometries.append(marker)

    for cand in detection["corners"]:
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(0.8, 0.85 * cfg.voxel_3d_mm)
        )
        marker.translate(np.asarray(cand["position"], dtype=float))
        marker.paint_uniform_color([1.0, 0.12, 0.12])
        marker.compute_vertex_normals()
        vis.add_geometry(marker)
        geometries.append(marker)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
    vis.add_geometry(frame, reset_bounding_box=False)
    geometries.append(frame)

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([0.03, 0.03, 0.03])
        opt.point_size = 4.0
    ctrl = vis.get_view_control()
    if len(pts):
        ctrl.set_lookat(np.mean(pts, axis=0))
        ctrl.set_front([0.0, 0.0, -1.0])
        ctrl.set_up([0.0, -1.0, 0.0])
        ctrl.set_zoom(0.7)
    return vis, geometries



if __name__ == "__main__":

    main()