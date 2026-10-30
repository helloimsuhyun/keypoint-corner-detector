"""STL depth -> single-scale, density-normalized 3D corner voting.

python stl_corner_multiscale_vote3d_global_sah.py \
  --input data/laser_pose_box.stl \
  --voxel-3d-mm 2 \
  --voxel-2d-mm 2 \
  --normal-radius-x 4 \
  --vote-radius-mm 60 \
  --vote-voter-step-mm 6 \
  --vote-max-voters 24 \
  --vote-density-radius-mm 8 \
  --vote-density-floor 0.25 \
  --vote-sigma-mm 4 \
  --vote-voxel-mm 2 \
  --vote-nms-mm 10 \
  --vote-threshold 0.04 \
  --vote-relative-threshold 0.30 \
  --vote-vis-min-relative 0.15 \
  --noise-mm 0.2

"""

from __future__ import annotations



import argparse

import importlib.util

import inspect

import sys

import time

from pathlib import Path

from dataclasses import dataclass



import numpy as np

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

BUILD = "single-scale-3d-corner-open3d-global-sah-v1"
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
    # Hough lambda_min absolute cutoff before Top-K NMS and 3D verification.
    # 0 disables absolute cutoff (20% relative cutoff stays active).
    hough_min_score: float = 0.020

    # 3-D Hough-peak verifier (original 3-D voxel cloud BEFORE surface filtering).
    # A sample is locally extreme along direction u when no neighbor extends
    # farther than tolerance epsilon along that direction.
    ext_radius_mm: float = 8.0
    ext_epsilon_mm: float = 0.5  # Used ONLY for the old hard-score comparison.
    ext_sigma_mm: float = 0.5    # Legacy max-projection score only; default uses physical projected length.
    ext_directions: int = 256
    ext_min_score: float = 0.08   # EXPERIMENTAL threshold; tune on real examples.
    # projection-length: 10% of the ROBUST PROJECTED LENGTH (not 10% of points).
    ext_method: str = "projection-length"
    ext_top_fraction: float = 0.10
    ext_trim_fraction: float = 0.05
    ext_min_span_mm: float = 1.0

    # Iterative corner localizer. These all live in tangent/UV space, so their
    # metric values are resolved from 2-D voxel multipliers in configure().
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





def build_tangent_context(points_mm, query_index, cfg, estimator, *, precomputed=None):

    points = np.asarray(points_mm, dtype=float)

    query = points[int(query_index)].copy()

    support_r = cfg.support_radius_mm



    # Surface threshold implies |height| <= sigma_r*sqrt(1/tau-1).

    # Include that height plus the local normal dependency halo in 3-D.

    if cfg.surface_weight_min > 0:

        height_bound = cfg.sigma_r_mm * np.sqrt(1.0 / cfg.surface_weight_min - 1.0)

        # In cached mode all normals are already computed from the complete scene.
        # There is no need to crop a halo for the SAH iterative dependencies.
        normal_halo = 0.0 if precomputed is not None else (cfg.normal_iterations + 1) * cfg.normal_radius_mm

        context_r = np.hypot(support_r, height_bound) + normal_halo

        tree = precomputed['tree'] if precomputed is not None else cKDTree(points)
        ids = np.asarray(tree.query_ball_point(query, context_r), dtype=np.int64)

    else:

        ids = np.arange(len(points), dtype=np.int64)

    local = points[ids]

    if len(local) < cfg.min_neighbors:

        raise ValueError("Too few points for surface-aware normal estimation.")

    query_local = int(np.flatnonzero(ids == query_index)[0])



    if precomputed is None:
        # Legacy single-query mode (kept for --no-vote3d compatibility).
        normals, normal_info, fit_weights = estimator(
            local, radius=cfg.normal_radius_mm,
            sigma_e_deg=cfg.sigma_e_deg, sigma_r=cfg.sigma_r_mm,
            normal_kernel=cfg.normal_kernel, max_iterations=cfg.normal_iterations,
            update_mode=cfg.normal_update_mode, min_neighbors=cfg.min_neighbors,
            return_info=True, return_weights=True,
        )
        query_fit = fit_weights[query_local]
        if not query_fit['valid'] or not np.isfinite(normals[query_local]).all():
            raise ValueError('Query has no successful surface-aware normal update')
        query_fit = dict(query_fit)
        query_fit['indices'] = ids[query_fit['indices']]
        del fit_weights
        normals = orient_normals_to_camera(local, normals)
    else:
        # Global SAH pass was performed ONCE on the downsampled frozen frame.
        # The indices here must reference that very same frame's points array.
        if len(points) != precomputed['count']:
            raise ValueError('Cached SAH point count differs from input point cloud')
        if not bool(precomputed['valid'][int(query_index)]):
            raise ValueError('Globally cached SAH normal is invalid at this voter')
        normals = precomputed['normals'][ids]
        normal_info = {'backend': precomputed.get('backend', 'global-cached-SAH')}
        # Compatibility metadata; no per-voter SAH or fit-weight recomputation.
        query_fit = {'valid': True, 'indices': np.asarray(
            precomputed['tree'].query_ball_point(query, cfg.normal_radius_mm),
            dtype=np.int64)}

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
    """Observed-candidate soft Hough votes with absolute lambda_min gating.

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

    finite_response = np.maximum(np.nan_to_num(response[candidate_ids], nan=0.0), 0.0)
    output["vote_max"] = float(np.max(finite_response))
    output["support_max"] = float(np.nanmax(support[candidate_ids]))
    if output["vote_max"] <= 1e-12:
        return output

    # Absolute lambda gate + relative 20% gate happen BEFORE Top-4 NMS and verifier.
    # Hough response/support heatmaps remain unchanged.
    cutoff = max(0.20*output["vote_max"], float(cfg.hough_min_score))
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


def directional_projection_length(projections, top_fraction, trim_fraction, min_span_mm):
    """Robust physical projection-position score for each sphere direction.

    z[j,k] = dot(P_j-C, u_k). Candidate C has projection 0 and is excluded.
    Robust endpoints are Q_trim and Q_(1-trim) of z. Let L=hi-lo.
    Remaining upper-tail length = max(hi,0)/L; score = clip(1-tail/q,0,1).
    This uses the TOP q FRACTION OF PROJECTED LENGTH, not q% OF POINT COUNT.
    Directions with span < min_span_mm are invalid and receive score zero.
    """
    z = np.asarray(projections, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or not np.isfinite(z).all():
        raise ValueError("projections must be finite with shape (N,K), N>=1")
    if not (0 < top_fraction < .5 and 0 <= trim_fraction < .25
            and np.isfinite(min_span_mm) and min_span_mm > 0):
        raise ValueError("Invalid projection-length configuration")
    lower, upper = np.quantile(z, [trim_fraction, 1.0-trim_fraction], axis=0)
    span = np.maximum(upper-lower, 0.0)
    tail = np.full(z.shape[1], np.inf, dtype=np.float64)
    good = span >= min_span_mm
    tail[good] = np.maximum(upper[good], 0.0) / span[good]
    score = np.clip(1.0-tail/top_fraction, 0.0, 1.0)
    return tail, score, span, lower, upper


def verify_hough_peaks_extremality_3d(ctx, result, points3d, point_tree=None, sphere_directions=None):
    """Projection-position 3D EXT verifier, applied AFTER Hough lambda gate.

    Default gate: upper 10% of robust projected length (5th--95th pct).
    Legacy max-projection score is retained ONLY for optional A/B diagnostics.
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
        comparison_ids = neighbor_ids[neighbor_ids != scene_id]
        delta = scene[comparison_ids] - center
        enough = len(delta) >= cfg.min_neighbors
        if enough:
            projections = delta @ directions.T
            margins = np.maximum(np.max(projections, axis=0), 0.0)
            max_direction_score = np.exp(-0.5*(margins/cfg.ext_sigma_mm)**2)
            hard = margins <= cfg.ext_epsilon_mm
            tail, length_direction_score, span, lower, upper = directional_projection_length(
                projections, cfg.ext_top_fraction, cfg.ext_trim_fraction, cfg.ext_min_span_mm)
            length_score = float(np.mean(length_direction_score))
            max_score = float(np.mean(max_direction_score))
            hard_score = float(np.mean(hard))
            method_scores = {"projection-length": length_score, "max": max_score}
            method_dirs = {"projection-length": length_direction_score, "max": max_direction_score}
            score = method_scores[cfg.ext_method]
            direction_scores = method_dirs[cfg.ext_method]
        else:
            margins = np.full(len(directions), np.nan)
            tail = span = lower = upper = np.full(len(directions), np.nan)
            length_direction_score = max_direction_score = np.full(len(directions), np.nan)
            hard = np.zeros(len(directions), dtype=bool)
            length_score = max_score = hard_score = score = float("nan")
            direction_scores = np.full(len(directions), np.nan)
        record = {
            "rank": rank, "uv_index": uv_idx, "scene_index": scene_id,
            "center": center.copy(), "hough_score": float(peak["value"]),
            "neighbor_indices": neighbor_ids, "neighbor_count": int(len(neighbor_ids)),
            "comparison_count": int(len(delta)),
            "directions": directions, "max_projection_mm": margins,
            "direction_scores": direction_scores, "extreme_mask": hard,
            "score": score, "hard_score": hard_score,
            "length_score": length_score, "max_score": max_score,
            "length_direction_scores": length_direction_score,
            "max_direction_scores": max_direction_score,
            "length_tail_fraction": tail, "length_span_mm": span,
            "length_robust_lower_mm": lower, "length_robust_upper_mm": upper,
            "method": cfg.ext_method,
            "top_fraction": float(cfg.ext_top_fraction),
            "trim_fraction": float(cfg.ext_trim_fraction),
            "min_span_mm": float(cfg.ext_min_span_mm),
            "valid": bool(enough),
            "pass": bool(enough and np.isfinite(score) and score >= cfg.ext_min_score),
            "radius_mm": float(cfg.ext_radius_mm),
            "epsilon_mm": float(cfg.ext_epsilon_mm),
            "sigma_mm": float(cfg.ext_sigma_mm),
            "threshold": float(cfg.ext_min_score),
        }
        records.append(record)
        peak["extremality_3d"] = record
    result["extremality_cloud_3d"] = scene
    return records


def estimate_context_boundary(ctx):

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
    result["corner_eigen"] = estimate_corner_eigenvalues_uv(ctx, result)
    # Diagnostic only: observed-point Hough does NOT move the original localizer.
    result["hough_vote"] = estimate_observed_point_hough_uv(ctx, result)

    return result





def estimate_corner_target_uv(ctx, result):
    """Estimate one corner target in the CURRENT tangent plane.

    The corner score is lambda_min of the boundary-direction tensor.  Search is
    evaluated over the entire query ROI. A small response plateau around
    the strongest sample is averaged to reduce voxel jitter.

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
    candidate &= radial <= cfg.query_radius_mm

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
        "query_radius_mm": float(cfg.query_radius_mm),
        "refine_radius_mm": float(cfg.corner_refine_radius_mm),
    }


def refine_corner_3d_subpix(points_mm, coarse_ctx, coarse_result, cfg, estimator):
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
            ctx_i = build_tangent_context(points, int(scene_id), cfg, estimator)
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


def localize_corner_iterative(points_mm, initial_query_index, cfg, estimator):
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
    # Construct once; the cloud is the voxelized 3-D scene, before tangent filtering.
    voxel_tree = cKDTree(points)

    # When iteration is disabled, run exactly one complete tangent/Hough/3-D
    # verifier pass and never move or snap the selected query.
    max_steps = int(cfg.corner_max_iterations) if cfg.enable_iteration else 1
    for iteration in range(max_steps):
        ctx = build_tangent_context(points, current, cfg, estimator)
        result = estimate_context_boundary(ctx)
        result["extremality_3d"] = verify_hough_peaks_extremality_3d(
            ctx, result, points, voxel_tree)
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
    """Eight diagnostics: original seven plus 3-D peak extremality results.

    Response color ceilings are fixed across iterations of a query.
    The Hough maps are diagnostic only; localization uses direction tensor.
    """
    cfg = ctx["config"]
    fig = Figure(figsize=(25.6, 11.8))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 12, left=0.035, right=0.96,
                          bottom=0.07, top=0.88, hspace=0.40, wspace=1.0)
    ax_surface, ax_dist, ax_dir, ax_query_tensor = [
        fig.add_subplot(gs[0, i*3:(i+1)*3]) for i in range(4)
    ]
    ax_dir_min, ax_support, ax_vote, ax_ext = [
        fig.add_subplot(gs[1, i*3:(i+1)*3]) for i in range(4)
    ]

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
                 "5. Direction tensor: lambda_min (ACTIVE)\n"
                 "independent color range; spatial weight OFF",
                 "Direction lambda_min")

    # 6. Soft line-vote strength, at OBSERVED candidates only.
    vote = result["hough_vote"]
    candidates = vote["candidate_indices"]
    sup = np.asarray(vote["support"], dtype=float)
    resp = np.asarray(vote["response"], dtype=float)
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
    ax_support.set_title(
        f"6. Hough line vote support (DEBUG)\n"
        f"{len(vote['source_indices'])} voters -> {len(candidates)} observed candidates")

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
        f"7. Observed-point HOUGH: lambda_min (DEBUG)\n"
        f"line sigma={cfg.vote_line_sigma_mm:g} mm | {len(vote['peaks'])} peaks"
        f" | max={vote['vote_max']:.4g}, abs gate={cfg.hough_min_score:g}")

    # 8. 3-D extremality per Hough peak; these results use the ORIGINAL prefilter
    # 3-D voxel cloud. Direction rays + local 3-D neighbors appear in a separate
    # rotatable Open3D diagnostic window (keys 1-4 select another peak).
    ext_records = result.get("extremality_3d", [])
    if ext_records:
        ranks = np.arange(1, len(ext_records) + 1)
        values = [r["score"] if r["valid"] else 0.0 for r in ext_records]
        colors = [(matplotlib.colormaps["viridis"](float(np.clip(r["score"], 0.0, 1.0)))
                   if r["valid"] else "#a0a0a0") for r in ext_records]
        bars = ax_ext.bar(ranks, values, width=0.72, color=colors)
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
        ax_ext.set(xlim=(0.3, max(6.7, len(ext_records)+0.7)),
                   ylim=(0, max(.35, max(values, default=0)*1.35)),
                   xticks=ranks, xlabel="Hough peak rank", ylabel="Mean 3D extremality score")
        ax_ext.legend(loc="upper right", fontsize=8)
    else:
        ax_ext.text(.5, .5, "No Hough peaks", ha="center", va="center", transform=ax_ext.transAxes)
    ax_ext.set_title(f"8. 3D {cfg.ext_method} EXT | selected #{selected_rank}\n"
                     f"r={cfg.ext_radius_mm:g} mm, top={cfg.ext_top_fraction:.0%} length, K={cfg.ext_directions}")

    # Common current query, trajectory and the ORIGINAL direction target.
    path_uv = None
    if localization is not None and len(localization.get("path_world", [])):
        pw = np.asarray(localization["path_world"], dtype=float)
        dq = pw - ctx["query_world"]
        path_uv = np.column_stack((dq @ ctx["u"], dq @ ctx["v"]))
    all_axes = (ax_surface, ax_dist, ax_dir, ax_query_tensor,
                ax_dir_min, ax_support, ax_vote)  # ax_ext uses a rank axis, not UV.
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
        "red star=ACTIVE target | Hough markers=soft 3D score (diagnostic)",
        fontsize=10,
    )
    return fig


def figure_to_rgb(fig):

    """Render to an owned RGB array, even for a Figure without an Agg canvas."""

    # Figure(...) initially owns FigureCanvasBase. Only the Agg canvas
    # implements buffer_rgba(), regardless of matplotlib.use('Agg').
    if not callable(getattr(fig.canvas, "buffer_rgba", None)):
        FigureCanvasAgg(fig)
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
    title = (f"3D EXT | Iter {iteration_number}/{total_iterations} | "
             f"Hough #{rank}/{len(candidates)} | E={score:.3f} ({verdict}) | "
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

    # Every ray encodes the active projected-LENGTH directional score, not a hard mask.
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
        # Red -> yellow -> green; 0=blocked strongly, 1=extreme.
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





def inspect_query(pixel_x, pixel_y, frozen, cfg, estimator):
    if getattr(cfg, "vote3d_enabled", False):
        return inspect_query_vote3d(pixel_x, pixel_y, frozen, cfg, estimator)

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

    if cfg.enable_iteration:
        print("[query] ITER ON: query -> tangent -> tensor target -> move -> repeat")
    else:
        print("[query] ITER OFF: single tangent/Hough/3D extremality pass, query stays fixed")

    try:

        ctx, result, localization = localize_corner_iterative(points, int(qi), cfg, estimator)

    except Exception as exc:

        print(f"[query] {type(exc).__name__}: {exc}")

        return

    frozen["selected_pixel"] = (pixel_x, pixel_y)

    frozen["context"], frozen["boundary"] = ctx, result
    frozen["corner_localization"] = localization
    # Refine only if enabled AND the tensor-based query verifier accepts.
    # Keep the original voxel position if LS constraints are unavailable.
    refinement = None
    if cfg.enable_refine and localization["verification"]["accepted"]:
        try:
            refinement = refine_corner_3d_subpix(points, ctx, result, cfg, estimator)
        except Exception as exc:
            print(f"  [refine3d] failed: {type(exc).__name__}: {exc}; keeping coarse position")
    if not cfg.enable_refine:
        print("  [refine3d] OFF: keep observed voxel query XYZ")
    frozen["corner_refinement_3d"] = refinement
    frozen["corner_position_3d"] = (
        np.asarray(refinement["position"], dtype=float).copy()
        if refinement is not None and refinement["success"]
        else np.asarray(ctx["query_world"], dtype=float).copy()
    )
    if refinement is not None:
        print(f"  [refine3d] {'OK' if refinement['success'] else 'FALLBACK'} "
              f"reason={refinement['reason']} "
              f"obs={refinement['observation_count']}/{refinement['tensor_contributor_count']} "
              f"shift={refinement['shift_mm']:.3f}mm rmse={refinement['rmse_mm']:.3f}mm")
        print(f"  [refine3d] final XYZ [mm]: {np.round(frozen['corner_position_3d'], 4)}")


    # Start at frame 1 (the only frame when iteration is disabled).
    # The final solution and verification still come from the last frame.
    frozen["extremality_rank"] = 1
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
        print(f"  [3D LENGTH EXT Hough #{record['rank']}] {vtext} "
              f"| hard={record['hard_score']:.3f} | "
              f"{'PASS' if record['pass'] else 'FAIL'} "
              f"(tentative thr={record['threshold']:.3f}) | "
              f"neighbors={record['neighbor_count']} | "
              f"XYZ={np.round(record['center'], 2)}")

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

    v = result["hough_vote"]
    vote_peaks = ", ".join(
        f"{np.round(peak['uv'], 1).tolist()}:{peak['value']:.4g}"
        for peak in v["peaks"][:4]) or "none"
    print(f"  [Hough DEBUG] voters={len(v['source_indices'])} "
          f"candidates={len(v['candidate_indices'])} "
          f"lambda_min_max={v['vote_max']:.5g}; peaks UV={vote_peaks}")

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
    view3d = frozen.pop("vote3d_window", None)
    if view3d is not None:
        try: view3d.destroy()
        except Exception: pass
    frozen.pop("vote3d_panel", None)
    frozen.pop("vote3d_image", None)





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
    instructions = ("Click: query | " + iter_instructions
                    + "A/D: previous/next Hough peak | 1-4: select peak "
                    "| R: recapture | Q/Esc: quit")
    status = tk.StringVar(master=root, value=instructions)
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x")
    frame_view = TkImagePanel(root, draw_frozen(frozen))
    set_image_window_size(root, frozen["rgb"], extra_height=35)

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
            frozen["pending_query"] = point

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
            peak_text.set(f"Hough {rank}/{len(entries)} | Soft E={entry['score']:.3f} | "
                          f"{verdict} | 3D neighbors={entry['neighbor_count']}")
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

            pending = frozen.pop("pending_query", None)
            if pending is not None:
                status.set("Computing corner localization, tangent frames and response maps...")
                root.update_idletasks()
                previous_image = frozen.get("boundary_image")
                inspect_query(pending[0], pending[1], frozen, cfg, estimator)
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

            for view_name in ("tangent_view", "extremality_view"):
                vis = frozen.get(view_name)
                if vis is not None:
                    if vis.poll_events():
                        vis.update_renderer()
                    else:
                        vis.destroy_window()
                        frozen[view_name] = None
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
            "STL -> query SAH normal -> tangent projection -> second-order boundary -> "
            "boundary-direction tensor. Spatial parameters are expressed as voxel multipliers."
        )
    )

    parser.add_argument("--version", action="version", version=f"{BUILD} | {Path(__file__).resolve()}")
    iter_group = parser.add_mutually_exclusive_group()
    iter_group.add_argument("--iteration", "--iter", dest="enable_iteration",
                            action="store_true", help="Enable iterative tensor-based localization")
    iter_group.add_argument("--no-iteration", "--no-iter", dest="enable_iteration",
                            action="store_false", help="Disable iteration (DEFAULT): one query evaluation")
    parser.set_defaults(enable_iteration=False)
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument("--refine", dest="enable_refine", action="store_true",
                              help="Enable accepted-query 3D subpixel refinement (DEFAULT)")
    refine_group.add_argument("--no-refine", dest="enable_refine", action="store_false",
                              help="Disable final 3D subpixel refinement")
    parser.set_defaults(enable_refine=True)

    parser.add_argument("--no-vote3d", dest="vote3d", action="store_false",
                        help="Disable single-radius 3D voting and use the original pipeline")
    parser.set_defaults(vote3d=True)
    parser.add_argument("--vote-radius-mm", type=float, default=60.,
                        help="Single 3D voter selection radius [mm]; no multi-scale")
    parser.add_argument("--vote-voter-step-mm", type=float, default=6.,
                        help="3D voxel spacing used to select representative voters")
    parser.add_argument("--vote-max-voters", type=int, default=24,
                        help="Max tangent-frame voters (all reuse one full-frame SAH computation)")
    parser.add_argument("--vote-density-radius-mm", type=float, default=8.,
                        help="Fixed physical radius to count global 3D density")
    parser.add_argument("--vote-density-floor", type=float, default=.25,
                        help="Minimum relative density to prevent giant weights on isolated points")
    parser.add_argument("--vote-weak-lambda", type=float, default=.015,
                        help="Soft 2D tensor lambda_min filter (low for high recall)")
    parser.add_argument("--vote-weak-balance", type=float, default=.08,
                        help="Soft 2D eigen-balance filter")
    parser.add_argument("--vote-uv-peaks", type=int, default=2,
                        help="2D tensor hypotheses per tangent plane, each voter mass divided across peaks")
    parser.add_argument("--vote-uv-nms-mm", type=float, default=8.,
                        help="2D NMS separation in mm")
    parser.add_argument("--vote-voxel-mm", type=float, default=2.,
                        help="3D heatmap voxel pitch [mm]")
    parser.add_argument("--vote-sigma-mm", type=float, default=4.,
                        help="3D isotropic voting kernel std in mm")
    parser.add_argument("--vote-nms-mm", type=float, default=10.,
                        help="3D NMS radius [mm]")
    parser.add_argument("--vote-threshold", type=float, default=.04,
                        help="Absolute normalized 3D heatmap cutoff [0..1]")
    parser.add_argument("--vote-relative-threshold", type=float, default=.30,
                        help="Additional threshold relative to strongest response")
    parser.add_argument("--vote-min-voters", type=int, default=3,
                        help="Minimum distinct 3D source voters supporting a 3D peak")
    parser.add_argument("--vote-max-grid-cells", type=int, default=2000000,
                        help="Safety limit for dense 3D heatmap volume")
    parser.add_argument("--vote-vis-min-relative", type=float, default=.15,
                        help="Open3D heatmap shows voxels >= fraction of field maximum")
    parser.add_argument("--vote-vis-max-points", type=int, default=30000,
                        help="Maximum heatmap voxels rendered as colored Open3D points")
    parser.add_argument("--input", "-i", required=True)

    parser.add_argument("--normal-module", help="Path to the updated surface_aware_harris3d.py")

    parser.add_argument("--stl-unit-mm", type=float, default=1.)

    parser.add_argument("--width", type=int, default=640)

    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--fov-deg", type=float, default=60.)

    # ------------------------------------------------------------------
    # Base scales. Keep them independent even when both defaults are 2 mm.
    # 3-D operations use voxel_3d_mm; tangent/UV operations use voxel_2d_mm.
    # ------------------------------------------------------------------
    parser.add_argument("--voxel-3d-mm", "--voxel-mm", dest="voxel_3d_mm",
                        type=float, default=2.0,
                        help="3-D point-cloud voxel size [mm]. Default: 2")

    parser.add_argument("--voxel-2d-mm", "--uv-voxel-mm", dest="voxel_2d_mm",
                        type=float, default=2.0,
                        help="2-D tangent/UV voxel size [mm]. Default: 2")

    # ------------------------------------------------------------------
    # 3-D multipliers: resolved length = multiplier * voxel_3d_mm.
    # ------------------------------------------------------------------
    parser.add_argument("--normal-radius-x", "--normal-radius-vox", dest="normal_radius_x",
                        type=float, default=4.0,
                        help="SAH normal radius / 3-D voxel. Default: 4 -> 8 mm at v3=2")

    parser.add_argument("--normal-sigma-r-x", type=float, default=0.5,
                        help="SAH/projection height sigma_r / 3-D voxel. Default: 0.5 -> 1 mm at v3=2")

    # ------------------------------------------------------------------
    # 2-D multipliers: resolved length = multiplier * voxel_2d_mm.
    # ------------------------------------------------------------------
    parser.add_argument("--query-radius-x", "--query-radius-vox", dest="query_radius_x",
                        type=float, default=20.0,
                        help="Displayed/evaluated query radius / 2-D voxel. Default: 20 -> 40 mm at v2=2")

    parser.add_argument("--boundary-radius-x", "--boundary-radius-vox", dest="boundary_radius_x",
                        type=float, default=8.0,
                        help="Boundary estimator radius / 2-D voxel. Default: 8 -> 16 mm at v2=2")

    parser.add_argument("--boundary-epsilon-x", "--boundary-epsilon-vox", dest="boundary_epsilon_x",
                        type=float, default=1.0,
                        help="Boundary epsilon / 2-D voxel. Threshold is 1.5*epsilon. Default: 1 -> 2 mm")

    parser.add_argument("--tensor-radius-x", type=float, default=8.0,
                        help="Harris-like tensor radius / 2-D voxel. Default: 8 -> 16 mm at v2=2")

    parser.add_argument("--tensor-sigma-d-x", type=float, default=1.5,
                        help="Boundary-distance weight sigma_d / 2-D voxel. Default: 1.5 -> 3 mm at v2=2")

    # The only new Hough scale is perpendicular line-consistency uncertainty.
    # --vote-sigma-x is accepted as a backward-compatible alias.
    parser.add_argument("--vote-line-sigma-x", "--vote-sigma-x",
                        dest="vote_line_sigma_x", type=float, default=1.0,
                        help="Soft line consistency sigma / 2-D voxel. Default: 1")

    parser.add_argument("--hough-min-score", type=float, default=0.020,
                        help="Absolute Hough lambda_min gate before Top-4 and 3D EXT. 0 disables absolute cutoff; default 0.020")
    parser.add_argument("--ext-method", choices=("projection-length", "max"),
                        default="projection-length",
                        help="3D EXT verifier: default robust projection-position (legacy max for diagnostics)")
    parser.add_argument("--ext-top-fraction", type=float, default=0.10,
                        help="Upper-tail fraction of robust projected LENGTH, default 0.10")
    parser.add_argument("--ext-trim-fraction", type=float, default=0.05,
                        help="Robust projection bounds, default Q05 and Q95")
    parser.add_argument("--ext-min-span-mm", type=float, default=1.0,
                        help="Zero score for degenerate projection directions with span < this mm")
    parser.add_argument("--ext-radius-x", type=float, default=4.0,
                        help="3D extremality neighborhood radius / 3D voxel (default 4 -> 8 mm)")
    parser.add_argument("--ext-epsilon-mm", type=float, default=.5,
                        help="Old binary score tolerance for comparison only [mm] (default 0.5)")
    parser.add_argument("--ext-sigma-mm", type=float, default=.5,
                        help="Legacy max-direction Gaussian width [mm]; not used by default projection-length")
    parser.add_argument("--ext-directions", type=int, default=256,
                        help="Even count of 3-D sphere directions (default 256)")
    parser.add_argument("--ext-min-score", type=float, default=.08,
                        help="Experimental 3D extremality acceptance fraction (default 0.08)")

    # Iterative corner-localizer scales, all in 2-D voxel multiples.
    parser.add_argument("--corner-step-max-x", type=float, default=6.0,
                        help="Maximum move per iteration / 2-D voxel. Default: 6 -> 12 mm")
    parser.add_argument("--corner-refine-radius-x", type=float, default=2.0,
                        help="Peak-centroid refinement radius / 2-D voxel. Default: 2 -> 4 mm")
    parser.add_argument("--corner-converge-x", type=float, default=1.0,
                        help="Convergence tolerance / 2-D voxel. Default: 1 -> 2 mm")
    parser.add_argument("--corner-peak-ratio", type=float, default=0.70,
                        help="Use samples >= ratio * local lambda_min peak for centroid refinement")
    parser.add_argument("--corner-max-iterations", type=int, default=10)
    parser.add_argument("--corner-lambda-min-threshold", type=float, default=0.03,
                        help="Final absolute lambda_min gate. Default: 0.03 (empirical starting value)")
    parser.add_argument("--corner-balance-threshold", type=float, default=0.15,
                        help="Final lambda_min/(lambda_max+lambda_min) gate. Default: 0.15")

    # Dimensionless / angular parameters.
    parser.add_argument("--normal-sigma-e-deg", type=float, default=15.)

    parser.add_argument("--normal-kernel", choices=("cauchy", "gaussian"), default="cauchy")

    parser.add_argument("--normal-max-iterations", type=int, default=5)

    parser.add_argument("--normal-update-mode", choices=("fixed", "synchronous"), default="synchronous")

    parser.add_argument("--surface-weight-min", type=float, default=.1,
                        help="Pre-projection w_e*w_r cutoff; 0 disables filtering")

    parser.add_argument("--min-neighbors", type=int, default=6)

    parser.add_argument("--no-3d-view", action="store_true")

    parser.add_argument("--min-depth-m", type=float, default=.01)

    parser.add_argument("--max-depth-m", type=float, default=5.)

    # Sensor corruption remains in physical units; it is not an algorithmic support scale.
    parser.add_argument("--noise-mm", type=float, default=0.)

    parser.add_argument("--outlier-rate", type=float, default=0.)

    parser.add_argument("--outlier-mm", type=float, default=8.)

    parser.add_argument("--dropout-rate", type=float, default=0.)

    parser.add_argument("--dropout-patches", type=int, default=0)

    parser.add_argument("--dropout-patch-radius-px", type=int, default=0)

    parser.add_argument("--quantization-mm", type=float, default=0.)

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)



def configure(args):

    global WIDTH, HEIGHT, MIN_DEPTH_M, MAX_DEPTH_M, VOXEL_3D_MM, VOXEL_2D_MM

    numeric = [value for value in vars(args).values() if isinstance(value, (int, float))]

    if not np.isfinite(numeric).all():

        raise ValueError("Numeric arguments must be finite.")

    if args.width < 64 or args.height < 64 or not 5 <= args.fov_deg <= 90:

        raise ValueError("Require width/height >=64 and FOV in [5,90].")

    positive = [
        args.stl_unit_mm, args.voxel_3d_mm, args.voxel_2d_mm,
        args.normal_radius_x, args.normal_sigma_r_x, args.ext_radius_x, args.ext_sigma_mm,
        args.query_radius_x, args.boundary_radius_x, args.boundary_epsilon_x,
        args.tensor_radius_x, args.tensor_sigma_d_x,
        args.vote_line_sigma_x,
        args.corner_step_max_x,
        args.corner_refine_radius_x, args.corner_converge_x,
        args.normal_sigma_e_deg,
    ]

    if min(positive) <= 0:

        raise ValueError("Voxel sizes, spatial multipliers, sigma, and epsilon must be positive.")

    if args.normal_max_iterations < 1 or args.min_neighbors < 3:

        raise ValueError("Require iterations>=1 and min-neighbors>=3.")

    if args.hough_min_score < 0:
        raise ValueError("--hough-min-score must be >= 0")
    if not (0.0 < args.ext_top_fraction < 0.5 and 0.0 <= args.ext_trim_fraction < 0.25
            and args.ext_min_span_mm > 0):
        raise ValueError("Require top-fraction in (0,0.5), trim-fraction in [0,0.25), min-span-mm > 0")
    if args.ext_epsilon_mm < 0 or not 0.0 <= args.ext_min_score <= 1.0:
        raise ValueError("Require ext-epsilon-mm >= 0 and ext-min-score in [0,1].")
    if args.ext_directions < 16 or args.ext_directions % 2:
        raise ValueError("ext-directions must be even and >= 16.")
    if args.corner_max_iterations < 1:
        raise ValueError("corner-max-iterations must be >= 1.")
    if not 0.0 < args.corner_peak_ratio <= 1.0:
        raise ValueError("corner-peak-ratio must be in (0,1].")
    if args.corner_lambda_min_threshold < 0.0:
        raise ValueError("corner-lambda-min-threshold must be >= 0.")
    if not 0.0 <= args.corner_balance_threshold <= 0.5:
        raise ValueError("corner-balance-threshold must be in [0, 0.5].")

    if not 0 <= args.surface_weight_min <= 1:

        raise ValueError("surface-weight-min must be in [0,1].")

    if not 0 < args.min_depth_m < args.max_depth_m:

        raise ValueError("Require 0 < min-depth < max-depth.")

    # Both are 2-D-voxel multipliers, so the consistency test is scale invariant.
    if 1.5 * args.boundary_epsilon_x >= .5 * args.boundary_radius_x:

        raise ValueError("Use boundary-radius-x > 3*boundary-epsilon-x.")

    if min(args.noise_mm, args.outlier_mm, args.dropout_patches,
           args.dropout_patch_radius_px, args.quantization_mm) < 0:

        raise ValueError("Noise, outlier magnitude, dropout patches, and quantization must be nonnegative.")

    validate_probability("outlier-rate", args.outlier_rate)

    validate_probability("dropout-rate", args.dropout_rate)

    WIDTH, HEIGHT = args.width, args.height

    MIN_DEPTH_M, MAX_DEPTH_M = args.min_depth_m, args.max_depth_m

    VOXEL_3D_MM = args.voxel_3d_mm

    VOXEL_2D_MM = args.voxel_2d_mm

    v3 = float(args.voxel_3d_mm)
    v2 = float(args.voxel_2d_mm)

    cfg = BoundaryConfig(

        voxel_3d_mm=v3,
        voxel_2d_mm=v2,
        ext_radius_mm=args.ext_radius_x*v3,
        ext_epsilon_mm=args.ext_epsilon_mm,
        ext_sigma_mm=args.ext_sigma_mm,
        ext_directions=args.ext_directions,
        ext_min_score=args.ext_min_score,
        ext_method=args.ext_method,
        ext_top_fraction=args.ext_top_fraction,
        ext_trim_fraction=args.ext_trim_fraction,
        ext_min_span_mm=args.ext_min_span_mm,
        hough_min_score=args.hough_min_score,

        # 3-D scales
        normal_radius_mm=args.normal_radius_x * v3,
        sigma_r_mm=args.normal_sigma_r_x * v3,

        # 2-D tangent/UV scales
        query_radius_mm=args.query_radius_x * v2,
        boundary_radius_mm=args.boundary_radius_x * v2,
        boundary_epsilon_mm=args.boundary_epsilon_x * v2,
        tensor_radius_mm=args.tensor_radius_x * v2,
        tensor_sigma_d_mm=args.tensor_sigma_d_x * v2,
        vote_line_sigma_mm=args.vote_line_sigma_x * v2,
        corner_step_max_mm=args.corner_step_max_x * v2,
        corner_refine_radius_mm=args.corner_refine_radius_x * v2,
        corner_converge_mm=args.corner_converge_x * v2,
        corner_peak_ratio=args.corner_peak_ratio,
        corner_max_iterations=args.corner_max_iterations,
        corner_lambda_min_threshold=args.corner_lambda_min_threshold,
        corner_balance_threshold=args.corner_balance_threshold,

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

    cfg.vote3d_enabled = bool(args.vote3d)
    cfg.vote3d_args = args
    if cfg.vote3d_enabled:
        pos = [args.vote_radius_mm, args.vote_voter_step_mm,
               args.vote_density_radius_mm, args.vote_weak_lambda,
               args.vote_uv_nms_mm, args.vote_voxel_mm,
               args.vote_sigma_mm, args.vote_nms_mm]
        if min(pos) <= 0 or args.vote_max_voters < 2 or args.vote_uv_peaks < 1:
            raise ValueError("Voting spacings, radii, score and voter counts must be positive")
        if (not 0 < args.vote_density_floor <= 1 or not 0 <= args.vote_weak_balance <= .5
                or not 0 <= args.vote_relative_threshold <= 1
                or not 0 <= args.vote_threshold <= 1 or args.vote_min_voters < 2
                or not 0 < args.vote_vis_min_relative <= 1
                or args.vote_vis_max_points < 100):
            raise ValueError("Invalid normalized voting thresholds/floors")
    noise = NoiseConfig(

        gaussian_sigma_mm=args.noise_mm, outlier_rate=args.outlier_rate,

        outlier_mm=args.outlier_mm, dropout_rate=args.dropout_rate,

        dropout_patches=args.dropout_patches, dropout_patch_radius_px=args.dropout_patch_radius_px,

        quantization_mm=args.quantization_mm, seed=args.seed,

    )

    return cfg, noise



def main():

    args = parse_args()

    cfg, noise = configure(args)

    print(f"[BUILD {BUILD}] {Path(__file__).resolve()}", flush=True)

    print("[GUI] Frozen-frame click: Tk | 3D voting heatmap + corners: Open3D", flush=True)

    if o3d is None:

        raise RuntimeError("The interactive STL demo requires open3d.")

    if tk is None:

        raise RuntimeError("Tk is unavailable. In this conda environment, run: conda install tk pillow")

    estimator = load_normal_estimator(args.normal_module)

    mesh = load_stl_as_mm(args.input, args.stl_unit_mm)

    scene = make_raycast_scene(mesh)

    if cfg.vote3d_enabled:
        print("[MODE] full-frame SAH ONCE -> cached per-voter tangents -> 2D tensor -> density-balanced 3D vote")
        print(f"[SINGLE SCALE] radius={args.vote_radius_mm:g} mm | max voters={args.vote_max_voters}")
    print(f"[SWITCHES] ITERATION={'ON' if cfg.enable_iteration else 'OFF'} | "
          f"REFINE={'ON' if cfg.enable_refine else 'OFF'} | HOUGH TOP-K={HOUGH_TOP_K}")
    if cfg.enable_iteration:
        print("QUERY -> TANGENT -> CORNER RESPONSE -> MOVE -> RETANGENT -> ... -> CONVERGE")
    else:
        print("QUERY -> TANGENT -> HOUGH TOP-4 + 3D EXT -> TENSOR VERIFIER (SINGLE PASS)")

    print(
        f"[3-D scale] voxel={cfg.voxel_3d_mm:g} mm | "
        f"normal r={cfg.normal_radius_mm:g} mm ({cfg.normal_radius_mm/cfg.voxel_3d_mm:g}x) | "
        f"sigma_r={cfg.sigma_r_mm:g} mm ({cfg.sigma_r_mm/cfg.voxel_3d_mm:g}x)"
    )

    print(
        f"[2-D scale] voxel={cfg.voxel_2d_mm:g} mm | "
        f"query r={cfg.query_radius_mm:g} ({cfg.query_radius_mm/cfg.voxel_2d_mm:g}x) | "
        f"boundary r={cfg.boundary_radius_mm:g} ({cfg.boundary_radius_mm/cfg.voxel_2d_mm:g}x) | "
        f"epsilon={cfg.boundary_epsilon_mm:g} ({cfg.boundary_epsilon_mm/cfg.voxel_2d_mm:g}x) mm"
    )

    print(
        f"[2-D tensor] r={cfg.tensor_radius_mm:g} mm ({cfg.tensor_radius_mm/cfg.voxel_2d_mm:g}x) | "
        f"sigma_d={cfg.tensor_sigma_d_mm:g} mm ({cfg.tensor_sigma_d_mm/cfg.voxel_2d_mm:g}x) | "
        f"threshold={1.5*cfg.boundary_epsilon_mm:g} mm "
        f"({1.5*cfg.boundary_epsilon_mm/cfg.voxel_2d_mm:g}x)"
    )

    print(f"[3D EXT {cfg.ext_method}] radius={cfg.ext_radius_mm:g} mm | "
          f"top={cfg.ext_top_fraction:.0%} projected length | directions={cfg.ext_directions} | "
          f"comparison hard epsilon={cfg.ext_epsilon_mm:g} mm | "
          f"tentative threshold={cfg.ext_min_score:g} | prefilter voxel cloud")
    print(f"[observed-point Hough] lambda_min gate={cfg.hough_min_score:g} | sigma_line={cfg.vote_line_sigma_mm:g} mm "
          f"| boundary sigma_d={cfg.tensor_sigma_d_mm:g} mm "
          f"| all valid points vote; only observed UV candidates receive votes")

    print(
        f"[2-D localizer] query/search={cfg.query_radius_mm:g} mm "
        f"({cfg.query_radius_mm/cfg.voxel_2d_mm:g}x) | "
        f"step<={cfg.corner_step_max_mm:g} mm ({cfg.corner_step_max_mm/cfg.voxel_2d_mm:g}x) | "
        f"refine={cfg.corner_refine_radius_mm:g} mm ({cfg.corner_refine_radius_mm/cfg.voxel_2d_mm:g}x) | "
        f"converge={cfg.corner_converge_mm:g} mm ({cfg.corner_converge_mm/cfg.voxel_2d_mm:g}x) | "
        f"max_iter={cfg.corner_max_iterations}"
    )

    print(
        f"[corner verifier] lambda_min >= {cfg.corner_lambda_min_threshold:g} | "
        f"lambda_min/(lambda_max+lambda_min) >= {cfg.corner_balance_threshold:g}"
    )
    print(
        f"surface cutoff={cfg.surface_weight_min:g} | padded support={cfg.support_radius_mm:g} mm "
        f"({cfg.support_radius_mm/cfg.voxel_2d_mm:.3f}x 2-D voxel)"
    )

    try:

        while True:

            capture = run_camera_selector(mesh, scene, noise, WIDTH, HEIGHT, args.fov_deg)

            if capture is None:

                break

            frozen = prepare_frozen(capture)
            if cfg.vote3d_enabled:
                _vote3d_prepare_global_sah(frozen, cfg, estimator)

            if run_frozen_click_stage(frozen, cfg, estimator) == "quit":

                break

    finally:

        if "frozen" in locals():

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

    img = Image.fromarray(frozen['rgb'])

    p = frozen.get('selected_pixel')

    if p is not None:

        x, y = p

        draw = ImageDraw.Draw(img)

        draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=3)

        draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=3)

    return np.asarray(img).copy()



# ---------------------------------------------------------------------------
# Single-scale corner-hypothesis voting extension. Input geometry and UV
# estimator are unchanged; Hough lines are NOT evaluated in this path.
# ---------------------------------------------------------------------------

def _vote3d_prepare_global_sah(frozen, cfg, estimator):
    """Full-frame SAH exactly once per frozen frame (and cached across clicks).

    SAH runs on 3D voxel-downsampled points. No SAH calls occur in voter loops
    or for another query clicked on the same frozen frame.
    """
    if 'vote3d_global_sah' in frozen:
        return frozen['vote3d_global_sah']

    if 'points_voxel_mm' not in frozen:
        frozen['points_voxel_mm'] = voxel_downsample_3d(
            frozen['points_raw_mm'], cfg.voxel_3d_mm)
    points = np.asarray(frozen['points_voxel_mm'], dtype=np.float64)
    if len(points) < cfg.min_neighbors:
        raise ValueError('Too few points for full-frame SAH normal computation')
    print(f'[SAH GLOBAL] start: {len(points):,} voxelized points | '
          f'radius={cfg.normal_radius_mm:g} mm, iterations={cfg.normal_iterations}',
          flush=True)
    start = time.perf_counter()
    normals, normal_info, fit_weights = estimator(
        points, radius=cfg.normal_radius_mm,
        sigma_e_deg=cfg.sigma_e_deg, sigma_r=cfg.sigma_r_mm,
        normal_kernel=cfg.normal_kernel, max_iterations=cfg.normal_iterations,
        update_mode=cfg.normal_update_mode, min_neighbors=cfg.min_neighbors,
        return_info=True, return_weights=True,
    )
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != points.shape or len(fit_weights) != len(points):
        raise ValueError('Global SAH backend returned mismatched normals/weights')
    valid_fit = np.array([bool(f.get('valid', False)) for f in fit_weights],
                         dtype=bool)
    lengths = np.linalg.norm(normals, axis=1)
    valid = (valid_fit & np.isfinite(normals).all(axis=1) &
             np.isfinite(lengths) & (lengths > 1e-12))
    # Align normals once, consistently, for all future tangent frames.
    normals = orient_normals_to_camera(points, normals)
    normals[~valid] = np.nan
    # Drop per-point fit weights after extracting validity to save RAM.
    del fit_weights
    cache = dict(normals=normals, valid=valid, count=len(points),
                 backend=normal_info.get('backend','global-SAH'),
                 tree=cKDTree(points))
    frozen['vote3d_global_sah'] = cache
    print(f'[SAH GLOBAL] done: {np.count_nonzero(valid):,}/{len(points):,} '
          f'valid normals | {time.perf_counter()-start:.2f}s | '
          'will reuse on every voter and click', flush=True)
    return cache


def _vote3d_select_voters(points, origin, radius, step, maximum):
    """Spatially distributed representatives within ONE 3-D search radius."""
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)
    ids = np.asarray(tree.query_ball_point(origin, float(radius)), dtype=np.int64)
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64)
    # Representative per fixed-grid cell, selecting the closest to its center.
    cells = np.floor((pts[ids] - origin) / step).astype(np.int64)
    unique, labels = np.unique(cells, axis=0, return_inverse=True)
    offset = (cells + .5)*step
    d2 = np.sum((pts[ids]-origin-offset)**2, axis=1)
    order = np.lexsort((d2, labels))
    _, first = np.unique(labels[order], return_index=True)
    ids = ids[order[first]]
    dist = np.linalg.norm(pts[ids]-origin, axis=1)
    ordered = np.argsort(dist, kind='stable')
    ids, dist = ids[ordered], dist[ordered]
    if len(ids) > maximum:
        # Stratified radial retention rather than selecting only the closest.
        cuts = np.linspace(0, len(ids)-1, maximum).round().astype(int)
        ids = ids[cuts]
    return ids


def _vote3d_global_inverse_density(points, selected_xyz, density_radius, floor_fraction):
    """GLOBAL 3D density, not density recalculated from scale's selected voters.

    Clamp the inverse count to avoid isolated outliers obtaining infinite votes.
    """
    tree = cKDTree(np.asarray(points, dtype=float))
    counts = np.asarray(tree.query_ball_point(selected_xyz, density_radius,
                                              return_length=True), dtype=float)
    base = max(1., float(np.median(counts))*floor_fraction)
    inv = 1. / np.maximum(counts, base)
    return inv, counts, base


def _vote3d_uv_hypotheses(ctx, boundary, args):
    """Extract several 2D tensor maxima; weak thresholds only, no Hough."""
    tensor = estimate_corner_eigenvalues_uv(ctx, boundary)
    uv = np.asarray(ctx['support_uv'])
    a = np.asarray(tensor['lambda_min'])
    b = np.asarray(tensor['lambda_max'])
    balance = a / np.maximum(a+b, 1e-12)
    valid = (boundary['evaluation_mask'] & np.isfinite(a) & np.isfinite(b)
             & (a >= args.vote_weak_lambda) & (balance >= args.vote_weak_balance))
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(a[ids])[::-1]]
    outputs = []
    for i in ids:
        c = uv[i]
        if any(np.linalg.norm(c - h['uv']) < args.vote_uv_nms_mm for h in outputs):
            continue
        # Keep the continuous projected hypothesis (do not snap back to XYZ).
        xyz = ctx['query_world'] + c[0]*ctx['u'] + c[1]*ctx['v']
        # 2D normalized strength in [0,1] (weighted tensor has trace <= 1).
        # Confidence references are anchored to the WEAK 2D cutoffs. A
        # moderately clear corner gets full weight; near-threshold candidates
        # still contribute without becoming hard-rejected.
        confidence = float(min(1., a[i]/max(2*args.vote_weak_lambda,1e-12))
                           * min(1., balance[i]/max(2*args.vote_weak_balance,1e-12)))
        outputs.append(dict(uv=c.copy(), xyz=xyz, lambda_min=float(a[i]),
                            balance=float(balance[i]), confidence=confidence))
        if len(outputs) >= args.vote_uv_peaks:
            break
    return outputs


def _vote3d_extract_raw_hypotheses(points, qidx, cfg, estimator, *, sah_cache=None):
    """Run the ORIGINAL boundary and tensor process at many tangent samples."""
    args = cfg.vote3d_args
    xyz = np.asarray(points, dtype=float)
    origin = xyz[qidx].copy()
    radius = float(args.vote_radius_mm)
    ids = _vote3d_select_voters(xyz, origin, radius, args.vote_voter_step_mm,
                               args.vote_max_voters)
    if len(ids) < 2:
        return origin, [], [], radius
    if sah_cache is None:
        raise ValueError('vote3d requires global SAH cache; call _vote3d_prepare_global_sah first')
    # Filter invalid SAH voter normals before density balancing, not after.
    ids = ids[sah_cache['valid'][ids]]
    if len(ids) < 2:
        return origin, [], [], radius
    inverse, density, floor = _vote3d_global_inverse_density(
        xyz, xyz[ids], args.vote_density_radius_mm, args.vote_density_floor)
    records, hypotheses = [], []
    for t,(pid,inv,count) in enumerate(zip(ids,inverse,density),1):
        rec = {'source_index': int(pid), 'source_xyz': xyz[pid].copy(),
               'radius_mm': float(np.linalg.norm(xyz[pid]-origin)),
               'density_count': float(count), 'density_floor': floor,
               'inverse_density': float(inv), 'valid': False}
        try:
            ctx = build_tangent_context(xyz, int(pid), cfg, estimator,
                                        precomputed=sah_cache)
            boundary = estimate_boundary_uv(
                ctx['support_uv'], cfg.boundary_radius_mm, cfg.boundary_epsilon_mm,
                min_neighbors=cfg.min_neighbors, evaluation_radius=cfg.query_radius_mm,
                crop_radius=cfg.support_radius_mm, voxel_mm=cfg.voxel_2d_mm)
            peaks = _vote3d_uv_hypotheses(ctx, boundary, args)
            rec['valid'] = bool(peaks)
            rec['hypothesis_count'] = len(peaks)
            # Confidence modulates the vote mass, while each voter has
            # at most one total unit before density correction. Splitting
            # across hypotheses prevents duplicate votes from one patch.
            for h in peaks:
                if h['confidence'] <= 0:
                    continue
                h.update(source_index=int(pid), source_xyz=xyz[pid].copy(),
                         voter_radius_mm=rec['radius_mm'],
                         density_count=float(count),
                         mass=float(inv*h['confidence']/len(peaks)))
                hypotheses.append(h)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            rec['error'] = str(exc)
        records.append(rec)
        print(f"  [vote3d] tangent {t:02d}/{len(ids):02d} "
              f"| distance={rec['radius_mm']:.1f}mm | density={count:g} "
              f"| hypothesis={rec.get('hypothesis_count', 0)}", flush=True)
    return origin, records, hypotheses, radius


def _vote3d_splat_linear(grid, locations, masses, lower, pitch):
    """3D trilinear cloud-in-cell deposition, conserving vote mass."""
    shape = np.asarray(grid.shape, dtype=int)
    gx = (np.asarray(locations)-lower)/pitch
    base = np.floor(gx).astype(int)
    frac = gx - base
    for di in (0,1):
        for dj in (0,1):
            for dk in (0,1):
                delta = np.array([di,dj,dk])
                idx = base+delta
                ok = np.all((idx>=0)&(idx<shape),axis=1)
                if not ok.any():
                    continue
                factors = np.where(delta[None,:]==1,frac,1-frac)
                weight = np.prod(factors,axis=1)*masses
                np.add.at(grid, (idx[ok,0],idx[ok,1],idx[ok,2]),weight[ok])


def _vote3d_make_grid(hypotheses, origin, args):
    points = np.asarray([h['xyz'] for h in hypotheses], dtype=float)
    # Include clicked query and pad by 3sigma for trustworthy Gaussian values.
    limits = np.vstack([points, origin[None,:]]) if len(points) else origin[None,:]
    pad = max(3*args.vote_sigma_mm, args.vote_voxel_mm*2)
    lower = np.min(limits,axis=0)-pad
    upper = np.max(limits,axis=0)+pad
    pitch = float(args.vote_voxel_mm)
    shape = np.maximum(5,np.ceil((upper-lower)/pitch).astype(int)+1)
    cells = int(np.prod(shape))
    if cells > args.vote_max_grid_cells:
        raise ValueError(f"3D grid {tuple(shape)}={cells:,} cells > "
                         f"--vote-max-grid-cells={args.vote_max_grid_cells}; increase --vote-voxel-mm")
    return lower, pitch, tuple(shape)


def _vote3d_find_peaks(field, lower, pitch, args, hypotheses):
    """3D NMS + relative and absolute thresholds + DISTINCT voter support."""
    from scipy.ndimage import maximum_filter
    mx = float(np.max(field)) if field.size else 0.
    cutoff = max(args.vote_threshold, args.vote_relative_threshold*mx)
    if mx <= 1e-12 or mx < cutoff:
        return []
    rad = max(1, int(np.ceil(args.vote_nms_mm/pitch)))
    localmax = maximum_filter(field,size=2*rad+1,mode='constant')
    ids = np.argwhere((field>=cutoff)&(field>=localmax-1e-12))
    ids = ids[np.argsort(field[tuple(ids.T)])[::-1]]
    sources = np.array([h['source_index'] for h in hypotheses],dtype=int)
    pos = np.array([h['xyz'] for h in hypotheses],dtype=float)
    peaks = []
    for idx in ids:
        xyz = lower+idx*pitch
        if any(np.linalg.norm(xyz-p['xyz'])<args.vote_nms_mm for p in peaks):
            continue
        # Count voters with a hypothesis within 2 sigma, independent of vote weight.
        nearby = np.linalg.norm(pos-xyz[None,:],axis=1)<=2*args.vote_sigma_mm
        voters = len(np.unique(sources[nearby]))
        if voters < args.vote_min_voters:
            continue
        peaks.append({'xyz':xyz,'score':float(field[tuple(idx)]),
                      'distinct_voters':int(voters),'cell':tuple(idx)})
        if len(peaks)>=12:
            break
    return peaks


def _vote3d_single_scale(hypotheses, records, origin, radius, args):
    """One global density-balanced 3D voting field (no persistence/multi-scale).

    The denominator includes ALL representative voters, even those yielding
    no 2D corner hypotheses. Otherwise rejecting a voter can increase the
    normalized peak score and make scores incomparable across viewpoints.
    """
    from scipy.ndimage import gaussian_filter

    lower, pitch, shape = _vote3d_make_grid(hypotheses, origin, args)
    sigma = float(args.vote_sigma_mm / pitch)
    # Normalize the Gaussian convolution to have unit peak at one exact voxel.
    ksize = max(9, int(2 * np.ceil(5 * sigma) + 1))
    impulse_reference = np.zeros((ksize, ksize, ksize), dtype=np.float32)
    center = ksize // 2
    impulse_reference[center, center, center] = 1.0
    kernel_peak = float(gaussian_filter(impulse_reference, sigma, mode='constant')[center, center, center])

    denominator = float(sum(r['inverse_density'] for r in records))
    impulse = np.zeros(shape, dtype=np.float32)
    if hypotheses and denominator > 0:
        locations = np.asarray([h['xyz'] for h in hypotheses], dtype=float)
        masses = np.asarray([h['mass'] for h in hypotheses], dtype=float) / denominator
        _vote3d_splat_linear(impulse, locations, masses, lower, pitch)
        field = gaussian_filter(impulse, sigma, mode='constant') / max(kernel_peak, 1e-12)
    else:
        field = impulse
    peaks = _vote3d_find_peaks(field, lower, pitch, args, hypotheses) if hypotheses else []
    return {
        'origin': np.asarray(origin, dtype=float).copy(),
        'radius': float(radius),
        'lower': lower, 'pitch': pitch, 'shape': shape,
        'field': field, 'peaks': peaks,
        'voter_count': len(records), 'hypothesis_count': len(hypotheses),
        'mass_normalizer': denominator, 'chosen': None,
    }


def _vote3d_verify_xyz(points,xyz,cfg,tree=None,directions=None):
    """Reuse the existing PROJECTION-LENGTH 3D verifier at arbitrary 3D peak."""
    tree=cKDTree(points) if tree is None else tree
    dirs=fibonacci_sphere_directions(cfg.ext_directions) if directions is None else directions
    # Do not delete the nearest observed point: candidate is continuous XYZ.
    idx=np.asarray(tree.query_ball_point(xyz,cfg.ext_radius_mm),dtype=int)
    delta=np.asarray(points[idx],dtype=float)-xyz
    if len(delta)<cfg.min_neighbors:
        return {'accepted':False,'score':float('nan'),'count':len(delta),'reason':'too_few_neighbors'}
    projections=delta@dirs.T
    if cfg.ext_method=='projection-length':
        _,direction_scores,*_=directional_projection_length(
            projections,cfg.ext_top_fraction,cfg.ext_trim_fraction,cfg.ext_min_span_mm)
    else:
        margin=np.maximum(np.max(projections,axis=0),0.)
        direction_scores=np.exp(-.5*(margin/cfg.ext_sigma_mm)**2)
    s=float(np.mean(direction_scores))
    return {'accepted':bool(s>=cfg.ext_min_score),'score':s,'count':len(delta),
            'reason':'passed' if s>=cfg.ext_min_score else 'extremality_below_threshold'}


def _vote3d_open3d(result, records, hypotheses, points, args):
    """Interactive Open3D 3D heatmap, voter hypotheses, and NMS detections.

    1: scene cloud only; 2: vote heatmap only; 3: both;
    V: toggle per-voter hypotheses; P: toggle NMS peaks. Escape: close.
    Colored heatmap is a sparse visualization of the actual 3D voxel field.
    No 2D projection or Matplotlib figure is used for the vote output.
    """
    if o3d is None:
        raise RuntimeError('Open3D is required for the 3D visualization')

    xyz = np.asarray(points, dtype=float)
    origin = np.asarray(result['origin'], dtype=float)
    radius = float(result['radius'])
    near = xyz[np.linalg.norm(xyz - origin, axis=1) <= 1.35 * radius]
    if len(near) > 60000:
        # Deterministic sampling prevents rendering latency on dense clouds.
        ids = np.linspace(0, len(near) - 1, 60000, dtype=int)
        near = near[ids]

    def cloud(coords, rgb):
        obj = o3d.geometry.PointCloud()
        obj.points = o3d.utility.Vector3dVector(np.asarray(coords, dtype=float).reshape(-1, 3))
        rgb = np.asarray(rgb, dtype=float)
        if rgb.ndim == 1:
            rgb = np.repeat(rgb[None, :], len(coords), axis=0)
        obj.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
        return obj

    scene_cloud = cloud(near, [0.52, 0.56, 0.62])
    # Actual dense 3D heatmap volume rendered by sparse colored voxels.
    field = np.asarray(result['field'])
    field_max = float(field.max()) if field.size else 0.
    mask = (field >= max(field_max * args.vote_vis_min_relative, 1e-9)) if field_max > 1e-12 else np.zeros_like(field, dtype=bool)
    voxels = np.argwhere(mask)
    if len(voxels) > args.vote_vis_max_points:
        # Deterministically thin visible voxels; do not change voting/NMS itself.
        rng = np.random.default_rng(42)
        voxels = voxels[np.sort(rng.choice(len(voxels), size=args.vote_vis_max_points, replace=False))]
    scores = field[tuple(voxels.T)] if len(voxels) else np.empty(0)
    positions = np.asarray(result['lower'])[None, :] + voxels * result['pitch']
    from matplotlib import colormaps
    colors = colormaps['inferno'](np.clip(scores / max(field_max, 1e-12), 0, 1))[:, :3]
    heatmap_cloud = cloud(positions, colors)

    hypothesis_xyz = (np.asarray([h['xyz'] for h in hypotheses], dtype=float).reshape(-1, 3)
                      if hypotheses else np.empty((0, 3)))
    hypothesis_cloud = cloud(hypothesis_xyz, [1.0, 0.66, 0.16])

    def ball(center, color, r):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(r), resolution=12)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        mesh.translate(np.asarray(center, dtype=float))
        return mesh

    base_r = max(0.8 * float(args.vote_voxel_mm), 1.0)
    query_sphere = ball(origin, [0.12, 0.46, 1.0], 1.5 * base_r)
    peak_meshes = []
    for peak in result['peaks']:
        accepted = peak.get('verification', {}).get('accepted', False)
        color = [0.11, 0.95, 0.38] if accepted else [1.0, 0.20, 0.20]
        peak_meshes.append(ball(peak['xyz'], color, 1.8 * base_r))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(window_name=(
        f'Open3D single-scale corner vote | R={radius:g} mm | '
        f'peaks={len(result["peaks"])}'), width=1360, height=850)
    if created is False:
        raise RuntimeError('Could not create Open3D window (check graphical DISPLAY)')

    view = {'mode': 3, 'hypotheses': False, 'peaks': True}
    active = set()
    bundles = {'scene': [scene_cloud], 'heatmap': [heatmap_cloud],
               'hypotheses': [hypothesis_cloud], 'peaks': peak_meshes,
               'query': [query_sphere]}

    def update_vis(v):
        wanted = {'query'}
        if view['mode'] in (1, 3):
            wanted.add('scene')
        if view['mode'] in (2, 3):
            wanted.add('heatmap')
        if view['hypotheses']:
            wanted.add('hypotheses')
        if view['peaks']:
            wanted.add('peaks')
        for label, geometries in bundles.items():
            if label in wanted and label not in active:
                for geom in geometries:
                    if isinstance(geom, o3d.geometry.PointCloud) and not geom.has_points():
                        continue
                    v.add_geometry(geom, reset_bounding_box=False)
                active.add(label)
            elif label not in wanted and label in active:
                for geom in geometries:
                    v.remove_geometry(geom, reset_bounding_box=False)
                active.remove(label)
        v.update_renderer()
        return False

    def set_mode(n):
        def cb(v):
            view['mode'] = n
            return update_vis(v)
        return cb

    def toggle(name):
        def cb(v):
            view[name] = not view[name]
            return update_vis(v)
        return cb

    try:
        vis.add_geometry(scene_cloud)
        active.add('scene')
        vis.add_geometry(query_sphere, reset_bounding_box=False)
        active.add('query')
        update_vis(vis)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.075, 0.085, 0.105])
        opt.point_size = 4.0
        opt.show_coordinate_frame = True
        vis.register_key_callback(ord('1'), set_mode(1))
        vis.register_key_callback(ord('2'), set_mode(2))
        vis.register_key_callback(ord('3'), set_mode(3))
        vis.register_key_callback(ord('V'), toggle('hypotheses'))
        vis.register_key_callback(ord('P'), toggle('peaks'))
        print('[Open3D] 1=cloud, 2=heatmap, 3=combined, V=2D hypotheses, '
              'P=NMS detections, ESC=close | '
              'blue=query, orange=hypothesis, red=rejected, green=accepted')
        print(f'[Open3D] Heatmap peak={field_max:.5f}, rendered voxels={len(voxels):,} '
              f'(of {np.count_nonzero(mask):,} above {args.vote_vis_min_relative:.0%} peak)')
        vis.run()
    finally:
        vis.destroy_window()


def inspect_query_vote3d(pixel_x, pixel_y, frozen, cfg, estimator):
    clicked = pixel_to_point_mm(pixel_x, pixel_y, frozen['depth_m'], frozen['intr'])
    if clicked is None:
        print('[vote3d] Invalid depth at clicked pixel')
        return
    if 'points_voxel_mm' not in frozen:
        frozen['points_voxel_mm'] = voxel_downsample_3d(
            frozen['points_raw_mm'], cfg.voxel_3d_mm)
    points = np.asarray(frozen['points_voxel_mm'], dtype=float)
    _, qi = cKDTree(points).query(clicked)
    args = cfg.vote3d_args
    start = time.perf_counter()
    print(f'[vote3d] clicked query index={qi}, XYZ={np.round(points[qi], 2)} mm')
    sah_cache = _vote3d_prepare_global_sah(frozen, cfg, estimator)
    origin, records, hypotheses, radius = _vote3d_extract_raw_hypotheses(
        points, int(qi), cfg, estimator, sah_cache=sah_cache)
    volume = _vote3d_single_scale(hypotheses, records, origin, radius, args)

    # Every threshold/NMS candidate gets the original 3D EXT verifier.
    # No scale persistence or multi-radius pass is required.
    tree = sah_cache['tree']
    dirs = fibonacci_sphere_directions(cfg.ext_directions)
    for peak in volume['peaks']:
        peak['verification'] = _vote3d_verify_xyz(points, peak['xyz'], cfg, tree, dirs)
    accepted = [p for p in volume['peaks'] if p['verification']['accepted']]
    volume['chosen'] = (max(accepted, key=lambda p: p['score']) if accepted else None)

    print(f"[vote3d SINGLE] R={radius:g} mm | voters={len(records)} "
          f"valid={sum(r['valid'] for r in records)} "
          f"| hypotheses={len(hypotheses)} | grid={volume['shape']} "
          f"| mass={volume['mass_normalizer']:.5g} "
          f"| peak={float(volume['field'].max()):.5f} "
          f"| NMS peaks={len(volume['peaks'])} "
          f"| accepted={len(accepted)} "
          f"| elapsed={time.perf_counter()-start:.1f}s")
    for k, p in enumerate(volume['peaks'], 1):
        vv = p['verification']
        print(f"  #{k:02d} {'ACCEPT' if vv['accepted'] else 'REJECT'} "
              f"XYZ={np.round(p['xyz'], 2)} "
              f"S={p['score']:.5f} voters={p['distinct_voters']} "
              f"EXT={vv['score']:.4f} reason={vv['reason']}")
    if volume['chosen'] is None:
        print('[vote3d] NO ACCEPTED CORNER (NMS threshold/support or 3D verifier)')
    else:
        print('[vote3d] SELECTED XYZ=', np.round(volume['chosen']['xyz'], 3))

    frozen['vote3d_output'] = volume
    frozen['selected_pixel'] = (pixel_x, pixel_y)
    # Viewer remains useful even when no corner is accepted.
    _vote3d_open3d(volume, records, hypotheses, points, args)


if __name__ == "__main__":

    main()