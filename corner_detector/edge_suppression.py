"""Whole-cloud bilateral-CED filtering, centered weighted PCA, and ratio scoring.

This module preserves the original normalized CED field and feeds only the
dense whole-cloud detector path.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


def saturate_pca_ratio(ratio, tau=None):
    """Return raw PCA ratios or apply ``tau * tanh(ratio / tau)``.

    Non-finite ratio entries are preserved so they can never become ordinary
    detector scores. ``tau`` itself must be omitted or finite and positive.
    """
    values = np.asarray(ratio, dtype=np.float64)
    if tau is None:
        return values.copy()
    tau_value = float(tau)
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau must be a positive finite number or None")
    transformed = tau_value * np.tanh(values / tau_value)
    return np.where(np.isfinite(values), transformed, values)


def ced_weighted_pca(points_mm, ced_weights):
    """Return centered weighted-PCA diagnostics for one neighborhood.

    Eigenvalues are returned in descending order and eigenvectors are the
    corresponding columns.  The caller is responsible for excluding invalid
    CED samples and deciding how many positive-weight samples are sufficient.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    weights = np.asarray(ced_weights, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(weights):
        raise ValueError("points_mm and ced_weights must have shapes (N, 3) and (N,)")
    if not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError("points_mm and ced_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("ced_weights must be non-negative")

    weight_sum = float(np.sum(weights))
    weight_epsilon = max(
        1e-15,
        64.0 * np.finfo(np.float64).eps * max(1.0, float(len(weights))),
    )
    if weight_sum <= weight_epsilon:
        return None

    centroid = np.sum(points * weights[:, None], axis=0) / weight_sum
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    negative_tolerance = max(
        1e-12,
        64.0 * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(eigenvalues)))),
    )
    if eigenvalues[-1] < -negative_tolerance:
        return None
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return {
        "weight_sum": weight_sum,
        "centroid_mm": centroid,
        "covariance_mm2": covariance,
        "eigenvalues_mm2": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def bilateral_filter_ced(
    points_mm,
    ced_score,
    ced_valid_mask,
    radius_mm,
    sigma_spatial_mm,
    sigma_ced,
    *,
    point_tree=None,
):
    """Simultaneously bilateral-filter a normalized CED scalar field."""
    points = np.asarray(points_mm, dtype=np.float64)
    source = np.asarray(ced_score, dtype=np.float64).reshape(-1)
    valid = np.asarray(ced_valid_mask, dtype=bool).reshape(-1)
    radius = float(radius_mm)
    spatial_sigma = float(sigma_spatial_mm)
    range_sigma = float(sigma_ced)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if len(source) != len(points) or len(valid) != len(points):
        raise ValueError("CED score and validity mask must align with points_mm")
    if not np.isfinite(source).all() or np.any(source < 0.0) or np.any(source > 1.0):
        raise ValueError("ced_score must be finite and in [0, 1]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and > 0")
    if not np.isfinite(spatial_sigma) or spatial_sigma <= 0.0:
        raise ValueError("sigma_spatial_mm must be finite and > 0")
    if not np.isfinite(range_sigma) or range_sigma <= 0.0:
        raise ValueError("sigma_ced must be finite and > 0")

    tree = point_tree if point_tree is not None else cKDTree(points)
    radius2 = radius * radius
    spatial_denominator = 2.0 * spatial_sigma * spatial_sigma
    range_denominator = 2.0 * range_sigma * range_sigma
    filtered = np.zeros(len(points), dtype=np.float64)
    neighbor_count = np.zeros(len(points), dtype=np.int32)
    weight_sum = np.zeros(len(points), dtype=np.float64)

    # Every output reads only ``source``. No already-filtered value is reused.
    for query_index, query in enumerate(points):
        if not valid[query_index]:
            continue
        neighbors = np.asarray(tree.query_ball_point(query, radius), dtype=np.int64)
        delta = points[neighbors] - query
        distance2 = np.einsum("ij,ij->i", delta, delta)
        inside = (distance2 <= radius2) & valid[neighbors]
        neighbors = neighbors[inside]
        distance2 = distance2[inside]
        neighbor_count[query_index] = len(neighbors)
        score_delta = source[neighbors] - source[query_index]
        weights = np.exp(-distance2 / spatial_denominator) * np.exp(
            -(score_delta * score_delta) / range_denominator
        )
        total = float(np.sum(weights))
        # A valid query is included with distance/score delta zero, so this is
        # only a numerical guard rather than a fallback smoothing policy.
        if not np.isfinite(total) or total <= 0.0:
            filtered[query_index] = source[query_index]
            weight_sum[query_index] = 1.0
            continue
        filtered[query_index] = np.clip(
            np.dot(weights, source[neighbors]) / total,
            0.0,
            1.0,
        )
        weight_sum[query_index] = total

    return {
        "score": filtered,
        "valid_mask": valid.copy(),
        "neighbor_count": neighbor_count,
        "weight_sum": weight_sum,
    }


def dense_ced_weighted_pca_scores(
    points_mm,
    ced_radius_mm,
    pca_radius_mm,
    min_neighbors,
    *,
    point_tree=None,
    bilateral_radius_mm=None,
    bilateral_sigma_spatial_mm=None,
    bilateral_sigma_ced=0.05,
    pca_ratio_saturation_tau=None,
):
    """Compute the whole-cloud Bilateral-CED × transformed-PCA-ratio score.

    Every CED/Bilateral-valid point is a PCA query. PCA support contains every
    in-radius sample with valid, positive original CED, and original CED is the
    covariance weight. The detector score is Bilateral CED times the raw PCA
    ratio, or its optional tanh saturation when ``tau`` is configured.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    ced_radius = float(ced_radius_mm)
    pca_radius = float(pca_radius_mm)
    minimum = int(min_neighbors)
    bilateral_radius = (
        ced_radius if bilateral_radius_mm is None else float(bilateral_radius_mm)
    )
    spatial_sigma = (
        0.5 * ced_radius
        if bilateral_sigma_spatial_mm is None
        else float(bilateral_sigma_spatial_mm)
    )
    range_sigma = float(bilateral_sigma_ced)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if not np.isfinite(ced_radius) or ced_radius <= 0.0:
        raise ValueError("ced_radius_mm must be finite and > 0")
    if not np.isfinite(pca_radius) or pca_radius <= 0.0:
        raise ValueError("pca_radius_mm must be finite and > 0")
    if minimum < 1:
        raise ValueError("min_neighbors must be >= 1")
    if not np.isfinite(bilateral_radius) or bilateral_radius <= 0.0:
        raise ValueError("bilateral_radius_mm must be finite and > 0")
    if not np.isfinite(spatial_sigma) or spatial_sigma <= 0.0:
        raise ValueError("bilateral_sigma_spatial_mm must be finite and > 0")
    if not np.isfinite(range_sigma) or range_sigma <= 0.0:
        raise ValueError("bilateral_sigma_ced must be finite and > 0")

    tree = point_tree if point_tree is not None else cKDTree(points)
    total_started = time.perf_counter()
    count = len(points)
    ced_radius2 = ced_radius * ced_radius
    pca_radius2 = pca_radius * pca_radius

    ced_valid = np.zeros(count, dtype=bool)
    ced_neighbor_count = np.zeros(count, dtype=np.int32)
    ced_centroid = points.copy()
    ced_distance = np.zeros(count, dtype=np.float64)
    ced = np.zeros(count, dtype=np.float64)

    ced_started = time.perf_counter()
    # Preserve the original CED numerical contract exactly: strict radius,
    # arithmetic centroid, query included, and radius-normalized distance.
    for query_index, query in enumerate(points):
        neighbors = np.asarray(tree.query_ball_point(query, ced_radius), dtype=np.int64)
        delta = points[neighbors] - query
        neighbors = neighbors[
            np.einsum("ij,ij->i", delta, delta) < ced_radius2
        ]
        ced_neighbor_count[query_index] = len(neighbors)
        if len(neighbors) < minimum:
            continue
        centroid = np.mean(points[neighbors], axis=0)
        distance = float(np.linalg.norm(query - centroid))
        ced_valid[query_index] = True
        ced_centroid[query_index] = centroid
        ced_distance[query_index] = distance
        ced[query_index] = np.clip(distance / ced_radius, 0.0, 1.0)
    ced_elapsed = time.perf_counter() - ced_started

    bilateral_started = time.perf_counter()
    bilateral_result = bilateral_filter_ced(
        points,
        ced,
        ced_valid,
        bilateral_radius,
        spatial_sigma,
        range_sigma,
        point_tree=tree,
    )
    bilateral = bilateral_result["score"]
    bilateral_valid = np.asarray(bilateral_result["valid_mask"], dtype=bool)
    bilateral_elapsed = time.perf_counter() - bilateral_started

    pca_query_mask = ced_valid & bilateral_valid

    pca_started = time.perf_counter()
    pca_valid = np.zeros(count, dtype=bool)
    pca_neighbor_count = np.zeros(count, dtype=np.int32)
    weighted_neighbor_count = np.zeros(count, dtype=np.int32)
    weight_sum = np.zeros(count, dtype=np.float64)
    effective_sample_size = np.zeros(count, dtype=np.float64)
    weighted_centroid = points.copy()
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    eigenvalues = np.zeros((count, 3), dtype=np.float64)
    eigenvectors = np.zeros((count, 3, 3), dtype=np.float64)
    pca_ratio_raw = np.zeros(count, dtype=np.float64)
    eigenvalue_epsilon = max(
        1e-12,
        64.0 * np.finfo(np.float64).eps * max(1.0, pca_radius2),
    )

    for query_index, query in enumerate(points):
        if not pca_query_mask[query_index]:
            continue
        neighbors = np.asarray(tree.query_ball_point(query, pca_radius), dtype=np.int64)
        delta = points[neighbors] - query
        neighbors = neighbors[
            np.einsum("ij,ij->i", delta, delta) <= pca_radius2
        ]
        pca_neighbor_count[query_index] = len(neighbors)
        usable = ced_valid[neighbors] & (ced[neighbors] > 0.0)
        neighbors = neighbors[usable]
        support_weights = ced[neighbors].copy()
        weighted_neighbor_count[query_index] = len(neighbors)
        if len(neighbors) < minimum:
            continue

        record = ced_weighted_pca(points[neighbors], support_weights)
        if record is None:
            continue
        values = record["eigenvalues_mm2"]
        if values[0] <= eigenvalue_epsilon:
            continue

        pca_valid[query_index] = True
        weight_sum[query_index] = record["weight_sum"]
        squared_weight_sum = float(np.dot(support_weights, support_weights))
        if squared_weight_sum > 0.0:
            effective_sample_size[query_index] = (
                record["weight_sum"] * record["weight_sum"]
                / squared_weight_sum
            )
        weighted_centroid[query_index] = record["centroid_mm"]
        covariance[query_index] = record["covariance_mm2"]
        eigenvalues[query_index] = values
        eigenvectors[query_index] = record["eigenvectors"]
        pca_ratio_raw[query_index] = np.clip(values[1] / values[0], 0.0, 1.0)
    pca_elapsed = time.perf_counter() - pca_started

    pca_ratio_saturated = saturate_pca_ratio(
        pca_ratio_raw,
        tau=pca_ratio_saturation_tau,
    )
    fusion_valid = pca_valid & bilateral_valid
    fusion_score = np.clip(bilateral * pca_ratio_saturated, 0.0, 1.0)
    fusion_score[~fusion_valid] = 0.0
    invalid_support = pca_query_mask & ~pca_valid
    total_elapsed = time.perf_counter() - total_started

    return {
        "ced_radius_mm": ced_radius,
        "pca_radius_mm": pca_radius,
        "min_neighbors": minimum,
        "bilateral_radius_mm": bilateral_radius,
        "bilateral_sigma_spatial_mm": spatial_sigma,
        "bilateral_sigma_ced": range_sigma,
        "pca_ratio_saturation_tau": pca_ratio_saturation_tau,
        "eigenvalue_epsilon_mm2": eigenvalue_epsilon,
        "ced_valid_mask": ced_valid,
        "pca_valid_mask": pca_valid,
        "fusion_valid_mask": fusion_valid,
        "pca_query_mask": pca_query_mask,
        "invalid_support_mask": invalid_support,
        "ced_neighbor_count": ced_neighbor_count,
        "pca_neighbor_count": pca_neighbor_count,
        "weighted_neighbor_count": weighted_neighbor_count,
        "original_point_indices": np.arange(count, dtype=np.int64),
        "query_position_mm": points.copy(),
        "ced_centroid_mm": ced_centroid,
        "ced_distance_mm": ced_distance,
        "ced_score": ced,
        "bilateral_ced_score": bilateral,
        "bilateral_valid_mask": bilateral_valid,
        "bilateral_neighbor_count": bilateral_result["neighbor_count"],
        "bilateral_weight_sum": bilateral_result["weight_sum"],
        "pca_weight_sum": weight_sum,
        "pca_effective_sample_size": effective_sample_size,
        "weighted_centroid_mm": weighted_centroid,
        "weighted_covariance_mm2": covariance,
        "eigenvalues_mm2": eigenvalues,
        "eigenvectors": eigenvectors,
        "pca_ratio_raw": pca_ratio_raw,
        "pca_ratio_saturated": pca_ratio_saturated,
        "fusion_score": fusion_score,
        "stage_elapsed_s": {
            "ced": ced_elapsed,
            "bilateral": bilateral_elapsed,
            "pca": pca_elapsed,
            "total": total_elapsed,
        },
    }


def weighted_pca_query_debug(score_result, query_index):
    """Return printable diagnostics for one whole-cloud heatmap query."""
    index = int(query_index)
    count = len(np.asarray(score_result["ced_score"]).reshape(-1))
    if not 0 <= index < count:
        raise IndexError("query_index is out of range")
    values = np.asarray(score_result["eigenvalues_mm2"])[index].copy()
    return {
        "query_index": index,
        "query_position_mm": np.asarray(
            score_result["query_position_mm"]
        )[index].copy(),
        "ced_score": float(score_result["ced_score"][index]),
        "bilateral_ced_score": float(
            score_result["bilateral_ced_score"][index]
        ),
        "fusion_score": float(score_result["fusion_score"][index]),
        "fusion_valid": bool(score_result["fusion_valid_mask"][index]),
        "pca_radius_mm": float(score_result["pca_radius_mm"]),
        "neighbor_count": int(score_result["pca_neighbor_count"][index]),
        "pca_requested": bool(score_result["pca_query_mask"][index]),
        "weighted_neighbor_count": int(
            score_result["weighted_neighbor_count"][index]
        ),
        "weight_sum": float(score_result["pca_weight_sum"][index]),
        "effective_sample_size": float(
            score_result["pca_effective_sample_size"][index]
        ),
        "weighted_centroid_mm": np.asarray(
            score_result["weighted_centroid_mm"]
        )[index].copy(),
        "valid": bool(score_result["pca_valid_mask"][index]),
        "invalid_support": bool(
            score_result["invalid_support_mask"][index]
        ),
        "eigenvalues_mm2": values,
        "pca_ratio_raw": float(score_result["pca_ratio_raw"][index]),
        "pca_ratio_saturated": float(
            score_result["pca_ratio_saturated"][index]
        ),
        "pca_ratio_saturation_tau": score_result["pca_ratio_saturation_tau"],
        "eigenvectors": np.asarray(score_result["eigenvectors"])[index].copy(),
        "candidate": bool(
            score_result.get("candidate_extraction", {}).get(
                "candidate_mask", np.zeros(count, dtype=bool)
            )[index]
        ),
    }


__all__ = [
    "saturate_pca_ratio",
    "ced_weighted_pca",
    "bilateral_filter_ced",
    "dense_ced_weighted_pca_scores",
    "weighted_pca_query_debug",
]
