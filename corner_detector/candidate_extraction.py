"""Dense local-maximum candidates and Fine-only greedy 3-D NMS."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _aligned_array(score_result, key, count, *, dtype=np.float64):
    values = np.asarray(score_result[key], dtype=dtype).reshape(-1)
    if len(values) != count:
        raise ValueError(f"{key} must align with points_mm")
    return values


def _candidate_record(
    points,
    score_result,
    index,
    radius,
    threshold,
    threshold_mode,
    configured_threshold,
):
    """Copy the score fields needed by a later geometric verifier."""
    count = len(points)
    ced = _aligned_array(score_result, "ced_score", count)
    bilateral = _aligned_array(score_result, "bilateral_ced_score", count)
    pca_raw = _aligned_array(score_result, "pca_ratio_raw", count)
    pca_saturated = _aligned_array(score_result, "pca_ratio_saturated", count)
    fusion = _aligned_array(score_result, "fusion_score", count)
    eigenvalues = np.asarray(
        score_result.get("eigenvalues_mm2", np.zeros((count, 3))),
        dtype=np.float64,
    )
    if eigenvalues.shape != (count, 3):
        raise ValueError("eigenvalues_mm2 must have shape (N, 3)")
    pca_valid = _aligned_array(
        score_result, "pca_valid_mask", count, dtype=bool,
    )
    fusion_valid = _aligned_array(
        score_result, "fusion_valid_mask", count, dtype=bool,
    )
    invalid_support = _aligned_array(
        score_result, "invalid_support_mask", count, dtype=bool,
    )
    pca_neighbor_count = np.asarray(
        score_result.get("pca_neighbor_count", np.zeros(count, dtype=np.int32)),
    ).reshape(-1)
    weighted_support_count = np.asarray(
        score_result.get("weighted_neighbor_count", np.zeros(count, dtype=np.int32)),
    ).reshape(-1)
    pca_weight_sum = np.asarray(
        score_result.get("pca_weight_sum", np.zeros(count, dtype=np.float64)),
    ).reshape(-1)
    effective_sample_size = np.asarray(
        score_result.get(
            "pca_effective_sample_size", np.zeros(count, dtype=np.float64),
        ),
    ).reshape(-1)
    for name, values in (
        ("pca_neighbor_count", pca_neighbor_count),
        ("weighted_neighbor_count", weighted_support_count),
        ("pca_weight_sum", pca_weight_sum),
        ("pca_effective_sample_size", effective_sample_size),
    ):
        if len(values) != count:
            raise ValueError(f"{name} must align with points_mm")
    return {
        "point_index": int(index),
        "position_mm": np.asarray(points[index], dtype=np.float64).copy(),
        "ced_score": float(ced[index]),
        "bilateral_ced_score": float(bilateral[index]),
        "pca_ratio_raw": float(pca_raw[index]),
        "pca_ratio_saturated": float(pca_saturated[index]),
        "pca_ratio_saturation_tau": score_result.get(
            "pca_ratio_saturation_tau"
        ),
        "fusion_score": float(fusion[index]),
        "eigenvalues_mm2": eigenvalues[index].copy(),
        "score": float(fusion[index]),
        "pca_valid": bool(pca_valid[index]),
        "fusion_valid": bool(fusion_valid[index]),
        "invalid_support": bool(invalid_support[index]),
        "pca_neighbor_count": int(pca_neighbor_count[index]),
        "weighted_support_count": int(weighted_support_count[index]),
        "pca_weight_sum": float(pca_weight_sum[index]),
        "pca_effective_sample_size": float(effective_sample_size[index]),
        "nms_radius_mm": float(radius),
        "fusion_score_threshold": float(threshold),
        "fusion_threshold_mode": str(threshold_mode),
        "configured_fusion_score_threshold": float(configured_threshold),
        "nms_method": "radius_local_maximum",
        "status": "candidate_unverified",
        "verified": False,
    }


def extract_corner_candidates(
    points_mm,
    score_result,
    nms_radius_mm,
    *,
    fusion_score_threshold=0.0,
    fusion_threshold_mode="fixed",
    fusion_peak_relative_ratio=0.5,
    point_tree=None,
):
    """Extract candidates by thresholding then deterministic local-max NMS.

    ``fixed`` uses ``fusion_score_threshold``. ``global_mean`` uses the
    arithmetic mean over all finite fusion-valid scores. ``peak_relative`` uses
    max(configured threshold, ratio * finite valid maximum). In all modes only
    strict ``score > threshold`` points are eligible. An eligible point is a
    local maximum exactly when no eligible point inside the inclusive 3-D
    radius has a higher score. Equal-score ties prefer the smaller original
    point index.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    radius = float(nms_radius_mm)
    configured_threshold = float(fusion_score_threshold)
    peak_ratio = float(fusion_peak_relative_ratio)
    threshold_mode = str(fusion_threshold_mode).strip().lower().replace("-", "_")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("nms_radius_mm must be finite and > 0")
    if not np.isfinite(configured_threshold) or not 0.0 <= configured_threshold <= 1.0:
        raise ValueError("fusion_score_threshold must be finite and in [0, 1]")
    if threshold_mode not in ("fixed", "global_mean", "peak_relative"):
        raise ValueError("fusion_threshold_mode must be fixed, global_mean or peak_relative")
    if not np.isfinite(peak_ratio) or not 0.0 <= peak_ratio <= 1.0:
        raise ValueError("fusion_peak_relative_ratio must be finite and in [0, 1]")

    count = len(points)
    scores = _aligned_array(score_result, "fusion_score", count)
    valid = _aligned_array(score_result, "fusion_valid_mask", count, dtype=bool)
    for key in (
        "ced_score", "bilateral_ced_score",
        "pca_ratio_raw", "pca_ratio_saturated",
    ):
        _aligned_array(score_result, key, count)

    threshold_source = valid & np.isfinite(scores)
    threshold_sample_count = int(np.count_nonzero(threshold_source))
    threshold = configured_threshold
    if threshold_sample_count:
        if threshold_mode == "global_mean":
            threshold = float(np.mean(scores[threshold_source]))
        elif threshold_mode == "peak_relative":
            threshold = max(configured_threshold, peak_ratio * float(np.max(scores[threshold_source])))
    eligible = threshold_source & (scores > threshold)
    eligible_indices = np.flatnonzero(eligible)

    tree = point_tree if point_tree is not None else cKDTree(points)
    radius2 = radius * radius
    selected = []
    for point_index in eligible_indices:
        point_index = int(point_index)
        neighbors = np.asarray(
            tree.query_ball_point(points[point_index], radius), dtype=np.int64,
        )
        delta = points[neighbors] - points[point_index]
        inside = np.einsum("ij,ij->i", delta, delta) <= radius2
        local = neighbors[inside]
        local = local[eligible[local]]
        higher = scores[local] > scores[point_index]
        tied_predecessor = (
            (scores[local] == scores[point_index]) & (local < point_index)
        )
        if not np.any(higher | tied_predecessor):
            selected.append(point_index)

    selected = np.asarray(selected, dtype=np.int64)
    if len(selected):
        order = np.lexsort((selected, -scores[selected]))
        selected = selected[order]

    candidate_indices = selected
    candidate_mask = np.zeros(count, dtype=bool)
    candidate_mask[candidate_indices] = True
    candidates = [
        _candidate_record(
            points,
            score_result,
            index,
            radius,
            threshold,
            threshold_mode,
            configured_threshold,
        )
        for index in candidate_indices
    ]
    return {
        "nms_radius_mm": radius,
        "fusion_score_threshold": threshold,
        "configured_fusion_score_threshold": configured_threshold,
        "fusion_threshold_mode": threshold_mode,
        "fusion_peak_relative_ratio": peak_ratio,
        "threshold_sample_count": threshold_sample_count,
        "nms_method": "radius_local_maximum",
        "ranking_score": "fusion_score",
        "total_point_count": int(count),
        "valid_score_count": int(np.count_nonzero(eligible)),
        "candidate_count": int(len(candidate_indices)),
        "eligible_score_mask": eligible,
        "candidate_mask": candidate_mask,
        "candidate_indices": candidate_indices,
        "candidates": candidates,
    }


def greedy_nms_indices(points_mm, scores, eligible_mask, radius_mm):
    """Suppress only around retained peaks; ties prefer the original array index."""
    points = np.asarray(points_mm, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if scores.shape != (len(points),) or eligible.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and eligibility must align with points_mm and be finite")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("NMS radius must be finite and positive")
    indices = np.flatnonzero(eligible)
    order = indices[np.lexsort((indices, -scores[indices]))]
    tree = cKDTree(points)
    suppressed = np.zeros(len(points), dtype=bool)
    selected = []
    for index in order:
        if suppressed[index]:
            continue
        selected.append(index)
        neighbors = np.asarray(tree.query_ball_point(points[index], radius_mm), dtype=np.int64)
        delta = points[neighbors] - points[index]
        neighbors = neighbors[np.einsum("ij,ij->i", delta, delta) <= radius_mm ** 2]
        suppressed[neighbors] = True
    return np.asarray(selected, dtype=np.int64)


__all__ = ["extract_corner_candidates", "greedy_nms_indices"]
