"""Continuous 2-D boundary evidence for the plane-edge detector."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

def estimate_boundary_uv(
    points_uv,
    radius,
    epsilon,
    *,
    min_neighbors=6,
    evaluation_radius=None,
    crop_radius=None,
    voxel_mm=0.0,
    roi_center_uv=None,
    evaluation_point_mask=None,
):
    """Density-corrected second-order Euclidean boundary statistic.

    Only the final classification is restricted to the evaluation ROI.
    Directions and density counts use the entire padded support.
    Invalid or symmetric neighborhoods yield NaN distance.
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

    roi_center = (
        np.zeros(2, dtype=np.float64)
        if roi_center_uv is None
        else np.asarray(roi_center_uv, dtype=np.float64).reshape(2)
    )
    if not np.isfinite(roi_center).all():
        raise ValueError("roi_center_uv must be finite")
    radial_distance = np.linalg.norm(p - roi_center, axis=1)
    evaluate = np.ones(len(p), dtype=bool)
    if evaluation_point_mask is not None:
        supplied = np.asarray(evaluation_point_mask, dtype=bool).reshape(-1)
        if len(supplied) != len(p):
            raise ValueError("evaluation_point_mask must align with points_uv")
        evaluate &= supplied
    if evaluation_radius is not None:
        evaluate &= radial_distance <= evaluation_radius
    if crop_radius is not None:
        # x -> distance neighbor (r) -> direction neighbor (r) -> density (r/2).
        evaluate &= radial_distance + 2.5 * radius + np.sqrt(2.0) * voxel_mm <= crop_radius

    distance = np.full(len(p), np.nan)
    best_id = np.full(len(p), -1, dtype=np.int64)
    for i in np.flatnonzero(evaluate & valid):
        j = np.asarray(neighborhoods[i], dtype=np.int64)
        pair = np.broadcast_to(normals[i], (len(j), 2)).copy()
        same = valid[j] & ((normals[j] @ normals[i]) > 0.0)
        pair[same] = 0.5 * (normals[i] + normals[j[same]])
        scores = np.sum((p[i] - p[j]) * pair, axis=1)
        argmax = int(np.argmax(scores))
        distance[i] = max(float(scores[argmax]), 0.0)
        best_id[i] = int(j[argmax])

    threshold = 1.5 * epsilon
    boundary = evaluate & valid & np.isfinite(distance) & (distance < threshold)
    return {
        "distance_mm": distance,
        "inward_normal_uv": normals,
        "boundary_mask": boundary,
        "valid_direction": valid,
        "evaluation_mask": evaluate,
        "best_support_index": best_id,
        "density_count": density_count,
        "neighbor_count": count,
        "direction_imbalance": imbalance,
        "threshold_mm": threshold,
    }
__all__ = ["estimate_boundary_uv"]
