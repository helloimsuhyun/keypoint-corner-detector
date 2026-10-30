"""First-point-guided, count-scored sequential three-point RANSAC in mm."""

from time import perf_counter

import numpy as np

from .settings import GuidedValidatorConfig


def guided_probabilities(points_mm, query_mm, sigma_mm):
    """Normalize in log space; use uniform sampling if no finite weight survives."""
    relative = np.asarray(points_mm, dtype=float) - np.asarray(query_mm, dtype=float)
    if not np.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("Guided sigma must be finite and positive")
    if not len(relative):
        return np.empty(0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        log_weights = -0.5 * np.sum((relative / sigma_mm) ** 2, axis=1)
        weights = np.exp(log_weights - np.max(log_weights))
        total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.full(len(relative), 1.0 / len(relative))
    return weights / total


def extract_guided_planes(points_mm, query_mm, cfg: GuidedValidatorConfig, *, point_tree=None):
    """Hard-remove inliers after each accepted plane; all support IDs address the RANSAC voxel cloud."""
    started = perf_counter()
    cfg.validate()
    cloud = np.asarray(points_mm, dtype=np.float64)
    query = np.asarray(query_mm, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("query_mm must be a finite 3-vector")
    if point_tree is None:
        point_ids = np.flatnonzero(np.linalg.norm(cloud - query, axis=1) <= cfg.ransac_radius_mm)
    else:
        point_ids = np.sort(point_tree.query_ball_point(query, cfg.ransac_radius_mm))
        point_ids = np.asarray(point_ids, dtype=np.int64)
        point_ids = point_ids[np.linalg.norm(cloud[point_ids] - query, axis=1) <= cfg.ransac_radius_mm]
    points = cloud[point_ids]
    count = len(points)
    minimum = max(cfg.min_inliers, int(np.ceil(cfg.min_inlier_ratio * count)))
    result = {"query_position_mm": query.copy(), "radius_mm": cfg.ransac_radius_mm,
              "roi_ransac_indices": point_ids, "roi_points_mm": points.copy(),
              "min_required": minimum, "planes": []}
    remaining = np.arange(count)
    rng = np.random.default_rng(cfg.random_seed)
    for _ in range(cfg.max_planes):
        if len(remaining) < max(3, minimum):
            break
        active = points[remaining]
        probabilities = guided_probabilities(active, query, cfg.guided_sigma_mm)
        best = None
        for _ in range(cfg.ransac_iterations):
            a = int(rng.choice(len(active), p=probabilities))
            others = rng.choice(len(active) - 1, size=2, replace=False)
            b, c = others + (others >= a)
            va, vb = active[b] - active[a], active[c] - active[a]
            lengths = (np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(active[c] - active[b]))
            normal = np.cross(va, vb)
            norm = np.linalg.norm(normal)
            if min(lengths) <= 1e-9 or norm <= 1e-12 * max(1., lengths[0] * lengths[1]):
                continue
            normal /= norm
            offset = -float(normal @ active[a])
            residual = np.abs(active @ normal + offset)
            mask = residual < cfg.inlier_threshold_mm
            support = int(np.count_nonzero(mask))
            if support < minimum:
                continue
            rms = float(np.sqrt(np.mean(residual[mask] ** 2)))
            if best is not None and (support, -rms) <= (best["inlier_count"], -best["inlier_rms_mm"]):
                continue
            roi_mask = np.zeros(count, dtype=bool)
            roi_mask[remaining[mask]] = True
            best = {"normal": normal.copy(), "offset_mm": offset,
                    "inlier_mask": roi_mask, "inlier_ransac_indices": point_ids[roi_mask],
                    "inlier_count": support, "inlier_ratio": support / count,
                    "inlier_rms_mm": rms, "query_to_plane_distance_mm": float(abs(normal @ query + offset))}
        if best is None:
            break
        result["planes"].append(best)
        remaining = remaining[~best["inlier_mask"][remaining]]
    result["ransac_elapsed_s"] = perf_counter() - started
    return result
