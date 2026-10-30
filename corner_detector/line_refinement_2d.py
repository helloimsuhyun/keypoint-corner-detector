"""Continuous 2-D Hough-line refinement and finite-segment suppression.

The optimizer owns only the infinite-line parameters ``rho`` and ``theta``.
Finite endpoints are derived afterwards from the stored continuous support
profile.  All coordinates and scales are in millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LineRefinement2DConfig:
    """Small set of controls; metric profile scales are estimated per plane."""

    enabled: bool = True
    max_iterations: int = 60
    final_nms_angle_deg: float = 5.0
    final_nms_overlap: float = 0.50

    def validate(self):
        if self.max_iterations < 1:
            raise ValueError("2-D line refinement max_iterations must be positive")
        if not (np.isfinite(self.final_nms_angle_deg)
                and 0.0 < self.final_nms_angle_deg < 90.0):
            raise ValueError("2-D final-NMS angle must be in (0, 90) degrees")
        if not (np.isfinite(self.final_nms_overlap)
                and 0.0 <= self.final_nms_overlap <= 1.0):
            raise ValueError("2-D final-NMS overlap must be in [0, 1]")


def _canonical_line(rho, theta, inside_sign=1):
    """Canonicalize ``(rho, theta) == (-rho, theta + pi)``."""
    rho = float(rho)
    theta = float(np.mod(theta, 2.0 * np.pi))
    sign = 1 if int(inside_sign) >= 0 else -1
    if theta >= np.pi:
        theta -= np.pi
        rho = -rho
        sign = -sign
    return rho, theta, sign


def _nearest_spacing(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 1.0
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    positive = distances[np.isfinite(distances) & (distances > 1e-7)]
    return float(np.median(positive)) if len(positive) else 1.0


def _otsu_threshold(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.inf
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= 1e-10:
        return low - 1e-12
    hist, edges = np.histogram(values, bins=128, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probability = hist.astype(np.float64) / max(1, int(np.sum(hist)))
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    total = mean[-1]
    denominator = omega * (1.0 - omega)
    variance = np.zeros_like(denominator)
    valid = denominator > 1e-15
    variance[valid] = (total * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(variance))])


def _longest_support_interval(grid, profile, threshold):
    mask = np.asarray(profile >= threshold, dtype=bool)
    if not np.any(mask):
        return None
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    candidates = []
    for start, stop in zip(starts, stops):
        # ``stop`` is exclusive. Prefer profile mass after interval length.
        mass = float(np.trapz(profile[start:stop], grid[start:stop])) if stop-start > 1 else 0.0
        candidates.append((stop-start, mass, int(start), int(stop)))
    _, _, start, stop = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 0.0
    return max(float(grid[0]), float(grid[start] - 0.5 * step)), min(
        float(grid[-1]), float(grid[stop-1] + 0.5 * step),
    ), start, stop


def _auto_scales(uv, boundary_weights, vote_mask, seeds, origin, core_mask,
                 rho_bin_mm):
    spacing = _nearest_spacing(uv)
    voter_ids = np.flatnonzero(vote_mask & (boundary_weights > 0.0))
    seed_residual = []
    if len(voter_ids):
        centered = uv[voter_ids] - origin
        for seed in seeds:
            theta = float(seed["theta_rad"])
            normal = np.array([np.cos(theta), np.sin(theta)])
            local_rho = float(seed.get("rho_mm", seed["rho"])) - float(normal @ origin)
            seed_residual.append(np.abs(centered @ normal - local_rho))
    positive_residual = (
        np.concatenate(seed_residual) if seed_residual else np.empty(0, dtype=float)
    )
    positive_residual = positive_residual[
        np.isfinite(positive_residual) & (positive_residual > 1e-7)
    ]
    residual_scale = (
        float(np.percentile(positive_residual, 25.0))
        if len(positive_residual) else 0.0
    )
    sigma_d = max(0.75 * spacing, 0.5 * float(rho_bin_mm), residual_scale)
    sigma_s = max(2.0 * spacing, 1.5 * sigma_d)
    h = 2.0 * sigma_d
    delta = 0.5 * sigma_d
    core_points = uv[np.asarray(core_mask, dtype=bool)]
    radial = np.linalg.norm((core_points if len(core_points) else uv) - origin, axis=1)
    radius = float(np.max(radial)) if len(radial) else max(spacing, 1.0)
    radius = max(radius, 2.0 * spacing)
    sample_step = max(spacing, (2.0 * radius) / 256.0)
    sample_count = max(17, int(np.ceil(2.0 * radius / sample_step)) + 1)
    grid = np.linspace(-radius, radius, sample_count)
    positive_weights = boundary_weights[voter_ids]
    representative_weight = (
        float(np.median(positive_weights)) if len(positive_weights) else 1.0
    )
    # Expected local 1-D edge mass and 2-D occupancy mass. These constants are
    # frozen once per plane so candidate scores remain directly comparable.
    c_b = max(
        0.25,
        0.5 * np.sqrt(2.0 * np.pi) * sigma_s / spacing * representative_weight,
    )
    c_o = max(0.25, np.pi * h * sigma_s / (spacing * spacing))
    return {
        "spacing_mm": spacing,
        "sigma_d_mm": sigma_d,
        "sigma_s_mm": sigma_s,
        "outside_h_mm": h,
        "inside_delta_mm": delta,
        "c_B": c_b,
        "c_O": c_o,
        "R_mm": radius,
        "s_mm": grid,
    }


def _profiles(centered_uv, weights, rho, theta, inside_sign, scales):
    normal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    distance = centered_uv @ normal - float(rho)
    along_coordinate = centered_uv @ tangent
    oriented_distance = int(inside_sign) * distance
    delta = float(scales["inside_delta_mm"])
    scaled = np.clip(oriented_distance / delta, -60.0, 60.0)
    inside = 1.0 / (1.0 + np.exp(-scaled))
    outside = 1.0 - inside
    sigma_d = float(scales["sigma_d_mm"])
    outside_h = float(scales["outside_h_mm"])
    boundary_cross = np.exp(-0.5 * (distance / sigma_d) ** 2)
    occupancy_cross = np.exp(-0.5 * (distance / outside_h) ** 2)
    grid = scales["s_mm"]
    sigma_s = float(scales["sigma_s_mm"])
    along = np.exp(-0.5 * ((grid[:, None] - along_coordinate[None, :]) / sigma_s) ** 2)
    boundary = along @ (weights * boundary_cross * inside)
    occupancy = along @ (occupancy_cross * outside)
    quality = (1.0 - np.exp(-boundary / float(scales["c_B"]))) * np.exp(
        -occupancy / float(scales["c_O"])
    )
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    smooth_quality = gaussian_filter1d(
        quality, sigma=max(1e-6, sigma_s / step), mode="nearest",
    )
    loss = -float(np.trapz(quality * smooth_quality, grid))
    return {
        "normal": normal,
        "direction": tangent,
        "distance_mm": distance,
        "along_mm": along_coordinate,
        "B": boundary,
        "O": occupancy,
        "q": quality,
        "q_smooth": smooth_quality,
        "loss": loss,
    }


def _optimize_seed(centered_uv, weights, seed, seed_index, inside_sign,
                   scales, config, rho_bound_mm, theta_bound_rad, origin):
    theta0 = float(seed["theta_rad"])
    normal0 = np.array([np.cos(theta0), np.sin(theta0)])
    rho0 = float(seed.get("rho_mm", seed["rho"])) - float(normal0 @ origin)
    start = np.array([rho0, theta0], dtype=np.float64)
    bounds = (
        (rho0 - rho_bound_mm, rho0 + rho_bound_mm),
        (theta0 - theta_bound_rad, theta0 + theta_bound_rad),
    )

    def objective(value):
        return _profiles(
            centered_uv, weights, value[0], value[1], inside_sign, scales,
        )["loss"]

    initial_loss = float(objective(start))
    path = [start.copy()]
    result = minimize(
        objective, start, method="L-BFGS-B", bounds=bounds,
        callback=lambda value: path.append(np.asarray(value, dtype=float).copy()),
        options={"maxiter": int(config.max_iterations), "ftol": 1e-10},
    )
    method = "L-BFGS-B"
    final = np.asarray(result.x, dtype=np.float64)
    final_loss = float(objective(final)) if np.isfinite(final).all() else np.inf
    unstable = (
        not bool(result.success) or not np.isfinite(final_loss)
        or final_loss >= initial_loss - 1e-8
    )
    if unstable:
        powell_path = [start.copy()]
        powell = minimize(
            objective, start, method="Powell", bounds=bounds,
            callback=lambda value: powell_path.append(np.asarray(value, dtype=float).copy()),
            options={"maxiter": int(config.max_iterations), "ftol": 1e-8, "xtol": 1e-6},
        )
        powell_final = np.asarray(powell.x, dtype=np.float64)
        powell_loss = (
            float(objective(powell_final)) if np.isfinite(powell_final).all() else np.inf
        )
        if powell_loss < final_loss or not np.isfinite(final_loss):
            result, final, final_loss = powell, powell_final, powell_loss
            path, method = powell_path, "Powell"
    if not np.array_equal(path[-1], final):
        path.append(final.copy())
    rho, theta, canonical_sign = _canonical_line(final[0], final[1], inside_sign)
    profiles = _profiles(centered_uv, weights, rho, theta, canonical_sign, scales)
    canonical_path = []
    for path_rho, path_theta in path:
        item_rho, item_theta, _ = _canonical_line(path_rho, path_theta, inside_sign)
        item_normal = np.array([np.cos(item_theta), np.sin(item_theta)])
        canonical_path.append([
            item_rho + float(item_normal @ origin), item_theta,
        ])
    normal = profiles["normal"]
    return {
        "seed_index": int(seed_index),
        "inside_sign": int(canonical_sign),
        "initial_rho_mm": float(seed.get("rho_mm", seed["rho"])),
        "initial_theta_rad": theta0,
        "final_rho_centered_mm": float(rho),
        "final_rho_mm": float(rho + normal @ origin),
        "final_theta_rad": float(theta),
        "initial_loss": initial_loss,
        "final_loss": float(profiles["loss"]),
        "iterations": int(getattr(result, "nit", max(0, len(path)-1))),
        "success": bool(result.success) and np.isfinite(profiles["loss"]),
        "message": str(result.message),
        "optimizer": method,
        "optimization_path": np.asarray(canonical_path, dtype=np.float64),
        "profiles": profiles,
    }


def _extract_segment(candidate, uv, origin, vote_mask, scales,
                     min_points, min_length_mm):
    profiles = candidate["profiles"]
    grid = scales["s_mm"]
    threshold = _otsu_threshold(profiles["q_smooth"])
    interval = _longest_support_interval(grid, profiles["q_smooth"], threshold)
    candidate = {key: value for key, value in candidate.items() if key != "profiles"}
    candidate.update({
        "s_mm": grid.copy(),
        "B": profiles["B"].copy(),
        "O": profiles["O"].copy(),
        "q": profiles["q"].copy(),
        "q_smooth": profiles["q_smooth"].copy(),
        "support_threshold": float(threshold),
        "score_pass": False,
        "rejection_reason": None,
    })
    if interval is None:
        candidate["rejection_reason"] = "no_continuous_support"
        return candidate
    lower, upper, start, stop = interval
    normal = profiles["normal"]
    direction = profiles["direction"]
    point = origin + candidate["final_rho_centered_mm"] * normal
    start_uv = point + lower * direction
    end_uv = point + upper * direction
    support = (
        np.asarray(vote_mask, dtype=bool)
        & (np.abs(profiles["distance_mm"]) <= 2.0 * float(scales["sigma_d_mm"]))
        & (profiles["along_mm"] >= lower)
        & (profiles["along_mm"] <= upper)
    )
    support_ids = np.flatnonzero(support)
    score = (
        float(np.trapz(
            profiles["q"][start:stop] * profiles["q_smooth"][start:stop],
            grid[start:stop],
        )) if stop-start > 1 else 0.0
    )
    mean_quality = float(np.mean(profiles["q_smooth"][start:stop]))
    length = float(upper - lower)
    candidate.update({
        "normal": normal.copy(),
        "direction": direction.copy(),
        "point_uv": point,
        "start_uv": start_uv,
        "end_uv": end_uv,
        "interval_s_mm": np.array([lower, upper], dtype=np.float64),
        "length_mm": length,
        "edge_support_score": score,
        "mean_quality": mean_quality,
        "boundary_support_uv_indices": support_ids,
        "support_count": int(len(support_ids)),
    })
    if not candidate["success"]:
        candidate["rejection_reason"] = "optimizer_failed"
    elif len(support_ids) < int(min_points):
        candidate["rejection_reason"] = "insufficient_boundary_support"
    elif length < float(min_length_mm):
        candidate["rejection_reason"] = "segment_too_short"
    elif not np.isfinite(score) or score <= 0.0:
        candidate["rejection_reason"] = "nonpositive_edge_score"
    else:
        candidate["score_pass"] = True
    return candidate


def _segment_overlap(left, right):
    direction = np.asarray(left["direction"], dtype=np.float64)
    left_interval = sorted([
        float(np.asarray(left["start_uv"]) @ direction),
        float(np.asarray(left["end_uv"]) @ direction),
    ])
    right_interval = sorted([
        float(np.asarray(right["start_uv"]) @ direction),
        float(np.asarray(right["end_uv"]) @ direction),
    ])
    overlap = max(0.0, min(left_interval[1], right_interval[1])
                  - max(left_interval[0], right_interval[0]))
    denominator = max(1e-12, min(
        left_interval[1] - left_interval[0],
        right_interval[1] - right_interval[0],
    ))
    return overlap / denominator


def _duplicates(left, right, angle_deg, distance_mm, overlap_threshold):
    dot = abs(float(np.asarray(left["direction"]) @ np.asarray(right["direction"])))
    angle = float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))
    if angle > angle_deg:
        return False
    distances = []
    for source, target in ((left, right), (right, left)):
        midpoint = 0.5 * (np.asarray(source["start_uv"]) + np.asarray(source["end_uv"]))
        distances.append(abs(float(
            (midpoint - np.asarray(target["point_uv"])) @ np.asarray(target["normal"])
        )))
    if max(distances) > distance_mm:
        return False
    return _segment_overlap(left, right) >= overlap_threshold


def refine_hough_lines_2d(
    uv,
    boundary_weights,
    vote_mask,
    hough_seeds,
    *,
    roi_center_uv,
    core_mask,
    rho_bin_mm,
    hough_nms_rho_mm,
    hough_nms_angle_deg,
    hough_theta_step_deg,
    min_points,
    min_length_mm,
    config=None,
):
    """Refine Hough seeds, extract valid finite support, then run final NMS."""
    cfg = config if config is not None else LineRefinement2DConfig()
    cfg.validate()
    uv = np.asarray(uv, dtype=np.float64)
    weights = np.asarray(boundary_weights, dtype=np.float64).reshape(-1)
    vote_mask = np.asarray(vote_mask, dtype=bool).reshape(-1)
    core_mask = np.asarray(core_mask, dtype=bool).reshape(-1)
    origin = np.asarray(roi_center_uv, dtype=np.float64).reshape(2)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("uv must contain finite (N,2) points")
    if any(value.shape != (len(uv),) for value in (weights, vote_mask, core_mask)):
        raise ValueError("2-D refinement arrays must align with uv")
    seeds = list(hough_seeds)
    empty = {
        "enabled": bool(cfg.enabled), "parameters": {},
        "refined_segments_before_nms": [], "valid_segments": [],
        "refined_segments_after_nms": [], "score_filter_threshold": None,
        "final_nms_suppressed": 0,
    }
    if not cfg.enabled or not len(uv) or not seeds:
        return empty
    scales = _auto_scales(
        uv, weights, vote_mask, seeds, origin, core_mask, rho_bin_mm,
    )
    centered = uv - origin
    rho_bound = max(float(hough_nms_rho_mm), 4.0 * scales["sigma_d_mm"])
    theta_bound = np.deg2rad(min(45.0, max(
        float(hough_nms_angle_deg), 4.0 * float(hough_theta_step_deg),
    )))
    candidates = []
    for seed_index, seed in enumerate(seeds):
        alternatives = [
            _optimize_seed(
                centered, weights, seed, seed_index, inside_sign, scales, cfg,
                rho_bound, theta_bound, origin,
            )
            for inside_sign in (-1, 1)
        ]
        chosen = min(alternatives, key=lambda item: item["final_loss"])
        candidates.append(_extract_segment(
            chosen, uv, origin, vote_mask, scales, min_points, min_length_mm,
        ))

    # No cross-candidate numeric quality threshold is applied. Candidates that
    # passed optimizer/support/length/positive-score validity enter final NMS.
    filtered = [item for item in candidates if item.get("score_pass", False)]
    filtered.sort(key=lambda item: (
        -float(item["edge_support_score"]), -float(item["length_mm"]),
        int(item["seed_index"]),
    ))
    final = []
    suppressed = 0
    distance_threshold = 2.0 * float(scales["sigma_d_mm"])
    for item in filtered:
        duplicate = next((kept for kept in final if _duplicates(
            item, kept, cfg.final_nms_angle_deg, distance_threshold,
            cfg.final_nms_overlap,
        )), None)
        if duplicate is None:
            item["suppressed_by_seed_index"] = None
            final.append(item)
        else:
            item["suppressed_by_seed_index"] = int(duplicate["seed_index"])
            suppressed += 1
    parameters = {
        key: (value.copy() if isinstance(value, np.ndarray) else float(value))
        for key, value in scales.items()
    }
    parameters.update({
        "rho_search_half_width_mm": float(rho_bound),
        "theta_search_half_width_deg": float(np.rad2deg(theta_bound)),
        "final_nms_angle_deg": float(cfg.final_nms_angle_deg),
        "final_nms_distance_mm": distance_threshold,
        "final_nms_overlap": float(cfg.final_nms_overlap),
    })
    return {
        "enabled": True,
        "parameters": parameters,
        "refined_segments_before_nms": candidates,
        "valid_segments": filtered,
        "refined_segments_after_nms": final,
        "score_filter_threshold": None,
        "final_nms_suppressed": int(suppressed),
    }


__all__ = ["LineRefinement2DConfig", "refine_hough_lines_2d"]
