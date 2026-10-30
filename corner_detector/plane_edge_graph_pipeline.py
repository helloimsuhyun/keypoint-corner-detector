"""Plane -> projected 2-D finite Hough lines -> 3-D PCA edges -> plane graph.

Coordinates and tolerances are in millimeters. Scene indices are retained through
projection and voting; no synthetic/back-projected points enter 3-D fitting.
This is the default query branch. Standalone geometric callers without padded
projection membership retain the Alpha-shape fallback path.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time

import numpy as np
from scipy.spatial import Delaunay, cKDTree, QhullError

try:  # Also supports the repository's direct-file numerical tests.
    from .corner_2d import estimate_boundary_uv
    from .line_refinement_2d import LineRefinement2DConfig, refine_hough_lines_2d
    from .boundary_support import boundary_weights
except ImportError:  # pragma: no cover - direct module loading compatibility
    from corner_detector.corner_2d import estimate_boundary_uv
    from corner_detector.line_refinement_2d import (
        LineRefinement2DConfig,
        refine_hough_lines_2d,
    )
    from corner_detector.boundary_support import boundary_weights


@dataclass(frozen=True)
class PlaneEdgeGraphConfig:
    plane_inlier_mm: float = 2.0
    alpha_radius_mm: float | None = None  # None = adaptive median spacing * alpha_factor
    alpha_factor: float = 2.5
    hough_theta_step_deg: float = 2.0
    hough_rho_bin_mm: float = 1.5
    hough_inlier_mm: float = 2.0
    hough_min_mean_weight: float = 0.0
    hough_score_threshold_mode: str = "off"
    hough_score_threshold: float = 0.0
    hough_nms_mode: str = "greedy"
    hough_nms_angle_deg: float = 5.0
    hough_nms_rho_mm: float = 2.0
    hough_min_votes: int = 7
    hough_max_peaks: int = 36
    line_refinement_enabled: bool = True
    line_refinement_max_iterations: int = 60
    line_refinement_final_nms_angle_deg: float = 5.0
    line_refinement_final_nms_overlap: float = 0.50
    line_min_points: int = 7
    line_min_length_mm: float = 30.0
    segment_gap_mm: float = 25.0
    direction_tolerance_deg: float = 15.0
    merge_angle_deg: float = 10.0
    merge_distance_mm: float = 3.0
    merge_gap_mm: float = 7.0
    corner_min_angle_deg: float = 25.0
    corner_endpoint_gap_mm: float = 12.0
    corner_line_residual_mm: float = 3.0
    corner_merge_distance_mm: float = 10.0
    projection_halo_mm: float = 0.0
    boundary_radius_mm: float = 8.0
    boundary_epsilon_mm: float = 2.5
    boundary_sigma_mm: float = 2.5
    boundary_min_neighbors: int = 6
    # Retained as inert compatibility fields; visualization never changes math.
    debug_2d: bool = False
    debug_max_planes: int = 0

    def validate(self):
        lengths = (self.plane_inlier_mm, self.alpha_factor, self.hough_theta_step_deg,
                   self.hough_rho_bin_mm, self.hough_inlier_mm, self.line_min_length_mm,
                   self.hough_nms_angle_deg, self.hough_nms_rho_mm,
                   self.segment_gap_mm, self.merge_angle_deg,
                   self.merge_distance_mm, self.merge_gap_mm, self.direction_tolerance_deg,
                   self.corner_min_angle_deg, self.corner_endpoint_gap_mm,
                   self.corner_line_residual_mm, self.corner_merge_distance_mm,
                   self.boundary_radius_mm, self.boundary_epsilon_mm,
                   self.boundary_sigma_mm)
        if not all(np.isfinite(x) and x > 0 for x in lengths):
            raise ValueError('Plane/edge numerical scales must be finite and positive')
        if self.alpha_radius_mm is not None and not (np.isfinite(self.alpha_radius_mm) and self.alpha_radius_mm > 0):
            raise ValueError('alpha_radius_mm must be positive or None')
        if (self.hough_theta_step_deg > 30 or self.hough_nms_angle_deg >= 90
                or self.merge_angle_deg >= 90 or self.direction_tolerance_deg >= 90):
            raise ValueError('Angular tolerances are invalid')
        if self.corner_min_angle_deg >= 90:
            raise ValueError('corner_min_angle_deg must be <90')
        if self.hough_inlier_mm < self.hough_rho_bin_mm * 0.5:
            raise ValueError('Hough inlier tolerance must be at least half a rho bin')
        if not (np.isfinite(self.hough_min_mean_weight)
                and 0.0 <= self.hough_min_mean_weight <= 1.0):
            raise ValueError('hough_min_mean_weight must be in [0,1]')
        if self.hough_score_threshold_mode not in (
                'off', 'fixed', 'peak_relative', 'global_mean'):
            raise ValueError(
                'hough_score_threshold_mode must be off, fixed, peak_relative, or global_mean'
            )
        if not (np.isfinite(self.hough_score_threshold)
                and self.hough_score_threshold >= 0.0):
            raise ValueError('hough_score_threshold must be finite and nonnegative')
        if (self.hough_score_threshold_mode == 'peak_relative'
                and self.hough_score_threshold > 1.0):
            raise ValueError('peak-relative Hough score threshold must be in [0,1]')
        if self.hough_nms_mode not in ('greedy', 'local_max'):
            raise ValueError('hough_nms_mode must be greedy or local_max')
        if min(self.hough_min_votes, self.hough_max_peaks, self.line_min_points) < 2:
            raise ValueError('Hough/line counts must be at least 2')
        LineRefinement2DConfig(
            enabled=self.line_refinement_enabled,
            max_iterations=self.line_refinement_max_iterations,
            final_nms_angle_deg=self.line_refinement_final_nms_angle_deg,
            final_nms_overlap=self.line_refinement_final_nms_overlap,
        ).validate()
        if not np.isfinite(self.projection_halo_mm) or self.projection_halo_mm < 0:
            raise ValueError('projection_halo_mm must be finite and nonnegative')
        if self.boundary_min_neighbors < 3:
            raise ValueError('boundary_min_neighbors must be at least 3')


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    mag = float(np.linalg.norm(v))
    if mag < 1e-12:
        raise ValueError('Degenerate geometric direction')
    return v / mag


def _fit_plane(points, seed_normal, threshold):
    """Iterative trimmed TLS on points supporting a normal-seeded plane."""
    if len(points) < 3:
        return None
    normal = _normalize(seed_normal)
    center = np.median(points, axis=0)
    selected = np.abs((points - center) @ normal) <= threshold
    for _ in range(4):
        if np.count_nonzero(selected) < 3:
            return None
        center = np.mean(points[selected], axis=0)
        _, s, vt = np.linalg.svd(points[selected] - center, full_matrices=False)
        if len(s) < 3 or s[1] < 1e-6:
            return None
        normal = vt[-1]
        if normal @ seed_normal < 0:
            normal = -normal
        next_sel = np.abs((points - center) @ normal) <= threshold
        if np.array_equal(next_sel, selected):
            break
        selected = next_sel
    if np.count_nonzero(selected) < 3:
        return None
    center = np.mean(points[selected], axis=0)
    err = (points[selected] - center) @ normal
    return center, normal, selected, float(np.sqrt(np.mean(err * err)))


def _plane_basis(normal):
    normal = _normalize(normal)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = _normalize(np.cross(normal, axis))
    return u, np.cross(normal, u)


def _boundary_indices(uv, alpha_radius):
    """Alpha-complex boundary vertices; holes/dropouts stay explicitly observable."""
    if len(uv) < 4:
        return np.arange(len(uv), dtype=np.int64)
    # Delaunay cannot handle exactly coincident samples; deduplicate with inverse index.
    _, ix = np.unique(np.round(uv, decimals=7), axis=0, return_index=True)
    uv_unique = uv[np.sort(ix)]
    ids_unique = np.sort(ix)
    if len(uv_unique) < 4:
        return ids_unique
    try:
        triangles = Delaunay(uv_unique, qhull_options='Qbb Qc Qz Q12').simplices
    except QhullError:
        return np.empty(0, dtype=np.int64)
    tri = uv_unique[triangles]
    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    first = tri[:, 1] - tri[:, 0]
    second = tri[:, 2] - tri[:, 0]
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (a * b * c) / (2.0 * cross)
    kept = triangles[np.isfinite(r) & (r <= alpha_radius)]
    if len(kept) == 0:
        return np.empty(0, dtype=np.int64)
    edges = np.sort(np.concatenate((kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [0, 2]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return ids_unique[np.unique(edges[counts == 1])]


def _tls_line(points, *, dimension):
    if len(points) < 2:
        return None
    center = np.mean(points, axis=0)
    _, s, vt = np.linalg.svd(points - center, full_matrices=False)
    if s[0] <= 1e-10:
        return None
    direction = _normalize(vt[0])
    return center, direction


def _line_residuals(points, center, direction):
    delta = points - center
    return np.linalg.norm(delta - (delta @ direction)[:, None] * direction, axis=1)


def _hough_nms_ranges(angle_index, rho_index, theta_count, rho_count, base, config):
    """Yield seam-aware accumulator slices suppressed/compared by one peak."""
    theta_guard = max(1, int(np.ceil(
        config.hough_nms_angle_deg / config.hough_theta_step_deg
    )))
    rho_guard = max(1, int(np.ceil(
        config.hough_nms_rho_mm / config.hough_rho_bin_mm
    )))
    rho_value = base + (rho_index + 0.5) * config.hough_rho_bin_mm
    for delta_angle in range(-theta_guard, theta_guard + 1):
        raw_angle = angle_index + delta_angle
        wrapped = raw_angle < 0 or raw_angle >= theta_count
        mapped_angle = raw_angle % theta_count
        if wrapped:
            reflected = (
                (-base - rho_value) / config.hough_rho_bin_mm - 0.5
            )
            center_rho = int(round(reflected))
        else:
            center_rho = int(rho_index)
        lower = max(0, center_rho - rho_guard)
        upper = min(rho_count, center_rho + rho_guard + 1)
        if lower < upper:
            yield mapped_angle, lower, upper


def _local_maximum_hough_cells(score, eligible, base, config):
    """Return deterministic plateau representatives in the configured NMS window."""
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    theta_count, rho_count = score.shape
    maxima = []
    for angle_index, rho_index in np.argwhere(eligible):
        angle_index = int(angle_index)
        rho_index = int(rho_index)
        current_score = float(score[angle_index, rho_index])
        current_flat = angle_index * rho_count + rho_index
        is_maximum = True
        for mapped_angle, lower, upper in _hough_nms_ranges(
            angle_index, rho_index, theta_count, rho_count, base, config,
        ):
            for neighbor_rho in range(lower, upper):
                if not eligible[mapped_angle, neighbor_rho]:
                    continue
                neighbor_score = float(score[mapped_angle, neighbor_rho])
                neighbor_flat = mapped_angle * rho_count + neighbor_rho
                if (neighbor_score > current_score
                        or (neighbor_score == current_score
                            and neighbor_flat < current_flat)):
                    is_maximum = False
                    break
            if not is_maximum:
                break
        if is_maximum:
            maxima.append((angle_index, rho_index))
    maxima.sort(key=lambda cell: (-float(score[cell]), cell[0], cell[1]))
    return maxima


def _select_hough_peak_cells(score, eligible, base, config):
    """Select accumulator cells using greedy suppression or local maxima."""
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    local_maxima = _local_maximum_hough_cells(score, eligible, base, config)
    if config.hough_nms_mode == 'local_max':
        return local_maxima[:config.hough_max_peaks], len(local_maxima)

    theta_count, rho_count = score.shape
    blocked = np.zeros_like(eligible, dtype=bool)
    selected = []
    for _ in range(config.hough_max_peaks):
        available = eligible & ~blocked
        if not np.any(available):
            break
        ranked = np.where(available, score, -np.inf)
        angle_index, rho_index = np.unravel_index(np.argmax(ranked), ranked.shape)
        if not np.isfinite(ranked[angle_index, rho_index]):
            break
        angle_index = int(angle_index)
        rho_index = int(rho_index)
        selected.append((angle_index, rho_index))
        for mapped_angle, lower, upper in _hough_nms_ranges(
            angle_index, rho_index, theta_count, rho_count, base, config,
        ):
            blocked[mapped_angle, lower:upper] = True
    return selected, len(local_maxima)


def _apply_hough_score_threshold(score, base_eligible, config):
    """Apply the selected weighted-accumulator threshold after count/mean gates."""
    score = np.asarray(score, dtype=np.float64)
    base_eligible = np.asarray(base_eligible, dtype=bool)
    values = score[base_eligible]
    mode = config.hough_score_threshold_mode
    if mode == 'off':
        effective = 0.0
    elif mode == 'fixed':
        effective = float(config.hough_score_threshold)
    elif not len(values):
        effective = 0.0
    elif mode == 'peak_relative':
        effective = float(config.hough_score_threshold) * float(np.max(values))
    else:  # global_mean
        effective = float(np.mean(values))
    return base_eligible & (score >= effective), effective


def _hough_2d(uv, config):
    """Finite point-set Hough, one vote / point / angle / rho cell."""
    if len(uv) < config.hough_min_votes:
        return []
    theta = np.deg2rad(np.arange(0, 180, config.hough_theta_step_deg))
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    rho = uv @ normals.T
    # Fixed global rho offset guarantees stable bins across all angles.
    base = float(np.min(rho)) - config.hough_rho_bin_mm
    bin_id = np.floor((rho - base) / config.hough_rho_bin_mm).astype(np.int32)
    bins = int(np.max(bin_id)) + 1
    table = np.zeros((len(theta), bins), dtype=np.int32)
    ai = np.broadcast_to(np.arange(len(theta)), bin_id.shape)
    np.add.at(table, (ai.ravel(), bin_id.ravel()), 1)
    mean_weight = np.ones_like(table, dtype=np.float64)
    count_mean_eligible = (
        (table >= int(config.hough_min_votes))
        & (mean_weight >= float(config.hough_min_mean_weight))
    )
    qualified, _effective_threshold = _apply_hough_score_threshold(
        table.astype(np.float64), count_mean_eligible, config,
    )
    cells, _local_maximum_count = _select_hough_peak_cells(
        table.astype(np.float64), qualified, base, config,
    )
    peaks = []
    for a, r in cells:
        # Use an expanded stripe around the selected bin before 2-D TLS.
        signed = rho[:, a] - (base + (r + 0.5) * config.hough_rho_bin_mm)
        members = np.flatnonzero(np.abs(signed) <= config.hough_inlier_mm)
        if len(members) >= config.hough_min_votes:
            peaks.append({'normal': normals[a].copy(), 'rho': base + (r + 0.5) * config.hough_rho_bin_mm,
                          'vote_count': int(table[a, r]), 'indices': members})
    return peaks


def _weighted_hough_2d(uv, weights, vote_mask, config):
    """Stored weighted Hough result over core boundary samples only."""
    uv = np.asarray(uv, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    vote_mask = np.asarray(vote_mask, dtype=bool).reshape(-1)
    if weights.shape != (len(uv),) or vote_mask.shape != (len(uv),):
        raise ValueError("Hough weights and vote mask must align with UV")
    voter_ids = np.flatnonzero(vote_mask & np.isfinite(weights) & (weights > 0.0))
    theta = np.deg2rad(np.arange(0, 180, config.hough_theta_step_deg))
    if len(theta) == 0 or len(voter_ids) < config.hough_min_votes:
        return {
            "accumulator": np.zeros((len(theta), 0), dtype=float),
            "count_accumulator": np.zeros((len(theta), 0), dtype=np.int32),
            "mean_weight_accumulator": np.zeros((len(theta), 0), dtype=float),
            "rho_mm": np.empty(0, dtype=float), "theta_rad": theta,
            "vote_mask": vote_mask, "boundary_weights": weights,
            "hough_peaks_after_nms": [], "orientation_guided": False,
            "orientation_window_deg": 90.0,
            "hough_min_votes": int(config.hough_min_votes),
            "hough_min_mean_weight": float(config.hough_min_mean_weight),
            "hough_score_threshold_mode": config.hough_score_threshold_mode,
            "hough_score_threshold_configured": float(config.hough_score_threshold),
            "hough_score_threshold_effective": 0.0,
            "hough_nms_mode": config.hough_nms_mode,
            "hough_nms_angle_deg": float(config.hough_nms_angle_deg),
            "hough_nms_rho_mm": float(config.hough_nms_rho_mm),
            "peak_count_above_threshold": 0,
            "peak_count_after_count_mean_gate": 0,
            "peak_count_local_maximum": 0,
            "peak_count_after_nms": 0,
            "peak_count_before_support": 0,
        }
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    voter_uv = uv[voter_ids]
    projected = voter_uv @ normals.T
    base = float(np.min(projected)) - config.hough_rho_bin_mm
    bin_id = np.floor((projected - base) / config.hough_rho_bin_mm).astype(np.int32)
    bins = int(np.max(bin_id)) + 1
    weighted = np.zeros((len(theta), bins), dtype=np.float64)
    counts = np.zeros((len(theta), bins), dtype=np.int32)
    ai = np.broadcast_to(np.arange(len(theta)), bin_id.shape)
    np.add.at(weighted, (ai.ravel(), bin_id.ravel()),
              np.repeat(weights[voter_ids], len(theta)))
    np.add.at(counts, (ai.ravel(), bin_id.ravel()), 1)
    mean_weight = np.zeros_like(weighted)
    np.divide(weighted, counts, out=mean_weight, where=counts > 0)
    count_mean_eligible = (
        (counts >= int(config.hough_min_votes))
        & (mean_weight >= float(config.hough_min_mean_weight))
    )
    qualified, effective_threshold = _apply_hough_score_threshold(
        weighted, count_mean_eligible, config,
    )
    after_count_mean = int(np.count_nonzero(count_mean_eligible))
    above = int(np.count_nonzero(qualified))
    cells, local_maximum_count = _select_hough_peak_cells(
        weighted, qualified, base, config,
    )
    before_support = len(cells)
    peaks = []
    for a, r in cells:
        rho_value = base + (r + 0.5) * config.hough_rho_bin_mm
        signed = projected[:, a] - rho_value
        local = np.flatnonzero(np.abs(signed) <= config.hough_inlier_mm)
        members = voter_ids[local]
        if len(members) >= config.hough_min_votes:
            peaks.append({
                "normal": normals[a].copy(), "rho": float(rho_value),
                "rho_mm": float(rho_value), "theta_rad": float(theta[a]),
                "vote_count": int(counts[a, r]),
                "weighted_vote": float(weighted[a, r]),
                "mean_weight": float(mean_weight[a, r]),
                "indices": members,
            })
    rho_centers = base + (np.arange(bins) + 0.5) * config.hough_rho_bin_mm
    return {
        "accumulator": weighted, "count_accumulator": counts,
        "mean_weight_accumulator": mean_weight,
        "rho_mm": rho_centers, "theta_rad": theta,
        "vote_mask": vote_mask, "boundary_weights": weights,
        "hough_peaks_after_nms": peaks, "orientation_guided": False,
        "orientation_window_deg": 90.0,
        "hough_min_votes": int(config.hough_min_votes),
        "hough_min_mean_weight": float(config.hough_min_mean_weight),
        "hough_score_threshold_mode": config.hough_score_threshold_mode,
        "hough_score_threshold_configured": float(config.hough_score_threshold),
        "hough_score_threshold_effective": float(effective_threshold),
        "hough_nms_mode": config.hough_nms_mode,
        "hough_nms_angle_deg": float(config.hough_nms_angle_deg),
        "hough_nms_rho_mm": float(config.hough_nms_rho_mm),
        "peak_count_above_threshold": above,
        "peak_count_after_count_mean_gate": after_count_mean,
        "peak_count_local_maximum": local_maximum_count,
        "peak_count_after_nms": len(peaks),
        "peak_count_before_support": before_support,
    }


def _split_by_gaps(indices, uv, direction, max_gap):
    if not len(indices):
        return []
    values = uv[indices] @ direction
    ordered = indices[np.argsort(values)]
    gaps = np.diff(np.sort(values))
    return np.split(ordered, np.flatnonzero(gaps > max_gap) + 1)


def _pca_line_3d(points, config, reference_direction):
    """Fit one 3-D PCA/TLS line to every observed point in a 2-D segment."""
    if len(points) < config.line_min_points:
        return None
    fit = _tls_line(points, dimension=3)
    if fit is None:
        return None
    center, direction = fit
    cos_max = np.cos(np.deg2rad(config.direction_tolerance_deg))
    if abs(direction @ reference_direction) < cos_max:
        return None
    rms = float(np.sqrt(np.mean(_line_residuals(points, center, direction) ** 2)))
    ts = (points - center) @ direction
    span = float(np.max(ts) - np.min(ts))
    if span < config.line_min_length_mm:
        return None
    return {'center_mm': center, 'direction': direction,
            'start_mm': center + float(np.min(ts)) * direction,
            'end_mm': center + float(np.max(ts)) * direction,
            'length_mm': span, 'rms_mm': rms}


def _finite_segment_match(a, b, cfg):
    d, e = a['direction'], b['direction']
    if abs(float(d @ e)) < np.cos(np.deg2rad(cfg.merge_angle_deg)):
        return False
    # Symmetric line-to-support distance: avoids skew lines merging at a crossing.
    for src, dst in ((a, b), (b, a)):
        mid = 0.5 * (src['start_mm'] + src['end_mm'])
        if float(_line_residuals(mid[None, :], dst['center_mm'], dst['direction'])[0]) > cfg.merge_distance_mm:
            return False
    ta = sorted([float(a['start_mm'] @ d), float(a['end_mm'] @ d)])
    tb = sorted([float(b['start_mm'] @ d), float(b['end_mm'] @ d)])
    return min(ta[1], tb[1]) + cfg.merge_gap_mm >= max(ta[0], tb[0])


def _merge_edges(proposals, points, cfg):
    """Greedy complete-link grouping prevents transitive merging of unrelated edges."""
    sorted_edges = sorted(proposals, key=lambda x: (-len(x['support_indices']), x['plane_id'], -x['length_mm']))
    groups = []
    for edge in sorted_edges:
        matches = [g for g in groups if all(_finite_segment_match(edge, other, cfg) for other in g)]
        if matches:
            matches[0].append(edge)
        else:
            groups.append([edge])
    merged = []
    for group in groups:
        ids = np.unique(np.concatenate([x['support_indices'] for x in group]))
        planes = sorted(set(int(x['plane_id']) for x in group))
        fit = _tls_line(points[ids], dimension=3)
        if fit is None:
            continue
        center, direction = fit
        ts = (points[ids] - center) @ direction
        start = center + float(np.min(ts)) * direction
        end = center + float(np.max(ts)) * direction
        rms = float(np.sqrt(np.mean(_line_residuals(points[ids], center, direction) ** 2)))
        merged.append({
            'edge_id': len(merged), 'plane_ids': planes,
            'member_proposal_ids': [x['proposal_id'] for x in group],
            'support_indices': ids, 'support_count': int(len(ids)),
            'center_mm': center, 'direction': direction,
            'start_mm': start, 'end_mm': end, 'length_mm': float(np.max(ts)-np.min(ts)),
            'rms_mm': rms, 'shared': len(planes) >= 2,
        })
    return merged



def _single_plane_corners(edges, planes, cfg):
    """Corners supported by two distinct finite boundary edges on ONE observed plane.

    These are *candidates*, not guaranteed physical 3-D convex corners: a depth
    occlusion may also create a 2-D boundary junction. Intersect two supporting
    lines in the fitted plane, and reject near-parallel lines, extrapolated
    intersections, skew 3-D lines and repeated junctions.
    """
    raw = []
    for plane in planes:
        plane_id = int(plane['plane_id'])
        center = plane['center_mm']; u = plane['u']; v = plane['v']
        eligible = [edge for edge in edges if plane_id in edge['plane_ids']]
        for left, right in combinations(eligible, 2):
            def uv3(point):
                d = np.asarray(point, dtype=np.float64) - center
                return np.asarray([d @ u, d @ v])
            a0, a1 = uv3(left['start_mm']), uv3(left['end_mm'])
            b0, b1 = uv3(right['start_mm']), uv3(right['end_mm'])
            da, db = a1-a0, b1-b0
            nda, ndb = np.linalg.norm(da), np.linalg.norm(db)
            if min(nda, ndb) < 1e-9:
                continue
            da, db = da/nda, db/ndb
            dot = np.clip(abs(float(da @ db)), 0.0, 1.0)
            angle = float(np.degrees(np.arccos(dot)))  # acute angle, orientation invariant
            if angle < cfg.corner_min_angle_deg:
                continue
            matrix = np.column_stack((da, -db))
            if abs(float(np.linalg.det(matrix))) < 1e-9:
                continue
            time_ab = np.linalg.solve(matrix, b0-a0)
            xy = a0 + time_ab[0]*da
            xyz = center + xy[0]*u + xy[1]*v
            # The junction should be close to the finite endpoints of BOTH edges;
            # intersecting infinite supporting lines is not sufficient.
            endpoint_a = min(np.linalg.norm(xyz-left['start_mm']), np.linalg.norm(xyz-left['end_mm']))
            endpoint_b = min(np.linalg.norm(xyz-right['start_mm']), np.linalg.norm(xyz-right['end_mm']))
            if max(endpoint_a, endpoint_b) > cfg.corner_endpoint_gap_mm:
                continue
            line_error_a = float(_line_residuals(xyz[None, :], left['center_mm'], left['direction'])[0])
            line_error_b = float(_line_residuals(xyz[None, :], right['center_mm'], right['direction'])[0])
            if max(line_error_a, line_error_b) > cfg.corner_line_residual_mm:
                continue
            raw.append({
                'plane_id': plane_id, 'edge_ids': [int(left['edge_id']), int(right['edge_id'])],
                'position_mm': xyz, 'angle_deg': angle,
                'endpoint_gap_mm': float(max(endpoint_a, endpoint_b)),
                'line_residual_mm': max(line_error_a, line_error_b),
                'support_count': int(left['support_count'] + right['support_count']),
                'type': 'single_plane_boundary_corner_candidate',
            })
    # Suppress repeat hypotheses generated by fragmented/duplicate Hough segments.
    raw.sort(key=lambda c: (c['line_residual_mm'], c['endpoint_gap_mm'], -c['support_count']))
    clusters = []
    for candidate in raw:
        match = next((g for g in clusters
                      if g['plane_id'] == candidate['plane_id']
                      and np.linalg.norm(g['position_mm'] - candidate['position_mm'])
                      <= cfg.corner_merge_distance_mm), None)
        if match is None:
            candidate['corner_id'] = len(clusters)
            clusters.append(candidate)
        else:
            match['edge_ids'] = sorted(set(match['edge_ids'] + candidate['edge_ids']))
    return clusters


def analyze_plane_edge_graph(points_mm, plane_analysis, config=None):
    """Run on an existing analyze_query_plane_clusters result (scene-index preserving)."""
    cfg = config if config is not None else PlaneEdgeGraphConfig()
    cfg.validate()
    start = time.perf_counter()
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite 3-D scene points')
    planes, proposals = [], []
    statistics = {'planes_rejected': 0, 'boundary_points': 0, 'hough_peaks': 0,
                  '3d_rejected': 0, '3d_accepted': 0, '2d_segments': 0,
                  '2d_short_segments': 0, 'refined_before_nms': 0,
                  'refined_valid': 0, 'refined_after_nms': 0,
                  'refined_final_nms_suppressed': 0}
    roi = np.unique(np.asarray(plane_analysis.get('roi_point_indices', np.arange(len(points))), dtype=np.int64))
    if np.any(roi < 0) or np.any(roi >= len(points)):
        raise ValueError('Invalid ROI scene index')
    if plane_analysis.get("search_scope") == "scene":
        raise ValueError("Plane-Edge Graph cannot accept scene-global plane clusters")
    # Production query analysis carries the radius and clicked query position.
    # Keep standalone geometric tests compatible when both are absent.
    center_query = plane_analysis.get("query_position_mm")
    roi_radius = plane_analysis.get("query_radius_mm")
    if center_query is not None and roi_radius is not None:
        q = np.asarray(center_query, dtype=np.float64).reshape(3)
        if not np.isfinite(roi_radius) or roi_radius <= 0:
            raise ValueError("Invalid query radius")
        if np.any(np.linalg.norm(points[roi] - q, axis=1) > float(roi_radius) + 1e-6):
            raise ValueError("3D plane projection ROI contains points outside query radius")
    in_roi = np.zeros(len(points), dtype=bool)
    in_roi[roi] = True
    projection_support = np.unique(np.asarray(
        plane_analysis.get('projection_support_indices', roi), dtype=np.int64,
    ))
    if np.any(projection_support < 0) or np.any(projection_support >= len(points)):
        raise ValueError('Invalid projection support scene index')
    projection_halo = float(plane_analysis.get('projection_halo_mm', 0.0))
    if center_query is not None and roi_radius is not None:
        support_distance = np.linalg.norm(points[projection_support] - q, axis=1)
        if np.any(support_distance > float(roi_radius) + projection_halo + 1e-6):
            raise ValueError("Projection support exceeds the declared padded radius")
    for raw_plane in plane_analysis['global_planes']:
        plane_id = int(raw_plane['plane_id'])
        original_ids = np.unique(np.asarray(raw_plane['support_indices'], dtype=np.int64))
        if len(original_ids) < cfg.line_min_points or np.any(original_ids < 0) or np.any(original_ids >= len(points)):
            statistics['planes_rejected'] += 1
            continue
        if not np.all(in_roi[original_ids]):
            raise ValueError("Cluster plane contains support outside query ROI")
        seed = np.asarray(raw_plane['representative_normal'], dtype=np.float64)
        fit = _fit_plane(points[original_ids], seed, cfg.plane_inlier_mm)
        if fit is None:
            statistics['planes_rejected'] += 1
            continue
        center, normal, mask, plane_rms = fit
        plane_fit_ids = original_ids[mask]
        if len(plane_fit_ids) < cfg.line_min_points:
            continue
        u, v = _plane_basis(normal)
        has_plane_membership_context = "projection_indices" in raw_plane
        use_padded_boundary = has_plane_membership_context
        candidate_pool = (
            np.asarray(raw_plane["projection_indices"], dtype=np.int64)
            if has_plane_membership_context
            else roi
        )
        # CornerVINS supplies explicit organized-cloud plane membership,
        # including a query-padding halo. Standalone geometric tests retain
        # their fitted-plane support fallback.
        ids = candidate_pool if has_plane_membership_context else plane_fit_ids
        mode = (
            'padded_cornervins_membership'
            if has_plane_membership_context
            else 'plane_fit_support'
        )
        if len(ids):
            uv = np.column_stack(((points[ids] - center) @ u, (points[ids] - center) @ v))
        else:
            uv = np.empty((0, 2), dtype=float)
        core_mask = in_roi[ids]
        query_uv = (
            np.asarray([(q - center) @ u, (q - center) @ v], dtype=float)
            if center_query is not None else np.zeros(2, dtype=float)
        )
        if len(uv) >= 2:
            nearest = cKDTree(uv).query(uv, k=2)[0][:, 1]
            spacing = float(np.median(nearest[nearest > 1e-7])) if np.any(nearest > 1e-7) else 1.0
        else:
            spacing = 1.0
        alpha_radius = cfg.alpha_radius_mm or (cfg.alpha_factor * spacing)
        if use_padded_boundary and len(uv):
            boundary = estimate_boundary_uv(
                uv, cfg.boundary_radius_mm, cfg.boundary_epsilon_mm,
                min_neighbors=cfg.boundary_min_neighbors,
                roi_center_uv=query_uv,
                evaluation_point_mask=core_mask,
            )
            continuous_weight = boundary_weights(
                boundary['distance_mm'], boundary['valid_direction'],
                cfg.boundary_sigma_mm,
            )
            vote_mask = core_mask & boundary['boundary_mask']
            line_vote = _weighted_hough_2d(uv, continuous_weight, vote_mask, cfg)
            peaks = line_vote['hough_peaks_after_nms']
            boundary_idx = np.flatnonzero(boundary['boundary_mask'])
        else:
            boundary_idx = _boundary_indices(uv, alpha_radius)
            fallback_mask = np.zeros(len(uv), dtype=bool)
            fallback_mask[boundary_idx] = True
            boundary = {
                'distance_mm': np.where(fallback_mask, 0.0, np.nan),
                'inward_normal_uv': np.zeros_like(uv),
                'boundary_mask': fallback_mask,
                'valid_direction': fallback_mask.copy(),
                'evaluation_mask': np.ones(len(uv), dtype=bool),
                'best_support_index': np.full(len(uv), -1, dtype=np.int64),
                'density_count': np.zeros(len(uv), dtype=np.int64),
                'neighbor_count': np.zeros(len(uv), dtype=np.int64),
                'direction_imbalance': np.zeros(len(uv), dtype=float),
                'threshold_mm': float('nan'),
            }
            continuous_weight = fallback_mask.astype(float)
            peaks = _hough_2d(uv[boundary_idx], cfg)
            # Fallback peaks index the compact boundary array. Convert them to
            # full projection indices so all downstream paths share one schema.
            for peak in peaks:
                peak['indices'] = boundary_idx[np.asarray(peak['indices'], dtype=int)]
                peak['rho_mm'] = peak['rho']
                peak['theta_rad'] = float(np.arctan2(peak['normal'][1], peak['normal'][0]))
                peak['weighted_vote'] = float(peak['vote_count'])
                peak['mean_weight'] = 1.0
            line_vote = {
                'accumulator': np.empty((0, 0)), 'count_accumulator': np.empty((0, 0)),
                'mean_weight_accumulator': np.empty((0, 0)),
                'rho_mm': np.empty(0), 'theta_rad': np.empty(0),
                'vote_mask': fallback_mask, 'boundary_weights': continuous_weight,
                'hough_peaks_after_nms': peaks, 'orientation_guided': False,
                'orientation_window_deg': 90.0,
                'hough_min_votes': int(cfg.hough_min_votes),
                'hough_min_mean_weight': float(cfg.hough_min_mean_weight),
                'hough_score_threshold_mode': cfg.hough_score_threshold_mode,
                'hough_score_threshold_configured': float(cfg.hough_score_threshold),
                'hough_score_threshold_effective': (
                    float(cfg.hough_score_threshold)
                    if cfg.hough_score_threshold_mode == 'fixed' else 0.0
                ),
                'hough_nms_mode': cfg.hough_nms_mode,
                'hough_nms_angle_deg': float(cfg.hough_nms_angle_deg),
                'hough_nms_rho_mm': float(cfg.hough_nms_rho_mm),
                'peak_count_above_threshold': len(peaks),
                'peak_count_after_count_mean_gate': len(peaks),
                'peak_count_local_maximum': len(peaks),
                'peak_count_after_nms': len(peaks),
                'peak_count_before_support': len(peaks),
            }
        line_refinement = refine_hough_lines_2d(
            uv,
            continuous_weight,
            line_vote['vote_mask'],
            peaks,
            roi_center_uv=query_uv,
            core_mask=core_mask,
            rho_bin_mm=cfg.hough_rho_bin_mm,
            hough_nms_rho_mm=cfg.hough_nms_rho_mm,
            hough_nms_angle_deg=cfg.hough_nms_angle_deg,
            hough_theta_step_deg=cfg.hough_theta_step_deg,
            min_points=cfg.line_min_points,
            min_length_mm=cfg.line_min_length_mm,
            config=LineRefinement2DConfig(
                enabled=cfg.line_refinement_enabled,
                max_iterations=cfg.line_refinement_max_iterations,
                final_nms_angle_deg=cfg.line_refinement_final_nms_angle_deg,
                final_nms_overlap=cfg.line_refinement_final_nms_overlap,
            ),
        )
        line_vote['line_refinement'] = line_refinement
        statistics['refined_before_nms'] += len(
            line_refinement['refined_segments_before_nms']
        )
        statistics['refined_valid'] += len(
            line_refinement['valid_segments']
        )
        statistics['refined_after_nms'] += len(
            line_refinement['refined_segments_after_nms']
        )
        statistics['refined_final_nms_suppressed'] += int(
            line_refinement['final_nms_suppressed']
        )
        statistics['boundary_points'] += int(len(boundary_idx))
        statistics['hough_peaks'] += len(peaks)
        record = {'plane_id': plane_id, 'support_indices': plane_fit_ids,
                  'projection_indices': ids, 'projection_mode': mode,
                  'projection_uv': uv, 'projection_core_mask': core_mask,
                  'projection_center_uv': query_uv,
                  'boundary': boundary, 'boundary_weights': continuous_weight,
                  'line_vote': line_vote,
                  'projection_roi_indices': candidate_pool,
                  'center_mm': center,
                  'normal': normal, 'u': u, 'v': v, 'rmse_mm': plane_rms,
                  'boundary_indices': ids[boundary_idx], 'alpha_radius_mm': alpha_radius,
                  'hough_peak_count': len(peaks), 'edge_proposal_ids': [],
                  'stage_counts': {
                      'input_support': int(len(original_ids)),
                      'plane_inliers': int(len(plane_fit_ids)),
                      'roi_candidates': int(len(roi)),
                      'projection_context': int(len(candidate_pool)),
                      'projection_core': int(np.count_nonzero(core_mask)),
                      'projection_halo': int(np.count_nonzero(~core_mask)),
                      'projection_kept': int(len(ids)),
                      'boundary': int(len(boundary_idx)),
                      'hough_peaks': int(len(peaks)),
                      'refined_before_nms': int(len(
                          line_refinement['refined_segments_before_nms']
                      )),
                      'refined_valid': int(len(
                          line_refinement['valid_segments']
                      )),
                      'refined_after_nms': int(len(
                          line_refinement['refined_segments_after_nms']
                      )),
                      'segments': 0, '3d_accepted': 0, '3d_rejected': 0,
                  }}
        planes.append(record)
        if cfg.line_refinement_enabled:
            finite_hypotheses = [
                {
                    'piece': np.asarray(
                        segment['boundary_support_uv_indices'], dtype=np.int64,
                    ),
                    'center': np.asarray(segment['point_uv'], dtype=np.float64),
                    'direction': np.asarray(segment['direction'], dtype=np.float64),
                    'peak': peaks[int(segment['seed_index'])],
                    'refinement': segment,
                }
                for segment in line_refinement['refined_segments_after_nms']
            ]
        else:
            finite_hypotheses = []
            for peak in peaks:
                candidates = np.asarray(peak['indices'], dtype=np.int64)
                refined = _tls_line(uv[candidates], dimension=2)
                if refined is None:
                    continue
                c2, d2 = refined
                # Compatibility path: the pre-refinement TLS/gap behavior is
                # retained exactly when continuous line refinement is disabled.
                eligible = np.flatnonzero(line_vote['vote_mask'])
                local_inliers = (
                    _line_residuals(uv[eligible], c2, d2) <= cfg.hough_inlier_mm
                )
                inliers2 = eligible[local_inliers]
                finite_hypotheses.extend({
                    'piece': np.asarray(piece, dtype=np.int64),
                    'center': c2,
                    'direction': d2,
                    'peak': peak,
                    'refinement': None,
                } for piece in _split_by_gaps(
                    inliers2, uv, d2, cfg.segment_gap_mm,
                ))
        for hypothesis in finite_hypotheses:
            piece = hypothesis['piece']
            peak = hypothesis['peak']
            c2 = hypothesis['center']
            d2 = hypothesis['direction']
            refinement = hypothesis['refinement']
            if cfg.line_refinement_enabled:
                if len(piece) < cfg.line_min_points:
                    statistics['2d_short_segments'] += 1
                    continue
                statistics['2d_segments'] += 1
                record['stage_counts']['segments'] += 1
            else:
                if len(piece) < cfg.line_min_points:
                    statistics['2d_short_segments'] += 1
                    continue
                statistics['2d_segments'] += 1
                record['stage_counts']['segments'] += 1
                local_fit = _tls_line(uv[piece], dimension=2)
                if local_fit is None:
                    continue
                c2, d2 = local_fit
            reference_dir3 = _normalize(d2[0] * u + d2[1] * v)
            # Crucial: the EXACT input scene IDs supporting this finite 2-D segment.
            source_ids = ids[piece]
            if np.any(~in_roi[source_ids]):
                raise RuntimeError("Projection halo leaked into 3-D edge support")
            fitted = _pca_line_3d(points[source_ids], cfg, reference_dir3)
            if fitted is None:
                statistics['3d_rejected'] += 1
                record['stage_counts']['3d_rejected'] += 1
                continue
            item = {'proposal_id': len(proposals), 'plane_id': plane_id,
                    'source_2d_indices': source_ids.copy(),
                    'support_indices': source_ids.copy(),
                    'hough_vote_count': peak['vote_count'],
                    'line_refinement_seed_index': (
                        None if refinement is None else int(refinement['seed_index'])
                    ),
                    'line_refinement_score': (
                        None if refinement is None
                        else float(refinement['edge_support_score'])
                    ),
                    **fitted}
            proposals.append(item)
            record['edge_proposal_ids'].append(item['proposal_id'])
            statistics['3d_accepted'] += 1
            record['stage_counts']['3d_accepted'] += 1
    merged = _merge_edges(proposals, points, cfg)
    same_plane_corners = _single_plane_corners(merged, planes, cfg)
    edge_corner_adjacency = {int(e['edge_id']): [] for e in merged}
    for corner in same_plane_corners:
        for edge_id in corner['edge_ids']:
            edge_corner_adjacency[edge_id].append(int(corner['corner_id']))
    graph_edges = []
    adjacency = {int(p['plane_id']): [] for p in planes}
    for edge in merged:
        for a, b in combinations(edge['plane_ids'], 2):
            graph_edges.append({'plane_a': a, 'plane_b': b, 'edge_id': edge['edge_id'],
                                'length_mm': edge['length_mm']})
            adjacency[a].append({'neighbor_plane_id': b, 'edge_id': edge['edge_id']})
            adjacency[b].append({'neighbor_plane_id': a, 'edge_id': edge['edge_id']})
    return {'method': 'plane_padded_boundary_weighted_hough_3d_edge_graph',
            'query_index': plane_analysis.get('query_index'),
            'query_position_mm': plane_analysis.get('query_position_mm'),
            'query_radius_mm': plane_analysis.get('query_radius_mm'),
            'search_scope': plane_analysis.get('search_scope', 'query'),
            'roi_point_indices': roi,
            'projection_support_indices': projection_support,
            'projection_halo_mm': projection_halo,
            'planes': planes, 'edge_proposals': proposals, 'edges': merged,
            'corners': same_plane_corners,
            'edge_corner_adjacency': edge_corner_adjacency,
            'graph_edges': graph_edges, 'adjacency': adjacency,
            'diagnostics': {**statistics, 'plane_count': len(planes),
                            'proposal_count': len(proposals), 'merged_edge_count': len(merged),
                            'shared_edge_count': sum(e['shared'] for e in merged),
                            'single_plane_corner_count': len(same_plane_corners),
                            'graph_link_count': len(graph_edges),
                            'elapsed_s': time.perf_counter()-start}}


__all__ = ['PlaneEdgeGraphConfig', 'analyze_plane_edge_graph', 'save_plane_edge_graph_json']


def save_plane_edge_graph_json(result, destination):
    """Export the numerical model, provenance indices, and plane adjacency."""
    import json
    from pathlib import Path

    def serialize(value):
        if isinstance(value, np.ndarray):
            return serialize(value.tolist())
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return serialize(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialize(v) for v in value]
        return value

    output = Path(destination).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serialize(result), indent=2, allow_nan=False), encoding='utf-8')
    return output
