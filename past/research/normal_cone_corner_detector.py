"""Normal-free 3-D corner detection with a robust local normal cone.

For a query point ``q`` and unit direction ``v``, a direction is accepted when
at least ``quantile`` of radius-neighborhood points satisfy::

    v.T @ (p - q) <= tolerance_mm

The fraction of accepted Fibonacci-sphere directions is the corner score.
The scene detector evaluates the score at downsampled points, tests local
maxima, and applies greedy radius NMS.  All coordinates are millimeters.

Run ``python3 normal_cone_corner_detector.py --self-test`` for the headless
synthetic validation.  STL input reuses the repository's virtual depth camera.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NormalConeConfig:
    """Metric and sampling parameters for the Phase A+B detector."""

    voxel_size_mm: float = 2.0
    query_radius_mm: float = 30.0
    sphere_samples: int = 1024
    tolerance_mm: float = 0.6
    quantile: float = 0.95
    min_neighbors: int = 12
    direction_chunk_size: int = 256
    min_score: float = 0.10
    local_max_radius_mm: float = 3.0
    nms_radius_mm: float = 6.0

    def validate(self) -> None:
        positive = (
            self.voxel_size_mm,
            self.query_radius_mm,
            self.tolerance_mm,
            self.local_max_radius_mm,
            self.nms_radius_mm,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("Voxel size, radii, and tolerance must be finite and positive")
        if self.sphere_samples < 16:
            raise ValueError("sphere_samples must be >= 16")
        if not 0.5 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0.5, 1]")
        if self.min_neighbors < 3:
            raise ValueError("min_neighbors must be >= 3")
        if self.direction_chunk_size < 1:
            raise ValueError("direction_chunk_size must be >= 1")
        if not np.isfinite(self.min_score) or not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")


@dataclass
class NormalConeScene:
    points_mm: np.ndarray
    directions: np.ndarray
    tree: cKDTree = field(repr=False)


@dataclass
class NormalConeResult:
    """Phase A diagnostic result for one query."""

    query_index: int
    query_point_mm: np.ndarray
    neighbor_indices: np.ndarray
    allowed_mask: np.ndarray
    direction_quantile_mm: np.ndarray
    inlier_count: np.ndarray
    required_inliers: int
    score: float


@dataclass
class SceneDetectionResult:
    """Phase B full-scene scores and selected candidates."""

    scores: np.ndarray
    neighbor_count: np.ndarray
    valid_mask: np.ndarray
    local_max_mask: np.ndarray
    local_max_indices: np.ndarray
    candidate_indices: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class MultiScaleConfig:
    """Consensus settings for evaluating the detector at several radii."""

    radii_mm: tuple[float, ...] = (12.0, 18.0, 24.0, 30.0)
    min_scale_count: int = 3
    min_consistency: float = 0.35

    def validate(self) -> None:
        radii = np.asarray(self.radii_mm, dtype=np.float64)
        if radii.ndim != 1 or len(radii) < 2:
            raise ValueError("radii_mm must contain at least two radii")
        if not np.isfinite(radii).all() or np.any(radii <= 0.0):
            raise ValueError("All multiscale radii must be finite and positive")
        if np.any(np.diff(radii) <= 0.0):
            raise ValueError("Multiscale radii must be strictly increasing")
        if not 1 <= self.min_scale_count <= len(radii):
            raise ValueError("min_scale_count must be between 1 and scale count")
        if not np.isfinite(self.min_consistency) or not 0.0 <= self.min_consistency <= 1.0:
            raise ValueError("min_consistency must be finite and in [0, 1]")


@dataclass
class MultiScaleDetectionResult:
    """Per-radius responses and their consensus Phase-B detection."""

    radii_mm: np.ndarray
    scale_scores: np.ndarray
    scale_neighbor_count: np.ndarray
    aggregate_scores: np.ndarray
    scale_pass_count: np.ndarray
    scale_consistency: np.ndarray
    valid_mask: np.ndarray
    local_max_mask: np.ndarray
    local_max_indices: np.ndarray
    candidate_indices: np.ndarray
    elapsed_seconds: float

    def as_scene_detection(self) -> SceneDetectionResult:
        return SceneDetectionResult(
            scores=self.aggregate_scores,
            neighbor_count=np.max(self.scale_neighbor_count, axis=0),
            valid_mask=self.valid_mask,
            local_max_mask=self.local_max_mask,
            local_max_indices=self.local_max_indices,
            candidate_indices=self.candidate_indices,
            elapsed_seconds=self.elapsed_seconds,
        )


@dataclass(frozen=True)
class ScaleSpaceStabilityConfig:
    """Require a corner maximum to stay localized across metric radii.

    ``detect_scene_multiscale`` checks whether the *score at one sampled point*
    persists.  That alone does not prevent the actual local maximum from
    sliding along a visibility boundary as the support grows.  This verifier
    independently extracts NMS maxima at every radius and tracks each
    aggregate candidate to the nearest per-radius maximum.
    """

    track_radius_mm: float = 30.0
    min_track_scale_count: int = 4
    max_median_drift_mm: float = 18.0
    max_p90_drift_mm: float = 30.0

    def validate(self, scale_count: int) -> None:
        if scale_count < 1:
            raise ValueError("scale_count must be positive")
        for name, value in (
            ("track_radius_mm", self.track_radius_mm),
            ("max_median_drift_mm", self.max_median_drift_mm),
            ("max_p90_drift_mm", self.max_p90_drift_mm),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.min_track_scale_count <= scale_count:
            raise ValueError("min_track_scale_count must be within the scale count")
        if self.max_median_drift_mm > self.track_radius_mm:
            raise ValueError("max_median_drift_mm cannot exceed track_radius_mm")
        if self.max_p90_drift_mm > self.track_radius_mm:
            raise ValueError("max_p90_drift_mm cannot exceed track_radius_mm")


@dataclass
class ScaleSpaceStabilityResult:
    """Spatial scale-track diagnostics stored in scene-point index space."""

    scale_local_max_mask: np.ndarray
    scale_candidate_indices: tuple[np.ndarray, ...]
    nearest_max_distance_mm: np.ndarray
    track_count: np.ndarray
    median_drift_mm: np.ndarray
    p90_drift_mm: np.ndarray
    accepted_mask: np.ndarray
    accepted_indices: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class GeometryScaleConfig:
    """Coordinate-only scale selection based on neighborhood geometry change.

    The dimensionality descriptor follows Demantke et al.: covariance
    standard deviations are converted to a probability-like 1D/2D/3D vector.
    A scale transition is declared when either the cumulative neighborhood or
    the newly added spherical shell differs too much from the smaller support.
    """

    min_clean_scale_count: int = 3
    min_score_consistency: float = 0.30
    min_shell_points: int = 12
    max_anchor_js: float = 0.20
    max_step_js: float = 0.14
    max_shell_js: float = 0.52

    def validate(self, scale_count: int | None = None) -> None:
        if self.min_clean_scale_count < 1:
            raise ValueError("min_clean_scale_count must be >= 1")
        if scale_count is not None and self.min_clean_scale_count > scale_count:
            raise ValueError("min_clean_scale_count exceeds the scale count")
        if self.min_shell_points < 3:
            raise ValueError("min_shell_points must be >= 3")
        for name, value in (
            ("min_score_consistency", self.min_score_consistency),
            ("max_anchor_js", self.max_anchor_js),
            ("max_step_js", self.max_step_js),
            ("max_shell_js", self.max_shell_js),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass
class GeometryScaleDetectionResult:
    """Geometry-aware multi-scale scores and diagnostics for every point."""

    radii_mm: np.ndarray
    dimensionality: np.ndarray
    dimensionality_entropy: np.ndarray
    shell_dimensionality: np.ndarray
    anchor_js: np.ndarray
    step_js: np.ndarray
    shell_js: np.ndarray
    transition_scale_index: np.ndarray
    selected_scale_index: np.ndarray
    selected_radius_mm: np.ndarray
    clean_scale_count: np.ndarray
    aggregate_scores: np.ndarray
    score_consistency: np.ndarray
    valid_mask: np.ndarray
    local_max_mask: np.ndarray
    local_max_indices: np.ndarray
    candidate_indices: np.ndarray
    elapsed_seconds: float

    def as_scene_detection(self, multiscale: MultiScaleDetectionResult) -> SceneDetectionResult:
        return SceneDetectionResult(
            scores=self.aggregate_scores,
            neighbor_count=np.max(multiscale.scale_neighbor_count, axis=0),
            valid_mask=self.valid_mask,
            local_max_mask=self.local_max_mask,
            local_max_indices=self.local_max_indices,
            candidate_indices=self.candidate_indices,
            elapsed_seconds=multiscale.elapsed_seconds + self.elapsed_seconds,
        )


@dataclass(frozen=True)
class PhaseCConfig:
    """Coordinate-only candidate verifier parameters.

    The verifier is intentionally evaluated only at Phase-B NMS candidates.
    Spatial support removes locally sparse responses, while split
    repeatability checks whether independently sampled halves of the same
    neighborhood yield similar allowed-direction sets.
    """

    min_support: int = 40
    outer_shell_start_ratio: float = 0.70
    min_outer_shell_fraction: float = 0.15
    min_radius_p90_ratio: float = 0.75
    min_centroid_offset_ratio: float = 0.42
    min_directional_concentration: float = 0.68
    repeatability_trials: int = 3
    min_repeatability: float = 0.55
    min_split_score_ratio: float = 0.50
    repeatability_seed: int = 7

    def validate(self) -> None:
        if self.min_support < 3:
            raise ValueError("min_support must be >= 3")
        if not 0.0 <= self.outer_shell_start_ratio < 1.0:
            raise ValueError("outer_shell_start_ratio must be in [0, 1)")
        for name, value in (
            ("min_outer_shell_fraction", self.min_outer_shell_fraction),
            ("min_radius_p90_ratio", self.min_radius_p90_ratio),
            ("min_centroid_offset_ratio", self.min_centroid_offset_ratio),
            ("min_directional_concentration", self.min_directional_concentration),
            ("min_repeatability", self.min_repeatability),
            ("min_split_score_ratio", self.min_split_score_ratio),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.repeatability_trials < 1:
            raise ValueError("repeatability_trials must be >= 1")


@dataclass
class PhaseCResult:
    """Verifier metrics aligned with ``input_indices``."""

    input_indices: np.ndarray
    support_pass: np.ndarray
    spatial_pass: np.ndarray
    repeatability_pass: np.ndarray
    outer_shell_fraction: np.ndarray
    radius_p90_ratio: np.ndarray
    centroid_offset_ratio: np.ndarray
    directional_concentration: np.ndarray
    repeatability: np.ndarray
    split_score_ratio: np.ndarray
    support_indices: np.ndarray
    spatial_indices: np.ndarray
    accepted_indices: np.ndarray
    elapsed_seconds: float


def _as_points(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points_mm must not be empty")
    if not np.isfinite(points).all():
        raise ValueError("points_mm must contain only finite coordinates")
    return np.ascontiguousarray(points)


def fibonacci_sphere(sample_count: int) -> np.ndarray:
    """Return deterministic, approximately equal-area unit directions."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    index = np.arange(sample_count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / float(sample_count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = index * golden_angle
    directions = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def voxel_downsample(points_mm: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """Average points in a deterministic world-aligned voxel grid."""
    points = _as_points(points_mm)
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be finite and positive")
    cell = np.floor(points / float(voxel_size_mm)).astype(np.int64)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    downsampled = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    return np.asarray(downsampled, dtype=np.float64)


def build_scene(
    points_mm: np.ndarray,
    config: NormalConeConfig,
    *,
    downsample: bool = True,
) -> NormalConeScene:
    config.validate()
    points = (
        voxel_downsample(points_mm, config.voxel_size_mm)
        if downsample
        else _as_points(points_mm).copy()
    )
    return NormalConeScene(
        points_mm=points,
        directions=fibonacci_sphere(config.sphere_samples),
        tree=cKDTree(points),
    )


def _allowed_direction_statistics(
    offsets_mm: np.ndarray,
    directions: np.ndarray,
    config: NormalConeConfig,
    *,
    return_quantiles: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Evaluate empirical-quantile constraints in bounded memory."""
    count = len(offsets_mm)
    required = int(math.ceil(config.quantile * count))
    kth = required - 1
    allowed = np.empty(len(directions), dtype=bool)
    inlier_count = np.empty(len(directions), dtype=np.int32)
    quantiles = np.empty(len(directions), dtype=np.float64) if return_quantiles else None

    chunk = config.direction_chunk_size
    for start in range(0, len(directions), chunk):
        stop = min(start + chunk, len(directions))
        projection = offsets_mm @ directions[start:stop].T
        counts = np.count_nonzero(projection <= config.tolerance_mm, axis=0)
        inlier_count[start:stop] = counts
        allowed[start:stop] = counts >= required
        if quantiles is not None:
            # Empirical inverse-CDF Q_q. Q_q <= delta is exactly equivalent
            # to at least ceil(q*N) observations being <= delta.
            partitioned = np.partition(projection, kth, axis=0)
            quantiles[start:stop] = partitioned[kth]
    return allowed, inlier_count, quantiles, required


def estimate_normal_cone(
    scene: NormalConeScene,
    query_index: int,
    config: NormalConeConfig,
) -> NormalConeResult:
    """Compute allowed directions and the solid-angle score for one point."""
    config.validate()
    index = int(query_index)
    if not 0 <= index < len(scene.points_mm):
        raise IndexError("query_index is outside the scene")
    query = scene.points_mm[index]
    neighbor_indices = np.asarray(
        scene.tree.query_ball_point(query, config.query_radius_mm), dtype=np.int64
    )
    if len(neighbor_indices) < config.min_neighbors:
        allowed = np.zeros(len(scene.directions), dtype=bool)
        return NormalConeResult(
            index,
            query.copy(),
            neighbor_indices,
            allowed,
            np.full(len(scene.directions), np.nan),
            np.zeros(len(scene.directions), dtype=np.int32),
            int(math.ceil(config.quantile * len(neighbor_indices))),
            0.0,
        )

    offsets = scene.points_mm[neighbor_indices] - query
    allowed, inliers, quantiles, required = _allowed_direction_statistics(
        offsets, scene.directions, config, return_quantiles=True
    )
    assert quantiles is not None
    return NormalConeResult(
        index,
        query.copy(),
        neighbor_indices,
        allowed,
        quantiles,
        inliers,
        required,
        float(np.mean(allowed)),
    )


def _score_query(
    scene: NormalConeScene, query_index: int, config: NormalConeConfig
) -> tuple[float, int]:
    query = scene.points_mm[query_index]
    neighbors = np.asarray(
        scene.tree.query_ball_point(query, config.query_radius_mm), dtype=np.int64
    )
    if len(neighbors) < config.min_neighbors:
        return 0.0, len(neighbors)
    offsets = scene.points_mm[neighbors] - query
    allowed, _, _, _ = _allowed_direction_statistics(
        offsets, scene.directions, config, return_quantiles=False
    )
    return float(np.mean(allowed)), len(neighbors)


def _local_maxima_and_nms(
    scene: NormalConeScene,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    config: NormalConeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = valid_mask & (scores >= config.min_score)
    local_max = np.zeros(len(scores), dtype=bool)
    for index in np.flatnonzero(eligible):
        nearby = np.asarray(
            scene.tree.query_ball_point(scene.points_mm[index], config.local_max_radius_mm),
            dtype=np.int64,
        )
        comparable = nearby[valid_mask[nearby]]
        if len(comparable) == 0 or scores[index] + 1e-12 >= np.max(scores[comparable]):
            local_max[index] = True

    local_indices = np.flatnonzero(local_max)
    order = local_indices[np.lexsort((local_indices, -scores[local_indices]))]
    suppressed = np.zeros(len(scores), dtype=bool)
    selected: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        selected.append(int(index))
        nearby = scene.tree.query_ball_point(scene.points_mm[index], config.nms_radius_mm)
        suppressed[np.asarray(nearby, dtype=np.int64)] = True
    return local_max, local_indices, np.asarray(selected, dtype=np.int64)


def detect_scene(
    scene: NormalConeScene,
    config: NormalConeConfig,
    *,
    query_indices: np.ndarray | Sequence[int] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> SceneDetectionResult:
    """Score scene points, find local maxima, and apply greedy radius NMS."""
    config.validate()
    count = len(scene.points_mm)
    if query_indices is None:
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.unique(np.asarray(query_indices, dtype=np.int64).reshape(-1))
        if np.any((indices < 0) | (indices >= count)):
            raise IndexError("query_indices contains an index outside the scene")

    scores = np.zeros(count, dtype=np.float64)
    neighbor_count = np.zeros(count, dtype=np.int32)
    valid = np.zeros(count, dtype=bool)
    start = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        score, neighbors = _score_query(scene, int(index), config)
        scores[index] = score
        neighbor_count[index] = neighbors
        valid[index] = neighbors >= config.min_neighbors
        if progress is not None:
            progress(position, len(indices), time.perf_counter() - start)

    local_mask, local_indices, candidates = _local_maxima_and_nms(
        scene, scores, valid, config
    )
    return SceneDetectionResult(
        scores,
        neighbor_count,
        valid,
        local_mask,
        local_indices,
        candidates,
        time.perf_counter() - start,
    )


def detect_scene_multiscale(
    scene: NormalConeScene,
    config: NormalConeConfig,
    multiscale: MultiScaleConfig | None = None,
    *,
    query_indices: np.ndarray | Sequence[int] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> MultiScaleDetectionResult:
    """Detect corners that persist across multiple neighborhood radii.

    A point must exceed ``config.min_score`` at ``min_scale_count`` radii.
    The median response across valid radii is used for local maxima and NMS.
    ``scale_consistency`` is the minimum divided by the maximum valid response;
    it suppresses responses that exist only at one small scale.
    """
    config.validate()
    scale_config = multiscale or MultiScaleConfig()
    scale_config.validate()
    count = len(scene.points_mm)
    if query_indices is None:
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.unique(np.asarray(query_indices, dtype=np.int64).reshape(-1))
        if np.any((indices < 0) | (indices >= count)):
            raise IndexError("query_indices contains an index outside the scene")

    radii = np.asarray(scale_config.radii_mm, dtype=np.float64)
    scale_scores = np.zeros((len(radii), count), dtype=np.float64)
    scale_neighbors = np.zeros((len(radii), count), dtype=np.int32)
    start = time.perf_counter()
    completed = 0
    total = len(radii) * len(indices)
    for scale_index, radius in enumerate(radii):
        radius_config = replace(config, query_radius_mm=float(radius))
        for index in indices:
            score, neighbors = _score_query(scene, int(index), radius_config)
            scale_scores[scale_index, index] = score
            scale_neighbors[scale_index, index] = neighbors
            completed += 1
            if progress is not None:
                progress(completed, total, time.perf_counter() - start)

    scale_valid = scale_neighbors >= config.min_neighbors
    valid_count = np.count_nonzero(scale_valid, axis=0)
    pass_count = np.count_nonzero(
        scale_valid & (scale_scores >= config.min_score), axis=0
    ).astype(np.int16)
    aggregate = np.zeros(count, dtype=np.float64)
    consistency = np.zeros(count, dtype=np.float64)
    for index in indices:
        values = scale_scores[scale_valid[:, index], index]
        if len(values) == 0:
            continue
        aggregate[index] = float(np.median(values))
        maximum = float(np.max(values))
        if maximum > 0.0:
            consistency[index] = float(np.min(values) / maximum)

    consensus = (
        (valid_count >= scale_config.min_scale_count)
        & (pass_count >= scale_config.min_scale_count)
        & (consistency >= scale_config.min_consistency)
    )
    local_mask, local_indices, candidates = _local_maxima_and_nms(
        scene, aggregate, consensus, config
    )
    return MultiScaleDetectionResult(
        radii_mm=radii,
        scale_scores=scale_scores,
        scale_neighbor_count=scale_neighbors,
        aggregate_scores=aggregate,
        scale_pass_count=pass_count,
        scale_consistency=consistency,
        valid_mask=consensus,
        local_max_mask=local_mask,
        local_max_indices=local_indices,
        candidate_indices=candidates,
        elapsed_seconds=time.perf_counter() - start,
    )


def evaluate_scale_space_stability(
    scene: NormalConeScene,
    config: NormalConeConfig,
    multiscale: MultiScaleDetectionResult,
    stability: ScaleSpaceStabilityConfig | None = None,
) -> ScaleSpaceStabilityResult:
    """Track aggregate candidates to independently detected scale maxima.

    A response is stable only when sufficiently many radius-specific NMS
    maxima remain close to the final aggregate maximum.  Distances are in
    physical units, so the test is independent of image depth and point count.
    The function reuses already-computed score maps; it does not evaluate any
    additional Normal Cones.
    """
    config.validate()
    if multiscale.scale_scores.ndim != 2:
        raise ValueError("multiscale.scale_scores must be a scale-by-point array")
    scale_count, point_count = multiscale.scale_scores.shape
    if point_count != len(scene.points_mm):
        raise ValueError("multiscale result and scene point counts differ")
    if multiscale.scale_neighbor_count.shape != (scale_count, point_count):
        raise ValueError("scale_neighbor_count shape does not match scale_scores")
    verify_config = stability or ScaleSpaceStabilityConfig()
    verify_config.validate(scale_count)

    started = time.perf_counter()
    scale_local_max = np.zeros((scale_count, point_count), dtype=bool)
    scale_candidates: list[np.ndarray] = []
    for scale_index in range(scale_count):
        scale_valid = (
            multiscale.scale_neighbor_count[scale_index] >= config.min_neighbors
        )
        local_mask, _, candidates = _local_maxima_and_nms(
            scene,
            multiscale.scale_scores[scale_index],
            scale_valid,
            config,
        )
        scale_local_max[scale_index] = local_mask
        scale_candidates.append(candidates)

    distances = np.full((scale_count, point_count), np.inf, dtype=np.float64)
    aggregate_candidates = np.asarray(multiscale.candidate_indices, dtype=np.int64)
    if len(aggregate_candidates):
        queries = scene.points_mm[aggregate_candidates]
        for scale_index, candidates in enumerate(scale_candidates):
            if len(candidates) == 0:
                continue
            maximum_tree = cKDTree(scene.points_mm[candidates])
            nearest, _ = maximum_tree.query(queries, k=1)
            nearest = np.asarray(nearest, dtype=np.float64)
            within = nearest <= verify_config.track_radius_mm
            distances[scale_index, aggregate_candidates[within]] = nearest[within]

    track_count = np.count_nonzero(np.isfinite(distances), axis=0).astype(np.int16)
    median_drift = np.full(point_count, np.inf, dtype=np.float64)
    p90_drift = np.full(point_count, np.inf, dtype=np.float64)
    for index in aggregate_candidates:
        tracked = distances[np.isfinite(distances[:, index]), index]
        if len(tracked) == 0:
            continue
        median_drift[index] = float(np.median(tracked))
        p90_drift[index] = float(np.percentile(tracked, 90))

    accepted = np.zeros(point_count, dtype=bool)
    accepted[aggregate_candidates] = (
        (track_count[aggregate_candidates] >= verify_config.min_track_scale_count)
        & (median_drift[aggregate_candidates] <= verify_config.max_median_drift_mm)
        & (p90_drift[aggregate_candidates] <= verify_config.max_p90_drift_mm)
    )
    return ScaleSpaceStabilityResult(
        scale_local_max_mask=scale_local_max,
        scale_candidate_indices=tuple(scale_candidates),
        nearest_max_distance_mm=distances,
        track_count=track_count,
        median_drift_mm=median_drift,
        p90_drift_mm=p90_drift,
        accepted_mask=accepted,
        accepted_indices=np.flatnonzero(accepted),
        elapsed_seconds=time.perf_counter() - started,
    )


def dimensionality_descriptor(points_mm: np.ndarray) -> np.ndarray:
    """Return Demantke-style probabilities for linear/planar/volumetric shape.

    If ``sigma1 >= sigma2 >= sigma3`` are the square roots of the centered
    covariance eigenvalues, the result is::

        [(sigma1-sigma2)/sigma1,
         (sigma2-sigma3)/sigma1,
          sigma3/sigma1]

    The three components are non-negative and sum to one.  Only point
    coordinates are used; no normal is estimated.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    if len(points) < 3:
        return np.full(3, np.nan, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(len(centered))
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    sigma = np.sqrt(np.maximum(eigenvalues, 0.0))
    if sigma[0] <= np.finfo(np.float64).eps:
        return np.full(3, np.nan, dtype=np.float64)
    descriptor = np.array(
        [sigma[0] - sigma[1], sigma[1] - sigma[2], sigma[2]],
        dtype=np.float64,
    ) / sigma[0]
    descriptor = np.maximum(descriptor, 0.0)
    return descriptor / np.sum(descriptor)


def dimensionality_entropy(descriptor: np.ndarray) -> float:
    """Return normalized Shannon entropy of a 1D/2D/3D descriptor."""
    values = np.asarray(descriptor, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        return float("nan")
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log(positive)) / np.log(3.0))


def jensen_shannon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return a [0, 1] Jensen-Shannon distance for probability vectors."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Probability vectors must have the same 1-D shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_sum = float(np.sum(left))
    right_sum = float(np.sum(right))
    if left_sum <= 0.0 or right_sum <= 0.0:
        return float("nan")
    left = np.maximum(left / left_sum, 0.0)
    right = np.maximum(right / right_sum, 0.0)
    middle = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log(values[mask] / middle[mask])))

    js_divergence = 0.5 * (divergence(left) + divergence(right))
    return float(np.sqrt(max(js_divergence, 0.0) / np.log(2.0)))


def detect_scene_geometry_adaptive(
    scene: NormalConeScene,
    config: NormalConeConfig,
    multiscale: MultiScaleDetectionResult,
    geometry: GeometryScaleConfig | None = None,
    *,
    query_indices: np.ndarray | Sequence[int] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> GeometryScaleDetectionResult:
    """Select the largest uncontaminated scale before a geometry transition.

    This differs from score-only consensus: the detector compares covariance
    dimensionality of cumulative balls and newly added annuli.  When a larger
    radius starts to include a geometrically different structure, the clean
    prefix ends at the preceding radius and only that prefix contributes to
    the aggregate Normal Cone score.
    """
    config.validate()
    scale_config = geometry or GeometryScaleConfig()
    radii = np.asarray(multiscale.radii_mm, dtype=np.float64)
    scale_count = len(radii)
    scale_config.validate(scale_count)
    point_count = len(scene.points_mm)
    if multiscale.scale_scores.shape != (scale_count, point_count):
        raise ValueError("multiscale result does not match the scene")
    if query_indices is None:
        indices = np.arange(point_count, dtype=np.int64)
    else:
        indices = np.unique(np.asarray(query_indices, dtype=np.int64).reshape(-1))
        if np.any((indices < 0) | (indices >= point_count)):
            raise IndexError("query_indices contains an index outside the scene")

    dimensionality = np.full((scale_count, point_count, 3), np.nan, dtype=np.float64)
    entropy = np.full((scale_count, point_count), np.nan, dtype=np.float64)
    shell_dimensionality = np.full(
        (max(scale_count - 1, 0), point_count, 3), np.nan, dtype=np.float64
    )
    anchor_js = np.full((scale_count, point_count), np.nan, dtype=np.float64)
    step_js = np.full((max(scale_count - 1, 0), point_count), np.nan, dtype=np.float64)
    shell_js = np.full((max(scale_count - 1, 0), point_count), np.nan, dtype=np.float64)
    transition_index = np.full(point_count, scale_count, dtype=np.int16)
    selected_index = np.full(point_count, -1, dtype=np.int16)
    selected_radius = np.full(point_count, np.nan, dtype=np.float64)
    clean_count = np.zeros(point_count, dtype=np.int16)
    aggregate = np.zeros(point_count, dtype=np.float64)
    consistency = np.zeros(point_count, dtype=np.float64)
    valid = np.zeros(point_count, dtype=bool)
    start = time.perf_counter()

    for position, index in enumerate(indices, start=1):
        center = scene.points_mm[index]
        outer_ids = np.asarray(
            scene.tree.query_ball_point(center, float(radii[-1])), dtype=np.int64
        )
        offsets = scene.points_mm[outer_ids] - center
        distances = np.linalg.norm(offsets, axis=1)
        scale_valid = multiscale.scale_neighbor_count[:, index] >= config.min_neighbors
        for scale_index, radius in enumerate(radii):
            inside = distances <= radius + 1e-12
            if np.count_nonzero(inside) < config.min_neighbors:
                continue
            descriptor = dimensionality_descriptor(offsets[inside])
            dimensionality[scale_index, index] = descriptor
            entropy[scale_index, index] = dimensionality_entropy(descriptor)

        valid_scales = np.flatnonzero(
            scale_valid & np.isfinite(dimensionality[:, index, 0])
        )
        if len(valid_scales) == 0:
            if progress is not None:
                progress(position, len(indices), time.perf_counter() - start)
            continue
        first_scale = int(valid_scales[0])
        anchor_js[first_scale, index] = 0.0
        clean_end = first_scale
        for scale_index in range(first_scale + 1, scale_count):
            if not scale_valid[scale_index] or not np.isfinite(
                dimensionality[scale_index, index, 0]
            ):
                break
            previous = dimensionality[scale_index - 1, index]
            current = dimensionality[scale_index, index]
            if not np.isfinite(previous).all():
                break
            anchor_js[scale_index, index] = jensen_shannon_distance(
                dimensionality[first_scale, index], current
            )
            step_js[scale_index - 1, index] = jensen_shannon_distance(previous, current)

            shell = (distances > radii[scale_index - 1] + 1e-12) & (
                distances <= radii[scale_index] + 1e-12
            )
            if np.count_nonzero(shell) >= scale_config.min_shell_points:
                shell_descriptor = dimensionality_descriptor(offsets[shell])
                shell_dimensionality[scale_index - 1, index] = shell_descriptor
                shell_js[scale_index - 1, index] = jensen_shannon_distance(
                    previous, shell_descriptor
                )

            changed = bool(
                anchor_js[scale_index, index] > scale_config.max_anchor_js
                or step_js[scale_index - 1, index] > scale_config.max_step_js
                or (
                    np.isfinite(shell_js[scale_index - 1, index])
                    and shell_js[scale_index - 1, index] > scale_config.max_shell_js
                )
            )
            if changed:
                transition_index[index] = scale_index
                break
            clean_end = scale_index

        clean_ids = valid_scales[
            (valid_scales >= first_scale) & (valid_scales <= clean_end)
        ]
        clean_count[index] = len(clean_ids)
        if len(clean_ids):
            selected_index[index] = int(clean_ids[-1])
            selected_radius[index] = radii[clean_ids[-1]]
            values = multiscale.scale_scores[clean_ids, index]
            aggregate[index] = float(np.median(values))
            maximum = float(np.max(values))
            if maximum > 0.0:
                consistency[index] = float(np.min(values) / maximum)
            pass_count = int(np.count_nonzero(values >= config.min_score))
            valid[index] = bool(
                len(clean_ids) >= scale_config.min_clean_scale_count
                and pass_count >= scale_config.min_clean_scale_count
                and consistency[index] >= scale_config.min_score_consistency
            )
        if progress is not None:
            progress(position, len(indices), time.perf_counter() - start)

    local_mask, local_indices, candidates = _local_maxima_and_nms(
        scene, aggregate, valid, config
    )
    return GeometryScaleDetectionResult(
        radii_mm=radii,
        dimensionality=dimensionality,
        dimensionality_entropy=entropy,
        shell_dimensionality=shell_dimensionality,
        anchor_js=anchor_js,
        step_js=step_js,
        shell_js=shell_js,
        transition_scale_index=transition_index,
        selected_scale_index=selected_index,
        selected_radius_mm=selected_radius,
        clean_scale_count=clean_count,
        aggregate_scores=aggregate,
        score_consistency=consistency,
        valid_mask=valid,
        local_max_mask=local_mask,
        local_max_indices=local_indices,
        candidate_indices=candidates,
        elapsed_seconds=time.perf_counter() - start,
    )


def allowed_mask_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Return Jaccard agreement for two allowed-direction masks."""
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("Allowed-direction masks must have the same shape")
    union = np.count_nonzero(left | right)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left & right) / union)


def _split_repeatability(
    scene: NormalConeScene,
    query_index: int,
    neighbor_indices: np.ndarray,
    base_score: float,
    config: NormalConeConfig,
    verifier: PhaseCConfig,
) -> tuple[float, float]:
    """Evaluate deterministic random half-splits of one neighborhood."""
    scores: list[float] = []
    agreements: list[float] = []
    seed = np.random.SeedSequence([verifier.repeatability_seed, int(query_index)])
    for child_seed in seed.spawn(verifier.repeatability_trials):
        shuffled = np.asarray(neighbor_indices, dtype=np.int64).copy()
        np.random.default_rng(child_seed).shuffle(shuffled)
        midpoint = len(shuffled) // 2
        halves = (shuffled[:midpoint], shuffled[midpoint:])
        masks: list[np.ndarray] = []
        trial_scores: list[float] = []
        for half in halves:
            if len(half) < config.min_neighbors:
                masks.append(np.zeros(len(scene.directions), dtype=bool))
                trial_scores.append(0.0)
                continue
            offsets = scene.points_mm[half] - scene.points_mm[query_index]
            allowed, _, _, _ = _allowed_direction_statistics(
                offsets,
                scene.directions,
                config,
                return_quantiles=False,
            )
            masks.append(allowed)
            trial_scores.append(float(np.mean(allowed)))
        agreements.append(allowed_mask_jaccard(masks[0], masks[1]))
        scores.append(min(trial_scores))

    denominator = max(float(base_score), 1.0 / len(scene.directions))
    return float(np.median(agreements)), float(np.median(scores) / denominator)


def verify_candidates_phase_c(
    scene: NormalConeScene,
    detection: SceneDetectionResult,
    config: NormalConeConfig,
    verifier: PhaseCConfig | None = None,
    *,
    query_radius_mm_by_index: np.ndarray | Sequence[float] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> PhaseCResult:
    """Verify NMS candidates using support, extent, and split stability.

    The support stages are evaluated before repeatability so expensive
    direction-set resampling is skipped for obvious sparse responses.  Every
    metric remains coordinate-only; no normal or tangent plane is estimated.
    """
    config.validate()
    verify_config = verifier or PhaseCConfig()
    verify_config.validate()
    indices = np.asarray(detection.candidate_indices, dtype=np.int64).copy()
    adaptive_radii: np.ndarray | None = None
    if query_radius_mm_by_index is not None:
        adaptive_radii = np.asarray(query_radius_mm_by_index, dtype=np.float64)
        if adaptive_radii.shape != (len(scene.points_mm),):
            raise ValueError("query_radius_mm_by_index must have one value per scene point")
        selected = adaptive_radii[indices]
        if not np.isfinite(selected).all() or np.any(selected <= 0.0):
            raise ValueError("Candidate query radii must be finite and positive")
    count = len(indices)
    support_pass = np.zeros(count, dtype=bool)
    spatial_pass = np.zeros(count, dtype=bool)
    repeatability_pass = np.zeros(count, dtype=bool)
    outer_fraction = np.zeros(count, dtype=np.float64)
    radius_p90_ratio = np.zeros(count, dtype=np.float64)
    centroid_offset_ratio = np.zeros(count, dtype=np.float64)
    directional_concentration = np.zeros(count, dtype=np.float64)
    repeatability = np.zeros(count, dtype=np.float64)
    split_score_ratio = np.zeros(count, dtype=np.float64)
    start = time.perf_counter()

    for position, index in enumerate(indices, start=1):
        query_radius = (
            float(adaptive_radii[index])
            if adaptive_radii is not None
            else config.query_radius_mm
        )
        neighbors = np.asarray(
            scene.tree.query_ball_point(
                scene.points_mm[index], query_radius
            ),
            dtype=np.int64,
        )
        offsets = scene.points_mm[neighbors] - scene.points_mm[index]
        distances = np.linalg.norm(offsets, axis=1)
        support_pass[position - 1] = len(neighbors) >= verify_config.min_support
        if len(distances):
            outer_fraction[position - 1] = float(
                np.mean(
                    distances
                    >= verify_config.outer_shell_start_ratio
                    * query_radius
                )
            )
            radius_p90_ratio[position - 1] = float(
                np.percentile(distances, 90) / query_radius
            )
            centroid_offset_ratio[position - 1] = float(
                np.linalg.norm(np.mean(offsets, axis=0)) / query_radius
            )
            nonzero = distances > 1e-12
            if np.any(nonzero):
                unit_offsets = offsets[nonzero] / distances[nonzero, None]
                directional_concentration[position - 1] = float(
                    np.linalg.norm(np.mean(unit_offsets, axis=0))
                )
        spatial_pass[position - 1] = bool(
            support_pass[position - 1]
            and outer_fraction[position - 1]
            >= verify_config.min_outer_shell_fraction
            and radius_p90_ratio[position - 1]
            >= verify_config.min_radius_p90_ratio
            and centroid_offset_ratio[position - 1]
            >= verify_config.min_centroid_offset_ratio
            and directional_concentration[position - 1]
            >= verify_config.min_directional_concentration
        )

        if spatial_pass[position - 1]:
            agreement, score_ratio = _split_repeatability(
                scene,
                int(index),
                neighbors,
                float(detection.scores[index]),
                config,
                verify_config,
            )
            repeatability[position - 1] = agreement
            split_score_ratio[position - 1] = score_ratio
            repeatability_pass[position - 1] = bool(
                agreement >= verify_config.min_repeatability
                and score_ratio >= verify_config.min_split_score_ratio
            )
        if progress is not None:
            progress(position, count, time.perf_counter() - start)

    after_support = support_pass
    after_spatial = support_pass & spatial_pass
    accepted = after_spatial & repeatability_pass
    return PhaseCResult(
        input_indices=indices,
        support_pass=support_pass,
        spatial_pass=spatial_pass,
        repeatability_pass=repeatability_pass,
        outer_shell_fraction=outer_fraction,
        radius_p90_ratio=radius_p90_ratio,
        centroid_offset_ratio=centroid_offset_ratio,
        directional_concentration=directional_concentration,
        repeatability=repeatability,
        split_score_ratio=split_score_ratio,
        support_indices=indices[after_support],
        spatial_indices=indices[after_spatial],
        accepted_indices=indices[accepted],
        elapsed_seconds=time.perf_counter() - start,
    )


def _grid_values(low: float, high: float, spacing: float) -> np.ndarray:
    return np.arange(low, high + 0.25 * spacing, spacing, dtype=np.float64)


def _unique_points(points: np.ndarray) -> np.ndarray:
    return np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=10), axis=0)


def synthetic_patch(
    kind: str, *, extent_mm: float = 45.0, spacing_mm: float = 2.0, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Create an analytic test patch and return ``(points, query_xyz)``."""
    positive = _grid_values(0.0, extent_mm, spacing_mm)
    signed = _grid_values(-extent_mm, extent_mm, spacing_mm)

    if kind == "plane":
        x, y = np.meshgrid(signed, signed, indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    elif kind == "plane_corner":
        x, y = np.meshgrid(positive, positive, indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    elif kind == "plane_edge":
        x, y = np.meshgrid(positive, signed, indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    elif kind == "two_face_edge":
        along, across = np.meshgrid(signed, positive, indexing="ij")
        first = np.column_stack((along.ravel(), across.ravel(), np.zeros(along.size)))
        second = np.column_stack((along.ravel(), np.zeros(along.size), across.ravel()))
        points = _unique_points(np.vstack((first, second)))
    elif kind == "two_face_corner":
        a, b = np.meshgrid(positive, positive, indexing="ij")
        first = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
        second = np.column_stack((a.ravel(), np.zeros(a.size), b.ravel()))
        points = _unique_points(np.vstack((first, second)))
    elif kind == "three_face_corner":
        a, b = np.meshgrid(positive, positive, indexing="ij")
        xy = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
        xz = np.column_stack((a.ravel(), np.zeros(a.size), b.ravel()))
        yz = np.column_stack((np.zeros(a.size), a.ravel(), b.ravel()))
        points = _unique_points(np.vstack((xy, xz, yz)))
    elif kind == "noisy_plane":
        x, y = np.meshgrid(signed, signed, indexing="ij")
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 0.35, x.size)
        center = int(np.argmin(x.ravel() ** 2 + y.ravel() ** 2))
        z[center] = 0.0
        points = np.column_stack((x.ravel(), y.ravel(), z))
    elif kind in {"sphere", "cylinder"}:
        radius = 100.0
        x, y = np.meshgrid(signed, signed, indexing="ij")
        if kind == "sphere":
            keep = x.ravel() ** 2 + y.ravel() ** 2 < radius ** 2
            xx, yy = x.ravel()[keep], y.ravel()[keep]
            zz = -radius + np.sqrt(radius ** 2 - xx ** 2 - yy ** 2)
        else:
            xx, yy = x.ravel(), y.ravel()
            zz = -radius + np.sqrt(radius ** 2 - xx ** 2)
        points = np.column_stack((xx, yy, zz))
    else:
        raise ValueError(f"Unknown synthetic patch: {kind}")
    return _unique_points(points), np.zeros(3)


def nearest_point_index(points_mm: np.ndarray, xyz_mm: np.ndarray) -> int:
    points = _as_points(points_mm)
    xyz = np.asarray(xyz_mm, dtype=np.float64).reshape(3)
    return int(np.argmin(np.sum((points - xyz) ** 2, axis=1)))


def run_synthetic_validation(
    config: NormalConeConfig | None = None,
) -> dict[str, float | bool]:
    """Run Phase A hypotheses on deterministic coordinate-only patches."""
    cfg = config or NormalConeConfig()
    cases = (
        "plane",
        "plane_edge",
        "two_face_edge",
        "plane_corner",
        "two_face_corner",
        "three_face_corner",
        "sphere",
        "cylinder",
        "noisy_plane",
    )
    result: dict[str, float | bool] = {}
    for name in cases:
        points, query = synthetic_patch(name)
        scene = build_scene(points, cfg, downsample=False)
        index = nearest_point_index(points, query)
        result[name] = estimate_normal_cone(scene, index, cfg).score

    low = ("plane", "plane_edge", "two_face_edge", "sphere", "cylinder", "noisy_plane")
    corners = ("plane_corner", "two_face_corner", "three_face_corner")
    result["separation_pass"] = bool(
        min(float(result[name]) for name in corners)
        > max(float(result[name]) for name in low) + 0.03
    )
    return result


def _load_open3d():
    # Open3D imports its optional ML package unconditionally.  Some geometry-
    # only installations have an incompatible sklearn/NumPy pair; a stub keeps
    # that unrelated optional component from breaking point-cloud visualization.
    if "open3d" not in sys.modules and "open3d.ml" not in sys.modules:
        sys.modules["open3d.ml"] = types.ModuleType("open3d.ml")
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - optional GUI package
        raise RuntimeError(
            "Open3D geometry/visualization could not be imported. "
            "Check the installed Open3D and NumPy versions."
        ) from exc
    return o3d


def load_points_mm(path: str | Path, args: argparse.Namespace) -> np.ndarray:
    """Load a point cloud or capture a partial STL view with the existing rig."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        return _as_points(np.load(input_path))
    if suffix == ".npz":
        with np.load(input_path) as data:
            for key in ("points_mm", "points", "xyz"):
                if key in data:
                    return _as_points(data[key])
        raise ValueError("NPZ must contain points_mm, points, or xyz")

    if suffix == ".stl":  # pragma: no cover - interactive GUI path
        _load_open3d()
        from stl_tangent_boundary_demo import (
            NoiseConfig,
            depth_to_xyz_mm,
            load_stl_as_mm,
            make_raycast_scene,
            run_camera_selector,
        )

        mesh = load_stl_as_mm(str(input_path), args.stl_unit_mm)
        ray_scene = make_raycast_scene(mesh)
        noise = NoiseConfig(gaussian_sigma_mm=args.noise_mm, seed=args.seed)
        capture = run_camera_selector(
            mesh, ray_scene, noise, args.width, args.height, args.fov_deg
        )
        if capture is None:
            raise RuntimeError("STL capture was cancelled")
        depth_m = np.asarray(capture["depth_mm"], dtype=np.float32) / 1000.0
        return _as_points(depth_to_xyz_mm(depth_m, capture["intr"]))

    o3d = _load_open3d()
    cloud = o3d.io.read_point_cloud(str(input_path))
    if cloud.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {input_path}")
    return _as_points(np.asarray(cloud.points))


def pick_query_index(points_mm: np.ndarray) -> int:  # pragma: no cover - GUI
    """Open an Open3D picking window and return the first picked point."""
    o3d = _load_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_mm)
    cloud.paint_uniform_color([0.72, 0.75, 0.80])
    viewer = o3d.visualization.VisualizerWithEditing()
    if not viewer.create_window(window_name="Pick query: Shift+left-click, then Q"):
        raise RuntimeError("Failed to create Open3D query picker")
    viewer.add_geometry(cloud)
    viewer.run()
    viewer.destroy_window()
    picked = viewer.get_picked_points()
    if not picked:
        raise RuntimeError("No query point was selected")
    return int(picked[0])


def _point_cloud(o3d, points: np.ndarray, colors: np.ndarray | Sequence[float]):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    color = np.asarray(colors, dtype=np.float64)
    if color.ndim == 1:
        color = np.repeat(color.reshape(1, 3), len(points), axis=0)
    cloud.colors = o3d.utility.Vector3dVector(color)
    return cloud


def _wire_sphere(o3d, center: np.ndarray, radius: float, rings: int = 10, segments: int = 24):
    vertices: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        z = radius * math.cos(phi)
        rr = radius * math.sin(phi)
        base = len(vertices)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(center + np.array([rr * math.cos(theta), rr * math.sin(theta), z]))
        for segment in range(segments):
            lines.append((base + segment, base + (segment + 1) % segments))
    for longitude in range(0, segments, max(1, segments // 12)):
        base = len(vertices)
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            theta = 2.0 * math.pi * longitude / segments
            vertices.append(
                center
                + radius
                * np.array(
                    [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]
                )
            )
        lines.extend((base + ring, base + ring + 1) for ring in range(rings))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    line_set.paint_uniform_color([0.20, 0.45, 0.95])
    return line_set


def _score_colors(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Small dependency-free blue/cyan/yellow/red score map."""
    stops = np.array(
        [[0.05, 0.10, 0.35], [0.00, 0.70, 0.95], [1.00, 0.92, 0.10], [0.90, 0.05, 0.02]]
    )
    value = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 0.30) / 0.30
    scaled = value * (len(stops) - 1)
    lower = np.minimum(scaled.astype(int), len(stops) - 2)
    fraction = scaled - lower
    colors = stops[lower] * (1.0 - fraction[:, None]) + stops[lower + 1] * fraction[:, None]
    colors[~valid] = [0.25, 0.25, 0.25]
    return colors


def make_visualization_geometries(
    scene: NormalConeScene,
    query: NormalConeResult,
    detection: SceneDetectionResult,
    config: NormalConeConfig,
):
    """Build the three Open3D views without opening windows."""
    o3d = _load_open3d()
    query_mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=max(0.8, 0.6 * config.voxel_size_mm)
    )
    query_mesh.translate(query.query_point_mm)
    query_mesh.paint_uniform_color([1.0, 0.05, 0.02])
    query_mesh.compute_vertex_normals()

    query_geometry = [
        _point_cloud(o3d, scene.points_mm, [0.55, 0.57, 0.60]),
        _point_cloud(o3d, scene.points_mm[query.neighbor_indices], [1.0, 0.72, 0.05]),
        _wire_sphere(o3d, query.query_point_mm, config.query_radius_mm),
        query_mesh,
    ]
    cone_geometry = [
        _wire_sphere(o3d, np.zeros(3), 1.0),
        _point_cloud(o3d, scene.directions[~query.allowed_mask], [0.48, 0.48, 0.48]),
        _point_cloud(o3d, scene.directions[query.allowed_mask], [1.0, 0.03, 0.02]),
    ]
    score_geometry = [
        _point_cloud(o3d, scene.points_mm, _score_colors(detection.scores, detection.valid_mask))
    ]
    for index in detection.candidate_indices[:256]:
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=max(0.9, 0.18 * config.nms_radius_mm)
        )
        marker.translate(scene.points_mm[index])
        marker.paint_uniform_color([1.0, 0.0, 0.85])
        marker.compute_vertex_normals()
        score_geometry.append(marker)
    return query_geometry, cone_geometry, score_geometry


def show_open3d_views(
    scene: NormalConeScene,
    query: NormalConeResult,
    detection: SceneDetectionResult,
    config: NormalConeConfig,
) -> None:  # pragma: no cover - GUI
    """Open Query, Normal Cone, and Corner Score windows together."""
    o3d = _load_open3d()
    geometry_sets = make_visualization_geometries(scene, query, detection, config)
    titles = (
        f"1 Query | neighbors={len(query.neighbor_indices)} r={config.query_radius_mm:g}mm",
        f"2 Normal Cone | score={query.score:.4f} allowed={np.count_nonzero(query.allowed_mask)}/{len(query.allowed_mask)}",
        f"3 Corner Score | NMS candidates={len(detection.candidate_indices)}",
    )
    windows = []
    for number, (title, geometries) in enumerate(zip(titles, geometry_sets)):
        viewer = o3d.visualization.Visualizer()
        ok = viewer.create_window(title, width=600, height=600, left=30 + number * 620, top=50)
        if not ok:
            for existing in windows:
                existing.destroy_window()
            raise RuntimeError("Failed to create Open3D result windows")
        for geometry in geometries:
            viewer.add_geometry(geometry)
        option = viewer.get_render_option()
        if option is not None:
            option.background_color = np.array([0.025, 0.025, 0.025])
            option.point_size = 4.0
        windows.append(viewer)

    active = [True] * len(windows)
    while any(active):
        for index, viewer in enumerate(windows):
            if active[index]:
                active[index] = bool(viewer.poll_events())
                if active[index]:
                    viewer.update_renderer()
    for viewer in windows:
        viewer.destroy_window()


def _progress_printer(every: int):
    last = {"position": 0}

    def report(position: int, total: int, elapsed: float) -> None:
        if position == total or position - last["position"] >= every:
            rate = position / max(elapsed, 1e-9)
            remaining = (total - position) / max(rate, 1e-9)
            print(
                f"score {position:>7}/{total} | {rate:7.1f} query/s | ETA {remaining:7.1f}s",
                flush=True,
            )
            last["position"] = position

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", help="STL, point-cloud, NPY, or NPZ input")
    parser.add_argument("--self-test", action="store_true", help="Run headless synthetic validation")
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--query-radius-mm", type=float, default=30.0)
    parser.add_argument("--sphere-samples", type=int, default=1024)
    parser.add_argument("--tolerance-mm", type=float, default=0.6)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-neighbors", type=int, default=12)
    parser.add_argument("--direction-chunk-size", type=int, default=256)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--local-max-radius-mm", type=float, default=3.0)
    parser.add_argument("--nms-radius-mm", type=float, default=6.0)
    parser.add_argument(
        "--multiscale-radii-mm",
        type=float,
        nargs="+",
        help="Use radius consensus, for example: 12 18 24 30",
    )
    parser.add_argument("--multiscale-min-scales", type=int)
    parser.add_argument("--multiscale-min-consistency", type=float, default=0.35)
    parser.add_argument(
        "--geometry-adaptive",
        action="store_true",
        help="Stop before a covariance-dimensionality transition enters the neighborhood",
    )
    parser.add_argument("--geometry-min-clean-scales", type=int, default=3)
    parser.add_argument("--geometry-min-shell-points", type=int, default=12)
    parser.add_argument("--geometry-max-anchor-js", type=float, default=0.20)
    parser.add_argument("--geometry-max-step-js", type=float, default=0.14)
    parser.add_argument("--geometry-max-shell-js", type=float, default=0.52)
    parser.add_argument("--query-index", type=int)
    parser.add_argument("--query-xyz-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--no-detect", action="store_true", help="Phase A only")
    parser.add_argument(
        "--phase-c",
        action="store_true",
        help="Verify Phase-B NMS candidates with coordinate-only Phase C",
    )
    parser.add_argument("--verifier-min-support", type=int, default=40)
    parser.add_argument("--outer-shell-start-ratio", type=float, default=0.70)
    parser.add_argument("--min-outer-shell-fraction", type=float, default=0.15)
    parser.add_argument("--min-radius-p90-ratio", type=float, default=0.75)
    parser.add_argument("--min-centroid-offset-ratio", type=float, default=0.42)
    parser.add_argument("--min-directional-concentration", type=float, default=0.68)
    parser.add_argument("--repeatability-trials", type=int, default=3)
    parser.add_argument("--min-repeatability", type=float, default=0.55)
    parser.add_argument("--min-split-score-ratio", type=float, default=0.50)
    parser.add_argument("--repeatability-seed", type=int, default=7)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--output-npz", help="Save points, scores, and candidates")
    parser.add_argument("--progress-every", type=int, default=100)

    # Existing STL virtual-depth-capture controls.
    parser.add_argument("--stl-unit-mm", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not args.self_test and not args.input:
        parser.error("--input is required unless --self-test is used")
    if args.geometry_adaptive and args.multiscale_radii_mm is None:
        parser.error("--geometry-adaptive requires --multiscale-radii-mm")
    return args


def _config_from_args(args: argparse.Namespace) -> NormalConeConfig:
    return NormalConeConfig(
        voxel_size_mm=args.voxel_mm,
        query_radius_mm=args.query_radius_mm,
        sphere_samples=args.sphere_samples,
        tolerance_mm=args.tolerance_mm,
        quantile=args.quantile,
        min_neighbors=args.min_neighbors,
        direction_chunk_size=args.direction_chunk_size,
        min_score=args.min_score,
        local_max_radius_mm=args.local_max_radius_mm,
        nms_radius_mm=args.nms_radius_mm,
    )


def _phase_c_config_from_args(args: argparse.Namespace) -> PhaseCConfig:
    return PhaseCConfig(
        min_support=args.verifier_min_support,
        outer_shell_start_ratio=args.outer_shell_start_ratio,
        min_outer_shell_fraction=args.min_outer_shell_fraction,
        min_radius_p90_ratio=args.min_radius_p90_ratio,
        min_centroid_offset_ratio=args.min_centroid_offset_ratio,
        min_directional_concentration=args.min_directional_concentration,
        repeatability_trials=args.repeatability_trials,
        min_repeatability=args.min_repeatability,
        min_split_score_ratio=args.min_split_score_ratio,
        repeatability_seed=args.repeatability_seed,
    )


def _geometry_scale_config_from_args(args: argparse.Namespace) -> GeometryScaleConfig:
    return GeometryScaleConfig(
        min_clean_scale_count=args.geometry_min_clean_scales,
        min_score_consistency=args.multiscale_min_consistency,
        min_shell_points=args.geometry_min_shell_points,
        max_anchor_js=args.geometry_max_anchor_js,
        max_step_js=args.geometry_max_step_js,
        max_shell_js=args.geometry_max_shell_js,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _config_from_args(args)
    config.validate()

    if args.self_test:
        result = run_synthetic_validation(config)
        print("Normal Cone synthetic validation")
        for name, value in result.items():
            if name != "separation_pass":
                print(f"  {name:20s} {float(value):.6f}")
        print(f"  {'separation_pass':20s} {result['separation_pass']}")
        if not result["separation_pass"]:
            raise SystemExit(1)
        return

    raw_points = load_points_mm(args.input, args)
    scene = build_scene(raw_points, config, downsample=True)
    print(f"points: raw={len(raw_points)} voxelized={len(scene.points_mm)}")

    if args.query_index is not None:
        query_index = int(args.query_index)
    elif args.query_xyz_mm is not None:
        query_index = nearest_point_index(scene.points_mm, np.asarray(args.query_xyz_mm))
    elif args.no_visualize:
        raise ValueError("Headless mode requires --query-index or --query-xyz-mm")
    else:
        query_index = pick_query_index(scene.points_mm)

    query = estimate_normal_cone(scene, query_index, config)
    print(
        f"query={query.query_index} xyz_mm={np.array2string(query.query_point_mm, precision=3)} "
        f"neighbors={len(query.neighbor_indices)} allowed={np.count_nonzero(query.allowed_mask)}/"
        f"{len(query.allowed_mask)} score={query.score:.6f}"
    )

    verification: PhaseCResult | None = None
    multiscale_detection: MultiScaleDetectionResult | None = None
    geometry_detection: GeometryScaleDetectionResult | None = None
    if args.no_detect:
        detection = SceneDetectionResult(
            scores=np.zeros(len(scene.points_mm)),
            neighbor_count=np.zeros(len(scene.points_mm), dtype=np.int32),
            valid_mask=np.zeros(len(scene.points_mm), dtype=bool),
            local_max_mask=np.zeros(len(scene.points_mm), dtype=bool),
            local_max_indices=np.empty(0, dtype=np.int64),
            candidate_indices=np.empty(0, dtype=np.int64),
            elapsed_seconds=0.0,
        )
        detection.scores[query_index] = query.score
        detection.valid_mask[query_index] = True
    else:
        if args.multiscale_radii_mm is not None:
            scale_count = len(args.multiscale_radii_mm)
            multiscale_config = MultiScaleConfig(
                radii_mm=tuple(args.multiscale_radii_mm),
                min_scale_count=(
                    args.multiscale_min_scales
                    if args.multiscale_min_scales is not None
                    else min(3, scale_count)
                ),
                min_consistency=args.multiscale_min_consistency,
            )
            multiscale_detection = detect_scene_multiscale(
                scene,
                config,
                multiscale_config,
                progress=_progress_printer(max(1, args.progress_every)),
            )
            if args.geometry_adaptive:
                geometry_detection = detect_scene_geometry_adaptive(
                    scene,
                    config,
                    multiscale_detection,
                    _geometry_scale_config_from_args(args),
                    progress=_progress_printer(max(1, args.progress_every)),
                )
                detection = geometry_detection.as_scene_detection(multiscale_detection)
                transitions = geometry_detection.transition_scale_index < len(
                    geometry_detection.radii_mm
                )
                selected = geometry_detection.selected_radius_mm[
                    geometry_detection.valid_mask
                ]
                selected_summary = (
                    f"{np.median(selected):.1f}mm median" if len(selected) else "none"
                )
                print(
                    f"geometry-adaptive: radii_mm={geometry_detection.radii_mm.tolist()} "
                    f"transitions={np.count_nonzero(transitions)} "
                    f"selected={selected_summary} "
                    f"valid={np.count_nonzero(detection.valid_mask)} "
                    f"local_max={len(detection.local_max_indices)} "
                    f"nms={len(detection.candidate_indices)} "
                    f"elapsed={detection.elapsed_seconds:.2f}s"
                )
            else:
                detection = multiscale_detection.as_scene_detection()
                print(
                    f"multiscale: radii_mm={multiscale_detection.radii_mm.tolist()} "
                    f"valid={np.count_nonzero(detection.valid_mask)} "
                    f"local_max={len(detection.local_max_indices)} "
                    f"nms={len(detection.candidate_indices)} "
                    f"elapsed={detection.elapsed_seconds:.2f}s"
                )
        else:
            detection = detect_scene(
                scene,
                config,
                progress=_progress_printer(max(1, args.progress_every)),
            )
            print(
                f"scene: valid={np.count_nonzero(detection.valid_mask)} "
                f"local_max={len(detection.local_max_indices)} "
                f"nms={len(detection.candidate_indices)} "
                f"elapsed={detection.elapsed_seconds:.2f}s"
            )
        for rank, index in enumerate(detection.candidate_indices[:20], start=1):
            print(
                f"  #{rank:02d} index={index:7d} score={detection.scores[index]:.6f} "
                f"xyz_mm={scene.points_mm[index]}"
            )

        if args.phase_c:
            verifier = _phase_c_config_from_args(args)
            verification = verify_candidates_phase_c(
                scene,
                detection,
                config,
                verifier,
                query_radius_mm_by_index=(
                    geometry_detection.selected_radius_mm
                    if geometry_detection is not None
                    else None
                ),
                progress=_progress_printer(max(1, args.progress_every)),
            )
            print(
                f"phase-c: input={len(verification.input_indices)} "
                f"support={len(verification.support_indices)} "
                f"spatial={len(verification.spatial_indices)} "
                f"repeatable={len(verification.accepted_indices)} "
                f"elapsed={verification.elapsed_seconds:.2f}s"
            )

    if args.output_npz:
        output = Path(args.output_npz).expanduser().resolve()
        saved = {
            "points_mm": scene.points_mm,
            "directions": scene.directions,
            "scores": detection.scores,
            "neighbor_count": detection.neighbor_count,
            "valid_mask": detection.valid_mask,
            "local_max_indices": detection.local_max_indices,
            "candidate_indices": detection.candidate_indices,
            "query_index": np.array(query_index, dtype=np.int64),
            "query_allowed_mask": query.allowed_mask,
        }
        if verification is not None:
            saved.update(
                phase_c_support_pass=verification.support_pass,
                phase_c_spatial_pass=verification.spatial_pass,
                phase_c_repeatability_pass=verification.repeatability_pass,
                phase_c_outer_shell_fraction=verification.outer_shell_fraction,
                phase_c_radius_p90_ratio=verification.radius_p90_ratio,
                phase_c_centroid_offset_ratio=verification.centroid_offset_ratio,
                phase_c_directional_concentration=verification.directional_concentration,
                phase_c_repeatability=verification.repeatability,
                phase_c_split_score_ratio=verification.split_score_ratio,
                verified_candidate_indices=verification.accepted_indices,
            )
        if multiscale_detection is not None:
            saved.update(
                multiscale_radii_mm=multiscale_detection.radii_mm,
                multiscale_scores=multiscale_detection.scale_scores,
                multiscale_neighbor_count=multiscale_detection.scale_neighbor_count,
                multiscale_pass_count=multiscale_detection.scale_pass_count,
                multiscale_consistency=multiscale_detection.scale_consistency,
            )
        if geometry_detection is not None:
            saved.update(
                geometry_dimensionality=geometry_detection.dimensionality,
                geometry_dimensionality_entropy=geometry_detection.dimensionality_entropy,
                geometry_shell_dimensionality=geometry_detection.shell_dimensionality,
                geometry_anchor_js=geometry_detection.anchor_js,
                geometry_step_js=geometry_detection.step_js,
                geometry_shell_js=geometry_detection.shell_js,
                geometry_transition_scale_index=geometry_detection.transition_scale_index,
                geometry_selected_scale_index=geometry_detection.selected_scale_index,
                geometry_selected_radius_mm=geometry_detection.selected_radius_mm,
                geometry_clean_scale_count=geometry_detection.clean_scale_count,
            )
        np.savez_compressed(output, **saved)
        print(f"saved: {output}")

    if not args.no_visualize:
        display_detection = detection
        if verification is not None:
            display_detection = SceneDetectionResult(
                scores=detection.scores,
                neighbor_count=detection.neighbor_count,
                valid_mask=detection.valid_mask,
                local_max_mask=detection.local_max_mask,
                local_max_indices=detection.local_max_indices,
                candidate_indices=verification.accepted_indices,
                elapsed_seconds=detection.elapsed_seconds,
            )
        show_open3d_views(scene, query, display_detection, config)


if __name__ == "__main__":
    main()
