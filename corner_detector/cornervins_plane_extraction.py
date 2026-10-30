"""Clean-room CornerVINS-style plane extraction for organized RGB-D clouds.

The AHC stage follows Algorithms 2--4 of Feng et al. (ICRA 2014) from the
paper description.  No PEAC source code is copied here; the published PEAC
implementation is AGPL-3.0-or-later and is used only as a behavioral
reference.  Coordinates, residuals, and thresholds in this module are in mm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class CornerVINSPlaneConfig:
    """Numerical configuration for AHC, RANSAC, merging, and hull fitting."""

    ahc_block_size: int = 10
    ahc_mse_threshold_mm2: float | None = None
    ahc_refinement: bool = True
    region_mode: str = "roi_test"
    min_region_points_paper: int = 20_000
    min_region_points_roi: int = 100
    ransac_threshold_mm: float = 1.0
    ransac_min_inlier_ratio: float = 0.90
    ransac_max_trials: int = 256
    ransac_seed: int = 0
    downsample_leaf_mm: float = 50.0
    merge_angle_deg: float = 5.0
    merge_distance_mm: float = 20.0

    # Feng et al. Kinect noise model.  These remain explicit because
    # ``ahc_mse_threshold_mm2=None`` selects this depth-dependent model.
    ahc_depth_sigma_per_mm: float = 1.6e-6
    ahc_init_tolerance_mm: float = 5.0
    ahc_merge_tolerance_mm: float = 8.0
    ahc_depth_change_factor: float = 0.04
    ahc_depth_change_tolerance_mm: float = 0.02
    ahc_angle_near_deg: float = 15.0
    ahc_angle_far_deg: float = 90.0
    ahc_angle_near_depth_mm: float = 100.0
    ahc_angle_far_depth_mm: float = 4_000.0

    @classmethod
    def from_settings(cls, settings: Any) -> "CornerVINSPlaneConfig":
        """Read resolved project settings without coupling to their class."""

        return cls(
            ahc_block_size=int(settings.cornervins_ahc_block_size),
            ahc_mse_threshold_mm2=settings.cornervins_ahc_mse_threshold_mm2,
            ahc_refinement=bool(settings.cornervins_ahc_refinement),
            region_mode=str(settings.cornervins_region_mode),
            min_region_points_paper=int(
                settings.cornervins_min_region_points_paper
            ),
            min_region_points_roi=int(settings.cornervins_min_region_points_roi),
            ransac_threshold_mm=float(settings.cornervins_ransac_threshold_mm),
            ransac_min_inlier_ratio=float(
                settings.cornervins_ransac_min_inlier_ratio
            ),
            ransac_max_trials=int(settings.cornervins_ransac_max_trials),
            ransac_seed=int(settings.cornervins_ransac_seed),
            downsample_leaf_mm=float(settings.cornervins_downsample_leaf_mm),
            merge_angle_deg=float(settings.cornervins_merge_angle_deg),
            merge_distance_mm=float(settings.cornervins_merge_distance_mm),
        )

    @property
    def min_region_points(self) -> int:
        if self.region_mode == "paper":
            return int(self.min_region_points_paper)
        if self.region_mode == "roi_test":
            return int(self.min_region_points_roi)
        raise ValueError("region_mode must be 'paper' or 'roi_test'")

    def validate(self) -> None:
        if self.ahc_block_size < 2:
            raise ValueError("AHC block size must be at least 2")
        if self.region_mode not in ("paper", "roi_test"):
            raise ValueError("CornerVINS region mode must be paper or roi_test")
        if min(self.min_region_points_paper, self.min_region_points_roi) < 3:
            raise ValueError("CornerVINS minimum region sizes must be at least 3")
        if (
            self.ahc_mse_threshold_mm2 is not None
            and (
                not np.isfinite(self.ahc_mse_threshold_mm2)
                or self.ahc_mse_threshold_mm2 <= 0.0
            )
        ):
            raise ValueError("AHC MSE threshold must be positive or None")
        positive = (
            self.ransac_threshold_mm,
            self.downsample_leaf_mm,
            self.merge_angle_deg,
            self.merge_distance_mm,
            self.ahc_depth_sigma_per_mm,
            self.ahc_init_tolerance_mm,
            self.ahc_merge_tolerance_mm,
            self.ahc_depth_change_factor,
            self.ahc_angle_far_depth_mm - self.ahc_angle_near_depth_mm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("CornerVINS metric thresholds must be finite and positive")
        if not 0.0 < self.ransac_min_inlier_ratio <= 1.0:
            raise ValueError("RANSAC minimum inlier ratio must be in (0, 1]")
        if self.ransac_max_trials < 1:
            raise ValueError("RANSAC max trials must be positive")
        if not 0.0 < self.merge_angle_deg < 90.0:
            raise ValueError("Plane merge angle must be in (0, 90) degrees")


@dataclass(frozen=True)
class PlanePatch:
    plane_id: int
    normal: np.ndarray
    offset: float
    center: np.ndarray
    point_indices: np.ndarray
    inlier_indices: np.ndarray
    hull_vertices_3d: np.ndarray
    residual_rmse: float
    inlier_ratio: float
    organized_point_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    organized_inlier_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


@dataclass(frozen=True)
class PlaneStagePatch:
    """Diagnostic plane state before compatible-patch merging."""

    plane_id: int
    normal: np.ndarray
    offset: float
    center: np.ndarray
    organized_point_indices: np.ndarray
    organized_inlier_indices: np.ndarray
    residual_rmse: float
    inlier_ratio: float


@dataclass(frozen=True)
class PlaneExtractionResult:
    patches: tuple[PlanePatch, ...]
    ahc_regions: tuple[np.ndarray, ...]
    ahc_membership_image: np.ndarray
    final_membership_image: np.ndarray
    diagnostics: Mapping[str, Any]
    ransac_patches: tuple[PlaneStagePatch, ...] = ()

    @property
    def refined_membership_image(self) -> np.ndarray:
        """Feng Algorithm 4 output retained under the compatible field name."""

        return self.final_membership_image


@dataclass(frozen=True)
class _PlaneFit:
    center: np.ndarray
    normal: np.ndarray
    offset: float
    mse: float


@dataclass(frozen=True)
class _Stats:
    count: int
    total: np.ndarray
    outer: np.ndarray


@dataclass
class _Region:
    region_id: int
    pixel_indices: np.ndarray
    stats: _Stats
    fit: _PlaneFit
    neighbors: set[int]
    active: bool = True


@dataclass(frozen=True)
class _RawPatch:
    region_pixels: np.ndarray
    inlier_pixels: np.ndarray
    fit: _PlaneFit
    residual_rmse: float
    inlier_ratio: float


def _stats_from_points(points: np.ndarray) -> _Stats:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return _Stats(
        count=int(len(pts)),
        total=np.sum(pts, axis=0),
        outer=pts.T @ pts,
    )


def _merge_stats(first: _Stats, second: _Stats) -> _Stats:
    return _Stats(
        count=first.count + second.count,
        total=first.total + second.total,
        outer=first.outer + second.outer,
    )


def _canonical_fit(stats: _Stats) -> _PlaneFit:
    if stats.count < 3:
        raise ValueError("At least three points are required to fit a plane")
    center = stats.total / float(stats.count)
    covariance = stats.outer / float(stats.count) - np.outer(center, center)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmin(values))]
    normal /= np.linalg.norm(normal)
    # PEAC fixes one arbitrary sign.  Camera-facing orientation is applied to
    # accepted final patches, where the camera-frame origin has meaning.
    if normal[2] < 0.0:
        normal = -normal
    mse = max(0.0, float(np.min(values)))
    return _PlaneFit(
        center=center,
        normal=normal,
        offset=-float(normal @ center),
        mse=mse,
    )


def _fit_points(points: np.ndarray, *, orient_to_camera: bool = False) -> _PlaneFit:
    fit = _canonical_fit(_stats_from_points(points))
    normal = fit.normal.copy()
    if orient_to_camera and float(normal @ fit.center) > 0.0:
        normal = -normal
    return _PlaneFit(
        center=fit.center.copy(),
        normal=normal,
        offset=-float(normal @ fit.center),
        mse=fit.mse,
    )


def _mse_threshold(config: CornerVINSPlaneConfig, depth_mm: float, phase: str) -> float:
    if config.ahc_mse_threshold_mm2 is not None:
        return float(config.ahc_mse_threshold_mm2)
    tolerance = (
        config.ahc_init_tolerance_mm
        if phase == "init"
        else config.ahc_merge_tolerance_mm
    )
    standard_deviation = config.ahc_depth_sigma_per_mm * depth_mm * depth_mm
    return float((standard_deviation + tolerance) ** 2)


def _initial_angle_threshold_deg(
    config: CornerVINSPlaneConfig, depth_mm: float
) -> float:
    clipped = np.clip(
        depth_mm,
        config.ahc_angle_near_depth_mm,
        config.ahc_angle_far_depth_mm,
    )
    fraction = (
        (clipped - config.ahc_angle_near_depth_mm)
        / (config.ahc_angle_far_depth_mm - config.ahc_angle_near_depth_mm)
    )
    return float(
        config.ahc_angle_near_deg
        + fraction * (config.ahc_angle_far_deg - config.ahc_angle_near_deg)
    )


def _depth_discontinuity_mask(
    organized: np.ndarray,
    valid: np.ndarray,
    config: CornerVINSPlaneConfig,
) -> np.ndarray:
    """Flag pixels discontinuous with at least one valid 4-neighbor."""

    depth = organized[..., 2]
    discontinuous = np.zeros(valid.shape, dtype=bool)
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair_valid = valid[first] & valid[second]
        first_depth = depth[first]
        second_depth = depth[second]
        first_bad = np.abs(first_depth - second_depth) > (
            config.ahc_depth_change_factor * np.abs(first_depth)
            + config.ahc_depth_change_tolerance_mm
        )
        second_bad = np.abs(first_depth - second_depth) > (
            config.ahc_depth_change_factor * np.abs(second_depth)
            + config.ahc_depth_change_tolerance_mm
        )
        bad = pair_valid & (first_bad | second_bad)
        discontinuous[first] |= bad
        discontinuous[second] |= bad
    return discontinuous


def _build_initial_graph(
    organized: np.ndarray,
    valid: np.ndarray,
    config: CornerVINSPlaneConfig,
) -> tuple[dict[int, _Region], dict[tuple[int, int], int], dict[str, int]]:
    height, width = valid.shape
    block = int(config.ahc_block_size)
    discontinuous = _depth_discontinuity_mask(organized, valid, config)
    regions: dict[int, _Region] = {}
    block_to_region: dict[tuple[int, int], int] = {}
    rejected = {"missing": 0, "depth_discontinuity": 0, "mse": 0}
    block_rows = (height + block - 1) // block
    block_cols = (width + block - 1) // block

    for block_row in range(block_rows):
        row_start = block_row * block
        row_stop = min(height, row_start + block)
        for block_col in range(block_cols):
            col_start = block_col * block
            col_stop = min(width, col_start + block)
            block_valid = valid[row_start:row_stop, col_start:col_stop]
            # Algorithm 2 REJECTNODE uses strict missing-data rejection.
            if not np.all(block_valid):
                rejected["missing"] += 1
                continue
            if np.any(discontinuous[row_start:row_stop, col_start:col_stop]):
                rejected["depth_discontinuity"] += 1
                continue
            yy, xx = np.mgrid[row_start:row_stop, col_start:col_stop]
            pixels = (yy * width + xx).reshape(-1).astype(np.int64)
            if len(pixels) < 3:
                rejected["missing"] += 1
                continue
            stats = _stats_from_points(organized.reshape(-1, 3)[pixels])
            fit = _canonical_fit(stats)
            if fit.mse > _mse_threshold(config, fit.center[2], "init"):
                rejected["mse"] += 1
                continue
            region_id = len(regions)
            regions[region_id] = _Region(
                region_id=region_id,
                pixel_indices=pixels,
                stats=stats,
                fit=fit,
                neighbors=set(),
            )
            block_to_region[(block_row, block_col)] = region_id

    # Algorithm 2 evaluates triples (left, center, right) and (up, center,
    # down).  Both incident edges are admitted only when all three nodes are
    # valid and the two outer fitted normals agree at the center depth.
    def admit_triple(keys: tuple[tuple[int, int], ...]) -> None:
        if any(key not in block_to_region for key in keys):
            return
        first_id, center_id, last_id = (block_to_region[key] for key in keys)
        first = regions[first_id]
        center = regions[center_id]
        last = regions[last_id]
        cosine = abs(float(first.fit.normal @ last.fit.normal))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if angle > _initial_angle_threshold_deg(config, center.fit.center[2]):
            return
        for left, right in ((first_id, center_id), (center_id, last_id)):
            regions[left].neighbors.add(right)
            regions[right].neighbors.add(left)

    for row in range(block_rows):
        for col in range(1, block_cols - 1):
            admit_triple(((row, col - 1), (row, col), (row, col + 1)))
    for col in range(block_cols):
        for row in range(1, block_rows - 1):
            admit_triple(((row - 1, col), (row, col), (row + 1, col)))
    return regions, block_to_region, rejected


def _cluster_graph(
    regions: dict[int, _Region],
    config: CornerVINSPlaneConfig,
    *,
    minimum_points: int,
) -> list[_Region]:
    """Algorithm 3: minimum-current-MSE AHC with graph edge contraction."""

    heap = [(region.fit.mse, region.region_id) for region in regions.values()]
    heapq.heapify(heap)
    next_id = max(regions, default=-1) + 1
    extracted: list[_Region] = []
    while heap:
        _mse, region_id = heapq.heappop(heap)
        region = regions.get(region_id)
        if region is None or not region.active:
            continue
        candidates: list[tuple[float, int, _Stats, _PlaneFit]] = []
        for neighbor_id in sorted(region.neighbors):
            neighbor = regions.get(neighbor_id)
            if neighbor is None or not neighbor.active:
                continue
            stats = _merge_stats(region.stats, neighbor.stats)
            fit = _canonical_fit(stats)
            candidates.append((fit.mse, neighbor_id, stats, fit))
        candidates.sort(key=lambda item: (item[0], item[1]))
        can_merge = False
        if candidates:
            best_mse, neighbor_id, merged_stats, merged_fit = candidates[0]
            threshold = _mse_threshold(config, merged_fit.center[2], "merge")
            can_merge = best_mse <= threshold
        if not can_merge:
            region.active = False
            for neighbor_id in tuple(region.neighbors):
                neighbor = regions.get(neighbor_id)
                if neighbor is not None:
                    neighbor.neighbors.discard(region_id)
            if region.stats.count >= minimum_points:
                extracted.append(region)
            continue

        neighbor = regions[neighbor_id]
        new_neighbors = (region.neighbors | neighbor.neighbors) - {
            region_id,
            neighbor_id,
        }
        region.active = False
        neighbor.active = False
        for other_id in tuple(new_neighbors):
            other = regions.get(other_id)
            if other is None or not other.active:
                new_neighbors.discard(other_id)
                continue
            other.neighbors.discard(region_id)
            other.neighbors.discard(neighbor_id)
            other.neighbors.add(next_id)
        merged = _Region(
            region_id=next_id,
            pixel_indices=np.concatenate(
                (region.pixel_indices, neighbor.pixel_indices)
            ),
            stats=merged_stats,
            fit=merged_fit,
            neighbors=set(new_neighbors),
        )
        regions[next_id] = merged
        heapq.heappush(heap, (merged.fit.mse, next_id))
        next_id += 1
    extracted.sort(key=lambda item: int(np.min(item.pixel_indices)))
    return extracted


def _membership_image(
    regions: list[_Region], height: int, width: int
) -> np.ndarray:
    membership = np.full(height * width, -1, dtype=np.int32)
    for label, region in enumerate(regions):
        membership[region.pixel_indices] = label
    return membership.reshape(height, width)


def _refine_regions(
    coarse: list[_Region],
    block_to_region: dict[tuple[int, int], int],
    organized: np.ndarray,
    valid: np.ndarray,
    config: CornerVINSPlaneConfig,
) -> list[_Region]:
    """Algorithm 4: block erosion, pixel region growing, and final AHC."""

    if not coarse:
        return []
    height, width = valid.shape
    block = int(config.ahc_block_size)
    source_to_coarse: dict[int, int] = {}
    for coarse_id, region in enumerate(coarse):
        # Initial region IDs are not retained after contractions, so recover
        # coarse block membership from pixel coordinates.
        block_keys = set(
            (int(pixel // width) // block, int(pixel % width) // block)
            for pixel in region.pixel_indices
        )
        for key in block_keys:
            source_id = block_to_region.get(key)
            if source_id is not None:
                source_to_coarse[source_id] = coarse_id

    coarse_blocks: list[set[tuple[int, int]]] = [set() for _ in coarse]
    for key, source_id in block_to_region.items():
        coarse_id = source_to_coarse.get(source_id)
        if coarse_id is not None:
            coarse_blocks[coarse_id].add(key)

    labels = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()
    active_plane: dict[int, _PlaneFit] = {}
    for coarse_id, (region, keys) in enumerate(zip(coarse, coarse_blocks)):
        interior = {
            key
            for key in keys
            if all(
                neighbor in keys
                for neighbor in (
                    (key[0] - 1, key[1]),
                    (key[0] + 1, key[1]),
                    (key[0], key[1] - 1),
                    (key[0], key[1] + 1),
                )
            )
        }
        if not interior:
            continue
        active_plane[coarse_id] = region.fit
        for block_row, block_col in interior:
            row_start, col_start = block_row * block, block_col * block
            row_stop, col_stop = min(height, row_start + block), min(
                width, col_start + block
            )
            labels[row_start:row_stop, col_start:col_stop] = coarse_id

    # Seed the queue with all pixel-level core boundaries.
    for row in range(height):
        for col in range(width):
            label = int(labels[row, col])
            if label < 0:
                continue
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + drow, col + dcol
                if not (0 <= nr < height and 0 <= nc < width) or labels[nr, nc] != label:
                    queue.append((row, col, label))
                    break

    best_distance = np.full((height, width), np.inf, dtype=np.float64)
    adjacency: set[tuple[int, int]] = set()
    while queue:
        row, col, label = queue.popleft()
        fit = active_plane[label]
        limit = 9.0 * max(fit.mse, np.finfo(np.float64).eps)
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + drow, col + dcol
            if not (0 <= nr < height and 0 <= nc < width) or not valid[nr, nc]:
                continue
            current = int(labels[nr, nc])
            if current == label:
                continue
            point = organized[nr, nc]
            distance = abs(float(fit.normal @ point + fit.offset))
            if distance * distance > limit:
                continue
            if current >= 0:
                adjacency.add(tuple(sorted((label, current))))
                other_fit = active_plane[current]
                other_distance = abs(float(other_fit.normal @ point + other_fit.offset))
                if distance >= other_distance:
                    continue
            if distance >= best_distance[nr, nc]:
                continue
            labels[nr, nc] = label
            best_distance[nr, nc] = distance
            queue.append((nr, nc, label))

    # Pixel adjacency is the final graph G' in Algorithm 4.  Scanning the
    # resulting label image also captures contacts not contested in the queue.
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        mask = (first >= 0) & (second >= 0) & (first != second)
        for left, right in zip(first[mask], second[mask]):
            adjacency.add(tuple(sorted((int(left), int(right)))))

    graph: dict[int, _Region] = {}
    flat_points = organized.reshape(-1, 3)
    for label in sorted(active_plane):
        pixels = np.flatnonzero(labels.reshape(-1) == label).astype(np.int64)
        if len(pixels) < 3:
            continue
        stats = _stats_from_points(flat_points[pixels])
        graph[label] = _Region(
            region_id=label,
            pixel_indices=pixels,
            stats=stats,
            fit=_canonical_fit(stats),
            neighbors=set(),
        )
    for first, second in adjacency:
        if first in graph and second in graph:
            graph[first].neighbors.add(second)
            graph[second].neighbors.add(first)
    return _cluster_graph(
        graph,
        config,
        minimum_points=config.min_region_points,
    )


def _voxel_representatives(points: np.ndarray, leaf_mm: float) -> np.ndarray:
    if len(points) <= 3:
        return np.arange(len(points), dtype=np.int64)
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / leaf_mm).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first.astype(np.int64))


def _ransac_plane(
    points: np.ndarray,
    config: CornerVINSPlaneConfig,
    *,
    seed_offset: int,
) -> tuple[_PlaneFit, np.ndarray, float] | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        return None
    sample_ids = _voxel_representatives(points, config.downsample_leaf_mm)
    if len(sample_ids) < 3:
        sample_ids = np.arange(len(points), dtype=np.int64)
    samples = points[sample_ids]
    rng = np.random.default_rng(int(config.ransac_seed) + int(seed_offset))
    best: tuple[int, float, np.ndarray, float] | None = None
    for _ in range(int(config.ransac_max_trials)):
        chosen = rng.choice(len(samples), size=3, replace=False)
        first, second, third = samples[chosen]
        normal = np.cross(second - first, third - first)
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            continue
        normal /= length
        offset = -float(normal @ first)
        residual = np.abs(points @ normal + offset)
        inliers = residual <= config.ransac_threshold_mm
        count = int(np.count_nonzero(inliers))
        score = float(np.mean(residual[inliers])) if count else np.inf
        key = (count, -score)
        if best is None or key > (best[0], -best[1]):
            best = (count, score, inliers, offset)
    if best is None or best[0] < 3:
        return None
    inliers = best[2]
    fit = _fit_points(points[inliers], orient_to_camera=True)
    # Reclassify once against the least-squares model, then refit exactly the
    # accepted full-resolution region inliers as requested by CornerVINS.
    inliers = np.abs(points @ fit.normal + fit.offset) <= config.ransac_threshold_mm
    if np.count_nonzero(inliers) < 3:
        return None
    fit = _fit_points(points[inliers], orient_to_camera=True)
    residual = np.abs(points @ fit.normal + fit.offset)
    inliers = residual <= config.ransac_threshold_mm
    ratio = float(np.count_nonzero(inliers) / len(points))
    if ratio < config.ransac_min_inlier_ratio:
        return None
    fit = _fit_points(points[inliers], orient_to_camera=True)
    inlier_residual = points[inliers] @ fit.normal + fit.offset
    rmse = float(np.sqrt(np.mean(inlier_residual * inlier_residual)))
    return fit, inliers, rmse


def _plane_distance(first: _RawPatch, second: _RawPatch) -> float:
    """Symmetric centroid-to-plane distance used for CornerVINS merging.

    CornerVINS gives the 20 mm gate but not a complete distance equation.
    We align signs through the angle test and use the larger of the two
    centroid-to-other-plane residuals.  This is symmetric, is exactly the
    parallel-plane separation for parallel fits, and avoids origin dependence.
    """

    return max(
        abs(float(first.fit.normal @ second.fit.center + first.fit.offset)),
        abs(float(second.fit.normal @ first.fit.center + second.fit.offset)),
    )


def _patches_mergeable(
    first: _RawPatch,
    second: _RawPatch,
    config: CornerVINSPlaneConfig,
) -> bool:
    cosine = abs(float(first.fit.normal @ second.fit.normal))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return (
        angle < config.merge_angle_deg
        and _plane_distance(first, second) < config.merge_distance_mm
    )


def _merge_raw_patches(
    patches: list[_RawPatch],
    flat_points: np.ndarray,
    config: CornerVINSPlaneConfig,
) -> tuple[list[_RawPatch], int]:
    merged_count = 0
    patches = list(patches)
    while True:
        selected: tuple[int, int, np.ndarray, _PlaneFit, np.ndarray, float] | None = None
        for first in range(len(patches)):
            for second in range(first + 1, len(patches)):
                if not _patches_mergeable(patches[first], patches[second], config):
                    continue
                pixels = np.unique(
                    np.concatenate(
                        (patches[first].region_pixels, patches[second].region_pixels)
                    )
                )
                fitted = _ransac_plane(
                    flat_points[pixels], config, seed_offset=10_000 + merged_count
                )
                if fitted is not None:
                    fit, inlier_mask, rmse = fitted
                    selected = (first, second, pixels, fit, inlier_mask, rmse)
                    break
            if selected is not None:
                break
        if selected is None:
            break
        first, second, pixels, fit, inlier_mask, rmse = selected
        combined = _RawPatch(
            region_pixels=pixels,
            inlier_pixels=pixels[inlier_mask],
            fit=fit,
            residual_rmse=rmse,
            inlier_ratio=float(np.count_nonzero(inlier_mask) / len(pixels)),
        )
        for index in sorted((first, second), reverse=True):
            patches.pop(index)
        patches.append(combined)
        patches.sort(key=lambda item: int(np.min(item.region_pixels)))
        merged_count += 1
    return patches, merged_count


def _convex_hull_3d(points: np.ndarray, fit: _PlaneFit) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    residual = points @ fit.normal + fit.offset
    projected = points - residual[:, None] * fit.normal
    axis = np.eye(3)[int(np.argmin(np.abs(fit.normal)))]
    u = np.cross(fit.normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(fit.normal, u)
    uv = np.column_stack(
        ((projected - fit.center) @ u, (projected - fit.center) @ v)
    )
    if len(uv) < 3:
        return projected.copy()
    try:
        hull = ConvexHull(uv)
    except QhullError:
        # Degenerate projected support is retained as its unique endpoints.
        _, unique = np.unique(np.round(uv, decimals=12), axis=0, return_index=True)
        return projected[np.sort(unique)].copy()
    return projected[np.asarray(hull.vertices, dtype=np.int64)].copy()


def _validate_inputs(
    points: np.ndarray,
    organized_points: np.ndarray | None,
    valid_mask: np.ndarray | None,
    organized_to_point_index: np.ndarray | None,
    roi_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    scene = np.asarray(points, dtype=np.float64)
    if scene.ndim != 2 or scene.shape[1] != 3 or not np.isfinite(scene).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if organized_points is None:
        raise ValueError(
            "CornerVINS AHC requires organized_points with shape (H, W, 3); "
            "an unorganized/KNN fallback is intentionally not implemented"
        )
    organized = np.asarray(organized_points, dtype=np.float64)
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise ValueError("organized_points must have shape (H, W, 3)")
    if valid_mask is None:
        valid = np.isfinite(organized).all(axis=2)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != organized.shape[:2]:
            raise ValueError("valid_mask must match organized_points[:2]")
        valid &= np.isfinite(organized).all(axis=2)
    if organized_to_point_index is None:
        mapping = np.full(valid.shape, -1, dtype=np.int64)
        valid_count = int(np.count_nonzero(valid))
        if valid_count != len(scene):
            raise ValueError(
                "organized_to_point_index is required when points is not the "
                "valid row-major flattening of organized_points"
            )
        mapping[valid] = np.arange(len(scene), dtype=np.int64)
    else:
        mapping = np.asarray(organized_to_point_index, dtype=np.int64)
        if mapping.shape != valid.shape:
            raise ValueError("organized_to_point_index must have shape (H, W)")
        if np.any(mapping[valid] < 0) or np.any(mapping[valid] >= len(scene)):
            raise ValueError("organized_to_point_index has invalid scene indices")
    roi = None
    if roi_indices is not None:
        roi = np.unique(np.asarray(roi_indices, dtype=np.int64).reshape(-1))
        if np.any(roi < 0) or np.any(roi >= len(scene)):
            raise ValueError("roi_indices contains an invalid scene index")
    return scene, organized, valid, mapping, roi


def _assign_scene_indices(
    raw_patches: list[_RawPatch],
    mapping: np.ndarray,
    scene: np.ndarray,
    roi: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map raw-pixel memberships into one non-overlapping scene-index label."""

    allowed = np.ones(len(scene), dtype=bool)
    if roi is not None:
        allowed[:] = False
        allowed[roi] = True
    votes: list[dict[int, int]] = [dict() for _ in range(len(scene))]
    inlier_votes: list[dict[int, int]] = [dict() for _ in range(len(scene))]
    flat_mapping = mapping.reshape(-1)
    for plane_id, patch in enumerate(raw_patches):
        for scene_id in flat_mapping[patch.region_pixels]:
            if allowed[scene_id]:
                votes[scene_id][plane_id] = votes[scene_id].get(plane_id, 0) + 1
        for scene_id in flat_mapping[patch.inlier_pixels]:
            if allowed[scene_id]:
                inlier_votes[scene_id][plane_id] = (
                    inlier_votes[scene_id].get(plane_id, 0) + 1
                )
    assigned = [[] for _ in raw_patches]
    assigned_inliers = [[] for _ in raw_patches]
    for scene_id, plane_votes in enumerate(votes):
        if not plane_votes:
            continue
        candidates = sorted(
            plane_votes,
            key=lambda plane_id: (
                -plane_votes[plane_id],
                abs(
                    float(
                        raw_patches[plane_id].fit.normal @ scene[scene_id]
                        + raw_patches[plane_id].fit.offset
                    )
                ),
                plane_id,
            ),
        )
        plane_id = candidates[0]
        assigned[plane_id].append(scene_id)
        if inlier_votes[scene_id].get(plane_id, 0) > 0:
            assigned_inliers[plane_id].append(scene_id)
    return [
        (
            np.asarray(indices, dtype=np.int64),
            np.asarray(assigned_inliers[plane_id], dtype=np.int64),
        )
        for plane_id, indices in enumerate(assigned)
    ]


def extract_planes(
    points: np.ndarray,
    organized_points: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    roi_indices: np.ndarray | None = None,
    settings: CornerVINSPlaneConfig | Any | None = None,
    *,
    organized_to_point_index: np.ndarray | None = None,
) -> PlaneExtractionResult:
    """Extract CornerVINS plane patches while preserving scene indices.

    AHC runs only on organized pixels selected by ``valid_mask``.  The caller
    can therefore provide an exact query-local mask without losing the source
    image's pixel adjacency.  ``roi_indices`` is applied when final raw-pixel
    memberships are adapted back into the caller's scene-index space.
    """

    config = (
        CornerVINSPlaneConfig()
        if settings is None
        else settings
        if isinstance(settings, CornerVINSPlaneConfig)
        else CornerVINSPlaneConfig.from_settings(settings)
    )
    config.validate()
    scene, organized, valid, mapping, roi = _validate_inputs(
        points,
        organized_points,
        valid_mask,
        organized_to_point_index,
        roi_indices,
    )
    started = time.perf_counter()
    graph_started = time.perf_counter()
    graph, block_to_region, rejected = _build_initial_graph(
        organized, valid, config
    )
    initial_region_count = len(graph)
    graph_elapsed = time.perf_counter() - graph_started

    cluster_started = time.perf_counter()
    coarse = _cluster_graph(
        graph,
        config,
        minimum_points=config.min_region_points,
    )
    cluster_elapsed = time.perf_counter() - cluster_started
    ahc_membership = _membership_image(coarse, *valid.shape)

    refine_started = time.perf_counter()
    if config.ahc_refinement:
        regions = _refine_regions(
            coarse,
            block_to_region,
            organized,
            valid,
            config,
        )
    else:
        regions = coarse
    refinement_elapsed = time.perf_counter() - refine_started
    final_membership = _membership_image(regions, *valid.shape)

    flat_points = organized.reshape(-1, 3)
    ransac_started = time.perf_counter()
    raw_patches: list[_RawPatch] = []
    ransac_rejected = 0
    for region_id, region in enumerate(regions):
        fitted = _ransac_plane(
            flat_points[region.pixel_indices], config, seed_offset=region_id
        )
        if fitted is None:
            ransac_rejected += 1
            continue
        fit, inlier_mask, rmse = fitted
        raw_patches.append(
            _RawPatch(
                region_pixels=region.pixel_indices.copy(),
                inlier_pixels=region.pixel_indices[inlier_mask],
                fit=fit,
                residual_rmse=rmse,
                inlier_ratio=float(np.count_nonzero(inlier_mask) / len(inlier_mask)),
            )
        )
    ransac_patches = tuple(
        PlaneStagePatch(
            plane_id=plane_id,
            normal=raw.fit.normal.copy(),
            offset=float(raw.fit.offset),
            center=raw.fit.center.copy(),
            organized_point_indices=raw.region_pixels.copy(),
            organized_inlier_indices=raw.inlier_pixels.copy(),
            residual_rmse=float(raw.residual_rmse),
            inlier_ratio=float(raw.inlier_ratio),
        )
        for plane_id, raw in enumerate(raw_patches)
    )
    ransac_elapsed = time.perf_counter() - ransac_started

    merge_started = time.perf_counter()
    raw_patches, merge_count = _merge_raw_patches(
        raw_patches, flat_points, config
    )
    merge_elapsed = time.perf_counter() - merge_started

    hull_started = time.perf_counter()
    scene_membership = _assign_scene_indices(raw_patches, mapping, scene, roi)
    patches: list[PlanePatch] = []
    dropped_after_roi = 0
    for raw_id, (raw, (point_indices, inlier_indices)) in enumerate(
        zip(raw_patches, scene_membership)
    ):
        if len(point_indices) < 3 or len(inlier_indices) < 3:
            dropped_after_roi += 1
            continue
        inlier_points = flat_points[raw.inlier_pixels]
        hull = _convex_hull_3d(inlier_points, raw.fit)
        patches.append(
            PlanePatch(
                plane_id=len(patches),
                normal=raw.fit.normal.copy(),
                offset=float(raw.fit.offset),
                center=raw.fit.center.copy(),
                point_indices=point_indices,
                inlier_indices=inlier_indices,
                hull_vertices_3d=hull,
                residual_rmse=float(raw.residual_rmse),
                inlier_ratio=float(raw.inlier_ratio),
                organized_point_indices=raw.region_pixels.copy(),
                organized_inlier_indices=raw.inlier_pixels.copy(),
            )
        )
    hull_elapsed = time.perf_counter() - hull_started
    elapsed = time.perf_counter() - started
    diagnostics = {
        "method": "cornervins_ahc_ransac_merge_hull",
        "organized_shape": tuple(int(value) for value in valid.shape),
        "valid_organized_point_count": int(np.count_nonzero(valid)),
        "scene_point_count": int(len(scene)),
        "roi_point_count": None if roi is None else int(len(roi)),
        "region_mode": config.region_mode,
        "min_region_points": int(config.min_region_points),
        "paper_region_size": config.region_mode == "paper",
        "ahc_block_size": int(config.ahc_block_size),
        "ahc_threshold_mode": (
            "feng_depth_noise_model"
            if config.ahc_mse_threshold_mm2 is None
            else "fixed_mm2"
        ),
        "initial_region_count": int(initial_region_count),
        "initial_rejected": dict(rejected),
        "coarse_region_count": int(len(coarse)),
        "refined_region_count": int(len(regions)),
        "ransac_rejected_region_count": int(ransac_rejected),
        "premerge_patch_count": int(len(raw_patches) + merge_count),
        "merge_count": int(merge_count),
        "dropped_after_roi_count": int(dropped_after_roi),
        "plane_count": int(len(patches)),
        "timings_seconds": {
            "initial_graph": float(graph_elapsed),
            "ahc": float(cluster_elapsed),
            "refinement": float(refinement_elapsed),
            "ransac": float(ransac_elapsed),
            "merging": float(merge_elapsed),
            "hulls_and_adapter": float(hull_elapsed),
            "total": float(elapsed),
        },
    }
    return PlaneExtractionResult(
        patches=tuple(patches),
        ahc_regions=tuple(region.pixel_indices.copy() for region in coarse),
        ahc_membership_image=ahc_membership,
        final_membership_image=final_membership,
        diagnostics=diagnostics,
        ransac_patches=ransac_patches,
    )


def as_plane_cluster_analysis(
    result: PlaneExtractionResult,
    points_mm: np.ndarray,
    query_index: int,
    query_radius_mm: float,
    roi_indices: np.ndarray,
    projection_halo_mm: float = 0.0,
    projection_inlier_mm: float | None = None,
) -> dict[str, Any]:
    """Adapt fitted query-core planes and residual-gated halo observations."""

    points = np.asarray(points_mm, dtype=np.float64)
    if not np.isfinite(projection_halo_mm) or projection_halo_mm < 0.0:
        raise ValueError("projection_halo_mm must be finite and nonnegative")
    roi = np.unique(np.asarray(roi_indices, dtype=np.int64).reshape(-1))
    query = points[int(query_index)]
    padded = np.flatnonzero(
        np.linalg.norm(points - query, axis=1)
        <= float(query_radius_mm) + float(projection_halo_mm) + 1e-9
    ).astype(np.int64)
    global_planes = []
    for patch in result.patches:
        support = np.intersect1d(patch.inlier_indices, roi, assume_unique=True)
        if len(support) < 3:
            continue
        if projection_inlier_mm is None:
            projection = np.intersect1d(
                patch.inlier_indices, padded, assume_unique=True
            )
        else:
            if not np.isfinite(projection_inlier_mm) or projection_inlier_mm <= 0:
                raise ValueError("projection_inlier_mm must be finite and positive")
            residual = np.abs(points[padded] @ patch.normal + patch.offset)
            compatible = padded[residual <= float(projection_inlier_mm)]
            # Query-core plane membership is immutable; the residual gate only
            # adds observed context in the projection halo.
            projection = np.union1d(support, compatible).astype(np.int64)
        global_planes.append(
            {
                "plane_id": len(global_planes),
                "member_reference_indices": np.empty(0, dtype=np.int64),
                "member_reference_ids": np.empty(0, dtype=np.int64),
                "member_plane_hypothesis_ids": np.empty(0, dtype=np.int64),
                "support_indices": support,
                "projection_indices": projection,
                "line_cluster_ids": [],
                "representative_center": patch.center.copy(),
                "representative_normal": patch.normal.copy(),
                "representative_offset": float(patch.offset),
                "hull_vertices_3d": patch.hull_vertices_3d.copy(),
                "residual_rmse": float(patch.residual_rmse),
                "inlier_ratio": float(patch.inlier_ratio),
            }
        )
    timings = result.diagnostics["timings_seconds"]
    projection_support = (
        np.unique(
            np.concatenate(
                [plane["projection_indices"] for plane in global_planes]
            )
        )
        if global_planes
        else np.empty(0, dtype=np.int64)
    )
    return {
        "method": "cornervins_query_plane_extraction",
        "query_index": int(query_index),
        "query_position_mm": points[int(query_index)].copy(),
        "query_radius_mm": float(query_radius_mm),
        "search_scope": "query",
        "roi_geometry": "3d_ball",
        "roi_point_indices": roi,
        "projection_support_indices": projection_support,
        "projection_halo_mm": float(projection_halo_mm),
        "projection_inlier_mm": (
            None if projection_inlier_mm is None else float(projection_inlier_mm)
        ),
        "global_planes": global_planes,
        "cornervins_result": result,
        "diagnostics": {
            **dict(result.diagnostics),
            "timings_seconds": {
                **dict(timings),
                "total": float(timings["total"]),
            },
        },
    }


__all__ = [
    "CornerVINSPlaneConfig",
    "PlaneExtractionResult",
    "PlanePatch",
    "as_plane_cluster_analysis",
    "extract_planes",
]
