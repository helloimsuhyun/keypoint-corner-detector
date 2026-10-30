"""Real-depth repeatability experiments for the normal-free Normal Cone detector.

This script evaluates three hypotheses that are not covered by the raw cone
solid angle alone:

1. a physical corner should remain a local maximum as the metric support grows;
2. a useful keypoint should survive voxel-grid phase, depth noise, and dropout;
3. detections from different RGB-D frames should return to the same world point.

RGB is loaded only for visualization.  Detection and verification use XYZ/depth
coordinates plus the dataset poses for evaluation.  The default input is the
locally available Microsoft 7-Scenes RedKitchen ``seq-03`` archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from normal_cone_corner_detector import (
    MultiScaleConfig,
    NormalConeConfig,
    ScaleSpaceStabilityConfig,
    build_scene,
    detect_scene_multiscale,
    evaluate_scale_space_stability,
)


DEFAULT_SEQUENCE = Path("/home/choisuhyun/datasets/7scenes/redkitchen/seq-03.zip")
DEFAULT_OUTPUT = Path("outputs/depth_repeatability_redkitchen")
FX = 585.0
FY = 585.0
CX = 320.0
CY = 240.0


@dataclass
class FrameData:
    frame: int
    rgb: np.ndarray
    depth_mm: np.ndarray
    pose_camera_to_world: np.ndarray


@dataclass
class DetectionBundle:
    frame: int
    variant: str
    points_mm: np.ndarray
    scores: np.ndarray
    candidate_indices: np.ndarray
    scale_stable_indices: np.ndarray
    scale_track_count: np.ndarray
    scale_median_drift_mm: np.ndarray
    scale_p90_drift_mm: np.ndarray
    elapsed_seconds: float


@dataclass
class SensorStability:
    hit_count: np.ndarray
    median_drift_mm: np.ndarray
    accepted_indices: np.ndarray


@dataclass
class RangeContrast:
    spread_mm: np.ndarray
    saliency: np.ndarray


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def read_frame(sequence: Path, frame: int) -> FrameData:
    prefix = f"seq-03/frame-{frame:06d}"
    with zipfile.ZipFile(sequence) as archive:
        rgb = np.asarray(
            Image.open(io.BytesIO(archive.read(f"{prefix}.color.png"))).convert("RGB")
        )
        depth = np.asarray(Image.open(io.BytesIO(archive.read(f"{prefix}.depth.png"))))
        pose = np.loadtxt(
            io.StringIO(archive.read(f"{prefix}.pose.txt").decode("utf-8"))
        )
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose for frame {frame}")
    pose = pose.copy()
    pose[:3, :3] = _project_rotation(pose[:3, :3])
    pose[3] = (0.0, 0.0, 0.0, 1.0)
    return FrameData(frame, rgb, depth, pose)


def depth_to_points_mm(
    depth_mm: np.ndarray,
    *,
    noise_sigma_mm: float = 0.0,
    dropout_fraction: float = 0.0,
    structured_dropout: bool = False,
    seed: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth_mm)
    rows, columns = np.indices(depth.shape)
    valid = (depth > 0) & (depth != 65535) & (depth < 5000)
    rng = np.random.default_rng(seed)
    if dropout_fraction > 0.0:
        valid &= rng.random(depth.shape) >= dropout_fraction
    if structured_dropout:
        # Missing scan rows mimic correlated holes rather than independent
        # point sampling alone.  The offset is seed-dependent but deterministic.
        row_period = 31
        valid &= (rows - seed % row_period) % row_period != 0
    z = depth.astype(np.float64)
    if noise_sigma_mm > 0.0:
        z = z + rng.normal(0.0, noise_sigma_mm, depth.shape)
        valid &= z > 0.0
    z_valid = z[valid]
    return np.column_stack(
        (
            (columns[valid] - CX) * z_valid / FX,
            (rows[valid] - CY) * z_valid / FY,
            z_valid,
        )
    )


def voxel_downsample_with_origin(
    points_mm: np.ndarray, voxel_size_mm: float, origin_mm: Sequence[float]
) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    keys = np.floor((points - origin) / voxel_size_mm).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centroids = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    return centroids


def virtual_view_subset(
    points_mm: np.ndarray,
    *,
    yaw_degrees: float,
    camera_shift_x_mm: float,
    image_shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Z-buffer a cloud from a nearby virtual camera, returned in base coordinates.

    A pure SE(3) transform is a trivial test for this detector because spherical
    neighborhoods and dot products are rigid-invariant.  Reprojection followed
    by a z-buffer is deliberately non-trivial: it changes sampling, field of
    view, and which foreground points survive near occlusion boundaries.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    angle = np.radians(yaw_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    camera_center = np.array([camera_shift_x_mm, 0.0, 0.0], dtype=np.float64)
    virtual = (points - camera_center) @ rotation
    z = virtual[:, 2]
    u = np.rint(FX * virtual[:, 0] / np.maximum(z, 1e-9) + CX).astype(np.int64)
    v = np.rint(FY * virtual[:, 1] / np.maximum(z, 1e-9) + CY).astype(np.int64)
    height, width = image_shape
    inside = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    source_indices = np.flatnonzero(inside)
    if len(source_indices) == 0:
        return np.empty((0, 3), dtype=np.float64)
    pixel_id = v[inside] * width + u[inside]
    # Sorting by pixel first and range second makes the first entry per pixel
    # the z-buffer winner.  Keep original/base coordinates for direct matching.
    order = np.lexsort((z[inside], pixel_id))
    sorted_pixel = pixel_id[order]
    first = np.r_[True, sorted_pixel[1:] != sorted_pixel[:-1]]
    kept = source_indices[order[first]]
    return points[kept]


def _variant_points(frame: FrameData, variant: str, voxel_size_mm: float) -> tuple[np.ndarray, np.ndarray]:
    if variant == "base":
        raw = depth_to_points_mm(frame.depth_mm)
        origin = np.zeros(3, dtype=np.float64)
    elif variant == "voxel_phase":
        raw = depth_to_points_mm(frame.depth_mm)
        origin = np.full(3, 0.5 * voxel_size_mm, dtype=np.float64)
    elif variant == "noise_dropout":
        raw = depth_to_points_mm(
            frame.depth_mm,
            noise_sigma_mm=2.5,
            dropout_fraction=0.10,
            structured_dropout=True,
            seed=10_000 + frame.frame,
        )
        origin = np.zeros(3, dtype=np.float64)
    elif variant == "view_left":
        raw = virtual_view_subset(
            depth_to_points_mm(frame.depth_mm),
            yaw_degrees=3.0,
            camera_shift_x_mm=35.0,
            image_shape=frame.depth_mm.shape,
        )
        origin = np.zeros(3, dtype=np.float64)
    elif variant == "view_right":
        raw = virtual_view_subset(
            depth_to_points_mm(frame.depth_mm),
            yaw_degrees=-3.0,
            camera_shift_x_mm=-35.0,
            image_shape=frame.depth_mm.shape,
        )
        origin = np.zeros(3, dtype=np.float64)
    elif variant == "view_left_strong":
        raw = virtual_view_subset(
            depth_to_points_mm(frame.depth_mm),
            yaw_degrees=5.0,
            camera_shift_x_mm=60.0,
            image_shape=frame.depth_mm.shape,
        )
        origin = np.zeros(3, dtype=np.float64)
    elif variant == "view_right_strong":
        raw = virtual_view_subset(
            depth_to_points_mm(frame.depth_mm),
            yaw_degrees=-5.0,
            camera_shift_x_mm=-60.0,
            image_shape=frame.depth_mm.shape,
        )
        origin = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError(f"Unknown perturbation variant: {variant}")
    return raw, origin


def _cache_path(
    output: Path,
    frame: int,
    variant: str,
    config: NormalConeConfig,
    multiscale: MultiScaleConfig,
    scale_stability: ScaleSpaceStabilityConfig,
) -> Path:
    payload = {
        "format": 2,
        "frame": frame,
        "variant": variant,
        "normal_cone": asdict(config),
        "multiscale": asdict(multiscale),
        "scale_stability": asdict(scale_stability),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return output / "cache" / f"frame-{frame:06d}-{variant}-{digest}.npz"


def detect_frame_variant(
    frame: FrameData,
    variant: str,
    config: NormalConeConfig,
    multiscale: MultiScaleConfig,
    scale_stability: ScaleSpaceStabilityConfig,
    output: Path,
    *,
    force: bool,
) -> DetectionBundle:
    cache = _cache_path(
        output,
        frame.frame,
        variant,
        config,
        multiscale,
        scale_stability,
    )
    if cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as saved:
            bundle = DetectionBundle(
                frame=frame.frame,
                variant=variant,
                points_mm=np.asarray(saved["points_mm"], dtype=np.float64),
                scores=np.asarray(saved["scores"], dtype=np.float64),
                candidate_indices=np.asarray(saved["candidate_indices"], dtype=np.int64),
                scale_stable_indices=np.asarray(
                    saved["scale_stable_indices"], dtype=np.int64
                ),
                scale_track_count=np.asarray(saved["scale_track_count"], dtype=np.int16),
                scale_median_drift_mm=np.asarray(
                    saved["scale_median_drift_mm"], dtype=np.float64
                ),
                scale_p90_drift_mm=np.asarray(
                    saved["scale_p90_drift_mm"], dtype=np.float64
                ),
                elapsed_seconds=float(saved["elapsed_seconds"]),
            )
        print(
            f"[CACHE] frame {frame.frame:06d} {variant}: "
            f"{len(bundle.candidate_indices)} candidates"
        )
        return bundle

    raw, origin = _variant_points(frame, variant, config.voxel_size_mm)
    points = voxel_downsample_with_origin(raw, config.voxel_size_mm, origin)
    scene = build_scene(points, config, downsample=False)
    print(
        f"[DETECT] frame {frame.frame:06d} {variant}: "
        f"raw={len(raw):,}, voxel={len(points):,}"
    )
    started = time.perf_counter()
    detection = detect_scene_multiscale(scene, config, multiscale)
    stability = evaluate_scale_space_stability(
        scene, config, detection, scale_stability
    )
    elapsed = time.perf_counter() - started
    bundle = DetectionBundle(
        frame=frame.frame,
        variant=variant,
        points_mm=scene.points_mm,
        scores=detection.aggregate_scores,
        candidate_indices=detection.candidate_indices,
        scale_stable_indices=stability.accepted_indices,
        scale_track_count=stability.track_count,
        scale_median_drift_mm=stability.median_drift_mm,
        scale_p90_drift_mm=stability.p90_drift_mm,
        elapsed_seconds=elapsed,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        points_mm=bundle.points_mm,
        scores=bundle.scores,
        candidate_indices=bundle.candidate_indices,
        scale_stable_indices=bundle.scale_stable_indices,
        scale_track_count=bundle.scale_track_count,
        scale_median_drift_mm=bundle.scale_median_drift_mm,
        scale_p90_drift_mm=bundle.scale_p90_drift_mm,
        elapsed_seconds=np.asarray(bundle.elapsed_seconds),
    )
    print(
        f"[DONE] frame {frame.frame:06d} {variant}: "
        f"score={len(bundle.candidate_indices)}, "
        f"scale-track={len(bundle.scale_stable_indices)}, {elapsed:.1f}s"
    )
    return bundle


def _mutual_nearest_distances(
    source: np.ndarray, target: np.ndarray, maximum_distance_mm: float
) -> np.ndarray:
    result = np.full(len(source), np.inf, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        return result
    target_tree = cKDTree(target)
    source_to_target, target_id = target_tree.query(source, k=1)
    source_tree = cKDTree(source)
    _, source_id = source_tree.query(target, k=1)
    source_indices = np.arange(len(source), dtype=np.int64)
    mutual = source_id[np.asarray(target_id, dtype=np.int64)] == source_indices
    accepted = mutual & (source_to_target <= maximum_distance_mm)
    result[accepted] = source_to_target[accepted]
    return result


def sensor_stability(
    base: DetectionBundle,
    perturbed: Sequence[DetectionBundle],
    *,
    match_radius_mm: float = 35.0,
    max_median_drift_mm: float = 25.0,
) -> SensorStability:
    base_indices = np.asarray(base.candidate_indices, dtype=np.int64)
    base_points = base.points_mm[base_indices]
    all_distances = []
    for variant in perturbed:
        target = variant.points_mm[variant.candidate_indices]
        all_distances.append(
            _mutual_nearest_distances(base_points, target, match_radius_mm)
        )
    distances = np.asarray(all_distances, dtype=np.float64)
    hit_count_small = np.count_nonzero(np.isfinite(distances), axis=0).astype(np.int16)
    median_small = np.full(len(base_indices), np.inf, dtype=np.float64)
    for position in range(len(base_indices)):
        values = distances[np.isfinite(distances[:, position]), position]
        if len(values):
            median_small[position] = float(np.median(values))
    passed = (hit_count_small == len(perturbed)) & (
        median_small <= max_median_drift_mm
    )
    hit_count = np.zeros(len(base.points_mm), dtype=np.int16)
    median = np.full(len(base.points_mm), np.inf, dtype=np.float64)
    hit_count[base_indices] = hit_count_small
    median[base_indices] = median_small
    return SensorStability(hit_count, median, base_indices[passed])


def camera_to_world_mm(points_mm: np.ndarray, pose_camera_to_world: np.ndarray) -> np.ndarray:
    points_m = np.asarray(points_mm, dtype=np.float64) / 1000.0
    world_m = points_m @ pose_camera_to_world[:3, :3].T + pose_camera_to_world[:3, 3]
    return world_m * 1000.0


def world_to_camera_mm(points_mm: np.ndarray, pose_camera_to_world: np.ndarray) -> np.ndarray:
    points_m = np.asarray(points_mm, dtype=np.float64) / 1000.0
    rotation = pose_camera_to_world[:3, :3]
    camera_m = (points_m - pose_camera_to_world[:3, 3]) @ rotation
    return camera_m * 1000.0


def project_camera(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)
    z = np.maximum(points[:, 2], 1e-9)
    return np.column_stack((FX * points[:, 0] / z + CX, FY * points[:, 1] / z + CY))


def visible_in_frame(
    world_points_mm: np.ndarray,
    target: FrameData,
    *,
    depth_tolerance_mm: float = 80.0,
    pixel_window: int = 2,
) -> np.ndarray:
    camera = world_to_camera_mm(world_points_mm, target.pose_camera_to_world)
    uv = project_camera(camera)
    height, width = target.depth_mm.shape
    visible = np.zeros(len(camera), dtype=bool)
    for index, (point, pixel) in enumerate(zip(camera, uv)):
        if point[2] <= 0.0:
            continue
        column = int(round(pixel[0]))
        row = int(round(pixel[1]))
        if not (0 <= column < width and 0 <= row < height):
            continue
        row0 = max(0, row - pixel_window)
        row1 = min(height, row + pixel_window + 1)
        column0 = max(0, column - pixel_window)
        column1 = min(width, column + pixel_window + 1)
        patch = target.depth_mm[row0:row1, column0:column1]
        valid = patch[(patch > 0) & (patch != 65535) & (patch < 5000)]
        if len(valid) and np.min(np.abs(valid.astype(float) - point[2])) <= depth_tolerance_mm:
            visible[index] = True
    return visible


def _rank_indices(bundle: DetectionBundle, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return indices
    # Primary detector saliency remains the Normal Cone score.  Deterministic
    # point index tie-breaking makes cached and fresh runs identical.
    order = np.lexsort((indices, -bundle.scores[indices]))
    return indices[order]


def range_contrast(
    frame: FrameData,
    bundle: DetectionBundle,
    *,
    pixel_radius: int = 4,
    physical_scale_mm: float = 15.0,
) -> RangeContrast:
    """Measure robust local range change without estimating surface normals.

    The p90-p10 range spread follows the NARF motivation that repeatable range
    keypoints need both a stable center and substantial nearby surface/border
    change.  It is used only to rank already detected Normal Cone candidates.
    """
    if pixel_radius < 1:
        raise ValueError("pixel_radius must be positive")
    if not np.isfinite(physical_scale_mm) or physical_scale_mm <= 0.0:
        raise ValueError("physical_scale_mm must be finite and positive")
    spread = np.zeros(len(bundle.points_mm), dtype=np.float64)
    saliency = np.zeros(len(bundle.points_mm), dtype=np.float64)
    indices = np.asarray(bundle.candidate_indices, dtype=np.int64)
    uv = project_camera(bundle.points_mm[indices])
    depth = frame.depth_mm
    height, width = depth.shape
    for index, pixel in zip(indices, uv):
        column, row = (int(round(pixel[0])), int(round(pixel[1])))
        row0, row1 = max(0, row - pixel_radius), min(height, row + pixel_radius + 1)
        col0, col1 = max(0, column - pixel_radius), min(width, column + pixel_radius + 1)
        patch = depth[row0:row1, col0:col1]
        valid = patch[(patch > 0) & (patch != 65535) & (patch < 5000)].astype(float)
        if len(valid):
            spread[index] = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    saliency[indices] = bundle.scores[indices] * np.log1p(
        spread[indices] / physical_scale_mm
    )
    return RangeContrast(spread, saliency)


def method_indices(
    base: DetectionBundle,
    sensor: SensorStability,
    viewpoint: SensorStability,
    contrast: RangeContrast,
) -> dict[str, np.ndarray]:
    baseline = np.asarray(base.candidate_indices, dtype=np.int64)
    scale = np.intersect1d(baseline, base.scale_stable_indices, assume_unique=False)
    perturb = np.intersect1d(baseline, sensor.accepted_indices, assume_unique=False)
    view = np.intersect1d(baseline, viewpoint.accepted_indices, assume_unique=False)
    joint = np.intersect1d(
        np.intersect1d(scale, perturb, assume_unique=False),
        view,
        assume_unique=False,
    )
    contrast_order = np.lexsort((view, -contrast.saliency[view]))
    return {
        "score_persistence": _rank_indices(base, baseline),
        "scale_max_track": _rank_indices(base, scale),
        "sensor_perturbation": _rank_indices(base, perturb),
        "viewpoint_resampling": _rank_indices(base, view),
        "view_contrast": view[contrast_order],
        "joint_stability": _rank_indices(base, joint),
    }


def pair_repeatability(
    source_frame: FrameData,
    target_frame: FrameData,
    source_bundle: DetectionBundle,
    target_bundle: DetectionBundle,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    match_radius_mm: float,
) -> dict[str, float | int]:
    source_world = camera_to_world_mm(
        source_bundle.points_mm[source_indices], source_frame.pose_camera_to_world
    )
    target_world = camera_to_world_mm(
        target_bundle.points_mm[target_indices], target_frame.pose_camera_to_world
    )
    source_visible = visible_in_frame(source_world, target_frame)
    target_visible = visible_in_frame(target_world, source_frame)
    source_distances = _mutual_nearest_distances(
        source_world, target_world, match_radius_mm
    )
    target_distances = _mutual_nearest_distances(
        target_world, source_world, match_radius_mm
    )
    source_match = source_visible & np.isfinite(source_distances)
    target_match = target_visible & np.isfinite(target_distances)
    denominator = int(np.count_nonzero(source_visible) + np.count_nonzero(target_visible))
    numerator = int(np.count_nonzero(source_match) + np.count_nonzero(target_match))
    errors = np.concatenate((source_distances[source_match], target_distances[target_match]))
    return {
        "source_count": int(len(source_indices)),
        "target_count": int(len(target_indices)),
        "visible_denominator": denominator,
        "repeatable_numerator": numerator,
        "repeatability": float(numerator / denominator) if denominator else 0.0,
        "median_localization_error_mm": float(np.median(errors)) if len(errors) else math.nan,
        "mean_localization_error_mm": float(np.mean(errors)) if len(errors) else math.nan,
    }


def _aggregate_pair_metrics(records: Sequence[dict[str, object]]) -> dict[str, float | int]:
    denominator = sum(int(record["visible_denominator"]) for record in records)
    numerator = sum(int(record["repeatable_numerator"]) for record in records)
    finite_medians = [
        float(record["median_localization_error_mm"])
        for record in records
        if np.isfinite(float(record["median_localization_error_mm"]))
    ]
    return {
        "visible_denominator": denominator,
        "repeatable_numerator": numerator,
        "repeatability": float(numerator / denominator) if denominator else 0.0,
        "median_pair_localization_error_mm": (
            float(np.median(finite_medians)) if finite_medians else math.nan
        ),
    }


def temporal_support_for_reference(
    reference_frame: FrameData,
    reference_bundle: DetectionBundle,
    reference_indices: np.ndarray,
    other_frames: Sequence[FrameData],
    other_bundles: Sequence[DetectionBundle],
    other_indices: Sequence[np.ndarray],
    *,
    match_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference_world = camera_to_world_mm(
        reference_bundle.points_mm[reference_indices],
        reference_frame.pose_camera_to_world,
    )
    visible_count = np.zeros(len(reference_indices), dtype=np.int16)
    hit_count = np.zeros(len(reference_indices), dtype=np.int16)
    for frame, bundle, indices in zip(other_frames, other_bundles, other_indices):
        visible = visible_in_frame(reference_world, frame)
        visible_count += visible.astype(np.int16)
        target_world = camera_to_world_mm(
            bundle.points_mm[indices], frame.pose_camera_to_world
        )
        if len(target_world):
            distance, _ = cKDTree(target_world).query(reference_world, k=1)
            hit_count += (visible & (distance <= match_radius_mm)).astype(np.int16)
    return visible_count, hit_count


def render_results(
    output: Path,
    frames: dict[int, FrameData],
    base: dict[int, DetectionBundle],
    methods: dict[int, dict[str, np.ndarray]],
    aggregate_metrics: dict[str, dict[str, object]],
    reference_frame: int,
    reference_visible_count: np.ndarray,
    reference_hit_count: np.ndarray,
) -> None:
    method_labels = {
        "score_persistence": "Score persistence",
        "scale_max_track": "Scale-max track",
        "sensor_perturbation": "Sensor perturbation",
        "viewpoint_resampling": "Virtual-view resampling",
        "view_contrast": "Virtual view + range contrast",
        "joint_stability": "Joint stability",
    }
    colors = {
        "score_persistence": "#00bcd4",
        "scale_max_track": "#ff9800",
        "sensor_perturbation": "#8bc34a",
        "viewpoint_resampling": "#7e57c2",
        "view_contrast": "#d81b60",
        "joint_stability": "#e91e63",
    }
    reference = frames[reference_frame]
    bundle = base[reference_frame]
    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.15))
    overlay_names = (
        "score_persistence",
        "scale_max_track",
        "sensor_perturbation",
        "viewpoint_resampling",
        "view_contrast",
    )
    overlay_cells = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    for name, cell in zip(overlay_names, overlay_cells):
        axis = fig.add_subplot(grid[cell])
        axis.imshow(reference.rgb)
        indices = methods[reference_frame][name]
        uv = project_camera(bundle.points_mm[indices])
        axis.scatter(
            uv[:, 0],
            uv[:, 1],
            s=30,
            facecolors="none",
            edgecolors=colors[name],
            linewidths=1.2,
        )
        axis.set_title(f"{method_labels[name]} — {len(indices)}")
        axis.set_xlim(0, reference.rgb.shape[1])
        axis.set_ylim(reference.rgb.shape[0], 0)
        axis.set_axis_off()

    bar_axis = fig.add_subplot(grid[1, 2])
    names = list(method_labels)
    values = [float(aggregate_metrics[name]["repeatability"]) for name in names]
    bar_axis.bar(
        np.arange(len(names)), values, color=[colors[name] for name in names]
    )
    bar_axis.set_xticks(np.arange(len(names)))
    bar_axis.set_xticklabels(
        [method_labels[name].replace(" ", "\n") for name in names], fontsize=9
    )
    bar_axis.set_ylim(0.0, 1.0)
    bar_axis.set_ylabel("Pose-aligned repeatability")
    bar_axis.set_title("Fixed-budget, visibility-aware")
    for position, value in enumerate(values):
        bar_axis.text(position, value + 0.025, f"{value:.2f}", ha="center")
    bar_axis.grid(axis="y", alpha=0.25)

    world_axis = fig.add_subplot(grid[2, :], projection="3d")
    cloud = bundle.points_mm[:: max(1, len(bundle.points_mm) // 8000)]
    world_cloud = camera_to_world_mm(cloud, reference.pose_camera_to_world)
    world_axis.scatter(
        world_cloud[:, 0], world_cloud[:, 1], world_cloud[:, 2],
        s=0.35, c="#aeb6bf", alpha=0.18,
    )
    final_indices = methods[reference_frame]["view_contrast"]
    world_joint = camera_to_world_mm(
        bundle.points_mm[final_indices], reference.pose_camera_to_world
    )
    support = 1 + reference_hit_count
    scatter = world_axis.scatter(
        world_joint[:, 0], world_joint[:, 1], world_joint[:, 2],
        c=support, cmap="viridis", vmin=1, vmax=3, s=35, depthshade=False,
    )
    world_axis.set_title("Virtual-view + range-contrast candidates: temporal support")
    world_axis.set_xlabel("world X (mm)")
    world_axis.set_ylabel("world Y (mm)")
    world_axis.set_zlabel("world Z (mm)")
    colorbar = fig.colorbar(scatter, ax=world_axis, fraction=0.04, pad=0.08)
    colorbar.set_label("views containing a matched keypoint")
    fig.suptitle(
        "Normal Cone repeatability on real RGB-D (RGB is visualization only)",
        fontsize=16,
    )
    figure_path = output / "normal-cone-repeatability-comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        output / "repeatable-corners-3d.npz",
        frame=np.asarray(reference_frame),
        points_camera_mm=bundle.points_mm,
        points_world_mm=camera_to_world_mm(
            bundle.points_mm, reference.pose_camera_to_world
        ),
        score_candidate_indices=methods[reference_frame]["score_persistence"],
        scale_candidate_indices=methods[reference_frame]["scale_max_track"],
        perturbation_candidate_indices=methods[reference_frame]["sensor_perturbation"],
        viewpoint_candidate_indices=methods[reference_frame]["viewpoint_resampling"],
        view_contrast_candidate_indices=final_indices,
        joint_candidate_indices=methods[reference_frame]["joint_stability"],
        final_candidate_indices=final_indices,
        final_visible_target_count=reference_visible_count,
        final_temporal_hit_count=reference_hit_count,
        final_view_support=support,
        scores=bundle.scores,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frames", type=int, nargs="+", default=[720, 740, 760])
    parser.add_argument("--reference-frame", type=int, default=740)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sphere-samples", type=int, default=128)
    parser.add_argument("--fixed-budget", type=int, default=32)
    parser.add_argument("--match-radius-mm", type=float, default=35.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_order = tuple(dict.fromkeys(args.frames))
    if args.reference_frame not in frames_order:
        raise ValueError("--reference-frame must be listed in --frames")
    if len(frames_order) < 2:
        raise ValueError("At least two frames are required")
    if not args.sequence.is_file():
        raise FileNotFoundError(args.sequence)
    args.output.mkdir(parents=True, exist_ok=True)

    config = NormalConeConfig(
        voxel_size_mm=15.0,
        query_radius_mm=150.0,
        sphere_samples=args.sphere_samples,
        tolerance_mm=4.0,
        min_neighbors=12,
        min_score=0.08,
        local_max_radius_mm=22.0,
        nms_radius_mm=45.0,
        direction_chunk_size=min(128, args.sphere_samples),
    )
    multiscale = MultiScaleConfig(
        radii_mm=(30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0),
        min_scale_count=5,
        min_consistency=0.35,
    )
    scale_stability = ScaleSpaceStabilityConfig(
        track_radius_mm=35.0,
        min_track_scale_count=4,
        max_median_drift_mm=20.0,
        max_p90_drift_mm=35.0,
    )

    frame_data = {frame: read_frame(args.sequence, frame) for frame in frames_order}
    detections: dict[int, dict[str, DetectionBundle]] = {}
    sensors: dict[int, SensorStability] = {}
    viewpoints: dict[int, SensorStability] = {}
    contrasts: dict[int, RangeContrast] = {}
    methods: dict[int, dict[str, np.ndarray]] = {}
    for frame in frames_order:
        variants: dict[str, DetectionBundle] = {}
        for variant in (
            "base",
            "voxel_phase",
            "noise_dropout",
            "view_left",
            "view_right",
        ):
            variants[variant] = detect_frame_variant(
                frame_data[frame],
                variant,
                config,
                multiscale,
                scale_stability,
                args.output,
                force=args.force,
            )
        detections[frame] = variants
        sensors[frame] = sensor_stability(
            variants["base"],
            [variants["voxel_phase"], variants["noise_dropout"]],
        )
        viewpoints[frame] = sensor_stability(
            variants["base"],
            [variants["view_left"], variants["view_right"]],
        )
        contrasts[frame] = range_contrast(
            frame_data[frame],
            variants["base"],
            physical_scale_mm=config.voxel_size_mm,
        )
        methods[frame] = method_indices(
            variants["base"], sensors[frame], viewpoints[frame], contrasts[frame]
        )
        print(
            f"[FILTER] frame {frame:06d}: "
            + ", ".join(f"{name}={len(ids)}" for name, ids in methods[frame].items())
        )

    method_names = tuple(next(iter(methods.values())).keys())
    smallest = min(
        len(methods[frame][method])
        for frame in frames_order
        for method in method_names
    )
    common_budget = min(args.fixed_budget, smallest)
    if common_budget < 1:
        raise RuntimeError("At least one method produced no candidates")
    print(f"[EVAL] common fixed keypoint budget={common_budget}")

    pairs = [
        (frames_order[left], frames_order[right])
        for left in range(len(frames_order))
        for right in range(left + 1, len(frames_order))
    ]
    def evaluate_budget(
        budget: int, *, verbose: bool
    ) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
        records: dict[str, list[dict[str, object]]] = {
            name: [] for name in method_names
        }
        for method in method_names:
            for source_frame, target_frame in pairs:
                source_indices = methods[source_frame][method][:budget]
                target_indices = methods[target_frame][method][:budget]
                record = pair_repeatability(
                    frame_data[source_frame],
                    frame_data[target_frame],
                    detections[source_frame]["base"],
                    detections[target_frame]["base"],
                    source_indices,
                    target_indices,
                    match_radius_mm=args.match_radius_mm,
                )
                record["source_frame"] = source_frame
                record["target_frame"] = target_frame
                records[method].append(record)
                if verbose:
                    print(
                        f"[PAIR] {method} {source_frame}->{target_frame}: "
                        f"repeatability={float(record['repeatability']):.3f}, "
                        f"visible={record['visible_denominator']}"
                    )
        aggregated = {
            method: _aggregate_pair_metrics(records[method])
            for method in method_names
        }
        return records, aggregated

    budgets = sorted({min(value, smallest) for value in (16, 32, 64, common_budget)})
    budget_sweep: dict[str, dict[str, object]] = {}
    pair_records: dict[str, list[dict[str, object]]] | None = None
    aggregate: dict[str, dict[str, object]] | None = None
    for budget in budgets:
        records, aggregated = evaluate_budget(budget, verbose=budget == common_budget)
        budget_sweep[str(budget)] = {
            "pair_results": records,
            "aggregate_results": aggregated,
        }
        if budget == common_budget:
            pair_records, aggregate = records, aggregated
    assert pair_records is not None and aggregate is not None
    for budget in budgets:
        summary = budget_sweep[str(budget)]["aggregate_results"]
        print(
            f"[BUDGET {budget}] "
            + ", ".join(
                f"{method}={float(summary[method]['repeatability']):.3f}"
                for method in method_names
            )
        )

    reference = args.reference_frame
    evaluated_methods = {
        frame: {
            method: indices[:common_budget]
            for method, indices in methods[frame].items()
        }
        for frame in frames_order
    }
    reference_final = evaluated_methods[reference]["view_contrast"]
    other_frames = [frame for frame in frames_order if frame != reference]
    visible_count, hit_count = temporal_support_for_reference(
        frame_data[reference],
        detections[reference]["base"],
        reference_final,
        [frame_data[frame] for frame in other_frames],
        [detections[frame]["base"] for frame in other_frames],
        [evaluated_methods[frame]["view_contrast"] for frame in other_frames],
        match_radius_mm=args.match_radius_mm,
    )
    repeatable_final = (visible_count > 0) & (hit_count == visible_count)

    payload = {
        "experiment": "Normal Cone real-depth repeatability",
        "sequence": str(args.sequence.resolve()),
        "frames": list(frames_order),
        "reference_frame": reference,
        "coordinate_input_only": True,
        "rgb_used_for_detection": False,
        "pose_used_for_detection": False,
        "pose_used_for_repeatability_evaluation": True,
        "config": asdict(config),
        "multiscale": asdict(multiscale),
        "scale_space_stability": asdict(scale_stability),
        "perturbations": {
            "voxel_phase_mm": [7.5, 7.5, 7.5],
            "depth_noise_sigma_mm": 2.5,
            "random_dropout_fraction": 0.10,
            "structured_dropout_period_rows": 31,
            "virtual_view_yaw_degrees": [-3.0, 3.0],
            "virtual_camera_shift_x_mm": [-35.0, 35.0],
        },
        "evaluation": {
            "fixed_budget_requested": args.fixed_budget,
            "common_fixed_budget": common_budget,
            "world_match_radius_mm": args.match_radius_mm,
            "visibility_depth_tolerance_mm": 80.0,
            "pairs": pairs,
            "reported_budget_sweep": budgets,
        },
        "candidate_counts": {
            str(frame): {method: len(ids) for method, ids in methods[frame].items()}
            for frame in frames_order
        },
        "pair_results": pair_records,
        "aggregate_results": aggregate,
        "budget_sweep": budget_sweep,
        "reference_final_temporal": {
            "method": "view_contrast",
            "candidate_count": int(len(reference_final)),
            "visible_in_at_least_one_other_frame": int(np.count_nonzero(visible_count > 0)),
            "repeatable_in_every_visible_frame": int(np.count_nonzero(repeatable_final)),
            "view_support_histogram": {
                str(support): int(np.count_nonzero(1 + hit_count == support))
                for support in range(1, len(frames_order) + 1)
            },
        },
        "limitations": [
            "Pose-aligned repeatability is a stability proxy, not physical-corner ground truth.",
            "Temporal consensus requires multiple depth frames and cannot resolve a single image alone.",
            "A persistently observed occlusion boundary may still pass all coordinate-only tests.",
        ],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    render_results(
        args.output,
        frame_data,
        {frame: detections[frame]["base"] for frame in frames_order},
        evaluated_methods,
        aggregate,
        reference,
        visible_count,
        hit_count,
    )
    print(f"[OUTPUT] {summary_path}")
    print(f"[OUTPUT] {args.output / 'normal-cone-repeatability-comparison.png'}")
    print(f"[OUTPUT] {args.output / 'repeatable-corners-3d.npz'}")


if __name__ == "__main__":
    main()
