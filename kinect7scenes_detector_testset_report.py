#!/home/choisuhyun/keypoint-corner-detector/.conda/kinect7scenes-minispinnet/bin/python
"""Stratified held-out evaluation of three 3-D keypoint detectors.

The evaluated sequence is Microsoft 7-Scenes RedKitchen seq-03.  The default
15 primary pairs span frames 10..930 and are stratified by GT dense-cloud
overlap (easy/moderate/hard, five pairs each).  Three near-zero-overlap pairs
are evaluated separately as negative controls.

Only the detector changes.  All methods use a common 150-keypoint spatial
budget, the same pretrained BUFFER-X Mini-SpinNet descriptor, strict mutual
top-1 matching, and the same deterministic correspondence RANSAC.  ICP and
temporal bridging are disabled.

Outputs include raw CSV/JSON, a six-panel figure, and a Korean Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import kinect7scenes_detector_benchmark as base
import kinect7scenes_minispinnet_registration as pipeline
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEQUENCE = Path.home() / "datasets/7scenes/redkitchen/seq-03.zip"
PRIMARY_PAIRS = base.parse_pairs(
    "10:50,210:250,410:450,730:770,770:810,"
    "50:170,210:290,410:650,730:810,770:930,"
    "50:250,290:450,450:690,650:730,690:930"
)
NEGATIVE_PAIRS = base.parse_pairs("10:410,210:690,410:730")
METHODS = base.METHODS
LABELS = base.LABELS
COLORS = base.COLORS
REPEATABILITY_THRESHOLDS_MM = (20, 30, 50, 75, 100)
REPORT_CACHE_VERSION = 2


@dataclass
class FrameBundle:
    method: str
    frame_index: int
    raw_count: int
    detected_keypoints: np.ndarray
    described_keypoints: np.ndarray
    descriptors: np.ndarray
    detection_time_s: float
    descriptor_time_s: float
    report_cache_hit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate a stratified held-out detector/registration report "
            "for FPS-projection, ISS3D, and Harris3D."
        ),
    )
    parser.add_argument("--capture", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--kinect-seq", default=None)
    parser.add_argument("--pairs", type=base.parse_pairs, default=PRIMARY_PAIRS)
    parser.add_argument(
        "--negative-pairs", type=base.parse_pairs, default=NEGATIVE_PAIRS
    )
    parser.add_argument("--skip-negative-controls", action="store_true")
    parser.add_argument("--keypoint-budget", type=int, default=150)
    parser.add_argument("--fps-count", type=int, default=500)
    parser.add_argument("--iss-salient-radius-m", type=float, default=0.09)
    parser.add_argument("--iss-nonmax-radius-m", type=float, default=0.06)
    parser.add_argument("--harris-radius-m", type=float, default=0.045)
    parser.add_argument("--descriptor-radius-m", type=float, default=0.30)
    parser.add_argument("--descriptor-batch-size", type=int, default=24)
    parser.add_argument("--ratio-threshold", type=float, default=0.98)
    parser.add_argument("--max-descriptor-distance", type=float, default=1.20)
    parser.add_argument("--ransac-distance-m", type=float, default=0.10)
    parser.add_argument("--ransac-iterations", type=int, default=100_000)
    parser.add_argument("--overlap-distance-m", type=float, default=0.040)
    parser.add_argument("--visibility-distance-m", type=float, default=0.030)
    parser.add_argument("--localization-distance-m", type=float, default=0.050)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=pipeline.DEFAULT_CHECKPOINT)
    parser.add_argument("--normal-module", type=Path, default=None)
    parser.add_argument("--kinect-fx", type=float, default=585.0)
    parser.add_argument("--kinect-fy", type=float, default=585.0)
    parser.add_argument("--kinect-cx", type=float, default=320.0)
    parser.add_argument("--kinect-cy", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.keypoint_budget,
        args.fps_count,
        args.iss_salient_radius_m,
        args.iss_nonmax_radius_m,
        args.harris_radius_m,
        args.descriptor_radius_m,
        args.descriptor_batch_size,
        args.max_descriptor_distance,
        args.ransac_distance_m,
        args.ransac_iterations,
        args.overlap_distance_m,
        args.visibility_distance_m,
        args.localization_distance_m,
        args.bootstrap_iterations,
    )
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("All counts, distances, and iteration limits must be positive")
    if not 0.0 < args.ratio_threshold <= 1.0:
        raise ValueError("--ratio-threshold must be in (0,1]")


def rigid_relative_transform(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return source-camera -> target-camera without inverting rounded matrices."""
    source_rotation = pipeline.project_rotation_matrix(source_pose[:3, :3])
    target_rotation = pipeline.project_rotation_matrix(target_pose[:3, :3])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = target_rotation.T @ source_rotation
    transform[:3, 3] = target_rotation.T @ (
        source_pose[:3, 3] - target_pose[:3, 3]
    )
    return transform


def deterministic_sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), maximum, replace=False))]


def dense_overlap(
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    transform: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, float]:
    source = deterministic_sample(source_cloud, 12_000, seed)
    target = deterministic_sample(target_cloud, 12_000, seed + 1)
    source_in_target = pipeline.transform_points(source, transform)
    target_in_source = pipeline.transform_points(target, np.linalg.inv(transform))
    source_fraction = float(
        np.mean(cKDTree(target_cloud).query(source_in_target, k=1)[0] <= threshold)
    )
    target_fraction = float(
        np.mean(cKDTree(source_cloud).query(target_in_source, k=1)[0] <= threshold)
    )
    return {
        "source_to_target_overlap": source_fraction,
        "target_to_source_overlap": target_fraction,
        "symmetric_overlap": min(source_fraction, target_fraction),
    }


def difficulty_from_overlap(overlap: float) -> str:
    if overlap >= 0.70:
        return "easy"
    if overlap >= 0.35:
        return "moderate"
    if overlap >= 0.10:
        return "hard"
    return "negative"


def report_cache_path(
    sequence: Path,
    method: str,
    frame_index: int,
    recipe: pipeline.DetectorRecipe,
    args: argparse.Namespace,
) -> Path:
    payload = {
        "version": REPORT_CACHE_VERSION,
        "sequence": str(sequence.resolve()),
        "method": method,
        "frame": frame_index,
        "recipe": asdict(recipe),
        "keypoint_budget": args.keypoint_budget,
        "fps_count": args.fps_count,
        "iss_salient_radius_m": args.iss_salient_radius_m,
        "iss_nonmax_radius_m": args.iss_nonmax_radius_m,
        "harris_radius_m": args.harris_radius_m,
        "descriptor_radius_m": args.descriptor_radius_m,
        "descriptor_batch_size": args.descriptor_batch_size,
        "descriptor_checkpoint": pipeline.CHECKPOINT_SHA256,
        "seed": args.seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return (
        pipeline.CACHE_ROOT
        / "detector_testset_report"
        / f"{sequence.stem}_{frame_index:06d}_{method}_{digest}.npz"
    )


def prepare_frame_bundle(
    method: str,
    frame_index: int,
    sequence: Path,
    cloud: np.ndarray,
    recipe: pipeline.DetectorRecipe,
    config: Any,
    normal_estimator: Any,
    model: pipeline.MiniSpinNet,
    device: torch.device,
    args: argparse.Namespace,
) -> FrameBundle:
    cache_path = report_cache_path(sequence, method, frame_index, recipe, args)
    if cache_path.is_file() and not args.force_recompute:
        with np.load(cache_path, allow_pickle=False) as saved:
            bundle = FrameBundle(
                method=method,
                frame_index=frame_index,
                raw_count=int(saved["raw_count"]),
                detected_keypoints=np.asarray(saved["detected_keypoints"], dtype=np.float64),
                described_keypoints=np.asarray(saved["described_keypoints"], dtype=np.float64),
                descriptors=np.asarray(saved["descriptors"], dtype=np.float64),
                detection_time_s=float(saved["detection_time_s"]),
                descriptor_time_s=float(saved["descriptor_time_s"]),
                report_cache_hit=True,
            )
        pipeline.log(
            f"[FRAME CACHE {frame_index:06d}] {LABELS[method]}: "
            f"{len(bundle.described_keypoints)} described keys"
        )
        return bundle

    # A report-cache miss always performs a cold detector run.  In particular,
    # do not time the FPS detector's older per-frame cache lookup.
    detector_args = SimpleNamespace(**vars(args))
    detector_args.force_redetect = method == "fps_projection" or args.force_recompute
    detection = base.detect_keypoints(
        method,
        sequence,
        frame_index,
        cloud,
        recipe,
        config,
        normal_estimator,
        detector_args,
    )
    started = time.perf_counter()
    descriptors, valid = pipeline.describe_keypoints(
        model,
        cloud,
        detection.keypoints,
        args.descriptor_radius_m,
        args.descriptor_batch_size,
        device,
        args.seed + frame_index,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    descriptor_time = time.perf_counter() - started
    described = detection.keypoints[valid]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        raw_count=np.int64(detection.raw_count),
        detected_keypoints=np.asarray(detection.keypoints, dtype=np.float64),
        described_keypoints=np.asarray(described, dtype=np.float64),
        descriptors=np.asarray(descriptors, dtype=np.float64),
        detection_time_s=np.float64(detection.elapsed_s),
        descriptor_time_s=np.float64(descriptor_time),
    )
    pipeline.log(
        f"[FRAME READY {frame_index:06d}] {LABELS[method]}: "
        f"raw={detection.raw_count}, budget={len(detection.keypoints)}, "
        f"described={len(described)}, cold detector={detection.elapsed_s:.2f}s"
    )
    return FrameBundle(
        method=method,
        frame_index=frame_index,
        raw_count=detection.raw_count,
        detected_keypoints=detection.keypoints,
        described_keypoints=described,
        descriptors=descriptors,
        detection_time_s=detection.elapsed_s,
        descriptor_time_s=descriptor_time,
        report_cache_hit=False,
    )


def frame_spatial_metrics(
    bundle: FrameBundle,
    cloud: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    sample = deterministic_sample(cloud, 12_000, seed + bundle.frame_index)
    distances = cKDTree(bundle.detected_keypoints).query(sample, k=1)[0]
    return {
        "frame": bundle.frame_index,
        "detector": bundle.method,
        "detector_label": LABELS[bundle.method],
        "raw_keypoints": bundle.raw_count,
        "budgeted_keypoints": len(bundle.detected_keypoints),
        "described_keypoints": len(bundle.described_keypoints),
        "descriptor_valid_fraction": len(bundle.described_keypoints)
        / len(bundle.detected_keypoints),
        "cloud_coverage_50mm": float(np.mean(distances <= 0.050)),
        "cloud_coverage_100mm": float(np.mean(distances <= 0.100)),
        "cloud_to_keypoint_median_m": float(np.median(distances)),
        "cloud_to_keypoint_p90_m": float(np.quantile(distances, 0.90)),
        "cold_detection_time_s": bundle.detection_time_s,
        "descriptor_time_s": bundle.descriptor_time_s,
        "report_cache_hit": bundle.report_cache_hit,
    }


def overlap_keypoint_metrics(
    source_keys: np.ndarray,
    target_keys: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    transform: np.ndarray,
    visibility_threshold: float,
    localization_threshold: float,
) -> dict[str, Any]:
    inverse = np.linalg.inv(transform)
    source_aligned = pipeline.transform_points(source_keys, transform)
    target_aligned = pipeline.transform_points(target_keys, inverse)
    source_visible = (
        cKDTree(target_cloud).query(source_aligned, k=1)[0] <= visibility_threshold
    )
    target_visible = (
        cKDTree(source_cloud).query(target_aligned, k=1)[0] <= visibility_threshold
    )
    source_visible_ids = np.flatnonzero(source_visible)
    target_visible_ids = np.flatnonzero(target_visible)

    if len(target_visible_ids):
        source_distances = cKDTree(target_keys[target_visible_ids]).query(
            source_aligned[source_visible_ids], k=1
        )[0]
    else:
        source_distances = np.full(len(source_visible_ids), np.inf)
    if len(source_visible_ids):
        target_distances = cKDTree(source_keys[source_visible_ids]).query(
            target_aligned[target_visible_ids], k=1
        )[0]
    else:
        target_distances = np.full(len(target_visible_ids), np.inf)

    metrics: dict[str, Any] = {
        "source_overlap_visible_keypoints": len(source_visible_ids),
        "target_overlap_visible_keypoints": len(target_visible_ids),
        "source_overlap_visible_fraction": float(np.mean(source_visible)),
        "target_overlap_visible_fraction": float(np.mean(target_visible)),
    }
    directional_sets = [source_distances, target_distances]
    for threshold_mm in REPEATABILITY_THRESHOLDS_MM:
        threshold_m = threshold_mm / 1000.0
        rates = [
            float(np.mean(values <= threshold_m))
            for values in directional_sets
            if len(values)
        ]
        metrics[f"repeatability_{threshold_mm}mm"] = (
            None if not rates else float(np.mean(rates))
        )
    auc_values = [
        float(np.mean(np.clip((0.050 - values) / 0.050, 0.0, 1.0)))
        for values in directional_sets
        if len(values)
    ]
    metrics["repeatability_auc_50mm"] = (
        None if not auc_values else float(np.mean(auc_values))
    )

    localization_errors = np.empty(0, dtype=np.float64)
    if len(source_visible_ids) and len(target_visible_ids):
        aligned_visible_source = source_aligned[source_visible_ids]
        visible_target = target_keys[target_visible_ids]
        forward_distances, forward_indices = cKDTree(visible_target).query(
            aligned_visible_source, k=1
        )
        _, reverse_indices = cKDTree(aligned_visible_source).query(visible_target, k=1)
        source_positions = np.arange(len(aligned_visible_source))
        mutual = reverse_indices[forward_indices] == source_positions
        mutual &= forward_distances <= localization_threshold
        localization_errors = forward_distances[mutual]

    metrics["localization_pair_count"] = len(localization_errors)
    metrics["localization_errors_m"] = localization_errors.tolist()
    metrics["localization_median_m"] = (
        None if not len(localization_errors) else float(np.median(localization_errors))
    )
    metrics["localization_p90_m"] = (
        None
        if not len(localization_errors)
        else float(np.quantile(localization_errors, 0.90))
    )
    metrics["localization_rmse_m"] = (
        None
        if not len(localization_errors)
        else float(np.sqrt(np.mean(localization_errors**2)))
    )
    return metrics


def evaluate_pair_method(
    method: str,
    source_frame: int,
    target_frame: int,
    primary: bool,
    pair_metadata: dict[str, Any],
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    source_bundle: FrameBundle,
    target_bundle: FrameBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gt_transform = np.asarray(
        pair_metadata["ground_truth_source_to_target"], dtype=np.float64
    )
    row: dict[str, Any] = {
        "pair": f"{source_frame}->{target_frame}",
        "source_frame": source_frame,
        "target_frame": target_frame,
        "primary_pair": primary,
        "difficulty": pair_metadata["difficulty"],
        "dense_overlap": pair_metadata["symmetric_overlap"],
        "gt_camera_delta_translation_m": pair_metadata[
            "gt_camera_delta_translation_m"
        ],
        "gt_camera_delta_rotation_deg": pair_metadata["gt_camera_delta_rotation_deg"],
        "detector": method,
        "detector_label": LABELS[method],
        "source_keypoints": len(source_bundle.detected_keypoints),
        "target_keypoints": len(target_bundle.detected_keypoints),
        "source_described_keypoints": len(source_bundle.described_keypoints),
        "target_described_keypoints": len(target_bundle.described_keypoints),
        "registration_status": "failed",
        "registration_error": None,
        "registration_success_strict": False,
        "registration_success_relaxed": False,
    }
    row.update(
        overlap_keypoint_metrics(
            source_bundle.detected_keypoints,
            target_bundle.detected_keypoints,
            source_cloud,
            target_cloud,
            gt_transform,
            args.visibility_distance_m,
            args.localization_distance_m,
        )
    )

    started = time.perf_counter()
    source_ids, target_ids, descriptor_distances = base.strict_mutual_matches(
        source_bundle.descriptors,
        target_bundle.descriptors,
        args.ratio_threshold,
        args.max_descriptor_distance,
    )
    row["matching_time_s"] = time.perf_counter() - started
    row["mutual_matches"] = len(source_ids)
    row["descriptor_distance_median"] = (
        None if not len(descriptor_distances) else float(np.median(descriptor_distances))
    )
    if len(source_ids):
        gt_residuals = np.linalg.norm(
            pipeline.transform_points(
                source_bundle.described_keypoints[source_ids], gt_transform
            )
            - target_bundle.described_keypoints[target_ids],
            axis=1,
        )
    else:
        gt_residuals = np.empty(0, dtype=np.float64)
    for threshold_mm in (30, 50, 100):
        correct = int(np.count_nonzero(gt_residuals <= threshold_mm / 1000.0))
        row[f"gt_correct_matches_{threshold_mm}mm"] = correct
        row[f"gt_match_precision_{threshold_mm}mm"] = (
            None if not len(gt_residuals) else correct / len(gt_residuals)
        )
    row["feature_match_recall_100mm"] = bool(
        row["gt_match_precision_100mm"] is not None
        and row["gt_match_precision_100mm"] > 0.05
    )

    row.update(
        {
            "ransac_inliers": 0,
            "ransac_inlier_ratio": None,
            "ransac_time_s": None,
            "dense_fitness_30mm": None,
            "dense_rmse_30mm_m": None,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "transform_source_to_target": None,
        }
    )
    try:
        registration = base.deterministic_ransac(
            source_bundle.described_keypoints,
            target_bundle.described_keypoints,
            source_ids,
            target_ids,
            source_cloud,
            target_cloud,
            args.ransac_distance_m,
            args.ransac_iterations,
            args.seed,
        )
        transform = registration["transform"]
        translation_error, rotation_error = pipeline.transform_error(
            transform, gt_transform
        )
        inlier_count = int(np.count_nonzero(registration["inliers"]))
        row.update(
            {
                "registration_status": "ok",
                "ransac_inliers": inlier_count,
                "ransac_inlier_ratio": inlier_count / len(source_ids),
                "ransac_time_s": registration["ransac_time_s"],
                "ransac_trials": registration["ransac_trials"],
                "ransac_hypotheses_evaluated": registration[
                    "ransac_hypotheses_evaluated"
                ],
                "dense_fitness_30mm": registration["final_dense_fitness"],
                "dense_rmse_30mm_m": registration["final_dense_rmse_m"],
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "transform_source_to_target": transform.tolist(),
                "registration_success_strict": bool(
                    translation_error <= 0.050 and rotation_error <= 5.0
                ),
                "registration_success_relaxed": bool(
                    translation_error <= 0.100 and rotation_error <= 10.0
                ),
            }
        )
    except Exception as error:
        row["registration_error"] = f"{type(error).__name__}: {error}"
    return row


def finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and np.isfinite(float(row[field]))
    ]
    return np.asarray(values, dtype=np.float64)


def quartiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if not len(values):
        return None, None, None
    return tuple(float(value) for value in np.quantile(values, (0.25, 0.50, 0.75)))


def bootstrap_ci(
    values: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    estimates = np.asarray([estimator(sample) for sample in samples])
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def make_summary(
    pair_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        method_frames = [row for row in frame_rows if row["detector"] == method]
        for group in ("all", "easy", "moderate", "hard"):
            rows = [
                row
                for row in pair_rows
                if row["detector"] == method
                and row["primary_pair"]
                and (group == "all" or row["difficulty"] == group)
            ]
            summary: dict[str, Any] = {
                "detector": method,
                "detector_label": LABELS[method],
                "difficulty": group,
                "pair_count": len(rows),
                "registration_estimates": sum(
                    row["registration_status"] == "ok" for row in rows
                ),
                "registration_failures": sum(
                    row["registration_status"] != "ok" for row in rows
                ),
            }
            for field in (
                "repeatability_20mm",
                "repeatability_30mm",
                "repeatability_50mm",
                "repeatability_75mm",
                "repeatability_100mm",
                "repeatability_auc_50mm",
                "localization_pair_count",
                "localization_median_m",
                "localization_p90_m",
                "mutual_matches",
                "gt_match_precision_50mm",
                "gt_match_precision_100mm",
                "ransac_inliers",
                "translation_error_m",
                "rotation_error_deg",
                "dense_fitness_30mm",
            ):
                values = finite_values(rows, field)
                q1, median, q3 = quartiles(values)
                summary[f"{field}_median"] = median
                summary[f"{field}_q1"] = q1
                summary[f"{field}_q3"] = q3
                low, high = bootstrap_ci(
                    values,
                    np.median,
                    args.bootstrap_iterations,
                    args.seed + method_index,
                )
                summary[f"{field}_median_ci95_low"] = low
                summary[f"{field}_median_ci95_high"] = high
            strict = sum(bool(row["registration_success_strict"]) for row in rows)
            relaxed = sum(bool(row["registration_success_relaxed"]) for row in rows)
            fmr = sum(bool(row["feature_match_recall_100mm"]) for row in rows)
            summary["strict_successes"] = strict
            summary["strict_success_rate"] = None if not rows else strict / len(rows)
            (
                summary["strict_success_ci95_low"],
                summary["strict_success_ci95_high"],
            ) = wilson_interval(strict, len(rows))
            summary["relaxed_successes"] = relaxed
            summary["relaxed_success_rate"] = None if not rows else relaxed / len(rows)
            (
                summary["relaxed_success_ci95_low"],
                summary["relaxed_success_ci95_high"],
            ) = wilson_interval(relaxed, len(rows))
            summary["fmr_100mm_successes"] = fmr
            summary["fmr_100mm_rate"] = None if not rows else fmr / len(rows)

            if group == "all":
                for field in (
                    "raw_keypoints",
                    "budgeted_keypoints",
                    "described_keypoints",
                    "descriptor_valid_fraction",
                    "cloud_coverage_50mm",
                    "cloud_coverage_100mm",
                    "cloud_to_keypoint_median_m",
                    "cloud_to_keypoint_p90_m",
                    "cold_detection_time_s",
                    "descriptor_time_s",
                ):
                    values = finite_values(method_frames, field)
                    q1, median, q3 = quartiles(values)
                    summary[f"frame_{field}_median"] = median
                    summary[f"frame_{field}_q1"] = q1
                    summary[f"frame_{field}_q3"] = q3
            summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    fields = sorted({key for row in rows for key in row if key not in excluded})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key not in excluded})


def format_median_iqr(
    summary: dict[str, Any], field: str, scale: float = 1.0, digits: int = 1
) -> str:
    values = [
        summary.get(f"{field}_median"),
        summary.get(f"{field}_q1"),
        summary.get(f"{field}_q3"),
    ]
    if any(value is None for value in values):
        return "—"
    median, q1, q3 = (float(value) * scale for value in values)
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---"] * len(headers)) + " |")
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def save_figure(
    path: Path,
    pair_metadata: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 16), dpi=160)
    primary_meta = [row for row in pair_metadata if row["primary_pair"]]
    difficulty_colors = {"easy": "#4daf4a", "moderate": "#ff9f1c", "hard": "#d62728"}

    axis = axes[0, 0]
    for difficulty in ("easy", "moderate", "hard"):
        rows = [row for row in primary_meta if row["difficulty"] == difficulty]
        axis.scatter(
            [100.0 * row["symmetric_overlap"] for row in rows],
            [row["gt_camera_delta_translation_m"] for row in rows],
            s=48,
            label=difficulty,
            color=difficulty_colors[difficulty],
        )
    axis.set_title("Stratified pair coverage")
    axis.set_xlabel("symmetric dense overlap [%]")
    axis.set_ylabel("GT camera translation [m]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    thresholds = np.asarray(REPEATABILITY_THRESHOLDS_MM)
    primary_rows = [row for row in pair_rows if row["primary_pair"]]
    for method_index, method in enumerate(METHODS):
        method_rows = [row for row in primary_rows if row["detector"] == method]
        means, lows, highs = [], [], []
        for threshold in thresholds:
            values = finite_values(method_rows, f"repeatability_{threshold}mm")
            means.append(float(np.mean(values)))
            low, high = bootstrap_ci(
                values,
                np.mean,
                args.bootstrap_iterations,
                args.seed + 20 + method_index,
            )
            lows.append(low)
            highs.append(high)
        axis.plot(thresholds, 100.0 * np.asarray(means), marker="o", label=LABELS[method], color=COLORS[method])
        axis.fill_between(
            thresholds,
            100.0 * np.asarray(lows),
            100.0 * np.asarray(highs),
            color=COLORS[method],
            alpha=0.14,
        )
    axis.set_title("Overlap-aware keypoint repeatability (mean, bootstrap 95% CI)")
    axis.set_xlabel("localization threshold [mm]")
    axis.set_ylabel("repeatability [%]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    location_data = [
        1000.0 * finite_values(
            [row for row in primary_rows if row["detector"] == method],
            "localization_median_m",
        )
        for method in METHODS
    ]
    box = axis.boxplot(location_data, tick_labels=[LABELS[m] for m in METHODS], patch_artist=True)
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.70)
    axis.set_title("Reciprocal-NN keypoint localization consistency")
    axis.set_ylabel("pair median error [mm] (lower is better)")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    groups = ("easy", "moderate", "hard", "all")
    x = np.arange(len(groups))
    width = 0.24
    summary_lookup = {
        (row["detector"], row["difficulty"]): row for row in summaries
    }
    for method_index, method in enumerate(METHODS):
        values, lower_errors, upper_errors = [], [], []
        for group in groups:
            summary = summary_lookup[(method, group)]
            rate = summary["strict_success_rate"]
            low = summary["strict_success_ci95_low"]
            high = summary["strict_success_ci95_high"]
            if rate is None:
                values.append(np.nan)
                lower_errors.append(0.0)
                upper_errors.append(0.0)
            else:
                values.append(100.0 * rate)
                lower_errors.append(100.0 * (rate - low))
                upper_errors.append(100.0 * (high - rate))
        axis.bar(
            x + (method_index - 1) * width,
            values,
            width,
            yerr=np.vstack((lower_errors, upper_errors)),
            capsize=3,
            label=LABELS[method],
            color=COLORS[method],
        )
    axis.set_xticks(x, [group.title() for group in groups])
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("registration success [%]")
    axis.set_title("RANSAC-only strict success (RTE ≤50 mm and RRE ≤5°)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[2, 0]
    for method in METHODS:
        rows = [
            row
            for row in primary_rows
            if row["detector"] == method and row["registration_status"] == "ok"
        ]
        axis.scatter(
            [1000.0 * row["translation_error_m"] for row in rows],
            [row["rotation_error_deg"] for row in rows],
            label=f"{LABELS[method]} (n={len(rows)})",
            color=COLORS[method],
            alpha=0.78,
            s=42,
        )
    axis.axvline(50.0, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Registration pose errors; dashed lines = strict boundary")
    axis.set_xlabel("RTE [mm]")
    axis.set_ylabel("RRE [deg]")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[2, 1]
    timing = [
        finite_values([row for row in frame_rows if row["detector"] == method], "cold_detection_time_s")
        for method in METHODS
    ]
    box = axis.boxplot(timing, tick_labels=[LABELS[m] for m in METHODS], patch_artist=True)
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.70)
    axis.set_yscale("log")
    axis.set_title("Cold detector runtime per frame")
    axis.set_ylabel("seconds (log scale)")
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Held-out RedKitchen seq-03 | common 150-keypoint budget | Mini-SpinNet + mutual top-1 + RANSAC | no ICP",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_report(
    path: Path,
    figure_name: str,
    sequence: Path,
    pair_metadata: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    overall = {
        row["detector"]: row for row in summaries if row["difficulty"] == "all"
    }

    keypoint_rows: list[list[str]] = []
    registration_rows: list[list[str]] = []
    for method in METHODS:
        row = overall[method]
        keypoint_rows.append(
            [
                LABELS[method],
                format_median_iqr(row, "repeatability_30mm", 100.0, 1),
                format_median_iqr(row, "repeatability_50mm", 100.0, 1),
                format_median_iqr(row, "localization_median_m", 1000.0, 1),
                format_median_iqr(row, "localization_p90_m", 1000.0, 1),
                format_median_iqr(row, "localization_pair_count", 1.0, 0),
                f"{100.0 * row['frame_cloud_coverage_100mm_median']:.1f}",
                f"{row['frame_cold_detection_time_s_median']:.3f}",
            ]
        )
        registration_rows.append(
            [
                LABELS[method],
                format_median_iqr(row, "mutual_matches", 1.0, 0),
                format_median_iqr(row, "gt_match_precision_50mm", 100.0, 1),
                f"{row['registration_estimates']}/{row['pair_count']}",
                f"{row['strict_successes']}/{row['pair_count']} "
                f"({100.0 * row['strict_success_rate']:.1f}%; "
                f"CI {100.0 * row['strict_success_ci95_low']:.1f}–"
                f"{100.0 * row['strict_success_ci95_high']:.1f}%)",
                f"{row['relaxed_successes']}/{row['pair_count']} ({100.0 * row['relaxed_success_rate']:.1f}%)",
                format_median_iqr(row, "translation_error_m", 1000.0, 1),
                format_median_iqr(row, "rotation_error_deg", 1.0, 2),
            ]
        )

    difficulty_rows: list[list[str]] = []
    for difficulty in ("easy", "moderate", "hard"):
        output = [difficulty.title()]
        for method in METHODS:
            row = next(
                item
                for item in summaries
                if item["detector"] == method and item["difficulty"] == difficulty
            )
            if row["strict_success_rate"] is None:
                output.append("—")
            else:
                output.append(
                    f"{row['strict_successes']}/{row['pair_count']} "
                    f"({100.0 * row['strict_success_rate']:.0f}%)"
                )
        difficulty_rows.append(output)

    pair_detail_rows: list[list[str]] = []
    primary_metadata = [row for row in pair_metadata if row["primary_pair"]]
    result_lookup = {(row["pair"], row["detector"]): row for row in pair_rows}
    for metadata in primary_metadata:
        pair = f"{metadata['source_frame']}->{metadata['target_frame']}"
        values = [
            pair,
            metadata["difficulty"],
            f"{100.0 * metadata['symmetric_overlap']:.1f}%",
        ]
        for method in METHODS:
            row = result_lookup[(pair, method)]
            if row["registration_status"] != "ok":
                values.append("FAIL")
            else:
                marker = "✓" if row["registration_success_strict"] else "✗"
                values.append(
                    f"{1000.0 * row['translation_error_m']:.1f} mm / "
                    f"{row['rotation_error_deg']:.2f}° {marker}"
                )
        pair_detail_rows.append(values)

    negative_rows = [row for row in pair_rows if not row["primary_pair"]]
    negative_text: list[str] = []
    if negative_rows:
        for method in METHODS:
            rows = [row for row in negative_rows if row["detector"] == method]
            hypotheses = sum(row["registration_status"] == "ok" for row in rows)
            negative_text.append(
                f"- {LABELS[method]}: {hypotheses}/{len(rows)}개에서 pose hypothesis를 반환"
            )
    else:
        negative_text.append("- 실행하지 않음")

    best_repeatability = max(
        METHODS,
        key=lambda method: overall[method]["repeatability_50mm_median"] or -math.inf,
    )
    localization_candidates = [
        method
        for method in METHODS
        if overall[method]["localization_median_m_median"] is not None
    ]
    best_localization = min(
        localization_candidates,
        key=lambda method: overall[method]["localization_median_m_median"],
    )
    best_success_value = max(overall[method]["strict_success_rate"] for method in METHODS)
    best_success = [
        LABELS[method]
        for method in METHODS
        if overall[method]["strict_success_rate"] == best_success_value
    ]

    primary_count = len(primary_metadata)
    negative_count = sum(not row["primary_pair"] for row in pair_metadata)
    evaluated_frames = sorted(
        {
            frame
            for row in pair_metadata
            for frame in (row["source_frame"], row["target_frame"])
        }
    )
    stratum_counts = {
        difficulty: sum(
            row["primary_pair"] and row["difficulty"] == difficulty
            for row in pair_metadata
        )
        for difficulty in ("easy", "moderate", "hard")
    }

    report = f"""# FPS-projection / ISS3D / Harris3D held-out 비교 보고서

## 결론

- **키포인트 반복성 최고:** Rep@50 mm 중앙값 기준 **{LABELS[best_repeatability]}**.
- **위치 동일성 오차 최저:** reciprocal-NN 중앙값 기준 **{LABELS[best_localization]}**.
- **RANSAC-only 정합 성공률:** 최고 방법은 **{', '.join(best_success)}**이며 strict 기준 성공률은 **{100.0 * best_success_value:.1f}%**입니다.
- 이 결과는 단일 종합점수가 아니라 `키포인트 재검출성`, `위치 정확도`, `최종 정합 성공률`을 분리해서 해석해야 합니다.

![종합 결과]({figure_name})

## 평가 범위

- 데이터: `{sequence}`
- 범위: held-out RedKitchen `seq-03` 한 개 sequence, 총 1000 frame 중 시간축 `{evaluated_frames[0]}..{evaluated_frames[-1]}`을 덮는 {len(evaluated_frames)}개 frame
- primary pair: {primary_count}개 — GT dense overlap 기준 easy {stratum_counts['easy']} / moderate {stratum_counts['moderate']} / hard {stratum_counts['hard']}
- negative control: near-zero-overlap {negative_count}개 pair, primary 집계에서 제외
- **공식 7Scenes test set 전체 결과가 아니라 현재 내려받은 RedKitchen seq-03에 대한 결과**입니다.

## 공통 프로토콜

- 입력 cloud: 15 mm voxel
- detector별 native 출력 수를 기록한 뒤 공통 spatial-FPS budgeter로 `{args.keypoint_budget}`개 제한
- descriptor: pretrained BUFFER-X Mini-SpinNet 32-D, radius `{args.descriptor_radius_m:.2f}` m
- matching: strict bidirectional mutual top-1 + bidirectional ratio test `{args.ratio_threshold:.2f}`
- pose: deterministic triple RANSAC + Kabsch, inlier threshold `{1000.0 * args.ransac_distance_m:.0f}` mm
- ICP: 사용하지 않음; temporal bridge: 사용하지 않음; matching relaxation: 사용하지 않음
- GT pose는 pair 선정과 **평가에만** 사용하며 특징 기술·매칭·RANSAC에는 사용하지 않음
- 7Scenes의 반올림된 pose 회전은 nearest SO(3)로 투영한 뒤 RRE 계산

## 키포인트 성능

아래 값은 {primary_count}개 pair의 `중앙값 [Q1, Q3]`입니다. Repeatability 분모는 GT 정렬 후 반대 cloud에서 `{1000.0 * args.visibility_distance_m:.0f}` mm 이내 표면이 확인되는 overlap-visible keypoint만 사용했습니다. 위치 동일성은 `{1000.0 * args.localization_distance_m:.0f}` mm 이내 reciprocal-NN keypoint 쌍의 3D 거리이며, 쌍 수를 함께 표시했습니다.

{markdown_table(
        ['Detector', 'Rep@30 %', 'Rep@50 %', '위치 median mm', '위치 p90 mm', '위치 pair 수', 'Cloud COV@100 %', 'Cold detect s/frame'],
        keypoint_rows,
    )}

## 매칭 및 정합 성능

Strict 성공은 `RTE ≤ 50 mm AND RRE ≤ 5°`, relaxed 성공은 `RTE ≤ 100 mm AND RRE ≤ 10°`입니다. RANSAC 실패도 성공률 분모 {primary_count}개에 포함했습니다. 괄호의 CI는 Wilson 95% 신뢰구간입니다. RTE/RRE 통계는 pose hypothesis가 나온 pair만 대상으로 하며, 그 수를 별도로 표시했습니다.

{markdown_table(
        ['Detector', 'Mutual matches', 'GT precision@50 %', f'Pose N/{primary_count}', 'Strict success', 'Relaxed success', 'RTE mm', 'RRE deg'],
        registration_rows,
    )}

### 난이도별 strict 정합 성공률

{markdown_table(
        ['Overlap stratum'] + [LABELS[method] for method in METHODS],
        difficulty_rows,
    )}

### Pair별 정합 오차

`✓`는 strict 성공, `✗`는 pose가 산출됐지만 strict 기준 실패입니다.

{markdown_table(
        ['Pair', '난이도', 'Overlap'] + [LABELS[method] for method in METHODS],
        pair_detail_rows,
    )}

## Near-zero-overlap negative control

정답 correspondence가 사실상 없는 pair에서 pose를 억지로 반환하는지 확인한 결과입니다. 낮을수록 보수적인 rejection입니다.

{chr(10).join(negative_text)}

## 해석상 주의점

1. 이 실험은 RedKitchen seq-03 한 sequence의 층화 표본 {primary_count}쌍입니다. scene 간 일반화 결론에는 여러 held-out scene이 추가로 필요합니다.
2. 동일한 keypoint 수를 맞추기 위한 spatial-FPS cap이 ISS/Harris의 native saliency 순위를 변경합니다. 따라서 결과는 엄밀히 `detector + 공통 budgeter` 비교입니다. native raw count도 원시 CSV에 보존했습니다.
3. Overlap-visible 판정은 projective z-buffer가 아니라 GT 정렬된 dense cloud의 nearest-neighbor 거리 proxy입니다.
4. Mini-SpinNet은 모든 detector에 동일하게 적용되지만 detector별 국소 패치와의 궁합이 결과에 반영됩니다.
5. RANSAC은 최소 6 inlier를 요구하므로 정확한 correspondence가 3–5개뿐인 경우도 실패 처리됩니다.
6. 검출 시간은 report cache가 처음 만들어질 때 측정한 cold detector 시간이며, 이후 실행은 저장된 원 측정값을 재사용합니다.
7. 여러 pair가 일부 frame을 공유하므로 bootstrap/Wilson 구간은 완전히 독립적인 표본을 가정한 참고용 구간입니다.

## 참고 자료

- [Microsoft 7-Scenes RGB-D Dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [BUFFER-X 공식 저장소](https://github.com/MIT-SPARK/BUFFER-X)
- [3DMatch 공식 사이트](https://3dmatch.cs.princeton.edu/)

## 산출물

- `detector_testset_pair_results.csv`: pair × detector 원시 결과
- `detector_testset_frame_results.csv`: frame × detector 검출/coverage/시간
- `detector_testset_summary.csv`: detector × 난이도 집계
- `detector_testset_results.json`: transform과 세부 배열을 포함한 전체 결과
"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pipeline.choose_device(args.device)
    sequence = pipeline.detector.resolve_kinect_sequence(args.capture, args.kinect_seq)
    recipe = pipeline.DetectorRecipe()
    config = pipeline.make_detector_config(recipe)
    normal_estimator = pipeline.detector.load_normal_estimator(args.normal_module)
    checkpoint = pipeline.ensure_checkpoint(args.checkpoint)
    model = pipeline.load_minispinnet(checkpoint, device)
    negative_pairs = [] if args.skip_negative_controls else args.negative_pairs
    all_pairs = list(args.pairs) + list(negative_pairs)
    frames = sorted({frame for pair in all_pairs for frame in pair})

    pipeline.log(f"[SOURCE] {sequence}")
    pipeline.log(
        f"[PROTOCOL] {len(args.pairs)} primary pairs + {len(negative_pairs)} controls; "
        f"{len(frames)} unique frames; budget={args.keypoint_budget}; ICP=OFF"
    )
    clouds: dict[int, np.ndarray] = {}
    poses: dict[int, np.ndarray] = {}
    for frame in frames:
        clouds[frame], _ = pipeline.load_frame_cloud(
            sequence,
            frame,
            recipe,
            args.kinect_fx,
            args.kinect_fy,
            args.kinect_cx,
            args.kinect_cy,
        )
        pose = pipeline.read_pose(sequence, frame)
        if pose is None:
            raise RuntimeError(f"Missing or invalid GT pose for frame {frame}")
        poses[frame] = pose

    bundles: dict[tuple[str, int], FrameBundle] = {}
    frame_rows: list[dict[str, Any]] = []
    for method in METHODS:
        pipeline.log(f"\n[PRECOMPUTE] {LABELS[method]}")
        for frame in frames:
            bundle = prepare_frame_bundle(
                method,
                frame,
                sequence,
                clouds[frame],
                recipe,
                config,
                normal_estimator,
                model,
                device,
                args,
            )
            bundles[(method, frame)] = bundle
            frame_rows.append(frame_spatial_metrics(bundle, clouds[frame], args.seed))

    pair_metadata: list[dict[str, Any]] = []
    pair_metadata_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for pair_index, (source_frame, target_frame) in enumerate(all_pairs):
        primary = pair_index < len(args.pairs)
        transform = rigid_relative_transform(poses[source_frame], poses[target_frame])
        overlap = dense_overlap(
            clouds[source_frame],
            clouds[target_frame],
            transform,
            args.overlap_distance_m,
            args.seed + source_frame * 1009 + target_frame,
        )
        delta_m, delta_deg = pipeline.transform_error(transform, np.eye(4))
        metadata = {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "primary_pair": primary,
            "ground_truth_source_to_target": transform.tolist(),
            "gt_camera_delta_translation_m": delta_m,
            "gt_camera_delta_rotation_deg": delta_deg,
            **overlap,
            "difficulty": difficulty_from_overlap(overlap["symmetric_overlap"]),
        }
        pair_metadata.append(metadata)
        pair_metadata_lookup[(source_frame, target_frame)] = metadata
        pipeline.log(
            f"[PAIR META {source_frame}->{target_frame}] "
            f"{metadata['difficulty']}, overlap={100.0 * overlap['symmetric_overlap']:.1f}%, "
            f"delta={delta_m:.3f}m/{delta_deg:.2f}deg"
        )

    pair_rows: list[dict[str, Any]] = []
    for pair_index, (source_frame, target_frame) in enumerate(all_pairs):
        primary = pair_index < len(args.pairs)
        for method in METHODS:
            pipeline.log(
                f"[EVALUATE {source_frame}->{target_frame}] {LABELS[method]}"
            )
            row = evaluate_pair_method(
                method,
                source_frame,
                target_frame,
                primary,
                pair_metadata_lookup[(source_frame, target_frame)],
                clouds[source_frame],
                clouds[target_frame],
                bundles[(method, source_frame)],
                bundles[(method, target_frame)],
                args,
            )
            pair_rows.append(row)
            status = (
                f"RTE={1000.0 * row['translation_error_m']:.1f}mm, "
                f"RRE={row['rotation_error_deg']:.2f}deg"
                if row["registration_status"] == "ok"
                else row["registration_error"]
            )
            repeatability = row["repeatability_50mm"]
            repeatability_text = (
                "--" if repeatability is None else f"{100.0 * repeatability:.1f}%"
            )
            pipeline.log(
                f"  rep50={repeatability_text}, "
                f"matches={row['mutual_matches']}, {status}"
            )

    summaries = make_summary(pair_rows, frame_rows, args)
    if args.output_dir is None:
        output_dir = (
            SCRIPT_DIR
            / "detector_testset_report_results"
            / f"{sequence.stem}_stratified{len(args.pairs)}_seed{args.seed}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pair_csv": output_dir / "detector_testset_pair_results.csv",
        "frame_csv": output_dir / "detector_testset_frame_results.csv",
        "summary_csv": output_dir / "detector_testset_summary.csv",
        "json": output_dir / "detector_testset_results.json",
        "figure": output_dir / "detector_testset_report.png",
        "report": output_dir / "detector_testset_report.md",
    }
    write_csv(paths["pair_csv"], pair_rows, {"localization_errors_m", "transform_source_to_target"})
    write_csv(paths["frame_csv"], frame_rows)
    write_csv(paths["summary_csv"], summaries)
    payload = {
        "protocol": {
            "scope": "held-out Microsoft 7-Scenes RedKitchen seq-03 only",
            "official_test_set_complete": False,
            "primary_pair_count": len(args.pairs),
            "negative_control_pair_count": len(negative_pairs),
            "unique_frame_count": len(frames),
            "pair_sampling": "GT dense-overlap stratified; easy/moderate/hard",
            "overlap_threshold_m": args.overlap_distance_m,
            "difficulty_thresholds": {
                "easy": "overlap >= 0.70",
                "moderate": "0.35 <= overlap < 0.70",
                "hard": "0.10 <= overlap < 0.35",
                "negative": "overlap < 0.10",
            },
            "visibility_threshold_m": args.visibility_distance_m,
            "localization_threshold_m": args.localization_distance_m,
            "common_keypoint_budget": args.keypoint_budget,
            "common_budgeter": "deterministic spatial FPS over native detector output",
            "descriptor": "BUFFER-X pretrained Mini-SpinNet 32-D",
            "checkpoint_sha256": pipeline.CHECKPOINT_SHA256,
            "matching": "strict bidirectional mutual top-1 + bidirectional ratio",
            "ratio_threshold": args.ratio_threshold,
            "matching_relaxation": False,
            "ransac": "deterministic exhaustive-or-capped triples + Kabsch",
            "ransac_distance_m": args.ransac_distance_m,
            "ransac_minimum_inliers": "max(6, ceil(0.10 * tentative_matches))",
            "strict_success": "RTE <= 0.050 m and RRE <= 5 deg",
            "relaxed_success": "RTE <= 0.100 m and RRE <= 10 deg",
            "icp_used": False,
            "temporal_bridge_used": False,
            "seed": args.seed,
            "detector_parameters": {
                "fps_count": args.fps_count,
                "iss_salient_radius_m": args.iss_salient_radius_m,
                "iss_nonmax_radius_m": args.iss_nonmax_radius_m,
                "harris_radius_m": args.harris_radius_m,
                "recipe": asdict(recipe),
            },
            "software": {
                "torch": torch.__version__,
                "open3d": o3d.__version__,
                "numpy": np.__version__,
            },
        },
        "sequence": str(sequence),
        "pair_metadata": pair_metadata,
        "frame_results": frame_rows,
        "pair_results": pair_rows,
        "summary": summaries,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_figure(paths["figure"], pair_metadata, pair_rows, frame_rows, summaries, args)
    save_report(
        paths["report"],
        paths["figure"].name,
        sequence,
        pair_metadata,
        pair_rows,
        frame_rows,
        summaries,
        args,
    )
    pipeline.log("\n[SUMMARY] strict RANSAC-only registration success")
    for method in METHODS:
        row = next(
            item
            for item in summaries
            if item["detector"] == method and item["difficulty"] == "all"
        )
        pipeline.log(
            f"  {LABELS[method]:<15} {row['strict_successes']}/{row['pair_count']} "
            f"({100.0 * row['strict_success_rate']:.1f}%), "
            f"Rep@50 median={100.0 * row['repeatability_50mm_median']:.1f}%"
        )
    for name, path in paths.items():
        pipeline.log(f"[OUTPUT {name}] {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pipeline.log("\nCancelled.")
        raise SystemExit(130)
    except Exception as error:
        pipeline.log(f"\nERROR: {type(error).__name__}: {error}")
        raise SystemExit(1)
