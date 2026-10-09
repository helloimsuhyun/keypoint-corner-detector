#!/home/choisuhyun/keypoint-corner-detector/.conda/kinect7scenes-minispinnet/bin/python
"""Lightweight FPS-projection / ISS3D / Harris3D registration benchmark.

Only the detector changes.  Every detector is evaluated with the same spatial
keypoint budget, pretrained BUFFER-X Mini-SpinNet descriptor, strict mutual
top-1 matching, and correspondence RANSAC.  ICP, temporal bridging, and match
filter relaxation are intentionally disabled.

Example:

    cd /home/choisuhyun/keypoint-corner-detector
    ./kinect7scenes_detector_benchmark.py

    ./kinect7scenes_detector_benchmark.py --pairs 10:50

The default input is Microsoft 7-Scenes RedKitchen ``seq-03``.  RedKitchen is
an official held-out 3DMatch test scene and seq-03 is also in the original
7-Scenes TestSplit, so the pretrained 3DMatch descriptor did not train on this
scene.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Open3D's correspondence RANSAC can otherwise choose different hypotheses in
# parallel despite resetting its random seed.  Keep all benchmark backends on
# one thread; the benchmark also uses its own deterministic RANSAC below.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import kinect7scenes_minispinnet_registration as pipeline

# Import after pipeline: it removes workstation ROS/user-site paths so this
# script consistently uses the dedicated Conda environment.
import matplotlib.pyplot as plt
import numpy as np
import pclpybridge as pcl
import torch
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REDKITCHEN_SEQUENCE = Path.home() / "datasets/7scenes/redkitchen/seq-03.zip"
REDKITCHEN_SOURCE_URL = (
    "https://download.microsoft.com/download/2/8/5/"
    "28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
)
REDKITCHEN_SEQ03_SHA256 = (
    "13969f608ede2f7d8bfefa23d042b0c4930d54e7d1bad8c1d54257a122eec5fc"
)
METHODS = ("fps_projection", "iss3d", "harris3d")
LABELS = {
    "fps_projection": "FPS-projection",
    "iss3d": "ISS3D",
    "harris3d": "Harris3D",
}
COLORS = {
    "fps_projection": "#ed7d31",
    "iss3d": "#4e79a7",
    "harris3d": "#59a14f",
}


@dataclass
class Detection:
    keypoints: np.ndarray
    raw_count: int
    elapsed_s: float
    cache_hit: bool | None


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "--pairs must look like SOURCE:TARGET[,SOURCE:TARGET...]"
            )
        try:
            source, target = map(int, fields)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"Invalid frame pair: {item}") from error
        if source < 0 or target < 0 or source == target:
            raise argparse.ArgumentTypeError(f"Invalid frame pair: {item}")
        pairs.append((source, target))
    if not pairs:
        raise argparse.ArgumentTypeError("At least one frame pair is required")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare three 3-D detectors with Mini-SpinNet + RANSAC only (no ICP).",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=DEFAULT_REDKITCHEN_SEQUENCE,
    )
    parser.add_argument("--kinect-seq", default=None)
    parser.add_argument(
        "--pairs",
        type=parse_pairs,
        default=parse_pairs("10:50,740:950"),
        help="Comma-separated SOURCE:TARGET pairs",
    )
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
    parser.add_argument("--force-redetect", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=pipeline.DEFAULT_CHECKPOINT)
    parser.add_argument("--normal-module", type=Path, default=None)
    parser.add_argument("--kinect-fx", type=float, default=585.0)
    parser.add_argument("--kinect-fy", type=float, default=585.0)
    parser.add_argument("--kinect-cx", type=float, default=320.0)
    parser.add_argument("--kinect-cy", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=7)
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
    )
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Counts, radii, distances, and iteration limits must be positive")
    if not 0.0 < args.ratio_threshold <= 1.0:
        raise ValueError("--ratio-threshold must be in (0,1]")


def spatial_fps_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic spatial cap applied identically to every detector."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    selected = np.empty(count, dtype=np.int64)
    center = points.mean(axis=0)
    selected[0] = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    minimum_squared = np.sum((points - points[selected[0]]) ** 2, axis=1)
    minimum_squared[selected[0]] = -1.0
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared))
        candidate_squared = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, candidate_squared)
        minimum_squared[selected[: index + 1]] = -1.0
    return selected


def detect_keypoints(
    method: str,
    sequence: Path,
    frame_index: int,
    cloud: np.ndarray,
    recipe: pipeline.DetectorRecipe,
    config: Any,
    normal_estimator: Any,
    args: argparse.Namespace,
) -> Detection:
    started = time.perf_counter()
    cache_hit: bool | None = None
    if method == "fps_projection":
        cache_path = pipeline.detector_cache_path(
            sequence, frame_index, recipe, args.fps_count
        )
        cache_hit = cache_path.is_file() and not args.force_redetect
        keypoints, _ = pipeline.detect_keypoints(
            sequence,
            frame_index,
            cloud,
            config,
            recipe,
            args.fps_count,
            normal_estimator,
            args.force_redetect,
        )
    elif method == "iss3d":
        keypoints = pcl.iss3d(
            np.asarray(cloud, dtype=np.float32),
            args.iss_salient_radius_m,
            args.iss_nonmax_radius_m,
            gamma21=0.975,
            gamma32=0.975,
            min_neighbors=5,
            threads=1,
        )
    elif method == "harris3d":
        keypoints, response = pcl.harris3d(
            np.asarray(cloud, dtype=np.float32),
            args.harris_radius_m,
            threshold=0.0,
            nonmax=True,
            refine=False,
            method="HARRIS",
            threads=1,
        )
        response = np.asarray(response)
        keep = np.isfinite(response) & (response > 0.0)
        keypoints = np.asarray(keypoints)[keep]
    else:
        raise ValueError(f"Unknown detector: {method}")

    keypoints = np.asarray(keypoints, dtype=np.float64)
    keypoints = keypoints[np.isfinite(keypoints).all(axis=1)]
    raw_count = len(keypoints)
    if raw_count < 3:
        raise RuntimeError(f"{LABELS[method]} found only {raw_count} keypoints")
    keypoints = keypoints[spatial_fps_indices(keypoints, args.keypoint_budget)]
    return Detection(keypoints, raw_count, time.perf_counter() - started, cache_hit)


def strict_mutual_matches(
    source: np.ndarray,
    target: np.ndarray,
    ratio_threshold: float,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual top-1 + bidirectional ratio test, with no automatic relaxation."""
    if len(source) < 2 or len(target) < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    distances = np.sqrt(
        np.maximum(2.0 - 2.0 * np.clip(source @ target.T, -1.0, 1.0), 0.0)
    )
    source_order = np.argsort(distances, axis=1)
    target_order = np.argsort(distances, axis=0)
    source_best = source_order[:, 0]
    target_best = target_order[0, :]
    source_ratio = distances[np.arange(len(source)), source_best] / (
        distances[np.arange(len(source)), source_order[:, 1]] + 1e-12
    )
    target_ratio = distances[target_best, np.arange(len(target))] / (
        distances[target_order[1, :], np.arange(len(target))] + 1e-12
    )
    source_ids = np.arange(len(source), dtype=np.int64)
    target_ids = source_best
    keep = target_best[target_ids] == source_ids
    keep &= source_ratio <= ratio_threshold
    keep &= target_ratio[target_ids] <= ratio_threshold
    keep &= distances[source_ids, target_ids] <= max_distance
    values = distances[source_ids[keep], target_ids[keep]]
    order = np.argsort(values)
    return source_ids[keep][order], target_ids[keep][order], values[order]


def symmetric_repeatability(
    source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float
) -> float:
    source_in_target = pipeline.transform_points(source, transform)
    target_in_source = pipeline.transform_points(target, np.linalg.inv(transform))
    source_distances = cKDTree(target).query(source_in_target, k=1)[0]
    target_distances = cKDTree(source).query(target_in_source, k=1)[0]
    repeatable = np.count_nonzero(source_distances <= threshold)
    repeatable += np.count_nonzero(target_distances <= threshold)
    return float(repeatable / (len(source) + len(target)))


def ransac_triplets(count: int, maximum: int, seed: int) -> tuple[Iterable[tuple[int, int, int]], int]:
    """Enumerate all triples when cheap, otherwise return a fixed random subset."""
    total = math.comb(count, 3)
    if total <= maximum:
        return itertools.combinations(range(count), 3), total
    rng = np.random.default_rng(seed)
    samples: set[tuple[int, int, int]] = set()
    while len(samples) < maximum:
        samples.add(tuple(sorted(rng.choice(count, 3, replace=False).tolist())))
    ordered = sorted(samples)
    return iter(ordered), len(ordered)


def deterministic_ransac(
    source_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    source_match_ids: np.ndarray,
    target_match_ids: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    distance_threshold: float,
    maximum_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic correspondence RANSAC followed by all-inlier Kabsch."""
    matched_source = source_keypoints[source_match_ids]
    matched_target = target_keypoints[target_match_ids]
    match_count = len(matched_source)
    if match_count < 3:
        raise RuntimeError(f"Only {match_count} strict mutual matches")

    triplets, trial_count = ransac_triplets(match_count, maximum_trials, seed)
    best_count = -1
    best_rmse = math.inf
    best_transform: np.ndarray | None = None
    evaluated_hypotheses = 0
    started = time.perf_counter()
    edge_pairs_a = np.array((0, 0, 1))
    edge_pairs_b = np.array((1, 2, 2))
    for ids in triplets:
        ids_array = np.asarray(ids, dtype=np.int64)
        source_sample = matched_source[ids_array]
        target_sample = matched_target[ids_array]
        source_edges = np.linalg.norm(
            source_sample[edge_pairs_a] - source_sample[edge_pairs_b], axis=1
        )
        target_edges = np.linalg.norm(
            target_sample[edge_pairs_a] - target_sample[edge_pairs_b], axis=1
        )
        edge_ratio = np.minimum(source_edges, target_edges) / (
            np.maximum(source_edges, target_edges) + 1e-12
        )
        if np.any(edge_ratio < 0.8):
            continue
        if (
            np.linalg.norm(
                np.cross(source_sample[1] - source_sample[0], source_sample[2] - source_sample[0])
            )
            < 1e-8
            or np.linalg.norm(
                np.cross(target_sample[1] - target_sample[0], target_sample[2] - target_sample[0])
            )
            < 1e-8
        ):
            continue
        transform = pipeline.fit_rigid_transform(source_sample, target_sample)
        residuals = np.linalg.norm(
            pipeline.transform_points(matched_source, transform) - matched_target,
            axis=1,
        )
        inliers = residuals <= distance_threshold
        inlier_count = int(np.count_nonzero(inliers))
        rmse = (
            math.inf
            if inlier_count == 0
            else float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        )
        evaluated_hypotheses += 1
        if inlier_count > best_count or (
            inlier_count == best_count and rmse < best_rmse
        ):
            best_count = inlier_count
            best_rmse = rmse
            best_transform = transform

    required_inliers = max(6, int(math.ceil(0.10 * match_count)))
    if best_transform is None or best_count < required_inliers:
        raise RuntimeError(
            f"RANSAC found only {max(0, best_count)} inlier descriptor matches; "
            f"need at least {required_inliers}"
        )

    transform = best_transform
    for _ in range(3):
        residuals = np.linalg.norm(
            pipeline.transform_points(matched_source, transform) - matched_target,
            axis=1,
        )
        inliers = residuals <= distance_threshold
        if int(np.count_nonzero(inliers)) < 3:
            break
        transform = pipeline.fit_rigid_transform(
            matched_source[inliers], matched_target[inliers]
        )
    residuals = np.linalg.norm(
        pipeline.transform_points(matched_source, transform) - matched_target,
        axis=1,
    )
    inliers = residuals <= distance_threshold
    final_inlier_count = int(np.count_nonzero(inliers))
    if final_inlier_count < required_inliers:
        raise RuntimeError(
            f"RANSAC refit retained only {final_inlier_count} inliers; "
            f"need at least {required_inliers}"
        )

    evaluation = pipeline.o3d.pipelines.registration.evaluate_registration(
        pipeline.make_point_cloud(source_cloud),
        pipeline.make_point_cloud(target_cloud),
        0.030,
        transform,
    )
    return {
        "transform": transform,
        "residuals": residuals,
        "inliers": inliers,
        "ransac_time_s": time.perf_counter() - started,
        "ransac_trials": trial_count,
        "ransac_hypotheses_evaluated": evaluated_hypotheses,
        "final_dense_fitness": float(evaluation.fitness),
        "final_dense_rmse_m": float(evaluation.inlier_rmse),
    }


def benchmark_detector(
    method: str,
    sequence: Path,
    source_frame: int,
    target_frame: int,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    gt_transform: np.ndarray | None,
    recipe: pipeline.DetectorRecipe,
    config: Any,
    normal_estimator: Any,
    model: pipeline.MiniSpinNet,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pair": f"{source_frame}->{target_frame}",
        "source_frame": source_frame,
        "target_frame": target_frame,
        "detector": method,
        "detector_label": LABELS[method],
        "status": "failed",
        "error": None,
    }
    try:
        source_detection = detect_keypoints(
            method,
            sequence,
            source_frame,
            source_cloud,
            recipe,
            config,
            normal_estimator,
            args,
        )
        target_detection = detect_keypoints(
            method,
            sequence,
            target_frame,
            target_cloud,
            recipe,
            config,
            normal_estimator,
            args,
        )
        source_keys = source_detection.keypoints
        target_keys = target_detection.keypoints
        result.update(
            {
                "source_keypoints_raw": source_detection.raw_count,
                "target_keypoints_raw": target_detection.raw_count,
                "source_keypoints_budgeted": len(source_keys),
                "target_keypoints_budgeted": len(target_keys),
                "detection_time_s": source_detection.elapsed_s
                + target_detection.elapsed_s,
                "source_detection_cache_hit": source_detection.cache_hit,
                "target_detection_cache_hit": target_detection.cache_hit,
            }
        )

        started = time.perf_counter()
        source_descriptors, source_valid = pipeline.describe_keypoints(
            model,
            source_cloud,
            source_keys,
            args.descriptor_radius_m,
            args.descriptor_batch_size,
            device,
            args.seed + source_frame,
        )
        target_descriptors, target_valid = pipeline.describe_keypoints(
            model,
            target_cloud,
            target_keys,
            args.descriptor_radius_m,
            args.descriptor_batch_size,
            device,
            args.seed + target_frame,
        )
        source_keys = source_keys[source_valid]
        target_keys = target_keys[target_valid]
        result["descriptor_time_s"] = time.perf_counter() - started
        result["source_keypoints_described"] = len(source_keys)
        result["target_keypoints_described"] = len(target_keys)

        started = time.perf_counter()
        source_ids, target_ids, descriptor_distances = strict_mutual_matches(
            source_descriptors,
            target_descriptors,
            args.ratio_threshold,
            args.max_descriptor_distance,
        )
        result["matching_time_s"] = time.perf_counter() - started
        result["mutual_matches"] = len(source_ids)
        result["descriptor_distance_mean"] = (
            None if len(descriptor_distances) == 0 else float(np.mean(descriptor_distances))
        )

        if gt_transform is not None:
            for distance_mm in (30, 50, 100):
                result[f"repeatability_{distance_mm}mm"] = symmetric_repeatability(
                    source_keys, target_keys, gt_transform, distance_mm / 1000.0
                )
            if len(source_ids):
                gt_residuals = np.linalg.norm(
                    pipeline.transform_points(source_keys[source_ids], gt_transform)
                    - target_keys[target_ids],
                    axis=1,
                )
                gt_correct = int(np.count_nonzero(gt_residuals <= args.ransac_distance_m))
            else:
                gt_correct = 0
            result["gt_correct_matches"] = gt_correct
            result["gt_match_precision"] = (
                None if len(source_ids) == 0 else gt_correct / len(source_ids)
            )

        if len(source_ids) < 3:
            raise RuntimeError(f"Only {len(source_ids)} strict mutual matches")
        registration = deterministic_ransac(
            source_keys,
            target_keys,
            source_ids,
            target_ids,
            source_cloud,
            target_cloud,
            args.ransac_distance_m,
            args.ransac_iterations,
            args.seed,
        )
        transform = registration["transform"]
        inlier_count = int(np.count_nonzero(registration["inliers"]))
        inlier_rmse = (
            None
            if inlier_count == 0
            else float(
                np.sqrt(np.mean(registration["residuals"][registration["inliers"]] ** 2))
            )
        )
        result.update(
            {
                "status": "ok",
                "ransac_inliers": inlier_count,
                "ransac_inlier_ratio": inlier_count / len(source_ids),
                "ransac_inlier_rmse_m": inlier_rmse,
                "ransac_time_s": registration["ransac_time_s"],
                "ransac_trials": registration["ransac_trials"],
                "ransac_hypotheses_evaluated": registration[
                    "ransac_hypotheses_evaluated"
                ],
                "dense_fitness_30mm": registration["final_dense_fitness"],
                "dense_rmse_30mm_m": registration["final_dense_rmse_m"],
                "transform_source_to_target": transform.tolist(),
            }
        )
        if gt_transform is not None:
            translation_error, rotation_error = pipeline.transform_error(
                transform, gt_transform
            )
            result["translation_error_m"] = translation_error
            result["rotation_error_deg"] = rotation_error
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def console_table(results: list[dict[str, Any]]) -> None:
    pipeline.log("\n[RESULTS] Mini-SpinNet + strict mutual top-1 + RANSAC only; ICP=OFF")
    pipeline.log(
        f"{'pair':>10}  {'detector':<15} {'kp S/T':>9} {'match':>6} {'inlier':>7} "
        f"{'GT-match':>9} {'RTE mm':>9} {'RRE deg':>9} {'status':>8}"
    )
    for row in results:
        keys = f"{row.get('source_keypoints_described', 0)}/{row.get('target_keypoints_described', 0)}"
        gt_precision = row.get("gt_match_precision")
        gt_text = "--" if gt_precision is None else f"{100.0 * gt_precision:.1f}%"
        rte = row.get("translation_error_m")
        rre = row.get("rotation_error_deg")
        pipeline.log(
            f"{row['pair']:>10}  {row['detector_label']:<15} {keys:>9} "
            f"{row.get('mutual_matches', 0):>6} {row.get('ransac_inliers', 0):>7} "
            f"{gt_text:>9} "
            f"{('--' if rte is None else f'{1000.0 * rte:.1f}'):>9} "
            f"{('--' if rre is None else f'{rre:.2f}'):>9} {row['status']:>8}"
        )
        if row["error"]:
            pipeline.log(f"{'':>12}  -> {row['error']}")


def save_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "pair",
        "detector",
        "status",
        "source_keypoints_raw",
        "target_keypoints_raw",
        "source_keypoints_described",
        "target_keypoints_described",
        "detection_time_s",
        "descriptor_time_s",
        "matching_time_s",
        "mutual_matches",
        "gt_correct_matches",
        "gt_match_precision",
        "repeatability_30mm",
        "repeatability_50mm",
        "repeatability_100mm",
        "ransac_inliers",
        "ransac_inlier_ratio",
        "ransac_inlier_rmse_m",
        "ransac_time_s",
        "ransac_trials",
        "ransac_hypotheses_evaluated",
        "dense_fitness_30mm",
        "translation_error_m",
        "rotation_error_deg",
        "source_detection_cache_hit",
        "target_detection_cache_hit",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def save_chart(path: Path, pairs: list[tuple[int, int]], results: list[dict[str, Any]]) -> None:
    pair_labels = [f"{source}->{target}" for source, target in pairs]
    lookup = {(row["pair"], row["detector"]): row for row in results}
    metrics = (
        ("repeatability_50mm", "GT keypoint repeatability @ 50 mm", 100.0, "%"),
        ("ransac_inliers", "RANSAC inliers", 1.0, "count"),
        ("translation_error_m", "GT translation error", 1000.0, "mm"),
        ("rotation_error_deg", "GT rotation error", 1.0, "deg"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    x = np.arange(len(pair_labels), dtype=np.float64)
    width = 0.24
    for axis, (field, title, scale, ylabel) in zip(axes.flat, metrics):
        for method_index, method in enumerate(METHODS):
            values = []
            for pair in pair_labels:
                value = lookup[(pair, method)].get(field)
                values.append(np.nan if value is None else float(value) * scale)
            axis.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=LABELS[method],
                color=COLORS[method],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, pair_labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    figure.suptitle(
        "Detector comparison | same 3-D keypoint budget + Mini-SpinNet + RANSAC | NO ICP",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


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
    using_default_redkitchen = (
        sequence.resolve() == DEFAULT_REDKITCHEN_SEQUENCE.resolve()
    )
    recipe = pipeline.DetectorRecipe()
    config = pipeline.make_detector_config(recipe)
    normal_estimator = pipeline.detector.load_normal_estimator(args.normal_module)
    checkpoint = pipeline.ensure_checkpoint(args.checkpoint)
    model = pipeline.load_minispinnet(checkpoint, device)
    pipeline.log(f"[SOURCE] {sequence}")
    pipeline.log(f"[DEVICE] {device}")
    pipeline.log(
        "[PROTOCOL] detector changes only; budget="
        f"{args.keypoint_budget}; Mini-SpinNet + strict mutual top-1 + RANSAC; ICP=OFF"
    )

    frames = sorted({frame for pair in args.pairs for frame in pair})
    clouds: dict[int, np.ndarray] = {}
    poses: dict[int, np.ndarray | None] = {}
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
        poses[frame] = pipeline.read_pose(sequence, frame)

    pair_metadata: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for source_frame, target_frame in args.pairs:
        source_pose = poses[source_frame]
        target_pose = poses[target_frame]
        gt_transform = (
            None
            if source_pose is None or target_pose is None
            else np.linalg.inv(target_pose) @ source_pose
        )
        metadata: dict[str, Any] = {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "ground_truth_source_to_target": (
                None if gt_transform is None else gt_transform.tolist()
            ),
        }
        if gt_transform is not None:
            delta_m, delta_deg = pipeline.transform_error(gt_transform, np.eye(4))
            metadata["gt_camera_delta_translation_m"] = delta_m
            metadata["gt_camera_delta_rotation_deg"] = delta_deg
            pipeline.log(
                f"\n[PAIR {source_frame}->{target_frame}] GT camera delta="
                f"{delta_m:.4f} m / {delta_deg:.3f} deg (evaluation only)"
            )
        pair_metadata.append(metadata)
        for method in METHODS:
            pipeline.log(f"[BENCH {source_frame}->{target_frame}] {LABELS[method]}")
            results.append(
                benchmark_detector(
                    method,
                    sequence,
                    source_frame,
                    target_frame,
                    clouds[source_frame],
                    clouds[target_frame],
                    gt_transform,
                    recipe,
                    config,
                    normal_estimator,
                    model,
                    device,
                    args,
                )
            )

    console_table(results)
    if args.output_dir is None:
        pair_tag = "__".join(f"{source:06d}-{target:06d}" for source, target in args.pairs)
        output_dir = SCRIPT_DIR / "detector_benchmark_results" / f"{sequence.stem}_{pair_tag}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "detector_benchmark.json"
    csv_path = output_dir / "detector_benchmark.csv"
    chart_path = output_dir / "detector_benchmark.png"
    payload = {
        "protocol": {
            "evaluation_dataset": (
                "Microsoft 7-Scenes RedKitchen seq-03"
                if using_default_redkitchen
                else str(sequence)
            ),
            "split_status": (
                "7-scenes-redkitchen is a 3DMatch test scene; seq-03 is in the "
                "original Microsoft 7-Scenes TestSplit"
                if using_default_redkitchen
                else "User-supplied dataset; verify its train/test provenance separately"
            ),
            "detectors": list(METHODS),
            "detector_labels": LABELS,
            "common_spatial_keypoint_budget": args.keypoint_budget,
            "common_cloud_voxel_m": recipe.voxel_3d_mm / 1000.0,
            "descriptor": "BUFFER-X pretrained Mini-SpinNet 32-D",
            "descriptor_checkpoint_sha256": pipeline.CHECKPOINT_SHA256,
            "matching": "strict bidirectional mutual top-1 + bidirectional ratio test",
            "matching_relaxation": False,
            "ransac_distance_m": args.ransac_distance_m,
            "ransac_iterations": args.ransac_iterations,
            "pose_solver": "deterministic exhaustive-or-capped triple RANSAC + Kabsch",
            "icp_used": False,
            "temporal_bridge_used": False,
            "fps_count": args.fps_count,
            "iss_salient_radius_m": args.iss_salient_radius_m,
            "iss_nonmax_radius_m": args.iss_nonmax_radius_m,
            "harris_radius_m": args.harris_radius_m,
            "detector_recipe": asdict(recipe),
        },
        "sequence": str(sequence),
        "data_provenance": (
            {
                "official_outer_archive_url": REDKITCHEN_SOURCE_URL,
                "selected_nested_archive": "redkitchen/seq-03.zip",
                "selected_archive_sha256": REDKITCHEN_SEQ03_SHA256,
                "selected_archive_size_bytes": 503_831_078,
            }
            if using_default_redkitchen
            else None
        ),
        "pairs": pair_metadata,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_csv(csv_path, results)
    save_chart(chart_path, args.pairs, results)
    pipeline.log(f"\n[OUTPUT JSON] {json_path}")
    pipeline.log(f"[OUTPUT CSV] {csv_path}")
    pipeline.log(f"[OUTPUT CHART] {chart_path}")
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
