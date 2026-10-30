"""Synthetic ground-truth benchmark for single- and multi-scale Normal Cones.

The benchmark creates finite one-, two-, and three-face objects with exact
vertex coordinates, plus negative scenes containing missing observations,
sensor-like protrusions, and smooth curved surfaces.  It reports one-to-one
corner matching metrics at a configurable localization tolerance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from normal_cone_corner_detector import (
    GeometryScaleConfig,
    MultiScaleConfig,
    NormalConeConfig,
    PhaseCConfig,
    SceneDetectionResult,
    build_scene,
    detect_scene,
    detect_scene_geometry_adaptive,
    detect_scene_multiscale,
    verify_candidates_phase_c,
)


@dataclass(frozen=True)
class SyntheticGTScene:
    name: str
    points_mm: np.ndarray
    corners_mm: np.ndarray
    evaluation_min_mm: np.ndarray
    evaluation_max_mm: np.ndarray


@dataclass(frozen=True)
class MatchMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    localization_errors_mm: np.ndarray

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def mean_error_mm(self) -> float:
        if len(self.localization_errors_mm) == 0:
            return float("nan")
        return float(np.mean(self.localization_errors_mm))


def _unique(points: np.ndarray) -> np.ndarray:
    return np.unique(np.round(np.asarray(points, dtype=np.float64), 8), axis=0)


def _square_xy(length: float, spacing: float) -> np.ndarray:
    axis = np.arange(0.0, length + 0.25 * spacing, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def _perturb(
    points: np.ndarray,
    *,
    noise_sigma_mm: float,
    dropout: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kept = rng.random(len(points)) >= dropout
    noisy = points[kept].copy()
    if noise_sigma_mm > 0.0:
        noisy += rng.normal(0.0, noise_sigma_mm, noisy.shape)
    return noisy


def make_synthetic_gt_scenes(
    *, length_mm: float = 60.0, spacing_mm: float = 2.0
) -> list[SyntheticGTScene]:
    """Return deterministic scenes with exact physical-corner GT."""
    length = float(length_mm)
    spacing = float(spacing_mm)
    box_min = np.array([-2.0, -2.0, -2.0])
    box_max = np.array([length + 2.0] * 3)

    plate = _square_xy(length, spacing)
    plate_gt = np.array(
        [[x, y, 0.0] for x in (0.0, length) for y in (0.0, length)]
    )

    axis = np.arange(0.0, length + 0.25 * spacing, spacing)
    a, b = np.meshgrid(axis, axis, indexing="ij")
    xy = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
    xz = np.column_stack((a.ravel(), np.zeros(a.size), b.ravel()))
    yz = np.column_stack((np.zeros(a.size), a.ravel(), b.ravel()))
    wedge = _unique(np.vstack((xy, xz)))
    wedge_gt = _unique(
        np.array(
            [
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [length, length, 0],
                [0, 0, length],
                [length, 0, length],
            ],
            dtype=float,
        )
    )
    trihedral = _unique(np.vstack((xy, xz, yz)))
    trihedral_gt = np.array(
        [
            [x, y, z]
            for x in (0.0, length)
            for y in (0.0, length)
            for z in (0.0, length)
            if not (x == length and y == length and z == length)
        ]
    )

    rounded_plate = plate.copy()
    corner_radius = 14.0
    rounded_keep = np.ones(len(rounded_plate), dtype=bool)
    for corner_x, corner_y in (
        (0.0, 0.0),
        (0.0, length),
        (length, 0.0),
        (length, length),
    ):
        center_x = corner_radius if corner_x == 0.0 else length - corner_radius
        center_y = corner_radius if corner_y == 0.0 else length - corner_radius
        in_corner_box = (
            (np.abs(rounded_plate[:, 0] - corner_x) <= corner_radius)
            & (np.abs(rounded_plate[:, 1] - corner_y) <= corner_radius)
        )
        inside_arc = (
            (rounded_plate[:, 0] - center_x) ** 2
            + (rounded_plate[:, 1] - center_y) ** 2
            <= corner_radius**2 + 1e-9
        )
        rounded_keep &= ~in_corner_box | inside_arc
    rounded_plate = rounded_plate[rounded_keep]

    hole_plate = _square_xy(80.0, spacing)
    hole = (
        (hole_plate[:, 0] >= 26.0)
        & (hole_plate[:, 0] <= 54.0)
        & (hole_plate[:, 1] >= 26.0)
        & (hole_plate[:, 1] <= 54.0)
    )
    hole_plate = hole_plate[~hole]
    hole_gt = np.array(
        [[x, y, 0.0] for x in (0.0, 80.0) for y in (0.0, 80.0)]
    )

    signed = np.arange(-70.0, 70.0 + 0.25 * spacing, spacing)
    px, py = np.meshgrid(signed, signed, indexing="ij")
    noisy_plane = np.column_stack((px.ravel(), py.ravel(), np.zeros(px.size)))
    rng = np.random.default_rng(91)
    noisy_plane[:, 2] += rng.normal(0.0, 0.30, len(noisy_plane))
    cluster_centers = np.array(
        [[-32, -20, 5], [-12, 24, 7], [18, -28, 6], [34, 20, 8]],
        dtype=float,
    )
    islands = np.vstack(
        [center + rng.normal(0.0, [2.0, 2.0, 1.3], (18, 3)) for center in cluster_centers]
    )
    noisy_plane = np.vstack((noisy_plane, islands))

    curve_axis = np.arange(-60.0, 60.0 + 0.25 * spacing, spacing)
    cx, cy = np.meshgrid(curve_axis, curve_axis, indexing="ij")
    sphere_radius = 120.0
    keep = cx.ravel() ** 2 + cy.ravel() ** 2 < sphere_radius**2
    sx, sy = cx.ravel()[keep], cy.ravel()[keep]
    sz = -sphere_radius + np.sqrt(sphere_radius**2 - sx**2 - sy**2)
    sphere = np.column_stack((sx, sy, sz))
    cylinder = np.column_stack(
        (
            cx.ravel(),
            cy.ravel(),
            -sphere_radius + np.sqrt(sphere_radius**2 - cx.ravel() ** 2),
        )
    )

    # Targeted scale-contamination scenes.  Their evaluation regions isolate
    # the origin so that a larger neighborhood can ingest another structure
    # without making that structure's own corners part of the local GT.
    wall_y, wall_z = np.meshgrid(
        np.arange(-30.0, 30.0 + 0.25 * spacing, spacing),
        np.arange(-30.0, 30.0 + 0.25 * spacing, spacing),
        indexing="ij",
    )
    nearby_wall = np.column_stack(
        (np.full(wall_y.size, 18.0), wall_y.ravel(), wall_z.ravel())
    )
    corner_near_wall = _unique(np.vstack((plate, nearby_wall)))

    elevated_x, elevated_y = np.meshgrid(
        np.arange(0.0, 36.0 + 0.25 * spacing, spacing),
        np.arange(0.0, 36.0 + 0.25 * spacing, spacing),
        indexing="ij",
    )
    elevated_plate = np.column_stack(
        (
            elevated_x.ravel(),
            elevated_y.ravel(),
            np.full(elevated_x.size, 22.0),
        )
    )
    corner_near_elevated = _unique(np.vstack((plate, elevated_plate)))

    signed_wall_y, signed_wall_z = np.meshgrid(
        np.arange(-60.0, 60.0 + 0.25 * spacing, spacing),
        np.arange(0.0, 40.0 + 0.25 * spacing, spacing),
        indexing="ij",
    )
    signed_wall = np.column_stack(
        (
            np.full(signed_wall_y.size, 22.0),
            signed_wall_y.ravel(),
            signed_wall_z.ravel(),
        )
    )
    clean_signed_plane = np.column_stack(
        (px.ravel(), py.ravel(), np.zeros(px.size, dtype=float))
    )
    plane_near_wall = _unique(np.vstack((clean_signed_plane, signed_wall)))

    empty_gt = np.empty((0, 3), dtype=float)
    return [
        SyntheticGTScene("plate_clean", plate, plate_gt, box_min, box_max),
        SyntheticGTScene(
            "plate_noisy_dropout",
            _perturb(plate, noise_sigma_mm=0.35, dropout=0.20, seed=11),
            plate_gt,
            box_min,
            box_max,
        ),
        SyntheticGTScene(
            "two_face_noisy",
            _perturb(wedge, noise_sigma_mm=0.30, dropout=0.12, seed=23),
            wedge_gt,
            box_min,
            box_max,
        ),
        SyntheticGTScene(
            "three_face_noisy",
            _perturb(trihedral, noise_sigma_mm=0.30, dropout=0.12, seed=37),
            trihedral_gt,
            box_min,
            box_max,
        ),
        SyntheticGTScene(
            "rounded_plate_virtual_gt",
            rounded_plate,
            plate_gt,
            box_min,
            box_max,
        ),
        SyntheticGTScene(
            "missing_patch",
            hole_plate,
            hole_gt,
            np.array([-2.0, -2.0, -2.0]),
            np.array([82.0, 82.0, 2.0]),
        ),
        SyntheticGTScene(
            "sensor_outlier_plane",
            noisy_plane,
            empty_gt,
            np.array([-50.0, -50.0, -15.0]),
            np.array([50.0, 50.0, 15.0]),
        ),
        SyntheticGTScene(
            "sphere_patch",
            sphere,
            empty_gt,
            np.array([-40.0, -40.0, -30.0]),
            np.array([40.0, 40.0, 5.0]),
        ),
        SyntheticGTScene(
            "cylinder_patch",
            cylinder,
            empty_gt,
            np.array([-40.0, -40.0, -30.0]),
            np.array([40.0, 40.0, 5.0]),
        ),
        SyntheticGTScene(
            "corner_near_perpendicular_plane",
            corner_near_wall,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([-2.0, -2.0, -2.0]),
            np.array([8.0, 8.0, 8.0]),
        ),
        SyntheticGTScene(
            "corner_near_elevated_plate",
            corner_near_elevated,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([-2.0, -2.0, -2.0]),
            np.array([8.0, 8.0, 8.0]),
        ),
        SyntheticGTScene(
            "plane_near_perpendicular_plane",
            plane_near_wall,
            empty_gt,
            np.array([-8.0, -8.0, -3.0]),
            np.array([8.0, 8.0, 8.0]),
        ),
    ]


def points_in_evaluation_region(
    points_mm: np.ndarray, scene: SyntheticGTScene
) -> np.ndarray:
    return np.all(
        (points_mm >= scene.evaluation_min_mm)
        & (points_mm <= scene.evaluation_max_mm),
        axis=1,
    )


def match_corner_candidates(
    candidates_mm: np.ndarray,
    corners_mm: np.ndarray,
    *,
    tolerance_mm: float = 6.0,
) -> MatchMetrics:
    candidates = np.asarray(candidates_mm, dtype=float).reshape(-1, 3)
    corners = np.asarray(corners_mm, dtype=float).reshape(-1, 3)
    if len(candidates) == 0 or len(corners) == 0:
        matched = 0
        errors = np.empty(0, dtype=float)
    else:
        distances = np.linalg.norm(
            candidates[:, None, :] - corners[None, :, :], axis=2
        )
        candidate_ids, corner_ids = linear_sum_assignment(distances)
        errors = distances[candidate_ids, corner_ids]
        errors = errors[errors <= tolerance_mm]
        matched = len(errors)
    return MatchMetrics(
        true_positive=matched,
        false_positive=len(candidates) - matched,
        false_negative=len(corners) - matched,
        localization_errors_mm=errors,
    )


def _phase_c_candidates(
    scene,
    detection: SceneDetectionResult,
    config: NormalConeConfig,
    query_radius_mm_by_index: np.ndarray | None = None,
) -> np.ndarray:
    result = verify_candidates_phase_c(
        scene,
        detection,
        replace(config, query_radius_mm=30.0),
        PhaseCConfig(
            min_support=40,
            min_outer_shell_fraction=0.10,
            min_radius_p90_ratio=0.70,
            min_centroid_offset_ratio=0.38,
            min_directional_concentration=0.65,
            repeatability_trials=3,
            min_repeatability=0.50,
            min_split_score_ratio=0.40,
        ),
        query_radius_mm_by_index=query_radius_mm_by_index,
    )
    return result.accepted_indices


def run_benchmark(
    *,
    sphere_samples: int = 256,
    match_tolerance_mm: float = 6.0,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    base = NormalConeConfig(
        sphere_samples=sphere_samples,
        min_score=0.10,
        tolerance_mm=0.6,
        local_max_radius_mm=3.0,
        nms_radius_mm=6.0,
    )
    variants = (
        "single_12mm",
        "single_20mm",
        "single_30mm",
        "multi_2of3",
        "multi_3of4",
        "multi_5of7",
        "multi_3of4_phase_c",
        "geometry_adaptive",
        "geometry_adaptive_phase_c",
    )
    rows: list[dict[str, object]] = []
    candidates_by_scene: dict[tuple[str, str], np.ndarray] = {}
    for gt_scene in make_synthetic_gt_scenes():
        scene = build_scene(gt_scene.points_mm, base, downsample=True)
        detections: dict[str, SceneDetectionResult] = {}
        for radius in (12.0, 20.0, 30.0):
            config = replace(base, query_radius_mm=radius)
            detections[f"single_{int(radius)}mm"] = detect_scene(scene, config)

        multi_2of3 = detect_scene_multiscale(
            scene,
            replace(base, query_radius_mm=30.0),
            MultiScaleConfig(
                radii_mm=(12.0, 20.0, 30.0),
                min_scale_count=2,
                min_consistency=0.30,
            ),
        ).as_scene_detection()
        multi_3of4 = detect_scene_multiscale(
            scene,
            replace(base, query_radius_mm=30.0),
            MultiScaleConfig(
                radii_mm=(12.0, 18.0, 24.0, 30.0),
                min_scale_count=3,
                min_consistency=0.35,
            ),
        ).as_scene_detection()
        detections["multi_2of3"] = multi_2of3
        detections["multi_3of4"] = multi_3of4

        geometry_multiscale = detect_scene_multiscale(
            scene,
            replace(base, query_radius_mm=32.0),
            MultiScaleConfig(
                radii_mm=(8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0),
                min_scale_count=5,
                min_consistency=0.35,
            ),
        )
        detections["multi_5of7"] = geometry_multiscale.as_scene_detection()
        geometry_result = detect_scene_geometry_adaptive(
            scene,
            replace(base, query_radius_mm=32.0),
            geometry_multiscale,
            GeometryScaleConfig(
                min_clean_scale_count=3,
                min_score_consistency=0.30,
                min_shell_points=8,
            ),
        )
        detections["geometry_adaptive"] = geometry_result.as_scene_detection(
            geometry_multiscale
        )

        phase_c_ids = _phase_c_candidates(scene, multi_3of4, base)
        geometry_phase_c_ids = _phase_c_candidates(
            scene,
            detections["geometry_adaptive"],
            replace(base, query_radius_mm=32.0),
            geometry_result.selected_radius_mm,
        )
        for variant in variants:
            if variant == "multi_3of4_phase_c":
                ids = phase_c_ids
            elif variant == "geometry_adaptive_phase_c":
                ids = geometry_phase_c_ids
            else:
                ids = detections[variant].candidate_indices
            candidates = scene.points_mm[ids]
            candidates = candidates[points_in_evaluation_region(candidates, gt_scene)]
            metrics = match_corner_candidates(
                candidates,
                gt_scene.corners_mm,
                tolerance_mm=match_tolerance_mm,
            )
            candidates_by_scene[(gt_scene.name, variant)] = candidates
            rows.append(
                {
                    "scene": gt_scene.name,
                    "variant": variant,
                    "gt_count": len(gt_scene.corners_mm),
                    "candidate_count": len(candidates),
                    "tp": metrics.true_positive,
                    "fp": metrics.false_positive,
                    "fn": metrics.false_negative,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "mean_error_mm": metrics.mean_error_mm,
                }
            )
    return rows, candidates_by_scene


def aggregate_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    result = []
    for variant, group in grouped.items():
        tp = sum(int(row["tp"]) for row in group)
        fp = sum(int(row["fp"]) for row in group)
        fn = sum(int(row["fn"]) for row in group)
        errors = [
            float(row["mean_error_mm"])
            for row in group
            if np.isfinite(float(row["mean_error_mm"])) and int(row["tp"]) > 0
        ]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append(
            {
                "variant": variant,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_scene_error_mm": float(np.mean(errors)) if errors else float("nan"),
            }
        )
    return result


def save_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere-samples", type=int, default=256)
    parser.add_argument("--match-tolerance-mm", type=float, default=6.0)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, _ = run_benchmark(
        sphere_samples=args.sphere_samples,
        match_tolerance_mm=args.match_tolerance_mm,
    )
    summary = aggregate_rows(rows)
    for row in summary:
        print(
            f"{row['variant']:24s} TP={row['tp']:2d} FP={row['fp']:3d} "
            f"FN={row['fn']:2d} P={row['precision']:.3f} "
            f"R={row['recall']:.3f} F1={row['f1']:.3f} "
            f"error={row['mean_scene_error_mm']:.3f}mm"
        )
    if args.output_csv:
        save_csv(rows, args.output_csv)
        print(f"saved: {Path(args.output_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
