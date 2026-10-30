"""Controlled query-level benchmark for 3-D directional corner evidence.

This experiment is deliberately smaller than a full scene detector.  Every
synthetic patch has one fixed query at the origin, so the measured quantities
answer a narrow question: does the local geometry around that query contain a
stable set of supporting directions?

The main statistic is the robust version of

    h_r(q, u) = max_i u.T @ (p_i - q)

already used by :mod:`normal_cone_corner_detector`.  The exact maximum and the
95 % empirical quantile are kept separate.  Several diagnostic comparators are
also reported:

* exact tangent-polar membership (mathematically identical to exact support);
* a normal-cone opening/corner test inspired by Kobbelt et al.;
* a one-pass normal-tensor proxy (not a reproduction of Park et al. 2012);
* the Gaussian Harris3D normal-moment response at the query;
* ISS scatter eigenvalue ratios and saliency at the query;
* a fixed-radius Difference-of-Normals query score.

Analytic surface normals are supplied to the normal-based diagnostics.  This
is intentionally an oracle-normal comparison: it tests geometric separability
without conflating it with a particular normal estimator.  The directional
support statistic itself uses coordinates only.

Run the complete deterministic experiment with::

    python3 directional_support_benchmark.py

Generated CSV/JSON/PNG artifacts are written under
``outputs/directional_support_experiment`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata

from normal_cone_corner_detector import fibonacci_sphere


@dataclass(frozen=True)
class SyntheticPatch:
    """A local surface observation with an exact query at ``(0, 0, 0)``."""

    name: str
    points_mm: np.ndarray
    normals: np.ndarray
    desired_corner: bool
    sharp_corner: bool
    description: str


def antipodal_sphere(sample_count: int) -> np.ndarray:
    """Return deterministic directions closed under ``u -> -u``."""
    if sample_count < 16 or sample_count % 2:
        raise ValueError("sample_count must be an even integer >= 16")
    half = fibonacci_sphere(sample_count // 2)
    return np.vstack((half, -half))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=1, keepdims=True)
    output = np.full_like(values, np.nan)
    valid = np.squeeze(length > 1e-12)
    output[valid] = values[valid] / length[valid]
    return output


def _grid(low: float, high: float, spacing: float) -> np.ndarray:
    return np.arange(low, high + 0.25 * spacing, spacing, dtype=np.float64)


def _plane_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z: float,
    normal: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(x_values, y_values, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
    normals = np.tile(np.asarray(normal, dtype=np.float64), (len(points), 1))
    return points, normals


def _plane_xz(
    x_values: np.ndarray,
    z_values: np.ndarray,
    y: float,
    normal: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    x, z = np.meshgrid(x_values, z_values, indexing="ij")
    points = np.column_stack((x.ravel(), np.full(x.size, y), z.ravel()))
    normals = np.tile(np.asarray(normal, dtype=np.float64), (len(points), 1))
    return points, normals


def _plane_yz(
    y_values: np.ndarray,
    z_values: np.ndarray,
    x: float,
    normal: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    y, z = np.meshgrid(y_values, z_values, indexing="ij")
    points = np.column_stack((np.full(y.size, x), y.ravel(), z.ravel()))
    normals = np.tile(np.asarray(normal, dtype=np.float64), (len(points), 1))
    return points, normals


def _stack(parts: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(parts)
    return np.vstack([item[0] for item in pairs]), np.vstack([item[1] for item in pairs])


def _rounded_trihedral(
    *, fillet_radius_mm: float, extent_mm: float, spacing_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rounded positive-octant box vertex.

    The patch contains the spherical vertex fillet, the three quarter-cylinder
    edge fillets, and the three planar faces.  The query is the point on the
    spherical fillet whose normal is ``(-1,-1,-1)/sqrt(3)``.
    """
    radius = float(fillet_radius_mm)
    center = np.full(3, radius)
    query_normal = -np.ones(3) / math.sqrt(3.0)
    query = center + radius * query_normal

    direction_count = max(4096, int(24.0 * math.pi * radius * radius / spacing_mm**2))
    sphere_directions = fibonacci_sphere(direction_count)
    sphere_directions = sphere_directions[np.all(sphere_directions <= 1e-12, axis=1)]
    sphere_points = center + radius * sphere_directions

    angle_count = max(8, int(math.ceil(0.5 * math.pi * radius / spacing_mm)) + 1)
    angle = np.linspace(0.0, 0.5 * math.pi, angle_count)
    first = -np.cos(angle)
    second = -np.sin(angle)
    longitudinal = _grid(radius, radius + extent_mm, spacing_mm)
    cylinders: list[tuple[np.ndarray, np.ndarray]] = []
    for axis in range(3):
        points_list = []
        normals_list = []
        for value in longitudinal:
            normal = np.zeros((angle_count, 3), dtype=np.float64)
            other = [index for index in range(3) if index != axis]
            normal[:, other[0]] = first
            normal[:, other[1]] = second
            point = center + radius * normal
            point[:, axis] = value
            points_list.append(point)
            normals_list.append(normal)
        cylinders.append((np.vstack(points_list), np.vstack(normals_list)))

    face_axis = _grid(radius, radius + extent_mm, spacing_mm)
    faces = (
        _plane_yz(face_axis, face_axis, 0.0, (-1.0, 0.0, 0.0)),
        _plane_xz(face_axis, face_axis, 0.0, (0.0, -1.0, 0.0)),
        _plane_xy(face_axis, face_axis, 0.0, (0.0, 0.0, -1.0)),
    )
    points, normals = _stack(
        [(sphere_points, sphere_directions), *cylinders, *faces]
    )
    points -= query
    return points, normals


def make_synthetic_patches(
    *, extent_mm: float = 40.0, spacing_mm: float = 2.0
) -> list[SyntheticPatch]:
    """Build the required face/edge/corner/curve/boundary counterexamples."""
    signed = _grid(-extent_mm, extent_mm, spacing_mm)
    positive = _grid(0.0, extent_mm, spacing_mm)

    face_points, face_normals = _plane_xy(
        signed, signed, 0.0, (0.0, 0.0, 1.0)
    )
    face_disk = np.linalg.norm(face_points[:, :2], axis=1) <= extent_mm + 1e-9
    face_points, face_normals = face_points[face_disk], face_normals[face_disk]

    boundary_points, boundary_normals = _plane_xy(
        positive, signed, 0.0, (0.0, 0.0, 1.0)
    )
    crop_corner_points, crop_corner_normals = _plane_xy(
        positive, positive, 0.0, (0.0, 0.0, 1.0)
    )

    edge_points, edge_normals = _stack(
        (
            _plane_xy(positive, signed, 0.0, (0.0, 0.0, -1.0)),
            _plane_yz(signed, positive, 0.0, (-1.0, 0.0, 0.0)),
        )
    )

    sharp_points, sharp_normals = _stack(
        (
            _plane_xy(positive, positive, 0.0, (0.0, 0.0, -1.0)),
            _plane_xz(positive, positive, 0.0, (0.0, -1.0, 0.0)),
            _plane_yz(positive, positive, 0.0, (-1.0, 0.0, 0.0)),
        )
    )

    rounded_points, rounded_normals = _rounded_trihedral(
        fillet_radius_mm=14.0,
        extent_mm=extent_mm,
        spacing_mm=spacing_mm,
    )

    sphere_radius = 70.0
    sx, sy = np.meshgrid(signed, signed, indexing="ij")
    keep = sx.ravel() ** 2 + sy.ravel() ** 2 <= extent_mm**2
    sx_flat = sx.ravel()[keep]
    sy_flat = sy.ravel()[keep]
    sz_flat = np.sqrt(sphere_radius**2 - sx_flat**2 - sy_flat**2) - sphere_radius
    sphere_points = np.column_stack((sx_flat, sy_flat, sz_flat))
    sphere_normals = _normalize_rows(
        np.column_stack((sx_flat, sy_flat, sz_flat + sphere_radius))
    )

    bump_period = 24.0
    bump_amplitude = 4.0
    bx, by = np.meshgrid(signed, signed, indexing="ij")
    phase_x = 2.0 * math.pi * bx.ravel() / bump_period
    phase_y = 2.0 * math.pi * by.ravel() / bump_period
    bz = bump_amplitude * (np.cos(phase_x) * np.cos(phase_y) - 1.0)
    bump_points = np.column_stack((bx.ravel(), by.ravel(), bz))
    dz_dx = -(2.0 * math.pi * bump_amplitude / bump_period) * np.sin(
        phase_x
    ) * np.cos(phase_y)
    dz_dy = -(2.0 * math.pi * bump_amplitude / bump_period) * np.cos(
        phase_x
    ) * np.sin(phase_y)
    bump_normals = _normalize_rows(
        np.column_stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)))
    )

    concave_mirror_points = -sharp_points
    concave_mirror_normals = -sharp_normals

    notch_depth = 12.0
    notch_width = 24.0
    notch_axis = _grid(0.0, notch_width, spacing_mm)
    notch_height = _grid(0.0, notch_depth, spacing_mm)
    bottom = _plane_xy(notch_axis, notch_axis, 0.0, (0.0, 0.0, 1.0))
    wall_x = _plane_yz(
        notch_axis, notch_height, 0.0, (1.0, 0.0, 0.0)
    )
    wall_y = _plane_xz(
        notch_axis, notch_height, 0.0, (0.0, 1.0, 0.0)
    )
    top_points, top_normals = _plane_xy(
        signed, signed, notch_depth, (0.0, 0.0, 1.0)
    )
    outside_opening = ~(
        (top_points[:, 0] > 0.0)
        & (top_points[:, 0] < notch_width)
        & (top_points[:, 1] > 0.0)
        & (top_points[:, 1] < notch_width)
    )
    notch_points, notch_normals = _stack(
        (
            bottom,
            wall_x,
            wall_y,
            (top_points[outside_opening], top_normals[outside_opening]),
        )
    )

    return [
        SyntheticPatch(
            "face_interior",
            face_points,
            face_normals,
            False,
            False,
            "Interior of one smooth planar face.",
        ),
        SyntheticPatch(
            "visibility_boundary",
            boundary_points,
            boundary_normals,
            False,
            False,
            "One-sided planar crop; a physical edge and an occlusion crop are locally identical.",
        ),
        SyntheticPatch(
            "visibility_crop_corner",
            crop_corner_points,
            crop_corner_normals,
            False,
            False,
            "A quadrant-shaped planar crop corner; locally identical to a one-face physical termination.",
        ),
        SyntheticPatch(
            "sharp_edge",
            edge_points,
            edge_normals,
            False,
            False,
            "Two perpendicular half-planes meeting along a line.",
        ),
        SyntheticPatch(
            "sharp_convex_corner",
            sharp_points,
            sharp_normals,
            True,
            True,
            "Three perpendicular faces meeting at an exposed vertex.",
        ),
        SyntheticPatch(
            "rounded_corner_r14",
            rounded_points,
            rounded_normals,
            True,
            False,
            "A 14 mm spherical/cylindrical fillet joining three planar faces.",
        ),
        SyntheticPatch(
            "sphere_r70",
            sphere_points,
            sphere_normals,
            False,
            False,
            "Smooth spherical surface at its top point.",
        ),
        SyntheticPatch(
            "bumpy_surface",
            bump_points,
            bump_normals,
            False,
            False,
            "Sinusoidal height field queried at a smooth local maximum.",
        ),
        SyntheticPatch(
            "concave_trihedral_mirror",
            concave_mirror_points,
            concave_mirror_normals,
            True,
            True,
            "Central reflection of the convex trihedral; point-only invariant scores cannot distinguish them.",
        ),
        SyntheticPatch(
            "concave_notch_with_rim",
            notch_points,
            notch_normals,
            True,
            False,
            "Bottom corner of a 12 mm recess plus the surrounding top surface.",
        ),
    ]


def perturb_patch(
    patch: SyntheticPatch,
    *,
    seed: int,
    noise_sigma_mm: float,
    normal_noise_degrees: float,
    dropout: float,
) -> SyntheticPatch:
    """Apply deterministic observation noise while preserving the exact query."""
    rng = np.random.default_rng(seed)
    keep = rng.random(len(patch.points_mm)) >= dropout
    points = patch.points_mm[keep].copy()
    normals = patch.normals[keep].copy()
    if noise_sigma_mm > 0.0:
        points += rng.normal(0.0, noise_sigma_mm, points.shape)
    if normal_noise_degrees > 0.0:
        scale = math.sin(math.radians(normal_noise_degrees))
        normals = _normalize_rows(normals + rng.normal(0.0, scale, normals.shape))

    # q belongs to conv({q} U P_r(q)); it need not have a unique surface normal.
    points = np.vstack((points, np.zeros((1, 3), dtype=np.float64)))
    normals = np.vstack((normals, np.full((1, 3), np.nan)))
    return SyntheticPatch(
        patch.name,
        points,
        normals,
        patch.desired_corner,
        patch.sharp_corner,
        patch.description,
    )


def empirical_projection_quantile(
    offsets_mm: np.ndarray,
    directions: np.ndarray,
    quantile: float,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return the inverse empirical CDF used by the production detector."""
    offsets = np.asarray(offsets_mm, dtype=np.float64)
    required = int(math.ceil(float(quantile) * len(offsets)))
    kth = required - 1
    output = np.empty(len(directions), dtype=np.float64)
    for start in range(0, len(directions), chunk_size):
        stop = min(start + chunk_size, len(directions))
        projection = offsets @ directions[start:stop].T
        output[start:stop] = np.partition(projection, kth, axis=0)[kth]
    return output


def exact_support_mask(
    offsets_mm: np.ndarray, directions: np.ndarray, *, tolerance_mm: float = 0.0
) -> np.ndarray:
    """Evaluate the exact finite-set support function in bounded memory."""
    maximum = np.full(len(directions), -np.inf, dtype=np.float64)
    for start in range(0, len(directions), 256):
        stop = min(start + 256, len(directions))
        maximum[start:stop] = np.max(
            np.asarray(offsets_mm) @ directions[start:stop].T, axis=0
        )
    return maximum <= float(tolerance_mm) + 1e-12


def tangent_polar_mask(offsets_mm: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Evaluate the polar of the cone generated by nonzero neighbor rays."""
    offsets = np.asarray(offsets_mm, dtype=np.float64)
    length = np.linalg.norm(offsets, axis=1)
    rays = offsets[length > 1e-12] / length[length > 1e-12, None]
    if len(rays) == 0:
        return np.ones(len(directions), dtype=bool)
    return exact_support_mask(rays, directions, tolerance_mm=0.0)


def _cone_shape(mask: np.ndarray, directions: np.ndarray) -> tuple[float, float]:
    """Return accepted-direction resultant and median aperture in degrees."""
    accepted = directions[np.asarray(mask, dtype=bool)]
    if len(accepted) == 0:
        return 0.0, float("nan")
    mean = np.mean(accepted, axis=0)
    resultant = float(np.linalg.norm(mean))
    if resultant < 1e-8:
        return resultant, float("nan")
    axis = mean / resultant
    angle = np.degrees(np.arccos(np.clip(accepted @ axis, -1.0, 1.0)))
    return resultant, float(np.median(angle))


def _normal_metrics(normals: np.ndarray, distances: np.ndarray, radius_mm: float) -> dict[str, float]:
    valid = np.all(np.isfinite(normals), axis=1)
    values = _normalize_rows(normals[valid])
    distance = np.asarray(distances, dtype=np.float64)[valid]
    finite_unit = np.all(np.isfinite(values), axis=1)
    values = values[finite_unit]
    distance = distance[finite_unit]
    if len(values) < 3:
        return {
            "normal_opening_deg": float("nan"),
            "normal_corner_phi": float("nan"),
            "normal_cone_corner_score": float("nan"),
            "tensor_cornerness": float("nan"),
            "harris3d_response": float("nan"),
        }

    if len(values) > 256:
        select = np.linspace(0, len(values) - 1, 256).round().astype(np.int64)
        sampled = values[select]
    else:
        sampled = values
    gram = sampled @ sampled.T
    pair = np.unravel_index(int(np.argmin(gram)), gram.shape)
    minimum_dot = float(np.clip(gram[pair], -1.0, 1.0))
    opening = math.degrees(math.acos(minimum_dot))
    cross = np.cross(sampled[pair[0]], sampled[pair[1]])
    cross_length = float(np.linalg.norm(cross))
    phi = 0.0 if cross_length < 1e-12 else float(np.max(np.abs(sampled @ (cross / cross_length))))
    cone_score = 0.5 * (1.0 - minimum_dot) * phi

    sigma = max(radius_mm * 0.5, np.finfo(float).eps)
    weight = np.exp(-(distance**2) / (2.0 * sigma**2))
    weight /= np.sum(weight)
    moment = (values * weight[:, None]).T @ values
    eigenvalues = np.linalg.eigvalsh(moment)
    tensor_cornerness = float(eigenvalues[0] / max(np.sum(eigenvalues), 1e-12))
    trace = float(np.trace(moment))
    harris = float(0.04 + np.linalg.det(moment) - 0.04 * trace * trace)
    return {
        "normal_opening_deg": opening,
        "normal_corner_phi": phi,
        "normal_cone_corner_score": cone_score,
        "tensor_cornerness": tensor_cornerness,
        "harris3d_response": harris,
    }


def _iss_metrics(points_mm: np.ndarray) -> dict[str, float | bool]:
    points = np.asarray(points_mm, dtype=np.float64)
    if len(points) < 3:
        return {
            "iss_ratio21": float("nan"),
            "iss_ratio32": float("nan"),
            "iss_saliency": float("nan"),
            "iss_pass": False,
        }
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / len(points)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    ratio21 = float(eigenvalues[1] / max(eigenvalues[0], 1e-12))
    ratio32 = float(eigenvalues[2] / max(eigenvalues[1], 1e-12))
    return {
        "iss_ratio21": ratio21,
        "iss_ratio32": ratio32,
        "iss_saliency": float(eigenvalues[2]),
        "iss_pass": bool(ratio21 < 0.975 and ratio32 < 0.975),
    }


def _pca_normal(points_mm: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_mm, dtype=np.float64)
    if len(points) < 3:
        return None
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / len(points)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    length = float(np.linalg.norm(normal))
    return None if length < 1e-12 else normal / length


def difference_of_normals(
    points_mm: np.ndarray, small_radius_mm: float, large_radius_mm: float
) -> float:
    """Fixed-radius query DoN magnitude with optimal sign alignment."""
    distance = np.linalg.norm(points_mm, axis=1)
    first = _pca_normal(points_mm[distance <= small_radius_mm + 1e-9])
    second = _pca_normal(points_mm[distance <= large_radius_mm + 1e-9])
    if first is None or second is None:
        return float("nan")
    dot = float(abs(np.clip(first @ second, -1.0, 1.0)))
    return math.sqrt(max(0.0, 0.5 * (1.0 - dot)))


def evaluate_patch(
    patch: SyntheticPatch,
    directions: np.ndarray,
    *,
    radius_mm: float,
    tolerance_mm: float,
    quantile: float,
    don_small_radius_mm: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Measure all query statistics at one radius."""
    distance = np.linalg.norm(patch.points_mm, axis=1)
    neighborhood = distance <= radius_mm + 1e-9
    offsets = patch.points_mm[neighborhood]
    normals = patch.normals[neighborhood]
    local_distance = distance[neighborhood]
    if len(offsets) < 3:
        raise ValueError(f"{patch.name} has too few points at radius {radius_mm}")

    quantile_values = empirical_projection_quantile(offsets, directions, quantile)
    robust_mask = quantile_values <= tolerance_mm
    strict_mask = exact_support_mask(offsets, directions)
    tangent_mask = tangent_polar_mask(offsets, directions)
    resultant, aperture = _cone_shape(robust_mask, directions)
    row: dict[str, object] = {
        "scene": patch.name,
        "desired_corner": patch.desired_corner,
        "sharp_corner": patch.sharp_corner,
        "radius_mm": float(radius_mm),
        "neighbor_count": int(len(offsets)),
        "support_score_q95": float(np.mean(robust_mask)),
        "exact_support_score": float(np.mean(strict_mask)),
        "tangent_polar_score": float(np.mean(tangent_mask)),
        "tangent_exact_mismatch": int(np.count_nonzero(strict_mask != tangent_mask)),
        "accepted_resultant": resultant,
        "accepted_median_aperture_deg": aperture,
        "h_q10_mm": float(np.quantile(quantile_values, 0.10)),
        "h_q50_mm": float(np.quantile(quantile_values, 0.50)),
        "h_q90_mm": float(np.quantile(quantile_values, 0.90)),
        "don_query": difference_of_normals(
            patch.points_mm, min(don_small_radius_mm, radius_mm), radius_mm
        ),
    }
    row.update(_normal_metrics(normals, local_distance, radius_mm))
    row.update(_iss_metrics(offsets))
    return row, quantile_values, robust_mask


def _median_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "support_score_q95",
        "exact_support_score",
        "accepted_resultant",
        "accepted_median_aperture_deg",
        "normal_cone_corner_score",
        "tensor_cornerness",
        "harris3d_response",
        "iss_saliency",
        "don_query",
    )
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), float(row["radius_mm"])), []).append(row)
    output: list[dict[str, object]] = []
    for (scene, radius), group in grouped.items():
        record: dict[str, object] = {
            "scene": scene,
            "radius_mm": radius,
            "repeats": len(group),
            "desired_corner": bool(group[0]["desired_corner"]),
            "sharp_corner": bool(group[0]["sharp_corner"]),
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            finite = values[np.isfinite(values)]
            for suffix, percentile in (("q25", 25), ("median", 50), ("q75", 75)):
                record[f"{metric}_{suffix}"] = (
                    float(np.percentile(finite, percentile)) if len(finite) else float("nan")
                )
        output.append(record)
    return sorted(output, key=lambda item: (str(item["scene"]), float(item["radius_mm"])))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive = int(np.count_nonzero(labels))
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (np.sum(ranks[labels]) - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def _persistent_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "support_score_q95",
        "normal_cone_corner_score",
        "tensor_cornerness",
        "harris3d_response",
        "iss_saliency",
        "don_query",
    )
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), int(row["repeat"])), []).append(row)
    output: list[dict[str, object]] = []
    for (scene, repeat), group in grouped.items():
        record: dict[str, object] = {
            "scene": scene,
            "repeat": repeat,
            "desired_corner": bool(group[0]["desired_corner"]),
            "sharp_corner": bool(group[0]["sharp_corner"]),
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            record[metric] = float(np.nanmedian(values))
        support = np.asarray([float(item["support_score_q95"]) for item in group])
        record["support_consistency"] = float(
            np.min(support) / max(np.max(support), np.finfo(float).eps)
        )
        output.append(record)
    return output


def _method_auc(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    persistent = _persistent_scores(rows)
    metrics = (
        "support_score_q95",
        "normal_cone_corner_score",
        "tensor_cornerness",
        "harris3d_response",
        "iss_saliency",
        "don_query",
    )
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        scores = np.asarray([float(item[metric]) for item in persistent])
        result[metric] = {
            "sharp_vs_noncorner_auc": _roc_auc(
                np.asarray([bool(item["sharp_corner"]) for item in persistent]), scores
            ),
            "all_desired_vs_noncorner_auc": _roc_auc(
                np.asarray([bool(item["desired_corner"]) for item in persistent]), scores
            ),
        }
    return result


def _threshold_rates(rows: list[dict[str, object]], threshold: float = 0.10) -> dict[str, float]:
    persistent = _persistent_scores(rows)
    grouped: dict[str, list[float]] = {}
    for row in persistent:
        grouped.setdefault(str(row["scene"]), []).append(float(row["support_score_q95"]))
    return {
        scene: float(np.mean(np.asarray(values) >= threshold))
        for scene, values in grouped.items()
    }


def _render_support_curves(summary: list[dict[str, object]], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-directional-support-mpl")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    scenes = list(dict.fromkeys(str(row["scene"]) for row in summary))
    row_count = int(math.ceil(len(scenes) / 3.0))
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(13.5, 3.2 * row_count + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, scene in zip(axes.ravel(), scenes):
        group = [row for row in summary if row["scene"] == scene]
        radii = np.asarray([float(row["radius_mm"]) for row in group])
        median = np.asarray([float(row["support_score_q95_median"]) for row in group])
        q25 = np.asarray([float(row["support_score_q95_q25"]) for row in group])
        q75 = np.asarray([float(row["support_score_q95_q75"]) for row in group])
        axis.fill_between(radii, q25, q75, color="#4C78A8", alpha=0.22)
        axis.plot(radii, median, marker="o", color="#1F5A94", linewidth=2)
        axis.axhline(0.10, color="#C44E52", linestyle="--", linewidth=1)
        axis.set_title(scene.replace("_", " "), fontsize=10)
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(scenes) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("support radius [mm]")
    for axis in axes[:, 0]:
        axis.set_ylabel("q95 support solid-angle fraction")
    figure.suptitle(
        "Robust 3-D directional support by local geometry\n"
        "line: median over perturbations; band: interquartile range; dashed: 0.10 gate",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _render_spherical_profiles(
    clean_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    directions: np.ndarray,
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-directional-support-mpl")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    longitude = np.arctan2(directions[:, 1], directions[:, 0])
    latitude = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    row_count = int(math.ceil(len(clean_profiles) / 3.0))
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(14.5, 3.1 * row_count + 1.1),
        squeeze=False,
        subplot_kw={"projection": "mollweide"},
    )
    color_handle = None
    for axis, (scene, (quantile_values, allowed)) in zip(
        axes.ravel(), clean_profiles.items()
    ):
        clipped = np.clip(quantile_values, -2.0, 8.0)
        color_handle = axis.scatter(
            longitude,
            latitude,
            c=clipped,
            s=6,
            cmap="coolwarm",
            vmin=-2.0,
            vmax=8.0,
            rasterized=True,
        )
        axis.scatter(
            longitude[allowed],
            latitude[allowed],
            s=8,
            facecolors="none",
            edgecolors="#00E5FF",
            linewidths=0.45,
        )
        axis.set_title(scene.replace("_", " "), fontsize=9)
        axis.grid(alpha=0.2)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in axes.ravel()[len(clean_profiles) :]:
        axis.set_visible(False)
    if color_handle is not None:
        color_axis = figure.add_axes((0.915, 0.20, 0.015, 0.58))
        figure.colorbar(
            color_handle,
            cax=color_axis,
            label="empirical q95 of u·(p-q) [mm] (cyan ring = accepted)",
        )
    figure.suptitle("Spherical directional-support profiles at 26 mm (clean patches)", fontsize=14)
    figure.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.91, wspace=0.08, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _render_method_heatmap(rows: list[dict[str, object]], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-directional-support-mpl")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    persistent = _persistent_scores(rows)
    scenes = list(dict.fromkeys(str(row["scene"]) for row in persistent))
    metrics = (
        ("support_score_q95", "Directional\nsupport"),
        ("normal_cone_corner_score", "Oracle normal-cone\nopening proxy"),
        ("tensor_cornerness", "Oracle tensor-inspired\ncornerness"),
        ("harris3d_response", "Oracle Harris3D\nresponse"),
        ("iss_saliency", "ISS\nsaliency"),
        ("don_query", "DoN query\nchange"),
    )
    matrix = np.zeros((len(scenes), len(metrics)), dtype=np.float64)
    for row_index, scene in enumerate(scenes):
        group = [row for row in persistent if row["scene"] == scene]
        for column, (metric, _) in enumerate(metrics):
            matrix[row_index, column] = float(
                np.nanmedian([float(item[metric]) for item in group])
            )
    normalized = np.zeros_like(matrix)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        low, high = np.nanmin(values), np.nanmax(values)
        if np.isfinite(low) and high > low:
            normalized[:, column] = (values - low) / (high - low)

    figure, axis = plt.subplots(figsize=(10.5, 6.7))
    image = axis.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    axis.set_yticks(np.arange(len(scenes)), [scene.replace("_", " ") for scene in scenes])
    for row_index in range(len(scenes)):
        for column in range(len(metrics)):
            raw = matrix[row_index, column]
            label = "nan" if not np.isfinite(raw) else f"{raw:.3g}"
            axis.text(
                column,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[row_index, column] < 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="within-method min-max normalized value")
    axis.set_title(
        "Query-level method diagnostics (median across radii and perturbations)\n"
        "normal-cone/tensor/Harris columns use analytic surface normals"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_benchmark(
    *,
    radii_mm: Sequence[float] = (8.0, 12.0, 18.0, 26.0, 34.0),
    sphere_samples: int = 1024,
    repeats: int = 8,
    tolerance_mm: float = 0.6,
    quantile: float = 0.95,
    noise_sigma_mm: float = 0.25,
    normal_noise_degrees: float = 3.0,
    dropout: float = 0.10,
    output_dir: str | Path = "outputs/directional_support_experiment",
    render_plots: bool = True,
) -> dict[str, object]:
    """Run the controlled benchmark and save all reproducible artifacts."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    radii = tuple(float(value) for value in radii_mm)
    if len(radii) < 2 or any(value <= 0.0 for value in radii):
        raise ValueError("radii_mm must contain at least two positive values")
    if any(second <= first for first, second in zip(radii, radii[1:])):
        raise ValueError("radii_mm must be strictly increasing")
    if not 0.5 < quantile <= 1.0:
        raise ValueError("quantile must be in (0.5, 1]")

    directions = antipodal_sphere(sphere_samples)
    patches = make_synthetic_patches()
    rows: list[dict[str, object]] = []
    clean_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    exact_mismatches = 0
    clean_scores: dict[str, list[float]] = {}
    profile_radius = min(radii, key=lambda value: abs(value - 26.0))

    for patch_index, patch in enumerate(patches):
        clean_with_query = SyntheticPatch(
            patch.name,
            np.vstack((patch.points_mm, np.zeros((1, 3)))),
            np.vstack((patch.normals, np.full((1, 3), np.nan))),
            patch.desired_corner,
            patch.sharp_corner,
            patch.description,
        )
        clean_scores[patch.name] = []
        for radius in radii:
            clean_row, quantile_values, allowed = evaluate_patch(
                clean_with_query,
                directions,
                radius_mm=radius,
                tolerance_mm=tolerance_mm,
                quantile=quantile,
                don_small_radius_mm=radii[0],
            )
            exact_mismatches += int(clean_row["tangent_exact_mismatch"])
            clean_scores[patch.name].append(float(clean_row["exact_support_score"]))
            if radius == profile_radius:
                clean_profiles[patch.name] = (quantile_values, allowed)

        for repeat in range(repeats):
            observed = perturb_patch(
                patch,
                seed=1009 * patch_index + repeat,
                noise_sigma_mm=noise_sigma_mm,
                normal_noise_degrees=normal_noise_degrees,
                dropout=dropout,
            )
            for radius in radii:
                row, _, _ = evaluate_patch(
                    observed,
                    directions,
                    radius_mm=radius,
                    tolerance_mm=tolerance_mm,
                    quantile=quantile,
                    don_small_radius_mm=radii[0],
                )
                row["repeat"] = repeat
                rows.append(row)

    summary_rows = _median_summary(rows)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output / "query_metrics.csv")
    _write_csv(summary_rows, output / "summary.csv")

    convex = np.asarray(clean_scores["sharp_convex_corner"])
    concave = np.asarray(clean_scores["concave_trihedral_mirror"])
    exact_monotonic_violations = {
        scene: int(np.count_nonzero(np.diff(values) > 1e-12))
        for scene, values in clean_scores.items()
    }
    result: dict[str, object] = {
        "experiment": "query-level 3-D directional support geometry benchmark",
        "parameters": {
            "radii_mm": list(radii),
            "sphere_samples": sphere_samples,
            "repeats": repeats,
            "tolerance_mm": tolerance_mm,
            "quantile": quantile,
            "noise_sigma_mm": noise_sigma_mm,
            "normal_noise_degrees": normal_noise_degrees,
            "dropout": dropout,
        },
        "scenes": {
            patch.name: {
                "desired_corner": patch.desired_corner,
                "sharp_corner": patch.sharp_corner,
                "description": patch.description,
            }
            for patch in patches
        },
        "invariants": {
            "tangent_polar_vs_exact_support_mask_mismatches": exact_mismatches,
            "convex_vs_mirrored_concave_max_exact_score_delta": float(
                np.max(np.abs(convex - concave))
            ),
            "exact_support_score_radius_monotonicity_violations": exact_monotonic_violations,
        },
        "persistent_method_auc": _method_auc(rows),
        "persistent_q95_support_gate_0p10_pass_rate": _threshold_rates(rows, 0.10),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=True)

    if render_plots:
        _render_support_curves(summary_rows, output / "support_score_by_radius.png")
        _render_spherical_profiles(
            clean_profiles, directions, output / "spherical_support_profiles.png"
        )
        _render_method_heatmap(rows, output / "method_comparison.png")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radii-mm", type=float, nargs="+", default=(8.0, 12.0, 18.0, 26.0, 34.0)
    )
    parser.add_argument("--sphere-samples", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--tolerance-mm", type=float, default=0.6)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--noise-sigma-mm", type=float, default=0.25)
    parser.add_argument("--normal-noise-degrees", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--output-dir", default="outputs/directional_support_experiment"
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_benchmark(
        radii_mm=args.radii_mm,
        sphere_samples=args.sphere_samples,
        repeats=args.repeats,
        tolerance_mm=args.tolerance_mm,
        quantile=args.quantile,
        noise_sigma_mm=args.noise_sigma_mm,
        normal_noise_degrees=args.normal_noise_degrees,
        dropout=args.dropout,
        output_dir=args.output_dir,
        render_plots=not args.no_plots,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"saved: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
