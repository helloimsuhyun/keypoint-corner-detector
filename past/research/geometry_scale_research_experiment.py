"""Research-driven geometry-transition experiments for Normal Cone corners.

The script generates two reproducible figures:

1. synthetic scale curves that separate scale growth from geometry ingestion;
2. an RGB overlay on the locally available 3DMatch RedKitchen sequence.

Only XYZ coordinates are used by the detector and scale selector.  RGB is used
afterwards solely as a visualization background.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-normal-cone-matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from normal_cone_corner_detector import (
    GeometryScaleConfig,
    MultiScaleConfig,
    NormalConeConfig,
    build_scene,
    detect_scene_geometry_adaptive,
    detect_scene_multiscale,
    nearest_point_index,
)
from synthetic_normal_cone_gt_benchmark import make_synthetic_gt_scenes


DEFAULT_SEQUENCE = Path("/home/choisuhyun/datasets/7scenes/redkitchen/seq-03.zip")
RADII_SYNTHETIC = (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)


def _plate_and_wall() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(0.0, 60.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    plate = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    signed = np.arange(-30.0, 30.1, 2.0)
    wall_y, wall_z = np.meshgrid(signed, signed, indexing="ij")
    wall = np.column_stack(
        (np.full(wall_y.size, 18.0), wall_y.ravel(), wall_z.ravel())
    )
    return plate, np.unique(np.vstack((plate, wall)), axis=0)


def _query_scale_diagnostics(
    points_mm: np.ndarray,
    query_mm: np.ndarray,
    config: NormalConeConfig,
):
    scene = build_scene(points_mm, config, downsample=False)
    query = nearest_point_index(scene.points_mm, query_mm)
    multiscale = detect_scene_multiscale(
        scene,
        config,
        MultiScaleConfig(
            radii_mm=RADII_SYNTHETIC,
            min_scale_count=5,
            min_consistency=0.35,
        ),
        query_indices=[query],
    )
    geometry = detect_scene_geometry_adaptive(
        scene,
        config,
        multiscale,
        GeometryScaleConfig(min_clean_scale_count=3, min_shell_points=6),
        query_indices=[query],
    )
    return scene, query, multiscale, geometry


def render_synthetic_scale_figure(output: Path) -> dict[str, object]:
    config = NormalConeConfig(
        query_radius_mm=32.0,
        sphere_samples=256,
        tolerance_mm=0.6,
        min_score=0.10,
    )
    plate, plate_wall = _plate_and_wall()
    scenes = {scene.name: scene for scene in make_synthetic_gt_scenes()}
    cases = [
        ("Same geometry", plate, np.array([0.0, 0.0, 0.0])),
        ("New plane enters at 18 mm", plate_wall, np.array([0.0, 0.0, 0.0])),
        (
            "Sensor protrusion",
            scenes["sensor_outlier_plane"].points_mm,
            np.array([-32.0, -20.0, 5.0]),
        ),
        (
            "Rounded boundary",
            scenes["rounded_plate_virtual_gt"].points_mm,
            np.array([0.0, 0.0, 0.0]),
        ),
    ]

    fig = plt.figure(figsize=(16, 7.8), constrained_layout=True)
    top_axes = [fig.add_subplot(2, 4, column + 1, projection="3d") for column in range(4)]
    bottom_axes = [fig.add_subplot(2, 4, column + 5) for column in range(4)]
    records: dict[str, object] = {}
    for column, (name, points, query_point) in enumerate(cases):
        scene, query, multiscale, geometry = _query_scale_diagnostics(
            points, query_point, config
        )
        q = scene.points_mm[query]
        selected = geometry.selected_radius_mm[query]
        transition_id = int(geometry.transition_scale_index[query])
        transition = (
            float(geometry.radii_mm[transition_id])
            if transition_id < len(geometry.radii_mm)
            else None
        )

        ax = top_axes[column]
        local = np.linalg.norm(scene.points_mm - q, axis=1) <= 35.0
        local_points = scene.points_mm[local]
        if len(local_points) > 1800:
            local_points = local_points[:: int(np.ceil(len(local_points) / 1800))]
        ax.scatter(
            local_points[:, 0] - q[0],
            local_points[:, 1] - q[1],
            local_points[:, 2] - q[2],
            s=3,
            c="#8a96a3",
            alpha=0.55,
            depthshade=False,
        )
        ax.scatter([0], [0], [0], s=55, c="#d62728", marker="*", depthshade=False)
        ax.set_title(name)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect((1, 1, 0.72))

        curve = bottom_axes[column]
        radii = geometry.radii_mm
        scores = multiscale.scale_scores[:, query]
        dims = geometry.dimensionality[:, query]
        curve.plot(radii, scores, "o-", color="#202020", label="Normal Cone score")
        curve.plot(radii, dims[:, 0], "s-", color="#1f77b4", label="1D")
        curve.plot(radii, dims[:, 1], "^-", color="#2ca02c", label="2D")
        curve.plot(radii, dims[:, 2], "D-", color="#9467bd", label="3D")
        curve.plot(radii, geometry.anchor_js[:, query], "x--", color="#d62728", label="geometry JS")
        curve.axvline(selected, color="#ff7f0e", linewidth=2, label="selected radius")
        if transition is not None:
            curve.axvline(transition, color="#d62728", linestyle=":", linewidth=2)
        curve.axhline(config.min_score, color="#555555", linestyle="--", linewidth=1)
        curve.set_xlabel("Neighborhood radius (mm)")
        curve.set_ylabel("Response / normalized distance")
        curve.set_ylim(-0.03, 1.03)
        curve.grid(alpha=0.22)
        if column == 0:
            curve.legend(loc="upper right", fontsize=8)

        records[name] = {
            "query_mm": q.tolist(),
            "scores": scores.tolist(),
            "dimensionality": dims.tolist(),
            "anchor_js": [
                float(value) if np.isfinite(value) else None
                for value in geometry.anchor_js[:, query]
            ],
            "selected_radius_mm": float(selected),
            "transition_radius_mm": transition,
            "accepted": bool(geometry.valid_mask[query]),
        }

    fig.suptitle(
        "Multi-scale Normal Cone: growing support vs. newly included geometry",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return records


def _load_redkitchen_frame(
    sequence: Path, frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"seq-03/frame-{frame:06d}"
    with zipfile.ZipFile(sequence) as archive:
        rgb = np.asarray(
            Image.open(io.BytesIO(archive.read(f"{prefix}.color.png"))).convert("RGB")
        )
        depth = np.asarray(Image.open(io.BytesIO(archive.read(f"{prefix}.depth.png"))))
    rows, columns = np.indices(depth.shape)
    valid = (depth > 0) & (depth != 65535) & (depth < 5000)
    z = depth[valid].astype(np.float64)
    points = np.column_stack(
        (
            (columns[valid] - 320.0) * z / 585.0,
            (rows[valid] - 240.0) * z / 585.0,
            z,
        )
    )
    return rgb, depth, points


def _project(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)
    z = np.maximum(points[:, 2], 1e-9)
    return np.column_stack((585.0 * points[:, 0] / z + 320.0, 585.0 * points[:, 1] / z + 240.0))


def render_redkitchen_figure(
    sequence: Path,
    frame: int,
    output: Path,
    npz_output: Path,
) -> dict[str, object]:
    rgb, depth, raw_points = _load_redkitchen_frame(sequence, frame)
    config = NormalConeConfig(
        voxel_size_mm=15.0,
        query_radius_mm=150.0,
        sphere_samples=128,
        tolerance_mm=4.0,
        min_neighbors=12,
        min_score=0.08,
        local_max_radius_mm=22.0,
        nms_radius_mm=45.0,
        direction_chunk_size=128,
    )
    radii = (30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0)
    scene = build_scene(raw_points, config, downsample=True)
    multiscale = detect_scene_multiscale(
        scene,
        config,
        MultiScaleConfig(
            radii_mm=radii,
            min_scale_count=5,
            min_consistency=0.35,
        ),
    )
    geometry = detect_scene_geometry_adaptive(
        scene,
        config,
        multiscale,
        GeometryScaleConfig(
            min_clean_scale_count=3,
            min_score_consistency=0.35,
            min_shell_points=12,
        ),
    )

    score_ids = multiscale.candidate_indices
    geometry_ids = geometry.candidate_indices
    score_uv = _project(scene.points_mm[score_ids])
    geometry_uv = _project(scene.points_mm[geometry_ids])
    transition_mask = geometry.transition_scale_index < len(radii)
    transition_ids = np.flatnonzero(transition_mask)
    if len(transition_ids) > 12000:
        transition_ids = transition_ids[:: int(np.ceil(len(transition_ids) / 12000))]
    transition_uv = _project(scene.points_mm[transition_ids])
    transition_scale = geometry.transition_scale_index[transition_ids]
    transition_radius = np.asarray(radii)[transition_scale]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.3), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"3DMatch RedKitchen seq-03 frame {frame:06d}")

    axes[0, 1].imshow(rgb)
    axes[0, 1].scatter(
        score_uv[:, 0], score_uv[:, 1], s=30, facecolors="none", edgecolors="#00bcd4", linewidths=1.2
    )
    axes[0, 1].set_title(f"Score persistence: {len(score_ids)} NMS candidates")

    axes[1, 0].imshow(rgb)
    scatter = axes[1, 0].scatter(
        geometry_uv[:, 0],
        geometry_uv[:, 1],
        c=geometry.selected_radius_mm[geometry_ids],
        cmap="plasma",
        vmin=min(radii),
        vmax=max(radii),
        s=34,
        edgecolors="white",
        linewidths=0.45,
    )
    axes[1, 0].set_title(f"Geometry-adaptive: {len(geometry_ids)} candidates")
    colorbar = fig.colorbar(scatter, ax=axes[1, 0], fraction=0.04, pad=0.02)
    colorbar.set_label("Selected clean radius (mm)")

    axes[1, 1].imshow(rgb)
    transitions = axes[1, 1].scatter(
        transition_uv[:, 0],
        transition_uv[:, 1],
        c=transition_radius,
        cmap="viridis",
        vmin=min(radii),
        vmax=max(radii),
        s=4,
        alpha=0.65,
        linewidths=0,
    )
    axes[1, 1].set_title("First radius that ingests different geometry")
    colorbar = fig.colorbar(transitions, ax=axes[1, 1], fraction=0.04, pad=0.02)
    colorbar.set_label("Transition radius (mm)")

    for axis in axes.ravel():
        axis.set_xlim(0, rgb.shape[1])
        axis.set_ylim(rgb.shape[0], 0)
        axis.set_axis_off()

    fig.suptitle(
        "Coordinate-only multi-scale geometry diagnostics (RGB used only for display)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz_output,
        points_mm=scene.points_mm,
        radii_mm=np.asarray(radii),
        scale_scores=multiscale.scale_scores,
        score_candidate_indices=score_ids,
        geometry_candidate_indices=geometry_ids,
        dimensionality=geometry.dimensionality,
        anchor_js=geometry.anchor_js,
        shell_js=geometry.shell_js,
        transition_scale_index=geometry.transition_scale_index,
        selected_radius_mm=geometry.selected_radius_mm,
    )
    return {
        "frame": frame,
        "raw_point_count": len(raw_points),
        "voxel_point_count": len(scene.points_mm),
        "score_candidate_count": len(score_ids),
        "geometry_candidate_count": len(geometry_ids),
        "transition_point_count": int(np.count_nonzero(transition_mask)),
        "score_only_geometry_overlap": int(len(np.intersect1d(score_ids, geometry_ids))),
        "median_selected_radius_mm": float(
            np.median(geometry.selected_radius_mm[geometry.valid_mask])
        ),
    }


def render_benchmark_figure(csv_path: Path, output: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    variants = ("multi_5of7", "geometry_adaptive")
    scene_names = [
        "plate_clean",
        "plate_noisy_dropout",
        "two_face_noisy",
        "three_face_noisy",
        "missing_patch",
        "sensor_outlier_plane",
        "sphere_patch",
        "cylinder_patch",
        "corner_near_perpendicular_plane",
        "corner_near_elevated_plate",
        "plane_near_perpendicular_plane",
    ]
    values = {variant: [] for variant in variants}
    for variant in variants:
        by_scene = {
            row["scene"]: row
            for row in rows
            if row["variant"] == variant and row["scene"] in scene_names
        }
        values[variant] = [int(by_scene[name]["fp"]) for name in scene_names]

    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    positions = np.arange(len(scene_names))
    width = 0.38
    ax.bar(
        positions - width / 2,
        values["multi_5of7"],
        width,
        label="Score persistence (5/7)",
        color="#4c78a8",
    )
    ax.bar(
        positions + width / 2,
        values["geometry_adaptive"],
        width,
        label="Geometry-adaptive prefix",
        color="#f58518",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "plate",
            "noisy plate",
            "2-face",
            "3-face",
            "missing patch",
            "sensor outlier",
            "sphere",
            "cylinder",
            "corner + wall",
            "corner + plate",
            "plane + wall",
        ],
        rotation=28,
        ha="right",
    )
    ax.set_ylabel("False positives in evaluation region")
    ax.set_title("Adaptive early stopping preserves transitions but relaxes FP suppression")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--frame", type=int, default=740)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_scale_research"))
    parser.add_argument("--skip-redkitchen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {
        "synthetic": render_synthetic_scale_figure(output / "synthetic-geometry-transitions.png")
    }
    benchmark_csv = Path("outputs/geometry_scale_benchmark_256.csv")
    if not benchmark_csv.exists():
        benchmark_csv = Path("outputs/geometry_scale_benchmark_128.csv")
    if benchmark_csv.exists():
        render_benchmark_figure(benchmark_csv, output / "geometry-adaptive-benchmark.png")
    if not args.skip_redkitchen:
        records["redkitchen"] = render_redkitchen_figure(
            args.sequence.expanduser().resolve(),
            args.frame,
            output / "redkitchen-geometry-transitions.png",
            output / "redkitchen-geometry-transitions.npz",
        )
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, allow_nan=False)
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
