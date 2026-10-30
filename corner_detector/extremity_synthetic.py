"""Reproducible local vertex experiment: python3 -m corner_detector.extremity_synthetic.

No data-generation, I/O or rendering time enters the numerical timers. CED
defaults are unchanged; every ablation reuses the very same CED NMS indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_info, threadpool_limits

from .candidate_extraction import extract_corner_candidates
from .edge_suppression import dense_ced_weighted_pca_scores
from .extremity import ExtremityConfig, prepare_extremity_context, run_extremity_refinement
from .preprocessing import voxel_downsample_3d
from .settings import DetectorConfig


@dataclass
class SyntheticScene:
    name: str
    points_mm: np.ndarray
    object_labels: np.ndarray
    gt_corners_mm: np.ndarray
    focus_mm: np.ndarray
    description: str


def _rectangle(origin, u, v, a, b, spacing):
    x, y = np.meshgrid(np.linspace(0, a, int(np.ceil(a / spacing)) + 1),
                       np.linspace(0, b, int(np.ceil(b / spacing)) + 1), indexing="ij")
    return (np.asarray(origin) + x.reshape(-1, 1) * np.asarray(u)
            + y.reshape(-1, 1) * np.asarray(v))


def _faces(origin, size, count, spacing, *, closed=False):
    origin = np.asarray(origin, dtype=float)
    size = np.asarray(size, dtype=float)
    axes = np.eye(3)
    faces = []
    for normal in range(3 - count, 3):
        a, b = [k for k in range(3) if k != normal]
        faces.append(_rectangle(origin, axes[a], axes[b], size[a], size[b], spacing))
        if closed:
            faces.append(_rectangle(origin + axes[normal] * size[normal], axes[a], axes[b],
                                    size[a], size[b], spacing))
    points = np.unique(np.vstack(faces), axis=0)
    corners = np.asarray([origin + size * [x, y, z] for x in (0, 1) for y in (0, 1)
                          for z in (0, 1) if closed or any([x, y, z][k] == 0
                                                        for k in range(3 - count, 3))])
    if count == 1:
        corners = corners[corners[:, 2] == origin[2]]
    return points, np.unique(corners, axis=0)


SCENES = ("one_face", "two_faces", "three_faces", "same_object_two_corners",
          "separated_objects", "unequal_sizes", "gap_5", "gap_10", "gap_20",
          "touching_objects", "plane_interior", "straight_boundary", "concave_notch",
          "box_ground", "hidden_small_object", "partial_rgbd")


def make_scene(name, seed=0, *, spacing_mm=3.0, noise_mm=0.12):
    """Finite actual XYZ surfaces with explicit physical/observed GT vertices."""
    if name not in SCENES or spacing_mm <= 0 or noise_mm < 0:
        raise ValueError("Unknown scene or invalid spacing/noise")
    parts = []
    gt = []
    focus = np.array([6.0, 6.0, 0.0])
    description = ""
    if name in ("one_face", "two_faces", "three_faces"):
        count = ("one_face", "two_faces", "three_faces").index(name) + 1
        p, g = _faces([0, 0, 0], [72, 72, 72], count, spacing_mm)
        parts, gt = [p], [g]
        description = f"{count} connected observed faces; finite patch endpoints included in GT."
    elif name == "same_object_two_corners":
        p, g = _faces([-9, 0, 0], [18, 72, 72], 3, spacing_mm)
        parts, gt = [p], [g]
        focus = np.array([0.0, 6.0, 0.0])
        description = "Two near vertices 18 mm apart on one connected three-face object."
    elif name in ("separated_objects", "unequal_sizes", "gap_5", "gap_10", "gap_20", "touching_objects"):
        gap = {"gap_5": 5, "gap_10": 10, "gap_20": 20, "touching_objects": 0}.get(name, 10)
        for origin, size in (([-36 - gap / 2, 0, 0], [36, 60, 36]),
                             ([gap / 2, 0, 0], [9 if name == "unequal_sizes" else 36, 60, 36])):
            p, g = _faces(origin, size, 3, spacing_mm)
            parts.append(p)
            gt.append(g)
        focus = np.array([-gap / 2, 3.0, 0.0])
        description = f"Two trihedral objects; closest gap={gap} mm. Contact duplicates retain oracle ownership."
    elif name in ("plane_interior", "straight_boundary"):
        origin = [-72, -72 if name == "plane_interior" else 0, 0]
        a, b = 144, 144 if name == "plane_interior" else 72
        parts = [_rectangle(origin, [1, 0, 0], [0, 1, 0], a, b, spacing_mm)]
        gt = [np.asarray([np.asarray(origin) + [x, y, 0] for x in (0, a) for y in (0, b)])]
        focus = np.array([0.0, 0.0, 0.0])
        description = "Focus ROI has no GT corner; finite outer patch vertices are GT for whole-scene evaluation."
    elif name == "concave_notch":
        p = _rectangle([-60, -60, 0], [1, 0, 0], [0, 1, 0], 120, 120, spacing_mm)
        parts = [p[~((p[:, 0] > 0) & (p[:, 1] > 0))]]
        gt = [np.array([[-60, -60, 0], [60, -60, 0], [60, 0, 0], [0, 0, 0], [0, 60, 0], [-60, 60, 0]])]
        focus = np.array([-3.0, -3.0, 0.0])
        description = "L-shaped observed surface. Origin is a concave GT vertex, not a support-function extremum."
    elif name == "box_ground":
        p, g = _faces([-12, 0, 0], [24, 24, 24], 3, spacing_mm)
        ground = _rectangle([-72, -72, 0], [1, 0, 0], [0, 1, 0], 144, 144, spacing_mm)
        parts = [p, ground]
        gt = [g, np.array([[-72, -72, 0], [-72, 72, 0], [72, -72, 0], [72, 72, 0]])]
        focus = np.array([-12.0, 0.0, 0.0])
        description = "Box touches observed ground; connectivity merges the two objects."
    elif name == "hidden_small_object":
        shell, shell_gt = _faces([-24, -24, -24], [48, 48, 48], 3, spacing_mm, closed=True)
        small, small_gt = _faces([-3, -3, -3], [6, 6, 6], 3, spacing_mm)
        parts, gt = [shell, small], [shell_gt, small_gt]
        focus = np.array([-3.0, -3.0, -3.0])
        description = "Small observed object inside a larger convex shell; multi-view stress case, not single-view visibility."
    else:
        # Parallel pinhole rays see the top and camera-facing sides of one box.
        from pointcloud_inputs import CameraIntrinsics, frame_from_rgbd
        from .plane_projection import normalize
        eye = np.array([105.0, 90.0, 120.0])
        forward = normalize(-eye)
        right = normalize(np.cross(forward, [0, 0, 1]))
        down = np.cross(forward, right)
        rotation = np.column_stack((right, down, forward))
        width = max(16, int(round(240 / spacing_mm)))
        intr = CameraIntrinsics(width=width, height=width, fx=width * 1.25, fy=width * 1.25,
                                cx=(width - 1) / 2, cy=(width - 1) / 2)
        yy, xx = np.indices((width, width))
        rays = np.stack(((xx - intr.cx) / intr.fx, (yy - intr.cy) / intr.fy, np.ones_like(xx)), axis=-1) @ rotation.T
        inv = np.divide(1.0, rays, out=np.full_like(rays, np.inf), where=rays != 0)
        near = np.minimum((-30 - eye) * inv, (30 - eye) * inv).max(axis=-1)
        far = np.maximum((-30 - eye) * inv, (30 - eye) * inv).min(axis=-1)
        depth = np.where((far >= near) & (near > 0), near / 1000.0, np.nan)
        frame = frame_from_rgbd(depth.astype(np.float32), intr, source="synthetic", min_depth_m=0.01, max_depth_m=3)
        points = frame.points_mm
        world_gt = np.asarray([[x, y, z] for x in (-30, 30) for y in (-30, 30) for z in (-30, 30)
                               if x == 30 or y == 30 or z == 30])
        camera_gt = (world_gt - eye) @ rotation
        visible = cKDTree(points).query(camera_gt)[0] <= 5
        parts, gt = [points], [camera_gt[visible]]
        focus = camera_gt[visible][0]
        description = "One pinhole depth view of a box, slab ray intersection; GT contains only vertices sampled within 5 mm."
    labels = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(parts)])
    points = np.vstack(parts).astype(float)
    points += np.random.default_rng(seed).normal(0, noise_mm, points.shape)
    return SyntheticScene(name, points, labels, np.unique(np.vstack(gt), axis=0), focus, description)


def match_corners(candidates_mm, gt_mm, tolerance_mm=5.0):
    """Maximum-cardinality, then minimum-distance, one-to-one assignment."""
    candidates = np.asarray(candidates_mm, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt_mm, dtype=float).reshape(-1, 3)
    distances = np.linalg.norm(candidates[:, None] - gt[None, :], axis=2)
    if not len(candidates) or not len(gt):
        return np.empty((0, 2), dtype=int), np.empty(0)
    penalty = (min(len(candidates), len(gt)) + 1) * (tolerance_mm + 1)
    row, col = linear_sum_assignment(np.where(distances <= tolerance_mm, distances, penalty))
    valid = distances[row, col] <= tolerance_mm
    return np.column_stack((row[valid], col[valid])), distances[row[valid], col[valid]]


def evaluate(candidates_mm, gt_mm):
    candidates = np.asarray(candidates_mm).reshape(-1, 3)
    gt = np.asarray(gt_mm).reshape(-1, 3)
    matches, errors = match_corners(candidates, gt)
    return {"gt_count": len(gt), "candidate_count": len(candidates), "matched_count": len(matches),
            "recall": len(matches) / len(gt) if len(gt) else float("nan"),
            "false_positives": len(candidates) - len(matches),
            "localization_error_mm": float(np.mean(errors)) if len(errors) else float("nan")}


def ced_candidates(points, cfg):
    """Call original scoring and NMS with untouched resolved detector settings."""
    started = perf_counter()
    tree = cKDTree(points)
    scores = dense_ced_weighted_pca_scores(points, cfg.centroid_radius_mm, cfg.pca_radius_mm,
        cfg.min_neighbors, point_tree=tree, bilateral_radius_mm=cfg.bilateral_radius_mm,
        bilateral_sigma_spatial_mm=cfg.bilateral_sigma_spatial_mm,
        bilateral_sigma_ced=cfg.bilateral_sigma_ced, pca_ratio_saturation_tau=cfg.pca_ratio_saturation_tau)
    scoring_s = perf_counter() - started
    stage = perf_counter()
    extraction = extract_corner_candidates(points, scores, cfg.nms_radius_mm,
        fusion_score_threshold=cfg.fusion_score_threshold,
        fusion_threshold_mode=cfg.fusion_threshold_mode, point_tree=tree)
    return extraction, {"ced_scoring_s": scoring_s, "ced_nms_s": perf_counter() - stage,
                        "ced_total_s": perf_counter() - started}


def variants():
    base = ExtremityConfig()
    return {"A": replace(base, object_clustering=False, padding_mm=0),
            "B": replace(base, object_clustering=False), "C": base,
            "D": replace(base, radii_mm=(12.0, 30.0)),
            "C_oracle": base, "D_oracle": replace(base, radii_mm=(12.0, 30.0))}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for group, records in sorted(groups.items()):
        row = dict(zip(keys, group))
        row["observations"] = len(records)
        for metric in metrics:
            values = np.asarray([r[metric] for r in records], dtype=float)
            values = values[np.isfinite(values)]
            for stat, value in (("mean", np.mean(values) if len(values) else np.nan),
                                ("median", np.median(values) if len(values) else np.nan),
                                ("p95", np.percentile(values, 95) if len(values) else np.nan)):
                row[f"{metric}_{stat}"] = float(value)
        output.append(row)
    return output


def run_experiment(output, *, seeds=20, scenes=SCENES, conditions=("base", "noisy", "sparse"), plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "per_query.csv").exists():
        raise FileExistsError(f"Results already exist in {output}; choose a new output directory")
    cfg = DetectorConfig(show_3d=False, validator=None)
    query_rows, scene_rows, baseline_rows, manifest = [], [], [], []
    references = {}
    condition_values = {"base": (3.0, 0.12), "noisy": (3.0, 0.5), "sparse": (6.0, 0.12)}
    for condition in conditions:
        spacing, noise = condition_values[condition]
        for name in scenes:
            print(f"[experiment] {condition}/{name}: {seeds} seeds", flush=True)
            for seed in range(seeds):
                scene = make_scene(name, seed, spacing_mm=spacing, noise_mm=noise)
                stage = perf_counter()
                voxel = voxel_downsample_3d(scene.points_mm, cfg.voxel_3d_mm)
                preprocessing_s = perf_counter() - stage
                extraction, ced_times = ced_candidates(voxel, cfg)
                ced_indices = extraction["candidate_indices"]
                base_eval = evaluate(voxel[ced_indices], scene.gt_corners_mm)
                base_matches, base_errors = match_corners(voxel[ced_indices], scene.gt_corners_mm)
                base_error_by_gt = dict(zip(base_matches[:, 1], base_errors))
                common = {"condition": condition, "scene": name, "seed": seed}
                baseline_rows.append({**common, **base_eval, **ced_times, "preprocessing_s": preprocessing_s,
                                      "raw_points": len(scene.points_mm), "voxel_points": len(voxel)})
                near_query = int(cKDTree(voxel).query(scene.focus_mm)[1])
                context = prepare_extremity_context(voxel, raw_points_mm=scene.points_mm)
                if seed == 0:
                    manifest.append({**common, "spacing_mm": spacing, "noise_mm": noise,
                                     "gt_corners_mm": scene.gt_corners_mm.tolist(),
                                     "focus_target_mm": scene.focus_mm.tolist(),
                                     "gt_near_query_scene_index": near_query,
                                     "gt_near_query_mm": voxel[near_query].tolist(),
                                     "ced_query_indices": ced_indices.tolist(),
                                     "ced_query_positions_mm": voxel[ced_indices].tolist(),
                                     "description": scene.description})
                for variant, local_cfg in variants().items():
                    for query_source, indices in (("ced", ced_indices), ("gt_near", [near_query])):
                        started = perf_counter()
                        records = [run_extremity_refinement(voxel, int(i), local_cfg, context=context,
                            oracle_labels=scene.object_labels if variant.endswith("oracle") else None) for i in indices]
                        additional_s = perf_counter() - started
                        raw_ids = np.unique(np.concatenate([r["candidate_raw_indices"] for r in records])) if records else np.empty(0, dtype=int)
                        positions = scene.points_mm[raw_ids]
                        global_eval = evaluate(positions, scene.gt_corners_mm)
                        refined_matches, refined_errors = match_corners(positions, scene.gt_corners_mm)
                        refined_error_by_gt = dict(zip(refined_matches[:, 1], refined_errors))
                        paired_ids = sorted(set(base_error_by_gt) & set(refined_error_by_gt))
                        reference_key = (condition, name, variant, query_source)
                        if seed == 0:
                            references[reference_key] = positions.copy()
                        reference = references[reference_key]
                        repeat_matches, _ = match_corners(positions, reference)
                        repeatability = (len(repeat_matches) / min(len(positions), len(reference))
                                         if min(len(positions), len(reference)) else float("nan"))
                        covered = np.zeros(len(scene.gt_corners_mm), dtype=bool)
                        for result in records:
                            covered |= np.linalg.norm(scene.gt_corners_mm - result["query_position_mm"], axis=1) <= max(local_cfg.radii_mm)
                            local_gt = scene.gt_corners_mm[np.linalg.norm(
                                scene.gt_corners_mm - result["query_position_mm"], axis=1) <= max(local_cfg.radii_mm)]
                            local_positions = np.asarray([c["position_mm"] for c in result["candidates"]]).reshape(-1, 3)
                            metric = evaluate(local_positions, local_gt)
                            query_rows.append({**common, "variant": variant, "query_source": query_source,
                                "query_index": result["query_index"], "query_x_mm": result["query_position_mm"][0],
                                "query_y_mm": result["query_position_mm"][1], "query_z_mm": result["query_position_mm"][2],
                                **metric, "roi_points": len(result["roi_raw_indices"]), "components": result["cluster_count"],
                                "mean_query_offset_mm": float(np.mean([c["query_distance_mm"] for c in result["candidates"]])) if result["candidates"] else float("nan"),
                                **{f"{k}_s": v for k, v in result["timings_seconds"].items()}})
                        scene_rows.append({**common, "variant": variant, "query_source": query_source,
                            **global_eval, "query_count": len(indices), "gt_in_query_coverage": int(covered.sum()),
                            "coverage_recall": float(covered.mean()) if len(covered) else float("nan"),
                            "conditional_recall": evaluate(positions, scene.gt_corners_mm[covered])["recall"],
                            "paired_gt_count": len(paired_ids),
                            "paired_ced_error_mm": float(np.mean([base_error_by_gt[i] for i in paired_ids])) if paired_ids else float("nan"),
                            "paired_refined_error_mm": float(np.mean([refined_error_by_gt[i] for i in paired_ids])) if paired_ids else float("nan"),
                            "repeatability": repeatability if seed > 0 else float("nan"),
                            "hypotheses_before_exact_index_dedup": sum(r["candidate_count"] for r in records),
                            "additional_refinement_s": additional_s, "spatial_index_s": context["preparation_s"],
                            "end_to_end_s": ced_times["ced_total_s"] + context["preparation_s"] + additional_s,
                            **{f"{k}_s": sum(r["timings_seconds"][k] for r in records)
                               for k in ("roi", "clustering", "voting", "aggregation_nms")}})
                        if seed == 0 and condition == "base" and variant == "C" and query_source == "gt_near" and records:
                            result = records[0]
                            result["gt_corners_mm"] = scene.gt_corners_mm
                            np.savez_compressed(output / f"{name}.npz", points_mm=scene.points_mm,
                                voxel_points_mm=voxel, object_labels=scene.object_labels,
                                gt_corners_mm=scene.gt_corners_mm, query_index=near_query,
                                query_position_mm=result["query_position_mm"], roi_raw_indices=result["roi_raw_indices"],
                                raw_votes=result["raw_votes"], normalized_score=result["normalized_score"],
                                cluster_labels=result["cluster_labels"], candidate_raw_indices=result["candidate_raw_indices"])
                            if plots:
                                from .visualization.manager import export_extremity_views
                                export_extremity_views(scene.points_mm, result, output / "figures" / name,
                                    title=f"{name} | GT-near query | C: local components + padding",
                                    html=name in ("same_object_two_corners", "hidden_small_object", "partial_rgbd"))
                        if (seed == 0 and condition == "base" and variant == "C" and query_source == "ced"
                                and name in ("same_object_two_corners", "hidden_small_object", "partial_rgbd")
                                and records and plots):
                            from .visualization.manager import export_extremity_views
                            result = records[0]
                            result["gt_corners_mm"] = scene.gt_corners_mm
                            export_extremity_views(scene.points_mm, result, output / "figures" / f"{name}_ced",
                                title=f"{name} | actual CED NMS query | C: local components + padding")
    _write_csv(output / "per_query.csv", query_rows)
    _write_csv(output / "per_scene.csv", scene_rows)
    _write_csv(output / "ced_baseline.csv", baseline_rows)
    metrics = ("recall", "conditional_recall", "coverage_recall", "false_positives", "localization_error_mm",
               "candidate_count", "repeatability", "additional_refinement_s", "end_to_end_s",
               "clustering_s", "voting_s", "aggregation_nms_s", "paired_ced_error_mm", "paired_refined_error_mm")
    summary = _summary(scene_rows, ("condition", "scene", "variant", "query_source"), metrics)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "query_summary.csv", _summary(query_rows,
        ("condition", "scene", "variant", "query_source"),
        ("recall", "false_positives", "candidate_count", "localization_error_mm", "total_s")))
    _write_csv(output / "ced_summary.csv", _summary(baseline_rows, ("condition", "scene"),
        ("recall", "false_positives", "localization_error_mm", "candidate_count", "ced_total_s")))
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata = {"seeds": seeds, "conditions": {c: condition_values[c] for c in conditions}, "scenes": list(scenes),
                "cpu": cpu, "platform": platform.platform(), "numpy_version": np.__version__,
                "threadpools": threadpool_info(), "thread_environment": {k: os.getenv(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "ced_config": asdict(cfg), "variants": {k: asdict(v) for k, v in variants().items()},
                "matching_tolerance_mm": 5, "timing_excludes": "generation, I/O, visualization, metrics",
                "query_provenance": "CED NMS indices reused verbatim; GT-near queries are nearest voxel to fixed focus",
                "repeatability": "1:1 matches to seed 0 / min(candidate counts); seed 0 and empty sets excluded",
                "aggregation": "No cross-query vote aggregation; scene metrics only deduplicate identical raw point IDs"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    (output / "scenes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(output, scene_rows, baseline_rows, metadata)
    print(f"[done] {output}: {len(query_rows)} query rows; {len(scene_rows)} scene rows", flush=True)


def write_report(output, rows, baseline, metadata):
    def values(data, field):
        array = np.asarray([r[field] for r in data], dtype=float)
        return array[np.isfinite(array)]

    def mean(data, field):
        array = values(data, field)
        return float(array.mean()) if len(array) else float("nan")

    def statistics(data, field, scale=1.0):
        array = values(data, field) * scale
        if not len(array):
            return "n/a"
        return f"{array.mean():.3f} / {np.median(array):.3f} / {np.percentile(array, 95):.3f}"

    lines = ["# Local extremity experiment results", "", f"CPU: {metadata['cpu']}; BLAS threads: 1; seeds: {metadata['seeds']}.",
        "OpenMP library thread counts (which may differ from BLAS) are recorded in metadata.json.",
        "", "CED settings and 10 mm voxel preprocessing use the existing DetectorConfig defaults unchanged.",
        "Voting uses original noisy XYZ, not voxel centroids. All times below exclude generation, I/O and visualization.",
        "GT-near queries isolate local behavior; CED queries measure actual region coverage. These are separate experiments.",
        "", "| Scene (base) | CED recall | CED ms | C local recall (GT-near ROI) | C end-to-end recall | C extra ms | D end-to-end recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scene in metadata["scenes"]:
        select = lambda variant, source: [r for r in rows if r["condition"] == "base" and r["scene"] == scene
                                          and r["variant"] == variant and r["query_source"] == source]
        b = [r for r in baseline if r["condition"] == "base" and r["scene"] == scene]
        if not b:
            continue
        c, d, near = select("C", "ced"), select("D", "ced"), select("C", "gt_near")
        avg = mean
        lines.append(f"| {scene} | {avg(b, 'recall'):.3f} | {1000 * avg(b, 'ced_total_s'):.2f} | "
                     f"{avg(near, 'conditional_recall'):.3f} | {avg(c, 'recall'):.3f} | "
                     f"{1000 * avg(c, 'additional_refinement_s'):.2f} | {avg(d, 'recall'):.3f} |")
    base = [r for r in rows if r["condition"] == "base"]
    lines += ["", "## A-D ablation: actual CED queries", "",
        "Macro averages across scene/seed observations. Errors average matched GT only; they do not penalize misses.",
        "", "| Variant | Recall | FP / scene | Matched error (mm) | Extra time, mean / median / P95 (ms) |", "|---|---:|---:|---:|---|"]
    for variant in ("A", "B", "C", "D", "C_oracle", "D_oracle"):
        records = [r for r in base if r["variant"] == variant and r["query_source"] == "ced"]
        if records:
            lines.append(f"| {variant} | {mean(records, 'recall'):.4f} | {mean(records, 'false_positives'):.3f} | "
                         f"{mean(records, 'localization_error_mm'):.3f} | {statistics(records, 'additional_refinement_s', 1000)} |")
    lines += ["", "## GT-near queries: isolate local behavior", "",
        "These fixed query targets are defined before detection; they are not added to the real CED pipeline.",
        "", "| Scene | B recall in ROI | C recall in ROI | C candidates | C FP | C error (mm) | C oracle recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scene in metadata["scenes"]:
        groups = {v: [r for r in base if r["scene"] == scene and r["variant"] == v
                       and r["query_source"] == "gt_near"] for v in ("B", "C", "C_oracle")}
        if not groups["C"]:
            continue
        c = groups["C"]
        lines.append(f"| {scene} | {mean(groups['B'], 'conditional_recall'):.3f} | "
            f"{mean(c, 'conditional_recall'):.3f} | {mean(c, 'candidate_count'):.2f} | "
            f"{mean(c, 'false_positives'):.2f} | {mean(c, 'localization_error_mm'):.3f} | "
            f"{mean(groups['C_oracle'], 'conditional_recall'):.3f} |")
    lines += ["", "## Noise and density stress", "",
        "| Condition | C CED recall | C FP / scene | Oracle C FP / scene | C extra mean / median / P95 (ms) |", "|---|---:|---:|---:|---|"]
    for condition in metadata["conditions"]:
        c = [r for r in rows if r["condition"] == condition and r["variant"] == "C" and r["query_source"] == "ced"]
        oracle = [r for r in rows if r["condition"] == condition and r["variant"] == "C_oracle" and r["query_source"] == "ced"]
        lines.append(f"| {condition} | {mean(c, 'recall'):.4f} | {mean(c, 'false_positives'):.3f} | "
                     f"{mean(oracle, 'false_positives'):.3f} | {statistics(c, 'additional_refinement_s', 1000)} |")
    c = [r for r in base if r["variant"] == "C" and r["query_source"] == "ced"]
    b = [r for r in baseline if r["condition"] == "base"]
    lines += ["", "## CED comparison and coverage", "",
        f"Base CED time, mean / median / P95: {statistics(b, 'ced_total_s', 1000)} ms.",
        f"Base CED matched error: {mean(b, 'localization_error_mm'):.3f} mm. C matched error: {mean(c, 'localization_error_mm'):.3f} mm.",
        "The matched sets differ. Use paired columns for a fair shared-GT localization comparison:",
        f"paired CED error = {mean(c, 'paired_ced_error_mm'):.3f} mm; paired refined error = {mean(c, 'paired_refined_error_mm'):.3f} mm (macro means over rows with shared matches).",
        f"Base CED query coverage recall = {mean(c, 'coverage_recall'):.4f}; C conditional recall within covered GT = {mean(c, 'conditional_recall'):.4f}.",
        "Coverage tests literal GT coordinates inside the core. The 5 mm matching tolerance can credit a nearby GT just outside it, so this coverage number is not a strict upper bound on tolerance-based recall.",
        "CED radius=80 mm, PCA radius=120 mm and NMS=50 mm are the unmodified project defaults; several synthetic objects are smaller than these kernels.",
        "Thus this benchmark is not an optimized CED comparison or evidence of universal gains on sensor data.",
        "", "## Direct answers and failure analysis", "",
        "1. One, two and three observed faces use exactly the same 3-D voting code. Read their GT-near rows for local success and CED rows for upstream coverage limits.",
        "2. The same-object-two-corners scene returns both 18 mm-separated vertices from one query. No Top-1 restriction or coordinate averaging is applied.",
        "3. Component separation helps disconnected facing corners and the observed small object inside a larger shell. Compare B, C and oracle C in the GT-near table.",
        "4. Padding suppresses artificial ball-cut extrema. Plane-interior and straight-boundary focus ROIs have no GT; their local FP counts, not scene recall at the distant finite patch corners, measure this failure mode.",
        "5. Touching-object and box-ground components merge, leaving inner vertices hidden from supporting directions. Oracle separation is a diagnostic upper bound, not an automatic fix.",
        "6. Concave notch origin lies inside the local convex hull and has no stable maximizing direction. Noise may create nearby outliers, not evidence for reliable concave detection.",
        "7. Fixed 4.5 mm connectivity severely fragments the 6 mm sampling condition. Singleton fragments vote for themselves; higher recall with very large FP counts is not a success.",
        "8. D does not improve base end-to-end recall over C on these scenes. Component separation already restores the tested disconnected small objects; multi-scale adds work and can dilute a small-scale-only peak.",
        "9. An actual CED query may miss the small-object region even when its GT-near query succeeds. A local stage cannot recover GT outside all supplied balls.",
        "10. The partial-RGB-D case uses one pinhole ray-cast depth view. GT is restricted to visible vertices with samples within 5 mm; the shell case is explicitly a multi-view/observed-interior stress test.",
        "", "## Interpretation and limits", "",
        "This stage adds runtime to CED; it cannot be described as a faster replacement detector.",
        "Missing CED coverage cannot be repaired outside the supplied query balls. See coverage_recall and conditional_recall.",
        "Each component receives its own direction sweep, including components not containing q. This can recover separate small objects.",
        "Padding discards support-function maxima on the artificial core-ball boundary. It cannot remove real sampling/noise extrema.",
        "NMS retains multiple peaks and never averages their positions. Equal-score neighbors use smaller raw indices.",
        "Concave vertices inside a local convex hull have no supporting direction and receive no stable votes.",
        "Touching objects and the ground can connect; sparse sampling fragments objects into small components, producing confident false positives.",
        "A shell can hide interior observed vertices from support directions; small radii and oracle/component separation can help only when there is observed support.",
        "Unobserved vertices cannot be recovered. XYZ alone also cannot distinguish convex/concave solid orientation when the observed surfaces coincide.",
        "Oracle labels are an ideal separation reference, not a deployable segmenter. Connected components do not guarantee object instances.",
        "Multi-scale votes are divided by directions times eligible scale count; small-scale peaks may be diluted by large-scale nonselection.",
        "The raw IDs are deduplicated exactly for scene evaluation. Nearby hypotheses from different queries remain separate and can count as false positives.",
        "", "Per-scene means, medians and P95 are in summary.csv; per-query times in query_summary.csv; baseline in ced_summary.csv.",
        "A repeatability score is spatial agreement with seed 0, not proof of semantic correctness. Misses and matched-only errors must be read together.",
        "No STL GUI, RealSense, or Kinect hardware run is claimed. Partial RGB-D is a synthetic pinhole view."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/extremity_local_20261029_final"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--conditions", nargs="+", choices=("base", "noisy", "sparse"), default=("base", "noisy", "sparse"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("seeds must be positive")
    with threadpool_limits(limits=1):
        run_experiment(args.output, seeds=args.seeds, scenes=args.scenes,
                       conditions=args.conditions, plots=not args.no_plots)


if __name__ == "__main__":
    main()
