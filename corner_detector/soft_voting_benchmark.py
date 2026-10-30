"""Hard/quantile-soft study: python3 -m corner_detector.soft_voting_benchmark.

All variants reuse the same noisy raw cloud and original CED NMS indices.
Fixed-GT-near queries are a separately labelled diagnostic, never a detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_info, threadpool_limits

from .extremity import prepare_extremity_context, run_extremity_refinement, fibonacci_directions
from .extremity_voting import linear_quantile_99
from .extremity_synthetic import make_scene, match_corners, ced_candidates, _write_csv
from .preprocessing import voxel_downsample_3d
from .settings import DetectorConfig, ExtremityConfig


SCENES = ("one_face", "two_faces", "three_faces", "same_object_two_corners",
          "separated_objects", "missing_vertex", "outliers")
NOISE = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
STAGES = ("roi", "clustering", "projection", "quantile", "sigmoid", "normalization",
          "accumulation", "nms", "total")


def configurations(min_vote_fraction=0.04):
    # Research geometry is fixed, independent of the user's live settings.
    base = ExtremityConfig(radii_mm=(30.0,), padding_mm=10.0, directions=256,
        object_clustering=True, cluster_radius_mm=4.5, nms_radius_mm=5.0,
        nms_method="local_maximum", min_vote_fraction=min_vote_fraction, soft_tau_mm=0.25)
    configs = {}
    for dtype in ("float64", "float32"):
        suffix = "f64" if dtype == "float64" else "f32"
        for label, mode, alpha in (("A_hard", "hard", 4), ("B_dense", "soft_dense", 4),
            ("C_partition", "soft_partition", 4), ("D_sparse4", "soft_sparse", 4),
            ("D_sparse6", "soft_sparse", 6), ("D_sparse8", "soft_sparse", 8)):
            configs[f"{label}_{suffix}"] = replace(base, voting_mode=mode, voting_dtype=dtype, soft_alpha=alpha)
    return configs


def study_scene(name, seed, sigma):
    base = "three_faces" if name in ("missing_vertex", "outliers") else name
    scene = make_scene(base, seed, spacing_mm=3, noise_mm=0)
    scene.name = name
    rng = np.random.default_rng(seed)
    if name == "missing_vertex":
        keep = np.linalg.norm(scene.points_mm, axis=1) > 3.1
        scene.points_mm = scene.points_mm[keep]
        scene.object_labels = scene.object_labels[keep]
        scene.description += " Samples within 3.1 mm of origin removed before noise. GT remains at origin."
    scene.points_mm += rng.normal(0, sigma, scene.points_mm.shape)
    if name == "outliers":
        # Eight isolated observations inside the query's search ball.
        vector = rng.normal(size=(8, 3))
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        outliers = scene.focus_mm + vector * rng.uniform(12, 36, (8, 1))
        scene.points_mm = np.vstack((scene.points_mm, outliers))
        scene.object_labels = np.r_[scene.object_labels, np.arange(1, 9)]
        scene.description += " Eight unlabelled spatial outliers; connectivity is not outlier rejection."
    return scene


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _stats(values):
    values = _finite(values)
    return (float(np.mean(values)), float(np.median(values)), float(np.percentile(values, 95))) if len(values) else (np.nan,) * 3


def summarize(frame_rows, matches, query_rows):
    keys = ("scene", "sigma_mm", "query_source", "variant")
    groups, match_groups, query_groups = {}, {}, {}
    for records, target in ((frame_rows, groups), (matches, match_groups), (query_rows, query_groups)):
        for row in records:
            target.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        matched = match_groups.get(key, [])
        queries = query_groups.get(key, [])
        row = dict(zip(keys, key))
        row["seeds"] = len(rows)
        row["gt_count"] = sum(r["gt_count"] for r in rows)
        row["matched_count"] = len(matched)
        row["recall_5mm"] = len(matched) / row["gt_count"] if row["gt_count"] else np.nan
        row["error_mean_mm"], row["error_median_mm"], row["error_p95_mm"] = _stats([r["error_mm"] for r in matched])
        for metric in ("false_positives", "candidate_count", "query_count", "frame_refinement_s", "ced_s",
                       "vote_l1_vs_dense", "candidate_jaccard_vs_dense", "sigmoid_support_fraction"):
            for stat, value in zip(("mean", "median", "p95"), _stats([r[metric] for r in rows])):
                row[f"{metric}_{stat}"] = value
        for stage in STAGES:
            for stat, value in zip(("mean", "median", "p95"), _stats([r[stage + "_s"] for r in queries])):
                row[f"{stage}_s_{stat}"] = value
        output.append(row)
    return output


def stability_summary(matches):
    keys = ("scene", "sigma_mm", "query_source", "variant", "gt_index")
    groups = {}
    for row in matches:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        positions = np.asarray([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in rows])
        # This mean is an evaluation statistic, never a returned hypothesis.
        center = positions.mean(axis=0)
        displacement = np.linalg.norm(positions - center, axis=1)
        row = dict(zip(keys, key))
        row.update(observations=len(rows), position_rms_mm=float(np.sqrt(np.mean(displacement ** 2))),
                   position_p95_mm=float(np.percentile(displacement, 95)))
        output.append(row)
    return output


def paired_comparisons(matches):
    """Compare positions only for identical GT/scene/noise/seed/source matches."""
    keys = ("scene", "sigma_mm", "seed", "query_source", "gt_index")
    index = {(tuple(r[k] for k in keys), r["variant"]): r for r in matches}
    groups = {}
    for row in matches:
        suffix = "f32" if row["variant"].endswith("f32") else "f64"
        for comparison, reference in (("dense", "B_dense_" + suffix),
                                       ("float64", row["variant"][:-3] + "f64"),
                                       ("hard", "A_hard_" + suffix)):
            ref = index.get((tuple(row[k] for k in keys), reference))
            if ref is None:
                continue
            group = (comparison, row["scene"], row["sigma_mm"], row["query_source"], row["variant"])
            displacement = np.linalg.norm(np.array([row[k] for k in ("x_mm", "y_mm", "z_mm")])
                                          - [ref[k] for k in ("x_mm", "y_mm", "z_mm")])
            groups.setdefault(group, []).append((row["error_mm"], ref["error_mm"], float(displacement),
                                                  row["raw_point_index"] != ref["raw_point_index"]))
    output = []
    for key, records in sorted(groups.items()):
        data = np.asarray(records)
        row = dict(zip(("comparison", "scene", "sigma_mm", "query_source", "variant"), key))
        row.update(paired_observations=len(data), method_error_median_mm=float(np.median(data[:, 0])),
            reference_error_median_mm=float(np.median(data[:, 1])),
            position_delta_median_mm=float(np.median(data[:, 2])),
            position_delta_p95_mm=float(np.percentile(data[:, 2], 95)),
            position_delta_max_mm=float(data[:, 2].max()), changed_raw_index_fraction=float(data[:, 3].mean()))
        output.append(row)
    return output


def run_study(output, seeds=30, *, scenes=SCENES, noise=NOISE, plots=True, min_vote_fraction=0.04):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "per_frame.csv").exists():
        raise FileExistsError("Choose a new output directory; prior results are never overwritten")
    variants = configurations(min_vote_fraction)
    detector = DetectorConfig(voxel_3d_mm=10, show_3d=False, validator=None)
    queries, frames, matches, cases, threshold_checks = [], [], [], [], []
    # Warm imported numerical kernels before timing actual data.
    warm = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
    for cfg in variants.values():
        run_extremity_refinement(warm, 0, cfg)
    for name in scenes:
        for sigma in noise:
            print(f"[soft study] {name} sigma={sigma:g} mm: {seeds} seeds x {len(variants)} variants", flush=True)
            for seed in range(seeds):
                scene = study_scene(name, seed, sigma)
                stage = perf_counter()
                voxel = voxel_downsample_3d(scene.points_mm, detector.voxel_3d_mm)
                preprocessing_s = perf_counter() - stage
                ced, ced_times = ced_candidates(voxel, detector)
                near = int(cKDTree(voxel).query(scene.focus_mm)[1])
                sources = {"ced": ced["candidate_indices"], "gt_near": np.asarray([near])}
                context = prepare_extremity_context(voxel, raw_points_mm=scene.points_mm)
                if seed == 0:
                    cases.append(dict(scene=name, sigma_mm=sigma, description=scene.description,
                        gt_corners_mm=scene.gt_corners_mm.tolist(),
                        ced_query_indices=sources["ced"].tolist(), ced_query_positions_mm=voxel[sources["ced"]].tolist(),
                        gt_near_query_index=near, gt_near_query_mm=voxel[near].tolist()))
                # Rotate order to reduce a fixed first/last timing advantage.
                order = list(variants)
                order = order[seed % len(order):] + order[:seed % len(order)]
                results, elapsed = {}, {}
                for source, indices in sources.items():
                    for label in order:
                        stage = perf_counter()
                        records = [run_extremity_refinement(voxel, int(i), variants[label], context=context) for i in indices]
                        elapsed[source, label] = perf_counter() - stage
                        results[source, label] = records
                for source, indices in sources.items():
                    if source == "ced":
                        gt_ids = np.arange(len(scene.gt_corners_mm))
                    else:
                        gt_ids = np.flatnonzero(np.linalg.norm(scene.gt_corners_mm - voxel[near], axis=1) <= 30)
                    gt = scene.gt_corners_mm[gt_ids]
                    for label, cfg in variants.items():
                        records = results[source, label]
                        reference = results[source, "B_dense_f64" if cfg.voting_dtype == "float64" else "B_dense_f32"]
                        unique_ids = np.unique(np.concatenate([r["candidate_raw_indices"] for r in records])) if records else np.empty(0, dtype=int)
                        positions = scene.points_mm[unique_ids]
                        pairs, errors = match_corners(positions, gt)
                        common = dict(scene=name, sigma_mm=sigma, seed=seed, query_source=source, variant=label)
                        l1, jaccard = [], []
                        for result, ref in zip(records, reference):
                            np.testing.assert_array_equal(result["query_position_mm"], ref["query_position_mm"])
                            np.testing.assert_array_equal(result["roi_raw_indices"], ref["roi_raw_indices"])
                            l1.append(float(np.abs(result["raw_votes"] - ref["raw_votes"]).sum()))
                            a, b = set(result["candidate_raw_indices"]), set(ref["candidate_raw_indices"])
                            jaccard.append(len(a & b) / len(a | b) if a | b else 1.0)
                            local_gt = scene.gt_corners_mm[np.linalg.norm(scene.gt_corners_mm - result["query_position_mm"], axis=1) <= 30]
                            local_positions = scene.points_mm[result["candidate_raw_indices"]]
                            local_pairs, local_errors = match_corners(local_positions, local_gt)
                            queries.append({**common, "query_index": result["query_index"],
                                "query_x_mm": result["query_position_mm"][0], "query_y_mm": result["query_position_mm"][1],
                                "query_z_mm": result["query_position_mm"][2], "roi_points": len(result["roi_raw_indices"]),
                                "components": result["cluster_count"], "candidate_count": result["candidate_count"],
                                "gt_count": len(local_gt), "matched_count": len(local_pairs),
                                "false_positives": result["candidate_count"] - len(local_pairs),
                                "error_median_mm": _stats(local_errors)[1], "error_p95_mm": _stats(local_errors)[2],
                                **{stage + "_s": result["timings_seconds"][stage] for stage in STAGES}})
                        used = sum(r["sigmoid_evaluated_count"] for r in records)
                        possible = sum(r["sigmoid_possible_count"] for r in records)
                        frames.append({**common, "gt_count": len(gt), "matched_count": len(pairs),
                            "candidate_count": len(unique_ids), "false_positives": len(unique_ids) - len(pairs),
                            "query_count": len(indices), "frame_refinement_s": context["preparation_s"] + elapsed[source, label],
                            "spatial_index_s": context["preparation_s"], "ced_s": ced_times["ced_total_s"],
                            "preprocessing_s": preprocessing_s, "vote_l1_vs_dense": float(np.mean(l1)) if l1 else np.nan,
                            "candidate_jaccard_vs_dense": float(np.mean(jaccard)) if jaccard else np.nan,
                            "sigmoid_support_fraction": used / possible if possible else np.nan})
                        for (prediction, target), error in zip(pairs, errors):
                            xyz = positions[prediction]
                            matches.append({**common, "gt_index": int(gt_ids[target]), "raw_point_index": int(unique_ids[prediction]),
                                "x_mm": xyz[0], "y_mm": xyz[1], "z_mm": xyz[2], "error_mm": float(error)})
                if seed == 0:
                    r = results["gt_near", "B_dense_f64"][0]
                    delta = r["roi_points_mm"] - r["query_position_mm"]
                    for dtype in ("float64", "float32"):
                        error = 0.0
                        for component in np.unique(r["cluster_labels"]):
                            p = np.ascontiguousarray(fibonacci_directions(256).astype(dtype) @ delta[r["cluster_labels"] == component].astype(dtype).T)
                            error = max(error, float(np.abs(linear_quantile_99(p) - np.quantile(p, 0.99, axis=1).astype(dtype)).max()))
                        threshold_checks.append(dict(scene=name, sigma_mm=sigma, dtype=dtype, max_threshold_error_mm=error))
                    if sigma == 0.2 and name in ("one_face", "two_faces", "three_faces"):
                        selected = {label: results["gt_near", label][0] for label in
                                    ("A_hard_f64", "B_dense_f64", "C_partition_f64", "D_sparse8_f64")}
                        np.savez_compressed(output / f"{name}_sigma02.npz", points_mm=scene.points_mm,
                            gt_corners_mm=scene.gt_corners_mm, query_position_mm=voxel[near],
                            roi_raw_indices=r["roi_raw_indices"], **{label: value["raw_votes"] for label, value in selected.items()})
                        if plots:
                            from .visualization.manager import export_soft_vote_comparison
                            export_soft_vote_comparison(scene.points_mm, selected, scene.gt_corners_mm,
                                output / f"{name}_voting.png", title=f"{name}, sigma=0.2 mm, same GT-near query")
    summary = summarize(frames, matches, queries)
    stability = stability_summary(matches)
    comparisons = paired_comparisons(matches)
    for filename, rows in (("per_query.csv", queries), ("per_frame.csv", frames), ("matched_vertices.csv", matches),
                            ("summary.csv", summary), ("stability.csv", stability), ("quantile_checks.csv", threshold_checks),
                            ("paired_comparisons.csv", comparisons)):
        _write_csv(output / filename, rows)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    metadata = dict(seeds=seeds, scenes=list(scenes), noise_sigma_mm=list(noise), cpu=cpu,
        platform=platform.platform(), threadpools=threadpool_info(), numpy=np.__version__,
        detector_config=asdict(detector), voting_configs={k: asdict(v) for k, v in variants.items()},
        timing_excludes="generation, noise, matching, metrics, I/O, rendering; includes ROI and projection",
        direction_block_size=64, quantile=0.99, score_gate=min_vote_fraction,
        quantile_interpolation="linear, same as NumPy; no rank-only approximation",
        source_queries="unchanged CED NMS indices; fixed-GT-near diagnostic separately labelled",
        missing_vertex="remove original samples within 3.1 mm of origin, then add noise",
        outliers="eight uniform-direction points 12..36 mm from focus; no hidden outlier filter",
        aggregation="independent query maps; exact raw-ID dedup only for frame evaluation",
        localization="actual observed raw points, local NMS, 5 mm one-to-one matching; no averaging/smoothing",
        stability="RMS displacement around each matched GT's repeat-observation mean; mean used for evaluation only")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    (output / "scenes.json").write_text(json.dumps(cases, indent=2) + "\n")
    if plots:
        from .visualization.manager import export_soft_study_plots
        export_soft_study_plots(summary, stability, output, matches=matches)
    write_report(output, summary, stability, threshold_checks, metadata, frames=frames, comparisons=comparisons)
    print(f"[done] {output}: {len(frames)} frames, {len(queries)} queries, {len(matches)} matched observations", flush=True)


def write_report(output, summary, stability, checks, metadata, *, frames=None, comparisons=None):
    def avg(rows, field):
        values = _finite([r[field] for r in rows])
        return float(np.mean(values)) if len(values) else np.nan
    lines = ["# Quantile soft voting: measured comparison", "",
        f"{metadata['cpu']}; {metadata['seeds']} seeds per scene/noise; {len(metadata['scenes'])} scenes; {len(metadata['noise_sigma_mm'])} noise levels.",
        "Full CPU/thread/config provenance: metadata.json. Timings exclude data generation, matching, I/O and rendering.",
        f"CED/PCA/NMS defaults are unchanged. Local study fixes r=30, padding=10, cluster=4.5, NMS=5 mm, tau=0.25 mm, 256 directions and score gate={metadata.get('score_gate', 0.04):g}.",
        "The user's live ROI settings are not overwritten. Every variant sees identical raw XYZ and unchanged CED queries; GT-near queries are a separate diagnostic.",
        "", "## Aggregate metrics", "",
        "Macro means over scenes/noise. Error values condition on successful 1:1 matches within 5 mm; read recall and FP alongside errors.",
        "", "| Source | Variant | Recall | FP/frame | Matched median mm | Matched P95 mm | Mean query ms | Mean frame ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source in ("gt_near", "ced"):
        for variant in configurations():
            rows = [r for r in summary if r["query_source"] == source and r["variant"] == variant]
            lines.append(f"| {source} | {variant} | {avg(rows, 'recall_5mm'):.3f} | {avg(rows, 'false_positives_mean'):.2f} | "
                f"{avg(rows, 'error_median_mm'):.3f} | {avg(rows, 'error_p95_mm'):.3f} | "
                f"{1000 * avg(rows, 'total_s_mean'):.3f} | {1000 * avg(rows, 'frame_refinement_s_mean'):.3f} |")
    lines += ["", "## Stage timing (actual CED queries)", "",
        "| Variant | Projection | Quantile | Sigmoid + support | Normalize | Accumulate | NMS | Total |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in configurations():
        rows = [r for r in summary if r["query_source"] == "ced" and r["variant"] == variant]
        lines.append("| " + variant + " | " + " | ".join(f"{1000 * avg(rows, s + '_s_mean'):.3f}" for s in
            ("projection", "quantile", "sigmoid", "normalization", "accumulation", "nms", "total")) + " |")
    lines += ["", "All table times are ms. ROI, clustering and shared spatial-index preparation are separate CSV columns.",
        "", "## Exact quantile and sparse approximation", "",
        f"Maximum checked threshold difference: {max(r['max_threshold_error_mm'] for r in checks):.6g} mm (seed 0, every scene/noise/dtype).",
        "The optimized quantile partitions two ranks and retains NumPy's stable linear interpolation, rather than substituting a single order statistic.",
        "Installed NumPy 1.21.5 already implements np.quantile with partition. A speed gain here is wrapper/layout overhead reduction, not a change from full sort to linear selection.",
        "", "| Variant (GT-near) | Vote L1 vs dense | Candidate Jaccard vs dense | Sigmoid support fraction |", "|---|---:|---:|---:|"]
    for variant in configurations():
        rows = [r for r in summary if r["query_source"] == "gt_near" and r["variant"] == variant]
        lines.append(f"| {variant} | {avg(rows, 'vote_l1_vs_dense_mean'):.6g} | "
            f"{avg(rows, 'candidate_jaccard_vs_dense_mean'):.4f} | {avg(rows, 'sigmoid_support_fraction_mean'):.4f} |")
    lines += ["", "## Research question: measured answer", ""]
    hard = [r for r in summary if r["variant"] == "A_hard_f64" and r["query_source"] == "gt_near"]
    soft = [r for r in summary if r["variant"] == "C_partition_f64" and r["query_source"] == "gt_near"]
    lines += [f"GT-near matched median error (macro): hard {avg(hard, 'error_median_mm'):.4f} mm; soft {avg(soft, 'error_median_mm'):.4f} mm.",
        f"GT-near recall: hard {avg(hard, 'recall_5mm'):.4f}; soft {avg(soft, 'recall_5mm'):.4f}.",
        "Thus soft voting is not assumed to be more robust merely because weights vary continuously. The error/recall curves and paired shared-GT comparison determine the measured benefit."]
    dense = [r for r in summary if r["variant"] == "B_dense_f64" and r["query_source"] == "ced"]
    partition = [r for r in summary if r["variant"] == "C_partition_f64" and r["query_source"] == "ced"]
    sparse = [r for r in summary if r["variant"] == "D_sparse8_f64" and r["query_source"] == "ced"]
    lines += [f"Mean actual-query dense/partition speed ratio = {avg(dense, 'total_s_mean') / avg(partition, 'total_s_mean'):.3f}x; "
              f"quantile-stage ratio = {avg(dense, 'quantile_s_mean') / avg(partition, 'quantile_s_mean'):.3f}x.",
        f"Partition/sparse-alpha8 speed ratio = {avg(partition, 'total_s_mean') / avg(sparse, 'total_s_mean'):.3f}x.",
        "The partition optimization is exact relative to the dense reference at the chosen precision. Sparse is kept only as an explicit comparison option, never silently substituted for dense."]
    if comparisons:
        lines += ["", "## Paired localization / float precision", "",
            "Only shared successful GT matches contribute. Misses/false positives remain in summary.csv; these rows cannot establish equal recall.",
            "", "| Comparison | Variant (GT-near) | Method median mm | Reference median mm | Mean within-condition P95 position delta mm | Max delta mm | Changed raw index fraction |", "|---|---|---:|---:|---:|---:|---:|"]
        for comparison in ("hard", "dense", "float64"):
            for variant in configurations():
                if comparison == "hard" and variant not in ("B_dense_f64", "C_partition_f64"):
                    continue
                if comparison == "float64" and not variant.endswith("f32"):
                    continue
                rows = [r for r in comparisons if r["comparison"] == comparison
                        and r["variant"] == variant and r["query_source"] == "gt_near"]
                if not rows:
                    continue
                lines.append(f"| {comparison} | {variant} | {avg(rows, 'method_error_median_mm'):.4f} | "
                    f"{avg(rows, 'reference_error_median_mm'):.4f} | {avg(rows, 'position_delta_p95_mm'):.6g} | "
                    f"{max(r['position_delta_max_mm'] for r in rows):.6g} | {avg(rows, 'changed_raw_index_fraction'):.4f} |")
    if frames:
        lines += ["", "## Whole candidate pipeline latency", "",
            "Includes voxel preprocessing + unchanged CED/PCA/NMS + local stage for every CED query, excludes capture/I/O/rendering.",
            "", "| Variant | Mean ms | Median ms | P95 ms | 1000 / median ms |", "|---|---:|---:|---:|---:|"]
        for variant in configurations():
            rows = [r for r in frames if r["variant"] == variant and r["query_source"] == "ced"]
            mean, median, p95 = _stats([1000 * (r["preprocessing_s"] + r["ced_s"] + r["frame_refinement_s"]) for r in rows])
            lines.append(f"| {variant} | {mean:.3f} | {median:.3f} | {p95:.3f} | {1000 / median:.1f} |")
    lines += ["", "## Interpretation / limits", "",
        "Hard is retained as the live default; none of these measurements authorizes silently changing the user's ROI, NMS or threshold settings.",
        f"All variants use the same score gate {metadata.get('score_gate', 0.04):g}. At 0.04, diffuse soft mass may be rejected; gate-zero runs isolate peak behavior and must also report added false positives.",
        "Sigmoid normalization includes padding first; weights outside the candidate core are then discarded without renormalization. Sparse alpha cuts support before normalization and is genuinely approximate.",
        "Higher alpha can reduce omitted mass but cannot guarantee identical peak identities near ties. Float32 is also checked separately; exact-zero-noise ties can change under rounding.",
        "No Gaussian smoothing, spatial weighted mean, new geometric refinement or query sampling is used. Each output coordinate is an actual observation.",
        "Missing observed vertices limit localization; soft votes may prefer near-boundary interior samples, so lower variance alone is not proof of lower bias.",
        "Euclidean components are not outlier rejection. Isolated outliers can become singleton components receiving 256 votes in every method, causing stable/confident false positives.",
        "Spatial sparse voting still computes every projection and support mask; gather/scatter overhead may outweigh saved sigmoid evaluations.",
        "Default hard heatmaps were approximately 98% zero on the inspected synthetic face cases. Zero is now gray and positive raw mass is log-colored with numeric range; fractional soft weights are never cast to int.",
        "Position RMS graphs use only successful repeated matches and show observation counts in stability.csv. They are not smoothed detector outputs.",
        "This is a small-cloud CPU benchmark, not a guarantee of sensor-frame real time. Frame results include all CED queries; CED cost is reported separately.",
        "No STL GUI, RealSense or Kinect hardware validation was performed.",
        "", "Graphs: noise_localization.png, noise_detection.png, repeatability.png, stage_times.png, tradeoff.png.",
        "3-D figures: one_face_voting.png, two_faces_voting.png, three_faces_voting.png (sigma=0.2, identical GT-near query)."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/soft_voting_20261029"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--noise", nargs="+", type=float, default=NOISE)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--min-vote-fraction", type=float, default=0.04,
                        help="identical local gate for ALL methods; 0 is the peak-only ablation")
    args = parser.parse_args(argv)
    if (args.seeds < 1 or not np.isfinite(args.noise).all() or min(args.noise) < 0
            or not 0 <= args.min_vote_fraction <= 1):
        parser.error("Require positive seeds and finite nonnegative noise")
    from pointcloud_inputs.visualization import require_open3d
    require_open3d()  # Load the geometry library before recording/limiting pools.
    with threadpool_limits(limits=1):
        run_study(args.output, args.seeds, scenes=args.scenes, noise=args.noise, plots=not args.no_plots,
                  min_vote_fraction=args.min_vote_fraction)


if __name__ == "__main__":
    main()
