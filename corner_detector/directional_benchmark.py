"""Held-out post-NMS FP study: python3 -m corner_detector.directional_benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_info, threadpool_limits

from .directional_features import FEATURES, SPATIAL_METHODS, candidate_direction_features
from .extremity import fibonacci_directions, prepare_extremity_context, run_extremity_refinement
from .extremity_synthetic import SCENES, SyntheticScene, ced_candidates, make_scene, match_corners
from .preprocessing import voxel_downsample_3d
from .settings import DetectorConfig, ExtremityConfig


STRESS_SCENES = ("smooth_sphere", "smooth_cylinder", "missing_vertex", "outliers")
CONDITIONS = {"clean": (3., 0.), "base": (3., 0.12), "noisy": (3., 0.5), "sparse": (6., 0.12)}
METHODS = ("none",) + FEATURES


@dataclass
class FrameCase:
    meta: dict
    raw_ids: np.ndarray
    points: np.ndarray
    gt: np.ndarray
    features: np.ndarray


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profiles():
    live = ExtremityConfig()
    reference = replace(live, radii_mm=(30.,), padding_mm=10., cluster_radius_mm=4.5,
                        nms_radius_mm=5., nms_method="local_maximum")
    return {"live": live, "reference": reference}


def rotated_scene(name, seed, spacing, noise):
    """Paired random pose and observation noise; explicit observed physical GT."""
    rng = np.random.default_rng(seed + 941)
    if name in ("smooth_sphere", "smooth_cylinder"):
        if name == "smooth_sphere":
            points = 35 * fibonacci_directions(int(np.ceil(4 * np.pi * 35 ** 2 / spacing ** 2)))
            focus = np.array([0., 0., 35.])
        else:
            angle = np.linspace(0, 2 * np.pi, int(np.ceil(2 * np.pi * 30 / spacing)), endpoint=False)
            a, z = np.meshgrid(angle, np.linspace(-36, 36, int(np.ceil(72 / spacing)) + 1))
            points = np.column_stack((30 * np.cos(a.ravel()), 30 * np.sin(a.ravel()), z.ravel()))
            focus = np.array([30., 0., 0.])
        scene = SyntheticScene(name, points.copy(), np.zeros(len(points), dtype=int), np.empty((0, 3)),
                               focus, "Smooth observed surface: no physical corner GT; cylinder rims are curves.")
    else:
        scene = make_scene("three_faces" if name in ("missing_vertex", "outliers") else name,
                           seed, spacing_mm=spacing, noise_mm=0)
        scene.name = name
        if name == "missing_vertex":
            keep = np.linalg.norm(scene.points_mm, axis=1) > 3.1
            scene.points_mm, scene.object_labels = scene.points_mm[keep], scene.object_labels[keep]
            scene.description += " Origin samples removed within 3.1 mm; GT is unchanged."
        if name == "outliers":
            vector = rng.normal(size=(8, 3))
            vector /= np.linalg.norm(vector, axis=1, keepdims=True)
            extra = scene.focus_mm + vector * rng.uniform(12, 36, (8, 1))
            scene.points_mm = np.vstack((scene.points_mm, extra))
            scene.object_labels = np.r_[scene.object_labels, np.arange(1, 9)]
            scene.description += " Eight unlabelled observed outliers added in the focus search ball."
    scene.points_mm += np.random.default_rng(seed).normal(0, noise, scene.points_mm.shape)
    rotation, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation *= np.sign(np.diag(triangular))
    rotation[:, 0] *= np.linalg.det(rotation)
    scene.points_mm = scene.points_mm @ rotation.T
    scene.gt_corners_mm = scene.gt_corners_mm @ rotation.T
    scene.focus_mm = scene.focus_mm @ rotation.T
    return scene


def merge_query_features(raw_points, records):
    """A raw point survives frame display if any independent query retains it."""
    if not records or not any(r[0]["candidate_count"] for r in records):
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty((0, len(FEATURES)))
    ids = np.concatenate([r[0]["candidate_raw_indices"] for r in records])
    values = np.vstack([np.column_stack([r[1][name] for name in FEATURES]) for r in records])
    unique, inverse = np.unique(ids, return_inverse=True)
    merged = np.zeros((len(unique), len(FEATURES)))
    np.maximum.at(merged, inverse, values)
    return unique, raw_points[unique].copy(), merged


def collect_cases(output, *, seeds, calibration_seeds, validation_seeds, names, conditions, selected_profiles):
    detector = DetectorConfig(show_3d=False, validator=None)
    cases, queries, manifest = [], [], []
    for name in names:
        for condition in conditions:
            spacing, noise = CONDITIONS[condition]
            first = calibration_seeds + validation_seeds if name in STRESS_SCENES else 0
            print(f"[direction study] {name} / {condition}: seeds {first}..{seeds - 1}", flush=True)
            for seed in range(first, seeds):
                split = ("stress_test" if name in STRESS_SCENES else "calibration" if seed < calibration_seeds
                         else "validation" if seed < calibration_seeds + validation_seeds else "test")
                scene = rotated_scene(name, seed, spacing, noise)
                started = perf_counter()
                voxel = voxel_downsample_3d(scene.points_mm, detector.voxel_3d_mm)
                preprocessing = perf_counter() - started
                ced, ced_times = ced_candidates(voxel, detector)
                near = int(cKDTree(voxel).query(scene.focus_mm)[1])
                sources = {"ced": ced["candidate_indices"], "gt_near": np.array([near])}
                context = prepare_extremity_context(voxel, raw_points_mm=scene.points_mm)
                if seed == first:
                    manifest.append(dict(scene=name, condition=condition, seed=seed,
                        description=scene.description, gt_corners_mm=scene.gt_corners_mm.tolist(),
                        ced_query_indices=sources["ced"].tolist(),
                        ced_queries_mm=voxel[sources["ced"]].tolist(), gt_near_query_mm=voxel[near].tolist()))
                for label, cfg in selected_profiles.items():
                    cache = {}
                    for source, indices in sources.items():
                        records = []
                        for query in indices:
                            query = int(query)
                            if query not in cache:
                                record = run_extremity_refinement(voxel, query, cfg, context=context)
                                started = perf_counter()
                                features, detail = candidate_direction_features(record)
                                suite_s = perf_counter() - started
                                cache[query] = record, features, detail, suite_s
                            record, features, detail, suite_s = cache[query]
                            records.append((record, features))
                            queries.append(dict(scene=name, condition=condition, seed=seed, split=split,
                                profile=label, query_source=source, query_index=query,
                                candidate_count=record["candidate_count"], refinement_s=record["timings_seconds"]["total"],
                                trace_s=detail["trace_s"], features_s=detail["features_s"], feature_suite_s=suite_s))
                        raw_ids, positions, values = merge_query_features(scene.points_mm, records)
                        gt = scene.gt_corners_mm
                        if source == "gt_near":
                            gt = gt[np.linalg.norm(gt - voxel[near], axis=1) <= max(cfg.radii_mm) + 5]
                        meta = dict(scene=name, condition=condition, seed=seed, split=split, profile=label,
                            query_source=source, query_count=len(indices), raw_point_count=len(scene.points_mm),
                            voxel_point_count=len(voxel), gt_scene_count=len(scene.gt_corners_mm),
                            ced_s=ced_times["ced_total_s"], preprocessing_s=preprocessing,
                            context_s=context["preparation_s"],
                            refinement_s=sum(r["timings_seconds"]["total"] for r, _ in records),
                            trace_s=sum(cache[int(q)][2]["trace_s"] for q in indices),
                            features_s=sum(cache[int(q)][2]["features_s"] for q in indices),
                            feature_suite_s=sum(cache[int(q)][3] for q in indices))
                        cases.append(FrameCase(meta, raw_ids, positions, gt.copy(), values))
    _csv(output / "per_query.csv", queries)
    (output / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))
    offsets = np.r_[0, np.cumsum([len(c.points) for c in cases])]
    gt_offsets = np.r_[0, np.cumsum([len(c.gt) for c in cases])]
    np.savez_compressed(output / "cases.npz", offsets=offsets, gt_offsets=gt_offsets,
        points=np.vstack([c.points for c in cases]), gt=np.vstack([c.gt for c in cases]),
        raw_ids=np.concatenate([c.raw_ids for c in cases]), features=np.vstack([c.features for c in cases]),
        feature_names=np.array(FEATURES), meta_json=np.array([json.dumps(c.meta) for c in cases]))
    return cases, queries, detector


def evaluate_case(case, method="none", threshold=None):
    keep = (np.ones(len(case.points), dtype=bool) if threshold is None or method == "none"
            else case.features[:, FEATURES.index(method)] >= threshold)
    pairs, errors = match_corners(case.points[keep], case.gt)
    return len(pairs), int(keep.sum()) - len(pairs), int(keep.sum()), errors


def totals(cases, method="none", threshold=None):
    measured = [evaluate_case(c, method, threshold) for c in cases]
    return dict(matches=sum(r[0] for r in measured), false_positives=sum(r[1] for r in measured),
                candidates=sum(r[2] for r in measured), gt=sum(len(c.gt) for c in cases), frames=len(cases))


def fit_threshold(cases, method, retain=0.95):
    """Largest observed cutoff preserving the specified calibration 1:1 matches."""
    baseline = totals(cases)
    if baseline["matches"] == 0:
        return None
    values = np.unique(np.concatenate([c.features[:, FEATURES.index(method)] for c in cases]))
    values = np.r_[values, np.nextafter(values[-1], np.inf)]
    required = int(np.ceil(retain * baseline["matches"]))
    low, high = -1, len(values)
    while high - low > 1:
        middle = (low + high) // 2
        if totals(cases, method, float(values[middle]))["matches"] >= required:
            low = middle
        else:
            high = middle
    return None if low < 0 else float(values[low])


def choose_validation_method(rows, allowed, retain=0.95):
    safe = [r for r in rows if r["method"] in allowed and r["validation_retention"] >= retain]
    if not safe:
        return "none"
    return min(safe, key=lambda r: (r["validation_fp"], -r["validation_retention"], r["method"]))["method"]


def tune(cases, output, retain):
    thresholds, tuning, selected = {}, [], {}
    for profile in sorted({c.meta["profile"] for c in cases}):
        train = [c for c in cases if c.meta["profile"] == profile and c.meta["split"] == "calibration"
                 and c.meta["query_source"] == "ced"]
        validation = [c for c in cases if c.meta["profile"] == profile and c.meta["split"] == "validation"
                      and c.meta["query_source"] == "ced"]
        train_base, val_base = totals(train), totals(validation)
        thresholds[profile] = {"none": None}
        for method in METHODS:
            threshold = None if method == "none" else fit_threshold(train, method, retain)
            thresholds[profile][method] = threshold
            calibrated, validated = totals(train, method, threshold), totals(validation, method, threshold)
            tuning.append(dict(profile=profile, method=method, threshold=threshold,
                calibration_retention=calibrated["matches"] / train_base["matches"] if train_base["matches"] else 1.,
                calibration_fp=calibrated["false_positives"],
                validation_retention=validated["matches"] / val_base["matches"] if val_base["matches"] else 1.,
                validation_fp=validated["false_positives"], validation_matches=validated["matches"]))
        candidates = [r for r in tuning if r["profile"] == profile]
        selected[profile] = dict(best_overall=choose_validation_method(candidates,
            ("none",) + tuple(m for m in FEATURES if not m.startswith("shuffled_")), retain),
            best_spatial=choose_validation_method(candidates, ("none",) + SPATIAL_METHODS, retain))
    _csv(output / "calibration_validation.csv", tuning)
    (output / "frozen_thresholds.json").write_text(json.dumps(
        dict(retain_target=retain, thresholds=thresholds, validation_selected=selected), indent=2))
    return thresholds, selected


def evaluate(cases, thresholds):
    rows = []
    for case in cases:
        base_matches = evaluate_case(case)[0]
        for method in METHODS:
            matched, fp, count, errors = evaluate_case(case, method, thresholds[case.meta["profile"]][method])
            rows.append({**case.meta, "method": method, "gt_count": len(case.gt), "matched_count": matched,
                "baseline_matched_count": base_matches, "false_positives": fp, "candidate_count": count,
                "error_median_mm": float(np.median(errors)) if len(errors) else np.nan,
                "error_p95_mm": float(np.percentile(errors, 95)) if len(errors) else np.nan})
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        for scene, condition in ((row["scene"], row["condition"]), (row["scene"], "ALL"),
                                  ("ALL", row["condition"]), ("ALL", "ALL")):
            key = (row["split"], row["profile"], row["query_source"], row["method"], scene, condition)
            groups.setdefault(key, []).append(row)
    summary = []
    for key, records in sorted(groups.items()):
        row = dict(zip(("split", "profile", "query_source", "method", "scene", "condition"), key))
        matched = sum(r["matched_count"] for r in records)
        baseline = sum(r["baseline_matched_count"] for r in records)
        gt = sum(r["gt_count"] for r in records)
        candidates = sum(r["candidate_count"] for r in records)
        row.update(frames=len(records), seeds=len({r["seed"] for r in records}), gt_count=gt,
            matched_count=matched, recall=matched / gt if gt else np.nan,
            recall_retention=matched / baseline if baseline else np.nan,
            precision=matched / candidates if candidates else np.nan)
        for metric in ("false_positives", "candidate_count", "refinement_s", "trace_s", "features_s",
                       "feature_suite_s", "ced_s", "error_median_mm", "error_p95_mm"):
            values = np.array([r[metric] for r in records], dtype=float)
            values = values[np.isfinite(values)]
            for name, value in zip(("mean", "median", "p95"),
                    (np.mean(values), np.median(values), np.percentile(values, 95)) if len(values) else (np.nan,) * 3):
                row[f"{metric}_{name}"] = float(value)
        summary.append(row)
    return summary


def paired_intervals(rows, selected, repetitions=2000):
    """Seed-cluster bootstrap keeps all paired scenes/conditions for each seed."""
    output = []
    for profile, choice in selected.items():
        for split in ("test", "stress_test"):
            for source in ("ced", "gt_near"):
                for method in set((choice["best_overall"], choice["best_spatial"], "vote_score")):
                    data = [r for r in rows if r["profile"] == profile and r["split"] == split
                            and r["query_source"] == source and r["method"] in ("none", method)]
                    seeds = sorted({r["seed"] for r in data})
                    if not seeds:
                        continue
                    matrix = np.zeros((len(seeds), 4))
                    for i, seed in enumerate(seeds):
                        for r in data:
                            if r["seed"] == seed:
                                side = 0 if r["method"] == "none" else 2
                                matrix[i, side:side + 2] += (r["matched_count"], r["false_positives"])
                        if method == "none":
                            matrix[i, 2:] = matrix[i, :2]
                    draws = np.random.default_rng(20261029).integers(0, len(seeds), (repetitions, len(seeds)))
                    sums = matrix[draws].sum(axis=1)
                    keep = np.divide(sums[:, 2], sums[:, 0], out=np.full(repetitions, np.nan), where=sums[:, 0] > 0)
                    reduction = 1 - np.divide(sums[:, 3], sums[:, 1], out=np.full(repetitions, np.nan), where=sums[:, 1] > 0)
                    row = dict(profile=profile, split=split, query_source=source, method=method, seeds=len(seeds))
                    for label, values in (("recall_retention", keep), ("fp_reduction", reduction)):
                        values = values[np.isfinite(values)]
                        lo, hi = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
                        row.update({label + "_ci_low": float(lo), label + "_ci_high": float(hi)})
                    output.append(row)
    return output


def run_study(output, *, seeds=60, calibration_seeds=20, validation_seeds=10,
              names=SCENES + STRESS_SCENES, conditions=tuple(CONDITIONS), profile_names=("live", "reference"),
              retain=0.95, plots=True):
    if min(calibration_seeds, validation_seeds) < 1 or seeds <= calibration_seeds + validation_seeds:
        raise ValueError("Calibration, validation and test require disjoint nonempty seed sets")
    if not 0 < retain <= 1:
        raise ValueError("retain must lie in (0, 1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; previous research results are preserved")
    configs = {k: v for k, v in profiles().items() if k in profile_names}
    metadata = dict(seeds=seeds, calibration_seeds=calibration_seeds, validation_seeds=validation_seeds,
        test_seeds=seeds - calibration_seeds - validation_seeds, scenes=list(names),
        stress_scenes=STRESS_SCENES, conditions={k: CONDITIONS[k] for k in conditions},
        profiles={k: asdict(v) for k, v in configs.items()}, numpy=np.__version__,
        platform=platform.platform(), processor=platform.processor(), retain_target=retain,
        cpu=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown"),
        scope="Post-NMS pruning only; exact raw-ID deduplication; no new queries or score/NMS changes")
    with threadpool_limits(limits=1):
        metadata["threadpools"] = threadpool_info()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        cases, queries, detector = collect_cases(output, seeds=seeds, calibration_seeds=calibration_seeds,
            validation_seeds=validation_seeds, names=names, conditions=conditions, selected_profiles=configs)
        metadata["detector"] = asdict(detector)
        thresholds, selected = tune(cases, output, retain)
        # All test metrics are computed only after thresholds and selection are frozen on disk.
        rows = evaluate(cases, thresholds)
        summary = summarize(rows)
        intervals = paired_intervals(rows, selected)
        _csv(output / "per_frame.csv", rows)
        _csv(output / "summary.csv", summary)
        _csv(output / "paired_seed_intervals.csv", intervals)
        metadata.update(frames=len(cases), query_records=len(queries), baseline_candidates=sum(len(c.points) for c in cases))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if plots:
            from .visualization.manager import export_directional_study_plots
            export_directional_study_plots(summary, selected, output)
    print(json.dumps(dict(selected=selected, output=str(output), frames=len(cases)), indent=2), flush=True)
    return summary, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--calibration-seeds", type=int, default=20)
    parser.add_argument("--validation-seeds", type=int, default=10)
    parser.add_argument("--retain", type=float, default=0.95)
    parser.add_argument("--scenes", nargs="+", choices=SCENES + STRESS_SCENES, default=SCENES + STRESS_SCENES)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=tuple(CONDITIONS))
    parser.add_argument("--profiles", nargs="+", choices=("live", "reference"), default=("live", "reference"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run_study(args.output, seeds=args.seeds, calibration_seeds=args.calibration_seeds,
              validation_seeds=args.validation_seeds, names=args.scenes, conditions=args.conditions,
              profile_names=args.profiles, retain=args.retain, plots=not args.no_plots)


if __name__ == "__main__":
    main()
