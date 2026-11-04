"""Whole-scene detector NMS followed by the canonical guided validator."""

from collections import Counter
from dataclasses import replace
from time import perf_counter

from .dense_heatmap import run_dense_heatmap_mode
from .settings import ValidatorDetectorConfig
from .validator_pipeline import prepare_validator_voxel_context, run_validator_candidates


def run_total_mode(points_mm, cfg, *, raw_points_mm):
    """Keep one final selected Corner per seed, without debug comparisons or new merging."""
    if not isinstance(cfg, ValidatorDetectorConfig) or not cfg.validator.geometry_enabled:
        raise ValueError("Total mode requires the guided validator with geometry enabled")
    started = perf_counter()
    context = prepare_validator_voxel_context(points_mm, raw_points_mm, cfg)
    prepared = perf_counter()
    scores = run_dense_heatmap_mode(
        points_mm, replace(cfg, show_3d=False, validator=None),
        point_tree=context["scene_tree"], print_candidates=False)
    detected = perf_counter()
    validation = run_validator_candidates(
        points_mm, scores["candidate_extraction"]["candidate_indices"], cfg,
        voxel_context=context, include_debug=False)
    finished = perf_counter()
    counts = Counter(record["validation_status"] for record in validation["records"]
                     if record["status"] == "ok")
    diagnostics = validation["diagnostics"]
    print(f"[TOTAL] NMS={diagnostics['candidate_count']} | ACCEPT={counts['ACCEPT']} | "
          f"REJECT={counts['REJECT']} | INCONCLUSIVE={counts['INCONCLUSIVE']} | "
          f"errors={diagnostics['failed_count']} | contributing edges={len(validation['final_edges'])}", flush=True)
    print(f"[TOTAL time] prepare={prepared - started:.3f}s | detector={detected - prepared:.3f}s | "
          f"validator={finished - detected:.3f}s | total={finished - started:.3f}s", flush=True)
    for record in validation["records"]:
        if record["status"] != "ok":
            print(f"[TOTAL query {record['query_index']}] {record['error_type']}: {record['error']}", flush=True)
    return {"method": "total_detector_to_guided_validator", "scores": scores,
            "validation": validation, "elapsed_s": finished - started,
            "stage_elapsed_s": {"prepare": prepared - started,
                                "detector": detected - prepared, "validator": finished - detected}}


__all__ = ["run_total_mode"]
