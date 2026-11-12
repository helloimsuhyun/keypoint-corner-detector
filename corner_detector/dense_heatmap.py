"""Whole-cloud CED/PCA scoring, selectable bilateral stage and candidates."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from .candidate_extraction import extract_corner_candidates
from .edge_suppression import dense_ced_weighted_pca_scores
from .q10_validator import run_q10_validator
from .validator_pipeline import prepare_validator_seeds, run_validator_candidates
from .visualization.manager import show_dense_heatmaps, show_detector_validator_results


def _score_summary(values, valid_mask):
    selected = np.asarray(values, dtype=np.float64)[valid_mask]
    if len(selected) == 0:
        return "none"
    return (
        f"min={np.min(selected):.4f} "
        f"median={np.median(selected):.4f} "
        f"max={np.max(selected):.4f}"
    )


def run_dense_heatmap_mode(
    points_mm,
    cfg,
    *,
    rgb_points_mm=None,
    rgb_colors_rgb=None,
    validator_estimator=None,
    organized_context=None,
    point_tree=None,
    print_candidates=True,
    raw_points_mm=None,
    raw_point_tree=None,
):
    """Score points, run NMS seeds through validator, and show final geometry."""
    q10_cfg = getattr(cfg, "q10_validator", None)
    if q10_cfg is not None:
        q10_cfg.validate()
        if raw_points_mm is None:
            raise ValueError("Q10 Lite requires the original raw_points_mm, not voxel points")
    points = np.asarray(points_mm, dtype=np.float64)
    started = time.perf_counter()
    point_tree = cKDTree(points) if point_tree is None else point_tree
    scores = dense_ced_weighted_pca_scores(
        points,
        cfg.centroid_radius_mm,
        cfg.pca_radius_mm,
        cfg.min_neighbors,
        point_tree=point_tree,
        bilateral_radius_mm=cfg.bilateral_radius_mm,
        bilateral_sigma_spatial_mm=cfg.bilateral_sigma_spatial_mm,
        bilateral_sigma_ced=cfg.bilateral_sigma_ced,
        bilateral_mode=getattr(cfg, "bilateral_mode", "ced"),
        pca_ratio_saturation_tau=cfg.pca_ratio_saturation_tau,
        pca_ced_cutoff=getattr(cfg, "pca_ced_cutoff", 0.0),
        fusion_mode=getattr(cfg, "fusion_mode", "pca_ratio"),
        local_minimum_tau=getattr(cfg, "local_minimum_tau", .005),
    )
    ced_valid = scores["ced_valid_mask"]
    pca_valid = scores["pca_valid_mask"]
    score_elapsed = time.perf_counter() - started
    extraction_started = time.perf_counter()
    extraction = extract_corner_candidates(
        points,
        scores,
        cfg.nms_radius_mm,
        fusion_score_threshold=cfg.fusion_score_threshold,
        fusion_threshold_mode=getattr(cfg, "fusion_threshold_mode", "fixed"),
        point_tree=point_tree,
        post_nms_lambda1_ratio_max=getattr(cfg, "post_nms_lambda1_ratio_max", None),
    )
    scores["candidate_extraction"] = extraction
    gate = extraction.get("post_nms_lambda1_gate")
    if gate is not None and gate["enabled"]:
        print(f"[POST-NMS lambda1/R²] NMS={len(gate['nms_indices'])} "
              f"kept={extraction['candidate_count']} rejected={len(gate['rejected_indices'])} "
              f"invalid-weighted-PCA={np.count_nonzero(gate['invalid_mask'])} "
              f"| <= {gate['threshold']:g}", flush=True)
    extraction_elapsed = time.perf_counter() - extraction_started
    lite = None
    if q10_cfg is not None:
        lite = run_q10_validator(points, extraction, cfg, q10_cfg,
                                 raw_points_mm=raw_points_mm, raw_tree=raw_point_tree)
        scores["q10_validator"] = lite
    if lite is not None:
        scores["lite_validator"] = lite
    seeds = scores["validator_seeds"] = prepare_validator_seeds(points, extraction, lite)
    validator_indices = seeds["candidate_indices"]
    validation = None
    validator_cfg = None if getattr(cfg, "only", False) else getattr(cfg, "validator", None)
    if validator_cfg is not None:
        validator_kwargs = {}
        if organized_context is not None:
            validator_kwargs["organized_context"] = organized_context
        if lite is not None:
            validator_kwargs["query_positions_mm"] = seeds["positions_mm"]
        validation = run_validator_candidates(
            points, validator_indices, validator_cfg,
            validator_estimator,
            **validator_kwargs,
        )
        scores["validator_results"] = validation
        records_by_query = {
            int(record["query_index"]): record for record in validation["records"]
        }
        for candidate in extraction["candidates"] if lite is None else ():
            record = records_by_query[int(candidate["point_index"])]
            candidate["validator_result"] = record
            candidate["status"] = (
                "validator_complete" if record["status"] == "ok"
                else "validator_error"
            )
    print(
        f"[HEATMAP] points={len(points):,} "
        f"CED-valid={np.count_nonzero(ced_valid):,} "
        f"PCA-queries={np.count_nonzero(scores['pca_query_mask']):,} "
        f"PCA-valid={np.count_nonzero(pca_valid):,} "
        f"invalid-support={np.count_nonzero(scores['invalid_support_mask']):,} "
        f"NMS-eligible={np.count_nonzero(scores['fusion_valid_mask']):,} "
        f"score-elapsed={score_elapsed:.2f}s",
        flush=True,
    )
    if validation is not None:
        diagnostics = validation["diagnostics"]
        print(
            f"[NMS -> VALIDATOR] candidates={diagnostics['candidate_count']:,} "
            f"completed={diagnostics['completed_count']:,} "
            f"failed={diagnostics['failed_count']:,} "
            f"final-edges={diagnostics['final_edge_count']:,} "
            f"final-corners={diagnostics['final_corner_count']:,} "
            f"elapsed={diagnostics['elapsed_s']:.2f}s",
            flush=True,
        )
        for record in validation["records"]:
            if record["status"] != "ok":
                print(
                    f"  validator query={record['query_index']} failed: "
                    f"{record['error_type']}: {record['error']}", flush=True,
                )
    stage = scores["stage_elapsed_s"]
    print(
        f"  elapsed: CED={stage['ced']:.2f}s "
        f"bilateral={stage['bilateral']:.2f}s "
        f"PCA={stage['pca']:.2f}s NMS={extraction_elapsed:.2f}s",
        flush=True,
    )
    print(
        f"[CANDIDATES] total={extraction['total_point_count']:,} "
        f"valid-score={extraction['valid_score_count']:,} "
        f"after-NMS={len(extraction.get('nms_candidate_indices', extraction['candidate_indices'])):,} "
        f"retained={extraction['candidate_count']:,} | "
        f"threshold={extraction['fusion_threshold_mode']} "
        f"score>{extraction['fusion_score_threshold']:.6g} | "
        f"NMS={extraction['nms_method']} | "
        f"NMS-radius={extraction['nms_radius_mm']:g} mm "
        f"elapsed={extraction_elapsed:.2f}s",
        flush=True,
    )
    for label, key, valid in (
        ("Original CED", "ced_score", ced_valid),
        ("Bilateral CED" if scores["bilateral_mode"] == "ced" else "CED (bypass)", "bilateral_ced_score", ced_valid),
        ("PCA Raw", "pca_ratio_raw", pca_valid),
        ("PCA Saturated", "pca_ratio_saturated", pca_valid),
        ("Fusion Score", "fusion_score", scores["fusion_valid_mask"]),
    ):
        print(f"  {label:12s} {_score_summary(scores[key], valid)}", flush=True)
    if scores.get("fusion_mode") == "local_minimum":
        print(f"  Local Minimum {_score_summary(scores['local_minimum_score'], scores['local_minimum_valid_mask'])} "
              f"| Normal PCA lambda1 | tau={scores['local_minimum_tau']:g} "
              f"| elapsed={scores['stage_elapsed_s']['local_minimum']:.2f}s", flush=True)
    print(
        f"  CED-radius={scores['ced_radius_mm']:g} mm "
        f"bilateral=(mode={scores['bilateral_mode']}, radius={scores['bilateral_radius_mm']:g} mm, "
        f"sigma-s={scores['bilateral_sigma_spatial_mm']:g} mm, "
        f"sigma-CED={scores['bilateral_sigma_ced']:g}) "
        f"PCA-radius={scores['pca_radius_mm']:g} mm "
        f"PCA-CED-cutoff={scores['pca_ced_cutoff']:g} "
        f"PCA-saturation-tau={scores['pca_ratio_saturation_tau']} "
        f"eigenvalue-epsilon={scores['eigenvalue_epsilon_mm2']:.3e} mm^2",
        flush=True,
    )
    for candidate in extraction["candidates"] if print_candidates else ():
        position = np.asarray(candidate["position_mm"], dtype=np.float64)
        print(
            f"  candidate index={candidate['point_index']} "
            f"xyz-mm={np.round(position, 6).tolist()} "
            f"CED={candidate['ced_score']:.6g} "
            f"bilateral={candidate['bilateral_ced_score']:.6g} "
            f"PCA-raw={candidate['pca_ratio_raw']:.6g} "
            f"PCA-saturated={candidate['pca_ratio_saturated']:.6g} "
            f"tau={candidate['pca_ratio_saturation_tau']} "
            f"fusion={candidate['fusion_score']:.6g} "
            f"status={candidate['status']}",
            flush=True,
        )
    if cfg.show_3d:
        show_dense_heatmaps(
            points,
            scores,
            cfg,
            rgb_points_mm=rgb_points_mm,
            rgb_colors_rgb=rgb_colors_rgb,
        )
        if validation is not None:
            show_detector_validator_results(points, validation, cfg)
    return scores




__all__ = ["run_dense_heatmap_mode"]
