"""Cached Q10 relocation and cleaned per-plane angular-span gates with optional CC."""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .preprocessing import voxel_downsample_3d
from .refiner import Q10PositionRefiner
from .spherical_antipodal import segment_origin_spherical_planes, evaluate_plane_max_gap


KIND = "q10_antipodal"


def _final_evidence(position, refiner, cfg):
    """Lab great-circle RANSAC and cleaned MaxGap, without a final spatial CC."""
    roi = np.sort(np.asarray(refiner.tree.query_ball_point(position, cfg.radius_mm + 1e-8), dtype=np.int64))
    offsets = refiner.points[roi] - position
    lengths = np.linalg.norm(offsets, axis=1)
    nonzero = lengths > 1e-7
    ids = roi[nonzero]
    evidence = {"roi_analysis_indices": roi, "vector_analysis_indices": ids,
                "geometry_analysis_indices": np.empty(0, dtype=np.int64),
                "component_analysis_indices": roi.copy(), "selected_component_mask": np.ones(len(roi), dtype=bool),
                "connected_analysis_mask": np.ones(len(ids), dtype=bool), "cc_evaluated": False,
                "median_nn_mm": None, "cc_radius_mm": None, "antipodal": None,
                "spherical_planes": None, "plane_antipodal": None}
    if len(ids) < 3:
        return evidence, "insufficient_spherical_directions"
    units = offsets[nonzero] / lengths[nonzero, None]
    try:
        planes = segment_origin_spherical_planes(
            units, k=cfg.sphere_max_planes, tolerance_deg=cfg.sphere_tolerance_deg,
            iterations=cfg.sphere_iterations, min_support=cfg.sphere_min_support,
            min_fraction=cfg.sphere_min_fraction, seed=cfg.sphere_seed)
        validation = evaluate_plane_max_gap(
            planes, max_link_gap_deg=cfg.gap_link_deg, edge_span_deg=cfg.gap_edge_span_deg,
            tail_filter_enabled=cfg.gap_tail_enabled, tail_bin_width_deg=cfg.gap_tail_bin_width_deg,
            tail_density_fraction=cfg.gap_tail_density_fraction, tail_min_bin_count=cfg.gap_tail_min_bin_count)
    except np.linalg.LinAlgError:
        return evidence, "spherical_plane_fit_failed"
    evidence.update(spherical_planes=asdict(planes), plane_antipodal=asdict(validation),
                    geometry_analysis_indices=ids[planes.membership_count > 0])
    if validation.tests:
        # A rejecting plane wins even if an unreliable plane has a wider span.
        best = max(validation.tests, key=lambda t: (t.edge_like, t.reliable, t.observed_span_deg,
                                                    t.kept_count, -t.plane_index))
        plane = planes.planes[best.plane_index]
        retained, discarded = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
        retained[plane.inlier_mask] = best.kept_mask
        discarded[plane.inlier_mask] = ~best.kept_mask
        evidence["antipodal"] = {"mode": "sphere_max_gap", "plane_index": best.plane_index,
                                 "max_gap_deg": best.max_gap_deg, "observed_span_deg": best.observed_span_deg,
                                 "raw_count": best.raw_count, "kept_count": best.kept_count,
                                 "discarded_count": best.discarded_count, "reliable": best.reliable,
                                 "retained_mask": retained, "discarded_mask": discarded}
    return evidence, "insufficient_spherical_plane_support" if validation.status == "UNKNOWN" else None


def run_q10_validator(points_mm, extraction, detector_cfg, cfg, *, raw_points_mm, raw_tree=None):
    cfg.validate()
    started = perf_counter()
    scene, raw = np.asarray(points_mm, dtype=float), np.asarray(raw_points_mm, dtype=float)
    if any(p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() for p in (scene, raw)):
        raise ValueError("Q10 scene/raw points must be finite (N,3)")
    indices = np.asarray(extraction["candidate_indices"])
    candidates = extraction["candidates"]
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(scene)) or len(np.unique(indices)) != len(indices)
            or len(candidates) != len(indices)
            or any(c["point_index"] != i or not np.isfinite(c["score"]) for c, i in zip(candidates, indices))):
        raise ValueError("Q10 candidates must preserve aligned unique scene IDs and finite scores")
    if len(indices) and not len(raw):
        raise ValueError("Q10 candidates require a nonempty raw cloud")
    analysis = voxel_downsample_3d(raw, cfg.voxel_mm)
    refiner = Q10PositionRefiner(analysis, cfg.radius_mm, detector_cfg.nms_radius_mm, cfg.cc_factor,
                                 cfg.refine_cc_enabled, cfg.direction_mode)
    records, final_cache = [], {}
    last_progress = started
    for index, candidate in zip(indices, candidates):
        t = perf_counter()
        if cfg.refine_enabled:
            start = int(refiner.tree.query(scene[index], k=1)[1])
            trace = refiner.refine(start, center_mm=scene[index])
        else:
            # The exact NMS coordinate need not be a sample in the analysis cloud.
            trace = {"analysis_indices": np.empty(0, dtype=np.int64), "positions_mm": scene[index][None, :].copy(),
                     "q10_values": np.array([np.nan]), "neighbor_counts": np.array([0], dtype=np.int64),
                     "mean_axes": np.full((1, 3), np.nan), "tested_positions": np.array([0], dtype=np.int64),
                     "support_analysis_indices": [np.empty(0, dtype=np.int64)], "roi_counts": np.array([0]),
                     "cc_median_nn_mm": np.array([np.nan]), "cc_radii_mm": np.array([np.nan]),
                     "converged": False, "stop_reason": "refine_disabled"}
        final_id = int(trace["analysis_indices"][-1]) if len(trace["analysis_indices"]) else None
        position = trace["positions_mm"][-1].copy()
        record = {"kind": KIND, "candidate_id": int(index), "point_index": int(index),
                  "position": scene[index].copy(), "refined_position_mm": position,
                  "refined_analysis_index": final_id, "score": float(candidate["score"]),
                  "fusion_score": float(candidate["score"]), "trace": trace,
                  "radius_mm": cfg.radius_mm, "step_mm": detector_cfg.nms_radius_mm,
                  "antipodal_mode": "sphere_max_gap", "direction_mode": cfg.direction_mode,
                  "refine_enabled": cfg.refine_enabled,
                  "refine_cc_enabled": cfg.refine_enabled and cfg.refine_cc_enabled,
                  "q10_value": None, "min_q10": cfg.min_q10, "q10_reject": None, "antipodal_reject": None,
                  "rejection_reasons": [],
                  "status": "INDETERMINATE", "accept": None, "keep": True,
                  "antipodal": None, "antipodal_evaluated": False,
                  "reason": trace["stop_reason"], "cc_evaluated": False}
        if not cfg.refine_enabled or trace["converged"]:
            quality = float(trace["q10_values"][-1]) if cfg.refine_enabled else None
            q10_reject = bool(quality < cfg.min_q10) if quality is not None and np.isfinite(quality) else None
            record.update(q10_value=quality, q10_reject=q10_reject)
            if q10_reject is True:
                record.update(status="REJECT", accept=False, keep=False, reason="q10_below_threshold",
                              rejection_reasons=["q10_below_threshold"])
            elif cfg.refine_enabled and q10_reject is None:
                record["reason"] = "invalid_final_q10"
            else:
                cache_key = final_id if cfg.refine_enabled else tuple(position)
                if cache_key not in final_cache:
                    final_cache[cache_key] = _final_evidence(position, refiner, cfg)
                evidence, reason = final_cache[cache_key]
                if not cfg.refine_enabled:
                    trace["roi_counts"][0] = len(evidence["roi_analysis_indices"])
                record.update(evidence, antipodal_evaluated=True)
                if reason is None:
                    rejected = record["plane_antipodal"]["status"] == "REJECT"
                    record.update(antipodal_reject=rejected, status="REJECT" if rejected else "ACCEPT",
                                  accept=not rejected, keep=not rejected,
                                  rejection_reasons=["angular_span"] if rejected else [],
                                  reason="angular_span" if rejected else "compact_angular_span")
                else:
                    record["reason"] = reason
        record["elapsed_s"] = perf_counter() - t
        records.append(record)
        if perf_counter() - last_progress >= 5.:
            print(f"[Q10 LITE] {len(records)}/{len(indices)} | Q-cache={len(refiner.cache)} "
                  f"hits={refiner.cache_hits} | elapsed={perf_counter() - started:.2f}s", flush=True)
            last_progress = perf_counter()
    result = {"kind": KIND, "records": records, "debug": cfg.debug, "candidate_count": len(records),
              "radius_mm": cfg.radius_mm, "step_mm": detector_cfg.nms_radius_mm,
              "direction_mode": cfg.direction_mode,
              "antipodal_mode": "sphere_max_gap",
              "gap_link_deg": cfg.gap_link_deg, "gap_edge_span_deg": cfg.gap_edge_span_deg,
              "gap_tail_enabled": cfg.gap_tail_enabled, "gap_tail_bin_width_deg": cfg.gap_tail_bin_width_deg,
              "gap_tail_density_fraction": cfg.gap_tail_density_fraction,
              "gap_tail_min_bin_count": cfg.gap_tail_min_bin_count,
              "min_q10": cfg.min_q10, "refine_enabled": cfg.refine_enabled,
              "refine_cc_enabled": cfg.refine_enabled and cfg.refine_cc_enabled,
              "sphere_max_planes": cfg.sphere_max_planes, "sphere_tolerance_deg": cfg.sphere_tolerance_deg,
              "sphere_iterations": cfg.sphere_iterations, "sphere_min_support": cfg.sphere_min_support,
              "sphere_min_fraction": cfg.sphere_min_fraction, "sphere_seed": cfg.sphere_seed,
              "voxel_mm": cfg.voxel_mm, "cache_hits": refiner.cache_hits,
              "cache_misses": refiner.cache_misses, "final_evaluation_count": len(final_cache),
              "q10_early_reject_count": sum(r["q10_reject"] is True for r in records),
              "elapsed_s": perf_counter() - started}
    for name, status in (("accept", "ACCEPT"), ("reject", "REJECT"), ("indeterminate", "INDETERMINATE")):
        ids = np.asarray([r["point_index"] for r in records if r["status"] == status], dtype=np.int64)
        result[{"accept": "accepted", "reject": "rejected"}.get(name, name) + "_indices"] = ids
        result[name + "_count"] = len(ids)
    result["kept_indices"] = np.asarray([r["point_index"] for r in records if r["keep"]], dtype=np.int64)
    result["keep_count"] = len(result["kept_indices"])
    if cfg.debug:
        result["context"] = {"analysis_points_mm": analysis, "analysis_tree": refiner.tree, "raw_points_mm": raw}
    print(f"[Q10 LITE] NMS={len(records)} ACCEPT={result['accept_count']} REJECT={result['reject_count']} "
          f"INDETERMINATE={result['indeterminate_count']} | fixed NMS radius={detector_cfg.nms_radius_mm:g} mm "
          f"| direction={cfg.direction_mode} | sphere_ransac max_planes={cfg.sphere_max_planes} "
          f"MaxGap link={cfg.gap_link_deg:g} deg span>={cfg.gap_edge_span_deg:g} deg "
          f"tail={cfg.gap_tail_enabled} "
          f"| Q10 refine={cfg.refine_enabled} CC={result['refine_cc_enabled']} "
          f"| final=angular only | Q10 min={cfg.min_q10 if cfg.refine_enabled else 'OFF'} "
          f"| cache misses/hits={refiner.cache_misses}/{refiner.cache_hits} "
          f"| final evaluations={len(final_cache)} Q10 early rejects={result['q10_early_reject_count']} "
          f"| elapsed={result['elapsed_s']:.3f}s", flush=True)
    return result
