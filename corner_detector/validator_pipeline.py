"""Shared numerical validator path for clicked queries and detector NMS seeds."""

from __future__ import annotations

import time
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree

from .cornervins_plane_extraction import (
    CornerVINSPlaneConfig,
    as_plane_cluster_analysis,
    extract_planes,
)
from .plane_edge_graph_pipeline import PlaneEdgeGraphConfig, analyze_plane_edge_graph
from .refiner import HoughEdgeRefinerConfig, refine_hough_edge_graph


def make_plane_edge_config(cfg):
    """Build the plane-edge stage from resolved validator settings."""
    return PlaneEdgeGraphConfig(
        plane_inlier_mm=cfg.plane_edge_plane_inlier_mm,
        alpha_radius_mm=cfg.plane_edge_alpha_radius_mm,
        hough_rho_bin_mm=cfg.plane_edge_hough_bin_mm,
        hough_theta_step_deg=cfg.plane_edge_hough_angle_deg,
        hough_min_mean_weight=cfg.plane_edge_hough_min_mean_weight,
        hough_score_threshold_mode=cfg.plane_edge_hough_score_threshold_mode,
        hough_score_threshold=cfg.plane_edge_hough_score_threshold,
        hough_nms_mode=cfg.plane_edge_hough_nms_mode,
        hough_nms_angle_deg=cfg.plane_edge_hough_nms_angle_deg,
        hough_nms_rho_mm=cfg.plane_edge_hough_nms_rho_mm,
        hough_inlier_mm=cfg.plane_edge_inlier_mm,
        hough_min_votes=cfg.plane_edge_min_points,
        hough_max_peaks=cfg.plane_edge_max_peaks,
        line_refinement_enabled=cfg.plane_edge_line_refinement_enabled,
        line_refinement_max_iterations=cfg.plane_edge_line_refinement_max_iterations,
        line_refinement_final_nms_angle_deg=cfg.plane_edge_line_final_nms_angle_deg,
        line_refinement_final_nms_overlap=cfg.plane_edge_line_final_nms_overlap,
        line_min_points=cfg.plane_edge_min_points,
        line_min_length_mm=cfg.plane_edge_min_length_mm,
        segment_gap_mm=cfg.plane_edge_segment_gap_mm,
        direction_tolerance_deg=cfg.plane_edge_direction_tolerance_deg,
        merge_distance_mm=cfg.plane_edge_merge_distance_mm,
        merge_gap_mm=cfg.plane_edge_merge_gap_mm,
        merge_angle_deg=cfg.plane_edge_merge_angle_deg,
        projection_halo_mm=cfg.projection_halo_mm,
        boundary_radius_mm=cfg.boundary_radius_mm,
        boundary_epsilon_mm=cfg.boundary_epsilon_mm,
        boundary_sigma_mm=cfg.boundary_sigma_mm,
        boundary_min_neighbors=cfg.min_neighbors,
    )


def make_refiner_config(cfg):
    """Build the graph-local refiner from resolved validator settings."""
    return HoughEdgeRefinerConfig(
        enabled=True,
        robust_loss=cfg.plane_edge_refine_loss,
        huber_delta_mm=cfg.plane_edge_refine_huber_delta_mm,
        plane_confidence_weight=cfg.plane_edge_refine_plane_weight,
        boundary_confidence_weight=cfg.plane_edge_refine_boundary_weight,
        max_nfev=cfg.plane_edge_refine_max_nfev,
        corner_intersection_mm=cfg.plane_edge_refine_corner_intersection_mm,
        corner_endpoint_gap_mm=cfg.plane_edge_refine_corner_endpoint_gap_mm,
        corner_merge_mm=cfg.plane_edge_refine_corner_merge_mm,
    )


def prepare_validator_seeds(points_mm, extraction, q10_result=None):
    """Keep NMS scene IDs while supplying separate, optionally refined XYZ centers."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Validator seeds require finite scene XYZ points")
    nms = np.asarray(extraction["candidate_indices"], dtype=np.int64)
    indices = nms if q10_result is None else np.asarray(q10_result["accepted_indices"], dtype=np.int64)
    if (indices.ndim != 1 or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(points)) or not np.isin(indices, nms).all()):
        raise ValueError("Validator seeds must retain unique detector NMS scene IDs")
    if q10_result is None or not len(indices):
        positions = points[indices].copy()
    else:
        records = {record["point_index"]: record for record in q10_result["records"]}
        positions = np.asarray([records[int(i)]["refined_position_mm"] for i in indices], dtype=float).reshape(-1, 3)
    if positions.shape != (len(indices), 3) or not np.isfinite(positions).all():
        raise ValueError("Validator seed positions must be aligned finite XYZ coordinates")
    return {"candidate_indices": indices.copy(), "positions_mm": positions.copy(),
            "original_positions_mm": points[indices].copy(),
            "source": ("detector_nms" if q10_result is None else
                       "q10_refined" if q10_result.get("refine_enabled", True) else "maxgap_nms")}


def _cornervins_clustering(points, query_index, cfg, organized_context, query_position_mm=None):
    if organized_context is None:
        raise ValueError(
            "CornerVINS mode requires organized RGB-D context; the voxel cloud "
            "alone has no pixel adjacency"
        )
    if not isinstance(organized_context, dict):
        raise TypeError("organized_context must be the mutable frame context dict")
    required = (
        "organized_points_mm",
        "valid_mask",
        "organized_to_scene_index",
    )
    missing = [name for name in required if name not in organized_context]
    if missing:
        raise ValueError(f"CornerVINS organized context is missing: {missing}")
    extraction_config = CornerVINSPlaneConfig.from_settings(cfg)
    query = points[int(query_index)] if query_position_mm is None else query_position_mm
    tree = cKDTree(points)
    roi = np.sort(
        np.asarray(
            tree.query_ball_point(query, float(cfg.query_radius_mm)),
            dtype=np.int64,
        )
    )
    extraction_radius_mm = float(cfg.query_radius_mm)
    organized = np.asarray(
        organized_context["organized_points_mm"], dtype=np.float64
    )
    base_valid = np.asarray(organized_context["valid_mask"], dtype=bool)
    query_valid = base_valid & (
        np.linalg.norm(organized - query, axis=2)
        <= extraction_radius_mm + 1e-9
    )

    # CornerVINS segmentation and plane fitting use exactly the query core.
    # No full-scene or projection-halo point participates in plane clustering.
    cache_key = (
        int(query_index),
        tuple(query),
        extraction_config,
        extraction_radius_mm,
    )
    cache = organized_context.setdefault("_cornervins_query_extractions", {})
    extraction = cache.get(cache_key)
    if extraction is None:
        extraction = extract_planes(
            points,
            organized_points=organized,
            valid_mask=query_valid,
            organized_to_point_index=organized_context[
                "organized_to_scene_index"
            ],
            settings=extraction_config,
        )
        cache[cache_key] = extraction
    analysis = as_plane_cluster_analysis(
        extraction,
        points,
        int(query_index),
        float(cfg.query_radius_mm),
        roi,
        projection_halo_mm=float(cfg.projection_halo_mm),
        projection_inlier_mm=float(cfg.plane_edge_plane_inlier_mm),
        query_position_mm=query,
    )
    analysis["organized_mapping_max_distance_mm"] = float(
        organized_context.get("mapping_max_distance_mm", np.nan)
    )
    analysis["organized_mapping_mean_distance_mm"] = float(
        organized_context.get("mapping_mean_distance_mm", np.nan)
    )
    analysis["organized_extraction_scope"] = "query_core"
    analysis["organized_extraction_radius_mm"] = extraction_radius_mm
    analysis["organized_extraction_pixel_count"] = int(
        np.count_nonzero(query_valid)
    )
    analysis["diagnostics"].update({
        "organized_extraction_scope": "query_core",
        "organized_extraction_radius_mm": extraction_radius_mm,
        "organized_extraction_pixel_count": int(np.count_nonzero(query_valid)),
    })
    return analysis


def prepare_validator_voxel_context(scene_points_mm, raw_points_mm, cfg):
    """Cache raw/Coarse trees; voxelize only after selecting a query."""
    scene = np.asarray(scene_points_mm, dtype=np.float64)
    raw = np.asarray(raw_points_mm, dtype=np.float64)
    for name, points in (("scene_points_mm", scene), ("raw_points_mm", raw)):
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{name} must be finite with shape (N, 3)")
    if not len(scene):
        raise ValueError("Validator requires a nonempty Coarse scene")
    cfg.validator.validate()
    return {"scene_points_mm": scene, "scene_tree": cKDTree(scene),
            "raw_points_mm": raw, "raw_tree": cKDTree(raw)}


def _query_voxel_cloud(context, query_index, cfg, query_position_mm=None):
    from .preprocessing import voxel_downsample_3d

    query = context["scene_points_mm"][query_index] if query_position_mm is None else query_position_mm
    radius = cfg.ransac_radius_mm
    support_radius = cfg.padded_support_radius_mm if cfg.geometry_enabled else radius
    key = (query_index, tuple(query), cfg.ransac_voxel_mm, radius, support_radius)
    if context.get("query_cloud_key") == key:
        return context["query_cloud"]
    raw_ids = np.sort(np.asarray(context["raw_tree"].query_ball_point(query, support_radius), dtype=np.int64))
    raw = context["raw_points_mm"][raw_ids]
    distances = np.linalg.norm(raw - query, axis=1)
    core_mask = distances <= radius
    # Separate the halo so outside points cannot move core voxel centroids.
    core = voxel_downsample_3d(raw[core_mask], cfg.ransac_voxel_mm)
    halo = voxel_downsample_3d(raw[(distances > radius) & (distances <= support_radius)], cfg.ransac_voxel_mm)
    points = np.concatenate((core, halo))
    cloud = {"points_mm": points, "point_tree": cKDTree(points),
             "core_points_mm": core, "core_tree": cKDTree(core),
             "scene_indices": np.asarray(context["scene_tree"].query(points)[1], dtype=np.int64),
             "raw_core_point_indices": raw_ids[core_mask],
             "core_voxel_count": len(core), "halo_voxel_count": len(halo)}
    context["query_cloud_key"], context["query_cloud"] = key, cloud
    return cloud


def _projection_q10_result(patches, threshold):
    from .validator_plane_support import projection_centroid_q10
    evidence = []
    for patch in patches:
        score = projection_centroid_q10(patch["projection_uv"])
        score.update(plane_id=patch["plane_id"],
                     passed=None if score["q10"] is None else bool(score["q10"] >= threshold))
        evidence.append(score)
    failed = any(score["passed"] is False for score in evidence)
    passed = bool(evidence) and all(score["passed"] is True for score in evidence)
    status = "ACCEPT" if passed else "REJECT" if failed else "INDETERMINATE"
    reason = (None if passed else "projection_q10_below_threshold" if failed else
              "undefined_projection_q10" if evidence else "no_fitted_planes")
    return {"plane_q10": evidence, "q10_threshold": threshold,
            "passed_plane_count": sum(score["passed"] is True for score in evidence),
            "validation_status": status, "accept": passed, "failure_reason": reason}


def run_validator_pipeline(
    points_mm,
    query_index,
    cfg,
    estimator=None,
    *,
    organized_context=None,
    raw_points_mm=None,
    voxel_context=None,
    include_debug=True,
    query_position_mm=None,
):
    """Keep scene IDs canonical; an optional XYZ center never moves scene samples."""
    points = np.asarray(points_mm, dtype=np.float64)
    query_index = int(query_index)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if not 0 <= query_index < len(points):
        raise IndexError("query_index is outside points_mm")
    query = points[query_index] if query_position_mm is None else np.asarray(query_position_mm, dtype=np.float64)
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("query_position_mm must be a finite XYZ coordinate")
    from .settings import ValidatorDetectorConfig
    if isinstance(cfg, ValidatorDetectorConfig):
        started = time.perf_counter()
        from .guided_ransac import extract_guided_planes
        from .validator_plane_support import reconstruct_plane_support
        from .validator_edge_graph import build_validator_edge_graph
        from .validator_corner_graph import empty_corner_generation, reconstruct_validator_corners
        cfg.validator.validate()
        if voxel_context is None:
            if raw_points_mm is None:
                raise ValueError("Guided validator requires raw_points_mm for query-local voxelization")
            voxel_context = prepare_validator_voxel_context(points, raw_points_mm, cfg)
        if (voxel_context["scene_points_mm"] is not points
                and not np.array_equal(voxel_context["scene_points_mm"], points)):
            raise ValueError("Voxel context belongs to a different Coarse scene")
        voxel_started = time.perf_counter()
        ransac_cloud = _query_voxel_cloud(voxel_context, query_index, cfg.validator, query)
        voxel_elapsed = time.perf_counter() - voxel_started
        result = extract_guided_planes(ransac_cloud["core_points_mm"], query, cfg.validator,
                                       point_tree=ransac_cloud["core_tree"])
        scene_ids = ransac_cloud["scene_indices"]
        result["roi_point_indices"] = scene_ids[result["roi_ransac_indices"]].copy()
        result["planes"] = [
            {**plane, "plane_id": plane_id,
             "inlier_point_indices": scene_ids[plane["inlier_ransac_indices"]].copy()}
            for plane_id, plane in enumerate(result["planes"])
        ]
        result.update(query_index=query_index, coarse_query_position_mm=points[query_index].copy(),
                      planes_computed=True, ransac_voxel_mm=cfg.validator.ransac_voxel_mm,
                      ransac_points_mm=ransac_cloud["points_mm"],
                      ransac_scene_indices=scene_ids,
                      roi_raw_point_indices=ransac_cloud["raw_core_point_indices"],
                      core_voxel_count=ransac_cloud["core_voxel_count"],
                      halo_voxel_count=ransac_cloud["halo_voxel_count"])
        result.update(empty_corner_generation(query))
        timers = {"voxel": voxel_elapsed, "ransac": result["ransac_elapsed_s"],
                  "plane_support": 0., "edge_graph": 0., "corner_graph": 0., "geometry": 0.}
        if getattr(cfg.validator, "projection_only", False):
            support = reconstruct_plane_support(ransac_cloud["points_mm"], query, result["planes"],
                                                cfg.validator, point_tree=ransac_cloud["point_tree"], projection_only=True)
            for patch in support["plane_patches"]:
                patch["support_point_indices"] = scene_ids[patch["support_ransac_indices"]].copy()
                patch["recollected_point_indices"] = scene_ids[patch["recollected_ransac_indices"]].copy()
            timers["plane_support"] = support["elapsed_s"]
            result.update(projection_only=True, projection_computed=True,
                          plane_patches=support["plane_patches"], support_radius_mm=support["support_radius_mm"],
                          support_roi_ransac_indices=support["support_roi_ransac_indices"],
                          support_roi_point_indices=scene_ids[support["support_roi_ransac_indices"]].copy(),
                          support_roi_points_mm=support["support_roi_points_mm"],
                          stage_elapsed_s=timers, elapsed_s=time.perf_counter() - started)
            quality_started = time.perf_counter()
            result.update(_projection_q10_result(support["plane_patches"], cfg.validator.min_q10))
            timers["projection_q10"] = time.perf_counter() - quality_started
            result["elapsed_s"] = time.perf_counter() - started
            return result
        if cfg.validator.geometry_enabled:
            geometry_started = time.perf_counter()
            support = reconstruct_plane_support(ransac_cloud["points_mm"], query, result["planes"],
                                                cfg.validator, point_tree=ransac_cloud["point_tree"])
            graph = build_validator_edge_graph(ransac_cloud["points_mm"], query, support, cfg.validator)
            geometry = reconstruct_validator_corners(graph, cfg.validator)
            before, before_elapsed = None, 0.
            before_support = {"plane_patches": []}
            before_graph = {"all_edges": [], "edge_candidates": []}
            if include_debug:
                # The core-only comparison is display data, never part of Corner selection.
                before_started = time.perf_counter()
                before_cfg = replace(cfg.validator, roi_padding_mm=0.)
                before_support = reconstruct_plane_support(ransac_cloud["core_points_mm"], query, result["planes"],
                                                           before_cfg, point_tree=ransac_cloud["core_tree"])
                before_graph = build_validator_edge_graph(ransac_cloud["core_points_mm"], query, before_support, before_cfg)
                before_corners = reconstruct_validator_corners(before_graph, before_cfg)
                before = {"plane_patches": before_support["plane_patches"], **before_graph,
                          "corners": before_corners["corners"]}
                before_elapsed = time.perf_counter() - before_started
            timers.update(plane_support=support["elapsed_s"], edge_graph=graph["elapsed_s"],
                          corner_graph=geometry.pop("elapsed_s"),
                          roi_debug_before=before_elapsed,
                          geometry=time.perf_counter() - geometry_started)
            scene_ids = ransac_cloud["scene_indices"]
            for patch in support["plane_patches"] + before_support["plane_patches"]:
                patch["support_point_indices"] = scene_ids[patch["support_ransac_indices"]].copy()
                patch["recollected_point_indices"] = scene_ids[patch["recollected_ransac_indices"]].copy()
            for edge in (graph["all_edges"] + graph["edge_candidates"]
                         + before_graph["all_edges"] + before_graph["edge_candidates"]):
                edge["support_point_indices"] = scene_ids[edge["support_ransac_indices"]].copy()
                for membership in edge["support_by_plane"]:
                    membership["support_point_indices"] = scene_ids[membership["support_ransac_indices"]].copy()
            result.update(geometry)
            result.update(plane_patches=support["plane_patches"], edges=graph["edges"],
                          edge_overlap_settings={"distance_mm": cfg.validator.edge_overlap_distance_mm,
                                                 "angle_deg": cfg.validator.edge_overlap_angle_deg,
                                                 "min_length_mm": cfg.validator.edge_overlap_min_length_mm},
                          corner_radius_mm=cfg.validator.corner_radius_mm,
                          all_edges=graph["all_edges"],
                          boundary_debug=graph["boundary_debug"], roi_debug_before=before,
                          debug_comparison_computed=bool(include_debug),
                          boundary_overlap_diagnostics=graph["boundary_overlap_diagnostics"],
                          edge_candidates=graph["edge_candidates"], plane_adjacency=graph["plane_adjacency"],
                          plane_pair_diagnostics=graph["plane_pair_diagnostics"],
                          support_radius_mm=support["support_radius_mm"],
                          core_radius_mm=cfg.validator.ransac_radius_mm,
                          roi_padding_mm=cfg.validator.roi_padding_mm,
                          support_roi_ransac_indices=support["support_roi_ransac_indices"],
                          support_roi_point_indices=scene_ids[support["support_roi_ransac_indices"]].copy(),
                          support_roi_points_mm=support["support_roi_points_mm"])
        result["stage_elapsed_s"] = timers
        result["elapsed_s"] = time.perf_counter() - started
        return result
    started = time.perf_counter()
    clustering = _cornervins_clustering(
        points, query_index, cfg, organized_context, query
    )
    edge_config = None
    plane_edge = None
    if cfg.plane_edge_graph_enabled:
        edge_config = make_plane_edge_config(cfg)
        plane_edge = analyze_plane_edge_graph(points, clustering, edge_config)
        if cfg.plane_edge_refiner_enabled:
            plane_edge = {
                **plane_edge,
                "refinement": refine_hough_edge_graph(
                    points, plane_edge, make_refiner_config(cfg),
                ),
            }
    return {
        "query_index": query_index,
        "query_position_mm": query.copy(),
        "coarse_query_position_mm": points[query_index].copy(),
        "search_scope": "query",
        "clustering": clustering,
        "edge_config": edge_config,
        "plane_edge": plane_edge,
        "elapsed_s": time.perf_counter() - started,
    }


def _final_geometry(query_result):
    if "geometry_computed" in query_result:
        edges, corners = [], []
        by_id = {edge["edge_id"]: edge for edge in query_result.get("edges", ())}
        for index, corner in enumerate(query_result["corners"]):
            corners.append({"query_index": int(query_result["query_index"]), "corner_id": index,
                            "position_mm": corner["position"].copy(),
                            "edge_ids": corner["edge_ids"].copy(), "endpoint_ids": corner["endpoint_ids"].copy(),
                            "intersection_residual": corner["intersection_residual"],
                            "endpoint_gap": corner["endpoint_gap"]})
            for identifier in corner["edge_ids"]:
                edge = by_id[identifier]
                edges.append({"query_index": int(query_result["query_index"]), "corner_id": index,
                              "source_edge_id": identifier, "endpoint_id": corner["endpoint_ids"][identifier],
                              "edge_type": edge["edge_type"], "plane_ids": edge["plane_ids"].copy(),
                              "start_mm": edge["start_mm"].copy(), "end_mm": edge["end_mm"].copy(),
                              "active_intervals_mm": list(edge["active_intervals_mm"]),
                              "corner_position_mm": corner["position"].copy()})
        return edges, corners
    graph = query_result["plane_edge"]
    source_by_id = {int(edge["edge_id"]): edge for edge in graph.get("edges", ())}
    refinement = graph.get("refinement")
    edges, corners = [], []
    if refinement is not None:
        for edge in refinement.get("edges", ()):
            source_id = int(edge.get("source_edge_id", -1))
            source = source_by_id.get(source_id, edge)
            plane_ids = sorted(set(int(value) for value in source.get("plane_ids", ())))
            refined = edge["refined"]
            edges.append({
                "query_index": int(query_result["query_index"]),
                "source_edge_id": source_id,
                "refined_edge_id": int(edge.get("refined_edge_id", -1)),
                "plane_ids": plane_ids,
                "plane_count": len(plane_ids),
                "start_mm": np.asarray(refined["start_mm"], dtype=np.float64).copy(),
                "end_mm": np.asarray(refined["end_mm"], dtype=np.float64).copy(),
            })
        for corner in refinement.get("corners", ()):
            corners.append({
                "query_index": int(query_result["query_index"]),
                "corner_id": int(corner.get("corner_id", -1)),
                "plane_ids": [int(value) for value in corner.get("plane_ids", ())],
                "position_mm": np.asarray(corner["position_mm"], dtype=np.float64).copy(),
            })
    else:
        for edge in graph.get("edges", ()):
            plane_ids = sorted(set(int(value) for value in edge.get("plane_ids", ())))
            edges.append({
                "query_index": int(query_result["query_index"]),
                "source_edge_id": int(edge.get("edge_id", -1)),
                "refined_edge_id": None,
                "plane_ids": plane_ids,
                "plane_count": len(plane_ids),
                "start_mm": np.asarray(edge["start_mm"], dtype=np.float64).copy(),
                "end_mm": np.asarray(edge["end_mm"], dtype=np.float64).copy(),
            })
        for corner in graph.get("corners", ()):
            corners.append({
                "query_index": int(query_result["query_index"]),
                "corner_id": int(corner.get("corner_id", -1)),
                "plane_ids": [int(corner["plane_id"])],
                "position_mm": np.asarray(corner["position_mm"], dtype=np.float64).copy(),
            })
    return edges, corners


def run_validator_candidates(
    points_mm,
    candidate_indices,
    cfg,
    estimator=None,
    *,
    organized_context=None,
    raw_points_mm=None,
    voxel_context=None,
    include_debug=True,
    query_positions_mm=None,
):
    """Invoke the canonical validator for every detector NMS candidate."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    positions = None if query_positions_mm is None else np.asarray(query_positions_mm, dtype=np.float64)
    if positions is not None and (positions.shape != (len(indices), 3) or not np.isfinite(positions).all()):
        raise ValueError("query_positions_mm must contain one finite XYZ coordinate per candidate")
    started = time.perf_counter()
    from .settings import ValidatorDetectorConfig
    guided = isinstance(cfg, ValidatorDetectorConfig)
    if guided and len(indices) and voxel_context is None:
        if raw_points_mm is None:
            raise ValueError("Guided batch validation requires raw_points_mm")
        voxel_context = prepare_validator_voxel_context(points_mm, raw_points_mm, cfg)
    records, final_edges, final_corners = [], [], []
    progress_started = started
    for ordinal, query_index in enumerate(indices, 1):
        try:
            pipeline_kwargs = {}
            if guided:
                pipeline_kwargs.update(voxel_context=voxel_context, include_debug=include_debug)
            if organized_context is not None:
                pipeline_kwargs["organized_context"] = organized_context
            if positions is not None:
                pipeline_kwargs["query_position_mm"] = positions[ordinal - 1]
            result = run_validator_pipeline(
                points_mm,
                int(query_index),
                cfg,
                estimator,
                **pipeline_kwargs,
            )
        except Exception as exc:
            records.append({
                "query_index": int(query_index), "status": "error",
                "error_type": type(exc).__name__, "error": str(exc),
            })
            continue
        edges, corners = _final_geometry(result)
        final_edges.extend(edges)
        final_corners.extend(corners)
        if guided and not include_debug and not result.get("projection_only", False):
            record = {key: result[key] for key in (
                "query_index", "query_position_mm", "coarse_query_position_mm", "validation_status", "verified",
                "corners",
                "failure_reason", "stage_elapsed_s", "elapsed_s")}
        else:
            record = result
        records.append({**record, "status": "ok"})
        if getattr(cfg, "mode", None) == "total" and (
                time.perf_counter() - progress_started >= 1. or ordinal == len(indices)):
            print(f"[TOTAL validator] {ordinal}/{len(indices)} queries | "
                  f"corners={len(final_corners)} | elapsed={time.perf_counter() - started:.2f}s", flush=True)
            progress_started = time.perf_counter()
    return {
        "method": "dense_nms_to_shared_validator_pipeline",
        "candidate_indices": indices.copy(),
        "records": records,
        "final_edges": final_edges,
        "final_corners": final_corners,
        "diagnostics": {
            "candidate_count": int(len(indices)),
            "completed_count": int(sum(record["status"] == "ok" for record in records)),
            "failed_count": int(sum(record["status"] != "ok" for record in records)),
            "final_edge_count": int(len(final_edges)),
            "final_corner_count": int(len(final_corners)),
            "elapsed_s": time.perf_counter() - started,
        },
    }


__all__ = [
    "prepare_validator_seeds",
    "make_plane_edge_config", "make_refiner_config",
    "run_validator_candidates", "run_validator_pipeline", "prepare_validator_voxel_context",
]
