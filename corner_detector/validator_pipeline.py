"""Shared numerical validator path for clicked queries and detector NMS seeds."""

from __future__ import annotations

import time
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


def _cornervins_clustering(points, query_index, cfg, organized_context):
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
    query = points[int(query_index)]
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
    """Cache stage clouds once; each keeps a nearest-Coarse scene-ID adapter."""
    from scipy.spatial import cKDTree
    from .preprocessing import voxel_downsample_3d

    scene = np.asarray(scene_points_mm, dtype=np.float64)
    raw = np.asarray(raw_points_mm, dtype=np.float64)
    for name, points in (("scene_points_mm", scene), ("raw_points_mm", raw)):
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{name} must be finite with shape (N, 3)")
    if not len(scene):
        raise ValueError("Validator requires a nonempty Coarse scene")
    cfg.validator.validate()
    scene_tree = cKDTree(scene)
    clouds = {float(cfg.voxel_3d_mm): {
        "points_mm": scene, "point_tree": scene_tree,
        "scene_indices": np.arange(len(scene), dtype=np.int64),
    }}
    context = {}
    for stage, voxel in (("fine", cfg.validator.fine_voxel_mm),
                         ("ransac", cfg.validator.ransac_voxel_mm)):
        if voxel not in clouds:
            points = voxel_downsample_3d(raw, voxel)
            clouds[voxel] = {"points_mm": points, "point_tree": cKDTree(points),
                             "scene_indices": np.asarray(scene_tree.query(points)[1], dtype=np.int64)}
        context[stage] = clouds[voxel]
        context[stage + "_voxel_mm"] = voxel
    return context


def run_validator_pipeline(
    points_mm,
    query_index,
    cfg,
    estimator=None,
    *,
    organized_context=None,
    raw_points_mm=None,
    voxel_context=None,
    fine_result=None,
    fine_index=None,
    stop_after_fine=False,
):
    """Keep Coarse scene IDs canonical and distinguish each stage's voxel IDs."""
    points = np.asarray(points_mm, dtype=np.float64)
    query_index = int(query_index)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    if not 0 <= query_index < len(points):
        raise IndexError("query_index is outside points_mm")
    from .settings import ValidatorDetectorConfig
    if isinstance(cfg, ValidatorDetectorConfig):
        from .guided_ransac import extract_guided_planes
        from .validator_fine import find_fine_queries
        if voxel_context is None:
            if raw_points_mm is None:
                raise ValueError("Guided validator requires raw_points_mm to prepare stage voxels")
            voxel_context = prepare_validator_voxel_context(points, raw_points_mm, cfg)
        if (voxel_context["fine_voxel_mm"] != cfg.validator.fine_voxel_mm
                or voxel_context["ransac_voxel_mm"] != cfg.validator.ransac_voxel_mm):
            raise ValueError("Voxel context sizes do not match validator settings")
        coarse = points[query_index]
        fine_cloud = voxel_context["fine"]
        ransac_cloud = voxel_context["ransac"]
        if fine_result is None:
            fine_result = {**find_fine_queries(fine_cloud["points_mm"], coarse, cfg,
                                              point_tree=fine_cloud["point_tree"]),
                           "coarse_query_index": query_index,
                           "fine_voxel_mm": cfg.validator.fine_voxel_mm}
            for source, target in (("candidate_fine_indices", "candidate_scene_indices"),
                                   ("evaluation_fine_indices", "evaluation_scene_indices"),
                                   ("support_fine_indices", "support_scene_indices")):
                fine_result[target] = fine_cloud["scene_indices"][fine_result[source]].copy()
        if (fine_result["coarse_query_index"] != query_index
                or not np.array_equal(fine_result["coarse_query_position_mm"], coarse)
                or fine_result["fine_voxel_mm"] != cfg.validator.fine_voxel_mm):
            raise ValueError("Fine results belong to a different coarse query")
        candidates = fine_result["candidate_fine_indices"]
        fallback = not len(candidates)
        if fine_index is not None and fine_index not in candidates:
            raise ValueError("Fine selection must be an existing Fine NMS voxel index")
        if fine_index is None and not fallback:
            distances = np.linalg.norm(fine_cloud["points_mm"][candidates] - coarse, axis=1)
            fine_index = int(candidates[np.lexsort((candidates, distances))[0]])
        query = coarse if fallback else fine_cloud["points_mm"][int(fine_index)]
        if stop_after_fine and not fallback:
            result = {"query_position_mm": query.copy(), "planes": [], "planes_computed": False}
        else:
            result = extract_guided_planes(ransac_cloud["points_mm"], query, cfg.validator,
                                            point_tree=ransac_cloud["point_tree"])
            result["roi_point_indices"] = ransac_cloud["scene_indices"][result["roi_ransac_indices"]].copy()
            for plane in result["planes"]:
                plane["inlier_point_indices"] = ransac_cloud["scene_indices"][plane["inlier_ransac_indices"]].copy()
            result["planes_computed"] = True
        result.update(query_index=query_index, coarse_query_position_mm=coarse.copy(),
                      fine_query_index=fine_index,
                      fine_query_scene_index=(query_index if fallback else int(fine_cloud["scene_indices"][fine_index])),
                      fine_voxel_mm=cfg.validator.fine_voxel_mm, ransac_voxel_mm=cfg.validator.ransac_voxel_mm,
                      fine_fallback=fallback,
                      fine_query_shift_mm=float(np.linalg.norm(query - coarse)), fine_result=fine_result)
        return result
    started = time.perf_counter()
    clustering = _cornervins_clustering(
        points, query_index, cfg, organized_context
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
        "query_position_mm": points[query_index].copy(),
        "search_scope": "query",
        "clustering": clustering,
        "edge_config": edge_config,
        "plane_edge": plane_edge,
        "elapsed_s": time.perf_counter() - started,
    }


def _final_geometry(query_result):
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
    estimator,
    *,
    organized_context=None,
):
    """Invoke the canonical validator for every detector NMS candidate."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    started = time.perf_counter()
    records, final_edges, final_corners = [], [], []
    for query_index in indices:
        try:
            pipeline_kwargs = {}
            if organized_context is not None:
                pipeline_kwargs["organized_context"] = organized_context
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
        records.append({**result, "status": "ok"})
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
    "make_plane_edge_config", "make_refiner_config",
    "run_validator_candidates", "run_validator_pipeline", "prepare_validator_voxel_context",
]
