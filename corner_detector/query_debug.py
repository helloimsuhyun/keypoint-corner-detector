"""Evaluate one selected query and report plane/edge diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from pointcloud_inputs import PointCloudFrame, query_from_pixel

from .preprocessing import voxel_downsample_3d
from .plane_edge_graph_pipeline import save_plane_edge_graph_json
from .validator_pipeline import run_validator_pipeline
from .visualization.manager import (
    prepare_cornervins_plane_views,
    prepare_plane_edge_graph_view,
)


def inspect_query(pixel_x, pixel_y, frozen, cfg):
    frame = frozen.get("pointcloud_frame")
    if not isinstance(frame, PointCloudFrame):
        raise TypeError("inspect_query requires frozen['pointcloud_frame']")
    selection = query_from_pixel(frame, pixel_x, pixel_y, max_radius_px=2)
    clicked = None if selection is None else selection.point_mm
    if clicked is None:
        print(f"[query] Invalid depth at ({pixel_x}, {pixel_y})")
        return

    if "points_voxel_mm" not in frozen:
        frozen["points_voxel_mm"] = voxel_downsample_3d(
            frozen["points_raw_mm"], cfg.voxel_3d_mm,
        )
    points = frozen["points_voxel_mm"]
    if len(points) < cfg.min_neighbors:
        print("[query] Too few downsampled 3-D points.")
        return
    point_tree = cKDTree(points)
    click_distance, query_index = point_tree.query(clicked)
    frozen["selected_pixel"] = (pixel_x, pixel_y)
    frozen["corner_position_3d"] = np.asarray(points[int(query_index)], dtype=float).copy()
    plane_frontend = "query-core AHC -> RANSAC/refit planes"
    if getattr(cfg, "plane_edge_graph_enabled", False):
        plane_frontend += " -> padded 2-D boundary -> 3-D graph"
    print(f"[query] PLANE PIPELINE: {plane_frontend}")
    plane_cluster_analysis = None
    try:
        validation = run_validator_pipeline(
            points,
            int(query_index),
            cfg,
            None,
            organized_context=frozen.get("organized_plane_context"),
        )
    except Exception as exc:
        print(f"[VALIDATOR PIPELINE] {type(exc).__name__}: {exc}")
    else:
        scope = validation["search_scope"]
        plane_cluster_analysis = validation["clustering"]
        edge_cfg = validation["edge_config"]
        plane_edge_result = validation["plane_edge"]
        prepare_cornervins_plane_views(
            frozen,
            plane_cluster_analysis,
            cfg=cfg,
            show_3d=cfg.show_3d,
        )
        if getattr(cfg, "plane_edge_graph_enabled", False):
                # Install stage 4 now; Query UI opens it after the stage-3 2-D
                # window has been created so the visible order matches the pipeline.
                prepare_plane_edge_graph_view(frozen, plane_edge_result, show_3d=False)
                if getattr(cfg, "plane_edge_debug_2d", False):
                    # View only. Do not pass debug flags to detector or change its results.
                    frozen["plane_edge_debug_clustering"] = plane_cluster_analysis
                    frozen["plane_edge_debug_config"] = edge_cfg

                if cfg.plane_edge_export_json:
                    filename = save_plane_edge_graph_json(plane_edge_result, cfg.plane_edge_export_json)
                    print(f"  Exported plane/edge graph: {filename}")
                stats = plane_edge_result["diagnostics"]
                print(f"[PLANE -> 2D HOUGH -> 3D EDGE GRAPH] scope={scope}")
                corner_diag = plane_cluster_analysis["diagnostics"]
                print(
                    f"  CornerVINS AHC regions coarse/refined="
                    f"{corner_diag['coarse_region_count']}/"
                    f"{corner_diag['refined_region_count']} | "
                    f"RANSAC rejected={corner_diag['ransac_rejected_region_count']} | "
                    f"merged={corner_diag['merge_count']} | "
                    f"region-mode={corner_diag['region_mode']} "
                    f"min={corner_diag['min_region_points']}"
                )
                print(
                    "  CornerVINS extraction scope="
                    f"{corner_diag['organized_extraction_scope']} | "
                    f"radius={corner_diag['organized_extraction_radius_mm']:.3f} mm | "
                    f"organized-pixels="
                    f"{corner_diag['organized_extraction_pixel_count']}"
                )
                print(
                    "  ROI->plane projection: CornerVINS organized membership; "
                    f"halo={edge_cfg.projection_halo_mm:.3g} mm"
                )
                print(f"  2D Hough: count>={edge_cfg.hough_min_votes} "
                      f"mean-weight>={edge_cfg.hough_min_mean_weight:g}; "
                      f"score-threshold={edge_cfg.hough_score_threshold_mode}"
                      f"({edge_cfg.hough_score_threshold:g}); "
                      f"NMS={edge_cfg.hough_nms_mode} "
                      f"angle=±{edge_cfg.hough_nms_angle_deg:g} deg "
                      f"rho=±{edge_cfg.hough_nms_rho_mm:g} mm; "
                      f"3D merge: angle={edge_cfg.merge_angle_deg:g} deg "
                      f"offset={edge_cfg.merge_distance_mm:g} mm "
                      f"gap={edge_cfg.merge_gap_mm:g} mm")
                print(f"  query_radius={plane_cluster_analysis['query_radius_mm']:.3f} mm "
                      f"scene={len(points)} "
                      f"ROI={len(plane_cluster_analysis['roi_point_indices'])} "
                      f"planes={len(plane_cluster_analysis['global_planes'])}")
                for plane in plane_edge_result['planes']:
                    details = plane['stage_counts']
                    vote = plane['line_vote']
                    print(f"    P{plane['plane_id']}: raw={details['input_support']} "
                          f"fit={details['plane_inliers']} roi={details['roi_candidates']} "
                          f"projected={details['projection_kept']} boundary={details['boundary']} "
                          f"hough={details['hough_peaks']} segments={details['segments']} "
                          f"score-T={vote.get('hough_score_threshold_effective', 0):.3g} "
                          f"pca-accepted={details['3d_accepted']} "
                          f"pca-rejected={details['3d_rejected']}")
                print(f"  planes={stats['plane_count']} boundary-points={stats['boundary_points']} "
                      f"hough-peaks={stats['hough_peaks']} "
                      f"3d-pca-accepted={stats['3d_accepted']} "
                      f"3d-pca-rejected={stats['3d_rejected']}")
                print(f"  proposals={stats['proposal_count']} merged={stats['merged_edge_count']} "
                      f"shared={stats['shared_edge_count']} graph-links={stats['graph_link_count']} "
                      f"elapsed={stats['elapsed_s']:.3f}s")
                if "refinement" in plane_edge_result:
                    refined_stats = plane_edge_result["refinement"]["diagnostics"]
                    print(f"  refiner: graphs={refined_stats['graph_count']} "
                          f"planes={refined_stats['plane_count']} "
                          f"edges={refined_stats['refined_edge_count']} "
                          f"plane-intersections={refined_stats['plane_intersection_count']} "
                          f"corners={refined_stats['corner_count']} "
                          f"elapsed={refined_stats['elapsed_s']:.3f}s")
                print(f"  single-plane boundary corner candidates={stats['single_plane_corner_count']}")
                for corner in plane_edge_result['corners']:
                    print(f"    C{corner['corner_id']} plane=P{corner['plane_id']} "
                          f"edges={corner['edge_ids']} angle={corner['angle_deg']:.1f}deg "
                          f"xyz_mm={np.asarray(corner['position_mm']).round(2).tolist()}")
                for edge in plane_edge_result['edges']:
                    print(f"    E{edge['edge_id']} planes={edge['plane_ids']} "
                          f"length={edge['length_mm']:.2f}mm rms={edge['rms_mm']:.2f}mm "
                          f"support={edge['support_count']}")
                if "refinement" in plane_edge_result:
                    for corner in plane_edge_result["refinement"]["corners"]:
                        print(f"    RC{corner['corner_id']} graph=G{corner['graph_id']} "
                              f"edges={corner['edge_ids']} planes={corner['plane_ids']} "
                              f"gap={corner['line_gap_mm']:.3f}mm "
                              f"xyz_mm={np.asarray(corner['position_mm']).round(2).tolist()}")
                    for edge in plane_edge_result["refinement"]["edges"]:
                        refined = edge["refined"]
                        print(f"    RE{edge['refined_edge_id']} graph=G{edge['graph_id']} "
                              f"planes={edge['plane_ids']} mode={edge['mode']} "
                              f"length={refined['length_mm']:.2f}mm "
                              f"rms={refined['rmse_mm']:.2f}mm")

    print(f"  raw / 3D voxel: {len(frozen['points_raw_mm']):,} / {len(points):,}")
    print(
        f"  clicked -> query: {click_distance:.3f} mm; "
        f"query-index={int(query_index)}"
    )
    if plane_cluster_analysis is not None:
        diagnostics = plane_cluster_analysis["diagnostics"]
        timings = diagnostics["timings_seconds"]
        print("\n[QUERY CORNERVINS PLANE EXTRACTION]")
        print(
            f"  ROI points / organized pixels: "
            f"{len(plane_cluster_analysis['roi_point_indices'])}/"
            f"{diagnostics['organized_extraction_pixel_count']}"
        )
        print(
            f"  Final planes: {len(plane_cluster_analysis['global_planes'])} | "
            f"AHC coarse/refined={diagnostics['coarse_region_count']}/"
            f"{diagnostics['refined_region_count']} | "
            f"RANSAC rejected={diagnostics['ransac_rejected_region_count']} | "
            f"merges={diagnostics['merge_count']}"
        )
        print(
            f"  Scope: {diagnostics['organized_extraction_scope']} | "
            f"radius={diagnostics['organized_extraction_radius_mm']:.3f} mm | "
            f"elapsed={timings['total']:.4f}s"
        )

__all__ = ["inspect_query"]
