"""Coarse/fine query selection and reporting for the guided validator."""

from dataclasses import replace

import numpy as np
from pointcloud_inputs import query_from_pixel

from .dense_heatmap import run_dense_heatmap_mode
from .validator_pipeline import prepare_validator_voxel_context, run_validator_pipeline
from .visualization.manager import prepare_guided_validator_result


def prepare_validator_queries(points_mm, cfg):
    """Retain unchanged CED/PCA/NMS scores for query selection and heatmap display."""
    return run_dense_heatmap_mode(points_mm, replace(cfg, show_3d=False, validator=None))


def inspect_validator_query(query_index, frozen, cfg, *, fine_index=None):
    indices = frozen["validator_query_indices"]
    if query_index not in indices:
        raise ValueError("Validator selection must be an existing CED NMS query")
    context = frozen.get("validator_voxel_context")
    if context is None:
        context = frozen["validator_voxel_context"] = prepare_validator_voxel_context(
            frozen["points_voxel_mm"], frozen["points_raw_mm"], cfg)
    cached = frozen.get("validator_fine_result") if fine_index is not None else None
    if fine_index is not None and cached is None:
        raise ValueError("Select a coarse query before selecting a Fine NMS query")
    result = run_validator_pipeline(
        frozen["points_voxel_mm"], query_index, cfg,
        voxel_context=context, fine_result=cached, fine_index=fine_index,
        stop_after_fine=fine_index is None and cfg.show_3d,
    )
    frozen["validator_fine_result"] = result["fine_result"]
    frozen["corner_position_3d"] = result["query_position_mm"].copy()
    prepare_guided_validator_result(frozen, result)
    print("\n[VALIDATOR]")
    print(f"Coarse Query: {result['coarse_query_position_mm'].tolist()}")
    print(f"Fine Query: {result['query_position_mm'].tolist()}")
    print(f"Fine Query Shift: {result['fine_query_shift_mm']:.6g} mm")
    print(f"Fine NMS Candidates: {len(result['fine_result']['candidate_fine_indices'])} | Fallback: {result['fine_fallback']}")
    print(f"Fine Voxel: {cfg.validator.fine_voxel_mm} mm | RANSAC Voxel: {cfg.validator.ransac_voxel_mm} mm")
    print(f"Fine CED Radius: {cfg.validator.fine_ced_radius_mm} mm")
    if not result["planes_computed"]:
        print("[VALIDATOR] Fine heatmap/NMS ready. Click a Fine NMS candidate to fit planes.")
        return result
    print(f"Guided Sigma: {cfg.validator.guided_sigma_mm} mm")
    print(f"ROI Radius: {result['radius_mm']} mm\nROI Points: {len(result['roi_ransac_indices'])}")
    print(f"\nDetected Planes: {len(result['planes'])}")
    for i, plane in enumerate(result["planes"], 1):
        print(f"\nPlane {i}\n  Normal: {plane['normal'].tolist()}")
        print(f"  Offset: {plane['offset_mm']:.6g}")
        print(f"  Inlier Count: {plane['inlier_count']} | Inlier Ratio: {plane['inlier_ratio']:.6g}")
        print(f"  Inlier RMS: {plane['inlier_rms_mm']:.6g} mm")
        print(f"  Query-to-Plane Distance: {plane['query_to_plane_distance_mm']:.6g} mm")
    print(f"RANSAC Total Time: {result['ransac_elapsed_s'] * 1000:.3f} ms")
    return result


def inspect_validator_pixel(pixel_x, pixel_y, frozen, cfg):
    indices = frozen["validator_query_indices"]
    if not len(indices):
        print("[VALIDATOR] No CED NMS query.")
        return None
    selection = query_from_pixel(frozen["pointcloud_frame"], pixel_x, pixel_y, max_radius_px=2)
    if selection is None:
        print("[VALIDATOR] No valid depth.")
        return None
    points = frozen["points_voxel_mm"]
    nearest = np.argmin(np.linalg.norm(points[indices] - selection.point_mm, axis=1))
    frozen["selected_pixel"] = (pixel_x, pixel_y)
    return inspect_validator_query(int(indices[nearest]), frozen, cfg)
