"""Coarse NMS query selection and reporting for the guided validator."""

from dataclasses import replace

import numpy as np
from pointcloud_inputs import query_from_pixel

from .dense_heatmap import run_dense_heatmap_mode
from .validator_pipeline import prepare_validator_voxel_context, run_validator_pipeline
from .visualization.manager import prepare_guided_validator_result


def prepare_validator_queries(points_mm, cfg, *, raw_points_mm=None):
    """Retain dense/NMS scores and prepare optional Q10 centers for kept queries."""
    options = {"raw_points_mm": raw_points_mm} if getattr(cfg, "q10_validator", None) is not None else {}
    return run_dense_heatmap_mode(points_mm, replace(cfg, show_3d=False, validator=None), **options)


def inspect_validator_query(query_index, frozen, cfg):
    indices = frozen["validator_query_indices"]
    if query_index not in indices:
        raise ValueError("Validator selection must be an existing kept detector seed")
    context = frozen.get("validator_voxel_context")
    if context is None:
        context = frozen["validator_voxel_context"] = prepare_validator_voxel_context(
            frozen["points_voxel_mm"], frozen["points_raw_mm"], cfg)
    positions = frozen.get("validator_query_positions_mm")
    options = {} if positions is None else {"query_position_mm": positions[int(np.flatnonzero(indices == query_index)[0])]}
    result = run_validator_pipeline(
        frozen["points_voxel_mm"], query_index, cfg, voxel_context=context,
        **options,
    )
    prepare_guided_validator_result(frozen, result)
    print("\n[VALIDATOR]")
    print(f"Coarse Query: {result['coarse_query_position_mm'].tolist()}")
    print(f"Validator start: {result['query_position_mm'].tolist()}")
    print(f"Query-local Voxel: {result['ransac_voxel_mm']} mm | "
          f"Raw core points: {len(result['roi_raw_point_indices'])} -> "
          f"{result['core_voxel_count']} core voxels | {result['halo_voxel_count']} halo voxels")
    print(f"Guided Sigma: {cfg.validator.guided_sigma_mm} mm")
    print(f"ROI Radius: {result['radius_mm']} mm\nROI Points: {len(result['roi_ransac_indices'])}")
    if result.get("geometry_computed"):
        nodes = [node for patch in result["boundary_debug"] for node in patch["nodes"].values()]
        print(f"Hull Support Radius: {result['support_radius_mm']:g} mm | "
              f"Padding: {result['support_radius_mm'] - result['radius_mm']:g} mm | "
              f"Hull nodes after clipping: {sum(node['corner_enabled'] for node in nodes)} active / "
              f"{sum(not node['corner_enabled'] for node in nodes)} inactive")
    print(f"\nDetected Planes: {len(result['planes'])}")
    for i, plane in enumerate(result["planes"], 1):
        print(f"\nPlane {i}\n  Normal: {plane['normal'].tolist()}")
        print(f"  Offset: {plane['offset_mm']:.6g}")
        print(f"  Inlier Count: {plane['inlier_count']} | Inlier Ratio: {plane['inlier_ratio']:.6g}")
        print(f"  Inlier RMS: {plane['inlier_rms_mm']:.6g} mm")
        print(f"  Query-to-Plane Distance: {plane['query_to_plane_distance_mm']:.6g} mm")
    print(f"RANSAC Total Time: {result['ransac_elapsed_s'] * 1000:.3f} ms")
    if result.get("projection_only"):
        print(f"2D projection: {len(result['plane_patches'])} planes | hull/edge/corner NOT_COMPUTED")
        print(f"2D centroid Q10: {result['validation_status']} | {result['passed_plane_count']}/{len(result['plane_q10'])} planes pass | threshold={result['q10_threshold']:g}")
        for score in result["plane_q10"]:
            print(f"  Plane {score['plane_id']}: Q10={score['q10']} | pass={score['passed']} | {score['reason']}")
    else:
        print(f"Corner generation: {result['validation_status']} | structure unchecked")
    print(f"Finite Edges: {len(result['edges'])} | Corners: {len(result['corners'])}")
    for index, corner in enumerate(result["corners"]):
        print(f"  Corner {index}: {corner['position'].tolist()} mm | edges={corner['edge_ids']} | "
              f"endpoints={corner['endpoint_ids']} | intersection residual={corner['intersection_residual']:.6g} mm | "
              f"endpoint gap={corner['endpoint_gap']:.6g} mm")
    print("Stage Time (ms): " + ", ".join(
        f"{name}={elapsed * 1000:.3f}" for name, elapsed in result["stage_elapsed_s"].items()))
    print(f"Validator Total Time: {result['elapsed_s'] * 1000:.3f} ms")
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
    positions = frozen.get("validator_query_positions_mm", points[indices])
    nearest = np.argmin(np.linalg.norm(positions - selection.point_mm, axis=1))
    frozen["selected_pixel"] = (pixel_x, pixel_y)
    return inspect_validator_query(int(indices[nearest]), frozen, cfg)
