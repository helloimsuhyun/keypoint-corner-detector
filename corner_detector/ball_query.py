"""Click and CED adapters for independent local vertex hypotheses."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pointcloud_inputs import PointCloudFrame, query_from_pixel

from .extremity import refine_ced_candidates
from .dense_heatmap import run_dense_heatmap_mode
from .visualization.manager import prepare_extremity_view


def inspect_ball_query(pixel_x, pixel_y, frozen, cfg):
    frame = frozen.get("pointcloud_frame")
    if not isinstance(frame, PointCloudFrame):
        raise TypeError("ball requires frozen['pointcloud_frame']")
    selection = query_from_pixel(frame, pixel_x, pixel_y, max_radius_px=2)
    if selection is None:
        print(f"[ball] Invalid depth at ({pixel_x}, {pixel_y})")
        return None
    points = frozen["points_voxel_mm"]
    batch = frozen["extremity_batch"]
    indices = batch["query_indices"]
    if not len(indices):
        print("[ball] CED returned no query; no vertex hypotheses can be generated.")
        return None
    # Clicking only selects an existing CED query; its position is unchanged.
    nearest = np.argmin(np.linalg.norm(points[indices] - selection.point_mm, axis=1))
    result = batch["records"][nearest]
    query = int(indices[nearest])
    distance = np.linalg.norm(points[query] - selection.point_mm)
    frozen["selected_pixel"] = (pixel_x, pixel_y)
    frozen["corner_position_3d"] = points[query].copy()
    prepare_extremity_view(frozen, result, show_3d=cfg.show_3d)
    decision = "HYPOTHESES PRESENT" if result["candidate_count"] else "NO HYPOTHESIS"
    print(f"[ball] {decision} | {result['candidate_count']} unverified peaks | "
          f"unchanged query={query} click->query={distance:.3f} mm | "
          f"elapsed={result['timings_seconds']['total'] * 1000:.2f} ms")
    for candidate in result["candidates"]:
        print(f"  raw={candidate['raw_point_index']} scene={candidate['point_index']} "
              f"xyz-mm={candidate['position_mm'].round(3).tolist()} "
              f"votes={candidate['raw_votes']} fraction={candidate['normalized_score']:.4f} "
              f"component={candidate['cluster_id']} offset={candidate['query_distance_mm']:.3f} mm")
    return result


def run_ced_extremity_mode(points_mm, cfg, *, raw_points_mm=None):
    """Use the original detector composition with its post-NMS validator disabled.

    CED, PCA, threshold and NMS settings remain exactly those supplied by cfg.
    Verification is a separate downstream consumer of the returned scene IDs.
    """
    from time import perf_counter
    started = perf_counter()
    scores = run_dense_heatmap_mode(points_mm, replace(cfg, show_3d=False, validator=None))
    ced_s = perf_counter() - started
    indices = scores["candidate_extraction"]["candidate_indices"]
    batch = refine_ced_candidates(points_mm, indices, cfg.extremity, raw_points_mm=raw_points_mm)
    batch["ced_candidate_extraction"] = scores["candidate_extraction"]
    batch["ced_score_result"] = scores
    batch["ced_wall_including_reporting_s"] = ced_s
    print(f"[CED -> LOCAL EXTREMITY] unchanged CED queries={len(indices)} "
          f"hypotheses={batch['candidate_count']} | CED-wall(reporting included)={ced_s:.6f}s "
          f"additional-refinement={batch['elapsed_s']:.6f}s")
    return batch
