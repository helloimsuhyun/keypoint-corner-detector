"""Fine-voxel CED queries with complete neighborhoods in the Fine voxel cloud."""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .candidate_extraction import extract_corner_candidates, greedy_nms_indices
from .edge_suppression import dense_ced_weighted_pca_scores


def find_fine_queries(fine_points_mm, coarse_query_mm, cfg, *, point_tree=None):
    started = perf_counter()
    cloud = np.asarray(fine_points_mm, dtype=np.float64)
    query = np.asarray(coarse_query_mm, dtype=np.float64)
    settings = cfg.validator
    settings.validate()
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError("fine_points_mm must be finite with shape (N, 3)")
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("coarse_query_mm must be a finite 3-vector")
    tree = cKDTree(cloud) if point_tree is None else point_tree
    pca_radius = settings.fine_pca_radius_mm
    bilateral_radius = settings.fine_bilateral_radius_mm
    # PCA/bilateral support needs original CED, whose own neighbors extend one more CED radius.
    halo_radius = (settings.fine_search_radius_mm + max(pca_radius, bilateral_radius)
                   + settings.fine_ced_radius_mm)
    support_ids = np.asarray(sorted(tree.query_ball_point(query, np.nextafter(halo_radius, np.inf))), dtype=np.int64)
    points = cloud[support_ids]
    scores = dense_ced_weighted_pca_scores(
        points, settings.fine_ced_radius_mm, pca_radius, settings.fine_min_neighbors,
        bilateral_radius_mm=bilateral_radius,
        bilateral_sigma_spatial_mm=settings.fine_bilateral_sigma_spatial_mm,
        bilateral_sigma_ced=settings.fine_bilateral_sigma_ced,
        pca_ratio_saturation_tau=settings.fine_pca_ratio_saturation_tau,
    )
    evaluation = np.linalg.norm(points - query, axis=1) <= settings.fine_search_radius_mm
    evaluation_ids = support_ids[evaluation]
    evaluated_scores = {key: value[evaluation].copy() if isinstance(value, np.ndarray) else value
                        for key, value in scores.items()}
    evaluated_scores["original_point_indices"] = evaluation_ids.copy()
    eligibility = extract_corner_candidates(
        cloud[evaluation_ids], evaluated_scores, settings.fine_nms_radius_mm,
        fusion_score_threshold=settings.fine_fusion_score_threshold,
        fusion_threshold_mode=settings.fine_fusion_threshold_mode,
        fusion_peak_relative_ratio=settings.fine_fusion_peak_relative_ratio,
    )
    if settings.fine_nms_method == "greedy":
        local_ids = greedy_nms_indices(cloud[evaluation_ids], evaluated_scores["fusion_score"],
                                        eligibility["eligible_score_mask"], settings.fine_nms_radius_mm)
    else:
        local_ids = eligibility["candidate_indices"]
    candidates = evaluation_ids[local_ids]
    return {"coarse_query_position_mm": query.copy(), "evaluation_fine_indices": evaluation_ids,
            "support_fine_indices": support_ids, "support_radius_mm": halo_radius,
            "scores": evaluated_scores, "candidate_fine_indices": candidates,
            "candidate_points_mm": cloud[candidates].copy(),
            "fusion_score_threshold": eligibility["fusion_score_threshold"],
            "fusion_threshold_mode": settings.fine_fusion_threshold_mode,
            "fusion_peak_relative_ratio": settings.fine_fusion_peak_relative_ratio,
            "nms_method": settings.fine_nms_method,
            "fine_ced_radius_mm": settings.fine_ced_radius_mm,
            "fine_search_radius_mm": settings.fine_search_radius_mm,
            "fine_pca_radius_mm": pca_radius, "fine_bilateral_radius_mm": bilateral_radius,
            "elapsed_s": perf_counter() - started}
