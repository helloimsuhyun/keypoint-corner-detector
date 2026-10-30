"""Single-source CLI defaults and mode-specific resolved detector configs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import BUILD
from .preprocessing import NoiseConfig, validate_probability


@dataclass(frozen=True)
class SettingsDefaults:
    """CLI defaults, grouped by consumer and ordered by pipeline execution."""

    # ------------------------------------------------------------------
    # Shared: input -> sensor geometry -> 3-D preprocessing -> display.
    # ------------------------------------------------------------------
    source: str = "stl"
    mode: str = "fine"
    only: bool = False
    input: str | None = None
    frame_index: int = 0
    fps: int = 30
    stl_unit_mm: float = 1.0
    width: int = 640
    height: int = 480
    fov_deg: float = 60.0
    min_depth_m: float = 0.01
    max_depth_m: float = 3.0

    voxel_3d_mm: float = 10.0
    min_neighbors: int = 6
    centroid_radius_x: float = 8.0
    no_3d_view: bool = False

    # Shared synthetic perturbation and reproducibility.
    noise_mm: float = 0.0
    outlier_rate: float = 0.0
    outlier_mm: float = 8.0
    dropout_rate: float = 0.0
    dropout_patches: int = 0
    dropout_patch_radius_px: int = 0
    quantization_mm: float = 0.0
    seed: int = 0

    # Validator: separate Fine/RANSAC voxels; each *_x uses its stage's voxel.
    validator_fine_voxel_mm: float = 5.0
    validator_fine_search_radius_x: float = 9.0
    validator_fine_ced_radius_x: float = 3.5

    validator_fine_min_neighbors: int = 6
    validator_fine_bilateral_radius_x: float = 3.0
    validator_fine_bilateral_sigma_spatial_x: float = 1.5
    validator_fine_bilateral_sigma_ced: float = 0.05

    validator_fine_pca_radius_x: float = 4.5
    validator_fine_pca_ratio_saturation_tau: float | None = 0.5
    validator_fine_fusion_threshold_mode: str = "peak-relative"
    # Peak-relative: max(absolute floor, ratio * valid Fine ROI peak).
    validator_fine_fusion_score_threshold: float = 0.10
    validator_fine_fusion_peak_relative_ratio: float = 0.5
    validator_fine_nms_method: str = "greedy"
    validator_fine_nms_radius_x: float = 3.0

    validator_ransac_voxel_mm: float = 5.0
    validator_ransac_radius_x: float = 6.0
    validator_max_planes: int = 3
    validator_ransac_iterations: int = 256
    validator_inlier_threshold_x: float = 0.5
    validator_guided_sigma_x: float = 1.0

    validator_min_inliers: int = 20
    validator_min_inlier_ratio: float = 0.08
    validator_random_seed: int = 42

    # ------------------------------------------------------------------
    # Validator step 0: query geometry.
    # ------------------------------------------------------------------
    query_radius_x: float = 10.0
    voxel_2d_mm: float = 10.0
    # ------------------------------------------------------------------
    # Validator step 1: query-core organized plane extraction.
    # Values marked as ROI/Kinect adaptations are not paper constants.
    # ------------------------------------------------------------------
    # Step 1: AHC planar-region segmentation (block side is in image pixels).
    cornervins_ahc_block_size: int = 3
    cornervins_ahc_mse_threshold_mm2: float | None = None
    cornervins_ahc_refinement: bool = True

    # Step 1 filtering: paper threshold or local-query adaptation.
    cornervins_region_mode: str = "roi_test"
    cornervins_min_region_points_paper: int = 20_000
    cornervins_min_region_points_roi: int = 100

    # Step 2: per-region RANSAC and least-squares refit (metric values are mm).
    cornervins_downsample_leaf_mm: float = 50.0
    cornervins_ransac_threshold_mm: float = 15.0
    cornervins_ransac_min_inlier_ratio: float = 0.90
    cornervins_ransac_max_trials: int = 256
    cornervins_ransac_seed: int = 0

    # Step 3: repeated compatible-plane merging.
    cornervins_merge_angle_deg: float = 5.0
    cornervins_merge_distance_mm: float = 20.0

    # Diagnostics only; these switches never change extracted plane numerics.
    cornervins_show_ahc_regions: bool = True
    cornervins_show_plane_patches: bool = True
    cornervins_show_convex_hulls: bool = True

    # Validator step 2: padded projection and continuous 2-D boundary evidence.
    plane_edge_graph_enabled: bool = True
    plane_edge_debug_2d: bool = True
    plane_edge_export_json: str | None = None

    # Extra observed context outside the query core; absolute millimetres.
    projection_halo_mm: float = 50.0

    boundary_radius_x: float = 9.0
    boundary_epsilon_x: float = 2.5
    boundary_sigma_x: float = 2.0
    plane_edge_alpha_x: float = 0.0  # compatibility fallback; padded boundary does not use it

    # Validator step 3: weighted Hough accumulator and seed NMS.
    plane_edge_hough_bin_x: float = 0.15
    plane_edge_hough_angle_deg: float = 0.5
    plane_edge_hough_min_mean_weight: float = 0.60
    plane_edge_hough_score_threshold_mode: str = "peak-relative"
    plane_edge_hough_score_threshold: float = 0.65

    plane_edge_hough_nms_mode: str = "local-max"
    plane_edge_hough_nms_angle_deg: float = 20.0
    plane_edge_hough_nms_rho_x: float = 3.0
    plane_edge_max_peaks: int = 7

    # Validator step 4: continuous line refinement and finite-segment NMS.
    plane_edge_line_refinement_enabled: bool = True
    plane_edge_line_refinement_max_iterations: int = 60
    plane_edge_line_final_nms_angle_deg: float = 5.0
    plane_edge_line_final_nms_overlap: float = 0.50

    # 2-D line support and finite-segment splitting.
    plane_edge_inlier_x: float = 1.0
    plane_edge_min_points: int = 3
    plane_edge_segment_gap_x: float = 2.5

    # Validator step 5: direct PCA/TLS 3-D line reconstruction and merging.
    plane_edge_plane_inlier_x: float = 1.5
    plane_edge_min_length_x: float = 3.0
    plane_edge_direction_tolerance_deg: float = 15.0

    # 3-D finite-edge complete-link merging.
    plane_edge_merge_distance_x: float = 5.0
    plane_edge_merge_gap_x: float = 5.0
    plane_edge_merge_angle_deg: float = 30.0

    # Validator step 6: graph-local plane/edge refinement and corners.
    plane_edge_refiner_enabled: bool = True
    plane_edge_refine_loss: str = "huber"
    plane_edge_refine_huber_delta_x: float = 1.0
    plane_edge_refine_plane_weight: float = 1.0
    plane_edge_refine_boundary_weight: float = 1.0
    plane_edge_refine_max_nfev: int = 100
    plane_edge_refine_corner_intersection_x: float = 0.5
    plane_edge_refine_corner_endpoint_gap_x: float = 1.2
    plane_edge_refine_corner_merge_x: float = 1.0

    # ------------------------------------------------------------------
    # Detector: dense CED -> bilateral/PCA fusion -> threshold -> NMS.
    # ------------------------------------------------------------------
    bilateral_sigma_spatial_mm: float | None = None
    bilateral_sigma_ced: float = 0.05
    bilateral_radius_mm: float | None = None
    pca_radius_factor: float = 1.5
    pca_ratio_saturation_tau: float | None = 0.5
    fusion_threshold_mode: str = "global-mean"
    fusion_score_threshold: float = 0.10
    nms_radius_mm: float | None = 50

    # ------------------------------------------------------------------
    # Ball/extremity256: local vertex hypotheses after CED/PCA/NMS.
    # Lengths are voxel_3d_mm multipliers, resolved to mm before voting.
    # ------------------------------------------------------------------
    ball_radii_x: tuple[float, ...] = (10.0,)
    ball_padding_x: float = 2.5
    ball_directions: int = 256
    ball_object_clustering: bool = True
    ball_cluster_radius_x: float = 1.5
    ball_nms_radius_x: float = 3.0
    # "local_maximum": 반경 내 모든 유효 이웃과 비교해 극대점 유지 (동점은 작은 원본 index 우선).
    # "greedy": 점수 내림차순으로 선택한 점 주변을 억제 (동점은 작은 원본 index 우선).
    ball_nms_method: str = "greedy"
    ball_min_vote_fraction: float = 0.000

    # "hard": 방향별 최대 투영점 1개에 1표.
    # "soft_dense": np.quantile의 99분위 기준, 모든 점에 sigmoid 가중치 투표.
    # "soft_partition": 선형 보간을 유지한 np.partition 분위수 + dense sigmoid (동일 수식).
    # "soft_sparse": partition 분위수 + t-alpha*tau 이상만 sigmoid 투표 (dense의 근사).
    # Soft 모드는 방향별 가중치 합을 1로 정규화한 뒤 padding 영역의 표를 버림.
    ball_voting_mode: str = "soft_partition"
    ball_soft_tau_x: float = 0.025  # 0.25 mm at the default 10 mm voxel

    ball_soft_alpha: float = 4.0
    ball_voting_dtype: str = "float64"

DEFAULTS = SettingsDefaults()


def _resolve_validator_values(args):
    """Resolve validator settings in the same order as the runtime stages."""
    v3 = float(args.voxel_3d_mm)
    v2 = float(args.voxel_2d_mm)
    return {
        # Shared/query geometry.
        "voxel_3d_mm": v3,
        "voxel_2d_mm": v2,
        "min_neighbors": args.min_neighbors,
        "show_3d": not args.no_3d_view,
        "centroid_radius_mm": args.centroid_radius_x * v3,
        "query_radius_mm": args.query_radius_x * v2,

        # Step 1: CornerVINS AHC -> filter -> RANSAC/refit -> merge/hull.
        "cornervins_ahc_block_size": args.cornervins_ahc_block_size,
        "cornervins_ahc_mse_threshold_mm2": args.cornervins_ahc_mse_threshold_mm2,
        "cornervins_ahc_refinement": args.cornervins_ahc_refinement,
        "cornervins_region_mode": args.cornervins_region_mode,
        "cornervins_min_region_points_paper": args.cornervins_min_region_points_paper,
        "cornervins_min_region_points_roi": args.cornervins_min_region_points_roi,
        "cornervins_downsample_leaf_mm": args.cornervins_downsample_leaf_mm,
        "cornervins_ransac_threshold_mm": args.cornervins_ransac_threshold_mm,
        "cornervins_ransac_min_inlier_ratio": args.cornervins_ransac_min_inlier_ratio,
        "cornervins_ransac_max_trials": args.cornervins_ransac_max_trials,
        "cornervins_ransac_seed": args.cornervins_ransac_seed,
        "cornervins_merge_angle_deg": args.cornervins_merge_angle_deg,
        "cornervins_merge_distance_mm": args.cornervins_merge_distance_mm,
        "cornervins_show_ahc_regions": args.cornervins_show_ahc_regions,
        "cornervins_show_plane_patches": args.cornervins_show_plane_patches,
        "cornervins_show_convex_hulls": args.cornervins_show_convex_hulls,

        # Step 2: padded projection and continuous boundary evidence.
        "plane_edge_graph_enabled": args.plane_edge_graph_enabled,
        "plane_edge_debug_2d": args.plane_edge_debug_2d,
        "plane_edge_export_json": args.plane_edge_export_json,
        "boundary_radius_mm": args.boundary_radius_x * v2,
        "boundary_epsilon_mm": args.boundary_epsilon_x * v2,
        "boundary_sigma_mm": args.boundary_sigma_x * v2,
        "projection_halo_mm": args.projection_halo_mm,
        "plane_edge_alpha_radius_mm": (
            args.plane_edge_alpha_x * v3 if args.plane_edge_alpha_x > 0 else None
        ),

        # Step 3: weighted Hough and accumulator NMS.
        "plane_edge_hough_bin_mm": args.plane_edge_hough_bin_x * v3,
        "plane_edge_hough_angle_deg": args.plane_edge_hough_angle_deg,
        "plane_edge_hough_min_mean_weight": args.plane_edge_hough_min_mean_weight,
        "plane_edge_hough_score_threshold_mode": args.plane_edge_hough_score_threshold_mode.replace("-", "_"),
        "plane_edge_hough_score_threshold": args.plane_edge_hough_score_threshold,
        "plane_edge_hough_nms_mode": args.plane_edge_hough_nms_mode.replace("-", "_"),
        "plane_edge_hough_nms_angle_deg": args.plane_edge_hough_nms_angle_deg,
        "plane_edge_hough_nms_rho_mm": args.plane_edge_hough_nms_rho_x * v3,
        "plane_edge_max_peaks": args.plane_edge_max_peaks,

        # Step 4: continuous line refinement and finite-segment extraction/NMS.
        "plane_edge_line_refinement_enabled": args.plane_edge_line_refinement_enabled,
        "plane_edge_line_refinement_max_iterations": args.plane_edge_line_refinement_max_iterations,
        "plane_edge_line_final_nms_angle_deg": args.plane_edge_line_final_nms_angle_deg,
        "plane_edge_line_final_nms_overlap": args.plane_edge_line_final_nms_overlap,
        "plane_edge_inlier_mm": args.plane_edge_inlier_x * v3,
        "plane_edge_min_points": args.plane_edge_min_points,
        "plane_edge_segment_gap_mm": args.plane_edge_segment_gap_x * v3,

        # Step 5: direct 3-D reconstruction and complete-link merging.
        "plane_edge_plane_inlier_mm": args.plane_edge_plane_inlier_x * v3,
        "plane_edge_min_length_mm": args.plane_edge_min_length_x * v3,
        "plane_edge_direction_tolerance_deg": args.plane_edge_direction_tolerance_deg,
        "plane_edge_merge_distance_mm": args.plane_edge_merge_distance_x * v3,
        "plane_edge_merge_gap_mm": args.plane_edge_merge_gap_x * v3,
        "plane_edge_merge_angle_deg": args.plane_edge_merge_angle_deg,

        # Step 6: graph-local plane/edge refinement and corners.
        "plane_edge_refiner_enabled": args.plane_edge_refiner_enabled,
        "plane_edge_refine_loss": args.plane_edge_refine_loss.replace("-", "_"),
        "plane_edge_refine_huber_delta_mm": args.plane_edge_refine_huber_delta_x * v3,
        "plane_edge_refine_plane_weight": args.plane_edge_refine_plane_weight,
        "plane_edge_refine_boundary_weight": args.plane_edge_refine_boundary_weight,
        "plane_edge_refine_max_nfev": args.plane_edge_refine_max_nfev,
        "plane_edge_refine_corner_intersection_mm": args.plane_edge_refine_corner_intersection_x * v3,
        "plane_edge_refine_corner_endpoint_gap_mm": args.plane_edge_refine_corner_endpoint_gap_x * v3,
        "plane_edge_refine_corner_merge_mm": args.plane_edge_refine_corner_merge_x * v3,

    }


def _resolve_detector_values(args):
    """Resolve settings consumed only by the whole-cloud detector."""
    v3 = float(args.voxel_3d_mm)
    ced_radius_mm = args.centroid_radius_x * v3
    return {
        "voxel_3d_mm": v3,
        "min_neighbors": args.min_neighbors,
        "show_3d": not args.no_3d_view,
        "only": args.only,
        "centroid_radius_mm": ced_radius_mm,
        "bilateral_sigma_spatial_mm": (
            0.5 * ced_radius_mm
            if args.bilateral_sigma_spatial_mm is None
            else args.bilateral_sigma_spatial_mm
        ),
        "bilateral_sigma_ced": args.bilateral_sigma_ced,
        "bilateral_radius_mm": (
            ced_radius_mm
            if args.bilateral_radius_mm is None
            else args.bilateral_radius_mm
        ),
        "pca_radius_factor": args.pca_radius_factor,
        "pca_radius_mm": ced_radius_mm * args.pca_radius_factor,
        "pca_ratio_saturation_tau": args.pca_ratio_saturation_tau,
        "fusion_threshold_mode": args.fusion_threshold_mode.replace("-", "_"),
        "fusion_score_threshold": args.fusion_score_threshold,
        "nms_radius_mm": (
            0.5 * ced_radius_mm
            if args.nms_radius_mm is None
            else args.nms_radius_mm
        ),
    }


def _resolve_extremity_values(args):
    """Convert local voting lengths from 3-D voxel multipliers to mm."""
    v3 = float(args.voxel_3d_mm)
    return {
        "radii_mm": tuple(radius * v3 for radius in args.ball_radii_x),
        "padding_mm": args.ball_padding_x * v3,
        "directions": args.ball_directions,
        "object_clustering": args.ball_object_clustering,
        "cluster_radius_mm": args.ball_cluster_radius_x * v3,
        "nms_radius_mm": args.ball_nms_radius_x * v3,
        "nms_method": args.ball_nms_method,
        "min_vote_fraction": args.ball_min_vote_fraction,
        "voting_mode": args.ball_voting_mode,
        "soft_tau_mm": args.ball_soft_tau_x * v3,
        "soft_alpha": args.ball_soft_alpha,
        "voting_dtype": args.ball_voting_dtype,
    }


_DEFAULT_VALIDATOR = _resolve_validator_values(DEFAULTS)
_DEFAULT_DETECTOR = _resolve_detector_values(DEFAULTS)
_DEFAULT_EXTREMITY = _resolve_extremity_values(DEFAULTS)


@dataclass
class ValidatorConfig:
    """Resolved settings for clicked-query plane/edge validation."""

    # Shared/query geometry.
    voxel_3d_mm: float = _DEFAULT_VALIDATOR["voxel_3d_mm"]
    voxel_2d_mm: float = _DEFAULT_VALIDATOR["voxel_2d_mm"]
    min_neighbors: int = _DEFAULT_VALIDATOR["min_neighbors"]
    show_3d: bool = _DEFAULT_VALIDATOR["show_3d"]
    centroid_radius_mm: float = _DEFAULT_VALIDATOR["centroid_radius_mm"]
    query_radius_mm: float = _DEFAULT_VALIDATOR["query_radius_mm"]

    # Step 1: CornerVINS AHC -> filter -> RANSAC/refit -> merge/hull.
    cornervins_ahc_block_size: int = _DEFAULT_VALIDATOR["cornervins_ahc_block_size"]
    cornervins_ahc_mse_threshold_mm2: float | None = _DEFAULT_VALIDATOR[
        "cornervins_ahc_mse_threshold_mm2"
    ]
    cornervins_ahc_refinement: bool = _DEFAULT_VALIDATOR["cornervins_ahc_refinement"]
    cornervins_region_mode: str = _DEFAULT_VALIDATOR["cornervins_region_mode"]
    cornervins_min_region_points_paper: int = _DEFAULT_VALIDATOR[
        "cornervins_min_region_points_paper"
    ]
    cornervins_min_region_points_roi: int = _DEFAULT_VALIDATOR[
        "cornervins_min_region_points_roi"
    ]
    cornervins_downsample_leaf_mm: float = _DEFAULT_VALIDATOR[
        "cornervins_downsample_leaf_mm"
    ]
    cornervins_ransac_threshold_mm: float = _DEFAULT_VALIDATOR[
        "cornervins_ransac_threshold_mm"
    ]
    cornervins_ransac_min_inlier_ratio: float = _DEFAULT_VALIDATOR[
        "cornervins_ransac_min_inlier_ratio"
    ]
    cornervins_ransac_max_trials: int = _DEFAULT_VALIDATOR[
        "cornervins_ransac_max_trials"
    ]
    cornervins_ransac_seed: int = _DEFAULT_VALIDATOR["cornervins_ransac_seed"]
    cornervins_merge_angle_deg: float = _DEFAULT_VALIDATOR[
        "cornervins_merge_angle_deg"
    ]
    cornervins_merge_distance_mm: float = _DEFAULT_VALIDATOR[
        "cornervins_merge_distance_mm"
    ]
    cornervins_show_ahc_regions: bool = _DEFAULT_VALIDATOR[
        "cornervins_show_ahc_regions"
    ]
    cornervins_show_plane_patches: bool = _DEFAULT_VALIDATOR[
        "cornervins_show_plane_patches"
    ]
    cornervins_show_convex_hulls: bool = _DEFAULT_VALIDATOR[
        "cornervins_show_convex_hulls"
    ]

    # Step 2: padded projection and continuous boundary evidence.
    plane_edge_graph_enabled: bool = _DEFAULT_VALIDATOR["plane_edge_graph_enabled"]
    plane_edge_debug_2d: bool = _DEFAULT_VALIDATOR["plane_edge_debug_2d"]
    plane_edge_export_json: str | None = _DEFAULT_VALIDATOR["plane_edge_export_json"]
    boundary_radius_mm: float = _DEFAULT_VALIDATOR["boundary_radius_mm"]
    boundary_epsilon_mm: float = _DEFAULT_VALIDATOR["boundary_epsilon_mm"]
    boundary_sigma_mm: float = _DEFAULT_VALIDATOR["boundary_sigma_mm"]
    projection_halo_mm: float = _DEFAULT_VALIDATOR["projection_halo_mm"]
    plane_edge_alpha_radius_mm: float | None = _DEFAULT_VALIDATOR["plane_edge_alpha_radius_mm"]

    # Step 3: weighted Hough and accumulator NMS.
    plane_edge_hough_bin_mm: float = _DEFAULT_VALIDATOR["plane_edge_hough_bin_mm"]
    plane_edge_hough_angle_deg: float = _DEFAULT_VALIDATOR["plane_edge_hough_angle_deg"]
    plane_edge_hough_min_mean_weight: float = _DEFAULT_VALIDATOR["plane_edge_hough_min_mean_weight"]
    plane_edge_hough_score_threshold_mode: str = _DEFAULT_VALIDATOR["plane_edge_hough_score_threshold_mode"]
    plane_edge_hough_score_threshold: float = _DEFAULT_VALIDATOR["plane_edge_hough_score_threshold"]
    plane_edge_hough_nms_mode: str = _DEFAULT_VALIDATOR["plane_edge_hough_nms_mode"]
    plane_edge_hough_nms_angle_deg: float = _DEFAULT_VALIDATOR["plane_edge_hough_nms_angle_deg"]
    plane_edge_hough_nms_rho_mm: float = _DEFAULT_VALIDATOR["plane_edge_hough_nms_rho_mm"]
    plane_edge_max_peaks: int = _DEFAULT_VALIDATOR["plane_edge_max_peaks"]

    # Step 4: continuous line refinement and finite-segment extraction/NMS.
    plane_edge_line_refinement_enabled: bool = _DEFAULT_VALIDATOR["plane_edge_line_refinement_enabled"]
    plane_edge_line_refinement_max_iterations: int = _DEFAULT_VALIDATOR["plane_edge_line_refinement_max_iterations"]
    plane_edge_line_final_nms_angle_deg: float = _DEFAULT_VALIDATOR["plane_edge_line_final_nms_angle_deg"]
    plane_edge_line_final_nms_overlap: float = _DEFAULT_VALIDATOR["plane_edge_line_final_nms_overlap"]
    plane_edge_inlier_mm: float = _DEFAULT_VALIDATOR["plane_edge_inlier_mm"]
    plane_edge_min_points: int = _DEFAULT_VALIDATOR["plane_edge_min_points"]
    plane_edge_segment_gap_mm: float = _DEFAULT_VALIDATOR["plane_edge_segment_gap_mm"]

    # Step 5: direct 3-D reconstruction and complete-link merging.
    plane_edge_plane_inlier_mm: float = _DEFAULT_VALIDATOR["plane_edge_plane_inlier_mm"]
    plane_edge_min_length_mm: float = _DEFAULT_VALIDATOR["plane_edge_min_length_mm"]
    plane_edge_direction_tolerance_deg: float = _DEFAULT_VALIDATOR["plane_edge_direction_tolerance_deg"]
    plane_edge_merge_distance_mm: float = _DEFAULT_VALIDATOR["plane_edge_merge_distance_mm"]
    plane_edge_merge_gap_mm: float = _DEFAULT_VALIDATOR["plane_edge_merge_gap_mm"]
    plane_edge_merge_angle_deg: float = _DEFAULT_VALIDATOR["plane_edge_merge_angle_deg"]

    # Step 6: graph-local plane/edge refinement and corners.
    plane_edge_refiner_enabled: bool = _DEFAULT_VALIDATOR["plane_edge_refiner_enabled"]
    plane_edge_refine_loss: str = _DEFAULT_VALIDATOR["plane_edge_refine_loss"]
    plane_edge_refine_huber_delta_mm: float = _DEFAULT_VALIDATOR["plane_edge_refine_huber_delta_mm"]
    plane_edge_refine_plane_weight: float = _DEFAULT_VALIDATOR["plane_edge_refine_plane_weight"]
    plane_edge_refine_boundary_weight: float = _DEFAULT_VALIDATOR["plane_edge_refine_boundary_weight"]
    plane_edge_refine_max_nfev: int = _DEFAULT_VALIDATOR["plane_edge_refine_max_nfev"]
    plane_edge_refine_corner_intersection_mm: float = _DEFAULT_VALIDATOR["plane_edge_refine_corner_intersection_mm"]
    plane_edge_refine_corner_endpoint_gap_mm: float = _DEFAULT_VALIDATOR["plane_edge_refine_corner_endpoint_gap_mm"]
    plane_edge_refine_corner_merge_mm: float = _DEFAULT_VALIDATOR["plane_edge_refine_corner_merge_mm"]

    @property
    def support_radius_mm(self):
        return self.query_radius_mm + self.projection_halo_mm


@dataclass
class DetectorConfig:
    """Resolved dense detector settings plus its shared validator config."""

    voxel_3d_mm: float = _DEFAULT_DETECTOR["voxel_3d_mm"]
    min_neighbors: int = _DEFAULT_DETECTOR["min_neighbors"]
    show_3d: bool = _DEFAULT_DETECTOR["show_3d"]
    only: bool = DEFAULTS.only
    centroid_radius_mm: float = _DEFAULT_DETECTOR["centroid_radius_mm"]
    bilateral_sigma_spatial_mm: float = _DEFAULT_DETECTOR["bilateral_sigma_spatial_mm"]
    bilateral_sigma_ced: float = _DEFAULT_DETECTOR["bilateral_sigma_ced"]
    bilateral_radius_mm: float = _DEFAULT_DETECTOR["bilateral_radius_mm"]
    pca_radius_factor: float = _DEFAULT_DETECTOR["pca_radius_factor"]
    pca_radius_mm: float = _DEFAULT_DETECTOR["pca_radius_mm"]
    pca_ratio_saturation_tau: float | None = _DEFAULT_DETECTOR[
        "pca_ratio_saturation_tau"
    ]
    fusion_threshold_mode: str = _DEFAULT_DETECTOR["fusion_threshold_mode"]
    fusion_score_threshold: float = _DEFAULT_DETECTOR["fusion_score_threshold"]
    nms_radius_mm: float = _DEFAULT_DETECTOR["nms_radius_mm"]
    validator: ValidatorConfig | None = field(default_factory=ValidatorConfig)


def _resolve_guided_validator_values(args):
    fine_voxel = args.validator_fine_voxel_mm
    ransac_voxel = args.validator_ransac_voxel_mm
    return {
        "fine_voxel_mm": fine_voxel,
        "fine_search_radius_mm": args.validator_fine_search_radius_x * fine_voxel,
        "fine_ced_radius_mm": args.validator_fine_ced_radius_x * fine_voxel,
        "fine_min_neighbors": args.validator_fine_min_neighbors,
        "fine_bilateral_radius_mm": args.validator_fine_bilateral_radius_x * fine_voxel,
        "fine_bilateral_sigma_spatial_mm": args.validator_fine_bilateral_sigma_spatial_x * fine_voxel,
        "fine_bilateral_sigma_ced": args.validator_fine_bilateral_sigma_ced,
        "fine_pca_radius_mm": args.validator_fine_pca_radius_x * fine_voxel,
        "fine_pca_ratio_saturation_tau": args.validator_fine_pca_ratio_saturation_tau,
        "fine_fusion_threshold_mode": args.validator_fine_fusion_threshold_mode.replace("-", "_"),
        "fine_fusion_score_threshold": args.validator_fine_fusion_score_threshold,
        "fine_fusion_peak_relative_ratio": args.validator_fine_fusion_peak_relative_ratio,
        "fine_nms_method": args.validator_fine_nms_method,
        "fine_nms_radius_mm": args.validator_fine_nms_radius_x * fine_voxel,
        "ransac_voxel_mm": ransac_voxel,
        "ransac_radius_mm": args.validator_ransac_radius_x * ransac_voxel,
        "inlier_threshold_mm": args.validator_inlier_threshold_x * ransac_voxel,
        "guided_sigma_mm": args.validator_guided_sigma_x * ransac_voxel,
        "max_planes": args.validator_max_planes,
        "ransac_iterations": args.validator_ransac_iterations,
        "min_inliers": args.validator_min_inliers,
        "min_inlier_ratio": args.validator_min_inlier_ratio,
        "random_seed": args.validator_random_seed,
    }


_DEFAULT_GUIDED_VALIDATOR = _resolve_guided_validator_values(DEFAULTS)


@dataclass(frozen=True)
class GuidedValidatorConfig:
    """Resolved stage-specific voxel sizes and mm lengths; CLI/defaults use *_x."""
    fine_voxel_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_voxel_mm"]
    fine_search_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_search_radius_mm"]
    fine_ced_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_ced_radius_mm"]
    fine_min_neighbors: int = _DEFAULT_GUIDED_VALIDATOR["fine_min_neighbors"]
    fine_bilateral_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_bilateral_radius_mm"]
    fine_bilateral_sigma_spatial_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_bilateral_sigma_spatial_mm"]
    fine_bilateral_sigma_ced: float = _DEFAULT_GUIDED_VALIDATOR["fine_bilateral_sigma_ced"]
    fine_pca_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_pca_radius_mm"]
    fine_pca_ratio_saturation_tau: float | None = _DEFAULT_GUIDED_VALIDATOR["fine_pca_ratio_saturation_tau"]
    fine_fusion_threshold_mode: str = _DEFAULT_GUIDED_VALIDATOR["fine_fusion_threshold_mode"]
    fine_fusion_score_threshold: float = _DEFAULT_GUIDED_VALIDATOR["fine_fusion_score_threshold"]
    fine_fusion_peak_relative_ratio: float = _DEFAULT_GUIDED_VALIDATOR["fine_fusion_peak_relative_ratio"]
    fine_nms_method: str = _DEFAULT_GUIDED_VALIDATOR["fine_nms_method"]
    fine_nms_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["fine_nms_radius_mm"]
    ransac_voxel_mm: float = _DEFAULT_GUIDED_VALIDATOR["ransac_voxel_mm"]
    ransac_radius_mm: float = _DEFAULT_GUIDED_VALIDATOR["ransac_radius_mm"]
    max_planes: int = DEFAULTS.validator_max_planes
    ransac_iterations: int = DEFAULTS.validator_ransac_iterations
    inlier_threshold_mm: float = _DEFAULT_GUIDED_VALIDATOR["inlier_threshold_mm"]
    guided_sigma_mm: float = _DEFAULT_GUIDED_VALIDATOR["guided_sigma_mm"]
    min_inliers: int = DEFAULTS.validator_min_inliers
    min_inlier_ratio: float = DEFAULTS.validator_min_inlier_ratio
    random_seed: int = DEFAULTS.validator_random_seed

    def validate(self):
        numeric = [value for value in vars(self).values() if isinstance(value, (int, float, np.number))]
        if not np.isfinite(numeric).all():
            raise ValueError("Guided validator settings must be finite")
        for name in ("fine_min_neighbors", "max_planes", "ransac_iterations", "min_inliers", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if min(self.fine_voxel_mm, self.ransac_voxel_mm, self.fine_search_radius_mm,
               self.fine_ced_radius_mm, self.fine_nms_radius_mm,
               self.fine_bilateral_radius_mm, self.fine_bilateral_sigma_spatial_mm,
               self.fine_bilateral_sigma_ced, self.fine_pca_radius_mm,
               self.ransac_radius_mm, self.inlier_threshold_mm, self.guided_sigma_mm) <= 0:
            raise ValueError("Validator radii, threshold and guided sigma must be positive")
        if not 0 <= self.max_planes <= 3 or self.ransac_iterations < 1:
            raise ValueError("Require 0..3 planes and at least one RANSAC iteration")
        if self.min_inliers < 3 or self.random_seed < 0:
            raise ValueError("Require at least 3 inliers and a nonnegative seed")
        if not 0 <= self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if self.fine_min_neighbors < 3:
            raise ValueError("fine_min_neighbors must be at least 3")
        if self.fine_pca_ratio_saturation_tau is not None and self.fine_pca_ratio_saturation_tau <= 0:
            raise ValueError("Fine PCA saturation tau must be positive or None")
        if self.fine_fusion_threshold_mode not in ("fixed", "global_mean", "peak_relative"):
            raise ValueError("Fine fusion threshold mode must be fixed, global_mean or peak_relative")
        if not 0 <= self.fine_fusion_score_threshold <= 1:
            raise ValueError("Fine fusion threshold must be in [0, 1]")
        if not 0 <= self.fine_fusion_peak_relative_ratio <= 1:
            raise ValueError("Fine peak-relative ratio must be in [0, 1]")
        if self.fine_nms_method not in ("greedy", "local_maximum"):
            raise ValueError("Fine NMS method must be greedy or local_maximum")


@dataclass
class ValidatorDetectorConfig(DetectorConfig):
    mode: str = "validator"
    validator: GuidedValidatorConfig = field(default_factory=GuidedValidatorConfig)


@dataclass(frozen=True)
class ExtremityConfig:
    """Resolved mm lengths; multiplier defaults live in SettingsDefaults above."""

    radii_mm: tuple[float, ...] = _DEFAULT_EXTREMITY["radii_mm"]
    padding_mm: float = _DEFAULT_EXTREMITY["padding_mm"]
    directions: int = _DEFAULT_EXTREMITY["directions"]
    object_clustering: bool = _DEFAULT_EXTREMITY["object_clustering"]
    cluster_radius_mm: float = _DEFAULT_EXTREMITY["cluster_radius_mm"]
    nms_radius_mm: float = _DEFAULT_EXTREMITY["nms_radius_mm"]
    nms_method: str = _DEFAULT_EXTREMITY["nms_method"]
    min_vote_fraction: float = _DEFAULT_EXTREMITY["min_vote_fraction"]
    voting_mode: str = _DEFAULT_EXTREMITY["voting_mode"]
    soft_tau_mm: float = _DEFAULT_EXTREMITY["soft_tau_mm"]
    soft_alpha: float = _DEFAULT_EXTREMITY["soft_alpha"]
    voting_dtype: str = _DEFAULT_EXTREMITY["voting_dtype"]

    def validate(self):
        values = (*self.radii_mm, self.padding_mm, self.cluster_radius_mm,
                  self.nms_radius_mm, self.min_vote_fraction, self.soft_tau_mm, self.soft_alpha)
        if not self.radii_mm or not np.isfinite(values).all():
            raise ValueError("Extremity radii and numeric settings must be finite")
        if min(self.radii_mm) <= 0 or len(set(self.radii_mm)) != len(self.radii_mm):
            raise ValueError("Extremity radii must be positive and distinct")
        if min(self.cluster_radius_mm, self.nms_radius_mm) <= 0:
            raise ValueError("Cluster and NMS radii must be positive")
        if self.padding_mm < 0:
            raise ValueError("Padding must be nonnegative")
        if not isinstance(self.directions, (int, np.integer)) or self.directions < 1:
            raise ValueError("Direction count must be a positive integer")
        if not 0 <= self.min_vote_fraction <= 1:
            raise ValueError("Minimum vote fraction must be in [0, 1]")
        if self.nms_method not in ("local_maximum", "greedy"):
            raise ValueError("NMS method must be local_maximum or greedy")
        if self.voting_mode not in ("hard", "soft_dense", "soft_partition", "soft_sparse"):
            raise ValueError("Unknown ball voting mode")
        if self.soft_tau_mm <= 0 or self.soft_alpha <= 0:
            raise ValueError("Soft tau and alpha must be positive")
        if self.voting_dtype not in ("float32", "float64"):
            raise ValueError("Voting dtype must be float32 or float64")


@dataclass
class ExtremityDetectorConfig(DetectorConfig):
    mode: str = "extremity256"
    extremity: ExtremityConfig = field(default_factory=ExtremityConfig)


def parse_args(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Point cloud -> query-local CornerVINS planes -> plane-edge graph "
            "or whole-cloud saliency candidates."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"{BUILD} | {Path(__file__).resolve()}",
    )

    # Shared input, preprocessing, and display options for every mode.
    shared = parser.add_argument_group("shared input and preprocessing")
    shared.add_argument("--input", "-i", help="STL path or Kinect 7-Scenes path")
    shared.add_argument("--source", choices=("stl", "kinect", "realsense"))
    shared.add_argument("--frame-index", type=int)
    shared.add_argument("--fps", type=int, help="Live RealSense camera FPS")
    shared.add_argument("--mode", choices=("fine", "detector", "ball", "extremity256", "validator"))
    shared.add_argument("--stl-unit-mm", type=float)
    shared.add_argument("--width", type=int)
    shared.add_argument("--height", type=int)
    shared.add_argument("--fov-deg", type=float)
    shared.add_argument("--voxel-3d-mm", "--voxel-mm", dest="voxel_3d_mm", type=float)
    shared.add_argument("--min-neighbors", type=int)
    shared.add_argument("--no-3d-view", action="store_true")
    shared.add_argument("--min-depth-m", type=float)
    shared.add_argument("--max-depth-m", type=float)
    shared.add_argument(
        "--centroid-radius-x", type=float,
        help="CED centroid radius / 3-D voxel (shared by fine and detector)",
    )
    perturbation = parser.add_argument_group("shared synthetic perturbation")
    guided = parser.add_argument_group("validator: separate Fine/RANSAC voxels and voxel multipliers")
    for name, kind in (
        ("fine_voxel_mm", float), ("fine_search_radius_x", float), ("fine_ced_radius_x", float),
        ("fine_nms_radius_x", float), ("ransac_voxel_mm", float), ("ransac_radius_x", float),
        ("fine_min_neighbors", int), ("fine_bilateral_radius_x", float),
        ("fine_bilateral_sigma_spatial_x", float), ("fine_bilateral_sigma_ced", float),
        ("fine_pca_radius_x", float), ("fine_pca_ratio_saturation_tau", float),
        ("fine_fusion_score_threshold", float),
        ("fine_fusion_peak_relative_ratio", float),
        ("max_planes", int), ("ransac_iterations", int), ("inlier_threshold_x", float),
        ("guided_sigma_x", float), ("min_inliers", int),
        ("min_inlier_ratio", float), ("random_seed", int),
    ):
        guided.add_argument("--validator-" + name.replace("_", "-"), type=kind)
    guided.add_argument("--validator-fine-fusion-threshold-mode", choices=("fixed", "global-mean", "peak-relative"))
    guided.add_argument("--validator-fine-nms-method", choices=("greedy", "local_maximum"))
    ball = parser.add_argument_group("ball / extremity256: query-local vertex hypotheses (3-D voxel multipliers)")
    ball.add_argument("--ball-radii-x", type=float, nargs="+",
                      help=f"candidate radii / voxel_3d_mm (default: {DEFAULTS.ball_radii_x})")
    ball.add_argument("--ball-padding-x", type=float, help="search padding / voxel_3d_mm")
    ball.add_argument("--ball-directions", type=int)
    ball.add_argument("--ball-object-clustering", action=argparse.BooleanOptionalAction)
    ball.add_argument("--ball-cluster-radius-x", type=float,
                      help=f"Euclidean connectivity radius / voxel_3d_mm (default: {DEFAULTS.ball_cluster_radius_x:g})")
    ball.add_argument("--ball-nms-radius-x", type=float, help="vertex NMS radius / voxel_3d_mm")
    ball.add_argument("--ball-nms-method", choices=("local_maximum", "greedy"))
    ball.add_argument("--ball-min-vote-fraction", type=float)
    ball.add_argument("--ball-voting-mode", choices=("hard", "soft_dense", "soft_partition", "soft_sparse"))
    ball.add_argument("--ball-soft-tau-x", type=float,
                      help=f"sigmoid temperature / voxel_3d_mm (default: {DEFAULTS.ball_soft_tau_x:g})")
    ball.add_argument("--ball-soft-alpha", type=float,
                      help=f"sparse cutoff in tau units (default: {DEFAULTS.ball_soft_alpha:g})")
    ball.add_argument("--ball-voting-dtype", choices=("float32", "float64"))

    # Fine mode options follow the clicked-query pipeline order exactly.
    validator_query = parser.add_argument_group("fine 0/6: query geometry")
    validator_cornervins = parser.add_argument_group(
        "fine 1/6: query-core CornerVINS AHC -> RANSAC/refit -> merge/hull"
    )
    validator_projection = parser.add_argument_group(
        "fine 2/6: padded projection and continuous boundary"
    )
    validator_hough = parser.add_argument_group(
        "fine 3/6: weighted Hough and accumulator NMS"
    )
    validator_line = parser.add_argument_group(
        "fine 4/6: continuous 2-D line refinement and final NMS"
    )
    validator_restore = parser.add_argument_group(
        "fine 5/6: 3-D line reconstruction and finite-edge merging"
    )
    validator_refiner = parser.add_argument_group(
        "fine 6/6: graph-local edge refinement and corners"
    )
    # CornerVINS Step 1: organized-image AHC segmentation and region filtering.
    validator_cornervins.add_argument(
        "--cornervins-ahc-block-size", type=int,
        help="AHC square-block side in organized depth pixels (default: 3 pixels)",
    )
    validator_cornervins.add_argument(
        "--cornervins-ahc-mse-threshold-mm2", type=float,
        help="fixed AHC plane-fit MSE in mm^2; omitted uses Feng's depth-noise model",
    )
    validator_cornervins.add_argument(
        "--cornervins-ahc-refinement", dest="cornervins_ahc_refinement",
        action="store_true", help="enable Feng Algorithm 4 erosion/region-growing refinement",
    )
    validator_cornervins.add_argument(
        "--no-cornervins-ahc-refinement", dest="cornervins_ahc_refinement",
        action="store_false", help="skip AHC erosion/region-growing refinement",
    )
    validator_cornervins.add_argument(
        "--cornervins-region-mode", choices=("paper", "roi_test"),
        help="minimum-region policy: paper=20000 points, roi_test=local threshold",
    )
    validator_cornervins.add_argument(
        "--cornervins-min-region-points-paper", type=int,
        help="paper-mode minimum AHC region size (default: 20000 points)",
    )
    validator_cornervins.add_argument(
        "--cornervins-min-region-points-roi", type=int,
        help="roi_test minimum AHC region size (default: 100 points)",
    )

    # CornerVINS Step 2: per-region RANSAC and least-squares plane refit.
    validator_cornervins.add_argument(
        "--cornervins-downsample-leaf-mm", type=float,
        help="voxel leaf used only for deterministic RANSAC sampling (default: 50 mm)",
    )
    validator_cornervins.add_argument(
        "--cornervins-ransac-threshold-mm", type=float,
        help="full-resolution point-to-plane inlier gate (default: 15 mm)",
    )
    validator_cornervins.add_argument(
        "--cornervins-ransac-min-inlier-ratio", type=float,
        help="minimum accepted inlier fraction in (0,1] (default: 0.90)",
    )
    validator_cornervins.add_argument(
        "--cornervins-ransac-max-trials", type=int,
        help="maximum random plane hypotheses per AHC region (default: 256)",
    )
    validator_cornervins.add_argument(
        "--cornervins-ransac-seed", type=int,
        help="nonnegative deterministic RANSAC seed (default: 0)",
    )

    # CornerVINS Step 3: repeated plane merging.
    validator_cornervins.add_argument(
        "--cornervins-merge-angle-deg", type=float,
        help="strict unoriented normal-angle merge gate (default: 5 deg)",
    )
    validator_cornervins.add_argument(
        "--cornervins-merge-distance-mm", type=float,
        help="strict symmetric plane-distance merge gate (default: 20 mm)",
    )

    # CornerVINS visualization switches; numeric results are unchanged.
    validator_cornervins.add_argument(
        "--no-cornervins-ahc-view", dest="cornervins_show_ahc_regions",
        action="store_false",
        help="hide 1/4 coarse-AHC and 2/4 refined-AHC views",
    )
    validator_cornervins.add_argument(
        "--no-cornervins-plane-view", dest="cornervins_show_plane_patches",
        action="store_false", help="hide 3/4 RANSAC/refit plane view",
    )
    validator_cornervins.add_argument(
        "--no-cornervins-hulls", dest="cornervins_show_convex_hulls",
        action="store_false", help="hide 4/4 merged-plane convex hull view",
    )
    validator_query.add_argument("--voxel-2d-mm", "--uv-voxel-mm", dest="voxel_2d_mm", type=float)
    validator_query.add_argument("--query-radius-x", "--query-radius-vox", dest="query_radius_x", type=float)
    validator_projection.add_argument("--boundary-radius-x", "--boundary-radius-vox", dest="boundary_radius_x", type=float)
    validator_projection.add_argument("--boundary-epsilon-x", "--boundary-epsilon-vox", dest="boundary_epsilon_x", type=float)
    validator_projection.add_argument(
        "--projection-halo-mm", type=float,
        help="extra projection/boundary context beyond the query radius in mm (default: 50)",
    )
    validator_projection.add_argument("--boundary-sigma-x", type=float)
    validator_projection.add_argument("--plane-edge-graph", dest="plane_edge_graph_enabled",
        action="store_true", help="CornerVINS planes -> padded 2-D boundary/Hough -> 3-D finite edge graph (default)")
    validator_projection.add_argument("--no-plane-edge-graph", dest="plane_edge_graph_enabled",
        action="store_false", help="disable the default plane -> edge graph workflow")
    validator_projection.add_argument("--plane-edge-debug-2d", dest="plane_edge_debug_2d", action="store_true",
        help="show the per-plane padded projection / boundary / Hough GUI (default)")
    validator_projection.add_argument("--no-plane-edge-debug-2d", dest="plane_edge_debug_2d", action="store_false",
        help="hide the per-plane 2-D diagnostic GUI")
    validator_projection.add_argument("--plane-edge-export-json", help="write full numerical plane/edge graph as JSON")
    validator_projection.add_argument("--plane-edge-alpha-x", type=float, help="alpha radius / voxel3D (0=adaptive)")
    validator_hough.add_argument("--plane-edge-hough-bin-x", type=float,
        help="2-D Hough rho bin / voxel_3d_mm")
    validator_hough.add_argument("--plane-edge-hough-angle-deg", type=float,
        help="2-D Hough theta bin in degrees")
    validator_hough.add_argument("--plane-edge-hough-min-mean-weight", type=float,
        help="minimum weighted-vote/count mean in [0,1] for an eligible Hough bin")
    validator_hough.add_argument("--plane-edge-hough-score-threshold-mode",
        choices=("off", "fixed", "peak-relative", "global-mean"),
        help="weighted Hough score gate: off, absolute, peak-relative, or eligible-bin mean")
    validator_hough.add_argument("--plane-edge-hough-score-threshold", type=float,
        help="absolute score for fixed mode or [0,1] peak fraction for peak-relative mode")
    validator_hough.add_argument("--plane-edge-hough-nms-mode", choices=("greedy", "local-max"),
        help="2-D Hough peak suppression: greedy selection or simultaneous local maxima")
    validator_hough.add_argument("--plane-edge-hough-nms-angle-deg", type=float,
        help="2-D Hough peak-NMS half-width in degrees")
    validator_hough.add_argument("--plane-edge-hough-nms-rho-x", type=float,
        help="2-D Hough peak-NMS rho half-width / voxel_3d_mm")
    validator_hough.add_argument("--plane-edge-max-peaks", type=int,
        help="maximum accepted Hough peaks per projected plane")
    validator_line.add_argument(
        "--plane-edge-line-refinement", dest="plane_edge_line_refinement_enabled",
        action="store_true",
        help="continuously refine each Hough seed in rho/theta before finite-segment NMS (default)",
    )
    validator_line.add_argument(
        "--no-plane-edge-line-refinement", dest="plane_edge_line_refinement_enabled",
        action="store_false", help="use the unrefined TLS/gap 2-D segment path",
    )
    validator_line.add_argument("--plane-edge-line-refinement-max-iterations", type=int,
        help="maximum iterations for each inside-direction rho/theta optimization")
    validator_line.add_argument("--plane-edge-line-final-nms-angle-deg", type=float,
        help="angle threshold for refined finite-segment NMS")
    validator_line.add_argument("--plane-edge-line-final-nms-overlap", type=float,
        help="minimum projected interval overlap ratio for refined finite-segment NMS")
    validator_line.add_argument("--plane-edge-inlier-x", type=float,
        help="2-D TLS line inlier distance / voxel_3d_mm")
    validator_line.add_argument("--plane-edge-min-points", type=int)
    validator_line.add_argument("--plane-edge-segment-gap-x", type=float,
        help="2-D finite-segment split gap / voxel_3d_mm")
    validator_restore.add_argument("--plane-edge-plane-inlier-x", type=float)
    validator_restore.add_argument("--plane-edge-min-length-x", type=float)
    validator_restore.add_argument("--plane-edge-direction-tolerance-deg", type=float,
        help="maximum angle between lifted 2-D direction and fitted 3-D PCA line")
    validator_restore.add_argument("--plane-edge-merge-distance-x", type=float,
        help="3-D line-offset threshold / voxel_3d_mm for complete-link merging")
    validator_restore.add_argument("--plane-edge-merge-gap-x", type=float,
        help="3-D finite-interval gap / voxel_3d_mm for merging")
    validator_restore.add_argument("--plane-edge-merge-angle-deg", type=float,
        help="3-D direction-angle threshold for merging")
    validator_refiner.add_argument("--plane-edge-refiner", dest="plane_edge_refiner_enabled",
        action="store_true", help="refine connected plane-edge graphs after Hough detection (default)")
    validator_refiner.add_argument("--no-plane-edge-refiner", dest="plane_edge_refiner_enabled",
        action="store_false", help="keep only the unrefined Hough/PCA edge graph")
    validator_refiner.add_argument("--plane-edge-refine-loss",
        choices=("huber", "soft-l1", "linear"),
        help="robust loss used by the post-processing refiner")
    validator_refiner.add_argument("--plane-edge-refine-huber-delta-x", type=float,
        help="refiner robust-loss transition / voxel_3d_mm")
    validator_refiner.add_argument("--plane-edge-refine-plane-weight", type=float,
        help="confidence multiplier for refined plane residuals")
    validator_refiner.add_argument("--plane-edge-refine-boundary-weight", type=float,
        help="confidence multiplier for 2-D boundary line residuals")
    validator_refiner.add_argument("--plane-edge-refine-max-nfev", type=int,
        help="maximum residual evaluations for each small refinement problem")
    validator_refiner.add_argument("--plane-edge-refine-corner-intersection-x", type=float,
        help="maximum closest-line gap for a refined corner / voxel_3d_mm")
    validator_refiner.add_argument("--plane-edge-refine-corner-endpoint-gap-x", type=float,
        help="finite-edge extension allowed for refined corners / voxel_3d_mm")
    validator_refiner.add_argument("--plane-edge-refine-corner-merge-x", type=float,
        help="spatial merge radius for repeated refined corners / voxel_3d_mm")
    # Detector options follow the dense whole-cloud pipeline order.
    detector = parser.add_argument_group("detector: dense CED -> fusion -> NMS")
    detector.add_argument("--only", action="store_true",
                          help="detector only: stop at NMS and show just the NMS Fusion heatmap")
    detector.add_argument(
        "--pca-radius-factor", type=float,
        help="CED-weighted PCA radius / CED radius (configured default: 1.5)",
    )
    detector.add_argument(
        "--bilateral-sigma-spatial-mm", type=float,
        help="bilateral spatial sigma in mm (default: 0.5 x CED radius)",
    )
    detector.add_argument(
        "--bilateral-sigma-ced", type=float,
        help="bilateral range sigma for normalized CED (default: 0.05)",
    )
    detector.add_argument(
        "--bilateral-radius-mm", type=float,
        help="bilateral neighborhood radius in mm (default: CED radius)",
    )
    detector.add_argument(
        "--pca-ratio-saturation-tau", type=float,
        help="tau for tau*tanh(PCA-ratio/tau) (configured default: 0.5)",
    )
    detector.add_argument(
        "--fusion-threshold-mode", choices=("fixed", "global-mean"),
        help=(
            "detector threshold source: configured constant or mean of all "
            "finite fusion-valid scores"
        ),
    )
    detector.add_argument(
        "--fusion-score-threshold", type=float,
        help="strict fixed fusion threshold used when mode=fixed",
    )
    detector.add_argument(
        "--nms-radius-mm", type=float,
        help="3-D candidate NMS radius in mm (configured default: 50)",
    )

    perturbation.add_argument("--noise-mm", type=float)
    perturbation.add_argument("--outlier-rate", type=float)
    perturbation.add_argument("--outlier-mm", type=float)
    perturbation.add_argument("--dropout-rate", type=float)
    perturbation.add_argument("--dropout-patches", type=int)
    perturbation.add_argument("--dropout-patch-radius-px", type=int)
    perturbation.add_argument("--quantization-mm", type=float)
    perturbation.add_argument("--seed", type=int)

    parser.set_defaults(**vars(DEFAULTS))
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.only and args.mode != "detector":
        parser.error("--only requires --mode detector")
    has_score_value = any(
        token == "--plane-edge-hough-score-threshold"
        or token.startswith("--plane-edge-hough-score-threshold=")
        for token in raw_argv
    )
    has_score_mode = any(
        token == "--plane-edge-hough-score-threshold-mode"
        or token.startswith("--plane-edge-hough-score-threshold-mode=")
        for token in raw_argv
    )
    if has_score_value and not has_score_mode:
        args.plane_edge_hough_score_threshold_mode = "fixed"
    if not any(flag in raw_argv for flag in ("--plane-edge-graph", "--no-plane-edge-graph")):
        args.plane_edge_graph_enabled = args.mode == "fine"
    if not any(flag in raw_argv for flag in ("--plane-edge-debug-2d", "--no-plane-edge-debug-2d")):
        args.plane_edge_debug_2d = args.mode == "fine"
    if args.source in ("stl", "kinect") and not args.input:
        parser.error(f"--input is required for --source {args.source}")
    if args.frame_index < 0:
        parser.error("--frame-index must be >= 0")
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    return args


def configure(args):
    numeric = [value for value in vars(args).values() if isinstance(value, (int, float))]
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric arguments must be finite.")
    if args.width < 64 or args.height < 64 or not 5 <= args.fov_deg <= 90:
        raise ValueError("Require width/height >=64 and FOV in [5,90].")
    if min(args.stl_unit_mm, args.voxel_3d_mm, args.centroid_radius_x) <= 0:
        raise ValueError("STL scale, 3-D voxel size, and CED radius must be positive.")
    if args.min_neighbors < 3:
        raise ValueError("min-neighbors must be >= 3.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("Require 0 < min-depth < max-depth.")
    if min(
        args.noise_mm, args.outlier_mm, args.dropout_patches,
        args.dropout_patch_radius_px, args.quantization_mm,
    ) < 0:
        raise ValueError("Noise, outlier magnitude, dropout, and quantization must be nonnegative.")
    validate_probability("outlier-rate", args.outlier_rate)
    validate_probability("dropout-rate", args.dropout_rate)
    # CornerVINS is the sole plane extraction path.
    if args.cornervins_ahc_block_size < 2:
        raise ValueError("cornervins-ahc-block-size must be at least 2")
    if (
        args.cornervins_ahc_mse_threshold_mm2 is not None
        and args.cornervins_ahc_mse_threshold_mm2 <= 0.0
    ):
        raise ValueError("cornervins-ahc-mse-threshold-mm2 must be positive")
    if args.cornervins_region_mode not in ("paper", "roi_test"):
        raise ValueError("cornervins-region-mode must be paper or roi_test")
    if min(
        args.cornervins_min_region_points_paper,
        args.cornervins_min_region_points_roi,
    ) < 3:
        raise ValueError("CornerVINS minimum region sizes must be at least 3")
    if min(
        args.cornervins_downsample_leaf_mm,
        args.cornervins_ransac_threshold_mm,
        args.cornervins_merge_angle_deg,
        args.cornervins_merge_distance_mm,
    ) <= 0.0:
        raise ValueError("CornerVINS metric thresholds must be positive")
    if not 0.0 < args.cornervins_ransac_min_inlier_ratio <= 1.0:
        raise ValueError("cornervins-ransac-min-inlier-ratio must be in (0, 1]")
    if args.cornervins_ransac_max_trials < 1 or args.cornervins_ransac_seed < 0:
        raise ValueError("CornerVINS RANSAC trials must be positive and seed nonnegative")
    if args.cornervins_merge_angle_deg >= 90.0:
        raise ValueError("cornervins-merge-angle-deg must be in (0, 90)")
    if args.projection_halo_mm < 0.0:
        raise ValueError("projection-halo-mm must be nonnegative")

    if args.mode in ("ball", "extremity256"):
        extremity = ExtremityConfig(**_resolve_extremity_values(args))
        extremity.validate()
    if args.mode in ("detector", "ball", "extremity256", "validator"):
        if args.pca_radius_factor <= 0:
            raise ValueError("pca-radius-factor must be > 0.")
        if args.nms_radius_mm is not None and args.nms_radius_mm <= 0:
            raise ValueError("nms-radius-mm must be > 0.")
        for name in ("bilateral_sigma_spatial_mm", "bilateral_radius_mm"):
            value = getattr(args, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be > 0.")
        if args.bilateral_sigma_ced <= 0:
            raise ValueError("bilateral-sigma-ced must be > 0.")
        tau = args.pca_ratio_saturation_tau
        if tau is not None and (not np.isfinite(tau) or tau <= 0.0):
            raise ValueError(
                "pca-ratio-saturation-tau must be a positive finite number or omitted."
            )
        if not 0.0 <= args.fusion_score_threshold <= 1.0:
            raise ValueError("fusion-score-threshold must be in [0, 1].")
        if args.fusion_threshold_mode.replace("-", "_") not in (
            "fixed", "global_mean",
        ):
            raise ValueError("fusion-threshold-mode must be fixed or global-mean.")
        if args.mode == "validator":
            guided_cfg = GuidedValidatorConfig(**_resolve_guided_validator_values(args))
            guided_cfg.validate()
            cfg = ValidatorDetectorConfig(**_resolve_detector_values(args), validator=guided_cfg)
        elif args.only:
            cfg = DetectorConfig(**_resolve_detector_values(args), validator=None)
        else:
            validator_values = _resolve_validator_values(args)
            validator_values.update({
                "plane_edge_graph_enabled": True,
                "plane_edge_debug_2d": False,
                "show_3d": False,
            })
            validator_cfg = ValidatorConfig(**validator_values)
            from .validator_pipeline import make_plane_edge_config, make_refiner_config
            make_plane_edge_config(validator_cfg).validate()
            if validator_cfg.plane_edge_refiner_enabled:
                make_refiner_config(validator_cfg).validate()
            config_class = ExtremityDetectorConfig if args.mode in ("ball", "extremity256") else DetectorConfig
            cfg = config_class(
                **_resolve_detector_values(args),
                validator=validator_cfg,
                **({"extremity": extremity, "mode": args.mode} if args.mode in ("ball", "extremity256") else {}),
            )
    else:
        positive = [
            args.voxel_2d_mm,
            args.query_radius_x,
            args.boundary_radius_x,
            args.boundary_epsilon_x,
            args.boundary_sigma_x,
        ]
        if min(positive) <= 0:
            raise ValueError(
                "Validator voxel sizes, radii, spatial multipliers, and sigma "
                "must be positive."
            )
        if args.plane_edge_graph_enabled:
            if args.plane_edge_alpha_x < 0:
                raise ValueError("--plane-edge-alpha-x must be >= 0")
            from .plane_edge_graph_pipeline import PlaneEdgeGraphConfig
            PlaneEdgeGraphConfig(
                alpha_radius_mm=args.plane_edge_alpha_x if args.plane_edge_alpha_x > 0 else None,
                hough_rho_bin_mm=args.plane_edge_hough_bin_x,
                hough_theta_step_deg=args.plane_edge_hough_angle_deg,
                hough_min_mean_weight=args.plane_edge_hough_min_mean_weight,
                hough_score_threshold_mode=args.plane_edge_hough_score_threshold_mode.replace("-", "_"),
                hough_score_threshold=args.plane_edge_hough_score_threshold,
                hough_nms_mode=args.plane_edge_hough_nms_mode.replace("-", "_"),
                hough_nms_angle_deg=args.plane_edge_hough_nms_angle_deg,
                hough_nms_rho_mm=args.plane_edge_hough_nms_rho_x,
                hough_inlier_mm=args.plane_edge_inlier_x,
                plane_inlier_mm=args.plane_edge_plane_inlier_x,
                line_min_points=args.plane_edge_min_points,
                hough_min_votes=args.plane_edge_min_points,
                line_min_length_mm=args.plane_edge_min_length_x,
                segment_gap_mm=args.plane_edge_segment_gap_x,
                merge_distance_mm=args.plane_edge_merge_distance_x,
                merge_gap_mm=args.plane_edge_merge_gap_x,
                merge_angle_deg=args.plane_edge_merge_angle_deg,
                hough_max_peaks=args.plane_edge_max_peaks,
                line_refinement_enabled=args.plane_edge_line_refinement_enabled,
                line_refinement_max_iterations=args.plane_edge_line_refinement_max_iterations,
                line_refinement_final_nms_angle_deg=args.plane_edge_line_final_nms_angle_deg,
                line_refinement_final_nms_overlap=args.plane_edge_line_final_nms_overlap,
                direction_tolerance_deg=args.plane_edge_direction_tolerance_deg,
            ).validate()
            from .refiner import HoughEdgeRefinerConfig
            HoughEdgeRefinerConfig(
                enabled=args.plane_edge_refiner_enabled,
                robust_loss=args.plane_edge_refine_loss.replace("-", "_"),
                huber_delta_mm=args.plane_edge_refine_huber_delta_x,
                plane_confidence_weight=args.plane_edge_refine_plane_weight,
                boundary_confidence_weight=args.plane_edge_refine_boundary_weight,
                max_nfev=args.plane_edge_refine_max_nfev,
                corner_intersection_mm=args.plane_edge_refine_corner_intersection_x,
                corner_endpoint_gap_mm=args.plane_edge_refine_corner_endpoint_gap_x,
                corner_merge_mm=args.plane_edge_refine_corner_merge_x,
            ).validate()
        if 1.5 * args.boundary_epsilon_x >= 0.5 * args.boundary_radius_x:
            raise ValueError("Use boundary-radius-x > 3*boundary-epsilon-x.")
        cfg = ValidatorConfig(**_resolve_validator_values(args))
    noise = NoiseConfig(
        gaussian_sigma_mm=args.noise_mm,
        outlier_rate=args.outlier_rate,
        outlier_mm=args.outlier_mm,
        dropout_rate=args.dropout_rate,
        dropout_patches=args.dropout_patches,
        dropout_patch_radius_px=args.dropout_patch_radius_px,
        quantization_mm=args.quantization_mm,
        seed=args.seed,
    )
    return cfg, noise


# Compatibility aliases for library callers. The removed names are not accepted
# by ``--mode``; new code should import ValidatorConfig / DetectorConfig.
BoundaryConfig = ValidatorConfig
HeatmapConfig = DetectorConfig
DetectorDefaults = SettingsDefaults


__all__ = [
    "GuidedValidatorConfig", "ValidatorDetectorConfig",
    "ValidatorConfig", "DetectorConfig", "ExtremityConfig", "ExtremityDetectorConfig",
    "SettingsDefaults", "DEFAULTS",
    "BoundaryConfig", "HeatmapConfig", "DetectorDefaults", "configure", "parse_args",
]
