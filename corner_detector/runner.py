"""Runtime composition for detector, total and query-local modes."""

from __future__ import annotations

from pathlib import Path

from pointcloud_inputs import (
    capture_realsense_point_cloud,
    capture_stl_point_cloud,
    load_kinect_point_cloud,
)
from pointcloud_inputs.visualization import require_open3d
from pointcloud_inputs.demo import make_demo_cloud

from .constants import BUILD
from .dense_heatmap import run_dense_heatmap_mode
from .ball_query import run_ced_extremity_mode
from .validator_query import prepare_validator_queries
from .total import run_total_mode
from .preprocessing import apply_noise_to_frame, prepare_frozen_from_frame, voxel_downsample_3d
from .settings import configure, parse_args
from .visualization.manager import close_query_views, show_total_results
from .visualization.query_ui import run_validator_mode, tk


def main(argv=None, *, prog=None):
    args = parse_args(argv, prog=prog)
    cfg, noise = configure(args)
    print(f"[BUILD {BUILD}] {Path(__file__).resolve()}", flush=True)
    if args.source == "demo":
        raw, colors, _ = make_demo_cloud()
        points = voxel_downsample_3d(raw, cfg.voxel_3d_mm)
        if len(points) < cfg.min_neighbors:
            raise SystemExit("[INPUT ERROR] Too few demo points after voxelization; reduce --voxel-3d-mm")
        print(f"[DEMO] Lab three-plane corner | seed=7 | raw={len(raw):,} "
              f"voxel={len(points):,} voxel-size={cfg.voxel_3d_mm:g} mm", flush=True)
        return run_dense_heatmap_mode(points, cfg, raw_points_mm=raw,
                                      rgb_points_mm=raw, rgb_colors_rgb=colors)
    if args.source == "dataset":
        from .dataset_input import run_dataset_detector
        try:
            return run_dataset_detector(args.input, cfg, region=args.dataset_region)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"[INPUT ERROR] {exc}") from None
    if args.mode in ("ball", "extremity256"):
        print(f"[MODE] {args.mode.upper()} | unchanged CED/PCA/NMS queries -> local padded 3-D "
              "extremity hypotheses (unverified)", flush=True)
    elif args.mode in ("validator", "new-validator"):
        print(f"[MODE] {args.mode.upper()} | detector NMS -> {'Q10 kept/refined' if cfg.q10_validator is not None else 'original NMS'} seeds -> raw radius crop -> local voxel -> guided sequential planes", flush=True)
    elif args.mode == "total":
        print(f"[MODE] TOTAL | detector NMS -> {'Q10 kept/refined' if cfg.q10_validator is not None else 'original NMS'} seeds -> guided validator -> final scene corners / contributing edges", flush=True)
    elif args.mode == "detector":
        print(
            "[MODE] DETECTOR | whole-cloud saliency + 3-D corner candidates | "
            "diagnostics: Open3D",
            flush=True,
        )
    else:
        validator_pipeline = "PLANE"
        if cfg.plane_edge_graph_enabled:
            validator_pipeline += "->PADDED-2D-EDGE"
        print(
            f"[MODE] FINE | pipeline={validator_pipeline} | "
            "diagnostics: Tk + Agg/Open3D",
            flush=True,
        )
    require_open3d()
    if (args.mode in ("fine", "ball", "validator", "new-validator") or args.mode == "extremity256" and cfg.show_3d) and tk is None:
        raise RuntimeError("Tk is unavailable. Install Tk and Pillow in the active environment.")
    if args.mode in ("ball", "extremity256"):
        print(f"[extremity] {cfg.extremity} | voxel={cfg.voxel_3d_mm:g} mm")
        print("[connectivity] Components do not guarantee object instances; touching objects merge.")
    elif args.mode in ("validator", "new-validator"):
        print(f"[guided planes] {cfg.validator} | voxel={cfg.voxel_3d_mm:g} mm")
    elif args.mode == "total":
        print(f"[total validator] {cfg.validator} | detector voxel={cfg.voxel_3d_mm:g} mm", flush=True)
        print("[total views] 1 Fusion heatmap / NMS | 2 raw scene + final corners / contributing edges", flush=True)
    elif args.mode == "detector":
        factor_path = ("NORMAL PCA -> MIN-MAX LOCAL MINIMUM -> PRODUCT FUSION"
                       if getattr(cfg, "fusion_mode", "pca_ratio") == "local_minimum" else
                       "CED-WEIGHTED PCA -> OPTIONAL TANH SATURATION -> PRODUCT FUSION")
        print(
            "VOXEL POINTS -> RADIUS NEIGHBORHOOD -> "
            f"CED -> OPTIONAL BILATERAL CED -> {factor_path} -> OPTIONAL BILATERAL FUSION -> 3-D NMS"
            + (" | NMS HEATMAP ONLY" if getattr(cfg, "only", False) else "")
        )
        print(
            f"[detector] voxel={cfg.voxel_3d_mm:g} mm | "
            f"CED-radius={cfg.centroid_radius_mm:g} mm | "
            f"bilateral={getattr(cfg, 'bilateral_mode', 'ced')} r={cfg.bilateral_radius_mm:g} mm "
            f"sigma-s={cfg.bilateral_sigma_spatial_mm:g} mm "
            f"sigma-CED={cfg.bilateral_sigma_ced:g} | "
            f"PCA-radius={cfg.pca_radius_mm:g} mm ({cfg.pca_radius_mm / cfg.centroid_radius_mm:g}x CED) | "
            f"PCA-tau={cfg.pca_ratio_saturation_tau} | "
            f"PCA-CED-cutoff={getattr(cfg, 'pca_ced_cutoff', 0.0):g} | "
            f"min-neighbors={cfg.min_neighbors} | "
            f"NMS-radius={cfg.nms_radius_mm:g} mm | "
            f"threshold-mode={getattr(cfg, 'fusion_threshold_mode', 'fixed')} | "
            f"fixed-threshold={cfg.fusion_score_threshold:g} | "
            f"fusion-mode={getattr(cfg, 'fusion_mode', 'pca_ratio')} | "
            f"post-NMS-lambda1/R²-max={getattr(cfg, 'post_nms_lambda1_ratio_max', None)} | "
            "NMS=radius-local-maximum"
        )
        if getattr(cfg, "q10_validator", None) is not None:
            print(f"[Q10 + antipodal Lite] {cfg.q10_validator} | fixed NMS radius={cfg.nms_radius_mm:g} mm "
                  "| CC only at convergence | final green=pass/red=reject/amber=indeterminate", flush=True)
        if getattr(cfg, "validator", None) is not None:
            validator = cfg.validator
            print(
                f"[validator after NMS] query-R={validator.query_radius_mm:g} mm | "
                "planes=CornerVINS | "
                f"boundary-R={validator.boundary_radius_mm:g} mm | "
                f"Hough-NMS={validator.plane_edge_hough_nms_mode} | "
                f"refiner={validator.plane_edge_refiner_enabled}"
            )
    else:
        plane_frontend = "QUERY-CORE ORGANIZED AHC -> RANSAC/REFIT/HULL"
        pipeline = f"QUERY -> {plane_frontend}"
        if cfg.plane_edge_graph_enabled:
            pipeline += (
                " -> PADDED 2-D BOUNDARY + WEIGHTED HOUGH"
                " -> 3-D PLANE/EDGE GRAPH"
            )
        print(pipeline)
        print(
            "[fine views] coarse-AHC/refined-AHC/RANSAC/final=True | "
            f"2-D GUI={cfg.plane_edge_debug_2d}"
        )
        print(
            f"[2-D scale] voxel={cfg.voxel_2d_mm:g} mm | "
            f"query r={cfg.query_radius_mm:g} mm | "
            f"boundary r={cfg.boundary_radius_mm:g} mm | "
            f"epsilon={cfg.boundary_epsilon_mm:g} mm"
        )
        print("[plane extraction] CornerVINS (automatic)")
        print(
            f"[CornerVINS ROI] exact query core={cfg.query_radius_mm:g} mm | "
            "clustering/fitting halo=0 mm | "
            f"projection halo={cfg.projection_halo_mm:g} mm | "
            f"projection max radius={cfg.support_radius_mm:g} mm"
        )
        threshold = (
            "Feng depth-noise model"
            if cfg.cornervins_ahc_mse_threshold_mm2 is None
            else f"{cfg.cornervins_ahc_mse_threshold_mm2:g} mm^2 fixed"
        )
        adaptation = (
            "PAPER 20k rule"
            if cfg.cornervins_region_mode == "paper"
            else (
                "ROI ADAPTATION (not paper): "
                f"min={cfg.cornervins_min_region_points_roi}"
            )
        )
        print(
            f"[CornerVINS] block={cfg.cornervins_ahc_block_size} | "
            f"AHC-MSE={threshold} | refinement={cfg.cornervins_ahc_refinement} | "
            f"region={adaptation} | "
            f"RANSAC={cfg.cornervins_ransac_threshold_mm:g}mm/"
            f"{cfg.cornervins_ransac_min_inlier_ratio:.2f} | "
            f"merge={cfg.cornervins_merge_angle_deg:g}deg/"
            f"{cfg.cornervins_merge_distance_mm:g}mm"
        )
        if cfg.plane_edge_graph_enabled:
            print("[plane->edge graph] query-local | "
                  "padded continuous boundary + weighted Hough -> "
                  "3D PCA/TLS -> finite edge merging -> plane adjacency")
            if cfg.plane_edge_refiner_enabled:
                print(f"[Hough edge refiner] loss={cfg.plane_edge_refine_loss} "
                      f"delta={cfg.plane_edge_refine_huber_delta_mm:g} mm | "
                      f"plane-weight={cfg.plane_edge_refine_plane_weight:g} "
                      f"boundary-weight={cfg.plane_edge_refine_boundary_weight:g} | "
                      "unit=connected plane-edge graph")

    try:
        while True:
            try:
                if args.source == "stl":
                    frame = capture_stl_point_cloud(
                        args.input,
                        stl_unit_mm=args.stl_unit_mm,
                        width=args.width,
                        height=args.height,
                        fov_deg=args.fov_deg,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                    )
                elif args.source == "kinect":
                    frame = load_kinect_point_cloud(
                        args.input,
                        frame_index=args.frame_index,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                    )
                else:
                    frame = capture_realsense_point_cloud(
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                    )
            except FileNotFoundError as exc:
                missing = str(exc)
                hint = (
                    "Set --input to an existing Kinect 7-Scenes sequence, scene "
                    "directory, or ZIP."
                    if args.source == "kinect"
                    else "Set --input to an existing STL file."
                )
                raise SystemExit(
                    f"[INPUT ERROR] File or frame not found: {missing}\n{hint}"
                ) from None
            except RuntimeError as exc:
                if "capture was cancelled" in str(exc).lower():
                    break
                raise
            if frame is None:
                break
            frame = apply_noise_to_frame(
                frame,
                noise,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
            )
            frozen, points = prepare_frozen_from_frame(
                frame, cfg, **({"include_organized_context": False} if args.mode in ("detector", "total", "new-validator") else {}))
            print(
                f"[INPUT] source={frame.source} raw={len(frame.points_mm):,} "
                f"voxel={len(points):,} voxel-size={cfg.voxel_3d_mm:g} mm",
                flush=True,
            )
            if args.mode == "total":
                result = run_total_mode(points, cfg, raw_points_mm=frozen["points_raw_mm"])
                if cfg.show_3d:
                    action = show_total_results(points, frozen["points_raw_mm"], result, cfg)
                    if action == "recapture":
                        del result
                        continue
            elif args.mode == "fine":
                action = run_validator_mode(frame, frozen, cfg)
                if action == "recapture":
                    close_query_views(frozen)
                    continue
            elif args.mode in ("validator", "new-validator"):
                frozen["validator_ced_scores"] = prepare_validator_queries(points, cfg, raw_points_mm=frozen["points_raw_mm"])
                frozen["validator_projection_only"] = bool(getattr(cfg.validator, "projection_only", False))
                seeds = frozen["validator_ced_scores"]["validator_seeds"]
                frozen["validator_query_indices"] = seeds["candidate_indices"].copy()
                frozen["validator_query_positions_mm"] = seeds["positions_mm"].copy()
                print(f"[VALIDATOR seeds] {len(seeds['candidate_indices'])} | {seeds['source']}", flush=True)
                action = run_validator_mode(frame, frozen, cfg)
                if action == "recapture":
                    close_query_views(frozen)
                    continue
            elif args.mode in ("ball", "extremity256"):
                frozen["extremity_batch"] = run_ced_extremity_mode(
                    points, cfg, raw_points_mm=frozen["points_raw_mm"])
                if args.mode == "ball" or cfg.show_3d:
                    action = run_validator_mode(frame, frozen, cfg)
                    if action == "recapture":
                        close_query_views(frozen)
                        continue
            else:
                heatmap_visualization = {}
                registered_colors = getattr(frame, "colors_rgb", None)
                if args.source == "realsense" and registered_colors is not None:
                    heatmap_visualization = {
                        "rgb_points_mm": frame.points_mm,
                        "rgb_colors_rgb": registered_colors,
                    }
                organized_context = frozen.get("organized_plane_context")
                if organized_context is not None:
                    heatmap_visualization["organized_context"] = organized_context
                if getattr(cfg, "q10_validator", None) is not None:
                    heatmap_visualization["raw_points_mm"] = frozen["points_raw_mm"]
                run_dense_heatmap_mode(points, cfg, **heatmap_visualization)
            break
    finally:
        if "frozen" in locals():
            close_query_views(frozen)


__all__ = ["main"]
