"""Centralized visualization controller for detector and validator views."""

from __future__ import annotations

import numpy as np
from .view3d import (
    dense_heatmap_colors,
    dense_heatmap_vmax,
    open_guided_validator_view,
    update_guided_validator_view,
    pick_guided_validator_query,
    pick_validator_fine_query,
    open_cornervins_plane_views,
    open_plane_edge_graph_view,
    open_plane_edge_refiner_view,
    show_dense_score_heatmaps,
    show_detector_validator_results as render_detector_validator_results,
    open_extremity_views,
    open_extremity_overview,
    save_extremity_figure,
    save_extremity_html,
    save_soft_vote_comparison,
    make_extremity_direction_figure,
    pick_extremity_roi_point,
    update_extremity_direction_figure,
    update_extremity_direction_scale,
)


_VIEW_KEYS = (
    "pending_query",
    "pending_validator_query",
    "pending_validator_fine_query",
    "validator_fine_result",
    "validator_view_action",
    "plane_cluster_analysis",
    "plane_edge_graph_result",
    "plane_edge_debug_clustering",
    "plane_edge_debug_config",
    "extremity_result",
    "guided_validator_result",
)


def prepare_guided_validator_view(frozen):
    close_guided_validator_view(frozen)
    display = open_guided_validator_view(
        frozen["points_voxel_mm"], frozen["validator_query_indices"],
        frozen["validator_scene_colors"])
    frozen["guided_validator_display"] = display
    return display


def start_guided_validator_view(frozen):
    from multiprocessing import get_context
    from .query_ui import run_guided_validator_view_process

    close_guided_validator_view(frozen)
    scores = frozen["validator_ced_scores"]
    colors = dense_heatmap_colors(
        scores["fusion_score"], scores["fusion_valid_mask"],
        vmax=dense_heatmap_vmax(scores)["fusion_score"],
    )
    context = get_context("spawn")
    connection, child = context.Pipe()
    process = context.Process(target=run_guided_validator_view_process, args=(
        child, frozen["points_voxel_mm"],
        frozen["validator_query_indices"], colors,
    ), daemon=True)
    try:
        process.start()
    except Exception:
        connection.close()
        raise
    finally:
        child.close()
    frozen["guided_validator_process"] = {"process": process, "connection": connection}


def close_guided_validator_view(frozen):
    worker = frozen.pop("guided_validator_process", None)
    if worker is not None:
        process, connection = worker["process"], worker["connection"]
        try:
            connection.send(("close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        process.join(timeout=0.5)
        # Open3D 0.19 can stall in Filament teardown; the worker owns only display data.
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
        connection.close()
    display = frozen.pop("guided_validator_display", None)
    if display is not None and not display["closed"]:
        display["window"].close()
        display["app"].run_one_tick()
        display["closed"] = True


def prepare_guided_validator_result(frozen, result):
    frozen["guided_validator_result"] = result
    worker = frozen.get("guided_validator_process")
    if worker is not None:
        try:
            worker["connection"].send(("result", result))
        except (BrokenPipeError, EOFError, OSError):
            close_guided_validator_view(frozen)
    display = frozen.get("guided_validator_display")
    if display is not None and not display["closed"]:
        update_guided_validator_view(display, result)


def prepare_plane_edge_graph_view(frozen, result, *, show_3d=True):
    """Install and visualize new opt-in plane/edge graph analysis."""
    frozen['plane_edge_graph_result'] = result
    if show_3d:
        frozen['plane_edge_graph_view'] = open_plane_edge_graph_view(
            frozen['points_voxel_mm'], result,
            frozen.get('plane_edge_graph_view'),
        )
        frozen['plane_edge_refiner_view'] = open_plane_edge_refiner_view(
            frozen['points_voxel_mm'], result,
            frozen.get('plane_edge_refiner_view'),
        )


def prepare_extremity_view(frozen, result, *, show_3d=True):
    """Install stored output and open the three independent ball stages."""
    frozen["extremity_result"] = result
    votes = np.asarray(result["raw_votes"])
    positive = votes[votes > 0]
    print(f"[ball heatmap] {result.get('voting_mode', 'hard')} | "
          f"zero={np.count_nonzero(votes == 0)}/{len(votes)} | "
          f"positive raw range={positive.min() if len(positive) else 0:.4g}.."
          f"{positive.max() if len(positive) else 0:.4g} | "
          f"linear colors [0, {votes.max(initial=0):.4g}]; zero=gray")
    if show_3d:
        frozen["extremity_views"] = open_extremity_views(
            frozen["points_raw_mm"], result, frozen.pop("extremity_views", ()))


def prepare_extremity_overview(frozen, *, show_3d=True):
    """Open the frame-wide cached Ball results alongside query-local views."""
    if show_3d and frozen.get("extremity_overview") is None:
        frozen["extremity_overview"] = open_extremity_overview(
            frozen["points_raw_mm"], frozen["points_voxel_mm"], frozen["extremity_batch"])


def prepare_extremity_direction_inspector(result):
    from ..extremity import fibonacci_directions
    return make_extremity_direction_figure(result, fibonacci_directions(result["config"].directions))


def inspect_extremity_point(display, result, pixel_x, pixel_y, *, scale_index):
    from ..extremity import inspect_point_directions
    local_index = pick_extremity_roi_point(display, pixel_x, pixel_y)
    if local_index is None:
        return None
    diagnostic = inspect_point_directions(result, local_index, scale_index=scale_index)
    update_extremity_direction_figure(display, diagnostic, result)
    return diagnostic


def set_extremity_inspector_scale(display, result, scale_index):
    from ..extremity import inspect_point_directions
    selected = display["diagnostic"]
    if selected is not None:
        diagnostic = inspect_point_directions(result, selected["roi_point_index"], scale_index=scale_index)
        update_extremity_direction_figure(display, diagnostic, result)
        return diagnostic
    update_extremity_direction_scale(display, result["config"].radii_mm[scale_index], result["config"].padding_mm)
    return None


def set_extremity_inspector_mode(display, result, mode):
    if mode not in ("contribution", "top1"):
        raise ValueError("Unknown direction sphere display mode")
    display["sphere_mode"] = mode
    diagnostic = display["diagnostic"]
    if diagnostic is not None:
        update_extremity_direction_figure(display, diagnostic, result)
    return diagnostic


def extremity_summary_rgb(result):
    from .plots import make_extremity_summary_figure, figure_to_rgb
    fig = make_extremity_summary_figure(result)
    try:
        return figure_to_rgb(fig)
    finally:
        fig.clear()


def export_extremity_views(points_mm, result, prefix, *, title="Extremity voting", html=False):
    from pathlib import Path
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_extremity_figure(points_mm, result, prefix.with_suffix(".png"), title=title)
    if html:
        save_extremity_html(points_mm, result, prefix.with_suffix(".html"), title=title)


def export_soft_vote_comparison(points_mm, results, gt_mm, path, *, title):
    save_soft_vote_comparison(points_mm, results, gt_mm, path, title=title)


def export_soft_study_plots(summary, stability, output, *, matches=None):
    from .plots import save_soft_study_plots
    save_soft_study_plots(summary, stability, output, matches=matches)


def show_plane_edge_graph_view(frozen, *, show_3d=True):
    """Open stored stage-4 output after the stage-3 2-D GUI is ready."""
    result = frozen.get("plane_edge_graph_result")
    if not show_3d or result is None:
        return
    frozen["plane_edge_graph_view"] = open_plane_edge_graph_view(
        frozen["points_voxel_mm"], result,
        frozen.get("plane_edge_graph_view"),
    )
    frozen["plane_edge_refiner_view"] = open_plane_edge_refiner_view(
        frozen["points_voxel_mm"], result,
        frozen.get("plane_edge_refiner_view"),
    )


def prepare_cornervins_plane_views(
    frozen,
    analysis,
    *,
    cfg,
    show_3d=True,
):
    """Show coarse/refined AHC, RANSAC planes, and final convex hulls."""
    frozen["plane_cluster_analysis"] = analysis
    if not show_3d:
        return
    context = frozen.get("organized_plane_context")
    if context is None:
        raise ValueError("CornerVINS visualization requires organized_plane_context")
    frozen["cornervins_plane_views"] = open_cornervins_plane_views(
        frozen["points_voxel_mm"],
        context["organized_points_mm"],
        analysis,
        frozen.get("cornervins_plane_views", ()),
        show_ahc=bool(getattr(cfg, "cornervins_show_ahc_regions", True)),
        show_patches=bool(getattr(cfg, "cornervins_show_plane_patches", True)),
        show_hulls=bool(getattr(cfg, "cornervins_show_convex_hulls", True)),
    )


def poll_view_events(frozen):
    """Service non-blocking Open3D windows."""
    worker = frozen.get("guided_validator_process")
    if worker is not None:
        try:
            while worker["connection"].poll():
                kind, value = worker["connection"].recv()
                if kind == "query":
                    frozen["pending_validator_query"] = value
                elif kind == "fine_query":
                    frozen["pending_validator_fine_query"] = value
                elif kind == "action":
                    frozen["validator_view_action"] = value
                elif kind == "error":
                    print(f"[VALIDATOR Open3D] {value}")
            if not worker["process"].is_alive():
                close_guided_validator_view(frozen)
        except (EOFError, OSError):
            close_guided_validator_view(frozen)
    display = frozen.get("guided_validator_display")
    if display is not None:
        running = display["app"].run_one_tick()
        if not running or display["closed"]:
            frozen.pop("guided_validator_display", None)
    for key in (
        "plane_edge_graph_view", "plane_edge_refiner_view",
        "extremity_overview",
    ):
        view = frozen.get(key)
        if view is None:
            continue
        if view.poll_events():
            view.update_renderer()
        else:
            view.destroy_window()
            frozen[key] = None
    for key in ("cornervins_plane_views", "extremity_views"):
        views = list(frozen.get(key, ()))
        if not views:
            continue
        for index, view in enumerate(views):
            if view is None:
                continue
            if view.poll_events():
                view.update_renderer()
            else:
                view.destroy_window()
                views[index] = None
        frozen[key] = tuple(views)


def close_query_views(frozen, *, keep_extremity_overview=False, keep_guided_validator=False):
    if not keep_guided_validator:
        close_guided_validator_view(frozen)
    for key in (
        "plane_edge_graph_view", "plane_edge_refiner_view",
        "extremity_overview",
    ):
        if key == "extremity_overview" and keep_extremity_overview:
            continue
        view = frozen.pop(key, None)
        if view is not None:
            view.destroy_window()
    for key in ("cornervins_plane_views", "extremity_views"):
        for view in frozen.pop(key, ()):
            if view is not None:
                view.destroy_window()
    for key in _VIEW_KEYS:
        frozen.pop(key, None)


def show_dense_heatmaps(
    points_mm,
    score_result,
    cfg,
    *,
    rgb_points_mm=None,
    rgb_colors_rgb=None,
):
    """Display score fields and an optional registered-RGB candidate view."""
    if not cfg.show_3d:
        return
    show_dense_score_heatmaps(
        np.asarray(points_mm, dtype=np.float64),
        score_result,
        only_nms=getattr(cfg, "only", False),
        marker_radius_mm=max(0.8, 0.65 * cfg.voxel_3d_mm),
        rgb_points_mm=rgb_points_mm,
        rgb_colors_rgb=rgb_colors_rgb,
    )


def show_detector_validator_results(points_mm, validation, cfg):
    """Open the detector's final-only edge/corner result window."""
    render_detector_validator_results(
        np.asarray(points_mm, dtype=np.float64), validation,
        voxel_3d_mm=cfg.voxel_3d_mm,
    )


__all__ = [
    "prepare_guided_validator_view", "start_guided_validator_view", "pick_guided_validator_query",
    "pick_validator_fine_query",
    "prepare_guided_validator_result",
    "prepare_extremity_view", "extremity_summary_rgb", "export_extremity_views",
    "prepare_extremity_overview",
    "prepare_extremity_direction_inspector", "inspect_extremity_point", "set_extremity_inspector_scale",
    "set_extremity_inspector_mode",
    "close_query_views",
    "poll_view_events",
    "prepare_cornervins_plane_views",
    "prepare_plane_edge_graph_view",
    "show_dense_heatmaps",
    "show_detector_validator_results",
]
