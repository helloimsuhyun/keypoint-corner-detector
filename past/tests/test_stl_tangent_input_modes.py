from __future__ import annotations

import numpy as np

import stl_tangent_boundary_demo as demo
from pointcloud_inputs import CameraIntrinsics, QuerySelection, frame_from_rgbd, query_from_pixel


def _frame():
    intrinsics = CameraIntrinsics(width=3, height=2, fx=2.0, fy=2.0, cx=1.0, cy=0.5)
    depth_m = np.ones((2, 3), dtype=np.float32)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    return frame_from_rgbd(depth_m, intrinsics, source="test", rgb=rgb)


def test_mode_arguments_are_resolved_into_config():
    args = demo.parse_args([
        "--input", "model.stl",
        "--mode", "fps",
        "--fps-count", "7",
        "--fps-min-spacing-mm", "3.5",
        "--corner-nms-radius-mm", "2.5",
        "--no-3d-view",
    ])
    config, _ = demo.configure(args)
    assert args.mode == "fps"
    assert config.fps_count == 7
    assert config.fps_min_spacing_mm == 3.5
    assert config.corner_nms_radius_mm == 2.5
    assert config.show_3d is False


def test_query_mode_passes_pointcloud_selection_to_diagnostic(monkeypatch):
    frame = _frame()
    selected = query_from_pixel(frame, 1, 1)
    assert selected is not None
    calls = {}

    monkeypatch.setattr(demo, "select_query_from_original", lambda *_a, **_k: selected)

    def fake_stage(frozen, config, estimator, **kwargs):
        calls.update(kwargs)
        return "quit"

    monkeypatch.setattr(demo, "run_frozen_click_stage", fake_stage)
    frozen = {"pointcloud_frame": frame, "selected_pixel": None}
    result = demo.run_query_mode(frame, frozen, demo.BoundaryConfig(), object())
    assert result == "quit"
    assert calls == {"initial_pixel": (1, 1), "allow_reselection": False}
    assert frozen["selected_pixel"] == (1, 1)


def test_fps_detection_runs_selected_indices_and_nms(monkeypatch):
    points = np.array([
        [0.0, 0.0, 100.0],
        [1.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
    ])
    queries = QuerySelection(
        indices=np.array([0, 1, 2]),
        points_mm=points.astype(np.float32),
        method="test-fps",
    )

    def fake_localize(scene, query_index, config, estimator):
        accepted = query_index in (0, 1)
        verification = {
            "accepted": accepted,
            "lambda_min": 0.2 if accepted else 0.0,
            "balance": 0.3 if accepted else 0.0,
            "score": 2.0 - query_index if accepted else 0.0,
        }
        context = {"query_world": scene[query_index].copy()}
        localization = {
            "verification": verification,
            "final_query_index": query_index,
        }
        return context, {}, localization

    monkeypatch.setattr(demo, "localize_corner_iterative", fake_localize)
    config = demo.BoundaryConfig(
        enable_refine=False,
        corner_nms_radius_mm=2.0,
    )
    result = demo.detect_fps_queries(points, queries, config, object())
    assert len(result["records"]) == 3
    assert len(result["accepted_before_nms"]) == 2
    assert len(result["corners"]) == 1
    assert result["corners"][0]["query_index"] == 0


def test_noise_disabled_preserves_frame_identity():
    frame = _frame()
    assert demo.apply_noise_to_frame(frame, demo.NoiseConfig()) is frame
