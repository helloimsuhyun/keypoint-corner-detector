from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from corner_detector.corner_2d import estimate_context_boundary
from corner_detector.plane_projection import build_tangent_context
from corner_detector.validator import (
    directional_projection_length,
    verify_corner_candidate,
)


def _config():
    return SimpleNamespace(
        support_radius_mm=20.0,
        surface_weight_min=0.0,
        normal_iterations=1,
        normal_radius_mm=3.0,
        sigma_e_deg=15.0,
        sigma_r_mm=1.0,
        normal_kernel="cauchy",
        normal_update_mode="synchronous",
        min_neighbors=3,
        voxel_2d_mm=1.0,
        boundary_radius_mm=3.0,
        boundary_epsilon_mm=0.5,
        query_radius_mm=8.0,
        tensor_radius_mm=4.0,
        tensor_sigma_d_mm=1.5,
        vote_line_sigma_mm=1.0,
        hough_min_score=0.0,
        corner_refine_radius_mm=2.0,
        corner_peak_ratio=0.7,
        corner_lambda_min_threshold=0.03,
        corner_balance_threshold=0.15,
        enable_iteration=False,
    )


def _normal_estimator(points, **_kwargs):
    points = np.asarray(points)
    normals = np.tile([0.0, 0.0, -1.0], (len(points), 1))
    indices = np.arange(len(points), dtype=np.int64)
    fits = [
        {"valid": True, "indices": indices.copy()}
        for _ in range(len(points))
    ]
    return normals, {"backend": "test"}, fits


def test_projection_and_2d_corner_stages_compose():
    xy = np.array(
        [(x, y) for y in range(-5, 6) for x in range(-5, 6)],
        dtype=np.float64,
    )
    points = np.column_stack((xy, np.full(len(xy), 100.0)))
    query_index = int(np.argmin(np.sum(xy * xy, axis=1)))

    context = build_tangent_context(
        points,
        query_index,
        _config(),
        _normal_estimator,
    )
    result = estimate_context_boundary(context)

    assert np.allclose(context["support_uv"][context["query_uv_index"]], 0.0)
    assert result["distance_mm"].shape == (len(context["support_uv"]),)
    assert result["corner_eigen"]["lambda_min"].shape == result["distance_mm"].shape
    assert result["hough_vote"]["response"].shape == result["distance_mm"].shape


def test_final_validator_is_independently_importable():
    config = _config()
    context = {"config": config, "query_uv_index": 0}
    result = {
        "corner_eigen": {
            "lambda_max": np.array([0.4]),
            "lambda_min": np.array([0.2]),
        }
    }
    verdict = verify_corner_candidate(context, result, status="single_pass")
    assert verdict["accepted"] is True
    assert verdict["reason"] == "accepted"


def test_projection_length_validator_returns_one_score_per_direction():
    projections = np.array([
        [-4.0, -2.0],
        [-1.0, 0.0],
        [2.0, 1.0],
        [5.0, 4.0],
    ])
    tail, score, span, lower, upper = directional_projection_length(
        projections,
        top_fraction=0.1,
        trim_fraction=0.0,
        min_span_mm=1.0,
    )
    assert tail.shape == score.shape == span.shape == lower.shape == upper.shape == (2,)
    assert np.all((0.0 <= score) & (score <= 1.0))


def test_main_pipeline_composes_stages_in_order(monkeypatch):
    import corner_detector.__main__ as pipeline

    calls = []
    context = {"config": object()}
    result = {"hough_vote": {"peaks": []}}

    def fake_projection(points, query_index, config, estimator):
        calls.append(("projection", query_index))
        return context

    def fake_corner_2d(received_context):
        assert received_context is context
        calls.append(("corner_2d", None))
        return result

    def fake_validator(received_context, received_result, points, **kwargs):
        assert received_context is context
        assert received_result is result
        calls.append(("validator", kwargs["point_tree"]))
        return [{"pass": True, "id": "winner"}]

    monkeypatch.setattr(pipeline, "build_tangent_context", fake_projection)
    monkeypatch.setattr(pipeline, "estimate_context_boundary", fake_corner_2d)
    monkeypatch.setattr(
        pipeline,
        "verify_hough_peaks_extremality_3d",
        fake_validator,
    )

    tree = object()
    actual_context, actual_result = pipeline.evaluate_query(
        np.zeros((1, 3)),
        0,
        object(),
        object(),
        point_tree=tree,
    )

    assert actual_context is context
    assert actual_result is result
    assert calls == [
        ("projection", 0),
        ("corner_2d", None),
        ("validator", tree),
    ]
    assert result["hough_verified_peak"]["id"] == "winner"


def test_stl_demo_uses_main_pipeline_functions():
    import stl_tangent_boundary_demo as demo
    from corner_detector.__main__ import (
        localize_corner_iterative,
        refine_corner_3d_subpix,
    )

    assert demo.localize_corner_iterative is localize_corner_iterative
    assert demo.refine_corner_3d_subpix is refine_corner_3d_subpix
