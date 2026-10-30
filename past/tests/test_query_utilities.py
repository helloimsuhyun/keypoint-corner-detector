from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_inputs import (
    CameraIntrinsics,
    QuerySelection,
    draw_queries_on_original,
    farthest_point_sampling,
    frame_from_rgbd,
    queries_from_indices,
    query_from_pixel,
    select_fps_queries,
)


def make_frame():
    intrinsics = CameraIntrinsics(3, 2, fx=2.0, fy=2.0, cx=1.0, cy=0.5)
    depth = np.array([[np.nan, 1.0, 2.0], [0.0, 1.5, 3.0]], dtype=np.float32)
    rgb = np.full((2, 3, 3), 80, dtype=np.uint8)
    return frame_from_rgbd(
        depth,
        intrinsics,
        source="test",
        rgb=rgb,
        min_depth_m=0.1,
        max_depth_m=5.0,
    )


def test_query_from_pixel_preserves_original_index_and_pixel() -> None:
    frame = make_frame()
    query = query_from_pixel(frame, 2, 1)
    assert isinstance(query, QuerySelection)
    assert query.index == 3
    np.testing.assert_array_equal(query.pixels_uv, [[2, 1]])
    np.testing.assert_allclose(query.point_mm, frame.points_mm[3])


def test_query_from_invalid_pixel_can_select_nearest_valid_pixel() -> None:
    frame = make_frame()
    assert query_from_pixel(frame, 0, 0, max_radius_px=0) is None
    query = query_from_pixel(frame, 0, 0, max_radius_px=1)
    assert query is not None
    assert query.index == 0
    np.testing.assert_array_equal(query.pixels_uv, [[1, 0]])


def test_fps_is_deterministic_and_uses_original_indices() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    indices = farthest_point_sampling(points, 3)
    np.testing.assert_array_equal(indices, [2, 3, 0])
    selection = select_fps_queries(points, 3)
    np.testing.assert_array_equal(selection.indices, indices)
    np.testing.assert_array_equal(selection.points_mm, points[indices])


def test_fps_min_spacing_stops_early() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    indices = farthest_point_sampling(points, 4, min_spacing_mm=5.0)
    np.testing.assert_array_equal(indices, [2, 3])


def test_fps_candidate_subset_keeps_global_indices() -> None:
    points = np.column_stack((np.arange(6.0), np.zeros((6, 2)))).astype(np.float32)
    indices = farthest_point_sampling(
        points,
        3,
        candidate_indices=[1, 3, 5],
        start_index=3,
    )
    assert set(indices.tolist()).issubset({1, 3, 5})
    assert indices[0] == 3


def test_fps_frame_selection_contains_source_pixels() -> None:
    frame = make_frame()
    selection = select_fps_queries(frame, 3)
    assert selection.pixels_uv.shape == (3, 2)
    for index, (u, v) in zip(selection.indices, selection.pixels_uv):
        assert frame.pixel_to_point_index[v, u] == index


def test_original_overlay_does_not_modify_frame_rgb() -> None:
    frame = make_frame()
    original = frame.rgb.copy()
    selection = queries_from_indices(frame, [0, 3], method="test")
    overlay = draw_queries_on_original(frame, selection, marker_radius_px=1)
    assert overlay.shape == frame.rgb.shape
    assert not np.array_equal(overlay, original)
    np.testing.assert_array_equal(frame.rgb, original)
