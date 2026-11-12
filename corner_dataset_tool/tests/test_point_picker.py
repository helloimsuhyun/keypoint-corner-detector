"""GPU-independent tests for point selection. OpenD3D itself is optional."""
import numpy as np
import pytest

from point_picker import pick_projected_point


def cam():
    # OpenGL-style perspective projection, camera looks toward -Z.
    proj = np.array([[1.0, 0, 0, 0],
                     [0, 1.0, 0, 0],
                     [0, 0, -1.002002, -0.2002002],
                     [0, 0, -1.0, 0]], dtype=float)
    return np.eye(4), proj


def test_exact_point_at_center():
    V, P = cam()
    pts = np.array([[-0.4, 0., -1], [0, 0, -1.0], [0.4, 0, -1.0]])
    picked = pick_projected_point(pts, V, P, 320, 240, 640, 480)
    assert picked is not None and picked.index == 1
    assert picked.distance_px < 0.001


def test_multiple_selections_choose_nearest_pixel():
    V, P = cam()
    pts = np.array([[-.3, .0, -1.0], [.3, .0, -1.0]])
    assert pick_projected_point(pts, V, P, 224, 240, 640, 480).index == 0
    assert pick_projected_point(pts, V, P, 416, 240, 640, 480).index == 1


def test_pick_returns_none_for_empty_area():
    V, P = cam()
    pts = np.array([[0, 0, -1.0]])
    assert pick_projected_point(pts, V, P, 50, 50, 640, 480) is None


def test_rejects_points_behind_camera():
    V, P = cam()
    pts = np.array([[0, 0, 1.0]])
    assert pick_projected_point(pts, V, P, 320, 240, 640, 480) is None


def test_prefers_front_surface_when_same_screen_pixel():
    V, P = cam()
    pts = np.array([[0, 0, -2], [0, 0, -1]])
    assert pick_projected_point(pts, V, P, 320, 240, 640, 480).index == 1


def test_top_left_y_axis():
    V, P = cam()
    pts = np.array([[0, 0.25, -1], [0, -0.25, -1]])
    assert pick_projected_point(pts, V, P, 320, 180, 640, 480).index == 0
    assert pick_projected_point(pts, V, P, 320, 300, 640, 480).index == 1


def test_chunking_stable_and_tolerance():
    V, P = cam()
    pts = np.array([[0.0, 0.0, -2.0], [0.005, 0.0, -1.0], [0.08, 0, -1]])
    a = pick_projected_point(pts, V, P, 320, 240, 640, 480, chunk_size=1)
    b = pick_projected_point(pts, V, P, 320, 240, 640, 480, chunk_size=500)
    assert a.index == b.index == 1
    assert pick_projected_point(pts, V, P, 320, 240, 640, 480, tolerance_px=0.1).index == 0


def test_reject_invalid_inputs():
    V, P = cam()
    with pytest.raises(ValueError):
        pick_projected_point(np.zeros((2, 2)), V, P, 0, 0, 100, 100)
