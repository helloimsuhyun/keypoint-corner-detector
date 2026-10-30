import numpy as np

from normal_cone_depth_repeatability import (
    DetectionBundle,
    FrameData,
    camera_to_world_mm,
    range_contrast,
    virtual_view_subset,
    world_to_camera_mm,
)


def test_camera_world_transform_round_trip():
    angle = np.radians(23.0)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    pose[:3, 3] = (1.2, -0.3, 2.1)
    points = np.array([[10.0, 20.0, 1000.0], [-30.0, 40.0, 1500.0]])
    np.testing.assert_allclose(
        world_to_camera_mm(camera_to_world_mm(points, pose), pose),
        points,
        atol=1e-9,
    )


def test_virtual_view_subset_z_buffers_the_farther_point():
    points = np.array(
        [
            [0.0, 0.0, 1000.0],
            [0.0, 0.0, 1200.0],
            [100.0, 0.0, 1000.0],
        ]
    )
    subset = virtual_view_subset(
        points, yaw_degrees=0.0, camera_shift_x_mm=0.0
    )
    assert len(subset) == 2
    assert np.any(np.all(subset == points[0], axis=1))
    assert not np.any(np.all(subset == points[1], axis=1))


def test_range_contrast_increases_saliency_near_depth_change():
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    depth[237:244, 321:325] = 1800
    frame = FrameData(0, np.zeros((480, 640, 3), dtype=np.uint8), depth, np.eye(4))
    bundle = DetectionBundle(
        frame=0,
        variant="base",
        points_mm=np.array([[0.0, 0.0, 1000.0]]),
        scores=np.array([0.2]),
        candidate_indices=np.array([0]),
        scale_stable_indices=np.array([0]),
        scale_track_count=np.array([3]),
        scale_median_drift_mm=np.array([0.0]),
        scale_p90_drift_mm=np.array([0.0]),
        elapsed_seconds=0.0,
    )
    result = range_contrast(frame, bundle, pixel_radius=4, physical_scale_mm=15.0)
    assert result.spread_mm[0] > 700.0
    assert result.saliency[0] > bundle.scores[0]
