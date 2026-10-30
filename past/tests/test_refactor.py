"""Targeted refactor checks; no camera/GUI needed."""
import numpy as np
from corner_detector.settings import parse_args, configure
from corner_detector.preprocessing import corrupt_depth, NoiseConfig
from corner_detector.fps import _nms_query_detections


def test_defaults():
    args = parse_args(["--input", "fake.stl"])
    cfg, noise = configure(args)
    assert args.source == "stl"
    assert cfg.query_radius_mm == 40.0
    assert cfg.voxel_3d_mm == 2.0
    assert cfg.enable_iteration is False
    assert cfg.enable_refine is True


def test_depth_no_noise():
    data = np.array([[700.0, np.nan]], dtype=np.float32)
    result = corrupt_depth(data, NoiseConfig(seed=0))
    np.testing.assert_allclose(result[0, 0], 700)
    assert np.isnan(result[0, 1])


def test_nms_order():
    candidates = [
        {"position": [0, 0, 0], "score": 0.2},
        {"position": [1, 0, 0], "score": 0.7},
        {"position": [10, 0, 0], "score": 0.5},
    ]
    result = _nms_query_detections(candidates, 3.0)
    assert [r["score"] for r in result] == [0.7, 0.5]


def test_numerical_pipeline_with_mock_normal_estimator():
    """Real pipeline stages execute without a sensor using a controlled normal backend."""
    from corner_detector.pipeline import localize_corner_iterative

    cfg, _ = configure(parse_args([
        "--input", "synthetic.stl", "--no-iteration", "--no-refine",
        "--ext-directions", "16",
    ]))
    xx, yy = np.meshgrid(np.arange(0, 41, 4), np.arange(0, 41, 4))
    points = np.column_stack([
        xx.ravel(), yy.ravel(), np.full(xx.size, 500.0)
    ]).astype(np.float64)

    def mock_sah(samples, **kwargs):
        normals = np.tile(np.array([0.0, 0.0, -1.0]), (len(samples), 1))
        fit_weights = [
            {"valid": True, "indices": np.array([i], dtype=np.int64)}
            for i in range(len(samples))
        ]
        return normals, {}, fit_weights

    context, result, localization = localize_corner_iterative(
        points, 0, cfg, mock_sah
    )
    assert localization["status"] == "single_pass"
    assert "corner_eigen" in result
    assert len(context["support_uv"]) > 0
    assert localization["initial_query_index"] == 0
    assert localization["final_query_index"] == 0
    assert "verification" in localization
