import numpy as np
from pathlib import Path

from synthetic_normal_cone_gt_benchmark import (
    make_synthetic_gt_scenes,
    match_corner_candidates,
)

from normal_cone_corner_detector import (
    GeometryScaleConfig,
    NormalConeConfig,
    MultiScaleConfig,
    PhaseCConfig,
    ScaleSpaceStabilityConfig,
    SceneDetectionResult,
    allowed_mask_jaccard,
    build_scene,
    detect_scene,
    detect_scene_geometry_adaptive,
    detect_scene_multiscale,
    evaluate_scale_space_stability,
    estimate_normal_cone,
    fibonacci_sphere,
    dimensionality_descriptor,
    jensen_shannon_distance,
    nearest_point_index,
    run_synthetic_validation,
    synthetic_patch,
    verify_candidates_phase_c,
    voxel_downsample,
)


def test_fibonacci_sphere_is_deterministic_and_unit_length():
    first = fibonacci_sphere(1024)
    second = fibonacci_sphere(1024)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(first, axis=0), 0.0, atol=2e-3)


def test_synthetic_gt_scene_corner_counts_are_explicit():
    scenes = make_synthetic_gt_scenes()
    assert [len(scene.corners_mm) for scene in scenes] == [
        4, 4, 6, 7, 4, 4, 0, 0, 0, 1, 1, 0
    ]
    assert all(scene.points_mm.shape[1] == 3 for scene in scenes)


def test_gt_matching_is_one_to_one_with_localization_error():
    ground_truth = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    candidates = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 2.0, 0.0]])
    metrics = match_corner_candidates(candidates, ground_truth, tolerance_mm=3.0)
    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 0
    np.testing.assert_allclose(np.sort(metrics.localization_errors_mm), [0.5, 2.0])


def test_voxel_downsample_uses_centroids():
    points = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [2.1, 0.0, 0.0]])
    actual = voxel_downsample(points, 2.0)
    expected = np.array([[0.4, 0.5, 0.6], [2.1, 0.0, 0.0]])
    np.testing.assert_allclose(actual, expected)


def test_default_synthetic_hypothesis_separates_corners_from_non_corners():
    result = run_synthetic_validation()
    assert result["separation_pass"]
    assert result["plane_corner"] > result["plane_edge"] + 0.10
    assert result["two_face_corner"] > result["two_face_edge"] + 0.05
    assert result["three_face_corner"] > result["plane"] + 0.05
    assert result["noisy_plane"] < 0.05


def test_query_result_matches_empirical_quantile_definition():
    config = NormalConeConfig(sphere_samples=256)
    points, query = synthetic_patch("plane_corner")
    scene = build_scene(points, config, downsample=False)
    result = estimate_normal_cone(scene, nearest_point_index(points, query), config)

    offsets = points[result.neighbor_indices] - result.query_point_mm
    projected = offsets @ scene.directions.T
    required = int(np.ceil(config.quantile * len(projected)))
    direct_quantile = np.partition(projected, required - 1, axis=0)[required - 1]
    np.testing.assert_allclose(result.direction_quantile_mm, direct_quantile)
    np.testing.assert_array_equal(
        result.allowed_mask, direct_quantile <= config.tolerance_mm
    )
    assert result.score == np.mean(result.allowed_mask)


def test_allowed_mask_jaccard_handles_overlap_and_empty_union():
    first = np.array([True, True, False, False])
    second = np.array([False, True, True, False])
    assert allowed_mask_jaccard(first, second) == 1.0 / 3.0
    assert allowed_mask_jaccard(np.zeros(4, bool), np.zeros(4, bool)) == 0.0


def test_dimensionality_descriptor_recovers_line_plane_and_volume():
    axis = np.linspace(-1.0, 1.0, 21)
    line = np.column_stack((axis, np.zeros_like(axis), np.zeros_like(axis)))
    x, y = np.meshgrid(axis, axis, indexing="ij")
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    np.testing.assert_allclose(dimensionality_descriptor(line), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(dimensionality_descriptor(plane), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(dimensionality_descriptor(volume), [0, 0, 1], atol=1e-12)
    assert jensen_shannon_distance([0, 1, 0], [0, 1, 0]) == 0.0
    assert jensen_shannon_distance([0, 1, 0], [0, 0, 1]) > 0.99


def test_geometry_adaptive_stops_before_nearby_perpendicular_surface():
    config = NormalConeConfig(
        query_radius_mm=32.0,
        sphere_samples=128,
        tolerance_mm=0.6,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
    )
    axis = np.arange(0.0, 60.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    plate = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    signed = np.arange(-30.0, 30.1, 2.0)
    wall_y, wall_z = np.meshgrid(signed, signed, indexing="ij")
    wall = np.column_stack(
        (np.full(wall_y.size, 18.0), wall_y.ravel(), wall_z.ravel())
    )
    radii = (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
    selected = {}
    for name, points in (
        ("stable", plate),
        ("contaminated", np.unique(np.vstack((plate, wall)), axis=0)),
    ):
        scene = build_scene(points, config, downsample=False)
        query = nearest_point_index(scene.points_mm, [0.0, 0.0, 0.0])
        multiscale = detect_scene_multiscale(
            scene,
            config,
            MultiScaleConfig(radii_mm=radii, min_scale_count=2, min_consistency=0.0),
            query_indices=[query],
        )
        result = detect_scene_geometry_adaptive(
            scene,
            config,
            multiscale,
            GeometryScaleConfig(min_clean_scale_count=3, min_shell_points=6),
            query_indices=[query],
        )
        selected[name] = result.selected_radius_mm[query]
        assert result.valid_mask[query]

    assert selected["stable"] == 32.0
    assert selected["contaminated"] == 16.0


def test_scene_detector_nms_localizes_planar_patch_corners():
    config = NormalConeConfig(
        query_radius_mm=12.0,
        sphere_samples=256,
        tolerance_mm=0.4,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
        direction_chunk_size=128,
    )
    axis = np.arange(0.0, 40.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    scene = build_scene(points, config, downsample=False)
    result = detect_scene(scene, config)

    candidates = scene.points_mm[result.candidate_indices]
    expected = np.array(
        [[0, 0, 0], [0, 40, 0], [40, 0, 0], [40, 40, 0]], dtype=float
    )
    for corner in expected:
        assert np.min(np.linalg.norm(candidates - corner, axis=1)) <= config.nms_radius_mm

    side_midpoints = np.array(
        [[0, 20, 0], [20, 0, 0], [40, 20, 0], [20, 40, 0]], dtype=float
    )
    for midpoint in side_midpoints:
        assert np.min(np.linalg.norm(candidates - midpoint, axis=1)) > config.nms_radius_mm


def test_multiscale_detector_localizes_planar_corners_at_consistent_scales():
    config = NormalConeConfig(
        query_radius_mm=18.0,
        sphere_samples=128,
        tolerance_mm=0.4,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
        direction_chunk_size=128,
    )
    axis = np.arange(0.0, 40.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    scene = build_scene(points, config, downsample=False)
    result = detect_scene_multiscale(
        scene,
        config,
        MultiScaleConfig(
            radii_mm=(8.0, 12.0, 18.0),
            min_scale_count=2,
            min_consistency=0.30,
        ),
    )

    assert result.scale_scores.shape == (3, len(points))
    candidates = scene.points_mm[result.candidate_indices]
    expected = np.array(
        [[0, 0, 0], [0, 40, 0], [40, 0, 0], [40, 40, 0]], dtype=float
    )
    for corner in expected:
        assert np.min(np.linalg.norm(candidates - corner, axis=1)) <= config.nms_radius_mm
    corner_index = nearest_point_index(points, expected[0])
    assert result.scale_pass_count[corner_index] >= 2
    assert result.scale_consistency[corner_index] >= 0.30


def test_scale_space_stability_tracks_planar_corner_maxima():
    config = NormalConeConfig(
        query_radius_mm=18.0,
        sphere_samples=128,
        tolerance_mm=0.4,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
        direction_chunk_size=128,
    )
    axis = np.arange(0.0, 40.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    scene = build_scene(points, config, downsample=False)
    multiscale = detect_scene_multiscale(
        scene,
        config,
        MultiScaleConfig(
            radii_mm=(8.0, 12.0, 18.0),
            min_scale_count=2,
            min_consistency=0.30,
        ),
    )
    stability = evaluate_scale_space_stability(
        scene,
        config,
        multiscale,
        ScaleSpaceStabilityConfig(
            track_radius_mm=5.0,
            min_track_scale_count=2,
            max_median_drift_mm=4.0,
            max_p90_drift_mm=5.0,
        ),
    )

    accepted = scene.points_mm[stability.accepted_indices]
    expected = np.array(
        [[0, 0, 0], [0, 40, 0], [40, 0, 0], [40, 40, 0]], dtype=float
    )
    for corner in expected:
        assert np.min(np.linalg.norm(accepted - corner, axis=1)) <= config.nms_radius_mm
    assert stability.scale_local_max_mask.shape == (3, len(points))
    assert np.all(stability.track_count[stability.accepted_indices] >= 2)


def test_phase_c_preserves_supported_repeatable_planar_corners():
    config = NormalConeConfig(
        query_radius_mm=12.0,
        sphere_samples=256,
        tolerance_mm=0.4,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
        direction_chunk_size=128,
    )
    axis = np.arange(0.0, 40.1, 2.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    scene = build_scene(points, config, downsample=False)
    detection = detect_scene(scene, config)
    verification = verify_candidates_phase_c(
        scene,
        detection,
        config,
        PhaseCConfig(
            min_support=25,
            min_outer_shell_fraction=0.10,
            min_radius_p90_ratio=0.70,
            repeatability_trials=3,
            min_repeatability=0.45,
            min_split_score_ratio=0.35,
        ),
        query_radius_mm_by_index=np.full(len(points), 12.0),
    )

    accepted = scene.points_mm[verification.accepted_indices]
    expected = np.array(
        [[0, 0, 0], [0, 40, 0], [40, 0, 0], [40, 40, 0]], dtype=float
    )
    for corner in expected:
        assert np.min(np.linalg.norm(accepted - corner, axis=1)) <= config.nms_radius_mm


def test_phase_c_support_stage_rejects_under_supported_candidates():
    config = NormalConeConfig(
        query_radius_mm=12.0,
        sphere_samples=128,
        tolerance_mm=0.4,
        min_score=0.10,
        local_max_radius_mm=2.1,
        nms_radius_mm=5.0,
    )
    points, _ = synthetic_patch("plane_corner", extent_mm=20.0, spacing_mm=2.0)
    scene = build_scene(points, config, downsample=False)
    detection = detect_scene(scene, config)
    verification = verify_candidates_phase_c(
        scene,
        detection,
        config,
        PhaseCConfig(min_support=len(points) + 1),
    )
    assert len(detection.candidate_indices) > 0
    assert len(verification.support_indices) == 0
    assert len(verification.accepted_indices) == 0


def test_phase_c_spatial_asymmetry_separates_corner_from_straight_edge():
    config = NormalConeConfig(query_radius_mm=30.0, sphere_samples=256)
    spatial_pass = {}
    for kind in ("plane_edge", "plane_corner"):
        points, query = synthetic_patch(kind)
        scene = build_scene(points, config, downsample=False)
        index = nearest_point_index(points, query)
        cone = estimate_normal_cone(scene, index, config)
        detection = SceneDetectionResult(
            scores=np.eye(1, len(points), index, dtype=float).ravel() * cone.score,
            neighbor_count=np.zeros(len(points), dtype=np.int32),
            valid_mask=np.ones(len(points), dtype=bool),
            local_max_mask=np.zeros(len(points), dtype=bool),
            local_max_indices=np.array([index]),
            candidate_indices=np.array([index]),
            elapsed_seconds=0.0,
        )
        result = verify_candidates_phase_c(
            scene,
            detection,
            config,
            PhaseCConfig(min_support=20),
        )
        spatial_pass[kind] = bool(result.spatial_pass[0])

    assert not spatial_pass["plane_edge"]
    assert spatial_pass["plane_corner"]


def test_stl_virtual_depth_capture_finds_visible_outer_box_corners():
    """Integration test the same STL -> depth -> voxel -> Phase B route as the UI."""
    import normal_cone_corner_detector as detector

    o3d = detector._load_open3d()
    path = Path(__file__).resolve().parent / "data" / "laser_pose_box.stl"
    mesh = o3d.io.read_triangle_mesh(str(path))
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = ray_scene.create_rays_pinhole(
        55.0,
        o3d.core.Tensor([0.0, 0.0, 15.0]),
        o3d.core.Tensor([105.0, -105.0, 95.0]),
        o3d.core.Tensor([0.0, 0.0, 1.0]),
        160,
        120,
    )
    hit = ray_scene.cast_rays(rays)["t_hit"].numpy()
    ray_array = rays.numpy()
    valid = np.isfinite(hit)
    points = (
        ray_array[..., :3][valid]
        + hit[valid, None] * ray_array[..., 3:][valid]
    )

    config = NormalConeConfig(sphere_samples=256)
    scene = build_scene(points, config)
    result = detect_scene(scene, config)
    candidate_points = scene.points_mm[result.candidate_indices]
    outer_corners = np.array(
        [[x, y, z] for x in (-30.0, 30.0) for y in (-30.0, 30.0) for z in (0.0, 30.0)]
    )
    candidate_corner_distance = np.min(
        np.linalg.norm(candidate_points[:, None, :] - outer_corners[None, :, :], axis=2),
        axis=1,
    )

    assert np.max(result.scores[result.candidate_indices]) > 0.25
    assert np.count_nonzero(candidate_corner_distance <= 3.5) >= 6
