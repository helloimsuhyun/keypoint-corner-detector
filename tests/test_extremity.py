"""Numerical contracts for local vertex hypotheses and unchanged CED queries."""

import contextlib
import io
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from corner_detector.extremity import (
    ExtremityConfig, fibonacci_directions, prepare_extremity_context,
    run_extremity_refinement, refine_ced_candidates, _local_peaks,
)
from corner_detector.extremity_synthetic import make_scene, match_corners, evaluate, ced_candidates
from corner_detector.settings import DEFAULTS, parse_args, configure, DetectorConfig
from corner_detector.visualization import manager
from corner_detector.visualization.view3d import extremity_primitives


def reference_config(**overrides):
    return ExtremityConfig(**{**dict(radii_mm=(30.0,), padding_mm=10.0,
        cluster_radius_mm=4.5, nms_radius_mm=5.0, voting_mode="hard",
        min_vote_fraction=0.04, nms_method="local_maximum"), **overrides})


class ExtremityTest(unittest.TestCase):
    def test_sphere_is_unit_deterministic_and_immutable(self):
        directions = fibonacci_directions()
        self.assertEqual(directions.shape, (256, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-15)
        self.assertIs(directions, fibonacci_directions())
        self.assertFalse(directions.flags.writeable)

    def test_vectorized_votes_equal_direction_by_direction_argmax(self):
        points = np.random.default_rng(7).normal(0, 15, (150, 3))
        cfg = reference_config(object_clustering=False)
        result = run_extremity_refinement(points, 3, cfg)
        expected = np.zeros(len(points), dtype=int)
        roi_ids = result["roi_raw_indices"]
        for direction in fibonacci_directions():
            winner = roi_ids[np.argmax((points[roi_ids] - points[3]) @ direction)]
            if np.linalg.norm(points[winner] - points[3]) <= 30:
                expected[winner] += 1
        np.testing.assert_array_equal(result["raw_votes"], expected[roi_ids])
        np.testing.assert_allclose(result["normalized_score"], expected[roi_ids] / 256)
        np.testing.assert_array_equal(result["query_position_mm"], points[3])

    def test_padding_discards_artificial_roi_cut_extrema(self):
        x, y = np.meshgrid(np.arange(-60, 61, 3), np.arange(-60, 61, 3))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        q = int(np.argmin(np.linalg.norm(points, axis=1)))
        cfg = reference_config(object_clustering=False)
        unpadded = run_extremity_refinement(points, q, replace(cfg, padding_mm=0))
        padded = run_extremity_refinement(points, q, cfg)
        self.assertGreater(unpadded["candidate_count"], 0)
        self.assertEqual(padded["candidate_count"], 0)
        self.assertEqual(padded["discarded_vote_count"], 256)

    def test_one_two_three_faces_share_the_same_detector(self):
        for name in ("one_face", "two_faces", "three_faces"):
            scene = make_scene(name, noise_mm=0)
            q = int(cKDTree(scene.points_mm).query([6, 6, 0])[1])
            result = run_extremity_refinement(scene.points_mm, q, reference_config())
            positions = np.array([c["position_mm"] for c in result["candidates"]])
            self.assertTrue(np.any(np.linalg.norm(positions, axis=1) < 1e-9), name)
            self.assertEqual(result["cluster_count"], 1, name)
            for c in result["candidates"]:
                self.assertLessEqual(c["query_distance_mm"], 30)

    def test_single_query_retains_two_separate_vertices(self):
        scene = make_scene("same_object_two_corners", noise_mm=0)
        q = int(cKDTree(scene.points_mm).query([0, 3, 0])[1])
        result = run_extremity_refinement(scene.points_mm, q, reference_config())
        positions = scene.points_mm[result["candidate_raw_indices"]]
        for target in ([-9, 0, 0], [9, 0, 0]):
            self.assertLess(np.linalg.norm(positions - target, axis=1).min(), 1e-9)

    def test_all_components_vote_not_only_the_query_component(self):
        scene = make_scene("hidden_small_object", noise_mm=0)
        q = int(cKDTree(scene.points_mm).query([-3, -3, -3])[1])
        off = run_extremity_refinement(scene.points_mm, q, reference_config(object_clustering=False))
        on = run_extremity_refinement(scene.points_mm, q, reference_config())
        small = np.flatnonzero(scene.object_labels == 1)
        self.assertEqual(len(np.intersect1d(off["candidate_raw_indices"], small)), 0)
        self.assertGreater(len(np.intersect1d(on["candidate_raw_indices"], small)), 0)
        self.assertGreater(on["direction_sweep_count"], 1)

    def test_touching_objects_merge_and_sparse_objects_fragment(self):
        touching = make_scene("touching_objects", noise_mm=0)
        q = int(cKDTree(touching.points_mm).query(touching.focus_mm)[1])
        merged = run_extremity_refinement(touching.points_mm, q, reference_config())
        oracle = run_extremity_refinement(touching.points_mm, q, reference_config(), oracle_labels=touching.object_labels)
        self.assertEqual(merged["cluster_count"], 1)
        self.assertEqual(oracle["cluster_count"], 2)
        sparse = make_scene("one_face", spacing_mm=6, noise_mm=0)
        fragmented = run_extremity_refinement(sparse.points_mm, 0, reference_config())
        self.assertGreater(fragmented["cluster_count"], 10)

    def test_concave_notch_has_no_stable_extremity_votes(self):
        scene = make_scene("concave_notch", noise_mm=0)
        q = int(cKDTree(scene.points_mm).query([-3, -3, 0])[1])
        result = run_extremity_refinement(scene.points_mm, q, reference_config())
        zero = np.flatnonzero(np.all(result["roi_points_mm"] == [0, 0, 0], axis=1))
        self.assertEqual(result["raw_votes"][zero[0]], 0)

    def test_multiscale_uses_one_query_and_tracks_scale_votes(self):
        scene = make_scene("hidden_small_object", noise_mm=0)
        cfg = reference_config(radii_mm=(12, 30), object_clustering=False)
        q = int(cKDTree(scene.points_mm).query([-3, -3, -3])[1])
        result = run_extremity_refinement(scene.points_mm, q, cfg)
        np.testing.assert_array_equal(result["raw_votes"], result["votes_per_scale"].sum(axis=0))
        np.testing.assert_array_equal(result["query_position_mm"], scene.points_mm[q])
        self.assertTrue(np.all(result["scale_support_count"] <= 2))
        # Padding still reaches the surrounding shell at the small scale.
        self.assertEqual(result["candidate_count"], 0)

    def test_plateau_and_greedy_chain_rules(self):
        points = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=float)
        votes = np.array([30, 30, 30])
        labels = np.zeros(3, dtype=int)
        cfg = reference_config()
        np.testing.assert_array_equal(_local_peaks(points, votes / 256, votes, labels, cfg), [0])
        np.testing.assert_array_equal(_local_peaks(points, votes / 256, votes, labels,
                                                   replace(cfg, nms_method="greedy")), [0, 2])
        labels[1] = 1
        np.testing.assert_array_equal(_local_peaks(points, votes / 256, votes, labels, cfg), [0, 1, 2])

    def test_raw_and_scene_indices_and_query_offsets_are_explicit(self):
        raw = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]], dtype=float)
        scene = np.array([[1.5, 1.5, 0]], dtype=float)
        result = run_extremity_refinement(scene, 0, raw_points_mm=raw)
        self.assertGreater(result["candidate_count"], 0)
        for c in result["candidates"]:
            self.assertEqual(c["point_index"], 0)
            np.testing.assert_array_equal(c["position_mm"], raw[c["raw_point_index"]])
            np.testing.assert_allclose(c["query_offset_mm"], c["position_mm"] - scene[0])
        np.testing.assert_array_equal(result["validator_seed_indices"], [0])

    def test_batch_preserves_supplied_query_indices_and_keeps_independent_maps(self):
        points = np.array([[0, 0, 0], [9, 0, 0], [0, 9, 0], [9, 9, 0]], dtype=float)
        original = points.copy()
        batch = refine_ced_candidates(points, np.array([2, 0]))
        np.testing.assert_array_equal(batch["query_indices"], [2, 0])
        self.assertEqual([r["query_index"] for r in batch["records"]], [2, 0])
        self.assertFalse(np.shares_memory(batch["records"][0]["raw_votes"], batch["records"][1]["raw_votes"]))
        np.testing.assert_array_equal(points, original)
        self.assertEqual(refine_ced_candidates(points, np.array([], dtype=int))["records"], [])

    def test_empty_roi_and_invalid_inputs(self):
        result = run_extremity_refinement(np.zeros((1, 3)), 0, raw_points_mm=np.empty((0, 3)))
        self.assertEqual(result["candidate_count"], 0)
        for cfg in (ExtremityConfig(padding_mm=-1), ExtremityConfig(radii_mm=(float("nan"),)),
                    ExtremityConfig(directions=0), ExtremityConfig(cluster_radius_mm=0)):
            with self.assertRaises(ValueError):
                cfg.validate()
        with self.assertRaises(IndexError):
            run_extremity_refinement(np.zeros((1, 3)), 1)
        with self.assertRaises(ValueError):
            run_extremity_refinement(np.array([[float("nan"), 0, 0]]), 0)
        context = prepare_extremity_context(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            run_extremity_refinement(np.ones((1, 3)), 0, context=context)

    def test_matching_is_one_to_one_and_maximizes_valid_matches(self):
        self.assertEqual(len(match_corners([[0, 0, 0]], [[0, 0, 0], [1, 0, 0]])[0]), 1)
        # Minimum raw-distance assignment can lose an admissible pair; gate costs first.
        matches, _ = match_corners([[0, 0, 0], [4, 0, 0]], [[0, 0, 0], [-4, 0, 0]])
        self.assertEqual(len(matches), 2)
        self.assertEqual(evaluate([[0, 0, 0], [1, 0, 0]], [[0, 0, 0]])["false_positives"], 1)

    def test_settings_preserve_ced_defaults_and_have_no_new_query_sampler(self):
        old, _ = configure(parse_args(["--source", "realsense", "--mode", "detector"]))
        new, _ = configure(parse_args(["--source", "realsense", "--mode", "extremity256"]))
        for key in vars(old):
            self.assertEqual(getattr(old, key), getattr(new, key), key)
        ball, _ = configure(parse_args(["--source", "realsense", "--mode", "ball"]))
        self.assertEqual(ball.voxel_3d_mm, old.voxel_3d_mm)
        self.assertEqual(ball.extremity.radii_mm, tuple(x * ball.voxel_3d_mm for x in DEFAULTS.ball_radii_x))
        for key in vars(old):
            self.assertEqual(getattr(old, key), getattr(ball, key), key)

    def test_ball_defaults_are_shared_by_cli_and_library(self):
        from corner_detector.settings import ExtremityConfig as SettingsExtremityConfig
        self.assertIs(ExtremityConfig, SettingsExtremityConfig)
        expected = {"radii_mm": tuple(x * DEFAULTS.voxel_3d_mm for x in DEFAULTS.ball_radii_x),
                    "padding_mm": DEFAULTS.ball_padding_x * DEFAULTS.voxel_3d_mm,
                    "cluster_radius_mm": DEFAULTS.ball_cluster_radius_x * DEFAULTS.voxel_3d_mm,
                    "nms_radius_mm": DEFAULTS.ball_nms_radius_x * DEFAULTS.voxel_3d_mm,
                    "soft_tau_mm": DEFAULTS.ball_soft_tau_x * DEFAULTS.voxel_3d_mm}
        for name in ("directions", "object_clustering", "nms_method", "min_vote_fraction",
                     "voting_mode", "soft_alpha", "voting_dtype"):
            expected[name] = getattr(DEFAULTS, "ball_" + name)
        expected_defaults = {k: v for k, v in vars(DEFAULTS).items() if k.startswith("ball_")}
        self.assertEqual(asdict(ExtremityConfig()), expected)
        for name, value in expected_defaults.items():
            self.assertEqual(getattr(DEFAULTS, name), value)
        for mode in ("ball", "extremity256"):
            args = parse_args(["--source", "realsense", "--mode", mode])
            cfg, _ = configure(args)
            self.assertEqual(asdict(cfg.extremity), expected)
            for name, value in expected_defaults.items():
                self.assertEqual(getattr(args, name), value)

    def test_ball_settings_defaults_feed_parser_and_cli_overrides(self):
        custom = replace(DEFAULTS, ball_radii_x=(1.2, 2.4), ball_padding_x=0.6,
                         ball_directions=128, ball_object_clustering=False,
                         ball_cluster_radius_x=0.65, ball_nms_radius_x=0.4,
                         ball_nms_method="greedy", ball_min_vote_fraction=0.125)
        with patch("corner_detector.settings.DEFAULTS", custom):
            args = parse_args(["--source", "realsense", "--mode", "ball"])
            cfg, _ = configure(args)
            self.assertEqual(cfg.extremity, ExtremityConfig(
                radii_mm=(12.0, 24.0), padding_mm=6.0, directions=128,
                object_clustering=False, cluster_radius_mm=6.5, nms_radius_mm=4.0,
                nms_method="greedy", min_vote_fraction=0.125))
            args = parse_args(["--source", "realsense", "--mode", "ball",
                               "--ball-radii-x", "1.5", "3", "--ball-object-clustering",
                               "--ball-padding-x", "0"])
            cfg, _ = configure(args)
        self.assertEqual(cfg.extremity.radii_mm, (15.0, 30.0))
        self.assertTrue(cfg.extremity.object_clustering)
        self.assertEqual(cfg.extremity.padding_mm, 0)
        self.assertEqual(cfg.extremity.directions, 128)

    def test_ball_lengths_scale_with_3d_voxel(self):
        for voxel, radius, padding, cluster, nms in ((3, 9.0, 3.0, 1.35, 1.5),
                                                   (10, 30.0, 10.0, 4.5, 5.0),
                                                   (20, 60.0, 20.0, 9.0, 10.0)):
            cfg, _ = configure(parse_args(["--source", "realsense", "--mode", "ball",
                                          "--voxel-3d-mm", str(voxel), "--ball-radii-x", "3",
                                          "--ball-padding-x", "1", "--ball-cluster-radius-x", "0.45",
                                          "--ball-nms-radius-x", "0.5"]))
            self.assertEqual(cfg.extremity.radii_mm, (radius,))
            self.assertEqual(cfg.extremity.padding_mm, padding)
            self.assertAlmostEqual(cfg.extremity.cluster_radius_mm, cluster)
            self.assertEqual(cfg.extremity.nms_radius_mm, nms)

    def test_ball_multiscale_and_metric_overrides_resolve_from_voxel(self):
        cfg, _ = configure(parse_args(["--source", "realsense", "--mode", "ball",
            "--voxel-3d-mm", "4", "--ball-radii-x", "1.2", "3",
            "--ball-padding-x", "0.5", "--ball-cluster-radius-x", "2",
            "--ball-nms-radius-x", "0.25"]))
        self.assertEqual(cfg.extremity.radii_mm, (4.8, 12.0))
        self.assertEqual(cfg.extremity.padding_mm, 2.0)
        self.assertEqual(cfg.extremity.cluster_radius_mm, 8.0)
        self.assertEqual(cfg.extremity.nms_radius_mm, 1.0)

    def test_ball_multiplier_validation_and_old_mm_flags_are_not_reinterpreted(self):
        for options in (["--ball-radii-x", "0"], ["--ball-radii-x", "nan"],
                        ["--ball-padding-x", "-1"], ["--ball-cluster-radius-x", "0"],
                        ["--ball-nms-radius-x", "0"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                configure(parse_args(["--source", "realsense", "--mode", "ball", *options]))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--source", "realsense", "--mode", "ball", "--ball-radii-mm", "30"])

    def test_ball_click_selects_existing_ced_query_not_the_clicked_raw_index(self):
        from corner_detector import ball_query
        from pointcloud_inputs import CameraIntrinsics, frame_from_rgbd
        intr = CameraIntrinsics(2, 2, 100, 100, 0, 0)
        frame = frame_from_rgbd(np.full((2, 2), 0.1), intr, source="synthetic")
        scene = frame.points_mm.astype(float)
        # Deliberately keep only a query far from pixel (0,0); no new query allowed.
        result = run_extremity_refinement(scene, 3)
        frozen = {"pointcloud_frame": frame, "points_voxel_mm": scene,
                  "points_raw_mm": frame.points_mm,
                  "extremity_batch": {"query_indices": np.array([3]), "records": [result]}}
        cfg, _ = configure(parse_args(["--mode", "ball", "--source", "realsense", "--no-3d-view"]))
        with patch.object(ball_query, "prepare_extremity_view") as show, contextlib.redirect_stdout(io.StringIO()):
            selected = ball_query.inspect_ball_query(0, 0, frozen, cfg)
        self.assertIs(selected, result)
        self.assertEqual(selected["query_index"], 3)
        np.testing.assert_array_equal(selected["query_position_mm"], scene[3])
        show.assert_called_once_with(frozen, result, show_3d=False)

    def test_ball_does_not_invent_queries_when_ced_returns_none(self):
        from corner_detector import ball_query
        from pointcloud_inputs import CameraIntrinsics, frame_from_rgbd
        frame = frame_from_rgbd(np.full((1, 1), 0.1), CameraIntrinsics(1, 1, 100, 100, 0, 0), source="synthetic")
        frozen = {"pointcloud_frame": frame, "points_voxel_mm": frame.points_mm,
                  "extremity_batch": {"query_indices": np.array([], dtype=int), "records": []}}
        with patch.object(ball_query, "prepare_extremity_view") as show, contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(ball_query.inspect_ball_query(0, 0, frozen, object()))
        show.assert_not_called()

    def test_clustering_is_once_per_query_and_view_cleanup_preserves_batch(self):
        import corner_detector.extremity as extremity
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=float)
        with patch.object(extremity, "_components", wraps=extremity._components) as cluster:
            result = run_extremity_refinement(points, 0, ExtremityConfig(radii_mm=(12, 30)))
        self.assertEqual(cluster.call_count, 1)
        batch = {"records": [result]}
        view = unittest.mock.Mock()
        frozen = {"extremity_batch": batch, "extremity_result": result, "extremity_views": (view,)}
        manager.close_query_views(frozen)
        view.destroy_window.assert_called_once()
        self.assertIs(frozen["extremity_batch"], batch)

    def test_ced_adapter_uses_original_pipeline_indices_verbatim(self):
        from corner_detector.ball_query import run_ced_extremity_mode
        from corner_detector.settings import ExtremityDetectorConfig
        scene = make_scene("one_face", spacing_mm=12, noise_mm=0)
        cfg = ExtremityDetectorConfig(show_3d=False)
        extraction, _ = ced_candidates(scene.points_mm, cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            batch = run_ced_extremity_mode(scene.points_mm, cfg)
        np.testing.assert_array_equal(batch["query_indices"], extraction["candidate_indices"])

    def test_ced_adapter_shares_settings_and_caches_existing_heatmap_without_recomputation(self):
        from corner_detector import ball_query
        cfg, _ = configure(parse_args(["--source", "realsense", "--mode", "ball",
                                       "--nms-radius-mm", "17", "--ball-nms-radius-x", "0.7"]))
        points = np.array([[0., 0., 0.], [3., 0., 0.]])
        indices = np.array([1])
        scores = {"candidate_extraction": {"candidate_indices": indices}}
        cached = {"candidate_count": 0, "elapsed_s": 0.0}
        with patch.object(ball_query, "run_dense_heatmap_mode", return_value=scores) as dense, \
                patch.object(ball_query, "refine_ced_candidates", return_value=cached) as refine, \
                contextlib.redirect_stdout(io.StringIO()):
            result = ball_query.run_ced_extremity_mode(points, cfg, raw_points_mm=points)
        dense.assert_called_once()
        upstream_cfg = dense.call_args.args[1]
        self.assertEqual(replace(upstream_cfg, show_3d=cfg.show_3d, validator=cfg.validator), cfg)
        self.assertFalse(upstream_cfg.show_3d)
        self.assertIsNone(upstream_cfg.validator)
        refine.assert_called_once_with(points, indices, cfg.extremity, raw_points_mm=points)
        self.assertIs(refine.call_args.args[2], cfg.extremity)
        self.assertIs(result["ced_score_result"], scores)
        self.assertIs(result["ced_candidate_extraction"], scores["candidate_extraction"])

    def test_visuals_do_not_mutate_votes_or_recompute(self):
        scene = make_scene("same_object_two_corners", noise_mm=0)
        result = run_extremity_refinement(scene.points_mm, 0)
        before = {k: v.copy() for k, v in result.items() if isinstance(v, np.ndarray)}
        for mode in ("heatmap", "clusters"):
            data = extremity_primitives(scene.points_mm, result, color_mode=mode)
            self.assertEqual(data["point_colors"].shape, scene.points_mm.shape)
        image = manager.extremity_summary_rgb(result)
        self.assertEqual(image.shape[2], 3)
        for key, value in before.items():
            np.testing.assert_array_equal(result[key], value)


if __name__ == "__main__":
    unittest.main()
