"""Expected quantiles, probability mass, support cutoff and index invariants."""

import unittest
from dataclasses import replace

import numpy as np
from scipy.special import expit

from corner_detector.extremity import fibonacci_directions, run_extremity_refinement
from corner_detector.extremity_voting import linear_quantile_99, soft_direction_votes
from corner_detector.settings import ExtremityConfig, parse_args, configure
from corner_detector.visualization.view3d import raw_vote_heatmap


class SoftVotingTest(unittest.TestCase):
    def cfg(self, **kwargs):
        return ExtremityConfig(**{**dict(radii_mm=(30.0,), padding_mm=10.0,
            cluster_radius_mm=4.5, nms_radius_mm=5.0, voting_mode="soft_partition",
            soft_tau_mm=0.25), **kwargs})

    def test_partition_matches_linear_quantile_and_does_not_reorder_points(self):
        for dtype in ("float32", "float64"):
            for count in (1, 2, 3, 50, 100, 101, 257):
                values = np.ascontiguousarray(np.random.default_rng(count).normal(size=(17, count)), dtype=dtype)
                before = values.copy()
                expected = np.quantile(values, 0.99, axis=1).astype(dtype)
                np.testing.assert_array_equal(linear_quantile_99(values), expected)
                np.testing.assert_array_equal(values, before)

    def test_dense_votes_match_literal_sigmoid_definition(self):
        points = np.random.default_rng(8).normal(size=(80, 3))
        directions = fibonacci_directions()
        projections = directions @ points.T
        weights = expit((projections - np.quantile(projections, 0.99, axis=1)[:, None]) / 0.25)
        expected = (weights / weights.sum(axis=1, keepdims=True)).sum(axis=0)
        for mode in ("soft_dense", "soft_partition"):
            votes, _, evaluated, possible = soft_direction_votes(points, directions, self.cfg(voting_mode=mode))
            np.testing.assert_allclose(votes, expected, rtol=1e-13, atol=1e-13)
            self.assertAlmostEqual(votes.sum(), 256, places=11)
            self.assertEqual(evaluated, possible)

    def test_sparse_cutoff_converges_and_alpha_four_is_not_assumed_exact(self):
        points = np.random.default_rng(4).normal(size=(300, 3))
        directions = fibonacci_directions()
        dense = soft_direction_votes(points, directions, self.cfg())[0]
        errors, counts = [], []
        for alpha in (4, 6, 8, 1000):
            votes, _, used, total = soft_direction_votes(points, directions,
                self.cfg(voting_mode="soft_sparse", soft_alpha=alpha))
            self.assertAlmostEqual(votes.sum(), 256, places=10)
            errors.append(np.abs(votes - dense).sum())
            counts.append(used)
            self.assertLessEqual(used, total)
        self.assertGreater(errors[0], errors[2])
        self.assertLess(errors[-1], 1e-10)
        self.assertEqual(counts, sorted(counts))

    def test_padding_normalizes_before_discarding_outer_support(self):
        points = np.array([[0., 0., 0.], [29., 0., 0.], [35., 0., 0.]])
        cfg = self.cfg(object_clustering=False)
        all_votes = soft_direction_votes(points, fibonacci_directions(), cfg)[0]
        result = run_extremity_refinement(points, 0, cfg)
        np.testing.assert_allclose(result["raw_votes"][:2], all_votes[:2])
        self.assertEqual(result["raw_votes"][2], 0)
        self.assertAlmostEqual(result["raw_votes"].sum() + result["discarded_vote_mass"], 256)
        self.assertIsNone(result["discarded_vote_count"])
        for candidate in result["candidates"]:
            np.testing.assert_array_equal(candidate["position_mm"], points[candidate["raw_point_index"]])

    def test_partition_preserves_peaks_and_float32_is_close(self):
        x, y = np.meshgrid(np.arange(0, 37, 3), np.arange(0, 37, 3))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        dense = run_extremity_refinement(points, 0, self.cfg(voting_mode="soft_dense"))
        partition = run_extremity_refinement(points, 0, self.cfg())
        low_precision = run_extremity_refinement(points, 0, self.cfg(voting_dtype="float32"))
        np.testing.assert_array_equal(dense["raw_votes"], partition["raw_votes"])
        np.testing.assert_array_equal(dense["candidate_raw_indices"], partition["candidate_raw_indices"])
        np.testing.assert_allclose(low_precision["raw_votes"], dense["raw_votes"], atol=2e-5, rtol=1e-5)

    def test_soft_local_maxima_keep_two_observed_vertices_without_coordinate_averaging(self):
        from corner_detector.extremity_synthetic import make_scene
        scene = make_scene("same_object_two_corners", noise_mm=0)
        query = int(np.argmin(np.linalg.norm(scene.points_mm - scene.focus_mm, axis=1)))
        result = run_extremity_refinement(scene.points_mm, query, self.cfg(min_vote_fraction=0))
        positions = scene.points_mm[result["candidate_raw_indices"]]
        for target in ([-9, 0, 0], [9, 0, 0]):
            self.assertEqual(np.linalg.norm(positions - target, axis=1).min(), 0)
        for candidate in result["candidates"]:
            np.testing.assert_array_equal(candidate["position_mm"], scene.points_mm[candidate["raw_point_index"]])

    def test_config_flags_temperature_units_and_validation(self):
        cfg, _ = configure(parse_args(["--source", "realsense", "--mode", "ball",
            "--voxel-3d-mm", "5", "--ball-voting-mode", "soft_partition",
            "--ball-soft-tau-x", "0.05", "--ball-voting-dtype", "float32"]))
        self.assertEqual(cfg.extremity.soft_tau_mm, 0.25)
        self.assertEqual(cfg.extremity.voting_dtype, "float32")
        for extra in (dict(soft_tau_mm=0), dict(soft_alpha=0), dict(voting_dtype="float16"), dict(voting_mode="other")):
            with self.assertRaises(ValueError):
                self.cfg(**extra).validate()

    def test_heatmap_does_not_truncate_fractional_votes_and_handles_empty_zero(self):
        values = np.array([0, 0.02, 0.1, 0.2])
        before = values.copy()
        colors, stats = raw_vote_heatmap(values)
        self.assertEqual(stats["max"], 0.2)
        self.assertEqual(len(np.unique(colors, axis=0)), 4)
        np.testing.assert_array_equal(values, before)
        for values in (np.zeros(4), np.empty(0), np.ones(3)):
            colors, stats = raw_vote_heatmap(values)
            self.assertTrue(np.isfinite(colors).all())

    def test_paired_comparison_matches_same_gt_seed_and_source_only(self):
        from corner_detector.soft_voting_benchmark import paired_comparisons
        base = dict(scene="test", sigma_mm=0.2, seed=0, query_source="gt_near", gt_index=0,
                    x_mm=0.0, y_mm=0.0, z_mm=0.0, error_mm=0.0, raw_point_index=0)
        rows = [{**base, "variant": "A_hard_f64"}, {**base, "variant": "B_dense_f64"},
                {**base, "variant": "C_partition_f64"},
                {**base, "variant": "C_partition_f32", "x_mm": 0.1, "error_mm": 0.1, "raw_point_index": 1},
                {**base, "variant": "C_partition_f32", "seed": 1}]
        paired = [r for r in paired_comparisons(rows) if r["comparison"] == "float64"
                  and r["variant"] == "C_partition_f32"]
        self.assertEqual(paired[0]["paired_observations"], 1)
        self.assertAlmostEqual(paired[0]["position_delta_max_mm"], 0.1)


if __name__ == "__main__":
    unittest.main()
