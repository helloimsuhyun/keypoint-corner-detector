"""Contracts for deterministic fusion-score 3-D candidate extraction."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from corner_detector.candidate_extraction import extract_corner_candidates
from corner_detector.edge_suppression import dense_ced_weighted_pca_scores
from corner_detector.settings import DEFAULTS, configure, parse_args


def score_record(scores, valid=None):
    values = np.asarray(scores, dtype=np.float64)
    mask = (
        np.ones(len(values), dtype=bool)
        if valid is None else np.asarray(valid, dtype=bool)
    )
    return {
        "ced_score": values.copy(),
        "bilateral_ced_score": values.copy(),
        "pca_ratio_raw": np.ones(len(values), dtype=np.float64),
        "pca_ratio_saturated": np.ones(len(values), dtype=np.float64),
        "pca_ratio_saturation_tau": None,
        "fusion_score": values.copy(),
        "pca_valid_mask": mask.copy(),
        "fusion_valid_mask": mask.copy(),
        "invalid_support_mask": ~mask,
    }


class CandidateExtractionTest(unittest.TestCase):
    def test_candidate_defaults_and_explicit_cli_values(self):
        default_args = parse_args([
            "--source", "realsense", "--mode", "detector", "--no-3d-view",
        ])
        default_cfg, _noise = configure(default_args)
        self.assertEqual(
            default_cfg.nms_radius_mm,
            (
                0.5 * default_cfg.centroid_radius_mm
                if DEFAULTS.nms_radius_mm is None
                else DEFAULTS.nms_radius_mm
            ),
        )
        self.assertEqual(
            default_cfg.pca_radius_mm,
            DEFAULTS.pca_radius_factor * default_cfg.centroid_radius_mm,
        )
        self.assertEqual(default_cfg.bilateral_radius_mm, default_cfg.centroid_radius_mm)
        self.assertEqual(
            default_cfg.bilateral_sigma_spatial_mm,
            0.5 * default_cfg.centroid_radius_mm,
        )
        self.assertEqual(
            default_cfg.pca_ratio_saturation_tau,
            DEFAULTS.pca_ratio_saturation_tau,
        )
        self.assertEqual(
            default_cfg.fusion_score_threshold,
            DEFAULTS.fusion_score_threshold,
        )
        self.assertEqual(
            default_cfg.fusion_threshold_mode,
            DEFAULTS.fusion_threshold_mode.replace("-", "_"),
        )
        self.assertFalse(hasattr(default_cfg, "query_radius_mm"))

        custom_args = parse_args([
            "--source", "realsense", "--mode", "detector", "--no-3d-view",
            "--nms-radius-mm", "12.5",
            "--pca-radius-factor", "1.4",
            "--bilateral-radius-mm", "18",
            "--bilateral-sigma-spatial-mm", "7",
            "--bilateral-sigma-ced", "0.08",
            "--pca-ratio-saturation-tau", "0.5",
            "--fusion-score-threshold", "0.025",
            "--fusion-threshold-mode", "global-mean",
        ])
        custom_cfg, _noise = configure(custom_args)
        self.assertEqual(custom_cfg.nms_radius_mm, 12.5)
        self.assertEqual(custom_cfg.pca_radius_factor, 1.4)
        self.assertEqual(custom_cfg.bilateral_radius_mm, 18.0)
        self.assertEqual(custom_cfg.bilateral_sigma_spatial_mm, 7.0)
        self.assertEqual(custom_cfg.bilateral_sigma_ced, 0.08)
        self.assertEqual(custom_cfg.pca_ratio_saturation_tau, 0.5)
        self.assertEqual(custom_cfg.fusion_score_threshold, 0.025)
        self.assertEqual(custom_cfg.fusion_threshold_mode, "global_mean")

    def test_heatmap_config_rejects_nonpositive_saturation_tau(self):
        for value in ("0", "-0.1", "nan", "inf"):
            args = parse_args([
                "--source", "realsense", "--mode", "detector",
                "--pca-ratio-saturation-tau", value,
            ])
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure(args)

    def test_heatmap_config_rejects_out_of_range_fusion_threshold(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            args = parse_args([
                "--source", "realsense", "--mode", "detector",
                "--fusion-score-threshold", value,
            ])
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure(args)

    def test_synthetic_two_direction_corner_keeps_the_corner_sample(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
             [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0],
             [0.0, 3.0, 0.0]],
        )
        scores = dense_ced_weighted_pca_scores(points, 1.6, 4.0, 2)
        originals = {
            key: scores[key].copy()
            for key in ("ced_score", "pca_ratio_raw", "fusion_score")
        }

        result = extract_corner_candidates(points, scores, 1.5)

        self.assertIn(0, result["candidate_indices"])
        for key, original in originals.items():
            np.testing.assert_array_equal(scores[key], original)

    def test_higher_fusion_score_wins_and_ties_prefer_smaller_index(self):
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
        higher = extract_corner_candidates(points, score_record([0.9, 0.8, 0.7]), 1.0)
        tied = extract_corner_candidates(points, score_record([0.8, 0.8, 0.7]), 1.0)

        np.testing.assert_array_equal(higher["candidate_indices"], [0, 2])
        np.testing.assert_array_equal(tied["candidate_indices"], [0, 2])

    def test_fusion_threshold_is_strict_and_applied_before_nms(self):
        points = np.column_stack((np.arange(4.0), np.zeros((4, 2))))
        scores = score_record([0.0, 0.2, 0.2001, 0.9])

        result = extract_corner_candidates(
            points,
            scores,
            0.1,
            fusion_score_threshold=0.2,
        )

        np.testing.assert_array_equal(result["candidate_indices"], [3, 2])
        self.assertEqual(result["valid_score_count"], 2)
        self.assertEqual(result["fusion_score_threshold"], 0.2)

    def test_global_mean_threshold_uses_all_finite_fusion_valid_scores(self):
        points = np.column_stack((10.0 * np.arange(4.0), np.zeros((4, 2))))
        scores = score_record([0.1, 0.2, 0.9, 1.0], valid=[True, True, True, False])

        result = extract_corner_candidates(
            points,
            scores,
            1.0,
            fusion_score_threshold=0.95,
            fusion_threshold_mode="global_mean",
        )

        self.assertAlmostEqual(result["fusion_score_threshold"], 0.4)
        self.assertEqual(result["configured_fusion_score_threshold"], 0.95)
        self.assertEqual(result["fusion_threshold_mode"], "global_mean")
        self.assertEqual(result["threshold_sample_count"], 3)
        np.testing.assert_array_equal(result["candidate_indices"], [2])

    def test_nms_requires_a_true_radius_local_maximum_not_greedy_survival(self):
        # Point 2 is outside point 0's radius, so greedy suppression retained it.
        # It is not a local maximum because its direct neighbor point 1 is higher.
        points = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [1.5, 0.0, 0.0]])
        result = extract_corner_candidates(
            points, score_record([0.9, 0.8, 0.7]), 1.0,
        )

        np.testing.assert_array_equal(result["candidate_indices"], [0])
        self.assertEqual(result["nms_method"], "radius_local_maximum")

    def test_peak_relative_uses_finite_valid_maximum_and_strict_threshold(self):
        points = np.column_stack((10. * np.arange(6), np.zeros((6, 2))))
        scores = score_record([np.nan, .8, .4, .3, .09, .99], valid=[True]*5 + [False])
        original = scores["fusion_score"].copy()
        result = extract_corner_candidates(points, scores, 1., fusion_threshold_mode="peak-relative",
            fusion_score_threshold=.1, fusion_peak_relative_ratio=.5)
        self.assertEqual(result["fusion_score_threshold"], .4)
        self.assertEqual(result["fusion_threshold_mode"], "peak_relative")
        self.assertEqual(result["fusion_peak_relative_ratio"], .5)
        self.assertEqual(result["threshold_sample_count"], 4)
        np.testing.assert_array_equal(result["candidate_indices"], [1])
        np.testing.assert_array_equal(scores["fusion_score"], original)

    def test_peak_relative_floor_zero_scores_empty_support_and_ratio_limits(self):
        points = np.column_stack((10. * np.arange(3), np.zeros((3, 2))))
        for values, valid, ratio, floor, threshold, expected in (
            ([.09, .05, .01], None, .5, .1, .1, []),
            ([0., 0., 0.], None, .5, 0., 0., []),
            ([.8, .4, .3], [False]*3, .5, .1, .1, []),
            ([.8, .4, .3], None, 0., .1, .1, [0, 1, 2]),
            ([.8, .4, .3], None, 1., .1, .8, []),
        ):
            with self.subTest(values=values, valid=valid, ratio=ratio):
                result = extract_corner_candidates(points, score_record(values, valid), 1.,
                    fusion_threshold_mode="peak_relative", fusion_score_threshold=floor,
                    fusion_peak_relative_ratio=ratio)
                self.assertEqual(result["fusion_score_threshold"], threshold)
                np.testing.assert_array_equal(result["candidate_indices"], expected)
        result = extract_corner_candidates(np.empty((0, 3)), score_record([]), 1.,
            fusion_threshold_mode="peak_relative", fusion_score_threshold=.1)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["fusion_score_threshold"], .1)

    def test_peak_relative_rejects_invalid_ratios(self):
        for ratio in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                extract_corner_candidates(np.zeros((1, 3)), score_record([.5]), 1.,
                    fusion_threshold_mode="peak_relative", fusion_peak_relative_ratio=ratio)

    def test_fusion_score_not_pca_ratio_controls_ranking(self):
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        scores = score_record([0.1, 0.7])
        scores["pca_ratio_raw"] = np.array([0.9, 0.8])
        scores["pca_ratio_saturated"] = np.array([0.4, 0.35])
        scores["bilateral_ced_score"] = np.array([0.1, 0.875])
        scores["fusion_score"] = np.array([0.09, 0.7])

        result = extract_corner_candidates(points, scores, 1.0)

        np.testing.assert_array_equal(result["candidate_indices"], [1])
        self.assertEqual(result["ranking_score"], "fusion_score")
        self.assertEqual(result["candidates"][0]["pca_ratio_raw"], 0.8)
        self.assertEqual(result["candidates"][0]["pca_ratio_saturated"], 0.35)
        self.assertEqual(result["candidates"][0]["fusion_score"], 0.7)

    def test_invalid_high_score_cannot_suppress_a_valid_candidate(self):
        points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        scores = score_record([0.99, 0.8], valid=[False, True])

        result = extract_corner_candidates(points, scores, 1.0)

        np.testing.assert_array_equal(result["candidate_indices"], [1])

    def test_larger_radius_does_not_increase_count(self):
        points = np.column_stack((np.arange(6.0), np.zeros((6, 2))))
        scores = score_record([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        counts = [
            extract_corner_candidates(points, scores, radius)["candidate_count"]
            for radius in (0.1, 1.1, 4.1)
        ]
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])

    def test_nonfinite_scores_are_excluded_without_mutation(self):
        points = np.column_stack((np.arange(5.0), np.zeros((5, 2))))
        scores = score_record([np.nan, np.inf, -0.1, 0.6, 0.5])
        originals = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in scores.items()
        }

        result = extract_corner_candidates(points, scores, 0.1)

        np.testing.assert_array_equal(result["candidate_indices"], [3, 4])
        self.assertEqual(result["valid_score_count"], 2)
        for key, original in originals.items():
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(scores[key], original)
            else:
                self.assertEqual(scores[key], original)

    def test_candidate_record_preserves_current_detector_fields(self):
        points = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        scores = score_record([0.175, 0.05])
        scores.update({
            "ced_score": np.array([0.4, 0.3]),
            "bilateral_ced_score": np.array([0.35, 0.25]),
            "pca_ratio_raw": np.array([0.5, 0.2]),
            "pca_ratio_saturated": np.array([0.4, 0.18]),
            "pca_ratio_saturation_tau": 0.5,
            "fusion_score": np.array([0.175, 0.05]),
            "eigenvalues_mm2": np.array([[3.0, 1.5, 0.1], [2.0, 0.4, 0.0]]),
        })
        result = extract_corner_candidates(
            points, scores, 1.0, point_tree=cKDTree(points),
        )
        candidate = result["candidates"][0]

        self.assertEqual(candidate["point_index"], 0)
        np.testing.assert_array_equal(candidate["position_mm"], points[0])
        self.assertEqual(candidate["ced_score"], 0.4)
        self.assertEqual(candidate["bilateral_ced_score"], 0.35)
        self.assertEqual(candidate["pca_ratio_raw"], 0.5)
        self.assertEqual(candidate["pca_ratio_saturated"], 0.4)
        self.assertEqual(candidate["pca_ratio_saturation_tau"], 0.5)
        self.assertEqual(candidate["fusion_score"], 0.175)
        np.testing.assert_array_equal(candidate["eigenvalues_mm2"], [3.0, 1.5, 0.1])
        self.assertEqual(candidate["nms_radius_mm"], 1.0)
        self.assertEqual(candidate["fusion_score_threshold"], 0.0)
        self.assertEqual(candidate["fusion_threshold_mode"], "fixed")
        self.assertEqual(candidate["nms_method"], "radius_local_maximum")
        self.assertFalse(candidate["verified"])


if __name__ == "__main__":
    unittest.main()
