"""Numerical and visualization contracts for the whole-cloud detector."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import numpy as np
from scipy.spatial import cKDTree

from corner_detector.dense_heatmap import run_dense_heatmap_mode
from corner_detector.edge_suppression import (
    bilateral_filter_ced,
    ced_weighted_pca,
    dense_ced_weighted_pca_scores,
    saturate_pca_ratio,
    weighted_pca_query_debug,
)
from corner_detector.visualization import manager
from corner_detector.visualization import view3d
from corner_detector.settings import DEFAULTS, DetectorConfig, configure, parse_args
from corner_detector.visualization.view3d import (
    corner_candidate_primitives,
    dense_heatmap_colors,
    dense_heatmap_panels,
    dense_heatmap_vmax,
    pca_ratio_heatmap_colors,
    synchronize_dense_camera_views,
)


def corner_points():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0],
         [0.0, 3.0, 0.0], [1.0, 1.0, 0.0]],
    )


class CedWeightedPcaTest(unittest.TestCase):
    def test_pca_ratio_saturation_identity_formula_and_invalid_preservation(self):
        ratio = np.array([0.0, 0.1, 0.5, 1.0, np.nan, np.inf])
        identity = saturate_pca_ratio(ratio, tau=None)
        saturated = saturate_pca_ratio(ratio, tau=0.3)

        np.testing.assert_array_equal(identity, ratio)
        np.testing.assert_allclose(
            saturated[:4], 0.3 * np.tanh(ratio[:4] / 0.3),
            rtol=0.0, atol=0.0,
        )
        self.assertEqual(saturated[0], 0.0)
        self.assertTrue(np.all((saturated[1:4] > 0.0) & (saturated[1:4] < ratio[1:4])))
        self.assertTrue(np.isnan(saturated[4]))
        self.assertTrue(np.isinf(saturated[5]))

    def test_pca_ratio_saturation_is_monotonic_and_approaches_raw_as_tau_grows(self):
        ratio = np.linspace(0.0, 1.0, 21)
        tau_small = saturate_pca_ratio(ratio, tau=0.3)
        tau_large = saturate_pca_ratio(ratio, tau=1.0)

        self.assertTrue(np.all(np.diff(tau_small) >= 0.0))
        self.assertTrue(np.all(np.diff(tau_large) >= 0.0))
        self.assertTrue(np.all(tau_large >= tau_small))
        self.assertLess(
            np.linalg.norm(ratio - tau_large),
            np.linalg.norm(ratio - tau_small),
        )

    def test_pca_ratio_saturation_rejects_invalid_tau(self):
        for tau in (0.0, -0.1, np.nan, np.inf, -np.inf):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                saturate_pca_ratio([0.2], tau=tau)

    def test_bilateral_constant_field_is_invariant_and_points_are_unchanged(self):
        points = np.column_stack((np.arange(5.0), np.zeros((5, 2))))
        original_points = points.copy()
        result = bilateral_filter_ced(
            points,
            np.full(5, 0.37),
            np.ones(5, dtype=bool),
            radius_mm=2.0,
            sigma_spatial_mm=1.0,
            sigma_ced=0.05,
        )

        np.testing.assert_allclose(result["score"], 0.37, atol=1e-15)
        np.testing.assert_array_equal(points, original_points)

    def test_bilateral_range_term_preserves_a_sharp_discontinuity(self):
        points = np.column_stack((np.arange(4.0), np.zeros((4, 2))))
        result = bilateral_filter_ced(
            points,
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.ones(4, dtype=bool),
            radius_mm=2.0,
            sigma_spatial_mm=2.0,
            sigma_ced=0.05,
        )

        self.assertLess(result["score"][1], 1e-12)
        self.assertGreater(result["score"][2], 1.0 - 1e-12)

    def test_bilateral_rejects_nonfinite_input(self):
        points = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            bilateral_filter_ced(
                points, [0.0, np.nan], [True, True], 1.0, 1.0, 0.05,
            )
        with self.assertRaises(ValueError):
            ced_weighted_pca(points, [1.0, np.inf])

    def test_centered_weighted_pca_suppresses_a_straight_line(self):
        line = np.array(
            [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        record = ced_weighted_pca(line, np.ones(len(line)))

        np.testing.assert_allclose(record["centroid_mm"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(record["eigenvalues_mm2"], [2.0, 0.0, 0.0])

    def test_two_direction_corner_has_a_nonzero_second_eigenvalue(self):
        record = ced_weighted_pca(corner_points()[:-1], np.ones(7))
        ratio = record["eigenvalues_mm2"][1] / record["eigenvalues_mm2"][0]

        self.assertGreater(ratio, 0.2)
        self.assertEqual(record["eigenvalues_mm2"][2], 0.0)

    def test_weighted_covariance_is_translation_invariant(self):
        points = np.array(
            [[10.0, 20.0, 30.0], [11.0, 20.0, 30.0],
             [12.0, 20.0, 30.0], [10.0, 21.0, 30.0]],
        )
        weights = np.array([0.1, 0.2, 0.8, 0.4])
        translated = ced_weighted_pca(points + [100.0, -50.0, 7.0], weights)
        original = ced_weighted_pca(points, weights)

        np.testing.assert_allclose(
            translated["covariance_mm2"], original["covariance_mm2"], atol=1e-12,
        )
        np.testing.assert_allclose(
            translated["eigenvalues_mm2"], original["eigenvalues_mm2"], atol=1e-12,
        )

    def test_pca_uses_all_positive_original_ced_support_without_threshold(self):
        points = corner_points()
        result = dense_ced_weighted_pca_scores(points, 1.6, 4.0, 2)
        valid_indices = np.flatnonzero(result["pca_valid_mask"])
        self.assertGreater(len(valid_indices), 0)
        index = int(valid_indices[0])
        tree = cKDTree(points)
        neighbors = np.asarray(tree.query_ball_point(points[index], 4.0), dtype=int)
        delta = points[neighbors] - points[index]
        neighbors = neighbors[np.einsum("ij,ij->i", delta, delta) <= 16.0]
        neighbors = neighbors[
            result["ced_valid_mask"][neighbors]
            & (result["ced_score"][neighbors] > 0.0)
        ]
        manual = ced_weighted_pca(points[neighbors], result["ced_score"][neighbors])

        np.testing.assert_allclose(
            result["weighted_centroid_mm"][index], manual["centroid_mm"],
        )
        np.testing.assert_allclose(
            result["weighted_covariance_mm2"][index], manual["covariance_mm2"],
        )
        np.testing.assert_allclose(
            result["eigenvalues_mm2"][index], manual["eigenvalues_mm2"],
        )

    def test_pca_ced_cutoff_is_inclusive_and_uses_raw_ced_in_every_bilateral_mode(self):
        points = corner_points()
        for mode in ("ced", "fusion", "off"):
            options = dict(bilateral_mode=mode, bilateral_sigma_ced=1.)
            baseline = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, **options)
            cutoff = baseline["ced_score"][3]
            result = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, pca_ced_cutoff=cutoff, **options)
            self.assertEqual(result["pca_ced_cutoff"], cutoff)
            for key in ("ced_score", "ced_valid_mask", "bilateral_ced_score", "pca_neighbor_count",
                        "pca_query_mask", "original_point_indices", "query_position_mm"):
                np.testing.assert_array_equal(result[key], baseline[key])
            self.assertEqual(result["weighted_neighbor_count"][0], 3)
            self.assertLess(result["ced_score"][1], cutoff)
            self.assertTrue(result["pca_valid_mask"][1])
            if mode == "ced":
                self.assertLess(result["bilateral_ced_score"][3], cutoff)
            for index in np.flatnonzero(result["pca_query_mask"]):
                ids = np.flatnonzero((np.sum((points-points[index])**2, axis=1) <= 16.)
                                    & result["ced_valid_mask"] & (result["ced_score"] > 0.)
                                    & (result["ced_score"] >= cutoff))
                self.assertEqual(result["weighted_neighbor_count"][index], len(ids))
                if len(ids) < 2:
                    self.assertFalse(result["pca_valid_mask"][index])
                    continue
                weights = result["ced_score"][ids]
                mean = np.average(points[ids], axis=0, weights=weights)
                delta = points[ids]-mean
                covariance = (delta*weights[:, None]).T @ delta / weights.sum()
                eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.)
                np.testing.assert_allclose(result["weighted_centroid_mm"][index], mean)
                np.testing.assert_allclose(result["weighted_covariance_mm2"][index], covariance, atol=1e-15)
                np.testing.assert_allclose(result["eigenvalues_mm2"][index], eigenvalues, atol=1e-15)
            self.assertGreater(np.max(np.abs(result["pca_ratio_raw"]-baseline["pca_ratio_raw"])), .01)

    def test_pca_ced_cutoff_empty_support_invalidates_fusion_and_zero_preserves_outputs(self):
        points = np.vstack((corner_points(), [[100., 100., 100.]]*2))
        baseline = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, bilateral_mode="fusion")
        zero = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, bilateral_mode="fusion", pca_ced_cutoff=0.)
        for key, value in baseline.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(zero[key], value)
        self.assertTrue(zero["ced_valid_mask"][-1])
        self.assertEqual(zero["weighted_neighbor_count"][-1], 0)
        empty = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, bilateral_mode="fusion", pca_ced_cutoff=1.)
        np.testing.assert_array_equal(empty["ced_score"], baseline["ced_score"])
        self.assertFalse(empty["weighted_neighbor_count"].any())
        self.assertFalse(empty["pca_valid_mask"].any())
        self.assertFalse(empty["fusion_valid_mask"].any())
        self.assertFalse(empty["fusion_score"].any())

    def test_pca_ced_cutoff_rejects_nonfinite_and_out_of_range_values(self):
        for cutoff in (-.01, 1.01, np.nan, np.inf, -np.inf):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "pca_ced_cutoff"):
                dense_ced_weighted_pca_scores(corner_points(), 1.6, 4., 2, pca_ced_cutoff=cutoff)

    def test_tau_none_fusion_is_exact_bilateral_ced_times_raw_pca_ratio(self):
        result = dense_ced_weighted_pca_scores(corner_points(), 1.6, 4.0, 2)
        valid = result["fusion_valid_mask"]

        np.testing.assert_array_equal(
            result["pca_ratio_saturated"], result["pca_ratio_raw"],
        )
        np.testing.assert_allclose(
            result["fusion_score"][valid],
            result["bilateral_ced_score"][valid]
            * result["pca_ratio_saturated"][valid],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            result["fusion_valid_mask"],
            result["pca_valid_mask"] & result["bilateral_valid_mask"],
        )
        self.assertTrue(np.all(result["fusion_score"][~valid] == 0.0))

    def test_bilateral_off_skips_filter_and_uses_original_ced_without_changing_pca(self):
        points = np.vstack((corner_points(), [100., 100., 100.]))
        for radius in (1., 5.):
            for tau in (None, 0.6):
                with self.subTest(radius=radius, tau=tau):
                    options = dict(bilateral_radius_mm=radius, pca_ratio_saturation_tau=tau)
                    enabled = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, **options)
                    self.assertEqual(enabled["bilateral_mode"], "ced")
                    with patch("corner_detector.edge_suppression._bilateral_neighborhood",
                               side_effect=AssertionError("bilateral must be skipped")):
                        disabled = dense_ced_weighted_pca_scores(
                            points, 1.6, 4., 2, bilateral_mode="off", **options)
                    self.assertEqual(disabled["bilateral_mode"], "off")
                    np.testing.assert_array_equal(disabled["bilateral_ced_score"], disabled["ced_score"])
                    for key in ("ced_score", "ced_valid_mask", "pca_query_mask", "pca_valid_mask",
                                "weighted_covariance_mm2", "pca_ratio_raw", "pca_ratio_saturated",
                                "fusion_valid_mask", "original_point_indices", "query_position_mm"):
                        np.testing.assert_array_equal(disabled[key], enabled[key])
                    valid = disabled["fusion_valid_mask"]
                    self.assertTrue(valid.any())
                    np.testing.assert_array_equal(disabled["fusion_score"][valid],
                                                  disabled["ced_score"][valid] * disabled["pca_ratio_saturated"][valid])
                    self.assertTrue(np.all(disabled["fusion_score"][~valid] == 0.))
                    self.assertFalse(disabled["bilateral_neighbor_count"].any())
                    self.assertFalse(disabled["bilateral_weight_sum"].any())

    def test_bilateral_mode_rejects_invalid_values(self):
        for value in ("False", "invalid", "CED", "", 0, 1, None, True, False):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bilateral_mode"):
                dense_ced_weighted_pca_scores(corner_points(), 1.6, 4., 2, bilateral_mode=value)

    def test_fusion_bilateral_filters_completed_product_and_excludes_invalid_support(self):
        points = np.vstack((corner_points(), [[100., 100., 100.]]*2, [300., 300., 300.]))
        for radius, spatial_sigma in ((2.5, .8), (200., 200.)):
            for tau in (None, .6):
                with self.subTest(radius=radius, tau=tau):
                    options = dict(bilateral_radius_mm=radius, bilateral_sigma_spatial_mm=spatial_sigma,
                                   bilateral_sigma_ced=.2, pca_ratio_saturation_tau=tau)
                    raw = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, bilateral_mode="off", **options)
                    result = dense_ced_weighted_pca_scores(points, 1.6, 4., 2, bilateral_mode="fusion", **options)
                    valid = raw["fusion_valid_mask"]
                    self.assertTrue(raw["ced_valid_mask"][-2])
                    self.assertFalse(valid[-2])
                    values = raw["fusion_score"]
                    delta = points[:, None, :] - points[None, :, :]
                    distance2 = np.sum(delta**2, axis=2)
                    support = (distance2 <= radius**2) & valid[:, None] & valid[None, :]
                    weights = (np.exp(-distance2/(2*spatial_sigma**2))
                               * np.exp(-(values[:, None]-values[None, :])**2/(2*.2**2)) * support)
                    expected = np.zeros(len(points))
                    expected[valid] = (weights @ values)[valid] / weights.sum(axis=1)[valid]
                    np.testing.assert_allclose(result["fusion_score"], expected, rtol=0, atol=1e-15)
                    self.assertGreater(np.max(np.abs(result["fusion_score"]-values)), 1e-4)
                    np.testing.assert_array_equal(result["fusion_score_raw"], values)
                    np.testing.assert_array_equal(result["bilateral_ced_score"], raw["ced_score"])
                    np.testing.assert_array_equal(result["bilateral_neighbor_count"], support.sum(axis=1))
                    np.testing.assert_allclose(result["bilateral_weight_sum"], weights.sum(axis=1), atol=1e-14)
                    for key in ("ced_score", "ced_valid_mask", "pca_query_mask", "pca_valid_mask",
                                "weighted_covariance_mm2", "pca_ratio_raw", "pca_ratio_saturated",
                                "fusion_valid_mask", "original_point_indices", "query_position_mm"):
                        np.testing.assert_array_equal(result[key], raw[key])
                    self.assertTrue(np.all(result["fusion_score"][~valid] == 0.))

    def test_positive_tau_changes_only_saturated_ratio_and_fusion(self):
        raw = dense_ced_weighted_pca_scores(corner_points(), 1.6, 4.0, 2)
        saturated = dense_ced_weighted_pca_scores(
            corner_points(), 1.6, 4.0, 2,
            pca_ratio_saturation_tau=0.3,
        )
        valid = saturated["fusion_valid_mask"]

        np.testing.assert_array_equal(
            saturated["pca_ratio_raw"], raw["pca_ratio_raw"],
        )
        np.testing.assert_array_equal(
            saturated["eigenvalues_mm2"], raw["eigenvalues_mm2"],
        )
        np.testing.assert_array_equal(saturated["ced_score"], raw["ced_score"])
        np.testing.assert_array_equal(
            saturated["bilateral_ced_score"], raw["bilateral_ced_score"],
        )
        np.testing.assert_allclose(
            saturated["pca_ratio_saturated"][valid],
            0.3 * np.tanh(saturated["pca_ratio_raw"][valid] / 0.3),
        )
        np.testing.assert_allclose(
            saturated["fusion_score"][valid],
            saturated["bilateral_ced_score"][valid]
            * saturated["pca_ratio_saturated"][valid],
        )
        self.assertTrue(np.all(saturated["fusion_score"][valid] >= 0.0))
        self.assertTrue(np.all(saturated["fusion_score"][valid] <= 1.0))

    def test_zero_factor_produces_zero_fusion_score(self):
        line = np.array(
            [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        result = dense_ced_weighted_pca_scores(line, 1.6, 3.0, 1)
        real_zero = result["pca_valid_mask"] & (result["pca_ratio_raw"] == 0.0)
        self.assertTrue(np.any(real_zero))
        self.assertTrue(np.all(result["fusion_score"][real_zero] == 0.0))

    def test_sparse_and_degenerate_support_stays_finite_and_invalid(self):
        self.assertIsNone(ced_weighted_pca(np.eye(3), np.zeros(3)))
        sparse = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = dense_ced_weighted_pca_scores(sparse, 1.0, 1.8, 2)

        self.assertFalse(np.any(result["ced_valid_mask"]))
        self.assertFalse(np.any(result["pca_valid_mask"]))
        self.assertFalse(np.any(result["fusion_valid_mask"]))
        for key in (
            "ced_score", "bilateral_ced_score", "pca_ratio_raw",
            "pca_ratio_saturated", "fusion_score",
            "eigenvalues_mm2", "eigenvectors", "weighted_covariance_mm2",
        ):
            self.assertTrue(np.all(np.isfinite(result[key])), key)

    def test_query_debug_exposes_only_current_score_stages(self):
        points = corner_points()
        result = dense_ced_weighted_pca_scores(points, 1.6, 4.0, 1)
        debug = weighted_pca_query_debug(result, 0)

        np.testing.assert_array_equal(debug["query_position_mm"], points[0])
        self.assertEqual(debug["eigenvalues_mm2"].shape, (3,))
        self.assertEqual(debug["eigenvectors"].shape, (3, 3))
        self.assertAlmostEqual(debug["pca_ratio_raw"], result["pca_ratio_raw"][0])
        self.assertAlmostEqual(
            debug["pca_ratio_saturated"], result["pca_ratio_saturated"][0],
        )
        self.assertIsNone(debug["pca_ratio_saturation_tau"])
        self.assertAlmostEqual(debug["fusion_score"], result["fusion_score"][0])
        self.assertNotIn("ced_soft_weight", debug)
        self.assertNotIn("combined_score", debug)

    def test_tree_is_built_once_and_shared_by_all_stages(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )
        config = SimpleNamespace(
            centroid_radius_mm=3.0,
            pca_radius_mm=5.4,
            nms_radius_mm=1.5,
            min_neighbors=2,
            show_3d=False,
            bilateral_radius_mm=3.0,
            bilateral_sigma_spatial_mm=1.5,
            bilateral_sigma_ced=0.05,
            pca_ratio_saturation_tau=None,
            fusion_score_threshold=0.0,
        )
        with patch("corner_detector.dense_heatmap.cKDTree", wraps=__import__(
            "scipy.spatial", fromlist=["cKDTree"],
        ).cKDTree) as tree_type:
            result = run_dense_heatmap_mode(points, config)

        self.assertEqual(tree_type.call_count, 1)
        self.assertEqual(len(result["ced_score"]), len(points))

    def test_dense_nms_output_is_forwarded_to_shared_validator_batch(self):
        points = corner_points()
        config = SimpleNamespace(
            centroid_radius_mm=1.6,
            pca_radius_mm=4.0,
            nms_radius_mm=1.5,
            min_neighbors=1,
            show_3d=False,
            bilateral_radius_mm=1.6,
            bilateral_sigma_spatial_mm=0.8,
            bilateral_sigma_ced=0.05,
            pca_ratio_saturation_tau=None,
            fusion_score_threshold=0.0,
            fusion_threshold_mode="fixed",
            validator=object(),
        )

        def fake_batch(_points, indices, validator_cfg, estimator):
            self.assertIs(validator_cfg, config.validator)
            self.assertIs(estimator, normal_estimator)
            return {
                "records": [
                    {"query_index": int(index), "status": "ok"}
                    for index in indices
                ],
                "final_edges": [],
                "final_corners": [],
                "diagnostics": {
                    "candidate_count": len(indices),
                    "completed_count": len(indices),
                    "failed_count": 0,
                    "final_edge_count": 0,
                    "final_corner_count": 0,
                    "elapsed_s": 0.0,
                },
            }

        normal_estimator = object()
        with patch(
            "corner_detector.dense_heatmap.run_validator_candidates",
            side_effect=fake_batch,
        ) as shared:
            result = run_dense_heatmap_mode(
                points, config, validator_estimator=normal_estimator,
            )

        shared.assert_called_once()
        np.testing.assert_array_equal(
            shared.call_args.args[1],
            result["candidate_extraction"]["candidate_indices"],
        )
        self.assertEqual(
            result["validator_results"]["diagnostics"]["candidate_count"],
            result["candidate_extraction"]["candidate_count"],
        )


class DetectorOnlyTest(unittest.TestCase):
    def test_pca_ced_cutoff_settings_cli_and_both_scoring_callers(self):
        from corner_detector.extremity_synthetic import ced_candidates

        self.assertEqual(DetectorConfig().pca_ced_cutoff, DEFAULTS.pca_ced_cutoff)
        with patch("corner_detector.settings.DEFAULTS", replace(DEFAULTS, pca_ced_cutoff=.1)):
            for mode in ("detector", "validator", "total", "new-validator", "ball", "extremity256"):
                cfg, _ = configure(parse_args(["--source", "realsense", "--mode", mode, "--no-q10-validator"]))
                self.assertEqual(cfg.pca_ced_cutoff, .1)
                cfg, _ = configure(parse_args(["--source", "realsense", "--mode", mode,
                                               "--no-q10-validator", "--pca-ced-cutoff", ".3125"]))
                self.assertEqual(cfg.pca_ced_cutoff, .3125)
        for value in ("-.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure(parse_args(["--source", "realsense", "--mode", "detector", "--pca-ced-cutoff", value]))
        cfg = DetectorConfig(centroid_radius_mm=1.6, pca_radius_mm=4., min_neighbors=2,
                             bilateral_radius_mm=1.6, nms_radius_mm=1.5, pca_ced_cutoff=.3125,
                             fusion_score_threshold=0., show_3d=False)
        with redirect_stdout(StringIO()):
            result = run_dense_heatmap_mode(corner_points(), cfg)
            candidates, _ = ced_candidates(corner_points(), cfg)
        self.assertEqual(result["pca_ced_cutoff"], .3125)
        np.testing.assert_array_equal(candidates["candidate_indices"],
                                      result["candidate_extraction"]["candidate_indices"])
        self.assertGreater(candidates["candidate_count"], 0)
        for candidate in candidates["candidates"]:
            self.assertEqual(candidate["weighted_support_count"],
                             result["weighted_neighbor_count"][candidate["point_index"]])

    def test_bilateral_cli_modes_reach_shared_configs_and_fusion_scoring_callers(self):
        from corner_detector.extremity_synthetic import ced_candidates

        for target in ("ced", "fusion", "off"):
            for mode in ("detector", "validator", "total", "new-validator", "ball", "extremity256"):
                cfg, _ = configure(parse_args(["--source", "realsense", "--mode", mode,
                                               "--bilateral-mode", target, "--no-q10-validator"]))
                self.assertEqual(cfg.bilateral_mode, target)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--source", "realsense", "--mode", "detector", "--bilateral-mode", "invalid"])
        cfg = DetectorConfig(centroid_radius_mm=1.6, pca_radius_mm=4., min_neighbors=2,
                             bilateral_radius_mm=1.6, bilateral_mode="fusion", nms_radius_mm=1.5,
                             fusion_score_threshold=0., show_3d=False)
        with redirect_stdout(StringIO()), patch("corner_detector.edge_suppression.bilateral_filter_ced",
                                                wraps=bilateral_filter_ced) as filtering:
            scores = run_dense_heatmap_mode(corner_points(), cfg)
            candidates, _ = ced_candidates(corner_points(), cfg)
        self.assertEqual(filtering.call_count, 2)
        self.assertEqual(scores["bilateral_mode"], "fusion")
        np.testing.assert_array_equal(candidates["candidate_indices"],
                                      scores["candidate_extraction"]["candidate_indices"])
        for candidate in candidates["candidates"]:
            self.assertEqual(candidate["score"], scores["fusion_score"][candidate["point_index"]])

    def test_settings_bilateral_mode_reaches_shared_configs_and_scoring_callers(self):
        from corner_detector.extremity_synthetic import ced_candidates

        self.assertEqual(DetectorConfig().bilateral_mode, DEFAULTS.bilateral_mode)
        self.assertFalse(hasattr(DetectorConfig(), "bilateral_enabled"))
        with patch("corner_detector.settings.DEFAULTS", replace(DEFAULTS, bilateral_mode="off")):
            for mode in ("detector", "validator", "total", "new-validator", "ball", "extremity256"):
                with self.subTest(mode=mode):
                    cfg, _ = configure(parse_args(["--source", "realsense", "--mode", mode,
                                                   "--no-q10-validator", "--no-3d-view"]))
                    self.assertEqual(cfg.bilateral_mode, "off")
            cfg, _ = configure(parse_args(["--source", "realsense", "--mode", "detector",
                                           "--no-q10-validator", "--no-3d-view"]))
        cfg = replace(cfg, centroid_radius_mm=1.6, pca_radius_mm=4., min_neighbors=2,
                      bilateral_radius_mm=1.6, nms_radius_mm=1.5, fusion_score_threshold=0.)
        with redirect_stdout(StringIO()), patch(
                "corner_detector.edge_suppression._bilateral_neighborhood",
                side_effect=AssertionError("bilateral must be skipped")):
            scores = run_dense_heatmap_mode(corner_points(), cfg)
            candidates, _ = ced_candidates(corner_points(), cfg)
        self.assertEqual(scores["bilateral_mode"], "off")
        np.testing.assert_array_equal(scores["bilateral_ced_score"], scores["ced_score"])
        self.assertGreater(candidates["candidate_count"], 0)
        np.testing.assert_array_equal(candidates["candidate_indices"],
                                      scores["candidate_extraction"]["candidate_indices"])

    def test_radius_migration_preserves_numerical_scores_and_nms(self):
        points = 10. * corner_points()
        expected_fusion = [0.029417604493211368, 0.025057179024822452, 0.03195901633004414,
                           0.054093700905800304, 0.025057179024822452, 0.03195901633004414,
                           0.054093700905800304, 0.010002296525230417]
        for voxel in (5., 10., 20.):
            for scale in (.5, 1., 2.):
                with self.subTest(voxel=voxel, scale=scale):
                    cfg, _ = configure(parse_args([
                        "--source", "realsense", "--mode", "detector", "--no-3d-view",
                        "--voxel-3d-mm", str(voxel), "--min-neighbors", "3", "--fusion-score-threshold", "0",
                        "--centroid-radius-mm", str(80. * scale), "--pca-radius-factor", "1.5",
                        "--bilateral-radius-factor", "1", "--bilateral-sigma-spatial-factor", "0.5",
                        "--nms-radius-factor", "0.375",
                    ]))
                    with redirect_stdout(StringIO()):
                        result = run_dense_heatmap_mode(points * scale, cfg)
                    np.testing.assert_allclose(result["pca_ratio_raw"], 0.20724107952479612, rtol=1e-12)
                    np.testing.assert_allclose(result["fusion_score"], expected_fusion, rtol=1e-12)
                    np.testing.assert_array_equal(result["candidate_extraction"]["candidate_indices"], [3, 6])

    def test_only_cli_is_detector_specific_and_skips_validator_config(self):
        args = parse_args(["--source", "realsense", "--mode", "detector", "--only"])
        cfg, _ = configure(args)
        self.assertTrue(cfg.only)
        self.assertIsNone(cfg.validator)
        default, _ = configure(parse_args(["--source", "realsense", "--mode", "detector"]))
        self.assertFalse(default.only)
        self.assertIsNone(default.validator)
        self.assertIsNone(DetectorConfig().validator)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--source", "realsense", "--mode", "fine", "--only"])

    def test_default_detector_preserves_nms_and_skips_validation(self):
        cfg, _ = configure(parse_args([
            "--source", "realsense", "--mode", "detector", "--no-3d-view",
        ]))
        cfg = replace(cfg, centroid_radius_mm=1.6, pca_radius_mm=4.,
                      nms_radius_mm=1.5, min_neighbors=1, bilateral_radius_mm=1.6,
                      fusion_threshold_mode="fixed", fusion_score_threshold=0.)
        with patch("corner_detector.dense_heatmap.run_validator_candidates") as validate, \
                patch("corner_detector.dense_heatmap.show_detector_validator_results") as final, \
                redirect_stdout(StringIO()):
            result = run_dense_heatmap_mode(corner_points(), cfg)
            only = run_dense_heatmap_mode(corner_points(), replace(cfg, only=True))
        validate.assert_not_called()
        final.assert_not_called()
        self.assertNotIn("validator_results", result)
        self.assertGreater(result["candidate_extraction"]["candidate_count"], 0)
        for key, value in only.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(result[key], value)
        np.testing.assert_array_equal(result["candidate_extraction"]["candidate_indices"],
                                      only["candidate_extraction"]["candidate_indices"])

    def test_only_preserves_scores_and_nms_and_never_runs_validator(self):
        points = corner_points()
        cfg = DetectorConfig(centroid_radius_mm=1.6, pca_radius_mm=4., nms_radius_mm=1.5,
                             min_neighbors=1, bilateral_radius_mm=1.6,
                             fusion_score_threshold=0., show_3d=False, validator=None)
        with redirect_stdout(StringIO()):
            baseline = run_dense_heatmap_mode(points, cfg)
        only_cfg = replace(cfg, only=True, show_3d=True, validator=object())
        with patch("corner_detector.dense_heatmap.run_validator_candidates") as validate, \
                patch("corner_detector.dense_heatmap.show_dense_heatmaps") as heatmap, \
                patch("corner_detector.dense_heatmap.show_detector_validator_results") as final, \
                redirect_stdout(StringIO()):
            result = run_dense_heatmap_mode(points, only_cfg)
        validate.assert_not_called()
        final.assert_not_called()
        heatmap.assert_called_once()
        self.assertTrue(heatmap.call_args.args[2].only)
        self.assertNotIn("validator_results", result)
        for key, value in baseline.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(result[key], value)
        np.testing.assert_array_equal(result["candidate_extraction"]["candidate_indices"],
                                      baseline["candidate_extraction"]["candidate_indices"])

    def test_only_opens_one_nms_heatmap_with_markers_and_no_rgb_window(self):
        points = corner_points()
        cfg = DetectorConfig(centroid_radius_mm=1.6, pca_radius_mm=4., nms_radius_mm=1.5,
                             min_neighbors=1, bilateral_radius_mm=1.6,
                             fusion_score_threshold=0., show_3d=False, validator=None)
        with redirect_stdout(StringIO()):
            scores = run_dense_heatmap_mode(points, cfg)
        fake = MagicMock()
        view = fake.visualization.VisualizerWithKeyCallback.return_value
        view.create_window.return_value = True
        view.poll_events.return_value = False
        fake.geometry.PointCloud.side_effect = Mock
        fake.utility.Vector3dVector.side_effect = lambda value: np.array(value, copy=True)
        with patch.object(view3d, "require_open3d", return_value=fake), \
                patch.object(view3d, "synchronize_dense_camera_views", return_value=(None, None)):
            manager.show_dense_heatmaps(points, scores, replace(cfg, show_3d=True, only=True),
                                        rgb_points_mm=points, rgb_colors_rgb=np.zeros_like(points))
        fake.visualization.VisualizerWithKeyCallback.assert_called_once()
        fake.visualization.Visualizer.assert_not_called()
        view.create_window.assert_called_once()
        self.assertIn("NMS Corner Candidates", view.create_window.call_args.kwargs["window_name"])
        cloud = view.add_geometry.call_args_list[0].args[0]
        np.testing.assert_array_equal(cloud.colors,
            view3d.fusion_rejection_primitives(points, scores)["point_colors"])
        self.assertGreater(scores["candidate_extraction"]["candidate_count"], 0)
        self.assertEqual(view.add_geometry.call_count, 2)
        view.register_key_callback.assert_called_once()
        view.destroy_window.assert_called_once()


class DenseHeatmapVisualizationTest(unittest.TestCase):
    def test_bilateral_panels_follow_mode_and_share_final_fusion_color_range(self):
        for mode, key in (("ced", "bilateral_ced_score"), ("fusion", "fusion_score_raw"),
                          ("off", "bilateral_ced_score")):
            panels = dense_heatmap_panels(mode)
            self.assertEqual(len(panels), 6)
            self.assertEqual(panels[2][1], key)
            self.assertEqual(panels[4][1], "fusion_score")
            self.assertEqual(panels[5][1], "fusion_score")
        scores = dense_ced_weighted_pca_scores(corner_points(), 1.6, 4., 2, bilateral_mode="fusion")
        saved = {key: value.copy() for key, value in scores.items() if isinstance(value, np.ndarray)}
        limits = dense_heatmap_vmax(scores)
        self.assertEqual(limits["fusion_score_raw"], limits["fusion_score"])
        for key, value in saved.items():
            np.testing.assert_array_equal(scores[key], value)

    def test_whole_cloud_visualization_stops_at_nms_before_final_geometry(self):
        panels = dense_heatmap_panels()

        self.assertEqual(len(panels), 6)
        self.assertEqual(
            [panel[0] for panel in panels],
            [
                "Original Point Cloud",
                "Point-wise CED",
                "Bilateral CED",
                "PCA Ratio Used for Fusion",
                "Final Fusion Score",
                "NMS Corner Candidates",
            ],
        )
        self.assertEqual(
            panels[3][1:3], ("pca_ratio_saturated", "pca_valid_mask"),
        )
        self.assertEqual(panels[4][1:3], ("fusion_score", "fusion_valid_mask"))

    def test_pca_colors_distinguish_uncomputed_invalid_and_real_zero(self):
        record = {
            "pca_ratio_saturated": np.array([0.0, 0.0, 0.0, 0.4]),
            "pca_valid_mask": np.array([False, False, True, True]),
            "pca_query_mask": np.array([False, True, True, True]),
            "invalid_support_mask": np.array([False, True, False, False]),
        }
        colors = pca_ratio_heatmap_colors(record, vmax=1.0)

        np.testing.assert_allclose(colors[0], [0.16, 0.16, 0.18])
        np.testing.assert_allclose(colors[1], [0.58, 0.58, 0.62])
        self.assertFalse(np.allclose(colors[2], colors[0]))

    def test_camera_rotation_sync_uses_whichever_window_changed(self):
        class Parameters:
            def __init__(self, extrinsic):
                self.extrinsic = np.asarray(extrinsic, dtype=float).copy()
                self.intrinsic = SimpleNamespace(intrinsic_matrix=np.eye(3))

        class Control:
            def __init__(self):
                self.extrinsic = np.eye(4)

            def convert_to_pinhole_camera_parameters(self):
                return Parameters(self.extrinsic)

            def convert_from_pinhole_camera_parameters(self, params, allow_arbitrary):
                self.extrinsic = np.asarray(params.extrinsic).copy()

        class Window:
            def __init__(self):
                self.control = Control()

            def get_view_control(self):
                return self.control

        windows = [Window() for _ in range(7)]
        active = [True] * len(windows)
        signatures, master = synchronize_dense_camera_views(windows, active, None)
        self.assertIsNone(master)
        windows[6].control.extrinsic[1, 3] = -7.0
        _signatures, master = synchronize_dense_camera_views(
            windows, active, signatures,
        )
        self.assertEqual(master, 6)
        for window in windows:
            self.assertEqual(window.control.extrinsic[1, 3], -7.0)

    def test_candidate_primitives_preserve_current_scores(self):
        points = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
        scores = {
            "ced_score": np.array([0.1, 0.6]),
            "bilateral_ced_score": np.array([0.15, 0.55]),
            "pca_ratio_raw": np.array([0.2, 0.5]),
            "pca_ratio_saturated": np.array([0.18, 0.4]),
            "pca_ratio_saturation_tau": 0.5,
            "fusion_score": np.array([0.03, 0.275]),
            "candidate_extraction": {
                "candidate_indices": np.array([1]),
                "nms_radius_mm": 4.0,
                "candidate_count": 1,
            },
        }

        debug = corner_candidate_primitives(points, scores)

        np.testing.assert_array_equal(debug["indices"], [1])
        np.testing.assert_array_equal(debug["positions_mm"], points[[1]])
        np.testing.assert_array_equal(debug["ced_score"], [0.6])
        np.testing.assert_array_equal(debug["bilateral_ced_score"], [0.55])
        np.testing.assert_array_equal(debug["pca_ratio_raw"], [0.5])
        np.testing.assert_array_equal(debug["pca_ratio_saturated"], [0.4])
        self.assertEqual(debug["pca_ratio_saturation_tau"], 0.5)
        np.testing.assert_array_equal(debug["fusion_score"], [0.275])

    def test_q99_ranges_exclude_invalid_and_include_fusion(self):
        scores = {
            "ced_score": np.array([0.0, 0.1, 0.2, 0.3, 1.0]),
            "ced_valid_mask": np.array([True, True, True, True, False]),
            "bilateral_ced_score": np.array([0.01, 0.09, 0.19, 0.28, 0.0]),
            "bilateral_valid_mask": np.array([True, True, True, True, False]),
            "pca_ratio_saturated": np.array([0.0, 0.18, 0.31, 0.46, 0.5]),
            "pca_valid_mask": np.array([True, True, True, True, False]),
            "fusion_score": np.array([0.0, 0.02, 0.08, 0.24, 0.0]),
            "fusion_valid_mask": np.array([True, True, True, True, False]),
        }
        originals = {key: value.copy() for key, value in scores.items()}
        limits = dense_heatmap_vmax(scores)

        self.assertAlmostEqual(
            limits["ced_score"], np.quantile([0.0, 0.1, 0.2, 0.3], 0.99),
        )
        self.assertAlmostEqual(
            limits["pca_ratio_saturated"],
            np.quantile([0.0, 0.18, 0.31, 0.46], 0.99),
        )
        self.assertAlmostEqual(
            limits["fusion_score"], np.quantile([0.0, 0.02, 0.08, 0.24], 0.99),
        )
        for key, original in originals.items():
            np.testing.assert_array_equal(scores[key], original)

    def test_dense_color_mapping_and_invalid_gray(self):
        colors = dense_heatmap_colors(
            [0.25, 0.5, 1.0], [True, False, True], vmax=0.5,
        )
        np.testing.assert_allclose(colors[1], [0.28, 0.28, 0.30])
        np.testing.assert_allclose(colors[0], dense_heatmap_colors(
            [0.25], [True], vmax=0.5,
        )[0])

    def test_manager_forwards_same_cloud_and_rgb_data(self):
        points = np.arange(12, dtype=float).reshape(4, 3)
        rgb_points = points + 0.25
        rgb_colors = np.full((4, 3), 128, dtype=np.uint8)
        scores = {"fusion_valid_mask": np.ones(4, dtype=bool)}
        config = SimpleNamespace(show_3d=True, voxel_3d_mm=2.0)
        with patch.object(manager, "show_dense_score_heatmaps") as show:
            manager.show_dense_heatmaps(
                points,
                scores,
                config,
                rgb_points_mm=rgb_points,
                rgb_colors_rgb=rgb_colors,
            )

        forwarded_points, forwarded_scores = show.call_args.args
        np.testing.assert_array_equal(forwarded_points, points)
        self.assertIs(forwarded_scores, scores)
        np.testing.assert_array_equal(show.call_args.kwargs["rgb_points_mm"], rgb_points)
        np.testing.assert_array_equal(show.call_args.kwargs["rgb_colors_rgb"], rgb_colors)


if __name__ == "__main__":
    unittest.main()
