"""Median contrast formula, outlier behavior and its separate lab views/NMS."""
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_corner_lab_roi_thresholds import comparison
import test_corner_lab_ui as ui
from test_corner_lab_ui import lab, make_window


class MedianMinimumTest(unittest.TestCase):
    def test_median_rejects_above_median_and_is_unchanged_by_one_large_outlier(self):
        xyz = np.column_stack((np.arange(5.)*.1, np.zeros((5, 2))))
        eigenvalues = np.array([10., 11., 11., 12., 100.])
        ced = np.array([.5, 1., 1., 1., 1.])
        minimum, score = lab.compute_local_minimum_fusion(
            xyz, eigenvalues, ced, 1., .005, statistic='median')
        np.testing.assert_allclose(minimum, [1/1.005, 0., 0., 0., 0.])
        np.testing.assert_allclose(score, minimum*ced)
        old, _ = lab.compute_local_minimum_fusion(xyz, eigenvalues, ced, 1., .005)
        self.assertAlmostEqual(old[1], 89/90.005)
        changed = eigenvalues.copy()
        changed[-1] = 1e9
        outlier, _ = lab.compute_local_minimum_fusion(xyz, changed, ced, 1., .005, statistic='median')
        np.testing.assert_array_equal(outlier, minimum)
        np.testing.assert_array_equal(eigenvalues, [10., 11., 11., 12., 100.])

    def test_even_finite_median_flat_nan_and_scale_behavior(self):
        xyz = np.column_stack((np.arange(5.)*.1, np.zeros((5, 2))))
        eigenvalues = np.array([0., 1., 2., 3., np.nan])
        ced = np.array([1., 1., np.nan, 1., 1.])
        a = lab.compute_local_minimum_fusion(xyz, eigenvalues, ced, 1., .5, statistic='median')
        np.testing.assert_allclose(a[0], [.75, .25, 0., 0., np.nan], equal_nan=True)
        np.testing.assert_allclose(a[1], [.75, .25, np.nan, 0., np.nan], equal_nan=True)
        b = lab.compute_local_minimum_fusion(xyz*10, eigenvalues*100, ced, 10., .5, statistic='median')
        np.testing.assert_allclose(a, b, equal_nan=True)
        flat, score = lab.compute_local_minimum_fusion(xyz, np.zeros(5), np.ones(5), 1., .005, statistic='median')
        np.testing.assert_array_equal(flat, np.zeros(5))
        np.testing.assert_array_equal(score, np.zeros(5))

    def test_median_nms_and_independent_tau_keep_minmax_unchanged(self):
        result = replace(comparison(), normal_lambda1_mm2=np.array([3., 3., 3., 1., 3., 3., 3.]),
                         ced_scores=np.ones(7), pca_radius_mm=3., nms_radius_mm=1.1)
        a = lab.with_local_minimum_fusion(result)
        self.assertEqual(a.contrast_tau, .005)
        self.assertEqual(a.median_contrast_tau, .005)
        self.assertAlmostEqual(a.ced_local_min_scores[3], 2/2.045)
        self.assertAlmostEqual(a.ced_median_min_scores[3], 2/2.045)
        np.testing.assert_array_equal(a.ced_median_min_maxima, [3])
        b = lab.with_local_minimum_fusion(result, median_contrast_tau=.2)
        np.testing.assert_array_equal(a.local_minimum_scores, b.local_minimum_scores)
        np.testing.assert_array_equal(a.ced_local_min_scores, b.ced_local_min_scores)
        np.testing.assert_array_equal(a.ced_local_min_maxima, b.ced_local_min_maxima)
        self.assertLess(b.median_minimum_scores[3], a.median_minimum_scores[3])
        c = lab.with_local_minimum_fusion(result, contrast_tau=.5)
        np.testing.assert_array_equal(a.median_minimum_scores, c.median_minimum_scores)
        local = lab.with_local_minimum_fusion(replace(result, nms_radius_mm=2.1), local_mode=True)
        np.testing.assert_array_equal(local.ced_median_min_maxima, [3])
        flat = lab.with_local_minimum_fusion(replace(result, normal_lambda1_mm2=np.ones(7)))
        self.assertEqual(len(flat.ced_median_min_maxima), 0)

    def test_full_and_local_entry_points_use_separate_median_tau(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5,
                      ced_cutoff=.05, contrast_tau=.02, median_contrast_tau=.007)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)
        for result in (full, local):
            self.assertEqual(result.contrast_tau, .02)
            self.assertEqual(result.median_contrast_tau, .007)
            fields = lab.compute_local_minimum_fusion(result.positions_mm, result.normal_lambda1_mm2,
                result.ced_scores, result.pca_radius_mm, .007, statistic='median')
            np.testing.assert_allclose(result.median_minimum_scores, fields[0], equal_nan=True)
            np.testing.assert_allclose(result.ced_median_min_scores, fields[1], equal_nan=True)

    def test_median_views_auto_contrast_and_own_inclusive_threshold(self):
        window, result, rendered = ui.CornerLabUiTest().comparison_window()
        window._compare_display_changed('', 7)
        self.assertTrue(window.compare_nms_check.enabled)
        self.assertTrue(window.compare_auto_contrast_check.enabled)
        self.assertIn('contrast tau=0.005', window.compare_summary.text)
        window._compare_display_changed('', 8)
        self.assertFalse(window.compare_nms_check.enabled)
        self.assertNotIn('median_minimum_scores', window.compare_threshold_edits)
        window.compare_auto_contrast_check.checked = True
        window._compare_auto_contrast_changed(True)
        colors, _ = lab.eigen_heatmap_colors(result.median_minimum_scores)
        np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), colors/255.)
        raw = comparison()
        kept = lab.threshold_extrema_comparison(raw, {'ced_median_min_scores': .4})
        np.testing.assert_array_equal(kept.ced_median_min_maxima, [0])
        removed = lab.threshold_extrema_comparison(raw, {'ced_median_min_scores': .4001})
        self.assertEqual(len(removed.ced_median_min_maxima), 0)
        self.assertIs(removed.ced_local_min_maxima, raw.ced_local_min_maxima)

    def test_median_validation_cancellation_and_gui_cache_key(self):
        xyz = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            lab.compute_local_minimum_fusion(xyz, np.ones(3), np.ones(3), 1., statistic='unknown')
        self.assertIsNone(lab.compute_local_minimum_fusion(
            xyz, np.ones(3), np.ones(3), 1., .005, lambda: True, statistic='median'))
        window = make_window()
        self.assertEqual(window.compare_contrast_tau_edit.text_value, '0.005')
        self.assertEqual(window.compare_median_tau_edit.text_value, '0.005')
        window.xyz = xyz
        key = window._compare_key()
        window.compare_median_tau_edit.text_value = '.02'
        self.assertNotEqual(window._compare_key(), key)
        window.compare_median_tau_edit.text_value = '0'
        window._error = Mock()
        with patch.object(lab.threading, 'Thread') as worker:
            window._toggle_compare_extrema()
        worker.assert_not_called()
        window._error.assert_called_once()
