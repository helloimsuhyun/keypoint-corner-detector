"""Contrast-regularized Normal-PCA lambda1 minima and independent MAX NMS."""
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_corner_lab_roi_thresholds import comparison
from test_corner_lab_ui import lab, make_window


class LocalMinimumFusionTest(unittest.TestCase):
    def test_given_flat_and_strong_examples_have_expected_contrast_and_ced_product(self):
        xyz = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
        ced = np.array([.5, .25, .1])
        weak, weak_score = lab.compute_local_minimum_fusion(xyz, [.010, .011, .012], ced, 1., .01)
        strong, strong_score = lab.compute_local_minimum_fusion(xyz, [.1, .2, .3], ced, 1., .01)
        np.testing.assert_allclose(weak, [1/6, 1/12, 0.], atol=1e-15)
        np.testing.assert_allclose(strong, [20/21, 10/21, 0.])
        np.testing.assert_allclose(weak_score, ced*weak)
        np.testing.assert_allclose(strong_score, ced*strong)
        self.assertGreater(strong[0], weak[0])

    def test_flat_response_is_zero_without_a_minimum_variance_gate_and_nan_is_preserved(self):
        xyz = np.column_stack((np.arange(4.), np.zeros((4, 2))))
        flat, score = lab.compute_local_minimum_fusion(xyz, [0., 0., 0., np.nan], np.ones(4), 5.)
        np.testing.assert_array_equal(flat[:3], [0., 0., 0.])
        np.testing.assert_array_equal(score[:3], [0., 0., 0.])
        self.assertTrue(np.isnan(score[3]))
        minimum, score = lab.compute_local_minimum_fusion(xyz, [0., 1., 2., np.nan], [1., np.nan, 1., 1.], 5.)
        self.assertGreater(minimum[0], 0.)
        self.assertTrue(np.isnan(score[1]))
        self.assertEqual(score[2], 0.)

    def test_radius_is_inclusive_and_r_squared_preserves_geometric_scaling(self):
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
        eigenvalues = np.array([1., 2., 10.])
        a = lab.compute_local_minimum_fusion(xyz, eigenvalues, np.ones(3), 1., .1)
        self.assertAlmostEqual(a[0][0], 1/1.1)
        self.assertEqual(a[0][2], 0.)
        b = lab.compute_local_minimum_fusion(xyz*10, eigenvalues*100, np.ones(3), 10., .1)
        np.testing.assert_allclose(a, b)
        lower = lab.compute_local_minimum_fusion(xyz, eigenvalues, np.ones(3), 1., 1.)
        self.assertLess(lower[0][0], a[0][0])

    def test_new_nms_finds_positive_minimum_and_retains_every_existing_field(self):
        result = replace(comparison(), normal_lambda1_mm2=np.array([3., 3., 3., 1., 3., 3., 3.]),
                         ced_scores=np.ones(7), pca_radius_mm=3., nms_radius_mm=1.1)
        updated = lab.with_local_minimum_fusion(result, .1)
        np.testing.assert_array_equal(updated.ced_local_min_maxima, [3])
        self.assertAlmostEqual(updated.ced_local_min_scores[3], 2/2.9)
        for name, value in vars(result).items():
            if name not in ('local_minimum_scores', 'ced_local_min_scores', 'ced_local_min_maxima', 'contrast_tau',
                            'median_minimum_scores', 'ced_median_min_scores', 'ced_median_min_maxima', 'median_contrast_tau'):
                self.assertIs(getattr(updated, name), value)
        flat = lab.with_local_minimum_fusion(replace(result, normal_lambda1_mm2=np.ones(7)))
        self.assertEqual(len(flat.ced_local_min_maxima), 0)
        local = lab.with_local_minimum_fusion(replace(result, nms_radius_mm=2.1), .1, local_mode=True)
        np.testing.assert_array_equal(local.ced_local_min_maxima, [3])

    def test_invalid_inputs_and_cancellation_do_not_modify_source_arrays(self):
        xyz = np.zeros((3, 3))
        eigenvalues = np.array([0., 1., 2.])
        for tau in (0., -1., np.nan, np.inf):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                lab.compute_local_minimum_fusion(xyz, eigenvalues, np.ones(3), 1., tau)
        with self.assertRaises(ValueError):
            lab.compute_local_minimum_fusion(xyz, eigenvalues[:2], np.ones(3), 1.)
        self.assertIsNone(lab.compute_local_minimum_fusion(
            xyz, eigenvalues, np.ones(3), 1., is_cancelled=lambda: True))
        np.testing.assert_array_equal(eigenvalues, [0., 1., 2.])

    def test_local_and_full_scene_entry_points_attach_the_requested_tau_and_field(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5,
                      ced_cutoff=.05, contrast_tau=.02)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)
        for result in (full, local):
            self.assertEqual(result.contrast_tau, .02)
            minimum, score = lab.compute_local_minimum_fusion(
                result.positions_mm, result.normal_lambda1_mm2, result.ced_scores, 15., .02)
            np.testing.assert_allclose(result.local_minimum_scores, minimum, equal_nan=True)
            np.testing.assert_allclose(result.ced_local_min_scores, score, equal_nan=True)
            self.assertTrue(np.all(result.ced_local_min_scores[result.ced_local_min_maxima] > 0))

    def test_gui_tau_is_in_cache_key_and_invalid_tau_never_starts_a_worker(self):
        window = make_window()
        window.xyz = np.zeros((12, 3))
        key = window._compare_key()
        window.compare_contrast_tau_edit.text_value = '.1'
        self.assertNotEqual(window._compare_key(), key)
        window.compare_contrast_tau_edit.text_value = 'nan'
        window._error = Mock()
        with patch.object(lab.threading, 'Thread') as worker:
            window._toggle_compare_extrema()
        worker.assert_not_called()
        window._error.assert_called_once()

    def test_new_heatmap_uses_a_fixed_scale_even_when_all_scores_are_low_or_invalid(self):
        _, limits = lab.eigen_heatmap_colors(np.array([.0001, .0002, .0003]), limits=(0., 1.))
        self.assertEqual(limits, (0., 1.))
        colors, _ = lab.eigen_heatmap_colors(np.array([np.nan]), limits=(0., 1.))
        np.testing.assert_array_equal(colors, [[95, 103, 115]])
