"""Contrast-regularized Normal-PCA lambda1 minima and independent MAX NMS."""
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
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
                            'median_minimum_scores', 'ced_median_min_scores', 'ced_median_min_maxima', 'median_contrast_tau',
                            'local_minimum_radius_mm', 'median_minimum_radius_mm'):
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


class MinimumNeighborhoodRadiusTest(unittest.TestCase):
    def test_neighborhood_radius_changes_statistics_without_changing_pca_regularizer(self):
        xyz = np.column_stack(([0., 1., 3.], np.zeros((3, 2))))
        eigenvalues = np.array([1., 2., 10.])
        for statistic, expected in (('max', (5/7, 45/47)), ('median', (5/9, 5/7))):
            for radius, value in zip((1., 3.), expected):
                field, _ = lab.compute_local_minimum_fusion(xyz, eigenvalues, np.ones(3), radius, .1,
                    statistic=statistic, regularization_radius_mm=2.)
                self.assertAlmostEqual(field[0], value)
                scaled, _ = lab.compute_local_minimum_fusion(xyz*10, eigenvalues*100, np.ones(3), radius*10, .1,
                    statistic=statistic, regularization_radius_mm=20.)
                np.testing.assert_allclose(scaled, field)
        for radius in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                lab.compute_local_minimum_fusion(xyz, eigenvalues, np.ones(3), 1., regularization_radius_mm=radius)

    def test_minmax_and_median_radii_change_only_their_own_fields(self):
        result = replace(comparison(), normal_lambda1_mm2=np.array([4., 1., 5., 2., 7., 3., 8.]),
                         ced_scores=np.ones(7), pca_radius_mm=3., nms_radius_mm=1.1)
        base = lab.with_local_minimum_fusion(result)
        minmax = lab.with_local_minimum_fusion(result, local_minimum_radius_mm=1.)
        median = lab.with_local_minimum_fusion(result, median_minimum_radius_mm=1.)
        np.testing.assert_array_equal(minmax.median_minimum_scores, base.median_minimum_scores)
        np.testing.assert_array_equal(minmax.ced_median_min_maxima, base.ced_median_min_maxima)
        np.testing.assert_array_equal(median.local_minimum_scores, base.local_minimum_scores)
        np.testing.assert_array_equal(median.ced_local_min_maxima, base.ced_local_min_maxima)
        self.assertFalse(np.array_equal(minmax.local_minimum_scores, base.local_minimum_scores))
        self.assertFalse(np.array_equal(median.median_minimum_scores, base.median_minimum_scores))
        self.assertEqual((minmax.local_minimum_radius_mm, minmax.median_minimum_radius_mm), (1., 3.))
        self.assertEqual((median.local_minimum_radius_mm, median.median_minimum_radius_mm), (3., 1.))
        self.assertIs(minmax.normal_lambda1_mm2, result.normal_lambda1_mm2)
        self.assertIs(median.fusion_maxima, result.fusion_maxima)
        for key in ('local_minimum_radius_mm', 'median_minimum_radius_mm'):
            for radius in (0., -1., np.nan, np.inf):
                with self.subTest(key=key, radius=radius), self.assertRaises(ValueError):
                    lab.with_local_minimum_fusion(result, **{key: radius})

    def test_local_and_full_entries_propagate_both_radii_and_defaults_follow_pca(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5, ced_cutoff=.05)
        for compute in (lambda **kw: lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common, **kw),
                        lambda **kw: lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common, **kw)):
            default = compute()
            self.assertEqual((default.local_minimum_radius_mm, default.median_minimum_radius_mm), (15., 15.))
            custom = compute(local_minimum_radius_mm=4., median_minimum_radius_mm=9.)
            for statistic, radius, name in (('max', 4., 'local_minimum_scores'), ('median', 9., 'median_minimum_scores')):
                expected, _ = lab.compute_local_minimum_fusion(custom.positions_mm, custom.normal_lambda1_mm2,
                    custom.ced_scores, radius, statistic=statistic, regularization_radius_mm=15.)
                np.testing.assert_allclose(getattr(custom, name), expected, equal_nan=True)
            for name in ('positions_mm', 'normal_lambda1_mm2', 'fusion_scores', 'fusion_maxima', 'harris_scores', 'harris_maxima'):
                np.testing.assert_array_equal(getattr(custom, name), getattr(default, name))

    def test_gui_radius_keys_validation_and_display_use_independent_resolved_values(self):
        window = make_window()
        window.xyz = np.zeros((12, 3))
        window.analysis_xyz = window.xyz.copy()
        window.active_index, window.corners = 0, [dict(id='corner_001', point_index=0)]
        window._closest_analysis_point = Mock(return_value=(0, np.zeros(3)))
        for scope in (0, 1):
            window.compare_scope_index = scope
            original_key = window._compare_key()
            for edit in (window.compare_minimum_radius_edit, window.compare_median_radius_edit):
                self.assertEqual(edit.text_value, '')
                edit.text_value = '7'
                self.assertNotEqual(window._compare_key(), original_key)
                edit.text_value = ''
                self.assertEqual(window._compare_key(), original_key)
        window._error = Mock()
        for edit in (window.compare_minimum_radius_edit, window.compare_median_radius_edit):
            for text in ('0', '-1', 'nan', 'inf', 'bad'):
                edit.text_value = text
                with patch.object(lab.threading, 'Thread') as worker:
                    window._toggle_compare_extrema()
                worker.assert_not_called()
                self.assertFalse(window.compare_running)
            edit.text_value = ''
        self.assertEqual(window._error.call_count, 10)
        window.compare_result = replace(comparison(), local_minimum_radius_mm=7., median_minimum_radius_mm=11.)
        window.compare_cache_key = window._compare_key()
        window.compare_show = True
        window._overlay = Mock()
        window._compare_display_changed('', 5)
        self.assertIn('local R=7 mm', window.compare_summary.text)
        window._compare_display_changed('', 7)
        self.assertIn('local R=11 mm', window.compare_summary.text)

    def test_local_and_scene_workers_snapshot_radius_inputs_and_discard_changed_settings(self):
        for scope, name in ((0, 'compute_local_fusion_pca_comparison'), (1, 'compute_full_scene_fusion_pca_comparison')):
            window = make_window()
            window.compare_scope_index = scope
            window.xyz = np.zeros((12, 3))
            window.analysis_xyz = window.xyz.copy()
            window.active_index, window.corners = 0, [dict(id='corner_001', point_index=0)]
            window._closest_analysis_point = Mock(return_value=(0, np.zeros(3)))
            window.compare_minimum_radius_edit.text_value = '5'
            window.compare_median_radius_edit.text_value = '9'
            window.app.post_to_main_thread.side_effect = lambda _win, callback: callback()
            with (patch.object(lab.threading, 'Thread') as thread,
                  patch.object(lab, name, return_value=comparison()) as compute,
                  patch.object(lab, 'voxel_downsample_analysis', return_value=(window.xyz, np.zeros((12, 3)))),
                  redirect_stdout(StringIO())):
                window._toggle_compare_extrema()
                window.compare_minimum_radius_edit.text_value = '7'
                window.compare_median_radius_edit.text_value = '11'
                thread.call_args.kwargs['target']()
            self.assertEqual(compute.call_args.kwargs['local_minimum_radius_mm'], 5.)
            self.assertEqual(compute.call_args.kwargs['median_minimum_radius_mm'], 9.)
            self.assertIsNone(window.compare_result)
            self.assertIn('Settings changed', window.compare_summary.text)
