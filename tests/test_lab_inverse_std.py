"""CED divided by unweighted PCA standard deviation: formula, cache and display."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.visualization import manager
from test_corner_lab_roi_thresholds import comparison
import test_corner_lab_ui as ui
from test_corner_lab_ui import lab, make_window


class InverseStdScoreTest(unittest.TestCase):
    def test_equation_uses_epsilon_inside_sqrt_and_preserves_zero_and_unbounded_scores(self):
        ced, eigenvalues = np.array([.5, .5, 0., 1.]), np.array([4., 9., 0., 0.])
        expected = np.array([.5/np.sqrt(5), .5/np.sqrt(10), 0., 1.])
        np.testing.assert_allclose(lab.compute_inverse_std_fusion(ced, eigenvalues, 1.), expected)
        self.assertEqual(lab.compute_inverse_std_fusion([1.], [0.], .01)[0], 10.)
        np.testing.assert_allclose(lab.compute_inverse_std_fusion(ced, eigenvalues*100, 100.), expected/10.)

    def test_invalid_samples_remain_nan_and_invalid_epsilon_or_shapes_fail(self):
        ced = np.array([.5, np.nan, 1., -.1, 1., 1., 0.])
        eigenvalues = np.array([4., 1., np.inf, 1., -1., np.nan, 0.])
        saved = ced.copy(), eigenvalues.copy()
        scores = lab.compute_inverse_std_fusion(ced, eigenvalues, 1.)
        self.assertAlmostEqual(scores[0], .5/np.sqrt(5))
        self.assertTrue(np.isnan(scores[1:6]).all())
        self.assertEqual(scores[6], 0.)
        np.testing.assert_array_equal(ced, saved[0])
        np.testing.assert_array_equal(eigenvalues, saved[1])
        for epsilon in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                lab.compute_inverse_std_fusion(ced, eigenvalues, epsilon)
        for invalid in (eigenvalues[:1], np.ones((7, 1))):
            with self.assertRaises(ValueError):
                lab.compute_inverse_std_fusion(ced, invalid)

    def test_attachment_has_positive_max_nms_without_changing_previous_fields(self):
        result = comparison()
        scores = np.array([0., 0., 0., .03, 0., 0., 0.])
        with patch.object(lab, 'compute_inverse_std_fusion', return_value=scores):
            for local_mode in (False, True):
                updated = lab.with_inverse_std_fusion(result, .25, local_mode=local_mode)
                np.testing.assert_array_equal(updated.ced_inverse_std_maxima, [3])
                self.assertEqual(updated.inverse_std_epsilon_mm2, .25)
                for name, value in vars(result).items():
                    if name not in ('ced_inverse_std_scores', 'ced_inverse_std_maxima', 'inverse_std_epsilon_mm2',
                                    'inverse_std_raw_scores', 'inverse_std_activation', 'inverse_std_neighbor_medians'):
                        self.assertIs(getattr(updated, name), value)
        with patch.object(lab, 'compute_inverse_std_fusion', return_value=np.zeros(7)):
            self.assertEqual(len(lab.with_inverse_std_fusion(result).ced_inverse_std_maxima), 0)
        self.assertIsNone(lab.with_inverse_std_fusion(result, is_cancelled=lambda: True))

    def test_local_and_full_entries_use_unweighted_lambda1_even_when_weighted_pca_is_invalid(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5,
                      ced_cutoff=100., inverse_std_epsilon_mm2=2.)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)
        for result in (full, local):
            self.assertTrue(np.isnan(result.ced_weighted_lambda1_mm2).all())
            expected = result.ced_scores/np.sqrt(result.normal_lambda1_mm2+2.)
            np.testing.assert_allclose(result.ced_inverse_std_scores, expected, equal_nan=True)
            self.assertEqual(result.inverse_std_epsilon_mm2, 2.)
            self.assertGreater(len(result.ced_inverse_std_maxima), 0)
        line = np.column_stack((np.arange(12.), np.zeros((12, 2))))
        independent = lab.compute_full_scene_fusion_pca_comparison(line, pca_radius_mm=100.,
            ced_radius_mm=3., nms_radius_mm=3., min_neighbors=3, ced_cutoff=100.)
        self.assertTrue(np.isnan(independent.fusion_scores).all())
        self.assertTrue(np.isnan(independent.harris_scores).all())
        self.assertTrue(np.isfinite(independent.ced_inverse_std_scores).all())

    def test_all_comparison_scores_use_raw_ced_without_bilateral_smoothing(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5, ced_cutoff=.05)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)
        for result in (full, local):
            expected = []
            for point in result.positions_mm:
                support = xyz[np.linalg.norm(xyz-point, axis=1) <= 10.+1e-8]
                expected.append(np.linalg.norm(support.mean(axis=0)-point)/10.)
            expected = np.asarray(expected)
            np.testing.assert_allclose(result.ced_scores, expected, rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(result.fusion_scores,
                np.clip(expected*result.tau*np.tanh(result.ced_weighted_ratio/result.tau), 0., 1.), equal_nan=True)
            np.testing.assert_allclose(result.ced_local_min_scores, expected*result.local_minimum_scores, equal_nan=True)
            np.testing.assert_allclose(result.ced_median_min_scores, expected*result.median_minimum_scores, equal_nan=True)
            np.testing.assert_allclose(result.ced_inverse_std_scores,
                expected/np.sqrt(result.normal_lambda1_mm2+result.inverse_std_epsilon_mm2), equal_nan=True)
            valid = np.isfinite(result.ced_normal_component) & np.isfinite(result.ced_tangent_component)
            np.testing.assert_allclose(result.ced_normal_component[valid]**2+result.ced_tangent_component[valid]**2,
                                       result.ced_scores[valid]**2, rtol=1e-13, atol=1e-15)
        self.assertEqual(lab.SCORE_VIEWS[1][0], 'Raw CED')

    def window(self, scope):
        window = make_window()
        window.compare_scope_index = scope
        window.xyz = np.zeros((12, 3))
        window.analysis_xyz = window.xyz.copy()
        window.active_index, window.corners = 0, [dict(id='corner_001', point_index=0)]
        window._closest_analysis_point = Mock(return_value=(0, np.zeros(3)))
        return window

    def test_epsilon_is_in_both_cache_keys_and_invalid_inputs_never_start_workers(self):
        for scope in (0, 1):
            window = self.window(scope)
            self.assertEqual(window.compare_inverse_std_epsilon_edit.text_value, '1e-6')
            key = window._compare_key()
            window.compare_inverse_std_epsilon_edit.text_value = '.25'
            self.assertNotEqual(key, window._compare_key())
            window._error = Mock()
            for text in ('', '0', '-1', 'nan', 'inf', 'bad'):
                window.compare_inverse_std_epsilon_edit.text_value = text
                with patch.object(lab.threading, 'Thread') as thread:
                    window._toggle_compare_extrema()
                thread.assert_not_called()
            self.assertEqual(window._error.call_count, 6)

    def test_both_worker_paths_snapshot_epsilon_and_discard_stale_output(self):
        for scope, name in ((0, 'compute_local_fusion_pca_comparison'), (1, 'compute_full_scene_fusion_pca_comparison')):
            window = self.window(scope)
            window.compare_inverse_std_epsilon_edit.text_value = '2'
            window.app.post_to_main_thread.side_effect = lambda _win, callback: callback()
            with (patch.object(lab.threading, 'Thread') as thread,
                  patch.object(lab, name, return_value=comparison()) as compute,
                  patch.object(lab, 'voxel_downsample_analysis', return_value=(window.xyz, np.zeros((12, 3)))),
                  redirect_stdout(StringIO())):
                window._toggle_compare_extrema()
                window.compare_inverse_std_epsilon_edit.text_value = '3'
                thread.call_args.kwargs['target']()
            self.assertEqual(compute.call_args.kwargs['inverse_std_epsilon_mm2'], 2.)
            self.assertIsNone(window.compare_result)
            self.assertIn('Settings changed', window.compare_summary.text)

    def test_cached_heatmap_gate_and_click_plot_use_the_new_score_and_units(self):
        window, _, rendered = ui.CornerLabUiTest().comparison_window()
        window.compare_result = result = comparison()
        saved = copy.deepcopy(vars(result))
        with patch.object(lab, 'compute_inverse_std_fusion', side_effect=AssertionError('recomputed')):
            window._compare_display_changed('', 12)
            self.assertTrue(window.compare_nms_check.enabled)
            expected, _ = lab.eigen_heatmap_colors(result.ced_inverse_std_scores)
            np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), expected/255.)
            np.testing.assert_array_equal(np.asarray(rendered['comparison_fusion_max'].points), result.positions_mm[[2]])
            self.assertIn('1/mm', window.eigen_heatmap_legend.text)
            self.assertIn('unweighted largest', window.compare_summary.text)
            kept = lab.threshold_extrema_comparison(result, {'ced_inverse_std_scores': .05})
            np.testing.assert_array_equal(kept.ced_inverse_std_maxima, [2])
            removed = lab.threshold_extrema_comparison(result, {'ced_inverse_std_scores': .05001})
            self.assertEqual(len(removed.ced_inverse_std_maxima), 0)
            display = dict(closed=False, window=Mock())
            with patch.object(manager, 'show_nms_peak_plot', return_value=display) as show, redirect_stdout(StringIO()):
                self.assertTrue(window._picked_comparison_nms(result.positions_mm[2]))
            self.assertIs(show.call_args.args[1], result.ced_inverse_std_scores)
            self.assertEqual(show.call_args.args[2], 2)
            self.assertEqual(show.call_args.kwargs['units'], ' 1/mm')
            window._close_comparison_peak()
        np.testing.assert_equal(vars(result), saved)
