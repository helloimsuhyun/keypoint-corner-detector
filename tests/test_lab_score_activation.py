"""Selectable inverse-std activations and PCA-neighbor median density evidence."""
import copy
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.visualization import manager
from test_corner_lab_roi_thresholds import comparison
from test_corner_lab_ui import lab, make_window
import test_corner_lab_ui as ui


class ScoreActivationTest(unittest.TestCase):
    def test_density_activation_boosts_sparse_queries_and_reduces_dense_queries(self):
        raw = np.array([.3, .3, .3, 0.])
        counts = np.array([5., 10., 20., 10.])
        medians = np.full(4, 10.)
        activated = lab.apply_inverse_std_activation(raw, 'median_density', neighbor_counts=counts, median_counts=medians)
        expected = 1.-np.exp(-raw*medians/counts)
        np.testing.assert_allclose(activated, expected)
        exponential = lab.apply_inverse_std_activation(raw, 'exponential')
        self.assertGreater(activated[0], exponential[0])
        self.assertEqual(activated[1], exponential[1])
        self.assertLess(activated[2], exponential[2])
        self.assertEqual(activated[3], 0.)
        np.testing.assert_array_equal(lab.apply_inverse_std_activation(raw), raw)
        np.testing.assert_allclose(lab.apply_inverse_std_activation(raw, 'median_density',
            neighbor_counts=counts*100, median_counts=medians*100), activated)
        self.assertAlmostEqual(lab.apply_inverse_std_activation([1e-15], 'exponential')[0], 1e-15, delta=1e-29)
        self.assertEqual(lab.apply_inverse_std_activation([1e300], 'exponential')[0], 1.)

    def test_invalid_density_evidence_stays_undefined_without_a_count_floor(self):
        raw = np.array([.3, .3, .3, np.nan, -.1, np.inf])
        counts, medians = np.array([0., 10., np.nan, 10., 10., 10.]), np.array([10., 0., 10., 10., 10., 10.])
        self.assertTrue(np.isnan(lab.apply_inverse_std_activation(raw, 'median_density',
            neighbor_counts=counts, median_counts=medians)).all())
        with self.assertRaises(ValueError):
            lab.apply_inverse_std_activation(raw, 'unknown')
        with self.assertRaises(ValueError):
            lab.apply_inverse_std_activation(raw, 'median_density', neighbor_counts=[10.], median_counts=medians)

    def test_local_median_excludes_self_includes_boundary_and_resists_one_outlier(self):
        xyz = np.column_stack(([0., 1., 3.], np.zeros((3, 2))))
        np.testing.assert_allclose(lab.compute_neighbor_count_medians(xyz, [1., 10., 100.], 1.),
                                   [10., 1., np.nan], equal_nan=True)
        cluster = np.column_stack((np.arange(5.)*.1, np.zeros((5, 2))))
        counts = np.array([1., 20., 20., 20., 10000.])
        self.assertEqual(lab.compute_neighbor_count_medians(cluster, counts, 1.)[0], 20.)
        counts[-1] = 1e9
        self.assertEqual(lab.compute_neighbor_count_medians(cluster, counts, 1.)[0], 20.)
        self.assertIsNone(lab.compute_neighbor_count_medians(cluster, counts, 1., is_cancelled=lambda: True))
        with self.assertRaises(ValueError):
            lab.compute_neighbor_count_medians(cluster, counts, 0.)

    def test_activation_precedes_nms_and_preserves_raw_ratio_and_other_fields(self):
        result = replace(comparison(), pca_neighbor_counts=np.array([20, 20, 40, 5, 20, 20, 20]))
        raw = np.array([0., 0., .4, .3, 0., 0., 0.])
        saved = raw.copy()
        with patch.object(lab, 'compute_inverse_std_fusion', return_value=raw):
            original = lab.with_inverse_std_fusion(result)
            activated = lab.with_inverse_std_fusion(result, activation='median_density')
        np.testing.assert_array_equal(original.ced_inverse_std_maxima, [2])
        np.testing.assert_array_equal(activated.ced_inverse_std_maxima, [3])
        np.testing.assert_array_equal(activated.inverse_std_raw_scores, raw)
        np.testing.assert_array_equal(raw, saved)
        self.assertEqual(activated.inverse_std_activation, 'median_density')
        for name, value in vars(result).items():
            if name not in ('ced_inverse_std_scores', 'ced_inverse_std_maxima', 'inverse_std_epsilon_mm2',
                            'inverse_std_raw_scores', 'inverse_std_activation', 'inverse_std_neighbor_medians'):
                self.assertIs(getattr(activated, name), value)

    def test_full_and_local_use_pca_counts_and_evaluated_query_medians(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5, ced_cutoff=.05)
        for compute in (lambda **kw: lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common, **kw),
                        lambda **kw: lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common, **kw)):
            baseline = compute()
            self.assertEqual(baseline.inverse_std_activation, 'none')
            activated = compute(inverse_std_activation='median_density')
            expected_counts = np.array([np.count_nonzero(np.linalg.norm(xyz-point, axis=1) <= 15.+1e-8)
                                        for point in activated.positions_mm])
            np.testing.assert_array_equal(activated.pca_neighbor_counts, expected_counts)
            for i, point in enumerate(activated.positions_mm):
                mask = np.linalg.norm(activated.positions_mm-point, axis=1) <= 15.+1e-8
                mask[i] = False
                self.assertEqual(activated.inverse_std_neighbor_medians[i], np.median(expected_counts[mask]))
            raw = activated.ced_scores/np.sqrt(activated.normal_lambda1_mm2+activated.inverse_std_epsilon_mm2)
            np.testing.assert_allclose(activated.inverse_std_raw_scores, raw)
            np.testing.assert_allclose(activated.ced_inverse_std_scores,
                -np.expm1(-raw*activated.inverse_std_neighbor_medians/expected_counts))
            for name in ('fusion_scores', 'fusion_maxima', 'normal_lambda1_mm2', 'harris_scores',
                         'ced_scores', 'local_minimum_scores', 'median_minimum_scores', 'positions_mm'):
                np.testing.assert_array_equal(getattr(activated, name), getattr(baseline, name))

    def window(self, scope):
        window = make_window()
        window.compare_scope_index = scope
        window.xyz = np.zeros((12, 3))
        window.analysis_xyz = window.xyz.copy()
        window.active_index, window.corners = 0, [dict(id='corner_001', point_index=0)]
        window._closest_analysis_point = Mock(return_value=(0, np.zeros(3)))
        return window

    def test_selector_updates_both_keys_clears_only_its_gate_and_is_idempotent(self):
        for scope in (0, 1):
            window = self.window(scope)
            self.assertEqual(window.compare_inverse_std_activation, 'none')
            self.assertEqual(window.compare_inverse_std_activation_combo.add_item.call_count, 3)
            key, job = window._compare_key(), window.compare_job_id
            window.compare_thresholds = {'ced_inverse_std_scores': .1, 'ced_scores': .2}
            window.compare_threshold_edits['ced_inverse_std_scores'].text_value = '.1'
            with redirect_stdout(StringIO()):
                window._inverse_std_activation_changed('', 2)
            self.assertNotEqual(window._compare_key(), key)
            self.assertEqual(window.compare_thresholds, {'ced_scores': .2})
            self.assertEqual(window.compare_threshold_edits['ced_inverse_std_scores'].text_value, '')
            self.assertGreater(window.compare_job_id, job)
            job = window.compare_job_id
            window._inverse_std_activation_changed('', 2)
            self.assertEqual(window.compare_job_id, job)

    def test_worker_paths_snapshot_activation_and_discard_stale_results(self):
        for scope, name in ((0, 'compute_local_fusion_pca_comparison'), (1, 'compute_full_scene_fusion_pca_comparison')):
            window = self.window(scope)
            window.compare_inverse_std_activation = 'median_density'
            window.app.post_to_main_thread.side_effect = lambda _win, callback: callback()
            with (patch.object(lab.threading, 'Thread') as thread,
                  patch.object(lab, name, return_value=comparison()) as compute,
                  patch.object(lab, 'voxel_downsample_analysis', return_value=(window.xyz, np.zeros((12, 3)))),
                  redirect_stdout(StringIO())):
                window._toggle_compare_extrema()
                window.compare_inverse_std_activation = 'exponential'
                thread.call_args.kwargs['target']()
            self.assertEqual(compute.call_args.kwargs['inverse_std_activation'], 'median_density')
            self.assertIsNone(window.compare_result)
            self.assertIn('Settings changed', window.compare_summary.text)

    def test_activated_heatmap_and_peak_plot_use_selected_score_and_dimensionless_units(self):
        window, _, rendered = ui.CornerLabUiTest().comparison_window()
        window.compare_result = result = lab.with_inverse_std_fusion(comparison(), activation='median_density')
        window.compare_inverse_std_activation = 'median_density'
        saved = copy.deepcopy(vars(result))
        with (patch.object(lab, 'apply_inverse_std_activation', side_effect=AssertionError('recomputed activation')),
              patch.object(lab, 'compute_neighbor_count_medians', side_effect=AssertionError('recomputed medians'))):
            window._compare_display_changed('', 12)
            expected, _ = lab.eigen_heatmap_colors(result.ced_inverse_std_scores, limits=(0., 1.))
            np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), expected/255.)
            self.assertTrue(window.compare_auto_contrast_check.enabled)
            self.assertIn('fixed 0-1', window.eigen_heatmap_legend.text)
            self.assertNotIn('1/mm', window.eigen_heatmap_legend.text)
            self.assertIn('Median density', window.compare_summary.text)
            display = dict(closed=False, window=Mock())
            point_id = int(result.ced_inverse_std_maxima[0])
            with patch.object(manager, 'show_nms_peak_plot', return_value=display) as show, redirect_stdout(StringIO()):
                self.assertTrue(window._picked_comparison_nms(result.positions_mm[point_id]))
            self.assertIs(show.call_args.args[1], result.ced_inverse_std_scores)
            self.assertEqual(show.call_args.kwargs['units'], '')
            self.assertIn('Activation: median_density', show.call_args.kwargs['title'])
            window._close_comparison_peak()
        np.testing.assert_equal(vars(result), saved)
