"""Selectable Min-Max Fusion and an optional weighted-lambda1 gate after NMS."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from corner_detector import dense_heatmap, extremity_synthetic
from corner_detector.candidate_extraction import extract_corner_candidates
from corner_detector.edge_suppression import (
    bilateral_filter_ced, dense_ced_weighted_pca_scores, local_minimum_contrast,
)
from corner_detector.settings import DEFAULTS, configure, parse_args
from corner_detector.validator_pipeline import prepare_validator_seeds
from corner_detector.visualization.view3d import dense_heatmap_colors, dense_heatmap_panels, dense_heatmap_vmax
from test_candidate_extraction import score_record


def corner_grid():
    grid = np.array([(x, y, 0.) for x in range(0, 10, 2) for y in range(0, 10, 2)])
    return np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)


class DetectorLocalMinimumTest(unittest.TestCase):
    def test_minmax_formula_flat_invalid_inclusive_radius_and_geometric_scaling(self):
        points = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
        values = np.array([.010, .011, .012])
        score = local_minimum_contrast(points, values, 1.)
        np.testing.assert_allclose(score, [2/7, 1/7, 0.])
        np.testing.assert_allclose(local_minimum_contrast(points*10, values*100, 10.), score)
        np.testing.assert_array_equal(local_minimum_contrast(points, np.zeros(3), 1.), np.zeros(3))
        invalid = local_minimum_contrast(points, [0., 1., np.nan], 1.)
        np.testing.assert_allclose(invalid[:2], [1/1.005, 0.])
        self.assertTrue(np.isnan(invalid[2]))
        boundary = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
        self.assertAlmostEqual(local_minimum_contrast(boundary, [1., 2., 100.], 1.)[0], 1/1.005)

    def test_new_mode_uses_normal_covariance_and_selected_ced_without_ratio_saturation(self):
        points = corner_grid()
        for mode in ('ced', 'fusion', 'off'):
            options = dict(fusion_mode='local_minimum', bilateral_mode=mode, pca_ced_cutoff=.1)
            scores = dense_ced_weighted_pca_scores(points, 4.1, 6.1, 3, **options)
            tree = cKDTree(points)
            expected = np.full(len(points), np.nan)
            for i, point in enumerate(points):
                ids = np.asarray(tree.query_ball_point(point, 6.1))
                delta = points[ids]-points[ids].mean(axis=0)
                expected[i] = np.linalg.eigvalsh(delta.T@delta/len(ids))[-1]
            np.testing.assert_allclose(scores['normal_pca_lambda1_mm2'], expected)
            minimum = local_minimum_contrast(points, expected, 6.1)
            np.testing.assert_allclose(scores['local_minimum_score'], minimum)
            raw = scores['bilateral_ced_score']*minimum
            raw[~scores['fusion_valid_mask']] = 0.
            np.testing.assert_allclose(scores['fusion_score_raw'], raw)
            expected_fusion = raw
            if mode == 'fusion':
                expected_fusion = bilateral_filter_ced(points, raw, scores['fusion_valid_mask'],
                    4.1, 2.05, .05)['score']
            np.testing.assert_allclose(scores['fusion_score'], expected_fusion)
            saturated = dense_ced_weighted_pca_scores(points, 4.1, 6.1, 3,
                pca_ratio_saturation_tau=.1, **options)
            np.testing.assert_array_equal(saturated['fusion_score'], scores['fusion_score'])

    def test_new_score_is_independent_of_weighted_pca_cutoff_but_still_requires_query_ced(self):
        points = corner_grid()
        a = dense_ced_weighted_pca_scores(points, 4.1, 6.1, 3,
            fusion_mode='local_minimum', bilateral_mode='off')
        b = dense_ced_weighted_pca_scores(points, 4.1, 6.1, 3,
            fusion_mode='local_minimum', bilateral_mode='off', pca_ced_cutoff=1.)
        self.assertFalse(b['pca_valid_mask'].any())
        self.assertTrue(b['fusion_valid_mask'].any())
        np.testing.assert_array_equal(a['fusion_score'], b['fusion_score'])
        no_ced = dense_ced_weighted_pca_scores(points, .1, 6.1, 3, fusion_mode='local_minimum')
        self.assertTrue(np.isfinite(no_ced['normal_pca_lambda1_mm2']).all())
        self.assertFalse(no_ced['fusion_valid_mask'].any())
        np.testing.assert_array_equal(no_ced['fusion_score'], np.zeros(len(points)))

    def test_gate_runs_after_nms_without_promoting_a_suppressed_neighbor_or_mutating_scores(self):
        points = np.array([[0., 0., 0.], [.5, 0., 0.], [5., 0., 0.]])
        scores = score_record([.9, .8, .7])
        scores.update(pca_radius_mm=2., eigenvalues_mm2=np.array([[1.2, .1, 0.], [.2, .1, 0.], [.8, .1, 0.]]))
        saved = copy.deepcopy(scores)
        baseline = extract_corner_candidates(points, scores, 1.)
        result = extract_corner_candidates(points, scores, 1., post_nms_lambda1_ratio_max=.2)
        np.testing.assert_array_equal(baseline['candidate_indices'], [0, 2])
        np.testing.assert_array_equal(result['nms_candidate_indices'], [0, 2])
        np.testing.assert_array_equal(result['candidate_indices'], [2])
        np.testing.assert_allclose(result['post_nms_lambda1_gate']['normalized_lambda1'], [.3, .2])
        np.testing.assert_array_equal(result['post_nms_lambda1_gate']['rejected_indices'], [0])
        np.testing.assert_array_equal(prepare_validator_seeds(points, result)['candidate_indices'], [2])
        self.assertEqual(result['candidates'][0]['weighted_lambda1_over_r2'], .2)
        np.testing.assert_equal(scores, saved)
        scaled = copy.deepcopy(scores)
        scaled['pca_radius_mm'] *= 10
        scaled['eigenvalues_mm2'] *= 100
        scaled_result = extract_corner_candidates(points*10, scaled, 10., post_nms_lambda1_ratio_max=.2)
        np.testing.assert_array_equal(scaled_result['candidate_indices'], [2])

    def test_gate_rejects_invalid_weighted_support_and_checks_missing_radius(self):
        points = np.column_stack((10*np.arange(3.), np.zeros((3, 2))))
        scores = score_record([.9, .8, .7])
        scores.update(pca_radius_mm=2., eigenvalues_mm2=np.array([[.1, 0., 0.], [np.nan, 0., 0.], [.1, 0., 0.]]))
        scores['pca_valid_mask'][0] = False
        result = extract_corner_candidates(points, scores, 1., post_nms_lambda1_ratio_max=.2)
        np.testing.assert_array_equal(result['candidate_indices'], [2])
        np.testing.assert_array_equal(result['post_nms_lambda1_gate']['invalid_mask'], [True, True, False])
        del scores['pca_radius_mm']
        with self.assertRaisesRegex(ValueError, 'PCA radius'):
            extract_corner_candidates(points, scores, 1., post_nms_lambda1_ratio_max=.2)

    def test_settings_modes_thresholds_gate_off_and_cli_propagation(self):
        for mode in ('detector', 'validator', 'total', 'new-validator', 'ball', 'extremity256'):
            args = parse_args(['--source', 'realsense', '--mode', mode, '--fusion-mode', 'local_minimum'])
            args.post_nms_lambda1_ratio_max = None
            cfg, _ = configure(args)
            self.assertEqual(cfg.fusion_mode, 'local_minimum')
            self.assertEqual(cfg.fusion_score_threshold, .2)
            self.assertEqual(cfg.local_minimum_tau, .005)
            self.assertIsNone(cfg.post_nms_lambda1_ratio_max)
            args.post_nms_lambda1_ratio_max = .15
            args.local_minimum_fusion_threshold = .3
            cfg, _ = configure(args)
            self.assertEqual(cfg.post_nms_lambda1_ratio_max, .15)
            self.assertEqual(cfg.fusion_score_threshold, .3)
        legacy_args = parse_args(['--source', 'realsense', '--mode', 'detector', '--fusion-mode', 'pca_ratio'])
        legacy_args.post_nms_lambda1_ratio_max = None
        legacy, _ = configure(legacy_args)
        self.assertEqual(legacy.fusion_mode, 'pca_ratio')
        self.assertEqual(legacy.fusion_score_threshold, DEFAULTS.fusion_score_threshold)
        self.assertIsNone(legacy.post_nms_lambda1_ratio_max)
        current, _ = configure(parse_args(['--source', 'realsense', '--mode', 'detector']))
        self.assertEqual(current.fusion_mode, DEFAULTS.fusion_mode)
        self.assertEqual(current.post_nms_lambda1_ratio_max, DEFAULTS.post_nms_lambda1_ratio_max)
        for flag, value in (('--local-minimum-tau', '0'), ('--local-minimum-tau', 'nan'),
                            ('--local-minimum-fusion-threshold', '1.1'), ('--post-nms-lambda1-ratio-max', '-.1')):
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                configure(parse_args(['--source', 'realsense', '--mode', 'detector', flag, value]))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(['--source', 'realsense', '--mode', 'detector', '--fusion-mode', 'unknown'])

    def test_shared_callers_and_managed_panels_use_new_score(self):
        points = corner_grid()
        cfg, _ = configure(parse_args(['--source', 'realsense', '--mode', 'detector',
            '--fusion-mode', 'local_minimum', '--no-q10-validator', '--no-3d-view',
            '--min-neighbors', '3', '--centroid-radius-mm', '4.1', '--pca-radius-factor', str(6.1/4.1)]))
        with (patch.object(dense_heatmap, 'dense_ced_weighted_pca_scores', wraps=dense_ced_weighted_pca_scores) as score,
              redirect_stdout(StringIO())):
            result = dense_heatmap.run_dense_heatmap_mode(points, cfg)
        self.assertEqual(score.call_args.kwargs['fusion_mode'], 'local_minimum')
        self.assertEqual(score.call_args.kwargs['local_minimum_tau'], .005)
        self.assertEqual(result['candidate_extraction']['configured_fusion_score_threshold'], .2)
        panels = dense_heatmap_panels('off', 'local_minimum')
        self.assertEqual(panels[3][1], 'local_minimum_score')
        limits = dense_heatmap_vmax(result)
        colors = dense_heatmap_colors(result['local_minimum_score'], result['local_minimum_valid_mask'],
                                       vmax=limits['local_minimum_score'])
        self.assertEqual(colors.shape, points.shape)
        with patch.object(extremity_synthetic, 'dense_ced_weighted_pca_scores', wraps=dense_ced_weighted_pca_scores) as score:
            extremity_synthetic.ced_candidates(points, cfg)
        self.assertEqual(score.call_args.kwargs['fusion_mode'], 'local_minimum')

    def test_q10_receives_only_post_gate_candidates(self):
        points = np.array([[0., 0., 0.], [.5, 0., 0.], [5., 0., 0.]])
        scores = dense_ced_weighted_pca_scores(points, 10., 10., 2)
        scores['fusion_score'] = np.array([.9, .8, .7])
        scores['fusion_valid_mask'][:] = True
        scores['pca_valid_mask'][:] = True
        scores['pca_radius_mm'] = 2.
        scores['eigenvalues_mm2'] = np.array([[1.2, .1, 0.], [.2, .1, 0.], [.8, .1, 0.]])
        cfg, _ = configure(parse_args(['--source', 'realsense', '--mode', 'detector', '--no-3d-view',
            '--centroid-radius-mm', '10', '--nms-radius-factor', '.1', '--post-nms-lambda1-ratio-max', '.2']))
        def q10(scene, extraction, *_args, **_kwargs):
            np.testing.assert_array_equal(extraction['candidate_indices'], [2])
            return {'accepted_indices': np.array([2]), 'refine_enabled': False,
                    'records': [{'point_index': 2, 'refined_position_mm': points[2]}]}
        with (patch.object(dense_heatmap, 'dense_ced_weighted_pca_scores', return_value=scores),
              patch.object(dense_heatmap, 'run_q10_validator', side_effect=q10) as validate,
              redirect_stdout(StringIO())):
            result = dense_heatmap.run_dense_heatmap_mode(points, cfg, raw_points_mm=points)
        validate.assert_called_once()
        np.testing.assert_array_equal(result['validator_seeds']['candidate_indices'], [2])
