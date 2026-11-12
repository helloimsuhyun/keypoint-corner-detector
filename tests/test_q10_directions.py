"""Direction definitions, default regression, cached world-space debug and CLI."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import copy
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from corner_detector.connectivity import radius_components
from corner_detector.refiner import mean_direction_q10, Q10PositionRefiner
from corner_detector import q10_validator
from corner_detector.settings import DEFAULTS, DetectorConfig, Q10ValidatorConfig, configure, parse_args
from corner_detector.visualization import manager, view3d


class Q10DirectionsTest(unittest.TestCase):
    def test_three_axes_and_expected_quantiles(self):
        p = np.array([10., 20., 30.])
        offsets = np.array([[1., 0, 0], [1., 0, 0], [0, 10., 0], [0, 0, 2.], [1., 1., 0]])
        mean = np.array([2. + 1 / np.sqrt(2), 1. + 1 / np.sqrt(2), 1.])
        mean /= np.linalg.norm(mean)
        centroid = np.array([3., 11., 2.]) / np.sqrt(134.)
        expected = {'unit_mean': (mean, .6 * mean[2] + .4 * mean[1]),
                    'unit_median': (np.array([1., 0, 0]), 0.),
                    'centroid': (centroid, 2.4 / np.sqrt(134.))}
        for mode, (axis, score) in expected.items():
            with self.subTest(mode=mode):
                actual, direction = mean_direction_q10(p, np.vstack((p, p + offsets)), mode)
                np.testing.assert_allclose(direction, axis, atol=1e-14)
                self.assertAlmostEqual(actual, score)
                self.assertAlmostEqual(np.linalg.norm(direction), 1.)

    def test_default_is_original_unit_mean_and_distance_scaling_only_changes_centroid(self):
        offsets = np.array([[1., 2., 3.], [2., 1., 1.], [3., 2., 1.], [1., 4., 2.]])
        units = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        axis = units.mean(axis=0)
        axis /= np.linalg.norm(axis)
        score, actual = mean_direction_q10(np.zeros(3), offsets)
        np.testing.assert_array_equal(actual, axis)
        self.assertEqual(score, float(np.quantile(np.clip(units @ axis, -1, 1), .1)))
        stretched = offsets * np.array([1., 8., 2., 5.])[:, None]
        for mode in ('unit_mean', 'unit_median'):
            np.testing.assert_allclose(mean_direction_q10(np.zeros(3), offsets, mode)[1],
                                       mean_direction_q10(np.zeros(3), stretched, mode)[1], atol=1e-14)
        self.assertFalse(np.allclose(mean_direction_q10(np.zeros(3), offsets, 'centroid')[1],
                                     mean_direction_q10(np.zeros(3), stretched, 'centroid')[1]))

    def test_undefined_axes_are_invalid_and_cached(self):
        symmetric = np.vstack((np.eye(3), -np.eye(3)))
        for mode in ('unit_mean', 'unit_median', 'centroid'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                mean_direction_q10(np.zeros(3), symmetric, mode)
        points = np.vstack((np.zeros(3), np.eye(3), -np.eye(3), 2 * np.eye(3), -2 * np.eye(3)))
        for mode in ('unit_mean', 'unit_median', 'centroid'):
            refiner = Q10PositionRefiner(points, 4., .1, cc_enabled=False, direction_mode=mode)
            self.assertIsNone(refiner.evaluate(0)[0])
            refiner.evaluate(0)
            self.assertEqual((refiner.cache_misses, refiner.cache_hits), (1, 1))
        with self.assertRaises(ValueError):
            Q10PositionRefiner(points, 4., .1, direction_mode='unknown')

    def test_moved_cc_preserves_inclusive_edges_and_duplicates(self):
        labels = radius_components(np.array([[0., 0, 0], [1., 0, 0], [1., 0, 0], [3., 0, 0]]), 1.)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1])

    def test_options_reach_refinement_and_cached_debug(self):
        points = np.vstack((np.zeros(3), [[x, y, .2 * x] for x in (1., 2., 3., 4.) for y in (1., 2., 3.)]))
        p = np.array([10., 20., 30.])
        points += p
        extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .5}]}
        for mode in ('unit_mean', 'unit_median', 'centroid'):
            cfg = configure(parse_args(['--mode', 'detector', '--source', 'realsense', '--q10-direction-mode', mode,
                                        '--q10-radius-mm', '10', '--q10-voxel-mm', '.01', '--q10-debug',
                                        '--q10-refine', '--no-q10-refine-cc', '--q10-min-q10', '-1']))[0]
            cfg.nms_radius_mm = .001
            with (patch.object(q10_validator, 'voxel_downsample_3d', return_value=points), redirect_stdout(StringIO())):
                result = q10_validator.run_q10_validator(points, extraction, cfg, cfg.q10_validator, raw_points_mm=points)
            record = result['records'][0]
            self.assertEqual((result['direction_mode'], record['direction_mode']), (mode, mode))
            expected, axis = mean_direction_q10(p, points, mode)
            self.assertAlmostEqual(record['q10_value'], expected)
            np.testing.assert_allclose(record['trace']['mean_axes'][-1], axis)
            saved = copy.deepcopy(record)
            with patch.object(Q10PositionRefiner, 'evaluate', side_effect=AssertionError('debug recomputed Q10')):
                display = manager.prepare_lite_validator_debug(points, record, result)
                for step in range(len(record['trace']['positions_mm'])):
                    display['set_step'](step)
                    arrow = view3d.q10_direction_primitives(record, step)
                    np.testing.assert_allclose(arrow['origin_mm'], p)
                    np.testing.assert_allclose(arrow['tip_mm'], p + 10. * axis)
                    self.assertEqual(arrow['mode'], mode)
                titles = ' '.join(a.get_xlabel() for a in display['figure'].axes)
                self.assertIn(mode, titles)
                display['figure'].clear()
            np.testing.assert_equal(record, saved)
        self.assertEqual(DEFAULTS.q10_direction_mode, 'unit_mean')

    def test_removed_modes_and_settings_are_unavailable(self):
        self.assertFalse(hasattr(DEFAULTS, 'cone_extremality_enabled'))
        self.assertFalse(hasattr(DEFAULTS, 'heatmap_min_radius_mm'))
        self.assertFalse(hasattr(DetectorConfig(), 'cone_extremality'))
        for args in (['--mode', 'heatmap'], ['--cone-extremality'], ['--heatmap-min-radius-mm', '30'],
                     ['--q10-direction-mode', 'unknown']):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(['--source', 'realsense', *args])
        with self.assertRaises(ValueError):
            Q10ValidatorConfig(direction_mode='unknown').validate()

    def test_native_arrow_uses_cached_axis_position_and_step(self):
        points = np.array([[1., 2., 3.], [2., 2., 3.]])
        record = {'kind': 'q10_antipodal', 'point_index': 7, 'voxel_mm': 5., 'radius_mm': 10., 'direction_mode': 'unit_median',
                  'position': points[0], 'refined_position_mm': points[1], 'status': 'ACCEPT', 'antipodal': None,
                  'trace': {'positions_mm': points, 'mean_axes': np.array([[1., 0, 0], [0, 1., 0]]),
                            'q10_values': np.array([.4, .5])},
                  'step_analysis_indices': [np.arange(2), np.arange(2)], 'step_points_mm': [points, points],
                  'roi_analysis_indices': np.arange(2), 'roi_analysis_points_mm': points}
        with patch.object(view3d, 'o3d', new=MagicMock()) as graphics:
            first, second = MagicMock(), MagicMock()
            graphics.geometry.TriangleMesh.create_arrow.side_effect = [first, second]
            graphics.utility.Vector3dVector.side_effect = lambda values: np.array(values, copy=True)
            display = view3d.open_q10_debug_view(record)
            display['set_step'](0)
            display['close']()
            np.testing.assert_array_equal(first.rotate.call_args.args[0][:, 2], [0., 1., 0.])
            np.testing.assert_array_equal(second.rotate.call_args.args[0][:, 2], [1., 0., 0.])
            np.testing.assert_array_equal(first.translate.call_args.args[0], points[1])
            np.testing.assert_array_equal(second.translate.call_args.args[0], points[0])
            for call in graphics.geometry.TriangleMesh.create_arrow.call_args_list:
                self.assertAlmostEqual(call.kwargs['cylinder_height'] + call.kwargs['cone_height'], 10.)
            self.assertIn('unit_median', graphics.visualization.VisualizerWithKeyCallback.return_value.create_window.call_args.kwargs['window_name'])


if __name__ == '__main__':
    unittest.main()
