"""The cached converged Q10 gates plane fitting, including equality and missing values."""
from contextlib import redirect_stdout
import copy
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from corner_detector import q10_validator as q10
from corner_detector.refiner import Q10PositionRefiner
from corner_detector.settings import DetectorConfig, Q10ValidatorConfig
from corner_detector.visualization import manager


class Q10EarlyGateTest(unittest.TestCase):
    def run_lite(self, quality):
        theta = np.radians(np.linspace(20., 70., 40))
        raw = np.vstack((np.zeros(3), np.column_stack((5*np.cos(theta), 5*np.sin(theta), np.zeros(40)))))
        cfg = Q10ValidatorConfig(refine_enabled=True, voxel_mm=.01, radius_mm=8., min_q10=.15, refine_cc_enabled=False)
        extraction = {'candidate_indices': np.array([0, 1]),
                      'candidates': [{'point_index': i, 'score': .9} for i in range(2)]}
        original_refine = Q10PositionRefiner.refine
        def refine(instance, index, **kwargs):
            trace = original_refine(instance, index, **kwargs)
            trace['q10_values'][-1] = quality
            return trace
        with patch.object(Q10PositionRefiner, 'refine', refine), redirect_stdout(StringIO()):
            return q10.run_q10_validator(np.zeros((2, 3)), extraction,
                DetectorConfig(nms_radius_mm=.0001), cfg, raw_points_mm=raw)

    def test_failing_q10_skips_fit_and_both_repeated_seeds_reject(self):
        with patch.object(q10, 'segment_origin_spherical_planes', side_effect=AssertionError('unnecessary RANSAC')):
            result = self.run_lite(.14)
        self.assertEqual(result['q10_early_reject_count'], 2)
        self.assertEqual(result['final_evaluation_count'], 0)
        np.testing.assert_array_equal(result['rejected_indices'], [0, 1])
        np.testing.assert_array_equal(result['kept_indices'], [])
        for r in result['records']:
            self.assertEqual(r['q10_value'], .14)
            self.assertEqual(r['status'], 'REJECT')
            self.assertTrue(r['q10_reject'])
            self.assertFalse(r['antipodal_evaluated'])
            self.assertIsNone(r['antipodal_reject'])
            self.assertIsNone(r['antipodal'])
            self.assertEqual(r['rejection_reasons'], ['q10_below_threshold'])

    def test_q10_equality_and_above_threshold_fit_once_and_reuse_evidence(self):
        for quality in (.15, .16):
            with self.subTest(quality=quality), patch.object(q10, '_final_evidence', wraps=q10._final_evidence) as final:
                result = self.run_lite(quality)
            final.assert_called_once()
            self.assertEqual(result['q10_early_reject_count'], 0)
            self.assertEqual(result['final_evaluation_count'], 1)
            self.assertEqual(result['accept_count'], 2)
            self.assertTrue(all(r['antipodal_evaluated'] for r in result['records']))

    def test_nan_q10_skips_fit_and_stays_indeterminate(self):
        with patch.object(q10, '_final_evidence', side_effect=AssertionError('undefined Q10 ran final evidence')):
            result = self.run_lite(float('nan'))
        self.assertEqual(result['indeterminate_count'], 2)
        self.assertEqual(result['final_evaluation_count'], 0)
        self.assertTrue(all(r['reason'] == 'invalid_final_q10' for r in result['records']))

    def test_early_reject_debug_explains_skipped_ransac_and_keeps_trace_read_only(self):
        with patch.object(q10, '_final_evidence', side_effect=AssertionError('unnecessary RANSAC')):
            result = self.run_lite(.14)
        record = result['records'][0]
        saved = copy.deepcopy(record)
        display = manager.prepare_lite_validator_debug(None, record, result)
        display['figure'].canvas.draw()
        texts = '\n'.join(t.get_text() for axis in display['figure'].axes for t in axis.texts)
        self.assertIn('RANSAC SKIPPED', texts)
        np.testing.assert_equal(record, saved)
        display['figure'].clear()
