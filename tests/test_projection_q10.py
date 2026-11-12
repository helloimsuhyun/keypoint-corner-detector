"""2-D centroid axes, cosine quantiles and strict all-fitted-planes composition."""

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from corner_detector.settings import DEFAULTS, NewValidatorConfig, configure, parse_args
from corner_detector.validator_plane_support import projection_centroid_q10
from corner_detector.validator_pipeline import _projection_q10_result
from corner_detector.visualization.plots import make_projection_validator_figure, figure_to_rgb


class ProjectionQ10Test(unittest.TestCase):
    def patch(self, points, i=0):
        return {'plane_id': i, 'projection_uv': np.asarray(points, dtype=float), 'status': 'VALID'}

    def test_centroid_axis_uses_original_2d_distances_and_expected_q10(self):
        uv = np.array([[1.,0.], [1.,0.], [0.,10.], [0.,1.]])
        result = projection_centroid_q10(uv)
        np.testing.assert_array_equal(result['centroid_uv_mm'], [.5, 2.75])
        np.testing.assert_allclose(result['direction_uv'], np.array([2.,11.])/np.sqrt(125.))
        self.assertAlmostEqual(result['q10'], 2/np.sqrt(125.))
        self.assertAlmostEqual(np.linalg.norm(result['direction_uv']), 1.)
        with_self = projection_centroid_q10(np.vstack((np.zeros(2), uv)))
        self.assertFalse(with_self['nonzero_mask'][0])
        self.assertAlmostEqual(with_self['q10'], result['q10'])

    def test_local_uv_axis_rotation_or_reflection_does_not_change_score(self):
        uv = np.array([[1.,.2], [2.,3.], [4.,1.], [-.1,2.]])
        result = projection_centroid_q10(uv)
        angle = .73
        for transform in (np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), np.diag([1.,-1.])):
            changed = projection_centroid_q10(uv @ transform)
            self.assertAlmostEqual(changed['q10'], result['q10'])
            np.testing.assert_allclose(changed['direction_uv'], result['direction_uv'] @ transform, atol=1e-14)

    def test_one_two_and_three_planes_require_every_plane_and_equality_passes(self):
        good = [[.6,.8], [.6,-.8], [1.2,1.6], [1.2,-1.6]]
        bad = [[.2,1.], [.2,-1.], [.4,2.], [.4,-2.]]
        for count in (1,2,3):
            patches = [self.patch(good,i) for i in range(count)]
            passed = _projection_q10_result(patches,.6)
            self.assertEqual((passed['validation_status'],passed['passed_plane_count']),('ACCEPT',count))
            self.assertTrue(passed['accept'])
            self.assertEqual(_projection_q10_result(patches,np.nextafter(.6,np.inf))['validation_status'],'REJECT')
            for failed in range(count):
                mixed = list(patches)
                mixed[failed] = self.patch(bad,failed)
                result = _projection_q10_result(mixed,.6)
                self.assertEqual(result['validation_status'],'REJECT')
                self.assertEqual(result['passed_plane_count'],count-1)
                self.assertFalse(result['accept'])

    def test_missing_planes_zero_centroid_and_sparse_directions_never_accept(self):
        unknowns = [np.empty((0,2)), np.zeros((5,2)), np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])]
        self.assertEqual(_projection_q10_result([],.1)['validation_status'],'INDETERMINATE')
        for uv in unknowns:
            result = _projection_q10_result([self.patch(uv)],.1)
            self.assertEqual(result['validation_status'],'INDETERMINATE')
            self.assertFalse(result['accept'])
            self.assertIsNone(result['plane_q10'][0]['q10'])
        mixed = _projection_q10_result([self.patch(unknowns[-1]), self.patch([[-1.,0.],[-1.,0.],[10.,0.]],1)],.1)
        self.assertEqual(mixed['validation_status'],'REJECT')
        for invalid in (np.zeros((4,3)), np.full((4,2),np.nan)):
            with self.assertRaises(ValueError):
                projection_centroid_q10(invalid)

    def test_threshold_is_independent_config_and_cli_and_plot_reads_cached_axis(self):
        cfg = configure(parse_args(['--mode','new-validator','--source','realsense','--new-validator-min-q10','.3']))[0]
        self.assertEqual(cfg.validator.min_q10,.3)
        self.assertEqual(NewValidatorConfig().min_q10,DEFAULTS.new_validator_min_q10)
        for threshold in (-1.01,1.01,np.nan):
            with self.assertRaises(ValueError):
                replace(NewValidatorConfig(),min_q10=threshold).validate()
        patches = [self.patch([[.6,.8],[.6,-.8],[1.2,1.6],[1.2,-1.6]])]
        result = {'query_index':0,'plane_patches':patches,**_projection_q10_result(patches,.3)}
        snapshot = deepcopy(result)
        figure = make_projection_validator_figure(result)
        self.assertIn('Q10=0.6000',figure.axes[0].get_title())
        self.assertGreater(figure_to_rgb(figure).size,0)
        figure.clear()
        np.testing.assert_equal(result,snapshot)


if __name__ == '__main__':
    unittest.main()
