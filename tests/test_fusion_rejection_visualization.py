"""Cached detector gate colors distinguish thresholds from NMS and invalid data."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from corner_detector.candidate_extraction import extract_corner_candidates
from corner_detector.visualization import manager, view3d
from test_candidate_extraction import score_record


class FusionRejectionVisualizationTest(unittest.TestCase):
    def scene(self, bound=.2, mode='fixed'):
        points = np.column_stack(([0., .5, 5., 10., 15., 20., 25., 30.], np.zeros((8, 2))))
        scores = score_record([.9, .8, .2, .7, .6, .4, 0., 0.],
                              valid=[True, True, True, True, True, True, False, False])
        scores.update(pca_radius_mm=2., eigenvalues_mm2=np.array([
            [1.2, .1, 0.], [.1, .1, 0.], [100., .1, 0.], [.8, .1, 0.],
            [np.nan, .1, 0.], [.4, .1, 0.], [0., 0., 0.], [0., 0., 0.]]),
            ced_valid_mask=np.ones(8, dtype=bool))
        scores['pca_valid_mask'][5] = False
        scores['candidate_extraction'] = extract_corner_candidates(points, scores, 1.,
            fusion_score_threshold=.2, fusion_threshold_mode=mode, post_nms_lambda1_ratio_max=bound)
        return points, scores

    def test_exact_gate_colors_equality_stage_order_indices_and_read_only(self):
        points, scores = self.scene()
        original = copy.deepcopy(scores)
        with patch('corner_detector.candidate_extraction.local_maximum_nms_indices',
                   side_effect=AssertionError('display reran NMS')):
            data = view3d.fusion_rejection_primitives(points, scores)
        np.testing.assert_allclose(data['point_colors'], [
            [1., .05, .75], [.40, .40, .42], [.10, .40, 1.], [.10, 1., .30],
            [1., .55, .05], [1., .55, .05], [.16, .16, .18], [.16, .16, .18]])
        self.assertEqual(data['counts'], dict(invalid_score=2, fusion_threshold=1,
            nms_suppressed=1, lambda1_upper=1, invalid_weighted_pca=2, gates_passed=1))
        np.testing.assert_array_equal(data['indices'], [0, 3, 4, 5])
        np.testing.assert_array_equal(data['positions_mm'], points[data['indices']])
        np.testing.assert_array_equal(data['marker_colors'], data['point_colors'][data['indices']])
        self.assertIn('Fusion <= 0.2', data['legend'])
        self.assertIn('lambda1/R_PCA² > 0.2', data['legend'])
        self.assertIn('pre-Q10', data['legend'])
        np.testing.assert_equal(scores, original)
        data['point_colors'][:] = 0
        np.testing.assert_equal(scores, original)

    def test_off_adaptive_threshold_nonfinite_and_empty_results(self):
        points, scores = self.scene(bound=None)
        data = view3d.fusion_rejection_primitives(points, scores)
        self.assertEqual(data['counts']['gates_passed'], 4)
        self.assertEqual(data['counts']['lambda1_upper'], 0)
        self.assertEqual(data['counts']['invalid_weighted_pca'], 0)
        self.assertIn('gate OFF', data['legend'])
        points, scores = self.scene(mode='global_mean')
        data = view3d.fusion_rejection_primitives(points, scores)
        self.assertEqual(data['counts']['fusion_threshold'], 2)
        scores['fusion_score'][7] = np.nan
        scores['fusion_valid_mask'][7] = True
        self.assertEqual(view3d.fusion_rejection_primitives(points, scores)['counts']['invalid_score'], 2)
        points = np.empty((0, 3))
        scores = score_record([])
        scores['candidate_extraction'] = extract_corner_candidates(points, scores, 1.)
        data = view3d.fusion_rejection_primitives(points, scores)
        self.assertEqual(data['point_colors'].shape, (0, 3))
        self.assertEqual(data['marker_colors'].shape, (0, 3))
        self.assertEqual(sum(data['counts'].values()), 0)

    def test_managed_nms_view_colors_every_raw_peak_and_retains_marker_toggle(self):
        points, scores = self.scene()
        original = copy.deepcopy(scores)
        fake = MagicMock()
        view = fake.visualization.VisualizerWithKeyCallback.return_value
        view.create_window.return_value = True
        view.poll_events.return_value = False
        fake.geometry.PointCloud.side_effect = SimpleNamespace
        fake.utility.Vector3dVector.side_effect = lambda values: np.array(values, copy=True)
        markers = []
        def sphere(**kwargs):
            marker = MagicMock()
            markers.append(marker)
            return marker
        fake.geometry.TriangleMesh.create_sphere.side_effect = sphere
        cfg = SimpleNamespace(show_3d=True, only=True, voxel_3d_mm=1.)
        with (patch.object(view3d, 'require_open3d', return_value=fake),
              patch.object(view3d, 'synchronize_dense_camera_views', return_value=(None, None)),
              redirect_stdout(StringIO())):
            manager.show_dense_heatmaps(points, scores, cfg)
        data = view3d.fusion_rejection_primitives(points, scores)
        cloud = view.add_geometry.call_args_list[0].args[0]
        np.testing.assert_array_equal(cloud.colors, data['point_colors'])
        self.assertEqual(len(markers), 4)
        for marker, position, color in zip(markers, data['positions_mm'], data['marker_colors']):
            np.testing.assert_array_equal(marker.translate.call_args.args[0], position)
            np.testing.assert_array_equal(marker.paint_uniform_color.call_args.args[0], color)
        self.assertIn('gate reasons', view.create_window.call_args.kwargs['window_name'])
        key, callback = view.register_key_callback.call_args.args
        self.assertEqual(key, ord('M'))
        with redirect_stdout(StringIO()):
            callback(view)
            callback(view)
        view.remove_geometry.assert_called_once()
        self.assertFalse(view.remove_geometry.call_args.kwargs['reset_bounding_box'])
        self.assertEqual(view.add_geometry.call_count, 3)
        np.testing.assert_equal(scores, original)


if __name__ == '__main__':
    unittest.main()
