"""Independent settings and unchanged shared calculations up to cached UV projection."""

import copy
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from corner_detector import runner, validator_pipeline as pipeline, validator_plane_support as support
from corner_detector.settings import DEFAULTS, GuidedValidatorConfig, NewValidatorConfig, configure, parse_args
from corner_detector.visualization import manager, plots, query_ui, view3d


def config(mode='new-validator', *options):
    return configure(parse_args(['--mode', mode, '--source', 'realsense', '--no-q10-validator', *options]))[0]


class NewValidatorTest(unittest.TestCase):
    def test_defaults_match_current_validator_until_projection_and_settings_are_independent(self):
        new, old = config(), config('validator')
        self.assertIsInstance(new.validator, NewValidatorConfig)
        self.assertTrue(new.validator.projection_only)
        for name in vars(new.validator):
            if name not in ('projection_only', 'min_q10'):
                self.assertEqual(getattr(new.validator, name), getattr(old.validator, name))
        changed_old = config('new-validator', '--validator-ransac-voxel-mm', '9', '--validator-plane-connect-radius-x', '99',
                             '--no-validator-geometry')
        self.assertEqual(changed_old.validator, new.validator)
        changed_new = config('new-validator', '--new-validator-ransac-voxel-mm', '4', '--new-validator-ransac-radius-x', '3',
                             '--new-validator-inlier-threshold-x', '.5', '--new-validator-plane-connect-radius-x', '2')
        self.assertEqual((changed_new.validator.ransac_radius_mm, changed_new.validator.inlier_threshold_mm,
                          changed_new.validator.plane_connect_radius_mm), (12., 2., 8.))
        untouched_old = config('validator', '--new-validator-ransac-voxel-mm', '4')
        self.assertEqual(untouched_old.validator, old.validator)
        default = configure(parse_args(['--mode', 'new-validator', '--source', 'realsense']))[0]
        self.assertEqual(default.q10_validator is not None, DEFAULTS.q10_validator_enabled)
        for flag, value in (('ransac-voxel-mm', '0'), ('ransac-radius-x', '0'), ('max-planes', '4'),
                            ('ransac-iterations', '0'), ('min-inliers', '2'), ('min-inlier-ratio', '1.1'),
                            ('roi-padding-mm', '-1'), ('plane-connect-radius-x', '0')):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                config('new-validator', '--new-validator-' + flag, value)

    def test_uv_is_identical_to_existing_support_and_hull_is_not_called(self):
        points = np.array([[0.,0.,2.], [2.,0.,2.], [2.,4.,2.], [0.,4.,2.]])
        query = np.array([1.,2.,7.])
        planes = [{'normal': np.array([0.,0.,2.]), 'offset_mm': -4., 'inlier_ransac_indices': np.arange(4)}]
        old = support.reconstruct_plane_support(points, query, planes, GuidedValidatorConfig())
        with patch.object(support, 'ConvexHull', side_effect=AssertionError('new path computed a hull')) as hull:
            new = support.reconstruct_plane_support(points, query, planes, NewValidatorConfig(), projection_only=True)
        hull.assert_not_called()
        a, b = old['plane_patches'][0], new['plane_patches'][0]
        for key in ('support_ransac_indices', 'recollected_ransac_indices', 'origin_mm', 'basis_u', 'basis_v', 'projection_uv'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(b['projection_uv'], [[-1.,-2.], [1.,-2.], [1.,2.], [-1.,2.]])
        self.assertAlmostEqual(a['hull_area_mm2'], 8.)
        self.assertEqual(b['hull_uv'].shape, (0, 2))
        self.assertEqual(b['hull_area_mm2'], 0.)
        result = {'plane_patches': new['plane_patches'], 'query_index': 0}
        saved = copy.deepcopy(result)
        figure = plots.make_projection_validator_figure(result)
        self.assertGreater(plots.figure_to_rgb(figure).size, 0)
        figure.clear()
        np.testing.assert_equal(result, saved)

    def test_pipeline_matches_ransac_and_projection_and_stops_before_all_later_stages(self):
        x, y = np.meshgrid(np.arange(-20., 21., 4.), np.arange(-20., 21., 4.))
        zero = np.zeros(x.size)
        points = np.unique(np.vstack((np.column_stack((x.ravel(), y.ravel(), zero)),
                                      np.column_stack((zero, x.ravel(), y.ravel())))), axis=0)
        index = int(np.argmin(np.linalg.norm(points, axis=1)))
        snapshot = points.copy()
        with patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p,v:p.copy()):
            old = pipeline.run_validator_pipeline(points, index, config('validator'), raw_points_mm=points, include_debug=False)
            with (patch.object(support, 'ConvexHull', side_effect=AssertionError('hull called')),
                  patch('corner_detector.validator_edge_graph.build_validator_edge_graph', side_effect=AssertionError('edge called')),
                  patch('corner_detector.validator_corner_graph.reconstruct_validator_corners', side_effect=AssertionError('corner called'))):
                new = pipeline.run_validator_pipeline(points, index, config(), raw_points_mm=points)
        self.assertGreater(len(new['planes']), 0)
        self.assertEqual(len(new['planes']), len(old['planes']))
        for key in ('query_position_mm', 'roi_raw_point_indices', 'ransac_points_mm', 'ransac_scene_indices'):
            np.testing.assert_array_equal(new[key], old[key])
        for a, b in zip(new['planes'], old['planes']):
            for key in ('normal', 'offset_mm', 'inlier_ransac_indices'):
                np.testing.assert_array_equal(a[key], b[key])
        for a, b in zip(new['plane_patches'], old['plane_patches']):
            for key in ('support_point_indices', 'support_ransac_indices', 'projection_uv', 'basis_u', 'basis_v', 'origin_mm'):
                np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(new['projection_computed'])
        self.assertFalse(new['geometry_computed'])
        self.assertIn(new['validation_status'], ('ACCEPT', 'REJECT', 'INDETERMINATE'))
        self.assertEqual(new['accept'], all(score['passed'] is True for score in new['plane_q10']))
        self.assertEqual(new['corners'], [])
        self.assertEqual((new['stage_elapsed_s']['edge_graph'], new['stage_elapsed_s']['corner_graph']), (0., 0.))
        np.testing.assert_array_equal(points, snapshot)

    def test_empty_planes_and_sparse_projection_are_safe_and_plot_is_read_only(self):
        points = np.array([[0.,0,0], [1.,0,0]])
        with patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p,v:p.copy()):
            result = pipeline.run_validator_pipeline(points, 0, config('new-validator', '--new-validator-max-planes', '0'),
                                                      raw_points_mm=points)
        self.assertEqual(result['plane_patches'], [])
        saved = copy.deepcopy(result)
        figure = plots.make_projection_validator_figure(result)
        self.assertGreater(plots.figure_to_rgb(figure).size, 0)
        figure.clear()
        np.testing.assert_equal(result, saved)
        planes = [{'normal': [0.,0,1.], 'offset_mm': 0., 'inlier_ransac_indices': np.arange(2)}]
        patch_result = support.reconstruct_plane_support(points, np.zeros(3), planes, NewValidatorConfig(), projection_only=True)
        self.assertFalse(patch_result['plane_patches'][0]['projection_computed'])
        with patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p,v:p.copy()):
            batch = pipeline.run_validator_candidates(points, [0], config(), raw_points_mm=points, include_debug=False)
        self.assertTrue(batch['records'][0]['projection_computed'])
        self.assertIn('plane_patches', batch['records'][0])

    def test_view_creates_projection_window_instead_of_edge_corner_windows(self):
        with patch.object(view3d, 'o3d', new=MagicMock()) as graphics:
            graphics.geometry.PointCloud.side_effect = MagicMock
            graphics.utility.Vector3dVector.side_effect = lambda p:np.array(p, copy=True)
            display = view3d.open_guided_validator_view(np.zeros((1,3)), np.array([0]), np.ones((1,3)), projection_only=True)
            app = graphics.visualization.gui.Application.instance
            self.assertEqual(app.create_window.call_count, 2)
            self.assertEqual(len(display['geometry_windows']), 1)
            self.assertEqual(display['geometry_windows'][0]['panels'], [])
            view3d.update_validator_geometry_view(display, {'planes_computed': True, 'geometry_computed': False})
            view3d.update_validator_projection_view(display, np.zeros((40,60,3), dtype=np.uint8))
            display['projection_image'].update_image.assert_called_once()
        frozen = {'points_voxel_mm':np.zeros((1,3)), 'validator_query_indices':np.array([0]),
                  'validator_scene_colors':np.ones((1,3)), 'validator_projection_only':True}
        with (patch.object(manager,'close_guided_validator_view'), patch.object(manager,'open_guided_validator_view') as open_view):
            manager.prepare_guided_validator_view(frozen)
        self.assertTrue(open_view.call_args.kwargs['projection_only'])

    def test_runner_and_spawned_view_keep_new_mode_and_seed_coordinates(self):
        scene = np.array([[0.,0,0], [1.,0,0], [0.,1.,0]])
        frozen = {'points_raw_mm':scene}
        seeds = {'candidate_indices':np.array([0]), 'positions_mm':scene[:1], 'source':'detector_nms'}
        frame = SimpleNamespace(source='realsense', points_mm=scene)
        with (patch.object(runner,'require_open3d'), patch.object(runner,'tk',object()),
              patch.object(runner,'capture_realsense_point_cloud',return_value=frame),
              patch.object(runner,'apply_noise_to_frame',return_value=frame),
              patch.object(runner,'prepare_frozen_from_frame',return_value=(frozen,scene)),
              patch.object(runner,'prepare_validator_queries',return_value={'validator_seeds':seeds}),
              patch.object(runner,'run_validator_mode',return_value='quit') as view, redirect_stdout(StringIO())):
            runner.main(['--mode','new-validator','--source','realsense','--no-q10-validator'])
        self.assertEqual(view.call_args.args[2].mode,'new-validator')
        self.assertTrue(frozen['validator_projection_only'])
        connection=Mock()
        def open_view(state,_callback):
            self.assertTrue(state['validator_projection_only'])
            state['guided_validator_display']=None
        with (patch.object(query_ui,'open_guided_validator_window',side_effect=open_view), patch.object(query_ui,'close_query_views')):
            query_ui.run_guided_validator_view_process(connection,scene,np.array([0]),np.ones_like(scene),
                                                       query_positions_mm=scene[:1],projection_only=True)
        connection.send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
