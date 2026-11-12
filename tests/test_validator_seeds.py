"""Detector/Q10 seeds reach validator centers without changing scene IDs or XYZ."""

import copy
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from corner_detector import dense_heatmap, total, validator_pipeline as pipeline, validator_query
from corner_detector.cornervins_plane_extraction import as_plane_cluster_analysis
from corner_detector.settings import DEFAULTS, configure, parse_args
from corner_detector.visualization import manager, query_ui, view3d


def config(mode='validator', enabled=True):
    flags = ['--mode', mode, '--source', 'realsense', '--q10-validator' if enabled else '--no-q10-validator',
             '--no-3d-view', '--no-q10-debug', '--validator-ransac-voxel-mm', '1',
             '--validator-ransac-radius-x', '2', '--validator-max-planes', '0',
             '--centroid-radius-mm', '8', '--nms-radius-factor', '.1', '--fusion-score-threshold', '0',
             '--min-neighbors', '3', '--fusion-mode', 'pca_ratio']
    if mode != 'total':
        flags += ['--no-validator-geometry']
    args = parse_args(flags)
    args.post_nms_lambda1_ratio_max = None
    return configure(args)[0]


def fake_q10(points, extraction, cfg, qcfg, **kwargs):
    records = [{'point_index': int(i), 'refined_position_mm': points[i] + [.25, .125, 0.],
                'status': ('ACCEPT', 'REJECT', 'INDETERMINATE')[j % 3], 'keep': j % 3 != 1}
               for j, i in enumerate(extraction['candidate_indices'])]
    return {'records': records, 'kept_indices': np.array([r['point_index'] for r in records if r['keep']], dtype=np.int64),
            'accepted_indices': np.array([r['point_index'] for r in records if r['status'] == 'ACCEPT'], dtype=np.int64)}


class ValidatorSeedTest(unittest.TestCase):
    def test_refine_off_retains_only_accepted_nms_coordinates_without_labeling_them_refined(self):
        scene = np.array([[.123, .2, 0.], [3., 4., 5.]])
        extraction = {'candidate_indices': np.array([0, 1])}
        result = {'refine_enabled': False, 'accepted_indices': np.array([0]), 'records': [
            {'point_index': 0, 'status': 'ACCEPT', 'refined_position_mm': scene[0].copy()},
            {'point_index': 1, 'status': 'REJECT', 'refined_position_mm': scene[1].copy()}]}
        seeds = pipeline.prepare_validator_seeds(scene, extraction, result)
        np.testing.assert_array_equal(seeds['candidate_indices'], [0])
        np.testing.assert_array_equal(seeds['positions_mm'], scene[[0]])
        np.testing.assert_array_equal(seeds['original_positions_mm'], scene[[0]])
        self.assertEqual(seeds['source'], 'maxgap_nms')

    def test_all_three_modes_share_q10_options_and_off_keeps_original_seeds(self):
        for mode in ('detector', 'validator', 'total'):
            default = configure(parse_args(['--mode', mode, '--source', 'realsense']))[0]
            self.assertEqual(default.q10_validator is not None, DEFAULTS.q10_validator_enabled)
            self.assertIsNotNone(config(mode, True).q10_validator)
            self.assertIsNone(config(mode, False).q10_validator)

    def test_only_accepts_supply_refined_centers_and_original_scene_ids(self):
        scene = np.array([[0., 0, 0], [100., 0, 0], [200., 0, 0], [300., 0, 0]])
        extraction = {'candidate_indices': np.array([3, 0, 2])}
        q10 = {'kept_indices': np.array([0, 2]), 'accepted_indices': np.array([0]), 'records': [
            {'point_index': 3, 'status': 'REJECT', 'refined_position_mm': [299., 0, 0]},
            {'point_index': 0, 'status': 'ACCEPT', 'refined_position_mm': [1.25, 0, 0]},
            {'point_index': 2, 'status': 'INDETERMINATE', 'refined_position_mm': [201.5, 0, 0]}]}
        saved = copy.deepcopy(q10)
        seeds = pipeline.prepare_validator_seeds(scene, extraction, q10)
        np.testing.assert_array_equal(seeds['candidate_indices'], [0])
        np.testing.assert_array_equal(seeds['positions_mm'][:, 0], [1.25])
        np.testing.assert_array_equal(seeds['original_positions_mm'], scene[[0]])
        self.assertEqual(seeds['source'], 'q10_refined')
        seeds['positions_mm'][0, 0] = 55.
        np.testing.assert_equal(q10, saved)
        off = pipeline.prepare_validator_seeds(scene, extraction)
        np.testing.assert_array_equal(off['positions_mm'], scene[[3, 0, 2]])
        self.assertEqual(off['source'], 'detector_nms')
        q10['accepted_indices'] = np.empty(0, dtype=np.int64)
        self.assertEqual(pipeline.prepare_validator_seeds(scene, extraction, q10)['positions_mm'].shape, (0, 3))
        for invalid in (np.zeros((4, 2)), np.full((4, 3), np.nan)):
            with self.assertRaises(ValueError):
                pipeline.prepare_validator_seeds(invalid, extraction)

    def test_refined_center_controls_numeric_raw_crop_and_cache_without_scene_mutation(self):
        scene = np.array([[0., 0, 0], [100., 0, 0]])
        raw = np.array([[x, 0., 0.] for x in (0., 1., 8., 9., 10., 12.)])
        snapshot = scene.copy()
        cfg = config()
        context = pipeline.prepare_validator_voxel_context(scene, raw, cfg)
        with patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p, v: p.copy()) as down:
            refined = pipeline.run_validator_pipeline(scene, 0, cfg, voxel_context=context, query_position_mm=[8., 0, 0])
            np.testing.assert_array_equal(refined['roi_raw_point_indices'], [2, 3, 4])
            np.testing.assert_array_equal(refined['query_position_mm'], [8., 0, 0])
            np.testing.assert_array_equal(refined['coarse_query_position_mm'], scene[0])
            np.testing.assert_array_equal(refined['roi_points_mm'][:, 0], [8., 9., 10.])
            np.testing.assert_array_equal(refined['ransac_scene_indices'], [0, 0, 0])
            calls = down.call_count
            pipeline.run_validator_pipeline(scene, 0, cfg, voxel_context=context, query_position_mm=[8., 0, 0])
            self.assertEqual(down.call_count, calls)
            original = pipeline.run_validator_pipeline(scene, 0, cfg, voxel_context=context)
            self.assertGreater(down.call_count, calls)
            np.testing.assert_array_equal(original['roi_raw_point_indices'], [0, 1])
            np.testing.assert_array_equal(original['query_position_mm'], scene[0])
        np.testing.assert_array_equal(scene, snapshot)

    def test_batch_applies_each_center_and_rejects_misaligned_coordinates(self):
        scene = np.array([[0., 0, 0], [100., 0, 0]])
        raw = np.array([[x, 0., 0.] for x in (0., 8., 9., 10., 100., 108., 109., 110.)])
        positions = np.array([[8., 0, 0], [108., 0, 0]])
        cfg = config()
        with patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p, v: p.copy()):
            result = pipeline.run_validator_candidates(scene, [0, 1], cfg, raw_points_mm=raw,
                                                       query_positions_mm=positions, include_debug=False)
        self.assertEqual(result['diagnostics']['failed_count'], 0)
        for i, record in enumerate(result['records']):
            self.assertEqual(record['query_index'], i)
            np.testing.assert_array_equal(record['query_position_mm'], positions[i])
            np.testing.assert_array_equal(record['coarse_query_position_mm'], scene[i])
        for invalid in (np.zeros((1, 3)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                pipeline.run_validator_candidates(scene, [0, 1], cfg, raw_points_mm=raw, query_positions_mm=invalid)

    def test_geometry_stages_receive_the_same_refined_center(self):
        from corner_detector import validator_plane_support, validator_edge_graph
        scene = np.array([[0., 0, 0], [100., 0, 0]])
        raw = np.array([[x, 0., 0.] for x in (0., 1., 8., 9., 10., 12.)])
        with (patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p, v: p.copy()),
              patch.object(validator_plane_support, 'reconstruct_plane_support', wraps=validator_plane_support.reconstruct_plane_support) as support,
              patch.object(validator_edge_graph, 'build_validator_edge_graph', wraps=validator_edge_graph.build_validator_edge_graph) as graph):
            result = pipeline.run_validator_pipeline(scene, 0, config('total'), raw_points_mm=raw,
                                                       query_position_mm=[8., 0, 0], include_debug=False)
        np.testing.assert_array_equal(support.call_args.args[1], [8., 0, 0])
        np.testing.assert_array_equal(graph.call_args.args[1], [8., 0, 0])
        np.testing.assert_array_equal(result['query_position_mm'], [8., 0, 0])
        self.assertTrue(result['geometry_computed'])

    def test_interactive_and_total_both_use_the_same_detector_seed_policy(self):
        x, y = np.meshgrid(np.arange(6.), np.arange(6.))
        scene = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        for enabled in (False, True):
            cfg = config(enabled=enabled)
            with (patch.object(dense_heatmap, 'run_q10_validator', side_effect=fake_q10) as q10,
                  redirect_stdout(StringIO())):
                scores = validator_query.prepare_validator_queries(scene, cfg, raw_points_mm=scene)
            seeds = scores['validator_seeds']
            self.assertEqual(q10.call_count, int(enabled))
            self.assertGreater(len(seeds['candidate_indices']), 0)
            if enabled:
                self.assertLess(len(seeds['candidate_indices']), len(scores['candidate_extraction']['candidate_indices']))
                np.testing.assert_array_equal(seeds['candidate_indices'], scores['q10_validator']['accepted_indices'])
            expected = scene[seeds['candidate_indices']] + ([.25, .125, 0.] if enabled else np.zeros(3))
            np.testing.assert_array_equal(seeds['positions_mm'], expected)
            frozen = {'points_voxel_mm': scene, 'points_raw_mm': scene,
                      'validator_query_indices': seeds['candidate_indices'], 'validator_query_positions_mm': seeds['positions_mm']}
            with (patch('corner_detector.preprocessing.voxel_downsample_3d', side_effect=lambda p, v: p.copy()),
                  patch.object(validator_query, 'prepare_guided_validator_result'), redirect_stdout(StringIO())):
                clicked = validator_query.inspect_validator_query(int(seeds['candidate_indices'][0]), frozen, cfg)
            np.testing.assert_array_equal(clicked['query_position_mm'], expected[0])
            batch = {'records': [], 'final_corners': [], 'final_edges': [],
                     'diagnostics': {'candidate_count': len(seeds['candidate_indices']), 'failed_count': 0}}
            with (patch.object(dense_heatmap, 'run_q10_validator', side_effect=fake_q10),
                  patch.object(total, 'run_validator_candidates', return_value=batch) as validate, redirect_stdout(StringIO())):
                result = total.run_total_mode(scene, config('total', enabled), raw_points_mm=scene)
            np.testing.assert_array_equal(validate.call_args.args[1], seeds['candidate_indices'])
            if enabled:
                np.testing.assert_array_equal(validate.call_args.kwargs['query_positions_mm'], expected)
            else:
                self.assertNotIn('query_positions_mm', validate.call_args.kwargs)
            np.testing.assert_array_equal(result['scores']['validator_seeds']['positions_mm'], expected)

    def test_total_does_not_fall_back_to_rejected_nms_when_no_seed_survives(self):
        x, y = np.meshgrid(np.arange(6.), np.arange(6.))
        scene = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        def reject_all(*args, **kwargs):
            result = fake_q10(*args, **kwargs)
            result['kept_indices'] = np.empty(0, dtype=np.int64)
            result['accepted_indices'] = np.empty(0, dtype=np.int64)
            for record in result['records']:
                record.update(status='REJECT', keep=False)
            return result
        with (patch.object(dense_heatmap, 'run_q10_validator', side_effect=reject_all),
              patch.object(pipeline, 'run_validator_pipeline') as validate, redirect_stdout(StringIO())):
            result = total.run_total_mode(scene, config('total'), raw_points_mm=scene)
        self.assertGreater(result['scores']['candidate_extraction']['candidate_count'], 0)
        self.assertEqual(result['validation']['diagnostics']['candidate_count'], 0)
        self.assertEqual(result['validation']['final_corners'], [])
        validate.assert_not_called()

    def test_total_excludes_indeterminate_even_when_q10_keeps_every_candidate(self):
        x, y = np.meshgrid(np.arange(6.), np.arange(6.))
        scene = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        def indeterminate_all(*args, **kwargs):
            result = fake_q10(*args, **kwargs)
            result['accepted_indices'] = np.empty(0, dtype=np.int64)
            result['kept_indices'] = np.array([r['point_index'] for r in result['records']], dtype=np.int64)
            for record in result['records']:
                record.update(status='INDETERMINATE', keep=True)
            return result
        with (patch.object(dense_heatmap, 'run_q10_validator', side_effect=indeterminate_all),
              patch.object(pipeline, 'run_validator_pipeline') as validate, redirect_stdout(StringIO())):
            result = total.run_total_mode(scene, config('total'), raw_points_mm=scene)
        self.assertGreater(len(result['scores']['q10_validator']['kept_indices']), 0)
        self.assertEqual(result['validation']['diagnostics']['candidate_count'], 0)
        validate.assert_not_called()

    def test_organized_adapter_uses_override_while_retaining_scene_indices(self):
        scene = np.array([[0., 0, 0], [8., 0, 0], [9., 0, 0]])
        extraction = SimpleNamespace(patches=[], diagnostics={'timings_seconds': {'total': 0.}})
        result = as_plane_cluster_analysis(extraction, scene, 0, 2., np.array([1, 2]), query_position_mm=[8., 0, 0])
        np.testing.assert_array_equal(result['query_position_mm'], [8., 0, 0])
        self.assertEqual(result['query_index'], 0)
        np.testing.assert_array_equal(result['roi_point_indices'], [1, 2])

    def test_organized_crop_and_cache_follow_center_for_the_same_scene_id(self):
        scene = np.array([[0., 0, 0], [8., 0, 0], [9., 0, 0]])
        cfg = configure(parse_args(['--mode', 'fine', '--source', 'realsense']))[0]
        cfg = replace(cfg, query_radius_mm=2., plane_edge_graph_enabled=False)
        context = {'organized_points_mm': scene[None, :], 'valid_mask': np.ones((1, 3), dtype=bool),
                   'organized_to_scene_index': np.array([[0, 1, 2]])}
        extraction = SimpleNamespace(patches=[], diagnostics={'timings_seconds': {'total': 0.}})
        with patch.object(pipeline, 'extract_planes', return_value=extraction) as extract:
            result = pipeline.run_validator_pipeline(scene, 0, cfg, organized_context=context, query_position_mm=[8., 0, 0])
            np.testing.assert_array_equal(result['clustering']['roi_point_indices'], [1, 2])
            np.testing.assert_array_equal(extract.call_args.kwargs['valid_mask'], [[False, True, True]])
            pipeline.run_validator_pipeline(scene, 0, cfg, organized_context=context, query_position_mm=[8., 0, 0])
            self.assertEqual(extract.call_count, 1)
            original = pipeline.run_validator_pipeline(scene, 0, cfg, organized_context=context)
            self.assertEqual(extract.call_count, 2)
            np.testing.assert_array_equal(original['clustering']['roi_point_indices'], [0])

    def test_rgb_pick_uses_refined_coordinates_instead_of_original_nms_coordinates(self):
        frozen = {'points_voxel_mm': np.array([[0., 0, 0], [10., 0, 0]]),
                  'validator_query_indices': np.array([0, 1]),
                  'validator_query_positions_mm': np.array([[9., 0, 0], [1., 0, 0]]),
                  'pointcloud_frame': object()}
        selection = SimpleNamespace(point_mm=np.array([9., 0, 0]))
        with (patch.object(validator_query, 'query_from_pixel', return_value=selection),
              patch.object(validator_query, 'inspect_validator_query') as inspect):
            validator_query.inspect_validator_pixel(4, 5, frozen, config())
        self.assertEqual(inspect.call_args.args[0], 0)

    def test_view_and_picker_use_refined_positions_with_original_ids(self):
        scene = np.array([[0., 0, 0], [100., 0, 0], [200., 0, 0]])
        ids, positions = np.array([0, 2]), np.array([[1.25, 0, 0], [201.5, 0, 0]])
        frozen = {'points_voxel_mm': scene, 'validator_query_indices': ids, 'validator_query_positions_mm': positions,
                  'validator_scene_colors': np.ones_like(scene)}
        with (patch.object(manager, 'close_guided_validator_view'), patch.object(manager, 'open_guided_validator_view') as open_view):
            manager.prepare_guided_validator_view(frozen)
        np.testing.assert_array_equal(open_view.call_args.kwargs['query_positions_mm'], positions)
        with patch.object(view3d, 'o3d', new=MagicMock()) as graphics:
            graphics.geometry.PointCloud.side_effect = MagicMock
            graphics.utility.Vector3dVector.side_effect = lambda p: np.array(p, copy=True)
            display = view3d.open_guided_validator_view(scene, ids, np.ones_like(scene), query_positions_mm=positions)
        np.testing.assert_array_equal(display['query_xyz'], positions)
        np.testing.assert_array_equal(display['query_indices'], ids)
        with patch.object(view3d, '_pick_validator_points', return_value=2) as pick:
            self.assertEqual(view3d.pick_guided_validator_query(display, 5, 6), 2)
        np.testing.assert_array_equal(pick.call_args.args[1], positions)
        np.testing.assert_array_equal(pick.call_args.args[2], ids)

    def test_spawned_view_receives_cached_positions_without_detector_recomputation(self):
        positions = np.array([[1.25, 0, 0]])
        connection = Mock()
        def open_view(frozen, _callback):
            np.testing.assert_array_equal(frozen['validator_query_positions_mm'], positions)
            frozen['guided_validator_display'] = None
        with (patch.object(query_ui, 'open_guided_validator_window', side_effect=open_view),
              patch.object(query_ui, 'close_query_views')):
            query_ui.run_guided_validator_view_process(connection, np.zeros((1, 3)), np.array([0]), np.ones((1, 3)),
                                                      query_positions_mm=positions)
        connection.send.assert_not_called()
        connection.close.assert_called_once()

    def test_parent_sends_refined_positions_to_spawned_view(self):
        positions = np.array([[1.25, 0, 0]])
        frozen = {'points_voxel_mm': np.zeros((1, 3)), 'points_raw_mm': np.zeros((1, 3)),
                  'validator_query_indices': np.array([0]), 'validator_query_positions_mm': positions,
                  'validator_ced_scores': {'fusion_score': np.array([.5]), 'fusion_valid_mask': np.array([True])}}
        context = Mock()
        context.Pipe.return_value = Mock(), Mock()
        with (patch('multiprocessing.get_context', return_value=context),
              patch.object(manager, 'close_guided_validator_view'),
              patch.object(manager, 'dense_heatmap_vmax', return_value={'fusion_score': 1.}),
              patch.object(manager, 'dense_heatmap_colors', return_value=np.ones((1, 3)))):
            manager.start_guided_validator_view(frozen)
        np.testing.assert_array_equal(context.Process.call_args.kwargs['args'][-1], positions)


if __name__ == '__main__':
    unittest.main()
