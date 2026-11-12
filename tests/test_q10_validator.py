"""Q10 reference equations, cache/CC ordering, index spaces and read-only debug."""

import copy
from dataclasses import replace
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from corner_detector.refiner import Q10PositionRefiner, mean_direction_q10
from corner_detector.connectivity import radius_components
from corner_detector import q10_validator as q10
from corner_detector.settings import DEFAULTS, DetectorConfig, Q10ValidatorConfig, configure, parse_args
from corner_detector.visualization import manager, plots, query_ui, view3d


def corner_points():
    a, b = np.meshgrid(np.arange(9.), np.arange(9.))
    zero = np.zeros(a.size)
    return np.unique(np.vstack((np.column_stack((zero, a.ravel(), b.ravel())),
                                np.column_stack((a.ravel(), zero, b.ravel())),
                                np.column_stack((a.ravel(), b.ravel(), zero)))), axis=0)


class Q10NumericTest(unittest.TestCase):
    def test_unit_mean_is_not_distance_weighted_and_self_is_excluded(self):
        points = np.vstack((np.zeros(3), np.eye(3) * 2., np.eye(3) * 7., np.eye(3) * 19., np.eye(3) * 50.))
        value, axis = mean_direction_q10(np.zeros(3), points)
        np.testing.assert_allclose(axis, np.ones(3) / np.sqrt(3), atol=1e-14)
        self.assertAlmostEqual(value, 1 / np.sqrt(3))
        with self.assertRaises(ValueError):
            mean_direction_q10(np.zeros(3), np.vstack((np.eye(3), -np.eye(3))))

    def test_refinement_matches_one_reference_search_in_fixed_nms_ball(self):
        from pointcloud_inputs.visualization import require_open3d
        require_open3d()
        import corner_lab_open3d_cone as reference
        points = corner_points()
        start = int(np.argmin(np.linalg.norm(points - [0., 5., 4.], axis=1)))
        refiner = Q10PositionRefiner(points, 8., 2.5)
        actual = refiner.refine(start)
        result = reference.discrete_refine_q10(points[start], start, points, np.arange(len(points)), 8., 2.5)
        path = [start] if result.refined_point_index == start else [start, result.refined_point_index]
        np.testing.assert_array_equal(actual['analysis_indices'], path)
        self.assertTrue(actual['converged'])
        self.assertTrue(np.all(np.diff(actual['q10_values']) > 1e-9))
        self.assertTrue(np.all(np.linalg.norm(np.diff(actual['positions_mm'], axis=0), axis=1) <= 2.5 + 1e-8))
        for position, expected in zip(actual['positions_mm'], actual['q10_values']):
            ids = refiner.tree.query_ball_point(position, 8. + 1e-8)
            self.assertAlmostEqual(reference.analyze_cone(position, points[ids]).cosine_q10, expected)

    def test_increasing_scores_do_not_move_search_window_beyond_original_nms_ball(self):
        points = np.column_stack((np.arange(5.), np.zeros((5, 2))))
        refiner = Q10PositionRefiner(points, 8., 1.25)
        def evaluate(index, **_kwargs):
            return .1 * (index + 1), 100, np.array([1., 0., 0.])
        with patch.object(refiner, 'evaluate', side_effect=evaluate) as response:
            trace = refiner.refine(0)
        np.testing.assert_array_equal(trace['analysis_indices'], [0, 1])
        np.testing.assert_allclose(trace['q10_values'], [.1, .2])
        self.assertEqual({call.args[0] for call in response.call_args_list}, {0, 1})
        self.assertEqual(len(trace['tested_positions']), 1)
        self.assertTrue(np.all(np.linalg.norm(trace['positions_mm'] - points[0], axis=1) <= 1.25))

    def test_search_is_centered_on_original_nms_not_snapped_analysis_point(self):
        points = np.array([[.9, 0., 0.], [1.5, 0., 0.], [.25, 0., 0.]])
        refiner = Q10PositionRefiner(points, 8., 1.)
        scores = [.1, .99, .2]
        with patch.object(refiner, 'evaluate', side_effect=lambda i, **_kwargs: (scores[i], 100, np.array([1., 0., 0.]))):
            trace = refiner.refine(0, center_mm=np.zeros(3))
        np.testing.assert_array_equal(trace['analysis_indices'], [0, 2])
        self.assertTrue(np.all(np.linalg.norm(trace['positions_mm'], axis=1) <= 1.))

    def test_response_cache_is_shared_between_seeds_but_not_between_radii(self):
        points = corner_points()
        refiner = Q10PositionRefiner(points, 8., 2.5)
        first = refiner.refine(30)
        misses = refiner.cache_misses
        second = refiner.refine(30)
        self.assertEqual(refiner.cache_misses, misses)
        self.assertGreater(refiner.cache_hits, 0)
        np.testing.assert_array_equal(first['positions_mm'], second['positions_mm'])
        other = Q10PositionRefiner(points, 10., 2.5)
        other.refine(30)
        self.assertIsNot(refiner.cache, other.cache)

    def test_q10_uses_only_spherical_query_component_and_caches_cc(self):
        connected = np.vstack((np.zeros(3), *(np.eye(3) * radius for radius in range(1, 5))))
        distant = np.column_stack((-np.arange(10., 21.), np.zeros((11, 2))))
        points = np.vstack((connected, distant))
        before, _ = mean_direction_q10(np.zeros(3), points)
        self.assertAlmostEqual(before, -7 / 9)
        refiner = Q10PositionRefiner(points, 25., 2.5, cc_factor=1.1)
        with patch('corner_detector.refiner._components', wraps=radius_components) as cc:
            value, count, axis = refiner.evaluate(0)
            repeated = refiner.evaluate(0)
        cc.assert_called_once()
        self.assertAlmostEqual(value, 1 / np.sqrt(3))
        self.assertEqual(count, 13)
        self.assertEqual(repeated[0], value)
        np.testing.assert_allclose(axis, np.ones(3) / np.sqrt(3))
        support = refiner.support_cache[0]
        np.testing.assert_array_equal(support['component_analysis_indices'], np.arange(13))
        self.assertEqual(support['roi_count'], 24)
        self.assertEqual(support['median_nn_mm'], 1.)
        self.assertEqual(support['cc_radius_mm'], 1.1)
        other = Q10PositionRefiner(points, 25., 2.5, cc_factor=12.)
        self.assertAlmostEqual(other.evaluate(0)[0], -7 / 9)
        self.assertEqual(other.evaluate(0)[1], 24)

    def test_invalid_q10_cc_spacing_is_cached_without_a_graph(self):
        refiner = Q10PositionRefiner(np.zeros((12, 3)), 8., 2.5)
        with patch('corner_detector.refiner._components') as cc:
            self.assertIsNone(refiner.evaluate(0)[0])
            self.assertIsNone(refiner.evaluate(0)[0])
        cc.assert_not_called()
        self.assertEqual((refiner.cache_misses, refiner.cache_hits), (1, 1))
        self.assertEqual(refiner.support_cache[0]['median_nn_mm'], 0.)
        self.assertEqual(refiner.support_cache[0]['cc_radius_mm'], 0.)
        np.testing.assert_array_equal(refiner.support_cache[0]['component_analysis_indices'], [])

    def test_q10_cc_off_uses_full_radius_without_nn_or_graph(self):
        connected = np.vstack((np.zeros(3), *(np.eye(3) * r for r in range(1, 5))))
        distant = np.column_stack((-np.arange(10., 21.), np.zeros((11, 2))))
        refiner = Q10PositionRefiner(np.vstack((connected, distant)), 25., .1, cc_factor=1.1, cc_enabled=False)
        with (patch('scipy.spatial.cKDTree', side_effect=AssertionError('disabled CC computed NN')),
              patch('corner_detector.refiner._components', side_effect=AssertionError('disabled CC ran a graph'))):
            value, count, _ = refiner.evaluate(0)
        self.assertAlmostEqual(value, -7 / 9)
        self.assertEqual(count, 24)
        self.assertTrue(np.isnan(refiner.support_cache[0]['cc_radius_mm']))


class Q10PipelineTest(unittest.TestCase):
    def run_pipeline(self, cfg=None, scene=None):
        raw = corner_points()
        scene = np.zeros((2, 3)) if scene is None else scene
        extraction = {'candidate_indices': np.arange(len(scene)),
                      'candidates': [{'point_index': i, 'score': .4} for i in range(len(scene))]}
        snapshot = copy.deepcopy(extraction)
        with (patch.object(q10, 'voxel_downsample_3d', return_value=raw.copy()), redirect_stdout(StringIO())):
            result = q10.run_q10_validator(scene, extraction, DetectorConfig(nms_radius_mm=2.5),
                                          Q10ValidatorConfig(refine_enabled=True, radius_mm=8.) if cfg is None else cfg, raw_points_mm=raw)
        np.testing.assert_array_equal(extraction['candidate_indices'], snapshot['candidate_indices'])
        self.assertEqual(extraction['candidates'], snapshot['candidates'])
        return result

    def test_refine_off_keeps_exact_nms_and_ignores_min_gate_without_q10_calls(self):
        theta = np.radians(np.linspace(20., 70., 40))
        raw = np.column_stack((5*np.cos(theta), 5*np.sin(theta), np.zeros(40)))
        scene = np.array([[.123, .2, 0.]])
        extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .4}]}
        self.assertGreater(np.linalg.norm(raw-scene[0], axis=1).min(), .0001)
        saved_scene, saved_raw = scene.copy(), raw.copy()
        results = []
        for minimum in (-1., 1.):
            cfg = Q10ValidatorConfig(refine_enabled=False, min_q10=minimum, voxel_mm=.001,
                                     radius_mm=8., refine_cc_enabled=True, sphere_tolerance_deg=1.,
                                     gap_link_deg=20., gap_edge_span_deg=160., gap_tail_enabled=False)
            with (patch.object(Q10PositionRefiner, 'refine', side_effect=AssertionError('refine OFF')),
                  patch.object(Q10PositionRefiner, 'evaluate', side_effect=AssertionError('Q10 OFF')),
                  patch('corner_detector.refiner._components', side_effect=AssertionError('CC OFF')),
                  patch.object(q10, 'voxel_downsample_3d', return_value=raw.copy()), redirect_stdout(StringIO())):
                result = q10.run_q10_validator(scene, extraction, DetectorConfig(nms_radius_mm=.0001),
                                               cfg, raw_points_mm=raw)
            record = result['records'][0]
            np.testing.assert_array_equal(record['position'], scene[0])
            np.testing.assert_array_equal(record['refined_position_mm'], scene[0])
            self.assertIsNone(record['refined_analysis_index'])
            self.assertFalse(record['refine_enabled'])
            self.assertFalse(record['refine_cc_enabled'])
            self.assertIsNone(record['q10_value'])
            self.assertIsNone(record['q10_reject'])
            self.assertTrue(record['antipodal_evaluated'])
            self.assertEqual(record['status'], 'ACCEPT')
            self.assertEqual(record['trace']['stop_reason'], 'refine_disabled')
            self.assertFalse(record['trace']['converged'])
            self.assertEqual(record['trace']['positions_mm'].shape, (1, 3))
            self.assertEqual(record['trace']['analysis_indices'].size, 0)
            self.assertTrue(np.isnan(record['trace']['q10_values']).all())
            self.assertEqual(result['q10_early_reject_count'], 0)
            self.assertEqual(result['cache_misses'], 0)
            self.assertEqual(result['final_evaluation_count'], 1)
            results.append(result)
        np.testing.assert_equal(results[0]['records'][0]['plane_antipodal'],
                                results[1]['records'][0]['plane_antipodal'])
        np.testing.assert_array_equal(scene, saved_scene)
        np.testing.assert_array_equal(raw, saved_raw)

    def test_refine_off_still_rejects_wide_span_and_keeps_insufficient_evidence_unknown(self):
        for angles, status in ((np.arange(0., 360., 2.), 'REJECT'), (np.linspace(20., 70., 6), 'INDETERMINATE')):
            theta = np.radians(angles)
            raw = np.column_stack((5*np.cos(theta), 5*np.sin(theta), np.zeros(len(theta))))
            cfg = Q10ValidatorConfig(refine_enabled=False, min_q10=1., voxel_mm=.001, radius_mm=8.,
                                     sphere_tolerance_deg=1., gap_link_deg=20., gap_edge_span_deg=160.)
            extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .4}]}
            with (patch.object(Q10PositionRefiner, 'refine', side_effect=AssertionError('refine OFF')),
                  patch.object(q10, 'voxel_downsample_3d', return_value=raw.copy()), redirect_stdout(StringIO())):
                result = q10.run_q10_validator(np.zeros((1, 3)), extraction, DetectorConfig(nms_radius_mm=.0001),
                                               cfg, raw_points_mm=raw)
            record = result['records'][0]
            self.assertEqual(record['status'], status)
            self.assertEqual(record['keep'], status != 'REJECT')
            self.assertIsNone(record['q10_reject'])
            self.assertTrue(record['antipodal_evaluated'])

    def test_refine_off_caches_by_original_coordinate_and_debug_remains_read_only(self):
        scene = np.array([[.1, 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
        cfg = Q10ValidatorConfig(radius_mm=8., refine_enabled=False, min_q10=1.)
        with patch.object(q10, '_final_evidence', wraps=q10._final_evidence) as final:
            result = self.run_pipeline(cfg, scene)
        self.assertEqual(final.call_count, 2)
        self.assertEqual(result['final_evaluation_count'], 2)
        np.testing.assert_array_equal(final.call_args_list[0].args[0], scene[0])
        np.testing.assert_array_equal(final.call_args_list[1].args[0], scene[2])
        record = result['records'][0]
        saved = copy.deepcopy(record)
        with patch.object(Q10PositionRefiner, 'evaluate', side_effect=AssertionError('display Q10 OFF')):
            display = manager.prepare_lite_validator_debug(None, record, result)
            display['figure'].canvas.draw()
        labels = '\n'.join(text.get_text() for text in display['figure'].texts)
        self.assertIn('Q10 refine OFF / min gate OFF', labels)
        self.assertIsNone(view3d.q10_direction_primitives(record, 0))
        np.testing.assert_equal(record, saved)
        display['figure'].clear()

    def test_q10_cc_precedes_spherical_ransac_and_final_evidence_is_cached(self):
        events = []
        original_refine = Q10PositionRefiner.refine
        original_fit = q10.segment_origin_spherical_planes

        def refine(instance, index, **options):
            trace = original_refine(instance, index, **options)
            self.assertTrue(trace['converged'])
            events.append('converged')
            return trace

        def spherical_cc(*args):
            events.append('q10_cc')
            return radius_components(*args)

        def fit(*args, **kwargs):
            self.assertEqual(events[-1], 'converged')
            events.append('ransac')
            return original_fit(*args, **kwargs)

        with (patch.object(Q10PositionRefiner, 'refine', refine),
              patch('corner_detector.refiner._components', spherical_cc),
              patch.object(q10, 'segment_origin_spherical_planes', fit)):
            result = self.run_pipeline(Q10ValidatorConfig(refine_enabled=True, radius_mm=8., refine_cc_enabled=True))
        self.assertGreater(len(events), 3)
        self.assertTrue(all(event == 'q10_cc' for event in events[:-3]))
        self.assertEqual(events[-3:], ['converged', 'ransac', 'converged'])
        self.assertEqual(result['final_evaluation_count'], 1)
        self.assertEqual(result['step_mm'], 2.5)
        self.assertGreater(result['cache_hits'], 0)
        for index, record in enumerate(result['records']):
            self.assertEqual(record['point_index'], index)
            np.testing.assert_array_equal(record['position'], np.zeros(3))
            self.assertFalse(record['cc_evaluated'])

    def test_pipeline_passes_unsnapped_scene_nms_coordinate_to_refiner(self):
        original_refine = Q10PositionRefiner.refine
        centers = []
        def refine(instance, index, **options):
            centers.append(options['center_mm'].copy())
            return original_refine(instance, index, **options)
        scene = np.array([[.2, 5.1, 4.15]])
        with patch.object(Q10PositionRefiner, 'refine', refine):
            result = self.run_pipeline(scene=scene)
        np.testing.assert_array_equal(centers, scene)
        for record in result['records']:
            self.assertLessEqual(np.linalg.norm(record['refined_position_mm'] - record['position']), 2.5 + 1e-8)
            self.assertLessEqual(len(record['trace']['positions_mm']), 2)

    def test_no_analysis_point_in_nms_ball_keeps_original_position_without_cc(self):
        scene = np.array([[100., 100., 100.]])
        with patch.object(q10, '_final_evidence') as final:
            result = self.run_pipeline(scene=scene)
        final.assert_not_called()
        record = result['records'][0]
        self.assertEqual(record['status'], 'INDETERMINATE')
        self.assertEqual(record['reason'], 'no_analysis_point_in_nms_ball')
        self.assertIsNone(record['refined_analysis_index'])
        np.testing.assert_array_equal(record['refined_position_mm'], scene[0])
        np.testing.assert_array_equal(result['kept_indices'], [0])

    def test_pipeline_uses_cached_gap_verdict_without_recalculating_it(self):
        for verdict, expected in (('REJECT', 'reject_count'), ('CORNER', 'accept_count')):
            evidence = {'plane_antipodal': {'status': verdict}, 'cc_evaluated': False}
            with patch.object(q10, '_final_evidence', return_value=(evidence, None)):
                result = self.run_pipeline()
            self.assertEqual(result[expected], 2)

    def test_q10_boundary_and_independent_rejection_reasons(self):
        original_refine = Q10PositionRefiner.refine
        cases = ((.15, .01, 'ACCEPT', []), (.14, .01, 'REJECT', ['q10_below_threshold']),
                 (.2, .08, 'REJECT', ['angular_span']),
                 (.14, .08, 'REJECT', ['q10_below_threshold']),
                 (np.nan, .01, 'INDETERMINATE', []), (np.nan, .08, 'INDETERMINATE', []),
                 (.14, None, 'REJECT', ['q10_below_threshold']), (.2, None, 'INDETERMINATE', []))
        for q10_value, score, status, reasons in cases:
            with self.subTest(q10=q10_value, score=score):
                def refine(instance, index, **options):
                    trace = original_refine(instance, index, **options)
                    trace['q10_values'][-1] = q10_value
                    return trace
                evidence = {'plane_antipodal': {'status': 'REJECT' if score == .08 else 'CORNER'}}
                reason = 'insufficient_final_directions' if score is None else None
                with (patch.object(q10, '_final_evidence', return_value=(evidence, reason)) as final,
                      patch.object(Q10PositionRefiner, 'refine', refine)):
                    result = self.run_pipeline(Q10ValidatorConfig(refine_enabled=True, radius_mm=8., min_q10=.15))
                self.assertEqual(final.call_count, int(np.isfinite(q10_value) and q10_value >= .15))
                for record in result['records']:
                    self.assertEqual(record['status'], status)
                    self.assertEqual(record['rejection_reasons'], reasons)
                    self.assertEqual(record['keep'], status != 'REJECT')

    def test_final_quality_reuses_q10_even_with_negative_one_percent_outliers(self):
        x = np.r_[0., np.arange(1., 98.), -np.arange(1., 4.)]
        points = np.column_stack((x, np.zeros((len(x), 2))))
        cfg = Q10ValidatorConfig(refine_enabled=True, radius_mm=100., voxel_mm=1., min_q10=.01, refine_cc_enabled=False)
        extraction = {'candidate_indices':np.array([0]),'candidates':[{'point_index':0,'score':.4}]}
        with (patch.object(q10,'voxel_downsample_3d',return_value=points),
              patch.object(np,'quantile',wraps=np.quantile) as quantiles, redirect_stdout(StringIO())):
            result = q10.run_q10_validator(np.zeros((1,3)),extraction,DetectorConfig(nms_radius_mm=.1),cfg,
                                           raw_points_mm=points)
        record = result['records'][0]
        self.assertEqual([call.args[1] for call in quantiles.call_args_list],[.1])
        self.assertEqual(np.quantile(np.r_[np.ones(97),-np.ones(3)],.01),-1.)
        self.assertEqual(record['trace']['q10_values'][-1],1.)
        self.assertEqual(record['q10_value'],1.)
        self.assertEqual(record['q10_reject'],False)
        self.assertEqual(record['status'],'INDETERMINATE')
        self.assertEqual(record['rejection_reasons'],[])
        self.assertNotIn('q1',record)


    def test_invalid_q10_never_runs_cc_and_keeps_original_scene_id(self):
        raw = np.eye(3)
        extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .4}]}
        with (patch.object(q10, 'voxel_downsample_3d', return_value=raw), patch.object(q10, '_final_evidence') as final,
              redirect_stdout(StringIO())):
            result = q10.run_q10_validator(np.zeros((1, 3)), extraction, DetectorConfig(),
                                          Q10ValidatorConfig(refine_enabled=True), raw_points_mm=raw)
        final.assert_not_called()
        self.assertEqual(result['indeterminate_count'], 1)
        np.testing.assert_array_equal(result['kept_indices'], [0])


    def test_q10_receives_original_nms_and_supplies_primary_result(self):
        from corner_detector import dense_heatmap as dense
        points = corner_points()
        cfg = DetectorConfig(show_3d=False, centroid_radius_mm=8., pca_radius_mm=12., bilateral_radius_mm=8.,
                             nms_radius_mm=2.5, fusion_score_threshold=0.,
                             q10_validator=Q10ValidatorConfig(refine_enabled=True))
        refined = {'kept_indices': np.array([], dtype=np.int64), 'accepted_indices': np.array([], dtype=np.int64)}
        with (patch.object(dense, 'run_q10_validator', return_value=refined) as new,
              redirect_stdout(StringIO())):
            result = dense.run_dense_heatmap_mode(points, cfg, raw_points_mm=points, print_candidates=False)
        original = result['candidate_extraction']
        self.assertGreater(len(original['candidate_indices']), 0)
        self.assertIs(new.call_args.args[1], original)
        self.assertIs(result['lite_validator'], refined)
        self.assertNotIn('scale_lite_validator', result)

    def test_q10_marker_colors_and_refined_coordinates_after_scale_backend_removal(self):
        statuses = ('ACCEPT', 'REJECT', 'INDETERMINATE')
        records = [{'status': status, 'refined_position_mm': np.array([10.*i, 2., 3.])}
                   for i, status in enumerate(statuses)]
        for only_accepted in (False, True):
            with patch.object(view3d, 'o3d', new=MagicMock()) as graphics:
                view3d._lite_marker_mesh({'records': records}, .5, accepted_only=only_accepted)
                sphere = graphics.geometry.TriangleMesh.create_sphere.return_value
                expected = records[:1] if only_accepted else records
                self.assertEqual(sphere.translate.call_count, len(expected))
                for call, record in zip(sphere.translate.call_args_list, expected):
                    np.testing.assert_array_equal(call.args[0], record['refined_position_mm'])
                colors = [[.1, .9, .25]] if only_accepted else [[.1, .9, .25], [1., .08, .12], [1., .65, .05]]
                self.assertEqual([call.args[0] for call in sphere.paint_uniform_color.call_args_list], colors)

    def test_debug_is_read_only_and_step_plot_renders(self):
        result = self.run_pipeline()
        record = result['records'][0]
        trace = copy.deepcopy(record['trace'])
        with patch('corner_detector.refiner._components', side_effect=AssertionError('debug recomputed Q10 CC')):
            display = manager.prepare_lite_validator_debug(None, record, result)
            display['figure'].canvas.draw()
            display['set_step'](0)
            display['figure'].canvas.draw()
        self.assertEqual(len([a for a in display['figure'].axes if a.name == 'polar']),
                         len(record['spherical_planes']['planes']))
        for key in trace:
            if isinstance(trace[key], np.ndarray):
                np.testing.assert_array_equal(record['trace'][key], trace[key])
        for step, ids in enumerate(trace['support_analysis_indices']):
            payload = display['record']
            np.testing.assert_array_equal(payload['step_q10_points_mm'][step], result['context']['analysis_points_mm'][ids])
            units, cosines = plots.q10_step_directions(payload, step)
            self.assertEqual(len(ids), trace['neighbor_counts'][step])
            self.assertEqual(len(units), len(ids) - 1)
            self.assertAlmostEqual(np.quantile(cosines, .1), trace['q10_values'][step])

    def test_debug_disabled_has_identical_outcomes(self):
        on = self.run_pipeline()
        off = self.run_pipeline(Q10ValidatorConfig(refine_enabled=True, radius_mm=8., debug=False))
        self.assertNotIn('context', off)
        np.testing.assert_array_equal(on['kept_indices'], off['kept_indices'])
        for a, b in zip(on['records'], off['records']):
            np.testing.assert_array_equal(a['trace']['positions_mm'], b['trace']['positions_mm'])
            self.assertEqual(a['antipodal']['observed_span_deg'], b['antipodal']['observed_span_deg'])


    def test_voxel_display_deduplicates_analysis_ids_and_uses_stored_masks(self):
        record = {
            'step_analysis_indices': [np.array([10, 30, 40])],
            'step_points_mm': [np.array([[1.,0.,0.], [3.,0.,0.], [4.,0.,0.]])],
            'roi_analysis_indices': np.array([20, 30, 40, 50]),
            'roi_analysis_points_mm': np.array([[2.,0.,0.], [3.,0.,0.], [4.,0.,0.], [5.,0.,0.]]),
            'vector_analysis_indices': np.array([20, 30, 50]),
            'roi_points_mm': np.full((1000, 3), 999.),
            'spherical_planes': {'planes': [{'inlier_mask': np.array([True, True, True])},
                                             {'inlier_mask': np.array([False, True, False])}]},
            'plane_antipodal': {'tests': [{'plane_index': 0, 'kept_mask': np.array([True, False, False])},
                                          {'plane_index': 1, 'kept_mask': np.array([True])}]}}
        snapshot = copy.deepcopy(record)
        data = view3d.q10_debug_voxel_primitives(record, 0)
        np.testing.assert_array_equal(data['analysis_indices'], [10, 20, 30, 40, 50])
        np.testing.assert_array_equal(data['points_mm'][:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(data['point_classes'], [0, 2, 2, 0, 3])
        np.testing.assert_array_equal(data['plane_memberships'].sum(axis=1), [0, 1, 2, 0, 1])
        np.testing.assert_allclose(data['point_colors'][1], np.array([255, 130, 62])/255.)
        np.testing.assert_array_equal(data['point_colors'][2], [1., 1., 1.])
        np.testing.assert_allclose(data['point_colors'][4], .3*np.array([255, 130, 62])/255.)
        self.assertEqual(data['retained_count'], 2)
        self.assertEqual(data['discarded_count'], 1)
        self.assertEqual(np.count_nonzero(data['scored_mask']), 3)
        all_discarded = copy.deepcopy(record)
        all_discarded['plane_antipodal']['tests'][1]['kept_mask'][:] = False
        alternate = view3d.q10_debug_voxel_primitives(all_discarded, 0)
        np.testing.assert_array_equal(alternate['point_colors'][2], [.3, .3, .3])
        self.assertEqual(alternate['point_classes'][2], 3)
        np.testing.assert_equal(record, snapshot)

    def test_debug_payload_matches_computation_voxels_and_exact_support_counts(self):
        result = self.run_pipeline(Q10ValidatorConfig(refine_enabled=True, radius_mm=8., gap_edge_span_deg=160.))
        record = result['records'][0]
        snapshot = copy.deepcopy(record)
        with patch('corner_detector.refiner._components', side_effect=AssertionError('display recomputed CC')):
            display = manager.prepare_lite_validator_debug(None, record, result)
            payload = display['record']
            self.assertEqual(result['gap_edge_span_deg'], 160.)
            self.assertEqual(payload['gap_edge_span_deg'], 160.)
            self.assertNotIn('threshold', payload)
            self.assertNotIn('roi_points_mm', payload)
            for step in range(len(record['trace']['positions_mm'])):
                data = view3d.q10_debug_voxel_primitives(payload, step)
                ids = data['analysis_indices']
                self.assertEqual(len(ids), len(np.unique(ids)))
                np.testing.assert_array_equal(data['points_mm'], result['context']['analysis_points_mm'][ids])
                np.testing.assert_array_equal(ids[data['scored_mask']], record['vector_analysis_indices'])
                for index, test in enumerate(record['plane_antipodal']['tests']):
                    self.assertEqual(data['plane_memberships'][:, index].sum(), test['raw_count'])
                    self.assertEqual(data['retained_memberships'][:, index].sum(), test['kept_count'])
                self.assertEqual(np.count_nonzero(data['scored_mask']), len(record['vector_analysis_indices']))
                plane = record['spherical_planes']['planes'][record['antipodal']['plane_index']]
                self.assertEqual(np.count_nonzero(plane['inlier_mask']), record['antipodal']['raw_count'])
            display['figure'].canvas.draw()
        np.testing.assert_equal(record, snapshot)

    def test_native_debug_renders_one_voxel_cloud_per_step_without_raw_overlay(self):
        from pointcloud_inputs.visualization import require_open3d
        o3d = require_open3d()
        result = self.run_pipeline(Q10ValidatorConfig(refine_enabled=True, radius_mm=8.))
        payload = manager.prepare_lite_validator_debug(None, result['records'][0], result)['record']
        vis = Mock()
        vis.create_window.return_value = True
        with patch.object(o3d.visualization, 'VisualizerWithKeyCallback', return_value=vis):
            view = view3d.open_q10_debug_view(payload)
            try:
                for step in range(len(payload['trace']['positions_mm'])):
                    vis.add_geometry.reset_mock()
                    view['set_step'](step)
                    clouds = [c.args[0] for c in vis.add_geometry.call_args_list
                              if isinstance(c.args[0], o3d.geometry.PointCloud)]
                    self.assertEqual(len(clouds), 1)
                    data = view3d.q10_debug_voxel_primitives(payload, step)
                    np.testing.assert_array_equal(np.asarray(clouds[0].points), data['points_mm'])
                    np.testing.assert_array_equal(np.asarray(clouds[0].colors), data['point_colors'])
            finally:
                view['close']()

    def test_final_marker_pick_uses_refined_coordinate_and_status_colors(self):
        vis = Mock()
        camera = vis.get_view_control.return_value.convert_to_pinhole_camera_parameters.return_value
        camera.extrinsic = np.eye(4)
        camera.intrinsic.intrinsic_matrix = np.eye(3)
        record = {'position': np.array([100., 100., 1.]), 'refined_position_mm': np.array([0., 0., 1.]),
                  'kind': 'q10_antipodal', 'status': 'REJECT'}
        self.assertIs(view3d.pick_lite_validator_candidate(vis, [record], 0., 0.,
                      position_key='refined_position_mm'), record)
        self.assertIsNone(view3d.pick_lite_validator_candidate(vis, [record], 0., 0.))
        self.assertEqual(view3d._lite_record_color(record), [1., .08, .12])
        record['status'] = 'ACCEPT'
        self.assertEqual(view3d._lite_record_color(record), [.1, .9, .25])

    def test_all_final_statuses_have_markers_at_refined_positions(self):
        from pointcloud_inputs.visualization import require_open3d
        o3d = require_open3d()
        records = [{'kind': 'q10_antipodal', 'status': status, 'position': np.array([100., 100., 100.]),
                    'refined_position_mm': np.array([3. * i, 0., 0.])}
                   for i, status in enumerate(('ACCEPT', 'REJECT', 'INDETERMINATE'))]
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=.5, resolution=6)
        mesh = view3d._lite_marker_mesh({'kind': 'q10_antipodal', 'records': records}, .5)
        vertices, colors = np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)
        n = len(marker.vertices)
        self.assertEqual(len(vertices), 3*n)
        for i, record in enumerate(records):
            np.testing.assert_allclose(vertices[i*n:(i+1)*n].mean(axis=0), record['refined_position_mm'], atol=1e-14)
            np.testing.assert_allclose(colors[i*n:(i+1)*n], np.tile(view3d._lite_record_color(record), (n, 1)))

    def test_rgb_overlay_mesh_contains_only_green_accepts_at_final_coordinates(self):
        from pointcloud_inputs.visualization import require_open3d
        o3d = require_open3d()
        records = [{'kind': 'q10_antipodal', 'status': status, 'position': np.array([100., 100., 100.]),
                    'refined_position_mm': np.array([3. * i, 0., 0.])}
                   for i, status in enumerate(('REJECT', 'ACCEPT', 'INDETERMINATE'))]
        result = {'kind': 'q10_antipodal', 'records': records}
        snapshot = copy.deepcopy(result)
        mesh = view3d._lite_marker_mesh(result, .5, accepted_only=True)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.5, resolution=6)
        self.assertEqual(len(mesh.vertices), len(sphere.vertices))
        np.testing.assert_allclose(np.asarray(mesh.vertices).mean(axis=0), records[1]['refined_position_mm'], atol=1e-14)
        np.testing.assert_allclose(np.asarray(mesh.vertex_colors), np.tile([.1, .9, .25], (len(mesh.vertices), 1)))
        np.testing.assert_equal(result, snapshot)

    def test_rgb_result_view_preserves_rgb_cloud_and_accept_only_refresh(self):
        rgb_points = np.arange(12, dtype=float).reshape(4, 3)
        rgb_colors = np.linspace(0, 1, 12).reshape(4, 3)
        for count in (0, 1):
            with (self.subTest(accepted=count), patch.object(view3d, 'o3d', new=MagicMock()) as graphics,
                  patch.object(view3d, '_lite_marker_mesh') as build,
                  patch.object(view3d, 'dense_heatmap_colors', side_effect=AssertionError('RGB replaced by heatmap'))):
                graphics.geometry.PointCloud.side_effect = Mock
                graphics.utility.Vector3dVector.side_effect = lambda values: np.array(values, copy=True)
                result = {'kind': 'q10_antipodal', 'accept_count': count, 'candidate_count': 3,
                          'reject_count': 2-count, 'indeterminate_count': 1}
                build.return_value.is_empty.return_value = count == 0
                state = {}
                vis = view3d.open_lite_validator_results(np.zeros((1, 3)), {'lite_validator': result}, .5,
                    state=state, rgb_points_mm=rgb_points, rgb_colors_rgb=rgb_colors)
                cloud = vis.add_geometry.call_args_list[0].args[0]
                np.testing.assert_array_equal(cloud.points, rgb_points)
                np.testing.assert_array_equal(cloud.colors, rgb_colors)
                self.assertIn('Final RealSense RGB point cloud', vis.create_window.call_args.kwargs['window_name'])
                self.assertEqual(vis.add_geometry.call_count, 1+count)
                build.assert_called_once_with(result, .5, accepted_only=True)
                self.assertTrue(state['accepted_only'])
                view3d.refresh_lite_markers(state, result, .5)
                self.assertTrue(build.call_args.kwargs['accepted_only'])

    def test_only_mode_forwards_registered_rgb_to_final_view_and_picks_accepts(self):
        result = self.run_pipeline()
        result['records'][0].update(status='ACCEPT', accept=True, keep=True)
        result['records'][1].update(status='REJECT', accept=False, keep=False)
        result.update(accept_count=1, reject_count=1)
        points = np.zeros((2, 3))
        rgb_points = np.arange(12, dtype=float).reshape(4, 3)
        rgb_colors = np.arange(12, dtype=np.uint8).reshape(4, 3)
        scores = {'lite_validator': result, 'candidate_extraction': {
            'candidate_indices': np.arange(2), 'nms_radius_mm': 2.5, 'candidate_count': 2,
            'fusion_score_threshold': .1}}
        for name in ('ced_score', 'bilateral_ced_score', 'pca_ratio_raw', 'pca_ratio_saturated', 'fusion_score'):
            scores[name] = np.full(2, .4)
        for name in ('ced_valid_mask', 'pca_valid_mask', 'fusion_valid_mask'):
            scores[name] = np.ones(2, dtype=bool)
        with (patch.object(view3d, 'o3d', new=MagicMock()) as graphics,
              patch.object(view3d, 'open_q10_convergence_results') as intermediate,
              patch.object(view3d, 'open_lite_validator_results') as final,
              patch.object(view3d, 'synchronize_dense_camera_views', return_value=(None, None)),
              patch.object(query_ui, 'bind_lite_validator_click') as bind):
            graphics.geometry.PointCloud.side_effect = Mock
            graphics.utility.Vector3dVector.side_effect = lambda values: np.array(values, copy=True)
            for view in (graphics.visualization.VisualizerWithKeyCallback.return_value,
                         intermediate.return_value, final.return_value):
                view.poll_events.return_value = False
            view3d.show_dense_score_heatmaps(points, scores, only_nms=True, rgb_points_mm=rgb_points,
                rgb_colors_rgb=rgb_colors, lite_debug_callback=lambda record: None)
            graphics.visualization.Visualizer.assert_not_called()
            graphics.visualization.VisualizerWithKeyCallback.assert_called_once()
            np.testing.assert_array_equal(final.call_args.kwargs['rgb_points_mm'], rgb_points)
            np.testing.assert_allclose(final.call_args.kwargs['rgb_colors_rgb'], rgb_colors/255.)
            self.assertEqual(len(bind.call_args.args[1]), 1)
            self.assertTrue(all(r['status']=='ACCEPT' for r in bind.call_args.args[1]))
            self.assertEqual(bind.call_args.kwargs['position_key'], 'refined_position_mm')
        np.testing.assert_array_equal(rgb_colors, np.arange(12, dtype=np.uint8).reshape(4, 3))

    def test_native_worker_dispatches_cached_step_changes(self):
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.side_effect = [('step', 0), ('close', None)]
        view = {'poll': Mock(return_value=True), 'close': Mock(), 'set_step': Mock()}
        with (patch.object(view3d, 'open_q10_debug_view', return_value=view), patch.object(query_ui.time, 'sleep')):
            query_ui.run_q10_debug_view_process(connection, {'kind': 'q10_antipodal'})
        view['set_step'].assert_called_once_with(0)
        view['close'].assert_called_once()


class Q10SettingsTest(unittest.TestCase):
    def config(self, *flags, mode='detector'):
        return configure(parse_args(['--source', 'realsense', '--mode', mode, *flags]))[0]

    def test_q10_switch_and_default_primary(self):
        cfg = self.config()
        self.assertIsNotNone(cfg.q10_validator)
        self.assertFalse(hasattr(cfg, 'lite_validator'))
        cfg = self.config('--no-q10-validator')
        self.assertIsNone(cfg.q10_validator)
        for mode in ('validator', 'total'):
            self.assertEqual(self.config(mode=mode).q10_validator, self.config().q10_validator)
        for mode in ('fine', 'ball', 'extremity256'):
            self.assertIsNone(getattr(self.config(mode=mode), 'q10_validator', None))
        with self.assertRaisesRegex(ValueError, 'detector'):
            self.config('--q10-validator', mode='fine')

    def test_scale_pca_config_cli_and_exclusive_numerics_are_removed(self):
        from corner_detector import settings, edge_suppression
        self.assertFalse(hasattr(settings, 'LiteValidatorConfig'))
        self.assertFalse(hasattr(edge_suppression, 'StableRadiusPCA'))
        for name in ('enabled', 'debug', 'min_scale_mm', 'max_radius_ratio', 'num_scales',
                     'ced_drop_threshold', 'direction_change_deg', 'pca_ratio_threshold'):
            self.assertFalse(hasattr(DEFAULTS, 'lite_validator_' + name))
        for flags in (['--lite-validator'], ['--no-lite-validator'], ['--lite-validator-debug'],
                      ['--lite-validator-min-scale-mm', '60'], ['--lite-validator-max-radius-ratio', '3'],
                      ['--lite-validator-num-scales', '8'], ['--lite-validator-ced-drop-threshold', '.15'],
                      ['--lite-validator-direction-change-deg', '20'], ['--lite-validator-pca-ratio-threshold', '.35']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(['--mode', 'detector', '--source', 'realsense', *flags])

    def test_validation(self):
        for values in ({'voxel_mm': 0}, {'radius_mm': float('nan')}, {'cc_factor': -1}, {'gap_edge_span_deg': 361},
                       {'debug': 1},
                       {'refine_cc_enabled': 1}, {'refine_enabled': 1}, {'refine_enabled': 'False'},
                       {'min_q10': -1.01}, {'min_q10': 1.01}, {'min_q10': np.nan}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(Q10ValidatorConfig(refine_enabled=True), **values).validate()

    def test_q10_cc_switch_and_quality_threshold_are_configurable(self):
        default = self.config().q10_validator
        self.assertEqual(default.refine_cc_enabled, DEFAULTS.q10_refine_cc_enabled)
        self.assertEqual(default.min_q10, DEFAULTS.q10_min_q10)
        for refine_cc in (False, True):
            flag = '--q10-refine-cc' if refine_cc else '--no-q10-refine-cc'
            cfg = self.config(flag, '--q10-min-q10', '.25').q10_validator
            self.assertEqual(cfg.refine_cc_enabled, refine_cc)
            self.assertEqual(cfg.min_q10, .25)

    def test_refine_switch_is_independent_of_validator_master_and_shared_across_modes(self):
        self.assertEqual(self.config().q10_validator.refine_enabled, DEFAULTS.q10_refine_enabled)
        for mode in ('detector', 'validator', 'total', 'new-validator'):
            for enabled in (True, False):
                flag = '--q10-refine' if enabled else '--no-q10-refine'
                cfg = self.config(flag, mode=mode).q10_validator
                self.assertIsNotNone(cfg)
                self.assertEqual(cfg.refine_enabled, enabled)
                self.assertEqual(cfg.min_q10, DEFAULTS.q10_min_q10)
            self.assertIsNone(self.config('--no-q10-validator', '--no-q10-refine', mode=mode).q10_validator)

    def test_cleaned_gap_settings_and_cli_propagate_to_config(self):
        values = dict(link_deg=18., edge_span_deg=170.,
                      tail_bin_width_deg=12., tail_density_fraction=.4, tail_min_bin_count=3)
        flags = ['--no-q10-gap-tail']
        for name, value in values.items():
            flags.extend(('--q10-gap-' + name.replace('_', '-'), str(value)))
        cfg = self.config(*flags).q10_validator
        self.assertFalse(cfg.gap_tail_enabled)
        for name, value in values.items():
            self.assertEqual(getattr(cfg, 'gap_' + name), value)
        self.assertEqual(cfg.min_q10, self.config().q10_validator.min_q10)


if __name__ == '__main__':
    unittest.main()
