"""Cached spherical-plane display, exact rose counts and read-only diagnostics."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector import q10_validator
from corner_detector.settings import DetectorConfig, Q10ValidatorConfig
from corner_detector.visualization import manager, view3d


def plane_points():
    a, b = np.meshgrid(np.arange(1., 7.), np.arange(1., 7.))
    zero = np.zeros(a.size)
    return np.vstack((np.zeros(3), np.eye(3)*3., np.column_stack((zero, a.ravel(), b.ravel())),
                      np.column_stack((a.ravel(), zero, b.ravel())), np.column_stack((a.ravel(), b.ravel(), zero))))


class LiteVisualizationTest(unittest.TestCase):
    def run_lite(self, *, max_planes=3, points=None, tail_min_bin_count=2):
        points = plane_points() if points is None else points
        cfg = Q10ValidatorConfig(refine_enabled=True, voxel_mm=.01, radius_mm=12., min_q10=-1.,
                                 sphere_tolerance_deg=1., sphere_max_planes=max_planes,
                                 gap_tail_min_bin_count=tail_min_bin_count, refine_cc_enabled=False)
        extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .9}]}
        with redirect_stdout(StringIO()):
            return q10_validator.run_q10_validator(np.zeros((1, 3)), extraction,
                DetectorConfig(nms_radius_mm=.0001), cfg, raw_points_mm=points)

    def test_voxel_colors_cover_every_plane_and_preserve_overlaps_without_summary(self):
        result = self.run_lite()
        record = result['records'][0]
        saved = copy.deepcopy(record)
        payload = manager.prepare_lite_validator_debug(None, record, result)['record']
        payload = dict(payload, antipodal=None)
        data = view3d.q10_debug_voxel_primitives(payload, 0)
        vector_rows = np.searchsorted(data['analysis_indices'], record['vector_analysis_indices'])
        count = data['plane_memberships'].sum(axis=1)
        kept = data['retained_memberships'].any(axis=1)
        palette = np.array([(255, 130, 62), (81, 218, 129), (159, 130, 255)]) / 255.
        for index, plane in enumerate(record['spherical_planes']['planes']):
            np.testing.assert_array_equal(data['plane_memberships'][vector_rows, index], plane['inlier_mask'])
            test = record['plane_antipodal']['tests'][index]
            members = vector_rows[plane['inlier_mask']]
            np.testing.assert_array_equal(data['retained_memberships'][members, index], test['kept_mask'])
            unique = (count == 1) & data['plane_memberships'][:, index] & kept
            self.assertTrue(unique.any())
            np.testing.assert_allclose(data['point_colors'][unique], np.tile(palette[index], (unique.sum(), 1)))
        overlap = count > 1
        self.assertTrue(overlap.any())
        expected = np.ones((overlap.sum(), 3))
        expected[~kept[overlap]] *= .3
        np.testing.assert_array_equal(data['point_colors'][overlap], expected)
        np.testing.assert_array_equal(data['analysis_indices'][data['scored_mask']], record['vector_analysis_indices'])
        np.testing.assert_equal(record, saved)

    def test_projected_exclusions_dim_their_own_plane_color(self):
        result = self.run_lite()
        record = copy.deepcopy(result['records'][0])
        for test in record['plane_antipodal']['tests']:
            test['kept_mask'][0] = False
        saved = copy.deepcopy(record)
        for plane in view3d.q10_spherical_plane_primitives(record):
            keep = plane['test']['kept_mask']
            np.testing.assert_allclose(plane['projected_colors'][~keep],
                                       np.tile(.3*plane['color'], ((~keep).sum(), 1)))
            np.testing.assert_allclose(plane['projected_colors'][keep],
                                       np.tile(plane['color'], (keep.sum(), 1)))
        np.testing.assert_equal(record, saved)

    def test_projected_memberships_colors_and_gap_evidence_are_preserved(self):
        result = self.run_lite()
        record = result['records'][0]
        saved = copy.deepcopy(record)
        planes = view3d.q10_spherical_plane_primitives(record)
        self.assertEqual(len(planes), 3)
        self.assertGreater(np.count_nonzero(record['spherical_planes']['membership_count'] > 1), 0)
        for index, plane in enumerate(planes):
            stored = record['spherical_planes']['planes'][index]
            np.testing.assert_allclose(plane['projected_points'], stored['projected_directions']*(1.11+.025*index))
            np.testing.assert_allclose(np.linalg.norm(plane['projected_points'], axis=1), 1.11+.025*index)
            np.testing.assert_allclose(plane['circle'] @ stored['normal'], 0., atol=1e-12)
            np.testing.assert_array_equal(plane['angles_deg'], stored['angles_deg'])
            self.assertEqual(plane['test']['plane_index'], index)
            self.assertEqual(plane['test']['raw_count'], len(stored['angles_deg']))
        self.assertEqual(len({tuple(p['color']) for p in planes}), 3)
        np.testing.assert_equal(record, saved)

    def test_rose_counts_cached_arcs_and_cards_match_for_one_two_three_planes(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                result = self.run_lite(max_planes=count)
                record = result['records'][0]
                saved = copy.deepcopy(record)
                with (patch.object(q10_validator, 'segment_origin_spherical_planes', side_effect=AssertionError('display refit')),
                      patch.object(q10_validator, 'evaluate_plane_max_gap', side_effect=AssertionError('display gate'))):
                    display = manager.prepare_lite_validator_debug(None, record, result)
                    display['figure'].canvas.draw()
                    display['set_step'](0)
                    display['figure'].canvas.draw()
                roses = [a for a in display['figure'].axes if a.name == 'polar']
                self.assertEqual(len(roses), count)
                for rose, plane in zip(roses, view3d.q10_spherical_plane_primitives(record)):
                    expected = np.histogram(plane['angles_deg'], bins=36, range=(0., 360.))[0]
                    np.testing.assert_array_equal([bar.get_height() for bar in rose.patches[:36]], expected)
                    self.assertEqual(int(expected.sum()), plane['test']['raw_count'])
                    test = plane['test']
                    self.assertAlmostEqual(rose.lines[0].get_xdata()[0], np.radians(test['arc_start_deg']))
                    self.assertAlmostEqual(rose.lines[0].get_xdata()[-1],
                                           np.radians(test['arc_start_deg'] + test['observed_span_deg']))
                    self.assertEqual(len(rose.lines), 1)
                all_text = '\n'.join(t.get_text() for a in display['figure'].axes for t in a.texts)
                for test in record['plane_antipodal']['tests']:
                    self.assertIn(f"SPAN {test['observed_span_deg']:.1f}°", all_text)
                    self.assertIn(f"keep {test['kept_count']}/{test['raw_count']}", all_text)
                np.testing.assert_equal(record, saved)
                display['figure'].clear()

    def test_unknown_planes_are_not_presented_as_pass_and_empty_evidence_renders(self):
        for minimum in (2, 100):
            result = self.run_lite(tail_min_bin_count=minimum)
            record = result['records'][0]
            if minimum == 2:
                record = copy.deepcopy(record)
                record.update(spherical_planes=None, plane_antipodal=None, antipodal=None,
                              status='INDETERMINATE', q10_value=None, antipodal_reject=None,
                              reason='spherical_plane_fit_failed')
            display = manager.prepare_lite_validator_debug(None, record, result)
            display['figure'].canvas.draw()
            texts = '\n'.join(t.get_text() for a in display['figure'].axes for t in a.texts)
            figure_text = '\n'.join(t.get_text() for t in display['figure'].texts)
            self.assertNotIn('retained', figure_text)
            self.assertNotIn('min 12 points', figure_text)
            self.assertIn('UNKNOWN', texts)
            if minimum == 100:
                self.assertEqual(len([a for a in display['figure'].axes if a.name == 'polar']), 3)
                self.assertFalse(any(t.get_text() == 'PASS' for a in display['figure'].axes for t in a.texts))
            else:
                self.assertEqual(view3d.q10_spherical_plane_primitives(record), [])
            display['figure'].clear()

    def test_mm_axes_have_equal_scale_for_planar_point_clouds(self):
        theta = np.radians(np.linspace(10., 80., 40))
        points = np.vstack((np.zeros(3), np.column_stack((5*np.cos(theta), 5*np.sin(theta), np.zeros(40)))))
        result = self.run_lite(points=points)
        display = manager.prepare_lite_validator_debug(None, result['records'][0], result)
        for index in (0, 3):
            axis = display['figure'].axes[index]
            spans = [np.ptp(limit) for limit in (axis.get_xlim(), axis.get_ylim(), axis.get_zlim())]
            np.testing.assert_allclose(spans, np.repeat(spans[0], 3))
        display['figure'].clear()

    def test_native_kept_arcs_follow_cached_final_origin_without_opening_a_window(self):
        from pointcloud_inputs.visualization import require_open3d
        o3d = require_open3d()
        result = self.run_lite()
        record = result['records'][0]
        saved = copy.deepcopy(record)
        display = manager.prepare_lite_validator_debug(None, record, result)
        vis = Mock()
        vis.create_window.return_value = True
        with patch.object(o3d.visualization, 'VisualizerWithKeyCallback', return_value=vis):
            native = view3d.open_q10_debug_view(display['record'])
            try:
                geometries = [call.args[0] for call in vis.add_geometry.call_args_list]
                self.assertEqual(sum(isinstance(g, o3d.geometry.PointCloud) for g in geometries), 1)
                lines = [g for g in geometries if isinstance(g, o3d.geometry.LineSet)]
                center = record['refined_position_mm']
                for plane in view3d.q10_spherical_plane_primitives(record):
                    expected = center + .8*record['radius_mm']*plane['kept_arc']
                    matches = [line for line in lines if np.asarray(line.points).shape == expected.shape
                               and np.allclose(line.points, expected)]
                    self.assertEqual(len(matches), 1)
                    self.assertEqual(len(matches[0].lines), len(expected)-1)
                    np.testing.assert_allclose(np.asarray(matches[0].colors),
                                               np.tile(plane['color'], (len(expected)-1, 1)))
            finally:
                native['close']()
                display['figure'].clear()
        np.testing.assert_equal(record, saved)
