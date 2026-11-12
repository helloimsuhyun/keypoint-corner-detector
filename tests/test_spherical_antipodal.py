"""Lab-equation parity, cleaned per-plane MaxGap and Lite integration (headless)."""
import ast
import copy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from types import ModuleType
import sys
import unittest
from unittest.mock import patch
import numpy as np
from corner_detector import q10_validator as lite, spherical_antipodal as sphere
from corner_detector.settings import DEFAULTS, DetectorConfig, Q10ValidatorConfig, configure, parse_args
from corner_detector.visualization import manager, view3d


def arc(angles, radius=5.):
    theta = np.radians(angles)
    return radius * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(theta))))


class SphericalAntipodalTest(unittest.TestCase):
    def test_numerics_match_lab_source_without_gui_import(self):
        source = Path('corner_lab_open3d_cone.py').read_text().split('\ndef main():')[0]
        names = {'SphericalPlane', 'SphericalPlaneAnalysis', '_sphere_plane_basis',
                 '_spherical_plane_residual_deg', '_refit_origin_plane', 'segment_origin_spherical_planes',
                 'PlaneGapTest', 'PlaneGapValidation', '_dominant_angular_component',
                 'trim_sparse_arc_ends', 'evaluate_plane_max_gap'}
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        reference = ModuleType('spherical_lab_numeric_reference')
        reference.__dict__.update(np=np, dataclass=dataclass)
        with patch.dict(sys.modules, {reference.__name__: reference}):
            exec(compile(tree, '<lab numerical functions>', 'exec'), reference.__dict__)
        units = np.vstack((arc(np.arange(0., 360., 3.), 1.),
                           arc(np.arange(0., 100., 2.), 1.)[:, [2, 0, 1]],
                           np.random.default_rng(37).normal(size=(15, 3))))
        expected = reference.segment_origin_spherical_planes(units, tolerance_deg=7.)
        actual = sphere.segment_origin_spherical_planes(units, tolerance_deg=7.)
        np.testing.assert_equal(asdict(actual), asdict(expected))
        np.testing.assert_equal(asdict(sphere.evaluate_plane_max_gap(actual)),
                                asdict(reference.evaluate_plane_max_gap(expected, min_points=2,
                                                                       min_retained_fraction=1e-12)))
        for angles in (np.r_[np.arange(91.), np.arange(100., 171., 10.)],
                       np.mod(np.arange(91.) + 320., 360.),
                       np.r_[np.arange(0., 61., 2.), [170., 175.]], np.arange(6.)):
            plane = replace(actual.planes[0], angles_deg=angles)
            fit = replace(actual, planes=(plane,))
            for enabled in (False, True):
                options = dict(tail_filter_enabled=enabled)
                np.testing.assert_equal(asdict(sphere.evaluate_plane_max_gap(fit, **options)),
                                        asdict(reference.evaluate_plane_max_gap(fit, min_points=2,
                                                                               min_retained_fraction=1e-12, **options)))

    def test_circular_component_joins_seam_and_discards_opposite_group(self):
        angles = np.array([170., 355., 0., 160., 10., 350., 5.])
        keep, components = sphere._dominant_angular_component(angles, 20.)
        np.testing.assert_array_equal(keep, [False, True, True, False, True, True, True])
        self.assertEqual(components, 2)
        keep, _ = sphere._dominant_angular_component([0., 20., 100., 105.], 20.)
        np.testing.assert_array_equal(keep, [False, False, True, True])

    def test_sparse_tails_trim_but_interior_sparse_bins_remain(self):
        angles = np.r_[np.arange(91.), np.arange(100., 171., 10.)]
        dominant = np.ones(len(angles), dtype=bool)
        kept, raw_span, bins, removed, threshold, median = sphere.trim_sparse_arc_ends(angles, dominant)
        np.testing.assert_array_equal(angles[kept], np.arange(90.))
        self.assertEqual((raw_span, bins, removed, threshold, median), (170., 8, 9, 5, 10.))
        unfiltered = sphere.trim_sparse_arc_ends(angles, dominant, enabled=False)
        np.testing.assert_array_equal(unfiltered[0], dominant)
        # A low-count middle bin is preserved when both ends have enough support.
        interior = np.array([1., 2., 3., 15., 21., 22., 23.])
        np.testing.assert_array_equal(sphere.trim_sparse_arc_ends(interior, np.ones(7, dtype=bool))[0], True)
        wrapped = np.mod(angles + 320., 360.)
        np.testing.assert_array_equal(sphere.trim_sparse_arc_ends(wrapped, dominant)[0], kept)

    def test_tail_filter_changes_span_and_verdict(self):
        fit = sphere.segment_origin_spherical_planes(arc(np.arange(0., 100., 2.)), k=1)
        angles = np.r_[np.arange(91.), np.arange(100., 171., 10.)]
        fit = replace(fit, planes=(replace(fit.planes[0], angles_deg=angles),))
        cleaned = sphere.evaluate_plane_max_gap(fit)
        self.assertEqual(cleaned.status, 'CORNER')
        self.assertEqual(cleaned.tests[0].observed_span_deg, 89.)
        self.assertEqual(cleaned.tests[0].raw_span_deg, 170.)
        self.assertEqual(cleaned.tests[0].tail_removed_count, 9)
        self.assertEqual(sphere.evaluate_plane_max_gap(fit, tail_filter_enabled=False).status, 'REJECT')

    def test_small_minority_arc_is_classified_without_count_or_fraction_gates(self):
        fit = sphere.segment_origin_spherical_planes(arc(np.arange(0., 100., 2.)), k=1)
        angles = np.r_[np.arange(6.), np.arange(90., 331., 30.)]
        fit = replace(fit, planes=(replace(fit.planes[0], angles_deg=angles),))
        result = sphere.evaluate_plane_max_gap(fit)
        test = result.tests[0]
        self.assertEqual((test.kept_count, test.raw_count, test.observed_span_deg), (6, 15, 5.))
        self.assertTrue(test.reliable)
        self.assertEqual(result.status, 'CORNER')
        for angles, options, status, span in (
                ([25.], {}, 'CORNER', 0.),
                (np.arange(0., 161., 40.), dict(max_link_gap_deg=45., tail_filter_enabled=False), 'REJECT', 160.),
                (np.arange(0., 31., 10.), dict(tail_min_bin_count=100), 'UNKNOWN', 0.)):
            with self.subTest(angles=angles):
                sample = replace(fit, planes=(replace(fit.planes[0], angles_deg=np.asarray(angles)),))
                result = sphere.evaluate_plane_max_gap(sample, **options)
                self.assertEqual(result.status, status)
                self.assertEqual(result.tests[0].observed_span_deg, span)

    def test_full_circle_rejects_quadrant_passes_with_expected_span(self):
        for angles, status in ((np.arange(0., 360., 2.), 'REJECT'), (np.arange(5., 86., 2.), 'CORNER')):
            fit = sphere.segment_origin_spherical_planes(arc(angles), tolerance_deg=.01)
            result = sphere.evaluate_plane_max_gap(fit)
            self.assertEqual(len(fit.planes), 1)
            self.assertEqual(result.status, status)
            np.testing.assert_allclose(fit.planes[0].normal, [0., 0., 1.], atol=1e-12)
            t = result.tests[0]
            self.assertEqual(t.raw_count, len(angles))
            self.assertEqual(t.kept_count, len(angles))
            self.assertAlmostEqual(t.observed_span_deg, float(angles[-1]-angles[0]))
            self.assertAlmostEqual(t.max_gap_deg + t.observed_span_deg, 360.)

    def test_intersections_keep_overlapping_memberships(self):
        a = np.arange(5., 86., 2.)
        units = np.vstack((arc(a, 1.), arc(a, 1.)[:, [2, 0, 1]], arc(a, 1.)[:, [1, 2, 0]], np.eye(3)))
        fit = sphere.segment_origin_spherical_planes(units, tolerance_deg=.01)
        self.assertEqual(len(fit.planes), 3)
        np.testing.assert_array_equal(fit.membership_count[-3:], [2, 2, 2])

    def test_reject_beats_insufficient_plane_and_equality_rejects(self):
        fit = sphere.segment_origin_spherical_planes(arc(np.arange(0., 360., 2.), 1.), k=1)
        span = sphere.evaluate_plane_max_gap(fit).tests[0].observed_span_deg
        self.assertEqual(sphere.evaluate_plane_max_gap(fit, edge_span_deg=span).status, 'REJECT')
        sparse = replace(fit.planes[0], angles_deg=np.arange(6.))
        self.assertEqual(sphere.evaluate_plane_max_gap(replace(fit, planes=(sparse, fit.planes[0]))).status, 'REJECT')
        self.assertEqual(sphere.evaluate_plane_max_gap(replace(fit, planes=(sparse,))).status, 'CORNER')
        empty = replace(sparse, angles_deg=np.array([]))
        for planes in ((empty,), ()):
            self.assertEqual(sphere.evaluate_plane_max_gap(replace(fit, planes=planes)).status, 'UNKNOWN')

    def run_lite(self, angles, **options):
        raw = np.vstack((np.zeros(3), arc(np.asarray(angles))))
        cfg = Q10ValidatorConfig(refine_enabled=True, voxel_mm=.001, radius_mm=8., refine_cc_enabled=False,
                                 min_q10=-1., sphere_tolerance_deg=7., **options)
        extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .9}]}
        with redirect_stdout(StringIO()):
            return lite.run_q10_validator(np.zeros((1, 3)), extraction, DetectorConfig(nms_radius_mm=.0001),
                                          cfg, raw_points_mm=raw)

    def test_pipeline_status_masks_indices_and_no_final_cc(self):
        cases = ((np.linspace(-90., 90., 91), 'REJECT'),
                 (np.r_[np.linspace(-20., 20., 40), np.linspace(160., 200., 12)], 'ACCEPT'),
                 (np.linspace(20., 70., 40), 'ACCEPT'), (np.linspace(20., 70., 10), 'INDETERMINATE'))
        for angles, status in cases:
            with patch('corner_detector.refiner._components', side_effect=AssertionError('final spatial CC')):
                result = self.run_lite(angles)
            r = result['records'][0]
            self.assertEqual(r['status'], status)
            self.assertFalse(r['cc_evaluated'])
            self.assertIsNone(r['median_nn_mm'])
            self.assertEqual(r['point_index'], 0)
            np.testing.assert_array_equal(r['position'], np.zeros(3))
            anti = r['antipodal']
            if anti is not None:
                for mask, count in (('retained_mask', 'kept_count'), ('discarded_mask', 'discarded_count')):
                    self.assertEqual(np.count_nonzero(anti[mask]), anti[count])
                    self.assertEqual(len(anti[mask]), len(r['vector_analysis_indices']))
                self.assertEqual(anti['raw_count'], len(angles))
                self.assertFalse(any(name in anti for name in ('axis', 'score', 'positive_count', 'negative_count')))
                self.assertEqual(anti['mode'], 'sphere_max_gap')
            np.testing.assert_array_equal(result['kept_indices'], [] if status == 'REJECT' else [0])

    def test_q10_trace_does_not_depend_on_final_angular_parameters(self):
        angles = np.linspace(20., 70., 40)
        spherical = self.run_lite(angles)['records'][0]
        other = self.run_lite(angles, sphere_max_planes=1, gap_link_deg=15.,
                              sphere_iterations=50, gap_edge_span_deg=180., gap_tail_enabled=False)['records'][0]
        np.testing.assert_equal(spherical['trace'], other['trace'])
        self.assertEqual(spherical['q10_value'], other['q10_value'])

    def test_debug_is_cached_read_only_and_renders(self):
        result = self.run_lite(np.linspace(20., 70., 40))
        r = result['records'][0]
        saved = copy.deepcopy(r)
        with (patch.object(lite, 'segment_origin_spherical_planes', side_effect=AssertionError('debug refit')),
              patch.object(lite, 'evaluate_plane_max_gap', side_effect=AssertionError('debug retest'))):
            display = manager.prepare_lite_validator_debug(None, r, result)
            display['figure'].canvas.draw()
            display['set_step'](0)
            display['figure'].canvas.draw()
        self.assertEqual(len([a for a in display['figure'].axes if a.name == 'polar']),
                         len(r['spherical_planes']['planes']))
        data = view3d.q10_debug_voxel_primitives(display['record'], 0)
        self.assertEqual(data['retained_count'], r['antipodal']['kept_count'])
        np.testing.assert_equal(r, saved)
        display['figure'].clear()

    def test_fit_failure_remains_unknown_and_q10_failure_still_rejects(self):
        for minimum, status in ((-1., 'INDETERMINATE'), (1., 'REJECT')):
            raw = np.vstack((np.zeros(3), arc(np.linspace(20., 70., 40))))
            cfg = Q10ValidatorConfig(refine_enabled=True, voxel_mm=.001, radius_mm=8., min_q10=minimum,
                                     refine_cc_enabled=False)
            extraction = {'candidate_indices': np.array([0]), 'candidates': [{'point_index': 0, 'score': .9}]}
            with (patch.object(lite, 'segment_origin_spherical_planes', side_effect=np.linalg.LinAlgError),
                  redirect_stdout(StringIO())):
                result = lite.run_q10_validator(np.zeros((1, 3)), extraction, DetectorConfig(nms_radius_mm=.0001),
                                               cfg, raw_points_mm=raw)
            record = result['records'][0]
            self.assertEqual(record['status'], status)
            self.assertIsNone(record['antipodal_reject'])
            self.assertIsNone(record['antipodal'])

    def test_settings_cli_resolves_all_new_fields_and_rejects_invalid_values(self):
        values = {'max_planes': 2, 'tolerance_deg': 9., 'iterations': 50, 'min_support': 15,
                  'min_fraction': .1, 'seed': 17}
        flags = ['--source', 'realsense', '--mode', 'detector']
        for name, value in values.items():
            flags.extend(('--q10-sphere-' + name.replace('_', '-'), str(value)))
        cfg = configure(parse_args(flags))[0].q10_validator
        self.assertFalse(hasattr(cfg, 'antipodal_mode'))
        self.assertEqual(cfg.gap_edge_span_deg, DEFAULTS.q10_gap_edge_span_deg)
        for name, value in values.items():
            self.assertEqual(getattr(cfg, 'sphere_'+name), value)
        for values in ({'sphere_max_planes': 4}, {'sphere_max_planes': True}, {'sphere_iterations': 9},
                       {'sphere_min_support': 2}, {'sphere_min_support': 3.5}, {'sphere_min_fraction': 1.1},
                       {'sphere_seed': -1}, {'sphere_seed': False}, {'sphere_tolerance_deg': 0},
                       {'sphere_tolerance_deg': 45}, {'gap_link_deg': 0}, {'gap_link_deg': 180},
                       {'gap_link_deg': np.nan}, {'gap_edge_span_deg': 0}, {'gap_edge_span_deg': 361},
                       {'gap_tail_enabled': 1}, {'gap_tail_bin_width_deg': 0}, {'gap_tail_bin_width_deg': 91},
                       {'gap_tail_density_fraction': -.1}, {'gap_tail_density_fraction': 1.1},
                       {'gap_tail_min_bin_count': 0}, {'gap_tail_min_bin_count': False}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()

    def test_removed_modes_have_no_config_cli_or_numerical_entry_points(self):
        for name in ('axis_count', 'antipodal_cc_enabled', 'tube_radius_mm', 'gap_factor', 'antipodal_mode',
                     'beta_deg', 'min_side_points', 'antipodal_threshold', 'sphere_axis_step_deg',
                     'gap_min_points', 'gap_min_retained_fraction'):
            self.assertFalse(hasattr(DEFAULTS, 'q10_' + name))
            self.assertFalse(hasattr(Q10ValidatorConfig(refine_enabled=True), name))
        for flags in (['--q10-gap-min-points', '12'], ['--q10-gap-min-retained-fraction', '.6'],
                      ['--q10-beta-deg', '15'], ['--q10-min-side-points', '3'],
                      ['--q10-antipodal-threshold', '.05'], ['--q10-threshold', '.05'],
                      ['--q10-sphere-axis-step-deg', '2'], ['--q10-axis-count', '642'], ['--q10-antipodal-cc'], ['--no-q10-antipodal-cc'],
                      ['--q10-tube-radius-mm', '4'], ['--q10-gap-factor', '2.5'],
                      ['--q10-antipodal-mode', 'cone'], ['--q10-antipodal-mode', 'tube'],
                      ['--q10-antipodal-mode', 'projection_continuity'], ['--q10-antipodal-mode', 'sphere_ransac']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(['--source', 'realsense', '--mode', 'detector', *flags])
        for name in ('antipodal_support', 'antipodal_tube_support', 'antipodal_axes', '_projection_evidence', 'evaluate_plane_antipodal'):
            self.assertFalse(hasattr(lite, name))
        self.assertFalse(Path('corner_detector/antipodal_support.py').exists())
