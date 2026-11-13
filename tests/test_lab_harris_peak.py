"""Normal Harris numerics and read-only inspection of cached NMS neighborhoods."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.harris3d import compute_harris3d, normal_harris_response
from corner_detector.visualization import manager, plots, view3d
from test_corner_lab_roi_thresholds import comparison
from test_corner_lab_ui import lab, make_window


class Harris3DTest(unittest.TestCase):
    def test_normal_second_moment_equation_and_sign_rotation_invariance(self):
        self.assertAlmostEqual(normal_harris_response(np.tile([0., 0., 1.], (6, 1))), 0.)
        self.assertAlmostEqual(normal_harris_response(np.tile(np.eye(3)[:2], (3, 1))), 0.)
        normals = np.tile(np.eye(3), (2, 1))
        self.assertAlmostEqual(normal_harris_response(normals), 1./27.)
        normals[::2] *= -1.
        rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        self.assertAlmostEqual(normal_harris_response(normals @ rotation), 1./27.)

    def test_planar_zero_collinear_undefined_queries_scale_and_cancellation(self):
        xyz = np.array([(x, y, 0.) for x in range(5) for y in range(5)])
        before = xyz.copy()
        scores = compute_harris3d(xyz, 2.)
        np.testing.assert_allclose(scores, 0., atol=1e-14)
        subset = np.array([0, 12, 24])
        np.testing.assert_allclose(compute_harris3d(xyz*100., 200., query_indices=subset), scores[subset], atol=1e-14)
        line = np.column_stack((np.arange(6.), np.zeros((6, 2))))
        self.assertTrue(np.isnan(compute_harris3d(line, 3.)).all())
        self.assertIsNone(compute_harris3d(xyz, 2., is_cancelled=lambda: True))
        np.testing.assert_array_equal(xyz, before)
        for radius in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_harris3d(xyz, radius)
        for ids in ([25], [-1], [1.5]):
            with self.assertRaises(ValueError):
                compute_harris3d(xyz, 2., query_indices=ids)

    def test_attach_harris_uses_own_positive_nms_and_preserves_other_fields(self):
        result = comparison()
        scores = np.array([0., 0., 0., .03, 0., 0., 0.])
        with patch.object(lab, 'compute_harris3d', return_value=scores):
            updated = lab.with_harris3d(result)
        np.testing.assert_array_equal(updated.harris_maxima, [3])
        for key, value in vars(result).items():
            if key not in ('harris_scores', 'harris_maxima'):
                self.assertIs(getattr(updated, key), value)
        with patch.object(lab, 'compute_harris3d', return_value=np.full(7, 1e-17)):
            self.assertEqual(len(lab.with_harris3d(result).harris_maxima), 0)

    def test_local_and_full_entry_points_share_harris_without_truncating_local_support(self):
        grid = np.array([(x, y, 0.) for x in range(0, 19, 3) for y in range(0, 19, 3)])
        xyz = np.unique(np.vstack((grid, grid[:, [0, 2, 1]], grid[:, [2, 0, 1]])), axis=0)
        common = dict(ced_radius_mm=10., pca_radius_mm=15., voxel_mm=3., min_neighbors=5, ced_cutoff=.05)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)
        np.testing.assert_allclose(full.harris_scores, compute_harris3d(xyz, 15.))
        query_ids = np.flatnonzero(np.linalg.norm(xyz, axis=1) <= 20.)
        np.testing.assert_allclose(local.harris_scores, compute_harris3d(xyz, 15., query_indices=query_ids))
        for result in (full, local):
            self.assertEqual(result.harris_scores.shape, (len(result.positions_mm),))
            self.assertTrue((result.harris_scores[result.harris_maxima] > 1e-12).all())
        common['ced_cutoff'] = 100.
        for result in (lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=5., **common),
                       lab.compute_local_fusion_pca_comparison(xyz, np.zeros(3), view_radius_mm=20., **common)):
            self.assertTrue(np.isnan(result.fusion_scores).all())
            self.assertTrue(np.isfinite(result.harris_scores).any())


class NmsPeakTest(unittest.TestCase):
    def test_exact_neighbor_statistics_exclude_self_include_boundary_and_ignore_nonfinite(self):
        xyz = np.column_stack(([0., .5, 1., 1.001, 0.], np.zeros((5, 2))))
        values = np.array([10., 2., 4., 100., np.nan])
        saved = values.copy()
        evidence = plots.nms_peak_statistics(xyz, values, 0, 1.)
        np.testing.assert_array_equal(evidence['neighbor_indices'], [1, 2])
        self.assertEqual(evidence['rank'], 1)
        self.assertEqual(evidence['percentile'], 100.)
        self.assertEqual(evidence['mean_delta'], 7.)
        self.assertAlmostEqual(evidence['mean_percent'], 700./3.)
        self.assertEqual(evidence['best_margin'], 6.)
        self.assertEqual(evidence['z_score'], 7.)
        minimum = plots.nms_peak_statistics(xyz, values, 0, 1., maximize=False)
        self.assertEqual(minimum['rank'], 3)
        self.assertEqual(minimum['percentile'], 0.)
        self.assertEqual(minimum['mean_delta'], -7.)
        self.assertEqual(minimum['best_margin'], -8.)
        np.testing.assert_array_equal(values, saved)

    def test_ties_zero_mean_empty_neighbors_and_min_plot_are_explicit(self):
        xyz = np.column_stack((np.arange(3.), np.zeros((3, 2))))
        tied = plots.nms_peak_statistics(xyz, np.ones(3), 0, 3.)
        self.assertEqual(tied['percentile'], 50.)
        self.assertEqual(tied['best_margin'], 0.)
        self.assertIsNone(tied['z_score'])
        zero = plots.nms_peak_statistics(xyz, np.zeros(3), 0, 3.)
        self.assertIsNone(zero['mean_percent'])
        empty = plots.nms_peak_statistics(xyz, [1., np.nan, np.inf], 0, 3.)
        self.assertIsNone(empty['rank'])
        figure = plots.make_nms_peak_figure(xyz, [1., np.nan, np.inf], 0, 3., title='Test MIN', maximize=False)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 2)
        self.assertIn('lower is better', figure._suptitle.get_text())
        figure.clear()

    def window(self, mode=0):
        window = make_window()
        window.compare_result = comparison()
        window.compare_cache_key = 'cached'
        window._compare_key = Mock(return_value='cached')
        window.compare_show = True
        window.compare_heatmap_mode = mode
        window._overlay = Mock()
        window._save_labels = Mock(side_effect=AssertionError('saved labels'))
        return window

    def test_pick_reads_exact_visible_voxel_field_and_thresholds_without_recompute_or_labels(self):
        for mode, point_id, key, maximize in ((0, 1, 'fusion_scores', True), (4, 5, 'normal_lambda1_mm2', False),
                                             (9, 3, 'harris_scores', True)):
            window = self.window(mode)
            saved = copy.deepcopy(vars(window.compare_result))
            display = dict(closed=False, window=Mock())
            with (patch.object(manager, 'show_nms_peak_plot', return_value=display) as show,
                  patch.object(lab, 'compute_harris3d', side_effect=AssertionError('recomputed')),
                  redirect_stdout(StringIO())):
                self.assertTrue(window._picked_comparison_nms(window.compare_result.positions_mm[point_id]))
                self.assertIs(show.call_args.args[1], getattr(window.compare_result, key))
                self.assertEqual(show.call_args.args[2], point_id)
                self.assertEqual(show.call_args.kwargs['maximize'], maximize)
                window.compare_thresholds = {key: 100. if maximize else 0.}
                self.assertFalse(window._picked_comparison_nms(window.compare_result.positions_mm[point_id]))
                show.assert_called_once()
            np.testing.assert_equal(vars(window.compare_result), saved)
            window._save_labels.assert_not_called()
            window._deactivate_extrema_compare()
            display['window'].close.assert_called_once()

    def test_ctrl_click_preserves_overlay_and_drops_stale_depth_callbacks(self):
        for stale in (False, True):
            window = self.window()
            window.xyz = window.compare_result.positions_mm
            window.scene_widget.frame = SimpleNamespace(x=0, y=0, width=100, height=100)
            window._picked_comparison_nms = Mock()
            window._picked_world_point = Mock()
            window._deactivate_extrema_compare = Mock()
            window.scene_widget.scene.camera.unproject.return_value = [1., 0., 0.]
            window.scene_widget.scene.scene.render_to_depth_image.side_effect = lambda callback: callback(np.full((100, 100), .5))
            def post(_win, callback):
                if stale:
                    window.compare_heatmap_mode = 2
                callback()
            window.app.post_to_main_thread.side_effect = post
            event = SimpleNamespace(type=lab.gui.MouseEvent.Type.BUTTON_DOWN, x=50, y=50,
                                    is_button_down=lambda _: True, is_modifier_down=lambda _: True)
            window._on_mouse(event)
            self.assertEqual(window._picked_comparison_nms.call_count, 0 if stale else 1)
            window._picked_world_point.assert_not_called()
            window._deactivate_extrema_compare.assert_not_called()

    def test_managed_image_render_and_native_window_reuse_and_reopen(self):
        result = comparison()
        with patch.object(view3d, 'open_nms_peak_view', return_value='display') as open_view:
            display = manager.show_nms_peak_plot(result.positions_mm, result.fusion_scores, 1, 10., title='Fusion')
        self.assertEqual(display, 'display')
        rgb = open_view.call_args.args[0]
        self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(rgb.dtype, np.uint8)
        with (patch.object(lab.gui, 'Application', SimpleNamespace(instance=Mock())) as application,
              patch.object(lab.gui, 'ImageWidget', side_effect=lambda: Mock())):
            first = view3d.open_nms_peak_view(rgb)
            self.assertIs(view3d.open_nms_peak_view(rgb, display=first), first)
            application.instance.create_window.assert_called_once()
            first['window'].set_on_close.call_args.args[0]()
            self.assertTrue(first['closed'])
            self.assertIsNot(view3d.open_nms_peak_view(rgb, display=first), first)
            self.assertEqual(application.instance.create_window.call_count, 2)
