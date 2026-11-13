"""Standalone score/NMS association, tab layout and cached inspection behavior."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from pointcloud_inputs.visualization import require_open3d

require_open3d()
import corner_lab_open3d_cone as lab


def make_window():
    gui = MagicMock()
    for name in ('SceneWidget', 'ImageWidget', 'Slider', 'Checkbox', 'Combobox', 'Button',
                 'Horiz', 'Vert', 'ScrollableVert', 'CollapsableVert', 'TabControl', 'TextEdit'):
        getattr(gui, name).side_effect = lambda *args, **kwargs: MagicMock()
    gui.Label.side_effect = lambda text: SimpleNamespace(text=text)
    native = gui.Application.instance.create_window.return_value
    native.theme.font_size, native.scaling = 14, 1.
    native.content_rect = lab.gui.Rect(0, 0, 1650, 940)
    with patch.object(lab, 'gui', gui), patch.object(lab, 'rendering', MagicMock()):
        window = lab.CornerLab(None, False, False)
    return window


def cached_comparison():
    return SimpleNamespace(
        positions_mm=np.column_stack((np.arange(7.), np.zeros((7, 2)))),
        fusion_scores=np.arange(7.)/7, ced_scores=np.arange(7.)[::-1]/7,
        ced_weighted_ratio=np.array([.2, .1, .8, .4, .3, .7, .6]),
        ced_weighted_lambda1_mm2=np.array([4., 3., 2., 1., 0., 6., 5.]),
        normal_lambda1_mm2=np.array([5., 6., 4., 3., 2., 1., 0.]),
        local_minimum_scores=np.linspace(0., 1., 7),
        ced_local_min_scores=np.array([0., .2, .4, .9, .3, .1, 0.]),
        ced_local_min_maxima=np.array([3]), contrast_tau=.005,
        median_minimum_scores=np.array([.7, .4, .2, 0., .1, .3, .6]),
        ced_median_min_scores=np.array([.4, .2, .1, 0., .05, .1, .3]),
        ced_median_min_maxima=np.array([0]), median_contrast_tau=.005,
        harris_scores=np.array([0., .01, 0., .04, .005, .002, 0.]), harris_maxima=np.array([3]),
        local_minimum_radius_mm=None, median_minimum_radius_mm=None,
        ced_normal_component=np.array([0., .1, .3, .2, .05, .4, np.nan]),
        ced_tangent_component=np.array([.3, .1, .2, .5, 0., .05, np.nan]),
        ced_inverse_std_scores=np.array([.01, .02, .05, .03, .015, 0., np.nan]),
        ced_inverse_std_maxima=np.array([2]), inverse_std_epsilon_mm2=1e-6,
        inverse_std_activation='none', inverse_std_raw_scores=np.empty(0),
        pca_neighbor_counts=np.array([4, 5, 6, 7, 6, 5, 4]), inverse_std_neighbor_medians=np.empty(0),
        fusion_maxima=np.array([1]), ced_maxima=np.array([2]), ratio_maxima=np.array([3]),
        weighted_lambda1_minima=np.array([4]), lambda1_minima=np.array([5]),
        match_ced_ratio=((2, 3, 1.),), match_ced_weighted_l1=((2, 4, 2.),),
        match_ced_normal_l1=((2, 5, 3.),), nms_radius_mm=10., match_radius_mm=8.,
        ced_radius_mm=10., pca_radius_mm=30.)


class CornerLabUiTest(unittest.TestCase):
    def comparison_window(self):
        window = make_window()
        result = cached_comparison()
        window.compare_result, window.compare_cache_key, window.compare_show = result, 'cached', True
        window._compare_key = Mock(return_value='cached')
        rendered = {}
        window._overlay = lambda name, geometry, material: rendered.__setitem__(name, geometry)
        return window, result, rendered

    def test_all_ten_heatmaps_use_their_cached_field_and_corresponding_extrema(self):
        window, result, rendered = self.comparison_window()
        saved = copy.deepcopy(result.__dict__)
        expected = (('fusion_scores', 1, 'MAX'), ('ced_scores', 2, 'MAX'),
                    ('ced_weighted_ratio', 3, 'MAX'), ('ced_weighted_lambda1_mm2', 4, 'MIN'),
                    ('normal_lambda1_mm2', 5, 'MIN'), ('ced_local_min_scores', 3, 'MAX'),
                    ('local_minimum_scores', None, 'HEATMAP'),
                    ('ced_median_min_scores', 0, 'MAX'), ('median_minimum_scores', None, 'HEATMAP'),
                    ('harris_scores', 3, 'MAX'))
        with (patch.object(lab, 'compute_full_scene_fusion_pca_comparison', side_effect=AssertionError('display recompute')),
              patch.object(lab, 'compute_local_fusion_pca_comparison', side_effect=AssertionError('display recompute'))):
            for index, (key, point_id, extreme) in enumerate(expected):
                rendered.clear()
                window._compare_display_changed('', index)
                colors, _ = lab.eigen_heatmap_colors(
                    getattr(result, key), limits=(0., 1.) if key in (
                        'ced_local_min_scores', 'local_minimum_scores', 'ced_median_min_scores', 'median_minimum_scores') else None)
                np.testing.assert_array_equal(np.asarray(rendered['comparison_heatmap'].points), result.positions_mm)
                np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), colors/255.)
                if point_id is None:
                    self.assertNotIn('comparison_fusion_max', rendered)
                else:
                    np.testing.assert_array_equal(np.asarray(rendered['comparison_fusion_max'].points),
                                                  result.positions_mm[[point_id]])
                self.assertNotIn('comparison_lambda1_min', rendered)
                self.assertNotIn('comparison_match_links', rendered)
                self.assertIn(f'({extreme})', window.compare_summary.text)
                window.scene_widget.scene.show_geometry.assert_any_call('cloud', False)
                window.win.set_needs_layout.assert_called()
        np.testing.assert_equal(result.__dict__, saved)

    def test_pure_local_minimum_heatmap_reuses_l_without_gating_nms_or_recomputation(self):
        window, result, rendered = self.comparison_window()
        saved = copy.deepcopy(vars(result))
        window.compare_thresholds = {'fusion_scores': .9}
        with (patch.object(lab, 'threshold_extrema_comparison', side_effect=AssertionError('applied gates')),
              patch.object(lab, 'compute_local_minimum_fusion', side_effect=AssertionError('recomputed L')),
              patch.object(lab, 'find_full_scene_scalar_extrema', side_effect=AssertionError('ran NMS'))):
            window._compare_display_changed('', 6)
            self.assertFalse(window.compare_nms_check.enabled)
            self.assertTrue(window.compare_auto_contrast_check.enabled)
            self.assertNotIn('local_minimum_scores', window.compare_threshold_edits)
            self.assertEqual(set(rendered), {'comparison_heatmap'})
            colors, _ = lab.eigen_heatmap_colors(result.local_minimum_scores, limits=(0., 1.))
            np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), colors/255.)
            self.assertIn('before CED multiplication', window.compare_summary.text)
            window.compare_auto_contrast_check.checked = True
            window._compare_auto_contrast_changed(True)
            colors, _ = lab.eigen_heatmap_colors(result.local_minimum_scores)
            np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), colors/255.)
            self.assertIn('Auto Contrast P2-P98', window.eigen_heatmap_legend.text)
        np.testing.assert_equal(vars(result), saved)
        window.compare_thresholds = {}
        window._compare_display_changed('', 5)
        self.assertTrue(window.compare_nms_check.enabled)

    def test_auto_contrast_only_recolors_cached_minimum_score_without_changing_nms(self):
        window, result, rendered = self.comparison_window()
        result.ced_local_min_scores = np.linspace(.0001, .0007, 7)
        saved = copy.deepcopy(vars(result))
        token, key = window.compare_job_id, window.compare_cache_key
        self.assertFalse(window.compare_auto_contrast_check.checked)
        self.assertFalse(window.compare_auto_contrast_check.enabled)
        with (patch.object(lab, 'compute_local_minimum_fusion', side_effect=AssertionError('recomputed score')),
              patch.object(lab, 'find_full_scene_scalar_extrema', side_effect=AssertionError('reran NMS'))):
            window._compare_display_changed('', 5)
            self.assertTrue(window.compare_auto_contrast_check.enabled)
            fixed_colors = np.asarray(rendered['comparison_heatmap'].colors).copy()
            nms = np.asarray(rendered['comparison_fusion_max'].points).copy()
            window.compare_auto_contrast_check.checked = True
            window._compare_auto_contrast_changed(True)
            colors, _ = lab.eigen_heatmap_colors(result.ced_local_min_scores)
            np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), colors/255.)
            self.assertFalse(np.array_equal(fixed_colors, np.asarray(rendered['comparison_heatmap'].colors)))
            np.testing.assert_array_equal(np.asarray(rendered['comparison_fusion_max'].points), nms)
            self.assertIn('Auto Contrast P2-P98', window.eigen_heatmap_legend.text)
            window.compare_auto_contrast_check.checked = False
            window._compare_auto_contrast_changed(False)
            np.testing.assert_array_equal(np.asarray(rendered['comparison_heatmap'].colors), fixed_colors)
            self.assertIn('fixed 0-1', window.eigen_heatmap_legend.text)
        self.assertIs(window.compare_result, result)
        np.testing.assert_equal(vars(result), saved)
        self.assertEqual(window.compare_job_id, token)
        self.assertEqual(window.compare_cache_key, key)
        self.assertEqual(window.compare_thresholds, {})
        window._compare_display_changed('', 0)
        self.assertFalse(window.compare_auto_contrast_check.enabled)

    def test_auto_contrast_handles_all_invalid_minimum_scores_without_changing_them(self):
        window, result, rendered = self.comparison_window()
        result.ced_local_min_scores = np.full(7, np.nan)
        result.ced_local_min_maxima = np.empty(0, dtype=np.int64)
        window.compare_auto_contrast_check.checked = True
        window._compare_display_changed('', 5)
        np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors),
                                   np.tile(np.array([95, 103, 115])/255., (7, 1)))
        self.assertTrue(np.isnan(result.ced_local_min_scores).all())
        self.assertNotIn('comparison_fusion_max', rendered)

    def test_layer_toggles_and_legacy_pairing_preserve_cached_counts_and_results(self):
        window, result, rendered = self.comparison_window()
        saved = copy.deepcopy(result.__dict__)
        window._compare_method_changed('', 2)
        self.assertTrue(window.compare_pairs_check.checked)
        self.assertEqual(window.compare_heatmap_mode, 3)
        np.testing.assert_array_equal(np.asarray(rendered['comparison_fusion_max'].points), result.positions_mm[[2]])
        np.testing.assert_array_equal(np.asarray(rendered['comparison_lambda1_min'].points), result.positions_mm[[4]])
        np.testing.assert_array_equal(np.asarray(rendered['comparison_match_links'].points), result.positions_mm[[2, 4]])
        rendered.clear()
        window.compare_heatmap_check.checked = False
        window.compare_nms_check.checked = False
        window._compare_layers_changed(False)
        self.assertEqual(rendered, {})
        self.assertIn('Pairs 1', window.compare_summary.text)
        self.assertIn('(hidden)', window.compare_summary.text)
        window.scene_widget.scene.show_geometry.assert_any_call('cloud', True)
        window._compare_display_changed('', 0)
        self.assertFalse(window.compare_pairs_check.checked)
        np.testing.assert_equal(result.__dict__, saved)

    def test_matched_markers_stay_larger_than_nms_even_at_zero_distance(self):
        for secondary_id, distance in ((4, 2.), (2, 0.)):
            with self.subTest(distance=distance):
                window, result, rendered = self.comparison_window()
                result.weighted_lambda1_minima = np.array([secondary_id])
                result.match_ced_weighted_l1 = ((2, secondary_id, distance),)
                saved = copy.deepcopy(result.__dict__)
                window._overlay = Mock(side_effect=window._overlay)
                window._compare_method_changed('', 2)
                materials = {call.args[0]: call.args[2]
                             for call in window._overlay.call_args_list}
                marker = materials['comparison_match_midpoints']
                self.assertGreater(marker.point_size, materials['comparison_fusion_max'].point_size)
                self.assertGreater(marker.point_size, materials['comparison_lambda1_min'].point_size)
                self.assertEqual(materials['comparison_match_links'].line_width, 6.)
                expected = (result.positions_mm[2] + result.positions_mm[secondary_id]) * .5
                np.testing.assert_array_equal(
                    np.asarray(rendered['comparison_match_midpoints'].points), expected[None])
                np.testing.assert_allclose(
                    np.asarray(rendered['comparison_match_midpoints'].colors), [[70/255, 1., 40/255]])
                self.assertIn('Pairs 1', window.compare_summary.text)
                np.testing.assert_equal(result.__dict__, saved)

    def test_controls_are_grouped_and_corner_tab_reveals_existing_inspection_views(self):
        window = make_window()
        tabs = window.panel.add_tab.call_args_list
        self.assertEqual([call.args[0] for call in tabs], ['Heatmap/NMS', 'Corner', 'PCA', 'Input'])
        heatmap_children = [call.args[0] for call in window.heatmap_page.add_child.call_args_list]
        input_children = [call.args[0] for call in window.input_page.add_child.call_args_list]
        self.assertIn(window.eigen_ced_slider, heatmap_children)
        self.assertIn(window.eigen_pca_slider, heatmap_children)
        self.assertIn(window.compare_display_combo, heatmap_children)
        self.assertIn(window.preview, input_children)
        self.assertNotIn(window.preview, heatmap_children)
        window._layout(None)
        full_width = window.scene_widget.frame.width
        self.assertFalse(window.unit_widget.visible)
        self.assertFalse(window.hist_widget.visible)
        saved = {'score': np.array([.3, .7])}
        window.last_gap_validation = saved
        window.visible_overlay_names.update(('local_cone_ring', 'comparison_heatmap', 'corner_markers'))
        window._workspace_changed(1)
        window._layout(None)
        self.assertTrue(window.unit_widget.visible)
        self.assertTrue(window.hist_widget.visible)
        self.assertLess(window.scene_widget.frame.width, full_width)
        window.scene_widget.scene.show_geometry.assert_any_call('local_cone_ring', True)
        window._workspace_changed(0)
        window._layout(None)
        self.assertEqual(window.scene_widget.frame.width, full_width)
        window.scene_widget.scene.show_geometry.assert_any_call('local_cone_ring', False)
        self.assertIs(window.last_gap_validation, saved)
        self.assertEqual(window.radius_mm, 30)
        self.assertTrue(window.spherical_planes_enabled)
        self.assertEqual(window.gap_edge_span_deg, 160.)
        window._eigen_mode_changed('', 1)
        self.assertEqual(window.eigen_mode, 'unweighted')
        self.assertTrue(window.eigen_ced_slider.enabled)

    def test_optional_inspection_views_follow_permanent_controls_in_native_child_order(self):
        window = make_window()
        children = [call.args[0] for call in window.win.add_child.call_args_list]
        permanent = (window.scene_widget, window.header, window.panel,
                     window.eigen_heatmap_legend, window.status)
        last_permanent = max(next(i for i, child in enumerate(children) if child is widget)
                             for widget in permanent)
        for widget in (window.unit_widget, window.hist_title, window.hist_widget, window.hist_legend):
            index = next(i for i, child in enumerate(children) if child is widget)
            self.assertGreater(index, last_permanent)

    def test_native_tab_callbacks_do_not_requeue_selection_or_change_worker_state(self):
        class NativeTabs:
            index = 0

            def __init__(self):
                self.selections = []

            @property
            def selected_tab_index(self):
                return self.index

            @selected_tab_index.setter
            def selected_tab_index(self, index):
                self.selections.append(index)
                self.index = index

        window = make_window()
        window.panel = NativeTabs()
        window.compare_running = True
        window.compare_job_id = 17
        result = window.compare_result = cached_comparison()
        for index in (1, 2, 3, 0, 2, 1, 0):
            # Open3D has already selected the clicked tab before invoking Python.
            window.panel.index = index
            window._workspace_changed(index)
            self.assertEqual(window.panel.selections, [])
            self.assertEqual(window.unit_widget.visible, index == 1)
        window._workspace_changed(1)
        window._workspace_changed(1)
        self.assertEqual(window.panel.selections, [1])
        self.assertEqual(window.panel.selected_tab_index, 1)
        self.assertTrue(window.compare_running)
        self.assertEqual(window.compare_job_id, 17)
        self.assertIs(window.compare_result, result)

    def test_scope_controls_and_click_inspection_keep_original_raw_indices(self):
        window = make_window()
        window._draw_eigen_field_overlay = Mock()
        window._compare_scope_changed('', 0)
        self.assertFalse(window.compare_scene_voxel_edit.enabled)
        self.assertFalse(window.compare_scene_nms_edit.enabled)
        window._compare_scope_changed('', 1)
        self.assertTrue(window.compare_scene_voxel_edit.enabled)
        window.xyz = np.array([[1., 2., 3.], [4., 5., 6.]])
        window.kdtree = Mock()
        window.kdtree.search_knn_vector_3d.return_value = (1, [1], [0.])
        window._save_labels, window._refresh_corner_combo, window._refresh_analysis = Mock(), Mock(), Mock()
        window._picked_world_point(window.xyz[1])
        self.assertEqual(window.panel.selected_tab_index, 1)
        self.assertEqual(window.corners, [{'id': 'corner_001', 'point_index': 1}])
        window._save_labels.assert_called_once()
        window._refresh_analysis.assert_called_once()
