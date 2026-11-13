"""Scalar CED component magnitudes relative to estimated positional-PCA normals."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from corner_detector.direction_fields import pca_normal_direction, ced_component_magnitudes
import test_corner_lab_ui as ui
from test_corner_lab_ui import lab


class LabCedComponentTest(unittest.TestCase):
    def test_pca_normal_is_smallest_eigenvector_and_degenerate_eigenspaces_are_undefined(self):
        normal = pca_normal_direction(np.diag([4., 9., 0.]))
        np.testing.assert_array_equal(np.abs(normal), [0., 0., 1.])
        for covariance in (np.zeros((3, 3)), np.diag([0., 0., 1.]), np.eye(3)):
            self.assertTrue(np.isnan(pca_normal_direction(covariance)).all())
        rotation, _ = np.linalg.qr(np.random.default_rng(18).normal(size=(3, 3)))
        rotated = pca_normal_direction(rotation @ np.diag([4., 9., 0.]) @ rotation.T)
        expected = rotation @ normal
        np.testing.assert_allclose(np.outer(rotated, rotated), np.outer(expected, expected), atol=1e-14)

    def test_component_magnitudes_preserve_strength_zero_and_pythagorean_decomposition(self):
        offsets = np.array([[3., 4., 12.], [0., 0., 13.], [13., 0., 0.], [0., 0., 0.], [np.nan]*3])
        normals = np.tile([0., 0., 1.], (5, 1))
        before = offsets.copy()
        normal, tangent = ced_component_magnitudes(offsets, normals, 13.)
        np.testing.assert_allclose(normal, [12/13, 1., 0., 0., np.nan], equal_nan=True)
        np.testing.assert_allclose(tangent, [5/13, 0., 1., 0., np.nan], equal_nan=True)
        np.testing.assert_allclose(normal[:4]**2+tangent[:4]**2, np.sum(offsets[:4]**2, axis=1)/13.**2)
        half_normal, half_tangent = ced_component_magnitudes(offsets*.5, normals, 13.)
        np.testing.assert_allclose(half_normal, normal*.5, equal_nan=True)
        np.testing.assert_allclose(half_tangent, tangent*.5, equal_nan=True)
        tiny = ced_component_magnitudes(offsets[:1]*1e-9, normals[:1], 13.)
        np.testing.assert_allclose(tiny, [[normal[0]*1e-9], [tangent[0]*1e-9]], rtol=1e-14, atol=0.)
        np.testing.assert_array_equal(offsets, before)

    def test_components_are_normal_sign_rotation_and_scale_invariant_with_explicit_invalidity(self):
        offsets = np.array([[3., 4., 12.], [0., 0., 0.]])
        normals = np.tile([0., 0., 1.], (2, 1))
        expected = ced_component_magnitudes(offsets, normals, 13.)
        np.testing.assert_allclose(ced_component_magnitudes(offsets, -normals*3., 13.), expected)
        np.testing.assert_allclose(ced_component_magnitudes(offsets*10., normals, 130.), expected)
        rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        np.testing.assert_allclose(ced_component_magnitudes(offsets @ rotation, normals @ rotation, 13.), expected, atol=1e-15)
        for undefined in (np.zeros((2, 3)), np.full((2, 3), np.nan)):
            self.assertTrue(np.isnan(ced_component_magnitudes(offsets, undefined, 13.)).all())
        for radius in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                ced_component_magnitudes(offsets, normals, radius)
        with self.assertRaises(ValueError):
            ced_component_magnitudes(offsets, normals[:1], 13.)

    def test_local_and_full_clouds_cache_scalar_components_with_zero_normal_on_a_plane(self):
        xyz = np.array([(x, y, 700.) for x in range(0, 19, 2) for y in range(0, 19, 2)])
        common = dict(ced_radius_mm=3., pca_radius_mm=6., voxel_mm=2., min_neighbors=3, ced_cutoff=0.)
        full = lab.compute_full_scene_fusion_pca_comparison(xyz, nms_radius_mm=3., **common)
        local = lab.compute_local_fusion_pca_comparison(xyz, xyz[0], view_radius_mm=10., **common)
        for result in (full, local):
            self.assertEqual(result.ced_normal_component.shape, (len(result.positions_mm),))
            self.assertEqual(result.ced_tangent_component.shape, (len(result.positions_mm),))
            np.testing.assert_allclose(result.ced_normal_component, 0., atol=1e-14)
            self.assertAlmostEqual(result.ced_tangent_component[0], np.sqrt(2)/3.)
            for i, point in enumerate(result.positions_mm):
                observed = xyz[np.linalg.norm(xyz-point, axis=1) <= 3.+1e-8]
                raw_ced = np.linalg.norm(observed.mean(axis=0)-point)/3.
                self.assertAlmostEqual(result.ced_normal_component[i]**2+result.ced_tangent_component[i]**2, raw_ced**2)
        interior = np.flatnonzero(np.all(full.positions_mm == [8., 8., 700.], axis=1))[0]
        self.assertEqual(full.ced_normal_component[interior], 0.)
        self.assertEqual(full.ced_tangent_component[interior], 0.)

    def test_component_views_use_cached_scalar_strength_and_fixed_or_auto_contrast_without_nms(self):
        window, result, rendered = ui.CornerLabUiTest().comparison_window()
        saved = copy.deepcopy(vars(result))
        window.compare_thresholds = {'fusion_scores': .9}
        with (patch.object(lab, 'pca_normal_direction', side_effect=AssertionError('recomputed normal')),
              patch.object(lab, 'ced_component_magnitudes', side_effect=AssertionError('recomputed components')),
              patch.object(lab, 'threshold_extrema_comparison', side_effect=AssertionError('gated heatmap'))):
            for mode, key in ((10, 'ced_normal_component'), (11, 'ced_tangent_component')):
                rendered.clear()
                window.compare_auto_contrast_check.checked = False
                window._compare_display_changed('', mode)
                self.assertFalse(window.compare_nms_check.enabled)
                self.assertTrue(window.compare_auto_contrast_check.enabled)
                self.assertFalse(window.compare_pairs_check.checked)
                self.assertNotIn(key, window.compare_threshold_edits)
                self.assertEqual(set(rendered), {'comparison_heatmap'})
                expected, _ = lab.eigen_heatmap_colors(getattr(result, key), limits=(0., 1.))
                np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), expected/255.)
                np.testing.assert_array_equal(np.asarray(rendered['comparison_heatmap'].points), result.positions_mm)
                self.assertIn('magnitude / CED radius; no NMS', window.compare_summary.text)
                self.assertIn('6/7', window.compare_summary.text)
                self.assertIn('fixed 0-1', window.eigen_heatmap_legend.text)
                window.compare_auto_contrast_check.checked = True
                window._compare_auto_contrast_changed(True)
                expected, _ = lab.eigen_heatmap_colors(getattr(result, key))
                np.testing.assert_allclose(np.asarray(rendered['comparison_heatmap'].colors), expected/255.)
                self.assertIn('Auto Contrast', window.eigen_heatmap_legend.text)
        np.testing.assert_equal(vars(result), saved)
