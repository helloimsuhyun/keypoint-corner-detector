"""Configurable demo noise/bumps, exact defaults and separate variant saves."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pointcloud_inputs.demo import make_demo_cloud
from test_corner_lab_ui import lab, make_window


class DemoSurfaceTest(unittest.TestCase):
    def test_zero_noise_makes_planes_and_sigma_changes_only_the_normal_offsets(self):
        clean, colors, corner = make_demo_cloud(noise_sigma_mm=0.)
        baseline, _, _ = make_demo_cloud()
        noisy, _, _ = make_demo_cloud(noise_sigma_mm=1.)
        origin = np.array([0., 0., 700.])
        for dim in range(3):
            block = slice(4200*dim, 4200*(dim+1))
            other = [i for i in range(3) if i != dim]
            np.testing.assert_array_equal(clean[block, dim], np.full(4200, origin[dim]))
            np.testing.assert_array_equal(clean[block][:, other], noisy[block][:, other])
            np.testing.assert_allclose(noisy[block, dim]-origin[dim],
                                       4*(baseline[block, dim]-origin[dim]), atol=3e-4)
        self.assertEqual(corner, 12600)
        np.testing.assert_array_equal(noisy[corner], origin)
        np.testing.assert_array_equal(colors[corner], [255, 255, 255])

    def test_smooth_bumps_follow_amplitude_and_wavelength_without_resampling(self):
        clean, colors, corner = make_demo_cloud(noise_sigma_mm=0.)
        rough, rough_colors, rough_corner = make_demo_cloud(
            noise_sigma_mm=0., roughness_amplitude_mm=2.5, roughness_wavelength_mm=20.)
        origin = np.array([0., 0., 700.])
        for dim in range(3):
            block = slice(4200*dim, 4200*(dim+1))
            other = [i for i in range(3) if i != dim]
            uv = clean[block][:, other].astype(float)-origin[other]
            expected = 2.5*np.sin(2*np.pi*uv[:, 0]/20)*np.sin(2*np.pi*uv[:, 1]/20)
            np.testing.assert_allclose(rough[block, dim]-origin[dim], expected, atol=2e-4)
            np.testing.assert_array_equal(rough[block][:, other], clean[block][:, other])
            self.assertLessEqual(np.abs(rough[block, dim]-origin[dim]).max(), 2.5001)
        np.testing.assert_array_equal(rough_colors, colors)
        self.assertEqual(rough_corner, corner)
        np.testing.assert_array_equal(rough[corner], clean[corner])
        shorter, _, _ = make_demo_cloud(noise_sigma_mm=0., roughness_amplitude_mm=2.5,
                                         roughness_wavelength_mm=10.)
        self.assertFalse(np.array_equal(rough, shorter))

    def test_same_seed_reproduces_geometry_and_other_seeds_change_sampling(self):
        params = dict(noise_sigma_mm=.7, roughness_amplitude_mm=1.5, seed=18)
        a, rgb, _ = make_demo_cloud(**params)
        b, _, _ = make_demo_cloud(**params)
        c, _, _ = make_demo_cloud(**{**params, 'seed': 19})
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a[:-1], c[:-1]))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(rgb.dtype, np.uint8)

    def test_fillets_have_exact_cylinder_and_sphere_radii_and_keep_flat_faces(self):
        origin = np.array([0., 0., 700.])
        sharp, colors, corner = make_demo_cloud(noise_sigma_mm=0.)
        for radius in (10., 35., 85.):
            with self.subTest(radius=radius):
                rounded, rgb, reference = make_demo_cloud(noise_sigma_mm=0., fillet_radius_mm=radius)
                p = rounded.astype(float) - origin
                for dim in range(3):
                    block = slice(4200*dim, 4200*(dim+1))
                    original = sharp[block].astype(float) - origin
                    curved_axes = original < radius
                    count = curved_axes.sum(axis=1)
                    for n_axes in (1, 2, 3):
                        mask = count == n_axes
                        if radius < 85.:
                            self.assertTrue(mask.any())
                        displacement = np.where(curved_axes[mask], p[block][mask] - radius, 0.)
                        np.testing.assert_allclose(np.linalg.norm(displacement, axis=1), radius, atol=4e-5)
                    np.testing.assert_array_equal(rounded[block][count == 1], sharp[block][count == 1])
                    np.testing.assert_allclose(p[block][~curved_axes], original[~curved_axes], atol=4e-5)
                np.testing.assert_allclose(p[reference], radius*(1.-1./np.sqrt(3.)), atol=4e-5)
                self.assertTrue((p >= -4e-5).all())
                self.assertEqual(reference, corner)
                np.testing.assert_array_equal(rgb, colors)

    def test_filleted_noise_and_bumps_follow_surface_normals_with_same_seed(self):
        radius = 25.
        origin = np.array([0., 0., 700.])
        clean, _, corner = make_demo_cloud(noise_sigma_mm=0., fillet_radius_mm=radius)
        perturbed, _, _ = make_demo_cloud(noise_sigma_mm=.7, roughness_amplitude_mm=2.,
                                        fillet_radius_mm=radius)
        flat, _, _ = make_demo_cloud(noise_sigma_mm=0.)
        flat_perturbed, _, _ = make_demo_cloud(noise_sigma_mm=.7, roughness_amplitude_mm=2.)
        for dim in range(3):
            block = slice(4200*dim, 4200*(dim+1))
            centers = np.maximum(flat[block].astype(float)-origin, radius)
            normal = (centers - (clean[block].astype(float)-origin))/radius
            height = flat_perturbed[block, dim].astype(float)-origin[dim]
            np.testing.assert_allclose(perturbed[block]-clean[block], height[:, None]*normal, atol=1e-4)
        np.testing.assert_array_equal(perturbed[corner], clean[corner])
        repeated, _, _ = make_demo_cloud(noise_sigma_mm=.7, roughness_amplitude_mm=2.,
                                       fillet_radius_mm=radius)
        np.testing.assert_array_equal(perturbed, repeated)

    def test_invalid_parameters_fail_explicitly(self):
        for kwargs in ({'noise_sigma_mm': -1.}, {'noise_sigma_mm': np.nan},
                       {'roughness_amplitude_mm': -1.}, {'roughness_amplitude_mm': np.inf},
                       {'roughness_wavelength_mm': 0.}, {'roughness_wavelength_mm': np.nan},
                       {'fillet_radius_mm': -1.}, {'fillet_radius_mm': 85.01},
                       {'fillet_radius_mm': np.nan}, {'fillet_radius_mm': np.inf},
                       {'seed': -1}, {'seed': 1.5}, {'seed': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_demo_cloud(**kwargs)

    def test_gui_apply_saves_variants_separately_and_preserves_default_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = make_window()
            window._open_dataset = Mock()
            with patch.object(lab, 'CAPTURES', root), redirect_stdout(StringIO()):
                window._make_demo()
                default = root/'demo_corner'
                window._open_dataset.assert_called_with(default)
                lab.save_corners(default, [{'id': 'my_corner', 'point_index': 10,
                                           'refined_xyz_mm': [1., 2., 700.]}])
                saved = {p.name: p.read_bytes() for p in default.iterdir()}
                window.demo_noise_slider.double_value = 1.
                window.demo_roughness_slider.double_value = 2.
                window.demo_wavelength_slider.double_value = 25.
                window.demo_fillet_slider.double_value = 12.
                window._make_demo()
                variant = window._open_dataset.call_args.args[0]
                self.assertNotEqual(variant, default)
                metadata = json.loads((variant/'metadata.json').read_text())
                self.assertEqual(metadata['noise_sigma_mm'], 1.)
                self.assertEqual(metadata['roughness_amplitude_mm'], 2.)
                self.assertEqual(metadata['roughness_wavelength_mm'], 25.)
                self.assertEqual(metadata['fillet_radius_mm'], 12.)
                self.assertEqual(metadata['seed'], 7)
                self.assertEqual(lab.load_corners(variant, 12601), [{'id': 'corner_001', 'point_index': 12600}])
                self.assertEqual(saved, {p.name: p.read_bytes() for p in default.iterdir()})
                window._make_demo()
                self.assertNotEqual(window._open_dataset.call_args.args[0], variant)

    def test_gui_fillet_alone_creates_separate_geometry_and_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lab, 'CAPTURES', Path(tmp)), redirect_stdout(StringIO()):
            window = make_window()
            window._open_dataset = Mock()
            window._make_demo()
            default = Path(tmp)/'demo_corner'
            saved = {p.name: p.read_bytes() for p in default.iterdir()}
            window.demo_fillet_slider.double_value = 15.
            window._make_demo()
            variant = window._open_dataset.call_args.args[0]
            self.assertNotEqual(variant, default)
            _, points, colors = lab.load_cloud(variant)
            expected, rgb, corner = make_demo_cloud(fillet_radius_mm=15.)
            np.testing.assert_array_equal(points, expected)
            np.testing.assert_array_equal(colors, rgb)
            np.testing.assert_array_equal(points[corner], expected[corner])
            self.assertEqual(saved, {p.name: p.read_bytes() for p in default.iterdir()})

    def test_gui_sliders_only_update_labels_and_invalid_apply_does_not_write(self):
        window = make_window()
        with patch.object(lab, 'make_demo_cloud', side_effect=AssertionError('regenerated')):
            window.demo_noise_slider.double_value = 1.
            window.demo_fillet_slider.double_value = 12.
            window._demo_surface_changed(1.)
        self.assertIn('1 mm', window.demo_noise_label.text)
        self.assertIn('12 mm', window.demo_fillet_label.text)
        with tempfile.TemporaryDirectory() as tmp, patch.object(lab, 'CAPTURES', Path(tmp)):
            window.demo_seed_edit.text_value = '3.5'
            window._error = Mock()
            window._open_dataset = Mock()
            window._make_demo()
            window._error.assert_called_once()
            window._open_dataset.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
            window.demo_seed_edit.text_value = '7'
            window.demo_fillet_slider.double_value = float('nan')
            window._error.reset_mock()
            window._make_demo()
            window._error.assert_called_once()
            window._open_dataset.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
