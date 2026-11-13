"""Configurable demo noise/bumps, exact defaults and separate variant saves."""
from contextlib import redirect_stdout
from io import StringIO
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pointcloud_inputs.demo import _round_demo_face, make_demo_cloud
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

    def test_corner_and_edge_radii_are_independent_and_zero_disables_each(self):
        points = np.array([[0., 0., 0.], [0., 1., 1.], [0., 1., 85.], [0., 50., 50.]])
        corners, _ = _round_demo_face(points, 0, 0., 10.)
        edges, _ = _round_demo_face(points, 0, 20., 0.)
        both, _ = _round_demo_face(points, 0, 20., 10.)
        np.testing.assert_array_equal(corners[:2], both[:2])
        np.testing.assert_array_equal(edges[2:], both[2:])
        np.testing.assert_array_equal(edges[0], points[0])
        np.testing.assert_array_equal(corners[2:], points[2:])
        np.testing.assert_allclose(np.linalg.norm(corners[:2]-10., axis=1), 10.)
        self.assertAlmostEqual(np.linalg.norm(edges[2, :2]-20.), 20.)
        np.testing.assert_array_equal(both[3], points[3])

    def test_equal_radii_keep_exact_spherical_and_cylindrical_geometry(self):
        points = np.array([[0., 1., 2.], [0., 1., 60.], [0., 60., 60.]])
        for radius in (10., 35., 42.5):
            with self.subTest(radius=radius):
                rounded, normal = _round_demo_face(points, 0, radius, radius)
                center = np.maximum(points, radius)
                np.testing.assert_allclose(np.linalg.norm(rounded-center, axis=1), radius)
                np.testing.assert_allclose(normal, (center-rounded)/radius, atol=1e-14)
                np.testing.assert_array_equal(rounded[2], points[2])

    def test_all_one_two_three_face_corners_are_rounded_and_reference_stays_on_surface(self):
        origin = np.array([0., 0., 700.])
        sharp, colors, _ = make_demo_cloud(noise_sigma_mm=0.)
        rounded, rgb, reference = make_demo_cloud(noise_sigma_mm=0.,
            **{f'corner_fillet_{faces}face_radius_mm': 20. for faces in (1, 2, 3)})
        self.assertEqual(rounded.shape, (12121, 3))
        self.assertEqual(hashlib.sha256(rounded.tobytes()).hexdigest(),
                         '6d336a57e42111539081777bf7ddcf099ef01c85feeb8bdbd67a749effc909a8')
        self.assertEqual(reference, 12120)
        np.testing.assert_allclose(rounded[reference]-origin, 20.*(1.-1./np.sqrt(3.)), atol=4e-5)
        for dim in range(3):
            face_color = colors[4200*dim]
            face = rounded[np.all(rgb == face_color, axis=1)].astype(float)-origin
            other = [i for i in range(3) if i != dim]
            uv = face[:, other]
            for vertex in ((0., 85.), (85., 0.), (85., 85.)):
                # Far from the central blend these are exact circular outline cuts.
                distances = np.abs(uv-vertex)
                near = (distances < 20.).all(axis=1)
                self.assertTrue(near.any())
                self.assertTrue((np.linalg.norm(distances[near]-20., axis=1) <= 20.+1e-4).all())
                original = sharp[4200*dim:4200*(dim+1)][:, other]-origin[other]
                original_near = (np.abs(original-vertex) < 20.).all(axis=1)
                self.assertGreater(original_near.sum(), near.sum())
            central = (face < 20.).all(axis=1)
            self.assertTrue(central.any())
            np.testing.assert_allclose(np.linalg.norm(face[central]-20., axis=1), 20., atol=4e-5)
        edge_only, edge_rgb, edge_reference = make_demo_cloud(noise_sigma_mm=0., edge_fillet_radius_mm=20.)
        self.assertEqual(edge_only.shape, sharp.shape)
        np.testing.assert_array_equal(edge_rgb, colors)
        np.testing.assert_array_equal(edge_only[edge_reference], origin)

    def test_each_corner_type_only_changes_its_own_cut_or_central_surface(self):
        sharp, colors, _ = make_demo_cloud(noise_sigma_mm=0.)
        origin = np.array([0., 0., 700.])
        for faces, expected_count in ((1, 12440), (2, 12282), (3, 12601)):
            with self.subTest(faces=faces):
                rounded, rgb, reference = make_demo_cloud(noise_sigma_mm=0.,
                    **{f'corner_fillet_{faces}face_radius_mm': 20.})
                self.assertEqual(len(rounded), expected_count)
                self.assertEqual(reference, expected_count-1)
                if faces == 3:
                    np.testing.assert_array_equal(rgb, colors)
                    np.testing.assert_allclose(rounded[reference]-origin, 20.*(1.-1./np.sqrt(3.)), atol=4e-5)
                    continue
                np.testing.assert_array_equal(rounded[reference], origin)
                for dim in range(3):
                    original = sharp[4200*dim:4200*(dim+1)]
                    retained = rounded[np.all(rgb == colors[4200*dim], axis=1)]
                    other = [i for i in range(3) if i != dim]
                    def affected(points):
                        uv = points[:, other]-origin[other]
                        if faces == 1:
                            return (uv > 65.).all(axis=1)
                        return ((uv[:, 0] < 20.) & (uv[:, 1] > 65.)
                                | (uv[:, 1] < 20.) & (uv[:, 0] > 65.))
                    np.testing.assert_array_equal(retained[~affected(retained)], original[~affected(original)])
                    self.assertEqual(len(np.unique(np.vstack((original, retained)), axis=0)), len(original))

    def test_transition_normals_match_surface_tangents_and_noise_keeps_seed_sampling(self):
        points = np.array([[0., 3., 25.], [0., 1., 45.], [0., 18., 26.]])
        for corner, edge in ((10., 30.), (30., 0.), (0., 30.)):
            rounded, normal = _round_demo_face(points, 0, edge, corner)
            np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.)
            for axis in (1, 2):
                shift = np.eye(3)[axis]*1e-4
                plus, _ = _round_demo_face(points+shift, 0, edge, corner)
                minus, _ = _round_demo_face(points-shift, 0, edge, corner)
                np.testing.assert_allclose(np.sum(normal*(plus-minus)/2e-4, axis=1), 0., atol=1e-8)
            seam = np.column_stack((np.zeros((4, 2)), [1., 15., 31., 65.]))
            first, first_normal = _round_demo_face(seam, 0, edge, corner)
            second, second_normal = _round_demo_face(seam, 1, edge, corner)
            np.testing.assert_allclose(first, second, atol=1e-12)
            curved = np.linalg.norm(first[:, :2], axis=1) > 0.
            np.testing.assert_allclose(first_normal[curved], second_normal[curved], atol=1e-12)
        params = dict(corner_fillet_1face_radius_mm=10., corner_fillet_2face_radius_mm=10.,
                      corner_fillet_3face_radius_mm=10., edge_fillet_radius_mm=30.)
        clean, rgb, reference = make_demo_cloud(noise_sigma_mm=0., **params)
        noisy, noisy_rgb, noisy_reference = make_demo_cloud(noise_sigma_mm=.7, **params)
        scaled, _, _ = make_demo_cloud(noise_sigma_mm=1.4, **params)
        np.testing.assert_allclose(scaled-clean, 2.*(noisy-clean), atol=1e-4)
        np.testing.assert_array_equal(noisy_rgb, rgb)
        self.assertEqual(noisy_reference, reference)
        np.testing.assert_array_equal(noisy[reference], clean[reference])
        repeated, _, _ = make_demo_cloud(noise_sigma_mm=.7, **params)
        np.testing.assert_array_equal(noisy, repeated)

    def test_invalid_parameters_fail_explicitly(self):
        for kwargs in ({'noise_sigma_mm': -1.}, {'noise_sigma_mm': np.nan},
                       {'roughness_amplitude_mm': -1.}, {'roughness_amplitude_mm': np.inf},
                       {'roughness_wavelength_mm': 0.}, {'roughness_wavelength_mm': np.nan},
                       {'edge_fillet_radius_mm': -1.}, {'edge_fillet_radius_mm': 85.01},
                       {'edge_fillet_radius_mm': np.nan}, {'edge_fillet_radius_mm': np.inf},
                       {'seed': -1}, {'seed': 1.5}, {'seed': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_demo_cloud(**kwargs)
        for faces in (1, 2, 3):
            for radius in (-1., 42.51, np.nan, np.inf):
                with self.subTest(faces=faces, radius=radius), self.assertRaises(ValueError):
                    make_demo_cloud(**{f'corner_fillet_{faces}face_radius_mm': radius})

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
                for faces, radius in ((1, 12.), (2, 8.), (3, 5.)):
                    window.demo_corner_fillet_sliders[faces].double_value = radius
                window.demo_edge_fillet_slider.double_value = 7.
                window._make_demo()
                variant = window._open_dataset.call_args.args[0]
                self.assertNotEqual(variant, default)
                metadata = json.loads((variant/'metadata.json').read_text())
                self.assertEqual(metadata['noise_sigma_mm'], 1.)
                self.assertEqual(metadata['roughness_amplitude_mm'], 2.)
                self.assertEqual(metadata['roughness_wavelength_mm'], 25.)
                for faces, radius in ((1, 12.), (2, 8.), (3, 5.)):
                    self.assertEqual(metadata[f'corner_fillet_{faces}face_radius_mm'], radius)
                self.assertEqual(metadata['edge_fillet_radius_mm'], 7.)
                self.assertEqual(metadata['seed'], 7)
                _, points, _ = lab.load_cloud(variant)
                self.assertEqual(lab.load_corners(variant, len(points)),
                                 [{'id': 'corner_001', 'point_index': len(points)-1}])
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
            for radii, edge_radius in (((15., 0., 0.), 0.), ((0., 15., 0.), 0.),
                                      ((0., 0., 15.), 0.), ((0., 0., 0.), 15.), ((12., 8., 5.), 7.)):
                params = {f'corner_fillet_{faces}face_radius_mm': radius for faces, radius in enumerate(radii, 1)}
                for faces, radius in enumerate(radii, 1):
                    window.demo_corner_fillet_sliders[faces].double_value = radius
                window.demo_edge_fillet_slider.double_value = edge_radius
                window._make_demo()
                variant = window._open_dataset.call_args.args[0]
                self.assertNotEqual(variant, default)
                _, points, colors = lab.load_cloud(variant)
                expected, rgb, corner = make_demo_cloud(**params, edge_fillet_radius_mm=edge_radius)
                np.testing.assert_array_equal(points, expected)
                np.testing.assert_array_equal(colors, rgb)
                self.assertEqual(lab.load_corners(variant, len(points)),
                                 [{'id': 'corner_001', 'point_index': corner}])
                self.assertEqual(saved, {p.name: p.read_bytes() for p in default.iterdir()})

    def test_gui_sliders_only_update_labels_and_invalid_apply_does_not_write(self):
        window = make_window()
        with patch.object(lab, 'make_demo_cloud', side_effect=AssertionError('regenerated')):
            window.demo_noise_slider.double_value = 1.
            for faces, radius in ((1, 12.), (2, 8.), (3, 5.)):
                window.demo_corner_fillet_sliders[faces].double_value = radius
            window.demo_edge_fillet_slider.double_value = 7.
            window._demo_surface_changed(1.)
        self.assertIn('1 mm', window.demo_noise_label.text)
        for faces, radius in ((1, 12), (2, 8), (3, 5)):
            self.assertIn(f'{radius} mm', window.demo_corner_fillet_labels[faces].text)
        self.assertIn('7 mm', window.demo_edge_fillet_label.text)
        with tempfile.TemporaryDirectory() as tmp, patch.object(lab, 'CAPTURES', Path(tmp)):
            window.demo_seed_edit.text_value = '3.5'
            window._error = Mock()
            window._open_dataset = Mock()
            window._make_demo()
            window._error.assert_called_once()
            window._open_dataset.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
            window.demo_seed_edit.text_value = '7'
            window.demo_corner_fillet_sliders[1].double_value = float('nan')
            window._error.reset_mock()
            window._make_demo()
            window._error.assert_called_once()
            window._open_dataset.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
