"""Deterministic three-plane corner shared by the lab and detector (mm)."""

import numpy as np


def _round_demo_face(points: np.ndarray, face_axis: int, edge_radius_mm: float,
                     corner_radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Round the origin vertex locally and blend to the independent edge radius."""
    distance = np.linalg.norm(points, axis=1)
    width = 2. * max(edge_radius_mm, corner_radius_mm)
    t = np.clip((distance - np.sqrt(2.) * corner_radius_mm) / width, 0., 1.)
    radius = corner_radius_mm + (edge_radius_mm-corner_radius_mm) * t*t*(3.-2.*t)
    normal = np.zeros_like(points)
    normal[:, face_axis] = 1.
    positive = radius > 0.
    normal[positive] = 1. - np.minimum(points[positive], radius[positive, None]) / radius[positive, None]
    length = np.linalg.norm(normal, axis=1)
    normal /= length[:, None]
    rounded = np.maximum(points, radius[:, None]) - radius[:, None]*normal

    # Include the varying radius in the tangents, so noise follows the actual surface.
    tangents = []
    for axis in (i for i in range(3) if i != face_axis):
        basis = np.eye(3)[axis]
        dr = np.divide(points[:, axis], distance, out=np.zeros(len(points)), where=distance > 0.)
        dr *= (edge_radius_mm-corner_radius_mm) * 6.*t*(1.-t) / width
        dc = np.where(points < radius[:, None], dr[:, None], basis)
        delta = dc - basis
        tangent = dc - dr[:, None]*normal
        tangent -= (delta - normal*np.sum(normal*delta, axis=1)[:, None]) / length[:, None]
        tangents.append(tangent)
    surface_normal = np.cross(*tangents)
    surface_normal *= np.where(np.sum(surface_normal*normal, axis=1) < 0., -1., 1.)[:, None]
    surface_normal /= np.linalg.norm(surface_normal, axis=1)[:, None]
    return rounded, surface_normal


def make_demo_cloud(*, noise_sigma_mm: float = .25,
                    roughness_amplitude_mm: float = 0.,
                    roughness_wavelength_mm: float = 20.,
                    corner_fillet_1face_radius_mm: float = 0.,
                    corner_fillet_2face_radius_mm: float = 0.,
                    corner_fillet_3face_radius_mm: float = 0.,
                    edge_fillet_radius_mm: float = 0.,
                    seed: int = 7) -> tuple[np.ndarray, np.ndarray, int]:
    """Three faces with independent 1/2/3-face corner and edge fillet radii."""
    if (not np.isfinite([noise_sigma_mm, roughness_amplitude_mm, roughness_wavelength_mm]).all()
            or noise_sigma_mm < 0 or roughness_amplitude_mm < 0 or roughness_wavelength_mm <= 0):
        raise ValueError('Demo noise/amplitude must be finite >=0; wavelength must be finite >0 (mm)')
    corner_radii = (corner_fillet_1face_radius_mm, corner_fillet_2face_radius_mm,
                    corner_fillet_3face_radius_mm)
    if not np.isfinite(corner_radii).all() or any(not 0. <= radius <= 42.5 for radius in corner_radii):
        raise ValueError('Demo 1/2/3-face corner fillet radii must be finite in [0, 42.5] mm')
    if not np.isfinite(edge_fillet_radius_mm) or not 0. <= edge_fillet_radius_mm <= 85.:
        raise ValueError('Demo edge fillet radius must be finite in [0, 85] mm')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Demo seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    origin = np.array([0., 0., 700.])
    axes = [(0, [120, 180, 245]), (1, [245, 160, 110]), (2, [125, 225, 170])]
    blocks = []
    colors = []
    for dim, color in axes:
        u = rng.uniform(0, 85, size=(4200, 2))
        plane = np.zeros((4200, 3))
        other = [i for i in range(3) if i != dim]
        plane[:, other] = u
        plane[:, dim] += rng.normal(0, noise_sigma_mm, size=4200)
        if roughness_amplitude_mm:
            phase = 2. * np.pi * u / roughness_wavelength_mm
            plane[:, dim] += roughness_amplitude_mm * np.sin(phase[:, 0]) * np.sin(phase[:, 1])
        if corner_fillet_3face_radius_mm or edge_fillet_radius_mm:
            height = plane[:, dim].copy()
            plane[:, dim] = 0.
            plane, normal = _round_demo_face(plane, dim, edge_fillet_radius_mm, corner_fillet_3face_radius_mm)
            plane += height[:, None] * normal
        keep = np.ones(len(plane), dtype=bool)
        for vertex, radius in (((0., 85.), corner_fillet_2face_radius_mm),
                               ((85., 0.), corner_fillet_2face_radius_mm),
                               ((85., 85.), corner_fillet_1face_radius_mm)):
            if radius:
                inset = np.maximum(1. - np.abs(u-vertex) / radius, 0.)
                keep &= np.sum(inset*inset, axis=1) <= 1.
        plane = plane[keep]
        blocks.append(plane + origin)
        colors.append(np.tile(np.array(color, dtype=np.uint8), (len(plane), 1)))
    xyz = np.vstack(blocks)
    rgb = np.vstack(colors)
    # Rounded demos use a surface reference; the former sharp corner is removed.
    corner_index = len(xyz)
    reference = origin + corner_fillet_3face_radius_mm * (1. - 1. / np.sqrt(3.))
    xyz = np.vstack([xyz, reference])
    rgb = np.vstack([rgb, np.array([255, 255, 255], dtype=np.uint8)])
    return xyz.astype(np.float32), rgb, corner_index

