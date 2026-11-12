"""Deterministic three-plane corner shared by the lab and detector (mm)."""

import numpy as np


def make_demo_cloud(*, noise_sigma_mm: float = .25,
                    roughness_amplitude_mm: float = 0.,
                    roughness_wavelength_mm: float = 20.,
                    fillet_radius_mm: float = 0.,
                    seed: int = 7) -> tuple[np.ndarray, np.ndarray, int]:
    """Three-plane corner with optional edge/corner fillets and normal noise/bumps."""
    if (not np.isfinite([noise_sigma_mm, roughness_amplitude_mm, roughness_wavelength_mm]).all()
            or noise_sigma_mm < 0 or roughness_amplitude_mm < 0 or roughness_wavelength_mm <= 0):
        raise ValueError('Demo noise/amplitude must be finite >=0; wavelength must be finite >0 (mm)')
    if not np.isfinite(fillet_radius_mm) or not 0. <= fillet_radius_mm <= 85.:
        raise ValueError('Demo fillet radius must be finite in [0, 85] mm')
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
        if fillet_radius_mm:
            height = plane[:, dim].copy()
            plane[:, dim] = 0.
            # Project onto the radius-r offset of the inset octant [r, infinity)^3.
            normal = 1. - np.minimum(plane, fillet_radius_mm) / fillet_radius_mm
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            plane = np.maximum(plane, fillet_radius_mm) - fillet_radius_mm * normal
            plane += height[:, None] * normal
        blocks.append(plane + origin)
        colors.append(np.tile(np.array(color, dtype=np.uint8), (4200, 1)))
    xyz = np.vstack(blocks)
    rgb = np.vstack(colors)
    # Rounded demos use a surface reference; the former sharp corner is removed.
    corner_index = len(xyz)
    reference = origin + fillet_radius_mm * (1. - 1. / np.sqrt(3.))
    xyz = np.vstack([xyz, reference])
    rgb = np.vstack([rgb, np.array([255, 255, 255], dtype=np.uint8)])
    return xyz.astype(np.float32), rgb, corner_index

