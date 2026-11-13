"""CED normal/tangent component magnitudes from centroid offsets and positional PCA."""

import numpy as np


def pca_normal_direction(covariance):
    """Return an unoriented unit normal, or NaNs when its eigenspace is ambiguous."""
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 64.*np.finfo(np.float64).eps*max(float(values[-1]), 0.)
    if values[1] <= tolerance or values[1]-values[0] <= tolerance:
        return np.full(3, np.nan)
    return vectors[:, 0]


def ced_component_magnitudes(centroid_offsets_mm, normals, radius_mm):
    """Return |normal projection|/R and tangent length/R; undefined planes stay NaN."""
    offsets = np.asarray(centroid_offsets_mm, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3 or normals.shape != offsets.shape:
        raise ValueError('CED offsets and PCA normals must have aligned (N,3) shapes')
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('CED component radius must be finite and >0 mm')
    lengths = np.linalg.norm(normals, axis=1)
    valid = (np.isfinite(offsets).all(axis=1) & np.isfinite(normals).all(axis=1)
             & np.isfinite(lengths) & (lengths > 0.))
    unit = normals[valid]/lengths[valid, None]
    projection = np.sum(offsets[valid]*unit, axis=1)
    normal = np.full(len(offsets), np.nan)
    tangent = np.full(len(offsets), np.nan)
    normal[valid] = np.abs(projection)/radius_mm
    tangent[valid] = np.linalg.norm(offsets[valid]-projection[:, None]*unit, axis=1)/radius_mm
    return normal, tangent
