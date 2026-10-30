"""Depth noise injection and 3-D voxel preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from scipy.spatial import cKDTree
from pointcloud_inputs import PointCloudFrame, frame_from_rgbd
from pointcloud_inputs.visualization import require_open3d
from .constants import MIN_DEPTH_M, MAX_DEPTH_M

if TYPE_CHECKING:
    from .settings import DetectorConfig, ValidatorConfig


@dataclass

class NoiseConfig:

    gaussian_sigma_mm: float = 0.0

    outlier_rate: float = 0.0

    outlier_mm: float = 0.0

    dropout_rate: float = 0.0

    dropout_patches: int = 0

    dropout_patch_radius_px: int = 0

    quantization_mm: float = 0.0

    seed: int = 0


def validate_probability(name: str, value: float):

    if not 0.0 <= value <= 1.0:

        raise ValueError(f'{name} must be in [0,1]')


def corrupt_depth(depth_mm: np.ndarray, cfg: NoiseConfig, *,
                  min_depth_m: float = MIN_DEPTH_M,
                  max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:

    rng = np.random.default_rng(cfg.seed)

    out = np.asarray(depth_mm, dtype=np.float64).copy()

    valid = np.isfinite(out)

    if cfg.gaussian_sigma_mm > 0.0 and np.any(valid):

        out[valid] += rng.normal(0.0, cfg.gaussian_sigma_mm, np.count_nonzero(valid))

    valid = np.isfinite(out)

    if cfg.outlier_rate > 0.0 and cfg.outlier_mm > 0.0 and np.any(valid):

        flat = np.flatnonzero(valid.ravel())

        n = min(len(flat), int(round(cfg.outlier_rate * len(flat))))

        if n > 0:

            ids = rng.choice(flat, n, replace=False)

            mag = rng.uniform(0.5 * cfg.outlier_mm, cfg.outlier_mm, n)

            sign = rng.choice(np.array([-1.0, 1.0]), n)

            out.ravel()[ids] += sign * mag

    if cfg.quantization_mm > 0.0:

        valid = np.isfinite(out)

        out[valid] = np.round(out[valid] / cfg.quantization_mm) * cfg.quantization_mm

    valid = np.isfinite(out)

    bad = valid & ((out < min_depth_m * 1000.0) | (out > max_depth_m * 1000.0))

    out[bad] = np.nan

    if cfg.dropout_rate > 0.0:

        valid = np.isfinite(out)

        drop = valid & (rng.random(out.shape) < cfg.dropout_rate)

        out[drop] = np.nan

    if cfg.dropout_patches > 0 and cfg.dropout_patch_radius_px > 0:

        yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]

        for _ in range(cfg.dropout_patches):

            ys, xs = np.nonzero(np.isfinite(out))

            if len(xs) == 0:

                break

            j = int(rng.integers(0, len(xs)))

            cx = int(xs[j])

            cy = int(ys[j])

            r = cfg.dropout_patch_radius_px

            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

            out[mask] = np.nan

    return out.astype(np.float32)


def _noise_is_enabled(noise: "NoiseConfig") -> bool:
    return any((
        noise.gaussian_sigma_mm > 0.0,
        noise.outlier_rate > 0.0 and noise.outlier_mm > 0.0,
        noise.dropout_rate > 0.0,
        noise.dropout_patches > 0 and noise.dropout_patch_radius_px > 0,
        noise.quantization_mm > 0.0,
    ))


def apply_noise_to_frame(frame: PointCloudFrame, noise: "NoiseConfig", *,
                         min_depth_m: float = MIN_DEPTH_M,
                         max_depth_m: float = MAX_DEPTH_M) -> PointCloudFrame:
    """Apply the demo's optional synthetic depth corruption to a shared frame."""
    if not _noise_is_enabled(noise):
        return frame
    if frame.depth_m is None or frame.intrinsics is None:
        raise ValueError("Depth noise requires an organized point-cloud frame")

    depth_mm = np.asarray(frame.depth_m, dtype=np.float32) * 1000.0
    depth_mm[~np.isfinite(depth_mm) | (depth_mm <= 0.0)] = np.nan
    noisy_depth_m = corrupt_depth(
        depth_mm, noise, min_depth_m=min_depth_m, max_depth_m=max_depth_m
    ) / 1000.0
    metadata = dict(frame.metadata)
    metadata["synthetic_noise"] = {
        "gaussian_sigma_mm": float(noise.gaussian_sigma_mm),
        "outlier_rate": float(noise.outlier_rate),
        "outlier_mm": float(noise.outlier_mm),
        "dropout_rate": float(noise.dropout_rate),
        "dropout_patches": int(noise.dropout_patches),
        "dropout_patch_radius_px": int(noise.dropout_patch_radius_px),
        "quantization_mm": float(noise.quantization_mm),
        "seed": int(noise.seed),
    }
    return frame_from_rgbd(
        noisy_depth_m,
        frame.intrinsics,
        source=frame.source,
        rgb=frame.rgb,
        unregistered_rgb=frame.unregistered_rgb,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        metadata=metadata,
    )


def voxel_downsample_3d(points_mm: np.ndarray, voxel_mm: float) -> np.ndarray:

    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)

    if len(pts) == 0:

        return np.empty((0, 3), dtype=np.float64)

    o3d = require_open3d()
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pts)

    down = pcd.voxel_down_sample(float(voxel_mm))

    return np.asarray(down.points, dtype=np.float64)


def prepare_frozen_from_frame(
    frame: PointCloudFrame,
    cfg: ValidatorConfig | DetectorConfig,
) -> tuple[dict, np.ndarray]:
    """Adapt the shared input model to the existing diagnostic data structure."""
    if frame.depth_m is None or frame.intrinsics is None:
        raise ValueError("The STL tangent demo requires an organized depth frame")
    points = voxel_downsample_3d(frame.points_mm, cfg.voxel_3d_mm)
    if len(points) < cfg.min_neighbors:
        raise RuntimeError("Too few points after 3-D voxel downsampling")
    frozen = {
        "rgb": frame.preview_rgb(),
        "depth_m": frame.depth_m.copy(),
        "intr": frame.intrinsics,
        "points_raw_mm": np.asarray(frame.points_mm, dtype=np.float64),
        "points_voxel_mm": points,
        "pointcloud_frame": frame,
        "selected_pixel": None,
    }
    frozen["organized_plane_context"] = build_organized_plane_context(
        frame, points
    )
    return frozen, points


def build_organized_plane_context(
    frame: PointCloudFrame,
    scene_points_mm: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Preserve RGB-D pixel adjacency while targeting voxel scene indices.

    Open3D voxel centroids do not retain source indices.  Each valid organized
    pixel is therefore mapped to its nearest voxel centroid.  CornerVINS AHC
    still fits the original full-resolution RGB-D points; this mapping is used
    only by the downstream adapter so its support IDs remain in the canonical
    voxel scene-index space.
    """

    if frame.pixel_to_point_index is None:
        raise ValueError(
            "CornerVINS mode requires the original pixel-to-point index map"
        )
    scene = np.asarray(scene_points_mm, dtype=np.float64)
    if scene.ndim != 2 or scene.shape[1] != 3 or not np.isfinite(scene).all():
        raise ValueError("scene_points_mm must be finite with shape (N, 3)")
    if len(scene) == 0:
        raise ValueError("CornerVINS mode cannot map an empty voxel cloud")
    raw_mapping = np.asarray(frame.pixel_to_point_index, dtype=np.int64)
    valid = raw_mapping >= 0
    organized = np.full((*raw_mapping.shape, 3), np.nan, dtype=np.float64)
    organized[valid] = np.asarray(frame.points_mm, dtype=np.float64)[raw_mapping[valid]]
    distance, nearest = cKDTree(scene).query(organized[valid], k=1)
    scene_mapping = np.full(raw_mapping.shape, -1, dtype=np.int64)
    scene_mapping[valid] = np.asarray(nearest, dtype=np.int64)
    return {
        "organized_points_mm": organized,
        "valid_mask": valid,
        "organized_to_scene_index": scene_mapping,
        "mapping_max_distance_mm": float(np.max(distance)),
        "mapping_mean_distance_mm": float(np.mean(distance)),
    }
