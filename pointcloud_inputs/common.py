"""Shared point-cloud input types and RGB-D projection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for an image of ``width`` x ``height``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=float)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if not np.all(np.isfinite(values)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")

    @property
    def ppx(self) -> float:
        """RealSense-compatible alias for ``cx``."""

        return self.cx

    @property
    def ppy(self) -> float:
        """RealSense-compatible alias for ``cy``."""

        return self.cy

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics scaled to another image resolution."""

        sx = float(width) / self.width
        sy = float(height) / self.height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass
class PointCloudFrame:
    """One input frame in camera coordinates.

    Coordinates are millimeters with +X right, +Y down, and +Z forward.
    ``rgb`` is pixel-aligned with ``depth_m`` when present. A Kinect RGB frame
    is instead stored in ``unregistered_rgb`` because 7-Scenes RGB and depth
    are not calibrated to the same pixel grid.
    """

    points_mm: np.ndarray
    source: str
    intrinsics: CameraIntrinsics | None = None
    depth_m: np.ndarray | None = None
    rgb: np.ndarray | None = None
    colors_rgb: np.ndarray | None = None
    pixel_to_point_index: np.ndarray | None = None
    unregistered_rgb: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_mm must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_mm contains non-finite coordinates")
        self.points_mm = np.ascontiguousarray(points)

        if self.colors_rgb is not None:
            colors = np.asarray(self.colors_rgb)
            if colors.shape != (len(points), 3):
                raise ValueError("colors_rgb must have shape (N, 3)")
            if colors.dtype != np.uint8:
                if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1.0:
                    colors = np.rint(np.clip(colors, 0.0, 1.0) * 255.0)
                colors = np.clip(colors, 0, 255).astype(np.uint8)
            self.colors_rgb = np.ascontiguousarray(colors)

        expected_shape: tuple[int, int] | None = None
        if self.intrinsics is not None:
            expected_shape = (self.intrinsics.height, self.intrinsics.width)
        if self.depth_m is not None:
            depth = np.asarray(self.depth_m, dtype=np.float32)
            if depth.ndim != 2:
                raise ValueError("depth_m must have shape (H, W)")
            if expected_shape is not None and depth.shape != expected_shape:
                raise ValueError("depth_m shape does not match intrinsics")
            expected_shape = depth.shape
            self.depth_m = np.ascontiguousarray(depth)
        if self.rgb is not None:
            rgb = _as_rgb_image(self.rgb, "rgb")
            if expected_shape is not None and rgb.shape[:2] != expected_shape:
                raise ValueError("rgb shape does not match depth/intrinsics")
            self.rgb = rgb
        if self.unregistered_rgb is not None:
            self.unregistered_rgb = _as_rgb_image(
                self.unregistered_rgb, "unregistered_rgb"
            )
        if self.pixel_to_point_index is not None:
            mapping = np.asarray(self.pixel_to_point_index, dtype=np.int64)
            if mapping.ndim != 2:
                raise ValueError("pixel_to_point_index must have shape (H, W)")
            if expected_shape is not None and mapping.shape != expected_shape:
                raise ValueError("pixel map shape does not match depth/intrinsics")
            valid = mapping >= 0
            if np.any(mapping[valid] >= len(points)):
                raise ValueError("pixel map contains an out-of-range point index")
            self.pixel_to_point_index = np.ascontiguousarray(mapping)
        self.metadata = dict(self.metadata)

    def query_index_at_pixel(
        self, u: int, v: int, *, max_radius_px: int = 0
    ) -> int | None:
        """Map a clicked pixel to a point index, optionally searching nearby."""

        mapping = self.pixel_to_point_index
        if mapping is None:
            raise ValueError("This frame has no pixel-to-point mapping")
        h, w = mapping.shape
        u, v = int(u), int(v)
        if not (0 <= u < w and 0 <= v < h):
            return None
        index = int(mapping[v, u])
        if index >= 0:
            return index
        radius = int(max_radius_px)
        if radius <= 0:
            return None
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
        patch = mapping[y0:y1, x0:x1]
        yy, xx = np.nonzero(patch >= 0)
        if not len(xx):
            return None
        distances2 = (xx + x0 - u) ** 2 + (yy + y0 - v) ** 2
        nearest = int(np.argmin(distances2))
        return int(patch[yy[nearest], xx[nearest]])

    def point_at_pixel(
        self, u: int, v: int, *, max_radius_px: int = 0
    ) -> np.ndarray | None:
        """Return the 3-D point for a clicked pixel, or ``None`` if invalid."""

        index = self.query_index_at_pixel(u, v, max_radius_px=max_radius_px)
        return None if index is None else self.points_mm[index].copy()

    def preview_rgb(self) -> np.ndarray:
        """Return a click-safe RGB preview aligned to the point-index map."""

        if self.rgb is not None:
            return self.rgb.copy()
        if self.depth_m is None:
            raise ValueError("This frame has neither aligned RGB nor depth")
        return depth_preview_rgb(self.depth_m)

    def save_ply(self, path: str | Path, *, binary: bool = True) -> Path:
        """Write this frame to a PLY file and return its resolved path."""

        return save_ply(path, self.points_mm, self.colors_rgb, binary=binary)


def _as_rgb_image(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def frame_from_rgbd(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    source: str,
    rgb: np.ndarray | None = None,
    unregistered_rgb: np.ndarray | None = None,
    min_depth_m: float = 0.01,
    max_depth_m: float = 5.0,
    metadata: Mapping[str, Any] | None = None,
) -> PointCloudFrame:
    """Deproject a depth image into a :class:`PointCloudFrame`."""

    if not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError("Require 0 <= min_depth_m < max_depth_m")
    depth = np.asarray(depth_m, dtype=np.float32)
    expected = (intrinsics.height, intrinsics.width)
    if depth.shape != expected:
        raise ValueError(f"Expected depth shape {expected}, got {depth.shape}")
    aligned_rgb = None if rgb is None else _as_rgb_image(rgb, "rgb")
    if aligned_rgb is not None and aligned_rgb.shape[:2] != depth.shape:
        raise ValueError("Aligned RGB and depth dimensions differ")

    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    vv, uu = np.nonzero(valid)
    z_m = depth[vv, uu].astype(np.float64)
    x_m = (uu.astype(np.float64) - intrinsics.cx) * z_m / intrinsics.fx
    y_m = (vv.astype(np.float64) - intrinsics.cy) * z_m / intrinsics.fy
    points_mm = np.column_stack((x_m, y_m, z_m)).astype(np.float32) * 1000.0

    pixel_map = np.full(depth.shape, -1, dtype=np.int64)
    pixel_map[vv, uu] = np.arange(len(points_mm), dtype=np.int64)
    colors = None if aligned_rgb is None else aligned_rgb[vv, uu]
    return PointCloudFrame(
        points_mm=points_mm,
        source=source,
        intrinsics=intrinsics,
        depth_m=depth,
        rgb=aligned_rgb,
        colors_rgb=colors,
        pixel_to_point_index=pixel_map,
        unregistered_rgb=unregistered_rgb,
        metadata={} if metadata is None else metadata,
    )


def depth_preview_rgb(depth_m: np.ndarray) -> np.ndarray:
    """Build a dependency-free RGB depth preview for pixel picking."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    valid = np.isfinite(depth) & (depth > 0)
    image = np.full((*depth.shape, 3), (18, 20, 28), dtype=np.uint8)
    if not np.any(valid):
        return image
    low, high = np.percentile(depth[valid], (1.0, 99.0))
    if high <= low:
        high = low + 1e-6
    value = np.clip((depth[valid] - low) / (high - low), 0.0, 1.0)
    # Compact blue -> cyan -> yellow -> red map; exact colors are display-only.
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    image[valid] = np.rint(np.column_stack((red, green, blue)) * 255.0).astype(np.uint8)
    return image


def save_ply(
    path: str | Path,
    points_mm: np.ndarray,
    colors_rgb: np.ndarray | None = None,
    *,
    binary: bool = True,
) -> Path:
    """Save points (and optional uint8 RGB colors) as a PLY file."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_mm, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_mm must be a finite (N, 3) array")
    colors: np.ndarray | None = None
    if colors_rgb is not None:
        colors = np.asarray(colors_rgb)
        if colors.shape != (len(points), 3):
            raise ValueError("colors_rgb must have shape (N, 3)")
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    encoding = "binary_little_endian" if binary else "ascii"
    properties = ["property float x", "property float y", "property float z"]
    if colors is not None:
        properties += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header = "\n".join(
        [
            "ply",
            f"format {encoding} 1.0",
            "comment coordinates: millimeters, camera frame (+X right, +Y down, +Z forward)",
            f"element vertex {len(points)}",
            *properties,
            "end_header",
            "",
        ]
    ).encode("ascii")

    with target.open("wb") as stream:
        stream.write(header)
        if binary:
            fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if colors is not None:
                fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
            records = np.empty(len(points), dtype=np.dtype(fields, align=False))
            records["x"], records["y"], records["z"] = points.T
            if colors is not None:
                records["red"], records["green"], records["blue"] = colors.T
            records.tofile(stream)
        else:
            if colors is None:
                np.savetxt(stream, points, fmt="%.7g %.7g %.7g")
            else:
                rows = np.column_stack((points, colors))
                np.savetxt(stream, rows, fmt="%.7g %.7g %.7g %d %d %d")
    return target
