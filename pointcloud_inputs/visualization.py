"""Visualization helpers for input clouds and selected queries."""

from __future__ import annotations

import sys
import types
from typing import Any, Sequence

import numpy as np

from .common import PointCloudFrame
from .query import QuerySelection, _source_points, queries_from_indices


def require_open3d():
    """Import geometry-only Open3D with its optional ML package disabled."""

    if "open3d" not in sys.modules and "open3d.ml" not in sys.modules:
        sys.modules["open3d.ml"] = types.ModuleType("open3d.ml")
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Open3D geometry/visualization could not be imported; check the "
            "installed Open3D and NumPy versions"
        ) from exc
    return o3d


def make_point_cloud_geometry(
    source: PointCloudFrame | np.ndarray,
    *,
    uniform_color: Sequence[float] = (0.55, 0.68, 0.82),
    o3d: Any | None = None,
):
    """Build an Open3D point cloud from a frame or ``(N, 3)`` array."""

    o3d = require_open3d() if o3d is None else o3d
    points = _source_points(source).astype(np.float64, copy=False)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if isinstance(source, PointCloudFrame) and source.colors_rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(
            source.colors_rgb.astype(np.float64) / 255.0
        )
    elif len(points):
        color = np.asarray(uniform_color, dtype=np.float64)
        if color.shape != (3,) or not np.all(np.isfinite(color)):
            raise ValueError("uniform_color must contain three finite values")
        cloud.paint_uniform_color(np.clip(color, 0.0, 1.0))
    return cloud


def make_query_marker_geometry(
    queries: QuerySelection,
    *,
    radius_mm: float,
    color: Sequence[float] = (1.0, 0.12, 0.08),
    o3d: Any | None = None,
):
    """Build one combined sphere mesh marking all selected query points."""

    if radius_mm <= 0 or not np.isfinite(radius_mm):
        raise ValueError("radius_mm must be finite and > 0")
    o3d = require_open3d() if o3d is None else o3d
    marker_color = np.asarray(color, dtype=np.float64)
    if marker_color.shape != (3,) or not np.all(np.isfinite(marker_color)):
        raise ValueError("color must contain three finite values")
    combined = o3d.geometry.TriangleMesh()
    for point in queries.points_mm:
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=float(radius_mm), resolution=6
        )
        sphere.translate(point.astype(np.float64))
        combined += sphere
    if len(combined.vertices):
        combined.compute_vertex_normals()
        combined.paint_uniform_color(np.clip(marker_color, 0.0, 1.0))
    return combined


def show_point_cloud(
    source: PointCloudFrame | np.ndarray,
    queries: QuerySelection | Sequence[int] | np.ndarray | None = None,
    *,
    query_radius_mm: float | None = None,
    point_size: float = 3.0,
    show_coordinate_frame: bool = True,
    window_name: str = "Point cloud input | red: selected queries",
) -> None:
    """Show an input cloud and optional FPS/click queries in Open3D."""

    if point_size <= 0:
        raise ValueError("point_size must be > 0")
    o3d = require_open3d()
    selection = _coerce_queries(source, queries)
    points = _source_points(source)
    geometries = [make_point_cloud_geometry(source, o3d=o3d)]
    if selection is not None and len(selection):
        radius = query_radius_mm
        if radius is None:
            extent = np.ptp(points.astype(np.float64), axis=0) if len(points) else np.ones(3)
            radius = float(np.clip(np.linalg.norm(extent) * 0.003, 0.5, 5.0))
        geometries.append(
            make_query_marker_geometry(selection, radius_mm=radius, o3d=o3d)
        )
    if show_coordinate_frame:
        extent = np.ptp(points.astype(np.float64), axis=0) if len(points) else np.ones(3)
        size = float(np.clip(np.linalg.norm(extent) * 0.08, 5.0, 100.0))
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size))

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(window_name=window_name, width=1100, height=800):
        raise RuntimeError("Failed to create the Open3D visualization window")
    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)
        render = visualizer.get_render_option()
        if render is not None:
            render.background_color = np.array([0.025, 0.025, 0.025])
            render.point_size = float(point_size)
        visualizer.run()
    finally:
        visualizer.destroy_window()


def draw_queries_on_original(
    frame: PointCloudFrame,
    queries: QuerySelection | Sequence[int] | np.ndarray | None = None,
    *,
    marker_radius_px: int = 8,
) -> np.ndarray:
    """Return the original click-safe RGB preview with query markers."""

    if marker_radius_px < 1:
        raise ValueError("marker_radius_px must be >= 1")
    image = frame.preview_rgb()
    selection = _coerce_queries(frame, queries)
    if selection is None or not len(selection) or selection.pixels_uv is None:
        return image
    height, width = image.shape[:2]
    radius = int(marker_radius_px)
    for u_value, v_value in selection.pixels_uv:
        u, v = int(u_value), int(v_value)
        if not (0 <= u < width and 0 <= v < height):
            continue
        x0, x1 = max(0, u - radius), min(width - 1, u + radius)
        y0, y1 = max(0, v - radius), min(height - 1, v + radius)
        image[max(0, v - 1) : min(height, v + 2), x0 : x1 + 1] = (255, 40, 40)
        image[y0 : y1 + 1, max(0, u - 1) : min(width, u + 2)] = (255, 40, 40)
        image[max(0, y0) : min(height, y0 + 2), x0 : x1 + 1] = (255, 255, 255)
        image[max(0, y1 - 1) : min(height, y1 + 1), x0 : x1 + 1] = (255, 255, 255)
        image[y0 : y1 + 1, max(0, x0) : min(width, x0 + 2)] = (255, 255, 255)
        image[y0 : y1 + 1, max(0, x1 - 1) : min(width, x1 + 1)] = (255, 255, 255)
    return image


def show_original(
    frame: PointCloudFrame,
    queries: QuerySelection | Sequence[int] | np.ndarray | None = None,
    *,
    window_name: str = "Original input | queries",
) -> None:
    """Show the RGB/depth-aligned original with optional query overlays."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required to show the original frame") from exc
    image = draw_queries_on_original(frame, queries)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    try:
        while True:
            key = int(cv2.waitKey(20)) & 0xFF
            if key in (ord("q"), ord("Q"), 27, 13, 32):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def _coerce_queries(
    source: PointCloudFrame | np.ndarray,
    queries: QuerySelection | Sequence[int] | np.ndarray | None,
) -> QuerySelection | None:
    if queries is None:
        return None
    if isinstance(queries, QuerySelection):
        return queries
    return queries_from_indices(source, np.asarray(queries, dtype=np.int64), method="display")
