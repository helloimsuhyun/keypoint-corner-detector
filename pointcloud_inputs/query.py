"""Common query representation and interactive query-picking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .common import PointCloudFrame


@dataclass(frozen=True)
class QuerySelection:
    """Point indices selected from one source point cloud."""

    indices: np.ndarray
    points_mm: np.ndarray
    method: str
    pixels_uv: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        points = np.asarray(self.points_mm, dtype=np.float32)
        if points.shape != (len(indices), 3):
            raise ValueError("points_mm must have shape (len(indices), 3)")
        if np.any(indices < 0):
            raise ValueError("Query indices must be non-negative")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Query indices must be unique")
        if not np.all(np.isfinite(points)):
            raise ValueError("Query points contain non-finite coordinates")
        pixels = self.pixels_uv
        if pixels is not None:
            pixels = np.asarray(pixels, dtype=np.int64)
            if pixels.shape != (len(indices), 2):
                raise ValueError("pixels_uv must have shape (len(indices), 2)")
            object.__setattr__(self, "pixels_uv", np.ascontiguousarray(pixels))
        object.__setattr__(self, "indices", np.ascontiguousarray(indices))
        object.__setattr__(self, "points_mm", np.ascontiguousarray(points))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def index(self) -> int:
        """Return the only index in a single-query selection."""

        if len(self) != 1:
            raise ValueError(f"Expected one query, found {len(self)}")
        return int(self.indices[0])

    @property
    def point_mm(self) -> np.ndarray:
        """Return the only point in a single-query selection."""

        if len(self) != 1:
            raise ValueError(f"Expected one query, found {len(self)}")
        return self.points_mm[0].copy()


def queries_from_indices(
    source: PointCloudFrame | np.ndarray,
    indices: np.ndarray | list[int] | tuple[int, ...],
    *,
    method: str = "indices",
    metadata: Mapping[str, Any] | None = None,
) -> QuerySelection:
    """Create a selection from indices into the original point array."""

    points = _source_points(source)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= len(points)):
        raise IndexError("Query index is outside the source point cloud")
    pixels = pixels_for_indices(source, selected) if isinstance(source, PointCloudFrame) else None
    return QuerySelection(
        indices=selected,
        points_mm=points[selected],
        pixels_uv=pixels,
        method=method,
        metadata={} if metadata is None else metadata,
    )


def query_from_pixel(
    frame: PointCloudFrame,
    u: int,
    v: int,
    *,
    max_radius_px: int = 2,
) -> QuerySelection | None:
    """Create one query from a pixel in ``frame.preview_rgb()``."""

    index = frame.query_index_at_pixel(u, v, max_radius_px=max_radius_px)
    if index is None:
        return None
    pixels = pixels_for_indices(frame, np.array([index], dtype=np.int64))
    return QuerySelection(
        indices=np.array([index], dtype=np.int64),
        points_mm=frame.points_mm[[index]],
        pixels_uv=pixels,
        method="original-pixel",
        metadata={"clicked_pixel_uv": [int(u), int(v)]},
    )


def select_query_from_original(
    frame: PointCloudFrame,
    *,
    max_radius_px: int = 2,
    confirm_on_click: bool = True,
    window_name: str = "Original input | left click: query | Enter: confirm | Q: cancel",
) -> QuerySelection | None:
    """Interactively select one query from the original RGB/depth pixel grid.

    The displayed image is always geometrically aligned with the frame's pixel
    map. For Kinect 7-Scenes this means the depth preview, not unregistered RGB.
    """

    if frame.pixel_to_point_index is None:
        raise ValueError("The frame has no original pixel-to-point mapping")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required for image query selection") from exc

    base_rgb = frame.preview_rgb()
    state: dict[str, QuerySelection | bool | None] = {
        "selection": None,
        "done": False,
    }

    def redraw() -> None:
        image = base_rgb.copy()
        selection = state["selection"]
        if isinstance(selection, QuerySelection) and selection.pixels_uv is not None:
            u, v = (int(value) for value in selection.pixels_uv[0])
            cv2.circle(image, (u, v), 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                image, (u, v), (255, 40, 40), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA
            )
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def on_mouse(event: int, u: int, v: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selection = query_from_pixel(frame, u, v, max_radius_px=max_radius_px)
        if selection is None:
            return
        state["selection"] = selection
        state["done"] = bool(confirm_on_click)
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    try:
        while not state["done"]:
            key = int(cv2.waitKey(20)) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key in (ord("r"), ord("R")):
                state["selection"] = None
                redraw()
            if key in (13, 32) and state["selection"] is not None:
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return None
            except cv2.error:
                return None
        selection = state["selection"]
        return selection if isinstance(selection, QuerySelection) else None
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def select_queries_from_point_cloud(
    source: PointCloudFrame | np.ndarray,
    *,
    max_queries: int | None = 1,
    window_name: str = "3D query picker | Shift+left click: pick | Q: finish",
) -> QuerySelection | None:
    """Pick queries directly from the original 3-D cloud with Open3D."""

    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be >= 1 or None")
    from .visualization import make_point_cloud_geometry, require_open3d

    o3d = require_open3d()
    editor = o3d.visualization.VisualizerWithEditing()
    if not editor.create_window(window_name=window_name, width=1100, height=800):
        raise RuntimeError("Failed to create the Open3D query picker")
    editor.add_geometry(make_point_cloud_geometry(source, o3d=o3d))
    try:
        editor.run()
        picked = [int(index) for index in editor.get_picked_points()]
    finally:
        editor.destroy_window()
    if max_queries is not None:
        picked = picked[:max_queries]
    return None if not picked else queries_from_indices(source, picked, method="original-3d")


def pixels_for_indices(frame: PointCloudFrame, indices: np.ndarray) -> np.ndarray:
    """Return exact ``(u, v)`` pixels for point indices, or ``(-1, -1)``."""

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    output = np.full((len(selected), 2), -1, dtype=np.int64)
    mapping = frame.pixel_to_point_index
    if mapping is None or not len(selected):
        return output
    vv, uu = np.nonzero(mapping >= 0)
    inverse = np.full(len(frame.points_mm), -1, dtype=np.int64)
    inverse[mapping[vv, uu]] = np.arange(len(vv), dtype=np.int64)
    valid = (selected >= 0) & (selected < len(inverse))
    locations = np.full(len(selected), -1, dtype=np.int64)
    locations[valid] = inverse[selected[valid]]
    found = locations >= 0
    output[found, 0] = uu[locations[found]]
    output[found, 1] = vv[locations[found]]
    return output


def _source_points(source: PointCloudFrame | np.ndarray) -> np.ndarray:
    points = source.points_mm if isinstance(source, PointCloudFrame) else source
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Point source must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("Point source contains non-finite coordinates")
    return array
