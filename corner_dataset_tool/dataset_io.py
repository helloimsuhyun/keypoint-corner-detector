"""Persistent scene and annotation operations; no camera / GUI import required.

Coordinates: millimetres in the RealSense *aligned color optical* frame
(X right, Y down, Z forward). Raw point indices always refer to the unchanged
row-major, valid-depth point array saved in points.npz.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _points_and_rgb(xyz_mm: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_mm, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz_mm must have shape (N, 3) with N > 0")
    if colors.shape != xyz.shape:
        raise ValueError("rgb must have shape (N, 3)")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("Point cloud contains non-finite coordinates")
    return xyz, colors


def make_o3d_cloud(xyz_mm: np.ndarray, rgb: np.ndarray):
    """Make Open3D point cloud without changing XYZ units/order."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz_mm, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64) / 255.0)
    return pcd


def write_ply(path: Path, xyz_mm: np.ndarray, rgb: np.ndarray) -> None:
    import open3d as o3d

    if not o3d.io.write_point_cloud(str(path), make_o3d_cloud(xyz_mm, rgb), write_ascii=False):
        raise IOError(f"PLY write failed: {path}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")
    os.replace(temp, path)


def create_scene(
    scene_dir: str | Path,
    xyz_mm: np.ndarray,
    rgb: np.ndarray,
    *,
    pixel_index: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    rgb_image: np.ndarray | None = None,
    depth_z16: np.ndarray | None = None,
) -> Path:
    """Create an immutable raw capture. Refuses to overwrite an existing scene."""
    root = Path(scene_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Scene already exists: {root}")
    xyz, colors = _points_and_rgb(xyz_mm, rgb)
    if pixel_index is None:
        pixel_index = np.arange(len(xyz), dtype=np.int32)
    pixel_index = np.asarray(pixel_index, dtype=np.int32)
    if pixel_index.shape != (len(xyz),):
        raise ValueError("pixel_index must have N entries")
    root.mkdir(parents=True, exist_ok=False)
    try:
        # NPZ maintains the exact original point ordering for persistent GT indices.
        np.savez_compressed(root / "points.npz", xyz_mm=xyz, rgb=colors, pixel_index=pixel_index)
        write_ply(root / "raw.ply", xyz, colors)
        meta = {"schema_version": 1, "coordinate_frame": "color_optical", "unit": "mm", "point_count": len(xyz)}
        if metadata:
            meta.update(metadata)
        _write_json_atomic(root / "metadata.json", meta)
        if rgb_image is not None:
            import cv2
            if not cv2.imwrite(str(root / "rgb.png"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)):
                raise IOError("Failed saving rgb.png")
        if depth_z16 is not None:
            np.save(root / "depth.npy", np.asarray(depth_z16, dtype=np.uint16))
    except Exception:
        # Partial scenes are not re-used: user can remove incomplete dir if disk fills.
        raise
    return root


class SceneData:
    def __init__(self, root: Path, xyz_mm: np.ndarray, rgb: np.ndarray, pixel_index: np.ndarray):
        self.root = root
        self.xyz_mm, self.rgb = _points_and_rgb(xyz_mm, rgb)
        self.pixel_index = np.asarray(pixel_index, dtype=np.int32)
        if self.pixel_index.shape != (len(self.xyz_mm),):
            raise ValueError("Malformed pixel_index array")
        self.full_min = self.xyz_mm.min(axis=0).astype(np.float64)
        self.full_max = self.xyz_mm.max(axis=0).astype(np.float64)
        self.roi_min = self.full_min.copy() - 0.01
        self.roi_max = self.full_max.copy() + 0.01
        self.corners: list[dict[str, int]] = []
        self.next_id = 1

    @classmethod
    def load(cls, path: str | Path) -> "SceneData":
        root = Path(path).expanduser().resolve()
        with np.load(root / "points.npz", allow_pickle=False) as data:
            result = cls(root, data["xyz_mm"], data["rgb"], data["pixel_index"])
        annot_path = root / "annotations.json"
        if annot_path.exists():
            payload = json.loads(annot_path.read_text(encoding="utf-8"))
            if payload.get("unit") != "mm":
                raise ValueError("Annotation unit must be mm")
            roi = payload["roi"]
            result.set_roi(roi["min_mm"], roi["max_mm"])
            used_ids, used_indices = set(), set()
            for entry in payload.get("corners", []):
                id_, ix = int(entry["id"]), int(entry["raw_point_index"])
                if id_ in used_ids or ix in used_indices or not (0 <= ix < len(result.xyz_mm)):
                    raise ValueError("Annotation has invalid/duplicate corner ID or index")
                saved_xyz = np.asarray(entry["xyz_mm"], dtype=np.float64)
                if saved_xyz.shape != (3,) or not np.allclose(saved_xyz, result.xyz_mm[ix], atol=0.005):
                    raise ValueError(f"GT coordinates mismatch raw point index {ix}")
                result.corners.append({"id": id_, "raw_point_index": ix})
                used_ids.add(id_)
                used_indices.add(ix)
            result.next_id = max(used_ids, default=0) + 1
        return result

    def roi_mask(self) -> np.ndarray:
        return np.logical_and(self.xyz_mm >= self.roi_min, self.xyz_mm <= self.roi_max).all(axis=1)

    def roi_indices(self) -> np.ndarray:
        return np.flatnonzero(self.roi_mask())

    def set_roi(self, low: np.ndarray | list, high: np.ndarray | list) -> None:
        a, b = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("ROI bounds must be finite 3D vectors")
        if np.any(a >= b):
            raise ValueError("ROI dimensions must be positive")
        self.roi_min, self.roi_max = a.copy(), b.copy()

    def add_corner(self, raw_point_index: int) -> int | None:
        ix = int(raw_point_index)
        if not 0 <= ix < len(self.xyz_mm):
            raise IndexError(ix)
        if any(c["raw_point_index"] == ix for c in self.corners):
            return None
        cid = self.next_id
        self.next_id += 1
        self.corners.append({"id": cid, "raw_point_index": ix})
        return cid

    def remove_corner(self, corner_id: int) -> bool:
        before = len(self.corners)
        self.corners = [c for c in self.corners if c["id"] != corner_id]
        return len(self.corners) != before

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scene_id": self.root.name,
            "unit": "mm",
            "coordinate_frame": "color_optical",
            "roi": {"type": "aabb", "min_mm": self.roi_min.tolist(), "max_mm": self.roi_max.tolist()},
            "corners": [
                {
                    "id": c["id"],
                    "raw_point_index": c["raw_point_index"],
                    "xyz_mm": self.xyz_mm[c["raw_point_index"]].astype(float).tolist(),
                    "pixel_index": int(self.pixel_index[c["raw_point_index"]]),
                    "inside_roi": bool(np.all((self.xyz_mm[c["raw_point_index"]] >= self.roi_min) & (self.xyz_mm[c["raw_point_index"]] <= self.roi_max))),
                }
                for c in self.corners
            ],
        }

    def save_annotations(self) -> None:
        _write_json_atomic(self.root / "annotations.json", self.to_json())

    def export_roi(self) -> int:
        indices = self.roi_indices()
        if len(indices) == 0:
            raise ValueError("ROI is empty; change its center or size")
        # Stable mapping into raw points is included in addition to the ROI PLY.
        np.save(self.root / "roi_indices.npy", indices.astype(np.int64))
        write_ply(self.root / "roi.ply", self.xyz_mm[indices], self.rgb[indices])
        self.save_annotations()
        return len(indices)
