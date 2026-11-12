#!/usr/bin/env python3
"""RealSense corner lab: Q10, RANSAC, four full-scene extrema NMS modes, PCA fields.

Commands:
  python corner_lab_robust_max_gap.py --demo
  python corner_lab_open3d_cone.py --camera
  python corner_lab_open3d_cone.py --open /path/to/cloud.npz
  python corner_lab_open3d_cone.py --scene 2
  python corner_lab_open3d_cone.py --scene 0002 --dataset /home/choisuhyun/keypoint-corner-detector/corner_dataset_tool/dataset

Ctrl+Left click a 3D point to label it / inspect a saved corner.
All 3D positions, vectors, and neighborhood radii use millimetres.
The spherical view uses unit directions, so its coordinates are dimensionless.
The histogram shows cos(theta) = unit-direction dot normalized mean direction,
with the Q10 threshold defining a cone containing approximately 90% of samples.
All geometric statistics and Q10 refinement use ONE globally voxel-downsampled cloud.
Q10 refinement, spherical RANSAC and gap validation use one fixed radius.
Each Refine click searches DOWNSAMPLED points around the PREVIOUS refined point.
Original points and label indices remain unchanged in saved files.
Red marker = original; lime marker = current refined point; white polyline = step history.
Each step is locally non-decreasing in Q10, but global localization is not guaranteed.
Discard isolated angular components before gap/span computation.
Per-plane span threshold is an experimental edge-like test, not a PCL corner detector.
Requires NumPy + Open3D; Matplotlib for the optional eigenvalue field plot.
Eigenvalue field is a diagnostic overlay; it does not modify saved corner labels,
Q10 refinement, or the Fusion detector in a separate repository.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
import tempfile
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np

from pointcloud_inputs.demo import make_demo_cloud
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

@dataclass(frozen=True)
class CentroidAnalysis:
    point_mm: np.ndarray
    centroid_mm: np.ndarray
    vector_mm: np.ndarray
    normal: np.ndarray | None  # unit vector; plane is normal·(x-p)=0
    neighbor_count: int
    vector_length_mm: float
    reliability_threshold_mm: float


def analyze_neighborhood(point_mm: np.ndarray, neighbors_mm: np.ndarray,
                         radius_mm: float) -> CentroidAnalysis:
    """Compute the centroid-offset vector and its perpendicular plane normal.

    The plane passes through *point_mm*. A normal is returned only when the
    centroid-offset magnitude exceeds a visualization stability threshold.
    This heuristic is not a statistical confidence interval.
    """
    p = np.asarray(point_mm, dtype=np.float64).reshape(3)
    q = np.asarray(neighbors_mm, dtype=np.float64).reshape(-1, 3)
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    q = q[np.isfinite(q).all(axis=1)]
    if not len(q):
        raise ValueError("No valid points in the selected neighborhood")
    centroid = q.mean(axis=0)
    vector = centroid - p
    length = float(np.linalg.norm(vector))
    threshold = max(0.5, 0.02 * float(radius_mm))
    normal = vector / length if length >= threshold else None
    return CentroidAnalysis(p, centroid, vector, normal, len(q), length, threshold)


def save_cloud(folder: str | Path, xyz_mm: np.ndarray, colors_rgb: np.ndarray,
               metadata: dict | None = None) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz_mm, dtype=np.float32)
    rgb = np.asarray(colors_rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("points_mm and colors_rgb must both have shape (N,3)")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Point cloud is empty or contains non-finite coordinates")
    target = folder / "cloud.npz"
    np.savez_compressed(target, points_mm=xyz, colors_rgb=rgb)
    if metadata is not None:
        atomic_json(folder / "metadata.json", metadata)
    if not (folder / "corners.json").exists():
        save_corners(folder, [])
    return target


def load_cloud(path: str | Path) -> tuple[Path, np.ndarray, np.ndarray]:
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / ("cloud.npz" if (path / "cloud.npz").is_file() else "points.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "xyz_mm" in archive and "rgb" in archive:
            xyz, rgb = archive["xyz_mm"].astype(np.float32), archive["rgb"].astype(np.uint8)
        elif "points_mm" in archive and "colors_rgb" in archive:
            xyz, rgb = archive["points_mm"].astype(np.float32), archive["colors_rgb"].astype(np.uint8)
        else:
            raise ValueError(f"Invalid {path.name}: expected saved XYZ and RGB arrays")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape != rgb.shape:
        raise ValueError(f"Invalid {path.name}: expected (N,3) positions and RGB")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid {path.name}: empty or non-finite point cloud")
    return path.parent, xyz, rgb


def resolve_dataset_scene(dataset_root: str | Path, scene_number: int) -> Path:
    """Select a numbered scene by its ID, not its position in a folder listing."""
    if isinstance(scene_number, bool) or not isinstance(scene_number, (int, np.integer)) or scene_number < 1:
        raise ValueError("Scene number must be a positive integer")
    root = Path(dataset_root).expanduser().resolve()
    if (root / "dataset").is_dir():
        root = root / "dataset"
    path = root / f"scene_{scene_number:04d}" / "points.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Scene {scene_number} not found: {path}")
    return path


def load_dataset_roi_indices(folder: str | Path, n_points: int) -> np.ndarray:
    """Select saved inclusive ROI rows without renumbering original raw IDs."""
    from corner_dataset_tool.dataset_io import SceneData
    folder = Path(folder)
    if not (folder / 'annotations.json').is_file():
        raise ValueError('Saved ROI requires annotations.json; disable Dataset ROI only')
    scene = SceneData.load(folder)
    if len(scene.xyz_mm) != n_points:
        raise ValueError('ROI must index the full original dataset cloud')
    indices = scene.roi_indices()
    if not len(indices):
        raise ValueError('Saved dataset ROI contains no points')
    return indices


def atomic_json(path: str | Path, content: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_corners(folder: str | Path, corners: list[dict]) -> None:
    atomic_json(Path(folder) / "corners.json", {"version": 1, "corners": corners})


def load_corners(folder: str | Path, n_points: int, *, dataset: bool = False) -> list[dict]:
    path = Path(folder) / "corners.json"
    if not path.exists():
        if dataset and (Path(folder) / "annotations.json").is_file():
            from corner_dataset_tool.dataset_io import SceneData
            scene = SceneData.load(folder)
            if len(scene.xyz_mm) != n_points:
                raise ValueError("Dataset GT must index the full original point cloud")
            return [{"id": f"gt_{corner['id']:03d}", "point_index": corner["raw_point_index"]}
                    for corner in scene.corners]
        return []
    with path.open("r", encoding="utf-8") as stream:
        obj = json.load(stream)
    records = obj.get("corners", [])
    if not isinstance(records, list):
        raise ValueError("Invalid corners.json")
    valid = []
    seen_ids = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        idx = item.get("point_index")
        key = item.get("id")
        if not (isinstance(idx, int) and 0 <= idx < n_points
                and isinstance(key, str) and key not in seen_ids):
            continue
        record = {"id": key, "point_index": idx}
        refined_index = item.get("refined_point_index")
        if isinstance(refined_index, int) and 0 <= refined_index < n_points:
            record["refined_point_index"] = refined_index
        xyz = item.get("refined_xyz_mm")
        if isinstance(xyz, list) and len(xyz) == 3:
            try:
                arr = np.asarray(xyz, dtype=np.float64)
                if np.isfinite(arr).all():
                    record["refined_xyz_mm"] = arr.tolist()
            except (TypeError, ValueError):
                pass
        if isinstance(item.get("refinement"), dict):
            # A locally converged label may be unchanged (no refined_point_index).
            record["refinement"] = item["refinement"]
        # New format: Q10 refinement follows voxel centroid coordinates, which
        # do not necessarily correspond to any original cloud point index.
        raw_xyz_path = item.get("refinement_path_xyz_mm")
        if isinstance(raw_xyz_path, list) and 1 <= len(raw_xyz_path) <= 10000:
            try:
                arr = np.asarray(raw_xyz_path, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 3 and np.isfinite(arr).all():
                    record["refinement_path_xyz_mm"] = arr.tolist()
            except (TypeError, ValueError):
                pass
        # All steps reference ORIGINAL saved-cloud indices, not discretized grid points.
        raw_path = item.get("refinement_path_indices")
        if isinstance(raw_path, list) and len(raw_path) <= 10000 and all(
            isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < n_points for value in raw_path
        ) and raw_path and raw_path[0] == idx:
            record["refinement_path_indices"] = raw_path
        valid.append(record)
        seen_ids.add(key)
    return valid


def current_refinement_point_index(entry: dict, n_points: int) -> int:
    """Use the most recent saved observed point, else the original label."""
    refined_index = entry.get("refined_point_index")
    if (isinstance(refined_index, int) and not isinstance(refined_index, bool)
            and 0 <= refined_index < n_points):
        return refined_index
    return int(entry["point_index"])


def refinement_path_indices(entry: dict, n_points: int) -> list[int]:
    """Legacy path of original saved-cloud indices (read-only compatibility)."""
    initial_index = int(entry["point_index"])
    result = [initial_index]
    raw = entry.get("refinement_path_indices", [])
    if isinstance(raw, list) and len(raw) <= 10000:
        for value in raw:
            if (isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value < n_points and value != result[-1]):
                result.append(value)
    current_index = current_refinement_point_index(entry, n_points)
    if result[-1] != current_index:
        result.append(current_index)
    return result


def refinement_path_xyz(entry: dict, raw_xyz_mm: np.ndarray) -> np.ndarray:
    """Reconstruct the display path; modern refinement uses voxel XYZ coordinates."""
    p0 = np.asarray(raw_xyz_mm[int(entry["point_index"])], dtype=np.float64)
    path = entry.get("refinement_path_xyz_mm")
    if path:
        positions = np.asarray(path, dtype=np.float64).reshape(-1, 3)
        if not np.allclose(positions[0], p0, atol=1e-5):
            positions = np.vstack([p0, positions])
    else:
        indices = refinement_path_indices(entry, len(raw_xyz_mm))
        positions = np.asarray(raw_xyz_mm[indices], dtype=np.float64)
    refined = entry.get("refined_xyz_mm")
    if refined is not None:
        p = np.asarray(refined, dtype=np.float64)
        if not np.allclose(positions[-1], p, atol=1e-5):
            positions = np.vstack([positions, p])
    return positions


def export_binary_ply(path: str | Path, points_mm: np.ndarray, colors_rgb: np.ndarray) -> None:
    """Export RGB point cloud with millimetre positions to standard binary PLY."""
    xyz = np.asarray(points_mm, dtype=np.float32)
    rgb = np.asarray(colors_rgb, dtype=np.uint8)
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Point and color arrays must both be N x 3")
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex = np.empty(len(xyz), dtype=vertex_dtype)
    for i, name in enumerate(("x", "y", "z")):
        vertex[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertex[name] = rgb[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Position units: millimeters\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with Path(path).open("wb") as output:
        output.write(header.encode("ascii"))
        vertex.tofile(output)


try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # --demo and --open work without RealSense SDK.

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 2.0
MAX_VIEW_POINTS = 160_000
MAX_HIGHLIGHT_POINTS = 15_000


def point_cloud(points_mm: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_mm, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb, dtype=np.float64) / 255.0)
    return cloud


def voxel_downsample_analysis(points_mm: np.ndarray, colors_rgb: np.ndarray,
                              voxel_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the ENTIRE saved cloud once, before any neighborhood statistics.

    Each returned point is a voxel centroid, not a voxel occupancy vote.
    The same global grid is reused for Q10, centroid, and spherical-plane analyses.
    """
    if voxel_mm <= 0:
        raise ValueError("Analysis voxel size must be positive")
    down = point_cloud(points_mm, colors_rgb).voxel_down_sample(float(voxel_mm))
    xyz = np.asarray(down.points, dtype=np.float64).copy()
    colors = np.clip(np.rint(np.asarray(down.colors) * 255.0), 0, 255).astype(np.uint8)
    if len(xyz) == 0:
        raise ValueError("Voxel downsampling produced an empty cloud")
    return xyz, colors


def sphere(center, radius_mm: float, color):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_mm), resolution=10)
    mesh.translate(np.asarray(center, dtype=float))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def plane_disk(center, normal, radius_mm: float, segments: int = 56):
    """Disk through center, with normal = normalized centroid-offset vector."""
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    reference = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    center = np.asarray(center, dtype=np.float64)
    rim = center + float(radius_mm) * (
        np.cos(angle)[:, None] * u[None, :] + np.sin(angle)[:, None] * v[None, :]
    )
    disk = o3d.geometry.TriangleMesh()
    disk.vertices = o3d.utility.Vector3dVector(np.vstack((center, rim)))
    disk.triangles = o3d.utility.Vector3iVector(
        np.array([(0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)], dtype=np.int32)
    )
    disk.compute_vertex_normals()
    ring = o3d.geometry.LineSet()
    ring.points = o3d.utility.Vector3dVector(rim)
    ring.lines = o3d.utility.Vector2iVector(
        np.array([(i, (i + 1) % segments) for i in range(segments)], dtype=np.int32)
    )
    ring.paint_uniform_color([0.10, 0.92, 0.95])
    return disk, ring


def projected_cloud(depth_raw, color_rgb, intrinsics, depth_scale_m, max_view=MAX_VIEW_POINTS):
    """Vectorized RGB-aligned depth -> XYZ, keeping mm coordinates."""
    h, w = depth_raw.shape
    if color_rgb.shape[:2] != (h, w):
        raise ValueError("Depth and RGB frames must have the same dimensions (use rs.align)")
    fx, fy, ppx, ppy = (float(intrinsics[k]) for k in ("fx", "fy", "ppx", "ppy"))
    if fx <= 0 or fy <= 0 or depth_scale_m <= 0:
        raise ValueError("Invalid camera intrinsics or depth scale")
    z_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    good = (z_m >= MIN_DEPTH_M) & (z_m <= MAX_DEPTH_M) & np.isfinite(z_m)
    rows, cols = np.nonzero(good)
    if not len(rows):
        raise ValueError("No depth points found between 0.15m and 2m")
    if len(rows) > max_view:
        # Deterministic pixel subsampling. Saved cloud corresponds exactly to labels.
        take = np.linspace(0, len(rows) - 1, max_view, dtype=np.int64)
        rows, cols = rows[take], cols[take]
    z = z_m[rows, cols]
    pts = np.column_stack(((cols - ppx) / fx * z, (rows - ppy) / fy * z, z))
    return (pts * 1000.0).astype(np.float32), color_rgb[rows, cols].astype(np.uint8)




# ------------ Directional mean and Q10(90% containment) analysis ------------

MAX_UNIT_SPHERE_POINTS = 10_000  # Display budget; statistics use ALL directions.
COS_HIST_BINS = 40


@dataclass(frozen=True)
class ConeAnalysis:
    directions: np.ndarray       # N x 3 unit vectors from corner -> neighbors
    mean_axis: np.ndarray        # normalized sum of ALL unit directions
    cosines: np.ndarray          # each u_i dot mean_axis, range [-1, +1]
    cosine_q10: float           # lower 10th percentile of cosines
    half_angle_deg: float       # arccos(Q10), ~90% of directions inside
    inlier_mask: np.ndarray
    resultant_length: float      # ||mean(u)||: direction concentration, range [0,1]
    zero_distance_count: int


def _unit(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / n if n > eps else None


def analyze_cone(point_mm: np.ndarray, neighbors_mm: np.ndarray) -> ConeAnalysis:
    """Compute direction mean and 90%-containing cone from ALL nearby samples.

    u_i = (q_i - p) / ||q_i - p||; mean_axis = sum(u_i) / ||sum(u_i)||.
    Q10 = 10th percentile of (u_i dot mean_axis), and alpha90=acos(Q10).
    No median optimization, radial-distance weighting or subsampling in statistics.
    A near-zero resultant makes the mean direction ill-defined, so fail clearly.
    """
    point = np.asarray(point_mm, dtype=np.float64).reshape(3)
    neighbors = np.asarray(neighbors_mm, dtype=np.float64).reshape(-1, 3)
    d = neighbors - point[None, :]
    distances = np.linalg.norm(d, axis=1)
    valid = np.isfinite(distances) & (distances > 1e-7)
    u = d[valid] / distances[valid, None]
    if len(u) < 3:
        raise ValueError(f'Need at least 3 valid nonzero directions, got {len(u)}')
    mean_vec = u.mean(axis=0)
    resultant = float(np.linalg.norm(mean_vec))
    if resultant < 1e-8:
        raise ValueError('Mean direction undefined: unit direction vectors cancel (R~0)')
    mean_axis = mean_vec / resultant
    cosines = np.clip(u @ mean_axis, -1.0, 1.0)
    q10 = float(np.quantile(cosines, 0.10))
    half_angle = float(np.rad2deg(np.arccos(np.clip(q10, -1.0, 1.0))))
    mask = cosines >= (q10 - 1e-12)
    return ConeAnalysis(u, mean_axis, cosines, q10, half_angle, mask,
                        resultant, int((~valid).sum()))


# ------ Origin-constrained spherical multi-plane fitting (pure NumPy) ------

# These are planes through the UNIT SPHERE ORIGIN: n.dot(u) = 0.
# They are NOT arbitrary offset planes fitted in the original 3D cloud.
SPHERE_PLANE_COLORS = ((255, 130, 62), (81, 218, 129), (159, 130, 255))
SPHERE_PLANE_HIST_BINS = 36  # 10 degrees per bin, 0..360.


@dataclass(frozen=True)
class SphericalPlane:
    normal: np.ndarray                # unit normal, plane offset d == 0 by construction
    inlier_mask: np.ndarray           # N bool; may overlap with OTHER planes
    projected_directions: np.ndarray  # M x 3 on great circle; one per inlier
    angles_deg: np.ndarray           # M in [0, 360), within this plane
    residual_rms_deg: float
    newly_explained_count: int


@dataclass(frozen=True)
class SphericalPlaneAnalysis:
    directions: np.ndarray       # N x 3, derived from same downsampled cloud
    planes: tuple[SphericalPlane, ...]
    membership_count: np.ndarray  # N, may be 0, 1, 2 or 3
    requested_k: int
    tolerance_deg: float


def _sphere_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal basis of a zero-offset plane."""
    n = np.asarray(normal, dtype=np.float64)
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = ref - np.dot(ref, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _spherical_plane_residual_deg(u: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Angle to GREAT CIRCLE, rather than a Euclidean offset in millimetres."""
    return np.rad2deg(np.arcsin(np.clip(np.abs(u @ normal), 0.0, 1.0)))


def _refit_origin_plane(u: np.ndarray, seed: np.ndarray,
                        tolerance_deg: float, max_iters: int = 8) -> np.ndarray:
    """Robust Huber IRLS total least squares, with NO mean subtraction."""
    n = np.asarray(seed, dtype=np.float64).copy()
    huber_deg = max(0.5, 0.50 * float(tolerance_deg))
    for _ in range(max_iters):
        r = _spherical_plane_residual_deg(u, n)
        support = r <= tolerance_deg
        if support.sum() < 3:
            break
        r_s = r[support]
        weights = np.minimum(1.0, huber_deg / np.maximum(r_s, 1e-9))
        v = u[support]
        # Homogeneous constraint: smallest eigenvector of sum w*u*u.T.
        scatter = (v * weights[:, None]).T @ v
        _, eigenvectors = np.linalg.eigh(scatter)
        n_new = eigenvectors[:, 0]
        if np.dot(n_new, n) < 0:
            n_new = -n_new
        change_deg = np.rad2deg(np.arccos(np.clip(abs(n_new @ n), 0, 1)))
        n = n_new
        if change_deg < 1e-5:
            break
    return n


def segment_origin_spherical_planes(directions: np.ndarray, k: int = 3,
                                    tolerance_deg: float = 15.0,
                                    iterations: int = 300,
                                    min_support: int = 12,
                                    min_fraction: float = 0.06,
                                    seed: int = 20261110) -> SphericalPlaneAnalysis:
    """Sequential robust Great-Circle RANSAC with multiple memberships.

    Stage A: each RANSAC round prioritizes currently UNEXPLAINED points.
    Stage B: refit each accepted plane with ALL matching directions.
    Stage C: classify ALL directions independently against EVERY plane.
    Points near plane intersections belong to multiple planes, not just one.
    """
    u = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(u).all():
        raise ValueError('Unit directions contain non-finite values')
    norms = np.linalg.norm(u, axis=1)
    if len(u) < 3 or np.any(norms < 1e-10):
        raise ValueError('Need at least three valid nonzero directions')
    u = u / norms[:, None]
    if k not in (1, 2, 3):
        raise ValueError('k must be 1, 2, or 3')
    if not (0 < tolerance_deg < 45) or iterations < 10:
        raise ValueError('Invalid angular tolerance or RANSAC iterations')

    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    explained = np.zeros(len(u), dtype=bool)
    minimum = max(int(min_support), int(np.ceil(min_fraction * len(u))))
    max_abs_cos = float(np.sin(np.deg2rad(tolerance_deg)))
    # Reject almost duplicate planes, but do NOT reject point-level overlap.
    duplicate_cos = float(np.cos(np.deg2rad(max(12.0, 1.5 * tolerance_deg))))

    for plane_index in range(k):
        available = np.flatnonzero(~explained)
        if len(available) < minimum:
            break
        # Sample representative residual directions to bound per-frame work.
        if len(available) > 4000:
            sample_ids = rng.choice(available, size=4000, replace=False)
        else:
            sample_ids = available
        sample_u = u[sample_ids]
        best_n, best_count, best_cost = None, -1, np.inf

        for _ in range(iterations):
            i, j = rng.choice(len(sample_u), size=2, replace=False)
            n = np.cross(sample_u[i], sample_u[j])
            mag = np.linalg.norm(n)
            if mag < 0.16:  # Avoid near-collinear minimal samples.
                continue
            n /= mag
            if any(abs(n @ old) > duplicate_cos for old in accepted):
                continue
            distances = np.abs(sample_u @ n)
            inliers = distances <= max_abs_cos
            cnt = int(inliers.sum())
            # Simple robust tie-break for similarly supported models.
            cost = float(np.sum(np.minimum(distances, max_abs_cos)**2))
            if cnt > best_count or (cnt == best_count and cost < best_cost):
                best_n, best_count, best_cost = n.copy(), cnt, cost

        if best_n is None:
            break
        # Refitting must use the complete neighborhood, not just RANSAC's sample.
        refined = _refit_origin_plane(u, best_n, tolerance_deg)
        if any(abs(refined @ old) > duplicate_cos for old in accepted):
            break
        residual = _spherical_plane_residual_deg(u, refined)
        inliers = residual <= tolerance_deg
        newly_explained = inliers & ~explained
        if newly_explained.sum() < minimum:
            break
        accepted.append(refined)
        explained |= inliers

    # Final full-set membership: independent threshold for each plane.
    # Even an intersection point already claimed by one plane is retested
    # against all other planes and can be included in every applicable plane.
    planes: list[SphericalPlane] = []
    membership = np.zeros(len(u), dtype=np.uint8)
    covered_final = np.zeros(len(u), dtype=bool)
    for n in accepted:
        # Refit with all current inliers; still constrained to d=0.
        n = _refit_origin_plane(u, n, tolerance_deg)
        # Fix normal sign for repeatable local azimuth histograms.
        if n[int(np.argmax(np.abs(n)))] < 0:
            n = -n
        r = _spherical_plane_residual_deg(u, n)
        mask = r <= tolerance_deg
        membership += mask.astype(np.uint8)
        newly_explained_count = int(np.count_nonzero(mask & ~covered_final))
        covered_final |= mask
        selected = u[mask]
        projected = selected - (selected @ n)[:, None] * n[None, :]
        magnitudes = np.linalg.norm(projected, axis=1)
        usable = magnitudes > 1e-10  # Always true for r < 45 degrees.
        projected = projected[usable] / magnitudes[usable, None]
        e1, e2 = _sphere_plane_basis(n)
        angle = np.mod(np.rad2deg(np.arctan2(projected @ e2, projected @ e1)), 360.)
        angle[angle >= 360.] = 0.  # Guard floating-point rounding at the seam.
        rms = float(np.sqrt(np.mean(r[mask]**2))) if mask.any() else float('nan')
        planes.append(SphericalPlane(n, mask, projected, angle, rms, newly_explained_count))
    return SphericalPlaneAnalysis(u, tuple(planes), membership, k, float(tolerance_deg))


# ------ PCL-style max angular gap + isolated angular outlier filtering ------

@dataclass(frozen=True)
class PlaneGapTest:
    plane_index: int
    raw_count: int
    kept_count: int
    kept_mask: np.ndarray  # original plane.angles_deg order; False = rejected angular noise
    discarded_count: int
    component_count: int
    max_gap_deg: float
    observed_span_deg: float  # 360 - max_gap after removing angular outliers
    arc_start_deg: float | None
    arc_end_deg: float | None
    pcl_boundary_like: bool  # classical >90 degree angular gap, NOT corner classification
    edge_like: bool          # experimental edge-vs-corner decision via span
    reliable: bool
    raw_span_deg: float = 0.0
    tail_removed_count: int = 0
    tail_trimmed_bins: int = 0
    density_threshold_count: int = 0
    density_reference_count: float = 0.0


@dataclass(frozen=True)
class PlaneGapValidation:
    status: str              # CORNER / REJECT / UNKNOWN
    tests: tuple[PlaneGapTest, ...]
    max_link_gap_deg: float
    edge_span_deg: float


def _dominant_angular_component(angles_deg: np.ndarray,
                                max_link_gap_deg: float = 20.0) -> tuple[np.ndarray, int]:
    """Keep largest connected component on a circular angular coordinate.

    Sorted neighbors are connected if their *circular angular separation* is
    <= max_link_gap_deg. Disconnected small clusters (e.g. a few points opposite
    the actual surface arc) cannot change the measured main-arc span.
    This is angular connectivity, NOT Euclidean point-cloud connectivity.
    """
    a = np.mod(np.asarray(angles_deg, dtype=np.float64).reshape(-1), 360.)
    if not 0. < max_link_gap_deg < 180.:
        raise ValueError('Angular connection gap must be between 0 and 180 degrees')
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite angular direction')
    order = np.argsort(a, kind='stable')
    theta = a[order]
    # cyclic gap after each sorted angle, including the 360/0 seam
    gaps = np.diff(np.r_[theta, theta[0] + 360.])
    cuts = np.flatnonzero(gaps > max_link_gap_deg + 1e-10)
    if len(cuts) == 0:
        return np.ones(n, dtype=bool), 1
    # Linearize just after one cut, then split at all other cuts.
    linear = (np.arange(n, dtype=int) + int(cuts[0]) + 1) % n
    breaks = np.flatnonzero(gaps[linear[:-1]] > max_link_gap_deg) + 1
    groups = np.split(linear, breaks)
    # Prefer larger group; equal size chooses compact span, then smallest angle.
    def rank(group):
        spread = float(np.mod(theta[group[-1]] - theta[group[0]], 360.))
        return (-len(group), spread, float(theta[group[0]]))
    winner = min(groups, key=rank)
    keep = np.zeros(n, dtype=bool)
    keep[order[winner]] = True
    return keep, len(groups)


def trim_sparse_arc_ends(angles_deg: np.ndarray, dominant_mask: np.ndarray,
                         bin_width_deg: float = 10.0,
                         density_fraction: float = 0.50,
                         minimum_bin_count: int = 2,
                         enabled: bool = True,
                         ) -> tuple[np.ndarray, float, int, int, int, float]:
    """Lightly trim low-count angular bins ONLY from both ends of dominant arc.

    Returns (kept_mask, raw_span_deg, trimmed_bins, removed_points,
             count_threshold, reference_median_count). The circular 0/360
    seam is handled by unwrapping relative to the largest angular gap.
    Internal sparse bins are deliberately retained; only tails are trimmed.
    """
    angles = np.mod(np.asarray(angles_deg, dtype=np.float64).reshape(-1), 360.)
    dominant = np.asarray(dominant_mask, dtype=bool).reshape(-1)
    if angles.size != dominant.size:
        raise ValueError('Angular points and component mask must match')
    if not (0 < bin_width_deg <= 90 and 0 <= density_fraction <= 1
            and minimum_bin_count >= 1):
        raise ValueError('Invalid angular tail filter parameters')
    if not dominant.any():
        return dominant.copy(), 0.0, 0, 0, minimum_bin_count, 0.0
    ids = np.flatnonzero(dominant)
    ordered = np.sort(angles[ids])
    circular_gaps = np.diff(np.r_[ordered, ordered[0] + 360.])
    gap_idx = int(np.argmax(circular_gaps))
    start = float(ordered[(gap_idx + 1) % len(ordered)])
    raw_span = float(np.clip(360. - circular_gaps[gap_idx], 0., 360.))
    # Angle relative to the observed arc's starting endpoint.
    offsets = np.mod(angles[ids] - start, 360.)
    # Avoid placing the very last sample into a phantom bin if raw_span is
    # exactly divisible by the requested bin width.
    unwrapped = start + np.minimum(offsets, max(0., raw_span - 1e-8))
    # Anchor bins to a fixed global angular grid, not the first sparse point.
    # Otherwise that first point can accidentally share a bin with dense data.
    base_bin = int(np.floor(start / float(bin_width_deg)))
    bins = (np.floor(unwrapped / float(bin_width_deg)).astype(np.int64)
            - base_bin)
    counts = np.bincount(bins, minlength=int(bins.max()) + 1)
    occupied = counts[counts > 0]
    ref = float(np.median(occupied)) if len(occupied) else 0.0
    threshold = max(int(minimum_bin_count),
                    int(np.ceil(float(density_fraction) * ref)))
    if not enabled or len(counts) <= 1:
        return dominant.copy(), raw_span, 0, 0, threshold, ref
    # Cut only the two tails. Never discard a low-density interior bin.
    left, right = 0, len(counts) - 1
    while left <= right and counts[left] < threshold:
        left += 1
    while right >= left and counts[right] < threshold:
        right -= 1
    cleaned = dominant.copy()
    if left > right:  # No bin has reliable support: status becomes UNKNOWN.
        cleaned[ids] = False
    else:
        cleaned[ids] = (bins >= left) & (bins <= right)
    removed = int(np.count_nonzero(dominant & ~cleaned))
    trimmed_bins = int(len(counts) if left > right else left + (len(counts) - 1 - right))
    return cleaned, raw_span, trimmed_bins, removed, threshold, ref


def evaluate_plane_max_gap(analysis: SphericalPlaneAnalysis,
                           max_link_gap_deg: float = 20.0,
                           edge_span_deg: float = 160.0,
                           min_points: int = 12,
                           min_retained_fraction: float = 0.6,
                           tail_filter_enabled: bool = True,
                           tail_bin_width_deg: float = 10.0,
                           tail_density_fraction: float = 0.50,
                           tail_min_bin_count: int = 2) -> PlaneGapValidation:
    """Measure largest circular gap and its complementary span per spherical plane.

    Find the largest linked angular component first. Then, optionally remove
    only its very sparse END bins, using the median of occupied bins as the
    reference. Interior bins are never dropped by this gentle tail filter.
    A plane is UNKNOWN if too few trustworthy points remain, including when
    the dominant group accounts for < min_retained_fraction of its inliers.
    The PCL-style gap flag uses the dominant component BEFORE tail trimming,
    preserving previous diagnostic behavior. Rejecting EDGE-LIKE uses the
    cleaned span >= edge_span_deg (OUR experimental criterion).
    """
    if not (0. < max_link_gap_deg < 180. and 0. < edge_span_deg <= 360.
            and min_points >= 2 and 0. < min_retained_fraction <= 1.):
        raise ValueError('Invalid maximum angular gap parameters')
    tests = []
    for i, plane in enumerate(analysis.planes):
        angles = np.mod(np.asarray(plane.angles_deg, dtype=np.float64), 360.)
        component_keep, components = _dominant_angular_component(angles, max_link_gap_deg)
        keep, raw_span, tail_bins, tail_points, density_min, density_ref = \
            trim_sparse_arc_ends(angles, component_keep,
                                 bin_width_deg=tail_bin_width_deg,
                                 density_fraction=tail_density_fraction,
                                 minimum_bin_count=tail_min_bin_count,
                                 enabled=tail_filter_enabled)
        kept = int(keep.sum())
        reliable = kept >= min_points and kept / max(1, len(angles)) >= min_retained_fraction
        max_gap = 360.0
        start = end = None
        span = 0.0
        if kept:
            order = np.sort(angles[keep])
            gaps = np.diff(np.r_[order, order[0] + 360.])
            gap_id = int(np.argmax(gaps))
            max_gap = float(gaps[gap_id])
            span = float(np.clip(360.0 - max_gap, 0.0, 360.0))
            start = float(order[(gap_id + 1) % kept])
            end = float(order[gap_id])
        tests.append(PlaneGapTest(
            i, len(angles), kept, keep, len(angles) - kept, components,
            max_gap, span, start, end, bool(360. - raw_span > 90.),
            bool(reliable and span >= edge_span_deg), bool(reliable),
            float(raw_span), int(tail_points), int(tail_bins),
            int(density_min), float(density_ref)))
    if not tests or any(not t.reliable for t in tests):
        status = 'REJECT' if any(t.edge_like for t in tests) else 'UNKNOWN'
    else:
        status = 'REJECT' if any(t.edge_like for t in tests) else 'CORNER'
    return PlaneGapValidation(status, tuple(tests), max_link_gap_deg, edge_span_deg)


def spherical_plane_histogram_image(analysis: SphericalPlaneAnalysis | None,
                                    width: int = 620, height: int = 330,
                                    bins: int = SPHERE_PLANE_HIST_BINS,
                                    gap_validation: PlaneGapValidation | None = None) -> np.ndarray:
    """Rose histogram with dominant angular component, dropped outliers, gap/span.

    Cyan arc: angular support span of dominant component after filtering.
    Magenta rim dots: angles discarded as disconnected outliers.
    Raw histogram remains visible for visual debugging.
    """
    canvas = np.full((height, width, 3), [17, 25, 38], dtype=np.uint8)
    if analysis is None or not analysis.planes:
        _tiny_text(canvas, 12, 12, 'NO PLANES')
        return canvas
    row_h = max(1, height // len(analysis.planes))
    for j, plane in enumerate(analysis.planes):
        top = j * row_h + 5
        bottom = height - 5 if j == len(analysis.planes)-1 else (j+1)*row_h-5
        usable_h = bottom - top
        if usable_h < 34:
            continue
        color = np.asarray(SPHERE_PLANE_COLORS[j], dtype=np.uint8)
        canvas[top:bottom, 4:width-4] = [22, 32, 46]
        if j:
            canvas[top-4:top-3, 4:width-4] = [52,65,81]
        cx, cy = int(width * .43), (top+bottom)//2
        radius = max(13, min(usable_h//2 - 12, 98))
        counts = np.histogram(plane.angles_deg, bins=bins, range=(0.,360.))[0]
        maximum = max(1, int(counts.max()))
        x0, x1 = max(0,cx-radius-3), min(width,cx+radius+4)
        y0, y1 = max(0,cy-radius-3), min(height,cy+radius+4)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx.astype(float)-cx, cy-yy.astype(float)
        rr = np.hypot(dx,dy)
        angle = np.mod(np.rad2deg(np.arctan2(dy,dx)),360.)
        bin_idx = np.clip((angle / 360. * bins).astype(int),0,bins-1)
        patch = canvas[y0:y1,x0:x1].copy()
        patch[rr <= radius * counts[bin_idx] / maximum] = color
        for f in (1/3,2/3,1.):
            patch[np.abs(rr-f*radius)<=.65]=[75,87,103]
        patch[((np.abs(dx)<=.5)|(np.abs(dy)<=.5))&(rr<=radius)]=[122,138,157]
        patch[np.abs(rr-radius)<=1.]=[184,196,210]
        patch[rr<=1.6]=[224,230,238]
        test = next((t for t in gap_validation.tests if t.plane_index==j),None) if gap_validation else None
        if test is not None and test.kept_count:
            # Cyan highlights smallest circular arc containing dominant cluster.
            start = test.arc_start_deg
            delta = np.mod(angle-start,360.)
            visible = (delta<=test.observed_span_deg+0.2) | (test.observed_span_deg>=359.99)
            ring = (rr>=radius*.77)&(rr<=radius*.87)
            patch[ring&visible]=[55,232,229]
            # Magenta ticks mark the rejected raw angles on outer rim.
            rejected = np.asarray(plane.angles_deg)[~test.kept_mask]
            if len(rejected):
                rim = (rr>=radius*.92)&(rr<=radius*1.02)
                close = np.zeros(angle.shape,dtype=bool)
                # One 2-D mask per rejected point avoids an enormous H x W x N array.
                for rejected_angle in rejected:
                    close |= (np.abs(np.mod(angle-rejected_angle+180.,360.)-180.)<=1.2)
                patch[rim&close]=[255,77,177]
        canvas[y0:y1,x0:x1] = patch
        _tiny_text(canvas,9,top+9,f'P{j+1}',color=color.tolist())
        _tiny_text(canvas,9,top+24,f'N={len(plane.angles_deg)}',color=[188,199,214])
        _tiny_text(canvas,max(0,cx-radius-22),cy-4,'180')
        _tiny_text(canvas,min(width-10,cx+radius+4),cy-4,'0')
        _tiny_text(canvas,max(0,cx-9),max(0,cy-radius-12),'90')
        _tiny_text(canvas,max(0,cx-13),min(height-12,cy+radius+3),'270')
        tx = cx+radius+44
        if test is None:
            _tiny_text(canvas,tx,top+10,'NO GAP TEST')
        else:
            state_color = [245,212,68] if test.edge_like else [95,169,255]
            _tiny_text(canvas,tx,top+8,f'SPAN={test.observed_span_deg:.0f}DEG',color=[55,232,229])
            _tiny_text(canvas,tx,top+23,f'GAP={test.max_gap_deg:.0f}DEG',color=[245,226,190])
            _tiny_text(canvas,tx,top+38,f'KEEP={test.kept_count}/{test.raw_count}',color=[200,223,237])
            _tiny_text(canvas,tx,top+53,f'TAIL={test.tail_removed_count}',color=[255,77,177])
            if usable_h>=85:
                _tiny_text(canvas,tx,top+69,
                           'UNKNOWN' if not test.reliable else ('EDGE' if test.edge_like else 'CORNER'),
                           color=state_color)
    return canvas


# -------- Q10 discrete refinement: globally downsampled point centroids --------

@dataclass(frozen=True)
class RefinementResult:
    initial_mm: np.ndarray
    refined_mm: np.ndarray
    refined_point_index: int
    q10_before: float
    q10_after: float
    count_before: int
    count_after: int
    tested_positions: int
    available_positions: int
    displacement_mm: float
    search_radius_mm: float
    candidate_spacing_mm: float
    neighborhood_radius_mm: float
    boundary_hit: bool


def observed_candidates(initial_mm: np.ndarray,
                        xyz_mm: np.ndarray,
                        point_indices: np.ndarray,
                        search_radius_mm: float,
                        spacing_mm: float = 0.0) -> np.ndarray:
    """Return input point indices inside search ball, with optional extra thinning.

    Called with GLOBAL DOWNSAMPLED cloud and its indices. spacing_mm=0 evaluates
    every downsampled point; nonzero spacing is optional further thinning.
    """
    center = np.asarray(initial_mm, dtype=np.float64).reshape(3)
    xyz = np.asarray(xyz_mm, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if xyz.shape[0] != ids.shape[0] or not len(ids):
        raise ValueError('Candidate coordinates and original indexes must match and be nonempty')
    if search_radius_mm <= 0 or spacing_mm < 0:
        raise ValueError('Search radius must be positive, spacing must be nonnegative')
    if not np.isfinite(center).all() or not np.isfinite(xyz).all():
        raise ValueError('Candidate coordinates must be finite')
    dist = np.linalg.norm(xyz - center[None, :], axis=1)
    inside = dist <= float(search_radius_mm) + 1e-8
    xyz, ids, dist = xyz[inside], ids[inside], dist[inside]
    if not len(ids):
        raise ValueError('No downsampled point within refinement search range')
    # Nearest-to-original ordering also makes thinning deterministic.
    order = np.lexsort((ids, dist))
    xyz, ids = xyz[order], ids[order]
    if spacing_mm > 0:
        voxels = np.floor((xyz - center[None, :]) / float(spacing_mm)).astype(np.int64)
        _, keep = np.unique(voxels, axis=0, return_index=True)
        keep.sort()
        ids = ids[keep]
    return ids


def discrete_refine_q10(initial_mm: np.ndarray,
                        initial_point_index: int,
                        support_xyz_mm: np.ndarray,
                        support_point_indices: np.ndarray,
                        radius_mm: float,
                        search_radius_mm: float = 6.0,
                        candidate_spacing_mm: float = 0.0,
                        min_support_ratio: float = 0.80,
                        max_candidates: int = 3500) -> RefinementResult:
    """Optimize Q10 over globally voxel-downsampled point centroids only.

    support_xyz_mm contains downsampled points within R+search of initial_mm;
    support_point_indices maps rows back to the FULL DOWNSAMPLED cloud.
    For each candidate, recalculate its exact radius-R neighbors, unit-direction
    mean axis, and Q10 using every neighbor (no support subsampling).

    If candidate_spacing_mm=0, every downsampled candidate is evaluated.
    If candidate count exceeds max_candidates, fail rather than silently skip.
    """
    initial = np.asarray(initial_mm, dtype=np.float64).reshape(3)
    points = np.asarray(support_xyz_mm, dtype=np.float64).reshape(-1, 3)
    point_indices = np.asarray(support_point_indices, dtype=np.int64).reshape(-1)
    if len(points) != len(point_indices) or len(points) == 0:
        raise ValueError('Support point/index lengths must match and be nonempty')
    if not np.isfinite(initial).all() or not np.isfinite(points).all():
        raise ValueError('Non-finite coordinates in refinement inputs')
    if radius_mm <= 0 or search_radius_mm <= 0:
        raise ValueError('Radii must be positive')
    if not 0 < min_support_ratio <= 1:
        raise ValueError('min_support_ratio must be in (0, 1]')
    if not np.any(point_indices == int(initial_point_index)):
        raise ValueError('Original labeled point is missing from support set')
    available_ids = observed_candidates(initial, points, point_indices, search_radius_mm, 0)
    candidate_ids = observed_candidates(initial, points, point_indices,
                                        search_radius_mm, candidate_spacing_mm)
    # Map each original index to its row in the local support set.
    row_from_id = {int(idx): row for row, idx in enumerate(point_indices)}
    if int(initial_point_index) not in candidate_ids:
        candidate_ids = np.insert(candidate_ids, 0, int(initial_point_index))
    candidate_ids = np.unique(candidate_ids)
    if len(candidate_ids) > max_candidates:
        raise ValueError(f'{len(candidate_ids)} observed candidates (limit {max_candidates}). '
                         'Reduce Search range or increase Candidate spacing.')

    # A local spatial tree avoids the full-cloud scan for each score evaluation.
    local_cloud = o3d.geometry.PointCloud()
    local_cloud.points = o3d.utility.Vector3dVector(points)
    local_tree = o3d.geometry.KDTreeFlann(local_cloud)

    def score(p):
        count, local_ids, _ = local_tree.search_radius_vector_3d(
            np.asarray(p, dtype=np.float64), float(radius_mm) + 1e-8)
        if count < 10:
            return None, count
        try:
            return analyze_cone(p, points[np.asarray(local_ids, dtype=np.int64)]).cosine_q10, count
        except ValueError:
            return None, count

    initial_q10, initial_count = score(initial)
    if initial_q10 is None:
        raise ValueError(f'Initial Q10 invalid (neighbors={initial_count}; need >=10 and a defined mean axis)')
    minimum_count = max(10, int(np.ceil(initial_count * min_support_ratio)))
    best_q10, best_count = initial_q10, initial_count
    best_index, best_point = int(initial_point_index), initial.copy()
    tested = 1  # Current downsampled position baseline.

    for idx in candidate_ids:
        idx = int(idx)
        if idx == int(initial_point_index):
            continue
        candidate = points[row_from_id[idx]]
        q10, count = score(candidate)
        tested += 1
        if q10 is None or count < minimum_count:
            continue
        distance = float(np.linalg.norm(candidate - initial))
        best_distance = float(np.linalg.norm(best_point - initial))
        if q10 > best_q10 + 1e-9 or (
            abs(q10 - best_q10) <= 1e-9 and
            (distance < best_distance - 1e-9 or
             (abs(distance-best_distance) <= 1e-9 and idx < best_index))
        ):
            best_q10, best_count = q10, count
            best_index, best_point = idx, candidate.copy()
    displacement = float(np.linalg.norm(best_point - initial))
    return RefinementResult(
        initial_mm=initial, refined_mm=best_point,
        refined_point_index=best_index,
        q10_before=float(initial_q10), q10_after=float(best_q10),
        count_before=int(initial_count), count_after=int(best_count),
        tested_positions=tested, available_positions=int(len(available_ids)),
        displacement_mm=displacement, search_radius_mm=float(search_radius_mm),
        candidate_spacing_mm=float(candidate_spacing_mm),
        neighborhood_radius_mm=float(radius_mm),
        boundary_hit=(displacement > 0 and displacement >= float(search_radius_mm) * .90),
    )


def _tiny_text(canvas, x, y, content, color=(185, 197, 214)):
    """5x7 bitmap numerals for plot ticks, avoiding drawing dependencies."""
    glyphs = {
        '0': ('111','101','101','101','111'),
        '1': ('010','110','010','010','111'),
        '2': ('111','001','111','100','111'),
        '3': ('111','001','111','001','111'),
        '4': ('101','101','111','001','001'),
        '5': ('111','100','111','001','111'),
        '6': ('111','100','111','101','111'),
        '7': ('111','001','001','001','001'),
        '8': ('111','101','111','101','111'),
        '9': ('111','101','111','001','111'),
        '-': ('000','000','111','000','000'),
        '+': ('000','010','111','010','000'),
        '.': ('000','000','000','000','010'),
        '%': ('101','001','010','100','101'),
        '=': ('000','111','000','111','000'),
        'A': ('010','101','111','101','101'),
        'B': ('110','101','110','101','110'),
        'C': ('011','100','100','100','011'),
        'D': ('110','101','101','101','110'),
        'E': ('111','100','110','100','111'),
        'G': ('011','100','101','101','011'),
        'I': ('111','010','010','010','111'),
        'M': ('101','111','111','101','101'),
        'N': ('101','111','111','111','101'),
        'O': ('010','101','101','101','010'),
        'P': ('110','101','110','100','100'),
        'R': ('110','101','110','101','101'),
        'S': ('011','100','010','001','110'),
        'T': ('111','010','010','010','010'),
        'X': ('101','101','010','101','101'),
        '/': ('001','001','010','100','100'),
        ' ': ('000','000','000','000','000'),
    }
    h,w = canvas.shape[:2]
    for char in str(content):
        rows = glyphs.get(char)
        if rows is not None:
            for iy, row in enumerate(rows):
                for ix, val in enumerate(row):
                    if val == '1' and 0 <= y + 2*iy < h and 0 <= x + 2*ix < w:
                        canvas[y + 2*iy:y + 2*iy + 2, x + 2*ix:x + 2*ix + 2] = color
        x += 9


def cosine_histogram_image(cone: ConeAnalysis | None, width: int = 560,
                           height: int = 280, bins: int = COS_HIST_BINS) -> np.ndarray:
    """Pure NumPy RGB histogram; no matplotlib/VTK/Pillow dependency.

    X: cos(theta) from -1 to +1. Y: relative frequency per bin.
    Gold = cone inliers, magenta = cone outliers, cyan = Q10,
    green = average cosine (equal to resultant length R).
    """
    canvas = np.full((height, width, 3), [17, 25, 38], dtype=np.uint8)
    left, right, top, bottom = 36, width - 16, 14, height - 26
    if right <= left or bottom <= top:
        return canvas
    # Subtle horizontal guides and coordinate axes.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        y = int(round(bottom - fraction * (bottom - top)))
        canvas[max(y, 0):min(y + 1, height), left:right] = [45, 55, 68]
    canvas[bottom:bottom + 2, left:right] = [160, 172, 186]
    canvas[top:bottom + 2, left:left + 2] = [160, 172, 186]
    for cosine_tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        x = int(round(left + (cosine_tick + 1.0) * 0.5 * (right - left - 1)))
        canvas[bottom:bottom + 7, max(0, x):min(width, x + 1)] = [185, 192, 202]
    _tiny_text(canvas, left - 5, bottom + 10, '-1')
    _tiny_text(canvas, (left + right)//2 - 3, bottom + 10, '0')
    _tiny_text(canvas, right - 16, bottom + 10, '+1')
    if cone is None or not len(cone.cosines):
        return canvas

    edges = np.linspace(-1.0, 1.0, bins + 1)
    inside, _ = np.histogram(cone.cosines[cone.inlier_mask], bins=edges)
    outside, _ = np.histogram(cone.cosines[~cone.inlier_mask], bins=edges)
    maximum = max(1, int((inside + outside).max()))
    plot_height = bottom - top - 6
    _tiny_text(canvas, 2, top, f"{100.0 * maximum / len(cone.cosines):.0f}%")
    for j in range(bins):
        x0 = left + int(j * (right - left) / bins) + 1
        x1 = left + int((j + 1) * (right - left) / bins) - 1
        if x1 < x0:
            x1 = x0
        in_h = int(round(plot_height * inside[j] / maximum))
        out_h = int(round(plot_height * outside[j] / maximum))
        # Stack distinct samples; no artificial threshold at histogram bin center.
        if in_h:
            canvas[bottom - in_h:bottom, x0:x1 + 1] = [241, 192, 54]
        if out_h:
            canvas[bottom - in_h - out_h:bottom - in_h, x0:x1 + 1] = [251, 71, 153]

    # Average cosine = m dot mean(u) = ||mean(u)|| = R.
    x_mean = int(round(left + (cone.resultant_length + 1.0) * 0.5 * (right - left - 1)))
    x_mean = np.clip(x_mean, left, right - 1)
    canvas[top:bottom, max(left, x_mean - 1):min(right, x_mean + 1)] = [72, 224, 115]
    x_q10 = int(round(left + (cone.cosine_q10 + 1.0) * 0.5 * (right - left - 1)))
    x_q10 = np.clip(x_q10, left, right - 1)
    canvas[top:bottom, max(left, x_q10 - 1):min(right, x_q10 + 1)] = [40, 220, 248]
    return canvas



def line_set(points, pairs, color):
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line.lines = o3d.utility.Vector2iVector(np.asarray(pairs, dtype=np.int32))
    line.paint_uniform_color(color)
    return line


def sphere_graticule(segments=48):
    """Wireframe unit sphere (latitudes and meridians)."""
    points = []
    pairs = []
    def polyline(arr, loop=False):
        begin = len(points)
        points.extend(np.asarray(arr).tolist())
        pairs.extend((begin + j, begin + j + 1) for j in range(len(arr) - 1))
        if loop:
            pairs.append((begin + len(arr) - 1, begin))
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        polyline(np.column_stack([np.cos(lat) * np.cos(t),
                                  np.cos(lat) * np.sin(t),
                                  np.full_like(t, np.sin(lat))]), loop=True)
    for lon in np.linspace(0, np.pi, 7, endpoint=False):
        lat = np.linspace(-np.pi/2, np.pi/2, segments)
        polyline(np.column_stack([np.cos(lat) * np.cos(lon),
                                  np.cos(lat) * np.sin(lon),
                                  np.sin(lat)]))
    return line_set(points, pairs, [0.48, 0.54, 0.63])


def cone_geometry(origin, axis, alpha_deg, scale=1.0, segments=64):
    """Surface of angular cone on sphere of radius scale; apex at origin.

    Half-angle may exceed 90 degrees; then the rim is behind the apex
    relative to the mean axis. This is intentional and unambiguous.
    """
    axis = _unit(axis)
    if axis is None:
        raise ValueError('Cone axis is zero')
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < .9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)
    alpha = np.deg2rad(float(alpha_deg))
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rim = origin + float(scale) * (np.cos(alpha) * axis[None, :] +
              np.sin(alpha) * (np.cos(t)[:, None] * e1[None, :] +
                               np.sin(t)[:, None] * e2[None, :]))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack([origin[None, :], rim]))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray([(0, i + 1, (i + 1) % segments + 1) for i in range(segments)], dtype=np.int32))
    mesh.compute_vertex_normals()
    ring = line_set(rim, [(i, (i+1) % segments) for i in range(segments)], [0.09, 0.86, 1.0])
    spokes = line_set(np.vstack([origin, rim]),
                      [(0, 1 + i) for i in range(0, segments, max(1, segments // 8))],
                      [0.08, 0.68, 0.93])
    return mesh, ring, spokes



# ----- Optional PCA eigenvalue field analysis (independent of Q10) -----

@dataclass(frozen=True)
class EigenFieldResult:
    positions_mm: np.ndarray
    eigenvalues_mm2: np.ndarray      # Descending: lambda1, lambda2, lambda3
    eigenvectors_xyz: np.ndarray     # N x 3 x 3, columns v1/v2/v3 (descending)
    lambda2_over_lambda1: np.ndarray
    gradients_lambda1: np.ndarray   # Full 3D least-squares estimates
    gradients_ratio: np.ndarray
    minima_lambda1: np.ndarray      # Indices into positions_mm
    maxima_ratio: np.ndarray
    center_mm: np.ndarray
    axes: np.ndarray                # 3x2 chart projection basis
    chart_xy_mm: np.ndarray
    center_index: int
    pca_radius_mm: float
    ced_radius_mm: float
    view_radius_mm: float
    mode: str
    total_candidates: int
    effective_counts: np.ndarray


def _eigen_field_tree(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return o3d.geometry.KDTreeFlann(cloud)


def _eigen_field_radius(tree, p, radius):
    n, idx, _ = tree.search_radius_vector_3d(
        np.asarray(p, dtype=np.float64), float(radius) + 1e-8)
    return np.asarray(idx, dtype=np.int64) if n else np.empty(0, dtype=np.int64)




def compute_full_scene_lambda1(points_mm: np.ndarray,
                               pca_radius_mm: float = 30.0,
                               ced_radius_mm: float = 10.0,
                               mode: str = 'ced_threshold',
                               ced_cutoff: float = 0.10,
                               min_neighbors: int = 12,
                               is_cancelled=None,
                               progress_callback=None) -> np.ndarray:
    """Compute first PCA eigenvalue for EVERY global analysis voxel point.

    This full-scene path deliberately skips local gradients, extrema detection,
    and chart generation. It uses the same mean-centered covariance and raw-CED
    weighting/cutoff as compute_eigen_field, with no ROI sampling or cap.
    Invalid points (insufficient support/effective samples) stay NaN.
    """
    xyz = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < min_neighbors or not np.isfinite(xyz).all():
        raise ValueError('Too few valid global analysis points')
    if mode not in ('unweighted', 'ced_weighted', 'ced_threshold'):
        raise ValueError(f'Invalid PCA mode: {mode}')
    if pca_radius_mm <= 0 or ced_radius_mm <= 0:
        raise ValueError('PCA/CED radius must be positive')

    tree = _eigen_field_tree(xyz)
    n = len(xyz)
    offsets = None
    if mode != 'unweighted':
        offsets = np.zeros(n, dtype=np.float64)
        for j, p in enumerate(xyz):
            if j % 64 == 0 and is_cancelled is not None and is_cancelled():
                return None
            ids = _eigen_field_radius(tree, p, ced_radius_mm)
            if len(ids) >= 3:
                offsets[j] = np.linalg.norm(xyz[ids].mean(axis=0) - p) / ced_radius_mm
            if progress_callback is not None and (j + 1) % 3000 == 0:
                progress_callback('CED', j + 1, n)

    lambda1 = np.full(n, np.nan, dtype=np.float64)
    for j, p in enumerate(xyz):
        if j % 64 == 0 and is_cancelled is not None and is_cancelled():
            return None
        ids = _eigen_field_radius(tree, p, pca_radius_mm)
        if len(ids) < min_neighbors:
            continue
        if offsets is None:
            support = xyz[ids]
            mu = support.mean(axis=0)
            delta = support - mu
            cov = delta.T @ delta / len(support)
        else:
            weights = offsets[ids]
            if mode == 'ced_threshold':
                active = weights >= ced_cutoff
            else:
                active = weights > 0.0
            if np.count_nonzero(active) < min_neighbors:
                continue
            support = xyz[ids[active]]
            weights = weights[active]
            sw = float(weights.sum())
            if sw <= 0 or sw * sw / np.dot(weights, weights) < min_neighbors:
                continue
            mu = np.average(support, axis=0, weights=weights)
            delta = support - mu
            cov = (delta * weights[:, None]).T @ delta / sw
        lambda1[j] = float(np.linalg.eigvalsh(cov)[-1])
        if progress_callback is not None and (j + 1) % 3000 == 0:
            progress_callback('PCA', j + 1, n)
    return lambda1


# ---- Local diagnostics: independent extrema in four detection strategies ----

@dataclass(frozen=True)
class FusionPcaExtremaComparison:
    positions_mm: np.ndarray
    fusion_scores: np.ndarray
    normal_lambda1_mm2: np.ndarray
    fusion_maxima: np.ndarray
    lambda1_minima: np.ndarray  # backward-compatible alias: normal PCA lambda1 MIN
    matches: tuple  # legacy Fusion MAX vs Normal lambda1 MIN comparison
    center_mm: np.ndarray
    view_radius_mm: float
    nms_radius_mm: float
    match_radius_mm: float
    ced_radius_mm: float
    pca_radius_mm: float
    sigma_ced: float
    tau: float
    ced_cutoff: float
    # Additional fields, all sampled at exactly the same local voxel positions.
    ced_scores: np.ndarray  # bilateral Raw CED, i.e. the CED used by Fusion
    ced_weighted_ratio: np.ndarray  # lambda2_weighted / lambda1_weighted
    ced_weighted_lambda1_mm2: np.ndarray
    ced_maxima: np.ndarray
    ratio_maxima: np.ndarray
    weighted_lambda1_minima: np.ndarray
    match_ced_ratio: tuple
    match_ced_weighted_l1: tuple
    match_ced_normal_l1: tuple
    local_minimum_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    ced_local_min_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    ced_local_min_maxima: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    contrast_tau: float = .005
    median_minimum_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    ced_median_min_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    ced_median_min_maxima: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    median_contrast_tau: float = .005


def compute_local_minimum_fusion(positions_mm, lambda1_mm2, ced_scores,
                                 radius_mm, contrast_tau=.005, is_cancelled=None,
                                 *, statistic='max'):
    """Local max/median contrast above the minimum; S=CED*L on measured voxels."""
    if statistic not in ('max', 'median'):
        raise ValueError('Local minimum statistic must be max or median')
    xyz = np.asarray(positions_mm, dtype=np.float64)
    eigenvalues = np.asarray(lambda1_mm2, dtype=np.float64)
    ced = np.asarray(ced_scores, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or eigenvalues.shape != (len(xyz),) or ced.shape != (len(xyz),)):
        raise ValueError('Minimum-fusion requires finite (N,3) XYZ and aligned scalar arrays')
    if not np.isfinite([radius_mm, contrast_tau]).all() or min(radius_mm, contrast_tau) <= 0:
        raise ValueError('Minimum-fusion radius and contrast tau must be finite and > 0')
    regularizer = float(contrast_tau) * float(radius_mm) * float(radius_mm)
    if not np.isfinite(regularizer) or regularizer <= 0:
        raise ValueError('Minimum-fusion tau * R² must be finite and > 0')
    minimum = np.full(len(xyz), np.nan)
    tree = _eigen_field_tree(xyz)
    finite = np.isfinite(eigenvalues)
    for step, i in enumerate(np.flatnonzero(finite)):
        if step % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        ids = _eigen_field_radius(tree, xyz[i], radius_mm)
        values = eigenvalues[ids[finite[ids]]]
        if statistic == 'median':
            low, median = float(values.min()), float(np.median(values))
            minimum[i] = max(0., (median - eigenvalues[i]) / (median - low + regularizer))
        else:
            low, high = float(values.min()), float(values.max())
            minimum[i] = (high - eigenvalues[i]) / (high - low + regularizer)
    scores = np.full(len(xyz), np.nan)
    valid = np.isfinite(minimum) & np.isfinite(ced)
    scores[valid] = ced[valid] * minimum[valid]
    return minimum, scores


def with_local_minimum_fusion(result: FusionPcaExtremaComparison, contrast_tau=.005,
                              *, local_mode=False, is_cancelled=None,
                              median_contrast_tau=.005):
    """Attach a separate Normal-PCA minimum score/MAX NMS; retain old fields."""
    from dataclasses import replace
    fields = compute_local_minimum_fusion(
        result.positions_mm, result.normal_lambda1_mm2, result.ced_scores,
        result.pca_radius_mm, contrast_tau, is_cancelled)
    if fields is None:
        return None
    minimum, scores = fields
    # Zero response is not a corner peak, including constant eigenvalue fields.
    if local_mode:
        maxima = find_local_scalar_extrema(
            result.positions_mm, scores, result.nms_radius_mm,
            result.center_mm, result.view_radius_mm, maximize=True)
    else:
        maxima = find_full_scene_scalar_extrema(
            result.positions_mm, scores, result.nms_radius_mm,
            maximize=True, is_cancelled=is_cancelled)
    if maxima is None:
        return None
    maxima = maxima[scores[maxima] > 0]
    median_fields = compute_local_minimum_fusion(
        result.positions_mm, result.normal_lambda1_mm2, result.ced_scores,
        result.pca_radius_mm, median_contrast_tau, is_cancelled, statistic='median')
    if median_fields is None:
        return None
    median_minimum, median_scores = median_fields
    if local_mode:
        median_maxima = find_local_scalar_extrema(
            result.positions_mm, median_scores, result.nms_radius_mm,
            result.center_mm, result.view_radius_mm, maximize=True)
    else:
        median_maxima = find_full_scene_scalar_extrema(
            result.positions_mm, median_scores, result.nms_radius_mm,
            maximize=True, is_cancelled=is_cancelled)
    if median_maxima is None:
        return None
    median_maxima = median_maxima[median_scores[median_maxima] > 0]
    return replace(result, local_minimum_scores=minimum, ced_local_min_scores=scores,
                   ced_local_min_maxima=maxima, contrast_tau=float(contrast_tau),
                   median_minimum_scores=median_minimum, ced_median_min_scores=median_scores,
                   ced_median_min_maxima=median_maxima, median_contrast_tau=float(median_contrast_tau))


def find_local_scalar_extrema(positions_mm, scores, neighborhood_radius_mm,
                              roi_center_mm, roi_radius_mm, maximize=False,
                              max_extrema=25):
    """Discrete neighborhood extrema with duplicate suppression in 3D.

    Exclude the artificial ROI boundary. Ignore NaN samples; all four scalar
    fields use the identical NMS radius, point positions, and interior mask.
    """
    xyz = np.asarray(positions_mm, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    center = np.asarray(roi_center_mm, dtype=np.float64)
    tree = _eigen_field_tree(xyz)
    candidates = []
    for i in np.flatnonzero(finite):
        if np.linalg.norm(xyz[i] - center) > roi_radius_mm - neighborhood_radius_mm:
            continue
        ids = _eigen_field_radius(tree, xyz[i], neighborhood_radius_mm)
        ids = ids[finite[ids]]
        if len(ids) < 5:
            continue
        extremum = np.max(values[ids]) if maximize else np.min(values[ids])
        if ((values[i] >= extremum - 1e-9) if maximize
                else (values[i] <= extremum + 1e-9)):
            candidates.append(int(i))
    candidates.sort(key=lambda i: (
        -values[i] if maximize else values[i],
        np.linalg.norm(xyz[i] - center), i))
    selected = []
    for i in candidates:
        if all(np.linalg.norm(xyz[i] - xyz[j]) > neighborhood_radius_mm
               for j in selected):
            selected.append(i)
        if len(selected) >= max_extrema:
            break
    return np.asarray(selected, dtype=np.int64)


def match_fusion_lambda1_extrema(positions_mm, fusion_maxima, lambda1_minima,
                                 max_distance_mm):
    """Generic one-to-one, nearest-first pairing between two sets of extrema.

    Kept under its original name for backward compatibility with old scripts.
    Matching is not an NMS operation and permits different 3D positions.
    """
    xyz = np.asarray(positions_mm, dtype=np.float64)
    possible = []
    for fi in np.asarray(fusion_maxima, dtype=np.int64):
        for li in np.asarray(lambda1_minima, dtype=np.int64):
            distance = float(np.linalg.norm(xyz[fi] - xyz[li]))
            if distance <= max_distance_mm + 1e-9:
                possible.append((distance, int(fi), int(li)))
    possible.sort()
    used_f, used_l, matched = set(), set(), []
    for distance, fi, li in possible:
        if fi not in used_f and li not in used_l:
            matched.append((fi, li, distance))
            used_f.add(fi)
            used_l.add(li)
    return tuple(matched)


def rematch_extrema_comparison(result: FusionPcaExtremaComparison,
                               radius_mm: float) -> FusionPcaExtremaComparison:
    """Recompute only the four pairings when the GUI matching slider changes."""
    from dataclasses import replace
    xyz = result.positions_mm
    pair = lambda a, b: match_full_scene_extrema(xyz, a, b, radius_mm)
    return replace(result,
        matches=pair(result.fusion_maxima, result.lambda1_minima),
        match_ced_ratio=pair(result.ced_maxima, result.ratio_maxima),
        match_ced_weighted_l1=pair(result.ced_maxima, result.weighted_lambda1_minima),
        match_ced_normal_l1=pair(result.ced_maxima, result.lambda1_minima),
        match_radius_mm=float(radius_mm))


def threshold_extrema_comparison(result: FusionPcaExtremaComparison,
                                 thresholds: dict[str, float]) -> FusionPcaExtremaComparison:
    """Filter cached NMS by inclusive score gates; rematch without mutating it."""
    from dataclasses import replace
    if not thresholds:
        return result
    changes = {}
    for _, value_key, extrema_key, extreme, _ in SCORE_VIEWS:
        if extrema_key is None or value_key not in thresholds:
            continue
        limit = float(thresholds[value_key])
        if not np.isfinite(limit) or limit < 0:
            raise ValueError('NMS thresholds must be finite and >= 0')
        ids = getattr(result, extrema_key)
        values = getattr(result, value_key)[ids]
        keep = np.isfinite(values) & (values >= limit if extreme == 'MAX' else values <= limit)
        changes[extrema_key] = ids[keep]
    return rematch_extrema_comparison(replace(result, **changes), result.match_radius_mm)


def extrema_comparison_mode(result: FusionPcaExtremaComparison, mode: int):
    """Return (primary extrema, secondary extrema, matched pairs) for GUI mode."""
    if mode == 0:
        return result.fusion_maxima, np.empty(0, dtype=np.int64), ()
    if mode == 1:
        return result.ced_maxima, result.ratio_maxima, result.match_ced_ratio
    if mode == 2:
        return result.ced_maxima, result.weighted_lambda1_minima, result.match_ced_weighted_l1
    if mode == 3:
        return result.ced_maxima, result.lambda1_minima, result.match_ced_normal_l1
    raise ValueError('Invalid extrema comparison mode')



# ---- Global four-method comparison (all voxels; independent of label/ROI) ----

def find_full_scene_scalar_extrema(positions_mm, scores, nms_radius_mm,
                                   maximize=False, is_cancelled=None):
    """Spatial NMS over every valid *observed* voxel in the full scene.

    No artificial spherical-ROI boundary exclusion and NO top-K extrema cap.
    Actual scan boundaries can still create misleading local minima/maxima.
    """
    xyz = np.asarray(positions_mm, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if nms_radius_mm <= 0:
        raise ValueError('Full-scene NMS radius must be positive')
    finite = np.isfinite(values)
    tree = _eigen_field_tree(xyz)
    candidate = []
    for step, i in enumerate(np.flatnonzero(finite)):
        if step % 512 == 0 and is_cancelled is not None and is_cancelled():
            return None
        ids = _eigen_field_radius(tree, xyz[i], nms_radius_mm)
        ids = ids[finite[ids]]
        if len(ids) < 3:
            continue
        extreme = np.max(values[ids]) if maximize else np.min(values[ids])
        if ((values[i] >= extreme - 1e-9) if maximize
                else (values[i] <= extreme + 1e-9)):
            candidate.append(int(i))
    if not candidate:
        return np.empty(0, dtype=np.int64)
    ids = np.asarray(candidate, dtype=np.int64)
    # Score-first order; deterministic point index breaks ties on plateaus.
    ordering = np.lexsort((ids, -values[ids] if maximize else values[ids]))
    suppressed = np.zeros(len(xyz), dtype=bool)
    kept = []
    for step, i in enumerate(ids[ordering]):
        if step % 512 == 0 and is_cancelled is not None and is_cancelled():
            return None
        if suppressed[i]:
            continue
        kept.append(int(i))
        suppressed[_eigen_field_radius(tree, xyz[i], nms_radius_mm)] = True
    return np.asarray(kept, dtype=np.int64)


def match_full_scene_extrema(positions_mm, first, second, radius_mm):
    """Fast radius-based, 1:1 matching; avoids O(N_first * N_second)."""
    xyz = np.asarray(positions_mm, dtype=np.float64)
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if not len(first) or not len(second):
        return ()
    sub_tree = _eigen_field_tree(xyz[second])
    pairs = []
    for a in first:
        indices = _eigen_field_radius(sub_tree, xyz[a], radius_mm)
        for j in indices:
            b = int(second[j])
            dist = float(np.linalg.norm(xyz[a] - xyz[b]))
            if dist <= radius_mm + 1e-9:
                pairs.append((dist, int(a), b))
    pairs.sort()  # Nearest pairing first, one-to-one.
    used_a, used_b, matches = set(), set(), []
    for dist, a, b in pairs:
        if a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            matches.append((a, b, dist))
    return tuple(matches)


def compute_full_scene_fusion_pca_comparison(points_mm,
                                              pca_radius_mm=30.0,
                                              ced_radius_mm=10.0,
                                              voxel_mm=5.0,
                                              nms_radius_mm=10.0,
                                              match_radius_mm=8.0,
                                              sigma_ced=0.05, tau=0.5,
                                              ced_cutoff=0.10,
                                              min_neighbors=12,
                                              is_cancelled=None,
                                              progress_callback=None,
                                              max_evaluation_points=120000,
                                              contrast_tau=.005,
                                              median_contrast_tau=.005):
    """Calculate four extrema methods at every voxel across the complete scene.

    Input points are already voxel-downsampled once from the ORIGINAL scene.
    Same score definitions as the local diagnostic (including CED cutoff).
    No ROI truncation, no random sampling, no top-K extrema cap.
    A large point budget causes an EXPLICIT error rather than silent sampling.
    """
    xyz = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if not np.isfinite(xyz).all() or n < min_neighbors:
        raise ValueError('Whole scene has too few valid voxel points')
    if min(pca_radius_mm, ced_radius_mm, voxel_mm, nms_radius_mm,
           match_radius_mm, sigma_ced, tau) <= 0:
        raise ValueError('Voxel/NMS/matching/PCA/CED parameters must be positive')
    if n > max_evaluation_points:
        raise ValueError(f'Scene contains {n:,} voxel points (safe limit '
                         f'{max_evaluation_points:,}). Increase full-scene voxel size.')
    tree = _eigen_field_tree(xyz)
    raw_ced = np.full(n, np.nan)
    bilateral_ced = np.full(n, np.nan)
    fusion = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    weighted_l1 = np.full(n, np.nan)
    normal_l1 = np.full(n, np.nan)

    def report(stage, index):
        if progress_callback is not None and index % 1500 == 0:
            progress_callback(stage, index, n)

    for i, p in enumerate(xyz):
        if i % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        ids = _eigen_field_radius(tree, p, ced_radius_mm)
        if len(ids) >= 3:
            raw_ced[i] = float(np.linalg.norm(xyz[ids].mean(axis=0) - p)
                               / ced_radius_mm)
        report('Raw CED', i + 1)
    ced_for_weights = np.nan_to_num(raw_ced, nan=0.0)

    for i, p in enumerate(xyz):
        if i % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        if np.isfinite(raw_ced[i]):
            ids = _eigen_field_radius(tree, p, ced_radius_mm)
            distances_sq = np.sum((xyz[ids] - p) ** 2, axis=1)
            diff = ced_for_weights[ids] - ced_for_weights[i]
            w = np.exp(-0.5 * distances_sq / ced_radius_mm**2
                       - 0.5 * (diff / sigma_ced)**2)
            sw = float(w.sum())
            if sw > 0:
                bilateral_ced[i] = float(w @ ced_for_weights[ids] / sw)
        report('Bilateral CED', i + 1)

    for i, p in enumerate(xyz):
        if i % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        ids = _eigen_field_radius(tree, p, pca_radius_mm)
        if len(ids) < min_neighbors:
            continue
        pts = xyz[ids]
        dd = pts - pts.mean(axis=0)
        normal_l1[i] = float(np.linalg.eigvalsh(dd.T @ dd / len(pts))[-1])
        weights = ced_for_weights[ids].copy()
        weights[weights < ced_cutoff] = 0.
        keep = weights > 0
        if np.count_nonzero(keep) < min_neighbors:
            continue
        weights = weights[keep]
        sw = float(weights.sum())
        if sw <= 0 or sw**2 / float(weights @ weights) < min_neighbors:
            continue
        pts = pts[keep]
        mu = np.average(pts, axis=0, weights=weights)
        dd = pts - mu
        eig = np.linalg.eigvalsh((dd * weights[:, None]).T @ dd / sw)
        weighted_l1[i] = float(eig[-1])
        if eig[-1] > 1e-5 * pca_radius_mm**2:
            ratio[i] = float(np.clip(eig[-2] / eig[-1], 0., 1.))
            if np.isfinite(bilateral_ced[i]):
                fusion[i] = float(np.clip(
                    bilateral_ced[i] * tau * np.tanh(ratio[i] / tau), 0., 1.))
        report('PCA / Fusion', i + 1)
    if np.isfinite(fusion).sum() < min_neighbors:
        raise ValueError('Too few valid Fusion voxels. Try smaller full-scene voxel '
                         'size, larger PCA/CED radius, or lower cutoff.')

    fields = [('Fusion MAX', fusion, True), ('CED MAX', bilateral_ced, True),
              ('CED-PCA ratio MAX', ratio, True),
              ('CED-PCA lambda1 MIN', weighted_l1, False),
              ('Normal-PCA lambda1 MIN', normal_l1, False)]
    extrema = []
    for name, field, maximize in fields:
        if is_cancelled is not None and is_cancelled():
            return None
        if progress_callback is not None:
            progress_callback('NMS: ' + name, 0, n)
        ids = find_full_scene_scalar_extrema(
            xyz, field, nms_radius_mm, maximize, is_cancelled)
        if ids is None:
            return None
        extrema.append(ids)
    fmax, cmax, rmax, wlmin, nlmin = extrema
    pair = lambda a,b: match_full_scene_extrema(xyz, a,b,match_radius_mm)
    center = xyz.mean(axis=0)
    view_radius = float(np.max(np.linalg.norm(xyz - center, axis=1)))
    result = FusionPcaExtremaComparison(
        positions_mm=xyz, fusion_scores=fusion,
        normal_lambda1_mm2=normal_l1, fusion_maxima=fmax,
        lambda1_minima=nlmin, matches=pair(fmax,nlmin),
        center_mm=center, view_radius_mm=view_radius,
        nms_radius_mm=float(nms_radius_mm),
        match_radius_mm=float(match_radius_mm),
        ced_radius_mm=float(ced_radius_mm),
        pca_radius_mm=float(pca_radius_mm),
        sigma_ced=float(sigma_ced), tau=float(tau), ced_cutoff=float(ced_cutoff),
        ced_scores=bilateral_ced, ced_weighted_ratio=ratio,
        ced_weighted_lambda1_mm2=weighted_l1,
        ced_maxima=cmax, ratio_maxima=rmax,
        weighted_lambda1_minima=wlmin,
        match_ced_ratio=pair(cmax,rmax),
        match_ced_weighted_l1=pair(cmax,wlmin),
        match_ced_normal_l1=pair(cmax,nlmin))
    if progress_callback is not None:
        progress_callback('Local lambda1 contrast / MAX NMS', 0, n)
    return with_local_minimum_fusion(result, contrast_tau, is_cancelled=is_cancelled,
                                     median_contrast_tau=median_contrast_tau)

def compute_local_fusion_pca_comparison(points_mm, center_mm,
                                        pca_radius_mm=30.0,
                                        ced_radius_mm=10.0,
                                        view_radius_mm=40.0,
                                        voxel_mm=2.0,
                                        match_radius_mm=8.0,
                                        sigma_ced=0.05,
                                        tau=0.5,
                                        ced_cutoff=0.10,
                                        min_neighbors=12,
                                        max_evaluation_points=6500,
                                        contrast_tau=.005,
                                        median_contrast_tau=.005):
    """Compare four extrema strategies on the SAME complete local support.

    0: Diagnostic Fusion MAX NMS
    1: Bilateral CED MAX + CED-weighted ratio MAX, proximity matched
    2: Bilateral CED MAX + CED-weighted lambda1 MIN, proximity matched
    3: Bilateral CED MAX + normal-PCA lambda1 MIN, proximity matched

    Fusion = bilateral(raw CED; sigma=0.05) * tau*tanh(weighted_ratio/tau).
    Weighted PCA uses raw CED weights with CED>=ced_cutoff, as in this file.
    The diagnostic Fusion is NOT guaranteed identical to the external detector.
    All scores use every analysis voxel point (never silently subsampled).
    """
    xyz = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_mm, dtype=np.float64).reshape(3)
    if min(pca_radius_mm, ced_radius_mm, view_radius_mm, voxel_mm,
           match_radius_mm, sigma_ced, tau) <= 0:
        raise ValueError('All analysis radii and sigma/tau must be positive')
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('No valid analysis points')
    distance = np.linalg.norm(xyz - center, axis=1)
    outer = view_radius_mm + pca_radius_mm + 2.0 * ced_radius_mm + 0.01
    local_xyz = xyz[distance <= outer]
    tree = _eigen_field_tree(local_xyz)
    rel = np.linalg.norm(local_xyz - center, axis=1)
    query_ids = np.flatnonzero(rel <= view_radius_mm)
    if len(query_ids) < 12:
        raise ValueError('Fewer than 12 observed points in comparison ROI')
    if len(query_ids) > max_evaluation_points:
        raise ValueError(f'ROI has {len(query_ids):,} voxels, limit {max_evaluation_points:,}. '
                         'Increase analysis voxel size or reduce the PCA search region.')
    positions = local_xyz[query_ids]

    raw_ced = np.full(len(local_xyz), np.nan, dtype=np.float64)
    ced_support_ids = np.flatnonzero(rel <= view_radius_mm + pca_radius_mm + ced_radius_mm + 0.01)
    for j in ced_support_ids:
        ids = _eigen_field_radius(tree, local_xyz[j], ced_radius_mm)
        if len(ids) >= 3:
            raw_ced[j] = (np.linalg.norm(local_xyz[ids].mean(axis=0) - local_xyz[j])
                          / ced_radius_mm)
    # For compatibility with the original diagnostic, unsampled/unsupported CED
    # values inside the support are zero; invalid query values themselves stay NaN.
    ced_for_weights = np.nan_to_num(raw_ced, nan=0.0)

    fusion = np.full(len(query_ids), np.nan, dtype=np.float64)
    bilateral_ced = np.full(len(query_ids), np.nan, dtype=np.float64)
    weighted_ratio = np.full(len(query_ids), np.nan, dtype=np.float64)
    weighted_l1 = np.full(len(query_ids), np.nan, dtype=np.float64)
    normal_l1 = np.full(len(query_ids), np.nan, dtype=np.float64)

    for row, i in enumerate(query_ids):
        p = local_xyz[i]
        # Bilateral CED is independent of PCA validity.
        if np.isfinite(raw_ced[i]):
            bid = _eigen_field_radius(tree, p, ced_radius_mm)
            dist2 = np.sum((local_xyz[bid] - p)**2, axis=1)
            dc = ced_for_weights[bid] - ced_for_weights[i]
            bw = np.exp(-0.5 * dist2 / ced_radius_mm**2
                        - 0.5 * (dc / sigma_ced)**2)
            sw_b = float(bw.sum())
            if sw_b > 0:
                bilateral_ced[row] = float(np.dot(bw, ced_for_weights[bid]) / sw_b)

        ids = _eigen_field_radius(tree, p, pca_radius_mm)
        if len(ids) < min_neighbors:
            continue
        support = local_xyz[ids]
        mu = support.mean(axis=0)
        delta = support - mu
        normal_l1[row] = float(np.linalg.eigvalsh(delta.T @ delta / len(support))[-1])

        weights = ced_for_weights[ids].copy()
        weights[weights < ced_cutoff] = 0.0
        active = weights > 0
        if np.count_nonzero(active) < min_neighbors:
            continue
        w = weights[active]
        sw = float(w.sum())
        if sw <= 0 or sw**2 / float(np.dot(w, w)) < min_neighbors:
            continue
        q = support[active]
        mu_w = np.average(q, axis=0, weights=w)
        dd = q - mu_w
        eig = np.linalg.eigvalsh((dd * w[:, None]).T @ dd / sw)
        weighted_l1[row] = float(eig[-1])
        if eig[-1] <= 1e-5 * pca_radius_mm**2:
            continue
        weighted_ratio[row] = float(np.clip(eig[-2] / eig[-1], 0.0, 1.0))
        if np.isfinite(bilateral_ced[row]):
            fusion[row] = float(np.clip(
                bilateral_ced[row] * tau * np.tanh(weighted_ratio[row] / tau),
                0.0, 1.0))

    if np.isfinite(fusion).sum() < 12 or np.isfinite(normal_l1).sum() < 12:
        raise ValueError('Insufficient valid Fusion/normal-PCA samples; '
                         'try a larger PCA radius or lower voxel size')
    nms_radius = max(2.5 * voxel_mm, 0.22 * pca_radius_mm)
    extrema = lambda vals, is_max: find_local_scalar_extrema(
        positions, vals, nms_radius, center, view_radius_mm, maximize=is_max)
    fmax = extrema(fusion, True)
    cmax = extrema(bilateral_ced, True)
    rmax = extrema(weighted_ratio, True)
    wlmin = extrema(weighted_l1, False)
    nlmin = extrema(normal_l1, False)
    pair = lambda a, b: match_fusion_lambda1_extrema(positions, a, b, match_radius_mm)
    result = FusionPcaExtremaComparison(
        positions_mm=positions, fusion_scores=fusion,
        normal_lambda1_mm2=normal_l1, fusion_maxima=fmax,
        lambda1_minima=nlmin, matches=pair(fmax, nlmin),
        center_mm=center, view_radius_mm=float(view_radius_mm),
        nms_radius_mm=float(nms_radius), match_radius_mm=float(match_radius_mm),
        ced_radius_mm=float(ced_radius_mm), pca_radius_mm=float(pca_radius_mm),
        sigma_ced=float(sigma_ced), tau=float(tau), ced_cutoff=float(ced_cutoff),
        ced_scores=bilateral_ced, ced_weighted_ratio=weighted_ratio,
        ced_weighted_lambda1_mm2=weighted_l1,
        ced_maxima=cmax, ratio_maxima=rmax,
        weighted_lambda1_minima=wlmin,
        match_ced_ratio=pair(cmax, rmax),
        match_ced_weighted_l1=pair(cmax, wlmin),
        match_ced_normal_l1=pair(cmax, nlmin))
    return with_local_minimum_fusion(result, contrast_tau, local_mode=True,
                                     median_contrast_tau=median_contrast_tau)

def compute_eigen_field(points_mm: np.ndarray, center_mm: np.ndarray,
                        pca_radius_mm: float = 30.0,
                        view_radius_mm: float = 40.0,
                        ced_radius_mm: float = 10.0,
                        mode: str = 'ced_threshold',
                        ced_cutoff: float = 0.10,
                        min_neighbors: int = 12,
                        max_evaluation_points: int = 1800,
                        gradient_radius_mm: float | None = None,
                        voxel_mm: float = 2.0,
                        dense_radius_mm: float | None = None) -> EigenFieldResult:
    """Local *diagnostic* PCA field, evaluated on actual downsampled points.

    mode: 'unweighted', 'ced_weighted', 'ced_threshold'. The CED used here is
    raw centroid offset ||mean(q-p)||/ced_radius. This is NOT bilateral CED or
    the exact Fusion weighted-PCA implementation from another codebase.

    Important: cutoffs can make small eigenvalues spuriously small. A minimum
    effective sample size is enforced, and invalid positions remain NaN.
    """
    xyz = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_mm, dtype=np.float64).reshape(3)
    if len(xyz) < min_neighbors or not np.isfinite(xyz).all():
        raise ValueError('Not enough valid global voxel points')
    if mode not in ('unweighted', 'ced_weighted', 'ced_threshold'):
        raise ValueError(f'Invalid PCA mode: {mode}')
    if min(pca_radius_mm, ced_radius_mm, view_radius_mm, voxel_mm) <= 0:
        raise ValueError('All radii and voxel size must be positive')
    # Local superset: chart center + PCA support + CED support. No cut-off
    # at the view boundary when calculating a support point's CED.
    distance = np.linalg.norm(xyz - center, axis=1)
    outer_radius = view_radius_mm + pca_radius_mm + ced_radius_mm + 0.02
    local_xyz = xyz[distance <= outer_radius]
    if len(local_xyz) < min_neighbors:
        raise ValueError('Insufficient points in eigen field support')
    tree = _eigen_field_tree(local_xyz)
    relative = np.linalg.norm(local_xyz - center, axis=1)
    all_candidates = np.flatnonzero(relative <= view_radius_mm)
    total_candidates = len(all_candidates)
    if total_candidates < min_neighbors:
        raise ValueError('Too few sampled points in visualization radius')
    # The eigen refiner must see EVERY voxel in its search neighborhood;
    # coarse 3D heatmap sampling outside that dense center is still allowed.
    if len(all_candidates) > max_evaluation_points:
        dense = min(float(view_radius_mm), max(0.,float(dense_radius_mm or 0.)))
        mandatory = all_candidates[relative[all_candidates] <= dense + 1e-8]
        center_voxel = all_candidates[np.argmin(relative[all_candidates])]
        mandatory = np.unique(np.append(mandatory, center_voxel))
        optional = np.setdiff1d(all_candidates, mandatory, assume_unique=False)
        room = max(0, max_evaluation_points - len(mandatory))
        if room and len(optional):
            order = np.lexsort((local_xyz[optional,2],
                                local_xyz[optional,1],
                                local_xyz[optional,0]))
            optional = optional[order]
            keep = np.linspace(0, len(optional)-1, min(room,len(optional)),
                               dtype=np.int64)
            candidate_indices = np.sort(np.r_[mandatory, optional[keep]])
        else:
            candidate_indices = np.sort(mandatory)
    else:
        candidate_indices = all_candidates
    positions = local_xyz[candidate_indices]
    # For unweighted mode CED is not calculated at all; same real 3D support
    # points, same covariance eigenvalue ordering and global refinement rules.
    # Calculate CED for ALL possible PCA support points, not just chart samples.
    offsets = np.zeros(len(local_xyz), dtype=np.float64)
    if mode != 'unweighted':
        ced_support = np.flatnonzero(relative <= view_radius_mm + pca_radius_mm + 0.01)
        for j in ced_support:
            nearby = _eigen_field_radius(tree, local_xyz[j], ced_radius_mm)
            if len(nearby) >= 3:
                offsets[j] = np.linalg.norm(local_xyz[nearby].mean(axis=0)
                                            - local_xyz[j]) / ced_radius_mm
    eigenvalues = np.full((len(positions), 3), np.nan, dtype=np.float64)
    eigenvectors = np.full((len(positions), 3, 3), np.nan, dtype=np.float64)
    n_effective = np.full(len(positions), np.nan, dtype=np.float64)
    for row, (i, p) in enumerate(zip(candidate_indices, positions)):
        ids = _eigen_field_radius(tree, p, pca_radius_mm)
        if len(ids) < min_neighbors:
            continue
        support = local_xyz[ids]
        if mode == 'unweighted':
            weights = np.ones(len(ids))
        else:
            weights = offsets[ids].copy()
            if mode == 'ced_threshold':
                weights[weights < ced_cutoff] = 0.0
        active = weights > 0.0
        if int(active.sum()) < min_neighbors:
            continue
        support = support[active]
        weights = weights[active]
        sum_w = weights.sum()
        effective = sum_w * sum_w / np.sum(weights * weights)
        if effective < min_neighbors:
            continue
        center_w = np.average(support, axis=0, weights=weights)
        delta = support - center_w
        covariance = (delta * weights[:, None]).T @ delta / sum_w
        vals, vecs = np.linalg.eigh(covariance)
        eigenvalues[row] = vals[::-1]
        eigenvectors[row] = vecs[:, ::-1]
        n_effective[row] = effective
    ratio = np.full(len(positions), np.nan)
    good_ratio = (np.isfinite(eigenvalues[:, 0]) &
                  (eigenvalues[:, 0] > 1e-5 * pca_radius_mm ** 2))
    ratio[good_ratio] = eigenvalues[good_ratio, 1] / eigenvalues[good_ratio, 0]
    # Use PCA axes only to project the diagram into 2D; gradient is fitted in 3D.
    # A 3-face corner can be ambiguous in a 2D projection; report that in UI.
    valid = np.isfinite(eigenvalues[:, 0])
    if np.count_nonzero(valid) < 12 or np.count_nonzero(np.isfinite(ratio)) < 12:
        raise ValueError('Too few valid PCA samples (try unweighted, larger radius, or lower cutoff)')
    local_centered = positions - center
    chart_cov = local_centered.T @ local_centered / len(positions)
    _, axes_all = np.linalg.eigh(chart_cov)
    axes = axes_all[:, [2, 1]]
    chart_xy = local_centered @ axes

    # Full 3D gradient of the measured scalar field, locally fitted from
    # neighboring scores. lstsq yields the minimum-norm tangential gradient
    # when observations lie in a 2D plane (rank deficiency is expected).
    chart_tree = _eigen_field_tree(positions)
    fit_radius = (max(2.5*voxel_mm, 0.30*pca_radius_mm)
                  if gradient_radius_mm is None else gradient_radius_mm)
    def local_gradient(values):
        grad = np.full((len(positions), 3), np.nan)
        good = np.isfinite(values)
        for j, p in enumerate(positions):
            if not good[j]:
                continue
            nbr = _eigen_field_radius(chart_tree, p, fit_radius)
            nbr = nbr[good[nbr]]
            if len(nbr) < 8:
                continue
            x = positions[nbr] - p
            y = values[nbr] - values[j]
            # Drop the center/near duplicates; solve for the local slope.
            keep = np.linalg.norm(x, axis=1) > 1e-5
            if keep.sum() < 6:
                continue
            x, y = x[keep], y[keep]
            d = np.linalg.norm(x, axis=1)
            weight = 1. / np.maximum(d, 0.5*voxel_mm)
            slope, _, rank, _ = np.linalg.lstsq(x * weight[:, None],
                                               y * weight, rcond=1e-3)
            if rank >= 2 and np.linalg.norm(slope) > 1e-12:
                grad[j] = slope
        return grad
    grad_l1 = local_gradient(eigenvalues[:, 0])
    grad_ratio = local_gradient(ratio)

    # Discrete local minima for lambda1, maxima for lambda2/lambda1.
    # Suppress close duplicate extrema for legibility.
    minimum_radius = max(2.5*voxel_mm, 0.22*pca_radius_mm)
    def find_extrema(values, maximize=False):
        valid = np.flatnonzero(np.isfinite(values))
        minima = []
        for j in valid:
            # Exclude artificial extrema caused by the display ROI boundary.
            if np.linalg.norm(positions[j] - center) > view_radius_mm - minimum_radius:
                continue
            neighbors = _eigen_field_radius(chart_tree, positions[j], minimum_radius)
            nbr_values = values[neighbors]
            if (np.any(np.isfinite(nbr_values)) and
                    (values[j] >= np.nanmax(nbr_values) - 1e-9 if maximize
                     else values[j] <= np.nanmin(nbr_values) + 1e-9)):
                minima.append(j)
        minima.sort(key=lambda j: ((-values[j] if maximize else values[j]),
                                   np.linalg.norm(positions[j] - center)))
        kept = []
        for j in minima:
            if all(np.linalg.norm(positions[j] - positions[k]) > minimum_radius
                   for k in kept):
                kept.append(j)
            if len(kept) >= 12:
                break
        return np.asarray(kept, dtype=np.int64)
    minima_l1 = find_extrema(eigenvalues[:, 0])
    maxima_ratio = find_extrema(ratio, maximize=True)
    center_idx = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
    return EigenFieldResult(positions, eigenvalues, eigenvectors, ratio, grad_l1, grad_ratio,
                            minima_l1, maxima_ratio, center, axes, chart_xy,
                            center_idx, float(pca_radius_mm), float(ced_radius_mm),
                            float(view_radius_mm), mode, total_candidates, n_effective)


@dataclass(frozen=True)
class EigenRefineChoice:
    index: int
    start_index: int
    method: str
    evaluated_positions: int
    shift_mm: float
    before: float
    after: float
    boundary_hit: bool


def select_eigen_global_extremum(field: EigenFieldResult,
                                 start_mm: np.ndarray,
                                 search_radius_mm: float = 15.0,
                                 metric: str = 'lambda1',
                                 min_support_ratio: float = 0.80
                                 ) -> EigenRefineChoice:
    """Select ONE global optimum among ALL measured voxels in a fixed sphere.

    metric='lambda1': minimize first weighted-PCA eigenvalue.
    metric='ratio'  : maximize lambda2/lambda1.

    No joint score, smaller extremum neighborhood, gradient steps, or
    hill-climbing. Ties choose the nearest position to the fixed origin.
    Reject points with insufficient effective weighted support. Search
    radius is ALWAYS relative to the initial eigen-refinement origin.
    """
    if metric not in ('lambda1', 'ratio'):
        raise ValueError(f'Unknown eigen refinement metric: {metric}')
    if search_radius_mm <= 0 or not 0 < min_support_ratio <= 1:
        raise ValueError('Invalid search radius or support ratio')
    pos = field.positions_mm
    origin = np.asarray(start_mm, dtype=np.float64).reshape(3)
    distances = np.linalg.norm(pos - origin[None, :], axis=1)
    start = int(np.argmin(distances))
    effective = field.effective_counts
    initial_effective = float(effective[start])
    if not np.isfinite(initial_effective) or initial_effective <= 0:
        raise ValueError('Initial PCA sample invalid: adjust PCA mode or support radius')
    scores = (field.eigenvalues_mm2[:, 0] if metric == 'lambda1'
              else field.lambda2_over_lambda1)
    valid = (np.isfinite(scores) & np.isfinite(effective) &
             (effective >= max(8.0, min_support_ratio*initial_effective)) &
             (distances <= search_radius_mm + 1e-8))
    if not valid[start]:
        raise ValueError('Initial eigenvalue metric is invalid')
    candidate_ids = np.flatnonzero(valid)
    if not len(candidate_ids):
        raise ValueError('No reliable measured points within the search radius')
    if metric == 'lambda1':
        extreme_value = np.min(scores[candidate_ids])
    else:
        extreme_value = np.max(scores[candidate_ids])
    ties = candidate_ids[np.isclose(scores[candidate_ids], extreme_value,
                                    rtol=1e-9, atol=1e-10)]
    best = int(min(ties, key=lambda i: (distances[i], int(i))))
    shift = float(distances[best])
    return EigenRefineChoice(best, start, metric, int(len(candidate_ids)),
                             shift, float(scores[start]), float(scores[best]),
                             bool(shift >= 0.95*search_radius_mm))


def eigen_heatmap_values(field: EigenFieldResult, metric: str) -> np.ndarray:
    """Return native eigenvalue samples; no spatial interpolation is performed."""
    if metric == 'lambda1':
        return field.eigenvalues_mm2[:, 0]
    if metric == 'ratio':
        return field.lambda2_over_lambda1
    raise ValueError(f'Unknown eigen heatmap metric: {metric}')


def eigen_heatmap_colors(values: np.ndarray, *, limits=None) -> tuple[np.ndarray, tuple[float, float]]:
    """Viridis on robust percentiles or explicit limits; invalid samples grey."""
    from matplotlib import colormaps
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if not np.any(finite) and limits is None:
        raise ValueError('No finite heatmap values')
    low, high = np.percentile(v[finite], [2, 98]) if limits is None else limits
    if high <= low:
        high = low + max(1e-9, abs(low) * 1e-6)
    rgba = colormaps['viridis'](np.clip((v - low) / (high - low), 0, 1))
    rgb = np.rint(255 * rgba[:, :3]).astype(np.uint8)
    rgb[~finite] = [95, 103, 115]
    return rgb, (float(low), float(high))


def render_eigen_field_3d_image(field: EigenFieldResult,
                                width: int = 1450, height: int = 690) -> np.ndarray:
    """True 3D point-scatter heatmaps; no 2D PCA flattening or fabricated voxels.

    Two independent spatial fields: lambda1 and lambda2/lambda1. Both are
    sampled on observed voxel centroids. Arrows descend lambda1 and ascend
    lambda2/lambda1, and magenta outlines indicate corresponding extrema.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig = Figure(figsize=(width / 110, height / 110), dpi=110,
                 facecolor='#f8fafc')
    FigureCanvasAgg(fig)
    positions = field.positions_mm - field.center_mm[None, :]
    datasets = [
        ('lambda1', field.gradients_lambda1, field.minima_lambda1,
         'PCA first eigenvalue  (mm²)'),
        ('ratio', field.gradients_ratio, field.maxima_ratio,
         'PCA eigenvalue ratio  λ2/λ1'),
    ]
    for col, (metric, gradients, minima, title) in enumerate(datasets):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        values = eigen_heatmap_values(field, metric)
        valid = np.isfinite(values)
        if not np.any(valid):
            ax.set_title(title + ' (no data)')
            continue
        rgb, (vmin, vmax) = eigen_heatmap_colors(values)
        xyz = positions[valid]
        # Use explicit point colors so these match the Open3D 3D heatmap.
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=rgb[valid] / 255.0, s=10, alpha=.95, depthshade=False,
                   linewidths=0, rasterized=True)
        ax.scatter([0], [0], [0], color='#ff9c31', marker='*', s=125,
                   edgecolors='#111827', linewidths=.5, depthshade=False)
        if len(minima):
            m = positions[minima]
            ax.scatter(m[:,0], m[:,1], m[:,2], s=100, marker='o',
                       facecolors='none', edgecolors='#ef2d72', linewidths=1.8,
                       depthshade=False)
        valid_arrow = np.flatnonzero(np.isfinite(gradients).all(axis=1) & valid)
        if len(valid_arrow) > 38:
            valid_arrow = valid_arrow[np.linspace(0, len(valid_arrow) - 1, 38, dtype=int)]
        if len(valid_arrow):
            d = (-gradients[valid_arrow] if metric == "lambda1"
                 else gradients[valid_arrow])
            lengths = np.linalg.norm(d, axis=1)
            keep = lengths > 1e-11
            if np.any(keep):
                starts = positions[valid_arrow[keep]]
                directions = d[keep] / lengths[keep,None]
                ax.quiver(starts[:,0], starts[:,1], starts[:,2],
                          directions[:,0], directions[:,1], directions[:,2],
                          length=max(1.5, field.view_radius_mm * .13),
                          arrow_length_ratio=.23, normalize=False,
                          linewidth=.7, color='#e03427', alpha=.72)
        # Consistent isotropic axes: a 3D diagonal is not misread as a corner.
        extent = max(1., field.view_radius_mm * .86)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1,1,1), zoom=.86)
        ax.set_xlabel('Δx (mm)', fontsize=8)
        ax.set_ylabel('Δy (mm)', fontsize=8)
        ax.set_zlabel('Δz (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=23, azim=-60)
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap='viridis'),
                          ax=ax, fraction=.037, pad=.015, shrink=.65)
        cb.ax.tick_params(labelsize=7)
        cb.set_label('high = yellow / low = purple', fontsize=7)
    weighted = {'unweighted':'Unweighted PCA',
                'ced_weighted':'Raw CED-weighted PCA',
                'ced_threshold':'Raw CED-weighted PCA + 0.10 cutoff'}[field.mode]
    fig.suptitle(f'3D sampled eigenvalue heatmaps  |  {weighted}  |  '
                 f'PCA R={field.pca_radius_mm:g} mm  |  '
                 f'CED R={field.ced_radius_mm:g} mm', fontsize=12)
    fig.text(.5, .025, 'Orange star = query  •  Magenta rings = lambda1 MIN / ratio MAX  •  Arrows: descend lambda1 / ascend ratio',
             ha='center', fontsize=9, color='#334155')
    fig.subplots_adjust(left=.01, right=.96, top=.87, bottom=.07, wspace=.07)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return np.ascontiguousarray(rgba[:,:,:3])


def render_eigen_field_image(field: EigenFieldResult, width: int = 1380,
                             height: int = 640) -> np.ndarray:
    """Render 2D PCA projections + projected objective gradients.

    Uses Matplotlib Agg only on demand; no change to the Open3D renderer.
    The plotted field is made of sampled points, NOT a fitted continuous
    surface or a proof of convergence to a geometrical corner.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    fig = Figure(figsize=(width/110, height/110), dpi=110, facecolor='#fafafa')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    datasets = [
        (field.eigenvalues_mm2[:, 0], field.gradients_lambda1,
         field.minima_lambda1, 'lambda1 (mm^2) (MIN)'),
        (field.lambda2_over_lambda1, field.gradients_ratio,
         field.maxima_ratio, 'lambda2 / lambda1 (MAX)'),
    ]
    for ax, (values, gradients, minima, title) in zip(axes, datasets):
        finite = np.isfinite(values)
        xy = field.chart_xy_mm
        if not finite.any():
            ax.set_title(title + ' (no valid data)')
            continue
        top = float(np.percentile(values[finite], 98))
        low = float(np.percentile(values[finite], 2))
        if top <= low:
            top = low + 1e-9
        # Sampling rather than interpolation avoids filling missing geometry.
        dots = ax.scatter(xy[finite, 0], xy[finite, 1], c=values[finite],
                          s=18, cmap='viridis', vmin=low, vmax=top,
                          linewidths=0.0, alpha=.93, rasterized=True)
        ax.scatter([0], [0], marker='*', s=210, c='#f97316',
                   edgecolors='black', linewidths=.7, zorder=9,
                   label='Selected corner')
        if len(minima):
            ax.scatter(xy[minima, 0], xy[minima, 1], s=85,
                       facecolors='none', edgecolors='#ff355e', linewidths=2,
                       zorder=10, label='Local extrema')
        good = np.flatnonzero(np.isfinite(gradients).all(axis=1) & finite)
        # Uniform deterministic spatial thinning for arrows.
        if len(good) > 75:
            good = good[np.linspace(0, len(good)-1, 75, dtype=int)]
        if len(good):
            d3 = (-gradients[good] if title.startswith("lambda1")
                  else gradients[good])
            d2 = d3 @ field.axes
            magnitude = np.linalg.norm(d2, axis=1)
            keep = magnitude > 1e-11
            if np.any(keep):
                d2 = d2[keep] / magnitude[keep, None] * max(2.0, field.view_radius_mm*.12)
                pts = xy[good[keep]]
                ax.quiver(pts[:, 0], pts[:, 1], d2[:, 0], d2[:, 1],
                          angles='xy', scale_units='xy', scale=1, width=.0035,
                          color='#d62828', alpha=.8, zorder=7)
        fig.colorbar(dots, ax=ax, shrink=.8, pad=.025)
        ax.set_xlim(-field.view_radius_mm*1.04, field.view_radius_mm*1.04)
        ax.set_ylim(-field.view_radius_mm*1.04, field.view_radius_mm*1.04)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.12)
        ax.set_title(title + '  |  arrows toward target extremum', fontsize=10)
        ax.set_xlabel('PCA projected axis 1 (mm)')
        ax.set_ylabel('PCA projected axis 2 (mm)')
        ax.legend(loc='upper left', fontsize=8)
    weighted = {'unweighted':'Unweighted PCA',
                'ced_weighted':'Raw CED-weighted PCA',
                'ced_threshold':'CED-weighted PCA + 0.10 cutoff'}[field.mode]
    fig.suptitle(f'{weighted}  |  PCA R={field.pca_radius_mm:g} mm'
                 f'  |  CED R={field.ced_radius_mm:g} mm'
                 f'  |  samples={len(field.positions_mm)}/{field.total_candidates}',
                 fontsize=11)
    fig.text(.5, .012, '2D PCA projection of 3D field; finite-difference arrows are diagnostic, not confirmed corner directions.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .04, 1, .92])
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return np.ascontiguousarray(rgba[:, :, :3])


SCORE_VIEWS = (
    ("Fusion", "fusion_scores", "fusion_maxima", "MAX", ""),
    ("Bilateral CED", "ced_scores", "ced_maxima", "MAX", ""),
    ("CED-weighted PCA ratio", "ced_weighted_ratio", "ratio_maxima", "MAX", ""),
    ("CED-weighted lambda1", "ced_weighted_lambda1_mm2", "weighted_lambda1_minima", "MIN", " mm²"),
    ("Normal PCA lambda1", "normal_lambda1_mm2", "lambda1_minima", "MIN", " mm²"),
    ("CED x Local lambda1 minimum", "ced_local_min_scores", "ced_local_min_maxima", "MAX", ""),
    ("Local lambda1 minimum", "local_minimum_scores", None, "HEATMAP", ""),
    ("CED x Median lambda1 minimum", "ced_median_min_scores", "ced_median_min_maxima", "MAX", ""),
    ("Median lambda1 minimum", "median_minimum_scores", None, "HEATMAP", ""),
)


class CornerLab:
    def __init__(self, startup_path: str | None, demo: bool, camera: bool, *, roi_only: bool = False):
        self.app = gui.Application.instance
        self.win = self.app.create_window("Corner Lab | Heatmap + NMS | Corner inspection", 1650, 940)
        self.win.set_on_layout(self._layout)
        self.win.set_on_close(self._on_close)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.scene_widget.scene.set_background([0.075, 0.095, 0.13, 1.0])
        self.scene_widget.set_on_mouse(self._on_mouse)
        self.win.add_child(self.scene_widget)
        self.unit_widget = gui.SceneWidget()
        self.unit_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.unit_widget.scene.set_background([0.062, 0.083, 0.115, 1.0])
        self.hist_title = gui.Label("cos(theta) histogram | -1 to +1")
        self.hist_widget = gui.ImageWidget(o3d.geometry.Image(cosine_histogram_image(None)))
        self.hist_legend = gui.Label("Gold in | Pink out | Cyan Q10 | Green mean cos")
        self._unit_dynamic_names = []
        self._setup_unit_view()

        self.folder = None
        self.xyz = None
        self.rgb = None
        self.input_path = None
        self.dataset_source = False
        self.roi_only = bool(roi_only)
        self.input_raw_indices = None  # ROI tree rows -> full original raw IDs.
        self.hidden_corners = []       # Preserve labels outside the current ROI on save.
        self.kdtree = None                  # Original cloud for label picking ONLY
        self.analysis_xyz = None
        self.analysis_rgb = None
        self.analysis_tree = None          # Downsampled cloud for ALL calculations
        self.analysis_voxel_mm = 2.0
        self.corners = []
        self.active_index = None
        self.radius_mm = 30
        self.refine_range_mm = 6
        self.candidate_spacing_mm = 0
        self.refine_job_id = 0
        self.refine_running = False
        self.last_cone = None
        self.last_spherical_planes = None
        self.spherical_planes_enabled = True
        self.spherical_plane_k = 3
        self.spherical_plane_tolerance_deg = 15.0
        self.spherical_plane_iterations = 300
        self.spherical_plane_histogram_mode = 0
        # The same user-selected fixed radius is used throughout the pipeline.
        self.gap_link_deg = 20.0
        self.gap_edge_span_deg = 160.0
        self.gap_rejection_enabled = True
        self.gap_tail_enabled = True
        self.gap_tail_bin_deg = 10
        self.gap_tail_density_pct = 50
        self.last_gap_validation = None
        self.visible_overlay_names = set()
        self.eigen_field = None             # Last diagnostic result (not a label update)
        self.eigen_plot_image = None
        self.eigen_job_id = 0
        self.eigen_running = False
        self.eigen_mode = 'ced_threshold'
        self.eigen_ced_radius_mm = 10
        self.eigen_pca_radius_mm = 30
        self.eigen_overlay_mode = 0         # Off / -grad lambda1 / +grad lambda2/lambda1
        self.eigen_heatmap_mode = 1         # 0 off / 1 lambda1 / 2 lambda2/lambda1
        self.eigen_refine_metric = 'lambda1'
        self.eigen_refine_radius_mm = 15
        # Whole-scene lambda1 diagnostic is independent of local corner refinement.
        self.scene_l1_job_id = 0
        self.scene_l1_running = False
        self.scene_l1_show = False
        self.scene_l1_cache_key = None
        self.scene_l1_values = None
        self.scene_l1_colors = None
        self.scene_l1_limits = None
        # This experimental overlay does not alter the original PCA/label state.
        self.compare_job_id = 0
        self.compare_running = False
        self.compare_show = False
        self.compare_result = None
        self.compare_cache_key = None
        self.compare_heatmap_mode = 0    # Index into SCORE_VIEWS.
        self.compare_method_index = 0    # Fusion alone / CED+ratio / CED+weighted-lambda1 / CED+normal-lambda1
        self.compare_match_mm = 8
        self.compare_scope_index = 1  # 0: local ROI; 1: complete original scene (default)
        self.compare_scene_voxel_mm = 5.0
        self.compare_scene_nms_mm = 10.0
        self.compare_thresholds = {}

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.camera_thread = None
        self.camera_stop_event = threading.Event()
        self.camera_active = False
        self.last_camera_error = None

        self._setup_panel()
        # Open3D IDs depend on visible child order; optional views follow the tabs.
        for widget in (self.unit_widget, self.hist_title, self.hist_widget, self.hist_legend):
            self.win.add_child(widget)

        if demo:
            self._make_demo()
        elif startup_path:
            self._open_dataset(Path(startup_path))
        if camera:
            self._start_camera()

    def _setup_unit_view(self):
        self.unit_widget.scene.add_geometry(
            "unit_sphere_grid", sphere_graticule(), self._mat(shader="unlitLine", line_width=1.0))
        self.unit_widget.scene.add_geometry(
            "unit_axes", line_set(
                [[0., 0., 0.], [1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]],
                [(0, 1), (0, 2), (0, 3)], [0.5, 0.5, 0.5]),
            self._mat(shader="unlitLine", line_width=1.6))
        bounds = o3d.geometry.AxisAlignedBoundingBox([-1.3, -1.3, -1.3], [1.3, 1.3, 1.3])
        self.unit_widget.setup_camera(50.0, bounds, [0., 0., 0.])
        self.unit_widget.look_at([0., 0., 0.], [2.4, -2.4, 2.1], [0., 0., 1.])

    def _clear_unit_overlay(self):
        for name in self._unit_dynamic_names:
            self.unit_widget.scene.remove_geometry(name)
        self._unit_dynamic_names = []

    def _unit_overlay(self, name, geom, mat):
        self.unit_widget.scene.add_geometry(name, geom, mat)
        self._unit_dynamic_names.append(name)

    def _draw_spherical_analysis(self, cone: ConeAnalysis):
        """Unit sphere: exact angular classifications; optional point subsampling."""
        u = cone.directions
        mask = cone.inlier_mask
        if len(u) > MAX_UNIT_SPHERE_POINTS:
            keep = np.linspace(0, len(u) - 1, MAX_UNIT_SPHERE_POINTS, dtype=np.int64)
            u, mask = u[keep], mask[keep]
        cols = np.empty((len(u), 3), dtype=np.uint8)
        cols[mask] = [247, 204, 64]  # inside cone: gold
        cols[~mask] = [255, 70, 158]  # outside: magenta
        self._unit_overlay("dir_points", point_cloud(u * 1.018, cols),
                           self._mat(point_size=4.5))
        self._unit_overlay("mean_axis", line_set(
            [[0., 0., 0.], (cone.mean_axis * 1.20).tolist()], [(0, 1)],
            [0.12, 1.0, 0.29]), self._mat(shader="unlitLine", line_width=4.0))
        self._unit_overlay("mean_tip", sphere(cone.mean_axis * 1.20, .045,
                            [0.15, 1.0, 0.25]), self._mat(shader="defaultLit"))
        # Q10 cone is a purely visual overlay; hiding it must not affect
        # the direction samples, their mean, Q10 statistics or refinement.
        if self.show_q10_sphere_cone.checked:
            mesh, ring, spokes = cone_geometry(
                [0., 0., 0.], cone.mean_axis, cone.half_angle_deg)
            self._unit_overlay("cone_shell", mesh, self._mat(
                shader="defaultLitTransparency", color=[.08, .72, 1.0, .18]))
            self._unit_overlay("cone_ring", ring,
                               self._mat(shader="unlitLine", line_width=3))
            self._unit_overlay("cone_spokes", spokes,
                               self._mat(shader="unlitLine", line_width=1.5))
        self.unit_widget.force_redraw()

    def _draw_spherical_planes(self, analysis: SphericalPlaneAnalysis):
        """Colored great circles and PER-PLANE duplicated projection points."""
        t = np.linspace(0, 2*np.pi, 144, endpoint=False)
        for i, plane in enumerate(analysis.planes):
            rgb = np.asarray(SPHERE_PLANE_COLORS[i], dtype=np.uint8)
            color = (rgb.astype(float)/255.0).tolist()
            e1, e2 = _sphere_plane_basis(plane.normal)
            # Full zero-offset circle is the fitted spherical plane.
            circle = 1.11 * (np.cos(t)[:, None]*e1 + np.sin(t)[:, None]*e2)
            ring = line_set(circle, [(j, (j+1) % len(t)) for j in range(len(t))], color)
            self._unit_overlay(f'spherical_plane_circle_{i}', ring,
                               self._mat(shader='unlitLine', line_width=3.0))
            # Include the same point in EVERY matching plane. Slightly different
            # radii reveal multi-membership instead of overwriting the colors.
            proj = plane.projected_directions
            if len(proj) > MAX_UNIT_SPHERE_POINTS:
                ids = np.linspace(0, len(proj)-1, MAX_UNIT_SPHERE_POINTS, dtype=int)
                proj = proj[ids]
            if len(proj):
                offset = 1.13 + 0.025 * i
                cols = np.tile(rgb, (len(proj), 1))
                self._unit_overlay(f'spherical_plane_projected_{i}',
                                   point_cloud(proj * offset, cols),
                                   self._mat(point_size=5.8))
        self.unit_widget.force_redraw()

    def _setup_panel(self):
        em = self.win.theme.font_size
        self.panel_width = int(30 * em)
        self.header = gui.Vert(.35 * em, gui.Margins(em, .5 * em, em, .5 * em))
        self.header.add_child(gui.Label("CORNER LAB | Heatmap + NMS"))
        self.dataset_label = gui.Label("No dataset loaded")
        self.header.add_child(self.dataset_label)
        self.win.add_child(self.header)
        row2 = gui.Horiz(0.4 * em)
        demo_btn = gui.Button("Demo")
        demo_btn.set_on_clicked(self._make_demo)
        row2.add_child(demo_btn)
        open_btn = gui.Button("Open cloud / points.npz")
        open_btn.set_on_clicked(self._show_open_dialog)
        row2.add_child(open_btn)
        self.header.add_child(row2)
        self.roi_only_check = gui.Checkbox('Dataset: saved ROI only')
        self.roi_only_check.checked = self.roi_only
        self.roi_only_check.set_on_checked(self._roi_only_changed)
        self.header.add_child(self.roi_only_check)


        self.panel = gui.TabControl()
        self.heatmap_page = gui.ScrollableVert(.45 * em, gui.Margins(em, em, em, em))
        self.corner_page = gui.ScrollableVert(.45 * em, gui.Margins(em, em, em, em))
        self.pca_page = gui.ScrollableVert(.45 * em, gui.Margins(em, em, em, em))
        self.input_page = gui.ScrollableVert(.45 * em, gui.Margins(em, em, em, em))
        for name, page in (("Heatmap/NMS", self.heatmap_page), ("Corner", self.corner_page),
                           ("PCA", self.pca_page), ("Input", self.input_page)):
            self.panel.add_tab(name, page)
        self.panel.selected_tab_index = 0
        self.panel.set_on_selected_tab_changed(self._workspace_changed)
        self.win.add_child(self.panel)
        self.eigen_heatmap_legend = gui.Label("Heatmap: low purple / high yellow | NMS: cyan")
        self.status = gui.Label("Open a cloud or choose Demo, then compute heatmap + NMS.")
        self.win.add_child(self.eigen_heatmap_legend)
        self.win.add_child(self.status)

        self.heatmap_page.add_child(gui.Label("1. Scope and shared score radii"))
        self.compare_scope_combo = gui.Combobox()
        for name in ('Scope: selected corner (local)', 'Scope: loaded scene / saved ROI'):
            self.compare_scope_combo.add_item(name)
        self.compare_scope_combo.selected_index = 1
        self.compare_scope_combo.set_on_selection_changed(self._compare_scope_changed)
        self.heatmap_page.add_child(self.compare_scope_combo)

        self.heatmap_page.add_child(gui.Label('FULL SCENE voxel size (mm), e.g. 5'))
        self.compare_scene_voxel_edit = gui.TextEdit()
        self.compare_scene_voxel_edit.text_value = '5.0'
        self.heatmap_page.add_child(self.compare_scene_voxel_edit)
        self.heatmap_page.add_child(gui.Label('FULL SCENE NMS radius (mm), e.g. 10'))
        self.compare_scene_nms_edit = gui.TextEdit()
        self.compare_scene_nms_edit.text_value = '10.0'
        self.heatmap_page.add_child(self.compare_scene_nms_edit)
        self.heatmap_page.add_child(gui.Label('Scene: uses the loaded region; local voxel unchanged.'))
        self.eigen_ced_label = gui.Label("CED radius: 10 mm")
        self.heatmap_page.add_child(self.eigen_ced_label)
        self.eigen_ced_slider = gui.Slider(gui.Slider.INT)
        self.eigen_ced_slider.set_limits(3, 60)
        self.eigen_ced_slider.int_value = 10
        self.eigen_ced_slider.set_on_value_changed(self._eigen_ced_radius_changed)
        self.heatmap_page.add_child(self.eigen_ced_slider)

        self.eigen_pca_label = gui.Label("PCA support radius: 30 mm")
        self.heatmap_page.add_child(self.eigen_pca_label)
        self.eigen_pca_slider = gui.Slider(gui.Slider.INT)
        self.eigen_pca_slider.set_limits(3, 120)
        self.eigen_pca_slider.int_value = 30
        self.eigen_pca_slider.set_on_value_changed(self._eigen_pca_radius_changed)
        self.heatmap_page.add_child(self.eigen_pca_slider)


        self.heatmap_page.add_child(gui.Label("2. Criterion: heatmap + its own NMS"))
        self.compare_display_combo = gui.Combobox()
        for title, _, _, extreme, _ in SCORE_VIEWS:
            self.compare_display_combo.add_item(f"{title} | {extreme}")
        self.compare_display_combo.selected_index = 0
        self.compare_display_combo.set_on_selection_changed(self._compare_display_changed)
        self.heatmap_page.add_child(self.compare_display_combo)
        self.heatmap_page.add_child(gui.Label('Min-Max contrast tau (>0; R = PCA radius)'))
        self.compare_contrast_tau_edit = gui.TextEdit()
        self.compare_contrast_tau_edit.text_value = '0.005'
        self.heatmap_page.add_child(self.compare_contrast_tau_edit)
        self.heatmap_page.add_child(gui.Label('Median contrast tau (>0; R = PCA radius)'))
        self.compare_median_tau_edit = gui.TextEdit()
        self.compare_median_tau_edit.text_value = '0.005'
        self.heatmap_page.add_child(self.compare_median_tau_edit)
        self.heatmap_page.add_child(gui.Label('Both use Normal PCA lambda1; recompute after tau changes.'))
        self.compare_heatmap_check = gui.Checkbox("Show heatmap")
        self.compare_heatmap_check.checked = True
        self.compare_heatmap_check.set_on_checked(self._compare_layers_changed)
        self.compare_nms_check = gui.Checkbox("Show NMS points")
        self.compare_nms_check.checked = True
        self.compare_nms_check.set_on_checked(self._compare_layers_changed)
        layers = gui.Horiz(.4 * em)
        layers.add_child(self.compare_heatmap_check)
        layers.add_child(self.compare_nms_check)
        self.heatmap_page.add_child(layers)
        self.compare_auto_contrast_check = gui.Checkbox('Auto Contrast')
        self.compare_auto_contrast_check.checked = False
        self.compare_auto_contrast_check.enabled = False  # L or CED*L only.
        self.compare_auto_contrast_check.set_on_checked(self._compare_auto_contrast_changed)
        self.heatmap_page.add_child(self.compare_auto_contrast_check)
        self.compare_btn = gui.Button(self._compare_button_label())
        self.compare_btn.set_on_clicked(self._toggle_compare_extrema)
        self.heatmap_page.add_child(self.compare_btn)
        self.heatmap_page.add_child(gui.Label("Compute once; switching criteria reuses the result."))
        self.compare_summary = gui.Label("Choose scope and criterion, then compute.\nCtrl+click a point to open Corner inspection.")
        self.heatmap_page.add_child(self.compare_summary)
        thresholds = gui.CollapsableVert('NMS score thresholds (blank = OFF)', .3 * em)
        thresholds.set_is_open(False)
        self.compare_threshold_edits = {}
        for title, value_key, _, extreme, units in SCORE_VIEWS:
            if extreme == 'HEATMAP':
                continue
            sign = '>=' if extreme == 'MAX' else '<='
            thresholds.add_child(gui.Label(f'{title}: score {sign} threshold{units}'))
            edit = gui.TextEdit()
            edit.text_value = ''
            self.compare_threshold_edits[value_key] = edit
            thresholds.add_child(edit)
        apply_thresholds = gui.Button('Apply NMS thresholds')
        apply_thresholds.set_on_clicked(self._compare_thresholds_changed)
        thresholds.add_child(apply_thresholds)
        self.heatmap_page.add_child(thresholds)
        pairs = gui.CollapsableVert("Optional: pair two criteria", .4 * em)
        pairs.set_is_open(False)
        self.compare_pairs_check = gui.Checkbox("Show paired-method comparison")
        self.compare_pairs_check.checked = False
        self.compare_pairs_check.set_on_checked(self._compare_layers_changed)
        pairs.add_child(self.compare_pairs_check)
        self.compare_method_combo = gui.Combobox()
        for name in ("1. Fusion MAX NMS",
                     "2. Bilateral CED MAX + CED-PCA ratio MAX",
                     "3. Bilateral CED MAX + CED-PCA lambda1 MIN",
                     "4. Bilateral CED MAX + Normal-PCA lambda1 MIN"):
            self.compare_method_combo.add_item(name)
        self.compare_method_combo.selected_index = 0
        self.compare_method_combo.set_on_selection_changed(self._compare_method_changed)
        pairs.add_child(self.compare_method_combo)
        self.compare_match_label = gui.Label("Extrema matching distance: 8 mm")
        pairs.add_child(self.compare_match_label)
        self.compare_match_slider = gui.Slider(gui.Slider.INT)
        self.compare_match_slider.set_limits(2, 30)
        self.compare_match_slider.int_value = 8
        self.compare_match_slider.set_on_value_changed(self._compare_match_changed)
        pairs.add_child(self.compare_match_slider)
        pairs.add_child(gui.Label("Cyan: primary | Pink: secondary\nLarge lime marker + thick line: matched pair"))
        self.heatmap_page.add_child(pairs)

        self.corner_page.add_child(gui.Label("Ctrl+click the cloud to label or inspect a corner."))
        self.corner_page.add_child(gui.Label("Click mode"))
        self.mode = gui.Combobox()
        self.mode.add_item("Label new corner")
        self.mode.add_item("Inspect saved corner")
        self.mode.selected_index = 0
        self.corner_page.add_child(self.mode)

        self.corner_page.add_child(gui.Label("Saved corners"))
        self.corner_combo = gui.Combobox()
        self.corner_combo.add_item("(none)")
        self.corner_combo.set_on_selection_changed(self._corner_selection_changed)
        self.corner_page.add_child(self.corner_combo)

        label_row = gui.Horiz(0.4 * em)
        save_btn = gui.Button("Save labels")
        save_btn.set_on_clicked(self._save_labels)
        label_row.add_child(save_btn)
        delete_btn = gui.Button("Delete selected")
        delete_btn.set_on_clicked(self._delete_selected)
        label_row.add_child(delete_btn)
        self.corner_page.add_child(label_row)

        self.radius_label = gui.Label("Neighborhood radius: 30 mm")
        self.corner_page.add_child(self.radius_label)
        self.radius = gui.Slider(gui.Slider.INT)
        self.radius.set_limits(2, 250)
        self.radius.int_value = 30
        self.radius.set_on_value_changed(self._radius_changed)
        self.corner_page.add_child(self.radius)

        self.voxel_label = gui.Label("Analysis voxel downsampling: 2 mm (shared)")
        self.corner_page.add_child(self.voxel_label)
        self.voxel_slider = gui.Slider(gui.Slider.INT)
        self.voxel_slider.set_limits(1, 10)
        self.voxel_slider.int_value = 2
        self.voxel_slider.set_on_value_changed(self._analysis_voxel_changed)
        self.corner_page.add_child(self.voxel_slider)
        q10_controls = gui.CollapsableVert("Q10 refinement", .4 * em)
        q10_controls.set_is_open(False)
        self.corner_page.add_child(q10_controls)
        q10_controls.add_child(gui.Label("Iterative Q10 refinement (downsampled points)"))
        self.search_range_label = gui.Label("Search range: 6 mm")
        q10_controls.add_child(self.search_range_label)
        self.search_range = gui.Slider(gui.Slider.INT)
        self.search_range.set_limits(2, 12)
        self.search_range.int_value = 6
        self.search_range.set_on_value_changed(self._refine_range_changed)
        q10_controls.add_child(self.search_range)
        self.spacing_label = gui.Label("Candidate spacing: 0 mm (ALL downsampled points)")
        q10_controls.add_child(self.spacing_label)
        self.spacing_slider = gui.Slider(gui.Slider.INT)
        self.spacing_slider.set_limits(0, 5)
        self.spacing_slider.int_value = 0
        self.spacing_slider.set_on_value_changed(self._candidate_spacing_changed)
        q10_controls.add_child(self.spacing_slider)
        refine_row = gui.Horiz(0.4 * em)
        self.refine_btn = gui.Button("Refine Q10")
        self.refine_btn.set_on_clicked(self._refine_selected)
        refine_row.add_child(self.refine_btn)
        reset_btn = gui.Button("Reset refined")
        reset_btn.set_on_clicked(self._reset_refinement)
        refine_row.add_child(reset_btn)
        q10_controls.add_child(refine_row)
        self.effective_radius_label = gui.Label("Analysis radius: 30 mm (fixed for all stages)")
        q10_controls.add_child(self.effective_radius_label)
        q10_controls.add_child(gui.Label("RED original | LIME latest | WHITE step path"))
        q10_controls.add_child(gui.Label("Each Refine click starts at previous result"))
        q10_controls.add_child(gui.Label("Q10 ↑ means cone half-angle ↓"))
        q10_controls.add_child(gui.Label("Caution: Q10 alone may shift off the corner"))

        plane_controls = gui.CollapsableVert("Spherical planes", .4 * em)
        plane_controls.set_is_open(False)
        self.corner_page.add_child(plane_controls)
        plane_controls.add_child(gui.Label("SPHERICAL PLANES: through origin (0,0,0)"))
        self.sphere_plane_show = gui.Checkbox("Fit great circles + projected inlier dots")
        self.sphere_plane_show.checked = True
        self.sphere_plane_show.set_on_checked(self._sphere_plane_enabled_changed)
        plane_controls.add_child(self.sphere_plane_show)
        plane_controls.add_child(gui.Label("Auto K: up to 3 planes (stops if new support is insufficient)"))
        self.sphere_plane_tolerance_label = gui.Label("Angular inlier tolerance: 15 deg")
        plane_controls.add_child(self.sphere_plane_tolerance_label)
        self.sphere_plane_tolerance_slider = gui.Slider(gui.Slider.INT)
        self.sphere_plane_tolerance_slider.set_limits(2, 25)
        self.sphere_plane_tolerance_slider.int_value = 15
        self.sphere_plane_tolerance_slider.set_on_value_changed(self._sphere_plane_tolerance_changed)
        plane_controls.add_child(self.sphere_plane_tolerance_slider)
        plane_controls.add_child(gui.Label("Histogram under sphere"))
        self.sphere_plane_hist_combo = gui.Combobox()
        self.sphere_plane_hist_combo.add_item("Per-plane circular azimuth (rose plot)")
        self.sphere_plane_hist_combo.add_item("Original Q10 cosine histogram")
        self.sphere_plane_hist_combo.selected_index = 0
        self.sphere_plane_hist_combo.set_on_selection_changed(self._sphere_plane_hist_changed)
        plane_controls.add_child(self.sphere_plane_hist_combo)
        self.sphere_plane_summary = gui.Label("Fit is pending (select a corner)")
        plane_controls.add_child(self.sphere_plane_summary)
        plane_controls.add_child(gui.Label("Orange=P1 | Green=P2 | Violet=P3"))
        plane_controls.add_child(gui.Label("Overlapping inliers appear in each plane"))

        gap_controls = gui.CollapsableVert("Bidirectionality / MaxGap", .4 * em)
        self.corner_page.add_child(gap_controls)
        gap_controls.add_child(gui.Label("MAX ANGULAR GAP | PCL-style + outlier removal"))
        gap_controls.add_child(gui.Label("Split angular groups at gaps above link threshold"))
        gap_controls.add_child(gui.Label("Keep largest angular group; ignore isolated noise"))
        self.gap_link_label = gui.Label("Angular connection threshold: 20 deg")
        gap_controls.add_child(self.gap_link_label)
        self.gap_link_slider = gui.Slider(gui.Slider.INT)
        self.gap_link_slider.set_limits(5, 60)
        self.gap_link_slider.int_value = 20
        self.gap_link_slider.set_on_value_changed(self._gap_link_changed)
        gap_controls.add_child(self.gap_link_slider)
        self.gap_tail_checkbox = gui.Checkbox("Trim only very sparse END bins")
        self.gap_tail_checkbox.checked = True
        self.gap_tail_checkbox.set_on_checked(self._gap_tail_enabled_changed)
        gap_controls.add_child(self.gap_tail_checkbox)
        self.gap_tail_bin_label = gui.Label("End-bin angular width: 10 deg")
        gap_controls.add_child(self.gap_tail_bin_label)
        self.gap_tail_bin_slider = gui.Slider(gui.Slider.INT)
        self.gap_tail_bin_slider.set_limits(5, 25)
        self.gap_tail_bin_slider.int_value = 10
        self.gap_tail_bin_slider.set_on_value_changed(self._gap_tail_bin_changed)
        gap_controls.add_child(self.gap_tail_bin_slider)
        self.gap_tail_density_label = gui.Label("End-bin density: 50% of nonempty-bin median (min 2 pts)")
        gap_controls.add_child(self.gap_tail_density_label)
        self.gap_tail_density_slider = gui.Slider(gui.Slider.INT)
        self.gap_tail_density_slider.set_limits(0, 50)
        self.gap_tail_density_slider.int_value = 50
        self.gap_tail_density_slider.set_on_value_changed(self._gap_tail_density_changed)
        gap_controls.add_child(self.gap_tail_density_slider)
        self.gap_rejection_checkbox = gui.Checkbox("Use cleaned span to REJECT Edge (experimental)")
        self.gap_rejection_checkbox.checked = True
        self.gap_rejection_checkbox.set_on_checked(self._gap_rejection_changed)
        gap_controls.add_child(self.gap_rejection_checkbox)
        self.gap_edge_span_label = gui.Label("Edge-like span threshold: 160 deg")
        gap_controls.add_child(self.gap_edge_span_label)
        self.gap_edge_span_slider = gui.Slider(gui.Slider.INT)
        self.gap_edge_span_slider.set_limits(100, 220)
        self.gap_edge_span_slider.int_value = 160
        self.gap_edge_span_slider.set_on_value_changed(self._gap_edge_span_changed)
        gap_controls.add_child(self.gap_edge_span_slider)
        self.gap_summary = gui.Label("Gap analysis pending (cyan kept arc; magenta noise)")
        gap_controls.add_child(self.gap_summary)
        gap_controls.add_child(gui.Label("PCL boundary: max angular gap > 90 deg (NOT corner/edge)"))
        gap_controls.add_child(gui.Label("Experiment: edge-like if cleaned span >= threshold"))

        overlay_controls = gui.CollapsableVert("Inspection overlays", .4 * em)
        overlay_controls.set_is_open(False)
        self.corner_page.add_child(overlay_controls)
        overlay_controls.add_child(gui.Label("yellow: neighbors | red: stored labels"))
        overlay_controls.add_child(gui.Label("Active: BLUE corner-like | YELLOW edge-like reject"))
        overlay_controls.add_child(gui.Label("green: centroid offset | blue: centroid"))
        overlay_controls.add_child(gui.Label("cyan disk: perpendicular plane"))
        self.show_plane = gui.Checkbox("Show centroid vector + disk")
        self.show_plane.checked = True
        self.show_plane.set_on_checked(lambda _value: self._refresh_analysis())
        overlay_controls.add_child(self.show_plane)
        self.show_q10_sphere_cone = gui.Checkbox("Show blue Q10 cone on unit sphere")
        self.show_q10_sphere_cone.checked = True
        self.show_q10_sphere_cone.set_on_checked(lambda _value: self._refresh_analysis())
        overlay_controls.add_child(self.show_q10_sphere_cone)
        self.show_cone = gui.Checkbox("Show Q10 cone in 3D cloud")
        self.show_cone.checked = True
        self.show_cone.set_on_checked(lambda _value: self._refresh_analysis())
        overlay_controls.add_child(self.show_cone)
        overlay_controls.add_child(gui.Label("Unit sphere: gold=in 90% | magenta=out"))
        overlay_controls.add_child(gui.Label("Sphere green=mean | cyan=cone rim"))
        overlay_controls.add_child(gui.Label("Cloud purple=direction mean | green=centroid"))
        details = gui.CollapsableVert("Selected corner details", .4 * em)
        details.set_is_open(False)
        self.info = gui.Label("Select a corner to inspect cached results.")
        details.add_child(self.info)
        self.corner_page.add_child(details)

        self.pca_page.add_child(gui.Label("Optional PCA position refinement.\nUses CED/PCA radii from Heatmap/NMS."))
        self.pca_page.add_child(gui.Label("PCA REFINEMENT | same parameters for both modes"))
        self.pca_page.add_child(gui.Label("PCA mode"))
        self.eigen_mode_combo = gui.Combobox()
        for name in ("CED-weighted PCA (cutoff 0.10)", "Normal PCA (no CED)"):
            self.eigen_mode_combo.add_item(name)
        self.eigen_mode_combo.selected_index = 0
        self.eigen_mode_combo.set_on_selection_changed(self._eigen_mode_changed)
        self.pca_page.add_child(self.eigen_mode_combo)

        self.eigen_refine_range_label = gui.Label("Refine search radius: 15 mm")
        self.pca_page.add_child(self.eigen_refine_range_label)
        self.eigen_refine_range_slider = gui.Slider(gui.Slider.INT)
        self.eigen_refine_range_slider.set_limits(2, 45)
        self.eigen_refine_range_slider.int_value = 15
        self.eigen_refine_range_slider.set_on_value_changed(self._eigen_refine_radius_changed)
        self.pca_page.add_child(self.eigen_refine_range_slider)

        self.pca_page.add_child(gui.Label("Refine objective (choose one; no joint score)"))
        self.eigen_refine_combo = gui.Combobox()
        for title in ("lambda1: MIN", "lambda2/lambda1: MAX"):
            self.eigen_refine_combo.add_item(title)
        self.eigen_refine_combo.selected_index = 0
        self.eigen_refine_combo.set_on_selection_changed(self._eigen_refine_metric_changed)
        self.pca_page.add_child(self.eigen_refine_combo)
        self.eigen_refine_btn = gui.Button("Refine PCA")
        self.eigen_refine_btn.set_on_clicked(self._refine_eigen_selected)
        self.pca_page.add_child(self.eigen_refine_btn)
        self.scene_l1_btn = gui.Button("Show FULL SCENE lambda1 heatmap")
        self.scene_l1_btn.set_on_clicked(self._toggle_scene_lambda1)
        self.pca_page.add_child(self.scene_l1_btn)
        self.scene_l1_status = gui.Label("Full scene uses ALL analysis voxel points; no corner needed")
        self.pca_page.add_child(self.scene_l1_status)


        self.pca_page.add_child(gui.Label("3D heatmap (updates automatically on Refine)"))
        self.eigen_heatmap_combo = gui.Combobox()
        for name in ("Heatmap: off (RGB)", "Heatmap: lambda1", "Heatmap: lambda2/lambda1"):
            self.eigen_heatmap_combo.add_item(name)
        self.eigen_heatmap_combo.selected_index = 1
        self.eigen_heatmap_combo.set_on_selection_changed(self._eigen_heatmap_changed)
        self.pca_page.add_child(self.eigen_heatmap_combo)
        self.eigen_overlay_combo = gui.Combobox()
        for name in ("3D arrows: off", "3D arrows: -grad lambda1",
                     "3D arrows: +grad lambda2/lambda1"):
            self.eigen_overlay_combo.add_item(name)
        self.eigen_overlay_combo.selected_index = 0
        self.eigen_overlay_combo.set_on_selection_changed(self._eigen_overlay_changed)
        self.pca_page.add_child(self.eigen_overlay_combo)
        self.show_eigen_axes = gui.Checkbox("Show refined eigenvectors v1 / v2 (3D)")
        self.show_eigen_axes.checked = True
        self.show_eigen_axes.set_on_checked(lambda _: self._refresh_analysis())
        self.pca_page.add_child(self.show_eigen_axes)
        self.pca_page.add_child(gui.Label("v1: orange | v2: cyan"))
        self.eigen_refine_status = gui.Label("Eigen refinement not run")
        self.pca_page.add_child(self.eigen_refine_status)
        self.eigen_summary = gui.Label("Run Refine PCA to compute the eigen-field")
        self.pca_page.add_child(self.eigen_summary)


        demo_controls = gui.CollapsableVert('Demo surface', .4 * em)
        self.input_page.add_child(demo_controls)
        self.demo_fillet_label = gui.Label('Corner / edge fillet radius: 0 mm (sharp)')
        demo_controls.add_child(self.demo_fillet_label)
        self.demo_fillet_slider = gui.Slider(gui.Slider.DOUBLE)
        self.demo_fillet_slider.set_limits(0., 85.)
        self.demo_fillet_slider.double_value = 0.
        demo_controls.add_child(self.demo_fillet_slider)
        demo_controls.add_child(gui.Label('One radius rounds all 3 edges and their corner.\nWhite point follows the rounded surface (reference only).'))
        self.demo_noise_label = gui.Label('Gaussian normal noise sigma: 0.25 mm')
        demo_controls.add_child(self.demo_noise_label)
        self.demo_noise_slider = gui.Slider(gui.Slider.DOUBLE)
        self.demo_noise_slider.set_limits(0., 5.)
        self.demo_noise_slider.double_value = .25
        demo_controls.add_child(self.demo_noise_slider)
        self.demo_roughness_label = gui.Label('Surface bumps amplitude: 0 mm')
        demo_controls.add_child(self.demo_roughness_label)
        self.demo_roughness_slider = gui.Slider(gui.Slider.DOUBLE)
        self.demo_roughness_slider.set_limits(0., 10.)
        self.demo_roughness_slider.double_value = 0.
        demo_controls.add_child(self.demo_roughness_slider)
        self.demo_wavelength_label = gui.Label('Surface bumps wavelength: 20 mm')
        demo_controls.add_child(self.demo_wavelength_label)
        self.demo_wavelength_slider = gui.Slider(gui.Slider.DOUBLE)
        self.demo_wavelength_slider.set_limits(2., 100.)
        self.demo_wavelength_slider.double_value = 20.
        demo_controls.add_child(self.demo_wavelength_slider)
        for slider in (self.demo_fillet_slider, self.demo_noise_slider,
                       self.demo_roughness_slider, self.demo_wavelength_slider):
            slider.set_on_value_changed(self._demo_surface_changed)
        demo_controls.add_child(gui.Label('Demo seed (same seed keeps the same sampling)'))
        self.demo_seed_edit = gui.TextEdit()
        self.demo_seed_edit.text_value = '7'
        demo_controls.add_child(self.demo_seed_edit)
        apply_demo = gui.Button('Generate / apply Demo')
        apply_demo.set_on_clicked(self._make_demo)
        demo_controls.add_child(apply_demo)
        demo_controls.add_child(gui.Label('Changes apply on Generate or the top Demo button.\nModified demos save separately; camera captures are unaffected.'))

        self.input_page.add_child(gui.Label("Camera preview"))
        self.preview = gui.ImageWidget(o3d.geometry.Image(np.zeros((240, 320, 3), dtype=np.uint8)))
        self.input_page.add_child(self.preview)

        row = gui.Horiz(0.4 * em)
        self.camera_btn = gui.Button("Start camera")
        self.camera_btn.set_on_clicked(self._toggle_camera)
        row.add_child(self.camera_btn)
        self.capture_btn = gui.Button("CAPTURE")
        self.capture_btn.enabled = False
        self.capture_btn.set_on_clicked(self._capture)
        row.add_child(self.capture_btn)
        self.input_page.add_child(row)

        self.input_page.add_child(gui.Label("Ctrl+left click: label / inspect\nDrag: rotate | Wheel: zoom\nLabels keep original raw point indices."))
        self._workspace_changed(0)

    def _workspace_changed(self, index):
        # Re-selecting the native callback's tab queues another selection in Open3D.
        if self.panel.selected_tab_index != int(index):
            self.panel.selected_tab_index = int(index)
        inspecting = int(index) == 1
        for widget in (self.unit_widget, self.hist_title, self.hist_widget, self.hist_legend):
            widget.visible = inspecting
        for name in self.visible_overlay_names:
            if name == 'corner_markers' or name.startswith(('comparison_', 'eigen_', 'scene_lambda1_')):
                continue
            visible = int(index) == 2 if name.startswith('refined_eigen_') else inspecting
            self.scene_widget.scene.show_geometry(name, visible)
        self.win.set_needs_layout()
        self.scene_widget.force_redraw()

    def _layout(self, ctx):
        r = self.win.content_rect
        em = self.win.theme.font_size
        w = min(self.panel_width, max(240, int(r.width * .37)))
        area = max(300, r.width - w)
        header_h = int(10 * em)
        footer_h = int(3.5 * em)
        view_h = max(200, r.height - footer_h)
        self.header.frame = gui.Rect(r.get_right() - w, r.y, w, header_h)
        self.panel.frame = gui.Rect(r.get_right() - w, r.y + header_h, w, max(120, r.height - header_h))
        self.eigen_heatmap_legend.frame = gui.Rect(r.x + 8, r.y + view_h, area - 16, int(2 * em))
        self.status.frame = gui.Rect(r.x + 8, r.y + view_h + int(2 * em), area - 16, int(1.5 * em))
        if self.panel.selected_tab_index != 1:
            self.scene_widget.frame = gui.Rect(r.x, r.y, area, view_h)
            return
        mid = max(230, int(area * .32))
        if area - mid < 270:
            mid = max(130, area - 270)
        self.scene_widget.frame = gui.Rect(r.x, r.y, max(170, area - mid), view_h)
        hist_area = min(max(175, int(view_h * .34)), max(130, view_h - 200))
        label_h = max(22, int(1.4 * em))
        mid_x = r.x + area - mid
        self.unit_widget.frame = gui.Rect(mid_x, r.y, mid, view_h - hist_area)
        self.hist_title.frame = gui.Rect(mid_x + 6, r.y + view_h - hist_area, mid - 12, label_h)
        self.hist_widget.frame = gui.Rect(mid_x, r.y + view_h - hist_area + label_h,
                                          mid, hist_area - 2 * label_h)
        self.hist_legend.frame = gui.Rect(mid_x + 6, r.y + view_h - label_h, mid - 12, label_h)

    def _status(self, message):
        self.status.text = str(message)
        print(message)

    def _error(self, message):
        print("ERROR:", message)
        self._status("ERROR: " + str(message)[:160])
        try:
            self.win.show_message_box("Corner Lab", str(message))
        except Exception:
            pass

    def _on_close(self):
        self.camera_stop_event.set()
        return True

    def _mat(self, shader="defaultUnlit", point_size=None, color=None, line_width=None):
        m = rendering.MaterialRecord()
        m.shader = shader
        if point_size is not None:
            m.point_size = float(point_size) * self.win.scaling
        if color is not None:
            m.base_color = list(color)
        if line_width is not None:
            m.line_width = float(line_width) * self.win.scaling
        return m

    def _overlay(self, name, geom, mat):
        self._remove_overlay(name)
        self.scene_widget.scene.add_geometry(name, geom, mat)
        self.visible_overlay_names.add(name)

    def _remove_overlay(self, name):
        if name in self.visible_overlay_names:
            self.scene_widget.scene.remove_geometry(name)
            self.visible_overlay_names.discard(name)

    def _clear_overlays(self):
        for name in list(self.visible_overlay_names):
            self._remove_overlay(name)

    def _demo_surface_changed(self, _value):
        radius = self.demo_fillet_slider.double_value
        self.demo_fillet_label.text = f'Corner / edge fillet radius: {radius:.3g} mm' + (' (sharp)' if radius == 0. else '')
        self.demo_noise_label.text = f'Gaussian normal noise sigma: {self.demo_noise_slider.double_value:.3g} mm'
        self.demo_roughness_label.text = f'Surface bumps amplitude: {self.demo_roughness_slider.double_value:.3g} mm'
        self.demo_wavelength_label.text = f'Surface bumps wavelength: {self.demo_wavelength_slider.double_value:.3g} mm'

    def _make_demo(self):
        try:
            params = dict(noise_sigma_mm=float(self.demo_noise_slider.double_value),
                          roughness_amplitude_mm=float(self.demo_roughness_slider.double_value),
                          roughness_wavelength_mm=float(self.demo_wavelength_slider.double_value),
                          fillet_radius_mm=float(self.demo_fillet_slider.double_value),
                          seed=int(self.demo_seed_edit.text_value.strip()))
            xyz, rgb, true_idx = make_demo_cloud(**params)
            defaults = params == dict(noise_sigma_mm=.25, roughness_amplitude_mm=0.,
                                     roughness_wavelength_mm=20., fillet_radius_mm=0., seed=7)
            folder = CAPTURES / ('demo_corner' if defaults else
                                 'demo_corner_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            save_cloud(folder, xyz, rgb, {"mode": "synthetic", "units": "mm", **params})
            export_binary_ply(folder / "cloud.ply", xyz, rgb)
            # Preserve labels if user already annotated the demo.
            if not load_corners(folder, len(xyz)):
                save_corners(folder, [{"id": "corner_001", "point_index": int(true_idx)}])
            self._open_dataset(folder)
            self._status(f"Demo loaded: fillet={params['fillet_radius_mm']:g} mm | "
                         f"noise sigma={params['noise_sigma_mm']:g} mm | "
                         f"bumps={params['roughness_amplitude_mm']:g} mm / "
                         f"{params['roughness_wavelength_mm']:g} mm | seed={params['seed']}")
        except Exception as exc:
            self._error(exc)

    def _rebuild_analysis_cloud(self):
        self._invalidate_eigen_field()
        self._invalidate_scene_lambda1()
        """Create the one shared downsampled point cloud and search index."""
        if self.xyz is None:
            return
        self.analysis_xyz, self.analysis_rgb = voxel_downsample_analysis(
            self.xyz if self.input_raw_indices is None else self.xyz[self.input_raw_indices],
            self.rgb if self.input_raw_indices is None else self.rgb[self.input_raw_indices],
            self.analysis_voxel_mm)
        self.analysis_tree = o3d.geometry.KDTreeFlann(
            point_cloud(self.analysis_xyz, self.analysis_rgb))

    def _closest_analysis_point(self, coordinate):
        """Snap an old/original label to an actual downsampled centroid."""
        if self.analysis_tree is None:
            raise ValueError("No downsampled analysis cloud")
        n, ids, _ = self.analysis_tree.search_knn_vector_3d(
            np.asarray(coordinate, dtype=np.float64), 1)
        if n == 0:
            raise ValueError("No downsampled point near label")
        idx = int(ids[0])
        return idx, self.analysis_xyz[idx].astype(np.float64)

    def _open_dataset(self, path):
        try:
            self.refine_job_id += 1
            folder, xyz, rgb = load_cloud(path)
            dataset = (Path(path).name == "points.npz" or
                       Path(path).is_dir() and not (Path(path) / "cloud.npz").is_file())
            corners = load_corners(folder, len(xyz), dataset=dataset)
            indices = load_dataset_roi_indices(folder, len(xyz)) if dataset and self.roi_only else None
            hidden = []
            if indices is not None:
                selected = np.zeros(len(xyz), dtype=bool)
                selected[indices] = True
                hidden = [c for c in corners if not selected[c['point_index']]]
                corners = [c for c in corners if selected[c['point_index']]]
            self.folder, self.xyz, self.rgb = folder, xyz, rgb
            self.dataset_source = dataset
            self.input_path = folder / ('points.npz' if dataset else 'cloud.npz')
            self.input_raw_indices = indices
            self.hidden_corners = hidden
            self.corners = corners
            self.active_index = 0 if self.corners else None
            input_xyz, input_rgb = (xyz, rgb) if indices is None else (xyz[indices], rgb[indices])
            self.kdtree = o3d.geometry.KDTreeFlann(point_cloud(input_xyz, input_rgb))
            self._rebuild_analysis_cloud()
            self.scene_widget.scene.clear_geometry()
            self.visible_overlay_names.clear()
            # Downsample display only. Stored point indexes remain exact.
            if len(input_xyz) > MAX_VIEW_POINTS:
                ind = np.linspace(0, len(input_xyz) - 1, MAX_VIEW_POINTS, dtype=np.int64)
                show_xyz, show_rgb = input_xyz[ind], input_rgb[ind]
            else:
                show_xyz, show_rgb = input_xyz, input_rgb
            self.scene_widget.scene.add_geometry(
                "cloud", point_cloud(show_xyz, show_rgb), self._mat(point_size=3.0)
            )
            bounds = self.scene_widget.scene.bounding_box
            center = bounds.get_center()
            self.scene_widget.setup_camera(60.0, bounds, center)
            distance = max(200.0, float(np.linalg.norm(bounds.get_extent())) * 1.8)
            # RealSense coordinates: +z is forward, +y is down.
            self.scene_widget.look_at(center, center + np.array([0., 0., -distance]), [0., -1., 0.])
            self.dataset_label.text = (f"{folder.name} | {'ROI' if indices is not None else 'scene'} "
                                       f"{len(input_xyz):,}/{len(xyz):,} | "
                                       f"voxel {len(self.analysis_xyz):,} "
                                       f"({self.analysis_voxel_mm:g} mm)")
            self._refresh_corner_combo()
            self._refresh_analysis()
            self._status(f"Loaded: {folder / ('points.npz' if dataset else 'cloud.npz')} | analysis: "
                         f"{len(self.analysis_xyz):,} voxel-downsampled points")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _roi_only_changed(self, enabled):
        previous = self.roi_only
        self.roi_only = bool(enabled)
        if self.dataset_source and self.input_path is not None:
            if not self._open_dataset(self.input_path):
                self.roi_only = previous
                self.roi_only_check.checked = previous

    def _show_open_dialog(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "Open saved cloud.npz / points.npz", self.win.theme)
        dialog.add_filter(".npz", "Point cloud dataset (.npz)")
        dialog.set_on_cancel(self.win.close_dialog)
        def completed(path):
            self.win.close_dialog()
            self._open_dataset(Path(path))
        dialog.set_on_done(completed)
        self.win.show_dialog(dialog)

    def _refresh_corner_combo(self):
        self.corner_combo.clear_items()
        if not self.corners:
            self.corner_combo.add_item("(none)")
            self.active_index = None
        else:
            for item in self.corners:
                self.corner_combo.add_item(item["id"])
            if self.active_index is None or self.active_index >= len(self.corners):
                self.active_index = 0
            self.corner_combo.selected_index = self.active_index

    def _corner_selection_changed(self, label, index):
        self._deactivate_scene_lambda1()
        self._invalidate_eigen_field()
        if 0 <= int(index) < len(self.corners):
            self.refine_job_id += 1
            self.active_index = int(index)
            self._refresh_analysis()

    def _save_labels(self):
        if self.folder is None:
            self._status("No dataset loaded")
            return
        try:
            save_corners(self.folder, self.hidden_corners + self.corners)
            self._status(f"Saved {len(self.corners)} labels to corners.json")
        except Exception as exc:
            self._error(exc)

    def _delete_selected(self):
        self._invalidate_eigen_field()
        if self.active_index is None or not self.corners:
            return
        self.refine_job_id += 1
        removed = self.corners.pop(self.active_index)
        self.active_index = min(self.active_index, len(self.corners) - 1) if self.corners else None
        self._save_labels()
        self._refresh_corner_combo()
        self._refresh_analysis()
        self._status(f"Removed: {removed['id']}")

    def _radius_changed(self, value):
        self.radius_mm = max(2, int(round(value)))
        self.refine_job_id += 1
        self.radius_label.text = f"Neighborhood radius: {self.radius_mm} mm"
        if self.xyz is not None:
            self._refresh_analysis()

    def _refine_range_changed(self, value):
        self.refine_range_mm = int(round(value))
        self.search_range_label.text = f"Search range: {self.refine_range_mm} mm"

    def _candidate_spacing_changed(self, value):
        self.candidate_spacing_mm = max(0, int(round(value)))
        if self.candidate_spacing_mm == 0:
            self.spacing_label.text = "Candidate spacing: 0 mm (ALL downsampled points)"
        else:
            self.spacing_label.text = (f"Candidate spacing: {self.candidate_spacing_mm} mm "
                                       "(additional thinning)")

    def _sphere_plane_enabled_changed(self, enabled):
        self.spherical_planes_enabled = bool(enabled)
        if self.xyz is not None:
            self._refresh_analysis()

    def _sphere_plane_tolerance_changed(self, value):
        self.spherical_plane_tolerance_deg = float(int(round(value)))
        self.sphere_plane_tolerance_label.text = (
            f"Angular inlier tolerance: {self.spherical_plane_tolerance_deg:g} deg")
        if self.xyz is not None:
            self._refresh_analysis()

    def _sphere_plane_hist_changed(self, label, index):
        self.spherical_plane_histogram_mode = int(index)
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_link_changed(self, value):
        self.gap_link_deg = float(int(round(value)))
        self.gap_link_label.text = f"Angular connection threshold: {self.gap_link_deg:g} deg"
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_tail_enabled_changed(self, enabled):
        self.gap_tail_enabled = bool(enabled)
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_tail_bin_changed(self, value):
        self.gap_tail_bin_deg = int(round(value))
        self.gap_tail_bin_label.text = f"End-bin angular width: {self.gap_tail_bin_deg} deg"
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_tail_density_changed(self, value):
        self.gap_tail_density_pct = int(round(value))
        self.gap_tail_density_label.text = (
            f"End-bin density: {self.gap_tail_density_pct}% of nonempty-bin median (min 2 pts)")
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_rejection_changed(self, value):
        self.gap_rejection_enabled = bool(value)
        if self.xyz is not None:
            self._refresh_analysis()

    def _gap_edge_span_changed(self, value):
        self.gap_edge_span_deg = float(int(round(value)))
        self.gap_edge_span_label.text = f"Edge-like span threshold: {self.gap_edge_span_deg:g} deg"
        if self.xyz is not None:
            self._refresh_analysis()

    def _compare_button_label(self, hide=False):
        scope = ('dataset ROI' if self.input_raw_indices is not None else 'full scene') if self.compare_scope_index == 1 else 'local corner'
        return f"{'Hide' if hide else 'Compute / show'} heatmap + NMS ({scope})"

    def _compare_scope_changed(self, _name, index):
        self.compare_scope_index = int(index)
        self.compare_scene_voxel_edit.enabled = int(index) == 1
        self.compare_scene_nms_edit.enabled = int(index) == 1
        self._invalidate_extrema_compare()
        self._draw_eigen_field_overlay()

    def _compare_key(self):
        if self.compare_scope_index == 1:
            if self.xyz is None:
                return None
            return ('full_scene', id(self.xyz), id(self.input_raw_indices), len(self.xyz),
                    self.compare_scene_voxel_edit.text_value.strip(),
                    self.compare_scene_nms_edit.text_value.strip(),
                    float(self.eigen_ced_radius_mm),
                    float(self.eigen_pca_radius_mm),
                    self.compare_contrast_tau_edit.text_value.strip(),
                    self.compare_median_tau_edit.text_value.strip())
        if self.analysis_xyz is None or self.active_index is None:
            return None
        entry = self.corners[self.active_index]
        _, center = self._closest_analysis_point(self._eigen_refinement_origin(entry))
        return (str(self.folder),len(self.analysis_xyz),float(self.analysis_voxel_mm),
                str(entry['id']),tuple(center.tolist()),
                float(self.eigen_ced_radius_mm),float(self.eigen_pca_radius_mm),
                float(self.eigen_refine_radius_mm),
                self.compare_contrast_tau_edit.text_value.strip(),
                self.compare_median_tau_edit.text_value.strip())

    def _clear_extrema_compare_geometry(self):
        for name in ('comparison_heatmap', 'comparison_fusion_max',
                     'comparison_lambda1_min', 'comparison_match_links',
                     'comparison_match_midpoints'):
            self._remove_overlay(name)
        self._restore_heatmap_context()

    def _invalidate_extrema_compare(self):
        self.compare_job_id += 1
        self.compare_running = False
        self.compare_show = False
        self.compare_result = None
        self.compare_cache_key = None
        self._clear_extrema_compare_geometry()
        if hasattr(self, 'compare_btn'):
            self.compare_btn.enabled = True
            self.compare_btn.text = self._compare_button_label()
        if hasattr(self, 'compare_summary'):
            self.compare_summary.text = 'Settings changed. Compute heatmap + NMS again.'

    def _deactivate_extrema_compare(self):
        if not self.compare_show and not self.compare_running:
            return
        self.compare_job_id += 1
        self.compare_running = False
        self.compare_show = False
        self._clear_extrema_compare_geometry()
        self.compare_btn.enabled = True
        self.compare_btn.text = self._compare_button_label()
        self._draw_eigen_field_overlay()

    def _compare_match_changed(self, value):
        self.compare_match_mm = int(round(value))
        self.compare_match_label.text = f'Extrema matching distance: {self.compare_match_mm} mm'
        if self.compare_result is not None:
            self.compare_result = rematch_extrema_comparison(
                self.compare_result, self.compare_match_mm)
            if self.compare_show:
                self._draw_extrema_compare_overlay()

    def _compare_method_changed(self, _name, index):
        self.compare_method_index = int(index)
        self.compare_heatmap_mode = (0, 2, 3, 4)[int(index)]
        self.compare_display_combo.selected_index = self.compare_heatmap_mode
        self.compare_auto_contrast_check.enabled = False
        self.compare_nms_check.enabled = True
        self.compare_pairs_check.checked = True
        if self.compare_show:
            self._draw_extrema_compare_overlay()

    def _compare_display_changed(self, _name, index):
        self.compare_heatmap_mode = int(index)
        self.compare_auto_contrast_check.enabled = SCORE_VIEWS[int(index)][1] in (
            'ced_local_min_scores', 'local_minimum_scores', 'ced_median_min_scores', 'median_minimum_scores')
        self.compare_nms_check.enabled = SCORE_VIEWS[int(index)][2] is not None
        self.compare_pairs_check.checked = False
        if self.compare_show:
            self._draw_extrema_compare_overlay()

    def _compare_layers_changed(self, _checked):
        if self.compare_pairs_check.checked:
            self._compare_method_changed(None, self.compare_method_index)
        elif self.compare_show:
            self._draw_extrema_compare_overlay()

    def _compare_auto_contrast_changed(self, _checked):
        if self.compare_show:
            self._draw_extrema_compare_overlay()

    def _compare_thresholds_changed(self):
        try:
            thresholds = {}
            for key, edit in self.compare_threshold_edits.items():
                text = edit.text_value.strip()
                if text:
                    value = float(text)
                    if not np.isfinite(value) or value < 0:
                        raise ValueError('NMS thresholds must be finite and >= 0; blank means OFF')
                    thresholds[key] = value
        except ValueError as exc:
            self._error(str(exc))
            return
        self.compare_thresholds = thresholds
        if self.compare_show:
            self._draw_extrema_compare_overlay()
        else:
            self._status('NMS thresholds applied; compute/show the comparison to see them')

    def _draw_extrema_compare_overlay(self):
        if not self.compare_show or self.compare_result is None:
            return
        if self.compare_cache_key != self._compare_key():
            self._invalidate_extrema_compare()
            return
        field_name, value_key, extrema_key, extreme, units = SCORE_VIEWS[self.compare_heatmap_mode]
        result = (self.compare_result if extrema_key is None else
                  threshold_extrema_comparison(self.compare_result, self.compare_thresholds))
        self._clear_extrema_compare_geometry()
        for name in ('scene_lambda1_heatmap', 'eigen_heatmap',
                     'eigen_grad_vectors', 'eigen_minima'):
            self._remove_overlay(name)
        if self.compare_heatmap_check.checked:
            values = getattr(result, value_key)
            limits = None
            contrast_label = ''
            if value_key in ('ced_local_min_scores', 'local_minimum_scores',
                             'ced_median_min_scores', 'median_minimum_scores'):
                auto = self.compare_auto_contrast_check.checked and np.isfinite(values).any()
                limits = None if auto else (0., 1.)
                contrast_label = ' | Auto Contrast P2-P98' if auto else ' | fixed 0-1'
            colors, (low, high) = eigen_heatmap_colors(values, limits=limits)
            self._overlay('comparison_heatmap', point_cloud(result.positions_mm, colors),
                          self._mat(shader='defaultUnlit', point_size=(6.0 if self.compare_scope_index == 1 else 8.0)))
            self.eigen_heatmap_legend.text = (
                f'{field_name} ({extreme}) | purple {low:.3g} -> yellow {high:.3g}{units}'
                + contrast_label)
        else:
            self.eigen_heatmap_legend.text = f'{field_name} ({extreme}) | RGB background'
        # Full-scene RGB otherwise covers the scored voxels. Local views keep
        # the surrounding cloud available for raw-point depth picking.
        self.scene_widget.scene.show_geometry(
            'cloud', not (self.compare_scope_index == 1 and self.compare_heatmap_check.checked))
        if 'radius_points' in self.visible_overlay_names:
            self.scene_widget.scene.show_geometry('radius_points', False)

        xyz = result.positions_mm
        if self.compare_pairs_check.checked:
            primary, secondary, matches = extrema_comparison_mode(result, self.compare_method_index)
        else:
            primary = getattr(result, extrema_key) if extrema_key is not None else np.empty(0, dtype=np.int64)
            secondary, matches = np.empty(0, dtype=np.int64), ()
        selected_count = len(getattr(result, extrema_key)) if extrema_key is not None else 0
        if self.compare_nms_check.checked and len(primary):
            primary_points = xyz[primary]
            self._overlay('comparison_fusion_max', point_cloud(primary_points, np.tile(
                np.array([0, 222, 255], dtype=np.uint8), (len(primary_points), 1))),
                self._mat(shader='defaultUnlit', point_size=15.0))
        if self.compare_nms_check.checked and len(secondary):
            secondary_points = xyz[secondary]
            self._overlay('comparison_lambda1_min', point_cloud(secondary_points, np.tile(
                np.array([255, 70, 160], dtype=np.uint8), (len(secondary_points), 1))),
                self._mat(shader='defaultUnlit', point_size=13.0))
        if self.compare_nms_check.checked and matches:
            segments = np.vstack([[xyz[a], xyz[b]] for a, b, _ in matches])
            lines = [(2 * i, 2 * i + 1) for i in range(len(matches))]
            self._overlay('comparison_match_links', line_set(segments, lines, [.18, 1., .25]),
                          self._mat(shader='unlitLine', line_width=6.0))
            midpoints = np.array([(xyz[a] + xyz[b]) * .5 for a, b, _ in matches])
            self._overlay('comparison_match_midpoints', point_cloud(midpoints, np.tile(
                np.array([70, 255, 40], dtype=np.uint8), (len(midpoints), 1))),
                # Larger than both NMS markers, including zero-distance pairs.
                self._mat(shader='defaultUnlit', point_size=28.0))

        summary = [
            f"{('DATASET ROI' if self.input_raw_indices is not None else 'FULL SCENE') if self.compare_scope_index == 1 else 'LOCAL CORNER'} | {len(xyz):,} voxels",
            f'Criterion: {field_name} ({extreme})',
            (f'NMS: {selected_count:,}/{len(getattr(self.compare_result, extrema_key)):,} points '
            f'| radius {result.nms_radius_mm:g} mm'
            + (' (hidden)' if not self.compare_nms_check.checked else '')
             if extrema_key is not None else 'Heatmap only: L(p), before CED multiplication'),
            f'CED R={result.ced_radius_mm:g} mm | PCA R={result.pca_radius_mm:g} mm',
        ]
        if self.compare_thresholds and extrema_key is not None:
            summary.extend(f'{title}: {">=" if extreme == "MAX" else "<="} '
                           f'{self.compare_thresholds[key]:g}{unit}'
                           for title, key, _, extreme, unit in SCORE_VIEWS
                           if key in self.compare_thresholds)
        if value_key in ('ced_local_min_scores', 'local_minimum_scores'):
            summary.append(f'Normal PCA lambda1 | local R={result.pca_radius_mm:g} mm '
                           f'| contrast tau={result.contrast_tau:g}')
        elif value_key in ('ced_median_min_scores', 'median_minimum_scores'):
            summary.append(f'Normal PCA lambda1 | Median | local R={result.pca_radius_mm:g} mm '
                           f'| contrast tau={result.median_contrast_tau:g}')
        if self.compare_pairs_check.checked:
            pair_names = ('Fusion MAX', 'CED MAX + ratio MAX', 'CED MAX + weighted lambda1 MIN',
                          'CED MAX + normal lambda1 MIN')
            summary.extend((f'Pair comparison: {pair_names[self.compare_method_index]}',
                            f'Primary {len(primary)} | Secondary {len(secondary)} | Pairs {len(matches)}',
                            f'Matching distance <= {result.match_radius_mm:g} mm'))
        if matches:
            summary.append(f'Mean matched distance: {np.mean([m[2] for m in matches]):.2f} mm')
        self.compare_summary.text = '\n'.join(summary)
        self.win.set_needs_layout()
        self.scene_widget.force_redraw()

    def _toggle_compare_extrema(self):
        scene_scope = self.compare_scope_index == 1
        if scene_scope:
            if self.xyz is None:
                self._status('Load a scene first')
                return
        elif self.analysis_xyz is None or self.active_index is None:
            self._status('Select a corner (Ctrl+click) for LOCAL comparison')
            return
        if self.compare_running:
            self._status('Extrema comparison is computing')
            return
        if self.compare_show:
            # New voxel/NMS text was entered while an old result was visible:
            # run the new settings on the first click, not the second.
            if self.compare_cache_key == self._compare_key():
                self._deactivate_extrema_compare()
                return
            self._deactivate_extrema_compare()
        # Prevent a full-scene job from racing the PCA heatmap or vice versa.
        self._deactivate_scene_lambda1()
        key = self._compare_key()
        if self.compare_result is not None and self.compare_cache_key == key:
            self.compare_show = True
            self.compare_btn.text = self._compare_button_label(hide=True)
            self._draw_extrema_compare_overlay()
            return
        try:
            contrast_tau = float(self.compare_contrast_tau_edit.text_value.strip())
            if not np.isfinite(contrast_tau) or contrast_tau <= 0:
                raise ValueError('Local minimum contrast tau must be finite and > 0')
            median_contrast_tau = float(self.compare_median_tau_edit.text_value.strip())
            if not np.isfinite(median_contrast_tau) or median_contrast_tau <= 0:
                raise ValueError('Median contrast tau must be finite and > 0')
            if scene_scope:
                scene_voxel = float(self.compare_scene_voxel_edit.text_value.strip())
                scene_nms = float(self.compare_scene_nms_edit.text_value.strip())
                if not (np.isfinite(scene_voxel) and .25 <= scene_voxel <= 100):
                    raise ValueError('Voxel size must be 0.25..100 mm')
                if not (np.isfinite(scene_nms) and .25 <= scene_nms <= 300):
                    raise ValueError('NMS radius must be 0.25..300 mm')
                # Independent full-scene voxelization; DO NOT change the local
                # analysis_xyz, label coordinates or existing refinement UI.
                raw = self.xyz if self.input_raw_indices is None else self.xyz[self.input_raw_indices]
                xyz = np.asarray(raw, dtype=np.float64).copy()
            else:
                _, center = self._closest_analysis_point(
                    self._eigen_refinement_origin(self.corners[self.active_index]))
                xyz = self.analysis_xyz.copy()
                view = float(max(40., 2 * self.eigen_refine_radius_mm))
                voxel = float(self.analysis_voxel_mm)
        except (ValueError, TypeError) as exc:
            self._error('Full-scene input: ' + str(exc))
            return
        pca_radius = float(self.eigen_pca_radius_mm)
        ced_radius = float(self.eigen_ced_radius_mm)
        match = float(self.compare_match_mm)
        self.compare_job_id += 1
        job_id = self.compare_job_id
        current_folder = self.folder
        self.compare_running = True
        self.compare_btn.enabled = False
        self.compare_summary.text = ('Voxelizing full original cloud...'
                                     if scene_scope else 'Computing local extrema fields...')
        self._status('Computing FULL-SCENE score fields and NMS...'
                     if scene_scope else 'Computing local score fields and NMS...')

        def worker():
            try:
                if scene_scope:
                    scene_xyz, _ = voxel_downsample_analysis(
                        xyz, np.zeros((len(xyz), 3), dtype=np.uint8), scene_voxel)
                    if job_id != self.compare_job_id:
                        return
                    def progress(stage, done_points, total_points):
                        def update():
                            if job_id == self.compare_job_id:
                                self.compare_summary.text = (
                                    f'FULL SCENE {stage}: {done_points:,}/{total_points:,}'
                                    f' | voxel {scene_voxel:g} mm | NMS {scene_nms:g} mm')
                        try:
                            self.app.post_to_main_thread(self.win, update)
                        except Exception:
                            pass
                    result = compute_full_scene_fusion_pca_comparison(
                        scene_xyz, pca_radius_mm=pca_radius, ced_radius_mm=ced_radius,
                        voxel_mm=scene_voxel, nms_radius_mm=scene_nms,
                        match_radius_mm=match, contrast_tau=contrast_tau,
                        median_contrast_tau=median_contrast_tau,
                        is_cancelled=lambda: job_id != self.compare_job_id,
                        progress_callback=progress)
                    if result is None:
                        return
                else:
                    result = compute_local_fusion_pca_comparison(
                        xyz, center, pca_radius, ced_radius, view, voxel,
                        match_radius_mm=match, contrast_tau=contrast_tau,
                        median_contrast_tau=median_contrast_tau)
                err = None
            except Exception as exc:
                result, err = None, str(exc)

            def done():
                if job_id != self.compare_job_id:
                    return
                self.compare_running = False
                self.compare_btn.enabled = True
                if self.folder != current_folder or self._compare_key() != key:
                    self.compare_summary.text = 'Settings changed during computation; run again'
                    return
                if err:
                    self.compare_summary.text = 'Comparison failed: ' + err
                    self._error('Four-method extrema comparison: ' + err)
                    return
                # Distinct variable name avoids closure UnboundLocalError.
                display_result = result
                if display_result.match_radius_mm != self.compare_match_mm:
                    display_result = rematch_extrema_comparison(
                        display_result, self.compare_match_mm)
                self.compare_cache_key = key
                self.compare_result = display_result
                self.compare_show = True
                self.compare_btn.text = self._compare_button_label(hide=True)
                self._draw_extrema_compare_overlay()
                self._status('Full-scene extrema ready' if scene_scope else 'Local extrema ready')

            try:
                self.app.post_to_main_thread(self.win, done)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _scene_lambda1_key(self):
        return (str(self.folder), int(len(self.analysis_xyz)),
                float(self.analysis_voxel_mm), str(self.eigen_mode),
                float(self.eigen_pca_radius_mm), float(self.eigen_ced_radius_mm))

    def _invalidate_scene_lambda1(self):
        """Called when the cloud, PCA mode, or PCA/CED support radius changes."""
        self.scene_l1_job_id += 1
        self.scene_l1_running = False
        self.scene_l1_show = False
        self.scene_l1_cache_key = None
        self.scene_l1_values = None
        self.scene_l1_colors = None
        self.scene_l1_limits = None
        self._remove_overlay('scene_lambda1_heatmap')
        if hasattr(self, 'scene_l1_btn'):
            self.scene_l1_btn.enabled = True
            self.scene_l1_btn.text = 'Show FULL SCENE lambda1 heatmap'
        if hasattr(self, 'scene_l1_status'):
            self.scene_l1_status.text = 'Full-scene field not computed for current PCA settings'
        self._restore_heatmap_context()

    def _deactivate_scene_lambda1(self):
        """Switch back to original corner-local PCA display, never mix heatmaps.

        Also cancels a pending full-scene job so it cannot unexpectedly take
        over the 3D view after a new corner is clicked or local Refine starts.
        """
        if not self.scene_l1_show and not self.scene_l1_running:
            return
        if self.scene_l1_running:
            self.scene_l1_job_id += 1
            self.scene_l1_running = False
        self.scene_l1_show = False
        self._remove_overlay('scene_lambda1_heatmap')
        self.scene_l1_btn.enabled = True
        self.scene_l1_btn.text = 'Show FULL SCENE lambda1 heatmap'
        self.scene_l1_status.text = 'Full-scene heatmap hidden (local corner view)'
        # The LOCAL rendering method below is deliberately unchanged: it uses
        # field.positions_mm and LOCAL 2nd..98th percentile color normalization.
        self._draw_eigen_field_overlay()
        if self.eigen_field is None or self.eigen_heatmap_mode == 0:
            self.eigen_heatmap_legend.text = 'Local PCA heatmap off; press Refine PCA'
        self.scene_widget.force_redraw()

    def _draw_scene_lambda1_overlay(self):
        if not (self.scene_l1_show and self.scene_l1_colors is not None
                and self.analysis_xyz is not None
                and self.scene_l1_cache_key == self._scene_lambda1_key()):
            return
        # Full-scene and local heatmap/gradient layers are mutually exclusive.
        # Keep the LOCAL FIELD and its statistics cached, not rendered.
        self._remove_overlay('eigen_heatmap')
        self._remove_overlay('eigen_grad_vectors')
        self._remove_overlay('eigen_minima')
        self._overlay('scene_lambda1_heatmap',
                      point_cloud(self.analysis_xyz, self.scene_l1_colors),
                      self._mat(shader='defaultUnlit', point_size=7.0))
        self.scene_widget.scene.show_geometry('cloud', True)
        if 'radius_points' in self.visible_overlay_names:
            self.scene_widget.scene.show_geometry('radius_points', False)
        vmin, vmax = self.scene_l1_limits
        self.eigen_heatmap_legend.text = (
            f'FULL SCENE lambda1: purple {vmin:.3g} -> yellow {vmax:.3g} mm²'
            f' | gray=invalid | {len(self.analysis_xyz):,} voxel points')
        self.scene_widget.force_redraw()

    def _toggle_scene_lambda1(self):
        self._deactivate_extrema_compare()
        if self.analysis_xyz is None:
            self._status('Load a scene before computing full-scene lambda1')
            return
        if self.scene_l1_running:
            self._status('Full-scene PCA already computing')
            return
        if self.scene_l1_show:
            self._deactivate_scene_lambda1()
            return
        key = self._scene_lambda1_key()
        if self.scene_l1_cache_key == key and self.scene_l1_colors is not None:
            self.scene_l1_show = True
            self.scene_l1_btn.text = 'Hide FULL SCENE lambda1 heatmap'
            self._draw_scene_lambda1_overlay()
            return
        xyz = self.analysis_xyz.copy()
        mode = str(self.eigen_mode)
        pca_r = float(self.eigen_pca_radius_mm)
        ced_r = float(self.eigen_ced_radius_mm)
        current_folder = self.folder
        self.scene_l1_job_id += 1
        job_id = self.scene_l1_job_id
        self.scene_l1_running = True
        self.scene_l1_btn.enabled = False
        self.scene_l1_status.text = f'Computing whole scene: {len(xyz):,} voxel points...'
        self._status(f'Computing FULL SCENE lambda1 ({mode}, {len(xyz):,} points)...')

        def worker():
            def cancelled():
                return job_id != self.scene_l1_job_id
            def progress(stage, count, total):
                def update():
                    if job_id == self.scene_l1_job_id:
                        self.scene_l1_status.text = f'{stage}: {count:,}/{total:,} points'
                try:
                    self.app.post_to_main_thread(self.win, update)
                except Exception:
                    pass
            try:
                values = compute_full_scene_lambda1(
                    xyz, pca_radius_mm=pca_r, ced_radius_mm=ced_r,
                    mode=mode, is_cancelled=cancelled,
                    progress_callback=progress)
                if values is None:
                    return
                colors, limits = eigen_heatmap_colors(values)
                error = None
            except Exception as exc:
                values, colors, limits, error = None, None, None, str(exc)
            def done():
                if job_id != self.scene_l1_job_id or self.folder != current_folder:
                    return
                self.scene_l1_running = False
                self.scene_l1_btn.enabled = True
                if error:
                    self.scene_l1_status.text = 'Full-scene PCA failed'
                    self._error('Full-scene lambda1: ' + error)
                    return
                self.scene_l1_cache_key = key
                self.scene_l1_values = values
                self.scene_l1_colors = colors
                self.scene_l1_limits = limits
                self.scene_l1_show = True
                self.scene_l1_btn.text = 'Hide FULL SCENE lambda1 heatmap'
                count = int(np.isfinite(values).sum())
                self.scene_l1_status.text = (
                    f'Full scene: {count:,}/{len(values):,} valid | '
                    + ('CED-weighted' if mode != 'unweighted' else 'Normal PCA'))
                self._draw_scene_lambda1_overlay()
                self._status('Full-scene lambda1 heatmap ready; Ctrl+click remains enabled')
            try:
                self.app.post_to_main_thread(self.win, done)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _invalidate_eigen_field(self):
        self._invalidate_extrema_compare()
        self.eigen_job_id += 1
        self.eigen_field = None
        self.eigen_plot_image = None
        for overlay_name in ('eigen_heatmap', 'eigen_grad_vectors', 'eigen_minima'):
            self._remove_overlay(overlay_name)
        self._restore_heatmap_context()
        if hasattr(self, 'eigen_summary'):
            self.eigen_summary.text = "Press Refine PCA to recompute for selected settings"
        if hasattr(self, 'eigen_refine_status'):
            self.eigen_refine_status.text = "PCA settings changed; previous refinement still saved"

    def _eigen_refine_metric_changed(self, _name, index):
        self.eigen_refine_metric = ('lambda1', 'ratio')[int(index)]
        self.eigen_heatmap_combo.selected_index = int(index) + 1
        self.eigen_heatmap_mode = int(index) + 1
        self._draw_eigen_field_overlay()

    def _eigen_refine_radius_changed(self, value):
        self.eigen_refine_radius_mm = int(round(value))
        self.eigen_refine_range_label.text = (
            f"Refine search radius: {self.eigen_refine_radius_mm} mm")
        self._invalidate_eigen_field()  # Rebuild all candidate samples in the search sphere.

    def _draw_refined_eigen_axes(self):
        """Draw exactly the first TWO PCA axes, centered at refined 3D position."""
        self._remove_overlay('refined_eigen_axis1')
        self._remove_overlay('refined_eigen_axis2')
        if not self.show_eigen_axes.checked or self.active_index is None:
            return
        entry = self.corners[self.active_index]
        saved = entry.get('refinement', {})
        if not str(saved.get('method','')).startswith(('eigen_extremum_', 'eigen_global_')):
            return
        payload = saved.get('eigenvectors_12')
        if payload is None or entry.get('refined_xyz_mm') is None:
            return
        basis = np.asarray(payload, dtype=np.float64)
        if basis.shape != (3,2) or not np.isfinite(basis).all():
            return
        p = np.asarray(entry['refined_xyz_mm'], dtype=np.float64)
        length = max(2.0, min(14.0, 0.38 * float(saved.get('radius_mm',self.radius_mm))))
        for k, (name, color) in enumerate((('refined_eigen_axis1',[1., .55, .10]),
                                            ('refined_eigen_axis2',[.10, .90, 1.]))):
            vec = basis[:,k]
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                continue
            vec = vec / norm
            xyz = np.vstack([p - length*vec, p + length*vec])
            self._overlay(name, line_set(xyz, [(0,1)], color),
                          self._mat(shader='unlitLine', line_width=5.0))

    def _eigen_refinement_origin(self, entry):
        """Keep a fixed start when switching between lambda1 and ratio modes."""
        saved = entry.get('refinement', {})
        if str(saved.get('method','')).startswith('eigen_global_'):
            initial = saved.get('eigen_origin_xyz_mm')
            if initial is not None:
                initial = np.asarray(initial, dtype=np.float64)
                if initial.shape == (3,) and np.isfinite(initial).all():
                    return initial
        return np.asarray(entry.get('refined_xyz_mm',
            self.xyz[int(entry['point_index'])]), dtype=np.float64)

    def _refine_eigen_selected(self):
        self._deactivate_extrema_compare()
        """Move once to global MIN(lambda1) or MAX(lambda2/lambda1)."""
        # Refinement always shows the original click-based LOCAL heatmap.
        self._deactivate_scene_lambda1()
        if self.active_index is None or self.analysis_xyz is None:
            self._status('Select a corner before eigen refinement')
            return
        if self.eigen_running or self.refine_running:
            self._status('Another field/refinement is still being computed')
            return
        entry = self.corners[self.active_index]
        center = self._eigen_refinement_origin(entry)
        _, start = self._closest_analysis_point(center)
        field = self.eigen_field
        # If absent or centered at an older refinement, compute first and refine
        # upon successful completion. This leaves existing Q10 UI untouched.
        if field is None or np.linalg.norm(field.center_mm-start) > max(0.2, self.analysis_voxel_mm*.1):
            self._compute_eigen_field(refine_after=True)
            return
        try:
            choice = select_eigen_global_extremum(
                field, start, float(self.eigen_refine_radius_mm),
                self.eigen_refine_metric)
            best_idx = int(choice.index)
            new_xyz = field.positions_mm[best_idx].copy()
            basis = field.eigenvectors_xyz[best_idx, :, :2]
            eig = field.eigenvalues_mm2[best_idx]
            if not np.isfinite(basis).all() or not np.isfinite(eig).all():
                raise ValueError('Invalid eigenvectors at the selected extremum')
            # Keep original raw GT and history. Switching metrics replaces the
            # previous EIGEN step rather than starting from its output.
            previous = entry.get('refinement', {})
            if str(previous.get('method','')).startswith('eigen_global_') and \
                    previous.get('eigen_base_path_xyz_mm'):
                base_path = np.asarray(previous['eigen_base_path_xyz_mm'], dtype=np.float64)
            else:
                base_path = refinement_path_xyz(entry, self.xyz)
            path = (np.vstack([base_path, new_xyz])
                    if not np.allclose(base_path[-1], new_xyz, atol=1e-5)
                    else base_path)
            entry.pop('refined_point_index', None)
            entry.pop('refinement_path_indices', None)
            entry['refinement_path_xyz_mm'] = path.tolist()
            entry['refined_xyz_mm'] = new_xyz.tolist()
            entry['refinement'] = {
                'method': 'eigen_global_'+choice.method,
                'eigen_origin_xyz_mm': start.tolist(),
                'eigen_base_path_xyz_mm': base_path.tolist(),
                'radius_mm': float(self.eigen_pca_radius_mm),
                'voxel_size_mm': float(self.analysis_voxel_mm),
                'ced_radius_mm': float(self.eigen_ced_radius_mm),
                'pca_mode': str(self.eigen_mode),
                'search_radius_mm': float(self.eigen_refine_radius_mm),
                'objective_before': float(choice.before),
                'objective_after': float(choice.after),
                'step_displacement_mm': float(choice.shift_mm),
                'total_displacement_mm': float(np.linalg.norm(
                    new_xyz - self.xyz[int(entry['point_index'])])),
                'search_boundary_hit': bool(choice.boundary_hit),
                'evaluated': int(choice.evaluated_positions),
                'lambda_values': eig.tolist(),
                'ratio': float(field.lambda2_over_lambda1[best_idx]),
                'eigenvectors_12': basis.tolist(),
                'iteration': int(len(path)-1),
                'converged': bool(choice.shift_mm < 1e-7),
            }
            self._save_labels()
            unstable = bool(eig[0] > 0 and (eig[0]-eig[1])/eig[0] < 0.05)
            self.eigen_refine_status.text = (
                f'{choice.method} full-radius optimum: shift {choice.shift_mm:.2f} mm; '
                f'λ1={eig[0]:.2f}, ratio={field.lambda2_over_lambda1[best_idx]:.3f}'
                + (' | v1/v2 degenerate: axes unstable' if unstable else ''))
            self._refresh_analysis()
            self._status(f'Eigen refinement ({choice.method}): '
                         f'selected global optimum at {choice.shift_mm:.2f} mm; '
                         f'3D v1/v2 shown (recompute field at new location if needed)')
        except Exception as exc:
            self._error('Eigen refinement: '+str(exc))

    def _eigen_mode_changed(self, _name, index):
        self.eigen_mode = ('ced_threshold', 'unweighted')[int(index)]
        # The shared CED radius still controls Fusion/CED heatmaps in Normal PCA mode.
        self.eigen_ced_slider.enabled = True
        self.eigen_ced_label.text = f"CED radius: {self.eigen_ced_radius_mm} mm"
        self._invalidate_eigen_field()
        self._invalidate_scene_lambda1()

    def _eigen_ced_radius_changed(self, value):
        self.eigen_ced_radius_mm = int(round(value))
        self.eigen_ced_label.text = f"CED radius: {self.eigen_ced_radius_mm} mm"
        self._invalidate_eigen_field()
        self._invalidate_scene_lambda1()

    def _eigen_pca_radius_changed(self, value):
        self.eigen_pca_radius_mm = int(round(value))
        self.eigen_pca_label.text = f"PCA support radius: {self.eigen_pca_radius_mm} mm"
        self._invalidate_eigen_field()
        self._invalidate_scene_lambda1()

    def _eigen_heatmap_changed(self, _name, index):
        self._deactivate_extrema_compare()
        # Changing local heatmap selection must not leave a full-scene overlay
        # covering the selected ROI's own color scale.
        self._deactivate_scene_lambda1()
        self.eigen_heatmap_mode = int(index)
        self._draw_eigen_field_overlay()

    def _restore_heatmap_context(self):
        # Restore source cloud and Q10 support points when heatmap is off.
        if self.xyz is not None:
            try:
                self.scene_widget.scene.show_geometry('cloud', True)
            except Exception:
                pass
        if 'radius_points' in self.visible_overlay_names:
            try:
                self.scene_widget.scene.show_geometry('radius_points', not self.scene_l1_show)
            except Exception:
                pass

    def _draw_eigen_heatmap_overlay(self):
        # Color measured voxel centroids in the rotatable Open3D scene.
        self._remove_overlay('eigen_heatmap')
        if self.scene_l1_show:
            return  # Whole-scene heatmap has exclusive control of the colors.
        field = self.eigen_field
        if field is None or self.eigen_heatmap_mode == 0:
            self._restore_heatmap_context()
            return
        metric = 'lambda1' if self.eigen_heatmap_mode == 1 else 'ratio'
        values = eigen_heatmap_values(field, metric)
        colors, (vmin, vmax) = eigen_heatmap_colors(values)
        self._overlay('eigen_heatmap', point_cloud(field.positions_mm, colors),
                      self._mat(shader='defaultUnlit', point_size=9.0))
        # Keep the original cloud VISIBLE. Mouse depth picking uses the visible
        # scene and raw-point KDTree; hiding it made Ctrl+click fail outside ROI.
        self.scene_widget.scene.show_geometry('cloud', True)
        if 'radius_points' in self.visible_overlay_names:
            self.scene_widget.scene.show_geometry('radius_points', False)
        self.eigen_heatmap_legend.text = (
            f'Viridis: purple {vmin:.3g} -> yellow {vmax:.3g}'
            + (' mm²' if metric == 'lambda1' else ' (ratio)')
            + ' | gray = invalid')

    def _eigen_overlay_changed(self, _name, index):
        self._deactivate_extrema_compare()
        self._deactivate_scene_lambda1()
        self.eigen_overlay_mode = int(index)
        self._draw_eigen_field_overlay()

    def _draw_eigen_field_overlay(self):
        if self.compare_show:
            self._draw_extrema_compare_overlay()
            return
        if self.scene_l1_show:
            # A full-scene PCA heatmap must not be overlaid with LOCAL extrema
            # or arrows, even on refresh after a new corner is selected.
            self._draw_scene_lambda1_overlay()
            return
        self._draw_eigen_heatmap_overlay()
        # Arrows are diagnostic field directions, not actual corner motion.
        self._remove_overlay('eigen_grad_vectors')
        self._remove_overlay('eigen_minima')
        field = self.eigen_field
        if field is None or self.eigen_overlay_mode == 0:
            self.scene_widget.force_redraw()
            return
        grad = (field.gradients_lambda1 if self.eigen_overlay_mode == 1
                else field.gradients_ratio)
        finite = np.flatnonzero(np.isfinite(grad).all(axis=1))
        if len(finite) > 110:
            finite = finite[np.linspace(0, len(finite)-1, 110, dtype=int)]
        if len(finite):
            start = field.positions_mm[finite]
            directions = (-grad[finite] if self.eigen_overlay_mode == 1
                          else grad[finite])
            mag = np.linalg.norm(directions, axis=1)
            good = mag > 1e-12
            if np.any(good):
                start = start[good]
                directions = directions[good] / mag[good, None] * max(2.,field.view_radius_mm*.12)
                end = start + directions
                coords = np.vstack([start, end])
                n = len(start)
                self._overlay('eigen_grad_vectors', line_set(coords,
                    [(i,i+n) for i in range(n)], [.99,.24,.20] if self.eigen_overlay_mode==1
                    else [.32,.88,.98]), self._mat(shader='unlitLine',line_width=2.0))
        chosen = (field.minima_lambda1 if self.eigen_overlay_mode == 1
                  else field.maxima_ratio)
        if len(chosen):
            self._overlay('eigen_minima', point_cloud(field.positions_mm[chosen],
                np.tile(np.array([255,55,115],dtype=np.uint8),(len(chosen),1))),
                self._mat(point_size=11))
        self.scene_widget.force_redraw()

    def _show_eigen_field_dialog(self):
        if self.eigen_plot_image is None:
            return
        dialog = gui.Dialog("3D PCA heatmaps | lambda1 MIN / lambda2/lambda1 MAX")
        content = gui.Vert(0.3*self.win.theme.font_size,
                           gui.Margins(12, 12, 12, 12))
        image_widget = gui.ImageWidget(o3d.geometry.Image(self.eigen_plot_image))
        content.add_child(image_widget)
        controls = gui.Horiz(0.4 * self.win.theme.font_size)
        save_btn = gui.Button("Save PNG")
        def save_image():
            try:
                if self.folder is None:
                    raise ValueError('Open a dataset before saving')
                target = Path(self.folder)/'pca_eigen_field_3d.png'
                o3d.io.write_image(str(target),o3d.geometry.Image(self.eigen_plot_image))
                self._status(f"Saved diagnostic: {target}")
            except Exception as exc:
                self._error(exc)
        save_btn.set_on_clicked(save_image)
        controls.add_child(save_btn)
        close_btn = gui.Button("Close")
        close_btn.set_on_clicked(self.win.close_dialog)
        controls.add_child(close_btn)
        content.add_child(controls)
        dialog.add_child(content)
        self.win.show_dialog(dialog)

    def _compute_eigen_field(self, refine_after=False):
        self._deactivate_extrema_compare()
        self._deactivate_scene_lambda1()
        if self.active_index is None or self.analysis_xyz is None:
            self._status('Select a corner first')
            return
        if self.eigen_running:
            self._status('Eigen field computation already in progress')
            return
        entry = self.corners[self.active_index]
        ref = self._eigen_refinement_origin(entry)
        _, p = self._closest_analysis_point(ref)
        snapshot_xyz = self.analysis_xyz.copy()
        radius = float(self.eigen_pca_radius_mm)
        # Display enough around the refinement sphere; no separate GUI view-radius slider.
        view = float(max(40.0, 2.0 * self.eigen_refine_radius_mm))
        ced_r = float(self.eigen_ced_radius_mm)
        mode = self.eigen_mode
        voxel = self.analysis_voxel_mm
        dense_radius = float(self.eigen_refine_radius_mm + 2.0*voxel)
        current_id = entry['id']
        current_folder = self.folder
        self.eigen_job_id += 1
        job_id = self.eigen_job_id
        self.eigen_running = True
        self.eigen_refine_btn.enabled = False
        self._status(f'Computing local PCA eigen-field: {mode}, view={view:g} mm...')
        def worker():
            try:
                result = compute_eigen_field(snapshot_xyz,p,radius,view,ced_r,
                                             mode=mode,voxel_mm=voxel,
                                             dense_radius_mm=dense_radius)
                image = render_eigen_field_3d_image(result)
                error = None
            except Exception as exc:
                result, image, error = None, None, str(exc)
            def done():
                self.eigen_running = False
                self.eigen_refine_btn.enabled = True
                if job_id != self.eigen_job_id or self.folder != current_folder:
                    return
                if self.active_index is None or self.corners[self.active_index]['id'] != current_id:
                    return
                if error:
                    self._error('Eigen-field: '+error)
                    return
                self.eigen_field = result
                self.eigen_plot_image = image
                at = result.center_index
                eig = result.eigenvalues_mm2[at]
                ratio = result.lambda2_over_lambda1[at]
                eig_txt = '/'.join(f'{x:.2f}' for x in eig)
                ratio_txt = f'{ratio:.3f}' if np.isfinite(ratio) else 'undefined'
                def extremum_description(indices, score_values):
                    if not len(indices):
                        return 'none (interior ROI)'
                    distances = np.linalg.norm(result.positions_mm[indices]-result.center_mm,
                                               axis=1)
                    best = int(indices[np.argmin(distances)])
                    shift = result.positions_mm[best] - result.center_mm
                    return (f'nearest Δ={np.round(shift,1).tolist()} mm, '
                            f'|Δ|={np.linalg.norm(shift):.1f} mm, '
                            f'value={score_values[best]:.3g}')
                l1_min_description = extremum_description(result.minima_lambda1,
                    result.eigenvalues_mm2[:,0])
                ratio_max_description = extremum_description(result.maxima_ratio,
                    result.lambda2_over_lambda1)
                self.eigen_summary.text = (
                    f'PCA values λ1/λ2/λ3 = {eig_txt} mm²\n'
                    f'λ2/λ1 = {ratio_txt}\n'
                    f'Valid field {np.count_nonzero(np.isfinite(result.eigenvalues_mm2[:,0]))}'
                    f'/{len(result.positions_mm)} (all ROI={result.total_candidates})\n'
                    f'λ1 minima: {len(result.minima_lambda1)}; '
                    f'ratio maxima: {len(result.maxima_ratio)}\n'
                    f'λ1: {l1_min_description}\n'
                    f'λ2/λ1 MAX: {ratio_max_description}\n'
                    f'Arrows = -grad λ1 / +grad ratio (not corner direction)')
                self._draw_eigen_field_overlay()
                # Do not open a MODAL dialog here; it blocks Ctrl+click.
                self._status('3D heatmap ready. Ctrl+click remains enabled; optional plot button is above.')
                if refine_after:
                    self._refine_eigen_selected()
            try:
                self.app.post_to_main_thread(self.win, done)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _analysis_voxel_changed(self, value):
        new_size = float(int(round(value)))
        if new_size == self.analysis_voxel_mm:
            return
        self.refine_job_id += 1     # Invalidate outstanding optimizer results.
        self.analysis_voxel_mm = new_size
        self.voxel_label.text = f"Analysis voxel downsampling: {new_size:g} mm (shared)"
        if self.xyz is not None:
            self._rebuild_analysis_cloud()
            selected_count = len(self.xyz) if self.input_raw_indices is None else len(self.input_raw_indices)
            self.dataset_label.text = (f"{self.folder.name} | {'ROI' if self.input_raw_indices is not None else 'scene'} "
                                       f"{selected_count:,}/{len(self.xyz):,} | "
                                       f"voxel {len(self.analysis_xyz):,} "
                                       f"({self.analysis_voxel_mm:g} mm)")
            self._refresh_analysis()
            self._status(f"Analysis cloud rebuilt: {len(self.analysis_xyz):,} points "
                         f"at {self.analysis_voxel_mm:g} mm. Re-refine labels if needed.")

    def _refine_selected(self):
        if self.analysis_xyz is None or self.active_index is None:
            self._status("Select a corner before refining")
            return
        if self.refine_running:
            self._status("A refinement is already running")
            return
        entry = self.corners[self.active_index]
        # The original label remains a RAW-cloud index; the optimizer works
        # exclusively on GLOBAL DOWNSAMPLED voxel centroid positions.
        p0 = self.xyz[int(entry["point_index"])].astype(np.float64)
        last_coord = entry.get("refined_xyz_mm")
        if last_coord is None:
            legacy_id = current_refinement_point_index(entry, len(self.xyz))
            last_coord = self.xyz[legacy_id]
        start_id, start = self._closest_analysis_point(last_coord)
        radius, search, spacing = self.radius_mm, self.refine_range_mm, self.candidate_spacing_mm
        # Candidates and supports are globally downsampled points.
        n_candidates, candidate_indices, _ = self.analysis_tree.search_radius_vector_3d(
            start, float(search) + 1e-8)
        if n_candidates < 1:
            self._status("No downsampled candidate points in the search range")
            return
        # Support = the union/superset of all candidate radius-R neighborhoods.
        _, support_indices, _ = self.analysis_tree.search_radius_vector_3d(
            start, float(radius + search + 0.001))
        support_ids = np.asarray(support_indices, dtype=np.int64).copy()
        support_xyz = self.analysis_xyz[support_ids].astype(np.float64).copy()
        try:
            candidates = observed_candidates(start, self.analysis_xyz[np.asarray(candidate_indices)],
                                              np.asarray(candidate_indices), search, spacing)
            if len(candidates) > 3500:
                self._status(f"Too many downsampled candidates ({len(candidates)}). "
                             "Reduce Search range or raise Candidate spacing.")
                return
        except ValueError as exc:
            self._error(exc)
            return
        self.refine_job_id += 1
        token = self.refine_job_id
        dataset = self.folder
        corner_id = entry["id"]
        self.refine_running = True
        self.refine_btn.enabled = False
        self._status(f"Optimizing Q10: {len(candidates)} downsampled points "
                     f"(support={len(support_xyz)}, spacing={spacing} mm)...")

        def worker():
            try:
                result = discrete_refine_q10(start, start_id, support_xyz, support_ids,
                                             float(radius), search, spacing)
                error = None
            except Exception as exc:
                result, error = None, str(exc)
            def done():
                self.refine_running = False
                self.refine_btn.enabled = True
                if token != self.refine_job_id or dataset != self.folder:
                    return  # Dataset/radius/corner changed during computation.
                current = next((c for c in self.corners if c["id"] == corner_id), None)
                if current is None or self.active_index is None or self.corners[self.active_index]["id"] != corner_id:
                    return
                if error is not None:
                    self._error(error)
                    return
                new_index = int(result.refined_point_index)
                if new_index == start_id:
                    # Local convergence is valid only for this radius, search and spacing.
                    previous = {k: v for k, v in current.get("refinement", {}).items()
                            if k not in ("progressive_enabled", "effective_radius_mm", "progressive_steps")}
                    current["refinement"] = {
                        **previous,
                        "method": "discrete_q10_global_voxel_points",
                        "voxel_size_mm": self.analysis_voxel_mm,
                        "radius_mm": radius,
                        "search_radius_mm": search,
                        "candidate_spacing_mm": spacing,
                        "converged": True,
                        "q10_after": result.q10_after,
                    }
                    self._save_labels()
                    self._refresh_analysis()
                    self._status(f"Local optimum at downsampled index {start_id}: "
                                 f"Q10={result.q10_before:+.4f}. No better downsampled point "
                                 f"within {search} mm (tested {result.tested_positions}).")
                    return
                path = refinement_path_xyz(current, self.xyz)
                new_xyz = self.analysis_xyz[new_index].astype(np.float64)
                path = np.vstack([path, new_xyz])
                # Do not store a downsampled index in an original-cloud-index field.
                current.pop("refined_point_index", None)
                current.pop("refinement_path_indices", None)
                current["refinement_path_xyz_mm"] = path.tolist()
                current["refined_xyz_mm"] = new_xyz.tolist()
                total_displacement = float(np.linalg.norm(
                    new_xyz - p0))
                current["refinement"] = {
                    "method": "discrete_q10_global_voxel_points",
                    "voxel_size_mm": self.analysis_voxel_mm,
                    "radius_mm": radius,
                    "search_radius_mm": search,
                    "candidate_spacing_mm": spacing,
                    "q10_before": result.q10_before,
                    "q10_after": result.q10_after,
                    "neighbors_before": result.count_before,
                    "neighbors_after": result.count_after,
                    "evaluated": result.tested_positions,
                    "available_downsampled_candidates": result.available_positions,
                    "displacement_mm": result.displacement_mm,
                    "step_displacement_mm": result.displacement_mm,
                    "total_displacement_mm": total_displacement,
                    "iteration": len(path) - 1,
                    "boundary_hit": result.boundary_hit,
                    "min_support_ratio": 0.80,
                    "converged": False,
                }
                self._save_labels()
                self._refresh_analysis()
                self._status(f"Step {len(path)-1}: Q10 {result.q10_before:+.4f} "
                             f"-> {result.q10_after:+.4f}; moved {result.displacement_mm:.2f} mm "
                             f"(total {total_displacement:.2f} mm), downsampled index {new_index}; "
                             f"tested {result.tested_positions} downsampled candidates"
                             + (" | near search boundary" if result.boundary_hit else ""))
            try:
                self.app.post_to_main_thread(self.win, done)
            except Exception:
                pass  # Window can be closed while the worker is running.
        threading.Thread(target=worker, daemon=True).start()

    def _reset_refinement(self):
        self._invalidate_eigen_field()
        if self.active_index is None:
            return
        self.refine_job_id += 1
        entry = self.corners[self.active_index]
        entry.pop("refined_xyz_mm", None)
        entry.pop("refined_point_index", None)
        entry.pop("refinement_path_indices", None)
        entry.pop("refinement_path_xyz_mm", None)
        entry.pop("refinement", None)
        self._save_labels()
        self._refresh_analysis()
        self._status("Refinement cleared: using original labeled point")

    def _on_mouse(self, event):
        if self.xyz is None:
            return gui.Widget.EventCallbackResult.IGNORED
        is_picking = (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_button_down(gui.MouseButton.LEFT)
            and event.is_modifier_down(gui.KeyModifier.CTRL)
        )
        if not is_picking:
            return gui.Widget.EventCallbackResult.IGNORED
        # The whole-scene point overlay is larger than the original RGB cloud.
        # Disable it before depth-picking and restore LOCAL PCA visualization.
        self._deactivate_scene_lambda1()
        self._deactivate_extrema_compare()
        local_x = int(event.x - self.scene_widget.frame.x)
        local_y = int(event.y - self.scene_widget.frame.y)
        width = int(self.scene_widget.frame.width)
        height = int(self.scene_widget.frame.height)
        if local_x < 0 or local_y < 0 or local_x >= width or local_y >= height:
            return gui.Widget.EventCallbackResult.IGNORED

        def callback(depth_image):
            depth_array = np.asarray(depth_image)
            if local_y >= depth_array.shape[0] or local_x >= depth_array.shape[1]:
                return
            depth = float(depth_array[local_y, local_x])
            if not np.isfinite(depth) or depth >= 0.999999:
                return
            world = self.scene_widget.scene.camera.unproject(
                local_x, local_y, depth, width, height
            )
            # Open3D renders asynchronously. The callback is NOT on the UI thread.
            def on_gui_thread():
                self._picked_world_point(np.asarray(world, dtype=np.float64))
            self.app.post_to_main_thread(self.win, on_gui_thread)

        self.scene_widget.scene.scene.render_to_depth_image(callback)
        return gui.Widget.EventCallbackResult.HANDLED

    def _picked_world_point(self, world):
        if self.xyz is None or self.kdtree is None:
            return
        count, indices, distances2 = self.kdtree.search_knn_vector_3d(world, 1)
        if count == 0 or np.sqrt(distances2[0]) > 12.0:
            self._status("No cloud point near clicked location")
            return
        picked_idx = int(indices[0])
        if self.input_raw_indices is not None:
            picked_idx = int(self.input_raw_indices[picked_idx])
        if self.mode.selected_index == 0:
            if any(int(item["point_index"]) == picked_idx for item in self.corners):
                self._status("Already labeled point")
                return
            used = {item["id"] for item in self.hidden_corners + self.corners}
            n = 1
            while f"corner_{n:03d}" in used:
                n += 1
            self.corners.append({"id": f"corner_{n:03d}", "point_index": picked_idx})
            self.active_index = len(self.corners) - 1
            self._save_labels()
            self._refresh_corner_combo()
            self._refresh_analysis()
            self._status(f"Added {self.corners[-1]['id']} at index {picked_idx}")
        elif self.corners:
            corner_positions = np.array([self.xyz[c["point_index"]] for c in self.corners])
            dist = np.linalg.norm(corner_positions.astype(float) - world[None, :], axis=1)
            best = int(np.argmin(dist))
            if dist[best] > max(12., 0.12 * self.radius_mm):
                self._status("Click near a RED saved corner or use the label list")
                return
            self.active_index = best
            self.corner_combo.selected_index = best
            self._refresh_analysis()
        else:
            self._status("No saved labels. Switch to 'Label new corner' first.")
        if self.active_index is not None:
            self._workspace_changed(1)

    def _refresh_analysis(self):
        self._clear_overlays()
        self._clear_unit_overlay()
        self.last_cone = None
        self.last_spherical_planes = None
        self.last_gap_validation = None
        self.gap_summary.text = "Gap analysis pending (cyan kept arc; magenta noise)"
        self.sphere_plane_summary.text = "Fit is pending (select a corner)"
        self.hist_widget.update_image(o3d.geometry.Image(cosine_histogram_image(None)))
        self.hist_title.text = "cos(theta) histogram | -1 to +1"
        self.effective_radius_label.text = f"Analysis radius: {self.radius_mm:.1f} mm (fixed for all stages)"
        if self.xyz is None:
            return
        if self.corners:
            centers = self.xyz[[c["point_index"] for c in self.corners]]
            colors = np.tile([255, 55, 65], (len(centers), 1))
            self._overlay("corner_markers", point_cloud(centers, colors),
                          self._mat(point_size=11))
        if self.active_index is None or self.active_index >= len(self.corners):
            self.info.text = "Ctrl+click a point to add a corner."
            self._draw_scene_lambda1_overlay()
            return
        entry = self.corners[self.active_index]
        p0 = self.xyz[entry["point_index"]].astype(np.float64)
        refined_idx = entry.get("refined_point_index")
        if "refined_xyz_mm" in entry:
            reference = np.asarray(entry["refined_xyz_mm"], dtype=np.float64)
        elif isinstance(refined_idx, int) and 0 <= refined_idx < len(self.xyz):
            reference = self.xyz[refined_idx].astype(np.float64)
        else:
            reference = p0
        # The query is also snapped onto the globally downsampled point cloud.
        _, p = self._closest_analysis_point(reference)
        analysis_radius = float(self.radius_mm)
        self.effective_radius_label.text = (
            f"Analysis radius: {analysis_radius:.1f} mm (Q10 / RANSAC / MaxGap)")
        self._overlay("active_corner", sphere(p0, max(1.6, self.radius_mm * .055),
                      [1., .18, .20]), self._mat(shader="defaultLit"))
        if "refined_xyz_mm" in entry or "refined_point_index" in entry:
            self._overlay("refined_corner", sphere(p, max(1.8, self.radius_mm * .062),
                          [.12, 1.0, .2]), self._mat(shader="defaultLit"))
            history_xyz = refinement_path_xyz(entry, self.xyz)
            if not np.allclose(history_xyz[-1], p, atol=1e-5):
                history_xyz = np.vstack([history_xyz, p])
            if len(history_xyz) >= 2:
                edges = [(i, i + 1) for i in range(len(history_xyz) - 1)]
                self._overlay("refinement_link", line_set(history_xyz, edges,
                              [1., 1., 1.]), self._mat(shader="unlitLine", line_width=4.5))
            if len(history_xyz) > 2:
                waypoints = history_xyz[1:-1]
                self._overlay("refinement_waypoints", point_cloud(
                    waypoints, np.tile([235, 235, 235], (len(waypoints), 1))),
                    self._mat(point_size=7.5))
        count, indices, _ = self.analysis_tree.search_radius_vector_3d(p, float(analysis_radius))
        if count == 0:
            self.info.text = "No neighbors found"
            return
        ids = np.asarray(indices, dtype=np.int64)
        neighbors = self.analysis_xyz[ids].astype(np.float64)
        result = analyze_neighborhood(p, neighbors, float(analysis_radius))
        if len(ids) > MAX_HIGHLIGHT_POINTS:
            show_ids = ids[np.linspace(0, len(ids) - 1, MAX_HIGHLIGHT_POINTS, dtype=np.int64)]
        else:
            show_ids = ids
        self._overlay("radius_points", point_cloud(self.analysis_xyz[show_ids],
                      np.tile([250, 204, 38], (len(show_ids), 1))),
                      self._mat(point_size=4))

        if self.show_plane.checked:
            self._overlay("centroid", sphere(result.centroid_mm,
                          max(1.8, analysis_radius * .05), [0.15, .50, 1.]),
                          self._mat(shader="defaultLit"))
            if result.normal is not None:
                line = line_set(np.vstack((p, result.centroid_mm)), [(0, 1)],
                                [.16, .95, .22])
                self._overlay("centroid_vector", line,
                              self._mat(shader="unlitLine", line_width=3))
                disk, outline = plane_disk(p, result.normal, analysis_radius * .72)
                self._overlay("perpendicular_disk", disk,
                              self._mat(shader="defaultLitTransparency",
                                        color=[.08, .83, .88, .23]))
                self._overlay("perpendicular_outline", outline,
                              self._mat(shader="unlitLine", line_width=2))

        cone = None
        err = None
        try:
            cone = analyze_cone(p, neighbors)
            self.last_cone = cone
        except ValueError as exc:
            err = str(exc)
        except Exception as exc:
            err = f"Cone analysis error: {exc}"

        if cone is not None:
            self._draw_spherical_analysis(cone)
            self.hist_widget.update_image(o3d.geometry.Image(cosine_histogram_image(cone)))
            self.hist_title.text = (f"cos(theta) | N={len(cone.cosines):,} | "
                                    f"Q10={cone.cosine_q10:+.3f} | "
                                    f"R={cone.resultant_length:.3f}")
            if self.show_cone.checked:
                scale = max(3., analysis_radius * 0.82)
                mesh, ring, spokes = cone_geometry(p, cone.mean_axis,
                                                   cone.half_angle_deg, scale=scale)
                self._overlay("local_cone_shell", mesh, self._mat(
                    shader="defaultLitTransparency", color=[.13, .74, 1., .24]))
                self._overlay("local_cone_ring", ring,
                              self._mat(shader="unlitLine", line_width=2.8))
                self._overlay("local_cone_spokes", spokes,
                              self._mat(shader="unlitLine", line_width=1.7))
                self._overlay("local_mean_axis", line_set(
                    [p, p + scale * cone.mean_axis * 1.07], [(0, 1)],
                    [.96, .36, .98]), self._mat(shader="unlitLine", line_width=3.5))
            coverage = float(np.mean(cone.inlier_mask) * 100.0)
            concentration_warning = ("\nWARNING: mean axis unstable (low R)."
                                     if cone.resultant_length < 0.07 else "")
            if cone.half_angle_deg > 90.0:
                concentration_warning += "\nWide cone (>90 deg): weak directionality."
            cone_text = (
                f"Directions (nonzero): {len(cone.directions):,}\n"
                f"Spherical mean axis m:\n"
                f"[{cone.mean_axis[0]:+.3f}, {cone.mean_axis[1]:+.3f}, "
                f"{cone.mean_axis[2]:+.3f}]\n"
                f"Q10 of (u . m): {cone.cosine_q10:+.4f}\n"
                f"90% CONE half-angle: {cone.half_angle_deg:.2f} deg\n"
                f"Observed coverage: {coverage:.1f}%\n"
                f"Mean resultant length R: {cone.resultant_length:.3f}\n"
                f"Mean cos(theta): {cone.resultant_length:+.4f}"
                + concentration_warning
            )
        else:
            cone_text = "CONE: undefined - " + str(err)

        # Fit origin-constrained great-circle models IN THE UNIT SPHERE.
        # This does not change Q10 refinement or saved labels.
        spherical_text = ""
        if self.spherical_planes_enabled:
            try:
                offsets = neighbors - p[None, :]
                lengths = np.linalg.norm(offsets, axis=1)
                good = np.isfinite(lengths) & (lengths > 1e-7)
                directions = offsets[good] / lengths[good, None]
                planes = segment_origin_spherical_planes(
                    directions, k=self.spherical_plane_k,
                    tolerance_deg=self.spherical_plane_tolerance_deg,
                    iterations=self.spherical_plane_iterations)
                self.last_spherical_planes = planes
                self._draw_spherical_planes(planes)
                gap_validation = evaluate_plane_max_gap(
                    planes, max_link_gap_deg=self.gap_link_deg,
                    edge_span_deg=self.gap_edge_span_deg,
                    tail_filter_enabled=self.gap_tail_enabled,
                    tail_bin_width_deg=self.gap_tail_bin_deg,
                    tail_density_fraction=self.gap_tail_density_pct / 100.0)
                self.last_gap_validation = gap_validation
                verdict = gap_validation.status if self.gap_rejection_enabled else 'UNKNOWN'
                desc = [
                    f"MAX ANGULAR GAP | Radius={analysis_radius:.1f} mm | K={len(planes.planes)}",
                    f"Link gap<={self.gap_link_deg:g} deg; dominant component first",
                    (f"END bins: {self.gap_tail_bin_deg} deg, "
                     f"{self.gap_tail_density_pct}% median, min 2 points; "
                     f"{'ON' if self.gap_tail_enabled else 'OFF'}"),
                    f"PCL-style angular gap>90 deg | experimental Edge span>={self.gap_edge_span_deg:g} deg",
                ]
                for t in gap_validation.tests:
                    desc.append(
                        f"P{t.plane_index+1}: keep={t.kept_count}/{t.raw_count}, "
                        f"outliers={t.discarded_count}, tail={t.tail_removed_count}, "
                        f"groups={t.component_count} | "
                        f"SPAN {t.raw_span_deg:.1f} -> {t.observed_span_deg:.1f} deg "
                        f"(min bin={t.density_threshold_count}, median={t.density_reference_count:.1f}) | "
                        f"PCL boundary={'YES' if t.pcl_boundary_like else 'NO'} | "
                        f"{'UNKNOWN' if not t.reliable else ('EDGE-LIKE' if t.edge_like else 'CORNER-LIKE')}")
                if not self.gap_rejection_enabled:
                    desc.append('Diagnostic only: experimental classification disabled')
                desc.append(f"FINAL: {verdict} ({'ON' if self.gap_rejection_enabled else 'OFF'})")
                self.gap_summary.text = '\n'.join(desc)
                if verdict in ('CORNER','REJECT'):
                    marker_color = ([0.12,0.47,1.0] if verdict=='CORNER'
                                    else [1.0,0.84,0.08])
                    marker_size = max(2.4, float(analysis_radius)*0.09)
                    self._overlay('active_corner', sphere(p,marker_size,marker_color),
                                  self._mat(shader='defaultLit'))
                    if 'refined_xyz_mm' in entry or 'refined_point_index' in entry:
                        self._overlay('refined_corner',sphere(p,marker_size,marker_color),
                                      self._mat(shader='defaultLit'))
                multiplane_count = int(np.count_nonzero(planes.membership_count >= 2))
                missing_count = int(np.count_nonzero(planes.membership_count == 0))
                detail_lines = [
                    f"Spherical great circles d=0: R={analysis_radius:.1f} mm | Auto K={len(planes.planes)} (max {planes.requested_k})",
                    f"angular tolerance={planes.tolerance_deg:g} deg, unassigned={missing_count}",
                    f"multi-plane members={multiplane_count} (not deduplicated)",
                ]
                for pi, plane in enumerate(planes.planes, 1):
                    detail_lines.append(
                        f"P{pi}: N={int(plane.inlier_mask.sum())}, "
                        f"RMS={plane.residual_rms_deg:.2f} deg, "
                        f"n={np.round(plane.normal, 2).tolist()}")
                if len(planes.planes) < planes.requested_k:
                    detail_lines.append("Remaining unsupported planes were not forced")
                self.sphere_plane_summary.text = "\n".join(detail_lines)
                spherical_text = "\n" + "\n".join(detail_lines) + "\n"
                if self.spherical_plane_histogram_mode == 0:
                    self.hist_widget.update_image(o3d.geometry.Image(
                        spherical_plane_histogram_image(
                            planes, gap_validation=self.last_gap_validation)))
                    self.hist_title.text = (
                        f"Per-plane circular angles | R={analysis_radius:.1f} mm | bins={SPHERE_PLANE_HIST_BINS} | "
                        f"Auto K={len(planes.planes)}")
                    self.hist_legend.text = (
                        "Rose: raw point count | cyan=kept angular arc | magenta=outlier directions")
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.sphere_plane_summary.text = f"Spherical plane fit unavailable: {exc}"
                self.gap_summary.text = f"UNKNOWN: plane fit failed ({exc})"
                spherical_text = "\nSpherical plane fit unavailable: " + str(exc) + "\n"
                if self.spherical_plane_histogram_mode == 0:
                    self.hist_widget.update_image(o3d.geometry.Image(
                        spherical_plane_histogram_image(None)))
                    self.hist_title.text = "Per-plane circular histograms unavailable"
                    self.hist_legend.text = "Try selecting more points or a larger radius"
        else:
            self.sphere_plane_summary.text = "Spherical plane fitting disabled"
            self.gap_summary.text = "UNKNOWN: plane fitting disabled"
        if self.spherical_plane_histogram_mode == 1 or not self.spherical_planes_enabled:
            # Preserve the original Q10 cosine histogram on demand.
            self.hist_widget.update_image(o3d.geometry.Image(cosine_histogram_image(cone)))
            self.hist_title.text = "cos(theta) histogram | -1 to +1"
            self.hist_legend.text = "Gold in | Pink out | Cyan Q10 | Green mean cos"

        refinement_text = ""
        if "refined_xyz_mm" in entry or "refined_point_index" in entry:
            original_q10 = None
            try:
                _, original_indices, _ = self.analysis_tree.search_radius_vector_3d(
                    self._closest_analysis_point(p0)[1], float(analysis_radius))
                original_q10 = analyze_cone(
                    self._closest_analysis_point(p0)[1],
                    self.analysis_xyz[np.asarray(original_indices, dtype=np.int64)]).cosine_q10
            except ValueError:
                pass
            shift = float(np.linalg.norm(p - p0))
            steps = max(0, len(refinement_path_xyz(entry, self.xyz)) - 1)
            before = f"{original_q10:+.4f}" if original_q10 is not None else "undefined"
            after = f"{cone.cosine_q10:+.4f}" if cone is not None else "undefined"
            saved_radius = entry.get("refinement", {}).get("radius_mm")
            saved_voxel = entry.get("refinement", {}).get("voxel_size_mm")
            note = (" (re-optimize for radius/voxel size)"
                    if saved_radius != self.radius_mm or saved_voxel != self.analysis_voxel_mm
                    else "")
            refinement_text = (
                f"\nINITIAL (red): {np.round(p0, 2).tolist()}\n"
                f"REFINED (lime): {np.round(p, 2).tolist()}"
                + (" [snapped to analysis cloud]\n")
                + f"Refinement steps: {steps}\n"
                + f"Net shift from original: {shift:.2f} mm\n"
                f"Q10 at R={analysis_radius:.1f} mm: {before} -> {after} {note}\n"
            )
            if entry.get("refinement", {}).get("boundary_hit"):
                refinement_text += "WARNING: optimum near search boundary\n"
        else:
            refinement_text = "\nNo position refinement (click Refine Q10).\n"
        g = result.vector_mm
        plane_state = ("defined" if result.normal is not None
                       else "undefined (centroid vector nearly zero)")
        self.info.text = (
            f"{entry['id']} | analysis R {analysis_radius:.1f} mm (fixed)\n"
            + refinement_text +
            f"Analysis voxel size: {self.analysis_voxel_mm:g} mm\n"
            f"Downsampled neighbors (at query): {count:,}\n"
            f"Centroid offset g (mm):\n"
            f"[{g[0]:+.2f}, {g[1]:+.2f}, {g[2]:+.2f}]\n"
            f"|g|: {result.vector_length_mm:.3f} mm\n"
            f"Perpendicular plane: {plane_state}\n\n"
            + cone_text + spherical_text
        )
        # Refined PCA eigenvectors are persisted with the point and do not
        # disappear when a field is invalidated after position refinement.
        self._draw_refined_eigen_axes()
        # The eigen-field is anchored to the original PCA refinement origin,
        # not the moved point. Preserve its heatmap after refinement and when
        # switching between the two independent eigen metrics.
        if self.eigen_field is not None:
            field_origin = self._eigen_refinement_origin(entry)
            _, snapped = self._closest_analysis_point(field_origin)
            if np.linalg.norm(self.eigen_field.center_mm - snapped) < max(0.2, self.analysis_voxel_mm*.1):
                self._draw_eigen_field_overlay()
            else:
                self._invalidate_eigen_field()
        self._draw_scene_lambda1_overlay()
        self._draw_extrema_compare_overlay()
        self._workspace_changed(self.panel.selected_tab_index)
        self.scene_widget.force_redraw()

    # ------------------------- RealSense capture -------------------------
    def _toggle_camera(self):
        if self.camera_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if rs is None:
            self._error("pyrealsense2 missing in this Python environment. --demo / --open work without it.")
            return
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self._status("Previous camera thread is still stopping")
            return
        self.camera_stop_event = threading.Event()
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_active = True
        self.camera_btn.text = "Stop camera"
        self.camera_thread.start()
        self._status("Starting RealSense camera...")

    def _stop_camera(self):
        self.camera_stop_event.set()
        self.camera_active = False
        self.camera_btn.text = "Start camera"
        self.capture_btn.enabled = False
        self._status("Stopping RealSense camera")

    def _camera_worker(self):
        pipeline = None
        stop_event = self.camera_stop_event
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            last_display = 0.0
            while not stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                frames = align.process(frames)
                dframe, cframe = frames.get_depth_frame(), frames.get_color_frame()
                if not dframe or not cframe:
                    continue
                rgb = np.asanyarray(cframe.get_data()).copy()
                depth = np.asanyarray(dframe.get_data()).copy()
                intr = cframe.profile.as_video_stream_profile().intrinsics
                intrinsics = {
                    "fx": float(intr.fx), "fy": float(intr.fy),
                    "ppx": float(intr.ppx), "ppy": float(intr.ppy),
                    "width": int(intr.width), "height": int(intr.height),
                }
                with self.frame_lock:
                    self.latest_frame = (rgb, depth, intrinsics, depth_scale)
                now = time.monotonic()
                if now - last_display >= 0.2:
                    last_display = now
                    preview = np.ascontiguousarray(rgb[::2, ::2])
                    def update(preview_data=preview):
                        if self.camera_active and not stop_event.is_set():
                            self.preview.update_image(o3d.geometry.Image(preview_data))
                            self.capture_btn.enabled = True
                    self.app.post_to_main_thread(self.win, update)
        except Exception as exc:
            message = f"RealSense camera failed: {exc}"
            def notify():
                self._error(message)
            self.app.post_to_main_thread(self.win, notify)
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            def finish():
                if self.camera_stop_event is stop_event:
                    self.camera_active = False
                    self.camera_btn.text = "Start camera"
                    self.capture_btn.enabled = False
            try:
                self.app.post_to_main_thread(self.win, finish)
            except Exception:
                pass

    def _capture(self):
        with self.frame_lock:
            frame = self.latest_frame
            if frame is not None:
                color, depth, intrinsics, depth_scale = frame
                color, depth = color.copy(), depth.copy()
                intrinsics = dict(intrinsics)
        if frame is None:
            self._status("No RealSense frame available yet")
            return
        try:
            xyz, rgb = projected_cloud(depth, color, intrinsics, depth_scale)
            folder = CAPTURES / datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
            metadata = {
                "mode": "realsense", "units": "mm", "depth_scale_m": depth_scale,
                "intrinsics": intrinsics, "alignment": "depth_to_color",
                "depth_filter_m": [MIN_DEPTH_M, MAX_DEPTH_M],
            }
            save_cloud(folder, xyz, rgb, metadata)
            # Keep synchronized RGB + raw depth for future reconstruction.
            np.savez_compressed(folder / "frames.npz", color_rgb=color, depth_raw=depth)
            export_binary_ply(folder / "cloud.ply", xyz, rgb)
            self._open_dataset(folder)
            self._status(f"Captured and saved: {folder}")
        except Exception as exc:
            self._error(exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Open synthetic 3-plane corner, no camera")
    parser.add_argument("--camera", action="store_true", help="Start RealSense on startup")
    parser.add_argument("--open", dest="open_path", help="Open cloud.npz, points.npz or a saved scene folder")
    parser.add_argument("--scene", type=int, help="Dataset scene number, e.g. 2 opens scene_0002")
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).resolve().parent / "corner_dataset_tool" / "dataset",
                        help="Dataset root (default: corner_dataset_tool/dataset beside this script)")
    parser.add_argument('--dataset-region', choices=('scene', 'roi'), default='scene',
                        help='Open the complete dataset scene or only its saved ROI')
    args = parser.parse_args(argv)
    if args.demo and args.dataset_region == 'roi':
        parser.error('--dataset-region roi requires a saved dataset, not --demo')
    if args.scene is not None:
        if args.demo or args.open_path:
            parser.error("--scene cannot be combined with --demo or --open")
        try:
            args.open_path = str(resolve_dataset_scene(args.dataset, args.scene))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    print("Open3D version:", o3d.__version__)
    print("Mean axis = normalize(sum(unit directions)); Q10=P10(cos(theta)), alpha90=acos(Q10).")
    print("Spherical plane RANSAC: automatic K up to 3, origin constrained n.u=0, angular threshold, Huber re-fit, overlap retained.")
    print("PCL-style max angular gap after largest-component filtering, per plane (multi-membership retained).")
    print("MaxGap validator: dominant angular component; rejects isolated angle groups before max-gap test.")
    print("All geometry: global voxel-downsampled cloud; refinement candidates are voxel centroids.")
    print("Refine Q10: neighboring voxel points, maximize Q10, >=80% support.")
    print("Heatmap/NMS: choose one criterion; its heatmap and extrema stay paired. Corner/PCA/Input tabs preserve existing tools.")
    gui.Application.instance.initialize()
    window = CornerLab(args.open_path, args.demo, args.camera, roi_only=args.dataset_region == 'roi')
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
