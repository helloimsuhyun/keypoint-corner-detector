#!/usr/bin/env python3
"""
Generate synthetic depth + visibility point sets from a CAD mesh.

Pipeline
--------
CAD mesh
  -> camera poses sampled on a sphere / upper hemisphere
  -> pinhole ray casting
  -> synthetic depth
  -> visibility point set with state:
       0 = FREE
       1 = SURFACE
       2 = UNKNOWN / OCCLUDED
       3 = INVALID  (reserved; not generated in the ideal renderer)
  -> one .npz per view + metadata.json

Important convention
--------------------
A ray that misses the object is NOT 'invalid'.
Within the local ROI it provides FREE-space evidence.
INVALID is reserved for later D435-like sensor dropout simulation.

Dependencies
------------
    pip install numpy open3d

Example
-------
python cad_visibility_dataset.py \
    --mesh /home/choisuhyun/keypoint-corner-detector/data/laser_pose_cross.stl \
    --out runs/visibility_obj \
    --num-views 256 \
    --hemisphere full \
    --width 160 \
    --height 120 \
    --hfov-deg 69 \
    --camera-radius-scale 2.5 \
    --roi-margin-scale 0.15 \
    --num-rays 12000 \
    --n-free 2 \
    --n-unknown 2 \
    --save-depth \
    --visualize \
    --visualize-view 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


FREE = 0
SURFACE = 1
UNKNOWN = 2
INVALID = 3
STATE_NAMES = ["free", "surface", "unknown", "invalid"]


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def fibonacci_directions(n: int, hemisphere: str = "upper") -> np.ndarray:
    """Approximately uniform directions on a sphere / upper hemisphere."""
    if n <= 0:
        raise ValueError("num_views must be > 0")

    i = np.arange(n, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * i

    if hemisphere == "full":
        z = 1.0 - 2.0 * (i + 0.5) / n
    elif hemisphere == "upper":
        # z in (0, 1): upper hemisphere relative to mesh center
        z = (i + 0.5) / n
    else:
        raise ValueError("hemisphere must be 'upper' or 'full'")

    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs.astype(np.float32)


def look_at_T_wc(camera_pos: np.ndarray,
                 target: np.ndarray,
                 world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Camera-to-world transform.

    Camera convention:
      +x = image right
      +y = image down
      +z = optical forward
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = normalize(target - camera_pos)
    if abs(float(np.dot(forward, normalize(world_up)))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    down = -up

    R_wc = np.column_stack([right, down, forward])
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = camera_pos
    return T_wc


def make_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = 0.5 * width / math.tan(0.5 * hfov)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def make_camera_rays(width: int,
                     height: int,
                     K: np.ndarray,
                     T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ray origins Nx3, normalized world directions Nx3, pixel uv Nx2."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    x = (u.reshape(-1) - cx) / fx
    y = (v.reshape(-1) - cy) / fy
    z = np.ones_like(x)

    dirs_cam = np.stack([x, y, z], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    R_wc = T_wc[:3, :3]
    dirs_world = dirs_cam @ R_wc.T
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)

    origin = T_wc[:3, 3]
    origins = np.repeat(origin[None, :], len(dirs_world), axis=0)
    uv = np.stack([u.reshape(-1), v.reshape(-1)], axis=1).astype(np.int32)
    return origins.astype(np.float32), dirs_world.astype(np.float32), uv


def ray_aabb_intersection(origins: np.ndarray,
                          dirs: np.ndarray,
                          bmin: np.ndarray,
                          bmax: np.ndarray,
                          eps: float = 1e-10):
    """Vectorized ray-AABB intersection. Returns valid, t_enter, t_exit."""
    n = len(origins)
    t_enter = np.full(n, -np.inf, dtype=np.float64)
    t_exit = np.full(n, np.inf, dtype=np.float64)
    invalid = np.zeros(n, dtype=bool)

    O = origins.astype(np.float64)
    D = dirs.astype(np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)

    for a in range(3):
        da = D[:, a]
        oa = O[:, a]
        parallel = np.abs(da) < eps
        invalid |= parallel & ((oa < bmin[a]) | (oa > bmax[a]))

        non = ~parallel
        t1 = np.empty(n, dtype=np.float64)
        t2 = np.empty(n, dtype=np.float64)
        t1[parallel] = -np.inf
        t2[parallel] = np.inf
        t1[non] = (bmin[a] - oa[non]) / da[non]
        t2[non] = (bmax[a] - oa[non]) / da[non]

        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        t_enter = np.maximum(t_enter, lo)
        t_exit = np.minimum(t_exit, hi)

    t_enter = np.maximum(t_enter, 0.0)
    valid = (~invalid) & (t_exit >= t_enter) & (t_exit > 0.0)
    return valid, t_enter.astype(np.float32), t_exit.astype(np.float32)


def choose_ray_indices(rng: np.random.Generator,
                       roi_valid: np.ndarray,
                       hit_mask: np.ndarray,
                       max_rays: int,
                       hit_ratio: float = 0.7) -> np.ndarray:
    """Stratified ray sampling: keep both object-hit and free-space-only rays."""
    valid_idx = np.flatnonzero(roi_valid)
    if len(valid_idx) <= max_rays:
        return valid_idx

    hit_idx = np.flatnonzero(roi_valid & hit_mask)
    miss_idx = np.flatnonzero(roi_valid & (~hit_mask))

    n_hit_target = int(round(max_rays * hit_ratio))
    n_hit = min(n_hit_target, len(hit_idx))
    n_miss = min(max_rays - n_hit, len(miss_idx))

    # Fill remaining quota from whichever class has capacity.
    remain = max_rays - n_hit - n_miss
    if remain > 0:
        extra_hit = min(remain, len(hit_idx) - n_hit)
        n_hit += extra_hit
        remain -= extra_hit
    if remain > 0:
        n_miss += min(remain, len(miss_idx) - n_miss)

    a = rng.choice(hit_idx, size=n_hit, replace=False) if n_hit > 0 else np.empty(0, int)
    b = rng.choice(miss_idx, size=n_miss, replace=False) if n_miss > 0 else np.empty(0, int)
    out = np.concatenate([a, b])
    rng.shuffle(out)
    return out


def sample_interval(rng: np.random.Generator,
                    lo: float,
                    hi: float,
                    n: int,
                    jitter: bool = True) -> np.ndarray:
    if n <= 0 or hi <= lo:
        return np.empty((0,), dtype=np.float32)
    if jitter:
        # Stratified samples, random position inside each bin.
        edges = np.linspace(lo, hi, n + 1, dtype=np.float64)
        u = rng.random(n)
        t = edges[:-1] + u * (edges[1:] - edges[:-1])
    else:
        t = np.linspace(lo, hi, n + 2, dtype=np.float64)[1:-1]
    return t.astype(np.float32)


def build_visibility_pointset(
    rng: np.random.Generator,
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_indices: np.ndarray,
    t_hit: np.ndarray,
    t_enter: np.ndarray,
    t_exit: np.ndarray,
    surface_band: float,
    n_free: int,
    n_unknown: int,
    n_free_miss: int,
):
    points = []
    states = []
    src_ray = []

    for ridx in ray_indices:
        o = origins[ridx]
        d = dirs[ridx]
        te = float(t_enter[ridx])
        tx = float(t_exit[ridx])
        th = float(t_hit[ridx])
        hit = np.isfinite(th) and (te <= th <= tx)

        if hit:
            # FREE samples: before measured surface.
            free_hi = min(tx, th - surface_band)
            ts = sample_interval(rng, te, free_hi, n_free)
            if len(ts):
                points.append(o[None, :] + ts[:, None] * d[None, :])
                states.append(np.full(len(ts), FREE, dtype=np.int64))
                src_ray.append(np.full(len(ts), ridx, dtype=np.int32))

            # SURFACE sample: exact synthetic hit.
            p = o + th * d
            points.append(p[None, :])
            states.append(np.array([SURFACE], dtype=np.int64))
            src_ray.append(np.array([ridx], dtype=np.int32))

            # UNKNOWN samples: behind first return, inside ROI.
            unk_lo = max(te, th + surface_band)
            ts = sample_interval(rng, unk_lo, tx, n_unknown)
            if len(ts):
                points.append(o[None, :] + ts[:, None] * d[None, :])
                states.append(np.full(len(ts), UNKNOWN, dtype=np.int64))
                src_ray.append(np.full(len(ts), ridx, dtype=np.int32))
        else:
            # No mesh hit, but ray traverses ROI -> FREE evidence.
            ts = sample_interval(rng, te, tx, n_free_miss)
            if len(ts):
                points.append(o[None, :] + ts[:, None] * d[None, :])
                states.append(np.full(len(ts), FREE, dtype=np.int64))
                src_ray.append(np.full(len(ts), ridx, dtype=np.int32))

    if not points:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int32))

    return (np.concatenate(points, axis=0).astype(np.float32),
            np.concatenate(states, axis=0),
            np.concatenate(src_ray, axis=0))


def one_hot_states(states: np.ndarray) -> np.ndarray:
    feat = np.zeros((len(states), 4), dtype=np.float32)
    if len(states):
        feat[np.arange(len(states)), states] = 1.0
    return feat


def visualize_visibility_view(
    mesh_legacy: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    states: np.ndarray,
    camera_pos: np.ndarray,
    T_wc: np.ndarray,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    diag: float,
    view_id: int,
    point_size: float = 4.0,
):
    """Visualize one generated visibility point set in Open3D."""
    if len(points) == 0:
        print(f"[visualize] view {view_id}: no visibility points to display")
        return

    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[states == FREE] = [0.15, 0.55, 1.00]       # BLUE
    colors[states == SURFACE] = [1.00, 0.15, 0.15]    # RED
    colors[states == UNKNOWN] = [0.60, 0.60, 0.60]    # GRAY
    colors[states == INVALID] = [1.00, 0.80, 0.10]    # YELLOW

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_legacy)
    mesh_wire.paint_uniform_color([0.92, 0.92, 0.92])

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(diag * 0.08, 1e-6)
    )
    cam_frame.transform(T_wc.astype(np.float64))

    roi_box = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(roi_min, dtype=np.float64),
        np.asarray(roi_max, dtype=np.float64),
    )
    roi_box.color = [0.15, 0.95, 0.25]

    counts = np.bincount(states.astype(np.int64), minlength=4)
    print("\\n[Visualization legend]")
    print(f"  BLUE   = FREE     ({counts[FREE]})")
    print(f"  RED    = SURFACE  ({counts[SURFACE]})")
    print(f"  GRAY   = UNKNOWN  ({counts[UNKNOWN]})")
    print(f"  YELLOW = INVALID  ({counts[INVALID]})")
    print("  WHITE  = CAD mesh wireframe")
    print("  GREEN  = ROI bounding box")
    print(f"  Camera = coordinate frame at {np.asarray(camera_pos).round(4).tolist()}")
    print("  Close the Open3D window to continue.\\n")

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name=f"Visibility view {view_id:05d}",
        width=1280,
        height=800,
    )
    if not ok:
        print("[visualize] Open3D window could not be created. "
              "Check that a desktop/display server is available.")
        return

    vis.add_geometry(mesh_wire)
    vis.add_geometry(pcd)
    vis.add_geometry(cam_frame)
    vis.add_geometry(roi_box)

    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = max(1.0, float(point_size))
        opt.background_color = np.array([0.03, 0.03, 0.03], dtype=np.float64)

    vis.run()
    vis.destroy_window()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True, help="CAD mesh: STL/OBJ/PLY/etc.")
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--num-views", type=int, default=256)
    p.add_argument("--hemisphere", choices=["upper", "full"], default="upper")
    p.add_argument("--camera-radius", type=float, default=None,
                   help="Absolute radius in mesh units. Overrides --camera-radius-scale")
    p.add_argument("--camera-radius-scale", type=float, default=2.5,
                   help="radius = bbox_diagonal * scale")

    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--hfov-deg", type=float, default=69.0,
                   help="Horizontal FOV. D435 is roughly in this range depending on stream/mode.")

    p.add_argument("--roi-margin", type=float, default=None,
                   help="Absolute AABB margin in mesh units")
    p.add_argument("--roi-margin-scale", type=float, default=0.15,
                   help="margin = bbox_diagonal * scale")

    p.add_argument("--num-rays", type=int, default=12000,
                   help="Visibility rays sampled per view")
    p.add_argument("--hit-ray-ratio", type=float, default=0.70,
                   help="Approx fraction of selected rays that hit the mesh")
    p.add_argument("--n-free", type=int, default=2,
                   help="FREE samples on a hit ray")
    p.add_argument("--n-unknown", type=int, default=2,
                   help="UNKNOWN samples behind a hit")
    p.add_argument("--n-free-miss", type=int, default=3,
                   help="FREE samples on a ray traversing ROI but missing mesh")
    p.add_argument("--surface-band", type=float, default=None,
                   help="Do not sample FREE/UNKNOWN within this distance of the hit. "
                        "Default = 0.002 * bbox diagonal")

    p.add_argument("--save-depth", action="store_true",
                   help="Save full synthetic z-depth + range for every view")

    p.add_argument("--visualize", action="store_true",
                   help="Open an Open3D viewer for generated visibility points")
    p.add_argument("--visualize-view", type=int, default=0,
                   help="View ID to visualize. Default: 0. Use -1 for every view.")
    p.add_argument("--visualize-point-size", type=float, default=4.0,
                   help="Point size in the Open3D viewer")

    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)

    if args.visualize_view >= args.num_views:
        raise ValueError(
            f"--visualize-view={args.visualize_view} is outside "
            f"the generated range [0, {args.num_views - 1}]"
        )

    out_dir = Path(args.out)
    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    # Legacy mesh -> tensor mesh -> raycasting scene.
    mesh_legacy = o3d.io.read_triangle_mesh(args.mesh)
    if mesh_legacy.is_empty() or len(mesh_legacy.triangles) == 0:
        raise RuntimeError(f"Failed to load a triangle mesh: {args.mesh}")

    mesh_legacy.compute_vertex_normals()
    vertices = np.asarray(mesh_legacy.vertices, dtype=np.float64)
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin
    diag = float(np.linalg.norm(extent))
    if diag <= 0:
        raise RuntimeError("Degenerate mesh bounding box")

    radius = args.camera_radius if args.camera_radius is not None else diag * args.camera_radius_scale
    margin = args.roi_margin if args.roi_margin is not None else diag * args.roi_margin_scale
    roi_min = bmin - margin
    roi_max = bmax + margin
    surface_band = args.surface_band if args.surface_band is not None else diag * 0.002

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    K = make_intrinsics(args.width, args.height, args.hfov_deg)
    sphere_dirs = fibonacci_directions(args.num_views, args.hemisphere)

    metadata = {
        "mesh": str(Path(args.mesh).resolve()),
        "state_encoding": {str(i): name for i, name in enumerate(STATE_NAMES)},
        "num_views": args.num_views,
        "hemisphere": args.hemisphere,
        "mesh_bounds_min": bmin.tolist(),
        "mesh_bounds_max": bmax.tolist(),
        "mesh_center": center.tolist(),
        "bbox_diagonal": diag,
        "camera_radius": radius,
        "roi_min": roi_min.tolist(),
        "roi_max": roi_max.tolist(),
        "surface_band": surface_band,
        "image_width": args.width,
        "image_height": args.height,
        "hfov_deg": args.hfov_deg,
        "K": K.tolist(),
        "visibility_point_definition": {
            "free": "sampled before first surface return, or along ROI-traversing rays with no mesh hit",
            "surface": "first synthetic mesh hit",
            "unknown": "behind first surface return until ROI exit",
            "invalid": "reserved for later sensor-dropout simulation; not generated by ideal renderer",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = []

    for vid, dcam in enumerate(sphere_dirs):
        camera_pos = center + radius * dcam.astype(np.float64)
        T_wc = look_at_T_wc(camera_pos, center)
        origins, dirs, uv = make_camera_rays(args.width, args.height, K, T_wc)

        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy().reshape(-1).astype(np.float32)

        roi_valid, t_enter, t_exit = ray_aabb_intersection(origins, dirs, roi_min, roi_max)
        hit_mask = np.isfinite(t_hit) & roi_valid & (t_hit >= t_enter) & (t_hit <= t_exit)

        selected = choose_ray_indices(
            rng=rng,
            roi_valid=roi_valid,
            hit_mask=hit_mask,
            max_rays=args.num_rays,
            hit_ratio=args.hit_ray_ratio,
        )

        pts, states, src_ray = build_visibility_pointset(
            rng=rng,
            origins=origins,
            dirs=dirs,
            ray_indices=selected,
            t_hit=t_hit,
            t_enter=t_enter,
            t_exit=t_exit,
            surface_band=surface_band,
            n_free=args.n_free,
            n_unknown=args.n_unknown,
            n_free_miss=args.n_free_miss,
        )
        features = one_hot_states(states)

        payload = {
            "points": pts,
            "state": states.astype(np.uint8),
            "features": features,
            "source_ray_index": src_ray,
            "selected_ray_index": selected.astype(np.int32),
            "camera_pos": camera_pos.astype(np.float32),
            "T_wc": T_wc.astype(np.float32),
            "K": K.astype(np.float32),
            "roi_min": roi_min.astype(np.float32),
            "roi_max": roi_max.astype(np.float32),
        }

        if args.save_depth:
            # t_hit is Euclidean range because world ray directions are normalized.
            range_map = t_hit.reshape(args.height, args.width).copy()
            range_map[~np.isfinite(range_map)] = np.nan

            depth_z = np.full_like(t_hit, np.nan, dtype=np.float32)
            finite = np.isfinite(t_hit)
            if np.any(finite):
                pw = origins[finite] + t_hit[finite, None] * dirs[finite]
                R_wc = T_wc[:3, :3]
                pc = (pw - camera_pos[None, :]) @ R_wc
                depth_z[finite] = pc[:, 2].astype(np.float32)
            depth_z = depth_z.reshape(args.height, args.width)

            payload["range"] = range_map.astype(np.float32)
            payload["depth_z"] = depth_z.astype(np.float32)

        path = views_dir / f"view_{vid:05d}.npz"
        np.savez_compressed(path, **payload)

        counts = np.bincount(states, minlength=4) if len(states) else np.zeros(4, dtype=int)
        row = {
            "view_id": vid,
            "file": path.name,
            "camera_pos": camera_pos.tolist(),
            "num_points": int(len(pts)),
            "num_free": int(counts[FREE]),
            "num_surface": int(counts[SURFACE]),
            "num_unknown": int(counts[UNKNOWN]),
            "num_invalid": int(counts[INVALID]),
            "num_hit_rays": int(np.count_nonzero(hit_mask)),
            "num_roi_rays": int(np.count_nonzero(roi_valid)),
        }
        summary.append(row)

        if args.visualize and (args.visualize_view == -1 or vid == args.visualize_view):
            visualize_visibility_view(
                mesh_legacy=mesh_legacy,
                points=pts,
                states=states,
                camera_pos=camera_pos,
                T_wc=T_wc,
                roi_min=roi_min,
                roi_max=roi_max,
                diag=diag,
                view_id=vid,
                point_size=args.visualize_point_size,
            )

        if vid % max(1, args.num_views // 20) == 0 or vid == args.num_views - 1:
            print(
                f"[{vid+1:4d}/{args.num_views}] points={len(pts):6d} "
                f"free={counts[FREE]:6d} surf={counts[SURFACE]:6d} "
                f"unk={counts[UNKNOWN]:6d}"
            )

    (out_dir / "views.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Dataset written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()