"""Run author-code corner baselines, optionally compare GT and methods in one window."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .dataset_io import SceneData
    from .scene_navigation import scene_sort_key
    from .baselines.setup import ROOT, SOURCES
    from .baselines.config import BASELINE_METHODS, AUTHOR_AHMED, DATASET_AHMED, SCENE2_AHMED, AUTHOR_SOMMER, DATASET_SOMMER, AhmedConfig
else:
    from dataset_io import SceneData
    from scene_navigation import scene_sort_key
    from baselines.setup import ROOT, SOURCES
    from baselines.config import BASELINE_METHODS, AUTHOR_AHMED, DATASET_AHMED, SCENE2_AHMED, AUTHOR_SOMMER, DATASET_SOMMER, AhmedConfig


def select_input(scene: SceneData, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode not in ("roi", "scene"):
        raise ValueError(f"Unknown input mode: {mode}")
    if mode == "roi" and not (scene.root / "annotations.json").is_file():
        raise ValueError(f"ROI mode requires saved annotations.json: {scene.root}")
    indices = scene.roi_indices() if mode == "roi" else np.arange(len(scene.xyz_mm), dtype=np.int64)
    if not len(indices):
        raise ValueError(f"Empty {mode} input: {scene.root}")
    return scene.xyz_mm[indices].astype(np.float64), indices


def load_csv(path: Path, columns: int) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)  # missing output is a backend failure, not no detections
        rows = list(reader)
    expected = ["x_m", "y_m", "z_m"]
    if columns == 4:
        expected += ["response"]
    elif columns == 7:
        expected += ["nx", "ny", "nz", "input_index"]
    if header != expected:
        raise ValueError(f"Unexpected backend CSV header: {path}")
    if any(len(row) != columns for row in rows):
        raise ValueError(f"Malformed backend output: {path}")
    values = np.asarray(rows, dtype=np.float64).reshape(-1, columns)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite backend output: {path}")
    return values


def voxel_representatives(points_mm, voxel_mm):
    """Select the observed point nearest each cell centroid, keeping raw IDs exact."""
    if voxel_mm == 0:
        return np.arange(len(points_mm), dtype=np.int64)
    cells = np.floor((points_mm - points_mm.min(axis=0)) / voxel_mm).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    means = np.column_stack([np.bincount(inverse, weights=points_mm[:, j]) / counts for j in range(3)])
    squared = np.sum((points_mm - means[inverse]) ** 2, axis=1)
    order = np.lexsort((np.arange(len(points_mm)), squared, inverse))
    return order[np.r_[True, np.diff(inverse[order]) != 0]]


def write_oriented_ply(path: Path, normals: np.ndarray) -> None:
    """Author tinyply reader requires float32 XYZ and normals in metres."""
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(normals)}\n"
              + "".join(f"property float {key}\n" for key in ("x", "y", "z", "nx", "ny", "nz"))
              + "end_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(np.asarray(normals[:, :6], dtype="<f4").tobytes())


def save_predictions(directory: Path, predictions_m: np.ndarray, points_mm: np.ndarray,
                     raw_indices: np.ndarray, response: np.ndarray | None = None) -> int:
    xyz_mm = predictions_m * 1000.0
    if xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3 or not np.isfinite(xyz_mm).all():
        raise ValueError("Predictions must be finite (K, 3) coordinates")
    distance, local = cKDTree(points_mm).query(xyz_mm)
    nearest_raw = raw_indices[local]
    if response is None:
        response = np.full(len(xyz_mm), np.nan)
    np.savez_compressed(directory / "predictions.npz", xyz_mm=xyz_mm, response=response,
                        nearest_raw_point_index=nearest_raw, nearest_raw_distance_mm=distance)
    with (directory / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_mm", "y_mm", "z_mm", "response", "nearest_raw_point_index", "nearest_raw_distance_mm"])
        for p, score, ix, dist in zip(xyz_mm, response, nearest_raw, distance):
            writer.writerow([*p, score, int(ix), dist])
    return len(xyz_mm)


def run_command(command: list[str], log: Path, timeout: float) -> None:
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def matlab_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def prepare_commands(args, method: str, directory: Path, input_file: Path) -> list[list[str]]:
    output = directory / "backend_corners_m.csv"
    if method == "harris3d":
        return [[str(ROOT / "build/harris3d/pcl_baseline"), "harris", str(input_file), str(output),
                 str(args.harris_radius_mm / 1000), str(args.harris_threshold), str(int(args.harris_refine))]]
    if method == "orthogonal":
        return [
            [str(ROOT / "build/orthogonal/pcl_baseline"), "normals",
             str(directory / "sampled_xyz_m.txt" if args.sommer_voxel_mm > 0 else input_file),
             str(directory / "normals_m.csv"), str(args.normal_radius_mm / 1000)],
            [str(ROOT / "build/orthogonal/orthogonal_baseline"), "--img", str(directory / "oriented_m.ply"),
             "--output", str(output), "--dmin", str(args.dmin_mm / 1000), "--dmax", str(args.dmax_mm / 1000),
             "--min-votes", str(args.min_votes), "--sampling", str(args.sampling), "--seed", str(args.seed)],
        ]
    if args.ahmed_backend == "python":
        return [[sys.executable, str(ROOT / "ahmed.py"), str(input_file), str(output),
                 "--level", str(args.ahmed_level), "--seed", str(args.seed),
                 "--voxel-mm", str(args.ahmed_voxel_mm), "--edge-neighbors", str(args.ahmed_edge_neighbors),
                 "--edge-threshold", str(args.ahmed_edge_threshold), "--thinning-neighbors", str(args.ahmed_thinning_neighbors),
                 "--corner-neighbors", str(args.ahmed_corner_neighbors),
                 "--pca-neighbors", str(args.ahmed_pca_neighbors), "--span-mm", str(args.ahmed_span_mm),
                 "--angle-min-deg", str(args.ahmed_angle_min_deg), "--angle-max-deg", str(args.ahmed_angle_max_deg)]]
    expression = (f"addpath({matlab_string(ROOT)}); run_ahmed({matlab_string(input_file)},"
                  f"{matlab_string(output)},{matlab_string(ROOT / 'build/ahmed')},"
                  f"{args.ahmed_level},{args.seed});")
    return [[args.matlab, "-batch", expression]]


def run_method(args, method: str, directory: Path, input_file: Path,
               points_mm: np.ndarray, raw_indices: np.ndarray) -> dict:
    directory.mkdir()
    commands = prepare_commands(args, method, directory, input_file)
    meta = {"method": method, "status": "prepared", "commands": commands,
            "parameters": {k: v for k, v in vars(args).items() if k not in ("input", "output", "scene", "method")},
            "unit": "mm", "input_points": len(points_mm), "input_selection": "../input_selection.npz"}
    if method in SOURCES:
        url, revision = SOURCES[method]
        meta["source"] = {"url": url, "commit": revision}
        if method == "ahmed":
            meta["source"]["implementation"] = "python_port" if args.ahmed_backend == "python" else "matlab_author"
            meta["source"]["matlab_parity_verified"] = False
            if args.ahmed_backend == "python":
                meta["source"]["port_sha256"] = hashlib.sha256((ROOT / "ahmed.py").read_bytes()).hexdigest()
    else:
        meta["source"] = {"url": "https://github.com/PointCloudLibrary/pcl", "class": "pcl::HarrisKeypoint3D"}
    meta["input_sha256"] = hashlib.sha256(input_file.read_bytes()).hexdigest()
    start = time.perf_counter()
    try:
        normal_raw_indices = raw_indices
        if method == "orthogonal" and args.sommer_voxel_mm > 0:
            selected = voxel_representatives(points_mm, args.sommer_voxel_mm)
            normal_raw_indices = raw_indices[selected]
            np.save(directory / "sampled_raw_indices.npy", normal_raw_indices)
            np.savetxt(directory / "sampled_xyz_m.txt", points_mm[selected] / 1000, fmt="%.17g")
            meta["sampled_points"] = len(selected)
        if not args.prepare_only:
            if method != "ahmed":
                meta["source"]["pcl_version"] = subprocess.check_output([commands[0][0], "--version"], text=True).strip()
            for i, command in enumerate(commands):
                run_command(command, directory / f"stage_{i + 1}.log", args.timeout)
                if method == "orthogonal" and i == 0:
                    normals = load_csv(directory / "normals_m.csv", 7)
                    if not len(normals):
                        raise ValueError("No finite normals; increase --normal-radius-mm or inspect input")
                    indices = normals[:, 6].astype(np.int64)
                    if (not np.array_equal(normals[:, 6], indices) or np.any(indices < 0)
                            or np.any(indices >= len(normal_raw_indices))):
                        raise ValueError("Invalid normal support indices")
                    np.save(directory / "normal_raw_indices.npy", normal_raw_indices[indices])
                    meta["oriented_points"] = len(normals)
                    meta["normal_viewpoint_m"] = [0, 0, 0]
                    write_oriented_ply(directory / "oriented_m.ply", normals)
            data = load_csv(directory / "backend_corners_m.csv", 4 if method == "harris3d" else 3)
            meta["prediction_count"] = save_predictions(directory, data[:, :3], points_mm, raw_indices,
                                                         data[:, 3] if method == "harris3d" else None)
            meta["status"] = "ok"
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as exc:
        meta.update(status="failed", error=str(exc))
    meta["elapsed_seconds"] = time.perf_counter() - start
    (directory / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="One scene directory or dataset root")
    parser.add_argument("--method", nargs="+", choices=BASELINE_METHODS, default=list(BASELINE_METHODS),
                        help="Methods/order; defaults to Sommer / Ahmed / Harris3D; visualization always includes all three")
    parser.add_argument("--mode", choices=["roi", "scene"], default="roi", help="Saved AABB crop or complete raw scene")
    parser.add_argument("--preset", choices=["dataset", "author", "scene2"], default="dataset",
                        help="Dataset, original author demo, or scene_0002 ROI tuning (not held-out evaluation)")
    parser.add_argument("--scene", action="append", help="Scene name/glob under dataset root; repeatable")
    parser.add_argument("--output", help="New output directory; never overwrites an existing run")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs/commands without requiring backend runtimes")
    parser.add_argument("--visualize", action="store_true", help="After each scene show GT and all three baselines in one synchronized window")
    parser.add_argument("--timeout", type=float, default=600, help="Timeout per backend stage, seconds")
    parser.add_argument("--harris-radius-mm", type=float, default=20)
    parser.add_argument("--harris-threshold", type=float, default=1e-6)
    parser.add_argument("--no-harris-refine", dest="harris_refine", action="store_false", default=True)
    parser.add_argument("--normal-radius-mm", type=float, help="PCL normal support for Sommer input")
    parser.add_argument("--dmin-mm", type=float, help="Sommer minimum pair distance")
    parser.add_argument("--dmax-mm", type=float, help="Sommer maximum pair distance")
    parser.add_argument("--min-votes", type=int)
    parser.add_argument("--sampling", type=int, help="Sommer refinement stride; >=2 avoids zero filtering stride")
    parser.add_argument("--sommer-voxel-mm", type=float, help="Sommer input sampling; 0 disables it")
    parser.add_argument("--ahmed-level", type=int, choices=range(1, 11), help="Export L level; scene2 defaults to 1, others 10; all ten retained in MAT")
    parser.add_argument("--ahmed-backend", choices=["python", "matlab"], default="python", help="Ahmed Python port (default, no MATLAB) or original MATLAB code")
    parser.add_argument("--ahmed-voxel-mm", type=float)
    parser.add_argument("--ahmed-edge-neighbors", type=int)
    parser.add_argument("--ahmed-edge-threshold", type=float)
    parser.add_argument("--ahmed-thinning-neighbors", type=int)
    parser.add_argument("--ahmed-corner-neighbors", type=int)
    parser.add_argument("--ahmed-pca-neighbors", type=int)
    parser.add_argument("--ahmed-span-mm", type=float)
    parser.add_argument("--ahmed-angle-min-deg", type=float)
    parser.add_argument("--ahmed-angle-max-deg", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable (Ahmed)")
    args = parser.parse_args(argv)
    if args.visualize:
        args.method = list(dict.fromkeys([*args.method, *BASELINE_METHODS]))
    sommer = AUTHOR_SOMMER if args.preset == "author" else DATASET_SOMMER
    for name, value in sommer.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.preset == "scene2" and args.ahmed_backend == "matlab" and "ahmed" in args.method:
        parser.error("--preset scene2 requires --ahmed-backend python for Ahmed")
    if args.ahmed_level is None:
        args.ahmed_level = 1 if args.preset == "scene2" else 10
    ahmed = {"dataset": DATASET_AHMED, "author": AUTHOR_AHMED, "scene2": SCENE2_AHMED}[args.preset]
    if args.ahmed_backend == "matlab":
        ahmed = AUTHOR_AHMED
    ahmed_names = ("voxel_mm", "edge_neighbors", "edge_threshold", "thinning_neighbors", "corner_neighbors",
                   "pca_neighbors", "span_mm", "angle_min_deg", "angle_max_deg")
    if args.ahmed_backend == "matlab" and any(getattr(args, "ahmed_" + name) is not None for name in ahmed_names):
        parser.error("Ahmed numeric overrides require --ahmed-backend python; MATLAB retains author settings")
    for name in ahmed_names:
        if getattr(args, "ahmed_" + name) is None:
            setattr(args, "ahmed_" + name, getattr(ahmed, name))
    try:
        AhmedConfig(**{name: getattr(args, "ahmed_" + name) for name in ahmed_names})
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.sommer_voxel_mm) or args.sommer_voxel_mm < 0:
        parser.error("--sommer-voxel-mm must be finite and nonnegative")
    if args.prepare_only and args.visualize:
        parser.error("--visualize requires computed results; omit --prepare-only")
    for name in ("harris_radius_mm", "normal_radius_mm", "dmin_mm", "dmax_mm", "timeout"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if (not np.isfinite(args.harris_threshold) or args.dmin_mm >= args.dmax_mm
            or args.min_votes < 1 or args.sampling < 2 or not 0 <= args.seed < 2**32):
        parser.error("Require finite Harris threshold, dmin < dmax, min-votes >=1, sampling >=2 and uint32 seed")
    source = Path(args.input).expanduser().resolve()
    if (source / "points.npz").is_file():
        scenes = [source]
    elif source.is_dir():
        scenes = sorted((p for p in source.iterdir() if p.is_dir() and (p / "points.npz").is_file()), key=scene_sort_key)
    else:
        parser.error(f"Input directory not found: {source}")
    if args.scene:
        scenes = [p for p in scenes if any(p.match(pattern) for pattern in args.scene)]
    if not scenes:
        parser.error("No matching scenes with points.npz")
    methods = list(dict.fromkeys(args.method))
    if not args.prepare_only:
        for method in methods:
            if method == "ahmed":
                if args.ahmed_backend == "matlab" and (not shutil.which(args.matlab) or not (ROOT / "build/ahmed/corner_detection.m").is_file()):
                    parser.error("Ahmed requires MATLAB and setup.py --method ahmed; see baselines/README.md")
            else:
                needed = [ROOT / f"build/{method}/pcl_baseline"]
                if method == "orthogonal":
                    needed.append(ROOT / "build/orthogonal/orthogonal_baseline")
                if any(not p.is_file() for p in needed):
                    parser.error(f"Run python3 corner_dataset_tool/baselines/setup.py --method {method} first")
    output = (Path(args.output) if args.output else Path(__file__).resolve().parent / "baseline_results" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).expanduser().resolve()
    # Results must never be written into the immutable input scene/dataset.
    if output == source or source in output.parents or any(output == p or p in output.parents for p in scenes):
        parser.error("--output must be outside the input scene/dataset")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    manifest = {"schema_version": 1, "input": str(source), "mode": args.mode, "unit": "mm",
                "created_utc": datetime.now(timezone.utc).isoformat(), "results": []}
    failed = False
    for path in scenes:
        try:
            scene = SceneData.load(path)
            points_mm, raw_indices = select_input(scene, args.mode)
            directory = output / path.name / args.mode
            directory.mkdir(parents=True)
            input_file = directory / "input_xyz_m.txt"
            np.savetxt(input_file, points_mm / 1000, fmt="%.17g")
            inside = scene.roi_mask()
            gt = [c for c in scene.corners if args.mode == "scene" or inside[c["raw_point_index"]]]
            gt_raw = np.array([c["raw_point_index"] for c in gt], dtype=np.int64)
            np.savez_compressed(directory / "input_selection.npz", raw_point_indices=raw_indices,
                                gt_raw_point_indices=gt_raw, gt_xyz_mm=scene.xyz_mm[gt_raw],
                                gt_ids=np.array([c["id"] for c in gt], dtype=np.int64))
            for method in methods:
                result = run_method(args, method, directory / method, input_file, points_mm, raw_indices)
                result.update(scene=str(path), output=str(directory / method), gt_count=len(gt))
                manifest["results"].append(result)
                failed |= result["status"] == "failed"
                print(f"{path.name}/{args.mode}/{method}: {result['status']} ({result.get('prediction_count', '-')} corners)", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            failed = True
            manifest["results"].append({"scene": str(path), "status": "failed", "error": str(exc)})
            print(f"{path.name}: failed: {exc}", flush=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.visualize and any(r.get("scene") == str(path) and "method" in r for r in manifest["results"]):
            try:
                if __package__:
                    from .view_baselines import show_run
                else:
                    from view_baselines import show_run
                show_run(output, methods=methods, scenes=[path.name])
            except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
                print(f"Visualization failed: {exc}. Saved results: {output}", flush=True)
                return 1
    print(f"Results: {output}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
