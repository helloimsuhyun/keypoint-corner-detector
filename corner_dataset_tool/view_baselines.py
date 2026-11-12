"""View saved baseline results as GT / method 1 / method 2 / ... panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corner_dataset_tool.dataset_io import SceneData
from corner_dataset_tool.baselines.config import BASELINE_METHODS


def load_comparisons(run_dir, *, methods=None, scenes=None):
    """Read saved selection/GT/predictions; never rerun a detector or load live labels."""
    root = Path(run_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("unit") != "mm":
        raise ValueError("Baseline result unit must be mm")
    groups = {}
    for result in manifest["results"]:
        if "method" not in result:
            continue
        scene_path = Path(result["scene"])
        if scenes and not any(scene_path.match(pattern) for pattern in scenes):
            continue
        groups.setdefault(scene_path, {})[result["method"]] = result
    if not groups:
        raise ValueError("No saved scene results match the selection")
    for scene_path, results in groups.items():
        selected = list(dict.fromkeys([*(methods or ()), *BASELINE_METHODS]))
        if set(selected) - set(BASELINE_METHODS):
            raise ValueError("Unknown baseline method")
        directory = root / scene_path.name / manifest["mode"]
        with np.load(scene_path / "points.npz", allow_pickle=False) as raw:
            scene = SceneData(scene_path, raw["xyz_mm"], raw["rgb"], raw["pixel_index"])
        with np.load(directory / "input_selection.npz", allow_pickle=False) as data:
            indices = data["raw_point_indices"]
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or not len(indices) or np.any(indices < 0) or np.any(indices >= len(scene.xyz_mm))):
                raise ValueError("Invalid saved raw point indices")
            gt = data["gt_xyz_mm"].copy()
        panels = []
        for method in selected:
            record = results.get(method, {"status": "not_saved", "error":
                "No saved result; rerun run_baselines.py --visualize to compute all three baselines"})
            xyz = np.empty((0, 3))
            if record["status"] == "ok":
                with np.load(directory / method / "predictions.npz", allow_pickle=False) as predictions:
                    xyz = predictions["xyz_mm"].copy()
            panels.append({"method": method, "status": record["status"], "xyz_mm": xyz,
                           "implementation": record.get("source", {}).get("implementation", ""),
                           "error": record.get("error", "")})
        yield {"points_mm": scene.xyz_mm[indices], "rgb": scene.rgb[indices],
               "gt_mm": gt, "results": panels, "title": f"{scene_path.name} | {manifest['mode']} | mm"}


def show_run(run_dir, *, methods=None, scenes=None):
    from corner_detector.visualization.manager import show_baseline_comparison
    for comparison in load_comparisons(run_dir, methods=methods, scenes=scenes):
        show_baseline_comparison(**comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Saved run directory containing manifest.json")
    parser.add_argument("--method", nargs="+", choices=BASELINE_METHODS,
                        help="Panel order; missing baseline names are appended so all three are always shown")
    parser.add_argument("--scene", action="append", help="Scene name/glob; repeatable")
    args = parser.parse_args(argv)
    try:
        show_run(args.run_dir, methods=args.method, scenes=args.scene)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Visualization failed: {exc}\nUse an environment with Open3D, Matplotlib, Pillow and a desktop display.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
