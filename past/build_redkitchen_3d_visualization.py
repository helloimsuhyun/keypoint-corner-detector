"""Inject compact RedKitchen point/corner data into the inline 3-D template."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
NPZ = ROOT / "outputs/geometry_scale_research/redkitchen-geometry-transitions.npz"
SEQUENCE = Path("/home/choisuhyun/datasets/7scenes/redkitchen/seq-03.zip")
OUTPUT = Path(
    "/home/choisuhyun/.codex/visualizations/2026/10/09/"
    "01a11f67-d2df-7b11-9fb1-5c165e6b5e3c/redkitchen-corners-canvas-3d.html"
)
PLACEHOLDER = "__REDKITCHEN_DATA__"


def main() -> None:
    archive = np.load(NPZ)
    points = archive["points_mm"].astype(np.float64)
    score_ids = archive["score_candidate_indices"].astype(np.int64)
    geometry_ids = archive["geometry_candidate_indices"].astype(np.int64)
    scale_scores = archive["scale_scores"].astype(np.float64)
    selected_radius = archive["selected_radius_mm"].astype(np.float64)

    with zipfile.ZipFile(SEQUENCE) as source:
        rgb = np.asarray(
            Image.open(
                io.BytesIO(source.read("seq-03/frame-000740.color.png"))
            ).convert("RGB")
        )

    center = np.median(points, axis=0)

    def transform(values: np.ndarray) -> np.ndarray:
        shifted = (values - center) / 1000.0
        return np.column_stack((shifted[:, 0], -shifted[:, 1], -shifted[:, 2]))

    transformed = transform(points)
    sample_ids = np.arange(0, len(points), 3, dtype=np.int64)
    sampled = transformed[sample_ids]
    sampled_source = points[sample_ids]
    z = np.maximum(sampled_source[:, 2], 1e-9)
    columns = np.clip(np.rint(585.0 * sampled_source[:, 0] / z + 320.0), 0, 639).astype(int)
    rows = np.clip(np.rint(585.0 * sampled_source[:, 1] / z + 240.0), 0, 479).astype(int)
    colors = rgb[rows, columns]

    score_set = set(score_ids.tolist())
    geometry_set = set(geometry_ids.tolist())
    overlap = sorted(score_set & geometry_set)
    score_only = sorted(score_set - geometry_set)
    geometry_only = sorted(geometry_set - score_set)

    def records(indices: list[int]) -> list[dict[str, object]]:
        result = []
        for index in indices:
            radius = selected_radius[index]
            result.append(
                {
                    "p": np.round(transformed[index], 5).tolist(),
                    "xyz": np.round(points[index], 1).tolist(),
                    "i": int(index),
                    "s": round(float(np.median(scale_scores[:, index])), 4),
                    "r": round(float(radius), 1) if np.isfinite(radius) else None,
                }
            )
        return result

    minimum = np.percentile(transformed, 1.0, axis=0)
    maximum = np.percentile(transformed, 99.0, axis=0)
    span = maximum - minimum
    data = {
        "meta": {
            "frame": "000740",
            "rawVoxelCount": int(len(points)),
            "renderPointCount": int(len(sample_ids)),
            "scoreCount": int(len(score_ids)),
            "geometryCount": int(len(geometry_ids)),
            "overlapCount": int(len(overlap)),
            "scoreOnlyCount": int(len(score_only)),
            "geometryOnlyCount": int(len(geometry_only)),
            "bounds": {
                "min": np.round(minimum, 4).tolist(),
                "max": np.round(maximum, 4).tolist(),
            },
            "gridSize": round(float(max(span[0], span[1]) * 1.12), 3),
            "cameraDistance": round(float(np.linalg.norm(span) * 0.78), 3),
        },
        "cloud": {
            "positions": np.round(sampled, 4).ravel().tolist(),
            "colors": colors.astype(int).ravel().tolist(),
        },
        "corners": {
            "scoreOnly": records(score_only),
            "geometryOnly": records(geometry_only),
            "overlap": records(overlap),
        },
    }

    source = OUTPUT.read_text(encoding="utf-8")
    if PLACEHOLDER not in source:
        source, replacements = re.subn(
            r"const DATA = \{.*?\};\n\n    const resolveColor",
            f"const DATA = {PLACEHOLDER};\n\n    const resolveColor",
            source,
            count=1,
            flags=re.DOTALL,
        )
        if replacements != 1:
            raise RuntimeError("Could not restore the visualization data placeholder")
    rendered = source.replace(
        PLACEHOLDER,
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    )
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "render_points": len(sample_ids),
                "score": len(score_ids),
                "geometry": len(geometry_ids),
                "overlap": len(overlap),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
