"""Inject real RedKitchen repeatability results into the inline 3-D fragment."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "outputs/depth_repeatability_redkitchen/repeatable-corners-3d.npz"
SUMMARY = ROOT / "outputs/depth_repeatability_redkitchen/summary.json"
SEQUENCE = Path("/home/choisuhyun/datasets/7scenes/redkitchen/seq-03.zip")
OUTPUT = Path(
    "/home/choisuhyun/.codex/visualizations/2026/10/09/"
    "01a11f67-d2df-7b11-9fb1-5c165e6b5e3c/normal-cone-repeatability-3d.html"
)
PLACEHOLDER = "__NORMAL_CONE_REPEATABILITY_DATA__"


def main() -> None:
    saved = np.load(RESULT)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    points = np.asarray(saved["points_camera_mm"], dtype=np.float64)
    scores = np.asarray(saved["scores"], dtype=np.float64)
    frame = int(saved["frame"])
    method_ids = {
        "score": np.asarray(saved["score_candidate_indices"], dtype=np.int64),
        "scale": np.asarray(saved["scale_candidate_indices"], dtype=np.int64),
        "sensor": np.asarray(saved["perturbation_candidate_indices"], dtype=np.int64),
        "view": np.asarray(saved["viewpoint_candidate_indices"], dtype=np.int64),
        "final": np.asarray(saved["final_candidate_indices"], dtype=np.int64),
    }
    final_support = np.asarray(saved["final_view_support"], dtype=np.int16)
    support_by_index = {
        int(index): int(support)
        for index, support in zip(method_ids["final"], final_support)
    }

    with zipfile.ZipFile(SEQUENCE) as source:
        rgb = np.asarray(
            Image.open(
                io.BytesIO(
                    source.read(f"seq-03/frame-{frame:06d}.color.png")
                )
            ).convert("RGB")
        )

    center = np.median(points, axis=0)

    def transform(values: np.ndarray) -> np.ndarray:
        shifted = (values - center) / 1000.0
        return np.column_stack((shifted[:, 0], -shifted[:, 1], -shifted[:, 2]))

    transformed = transform(points)
    sample_ids = np.arange(0, len(points), 4, dtype=np.int64)
    sampled = transformed[sample_ids]
    sampled_source = points[sample_ids]
    z = np.maximum(sampled_source[:, 2], 1e-9)
    columns = np.clip(np.rint(585.0 * sampled_source[:, 0] / z + 320.0), 0, 639).astype(int)
    rows = np.clip(np.rint(585.0 * sampled_source[:, 1] / z + 240.0), 0, 479).astype(int)
    colors = rgb[rows, columns]

    def records(indices: np.ndarray) -> list[dict[str, object]]:
        return [
            {
                "p": np.round(transformed[index], 5).tolist(),
                "xyz": np.round(points[index], 1).tolist(),
                "i": int(index),
                "s": round(float(scores[index]), 4),
                "v": support_by_index.get(int(index), 1),
            }
            for index in np.asarray(indices, dtype=np.int64)
        ]

    score_set = set(method_ids["score"].tolist())
    final_set = set(method_ids["final"].tolist())
    overlap = np.asarray(sorted(score_set & final_set), dtype=np.int64)
    score_only = np.asarray(sorted(score_set - final_set), dtype=np.int64)
    final_only = np.asarray(sorted(final_set - score_set), dtype=np.int64)
    support_groups = {
        f"support{support}": method_ids["final"][final_support == support]
        for support in (1, 2, 3)
    }

    minimum = np.percentile(transformed, 1.0, axis=0)
    maximum = np.percentile(transformed, 99.0, axis=0)
    span = maximum - minimum
    aggregate = summary["aggregate_results"]
    data = {
        "meta": {
            "frame": f"{frame:06d}",
            "voxelCount": int(len(points)),
            "renderPointCount": int(len(sample_ids)),
            "budget": int(summary["evaluation"]["common_fixed_budget"]),
            "bounds": {
                "min": np.round(minimum, 4).tolist(),
                "max": np.round(maximum, 4).tolist(),
            },
            "gridSize": round(float(max(span[0], span[1]) * 1.12), 3),
        },
        "metrics": {
            "score": aggregate["score_persistence"]["repeatability"],
            "scale": aggregate["scale_max_track"]["repeatability"],
            "sensor": aggregate["sensor_perturbation"]["repeatability"],
            "view": aggregate["viewpoint_resampling"]["repeatability"],
            "final": aggregate["view_contrast"]["repeatability"],
        },
        "cloud": {
            "positions": np.round(sampled, 4).ravel().tolist(),
            "colors": colors.astype(int).ravel().tolist(),
        },
        "corners": {
            **{name: records(indices) for name, indices in method_ids.items()},
            "scoreOnly": records(score_only),
            "finalOnly": records(final_only),
            "overlap": records(overlap),
            **{name: records(indices) for name, indices in support_groups.items()},
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
            raise RuntimeError("Could not restore the visualization placeholder")
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
                "budget": data["meta"]["budget"],
                "final_support": {
                    str(value): int(np.count_nonzero(final_support == value))
                    for value in (1, 2, 3)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
