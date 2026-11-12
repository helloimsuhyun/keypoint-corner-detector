import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_io import SceneData


@pytest.fixture
def scene_data(tmp_path):
    root = tmp_path / "scene_0001"
    root.mkdir()
    xyz = np.array([[0, 0, 500], [10, 0, 500], [0, 20, 500], [0, 0, 550], [100, 100, 900]], dtype=np.float32)
    colors = np.tile(np.array([100, 150, 200], dtype=np.uint8), (5, 1))
    np.savez_compressed(root / "points.npz", xyz_mm=xyz, rgb=colors, pixel_index=np.arange(5, dtype=np.int32) + 100)
    return root


def test_multiple_gt_roundtrip(scene_data):
    m = SceneData.load(scene_data)
    assert m.add_corner(1) == 1
    assert m.add_corner(3) == 2
    assert m.add_corner(1) is None
    m.save_annotations()
    n = SceneData.load(scene_data)
    assert [(c["id"], c["raw_point_index"]) for c in n.corners] == [(1, 1), (2, 3)]
    assert n.next_id == 3
    payload = json.loads((scene_data / "annotations.json").read_text())
    assert payload["corners"][1]["xyz_mm"] == [0, 0, 550]
    assert payload["corners"][1]["pixel_index"] == 103


def test_non_destructive_roi_preserves_gt(scene_data):
    m = SceneData.load(scene_data)
    m.add_corner(4)  # background
    m.add_corner(0)
    m.set_roi([-1, -1, 499], [25, 25, 555])
    assert m.roi_indices().tolist() == [0, 1, 2, 3]
    m.save_annotations()
    back = SceneData.load(scene_data)
    assert len(back.xyz_mm) == 5
    assert {c["raw_point_index"] for c in back.corners} == {0, 4}
    assert back.to_json()["corners"][0]["inside_roi"] is False
    assert back.to_json()["corners"][1]["inside_roi"] is True
    back.set_roi([-1, -1, 499], [101, 101, 901])
    assert back.roi_indices().tolist() == [0, 1, 2, 3, 4]
    assert back.corners[0]["raw_point_index"] == 4


def test_delete_and_monotonic_ids(scene_data):
    m = SceneData.load(scene_data)
    m.add_corner(0)
    m.add_corner(1)
    assert m.remove_corner(1)
    assert m.add_corner(2) == 3
    assert [c["id"] for c in m.corners] == [2, 3]
    assert not m.remove_corner(999)


def test_invalid_roi_and_tampered_gt_rejected(scene_data):
    m = SceneData.load(scene_data)
    with pytest.raises(ValueError):
        m.set_roi([0, 0, 0], [0, 20, 20])
    m.add_corner(1)
    m.save_annotations()
    payload = json.loads((scene_data / "annotations.json").read_text())
    payload["corners"][0]["xyz_mm"] = [999, 999, 999]
    (scene_data / "annotations.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch"):
        SceneData.load(scene_data)
