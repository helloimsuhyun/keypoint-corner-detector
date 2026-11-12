import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_navigation import adjacent_scene, list_scene_folders, scene_sort_key
from dataset_io import SceneData
import numpy as np


def make_scene(parent, name):
    scene = parent / name
    scene.mkdir()
    xyz = np.array([[0, 0, 500], [2, 3, 510]], np.float32)
    rgb = np.zeros((2, 3), dtype=np.uint8)
    np.savez_compressed(scene / 'points.npz', xyz_mm=xyz, rgb=rgb,
                        pixel_index=np.arange(2, dtype=np.int32))
    return scene


def test_scene_number_order_and_skip_missing(tmp_path):
    s2 = make_scene(tmp_path, 'scene2')
    s9 = make_scene(tmp_path, 'scene9')
    s10 = make_scene(tmp_path, 'scene10')
    make_scene(tmp_path, 'demo_corner')
    (tmp_path / 'scene11').mkdir()  # missing points.npz must be ignored
    paths = list_scene_folders(s9)
    assert paths == [s2, s9, s10]
    assert adjacent_scene(paths, s9, +1) == s10
    assert adjacent_scene(paths, s9, -1) == s2
    assert adjacent_scene(paths, s10, +1) is None
    assert adjacent_scene(paths, s2, -1) is None


def test_scene_199_to_200_and_no_other_family(tmp_path):
    a = make_scene(tmp_path, 'scene199')
    b = make_scene(tmp_path, 'scene200')
    make_scene(tmp_path, 'scene_201')
    assert adjacent_scene(list_scene_folders(a), a, 1) == b


def test_zero_padded_order_and_saved_gt_survives_scene_change(tmp_path):
    a = make_scene(tmp_path, 'scene_0199')
    b = make_scene(tmp_path, 'scene_0200')
    m = SceneData.load(a)
    m.add_corner(1)
    m.set_roi([-2, -2, 498], [3, 4, 515])
    m.save_annotations()
    assert adjacent_scene(list_scene_folders(a), a, +1) == b
    n = SceneData.load(a)
    assert n.corners == [{'id': 1, 'raw_point_index': 1}]
    assert n.roi_min.tolist() == [-2, -2, 498]
    assert SceneData.load(b).corners == []


def test_bad_offset_or_missing_points(tmp_path):
    s1 = make_scene(tmp_path, 'scene1')
    with pytest.raises(ValueError, match='offset'):
        adjacent_scene(list_scene_folders(s1), s1, 3)
    empty = tmp_path / 'scene2'
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        list_scene_folders(empty)
