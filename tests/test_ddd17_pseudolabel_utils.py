from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.ddd17_pseudolabel_utils import (
    build_pseudolabel_path,
    load_pseudolabel_mask,
    remap_to_ddd17_six_classes,
)


def test_build_pseudolabel_path_maps_segmentation_mask_to_split_dir_layout():
    segmentation_mask_path = Path(
        "/data/DDD17/scene_0001/dir0/segmentation_masks/frame_000123.png"
    )
    pseudolabel_root = Path("/data/DDD17/pseudolabels")

    expected = pseudolabel_root / "train" / "dir0" / "segmentation_masks" / "frame_000123.png"

    assert build_pseudolabel_path(segmentation_mask_path, pseudolabel_root) == expected


def test_load_pseudolabel_mask_returns_int64_array(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1], [2, 255]], dtype=np.uint8)).save(mask_path)

    mask = load_pseudolabel_mask(mask_path)

    assert mask.dtype == np.int64
    assert mask.tolist() == [[0, 1], [2, 255]]


def test_load_pseudolabel_mask_rejects_non_2d_masks(tmp_path):
    mask_path = tmp_path / "mask_rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(mask_path)

    with pytest.raises(ValueError, match="2D"):
        load_pseudolabel_mask(mask_path)


def test_remap_to_ddd17_six_classes_sets_unmapped_ids_to_ignore():
    mask = np.array([[0, 1], [2, 7], [255, 3]], dtype=np.int64)
    lut = {0: 10, 1: 11, 2: 12, 255: 0}

    remapped = remap_to_ddd17_six_classes(mask, lut)

    assert remapped.dtype == np.int64
    assert remapped.tolist() == [[10, 11], [12, 255], [255, 255]]


def test_remap_to_ddd17_six_classes_supports_ndarray_lut_branch():
    mask = np.array([[0, 1], [2, 3], [255, 4]], dtype=np.int64)
    lut = np.array([10, 11, 12, 13], dtype=np.int64)

    remapped = remap_to_ddd17_six_classes(mask, lut)

    assert remapped.dtype == np.int64
    assert remapped.tolist() == [[10, 11], [12, 13], [255, 255]]