from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fcclip_top2_soft_utils import (
    load_top2_soft_artifact,
    save_top2_soft_artifact,
    top2_soft_artifact_path,
)
from utils.fcclip_confidence_utils import sem_seg_to_top2_soft_maps


def test_top2_soft_artifact_path_swaps_png_suffix_for_npz():
    mask_path = Path('/tmp/fcclip_top2_soft_cropped/train/dir2/segmentation_masks/segmentation_00000042.png')

    artifact_path = top2_soft_artifact_path(mask_path)

    assert artifact_path == Path(
        '/tmp/fcclip_top2_soft_cropped/train/dir2/segmentation_masks/segmentation_00000042.npz'
    )


def test_top2_soft_artifact_roundtrip_preserves_expected_arrays(tmp_path: Path):
    artifact_path = tmp_path / 'train' / 'dir1' / 'segmentation_masks' / 'segmentation_00000001.npz'
    top1_id = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    top2_id = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    p1 = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    p2 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    save_top2_soft_artifact(artifact_path, top1_id=top1_id, top2_id=top2_id, p1=p1, p2=p2)
    artifact = load_top2_soft_artifact(artifact_path)

    assert artifact_path.is_file()
    np.testing.assert_array_equal(artifact.top1_id, top1_id)
    np.testing.assert_array_equal(artifact.top2_id, top2_id)
    np.testing.assert_allclose(artifact.p1, p1)
    np.testing.assert_allclose(artifact.p2, p2)


def test_save_top2_soft_artifact_rejects_shape_mismatch(tmp_path: Path):
    artifact_path = tmp_path / 'bad.npz'
    top1_id = np.zeros((2, 2), dtype=np.uint8)
    top2_id = np.zeros((2, 2), dtype=np.uint8)
    p1 = np.zeros((2, 2), dtype=np.float32)
    p2 = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match='same shape'):
        save_top2_soft_artifact(artifact_path, top1_id=top1_id, top2_id=top2_id, p1=p1, p2=p2)


def test_sem_seg_to_top2_soft_maps_extracts_ordered_top2_predictions():
    sem_seg = torch.tensor(
        [
            [[3.0, 1.0], [0.0, 0.0]],
            [[1.0, 2.0], [4.0, 3.0]],
            [[0.0, 0.0], [5.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    top1_id, top2_id, p1, p2 = sem_seg_to_top2_soft_maps(sem_seg)

    assert top1_id.tolist() == [[0, 1], [2, 1]]
    assert top2_id.tolist() == [[1, 0], [1, 2]]
    assert np.all(p1 >= p2)
    np.testing.assert_allclose(
        p1[1, 0],
        torch.softmax(sem_seg[:, 1, 0], dim=0).topk(k=2).values[0].item(),
    )
    np.testing.assert_allclose(
        p2[1, 0],
        torch.softmax(sem_seg[:, 1, 0], dim=0).topk(k=2).values[1].item(),
    )
