from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fcclip_confidence_utils import (
    dequantize_confidence_map,
    quantize_confidence_map,
    sem_seg_to_mask_and_confidence,
    threshold_mask,
)


def test_sem_seg_to_mask_and_confidence_extracts_top1_probabilities():
    sem_seg = torch.tensor(
        [
            [[3.0, 1.0], [0.0, 0.0]],
            [[1.0, 2.0], [0.0, 3.0]],
            [[0.0, 0.0], [5.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    mask, confidence = sem_seg_to_mask_and_confidence(sem_seg)

    assert mask.shape == (2, 2)
    assert confidence.shape == (2, 2)
    assert mask.tolist() == [[0, 1], [2, 1]]
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)
    assert np.isclose(confidence[1, 0], torch.softmax(sem_seg[:, 1, 0], dim=0).max().item())


def test_threshold_mask_applies_ignore_label_from_confidence():
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    confidence = np.array([[0.9, 0.49], [1.0, 0.0]], dtype=np.float32)

    thresholded = threshold_mask(mask, confidence, threshold=0.5, ignore_label=255)

    assert thresholded.dtype == np.uint8
    assert thresholded.tolist() == [[0, 255], [2, 255]]


def test_quantize_confidence_roundtrip_stays_within_one_byte_precision():
    confidence = np.array([[0.0, 0.125], [0.5, 0.999]], dtype=np.float32)

    quantized = quantize_confidence_map(confidence)
    restored = dequantize_confidence_map(quantized)

    assert quantized.dtype == np.uint8
    assert restored.shape == confidence.shape
    assert np.max(np.abs(restored - np.clip(confidence, 0.0, 1.0))) <= (1.0 / 255.0 + 1e-6)
