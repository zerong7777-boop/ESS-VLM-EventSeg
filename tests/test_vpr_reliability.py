from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.vpr_reliability import (
    class_reliability_map,
    normalize_class_reliability,
    normalize_confidence_map,
)


def test_normalize_class_reliability_keeps_values_in_requested_range():
    raw = [0.6898, 0.8795, 0.2027, 0.5317, 0.5046, 0.6551]
    weights = normalize_class_reliability(raw, min_weight=0.35, max_weight=1.0)

    assert weights.shape == (6,)
    assert torch.all(weights >= 0.35)
    assert torch.all(weights <= 1.0)
    assert weights[1] > weights[0] > weights[2]


def test_class_reliability_map_uses_ignore_zero_weight():
    labels = torch.tensor([[[0, 1, 2], [5, 255, 3]]], dtype=torch.long)
    class_weights = torch.tensor([0.7, 1.0, 0.35, 0.6, 0.6, 0.8], dtype=torch.float32)

    weight_map = class_reliability_map(labels, class_weights, ignore_index=255)

    assert weight_map.shape == labels.shape
    assert torch.isclose(weight_map[0, 0, 0], torch.tensor(0.7))
    assert torch.isclose(weight_map[0, 0, 1], torch.tensor(1.0))
    assert torch.isclose(weight_map[0, 0, 2], torch.tensor(0.35))
    assert torch.isclose(weight_map[0, 1, 1], torch.tensor(0.0))


def test_normalize_confidence_map_handles_range_and_constant_tensor():
    confidence = torch.tensor([[0.1, 0.2], [0.4, 0.3]], dtype=torch.float32)
    normalized = normalize_confidence_map(confidence, min_weight=0.25, max_weight=0.95)

    assert normalized.shape == confidence.shape
    assert torch.all(normalized >= 0.25)
    assert torch.all(normalized <= 0.95)
    assert normalized[0, 0] < normalized[1, 0]

    constant = torch.full((2, 2), 0.7, dtype=torch.float32)
    constant_norm = normalize_confidence_map(constant, min_weight=0.25, max_weight=0.95)

    assert constant_norm.shape == constant.shape
    assert torch.all(constant_norm == 0.25)
    assert not torch.isnan(constant_norm).any()
