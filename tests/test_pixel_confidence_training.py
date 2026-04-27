import torch
import torch.nn.functional as f

from utils.pixel_confidence_training import confidence_weighted_cross_entropy


def test_confidence_weighted_cross_entropy_matches_manual_reduction():
    logits = torch.tensor(
        [[[[2.0, 0.2], [0.4, 1.2]], [[0.1, 1.5], [1.6, 0.3]]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [1, 255]]], dtype=torch.long)
    confidence = torch.tensor([[[1.0, 0.5], [0.25, 0.9]]], dtype=torch.float32)

    actual = confidence_weighted_cross_entropy(logits, labels, confidence, ignore_index=255)

    ce = f.cross_entropy(logits, labels, ignore_index=255, reduction='none')
    weights = confidence * (labels != 255).float()
    expected = (ce * weights).sum() / weights.sum()

    assert torch.isclose(actual, expected)


def test_confidence_weighted_cross_entropy_ignores_ignore_index_pixels():
    logits = torch.tensor([[[[3.0]], [[0.1]]]], dtype=torch.float32)
    labels = torch.tensor([[[255]]], dtype=torch.long)
    confidence = torch.tensor([[[0.7]]], dtype=torch.float32)

    actual = confidence_weighted_cross_entropy(logits, labels, confidence, ignore_index=255)

    assert torch.isclose(actual, torch.tensor(0.0))
