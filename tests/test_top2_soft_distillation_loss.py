import torch

from utils.top2_soft_distillation import top2_soft_cross_entropy


def test_top2_soft_cross_entropy_normalizes_teacher_mass_over_top2():
    logits = torch.tensor([[[[2.0]], [[1.0]], [[0.0]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0]]], dtype=torch.long)
    top2 = torch.tensor([[[1]]], dtype=torch.long)
    p1 = torch.tensor([[[0.8]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.1]]], dtype=torch.float32)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2)

    log_probs = torch.log_softmax(logits, dim=1)
    expected = -((0.8 / 0.9) * log_probs[0, 0, 0, 0] + (0.1 / 0.9) * log_probs[0, 1, 0, 0])
    assert torch.isclose(actual, expected)


def test_top2_soft_cross_entropy_temperature_softens_teacher_distribution():
    logits = torch.tensor([[[[2.0]], [[0.0]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0]]], dtype=torch.long)
    top2 = torch.tensor([[[1]]], dtype=torch.long)
    p1 = torch.tensor([[[0.8]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.2]]], dtype=torch.float32)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, temperature=2.0)

    teacher_logits = torch.stack([p1.log(), p2.log()], dim=0)
    teacher_probs = torch.softmax(teacher_logits / 2.0, dim=0)
    log_probs = torch.log_softmax(logits, dim=1)
    expected = -(
        teacher_probs[0, 0, 0, 0] * log_probs[0, 0, 0, 0]
        + teacher_probs[1, 0, 0, 0] * log_probs[0, 1, 0, 0]
    )
    default_loss = top2_soft_cross_entropy(logits, top1, top2, p1, p2)
    assert torch.isclose(actual, expected)
    assert actual > default_loss


def test_top2_soft_cross_entropy_ignores_invalid_pixels():
    logits = torch.tensor([[[[3.0, 1.0]], [[0.2, 2.0]], [[0.1, 0.3]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0, 1]]], dtype=torch.long)
    top2 = torch.tensor([[[1, 0]]], dtype=torch.long)
    p1 = torch.tensor([[[0.7, 0.6]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.2, 0.3]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[1, 0]]], dtype=torch.bool)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, valid_mask=valid_mask)
    reference = top2_soft_cross_entropy(logits[:, :, :, :1], top1[:, :, :1], top2[:, :, :1], p1[:, :, :1], p2[:, :, :1])
    assert torch.isclose(actual, reference)


def test_top2_soft_cross_entropy_min_p1_gates_low_confidence_pixels():
    logits = torch.tensor([[[[3.0, 0.1]], [[0.2, 2.0]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0, 1]]], dtype=torch.long)
    top2 = torch.tensor([[[1, 0]]], dtype=torch.long)
    p1 = torch.tensor([[[0.8, 0.4]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.1, 0.3]]], dtype=torch.float32)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, min_p1=0.5)
    reference = top2_soft_cross_entropy(
        logits[:, :, :, :1],
        top1[:, :, :1],
        top2[:, :, :1],
        p1[:, :, :1],
        p2[:, :, :1],
    )
    assert torch.isclose(actual, reference)


def test_top2_soft_cross_entropy_min_margin_gates_ambiguous_pixels():
    logits = torch.tensor([[[[3.0, 0.1]], [[0.2, 2.0]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0, 1]]], dtype=torch.long)
    top2 = torch.tensor([[[1, 0]]], dtype=torch.long)
    p1 = torch.tensor([[[0.8, 0.55]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.1, 0.5]]], dtype=torch.float32)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, min_margin=0.1)
    reference = top2_soft_cross_entropy(
        logits[:, :, :, :1],
        top1[:, :, :1],
        top2[:, :, :1],
        p1[:, :, :1],
        p2[:, :, :1],
    )
    assert torch.isclose(actual, reference)


def test_top2_soft_cross_entropy_combines_original_valid_mask_with_gate():
    logits = torch.tensor([[[[3.0, 0.1, 1.0]], [[0.2, 2.0, 0.8]]]], dtype=torch.float32)
    top1 = torch.tensor([[[0, 1, 0]]], dtype=torch.long)
    top2 = torch.tensor([[[1, 0, 1]]], dtype=torch.long)
    p1 = torch.tensor([[[0.8, 0.7, 0.4]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[0, 1, 1]]], dtype=torch.bool)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, valid_mask=valid_mask, min_p1=0.5)
    reference = top2_soft_cross_entropy(
        logits[:, :, :, 1:2],
        top1[:, :, 1:2],
        top2[:, :, 1:2],
        p1[:, :, 1:2],
        p2[:, :, 1:2],
    )
    assert torch.isclose(actual, reference)


def test_top2_soft_cross_entropy_returns_zero_with_grad_when_all_pixels_gated_out():
    logits = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32, requires_grad=True)
    top1 = torch.tensor([[[0]]], dtype=torch.long)
    top2 = torch.tensor([[[1]]], dtype=torch.long)
    p1 = torch.tensor([[[0.4]]], dtype=torch.float32)
    p2 = torch.tensor([[[0.3]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[0]]], dtype=torch.bool)

    actual = top2_soft_cross_entropy(logits, top1, top2, p1, p2, valid_mask=valid_mask, min_p1=0.5)
    assert torch.isclose(actual, torch.tensor(0.0))
    actual.backward()
    assert logits.grad is not None
