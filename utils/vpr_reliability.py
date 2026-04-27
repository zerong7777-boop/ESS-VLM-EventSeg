from __future__ import annotations

from typing import Sequence

import torch


def normalize_class_reliability(
    raw_reliability: Sequence[float] | torch.Tensor,
    *,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    raw = torch.as_tensor(raw_reliability, dtype=torch.float32)
    if raw.ndim != 1:
        raise ValueError(f"raw_reliability must be 1-D, got shape {tuple(raw.shape)}")
    if raw.numel() == 0:
        raise ValueError("raw_reliability must contain at least one class")
    if min_weight > max_weight:
        raise ValueError("expected min_weight <= max_weight")

    raw_min = torch.min(raw)
    raw_max = torch.max(raw)
    if torch.isclose(raw_min, raw_max):
        return torch.full_like(raw, fill_value=max_weight)

    normalized = (raw - raw_min) / (raw_max - raw_min)
    return min_weight + normalized * (max_weight - min_weight)


def normalize_confidence_map(
    confidence: torch.Tensor | Sequence[float],
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    conf = torch.as_tensor(confidence, dtype=torch.float32)
    if conf.numel() == 0:
        raise ValueError("confidence map must contain at least one value")
    if min_weight > max_weight:
        raise ValueError("expected min_weight <= max_weight")

    conf_min = torch.min(conf)
    conf_max = torch.max(conf)
    if torch.isclose(conf_min, conf_max):
        return torch.full_like(conf, fill_value=min_weight)

    normalized = (conf - conf_min) / (conf_max - conf_min + eps)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return min_weight + normalized * (max_weight - min_weight)


def class_reliability_map(
    labels: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
    *,
    ignore_index: int,
) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=labels.device)
    if weights.ndim != 1:
        raise ValueError(f"class_weights must be 1-D, got shape {tuple(weights.shape)}")
    if weights.numel() == 0:
        raise ValueError("class_weights must contain at least one class")

    safe_labels = labels.clamp(min=0, max=weights.numel() - 1)
    weight_map = weights[safe_labels]
    return weight_map.masked_fill(labels == ignore_index, 0.0)
