from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sem_seg_to_top2_soft_maps(sem_seg: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sem_seg.ndim != 3:
        raise ValueError(f"expected sem_seg with shape [C, H, W], got {tuple(sem_seg.shape)}")
    if sem_seg.shape[0] < 2:
        raise ValueError(f'expected at least 2 classes in sem_seg, got {sem_seg.shape[0]}')

    probabilities = torch.softmax(sem_seg, dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, k=2, dim=0)
    return (
        top_indices[0].to(torch.uint8).cpu().numpy(),
        top_indices[1].to(torch.uint8).cpu().numpy(),
        top_probabilities[0].to(torch.float32).cpu().numpy(),
        top_probabilities[1].to(torch.float32).cpu().numpy(),
    )


def sem_seg_to_mask_and_confidence(sem_seg: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if sem_seg.ndim != 3:
        raise ValueError(f"expected sem_seg with shape [C, H, W], got {tuple(sem_seg.shape)}")

    probabilities = torch.softmax(sem_seg, dim=0)
    confidence, class_index = probabilities.max(dim=0)
    return class_index.to(torch.uint8).cpu().numpy(), confidence.to(torch.float32).cpu().numpy()


def threshold_mask(
    mask: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float,
    ignore_label: int = 255,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must be in [0, 1]')

    dense_mask = np.asarray(mask, dtype=np.uint8)
    confidence_map = np.asarray(confidence, dtype=np.float32)
    if dense_mask.shape != confidence_map.shape:
        raise ValueError(
            f'mask and confidence shapes must match, got {dense_mask.shape} and {confidence_map.shape}'
        )

    thresholded = dense_mask.copy()
    thresholded[confidence_map < threshold] = ignore_label
    return thresholded


def quantize_confidence_map(confidence: np.ndarray) -> np.ndarray:
    confidence_map = np.asarray(confidence, dtype=np.float32)
    if confidence_map.ndim != 2:
        raise ValueError(f'confidence map must be 2D, got shape {confidence_map.shape}')
    if np.isnan(confidence_map).any():
        raise ValueError('confidence map contains NaN values')

    clipped = np.clip(confidence_map, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def dequantize_confidence_map(confidence_u8: np.ndarray) -> np.ndarray:
    quantized = np.asarray(confidence_u8, dtype=np.uint8)
    if quantized.ndim != 2:
        raise ValueError(f'quantized confidence map must be 2D, got shape {quantized.shape}')
    return quantized.astype(np.float32) / 255.0


def save_confidence_map(path: str | Path, confidence: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantize_confidence_map(confidence), mode='L').save(output_path)


def load_confidence_map(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        quantized = np.asarray(image, dtype=np.uint8)
    if quantized.ndim != 2:
        raise ValueError('confidence image must be single-channel')
    return dequantize_confidence_map(quantized)
