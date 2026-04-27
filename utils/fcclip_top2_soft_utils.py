from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Top2SoftArtifact:
    top1_id: np.ndarray
    top2_id: np.ndarray
    p1: np.ndarray
    p2: np.ndarray


def top2_soft_artifact_path(mask_like_path: str | Path) -> Path:
    path = Path(mask_like_path)
    if path.suffix.lower() != '.png':
        raise ValueError(f'expected .png mask path, got {path}')
    return path.with_suffix('.npz')


def _as_uint8_map(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.uint8)
    if value.ndim != 2:
        raise ValueError(f'{name} must be a 2D array, got shape {value.shape}')
    return value


def _as_probability_map(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f'{name} must be a 2D array, got shape {value.shape}')
    if np.isnan(value).any():
        raise ValueError(f'{name} contains NaN values')
    return value


def _validate_top2_soft_artifact(
    *,
    top1_id: np.ndarray,
    top2_id: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> Top2SoftArtifact:
    artifact = Top2SoftArtifact(
        top1_id=_as_uint8_map(top1_id, name='top1_id'),
        top2_id=_as_uint8_map(top2_id, name='top2_id'),
        p1=_as_probability_map(p1, name='p1'),
        p2=_as_probability_map(p2, name='p2'),
    )
    shape = artifact.top1_id.shape
    if artifact.top2_id.shape != shape or artifact.p1.shape != shape or artifact.p2.shape != shape:
        raise ValueError('top-2 soft artifact arrays must have the same shape')
    return artifact


def save_top2_soft_artifact(
    path: str | Path,
    *,
    top1_id: np.ndarray,
    top2_id: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> None:
    artifact = _validate_top2_soft_artifact(top1_id=top1_id, top2_id=top2_id, p1=p1, p2=p2)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        top1_id=artifact.top1_id,
        top2_id=artifact.top2_id,
        p1=artifact.p1,
        p2=artifact.p2,
    )


def load_top2_soft_artifact(path: str | Path) -> Top2SoftArtifact:
    with np.load(Path(path)) as payload:
        return _validate_top2_soft_artifact(
            top1_id=payload['top1_id'],
            top2_id=payload['top2_id'],
            p1=payload['p1'],
            p2=payload['p2'],
        )
