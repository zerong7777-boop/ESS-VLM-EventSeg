from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

SPLIT_NAMES = ('train', 'valid', 'test')


def discover_split_dirs(dataset_root: Path, split: str) -> List[Path]:
    data_dirs = sorted(Path(path) for path in glob.glob(str(dataset_root / 'dir*')))
    if not data_dirs:
        raise FileNotFoundError(f'no DDD17 directories found under {dataset_root}')
    split_map = {
        'train': [data_dirs[0], data_dirs[2], data_dirs[3], data_dirs[5], data_dirs[6]],
        'valid': [data_dirs[1]],
        'test': [data_dirs[4]],
    }
    return split_map[split]


def resolve_output_root(output_root: Path, split: str) -> Path:
    if output_root.name == split:
        return output_root
    if output_root.name in SPLIT_NAMES and output_root.name != split:
        raise ValueError(f'output root {output_root} conflicts with requested split {split}')
    return output_root / split


def iter_reference_masks(sequence_dir: Path, max_images_per_sequence: Optional[int]) -> List[Path]:
    mask_files = sorted((sequence_dir / 'segmentation_masks').glob('*.png'))
    if not mask_files:
        raise FileNotFoundError(f'no segmentation masks found under {sequence_dir / "segmentation_masks"}')
    if max_images_per_sequence is not None:
        mask_files = mask_files[:max_images_per_sequence]
    return mask_files


def resolve_frame_path(sequence_dir: Path, segmentation_mask_path: Path) -> Path:
    stem = segmentation_mask_path.stem
    if not stem.startswith('segmentation_'):
        raise ValueError(f'unexpected DDD17 mask basename: {segmentation_mask_path.name}')
    frame_index = int(stem.split('_')[-1])
    candidates = [
        sequence_dir / 'imgs' / f'img_{frame_index:08d}.png',
        sequence_dir / 'imgs' / f'{frame_index:010d}.png',
        sequence_dir / 'imgs' / f'{frame_index:08d}.png',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'could not resolve frame for {segmentation_mask_path.name} in {sequence_dir / "imgs"}'
    )


def read_mask_shape(mask_path: Path) -> tuple[int, int]:
    with Image.open(mask_path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f'expected single-channel mask at {mask_path}, got ndim={array.ndim}')
    return tuple(int(value) for value in array.shape)


def align_prediction_to_reference(prediction: np.ndarray, reference_mask_path: Path) -> np.ndarray:
    prediction_array = np.asarray(prediction)
    reference_shape = read_mask_shape(reference_mask_path)
    if tuple(prediction_array.shape) == reference_shape:
        return prediction_array

    pred_height, pred_width = prediction_array.shape
    ref_height, ref_width = reference_shape
    if pred_width != ref_width:
        raise ValueError(
            f'prediction width {pred_width} does not match reference width {ref_width} for {reference_mask_path}'
        )
    if pred_height < ref_height:
        raise ValueError(
            f'prediction height {pred_height} is smaller than reference height {ref_height} for {reference_mask_path}'
        )

    cropped = prediction_array[:ref_height, :]
    if tuple(cropped.shape) != reference_shape:
        raise ValueError(
            f'failed to align prediction shape {prediction_array.shape} to reference {reference_shape} '
            f'for {reference_mask_path}'
        )
    return cropped


def output_artifact_path(split_output_root: Path, sequence_dir: Path, reference_mask_path: Path) -> Path:
    return split_output_root / sequence_dir.name / 'segmentation_masks' / reference_mask_path.name
