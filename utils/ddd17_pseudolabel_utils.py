from pathlib import Path

import numpy as np
from PIL import Image

_DDD17_SEQUENCE_SPLITS = {
    "dir0": "train",
    "dir3": "train",
    "dir4": "train",
    "dir6": "train",
    "dir7": "train",
    "dir1": "valid",
}


def build_pseudolabel_path(segmentation_mask_path, pseudolabel_root):
    segmentation_mask_path = Path(segmentation_mask_path)
    pseudolabel_root = Path(pseudolabel_root)

    if segmentation_mask_path.parent.name != "segmentation_masks":
        raise ValueError("segmentation mask path must end in dirX/segmentation_masks/<filename>")

    sequence_dir = segmentation_mask_path.parent.parent.name
    try:
        split = _DDD17_SEQUENCE_SPLITS[sequence_dir]
    except KeyError as exc:
        raise ValueError(f"unsupported DDD17 sequence directory: {sequence_dir}") from exc

    return pseudolabel_root / split / sequence_dir / "segmentation_masks" / segmentation_mask_path.name


def load_pseudolabel_mask(mask_path):
    with Image.open(mask_path) as image:
        mask = np.asarray(image, dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError("pseudolabel mask must be 2D")
    return mask


def remap_to_ddd17_six_classes(mask, lut, ignore_label=255):
    mask = np.asarray(mask, dtype=np.int64)
    remapped = np.full(mask.shape, ignore_label, dtype=np.int64)
    valid = mask != ignore_label

    if hasattr(lut, "items"):
        for source_id, target_id in lut.items():
            if source_id == ignore_label:
                continue
            remapped[mask == source_id] = target_id
    else:
        lut = np.asarray(lut, dtype=np.int64)
        valid &= (mask >= 0) & (mask < lut.shape[0])
        remapped[valid] = lut[mask[valid]]

    remapped[mask == ignore_label] = ignore_label
    return remapped