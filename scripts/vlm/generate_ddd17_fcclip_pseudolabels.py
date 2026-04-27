from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from detectron2.data.detection_utils import read_image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ddd17_events_loader import get_split
from scripts.vlm.fcclip_dense_predictor import FCCLIPDensePredictor

SPLIT_NAMES = ("train", "valid", "test")
IGNORE_LABEL = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline FC-CLIP pseudolabels for DDD17 frames")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--max-images-per-sequence", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit the number of split directories processed for smoke runs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_split_dirs(dataset_root: Path, split: str) -> List[Path]:
    data_dirs = sorted(Path(path) for path in glob.glob(str(dataset_root / "dir*")))
    if not data_dirs:
        raise FileNotFoundError(f"no DDD17 directories found under {dataset_root}")
    return [Path(path) for path in get_split([str(path) for path in data_dirs], split)]


def resolve_output_root(output_root: Path, split: str) -> Path:
    if output_root.name == split:
        return output_root
    if output_root.name in SPLIT_NAMES and output_root.name != split:
        raise ValueError(f"output root {output_root} conflicts with requested split {split}")
    return output_root / split


def iter_reference_masks(sequence_dir: Path, max_images_per_sequence: Optional[int]) -> List[Path]:
    mask_files = sorted((sequence_dir / "segmentation_masks").glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"no segmentation masks found under {sequence_dir / 'segmentation_masks'}")
    if max_images_per_sequence is not None:
        mask_files = mask_files[:max_images_per_sequence]
    return mask_files


def resolve_frame_path(sequence_dir: Path, segmentation_mask_path: Path) -> Path:
    stem = segmentation_mask_path.stem
    if not stem.startswith("segmentation_"):
        raise ValueError(f"unexpected DDD17 mask basename: {segmentation_mask_path.name}")
    frame_index = int(stem.split("_")[-1])
    candidates = [
        sequence_dir / "imgs" / f"img_{frame_index:08d}.png",
        sequence_dir / "imgs" / f"{frame_index:010d}.png",
        sequence_dir / "imgs" / f"{frame_index:08d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not resolve frame for {segmentation_mask_path.name} in {sequence_dir / 'imgs'}"
    )


def output_mask_path(split_output_root: Path, sequence_dir: Path, reference_mask_path: Path) -> Path:
    return split_output_root / sequence_dir.name / "segmentation_masks" / reference_mask_path.name


def build_mask(sem_seg, threshold: float) -> np.ndarray:
    probabilities = sem_seg.softmax(dim=0)
    confidence, class_index = probabilities.max(dim=0)
    dense_mask = class_index.cpu().numpy().astype(np.uint8)
    dense_mask[confidence.cpu().numpy() < threshold] = IGNORE_LABEL
    return dense_mask


def generate_for_split(args: argparse.Namespace) -> None:
    predictor = FCCLIPDensePredictor(
        config_file=str(args.config_file),
        opts=args.opts,
        device=args.device,
    )
    split_dirs = discover_split_dirs(args.dataset_root, args.split)
    if args.max_sequences is not None:
        split_dirs = split_dirs[: args.max_sequences]
    split_output_root = resolve_output_root(args.output_root, args.split)

    print(
        f"Generating DDD17 pseudolabels for split={args.split} dirs={[path.name for path in split_dirs]} "
        f"threshold={args.threshold} output_root={split_output_root} config_file={args.config_file}"
    )

    written = 0
    for sequence_dir in split_dirs:
        reference_masks = iter_reference_masks(sequence_dir, args.max_images_per_sequence)
        for reference_mask_path in tqdm(reference_masks, desc=sequence_dir.name):
            mask_path = output_mask_path(split_output_root, sequence_dir, reference_mask_path)
            if mask_path.exists() and not args.overwrite:
                continue

            frame_path = resolve_frame_path(sequence_dir, reference_mask_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            image = read_image(str(frame_path), format="BGR")
            mask = build_mask(predictor.predict_sem_seg(image), args.threshold)
            Image.fromarray(mask, mode="L").save(mask_path)
            written += 1

    print(f"WROTE_PSEUDOLABELS split={args.split} count={written} output_root={split_output_root}")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    generate_for_split(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
