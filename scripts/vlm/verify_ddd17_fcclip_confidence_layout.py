from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ddd17_fcclip_export_utils import (
    discover_split_dirs,
    iter_reference_masks,
    output_artifact_path,
    resolve_output_root,
)


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str


def read_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f'expected single-channel image at {path}, got ndim={array.ndim}')
    return tuple(int(value) for value in array.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify exported FC-CLIP confidence maps for DDD17')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--confidence-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid'), required=True)
    parser.add_argument('--pseudolabel-root', type=Path, default=None)
    parser.add_argument('--max-images-per-sequence', type=int, default=None)
    parser.add_argument('--max-sequences', type=int, default=None)
    return parser.parse_args()


def build_results(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    split_dirs = discover_split_dirs(args.dataset_root, args.split)
    if args.max_sequences is not None:
        split_dirs = split_dirs[: args.max_sequences]

    confidence_root = resolve_output_root(args.confidence_root, args.split)
    pseudolabel_root = resolve_output_root(args.pseudolabel_root, args.split) if args.pseudolabel_root else None

    results.append(CheckResult(confidence_root.is_dir(), 'confidence_root', f'expected {confidence_root}'))
    if pseudolabel_root is not None:
        results.append(CheckResult(pseudolabel_root.is_dir(), 'pseudolabel_root', f'expected {pseudolabel_root}'))

    checked = 0
    for sequence_dir in split_dirs:
        for reference_mask_path in iter_reference_masks(sequence_dir, args.max_images_per_sequence):
            checked += 1
            confidence_path = output_artifact_path(confidence_root, sequence_dir, reference_mask_path)
            results.append(CheckResult(confidence_path.is_file(), f'{sequence_dir.name}:{reference_mask_path.name}:confidence_file', f'expected {confidence_path}'))
            if not confidence_path.is_file():
                continue

            try:
                reference_shape = read_shape(reference_mask_path)
                confidence_shape = read_shape(confidence_path)
            except Exception as exc:
                results.append(CheckResult(False, f'{sequence_dir.name}:{reference_mask_path.name}:shape_read', str(exc)))
                continue

            results.append(CheckResult(confidence_shape == reference_shape, f'{sequence_dir.name}:{reference_mask_path.name}:shape_match', f'confidence={confidence_shape} reference={reference_shape}'))

            if pseudolabel_root is not None:
                pseudolabel_path = output_artifact_path(pseudolabel_root, sequence_dir, reference_mask_path)
                results.append(CheckResult(pseudolabel_path.is_file(), f'{sequence_dir.name}:{reference_mask_path.name}:pseudolabel_file', f'expected {pseudolabel_path}'))
                if pseudolabel_path.is_file():
                    try:
                        pseudolabel_shape = read_shape(pseudolabel_path)
                    except Exception as exc:
                        results.append(CheckResult(False, f'{sequence_dir.name}:{reference_mask_path.name}:pseudolabel_shape_read', str(exc)))
                    else:
                        results.append(CheckResult(confidence_shape == pseudolabel_shape, f'{sequence_dir.name}:{reference_mask_path.name}:pseudolabel_shape_match', f'confidence={confidence_shape} pseudolabel={pseudolabel_shape}'))

    results.append(CheckResult(checked > 0, 'checked_files', f'checked {checked} reference masks'))
    return results


def main() -> int:
    args = parse_args()
    results = build_results(args)
    failed = [item for item in results if not item.ok]

    print(f'Confidence layout check split={args.split} dataset_root={args.dataset_root}')
    print(f'Confidence root={args.confidence_root}')
    if args.pseudolabel_root:
        print(f'Pseudolabel root={args.pseudolabel_root}')

    for item in results:
        status = 'PASS' if item.ok else 'FAIL'
        print(f'[{status}] {item.name}: {item.detail}')

    if failed:
        print(f'RESULT: FAIL ({len(failed)} failing checks)')
        return 1

    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
