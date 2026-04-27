from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ddd17_fcclip_export_utils import (
    discover_split_dirs,
    iter_reference_masks,
    output_artifact_path,
    read_mask_shape,
    resolve_output_root,
)
from utils.fcclip_top2_soft_utils import load_top2_soft_artifact, top2_soft_artifact_path

DEFAULT_NUM_CLASSES = 6


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify exported FC-CLIP top-2 soft artifacts for DDD17')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--top2-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid'), required=True)
    parser.add_argument('--num-classes', type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument('--max-images-per-sequence', type=int, default=None)
    parser.add_argument('--max-sequences', type=int, default=None)
    return parser.parse_args()


def _count_artifacts(top2_root: Path) -> int:
    return sum(1 for _ in top2_root.rglob('*.npz'))


def build_results(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    split_dirs = discover_split_dirs(args.dataset_root, args.split)
    if args.max_sequences is not None:
        split_dirs = split_dirs[: args.max_sequences]

    top2_root = resolve_output_root(args.top2_root, args.split)
    results.append(CheckResult(top2_root.is_dir(), 'top2_root', f'expected {top2_root}'))

    expected_count = 0
    for sequence_dir in split_dirs:
        expected_count += len(iter_reference_masks(sequence_dir, args.max_images_per_sequence))

    actual_count = _count_artifacts(top2_root) if top2_root.is_dir() else 0
    results.append(
        CheckResult(
            actual_count == expected_count,
            'file_count_match',
            f'expected={expected_count} actual={actual_count}',
        )
    )

    checked = 0
    for sequence_dir in split_dirs:
        for reference_mask_path in iter_reference_masks(sequence_dir, args.max_images_per_sequence):
            checked += 1
            artifact_path = top2_soft_artifact_path(
                output_artifact_path(top2_root, sequence_dir, reference_mask_path)
            )
            results.append(
                CheckResult(
                    artifact_path.is_file(),
                    f'{sequence_dir.name}:{reference_mask_path.name}:artifact_file',
                    f'expected {artifact_path}',
                )
            )
            if not artifact_path.is_file():
                continue

            try:
                artifact = load_top2_soft_artifact(artifact_path)
                reference_shape = read_mask_shape(reference_mask_path)
            except Exception as exc:
                results.append(
                    CheckResult(False, f'{sequence_dir.name}:{reference_mask_path.name}:artifact_read', str(exc))
                )
                continue

            artifact_shapes = {
                'top1_id': tuple(int(value) for value in artifact.top1_id.shape),
                'top2_id': tuple(int(value) for value in artifact.top2_id.shape),
                'p1': tuple(int(value) for value in artifact.p1.shape),
                'p2': tuple(int(value) for value in artifact.p2.shape),
            }
            shape_ok = all(shape == reference_shape for shape in artifact_shapes.values())
            results.append(
                CheckResult(
                    shape_ok,
                    f'{sequence_dir.name}:{reference_mask_path.name}:shape_match',
                    f'artifact_shapes={artifact_shapes} reference={reference_shape}',
                )
            )

            top1_valid = bool(
                np.all(artifact.top1_id >= 0) and np.all(artifact.top1_id < args.num_classes)
            )
            top2_valid = bool(
                np.all(artifact.top2_id >= 0) and np.all(artifact.top2_id < args.num_classes)
            )
            results.append(
                CheckResult(
                    top1_valid and top2_valid,
                    f'{sequence_dir.name}:{reference_mask_path.name}:class_id_range',
                    f'num_classes={args.num_classes}',
                )
            )

            p1_range = bool(np.all(artifact.p1 >= 0.0) and np.all(artifact.p1 <= 1.0))
            p2_range = bool(np.all(artifact.p2 >= 0.0) and np.all(artifact.p2 <= 1.0))
            results.append(
                CheckResult(
                    p1_range and p2_range,
                    f'{sequence_dir.name}:{reference_mask_path.name}:probability_range',
                    'expected p1 and p2 in [0, 1]',
                )
            )
            results.append(
                CheckResult(
                    bool(np.all(artifact.p1 >= artifact.p2)),
                    f'{sequence_dir.name}:{reference_mask_path.name}:probability_order',
                    'expected p1 >= p2 per pixel',
                )
            )

    results.append(CheckResult(checked > 0, 'checked_files', f'checked {checked} reference masks'))
    return results


def main() -> int:
    args = parse_args()
    results = build_results(args)
    failed = [item for item in results if not item.ok]

    print(f'Top-2 soft layout check split={args.split} dataset_root={args.dataset_root}')
    print(f'Top-2 root={args.top2_root}')
    print(f'Num classes={args.num_classes}')

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
