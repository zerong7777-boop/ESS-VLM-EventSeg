from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ddd17_fcclip_export_utils import (
    align_prediction_to_reference,
    discover_split_dirs,
    iter_reference_masks,
    output_artifact_path,
    resolve_frame_path,
    resolve_output_root,
)
from utils.fcclip_top2_soft_utils import save_top2_soft_artifact, top2_soft_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export offline FC-CLIP top-2 soft artifacts for DDD17 frames')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid'), required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--config-file', type=Path, required=True)
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--max-images-per-sequence', type=int, default=None)
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def generate_for_split(args: argparse.Namespace) -> None:
    from detectron2.data.detection_utils import read_image

    from scripts.vlm.fcclip_dense_predictor import FCCLIPDensePredictor

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
        f'Generating DDD17 top-2 soft artifacts for split={args.split} '
        f'dirs={[path.name for path in split_dirs]} '
        f'output_root={split_output_root} config_file={args.config_file}'
    )

    written = 0
    for sequence_dir in split_dirs:
        reference_masks = iter_reference_masks(sequence_dir, args.max_images_per_sequence)
        for reference_mask_path in tqdm(reference_masks, desc=sequence_dir.name):
            artifact_path = top2_soft_artifact_path(
                output_artifact_path(split_output_root, sequence_dir, reference_mask_path)
            )
            if artifact_path.exists() and not args.overwrite:
                continue

            frame_path = resolve_frame_path(sequence_dir, reference_mask_path)
            image = read_image(str(frame_path), format='BGR')
            top1_id, top2_id, p1, p2 = predictor.predict_top2_soft(image)
            save_top2_soft_artifact(
                artifact_path,
                top1_id=align_prediction_to_reference(top1_id, reference_mask_path),
                top2_id=align_prediction_to_reference(top2_id, reference_mask_path),
                p1=align_prediction_to_reference(p1, reference_mask_path),
                p2=align_prediction_to_reference(p2, reference_mask_path),
            )
            written += 1

    print(f'WROTE_TOP2_SOFT split={args.split} count={written} output_root={split_output_root}')


def main() -> int:
    args = parse_args()
    generate_for_split(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
