from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ddd17_events_loader import get_split
from scripts.vlm.generate_ddd17_pseudolabels import iter_reference_masks, resolve_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DDD17 pseudolabel layout and basename contract")
    parser.add_argument("pseudolabel_root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--max-reference-masks-per-sequence", type=int, default=None)
    return parser.parse_args()


def discover_split_dirs(dataset_root: Path, split: str) -> list[Path]:
    data_dirs = sorted(Path(path) for path in glob.glob(str(dataset_root / "dir*")))
    if not data_dirs:
        raise FileNotFoundError(f"no DDD17 directories found under {dataset_root}")
    return [Path(path) for path in get_split([str(path) for path in data_dirs], split)]


def main() -> int:
    args = parse_args()
    split_root = resolve_output_root(args.pseudolabel_root, args.split)
    split_dirs = discover_split_dirs(args.dataset_root, args.split)

    missing = []
    for sequence_dir in split_dirs:
        expected_dir = split_root / sequence_dir.name / "segmentation_masks"
        if not expected_dir.is_dir():
            missing.append(str(expected_dir))
            continue
        for reference_mask in iter_reference_masks(sequence_dir, args.max_reference_masks_per_sequence):
            candidate = expected_dir / reference_mask.name
            if not candidate.is_file():
                missing.append(str(candidate))

    if missing:
        print(f"PSEUDOLABEL_LAYOUT_FAIL {split_root}")
        for path in missing:
            print(path)
        return 1

    print(f"PSEUDOLABEL_LAYOUT_OK {split_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
