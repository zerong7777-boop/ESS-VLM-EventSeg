#!/usr/bin/env python3
"""Verify the DDD17 sensor-B layout expected by ESS bootstrap runs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("data/ddd17_seg/data")
EVENT_FILES = ("events.dat.t", "events.dat.xyp")
INDEX_FILES = ("index/index_50ms.npy",)
OPTIONAL_SIGNS = ("segmentation_masks.zip", "video.mp4")


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str


def describe_matches(sequence_dir: Path, patterns: tuple[str, ...]) -> str:
    found = [pattern for pattern in patterns if (sequence_dir / pattern).exists()]
    if found:
        return ", ".join(found)
    return "none"


def inspect_sequence(sequence_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    masks_dir = sequence_dir / "segmentation_masks"
    mask_files = sorted(masks_dir.glob("*.png")) if masks_dir.is_dir() else []

    results.append(
        CheckResult(
            ok=masks_dir.is_dir(),
            name=f"{sequence_dir.name}:segmentation_masks_dir",
            detail=f"expected {masks_dir}",
        )
    )
    results.append(
        CheckResult(
            ok=bool(mask_files),
            name=f"{sequence_dir.name}:segmentation_masks_png",
            detail=f"found {len(mask_files)} png files",
        )
    )

    for relative_path in EVENT_FILES:
        path = sequence_dir / relative_path
        results.append(
            CheckResult(
                ok=path.is_file(),
                name=f"{sequence_dir.name}:{relative_path}",
                detail=f"expected {path}",
            )
        )

    for relative_path in INDEX_FILES:
        path = sequence_dir / relative_path
        results.append(
            CheckResult(
                ok=path.is_file(),
                name=f"{sequence_dir.name}:{relative_path}",
                detail=f"expected {path}",
            )
        )

    optional_found = describe_matches(sequence_dir, OPTIONAL_SIGNS)
    results.append(
        CheckResult(
            ok=True,
            name=f"{sequence_dir.name}:optional_signals",
            detail=f"detected {optional_found}; searched {', '.join(OPTIONAL_SIGNS)}",
        )
    )
    return results


def build_results(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(CheckResult(root.is_dir(), "dataset_root", f"expected {root}"))

    if not root.is_dir():
        return results

    sequences = sorted(path for path in root.glob("dir*") if path.is_dir())
    results.append(
        CheckResult(
            ok=bool(sequences),
            name="sequence_dirs",
            detail=f"found {len(sequences)} directories matching dir*",
        )
    )

    if not sequences:
        return results

    complete_sequences = 0
    for sequence_dir in sequences:
        sequence_results = inspect_sequence(sequence_dir)
        if all(item.ok for item in sequence_results if item.name.endswith(("segmentation_masks_dir", "segmentation_masks_png", "events.dat.t", "events.dat.xyp", "index/index_50ms.npy"))):
            complete_sequences += 1
        results.extend(sequence_results)

    results.append(
        CheckResult(
            ok=complete_sequences > 0,
            name="complete_sequences",
            detail=(
                f"found {complete_sequences}/{len(sequences)} sequences with masks, event binaries, "
                f"and index/index_50ms.npy"
            ),
        )
    )
    return results


def print_results(results: list[CheckResult], root: Path) -> int:
    failed = [item for item in results if not item.ok]
    print(f"DDD17 layout check root: {root}")
    print("Scope: DDD17 sensor-B only. This check does not validate Cityscapes sensor-A readiness.")
    print(f"Required event files: {', '.join(EVENT_FILES)}")
    print(f"Required index files: {', '.join(INDEX_FILES)}")
    print(f"Optional signs searched: {', '.join(OPTIONAL_SIGNS)}")

    for item in results:
        status = "PASS" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")

    if failed:
        print(f"RESULT: FAIL ({len(failed)} failing checks)")
        return 1

    print("RESULT: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the DDD17 sensor-B layout expected by ESS bootstrap runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="DDD17 dataset root. Defaults to the ESS project-local bootstrap path.",
    )
    args = parser.parse_args()

    results = build_results(args.root)
    return print_results(results, args.root)


if __name__ == "__main__":
    sys.exit(main())
