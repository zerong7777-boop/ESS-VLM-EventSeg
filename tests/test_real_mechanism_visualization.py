from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlm.make_real_mechanism_visualization import (
    CLASS_COLORS,
    SampleId,
    build_sample_paths,
    colorize_semantic_mask,
    compute_top2_ambiguity,
    make_agreement_overlay,
    parse_sample_id,
)


def test_parse_sample_id_accepts_dir_and_zero_padded_id():
    sample = parse_sample_id("dir0:00003726")

    assert sample == SampleId(sequence="dir0", frame_id="00003726")


def test_parse_sample_id_rejects_bad_format():
    try:
        parse_sample_id("dir0")
    except ValueError as exc:
        assert "dirX:00000000" in str(exc)
    else:
        raise AssertionError("parse_sample_id should reject missing frame id")


def test_build_sample_paths_uses_existing_ddd17_naming():
    data_root = Path("ddd17_seg") / "data"
    hard_root = Path("fcclip_no_filter_cropped") / "train"
    top2_root = Path("fcclip_top2_soft_cropped") / "train"
    sample = SampleId(sequence="dir0", frame_id="00003726")

    paths = build_sample_paths(data_root=data_root, hard_root=hard_root, top2_root=top2_root, sample=sample)

    assert paths.frame == data_root / "dir0" / "imgs" / "img_00003726.png"
    assert paths.label == data_root / "dir0" / "segmentation_masks" / "segmentation_00003726.png"
    assert paths.hard_pseudolabel == hard_root / "dir0" / "segmentation_masks" / "segmentation_00003726.png"
    assert paths.top2_artifact == top2_root / "dir0" / "segmentation_masks" / "segmentation_00003726.npz"


def test_compute_top2_ambiguity_highlights_small_probability_margin():
    p1 = np.array([[0.90, 0.55], [0.60, 0.51]], dtype=np.float32)
    p2 = np.array([[0.10, 0.40], [0.30, 0.50]], dtype=np.float32)

    ambiguity = compute_top2_ambiguity(p1, p2)

    assert ambiguity.shape == p1.shape
    assert ambiguity.dtype == np.float32
    assert np.all(ambiguity >= 0.0)
    assert np.all(ambiguity <= 1.0)
    assert ambiguity[1, 1] > ambiguity[0, 0]


def test_colorize_semantic_mask_maps_classes_and_ignore_label():
    mask = np.array([[0, 1, 255], [2, 5, 9]], dtype=np.uint8)

    rgb = colorize_semantic_mask(mask)

    assert rgb.shape == (2, 3, 3)
    assert tuple(rgb[0, 0]) == tuple(CLASS_COLORS[0])
    assert tuple(rgb[0, 1]) == tuple(CLASS_COLORS[1])
    assert tuple(rgb[0, 2]) == (0, 0, 0)
    assert tuple(rgb[1, 2]) == (0, 0, 0)


def test_make_agreement_overlay_marks_match_green_and_mismatch_red():
    base = np.zeros((2, 2, 3), dtype=np.uint8) + 100
    prediction = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    reference = np.array([[0, 2], [2, 1]], dtype=np.uint8)

    overlay = make_agreement_overlay(base, prediction, reference, ignore_label=255)

    assert overlay.shape == base.shape
    assert overlay[0, 0, 1] > overlay[0, 0, 0]
    assert overlay[0, 1, 0] > overlay[0, 1, 1]
    assert np.allclose(overlay[1, 1], base[1, 1], atol=1)
