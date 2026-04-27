import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fcclip_top2_soft_utils import load_top2_soft_artifact


RESAMPLING = getattr(Image, "Resampling", Image)


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


CLASS_NAMES = ("flat", "background", "object", "vegetation", "human", "vehicle")
CLASS_COLORS = np.array(
    [
        [128, 64, 128],
        [70, 70, 70],
        [220, 220, 0],
        [107, 142, 35],
        [220, 20, 60],
        [0, 0, 142],
    ],
    dtype=np.uint8,
)


class SampleId:
    def __init__(self, sequence, frame_id):
        self.sequence = sequence
        self.frame_id = frame_id

    def __eq__(self, other):
        return (
            isinstance(other, SampleId)
            and self.sequence == other.sequence
            and self.frame_id == other.frame_id
        )

    def __repr__(self):
        return "SampleId(sequence={!r}, frame_id={!r})".format(self.sequence, self.frame_id)


class SamplePaths:
    def __init__(self, frame, label, hard_pseudolabel, top2_artifact, prediction=None):
        self.frame = frame
        self.label = label
        self.hard_pseudolabel = hard_pseudolabel
        self.top2_artifact = top2_artifact
        self.prediction = prediction


def parse_sample_id(value):
    if ":" not in value:
        raise ValueError("sample id must use format dirX:00000000")
    sequence, frame_id = value.split(":", 1)
    if not sequence.startswith("dir") or not frame_id.isdigit():
        raise ValueError("sample id must use format dirX:00000000")
    return SampleId(sequence=sequence, frame_id="{:08d}".format(int(frame_id)))


def build_sample_paths(
    data_root,
    hard_root,
    top2_root,
    sample,
    prediction_root=None,
):
    mask_name = "segmentation_{}.png".format(sample.frame_id)
    prediction = None
    if prediction_root is not None:
        prediction = prediction_root / sample.sequence / "segmentation_masks" / mask_name
    return SamplePaths(
        frame=data_root / sample.sequence / "imgs" / "img_{}.png".format(sample.frame_id),
        label=data_root / sample.sequence / "segmentation_masks" / mask_name,
        hard_pseudolabel=hard_root / sample.sequence / "segmentation_masks" / mask_name,
        top2_artifact=top2_root / sample.sequence / "segmentation_masks" / "segmentation_{}.npz".format(sample.frame_id),
        prediction=prediction,
    )


def require_file(path, name):
    if not path.is_file():
        raise IOError("missing {}: {}".format(name, path))


def read_image_rgb(path, size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize(size, RESAMPLING.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def read_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def resize_rgb(image, shape, resample=None):
    if resample is None:
        resample = RESAMPLING.BILINEAR
    if image.shape[:2] == shape:
        return image
    return np.asarray(Image.fromarray(image).resize((shape[1], shape[0]), resample), dtype=np.uint8)


def resize_mask(mask, shape):
    if mask.shape == shape:
        return mask
    return np.asarray(Image.fromarray(mask).resize((shape[1], shape[0]), RESAMPLING.NEAREST), dtype=np.uint8)


def colorize_semantic_mask(mask, ignore_label=255):
    value = np.asarray(mask)
    rgb = np.zeros(tuple(value.shape) + (3,), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        rgb[value == class_id] = color
    rgb[value == ignore_label] = (0, 0, 0)
    rgb[(value < 0) | (value >= len(CLASS_COLORS))] = (0, 0, 0)
    return rgb


def compute_top2_ambiguity(p1, p2):
    margin = np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
    if margin.ndim != 2:
        raise ValueError("expected 2D top-2 margin, got {}".format(margin.shape))
    if np.isnan(margin).any():
        raise ValueError("top-2 margin contains NaN values")
    lo = float(np.percentile(margin, 2))
    hi = float(np.percentile(margin, 98))
    if hi <= lo:
        return (1.0 - np.clip(margin, 0.0, 1.0)).astype(np.float32)
    normalized_margin = np.clip((margin - lo) / (hi - lo), 0.0, 1.0)
    return (1.0 - normalized_margin).astype(np.float32)


def colorize_heatmap(value):
    heat = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    red = np.clip(255.0 * heat, 0, 255)
    green = np.clip(180.0 * (1.0 - np.abs(heat - 0.5) * 2.0), 0, 180)
    blue = np.clip(255.0 * (1.0 - heat), 0, 255)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def blend(base, overlay, alpha):
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def make_agreement_overlay(
    base_rgb,
    prediction,
    reference,
    ignore_label=255,
):
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference shapes differ: {} vs {}".format(prediction.shape, reference.shape))
    if base_rgb.shape[:2] != prediction.shape:
        base_rgb = resize_rgb(base_rgb, prediction.shape)

    color = np.zeros_like(base_rgb, dtype=np.uint8)
    valid = (reference != ignore_label) & (prediction != ignore_label)
    match = (prediction == reference) & valid
    mismatch = (prediction != reference) & valid
    color[match] = (20, 210, 80)
    color[mismatch] = (230, 50, 40)

    result = base_rgb.copy()
    result[valid] = blend(base_rgb, color, 0.55)[valid]
    return result


def panel(image, title, subtitle="", width=346, height=230):
    body = Image.fromarray(image).resize((width, height), RESAMPLING.NEAREST)
    header_h = 70
    canvas = Image.new("RGB", (width, height + header_h), (246, 248, 251))
    canvas.paste(body, (0, header_h - 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill=(9, 22, 48), font=load_font(18, bold=True))
    if subtitle:
        draw.text((10, 36), subtitle, fill=(70, 84, 105), font=load_font(15))
    return canvas


def hstack(images, gap=14):
    items = list(images)
    width = sum(item.width for item in items) + gap * (len(items) - 1)
    height = max(item.height for item in items)
    canvas = Image.new("RGB", (width, height), (246, 248, 251))
    x = 0
    for item in items:
        canvas.paste(item, (x, 0))
        x += item.width + gap
    return canvas


def vstack(images, gap=18):
    items = list(images)
    width = max(item.width for item in items)
    height = sum(item.height for item in items) + gap * (len(items) - 1)
    canvas = Image.new("RGB", (width, height), (246, 248, 251))
    y = 0
    for item in items:
        canvas.paste(item, (0, y))
        y += item.height + gap
    return canvas


def build_sample_row(paths, sample, agreement_source):
    del agreement_source
    for name, path in (
        ("frame", paths.frame),
        ("DDD17 label", paths.label),
        ("FC-CLIP hard pseudo label", paths.hard_pseudolabel),
        ("Top-2 artifact", paths.top2_artifact),
    ):
        require_file(path, name)

    frame = read_image_rgb(paths.frame)
    label = read_mask(paths.label)
    hard = resize_mask(read_mask(paths.hard_pseudolabel), label.shape)
    artifact = load_top2_soft_artifact(paths.top2_artifact)

    ambiguity = compute_top2_ambiguity(artifact.p1, artifact.p2)
    ambiguity_rgb = colorize_heatmap(ambiguity)
    ambiguity_rgb = resize_rgb(ambiguity_rgb, frame.shape[:2], resample=RESAMPLING.BILINEAR)
    ambiguity_rgb = blend(frame, ambiguity_rgb, 0.55)

    prediction_for_overlay = hard
    overlay_title = "Pseudo-label vs DDD17"
    if paths.prediction is not None and paths.prediction.is_file():
        prediction_for_overlay = resize_mask(read_mask(paths.prediction), label.shape)
        overlay_title = "ESS prediction vs DDD17"

    overlay_base = resize_rgb(frame, label.shape)
    overlay = make_agreement_overlay(overlay_base, prediction_for_overlay, label)

    return hstack(
        [
            panel(frame, "DDD17 frame", "{}:{}".format(sample.sequence, sample.frame_id)),
            panel(colorize_semantic_mask(label), "DDD17 label", "6-class mask"),
            panel(colorize_semantic_mask(hard), "FC-CLIP hard label", "offline teacher"),
            panel(ambiguity_rgb, "Top-2 ambiguity heatmap", "high = small p1-p2"),
            panel(overlay, overlay_title, "green=agree red=diff"),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Create a real DDD17 mechanism visualization for ESS-VLM")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--top2-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--sample", action="append", required=True, help="Sample id such as dir0:00003726")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = [parse_sample_id(value) for value in args.sample]
    rows = []
    for sample in samples:
        paths = build_sample_paths(
            data_root=args.data_root,
            hard_root=args.hard_root,
            top2_root=args.top2_root,
            prediction_root=args.prediction_root,
            sample=sample,
        )
        rows.append(build_sample_row(paths, sample, agreement_source="prediction-or-hard"))

    title_h = 118
    grid = vstack(rows)
    canvas = Image.new("RGB", (grid.width, grid.height + title_h), (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 18),
        "Real DDD17 Top-2 Soft Mechanism Visualization",
        fill=(9, 22, 48),
        font=load_font(30, bold=True),
    )
    draw.text(
        (18, 62),
        "Ambiguity is derived from FC-CLIP top-1/top-2 probability margin (p1-p2), not attention.",
        fill=(70, 84, 105),
        font=load_font(18),
    )
    draw.text(
        (18, 88),
        "Green/red overlay shows agreement/difference between FC-CLIP hard pseudo label and DDD17 label.",
        fill=(70, 84, 105),
        font=load_font(18),
    )
    canvas.paste(grid, (0, title_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print("WROTE_REAL_MECHANISM_VIS output={} samples={}".format(args.output, ",".join(args.sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
