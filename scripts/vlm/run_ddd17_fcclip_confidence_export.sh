#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv-fcclip/bin/python"
DATASET_ROOT="$REPO_ROOT/data/ddd17_seg/data"
OUTPUT_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_confidence_cropped"
PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_no_filter_cropped"
CONFIG_FILE="$REPO_ROOT/third_party/fc-clip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml"
MODEL_WEIGHTS="$REPO_ROOT/third_party/fc-clip/checkpoints/fcclip_convnext_large_eval_ade20k.pth"
OPEN_CLIP_WEIGHTS="$REPO_ROOT/third_party/fc-clip/checkpoints/open_clip_model.safetensors"

cd "$REPO_ROOT"

COMMON_OPTS=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --config-file "$CONFIG_FILE"
  --opts MODEL.WEIGHTS "$MODEL_WEIGHTS" MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS "$OPEN_CLIP_WEIGHTS"
)

"$PYTHON_BIN" scripts/vlm/export_ddd17_fcclip_confidence.py "${COMMON_OPTS[@]}" --split train
"$PYTHON_BIN" scripts/vlm/export_ddd17_fcclip_confidence.py "${COMMON_OPTS[@]}" --split valid

"$PYTHON_BIN" scripts/vlm/verify_ddd17_fcclip_confidence_layout.py   --dataset-root "$DATASET_ROOT"   --confidence-root "$OUTPUT_ROOT"   --pseudolabel-root "$PSEUDOLABEL_ROOT"   --split train

"$PYTHON_BIN" scripts/vlm/verify_ddd17_fcclip_confidence_layout.py   --dataset-root "$DATASET_ROOT"   --confidence-root "$OUTPUT_ROOT"   --split valid
