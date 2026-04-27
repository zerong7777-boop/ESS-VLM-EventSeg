#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
DATASET_ROOT="$REPO_ROOT/data/ddd17_seg/data"
PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_no_filter_cropped"
RUN_TRAIN="${ESS_RUN_TRAIN:-0}"

CONFIGS=(
  "config/settings_DDD17_fcclip_pseudolabel_tiny_weight.yaml"
  "config/settings_DDD17_fcclip_pseudolabel_zero_weight.yaml"
)

cd "$REPO_ROOT"
source ./.env.bootstrap.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FCCLIP-ABLATIONS] missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$PSEUDOLABEL_ROOT" ]]; then
  echo "[FCCLIP-ABLATIONS] missing pseudolabel root: $PSEUDOLABEL_ROOT" >&2
  exit 1
fi
for config_path in "${CONFIGS[@]}"; do
  if [[ ! -f "$config_path" ]]; then
    echo "[FCCLIP-ABLATIONS] missing config: $config_path" >&2
    exit 1
  fi
done

echo "[FCCLIP-ABLATIONS] verifying pseudolabel layout root=$PSEUDOLABEL_ROOT"
"$PYTHON_BIN" scripts/vlm/verify_ddd17_pseudolabel_layout.py   "$PSEUDOLABEL_ROOT"   --dataset-root "$DATASET_ROOT"   --split train

echo "[FCCLIP-ABLATIONS] project_root=$(pwd)"
echo "[FCCLIP-ABLATIONS] python=$PYTHON_BIN"

for config_path in "${CONFIGS[@]}"; do
  cmd=("$PYTHON_BIN" train.py --settings_file "$config_path")
  printf '[FCCLIP-ABLATIONS] command='
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$RUN_TRAIN" == "1" ]]; then
    "${cmd[@]}"
  fi
done

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[FCCLIP-ABLATIONS] preflight only; set ESS_RUN_TRAIN=1 to start training"
fi
