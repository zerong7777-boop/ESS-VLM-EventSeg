#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG_PATH="config/settings_DDD17_fcclip_vpr.yaml"
DATASET_ROOT="$REPO_ROOT/data/ddd17_seg/data"
PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_no_filter_cropped"
RUN_TRAIN="${ESS_RUN_TRAIN:-0}"

cd "$REPO_ROOT"
source ./.env.bootstrap.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[VPR] missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[VPR] missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$PSEUDOLABEL_ROOT" ]]; then
  echo "[VPR] missing pseudolabel root: $PSEUDOLABEL_ROOT" >&2
  exit 1
fi

echo "[VPR] verifying pseudolabel layout root=$PSEUDOLABEL_ROOT"
"$PYTHON_BIN" scripts/vlm/verify_ddd17_pseudolabel_layout.py   "$PSEUDOLABEL_ROOT"   --dataset-root "$DATASET_ROOT"   --split train

TRAIN_CMD=(/usr/bin/python3 train.py --settings_file "$CONFIG_PATH")
echo "[VPR] project_root=$(pwd)"
echo "[VPR] config=$CONFIG_PATH"
echo "[VPR] python=$PYTHON_BIN"
printf '[VPR] command='
printf '%q ' "${TRAIN_CMD[@]}"
printf '
'

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[VPR] preflight only; set ESS_RUN_TRAIN=1 to start training"
  exit 0
fi

export WANDB_MODE=offline
exec /usr/bin/python3 train.py --settings_file "$CONFIG_PATH"
