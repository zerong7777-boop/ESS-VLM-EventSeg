#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-full}"
case "$MODE" in
  smoke)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_smoke.yaml"
    ;;
  full)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft.yaml"
    ;;
  ablation)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_ablation.yaml"
    ;;
  *)
    CONFIG_PATH="$MODE"
    ;;
esac
HARD_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_no_filter_cropped"
SOFT_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/fcclip_top2_soft_cropped"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-fcclip/bin/python}"
RUN_TRAIN="${ESS_RUN_TRAIN:-0}"

cd "$REPO_ROOT"
source ./.env.bootstrap.sh
export WANDB_MODE=offline

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[TOP2-SOFT] missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[TOP2-SOFT] missing config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$HARD_ROOT" ]]; then
  echo "[TOP2-SOFT] missing hard pseudolabel root: $HARD_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SOFT_ROOT" ]]; then
  echo "[TOP2-SOFT] missing top2 soft root: $SOFT_ROOT" >&2
  exit 1
fi

TRAIN_CMD=("$PYTHON_BIN" train.py --settings_file "$CONFIG_PATH")
echo "[TOP2-SOFT] project_root=$(pwd)"
echo "[TOP2-SOFT] config=$CONFIG_PATH"
echo "[TOP2-SOFT] python=$PYTHON_BIN"
printf '[TOP2-SOFT] command='
printf '%q ' "${TRAIN_CMD[@]}"
printf '
'

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[TOP2-SOFT] preflight only; set ESS_RUN_TRAIN=1 to start training"
  exit 0
fi

exec "${TRAIN_CMD[@]}"
