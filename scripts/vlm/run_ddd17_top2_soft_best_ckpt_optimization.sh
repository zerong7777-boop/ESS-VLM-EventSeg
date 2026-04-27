#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-w0015}"

case "$MODE" in
  smoke)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_smoke.yaml"
    ;;
  w0015)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_best_w0015.yaml"
    ;;
  w001)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_best_w001.yaml"
    ;;
  temp15)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_best_temp15.yaml"
    ;;
  gate)
    CONFIG_PATH="config/settings_DDD17_fcclip_top2_soft_best_gate.yaml"
    ;;
  -h|--help|help)
    cat <<'USAGE'
Usage: bash scripts/vlm/run_ddd17_top2_soft_best_ckpt_optimization.sh [smoke|w0015|w001|temp15|gate|CONFIG_PATH]

Default mode is w0015. The launcher always runs preflight only unless ESS_RUN_TRAIN=1.
USAGE
    exit 0
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
if [[ -f ./.env.bootstrap.sh ]]; then
  source ./.env.bootstrap.sh
fi
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[TOP2-SOFT-BEST] missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$HARD_ROOT" ]]; then
  echo "[TOP2-SOFT-BEST] missing hard pseudolabel root: $HARD_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SOFT_ROOT" ]]; then
  echo "[TOP2-SOFT-BEST] missing top2 soft root: $SOFT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[TOP2-SOFT-BEST] missing config: $CONFIG_PATH" >&2
  exit 1
fi

TRAIN_CMD=("$PYTHON_BIN" train.py --settings_file "$CONFIG_PATH")

echo "[TOP2-SOFT-BEST] project_root=$(pwd)"
echo "[TOP2-SOFT-BEST] mode=$MODE"
echo "[TOP2-SOFT-BEST] config=$CONFIG_PATH"
echo "[TOP2-SOFT-BEST] python=$PYTHON_BIN"
printf '[TOP2-SOFT-BEST] command='
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[TOP2-SOFT-BEST] preflight only; set ESS_RUN_TRAIN=1 to start training"
  exit 0
fi

exec "${TRAIN_CMD[@]}"

