#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG_PATH="config/settings_DDD17_vlm_pseudolabel.yaml"
DATASET_ROOT="$REPO_ROOT/data/ddd17_seg/data"
PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/clipseg"
RUN_TRAIN="${ESS_RUN_TRAIN:-0}"

cd "$REPO_ROOT"
source ./.env.bootstrap.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[VLM-MAIN] missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

summarize_preflight_failure() {
  local label="$1"
  local launcher_input_root="$2"
  local verifier_log="$3"
  local effective_root
  local missing_count
  local first_missing

  effective_root=$(sed -n 's/^PSEUDOLABEL_LAYOUT_FAIL //p' "$verifier_log" | head -n 1)
  missing_count=$(tail -n +2 "$verifier_log" | wc -l | tr -d ' ')
  first_missing=$(sed -n '2p' "$verifier_log")

  echo "[VLM-MAIN] preflight failed for ${label} pseudolabels"
  echo "[VLM-MAIN] launcher_input_root=${launcher_input_root}"
  echo "[VLM-MAIN] loader_consumed_root=${effective_root:-unknown}"
  echo "[VLM-MAIN] missing_reference_masks=${missing_count:-0}"
  if [[ -n "$first_missing" ]]; then
    echo "[VLM-MAIN] first_missing_path=${first_missing}"
  fi
  echo "[VLM-MAIN] current remote pseudolabel roots are incomplete; Task 5 only generated a dry subset, not the full train split."
  echo "[VLM-MAIN] action: finish full pseudolabel generation before setting ESS_RUN_TRAIN=1."
}

verify_layout() {
  local label="$1"
  local split="$2"
  local launcher_input_root="$3"
  local verifier_log

  verifier_log=$(mktemp)
  echo "[VLM-MAIN] verifying ${label} pseudolabel layout split=${split} launcher_input_root=${launcher_input_root}"
  if "$PYTHON_BIN" scripts/vlm/verify_ddd17_pseudolabel_layout.py \
    "$launcher_input_root" \
    --dataset-root "$DATASET_ROOT" \
    --split "$split" >"$verifier_log" 2>&1; then
    local effective_root
    effective_root=$(sed -n 's/^PSEUDOLABEL_LAYOUT_OK //p' "$verifier_log" | head -n 1)
    echo "[VLM-MAIN] preflight ok loader_consumed_root=${effective_root:-unknown}"
    rm -f "$verifier_log"
    return 0
  fi

  summarize_preflight_failure "$label" "$launcher_input_root" "$verifier_log"
  rm -f "$verifier_log"
  return 1
}

TRAIN_CMD=(
  "$PYTHON_BIN"
  train.py
  --settings_file
  "$CONFIG_PATH"
)

verify_layout main train "$PSEUDOLABEL_ROOT"

echo "[VLM-MAIN] project_root=$(pwd)"
echo "[VLM-MAIN] config=$CONFIG_PATH"
echo "[VLM-MAIN] python=$PYTHON_BIN"
printf '[VLM-MAIN] command='
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[VLM-MAIN] preflight only; export ESS_RUN_TRAIN=1 after full pseudolabel generation completes"
  exit 0
fi

exec "${TRAIN_CMD[@]}"
