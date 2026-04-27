#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
DATASET_ROOT="$REPO_ROOT/data/ddd17_seg/data"
COMMON_PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/clipseg"
NO_FILTER_PSEUDOLABEL_ROOT="$REPO_ROOT/data/ddd17_pseudolabels/clipseg/train_no_filter"
RUN_TRAIN="${ESS_RUN_TRAIN:-0}"

CONFIGS=(
  "config/settings_DDD17_vlm_pseudolabel_no_filter.yaml"
  "config/settings_DDD17_vlm_pseudolabel_low_weight.yaml"
  "config/settings_DDD17_vlm_pseudolabel_medium_weight.yaml"
)

cd "$REPO_ROOT"
source ./.env.bootstrap.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[VLM-ABLATIONS] missing python executable: $PYTHON_BIN" >&2
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

  echo "[VLM-ABLATIONS] preflight failed for ${label} pseudolabels"
  echo "[VLM-ABLATIONS] launcher_input_root=${launcher_input_root}"
  echo "[VLM-ABLATIONS] loader_consumed_root=${effective_root:-unknown}"
  echo "[VLM-ABLATIONS] missing_reference_masks=${missing_count:-0}"
  if [[ -n "$first_missing" ]]; then
    echo "[VLM-ABLATIONS] first_missing_path=${first_missing}"
  fi
  echo "[VLM-ABLATIONS] current remote pseudolabel roots are incomplete; Task 5 only generated a dry subset, not the full train split."
  echo "[VLM-ABLATIONS] action: finish full pseudolabel generation before setting ESS_RUN_TRAIN=1."
}

verify_layout() {
  local label="$1"
  local launcher_input_root="$2"
  local verifier_log

  verifier_log=$(mktemp)
  echo "[VLM-ABLATIONS] verifying ${label} pseudolabel layout split=train launcher_input_root=${launcher_input_root}"
  if "$PYTHON_BIN" scripts/vlm/verify_ddd17_pseudolabel_layout.py \
    "$launcher_input_root" \
    --dataset-root "$DATASET_ROOT" \
    --split train >"$verifier_log" 2>&1; then
    local effective_root
    effective_root=$(sed -n 's/^PSEUDOLABEL_LAYOUT_OK //p' "$verifier_log" | head -n 1)
    echo "[VLM-ABLATIONS] preflight ok loader_consumed_root=${effective_root:-unknown}"
    rm -f "$verifier_log"
    return 0
  fi

  summarize_preflight_failure "$label" "$launcher_input_root" "$verifier_log"
  rm -f "$verifier_log"
  return 1
}

verify_layout no-filter "$NO_FILTER_PSEUDOLABEL_ROOT"
verify_layout filtered "$COMMON_PSEUDOLABEL_ROOT"

echo "[VLM-ABLATIONS] project_root=$(pwd)"
echo "[VLM-ABLATIONS] python=$PYTHON_BIN"

for config_path in "${CONFIGS[@]}"; do
  cmd=("$PYTHON_BIN" train.py --settings_file "$config_path")
  printf '[VLM-ABLATIONS] command='
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "$RUN_TRAIN" == "1" ]]; then
    "${cmd[@]}"
  fi
done

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[VLM-ABLATIONS] preflight only; export ESS_RUN_TRAIN=1 after full pseudolabel generation completes"
fi
