#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source ./.env.bootstrap.sh

LOG_ROOT="log/bootstrap_ddd17"
STDOUT_LOG="$LOG_ROOT/smoke.stdout.log"

mkdir -p "$LOG_ROOT"

run_cityscapes_precheck() {
  python3 - <<'PY'
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config/settings_DDD17_bootstrap.yaml").read_text())
cityscapes_root = Path(config["dataset"]["cityscapes_img"]["dataset_path"])
required_paths = [
    cityscapes_root,
    cityscapes_root / "leftImg8bit",
    cityscapes_root / "gtFine",
]
missing = [str(path) for path in required_paths if not path.exists()]

if missing:
    print("CITYSCAPES_PRECHECK_FAIL")
    for path in missing:
        print(f"- missing: {path}")
    raise SystemExit(2)

print(f"CITYSCAPES_PRECHECK_OK {cityscapes_root}")
PY
}

run_smoke() {
  set -euo pipefail

  echo "[SMOKE] DDD17 bootstrap launcher"
  echo "[SMOKE] project_root=$(pwd)"
  echo "[SMOKE] config=config/settings_DDD17_bootstrap.yaml"

  python3 scripts/bootstrap/check_env.py
  python3 scripts/bootstrap/verify_ddd17_layout.py
  run_cityscapes_precheck

  echo "[SMOKE] preflight passed; starting train.py"
  python3 train.py --settings_file config/settings_DDD17_bootstrap.yaml
}

set +e
run_smoke 2>&1 | tee "$STDOUT_LOG"
status=${PIPESTATUS[0]}
set -e

echo "[SMOKE] exit_code=$status" | tee -a "$STDOUT_LOG"
exit "$status"
