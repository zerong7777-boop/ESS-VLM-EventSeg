#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-config/settings_DDD17_fcclip_pixel_confidence.yaml}"

cd "$REPO_ROOT"
source ./.env.bootstrap.sh

export WANDB_MODE=offline
exec /usr/bin/python3 train.py --settings_file "$CONFIG_PATH"
