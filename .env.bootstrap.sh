#!/usr/bin/env bash

# Bootstrap environment for ESS remote work.
export ESS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ESS_DATA_ROOT="$ESS_PROJECT_ROOT/data"
export ESS_CACHE_ROOT="$ESS_PROJECT_ROOT/.cache"
export TORCH_HOME="$ESS_CACHE_ROOT/torch"
export HF_HOME="$ESS_CACHE_ROOT/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="$ESS_PROJECT_ROOT/.wandb"
export PYTHONUNBUFFERED=1

mkdir -p "$ESS_DATA_ROOT" "$ESS_CACHE_ROOT" "$TORCH_HOME" "$HF_HOME" "$WANDB_DIR"