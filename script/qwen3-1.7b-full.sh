#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
export MODEL_TAG="${MODEL_TAG:-qwen3-1.7b}"
export BLOCK_BATCH_SIZE="${BLOCK_BATCH_SIZE:-2}"

exec "${SCRIPT_DIR}/qwen3-full-pipeline.sh" "$@"
