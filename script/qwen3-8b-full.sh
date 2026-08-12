#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/share/Qwen3-8B}"
export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export BLOCK_BATCH_SIZE="${BLOCK_BATCH_SIZE:-1}"

exec "${SCRIPT_DIR}/qwen3-full-pipeline.sh" "$@"
