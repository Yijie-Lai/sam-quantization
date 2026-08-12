#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reasoningqat}"
GPU="${GPU:-0}"

: "${MODEL_PATH:?Set MODEL_PATH to the original Qwen3 checkpoint}"
: "${MODEL_TAG:?Set MODEL_TAG, for example qwen3-4b}"

START_BITS="${START_BITS:-8}"
TARGET_BITS="${TARGET_BITS:-2}"
GROUP_SIZE="${GROUP_SIZE:-128}"
SWEEP_TAG="${SWEEP_TAG:-sweep06}"
RUN_TAG="${RUN_TAG:-${MODEL_TAG}-w${TARGET_BITS}g${GROUP_SIZE}-${SWEEP_TAG}-${START_BITS}to${TARGET_BITS}-asam-mix75d25g}"

BLOCK_SAVE_DIR="${BLOCK_SAVE_DIR:-/share/MY-DAPO/block_qat/${RUN_TAG}-block}"
BLOCK_LOG_DIR="${BLOCK_LOG_DIR:-${PROJECT_ROOT}/log/block_qat/${RUN_TAG}-block}"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-/share/MY-DAPO/e2e/${RUN_TAG}}"

TRAIN_SIZE="${TRAIN_SIZE:-4096}"
VAL_SIZE="${VAL_SIZE:-64}"
TRAINING_SEQLEN="${TRAINING_SEQLEN:-2048}"
BLOCK_EPOCHS="${BLOCK_EPOCHS:-10}"
BLOCK_BATCH_SIZE="${BLOCK_BATCH_SIZE:-2}"
WEIGHT_LR="${WEIGHT_LR:-2e-5}"
QUANT_LR="${QUANT_LR:-1e-4}"
PROGRESSIVE_RHO="${PROGRESSIVE_RHO:-0.05}"
BLOCK_ASAM_RHO="${BLOCK_ASAM_RHO:-0.05}"
CALIB_DATASET="${CALIB_DATASET:-sweep_0.6}"

E2E_DATASET="${E2E_DATASET:-mix_deita_redpajama}"
REDPAJAMA_DATA_PATH="${REDPAJAMA_DATA_PATH:-/share/MY-DAPO/data/redpajama_1t_sample_text.jsonl}"
E2E_GENERAL_RATIO="${E2E_GENERAL_RATIO:-0.25}"
E2E_CONTEXT_LEN="${E2E_CONTEXT_LEN:-4096}"
E2E_MAX_STEPS="${E2E_MAX_STEPS:-3000}"
E2E_BATCH_SIZE="${E2E_BATCH_SIZE:-4}"
E2E_GRAD_ACCUM="${E2E_GRAD_ACCUM:-4}"
E2E_LR="${E2E_LR:-2e-5}"
E2E_ASAM_RHO="${E2E_ASAM_RHO:-0.05}"
E2E_SAVE_STEPS="${E2E_SAVE_STEPS:-250}"
E2E_SAVE_TOTAL_LIMIT="${E2E_SAVE_TOTAL_LIMIT:-12}"
E2E_TRUST_LOCAL_CHECKPOINT="${E2E_TRUST_LOCAL_CHECKPOINT:-false}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-80000}"

SKIP_BLOCK="${SKIP_BLOCK:-0}"
SKIP_E2E="${SKIP_E2E:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

mkdir -p "${BLOCK_LOG_DIR}" "${E2E_OUTPUT_DIR}"

if [[ "${SKIP_BLOCK}" != "1" ]]; then
    echo "[Qwen3] Starting full block-wise QAT: ${MODEL_TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" python block_main.py \
        --model "${MODEL_PATH}" \
        --wbits "${START_BITS}" \
        --target_bits "${TARGET_BITS}" \
        --group_size "${GROUP_SIZE}" \
        --calib_dataset "${CALIB_DATASET}" \
        --train_size "${TRAIN_SIZE}" \
        --val_size "${VAL_SIZE}" \
        --training_seqlen "${TRAINING_SEQLEN}" \
        --epochs "${BLOCK_EPOCHS}" \
        --batch_size "${BLOCK_BATCH_SIZE}" \
        --weight_lr "${WEIGHT_LR}" \
        --quant_lr "${QUANT_LR}" \
        --save_quant_dir "${BLOCK_SAVE_DIR}" \
        --output_dir "${BLOCK_LOG_DIR}" \
        --use_progressive \
        --r_shape_mode group \
        --rho "${PROGRESSIVE_RHO}" \
        --progressive_finalize_epochs 1 \
        --use_sam \
        --sam_rho "${BLOCK_ASAM_RHO}" \
        --real_quant
fi

if [[ "${SKIP_E2E}" != "1" ]]; then
    if [[ ! -f "${BLOCK_SAVE_DIR}/config.json" ]]; then
        echo "Missing block-wise checkpoint: ${BLOCK_SAVE_DIR}/config.json" >&2
        exit 1
    fi

    resume_args=()
    if [[ -n "${E2E_RESUME_FROM_CHECKPOINT:-}" ]]; then
        resume_args+=(--resume_from_checkpoint "${E2E_RESUME_FROM_CHECKPOINT}")
    fi

    echo "[Qwen3] Starting full e2e ASAM QAT: ${MODEL_TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" python e2e_main.py \
        --quant_model_path "${BLOCK_SAVE_DIR}" \
        --model_family "${MODEL_TAG}" \
        --wbits "${TARGET_BITS}" \
        --group_size "${GROUP_SIZE}" \
        --dataset "${E2E_DATASET}" \
        --redpajama_data_path "${REDPAJAMA_DATA_PATH}" \
        --dataset_format pt \
        --general_data_ratio "${E2E_GENERAL_RATIO}" \
        --pt_context_len "${E2E_CONTEXT_LEN}" \
        --output_dir "${E2E_OUTPUT_DIR}" \
        --max_steps "${E2E_MAX_STEPS}" \
        --per_device_train_batch_size "${E2E_BATCH_SIZE}" \
        --gradient_accumulation_steps "${E2E_GRAD_ACCUM}" \
        --learning_rate "${E2E_LR}" \
        --use_asam true \
        --asam_rho "${E2E_ASAM_RHO}" \
        --bf16 true \
        --gradient_checkpointing true \
        --max_memory_MB "${MAX_MEMORY_MB}" \
        --logging_steps 10 \
        --save_strategy steps \
        --save_steps "${E2E_SAVE_STEPS}" \
        --save_total_limit "${E2E_SAVE_TOTAL_LIMIT}" \
        --trust_local_checkpoint "${E2E_TRUST_LOCAL_CHECKPOINT}" \
        --report_to none \
        --do_train true \
        "${resume_args[@]}"
fi
