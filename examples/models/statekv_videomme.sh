#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-1B}"
METHOD="${2:-statekv}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/videomme}"

case "${MODEL_SIZE}" in
  1B|2B|8B) ;;
  *) echo "MODEL_SIZE must be one of: 1B, 2B, 8B" >&2; exit 2 ;;
esac

PRETRAINED="OpenGVLab/InternVL3-${MODEL_SIZE}"
MODEL_ARGS="pretrained=${PRETRAINED},max_num_frames=512,max_fps=1"

case "${METHOD}" in
  full)
    LMMS_MODEL="internvl3"
    ;;
  rekv)
    LMMS_MODEL="internvl3_rekv"
    MODEL_ARGS="${MODEL_ARGS},retrieved_frames=16"
    ;;
  statekv)
    LMMS_MODEL="statekv_internvl3"
    MODEL_ARGS="${MODEL_ARGS},cstate_size=4096,cache_attn_implementation=triton"
    ;;
  *) echo "METHOD must be one of: full, rekv, statekv" >&2; exit 2 ;;
esac

MODEL_SIZE_LOWER="$(printf '%s' "${MODEL_SIZE}" | tr '[:upper:]' '[:lower:]')"
OUTPUT_PATH="${OUTPUT_ROOT}/internvl3-${MODEL_SIZE_LOWER}/${METHOD}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  -m lmms_eval \
  --model "${LMMS_MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_PATH}"
