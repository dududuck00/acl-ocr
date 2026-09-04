#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

WEIGHTS_DIR="${WEIGHTS_DIR:-/home/liangyunhao/shared/models/echo840/MonkeyOCR-pro-3B}"
MONKEYOCR_REPO="${MONKEYOCR_REPO:-/home/liangyunhao/shared/liangyunhao/code/MonkeyOCR}"

DATA_BASE="${DATA_BASE:-fox_data/deepseek_mode_images}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-output/monkeyocr_deepseek_experiments}"
RESULT_DIR="${RESULT_DIR:-results/other/monkeyocr_deepseek_experiments}"

BACKEND="${BACKEND:-lmdeploy}"
TASK="${TASK:-parse}"
MODES="${MODES:-tiny base small}"
DATASETS="${DATASETS:-distort replace_swap_5 replace_swap_10 replace_shuffle_5 replace_shuffle_10 random}"

read -r -a MODE_LIST <<< "${MODES}"
read -r -a DATASET_LIST <<< "${DATASETS}"

for dataset in "${DATASET_LIST[@]}"; do
  echo "==> Running MonkeyOCR 3B on ${dataset}: ${MODE_LIST[*]}"
  "${PYTHON_BIN}" main/monkey_ocr.py \
    --monkeyocr-repo "${MONKEYOCR_REPO}" \
    --weights-dir "${WEIGHTS_DIR}" \
    --model-label "monkeyocr_3B_${dataset}" \
    --data-root "${DATA_BASE}/${dataset}" \
    --raw-output-dir "${RAW_OUTPUT_DIR}" \
    --result-dir "${RESULT_DIR}" \
    --modes "${MODE_LIST[@]}" \
    --cuda-visible-devices "${GPU_ID}" \
    --backend "${BACKEND}" \
    --task "${TASK}" \
    --resume
done
