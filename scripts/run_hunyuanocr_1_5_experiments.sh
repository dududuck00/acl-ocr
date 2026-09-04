#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/.codex/tmp"
export TMPDIR="$REPO_ROOT/.codex/tmp"
export TMP="$REPO_ROOT/.codex/tmp"
export TEMP="$REPO_ROOT/.codex/tmp"

PYTHON_BIN="${HUNYUAN_PYTHON:-python}"
EVAL_PYTHON_BIN="${HUNYUAN_EVAL_PYTHON:-$PYTHON_BIN}"
MODEL_PATH="${HUNYUAN_MODEL_PATH:-/home/liangyunhao/shared/models/tencent/HunyuanOCR}"
GPU_IDS="${HUNYUAN_GPU:-0}"
ATTENTION="${HUNYUAN_ATTN:-auto}"
ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then
    shift
fi

COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --cuda-visible-devices "$GPU_IDS"
    --attn-implementation "$ATTENTION"
    --torch-dtype bfloat16
    --max-new-tokens 8192
    --checkpoint-every 5
    --resume
)

run_native() {
    "$PYTHON_BIN" main/hunyuan_ocr_1_5.py \
        --protocol native \
        "${COMMON_ARGS[@]}" \
        "$@"
}

run_cross_arch() {
    "$PYTHON_BIN" main/hunyuan_ocr_1_5.py \
        --protocol cross_arch \
        "${COMMON_ARGS[@]}" \
        "$@"
}

case "$ACTION" in
    check)
        "$PYTHON_BIN" main/hunyuan_ocr_1_5.py \
            --protocol all \
            --model-path "$MODEL_PATH" \
            --dry-run \
            "$@"
        ;;
    smoke)
        "$PYTHON_BIN" main/hunyuan_ocr_1_5.py \
            --protocol native \
            "${COMMON_ARGS[@]}" \
            --native-datasets from_text \
            --limit 1 \
            --results-dir results/smoke/hunyuanocr_1.5 \
            "$@"
        ;;
    main)
        run_native "$@"
        ;;
    cross)
        run_cross_arch "$@"
        ;;
    all)
        run_native "$@"
        run_cross_arch "$@"
        ;;
    eval-main)
        "$EVAL_PYTHON_BIN" scripts/evaluate_hunyuanocr_1_5.py --protocol native "$@"
        ;;
    eval-cross)
        "$EVAL_PYTHON_BIN" scripts/evaluate_hunyuanocr_1_5.py --protocol cross_arch "$@"
        ;;
    eval-all)
        "$EVAL_PYTHON_BIN" scripts/evaluate_hunyuanocr_1_5.py --protocol all "$@"
        ;;
    *)
        echo "Usage: $0 {check|smoke|main|cross|all|eval-main|eval-cross|eval-all} [extra arguments]" >&2
        exit 2
        ;;
esac
