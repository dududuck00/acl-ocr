#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PADDLEOCR_PYTHON:-python}"
ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then
    shift
fi

COMMON_ARGS=(
    --max-workers "${PADDLEOCR_WORKERS:-3}"
    --checkpoint-every 5
    --resume
)

case "$ACTION" in
    check)
        "$PYTHON_BIN" main/paddle_ocr_v6_api_cross.py --dry-run --resume "$@"
        ;;
    run)
        "$PYTHON_BIN" main/paddle_ocr_v6_api_cross.py "${COMMON_ARGS[@]}" "$@"
        ;;
    eval)
        "$PYTHON_BIN" scripts/evaluate_paddle_ocr_v6_cross.py "$@"
        ;;
    all)
        "$PYTHON_BIN" main/paddle_ocr_v6_api_cross.py "${COMMON_ARGS[@]}" "$@"
        "$PYTHON_BIN" scripts/evaluate_paddle_ocr_v6_cross.py
        ;;
    *)
        echo "Usage: $0 {check|run|eval|all} [extra arguments]" >&2
        exit 2
        ;;
esac
