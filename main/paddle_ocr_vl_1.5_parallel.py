#!/bin/bash
# 并行调度脚本：在多个GPU上同时运行PaddleOCR-VL-1.6任务（保留旧文件名以兼容已有命令）
# GPU分配：0用于--resume恢复，其他GPU处理不同数据集

set -e

SCRIPT="main/paddle_ocr_vl_1.5.py"
MODES="tiny small base"
LOG_DIR="output/parallel_logs"
mkdir -p "$LOG_DIR"

# 定义任务队列：(GPU编号, 数据集, 模式)
# 可以根据需要调整任务分配
TASKS=(
    "2:distort"
    "3:replace_swap_5"
    "4:replace_swap_10"
    "5:replace_shuffle_5"
    "2:replace_shuffle_10"
    "3:random"
    "4:from_text"
)

run_task() {
    local gpu=$1
    local dataset=$2
    shift 2
    local mode=$1
    local logfile="$LOG_DIR/gpu${gpu}_${dataset}_${mode}.log"

    echo "[$(date)] GPU $gpu: Starting $dataset / $mode"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$SCRIPT" \
        --dataset-preset single \
        --datasets "$dataset" \
        --modes "$mode" \
        --pipeline-version v1.6 \
        --device "gpu:0" \
        --resume \
        2>&1 | tee "$logfile"
    echo "[$(date)] GPU $gpu: Finished $dataset / $mode"
}

# 先在GPU 0-1上恢复已完成的任务
echo "=== Resume phase on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT" \
    --dataset-preset paper-experiments \
    --modes "$MODES" \
    --pipeline-version v1.6 \
    --device gpu:0 \
    --resume \
    2>&1 | tee "$LOG_DIR/resume_gpu0.log" &

# 在其他GPU上并行处理不同数据集的特定模式
echo "=== Parallel phase ==="
for task in "${TASKS[@]}"; do
    IFS=':' read -r gpu dataset <<< "$task"

    # 每个GPU分配一个数据集，先处理tiny模式
    (
        CUDA_VISIBLE_DEVICES=$gpu python3 "$SCRIPT" \
            --dataset-preset single \
            --datasets "$dataset" \
            --modes tiny \
            --pipeline-version v1.6 \
            --device gpu:0 \
            --resume \
            2>&1 | tee "$LOG_DIR/gpu${gpu}_${dataset}_tiny.log"
    ) &

    sleep 2  # 简单避免同时启动造成的IO竞争
done

echo "=== All parallel tasks started ==="
echo "Logs available in: $LOG_DIR/"
echo "Use 'tail -f $LOG_DIR/*.log' to monitor progress"
echo "Use 'jobs' to see running tasks"
echo "Use 'kill \$(jobs -p)' to cancel all"
wait
echo "=== All tasks completed ==="
