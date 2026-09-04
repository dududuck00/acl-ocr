#!/usr/bin/env python3
"""
MonkeyOCR 多GPU并行执行脚本
使用方法:
    python3 main/monkey_ocr_multigpu.py \
        --datasets replace_swap_5 replace_swap_10 replace_shuffle_5 replace_shuffle_10 random distort \
        --modes tiny small base \
        --gpus 2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

REPO_ROOT = Path(__file__).resolve().parents[1]

# 默认数据集（paper-experiments 不含 from_text）
PAPER_DATASETS = ["distort", "replace_swap_5", "replace_swap_10", "replace_shuffle_5", "replace_shuffle_10", "random"]

SCRIPT_DIR = REPO_ROOT / "main"


def build_parser():
    parser = argparse.ArgumentParser(description="MonkeyOCR 多GPU并行执行")
    parser.add_argument("--monkeyocr-repo", default="/home/liangyunhao/shared/liangyunhao/code/MonkeyOCR")
    parser.add_argument("--weights-dir", default="/home/liangyunhao/shared/models/echo840/MonkeyOCR-pro-1.2B")
    parser.add_argument("--gpus", default="2,3,4,5,6,7", help="GPU ID列表，逗号分隔")
    parser.add_argument("--datasets", nargs="+", default=PAPER_DATASETS, help="数据集名称")
    parser.add_argument("--modes", nargs="+", default=["tiny", "small", "base"], help="模式: tiny small base")
    parser.add_argument("--backend", default="transformers", choices=["lmdeploy", "vllm", "transformers"])
    parser.add_argument("--task", default="parse")
    parser.add_argument("--resume", action="store_true", help="跳过已处理的图片")
    return parser


def run_single_task(gpu_id, dataset, mode, args):
    """在指定GPU上运行单个任务"""
    model_label = f"monkeyocr_1.2B_{dataset}"
    data_root = f"fox_data/deepseek_mode_images/{dataset}"

    cmd = [
        "python3",
        str(SCRIPT_DIR / "monkey_ocr.py"),
        "--monkeyocr-repo", args.monkeyocr_repo,
        "--weights-dir", args.weights_dir,
        "--model-label", model_label,
        "--data-root", data_root,
        "--raw-output-dir", "output/monkeyocr_deepseek_experiments",
        "--result-dir", "results/other/monkeyocr_deepseek_experiments",
        "--modes", mode,
        "--cuda-visible-devices", str(gpu_id),
        "--backend", args.backend,
        "--task", args.task,
    ]

    if args.resume:
        cmd.append("--resume")

    print(f"[GPU {gpu_id}] {dataset}/{mode}: 开始")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] {dataset}/{mode}: 完成")
            return True, None
        else:
            print(f"[GPU {gpu_id}] {dataset}/{mode}: 失败 - {result.stderr[:200]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] {dataset}/{mode}: 超时")
        return False, "Timeout"
    except Exception as e:
        print(f"[GPU {gpu_id}] {dataset}/{mode}: 错误 - {e}")
        return False, str(e)


def main():
    args = build_parser().parse_args()

    # 解析GPU列表
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"使用GPU: {gpu_ids}")
    print(f"数据集: {args.datasets}")
    print(f"模式: {args.modes}")

    # 构建任务列表 (dataset, mode)
    tasks = list(product(args.datasets, args.modes))
    print(f"总任务数: {len(tasks)}")

    # 分配任务到GPU（轮询）
    task_assignments = []
    for i, (dataset, mode) in enumerate(tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        task_assignments.append((gpu_id, dataset, mode))

    # 统计
    completed = 0
    failed = 0

    # 并行执行（每个GPU一个worker）
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run_single_task, gpu_id, dataset, mode, args): (gpu_id, dataset, mode)
            for gpu_id, dataset, mode in task_assignments
        }

        for future in as_completed(futures):
            gpu_id, dataset, mode = futures[future]
            try:
                success, error = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[GPU {gpu_id}] {dataset}/{mode}: 异常 - {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"完成: {completed}/{len(tasks)}")
    print(f"失败: {failed}/{len(tasks)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())