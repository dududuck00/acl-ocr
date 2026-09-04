#!/usr/bin/env python3
"""
PaddleOCR-VL 多GPU并行推理脚本
使用方法：
    # 单GPU模式（默认）
    CUDA_VISIBLE_DEVICES=0 python3 main/paddle_ocr_vl_1.5_multigpu.py ...

    # 使用多GPU并行（自动分配任务）
    python3 main/paddle_ocr_vl_1.5_multigpu.py \
        --dataset-preset paper-experiments \
        --modes tiny small base \
        --num-gpus 4 \
        --gpus 2,3,4,5

    # 恢复到指定GPU继续（不影响并行worker）
    CUDA_VISIBLE_DEVICES=2 python3 main/paddle_ocr_vl_1.5.py \
        --dataset-preset paper-experiments \
        --modes tiny small base \
        --device gpu:0 \
        --resume
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# ========== 复用的辅助函数 ==========
DEFAULT_MODES = ["tiny", "small", "base"]
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_MODEL_LABEL = "paddleocr_vl_1.6"
PAPER_EXPERIMENT_DATASETS = [
    "distort",
    "replace_swap_5",
    "replace_swap_10",
    "replace_shuffle_5",
    "replace_shuffle_10",
    "random",
]
DATASET_PRESETS = {
    "single": [],
    "paper-experiments": PAPER_EXPERIMENT_DATASETS,
    "all": ["from_text", *PAPER_EXPERIMENT_DATASETS],
}
TEXT_KEYS = {
    "markdown", "md", "text", "content", "block_content",
    "rec_text", "rec_texts", "ocr_text", "html", "latex",
}


def resolve_path(path, repo_root=REPO_ROOT):
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ========== 单GPU推理的核心逻辑 ==========
def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value).strip()


def collect_text_from_result(value, depth=0, seen=None):
    if seen is None:
        seen = set()
    if depth > 8 or value is None:
        return []

    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, (int, float, bool)):
        return []

    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if str(key).lower() in TEXT_KEYS:
                parts.extend(collect_text_from_result(item, depth + 1, seen))
        if parts:
            return parts
        for item in value.values():
            parts.extend(collect_text_from_result(item, depth + 1, seen))
        return parts

    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(collect_text_from_result(item, depth + 1, seen))
        return parts

    for attr in ("markdown", "md", "text", "content", "res", "json"):
        if hasattr(value, attr):
            try:
                parts = collect_text_from_result(getattr(value, attr), depth + 1, seen)
            except Exception:
                parts = []
            if parts:
                return parts

    for method in ("to_dict", "dict", "model_dump"):
        if hasattr(value, method):
            try:
                parts = collect_text_from_result(getattr(value, method)(), depth + 1, seen)
            except Exception:
                parts = []
            if parts:
                return parts

    if hasattr(value, "__dict__"):
        try:
            parts = collect_text_from_result(vars(value), depth + 1, seen)
        except Exception:
            parts = []
        if parts:
            return parts

    text = normalize_text(value)
    return [text] if text and text != repr(value) else []


def extract_ocr_text(prediction):
    parts = collect_text_from_result(prediction)
    deduped = []
    seen = set()
    for part in parts:
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            deduped.append(part)
    return "\n".join(deduped).strip()


def load_pipeline(device="gpu:0", **kwargs):
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise ImportError(
            "PaddleOCRVL is required. Activate the paddleocrvl environment "
            "or install paddleocr[doc-parser] before running this script."
        ) from exc

    final_kwargs = {"device": device}
    # 只传递非None的值
    for k, v in kwargs.items():
        if v is not None:
            final_kwargs[k] = v

    return PaddleOCRVL(**final_kwargs)


# ========== 多GPU并行任务函数 ==========
def worker_task(args_tuple):
    """单个worker进程的执行函数"""
    (task_id, dataset, mode, task_args) = args_tuple

    # 每个worker使用独立的GPU
    gpu_id = task_args["gpu_id"]
    print(f"[Worker {task_id}] GPU {gpu_id}: Processing {dataset}/{mode}")

    # 设置该worker的环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TMPDIR"] = str(REPO_ROOT / ".codex" / "tmp" / f"worker_{gpu_id}")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        return {
            "task_id": task_id,
            "dataset": dataset,
            "mode": mode,
            "gpu": gpu_id,
            "success": False,
            "error": str(exc),
            "processed": 0,
        }

    # 加载模型（每个worker独立加载）
    pipeline_kwargs = {
        "pipeline_version": task_args.get("pipeline_version"),
        "device": "gpu:0",  # 在这个进程内，GPU 0 就是分配的GPU
    }
    for k in ["use_doc_orientation_classify", "use_doc_unwarping",
              "use_layout_detection", "use_chart_recognition",
              "use_seal_recognition", "use_ocr_for_image_block",
              "format_block_content", "merge_layout_blocks"]:
        if k in task_args:
            pipeline_kwargs[k] = task_args[k]

    try:
        pipeline = PaddleOCRVL(**pipeline_kwargs)
    except Exception as exc:
        return {
            "task_id": task_id,
            "dataset": dataset,
            "mode": mode,
            "gpu": gpu_id,
            "success": False,
            "error": f"Failed to load pipeline: {exc}",
            "processed": 0,
        }

    # 准备路径结构: data_base/dataset/mode/
    mode_dir = resolve_path(task_args["data_base"]) / dataset / mode
    image_dir = mode_dir / "images"
    output_base = REPO_ROOT / "results" / "other" / f"{task_args['model_label']}_{dataset}_deepseek_modes"
    output_path = output_base / f"{task_args['model_label']}_{dataset}_{mode}.json"

    # 加载数据
    data_path = mode_dir / "data.json"
    if not data_path.exists():
        return {
            "task_id": task_id,
            "dataset": dataset,
            "mode": mode,
            "gpu": gpu_id,
            "success": False,
            "error": f"Data file not found: {data_path}",
            "processed": 0,
        }

    data = load_json(data_path)
    if task_args.get("limit"):
        data = data[:task_args["limit"]]

    # 读取已完成的图片
    completed = set()
    existing = []
    if output_path.exists():
        try:
            existing = load_json(output_path)
            completed = {
                item.get("processed_image", item.get("image"))
                for item in existing
                if item.get("ocr_text") and not item.get("error")
            }
        except Exception:
            pass

    # 过滤待处理项
    todo = [item for item in data if item.get("image") not in completed]
    print(f"[Worker {task_id}] GPU {gpu_id}: {dataset}/{mode}: {len(todo)}/{len(data)} remaining")

    # 处理图片
    results = []
    processed_count = 0
    for item in tqdm(todo, desc=f"{dataset}/{mode}", unit="image"):
        image_name = item["image"]
        image_path = image_dir / image_name

        if not image_path.exists():
            continue

        last_error = None
        for attempt in range(task_args.get("retries", 1) + 1):
            try:
                output = pipeline.predict(str(image_path))
                ocr_text = extract_ocr_text(output)

                result = dict(item)
                if item.get("source_image") and not task_args.get("keep_processed_image_name"):
                    result["processed_image"] = image_name
                    result["image"] = item["source_image"]
                else:
                    result["image"] = image_name
                result["ocr_text"] = ocr_text
                result["model_name"] = "PaddleOCR-VL-1.6"
                result["pipeline_version"] = task_args.get("pipeline_version")
                results.append(result)
                processed_count += 1
                break
            except Exception as exc:
                last_error = exc
                if attempt < task_args.get("retries", 1):
                    time.sleep(task_args.get("retry_sleep", 2.0) * (attempt + 1))

    # 合并并保存结果
    if results:
        all_items = existing + results
        # 去重
        seen = set()
        deduped = []
        for item in all_items:
            key = item.get("processed_image", item.get("image"))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        # 按原始顺序排序
        order = {item["image"]: idx for idx, item in enumerate(data)}
        deduped.sort(key=lambda x: order.get(x.get("processed_image", x.get("image")), 10**9))
        save_json(deduped, output_path)

    print(f"[Worker {task_id}] GPU {gpu_id}: {dataset}/{mode} done, processed {processed_count}")
    return {
        "task_id": task_id,
        "dataset": dataset,
        "mode": mode,
        "gpu": gpu_id,
        "success": True,
        "processed": processed_count,
        "total": len(todo),
    }


def str_to_optional_bool(value):
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL-1.6 on multiple GPUs in parallel (legacy filename)."
    )
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use (0=all available)")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs, e.g. '2,3,4,5'")
    parser.add_argument("--dataset-preset", default="single",
                        choices=sorted(DATASET_PRESETS))
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset directory names")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--pipeline-version", default="v1.6", choices=["v1.6"])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--keep-processed-image-name", action="store_true")
    # 高级选项
    parser.add_argument("--use-doc-orientation-classify", type=str_to_optional_bool, default=False)
    parser.add_argument("--use-doc-unwarping", type=str_to_optional_bool, default=False)
    parser.add_argument("--use-layout-detection", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-chart-recognition", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-seal-recognition", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-ocr-for-image-block", type=str_to_optional_bool, default=None)
    parser.add_argument("--format-block-content", type=str_to_optional_bool, default=None)
    parser.add_argument("--merge-layout-blocks", type=str_to_optional_bool, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    # 确定要使用的数据集
    datasets = args.datasets if args.datasets else DATASET_PRESETS[args.dataset_preset]
    if not datasets:
        datasets = ["from_text"]

    # 确定GPU列表
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        # 默认使用所有可见GPU
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            gpu_ids = [int(g) for g in cuda_visible.split(",") if g.strip()]
        else:
            # 尝试检测
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--list-gpus"],
                    capture_output=True, text=True, timeout=5
                )
                gpu_ids = list(range(len(result.stdout.strip().split("\n"))))
            except Exception:
                gpu_ids = [0]

    # 只使用指定的GPU数量
    if args.num_gpus > 0 and args.num_gpus < len(gpu_ids):
        gpu_ids = gpu_ids[:args.num_gpus]

    # 构建任务队列：(dataset, mode)
    tasks = []
    for dataset in datasets:
        for mode in args.modes:
            tasks.append((dataset, mode))

    print(f"=== Multi-GPU Parallel Mode ===")
    print(f"GPUs: {gpu_ids}")
    print(f"Datasets: {datasets}")
    print(f"Modes: {args.modes}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Tasks per GPU: ~{len(tasks) // len(gpu_ids)}")

    # 构建worker参数
    worker_args = {
        "data_base": DEFAULT_DATA_BASE,
        "model_label": DEFAULT_MODEL_LABEL,
        "pipeline_version": args.pipeline_version,
        "retries": args.retries,
        "retry_sleep": args.retry_sleep,
        "limit": args.limit,
        "keep_processed_image_name": args.keep_processed_image_name,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
        "use_layout_detection": args.use_layout_detection,
        "use_chart_recognition": args.use_chart_recognition,
        "use_seal_recognition": args.use_seal_recognition,
        "use_ocr_for_image_block": args.use_ocr_for_image_block,
        "format_block_content": args.format_block_content,
        "merge_layout_blocks": args.merge_layout_blocks,
    }

    # 将任务分配到GPU（轮询分配）
    task_assignments = []
    for idx, (dataset, mode) in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        task_assignments.append((idx, dataset, mode, {**worker_args, "gpu_id": gpu_id}))

    # 使用进程池执行
    start_time = time.time()
    completed_tasks = []
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(worker_task, task): task for task in task_assignments}

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                completed_tasks.append(result)
                print(f"✓ {result['dataset']}/{result['mode']} on GPU {result['gpu']}: "
                      f"{result['processed']}/{result['total']} processed")
            else:
                failed_tasks.append(result)
                print(f"✗ {result['dataset']}/{result['mode']}: {result.get('error', 'Unknown error')}")

    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Completed: {len(completed_tasks)}/{len(tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    if failed_tasks:
        print("Failed tasks:")
        for t in failed_tasks:
            print(f"  - {t['dataset']}/{t['mode']}: {t.get('error')}")

    return 0 if not failed_tasks else 1


if __name__ == "__main__":
    sys.exit(main())
