import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ["tiny", "small", "base"]
DEFAULT_MONKEYOCR_REPO = Path("/home/liangyunhao/shared/liangyunhao/code/MonkeyOCR")
DEFAULT_WEIGHTS_DIR = Path("/home/liangyunhao/shared/models/echo840/MonkeyOCR")
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = REPO_ROOT / "fox_data" / "deepseek_mode_images" / "from_text"
DEFAULT_RAW_OUTPUT_DIR = REPO_ROOT / "output" / "monkeyocr_deepseek_modes"
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "other" / "monkeyocr_from_text_deepseek_modes"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
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


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_config(args):
    config_dir = REPO_ROOT / ".codex" / "tmp" / "monkeyocr_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.model_label}_{args.backend}.yaml"
    config_text = f"""device: {args.device}
weights:
  PP-DocLayoutV2: Structure/PP-DocLayoutV2
  layoutreader: Relation
models_dir: {args.weights_dir}
layout_config:
  model: PP-DocLayoutV2
  reader:
    name: layoutreader
chat_config:
  weight_path: Recognition
  backend: {args.backend}
  data_parallelism: {args.data_parallelism}
  model_parallelism: {args.model_parallelism}
  batch_size: {args.batch_size}
  queue_config:
    max_batch_size: 256
    queue_timeout: 1
    max_queue_size: 2000
"""
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def import_monkeyocr_parse(monkeyocr_repo):
    monkeyocr_repo = str(monkeyocr_repo)
    if monkeyocr_repo not in sys.path:
        sys.path.insert(0, monkeyocr_repo)
    from parse import parse_folder

    return parse_folder


def prepare_runtime_dirs(monkeyocr_repo):
    tmp_dir = REPO_ROOT / ".codex" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))

    # Some MonkeyOCR API/demo paths mount a relative "static/" directory at import time.
    # Create both likely locations so accidental imports do not fail before batch parsing starts.
    (REPO_ROOT / "static").mkdir(parents=True, exist_ok=True)
    (monkeyocr_repo / "static").mkdir(parents=True, exist_ok=True)


def disable_gradio_mount_if_present():
    try:
        import gradio as gr
    except Exception:
        return

    def _skip_mount_gradio_app(app, *args, **kwargs):
        return app

    gr.mount_gradio_app = _skip_mount_gradio_app


def patch_transformers_attn_implementation(attn_implementation):
    if not attn_implementation:
        return

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        return

    original_from_pretrained = Qwen2_5_VLForConditionalGeneration.from_pretrained

    def _from_pretrained_with_attn_override(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = attn_implementation
        return original_from_pretrained(*args, **kwargs)

    Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(_from_pretrained_with_attn_override)


def read_parse_result(raw_mode_dir, image_name):
    stem = Path(image_name).stem
    result_path = raw_mode_dir / stem / f"{stem}_content_list.json"
    if not result_path.exists():
        return None, result_path
    content = load_json(result_path)
    text_parts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") in {"text", "equation", "table"} and item.get("text")
    ]
    return "\n".join(text_parts).strip(), result_path


def read_text_result(raw_mode_dir, image_name):
    stem = Path(image_name).stem
    result_path = raw_mode_dir / stem / f"{stem}_text_result.md"
    if not result_path.exists():
        return None, result_path
    return result_path.read_text(encoding="utf-8").strip(), result_path


def read_ocr_result(args, raw_mode_dir, image_name):
    if args.task == "text":
        return read_text_result(raw_mode_dir, image_name)
    return read_parse_result(raw_mode_dir, image_name)


def output_item_for_eval(item, ocr_text, keep_processed_image_name):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    return output_item


def aggregate_mode(args, mode):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    raw_label = args.raw_label or args.model_label
    result_prefix = args.result_prefix or f"{args.model_label}_from_text"
    raw_mode_dir = resolve_path(args.raw_output_dir) / raw_label / mode
    result_path = resolve_path(args.result_dir) / f"{result_prefix}_{mode}.json"

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    results = []
    missing = []
    for item in data:
        ocr_text, text_path = read_ocr_result(args, raw_mode_dir, item["image"])
        if ocr_text is None:
            missing.append(str(text_path))
            continue
        results.append(output_item_for_eval(item, ocr_text, args.keep_processed_image_name))

    save_json(results, result_path)
    print(f"[{args.model_label}][{mode}] aggregated {len(results)}/{len(data)} predictions to {result_path}")
    if missing:
        print(f"[{args.model_label}][{mode}] missing {len(missing)} text results; first missing: {missing[0]}")


def run_mode(args, mode, parse_folder):
    mode_dir = resolve_path(args.data_root) / mode
    image_dir = mode_dir / "images"
    raw_label = args.raw_label or args.model_label
    raw_mode_dir = resolve_path(args.raw_output_dir) / raw_label / mode

    if args.limit is not None:
        limited_image_dir = raw_mode_dir / "_limited_input"
        limited_image_dir.mkdir(parents=True, exist_ok=True)
        data = load_json(mode_dir / "data.json")[: args.limit]
        for item in tqdm(data, desc=f"Link {mode}", unit="image"):
            src = image_dir / item["image"]
            dst = limited_image_dir / item["image"]
            if not dst.exists():
                dst.symlink_to(src)
        image_dir = limited_image_dir

    print(f"[{args.model_label}][{mode}] images: {image_dir}")
    print(f"[{args.model_label}][{mode}] raw output: {raw_mode_dir}")

    if not args.aggregate_only:
        parse_folder(
            folder_path=str(image_dir),
            output_dir=str(raw_mode_dir),
            config_path=str(args.config_path),
            task=None if args.task == "parse" else args.task,
            group_size=args.group_size,
            skip_processed=args.resume,
        )

    aggregate_mode(args, mode)


def run_dataset(args, dataset, parse_folder):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.result_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.result_prefix = f"{args.model_label}_{dataset}"
    dataset_args.raw_label = f"{args.model_label}_{dataset}"

    print(f"[{args.model_label}][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, parse_folder)
    print(f"[{args.model_label}][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def split_evenly(items, num_groups):
    groups = [[] for _ in range(num_groups)]
    for index, item in enumerate(items):
        groups[index % num_groups].append(item)
    return [group for group in groups if group]


def add_flag(command, flag, value):
    if value is None:
        return
    if isinstance(value, Path):
        value = str(value)
    command.extend([flag, str(value)])


def build_launch_command(args, gpu_id, datasets):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--monkeyocr-repo",
        str(args.monkeyocr_repo),
        "--weights-dir",
        str(args.weights_dir),
        "--model-label",
        args.model_label,
        "--data-base",
        str(resolve_path(args.data_base)),
        "--raw-output-dir",
        str(resolve_path(args.raw_output_dir)),
        "--results-base",
        str(resolve_path(args.results_base)),
        "--cuda-visible-devices",
        str(gpu_id),
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--task",
        args.task,
        "--data-parallelism",
        str(args.data_parallelism),
        "--model-parallelism",
        str(args.model_parallelism),
        "--batch-size",
        str(args.batch_size),
        "--transformers-attn-implementation",
        args.transformers_attn_implementation,
        "--datasets",
        *datasets,
        "--modes",
        *args.modes,
    ]
    add_flag(command, "--group-size", args.group_size)
    add_flag(command, "--limit", args.limit)
    if args.resume:
        command.append("--resume")
    if args.aggregate_only:
        command.append("--aggregate-only")
    if args.keep_processed_image_name:
        command.append("--keep-processed-image-name")
    return command


def launch_across_gpus(args):
    datasets = selected_datasets(args)
    if not datasets:
        raise ValueError("--launch-gpus requires --datasets or a non-single --dataset-preset.")

    gpu_ids = [str(gpu_id) for gpu_id in args.launch_gpus]
    assignments = split_evenly(datasets, min(len(gpu_ids), len(datasets)))
    log_dir = REPO_ROOT / ".codex" / "tmp" / "monkeyocr_launch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    print(f"[launcher] datasets: {' '.join(datasets)}")
    print(f"[launcher] gpus: {' '.join(gpu_ids)}")
    for gpu_id, dataset_group in zip(gpu_ids, assignments):
        command = build_launch_command(args, gpu_id, dataset_group)
        log_path = log_dir / f"{args.model_label}_gpu{gpu_id}_{'_'.join(dataset_group)}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
        )
        processes.append((process, log_file, log_path, gpu_id, dataset_group))
        print(f"[launcher] gpu {gpu_id}: {' '.join(dataset_group)} -> {log_path}")

    failed = []
    try:
        while processes:
            still_running = []
            for process, log_file, log_path, gpu_id, dataset_group in processes:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((process, log_file, log_path, gpu_id, dataset_group))
                    continue
                log_file.close()
                status = "ok" if return_code == 0 else f"failed({return_code})"
                print(f"[launcher] gpu {gpu_id}: {' '.join(dataset_group)} {status}; log: {log_path}")
                if return_code != 0:
                    failed.append((gpu_id, dataset_group, log_path, return_code))
            processes = still_running
            if processes:
                time.sleep(args.launch_poll_interval)
    except KeyboardInterrupt:
        print("[launcher] interrupted; terminating child processes...")
        for process, log_file, _, _, _ in processes:
            process.terminate()
            log_file.close()
        raise

    if failed:
        print("[launcher] failed jobs:")
        for gpu_id, dataset_group, log_path, return_code in failed:
            print(f"  gpu {gpu_id}: {' '.join(dataset_group)} exit={return_code} log={log_path}")
        raise SystemExit(1)

    print("[launcher] all jobs finished successfully.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MonkeyOCR text recognition on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--monkeyocr-repo", default=str(DEFAULT_MONKEYOCR_REPO))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--model-label", default="monkeyocr_3B")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR))
    parser.add_argument("--raw-label", default=None)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--result-prefix", default=None)
    parser.add_argument("--results-base", default=str(DEFAULT_RESULTS_BASE))
    parser.add_argument("--dataset-preset", default="single", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directory names under --data-base, e.g. from_text distort random.",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--backend", default="lmdeploy", choices=["lmdeploy", "vllm", "transformers"])
    parser.add_argument(
        "--task",
        default="parse",
        choices=["parse", "text"],
        help="parse uses MonkeyOCR end-to-end content_list output; text uses direct single-task text recognition.",
    )
    parser.add_argument("--data-parallelism", type=int, default=1)
    parser.add_argument("--model-parallelism", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--transformers-attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Override MonkeyOCR transformers backend attention implementation. sdpa avoids flash-attn.",
    )
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples per mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip MonkeyOCR raw output folders that already exist.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild result JSON from existing raw outputs.")
    parser.add_argument(
        "--launch-gpus",
        nargs="+",
        default=None,
        help="Launch one child process per GPU and split selected datasets automatically, e.g. --launch-gpus 2 3 4 5 6 7.",
    )
    parser.add_argument("--launch-poll-interval", type=float, default=30.0)
    parser.add_argument(
        "--keep-processed-image-name",
        action="store_true",
        help="Keep image=en_1_tiny.png in output. By default image is restored to source_image for eval/eval.py.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.monkeyocr_repo = resolve_path(args.monkeyocr_repo)
    args.weights_dir = resolve_path(args.weights_dir)

    if not args.monkeyocr_repo.exists():
        raise FileNotFoundError(f"MonkeyOCR repo not found: {args.monkeyocr_repo}")
    if not args.weights_dir.exists():
        raise FileNotFoundError(f"MonkeyOCR weights dir not found: {args.weights_dir}")

    if args.launch_gpus:
        prepare_runtime_dirs(args.monkeyocr_repo)
        launch_across_gpus(args)
        return

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    prepare_runtime_dirs(args.monkeyocr_repo)
    disable_gradio_mount_if_present()
    if args.backend == "transformers":
        patch_transformers_attn_implementation(args.transformers_attn_implementation)
    args.config_path = write_config(args)
    parse_folder = import_monkeyocr_parse(args.monkeyocr_repo)

    datasets = selected_datasets(args)
    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset, parse_folder)
    else:
        for mode in args.modes:
            run_mode(args, mode, parse_folder)


if __name__ == "__main__":
    main()
