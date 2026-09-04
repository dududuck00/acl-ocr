import argparse
import json
import os
import queue as queue_module
import re
import time
from pathlib import Path


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ["tiny", "small", "base"]
DEFAULT_MODEL_PATH = "/home/liangyunhao/shared/models/deepseek-ai/DeepSeek-OCR"
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_BASE / "deepseek_ocr_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = "deepseek_ocr_from_text"
DEFAULT_MODEL_LABEL = "deepseek_ocr"
DEFAULT_PROMPT = "<image>\nFree OCR. "

PAPER_EXPERIMENT_DATASETS = [
    "distort",
    "replace_swap_5",
    "replace_swap_10",
    "replace_shuffle_5",
    "replace_shuffle_10",
    "random",
]
MASK_ABLATION_DATASETS = [
    "mask_clean",
    "mask_word_25",
    "mask_char_25",
    "mask_word_50",
    "mask_char_50",
    "mask_word_75",
    "mask_char_75",
    "mask_word_100",
    "mask_char_100",
    "noise_25",
    "noise_50",
    "noise_75",
    "noise_100",
]
DATASET_PRESETS = {
    "single": [],
    "paper-experiments": PAPER_EXPERIMENT_DATASETS,
    "all": ["from_text", *PAPER_EXPERIMENT_DATASETS],
    "mask-ablation": MASK_ABLATION_DATASETS,
}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def prepare_runtime_env(cuda_visible_devices):
    tmp_dir = REPO_ROOT / ".codex" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def check_runtime_dependencies():
    try:
        import transformers
        from transformers.models.llama import modeling_llama
    except ImportError as exc:
        raise ImportError(
            "DeepSeek-OCR local inference requires transformers. "
            "Install the project requirements before running this script."
        ) from exc

    if not hasattr(modeling_llama, "LlamaFlashAttention2"):
        raise RuntimeError(
            "Current transformers version is incompatible with DeepSeek-OCR remote code: "
            f"transformers=={transformers.__version__} does not expose LlamaFlashAttention2. "
            "Use transformers==4.46.3 for this local DeepSeek-OCR script, as pinned in requirements.txt."
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def mode_config(mode):
    configs = {
        "tiny": {"image_size": 512, "base_size": 512, "test_compress": True},
        "small": {"image_size": 640, "base_size": 640, "test_compress": True},
        "base": {"image_size": 1024, "base_size": 1024, "test_compress": True},
        "large": {"image_size": 1280, "base_size": 1280, "test_compress": True},
        "raw": {"image_size": 1024, "base_size": 1024, "test_compress": False},
    }
    return configs[mode]


def resolve_model_path_or_id(model_path):
    path = Path(model_path)
    if path.is_absolute():
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Local model path does not exist: {path}")

    repo_relative_path = REPO_ROOT / path
    if repo_relative_path.exists():
        return str(repo_relative_path)

    looks_like_local_path = (
        model_path.startswith(".")
        or model_path.startswith("model/")
        or model_path.startswith("models/")
        or "/" not in model_path
    )
    if looks_like_local_path:
        raise FileNotFoundError(
            "Local model path does not exist: "
            f"{repo_relative_path}\n"
            f"Pass the actual local checkpoint directory with --model-path, e.g. {DEFAULT_MODEL_PATH}"
        )
    return model_path


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for match in matches:
        if "<|ref|>image<|/ref|>" in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    return matches, matches_image, matches_other


def clean_ocr_output(text):
    _, matches_images, matches_other = re_match(text)
    for idx, match_image in enumerate(matches_images):
        text = text.replace(match_image, f"![](images/{idx}.jpg)\n")
    for match_other in matches_other:
        text = text.replace(match_other, "")

    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = text.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")
    text = text.replace("<center>", "").replace("</center>", "")
    return text.strip()


def output_item_for_eval(item, ocr_text, keep_processed_image_name, error=None):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    if error:
        output_item["error"] = error
    return output_item


def run_shard(worker_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_args["physical_gpu_id"])

    import torch
    from transformers import AutoModel, AutoTokenizer

    data = worker_args["data"]
    model_path = worker_args["model_path"]
    image_dir = Path(worker_args["image_dir"])
    output_path = Path(worker_args["output_path"])
    shard_file = Path(worker_args["shard_file"])
    cfg = worker_args["cfg"]
    gpu_id = worker_args["gpu_id"]
    physical_gpu_id = worker_args["physical_gpu_id"]
    shard_index = worker_args["shard_index"]
    progress_queue = worker_args.get("progress_queue")
    show_progress = worker_args.get("show_progress", True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.eval().to(device).to(dtype)

    results = []
    for item in tqdm(
        data,
        desc=f"GPU {gpu_id}({physical_gpu_id})",
        unit="image",
        position=shard_index,
        leave=True,
        disable=not show_progress,
    ):
        image_name = item["image"]
        image_file = image_dir / image_name
        if not image_file.exists():
            results.append(
                output_item_for_eval(
                    item,
                    "",
                    keep_processed_image_name=worker_args["keep_processed_image_name"],
                    error=f"Image not found: {image_file}",
                )
            )
            if progress_queue is not None:
                progress_queue.put(1)
            continue

        try:
            raw_text = model.infer(
                tokenizer=tokenizer,
                prompt=worker_args["prompt"],
                image_file=str(image_file),
                output_path=str(output_path),
                base_size=cfg["base_size"],
                image_size=cfg["image_size"],
                crop_mode=worker_args["crop_mode"],
                save_results=False,
                eval_mode=True,
                test_compress=cfg["test_compress"],
            )
            result = output_item_for_eval(
                item,
                clean_ocr_output(raw_text),
                keep_processed_image_name=worker_args["keep_processed_image_name"],
            )
        except Exception as exc:
            print(f"GPU {gpu_id}({physical_gpu_id}) error processing {image_name}: {exc}")
            result = output_item_for_eval(
                item,
                "",
                keep_processed_image_name=worker_args["keep_processed_image_name"],
                error=str(exc),
            )
        results.append(result)
        if progress_queue is not None:
            progress_queue.put(1)

    save_json(results, shard_file)


def split_round_robin(items, num_shards):
    shards = [[] for _ in range(num_shards)]
    for index, item in enumerate(items):
        shards[index % num_shards].append(item)
    return shards


def resolve_physical_gpu_ids(gpu_ids, cuda_visible_devices=None):
    if not cuda_visible_devices:
        return gpu_ids

    visible = [int(item.strip()) for item in cuda_visible_devices.split(",") if item.strip()]
    physical_gpu_ids = []
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= len(visible):
            raise ValueError(
                f"GPU id {gpu_id} is outside CUDA_VISIBLE_DEVICES={cuda_visible_devices}. "
                "When --cuda-visible-devices is set, --gpu-ids should use visible indices."
            )
        physical_gpu_ids.append(visible[gpu_id])
    return physical_gpu_ids


def visible_gpu_ids_from_args(args):
    if args.gpu_ids:
        return args.gpu_ids

    if args.cuda_visible_devices:
        visible = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
        if len(visible) > 1:
            return list(range(len(visible)))

    return [args.gpu_id]


def existing_completed_images(output_path):
    if not output_path.exists():
        return set(), []

    existing = load_json(output_path)
    completed = {
        item.get("processed_image", item.get("image"))
        for item in existing
        if item.get("ocr_text") and not item.get("error")
    }
    return completed, existing


def merge_results(data, existing, results, resume):
    if resume:
        by_key = {}
        for item in existing + results:
            key = item.get("processed_image", item.get("image"))
            by_key[key] = item
        merged = list(by_key.values())
    else:
        merged = results

    order = {item["image"]: idx for idx, item in enumerate(data)}
    merged.sort(key=lambda item: order.get(item.get("processed_image", item.get("image")), 10**9))
    return merged


def monitor_process_progress(processes, progress_queue, total, desc):
    completed = 0
    progress = tqdm(total=total, desc=desc, unit="image")
    try:
        while completed < total:
            try:
                increment = progress_queue.get(timeout=0.5)
                completed += increment
                progress.update(increment)
                continue
            except queue_module.Empty:
                pass

            if all(not process.is_alive() for process in processes):
                while True:
                    try:
                        increment = progress_queue.get_nowait()
                    except queue_module.Empty:
                        break
                    completed += increment
                    progress.update(increment)
                break
    finally:
        progress.close()

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)
    return failed


def run_mode(args, mode):
    data_root = resolve_path(args.data_root)
    single_data_path = data_root / "data.json"
    mode_data_path = data_root / mode / "data.json"
    if args.input_layout == "single" or (args.input_layout == "auto" and single_data_path.exists()):
        data_path = single_data_path
        image_dir = data_root / "images"
        layout_name = "single"
    else:
        mode_dir = data_root / mode
        data_path = mode_data_path
        image_dir = mode_dir / "images"
        layout_name = "deepseek-modes"
    output_prefix = args.output_prefix or f"{args.model_label}_from_text"
    output_path = resolve_path(args.output_dir) / f"{output_prefix}_{mode}.json"

    if not data_path.exists():
        message = f"[deepseek_ocr][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]
    data = [dict(item, _input_index=index) for index, item in enumerate(data)]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[deepseek_ocr][{mode}] model: {args.model_path}")
    print(f"[deepseek_ocr][{mode}] input layout: {layout_name}")
    print(f"[deepseek_ocr][{mode}] data: {data_path}")
    print(f"[deepseek_ocr][{mode}] images: {image_dir}")
    print(f"[deepseek_ocr][{mode}] output: {output_path}")
    print(f"[deepseek_ocr][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[deepseek_ocr][{mode}] nothing to do.")
        return

    start = time.time()
    cfg = mode_config(mode)
    model_path = resolve_model_path_or_id(args.model_path)
    raw_output_dir = resolve_path(args.raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = raw_output_dir / "shards" / output_path.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = visible_gpu_ids_from_args(args)
    physical_gpu_ids = resolve_physical_gpu_ids(gpu_ids, args.cuda_visible_devices)
    shard_files = []

    if len(gpu_ids) == 1:
        shard_file = shard_dir / f"shard_00_gpu{physical_gpu_ids[0]}.json"
        run_shard({
            "data": todo,
            "gpu_id": gpu_ids[0],
            "physical_gpu_id": physical_gpu_ids[0],
            "shard_index": 0,
            "model_path": model_path,
            "image_dir": str(image_dir),
            "output_path": str(raw_output_dir),
            "shard_file": str(shard_file),
            "cfg": cfg,
            "prompt": args.prompt,
            "crop_mode": args.crop_mode,
            "keep_processed_image_name": args.keep_processed_image_name,
            "show_progress": True,
        })
        shard_files = [shard_file]
    else:
        import multiprocessing as mp

        shards = split_round_robin(todo, len(gpu_ids))
        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()
        processes = []
        print(
            f"[deepseek_ocr][{mode}] running on {len(gpu_ids)} GPUs: "
            f"visible ids {gpu_ids}, physical ids {physical_gpu_ids}"
        )
        for shard_index, (gpu_id, physical_gpu_id, shard_data) in enumerate(
            zip(gpu_ids, physical_gpu_ids, shards)
        ):
            shard_file = shard_dir / f"shard_{shard_index:02d}_gpu{physical_gpu_id}.json"
            shard_files.append(shard_file)
            worker_args = {
                "data": shard_data,
                "gpu_id": gpu_id,
                "physical_gpu_id": physical_gpu_id,
                "shard_index": shard_index,
                "model_path": model_path,
                "image_dir": str(image_dir),
                "output_path": str(raw_output_dir),
                "shard_file": str(shard_file),
                "cfg": cfg,
                "prompt": args.prompt,
                "crop_mode": args.crop_mode,
                "keep_processed_image_name": args.keep_processed_image_name,
                "progress_queue": progress_queue,
                "show_progress": False,
            }
            process = ctx.Process(target=run_shard, args=(worker_args,))
            process.start()
            processes.append(process)

        failed = monitor_process_progress(
            processes=processes,
            progress_queue=progress_queue,
            total=len(todo),
            desc=f"DeepSeek-OCR {mode}",
        )
        if failed:
            raise RuntimeError(f"{len(failed)} OCR worker process(es) failed: {failed}")

    results = []
    for shard_file in shard_files:
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing OCR shard file: {shard_file}")
        results.extend(load_json(shard_file))

    for item in results:
        item.pop("_input_index", None)
    merged = merge_results(data, existing, results, args.resume)
    for item in merged:
        item.pop("_input_index", None)
    save_json(merged, output_path)
    print(f"[deepseek_ocr][{mode}] saved {len(merged)} predictions to {output_path}")
    print(f"[deepseek_ocr][{mode}] saved temporary shards to {shard_dir}")
    print(f"[deepseek_ocr][{mode}] elapsed: {(time.time() - start) / 60:.1f} min")


def run_dataset(args, dataset):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_root = resolve_path(args.data_base) / dataset
    dataset_args.data_root = str(dataset_root)
    effective_single_layout = (
        args.input_layout == "single"
        or (args.input_layout == "auto" and (dataset_root / "data.json").exists())
    )
    if effective_single_layout:
        output_dir_name = f"{args.model_label}_{dataset}"
    else:
        output_dir_name = f"{args.model_label}_{dataset}_deepseek_modes"
    dataset_args.output_dir = str(resolve_path(args.results_base) / output_dir_name)
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[deepseek_ocr][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode)
    print(f"[deepseek_ocr][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run local DeepSeek-OCR on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    parser.add_argument(
        "--input-layout",
        choices=["auto", "deepseek-modes", "single"],
        default="auto",
        help=(
            "Input data layout. deepseek-modes expects {data_root}/{mode}/data.json; "
            "single expects {data_root}/data.json. auto uses single when data.json exists."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--results-base", default=str(DEFAULT_RESULTS_BASE))
    parser.add_argument("--raw-output-dir", default=str(REPO_ROOT / "output" / "deepseek_ocr_deepseek_modes"))
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--dataset-preset", default="single", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directory names under --data-base, e.g. from_text distort mask_clean noise_100.",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large", "raw"])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Use selected visible GPU ids in parallel, e.g. --gpu-ids 0 1 2 3. "
            "If omitted and --cuda-visible-devices lists multiple GPUs, all visible GPUs are used."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--crop-mode", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples per mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have non-empty ocr_text.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing dataset/mode data.json files.")
    parser.add_argument(
        "--keep-processed-image-name",
        action="store_true",
        help="Keep image=en_1_tiny.png in output. By default image is restored to source_image for eval/eval.py.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    prepare_runtime_env(args.cuda_visible_devices)
    check_runtime_dependencies()
    datasets = selected_datasets(args)
    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset)
    else:
        for mode in args.modes:
            run_mode(args, mode)


if __name__ == "__main__":
    main()
