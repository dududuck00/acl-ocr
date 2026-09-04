import argparse
import json
import os
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ["tiny", "small", "base"]
DEFAULT_MODEL_PATH = "/home/liangyunhao/shared/models/facebook/nougat-base"
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_BASE / "nougat_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = "nougat_from_text"
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


def prepare_runtime_env(cuda_visible_devices):
    tmp_dir = REPO_ROOT / ".codex" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def resolve_torch_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"

    import torch

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def load_model(args):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    model = model.to(args.device).eval()
    return processor, model


def load_image(image_path):
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def infer(model, processor, image_path, args):
    import torch

    image = load_image(image_path)
    model_param = next(model.parameters())
    inputs = processor(images=image, return_tensors="pt").to(model_param.device)
    pixel_values = inputs.pixel_values.to(dtype=model_param.dtype)
    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            early_stopping=True,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0] or ""


def output_item_for_eval(item, ocr_text, keep_processed_image_name):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    return output_item


def process_single_image(item, image_dir, model, processor, args):
    image_name = item["image"]
    image_path = image_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            ocr_text = infer(model=model, processor=processor, image_path=image_path, args=args)
            return output_item_for_eval(item, ocr_text, args.keep_processed_image_name)
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (attempt + 1))

    print(f"Error processing {image_name}: {last_error}")
    return None


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


def run_mode(args, mode, model, processor):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    image_dir = mode_dir / "images"
    output_path = resolve_path(args.output_dir) / f"{args.output_prefix}_{mode}.json"

    if not data_path.exists():
        message = f"[nougat][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[nougat][{mode}] model: {args.model_path}")
    print(f"[nougat][{mode}] data: {data_path}")
    print(f"[nougat][{mode}] images: {image_dir}")
    print(f"[nougat][{mode}] output: {output_path}")
    print(f"[nougat][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[nougat][{mode}] nothing to do.")
        return

    results = []
    for item in tqdm(todo, desc=f"Nougat {mode}", unit="image"):
        result = process_single_image(item, image_dir, model, processor, args)
        if result is not None:
            results.append(result)

    if not results and not existing:
        print(f"[nougat][{mode}] no predictions were produced; output was not written.")
        return

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    print(f"[nougat][{mode}] saved {len(merged)} predictions to {output_path}")


def run_dataset(args, dataset, model, processor):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.output_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[nougat][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, model, processor)
    print(f"[nougat][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Nougat on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--results-base", default=str(DEFAULT_RESULTS_BASE))
    parser.add_argument("--model-label", default="nougat")
    parser.add_argument("--dataset-preset", default="single", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directory names under --data-base, e.g. from_text distort random.",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples per mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have ocr_text in output JSON.")
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

    processor, model = load_model(args)
    datasets = selected_datasets(args)
    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset, model, processor)
    else:
        for mode in args.modes:
            run_mode(args, mode, model, processor)


if __name__ == "__main__":
    main()
