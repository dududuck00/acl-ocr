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
DEFAULT_MODEL_PATH = "tencent/HunyuanOCR"
DEFAULT_MODEL_LABEL = "hunyuanocr_1.5"
DEFAULT_MODEL_VERSION = "HunyuanOCR-1.5"
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_BASE / f"{DEFAULT_MODEL_LABEL}_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = f"{DEFAULT_MODEL_LABEL}_from_text"
DEFAULT_PROMPT = "请提取图片中的文字内容。"
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


def resolve_model_path_or_id(model_path):
    """Resolve a local path while leaving Hugging Face/ModelScope repo IDs intact."""
    path = Path(model_path)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        return str(path)

    local_candidate = REPO_ROOT / path
    if local_candidate.exists():
        return str(local_candidate)
    if "/" in model_path:
        return model_path
    raise FileNotFoundError(f"Local model path does not exist: {local_candidate}")


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


def clean_repeated_substrings(text):
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[: n - length * (count - 1)]

    return text


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
    import torch
    import transformers
    from packaging.version import Version

    minimum_transformers = Version("5.13.0")
    if Version(transformers.__version__) < minimum_transformers:
        raise RuntimeError(
            f"{DEFAULT_MODEL_VERSION} requires transformers>=5.13.0; "
            f"found {transformers.__version__}. Upgrade the Hunyuan environment before inference."
        )

    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    dtype = resolve_torch_dtype(args.torch_dtype)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def load_image(image_path):
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def infer(model, processor, image_path, args):
    import torch

    image = load_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" not in inputs:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
        )

    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        input_ids = inputs.inputs
    generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_repeated_substrings(output_text or "")


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
            output_item = output_item_for_eval(item, ocr_text, args.keep_processed_image_name)
            output_item["model_name"] = args.model_path
            output_item["model_version"] = DEFAULT_MODEL_VERSION
            return output_item
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
        merged = existing + results
        seen = set()
        deduped = []
        for item in merged:
            key = item.get("processed_image", item.get("image"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        merged = deduped
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
        message = f"[hunyuan_ocr][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[hunyuan_ocr][{mode}] model: {args.model_path}")
    print(f"[hunyuan_ocr][{mode}] data: {data_path}")
    print(f"[hunyuan_ocr][{mode}] images: {image_dir}")
    print(f"[hunyuan_ocr][{mode}] output: {output_path}")
    print(f"[hunyuan_ocr][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[hunyuan_ocr][{mode}] nothing to do.")
        return

    started = time.time()
    results = []
    for item in tqdm(todo, desc=f"HunyuanOCR {mode}", unit="image"):
        result = process_single_image(item, image_dir, model, processor, args)
        if result is not None:
            results.append(result)

    if not results and not existing:
        print(f"[hunyuan_ocr][{mode}] no predictions were produced; output was not written.")
        return

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    elapsed = time.time() - started
    speed = len(results) / elapsed if elapsed > 0 else 0.0
    print(f"[hunyuan_ocr][{mode}] saved {len(merged)} predictions to {output_path}")
    print(f"[hunyuan_ocr][{mode}] processed {len(results)} new images in {elapsed:.1f}s ({speed:.3f} image/s)")


def run_dataset(args, dataset, model, processor):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.output_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[hunyuan_ocr][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, model, processor)
    print(f"[hunyuan_ocr][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run HunyuanOCR-1.5 local HF/ModelScope inference on DeepSeek-mode images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--results-base", default=str(DEFAULT_RESULTS_BASE))
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--dataset-preset", default="single", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directory names under --data-base, e.g. from_text distort random.",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
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
    args.model_path = resolve_model_path_or_id(args.model_path)

    model, processor = load_model(args)
    datasets = selected_datasets(args)
    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset, model, processor)
    else:
        for mode in args.modes:
            run_mode(args, mode, model, processor)


if __name__ == "__main__":
    main()
