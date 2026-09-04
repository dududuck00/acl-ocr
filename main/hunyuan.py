import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ["tiny", "small", "base"]
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_MODEL_LABEL = "hunyuanocr_1.5"
DEFAULT_MODEL_VERSION = "HunyuanOCR-1.5"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_BASE / f"{DEFAULT_MODEL_LABEL}_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = f"{DEFAULT_MODEL_LABEL}_from_text"
DEFAULT_MODEL_NAME = "tencent/HunyuanOCR"
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def encode_image_data_url(image_path):
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"

    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{image}"


def create_openai_client(api_key, base_url):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for HunyuanOCR vLLM API inference. "
            "Install it in the environment you use to run this script."
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(image_path, client, args):
    image = encode_image_data_url(image_path)
    response = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body={
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    )
    return response.choices[0].message.content or ""


def output_item_for_eval(item, ocr_text, keep_processed_image_name):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    return output_item


def process_single_image(item, image_dir, client, args):
    image_name = item["image"]
    image_path = image_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            ocr_text = chat(image_path, client, args)
            output_item = output_item_for_eval(item, ocr_text, args.keep_processed_image_name)
            output_item["model_name"] = args.model_name
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


def run_mode(args, mode, client):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    image_dir = mode_dir / "images"
    output_path = resolve_path(args.output_dir) / f"{args.output_prefix}_{mode}.json"

    if not data_path.exists():
        message = f"[hunyuan][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[hunyuan][{mode}] model: {args.model_name}")
    print(f"[hunyuan][{mode}] base_url: {args.base_url}")
    print(f"[hunyuan][{mode}] data: {data_path}")
    print(f"[hunyuan][{mode}] images: {image_dir}")
    print(f"[hunyuan][{mode}] output: {output_path}")
    print(f"[hunyuan][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[hunyuan][{mode}] nothing to do.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_single_image, item, image_dir, client, args)
            for item in todo
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"HunyuanOCR {mode}", unit="image"):
            result = future.result()
            if result is not None:
                results.append(result)

    if not results and not existing:
        print(f"[hunyuan][{mode}] no predictions were produced; output was not written.")
        return

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    print(f"[hunyuan][{mode}] saved {len(merged)} predictions to {output_path}")


def run_dataset(args, dataset, client):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.output_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[hunyuan][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, client)
    print(f"[hunyuan][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run HunyuanOCR-1.5 vLLM API on DeepSeek-mode preprocessed rendered text images."
    )
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
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
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
    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)
    datasets = selected_datasets(args)
    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset, client)
    else:
        for mode in args.modes:
            run_mode(args, mode, client)


if __name__ == "__main__":
    main()
