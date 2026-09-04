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
DEFAULT_MODELS = ["8B", "32B"]
DEFAULT_DATA_ROOT = REPO_ROOT / "fox_data" / "deepseek_mode_images" / "from_text"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "other" / "qwen3_vl_from_text_deepseek_modes"
DEFAULT_NATIVE_OUTPUT_DIR = REPO_ROOT / "results" / "other"
DEFAULT_API_KEY = ""
NATIVE_DATASETS = {
    "from_text": {
        "data_file": REPO_ROOT / "fox_data" / "data.json",
        "image_dir": REPO_ROOT / "fox_data" / "from_text",
        "output_suffix": "from_text",
    },
    "random": {
        "data_file": REPO_ROOT / "fox_data" / "random.json",
        "image_dir": REPO_ROOT / "fox_data" / "random",
        "output_suffix": "random_ocr",
    },
}
MODEL_ALIASES = {
    "8B": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "label": "qwen3_vl_8b",
    },
    "32B": {
        "name": "Qwen/Qwen3-VL-32B-Instruct",
        "label": "qwen3_vl_32b",
    },
}

SYSTEM_PROMPT = (
    "You are a professional OCR tool. Your task is to transcribe the text in "
    "the image exactly as it appears. Do not interpret, summarize, or comment "
    "on the text. Even if the text is random, nonsensical, or gibberish, "
    "output it exactly. Do not add any conversational filler."
)

USER_PROMPT = "Transcribe all the text in this image exactly as it is. Output ONLY the transcribed text."


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
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{image}"


def sanitize_model_label(model_name):
    label = model_name.split("/")[-1]
    return (
        label.replace("-Instruct", "")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )


def resolve_model_config(model_spec):
    if model_spec in MODEL_ALIASES:
        return MODEL_ALIASES[model_spec]
    return {"name": model_spec, "label": sanitize_model_label(model_spec)}


def chat(image_path, client, model_name, max_tokens):
    image = encode_image_data_url(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
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


def process_single_image(
    item,
    image_dir,
    client,
    model_name,
    max_tokens,
    retries,
    retry_sleep,
    keep_processed_image_name,
):
    image_name = item["image"]
    image_path = image_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    last_error = None
    for attempt in range(retries + 1):
        try:
            ocr_text = chat(image_path, client, model_name, max_tokens=max_tokens)
            output_item = output_item_for_eval(item, ocr_text, keep_processed_image_name)
            output_item["model_name"] = model_name
            output_item["model_family"] = "Qwen3-VL"
            return output_item
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))

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


def run_mode(args, mode, model_config, client):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    image_dir = mode_dir / "images"
    output_path = resolve_path(args.output_dir) / f"{model_config['label']}_from_text_{mode}.json"

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[{model_config['label']}][{mode}] model: {model_config['name']}")
    print(f"[{model_config['label']}][{mode}] data: {data_path}")
    print(f"[{model_config['label']}][{mode}] images: {image_dir}")
    print(f"[{model_config['label']}][{mode}] output: {output_path}")
    print(f"[{model_config['label']}][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[{model_config['label']}][{mode}] nothing to do.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                item,
                image_dir,
                client,
                model_config["name"],
                args.max_tokens,
                args.retries,
                args.retry_sleep,
                args.keep_processed_image_name,
            )
            for item in todo
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"OCR {mode}", unit="image"):
            result = future.result()
            if result is not None:
                results.append(result)

    if args.resume:
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
    save_json(merged, output_path)
    print(f"[{model_config['label']}][{mode}] saved {len(merged)} predictions to {output_path}")


def run_native_dataset(args, dataset, model_config, client):
    dataset_config = NATIVE_DATASETS[dataset]
    data_path = dataset_config["data_file"]
    image_dir = dataset_config["image_dir"]
    output_path = (
        resolve_path(args.native_output_dir)
        / f"{model_config['label']}_{dataset_config['output_suffix']}.json"
    )

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[{model_config['label']}][native:{dataset}] model: {model_config['name']}")
    print(f"[{model_config['label']}][native:{dataset}] data: {data_path}")
    print(f"[{model_config['label']}][native:{dataset}] images: {image_dir}")
    print(f"[{model_config['label']}][native:{dataset}] output: {output_path}")
    print(
        f"[{model_config['label']}][native:{dataset}] "
        f"total={len(data)} completed={len(completed)} todo={len(todo)}"
    )

    if not todo:
        print(f"[{model_config['label']}][native:{dataset}] nothing to do.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                item,
                image_dir,
                client,
                model_config["name"],
                args.max_tokens,
                args.retries,
                args.retry_sleep,
                True,
            )
            for item in todo
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"OCR native {dataset}",
            unit="image",
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    merged = existing + results if args.resume else results
    by_image = {item["image"]: item for item in merged}
    order = {item["image"]: idx for idx, item in enumerate(data)}
    merged = sorted(by_image.values(), key=lambda item: order.get(item["image"], 10**9))
    save_json(merged, output_path)
    print(f"[{model_config['label']}][native:{dataset}] saved {len(merged)} predictions to {output_path}")


def create_openai_client(api_key, base_url):
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for API inference. "
            "Install it in the environment you use to run this script, e.g. pip install openai."
        ) from exc
    return openai.Client(api_key=api_key, base_url=base_url)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL OCR API on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--protocol",
        default="resized",
        choices=["resized", "native"],
        help="resized runs tiny/small/base rebuttal inputs; native reproduces the main-paper Table 6 protocol.",
    )
    parser.add_argument(
        "--native-datasets",
        nargs="+",
        default=["from_text", "random"],
        choices=sorted(NATIVE_DATASETS),
    )
    parser.add_argument("--native-output-dir", default=str(DEFAULT_NATIVE_OUTPUT_DIR))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to run. Use aliases 8B/32B or full OpenAI-compatible model names.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Run a single full model name. This overrides --models and keeps compatibility with older commands.",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples per mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have ocr_text in output JSON.")
    parser.add_argument(
        "--keep-processed-image-name",
        action="store_true",
        help="Keep image=en_1_tiny.png in output. By default image is restored to source_image for eval/eval.py.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)
    model_specs = [args.model_name] if args.model_name else args.models
    model_configs = [resolve_model_config(model_spec) for model_spec in model_specs]
    for model_config in model_configs:
        if args.protocol == "native":
            for dataset in args.native_datasets:
                run_native_dataset(args, dataset, model_config, client)
        else:
            for mode in args.modes:
                run_mode(args, mode, model_config, client)


if __name__ == "__main__":
    main()
