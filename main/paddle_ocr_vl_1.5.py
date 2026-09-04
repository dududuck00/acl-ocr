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
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_MODEL_LABEL = "paddleocr_vl_1.6"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_BASE / f"{DEFAULT_MODEL_LABEL}_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = f"{DEFAULT_MODEL_LABEL}_from_text"
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
    "markdown",
    "md",
    "text",
    "content",
    "block_content",
    "rec_text",
    "rec_texts",
    "ocr_text",
    "html",
    "latex",
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
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_pipeline(args):
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise ImportError(
            "PaddleOCRVL is required. Activate the paddleocrvl environment "
            "or install paddleocr[doc-parser] before running this script."
        ) from exc

    kwargs = {
        "pipeline_version": args.pipeline_version,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
        "use_layout_detection": args.use_layout_detection,
        "use_chart_recognition": args.use_chart_recognition,
        "use_seal_recognition": args.use_seal_recognition,
        "use_ocr_for_image_block": args.use_ocr_for_image_block,
        "format_block_content": args.format_block_content,
        "merge_layout_blocks": args.merge_layout_blocks,
    }
    if args.device:
        kwargs["device"] = args.device
    if args.vl_rec_model_dir:
        kwargs["vl_rec_model_dir"] = args.vl_rec_model_dir
    if args.layout_detection_model_dir:
        kwargs["layout_detection_model_dir"] = args.layout_detection_model_dir

    return PaddleOCRVL(**kwargs)


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
        if not part or part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return "\n".join(deduped).strip()


def ocr_image(pipeline, image_path):
    output = pipeline.predict(str(image_path))
    return extract_ocr_text(output)


def output_item_for_eval(item, ocr_text, keep_processed_image_name):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    return output_item


def process_single_image(item, image_dir, pipeline, args):
    image_name = item["image"]
    image_path = image_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            ocr_text = ocr_image(pipeline, image_path)
            output_item = output_item_for_eval(item, ocr_text, args.keep_processed_image_name)
            output_item["model_name"] = "PaddleOCR-VL-1.6"
            output_item["pipeline_version"] = args.pipeline_version
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


def run_mode(args, mode, pipeline):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    image_dir = mode_dir / "images"
    output_path = resolve_path(args.output_dir) / f"{args.output_prefix}_{mode}.json"

    if not data_path.exists():
        message = f"[paddleocr-vl][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[paddleocr-vl][{mode}] pipeline_version: {args.pipeline_version}")
    print(f"[paddleocr-vl][{mode}] data: {data_path}")
    print(f"[paddleocr-vl][{mode}] images: {image_dir}")
    print(f"[paddleocr-vl][{mode}] output: {output_path}")
    print(f"[paddleocr-vl][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[paddleocr-vl][{mode}] nothing to do.")
        return

    results = []
    for item in tqdm(todo, desc=f"PaddleOCR-VL {mode}", unit="image"):
        result = process_single_image(item, image_dir, pipeline, args)
        if result is not None:
            results.append(result)

    if not results and not existing:
        print(f"[paddleocr-vl][{mode}] no predictions were produced; output was not written.")
        return

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    print(f"[paddleocr-vl][{mode}] saved {len(merged)} predictions to {output_path}")


def run_dataset(args, dataset, pipeline):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.output_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[paddleocr-vl][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, pipeline)
    print(f"[paddleocr-vl][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


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
        description="Run local PaddleOCR-VL-1.6 on DeepSeek-mode preprocessed rendered text images (legacy filename)."
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
    parser.add_argument("--pipeline-version", default="v1.6", choices=["v1.6"])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--device", default=None, help="Optional Paddle device, e.g. gpu:0 or cpu.")
    parser.add_argument("--vl-rec-model-dir", default=None)
    parser.add_argument("--layout-detection-model-dir", default=None)
    parser.add_argument("--use-doc-orientation-classify", type=str_to_optional_bool, default=False)
    parser.add_argument("--use-doc-unwarping", type=str_to_optional_bool, default=False)
    parser.add_argument("--use-layout-detection", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-chart-recognition", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-seal-recognition", type=str_to_optional_bool, default=None)
    parser.add_argument("--use-ocr-for-image-block", type=str_to_optional_bool, default=None)
    parser.add_argument("--format-block-content", type=str_to_optional_bool, default=None)
    parser.add_argument("--merge-layout-blocks", type=str_to_optional_bool, default=None)
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
    pipeline = load_pipeline(args)
    datasets = selected_datasets(args)

    if datasets:
        for dataset in datasets:
            run_dataset(args, dataset, pipeline)
    else:
        for mode in args.modes:
            run_mode(args, mode, pipeline)


if __name__ == "__main__":
    main()
