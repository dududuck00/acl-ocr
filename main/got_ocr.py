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
DEFAULT_MODEL_PATH = "/home/liangyunhao/shared/models/stepfun-ai/GOT-OCR-2.0-hf"
DEFAULT_DATA_ROOT = REPO_ROOT / "fox_data" / "deepseek_mode_images" / "from_text"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "other" / "got_ocr_from_text_deepseek_modes"


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


def load_model(model_path, device_map, use_safetensors, torch_dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map=device_map,
        use_safetensors=use_safetensors,
        torch_dtype=resolve_torch_dtype(torch_dtype),
    )
    model = model.eval()
    return processor, model


def get_input_device(model, device_map):
    if device_map in {"cpu", "cuda"} or device_map.startswith("cuda:"):
        return device_map
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def infer(model, processor, image_path, args):
    import torch

    processor_kwargs = {
        "return_tensors": "pt",
        "format": args.ocr_type == "format",
    }
    if args.crop:
        processor_kwargs["crop_to_patches"] = True
        processor_kwargs["max_patches"] = args.max_patches

    inputs = processor(str(image_path), **processor_kwargs)
    input_device = get_input_device(model, args.device_map)
    inputs = inputs.to(input_device)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=args.max_new_tokens,
        )

    input_length = inputs["input_ids"].shape[1]
    if generate_ids.ndim == 1:
        generated = generate_ids[input_length:]
    else:
        generated = generate_ids[0, input_length:]
    return processor.decode(generated, skip_special_tokens=True) or ""


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
    output_path = resolve_path(args.output_dir) / f"got_ocr_from_text_{mode}.json"

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[got_ocr][{mode}] model: {args.model_path}")
    print(f"[got_ocr][{mode}] data: {data_path}")
    print(f"[got_ocr][{mode}] images: {image_dir}")
    print(f"[got_ocr][{mode}] output: {output_path}")
    print(f"[got_ocr][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[got_ocr][{mode}] nothing to do.")
        return

    results = []
    for item in tqdm(todo, desc=f"GOT-OCR {mode}", unit="image"):
        result = process_single_image(item, image_dir, model, processor, args)
        if result is not None:
            results.append(result)

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    print(f"[got_ocr][{mode}] saved {len(merged)} predictions to {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run GOT-OCR on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--no-use-safetensors", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--ocr-type", default="ocr", choices=["ocr", "format"])
    parser.add_argument("--crop", action="store_true", help="Use GOT-OCR HF crop_to_patches inference.")
    parser.add_argument("--max-patches", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=1)
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
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    model_path = resolve_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    args.model_path = str(model_path)

    processor, model = load_model(
        model_path=args.model_path,
        device_map=args.device_map,
        use_safetensors=not args.no_use_safetensors,
        torch_dtype=args.torch_dtype,
    )
    for mode in args.modes:
        run_mode(args, mode, model, processor)


if __name__ == "__main__":
    main()
