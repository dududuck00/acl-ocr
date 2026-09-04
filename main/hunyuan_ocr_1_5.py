#!/usr/bin/env python3
"""Run HunyuanOCR-1.5 on the paper's native and cross-architecture inputs.

The two protocols deliberately use version-isolated result paths so rerunning
HunyuanOCR-1.5 cannot overwrite the HunyuanOCR-1.0 predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path("/home/liangyunhao/shared/models/tencent/HunyuanOCR")
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "other"
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
MODEL_ID = "tencent/HunyuanOCR"
MODEL_LABEL = "hunyuanocr_1.5"
MODEL_VERSION = "HunyuanOCR-1.5"
INFERENCE_REVISION = "transformers-tail-repeat-v1"
EXPECTED_SAMPLES = 112
DEFAULT_PROMPT = "请提取图片中的文字内容。"
DEFAULT_MODES = ("tiny", "small", "base")

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

# These are the five conditions used by the version-pinned cross-architecture
# comparison. The 10% swap/shuffle inputs can still be selected explicitly.
DEFAULT_CROSS_DATASETS = (
    "from_text",
    "distort",
    "replace_swap_5",
    "replace_shuffle_5",
    "random",
)
ALL_CROSS_DATASETS = (
    *DEFAULT_CROSS_DATASETS,
    "replace_swap_10",
    "replace_shuffle_10",
)


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def prepare_runtime_env(cuda_visible_devices: str | None) -> None:
    tmp_dir = REPO_ROOT / ".codex" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return data


def save_json_atomic(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def record_key(item: dict[str, Any]) -> str | None:
    return item.get("processed_image") or item.get("image")


def has_tail_repetition(text: str, min_repeats: int = 8, max_unit: int = 256) -> bool:
    """Detect greedy-decoding degeneration using HunyuanOCR's official rule."""
    length = len(text)
    if length < min_repeats * 2:
        return False
    upper = min(max_unit, length // min_repeats)
    for unit_length in range(1, upper + 1):
        unit = text[-unit_length:]
        if not unit.strip():
            continue
        if all(
            text[-unit_length * repeat : -unit_length * (repeat - 1)] == unit
            for repeat in range(2, min_repeats + 1)
        ):
            return True
    return False


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim a repeated suffix using HunyuanOCR's official final safety net."""
    length = len(text)
    if length < 2000:
        return text
    for unit_length in range(2, length // min_repeats + 1):
        candidate = text[-unit_length:]
        count = 0
        cursor = length - unit_length
        while cursor >= 0 and text[cursor : cursor + unit_length] == candidate:
            count += 1
            cursor -= unit_length
        if count >= min_repeats:
            return text[: length - unit_length * (count - 1)]
    return text


def make_tail_repetition_stopper(tokenizer, prompt_length: int, args):
    """Adapt the official streaming text check to native Transformers generation."""
    import torch
    from transformers import StoppingCriteria

    class TailRepetitionStoppingCriteria(StoppingCriteria):
        def __init__(self):
            self.next_check_at = args.repeat_check_interval_tokens
            self.triggered = False

        def __call__(self, input_ids, scores, **kwargs):
            stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            generated_tokens = input_ids.shape[1] - prompt_length
            if generated_tokens < self.next_check_at:
                return stop
            self.next_check_at = generated_tokens + args.repeat_check_interval_tokens

            for row_index, token_ids in enumerate(input_ids):
                generated = token_ids[prompt_length:]
                generated = generated[-args.repeat_decode_tokens :]
                tail = tokenizer.decode(generated, skip_special_tokens=True)[-8000:]
                if len(tail) >= args.repeat_check_start_chars and has_tail_repetition(
                    tail,
                    min_repeats=args.repeat_min_repeats,
                    max_unit=args.repeat_max_unit_chars,
                ):
                    stop[row_index] = True
                    self.triggered = True
            return stop

    return TailRepetitionStoppingCriteria()


def validate_model_path(model_path: Path) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"HunyuanOCR-1.5 model directory does not exist: {model_path}")
    if model_path.name == "v1.0":
        raise ValueError(f"The selected path is the archived HunyuanOCR-1.0 directory: {model_path}")

    required_files = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete HunyuanOCR-1.5 download; missing: {', '.join(missing)}")

    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    architectures = config.get("architectures", [])
    if "HunYuanVLForConditionalGeneration" not in architectures:
        raise ValueError(f"Unexpected model architecture in {model_path / 'config.json'}: {architectures}")

    readme = model_path / "README.md"
    if readme.is_file() and "HunyuanOCR-1.5" not in readme.read_text(encoding="utf-8"):
        raise ValueError(f"The model README does not identify the root checkpoint as HunyuanOCR-1.5: {readme}")


def validate_dataset(
    data_file: Path,
    image_dir: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing data file: {data_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    data = load_json(data_file)
    if limit is None and len(data) != EXPECTED_SAMPLES:
        raise ValueError(f"{data_file} contains {len(data)} records; expected {EXPECTED_SAMPLES}")
    if limit is not None:
        data = data[:limit]

    image_names = [item.get("image") for item in data]
    if None in image_names or len(image_names) != len(set(image_names)):
        raise ValueError(f"{data_file} has missing or duplicate image identifiers")
    missing = [name for name in image_names if not (image_dir / str(name)).is_file()]
    if missing:
        raise FileNotFoundError(f"{image_dir} is missing {len(missing)} images; first: {missing[0]}")
    return data


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def validate_runtime() -> None:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "Missing HunyuanOCR-1.5 runtime packages. Install torch and transformers>=5.13.0 "
            "in a dedicated environment."
        ) from error

    try:
        import torchvision
    except Exception as error:
        raise RuntimeError(
            "HunyuanOCR-1.5 image preprocessing requires a torchvision build compatible with torch. "
            "For torch 2.10.0+cu128, install torchvision==0.25.0 from the PyTorch cu128 index."
        ) from error

    from packaging.version import Version

    minimum_version = Version("5.13.0")
    installed_version = Version(transformers.__version__)
    if installed_version < minimum_version:
        raise RuntimeError(
            f"{MODEL_VERSION} requires transformers>=5.13.0; found {transformers.__version__}. "
            "Run this script in the new HunyuanOCR-1.5 environment."
        )

    print(f"[{MODEL_VERSION}] python: {sys.executable}")
    print(f"[{MODEL_VERSION}] transformers: {transformers.__version__}")
    print(f"[{MODEL_VERSION}] torch: {torch.__version__} (CUDA build: {torch.version.cuda})")
    print(f"[{MODEL_VERSION}] torchvision: {torchvision.__version__}")
    print(
        f"[{MODEL_VERSION}] CUDA available: {torch.cuda.is_available()} "
        f"(visible devices: {torch.cuda.device_count()})"
    )
    if not torch.cuda.is_available():
        print(f"WARNING: [{MODEL_VERSION}] CUDA is not visible; full OCR inference will be impractically slow.")


def load_model(args):
    import torch

    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"[{MODEL_VERSION}] loading target weights from {args.model_path}")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        **model_kwargs,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        backend="pil",
    )
    torch.set_grad_enabled(False)
    return model, processor


def infer(model, processor, image_path: Path, args) -> tuple[str, dict[str, Any]]:
    import torch
    from transformers import StoppingCriteriaList

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
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    prompt_length = inputs["input_ids"].shape[1]
    repetition_stopper = make_tail_repetition_stopper(
        processor.tokenizer,
        prompt_length,
        args,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            stopping_criteria=StoppingCriteriaList([repetition_stopper]),
        )
    generated_ids = output_ids[:, prompt_length:]
    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    cleaned_text = clean_repeated_substrings(raw_text).strip()
    generated_token_count = int(generated_ids.shape[1])
    metadata = {
        "inference_revision": INFERENCE_REVISION,
        "generation_max_new_tokens": args.max_new_tokens,
        "generation_tokens": generated_token_count,
        "generation_limit_reached": generated_token_count >= args.max_new_tokens,
        "repetition_early_stopped": repetition_stopper.triggered,
        "repetition_suffix_cleaned": cleaned_text != raw_text,
        "raw_output_chars": len(raw_text),
        "output_chars": len(cleaned_text),
    }
    return cleaned_text, metadata


def output_record(
    item: dict[str, Any],
    ocr_text: str,
    args,
    error: Exception | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(item)
    if item.get("source_image"):
        result["processed_image"] = item["image"]
        result["image"] = item["source_image"]
    result["ocr_text"] = ocr_text
    result["model_name"] = MODEL_ID
    result["model_version"] = MODEL_VERSION
    result["model_path"] = str(args.model_path)
    result["inference_backend"] = "transformers"
    result["inference_revision"] = INFERENCE_REVISION
    result["generation_max_new_tokens"] = args.max_new_tokens
    if metadata:
        result.update(metadata)
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    else:
        result.pop("error", None)
    return result


def process_image(item, image_dir: Path, model, processor, args) -> dict[str, Any]:
    image_path = image_dir / item["image"]
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            text, metadata = infer(model, processor, image_path, args)
            if not text:
                raise ValueError("model returned empty OCR text")
            return output_record(item, text, args, metadata=metadata)
        except Exception as error:
            last_error = error
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    assert last_error is not None
    print(f"[{MODEL_VERSION}] failed {image_path.name}: {last_error}")
    return output_record(item, "", args, error=last_error)


def is_completed_record(item: dict[str, Any], args) -> bool:
    """Only resume results produced with the current safe decoding contract."""
    return bool(item.get("ocr_text")) and not item.get("error") and (
        item.get("inference_revision") == INFERENCE_REVISION
        and item.get("generation_max_new_tokens") == args.max_new_tokens
    )


def merge_results(
    data: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    new_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        record_key(item): item
        for item in existing + new_results
        if record_key(item) is not None
    }
    order = {item["image"]: index for index, item in enumerate(data)}
    return sorted(by_key.values(), key=lambda item: order.get(record_key(item), 10**9))


def run_condition(
    args,
    label: str,
    data_file: Path,
    image_dir: Path,
    output_file: Path,
    model,
    processor,
) -> int:
    data = validate_dataset(data_file, image_dir, args.limit)
    existing = load_json(output_file) if args.resume and output_file.is_file() else []
    completed = {
        record_key(item)
        for item in existing
        if is_completed_record(item, args)
    }
    stale = sum(bool(item.get("ocr_text")) and not is_completed_record(item, args) for item in existing)
    todo = [item for item in data if item["image"] not in completed]

    print(f"[{MODEL_VERSION}][{label}] data: {data_file}")
    print(f"[{MODEL_VERSION}][{label}] images: {image_dir}")
    print(f"[{MODEL_VERSION}][{label}] output: {output_file}")
    print(
        f"[{MODEL_VERSION}][{label}] total={len(data)} completed={len(completed)} "
        f"stale={stale} todo={len(todo)}"
    )
    if args.dry_run or not todo:
        return 0

    started = time.time()
    new_results: list[dict[str, Any]] = []
    try:
        for item in tqdm(todo, desc=f"{MODEL_VERSION} {label}", unit="image"):
            new_results.append(process_image(item, image_dir, model, processor, args))
            if len(new_results) % args.checkpoint_every == 0:
                save_json_atomic(merge_results(data, existing, new_results), output_file)
    except KeyboardInterrupt:
        if new_results:
            save_json_atomic(merge_results(data, existing, new_results), output_file)
            print(f"\n[{MODEL_VERSION}][{label}] interruption checkpoint saved ({len(new_results)} new records)")
        raise

    merged = merge_results(data, existing, new_results)
    save_json_atomic(merged, output_file)
    errors = sum(bool(item.get("error")) or not item.get("ocr_text") for item in merged)
    elapsed = time.time() - started
    print(
        f"[{MODEL_VERSION}][{label}] saved={len(merged)} errors={errors} "
        f"elapsed={elapsed:.1f}s"
    )
    return errors


def run_native(args, model, processor) -> int:
    errors = 0
    for dataset in args.native_datasets:
        config = NATIVE_DATASETS[dataset]
        output_file = args.results_dir / f"{MODEL_LABEL}_{config['output_suffix']}.json"
        errors += run_condition(
            args,
            f"native:{dataset}",
            config["data_file"],
            config["image_dir"],
            output_file,
            model,
            processor,
        )
    return errors


def run_cross_arch(args, model, processor) -> int:
    errors = 0
    for dataset in args.cross_datasets:
        for mode in args.modes:
            mode_dir = args.data_base / dataset / mode
            output_dir = args.results_dir / f"{MODEL_LABEL}_{dataset}_deepseek_modes"
            output_file = output_dir / f"{MODEL_LABEL}_{dataset}_{mode}.json"
            errors += run_condition(
                args,
                f"cross:{dataset}:{mode}",
                mode_dir / "data.json",
                mode_dir / "images",
                output_file,
                model,
                processor,
            )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local HunyuanOCR-1.5 for the paper's native Table 6 and cross-architecture protocols."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--protocol", choices=("native", "cross_arch", "all"), default="native")
    parser.add_argument(
        "--native-datasets",
        nargs="+",
        default=list(NATIVE_DATASETS),
        choices=sorted(NATIVE_DATASETS),
    )
    parser.add_argument(
        "--cross-datasets",
        nargs="+",
        default=list(DEFAULT_CROSS_DATASETS),
        choices=sorted(ALL_CROSS_DATASETS),
    )
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=("tiny", "small", "base", "large"))
    parser.add_argument("--data-base", type=Path, default=DEFAULT_DATA_BASE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="8192 safely covers the dataset maximum of 2862 target tokens while bounding degeneration.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--repeat-min-repeats", type=int, default=8)
    parser.add_argument("--repeat-max-unit-chars", type=int, default=256)
    parser.add_argument("--repeat-check-start-chars", type=int, default=4000)
    parser.add_argument("--repeat-check-interval-tokens", type=int, default=128)
    parser.add_argument("--repeat-decode-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    args.model_path = resolve_path(args.model_path)
    args.data_base = resolve_path(args.data_base)
    args.results_dir = resolve_path(args.results_dir)
    validate_model_path(args.model_path)
    prepare_runtime_env(args.cuda_visible_devices)
    validate_runtime()

    model = processor = None
    if not args.dry_run:
        model, processor = load_model(args)

    errors = 0
    if args.protocol in {"native", "all"}:
        errors += run_native(args, model, processor)
    if args.protocol in {"cross_arch", "all"}:
        errors += run_cross_arch(args, model, processor)

    if args.dry_run:
        print("dry run complete; model was not loaded and no result files were written")
    elif errors:
        message = f"inference finished with {errors} incomplete predictions; rerun with --resume"
        if args.fail_on_error:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


if __name__ == "__main__":
    main()
