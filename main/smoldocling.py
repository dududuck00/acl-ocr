import argparse
import json
import os
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
DEFAULT_MODEL_PATH = "/home/liangyunhao/shared/models/SmolDocling-256M-preview"
DEFAULT_DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
DEFAULT_DATA_ROOT = DEFAULT_DATA_BASE / "from_text"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "other"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "other" / "smoldocling_from_text_deepseek_modes"
DEFAULT_OUTPUT_PREFIX = "smoldocling_from_text"
DEFAULT_PROMPT = "Convert this page to docling."
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
_WORKER_ARGS = None
_WORKER_PROCESSOR = None
_WORKER_MODEL = None


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


def resolve_model_path(model_path):
    model_path = resolve_path(model_path)
    if (model_path / "config.json").exists():
        return model_path

    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if (path / "config.json").exists())
        if snapshots:
            return snapshots[-1]

    raise FileNotFoundError(
        f"Could not find config.json under model path or snapshots: {model_path}"
    )


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


def strip_doctags(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_model(args):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq as AutoModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModel

    processor = AutoProcessor.from_pretrained(args.model_path)
    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation != "auto":
        model_kwargs["_attn_implementation"] = args.attn_implementation

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model = model.to(args.device).eval()
    torch.set_grad_enabled(False)
    return processor, model


def load_image(image_path, max_side):
    from PIL import Image, ImageOps

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if max_side and max(image.size) > max_side:
        width, height = image.size
        scale = max_side / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return image


def doctags_to_text(doctags, image):
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    return doc.export_to_text()


def infer(model, processor, image_path, args):
    import torch

    image = load_image(image_path, args.max_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", do_resize=True)
    inputs = inputs.to(args.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    prompt_length = inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].lstrip()

    if args.return_doctags:
        return doctags, doctags

    try:
        return doctags_to_text(doctags, image), doctags
    except Exception as exc:
        print(f"DocTags parse failed for {image_path.name}: {exc}; falling back to tag stripping.")
        return strip_doctags(doctags), doctags


def output_item_for_eval(item, ocr_text, keep_processed_image_name, doctags=None):
    output_item = dict(item)
    processed_image = item["image"]

    if not keep_processed_image_name and item.get("source_image"):
        output_item["processed_image"] = processed_image
        output_item["image"] = item["source_image"]

    output_item["ocr_text"] = ocr_text
    if doctags is not None:
        output_item["doctags"] = doctags
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
            ocr_text, doctags = infer(model=model, processor=processor, image_path=image_path, args=args)
            return output_item_for_eval(
                item,
                ocr_text,
                keep_processed_image_name=args.keep_processed_image_name,
                doctags=doctags if args.save_doctags else None,
            )
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (attempt + 1))

    print(f"Error processing {image_name}: {last_error}")
    return None


def init_worker(args):
    global _WORKER_ARGS, _WORKER_PROCESSOR, _WORKER_MODEL

    _WORKER_ARGS = argparse.Namespace(**vars(args))
    prepare_runtime_env(_WORKER_ARGS.cuda_visible_devices)
    _WORKER_PROCESSOR, _WORKER_MODEL = load_model(_WORKER_ARGS)


def worker_process_single_image(job):
    item, image_dir = job
    return process_single_image(
        item=item,
        image_dir=Path(image_dir),
        model=_WORKER_MODEL,
        processor=_WORKER_PROCESSOR,
        args=_WORKER_ARGS,
    )


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


def run_mode(args, mode, model=None, processor=None, pool=None):
    mode_dir = resolve_path(args.data_root) / mode
    data_path = mode_dir / "data.json"
    image_dir = mode_dir / "images"
    output_path = resolve_path(args.output_dir) / f"{args.output_prefix}_{mode}.json"

    if not data_path.exists():
        message = f"[smoldocling][{mode}] missing data file: {data_path}"
        if args.skip_missing:
            print(f"{message}; skipped.")
            return
        raise FileNotFoundError(message)

    data = load_json(data_path)
    if args.limit is not None:
        data = data[: args.limit]

    completed, existing = existing_completed_images(output_path) if args.resume else (set(), [])
    todo = [item for item in data if item["image"] not in completed]

    print(f"[smoldocling][{mode}] model: {args.model_path}")
    print(f"[smoldocling][{mode}] data: {data_path}")
    print(f"[smoldocling][{mode}] images: {image_dir}")
    print(f"[smoldocling][{mode}] output: {output_path}")
    print(f"[smoldocling][{mode}] total={len(data)} completed={len(completed)} todo={len(todo)}")

    if not todo:
        print(f"[smoldocling][{mode}] nothing to do.")
        return

    results = []
    if pool is None:
        for item in tqdm(todo, desc=f"SmolDocling {mode}", unit="image"):
            result = process_single_image(item, image_dir, model, processor, args)
            if result is not None:
                results.append(result)
    else:
        jobs = ((item, str(image_dir)) for item in todo)
        iterator = pool.imap_unordered(worker_process_single_image, jobs, chunksize=args.worker_chunksize)
        for result in tqdm(iterator, total=len(todo), desc=f"SmolDocling {mode}", unit="image"):
            if result is not None:
                results.append(result)

    if not results and not existing:
        print(f"[smoldocling][{mode}] no predictions were produced; output was not written.")
        return

    merged = merge_results(data, existing, results, args.resume)
    save_json(merged, output_path)
    print(f"[smoldocling][{mode}] saved {len(merged)} predictions to {output_path}")


def run_dataset(args, dataset, model=None, processor=None, pool=None):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.data_root = str(resolve_path(args.data_base) / dataset)
    dataset_args.output_dir = str(resolve_path(args.results_base) / f"{args.model_label}_{dataset}_deepseek_modes")
    dataset_args.output_prefix = f"{args.model_label}_{dataset}"

    print(f"[smoldocling][{dataset}] start")
    for mode in dataset_args.modes:
        run_mode(dataset_args, mode, model, processor, pool=pool)
    print(f"[smoldocling][{dataset}] done")


def selected_datasets(args):
    if args.datasets:
        return args.datasets
    return DATASET_PRESETS[args.dataset_preset]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SmolDocling on DeepSeek-mode preprocessed rendered text images."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--results-base", default=str(DEFAULT_RESULTS_BASE))
    parser.add_argument("--model-label", default="smoldocling")
    parser.add_argument("--dataset-preset", default="single", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset directory names under --data-base, e.g. from_text distort random.",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["tiny", "small", "base", "large"])
    parser.add_argument("--cuda-visible-devices", default="5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["auto", "sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes. Each worker loads one model copy.")
    parser.add_argument("--worker-chunksize", type=int, default=1, help="Number of images assigned per multiprocessing chunk.")
    parser.add_argument("--mp-start-method", default="spawn", choices=["spawn", "forkserver"])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples per mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have ocr_text in output JSON.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing dataset/mode data.json files.")
    parser.add_argument("--return-doctags", action="store_true", help="Use raw DocTags as ocr_text instead of exporting plain text.")
    parser.add_argument("--save-doctags", action="store_true", help="Also save raw DocTags in the output JSON.")
    parser.add_argument(
        "--keep-processed-image-name",
        action="store_true",
        help="Keep image=en_1_tiny.png in output. By default image is restored to source_image for eval/eval.py.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    prepare_runtime_env(args.cuda_visible_devices)
    args.model_path = str(resolve_model_path(args.model_path))

    datasets = selected_datasets(args)

    if args.num_workers > 1:
        import multiprocessing as mp

        print(f"[smoldocling] starting {args.num_workers} worker processes on CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
        ctx = mp.get_context(args.mp_start_method)
        with ctx.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args,)) as pool:
            if datasets:
                for dataset in datasets:
                    run_dataset(args, dataset, pool=pool)
            else:
                for mode in args.modes:
                    run_mode(args, mode, pool=pool)
    else:
        processor, model = load_model(args)
        if datasets:
            for dataset in datasets:
                run_dataset(args, dataset, model, processor)
        else:
            for mode in args.modes:
                run_mode(args, mode, model, processor)


if __name__ == "__main__":
    main()
