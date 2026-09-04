import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_FILE = REPO_ROOT / "fox_data" / "qa" / "qa_checked.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "qa" / "deepseek_ocr_decoder_gt_text_qa.json"
DEFAULT_MODEL_PATHS = [
    Path("/data/shared/models/deepseek-ai/DeepSeek-OCR"),
    Path("/home/liangyunhao/shared/models/deepseek-ai/DeepSeek-OCR"),
]
DEFAULT_SUITE_SPECS = {
    "gt": {
        "source_field": "gt_text",
        "ocr_file": None,
        "output": REPO_ROOT / "results" / "qa" / "deepseek_ocr_decoder_gt_text_qa.json",
    },
    "tiny": {
        "source_field": "ocr_text",
        "ocr_file": REPO_ROOT / "results" / "ocr" / "from_text_tiny_eval.json",
        "output": REPO_ROOT / "results" / "qa" / "deepseek_ocr_decoder_ocr_text_tiny_qa.json",
    },
    "small": {
        "source_field": "ocr_text",
        "ocr_file": REPO_ROOT / "results" / "ocr" / "from_text_small_eval.json",
        "output": REPO_ROOT / "results" / "qa" / "deepseek_ocr_decoder_ocr_text_small_qa.json",
    },
    "base": {
        "source_field": "ocr_text",
        "ocr_file": REPO_ROOT / "results" / "ocr" / "from_text_1024_eval.json",
        "output": REPO_ROOT / "results" / "qa" / "deepseek_ocr_decoder_ocr_text_base_qa.json",
    },
}


def prepare_runtime_env(cuda_visible_devices=None):
    tmp_dir = REPO_ROOT / ".codex" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Model path does not exist: {path}")

    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find DeepSeek-OCR checkpoint. Pass it explicitly with --model-path."
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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
            "Use transformers==4.46.3 for this local DeepSeek-OCR script."
        )


def normalize_image_key(name):
    if not name:
        return None
    return Path(str(name)).name


def merge_ocr_source(qa_data, ocr_file, source_field):
    ocr_data = load_json(ocr_file)
    ocr_by_image = {}
    for item in ocr_data:
        for key in ("image", "source_image", "processed_image"):
            image_key = normalize_image_key(item.get(key))
            if image_key:
                ocr_by_image[image_key] = item

    missing = 0
    merged = []
    for item in qa_data:
        output_item = deepcopy(item)
        image_key = normalize_image_key(item.get("image"))
        ocr_item = ocr_by_image.get(image_key)
        if ocr_item and source_field in ocr_item:
            output_item[source_field] = ocr_item[source_field]
        else:
            missing += 1
        merged.append(output_item)

    if missing:
        print(
            f"Warning: {missing}/{len(qa_data)} QA items did not find '{source_field}' "
            f"from OCR file {ocr_file}."
        )
    return merged


def format_options(options):
    if isinstance(options, dict):
        return "\n".join(f"{key}. {options[key]}" for key in sorted(options.keys()))
    return "\n".join(str(option) for option in options)


def build_prompt(source_text, question, options):
    return (
        "You are a strict multiple-choice QA system. "
        "Answer using only the provided source text. "
        "Output only one uppercase letter: A, B, C, or D.\n\n"
        "Source text:\n"
        f"{source_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Options:\n"
        f"{format_options(options)}\n\n"
        "Answer:"
    )


def parse_answer(text):
    if text is None:
        return ""
    normalized = str(text).strip().upper()
    if not normalized:
        return ""

    match = re.search(r"(?:ANSWER|OPTION)?\s*[:：]?\s*([ABCD])(?:\b|[\.\):：、])", normalized)
    if match:
        return match.group(1)

    for char in normalized:
        if char in "ABCD":
            return char
    return ""


def trim_to_max_tokens(tokenizer, prompt, max_input_tokens):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
        input_ids = [bos_token_id] + input_ids

    if max_input_tokens and len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
        if bos_token_id is not None:
            input_ids[0] = bos_token_id
    return input_ids


def load_existing_results(path):
    path = Path(path)
    if not path.exists():
        return {}

    existing = {}
    for item in load_json(path):
        key = item.get("id") or item.get("image")
        if key:
            existing[key] = item
    return existing


def qa_item_is_complete(item, answer_field):
    return all(answer_field in qa for qa in item.get("qa_pairs", []))


def split_round_robin(items, num_shards):
    return [items[index::num_shards] for index in range(num_shards)]


def visible_gpu_ids_from_args(args):
    if args.gpu_ids:
        return args.gpu_ids
    if args.cuda_visible_devices:
        visible = [part.strip() for part in args.cuda_visible_devices.split(",") if part.strip()]
        return list(range(len(visible))) if visible else [0]
    return [0]


def resolve_physical_gpu_ids(gpu_ids, cuda_visible_devices=None):
    if not cuda_visible_devices:
        return gpu_ids

    visible = [int(part.strip()) for part in cuda_visible_devices.split(",") if part.strip()]
    physical_gpu_ids = []
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= len(visible):
            raise ValueError(
                f"GPU id {gpu_id} is outside CUDA_VISIBLE_DEVICES={cuda_visible_devices}. "
                f"Use visible ids in 0..{len(visible) - 1}."
            )
        physical_gpu_ids.append(visible[gpu_id])
    return physical_gpu_ids


def dtype_from_arg(torch, dtype_arg):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(args):
    check_runtime_dependencies()

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = resolve_model_path(args.model_path)
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

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_arg(torch, args.torch_dtype)
    model = model.eval().to(device).to(dtype)
    return model, tokenizer, device, dtype, torch


def worker_model_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        device="auto",
        torch_dtype=args.torch_dtype,
    )


def generate_decoder_answer(model, tokenizer, torch, device, dtype, prompt, args):
    input_ids = trim_to_max_tokens(tokenizer, prompt, args.max_input_tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # DeepSeek-OCR's forward accesses images[0][1]. A zero image plus an all-False
    # image token mask bypasses the visual branch and leaves only the decoder LM active.
    dummy_crop = torch.zeros((1, 3, 1, 1), dtype=dtype, device=device)
    dummy_original = torch.zeros((1, 3, 1, 1), dtype=dtype, device=device)
    images_seq_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    images_spatial_crop = torch.zeros((1, 2), dtype=torch.long, device=device)

    generation_kwargs = {
        "attention_mask": attention_mask,
        "images": [(dummy_crop, dummy_original)],
        "images_seq_mask": images_seq_mask,
        "images_spatial_crop": images_spatial_crop,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "use_cache": True,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generation_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def process_qa_item(item, source_field, answer_field, model, tokenizer, torch, device, dtype, args):
    output_item = deepcopy(item)
    parsed_field = f"{answer_field}Parsed"
    correct_field = f"{answer_field}Correct"
    source_text = output_item.get(source_field, "")
    if not source_text:
        output_item["error"] = f"Missing source field: {source_field}"
        return output_item

    for qa in output_item.get("qa_pairs", []):
        prompt = build_prompt(source_text, qa.get("question", ""), qa.get("options", {}))
        try:
            answer = generate_decoder_answer(model, tokenizer, torch, device, dtype, prompt, args)
            parsed = parse_answer(answer)
            qa[answer_field] = answer
            qa[parsed_field] = parsed
            qa[correct_field] = parsed == qa.get("correct_answer")
        except Exception as exc:
            qa[answer_field] = None
            qa[parsed_field] = ""
            qa[correct_field] = False
            qa[f"{answer_field}Error"] = str(exc)
    return output_item


def evaluate_results(results, answer_field, parsed_field):
    total = 0
    correct = 0
    missing = 0
    answer_counts = {letter: 0 for letter in "ABCD"}

    for item in results:
        for qa in item.get("qa_pairs", []):
            total += 1
            parsed = qa.get(parsed_field, "")
            if parsed in answer_counts:
                answer_counts[parsed] += 1
            else:
                missing += 1
            if parsed and parsed == qa.get("correct_answer"):
                correct += 1

    accuracy = correct / total if total else 0.0
    return {
        "eval_question_num": total,
        "correct": correct,
        "accuracy": accuracy,
        "missing_or_unparsed": missing,
        "answer_counts": answer_counts,
        "answer_field": answer_field,
        "parsed_field": parsed_field,
    }


def expand_suite_names(suite_names):
    if not suite_names:
        return []

    expanded = []
    for name in suite_names:
        if name == "all":
            for default_name in ("gt", "tiny", "small", "base"):
                if default_name not in expanded:
                    expanded.append(default_name)
        elif name not in expanded:
            expanded.append(name)
    return expanded


def prepare_job(job_name, spec, args):
    qa_file = resolve_path(args.qa_file)
    data = load_json(qa_file)
    if args.limit is not None:
        data = data[:args.limit]

    source_field = spec["source_field"]
    ocr_file = spec.get("ocr_file")
    if ocr_file:
        data = merge_ocr_source(data, resolve_path(ocr_file), source_field)

    output = resolve_path(spec["output"])
    existing = load_existing_results(output) if args.resume else {}
    answer_field = args.answer_field
    results = [None] * len(data)
    todo = []

    for item_index, item in enumerate(data):
        key = item.get("id") or item.get("image")
        if args.resume and key in existing and qa_item_is_complete(existing[key], answer_field):
            results[item_index] = existing[key]
        else:
            todo.append(
                {
                    "job_name": job_name,
                    "item_index": item_index,
                    "item": item,
                    "source_field": source_field,
                    "answer_field": answer_field,
                }
            )

    return {
        "name": job_name,
        "source_field": source_field,
        "ocr_file": str(resolve_path(ocr_file)) if ocr_file else None,
        "output": output,
        "data_len": len(data),
        "results": results,
        "todo": todo,
    }


def prepare_suite_jobs(args):
    suite_names = expand_suite_names(args.suite)
    if not suite_names:
        return [
            prepare_job(
                "single",
                {
                    "source_field": args.source_field,
                    "ocr_file": resolve_path(args.ocr_file) if args.ocr_file else None,
                    "output": resolve_path(args.output),
                },
                args,
            )
        ]

    jobs = []
    for suite_name in suite_names:
        if suite_name not in DEFAULT_SUITE_SPECS:
            raise ValueError(
                f"Unknown suite entry '{suite_name}'. Choose from gt, tiny, small, base, all."
            )
        jobs.append(prepare_job(suite_name, DEFAULT_SUITE_SPECS[suite_name], args))
    return jobs


def run_batch_worker(worker_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_args["physical_gpu_id"])
    prepare_runtime_env(None)

    model_args = worker_model_args(worker_args["args"])
    model, tokenizer, device, dtype, torch = load_model_and_tokenizer(model_args)

    results = []
    tasks = worker_args["tasks"]
    desc = f"GPU {worker_args['gpu_id']}({worker_args['physical_gpu_id']})"
    for task in tqdm(
        tasks,
        desc=desc,
        unit="doc",
        position=worker_args["worker_index"],
        leave=True,
        disable=not worker_args.get("show_progress", True),
    ):
        processed_item = process_qa_item(
            task["item"],
            task["source_field"],
            task["answer_field"],
            model,
            tokenizer,
            torch,
            device,
            dtype,
            worker_args["args"],
        )
        results.append(
            {
                "job_name": task["job_name"],
                "item_index": task["item_index"],
                "item": processed_item,
            }
        )

    save_json(results, worker_args["shard_file"])


def finalize_batch_jobs(jobs, shard_files, answer_field):
    job_by_name = {job["name"]: job for job in jobs}
    for shard_file in shard_files:
        shard_file = Path(shard_file)
        if not shard_file.exists():
            continue
        for record in load_json(shard_file):
            job = job_by_name[record["job_name"]]
            job["results"][record["item_index"]] = record["item"]

    for job in jobs:
        missing = [idx for idx, item in enumerate(job["results"]) if item is None]
        if missing:
            raise RuntimeError(
                f"Job {job['name']} is missing {len(missing)} items after merging shards."
            )

        parsed_field = f"{answer_field}Parsed"
        metrics = evaluate_results(job["results"], answer_field, parsed_field)
        save_json(job["results"], job["output"])
        save_json(metrics, job["output"].with_name(job["output"].stem + "_metrics.json"))
        print(f"\n[{job['name']}] {job['output']}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_batch(args):
    prepare_runtime_env(args.cuda_visible_devices)
    args.model_path = resolve_model_path(args.model_path)

    import multiprocessing as mp

    jobs = prepare_suite_jobs(args)
    tasks = []
    for job in jobs:
        tasks.extend(job["todo"])

    if not tasks:
        finalize_batch_jobs(jobs, [], args.answer_field)
        return

    gpu_ids = visible_gpu_ids_from_args(args)
    physical_gpu_ids = resolve_physical_gpu_ids(gpu_ids, args.cuda_visible_devices)
    shard_dir = resolve_path(args.shard_dir) / time.strftime("%Y%m%d_%H%M%S")
    shard_dir.mkdir(parents=True, exist_ok=True)
    task_shards = split_round_robin(tasks, len(gpu_ids))
    shard_files = [
        shard_dir / f"shard_{worker_index:02d}_gpu{physical_gpu_ids[worker_index]}.json"
        for worker_index in range(len(gpu_ids))
    ]

    print(
        f"Running decoder QA suite on {len(gpu_ids)} GPUs: "
        f"visible ids {gpu_ids}, physical ids {physical_gpu_ids}. "
        f"Jobs: {[job['name'] for job in jobs]}; todo docs: {len(tasks)}"
    )

    if len(gpu_ids) == 1:
        run_batch_worker(
            {
                "args": args,
                "gpu_id": gpu_ids[0],
                "physical_gpu_id": physical_gpu_ids[0],
                "worker_index": 0,
                "tasks": task_shards[0],
                "shard_file": str(shard_files[0]),
                "show_progress": True,
            }
        )
    else:
        ctx = mp.get_context("spawn")
        processes = []
        for worker_index, (gpu_id, physical_gpu_id, task_shard, shard_file) in enumerate(
            zip(gpu_ids, physical_gpu_ids, task_shards, shard_files)
        ):
            worker_args = {
                "args": args,
                "gpu_id": gpu_id,
                "physical_gpu_id": physical_gpu_id,
                "worker_index": worker_index,
                "tasks": task_shard,
                "shard_file": str(shard_file),
                "show_progress": args.show_progress,
            }
            process = ctx.Process(target=run_batch_worker, args=(worker_args,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"Decoder QA worker processes failed: {failed}")

    finalize_batch_jobs(jobs, shard_files, args.answer_field)


def run(args):
    prepare_runtime_env(args.cuda_visible_devices)

    qa_file = resolve_path(args.qa_file)
    output = resolve_path(args.output)
    data = load_json(qa_file)
    if args.limit is not None:
        data = data[:args.limit]

    if args.ocr_file:
        data = merge_ocr_source(data, resolve_path(args.ocr_file), args.source_field)

    existing = load_existing_results(output) if args.resume else {}
    answer_field = args.answer_field
    parsed_field = f"{answer_field}Parsed"

    model, tokenizer, device, dtype, torch = load_model_and_tokenizer(args)

    results = []
    for item in tqdm(data, desc="DeepSeek-OCR decoder QA"):
        key = item.get("id") or item.get("image")
        if args.resume and key in existing and qa_item_is_complete(existing[key], answer_field):
            results.append(existing[key])
            continue

        results.append(
            process_qa_item(
                item,
                args.source_field,
                answer_field,
                model,
                tokenizer,
                torch,
                device,
                dtype,
                args,
            )
        )
        if args.save_every and len(results) % args.save_every == 0:
            save_json(results, output)

    metrics = evaluate_results(results, answer_field, parsed_field)
    save_json(results, output)
    save_json(metrics, output.with_name(output.stem + "_metrics.json"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the DeepSeek-OCR checkpoint's decoder as a text-only QA model."
    )
    parser.add_argument("--qa-file", default=str(DEFAULT_QA_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--source-field",
        default="gt_text",
        help="Text field used as QA evidence. Use gt_text for decoder-only upper bound.",
    )
    parser.add_argument(
        "--ocr-file",
        default=None,
        help="Optional OCR result JSON. If set, --source-field is copied from this file by image name.",
    )
    parser.add_argument("--answer-field", default="DeepSeekDecoderAnswer")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--gpu-ids",
        nargs="*",
        type=int,
        default=None,
        help="Visible GPU ids to use. If omitted with --cuda-visible-devices, use all listed GPUs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-input-tokens", type=int, default=7600)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=["gt", "tiny", "small", "base", "all"],
        default=None,
        help=(
            "Run a reusable-model batch suite. 'all' runs gt plus DeepSeek-OCR "
            "OCR-text tiny/small/base jobs."
        ),
    )
    parser.add_argument(
        "--shard-dir",
        default=str(REPO_ROOT / "results" / "qa" / ".deepseek_decoder_qa_shards"),
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.suite or len(visible_gpu_ids_from_args(parsed_args)) > 1:
        run_batch(parsed_args)
    else:
        run(parsed_args)
