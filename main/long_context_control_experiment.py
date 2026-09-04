import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path


DEFAULT_FONT = "fonts/NotoSans-Regular.ttf"
DEFAULT_DEEPSEEK_OCR_MODEL = "/home/liangyunhao/shared/models/deepseek-ai/DeepSeek-OCR"
REPO_ROOT = Path(__file__).resolve().parents[1]


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def wrap_text_by_pixel_width(text, font, max_width):
    """Same renderer used in the original notebook."""
    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_width = font.getbbox(word + " ")[2]
        if current_width + word_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width
        else:
            current_line.append(word)
            current_width += word_width

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def render_text_fixed_width(text, font_path, font_size=16, width=900, padding=20, line_spacing=4):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, font_size)
    text_width = width - 2 * padding
    lines = wrap_text_by_pixel_width(text, font, text_width)

    total_height = padding
    for line in lines:
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        total_height += line_height + line_spacing
    total_height = max(total_height - line_spacing + padding, 1)

    img = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y_offset = padding
    for line in lines:
        draw.text((padding, y_offset), line, font=font, fill=(0, 0, 0))
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y_offset += line_height + line_spacing

    return img


def get_image_size(image_path):
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def split_text_into_pages(text, page_count):
    words = text.split()
    if page_count <= 1 or not words:
        return [text.strip()]

    page_size = math.ceil(len(words) / page_count)
    pages = []
    for start in range(0, len(words), page_size):
        pages.append(" ".join(words[start:start + page_size]).strip())
    return pages


def story_sort_key(path):
    match = re.search(r"story_(\d+)", Path(path).stem)
    if match:
        return int(match.group(1))
    return Path(path).stem


def load_story_items(story_data_dir, stories=None):
    story_data_dir = Path(story_data_dir)
    paths = sorted(story_data_dir.glob("story_*_data.json"), key=story_sort_key)
    if stories:
        wanted = {f"story_{s}" if str(s).isdigit() else str(s).replace(".json", "") for s in stories}
        paths = [p for p in paths if p.stem.replace("_data", "") in wanted]
    if not paths:
        raise FileNotFoundError(f"No story_*_data.json files found in {story_data_dir}")

    for path in paths:
        story_name = path.stem.replace("_data", "")
        for item in load_json(path):
            yield story_name, item


def make_multipage(args):
    output_dir = repo_path(args.output_dir)
    font_path = repo_path(args.font_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    page_records = []
    doc_records = []
    serial = 1

    story_items = list(load_story_items(repo_path(args.story_data_dir), args.stories))
    story_items = [
        (story_name, item)
        for story_name, item in story_items
        if args.min_tokens <= int(item["token_count"]) <= args.max_tokens
        and (int(item["token_count"]) - args.min_tokens) % args.token_step == 0
    ]

    for story_name, item in tqdm(story_items, desc="Rendering fixed-density docs", unit="doc"):
        token_count = int(item["token_count"])

        doc_id = f"{story_name}_tokens_{token_count}"
        gt_text = item["gt_text"].strip()
        page_count = max(1, math.ceil(token_count / args.page_tokens))
        page_texts = split_text_into_pages(gt_text, page_count)
        page_images = []

        for page_index, page_text in enumerate(
            tqdm(
                page_texts,
                desc=f"{doc_id} pages",
                unit="page",
                leave=False,
            ),
            start=1,
        ):
            image_name = (
                f"{story_name}_t{token_count:05d}_p{page_index:04d}_"
                f"{serial:06d}.png"
            )
            image_path = images_dir / image_name
            if args.overwrite_images or not image_path.exists():
                img = render_text_fixed_width(
                    text=page_text,
                    font_path=font_path,
                    font_size=args.font_size,
                    width=args.width,
                    padding=args.padding,
                    line_spacing=args.line_spacing,
                )
                img.save(image_path)
            image_width, image_height = get_image_size(image_path)

            page_records.append({
                "image": image_name,
                "doc_id": doc_id,
                "story": story_name,
                "token_count": token_count,
                "page_index": page_index,
                "page_count": len(page_texts),
                "target_page_tokens": args.page_tokens,
                "font_size": args.font_size,
                "width": args.width,
                "image_width": image_width,
                "image_height": image_height,
                "padding": args.padding,
                "line_spacing": args.line_spacing,
                "gt_text": page_text,
                "doc_gt_text": gt_text,
            })
            page_images.append(image_name)
            serial += 1

        doc_records.append({
            "doc_id": doc_id,
            "story": story_name,
            "image": item["image"],
            "token_count": token_count,
            "page_count": len(page_texts),
            "target_page_tokens": args.page_tokens,
            "font_size": args.font_size,
            "width": args.width,
            "padding": args.padding,
            "line_spacing": args.line_spacing,
            "pages": page_images,
            "gt_text": gt_text,
        })
        save_json(page_records, output_dir / "pages.json")
        save_json(doc_records, output_dir / "docs.json")

    save_json(page_records, output_dir / "pages.json")
    save_json(doc_records, output_dir / "docs.json")
    print(f"Saved {len(page_records)} page images to {images_dir}")
    print(f"Saved page metadata to {output_dir / 'pages.json'}")
    print(f"Saved doc metadata to {output_dir / 'docs.json'}")


def make_font_sweep(args):
    output_dir = repo_path(args.output_dir)
    font_path = repo_path(args.font_path)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    token_set = set(args.tokens)
    records = []
    serial = 1

    story_items = [
        (story_name, item)
        for story_name, item in load_story_items(repo_path(args.story_data_dir), args.stories)
        if int(item["token_count"]) in token_set
    ]

    for story_name, item in tqdm(story_items, desc="Rendering font sweep docs", unit="doc"):
        token_count = int(item["token_count"])
        gt_text = item["gt_text"].strip()

        settings = [(font_size, width) for font_size in args.font_sizes for width in args.widths]
        for font_size, width in tqdm(
            settings,
            desc=f"{story_name} {token_count} settings",
            unit="setting",
            leave=False,
        ):
            image_name = (
                f"{story_name}_t{token_count:05d}_f{font_size:02d}_"
                f"w{width:04d}_{serial:06d}.png"
            )
            image_path = images_dir / image_name
            if args.overwrite_images or not image_path.exists():
                img = render_text_fixed_width(
                    text=gt_text,
                    font_path=font_path,
                    font_size=font_size,
                    width=width,
                    padding=args.padding,
                    line_spacing=args.line_spacing,
                )
                img.save(image_path)
            image_width, image_height = get_image_size(image_path)
            records.append({
                "image": image_name,
                "story": story_name,
                "token_count": token_count,
                "font_size": font_size,
                "width": width,
                "image_width": image_width,
                "image_height": image_height,
                "padding": args.padding,
                "line_spacing": args.line_spacing,
                "gt_text": gt_text,
            })
            serial += 1
        save_json(records, output_dir / "font_sweep.json")

    save_json(records, output_dir / "font_sweep.json")
    print(f"Saved {len(records)} font-sweep images to {images_dir}")
    print(f"Saved metadata to {output_dir / 'font_sweep.json'}")


def contain_chinese_string(text):
    return bool(re.search(r"[\u4e00-\u9fa5]", text or ""))


def tokenize_for_eval(text):
    text = text or ""
    if contain_chinese_string(text):
        try:
            import jieba
            return jieba.lcut(text)
        except ImportError:
            return list(text)
    return text.split()


def calc_set_metrics(pred, gt):
    reference = set(tokenize_for_eval(gt))
    hypothesis = set(tokenize_for_eval(pred))

    precision = len(reference & hypothesis) / len(hypothesis) if hypothesis else 0.0
    recall = len(reference & hypothesis) / len(reference) if reference else 0.0
    f_measure = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
    }


def aggregate_pages(args):
    page_data = load_json(repo_path(args.page_data))
    page_predictions = load_json(repo_path(args.page_predictions))
    pred_by_image = {item["image"]: item for item in page_predictions if "image" in item}

    grouped = defaultdict(list)
    missing = []
    for page in tqdm(page_data, desc="Indexing page predictions", unit="page"):
        pred = pred_by_image.get(page["image"])
        if pred is None or "ocr_text" not in pred:
            missing.append(page["image"])
            continue
        merged_page = dict(page)
        merged_page["ocr_text"] = pred.get("ocr_text", "")
        grouped[page["doc_id"]].append(merged_page)

    doc_results = []
    for doc_id, pages in tqdm(grouped.items(), desc="Aggregating documents", unit="doc"):
        pages = sorted(pages, key=lambda x: x["page_index"])
        first = pages[0]
        ocr_text = "\n".join(page.get("ocr_text", "") for page in pages).strip()
        gt_text = first["doc_gt_text"]
        metrics = calc_set_metrics(ocr_text, gt_text)
        doc_results.append({
            "doc_id": doc_id,
            "image": first.get("image"),
            "story": first["story"],
            "token_count": first["token_count"],
            "page_count": first["page_count"],
            "target_page_tokens": first["target_page_tokens"],
            "gt_text": gt_text,
            "ocr_text": ocr_text,
            **metrics,
        })

    doc_results = sorted(doc_results, key=lambda x: (x["story"], x["token_count"]))
    overall = average_metrics(doc_results)
    output = doc_results + [{"overall_metrics": overall, "missing_pages": missing}]

    output_path = repo_path(args.output)
    save_json(output, output_path)
    print(f"Saved {len(doc_results)} aggregated document results to {output_path}")
    if missing:
        print(f"Warning: {len(missing)} pages were missing OCR predictions.")


def evaluate_flat_predictions(args):
    data = load_json(repo_path(args.input))
    evaluated = []
    missing_gt = []
    missing_pred = []

    for item in tqdm(data, desc="Evaluating OCR predictions", unit="sample"):
        if "overall_metrics" in item:
            continue
        image = item.get("image", "<unknown>")
        if args.gt_field not in item:
            missing_gt.append(image)
            continue
        if args.pred_field not in item:
            missing_pred.append(image)
            continue

        merged = dict(item)
        metrics = calc_set_metrics(
            pred=merged.get(args.pred_field, ""),
            gt=merged.get(args.gt_field, ""),
        )
        merged.update(metrics)
        evaluated.append(merged)

    overall = average_metrics(evaluated)
    output = evaluated + [{
        "overall_metrics": overall,
        "missing_gt": missing_gt,
        "missing_predictions": missing_pred,
    }]
    output_path = repo_path(args.output)
    save_json(output, output_path)
    print(f"Saved {len(evaluated)} evaluated predictions to {output_path}")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} samples were missing {args.gt_field}.")
    if missing_pred:
        print(f"Warning: {len(missing_pred)} samples were missing {args.pred_field}.")


def average_metrics(items):
    metric_keys = ["precision", "recall", "f_measure"]
    if not items:
        return {"eval question num": 0, **{key: 0.0 for key in metric_keys}}

    result = {"eval question num": len(items)}
    for key in metric_keys:
        result[key] = sum(float(item.get(key, 0.0)) for item in items) / len(items)
    return result


def mean_numeric(items, key):
    values = [float(item[key]) for item in items if key in item and item[key] is not None]
    if not values:
        return ""
    return sum(values) / len(values)


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
            return path
        raise FileNotFoundError(f"Local model path does not exist: {path}")

    repo_relative_path = REPO_ROOT / path
    if repo_relative_path.exists():
        return repo_relative_path

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
            "Pass the actual local checkpoint directory with --model-path, "
            f"for example: --model-path {DEFAULT_DEEPSEEK_OCR_MODEL}"
        )

    return model_path


def run_deepseek_ocr_shard(worker_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_args["physical_gpu_id"])

    import torch
    from transformers import AutoModel, AutoTokenizer

    data = worker_args["data"]
    gpu_id = worker_args["gpu_id"]
    physical_gpu_id = worker_args["physical_gpu_id"]
    shard_index = worker_args["shard_index"]
    model_path = worker_args["model_path"]
    image_dir = worker_args["image_dir"]
    output_path = worker_args["output_path"]
    shard_file = worker_args["shard_file"]
    cfg = worker_args["cfg"]

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
    ):
        image_name = item["image"]
        image_file = str(image_dir / image_name)
        try:
            raw_text = model.infer(
                tokenizer=tokenizer,
                prompt=worker_args["prompt"],
                image_file=image_file,
                output_path=str(output_path),
                base_size=cfg["base_size"],
                image_size=cfg["image_size"],
                crop_mode=worker_args["crop_mode"],
                save_results=False,
                eval_mode=True,
                test_compress=cfg["test_compress"],
            )
            ocr_text = clean_ocr_output(raw_text)
        except Exception as exc:
            print(f"GPU {gpu_id}({physical_gpu_id}) error processing {image_name}: {exc}")
            ocr_text = ""

        merged = dict(item)
        merged["ocr_text"] = ocr_text
        results.append(merged)

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


def run_deepseek_ocr(args):
    data = load_json(repo_path(args.data))
    if args.limit is not None:
        data = data[:args.limit]
    data = [dict(item, _input_index=index) for index, item in enumerate(data)]

    cfg = mode_config(args.mode)
    model_path = resolve_model_path_or_id(args.model_path)
    image_dir = repo_path(args.image_dir)
    output_file = repo_path(args.output)
    output_path = repo_path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards" / output_file.stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    gpu_ids = args.gpu_ids if args.gpu_ids else [args.gpu_id]
    physical_gpu_ids = resolve_physical_gpu_ids(gpu_ids, args.cuda_visible_devices)
    if len(gpu_ids) == 1:
        shard_file = shard_dir / f"shard_00_gpu{physical_gpu_ids[0]}.json"
        run_deepseek_ocr_shard({
            "data": data,
            "gpu_id": gpu_ids[0],
            "physical_gpu_id": physical_gpu_ids[0],
            "shard_index": 0,
            "model_path": model_path,
            "image_dir": image_dir,
            "output_path": output_path,
            "shard_file": shard_file,
            "cfg": cfg,
            "prompt": args.prompt,
            "crop_mode": args.crop_mode,
        })
        shard_files = [shard_file]
    else:
        import multiprocessing as mp

        shards = split_round_robin(data, len(gpu_ids))
        ctx = mp.get_context("spawn")
        processes = []
        shard_files = []

        print(
            f"Running DeepSeek-OCR on {len(gpu_ids)} GPUs: "
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
                "image_dir": image_dir,
                "output_path": output_path,
                "shard_file": shard_file,
                "cfg": cfg,
                "prompt": args.prompt,
                "crop_mode": args.crop_mode,
            }
            process = ctx.Process(target=run_deepseek_ocr_shard, args=(worker_args,))
            process.start()
            processes.append(process)

        failed = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed.append(process.exitcode)
        if failed:
            raise RuntimeError(f"{len(failed)} OCR worker process(es) failed: {failed}")

    results = []
    for shard_file in shard_files:
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing OCR shard file: {shard_file}")
        results.extend(load_json(shard_file))

    results = sorted(results, key=lambda item: item.get("_input_index", 0))
    for item in results:
        item.pop("_input_index", None)
    save_json(results, output_file)
    elapsed = time.time() - start
    print(f"Saved OCR results to {output_file}")
    print(f"Saved temporary OCR shards to {shard_dir}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


def completed_images_from_output(output_file):
    if not output_file.exists():
        return set(), []

    existing = load_json(output_file)
    completed = {
        item.get("image")
        for item in existing
        if isinstance(item, dict) and item.get("image") and item.get("ocr_text")
    }
    return completed, existing


def batch_output_file(output_prefix, mode):
    output_prefix = repo_path(output_prefix)
    return output_prefix.with_name(f"{output_prefix.name}_{mode}.json")


def prepare_batch_jobs(args):
    jobs = []
    gpu_ids = args.gpu_ids if args.gpu_ids else [args.gpu_id]

    for job_index, (data_path, image_dir, output_prefix) in enumerate(args.job):
        data_path = repo_path(data_path)
        image_dir = repo_path(image_dir)
        data = load_json(data_path)
        if args.limit is not None:
            data = data[:args.limit]
        data = [dict(item, _input_index=index) for index, item in enumerate(data)]

        for mode in args.modes:
            output_file = batch_output_file(output_prefix, mode)
            completed, existing = completed_images_from_output(output_file) if args.resume else (set(), [])
            todo = [item for item in data if item.get("image") not in completed]
            shards = split_round_robin(todo, len(gpu_ids))
            shard_dir = repo_path(args.output_path) / "shards" / output_file.stem
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_files = [
                shard_dir / f"shard_{shard_index:02d}_gpu{{physical_gpu_id}}.json"
                for shard_index in range(len(gpu_ids))
            ]

            print(
                f"[batch][{job_index}][{mode}] data={data_path} "
                f"output={output_file} total={len(data)} completed={len(completed)} todo={len(todo)}"
            )

            jobs.append({
                "job_index": job_index,
                "mode": mode,
                "data": data,
                "image_dir": image_dir,
                "output_file": output_file,
                "existing": existing,
                "shards": shards,
                "shard_files": shard_files,
            })

    return jobs


def run_deepseek_ocr_batch_worker(worker_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_args["physical_gpu_id"])

    import torch
    from transformers import AutoModel, AutoTokenizer

    gpu_id = worker_args["gpu_id"]
    physical_gpu_id = worker_args["physical_gpu_id"]
    model_path = worker_args["model_path"]

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

    for task in worker_args["tasks"]:
        mode = task["mode"]
        data = task["data"]
        image_dir = Path(task["image_dir"])
        shard_file = Path(task["shard_file"])
        cfg = mode_config(mode)

        if not data:
            save_json([], shard_file)
            continue

        results = []
        for item in tqdm(
            data,
            desc=f"GPU {gpu_id}({physical_gpu_id}) {Path(task['output_file']).stem}",
            unit="image",
            position=worker_args["worker_index"],
            leave=True,
        ):
            image_name = item["image"]
            image_file = str(image_dir / image_name)
            try:
                raw_text = model.infer(
                    tokenizer=tokenizer,
                    prompt=worker_args["prompt"],
                    image_file=image_file,
                    output_path=str(worker_args["output_path"]),
                    base_size=cfg["base_size"],
                    image_size=cfg["image_size"],
                    crop_mode=worker_args["crop_mode"],
                    save_results=False,
                    eval_mode=True,
                    test_compress=cfg["test_compress"],
                )
                ocr_text = clean_ocr_output(raw_text)
            except Exception as exc:
                print(f"GPU {gpu_id}({physical_gpu_id}) error processing {image_name}: {exc}")
                ocr_text = ""

            merged = dict(item)
            merged["ocr_text"] = ocr_text
            results.append(merged)

        save_json(results, shard_file)


def merge_batch_job(job, physical_gpu_ids, resume):
    results = []
    concrete_shard_files = []
    for shard_file_template, physical_gpu_id in zip(job["shard_files"], physical_gpu_ids):
        shard_file = Path(str(shard_file_template).format(physical_gpu_id=physical_gpu_id))
        concrete_shard_files.append(shard_file)
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing OCR shard file: {shard_file}")
        results.extend(load_json(shard_file))

    if resume:
        by_key = {}
        for item in job["existing"] + results:
            if isinstance(item, dict) and item.get("image"):
                by_key[item["image"]] = item
        merged = list(by_key.values())
    else:
        merged = results

    merged = sorted(merged, key=lambda item: item.get("_input_index", 0))
    for item in merged:
        item.pop("_input_index", None)
    save_json(merged, job["output_file"])
    print(f"Saved {len(merged)} OCR results to {job['output_file']}")
    print(f"Saved temporary OCR shards to {concrete_shard_files[0].parent}")


def run_deepseek_ocr_batch(args):
    import multiprocessing as mp

    start = time.time()
    output_path = repo_path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path_or_id(args.model_path)
    gpu_ids = args.gpu_ids if args.gpu_ids else [args.gpu_id]
    physical_gpu_ids = resolve_physical_gpu_ids(gpu_ids, args.cuda_visible_devices)
    jobs = prepare_batch_jobs(args)

    if not jobs:
        print("No batch jobs to run.")
        return

    worker_tasks = [[] for _ in gpu_ids]
    for job in jobs:
        for worker_index, shard_data in enumerate(job["shards"]):
            shard_file = Path(str(job["shard_files"][worker_index]).format(
                physical_gpu_id=physical_gpu_ids[worker_index]
            ))
            worker_tasks[worker_index].append({
                "mode": job["mode"],
                "data": shard_data,
                "image_dir": str(job["image_dir"]),
                "output_file": str(job["output_file"]),
                "shard_file": str(shard_file),
            })

    ctx = mp.get_context("spawn")
    processes = []
    print(
        f"Running batched DeepSeek-OCR on {len(gpu_ids)} GPUs: "
        f"visible ids {gpu_ids}, physical ids {physical_gpu_ids}"
    )
    for worker_index, (gpu_id, physical_gpu_id, tasks) in enumerate(
        zip(gpu_ids, physical_gpu_ids, worker_tasks)
    ):
        worker_args = {
            "worker_index": worker_index,
            "gpu_id": gpu_id,
            "physical_gpu_id": physical_gpu_id,
            "model_path": model_path,
            "output_path": output_path,
            "prompt": args.prompt,
            "crop_mode": args.crop_mode,
            "tasks": tasks,
        }
        process = ctx.Process(target=run_deepseek_ocr_batch_worker, args=(worker_args,))
        process.start()
        processes.append(process)

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)
    if failed:
        raise RuntimeError(f"{len(failed)} OCR worker process(es) failed: {failed}")

    for job in jobs:
        merge_batch_job(job, physical_gpu_ids, args.resume)

    elapsed = time.time() - start
    print(f"Batched OCR complete. Elapsed: {elapsed / 60:.1f} min")


def summarize_by_token(args):
    data = load_json(repo_path(args.input))
    rows = []
    by_token = defaultdict(list)
    for item in data:
        if "overall_metrics" in item:
            continue
        by_token[int(item["token_count"])].append(item)

    for token_count in sorted(by_token):
        items = by_token[token_count]
        avg = average_metrics(items)
        rows.append({
            "token_count": token_count,
            "n": len(items),
            "precision": avg["precision"],
            "recall": avg["recall"],
            "f_measure": avg["f_measure"],
        })

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("token_count,n,precision,recall,f_measure\n")
        for row in rows:
            f.write(
                f"{row['token_count']},{row['n']},"
                f"{row['precision']:.6f},{row['recall']:.6f},{row['f_measure']:.6f}\n"
            )
    print(f"Saved summary CSV to {out}")


def summarize_font_sweep(args):
    data = load_json(repo_path(args.input))
    groups = defaultdict(list)
    for item in data:
        if "overall_metrics" in item:
            continue
        key = (
            int(item["token_count"]),
            int(item["font_size"]),
            int(item["width"]),
        )
        groups[key].append(item)

    rows = []
    for token_count, font_size, width in sorted(groups):
        items = groups[(token_count, font_size, width)]
        avg = average_metrics(items)
        rows.append({
            "token_count": token_count,
            "font_size": font_size,
            "width": width,
            "n": len(items),
            "precision": avg["precision"],
            "recall": avg["recall"],
            "f_measure": avg["f_measure"],
            "mean_image_width": mean_numeric(items, "image_width"),
            "mean_image_height": mean_numeric(items, "image_height"),
        })

    out = repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "token_count,font_size,width,n,precision,recall,f_measure,"
            "mean_image_width,mean_image_height\n"
        )
        for row in rows:
            f.write(
                f"{row['token_count']},{row['font_size']},{row['width']},{row['n']},"
                f"{row['precision']:.6f},{row['recall']:.6f},{row['f_measure']:.6f},"
                f"{format_csv_number(row['mean_image_width'])},"
                f"{format_csv_number(row['mean_image_height'])}\n"
            )
    print(f"Saved font-sweep summary CSV to {out}")


def format_csv_number(value):
    if value == "":
        return ""
    return f"{value:.2f}"


def add_render_args(parser):
    parser.add_argument("--font-path", default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--line-spacing", type=int, default=4)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Controlled supplementary experiments for long rendered context OCR."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_pages = subparsers.add_parser(
        "make-pages",
        help="Create fixed-density multi-page controls from fox_data/story_data.",
    )
    make_pages.add_argument("--story-data-dir", default="fox_data/story_data")
    make_pages.add_argument("--output-dir", default="fox_data/story_multipage/page_1000")
    make_pages.add_argument("--page-tokens", type=int, default=1000)
    make_pages.add_argument("--min-tokens", type=int, default=500)
    make_pages.add_argument("--max-tokens", type=int, default=20000)
    make_pages.add_argument("--token-step", type=int, default=500)
    make_pages.add_argument("--stories", nargs="*", default=None)
    make_pages.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Re-render images even if they already exist.",
    )
    add_render_args(make_pages)
    make_pages.set_defaults(func=make_multipage)

    font_sweep = subparsers.add_parser(
        "make-font-sweep",
        help="Create single-image font-size/width controls.",
    )
    font_sweep.add_argument("--story-data-dir", default="fox_data/story_data")
    font_sweep.add_argument("--output-dir", default="fox_data/story_font_sweep")
    font_sweep.add_argument("--tokens", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10000])
    font_sweep.add_argument("--font-sizes", nargs="+", type=int, default=[12, 16, 20, 24])
    font_sweep.add_argument("--widths", nargs="+", type=int, default=[900])
    font_sweep.add_argument("--stories", nargs="*", default=None)
    font_sweep.add_argument("--font-path", default=DEFAULT_FONT)
    font_sweep.add_argument("--padding", type=int, default=20)
    font_sweep.add_argument("--line-spacing", type=int, default=4)
    font_sweep.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Re-render images even if they already exist.",
    )
    font_sweep.set_defaults(func=make_font_sweep)

    ocr = subparsers.add_parser(
        "run-deepseek-ocr",
        help="Run local DeepSeek-OCR on a generated image folder.",
    )
    ocr.add_argument("--data", required=True)
    ocr.add_argument("--image-dir", required=True)
    ocr.add_argument("--output", required=True)
    ocr.add_argument("--output-path", default="output/long_context_control")
    ocr.add_argument("--model-path", default=DEFAULT_DEEPSEEK_OCR_MODEL)
    ocr.add_argument("--mode", choices=["tiny", "small", "base", "large", "raw"], default="tiny")
    ocr.add_argument("--gpu-id", type=int, default=0)
    ocr.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Use multiple visible GPU ids in parallel, e.g. --gpu-ids 0 1 2 3 4 5 6 7.",
    )
    ocr.add_argument("--cuda-visible-devices", default=None)
    ocr.add_argument("--prompt", default="<image>\nFree OCR. ")
    ocr.add_argument("--crop-mode", action="store_true")
    ocr.add_argument("--limit", type=int, default=None)
    ocr.set_defaults(func=run_deepseek_ocr)

    batch_ocr = subparsers.add_parser(
        "run-deepseek-ocr-batch",
        help=(
            "Run local DeepSeek-OCR over multiple datasets/modes while keeping "
            "one loaded model per GPU worker."
        ),
    )
    batch_ocr.add_argument(
        "--job",
        nargs=3,
        action="append",
        required=True,
        metavar=("DATA", "IMAGE_DIR", "OUTPUT_PREFIX"),
        help=(
            "Add one dataset job. OUTPUT_PREFIX is written as "
            "OUTPUT_PREFIX_<mode>.json, e.g. results/compress/story_font_density_sweep."
        ),
    )
    batch_ocr.add_argument("--output-path", default="output/long_context_control")
    batch_ocr.add_argument("--model-path", default=DEFAULT_DEEPSEEK_OCR_MODEL)
    batch_ocr.add_argument(
        "--modes",
        nargs="+",
        choices=["tiny", "small", "base", "large", "raw"],
        default=["tiny"],
    )
    batch_ocr.add_argument("--gpu-id", type=int, default=0)
    batch_ocr.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Use multiple visible GPU ids in parallel, e.g. --gpu-ids 0 1 2 3.",
    )
    batch_ocr.add_argument("--cuda-visible-devices", default=None)
    batch_ocr.add_argument("--prompt", default="<image>\nFree OCR. ")
    batch_ocr.add_argument("--crop-mode", action="store_true")
    batch_ocr.add_argument("--limit", type=int, default=None)
    batch_ocr.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present with non-empty ocr_text in each final output JSON.",
    )
    batch_ocr.set_defaults(func=run_deepseek_ocr_batch)

    aggregate = subparsers.add_parser(
        "aggregate-pages",
        help="Concatenate page-level OCR predictions and evaluate at document level.",
    )
    aggregate.add_argument("--page-data", required=True)
    aggregate.add_argument("--page-predictions", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(func=aggregate_pages)

    eval_flat = subparsers.add_parser(
        "eval-flat",
        help="Evaluate flat OCR predictions with gt_text and ocr_text fields.",
    )
    eval_flat.add_argument("--input", required=True)
    eval_flat.add_argument("--output", required=True)
    eval_flat.add_argument("--gt-field", default="gt_text")
    eval_flat.add_argument("--pred-field", default="ocr_text")
    eval_flat.set_defaults(func=evaluate_flat_predictions)

    summary = subparsers.add_parser(
        "summarize-by-token",
        help="Write token-count averages from an evaluated JSON file.",
    )
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(func=summarize_by_token)

    font_summary = subparsers.add_parser(
        "summarize-font-sweep",
        help="Write font-size/width/token averages from an evaluated font-sweep JSON file.",
    )
    font_summary.add_argument("--input", required=True)
    font_summary.add_argument("--output", required=True)
    font_summary.set_defaults(func=summarize_font_sweep)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
