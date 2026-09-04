import argparse
import json
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


def wrap_text_by_pixel_width(text, font, max_width):
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


def make_data(args):
    output_dir = repo_path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    font_path = repo_path(args.font_path)

    token_set = set(args.tokens)
    story_items = [
        (story_name, item)
        for story_name, item in load_story_items(repo_path(args.story_data_dir), args.stories)
        if int(item["token_count"]) in token_set
    ]

    records = []
    serial = 1
    settings = [(font_size, width) for font_size in args.font_sizes for width in args.widths]

    for story_name, item in tqdm(story_items, desc="Rendering font-density docs", unit="doc"):
        token_count = int(item["token_count"])
        gt_text = item["gt_text"].strip()

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
            image_path = image_dir / image_name
            if args.overwrite_images or not image_path.exists():
                image = render_text_fixed_width(
                    text=gt_text,
                    font_path=font_path,
                    font_size=font_size,
                    width=width,
                    padding=args.padding,
                    line_spacing=args.line_spacing,
                )
                image.save(image_path)

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

        save_json(records, output_dir / "font_density_sweep.json")

    save_json(records, output_dir / "font_density_sweep.json")
    print(f"Saved {len(records)} images to {image_dir}")
    print(f"Saved metadata to {output_dir / 'font_density_sweep.json'}")


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
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
    }


def average_metrics(items):
    if not items:
        return {"eval question num": 0, "precision": 0.0, "recall": 0.0, "f_measure": 0.0}
    return {
        "eval question num": len(items),
        "precision": sum(float(item.get("precision", 0.0)) for item in items) / len(items),
        "recall": sum(float(item.get("recall", 0.0)) for item in items) / len(items),
        "f_measure": sum(float(item.get("f_measure", 0.0)) for item in items) / len(items),
    }


def eval_predictions(args):
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
        merged.update(calc_set_metrics(merged.get(args.pred_field, ""), merged.get(args.gt_field, "")))
        evaluated.append(merged)

    output = evaluated + [{
        "overall_metrics": average_metrics(evaluated),
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


def mean_numeric(items, key):
    values = [float(item[key]) for item in items if key in item and item[key] is not None]
    if not values:
        return ""
    return sum(values) / len(values)


def format_csv_number(value):
    if value == "":
        return ""
    return f"{value:.2f}"


def summarize(args):
    data = load_json(repo_path(args.input))
    groups = defaultdict(list)
    for item in data:
        if "overall_metrics" in item:
            continue
        key = (int(item["token_count"]), int(item["font_size"]), int(item["width"]))
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

    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
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
    print(f"Saved summary CSV to {output_path}")


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


def run_ocr(args):
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
    shard_files = []

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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment 2: font-size and image-width density sweep for long rendered OCR."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-data", help="Render font-density sweep images.")
    make.add_argument("--story-data-dir", default="fox_data/story_data")
    make.add_argument("--output-dir", default="fox_data/story_font_density_sweep")
    make.add_argument("--tokens", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10000])
    make.add_argument("--font-sizes", nargs="+", type=int, default=[12, 16, 20, 24])
    make.add_argument("--widths", nargs="+", type=int, default=[900, 1200])
    make.add_argument("--stories", nargs="*", default=None)
    make.add_argument("--font-path", default=DEFAULT_FONT)
    make.add_argument("--padding", type=int, default=20)
    make.add_argument("--line-spacing", type=int, default=4)
    make.add_argument("--overwrite-images", action="store_true")
    make.set_defaults(func=make_data)

    ocr = subparsers.add_parser("run-ocr", help="Run local DeepSeek-OCR on sweep images.")
    ocr.add_argument("--data", required=True)
    ocr.add_argument("--image-dir", required=True)
    ocr.add_argument("--output", required=True)
    ocr.add_argument("--output-path", default="output/font_density_sweep")
    ocr.add_argument("--model-path", default=DEFAULT_DEEPSEEK_OCR_MODEL)
    ocr.add_argument("--mode", choices=["tiny", "small", "base", "large", "raw"], default="tiny")
    ocr.add_argument("--gpu-id", type=int, default=0)
    ocr.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    ocr.add_argument("--cuda-visible-devices", default=None)
    ocr.add_argument("--prompt", default="<image>\nFree OCR. ")
    ocr.add_argument("--crop-mode", action="store_true")
    ocr.add_argument("--limit", type=int, default=None)
    ocr.set_defaults(func=run_ocr)

    eval_cmd = subparsers.add_parser("eval", help="Evaluate OCR predictions.")
    eval_cmd.add_argument("--input", required=True)
    eval_cmd.add_argument("--output", required=True)
    eval_cmd.add_argument("--gt-field", default="gt_text")
    eval_cmd.add_argument("--pred-field", default="ocr_text")
    eval_cmd.set_defaults(func=eval_predictions)

    summary = subparsers.add_parser("summarize", help="Summarize by token/font/width.")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(func=summarize)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
