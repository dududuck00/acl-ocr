import argparse
import json
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODE_SIZES = {
    "tiny": 512,
    "small": 640,
    "base": 1024,
    "large": 1280,
}
PAD_COLOR = (127, 127, 127)
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


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


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to prepare mask ablation images. Use an environment "
            "with pillow installed, for example .dpsk-ocr/bin/python."
        ) from exc
    return Image, ImageDraw, ImageFont, ImageOps


def load_image(path):
    Image, _, _, ImageOps = require_pillow()
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def official_native_preprocess(image, mode):
    target_size = MODE_SIZES[mode]
    if target_size <= 640:
        return image.resize((target_size, target_size))
    _, _, _, ImageOps = require_pillow()
    return ImageOps.pad(image, (target_size, target_size), color=PAD_COLOR)


def build_output_name(image_name, dataset_name, mode=None):
    path = Path(image_name)
    if mode is None:
        return f"{path.stem}_{dataset_name}{path.suffix.lower() or '.png'}"
    return f"{path.stem}_{dataset_name}_{mode}{path.suffix.lower() or '.png'}"


def font_bbox(font, text):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_paired_text_by_pixel_width(source_text, draw_text, font, max_width):
    """Wrap using source text widths while returning the same spans from draw_text."""
    source_paragraphs = source_text.splitlines() or [source_text]
    draw_paragraphs = draw_text.splitlines() or [draw_text]
    lines = []
    for paragraph, draw_paragraph in zip(source_paragraphs, draw_paragraphs):
        word_spans = [match.span() for match in re.finditer(r"\S+", paragraph)]
        if not word_spans:
            lines.append("")
            continue

        current_line_parts = []
        current_width = 0
        for start, end in word_spans:
            source_word = paragraph[start:end]
            draw_word = draw_paragraph[start:end]
            word_width, _ = font_bbox(font, source_word + " ")
            if current_line_parts and current_width + word_width > max_width:
                lines.append(" ".join(current_line_parts))
                current_line_parts = [draw_word]
                current_width = word_width
            else:
                current_line_parts.append(draw_word)
                current_width += word_width
        if current_line_parts:
            lines.append(" ".join(current_line_parts))
    return lines


def render_text_fixed_width(source_text, draw_text, font_path, font_size, width, padding, line_spacing):
    Image, ImageDraw, ImageFont, _ = require_pillow()
    font = ImageFont.truetype(font_path, font_size)
    text_width = width - 2 * padding
    lines = wrap_paired_text_by_pixel_width(source_text, draw_text, font, text_width)

    _, line_height = font_bbox(font, "Ag")
    total_height = padding + len(lines) * line_height + max(len(lines) - 1, 0) * line_spacing + padding
    total_height = max(total_height, 1)

    image = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    y_offset = padding
    for line in lines:
        draw.text((padding, y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += line_height + line_spacing
    return image


def mask_spans(text, spans):
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] not in "\r\n\t ":
                chars[index] = " "
    return "".join(chars)


def word_mask_text(text, ratio, rng):
    spans = [match.span() for match in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)]
    if not spans or ratio <= 0:
        return text, 0, len(spans)
    mask_count = min(len(spans), round(len(spans) * ratio))
    selected = rng.sample(spans, mask_count)
    return mask_spans(text, selected), mask_count, len(spans)


def char_mask_text(text, ratio, rng):
    candidates = [index for index, char in enumerate(text) if char.isalnum()]
    if not candidates or ratio <= 0:
        return text, 0, len(candidates)
    mask_count = min(len(candidates), round(len(candidates) * ratio))
    selected = rng.sample(candidates, mask_count)
    chars = list(text)
    for index in selected:
        chars[index] = " "
    return "".join(chars), mask_count, len(candidates)


def token_count(item):
    return int(item.get("token_count", 0))


def select_items(items, sample_size, min_token_count, target_token_count, selection, seed, fill_below_threshold):
    if selection == "closest-token":
        ranked = sorted(
            items,
            key=lambda item: (abs(token_count(item) - target_token_count), token_count(item), item_sort_key(item)),
        )
        selected = ranked[:sample_size]
        selected.sort(key=item_sort_key)
        return selected, sum(1 for item in items if token_count(item) >= min_token_count)

    eligible = [item for item in items if int(item.get("token_count", 0)) >= min_token_count]
    rng = random.Random(seed)
    if len(eligible) >= sample_size:
        selected = rng.sample(eligible, sample_size)
    elif fill_below_threshold:
        selected = list(eligible)
        selected_ids = {item.get("id", item.get("image")) for item in selected}
        remainder = [
            item for item in items
            if item.get("id", item.get("image")) not in selected_ids
        ]
        remainder.sort(key=lambda item: int(item.get("token_count", 0)), reverse=True)
        selected.extend(remainder[:sample_size - len(selected)])
    else:
        selected = list(eligible)

    selected.sort(key=item_sort_key)
    return selected, len(eligible)


def item_sort_key(item):
    value = str(item.get("id") or item.get("image") or "")
    match = re.search(r"(\d+)", value)
    if match:
        return int(match.group(1)), value
    return 10**9, value


def dataset_name(mask_type, ratio):
    if mask_type == "clean":
        return "mask_clean"
    if mask_type == "noise":
        ratio_int = int(round(ratio * 100))
        return f"noise_{ratio_int}"
    ratio_int = int(round(ratio * 100))
    return f"mask_{mask_type}_{ratio_int}"


def build_masked_text(item, mask_type, ratio, seed):
    text = item["gt_text"]
    if mask_type == "clean":
        return text, 0, 0

    stable_seed = f"{seed}:{item.get('id', item.get('image'))}:{mask_type}:{ratio:.4f}"
    rng = random.Random(stable_seed)
    if mask_type == "word":
        return word_mask_text(text, ratio, rng)
    if mask_type == "char":
        return char_mask_text(text, ratio, rng)
    raise ValueError(f"Unknown mask type: {mask_type}")


def source_canvas_width(item, source_image_dir, default_width):
    image_path = source_image_dir / item["image"]
    if not image_path.exists():
        return default_width
    with load_image(image_path) as image:
        return image.size[0]


def make_image_for_item(item, masked_text, mask_type, source_image_dir, args):
    if mask_type == "clean" and args.clean_from_existing_image:
        image_path = source_image_dir / item["image"]
        if image_path.exists():
            return load_image(image_path)

    width = source_canvas_width(item, source_image_dir, args.width)
    return render_text_fixed_width(
        item["gt_text"],
        masked_text,
        font_path=str(repo_path(args.font_path)),
        font_size=args.font_size,
        width=width,
        padding=args.padding,
        line_spacing=args.line_spacing,
    )


def make_noise_image(image, item, ratio, seed):
    Image, _, _, _ = require_pillow()
    width, height = image.size
    stable_seed = f"{seed}:{item.get('id', item.get('image'))}:noise:{ratio:.4f}"
    rng = random.Random(stable_seed)
    noise_bytes = bytes(rng.randrange(256) for _ in range(width * height * 3))
    noise = Image.frombytes("RGB", (width, height), noise_bytes)
    return Image.blend(image, noise, ratio)


def process_noise_dataset(selected_items, ratio, args):
    name = dataset_name("noise", ratio)
    output_root = repo_path(args.output_base) / name
    source_image_dir = repo_path(args.source_image_dir)

    if args.layout == "single":
        records = []
        image_dir = output_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for item in tqdm(selected_items, desc=f"Rendering {name} ({ratio:.0%})", unit="sample"):
            image_path = source_image_dir / item["image"]
            image = load_image(image_path)
            image = make_noise_image(image, item, ratio, args.seed)
            original_width, original_height = image.size

            out_name = build_output_name(item["image"], name)
            out_path = image_dir / out_name
            if args.overwrite or not out_path.exists():
                image.save(out_path)
                processed_width, processed_height = image.size
            else:
                with load_image(out_path) as existing:
                    processed_width, processed_height = existing.size

            new_item = dict(item)
            new_item["image"] = out_name
            new_item["source_image"] = item["image"]
            new_item["mask_ablation"] = True
            new_item["perturbation_type"] = "noise"
            new_item["noise_ratio"] = ratio
            new_item["mask_type"] = "none"
            new_item["mask_ratio"] = 0.0
            new_item["mask_dataset"] = name
            new_item["masked_text"] = item["gt_text"]
            new_item["masked_count"] = 0
            new_item["mask_candidate_count"] = 0
            new_item["mask_seed"] = args.seed
            new_item["ablation_layout"] = "single"
            new_item["preprocess_strategy"] = "none"
            new_item["original_width"] = original_width
            new_item["original_height"] = original_height
            new_item["processed_width"] = processed_width
            new_item["processed_height"] = processed_height
            records.append(new_item)

        save_json(records, output_root / "data.json")
        print(f"Saved {len(records)} records: {output_root / 'data.json'}")
        return

    processed_by_mode = {mode: [] for mode in args.modes}

    for item in tqdm(selected_items, desc=f"Rendering {name} ({ratio:.0%})", unit="sample"):
        image_path = source_image_dir / item["image"]
        image = load_image(image_path)
        image = make_noise_image(image, item, ratio, args.seed)
        original_width, original_height = image.size

        for mode in args.modes:
            mode_dir = output_root / mode
            image_dir = mode_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            out_name = build_output_name(item["image"], name, mode)
            out_path = image_dir / out_name
            if args.overwrite or not out_path.exists():
                processed_image = official_native_preprocess(image, mode)
                processed_image.save(out_path)
                processed_width, processed_height = processed_image.size
            else:
                with load_image(out_path) as existing:
                    processed_width, processed_height = existing.size

            new_item = dict(item)
            new_item["image"] = out_name
            new_item["source_image"] = item["image"]
            new_item["mask_ablation"] = True
            new_item["perturbation_type"] = "noise"
            new_item["noise_ratio"] = ratio
            new_item["mask_type"] = "none"
            new_item["mask_ratio"] = 0.0
            new_item["mask_dataset"] = name
            new_item["masked_text"] = item["gt_text"]
            new_item["masked_count"] = 0
            new_item["mask_candidate_count"] = 0
            new_item["mask_seed"] = args.seed
            new_item["deepseek_mode"] = mode
            new_item["deepseek_target_size"] = MODE_SIZES[mode]
            new_item["preprocess_strategy"] = "official-native"
            new_item["original_width"] = original_width
            new_item["original_height"] = original_height
            new_item["processed_width"] = processed_width
            new_item["processed_height"] = processed_height
            processed_by_mode[mode].append(new_item)

    for mode, records in processed_by_mode.items():
        save_json(records, output_root / mode / "data.json")
        print(f"Saved {len(records)} records: {output_root / mode / 'data.json'}")


def process_dataset(selected_items, mask_type, ratio, args):
    name = dataset_name(mask_type, ratio)
    output_root = repo_path(args.output_base) / name
    source_image_dir = repo_path(args.source_image_dir)

    if args.layout == "single":
        records = []
        image_dir = output_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        desc = name if mask_type == "clean" else f"{name} ({ratio:.0%})"
        for item in tqdm(selected_items, desc=f"Rendering {desc}", unit="sample"):
            masked_text, masked_count, candidate_count = build_masked_text(
                item=item,
                mask_type=mask_type,
                ratio=ratio,
                seed=args.seed,
            )
            image = make_image_for_item(item, masked_text, mask_type, source_image_dir, args)
            original_width, original_height = image.size

            out_name = build_output_name(item["image"], name)
            out_path = image_dir / out_name
            if args.overwrite or not out_path.exists():
                image.save(out_path)
                processed_width, processed_height = image.size
            else:
                with load_image(out_path) as existing:
                    processed_width, processed_height = existing.size

            new_item = dict(item)
            new_item["image"] = out_name
            new_item["source_image"] = item["image"]
            new_item["mask_ablation"] = True
            new_item["mask_type"] = mask_type
            new_item["mask_ratio"] = ratio
            new_item["mask_dataset"] = name
            new_item["masked_text"] = masked_text
            new_item["masked_count"] = masked_count
            new_item["mask_candidate_count"] = candidate_count
            new_item["mask_seed"] = args.seed
            new_item["ablation_layout"] = "single"
            new_item["preprocess_strategy"] = "none"
            new_item["original_width"] = original_width
            new_item["original_height"] = original_height
            new_item["processed_width"] = processed_width
            new_item["processed_height"] = processed_height
            records.append(new_item)

        save_json(records, output_root / "data.json")
        print(f"Saved {len(records)} records: {output_root / 'data.json'}")
        return

    processed_by_mode = {mode: [] for mode in args.modes}

    desc = name if mask_type == "clean" else f"{name} ({ratio:.0%})"
    for item in tqdm(selected_items, desc=f"Rendering {desc}", unit="sample"):
        masked_text, masked_count, candidate_count = build_masked_text(
            item=item,
            mask_type=mask_type,
            ratio=ratio,
            seed=args.seed,
        )
        image = make_image_for_item(item, masked_text, mask_type, source_image_dir, args)
        original_width, original_height = image.size

        for mode in args.modes:
            mode_dir = output_root / mode
            image_dir = mode_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            out_name = build_output_name(item["image"], name, mode)
            out_path = image_dir / out_name
            if args.overwrite or not out_path.exists():
                processed_image = official_native_preprocess(image, mode)
                processed_image.save(out_path)
                processed_width, processed_height = processed_image.size
            else:
                with load_image(out_path) as existing:
                    processed_width, processed_height = existing.size

            new_item = dict(item)
            new_item["image"] = out_name
            new_item["source_image"] = item["image"]
            new_item["mask_ablation"] = True
            new_item["mask_type"] = mask_type
            new_item["mask_ratio"] = ratio
            new_item["mask_dataset"] = name
            new_item["masked_text"] = masked_text
            new_item["masked_count"] = masked_count
            new_item["mask_candidate_count"] = candidate_count
            new_item["mask_seed"] = args.seed
            new_item["deepseek_mode"] = mode
            new_item["deepseek_target_size"] = MODE_SIZES[mode]
            new_item["preprocess_strategy"] = "official-native"
            new_item["original_width"] = original_width
            new_item["original_height"] = original_height
            new_item["processed_width"] = processed_width
            new_item["processed_height"] = processed_height
            processed_by_mode[mode].append(new_item)

    for mode, records in processed_by_mode.items():
        save_json(records, output_root / mode / "data.json")
        print(f"Saved {len(records)} records: {output_root / mode / 'data.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare visual-evidence mask ablation datasets."
    )
    parser.add_argument("--data", default="fox_data/data.json")
    parser.add_argument("--source-image-dir", default="fox_data/from_text")
    parser.add_argument("--output-base", default="fox_data/mask_ablation_images")
    parser.add_argument(
        "--layout",
        choices=["single", "deepseek-modes"],
        default="single",
        help=(
            "single writes {dataset}/data.json and original-scale images; "
            "deepseek-modes writes {dataset}/{tiny,small,base}/data.json."
        ),
    )
    parser.add_argument("--modes", nargs="+", default=["tiny", "small", "base"], choices=sorted(MODE_SIZES))
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--include-noise", action="store_true", help="Also generate random-noise blended images.")
    parser.add_argument("--only-noise", action="store_true", help="Only generate random-noise blended images.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--min-token-count", type=int, default=1000)
    parser.add_argument("--target-token-count", type=int, default=1000)
    parser.add_argument(
        "--selection",
        choices=["above-threshold", "closest-token"],
        default="above-threshold",
        help="Sample selection rule: strict token threshold or nearest to target-token-count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fill-below-threshold",
        action="store_true",
        help="If fewer than sample-size items meet min-token-count, fill with the highest-token remaining items.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not write the clean subset baseline.")
    parser.add_argument("--font-path", default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--line-spacing", type=int, default=4)
    parser.add_argument(
        "--clean-from-existing-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use original rendered images for mask_clean when available.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_json(repo_path(args.data))
    selected_items, eligible_count = select_items(
        items=items,
        sample_size=args.sample_size,
        min_token_count=args.min_token_count,
        target_token_count=args.target_token_count,
        selection=args.selection,
        seed=args.seed,
        fill_below_threshold=args.fill_below_threshold,
    )
    if not selected_items:
        raise ValueError("No samples selected.")

    if args.selection == "closest-token":
        counts = [token_count(item) for item in selected_items]
        print(
            f"Selected {len(selected_items)} samples closest to token_count={args.target_token_count}. "
            f"Range: {min(counts)}-{max(counts)}, mean: {sum(counts) / len(counts):.1f}. "
            f"{eligible_count} samples meet token_count >= {args.min_token_count}."
        )
    else:
        print(
            f"Selected {len(selected_items)} samples. "
            f"{eligible_count} samples meet token_count >= {args.min_token_count}."
        )
    if args.selection == "above-threshold" and len(selected_items) < args.sample_size:
        print(
            f"Warning: requested {args.sample_size} samples, but only {len(selected_items)} "
            "strictly satisfy the selection rule. Use --fill-below-threshold to fill to sample-size."
        )

    if args.only_noise:
        for ratio in args.ratios:
            process_noise_dataset(selected_items, ratio, args)
        return

    if not args.no_clean:
        process_dataset(selected_items, "clean", 0.0, args)
    for ratio in args.ratios:
        process_dataset(selected_items, "word", ratio, args)
        process_dataset(selected_items, "char", ratio, args)
    if args.include_noise:
        for ratio in args.ratios:
            process_noise_dataset(selected_items, ratio, args)


if __name__ == "__main__":
    main()
