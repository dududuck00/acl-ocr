import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MODE_SIZES = {
    "tiny": 512,
    "small": 640,
    "base": 1024,
    "large": 1280,
}
PAD_COLOR = (127, 127, 127)


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
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to prepare images. Use the DeepSeek-OCR environment "
            "or install pillow in your current Python environment."
        ) from exc
    return Image, ImageOps


def load_image(path):
    Image, ImageOps = require_pillow()
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def official_native_preprocess(image, mode):
    """Mirror DeepSeek-OCR native-resolution preprocessing for crop_mode=False.

    In the released code, image_size <= 640 is directly resized to a square.
    Larger native modes use ImageOps.pad to preserve aspect ratio and fill
    remaining area with the transform mean color.
    """
    target_size = MODE_SIZES[mode]
    if target_size <= 640:
        return image.resize((target_size, target_size))
    _, ImageOps = require_pillow()
    return ImageOps.pad(image, (target_size, target_size), color=PAD_COLOR)


def direct_resize_preprocess(image, mode):
    target_size = MODE_SIZES[mode]
    return image.resize((target_size, target_size))


def pad_preprocess(image, mode):
    target_size = MODE_SIZES[mode]
    _, ImageOps = require_pillow()
    return ImageOps.pad(image, (target_size, target_size), color=PAD_COLOR)


def preprocess_image(image, mode, strategy):
    if strategy == "official-native":
        return official_native_preprocess(image, mode)
    if strategy == "direct-resize":
        return direct_resize_preprocess(image, mode)
    if strategy == "pad":
        return pad_preprocess(image, mode)
    raise ValueError(f"Unknown strategy: {strategy}")


def image_items_from_dir(input_dir):
    paths = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [{"image": path.name} for path in paths]


def build_output_name(image_name, mode, keep_names):
    if keep_names:
        return image_name
    path = Path(image_name)
    return f"{path.stem}_{mode}{path.suffix.lower() or '.png'}"


def process_mode(items, input_dir, output_dir, mode, strategy, keep_names, overwrite):
    mode_dir = output_dir / mode
    image_dir = mode_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    missing = []
    for item in tqdm(items, desc=f"Preparing {mode}", unit="image"):
        image_name = item["image"]
        src = input_dir / image_name
        if not src.exists():
            missing.append(image_name)
            continue

        out_name = build_output_name(image_name, mode, keep_names)
        dst = image_dir / out_name

        image = load_image(src)
        original_width, original_height = image.size
        if overwrite or not dst.exists():
            processed_image = preprocess_image(image, mode, strategy)
            processed_image.save(dst)
            output_width, output_height = processed_image.size
        else:
            Image, _ = require_pillow()
            with Image.open(dst) as existing:
                output_width, output_height = existing.size

        new_item = dict(item)
        new_item["source_image"] = image_name
        new_item["image"] = out_name
        new_item["deepseek_mode"] = mode
        new_item["deepseek_target_size"] = MODE_SIZES[mode]
        new_item["preprocess_strategy"] = strategy
        new_item["original_width"] = original_width
        new_item["original_height"] = original_height
        new_item["processed_width"] = output_width
        new_item["processed_height"] = output_height
        processed.append(new_item)

    metadata_path = mode_dir / "data.json"
    save_json(processed, metadata_path)
    print(f"Saved {len(processed)} {mode} images to {image_dir}")
    print(f"Saved {mode} metadata to {metadata_path}")
    if missing:
        print(f"Warning: {len(missing)} source images were missing for {mode}.")


def load_items_for_config(cfg):
    input_dir = repo_path(cfg["input_dir"])
    data_path = cfg.get("data")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if data_path:
        items = load_json(repo_path(data_path))
    else:
        items = image_items_from_dir(input_dir)

    if not items:
        raise ValueError(f"No input images found for dataset: {cfg['name']}")

    return input_dir, items


def process_dataset_config(cfg, output_root, args):
    name = cfg["name"]
    input_dir, items = load_items_for_config(cfg)

    if args.limit is not None:
        items = items[:args.limit]

    dataset_output_dir = output_root / name
    print(f"\n{'='*60}")
    print(f"Processing dataset: {name}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {dataset_output_dir}")
    print(f"Found {len(items)} images")
    print(f"{'='*60}")

    for mode in args.modes:
        process_mode(
            items=items,
            input_dir=input_dir,
            output_dir=dataset_output_dir,
            mode=mode,
            strategy=args.strategy,
            keep_names=args.keep_names,
            overwrite=args.overwrite,
        )


def main(args):
    if args.preset:
        prepare_preset(args)
        return

    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)

    if args.data:
        items = load_json(repo_path(args.data))
    else:
        items = image_items_from_dir(input_dir)

    if args.limit is not None:
        items = items[:args.limit]

    if not items:
        raise ValueError("No input images found.")

    for mode in args.modes:
        process_mode(
            items=items,
            input_dir=input_dir,
            output_dir=output_dir,
            mode=mode,
            strategy=args.strategy,
            keep_names=args.keep_names,
            overwrite=args.overwrite,
        )


# 预定义的数据集配置
# 格式: {"name": 数据集名, "input_dir": 相对路径, "data": JSON文件路径或None}
DATASET_CONFIGS = [
    # 基础图片目录（从目录读取图片列表）
    {"name": "compress", "input_dir": "fox_data/compress"},
    {"name": "distort", "input_dir": "fox_data/distort"},
    {"name": "en_png", "input_dir": "fox_data/en_png"},
    # {"name": "from_text", "input_dir": "fox_data/from_text"},  # 与 en_png 内容相同，跳过
    {"name": "random", "input_dir": "fox_data/random"},
    {"name": "test", "input_dir": "fox_data/test"},
    {"name": "replace_shuffle_5", "input_dir": "fox_data/replace_shuffle_5"},
    {"name": "replace_shuffle_10", "input_dir": "fox_data/replace_shuffle_10"},
    {"name": "replace_swap_5", "input_dir": "fox_data/replace_swap_5"},
    {"name": "replace_swap_10", "input_dir": "fox_data/replace_swap_10"},
    # 字体密度相关
    {"name": "story_font_density_sweep", "input_dir": "fox_data/story_font_density_sweep/images"},
    # 多页文档
    {"name": "story_multipage", "input_dir": "fox_data/story_multipage/page_1000"},
    # 有JSON数据文件的数据集
    {"name": "compress_from_json", "input_dir": "fox_data", "data": "fox_data/compress.json"},
    {"name": "random_from_json", "input_dir": "fox_data", "data": "fox_data/random.json"},
    {"name": "replace_from_json", "input_dir": "fox_data", "data": "fox_data/replace.json"},
    {"name": "replace_shuffle_5_from_json", "input_dir": "fox_data", "data": "fox_data/replace_shuffle_5.json"},
    {"name": "replace_shuffle_10_from_json", "input_dir": "fox_data", "data": "fox_data/replace_shuffle_10.json"},
    {"name": "replace_swap_5_from_json", "input_dir": "fox_data", "data": "fox_data/replace_swap_5.json"},
    {"name": "replace_swap_10_from_json", "input_dir": "fox_data", "data": "fox_data/replace_swap_10.json"},
]


PAPER_EXPERIMENT_CONFIGS = [
    {
        "name": "distort",
        "input_dir": "fox_data/distort",
        "data": "fox_data/data.json",
        "description": "semantic perturbation",
    },
    {
        "name": "replace_swap_5",
        "input_dir": "fox_data/replace_swap_5",
        "data": "fox_data/replace_swap_5.json",
        "description": "swap letters inside words, 5 percent",
    },
    {
        "name": "replace_swap_10",
        "input_dir": "fox_data/replace_swap_10",
        "data": "fox_data/replace_swap_10.json",
        "description": "swap letters inside words, 10 percent",
    },
    {
        "name": "replace_shuffle_5",
        "input_dir": "fox_data/replace_shuffle_5",
        "data": "fox_data/replace_shuffle_5.json",
        "description": "shuffle words, 5 percent",
    },
    {
        "name": "replace_shuffle_10",
        "input_dir": "fox_data/replace_shuffle_10",
        "data": "fox_data/replace_shuffle_10.json",
        "description": "shuffle words, 10 percent",
    },
    {
        "name": "random",
        "input_dir": "fox_data/random",
        "data": "fox_data/random.json",
        "description": "fully random text",
    },
]


PRESET_CONFIGS = {
    "paper-experiments": PAPER_EXPERIMENT_CONFIGS,
    "all": DATASET_CONFIGS,
}


def prepare_all_datasets(args):
    """批量处理所有预定义的数据集"""
    output_root = repo_path(args.output_dir)
    failed = []

    for cfg in DATASET_CONFIGS:
        try:
            process_dataset_config(cfg, output_root, args)
        except Exception as e:
            print(f"Error processing {cfg['name']}: {e}")
            failed.append(cfg["name"])

    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"Failed datasets: {failed if failed else 'None'}")
    print(f"Output root: {output_root}")


def prepare_preset(args):
    output_root = repo_path(args.output_dir)
    configs = PRESET_CONFIGS[args.preset]
    failed = []

    for cfg in configs:
        try:
            process_dataset_config(cfg, output_root, args)
        except Exception as e:
            print(f"Error processing {cfg['name']}: {e}")
            failed.append(cfg["name"])

    print(f"\n{'='*60}")
    print(f"Preset processing complete: {args.preset}")
    print(f"Failed datasets: {failed if failed else 'None'}")
    print(f"Output root: {output_root}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare images resized/padded to DeepSeek-OCR native mode sizes for fair cross-model comparisons."
    )
    parser.add_argument("--input-dir", default=None, help="Directory containing source images.")
    parser.add_argument(
        "--output-dir",
        default="fox_data/deepseek_mode_images",
        help="Output root. Single datasets write to output-dir/<mode>; presets write to output-dir/<dataset>/<mode>.",
    )
    parser.add_argument("--data", default=None, help="Optional JSON metadata with an image field.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        default=None,
        help="Batch prepare a predefined dataset group. paper-experiments covers the paper comparison datasets.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_SIZES),
        default=["tiny", "small", "base"],
        help="DeepSeek-OCR native modes to prepare.",
    )
    parser.add_argument(
        "--strategy",
        choices=["official-native", "direct-resize", "pad"],
        default="official-native",
        help="official-native matches DeepSeek-OCR crop_mode=False preprocessing.",
    )
    parser.add_argument("--keep-names", action="store_true", help="Keep original image filenames in each mode folder.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if outputs already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items for debugging.")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.list_presets:
        for preset_name, configs in PRESET_CONFIGS.items():
            print(preset_name)
            for cfg in configs:
                description = cfg.get("description", "")
                suffix = f" - {description}" if description else ""
                print(f"  {cfg['name']}{suffix}")
    else:
        if not parsed_args.preset and not parsed_args.input_dir:
            raise SystemExit("--input-dir is required unless --preset is set.")
        main(parsed_args)
