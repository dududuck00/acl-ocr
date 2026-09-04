import argparse
import json
import random
import re
import string
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "fox_data" / "low_prior_stress"
DEFAULT_PROSE_FILE = REPO_ROOT / "fox_data" / "qa" / "qa_checked.json"
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DEFAULT_TYPES = ["prose", "ids", "names", "code", "tables"]


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
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to render low-prior stress images."
        ) from exc
    return Image, ImageDraw, ImageFont


def token_count(text):
    return len(re.findall(r"\S+", text))


def random_code(rng, length):
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def random_hex(rng, length):
    return "".join(rng.choice("0123456789ABCDEF") for _ in range(length))


def make_identifier_lines(rng, page_index, target_lines):
    labels = [
        "Customer ID",
        "Invoice No.",
        "Device SN",
        "Access Key",
        "Batch Code",
        "Txn Ref",
        "Auth Token",
        "Case ID",
        "Asset Tag",
        "Checksum",
    ]
    lines = [f"LOW-PRIOR IDENTIFIER RECORD {page_index:03d}", ""]
    for idx in range(target_lines):
        label = labels[idx % len(labels)]
        style = idx % 5
        if style == 0:
            value = f"{random_code(rng, 4)}-{random_code(rng, 4)}-{random_code(rng, 4)}"
        elif style == 1:
            value = f"{random_code(rng, 3)}-{rng.randrange(1000, 9999)}-{random_code(rng, 5)}"
        elif style == 2:
            value = f"{random_hex(rng, 8)}-{random_hex(rng, 4)}-{random_hex(rng, 4)}"
        elif style == 3:
            value = f"{rng.randrange(10,99)}.{rng.randrange(100,999)}.{rng.randrange(1000,9999)}-{random_code(rng, 2)}"
        else:
            value = f"{random_code(rng, 6)}_{random_code(rng, 6)}_{rng.randrange(100,999)}"
        lines.append(f"{label:<14}: {value}")
    return "\n".join(lines)


def make_name_lines(rng, page_index, target_lines):
    first_names = [
        "Amina", "Oksana", "Mateusz", "Xiangrui", "Adebayo", "Soren",
        "Priyanka", "Levent", "Marisol", "Noor", "Anika", "Emeka",
        "Yara", "Tomasz", "Farida", "Kenji", "Mirela", "Zahra",
    ]
    middle = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    surnames = [
        "Mykhailenko", "Kristensen", "Feldmann", "Okonkwo", "Novakova",
        "Ben-Ari", "Qureshi", "Haddad", "Kowalczyk", "Sato-Wilkes",
        "Adeyemi", "Petrescu", "Nakamura", "Ibrahimovic", "Takahashi",
    ]
    streets = [
        "Kestrel Way", "Juniper Ridge", "North Solano Ave", "Marble Quay",
        "East Larch Street", "Orchid Hollow", "Mica Terrace", "Cobalt Lane",
    ]
    cities = ["Dayton", "Reno", "Tucson", "Madison", "Plano", "Albany", "Tempe"]
    states = ["OH", "NV", "AZ", "WI", "TX", "NY", "CA", "OR"]

    lines = [f"SYNTHETIC NAME AND ADDRESS LEDGER {page_index:03d}", ""]
    for idx in range(target_lines):
        first = rng.choice(first_names)
        last = rng.choice(surnames)
        name = f"{first} {rng.choice(middle)}. {last}"
        address = (
            f"{rng.randrange(100, 9999)} {rng.choice(streets)}, "
            f"Apt {rng.randrange(1, 48)}{rng.choice('ABCDE')}, "
            f"{rng.choice(cities)}, {rng.choice(states)} {rng.randrange(10000, 99999)}"
        )
        if idx % 2 == 0:
            lines.append(f"Name: {name}")
            lines.append(f"Address: {address}")
        else:
            lines.append(f"{idx:03d} | {last}, {first} | {address}")
    return "\n".join(lines)


def make_code_block(rng, page_index, target_blocks):
    variable_roots = ["user", "order", "item", "token", "row", "config", "payload", "sku"]
    lines = [
        f"# Synthetic code snippet {page_index:03d}",
        "import re",
        "from decimal import Decimal",
        "",
    ]
    for block in range(target_blocks):
        root = rng.choice(variable_roots)
        suffix = random_code(rng, 3).lower()
        lines.extend([
            f"def normalize_{root}_{suffix}(value, strict=True):",
            f"    pattern = r\"^[A-Z0-9]{{4}}-[A-Z0-9]{{4}}-{rng.randrange(10,99)}$\"",
            "    if value is None:",
            "        return None",
            "    text = str(value).strip().upper()",
            "    if strict and not re.match(pattern, text):",
            f"        raise ValueError(\"invalid {root} code: %s\" % text)",
            f"    meta = {{\"field\": \"{root}\", \"rank\": {rng.randrange(1, 9)}, \"ok\": True}}",
            "    return text, meta",
            "",
            f"record_{block} = normalize_{root}_{suffix}(\"{random_code(rng, 4)}-{random_code(rng, 4)}-{rng.randrange(10,99)}\", strict=False)",
            f"price_{block} = Decimal(\"{rng.randrange(1, 500)}.{rng.randrange(0, 99):02d}\")",
            "",
        ])
    return "\n".join(lines)


def make_table(rng, page_index, target_rows):
    statuses = ["PENDING", "PAID", "VOID", "HOLD", "RETRY", "SHIP"]
    lines = [
        f"STRUCTURED INVENTORY TABLE {page_index:03d}",
        "",
        "| SKU        | Qty | Price  | Date       | Status  | Ref        |",
        "|------------|----:|-------:|------------|---------|------------|",
    ]
    for _ in range(target_rows):
        sku = f"{random_code(rng, 3)}-{rng.randrange(100, 999)}-{random_code(rng, 2)}"
        qty = rng.randrange(1, 250)
        price = f"{rng.randrange(1, 999)}.{rng.randrange(0, 99):02d}"
        date = f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
        status = rng.choice(statuses)
        ref = f"{random_code(rng, 4)}-{random_code(rng, 4)}"
        lines.append(f"| {sku:<10} | {qty:>3} | {price:>6} | {date} | {status:<7} | {ref:<10} |")
    return "\n".join(lines)


def make_prose_text(prose_items, index, target_chars):
    item = prose_items[index % len(prose_items)]
    text = re.sub(r"\s+", " ", item.get("gt_text", "")).strip()
    if len(text) >= target_chars:
        return text[:target_chars].rsplit(" ", 1)[0]

    parts = [text]
    cursor = index + 1
    while sum(len(part) for part in parts) < target_chars:
        next_item = prose_items[cursor % len(prose_items)]
        parts.append(re.sub(r"\s+", " ", next_item.get("gt_text", "")).strip())
        cursor += 1
    return " ".join(parts)[:target_chars].rsplit(" ", 1)[0]


def wrap_text(text, chars_per_line, preserve_lines):
    if preserve_lines:
        lines = []
        for line in text.splitlines():
            if not line:
                lines.append("")
            elif len(line) <= chars_per_line:
                lines.append(line)
            else:
                lines.extend(textwrap.wrap(line, width=chars_per_line, replace_whitespace=False))
        return lines
    return textwrap.wrap(text, width=chars_per_line)


def render_text_image(text, output_path, args, preserve_lines=False):
    Image, ImageDraw, ImageFont = require_pillow()
    font = ImageFont.truetype(str(repo_path(args.font_path)), args.font_size)
    chars_per_line = max(20, (args.width - 2 * args.padding) // max(1, int(args.font_size * 0.62)))
    lines = wrap_text(text, chars_per_line, preserve_lines)

    bbox = font.getbbox("Ag")
    line_height = bbox[3] - bbox[1]
    height = args.padding * 2 + len(lines) * line_height + max(0, len(lines) - 1) * args.line_spacing
    image = Image.new("RGB", (args.width, max(height, 1)), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = args.padding
    for line in lines:
        draw.text((args.padding, y), line, font=font, fill=(0, 0, 0))
        y += line_height + args.line_spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image.size


def build_text(kind, index, rng, args, prose_items):
    if kind == "prose":
        return make_prose_text(prose_items, index, args.prose_chars), False
    if kind == "ids":
        return make_identifier_lines(rng, index + 1, args.id_lines), True
    if kind == "names":
        return make_name_lines(rng, index + 1, args.name_lines), True
    if kind == "code":
        return make_code_block(rng, index + 1, args.code_blocks), True
    if kind == "tables":
        return make_table(rng, index + 1, args.table_rows), True
    raise ValueError(f"Unknown low-prior type: {kind}")


def prepare_dataset(kind, args, prose_items):
    output_root = repo_path(args.output_base) / kind
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rng = random.Random(f"{args.seed}:{kind}")
    for index in tqdm(range(args.samples_per_type), desc=f"Rendering {kind}", unit="sample"):
        text, preserve_lines = build_text(kind, index, rng, args, prose_items)
        item_id = f"{kind}_{index + 1:03d}"
        image_name = f"{item_id}.png"
        image_path = image_dir / image_name
        width, height = render_text_image(text, image_path, args, preserve_lines=preserve_lines)
        records.append({
            "image": image_name,
            "id": item_id,
            "gt_text": text,
            "token_count": token_count(text),
            "len": len(text),
            "low_prior_stress": True,
            "low_prior_type": kind,
            "synthetic": kind != "prose",
            "seed": args.seed,
            "font_size": args.font_size,
            "render_width": args.width,
            "render_height": height,
            "original_width": width,
            "original_height": height,
        })

    save_json(records, output_root / "data.json")
    print(f"Saved {len(records)} records to {output_root / 'data.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a small low-prior OCR stress test for IDs, names, code, and tables."
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--prose-file", default=str(DEFAULT_PROSE_FILE))
    parser.add_argument("--types", nargs="+", default=DEFAULT_TYPES, choices=DEFAULT_TYPES)
    parser.add_argument("--samples-per-type", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--font-path", default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--padding", type=int, default=42)
    parser.add_argument("--line-spacing", type=int, default=7)
    parser.add_argument("--prose-chars", type=int, default=4200)
    parser.add_argument("--id-lines", type=int, default=90)
    parser.add_argument("--name-lines", type=int, default=60)
    parser.add_argument("--code-blocks", type=int, default=9)
    parser.add_argument("--table-rows", type=int, default=70)
    return parser.parse_args()


def main():
    args = parse_args()
    prose_items = load_json(repo_path(args.prose_file))
    for kind in args.types:
        prepare_dataset(kind, args, prose_items)


if __name__ == "__main__":
    main()
