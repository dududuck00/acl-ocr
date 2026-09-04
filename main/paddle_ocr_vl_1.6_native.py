#!/usr/bin/env python3
"""Run PaddleOCR-VL-1.6 on native-resolution Table 6 inputs via the official API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from main.paddle_ocr_vl_api import PaddleOCRAPI, SUPPORTED_MODEL
except ModuleNotFoundError:
    from paddle_ocr_vl_api import PaddleOCRAPI, SUPPORTED_MODEL


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "other"
EXPECTED_SAMPLES = 112
DATASETS = {
    "from_text": {
        "data_file": REPO_ROOT / "fox_data" / "data.json",
        "image_dir": REPO_ROOT / "fox_data" / "from_text",
        "output_file": RESULTS_DIR / "paddleocr_vl_1.6_from_text.json",
    },
    "random": {
        "data_file": REPO_ROOT / "fox_data" / "random.json",
        "image_dir": REPO_ROOT / "fox_data" / "random",
        "output_file": RESULTS_DIR / "paddleocr_vl_1.6_random_ocr.json",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def record_key(item: dict) -> str | None:
    return item.get("processed_image") or item.get("image")


def validate_dataset(dataset: str, limit: int | None = None) -> tuple[list[dict], Path, Path]:
    config = DATASETS[dataset]
    data_file = config["data_file"]
    image_dir = config["image_dir"]
    output_file = config["output_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    data = load_json(data_file)
    if limit is None and len(data) != EXPECTED_SAMPLES:
        raise ValueError(f"{data_file} contains {len(data)} records; expected {EXPECTED_SAMPLES}")
    if limit is not None:
        data = data[:limit]

    missing = [item["image"] for item in data if not (image_dir / item["image"]).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} is missing {len(missing)} images; first: {missing[0]}")
    return data, image_dir, output_file


def load_existing(output_file: Path, resume: bool) -> list[dict]:
    if not resume or not output_file.exists():
        return []
    existing = load_json(output_file)
    return [item for item in existing if isinstance(item, dict)]


def merge_results(data: list[dict], existing: list[dict], new_results: list[dict]) -> list[dict]:
    by_key = {record_key(item): item for item in existing + new_results if record_key(item)}
    order = {item["image"]: index for index, item in enumerate(data)}
    merged = sorted(by_key.values(), key=lambda item: order.get(record_key(item), 10**9))
    return merged


def run_dataset(args, api: PaddleOCRAPI, dataset: str) -> None:
    data, image_dir, output_file = validate_dataset(dataset, args.limit)
    existing = load_existing(output_file, args.resume)
    completed = {
        record_key(item)
        for item in existing
        if item.get("ocr_text") and not item.get("error")
    }
    todo = [item for item in data if item["image"] not in completed]

    print(f"[PaddleOCR-VL-1.6][native:{dataset}] data: {DATASETS[dataset]['data_file']}")
    print(f"[PaddleOCR-VL-1.6][native:{dataset}] images: {image_dir}")
    print(f"[PaddleOCR-VL-1.6][native:{dataset}] output: {output_file}")
    print(
        f"[PaddleOCR-VL-1.6][native:{dataset}] "
        f"total={len(data)} completed={len(completed)} todo={len(todo)}"
    )

    if args.dry_run or not todo:
        return

    items = [(str(image_dir / item["image"]), item) for item in todo]
    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }
    started = time.time()
    results = api.process_batch_optimized(
        items,
        optional_payload=optional_payload,
        max_workers=args.max_workers,
        poll_interval=args.poll_interval,
        submit_interval=args.submit_interval,
    )
    merged = merge_results(data, existing, results)
    save_json(merged, output_file)

    successful = sum(bool(item.get("ocr_text")) and not item.get("error") for item in merged)
    errors = sum(bool(item.get("error")) for item in merged)
    elapsed = time.time() - started
    print(
        f"[PaddleOCR-VL-1.6][native:{dataset}] saved={len(merged)} "
        f"successful={successful} errors={errors} elapsed={elapsed:.1f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL-1.6 native-resolution natural/random OCR for main-paper Table 6."
    )
    parser.add_argument("--token", default=os.environ.get("PADDLEOCR_TOKEN", ""))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=3)
    parser.add_argument("--submit-interval", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and not args.token:
        raise ValueError("Missing PaddleOCR API token. Set PADDLEOCR_TOKEN or pass --token.")

    api = PaddleOCRAPI(
        token=args.token,
        model=SUPPORTED_MODEL,
        poll_interval=args.poll_interval,
        max_concurrent=args.max_workers,
    )
    for dataset in args.datasets:
        run_dataset(args, api, dataset)

    if args.dry_run:
        print("dry run complete; no API requests or result files were written")


if __name__ == "__main__":
    main()
