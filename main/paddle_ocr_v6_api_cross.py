#!/usr/bin/env python3
"""Run PP-OCRv6 on the rebuttal's matched DeepSeek-mode images via the official API."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from paddle_ocr_v6_api_native import (
    MODEL,
    PaddleOCRV6API,
    load_json,
    process_image,
    record_key,
    save_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = REPO_ROOT / "fox_data" / "deepseek_mode_images"
RESULTS_DIR = REPO_ROOT / "results" / "other"
MODEL_LABEL = "paddle_ocr_v6"
MODEL_VERSION = "PP-OCRv6"
EXPECTED_SAMPLES = 112
DEFAULT_DATASETS = (
    "from_text",
    "distort",
    "replace_swap_5",
    "replace_shuffle_5",
    "random",
)
ALL_DATASETS = (*DEFAULT_DATASETS, "replace_swap_10", "replace_shuffle_10")
DEFAULT_MODES = ("tiny", "small", "base")


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def condition_paths(args, dataset: str, mode: str) -> tuple[Path, Path, Path]:
    condition_dir = args.data_base / dataset / mode
    output_dir = args.results_dir / f"{MODEL_LABEL}_{dataset}_deepseek_modes"
    output_file = output_dir / f"{MODEL_LABEL}_{dataset}_{mode}.json"
    return condition_dir / "data.json", condition_dir / "images", output_file


def validate_condition(args, dataset: str, mode: str) -> tuple[list[dict], Path, Path]:
    data_file, image_dir, output_file = condition_paths(args, dataset, mode)
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing data file: {data_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    data = load_json(data_file)
    if args.limit is None and len(data) != EXPECTED_SAMPLES:
        raise ValueError(f"{data_file} contains {len(data)} records; expected {EXPECTED_SAMPLES}")
    if args.limit is not None:
        data = data[: args.limit]

    images = [item.get("image") for item in data]
    if None in images or len(images) != len(set(images)):
        raise ValueError(f"{data_file} has missing or duplicate image identifiers")
    missing = [name for name in images if not (image_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset}/{mode} is missing {len(missing)} images; first: {missing[0]}")
    return data, image_dir, output_file


def result_is_complete(item: dict) -> bool:
    return (
        item.get("model_version") == MODEL_VERSION
        and not item.get("error")
        and (bool(item.get("ocr_text")) or item.get("recognition_empty") is True)
    )


def process_cross_image(item: dict, image_dir: Path, api: PaddleOCRV6API, args) -> dict:
    output = process_image(item, image_dir, api, args.retries, args.retry_sleep)
    processed_image = output.get("image")
    source_image = output.get("source_image")
    if source_image:
        output["processed_image"] = processed_image
        output["image"] = source_image
    output["protocol"] = "cross_arch"
    return output


def merge_results(data: list[dict], existing: list[dict], new_results: list[dict]) -> list[dict]:
    by_key = {record_key(item): item for item in existing + new_results if record_key(item)}
    order = {item["image"]: index for index, item in enumerate(data)}
    return sorted(by_key.values(), key=lambda item: order.get(record_key(item), 10**9))


def run_condition(args, api: PaddleOCRV6API, dataset: str, mode: str) -> int:
    data, image_dir, output_file = validate_condition(args, dataset, mode)
    existing = load_json(output_file) if args.resume and output_file.is_file() else []
    completed = {record_key(item) for item in existing if result_is_complete(item)}
    stale = sum(record_key(item) not in completed for item in existing)
    todo = [item for item in data if item["image"] not in completed]

    label = f"cross:{dataset}:{mode}"
    print(f"[{MODEL_VERSION}][{label}] images: {image_dir}")
    print(f"[{MODEL_VERSION}][{label}] output: {output_file}")
    print(
        f"[{MODEL_VERSION}][{label}] total={len(data)} "
        f"completed={len(completed)} stale_or_failed={stale} todo={len(todo)}"
    )
    if args.dry_run or not todo:
        return 0

    new_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_cross_image, item, image_dir, api, args): item["image"]
            for item in todo
        }
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{MODEL_VERSION} {dataset}/{mode}",
                unit="image",
            ):
                new_results.append(future.result())
                if len(new_results) % args.checkpoint_every == 0:
                    save_json(merge_results(data, existing, new_results), output_file)
        except KeyboardInterrupt:
            save_json(merge_results(data, existing, new_results), output_file)
            for future in futures:
                future.cancel()
            print(f"\n[{MODEL_VERSION}][{label}] interruption checkpoint saved")
            raise

    merged = merge_results(data, existing, new_results)
    save_json(merged, output_file)
    api_errors = sum(bool(item.get("error")) for item in merged)
    empty = sum(item.get("recognition_empty") is True and not item.get("error") for item in merged)
    print(
        f"[{MODEL_VERSION}][{label}] saved={len(merged)} "
        f"empty_predictions={empty} api_errors={api_errors}"
    )
    return api_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official PP-OCRv6 API on the 5x3 matched cross-architecture rebuttal grid."
    )
    parser.add_argument("--token", default=os.environ.get("PADDLEOCR_TOKEN", ""))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(ALL_DATASETS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=("tiny", "small", "base"))
    parser.add_argument("--data-base", type=resolve_path, default=DATA_BASE)
    parser.add_argument("--results-dir", type=resolve_path, default=RESULTS_DIR)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--poll-timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1 or args.checkpoint_every < 1:
        raise ValueError("--max-workers and --checkpoint-every must be at least 1")
    if not args.dry_run and not args.token:
        raise ValueError("Missing PaddleOCR API token. Set PADDLEOCR_TOKEN or pass --token.")

    api = PaddleOCRV6API(
        token=args.token,
        request_timeout=args.request_timeout,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )
    errors = 0
    for dataset in args.datasets:
        for mode in args.modes:
            errors += run_condition(args, api, dataset, mode)

    if args.dry_run:
        print("dry run complete; no API requests or result files were written")
    elif errors:
        raise SystemExit(
            f"Completed the grid with {errors} API failures. Rerun with --resume to retry only failed records."
        )


if __name__ == "__main__":
    main()
