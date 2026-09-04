#!/usr/bin/env python3
"""Run PP-OCRv6 on native-resolution Table 6 inputs via PaddleOCR's official API."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL = "PP-OCRv6"
MODEL_LABEL = "paddle_ocr_v6"
RESULTS_DIR = REPO_ROOT / "results" / "other"
EXPECTED_SAMPLES = 112
DATASETS = {
    "from_text": {
        "data_file": REPO_ROOT / "fox_data" / "data.json",
        "image_dir": REPO_ROOT / "fox_data" / "from_text",
        "output_file": RESULTS_DIR / f"{MODEL_LABEL}_from_text.json",
    },
    "random": {
        "data_file": REPO_ROOT / "fox_data" / "random.json",
        "image_dir": REPO_ROOT / "fox_data" / "random",
        "output_file": RESULTS_DIR / f"{MODEL_LABEL}_random_ocr.json",
    },
}
OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def record_key(item: dict) -> str | None:
    return item.get("processed_image") or item.get("image")


def validate_dataset(dataset: str, limit: int | None) -> tuple[list[dict], Path, Path]:
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

    images = [item.get("image") for item in data]
    if None in images or len(images) != len(set(images)):
        raise ValueError(f"{data_file} has missing or duplicate image identifiers")
    missing = [name for name in images if not (image_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} is missing {len(missing)} images; first: {missing[0]}")
    return data, image_dir, output_file


class PaddleOCRV6API:
    def __init__(self, token: str, request_timeout: int, poll_interval: float, poll_timeout: int):
        self.headers = {"Authorization": f"bearer {token}"}
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def submit(self, image_path: Path, retries: int, retry_sleep: float) -> str:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                data = {"model": MODEL, "optionalPayload": json.dumps(OPTIONAL_PAYLOAD)}
                with image_path.open("rb") as image_file:
                    response = requests.post(
                        JOB_URL,
                        headers=self.headers,
                        data=data,
                        files={"file": image_file},
                        timeout=self.request_timeout,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"submit HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                return response.json()["data"]["jobId"]
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_sleep * (attempt + 1))
        raise RuntimeError(f"failed to submit {image_path}: {last_error}")

    def wait_for_result_url(self, job_id: str) -> str:
        started = time.monotonic()
        while True:
            if time.monotonic() - started > self.poll_timeout:
                raise TimeoutError(f"poll timeout for job {job_id}")
            response = requests.get(
                f"{JOB_URL}/{job_id}",
                headers=self.headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            state = data["state"]
            if state == "done":
                return data["resultUrl"]["jsonUrl"]
            if state == "failed":
                raise RuntimeError(f"job {job_id} failed: {data.get('errorMsg', 'unknown error')}")
            if state not in {"pending", "running"}:
                raise RuntimeError(f"job {job_id} returned unknown state: {state}")
            time.sleep(self.poll_interval)

    def download_jsonl(self, result_url: str) -> list[dict]:
        response = requests.get(result_url, timeout=self.request_timeout)
        response.raise_for_status()
        records = []
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
        if not records:
            raise ValueError("the result JSONL is empty")
        return records


def find_rec_texts(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        for key in ("rec_texts", "recTexts"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [str(text) for text in candidate if text is not None and str(text) != ""]
        for child in value.values():
            found = find_rec_texts(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_rec_texts(child)
            if found is not None:
                return found
    return None


def extract_ocr_text(jsonl_records: list[dict]) -> str:
    page_texts = []
    for record in jsonl_records:
        result = record.get("result", record)
        ocr_results = result.get("ocrResults") if isinstance(result, dict) else None
        pages = ocr_results if isinstance(ocr_results, list) else [result]
        for page in pages:
            pruned_result = page.get("prunedResult", page) if isinstance(page, dict) else page
            rec_texts = find_rec_texts(pruned_result)
            if rec_texts:
                page_texts.append("\n".join(rec_texts))
    # A successful OCR job may legitimately contain no recognized lines,
    # especially for aggressively resized rebuttal inputs.  Keep that as an
    # auditable empty model prediction instead of misclassifying it as an API
    # failure and retrying it forever.
    return "\n".join(page_texts).strip()


def process_image(
    item: dict,
    image_dir: Path,
    api: PaddleOCRV6API,
    retries: int,
    retry_sleep: float,
) -> dict:
    image_path = image_dir / item["image"]
    try:
        job_id = api.submit(image_path, retries=retries, retry_sleep=retry_sleep)
        result_url = api.wait_for_result_url(job_id)
        records = api.download_jsonl(result_url)
        output = dict(item)
        output["ocr_text"] = extract_ocr_text(records)
        output["recognition_empty"] = not bool(output["ocr_text"])
        output["model_name"] = MODEL
        output["model_version"] = "PP-OCRv6"
        output["job_id"] = job_id
        return output
    except Exception as exc:
        output = dict(item)
        output["ocr_text"] = ""
        output["model_name"] = MODEL
        output["model_version"] = "PP-OCRv6"
        output["error"] = str(exc)
        return output


def merge_results(data: list[dict], existing: list[dict], new_results: list[dict]) -> list[dict]:
    by_key = {record_key(item): item for item in existing + new_results if record_key(item)}
    order = {item["image"]: index for index, item in enumerate(data)}
    return sorted(by_key.values(), key=lambda item: order.get(record_key(item), 10**9))


def run_dataset(args, api: PaddleOCRV6API, dataset: str) -> None:
    data, image_dir, output_file = validate_dataset(dataset, args.limit)
    existing = load_json(output_file) if args.resume and output_file.exists() else []
    completed = {
        record_key(item)
        for item in existing
        if not item.get("error")
        and (item.get("ocr_text") or item.get("recognition_empty") is True)
    }
    todo = [item for item in data if item["image"] not in completed]

    print(f"[PP-OCRv6][native:{dataset}] data: {DATASETS[dataset]['data_file']}")
    print(f"[PP-OCRv6][native:{dataset}] images: {image_dir}")
    print(f"[PP-OCRv6][native:{dataset}] output: {output_file}")
    print(f"[PP-OCRv6][native:{dataset}] total={len(data)} completed={len(completed)} todo={len(todo)}")
    if args.dry_run or not todo:
        return

    new_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_image,
                item,
                image_dir,
                api,
                args.retries,
                args.retry_sleep,
            ): item["image"]
            for item in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"PP-OCRv6 {dataset}", unit="image"):
            new_results.append(future.result())
            if len(new_results) % args.checkpoint_every == 0:
                save_json(merge_results(data, existing, new_results), output_file)

    merged = merge_results(data, existing, new_results)
    save_json(merged, output_file)
    successful = sum(not item.get("error") for item in merged)
    empty = sum(item.get("recognition_empty") is True and not item.get("error") for item in merged)
    errors = sum(bool(item.get("error")) for item in merged)
    print(
        f"[PP-OCRv6][native:{dataset}] saved={len(merged)} "
        f"successful_jobs={successful} empty_predictions={empty} errors={errors}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official PP-OCRv6 API on native natural/random inputs for main-paper Table 6."
    )
    parser.add_argument("--token", default=os.environ.get("PADDLEOCR_TOKEN", ""))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(DATASETS))
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
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if not args.dry_run and not args.token:
        raise ValueError("Missing PaddleOCR API token. Set PADDLEOCR_TOKEN or pass --token.")

    api = PaddleOCRV6API(
        token=args.token,
        request_timeout=args.request_timeout,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )
    for dataset in args.datasets:
        run_dataset(args, api, dataset)
    if args.dry_run:
        print("dry run complete; no API requests or result files were written")


if __name__ == "__main__":
    main()
