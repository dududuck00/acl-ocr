#!/usr/bin/env python3
"""Evaluate native-resolution Qwen3-VL OCR results for main-paper Table 6."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "other"
EVALUATOR = REPO_ROOT / "eval" / "eval.py"
EXPECTED_SAMPLES = 112
MODEL_IDS = {
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
}
CONDITIONS = {
    "natural": "from_text",
    "zero_prior": "random_ocr",
}


def prediction_path(model: str, suffix: str) -> Path:
    return RESULTS_DIR / f"{model}_{suffix}.json"


def evaluation_path(model: str, suffix: str) -> Path:
    return RESULTS_DIR / f"{model}_{suffix}_eval.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_predictions(path: Path, expected_model_id: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    data = load_json(path)
    records = [item for item in data if isinstance(item, dict) and "overall_metrics" not in item]
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} contains {len(records)} records; expected {EXPECTED_SAMPLES}")

    images = [item.get("image") for item in records]
    if None in images or len(set(images)) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} has missing or duplicate image identifiers")

    incomplete = [item.get("image") for item in records if not item.get("ocr_text") or item.get("error")]
    if incomplete:
        raise ValueError(f"{path} has {len(incomplete)} incomplete predictions")

    model_ids = {item.get("model_name") for item in records if item.get("model_name")}
    if model_ids and model_ids != {expected_model_id}:
        raise ValueError(f"{path} has unexpected model IDs: {sorted(model_ids)}")


def run_evaluator(predict_file: Path, output_file: Path, max_workers: int, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--predict_file",
        str(predict_file),
        "--output_file",
        str(output_file),
        "--reference-field",
        "gt_text",
        "--max_workers",
        str(max_workers),
    ]
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_overall_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    data = load_json(path)
    overall = [item["overall_metrics"] for item in data if isinstance(item, dict) and "overall_metrics" in item]
    if len(overall) != 1:
        raise ValueError(f"{path} must contain exactly one overall_metrics record")
    if int(overall[0].get("eval question num", -1)) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} does not summarize {EXPECTED_SAMPLES} samples")
    return overall[0]


def write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "natural_precision_pct",
        "zero_prior_precision_pct",
        "precision_drop_points",
        "natural_cer_pct",
        "zero_prior_cer_pct",
        "natural_wer_pct",
        "zero_prior_wer_pct",
        "samples_per_condition",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B/32B native natural and zero-prior OCR results."
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS_DIR / "qwen3_vl_native_table6_summary.csv",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only validate predictions and summarize existing *_eval.json files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print evaluator commands only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    for model, expected_model_id in MODEL_IDS.items():
        for suffix in CONDITIONS.values():
            predict_file = prediction_path(model, suffix)
            validate_predictions(predict_file, expected_model_id)
            print(f"validated: {predict_file.relative_to(REPO_ROOT)} ({EXPECTED_SAMPLES}/{EXPECTED_SAMPLES})")
            if not args.skip_eval:
                run_evaluator(
                    predict_file,
                    evaluation_path(model, suffix),
                    max_workers=args.max_workers,
                    dry_run=args.dry_run,
                )

    if args.dry_run:
        print("dry run complete; no evaluation or summary files were written")
        return

    rows = []
    for model in MODEL_IDS:
        natural = load_overall_metrics(evaluation_path(model, CONDITIONS["natural"]))
        zero_prior = load_overall_metrics(evaluation_path(model, CONDITIONS["zero_prior"]))
        natural_precision = float(natural["precision"]) * 100.0
        zero_prior_precision = float(zero_prior["precision"]) * 100.0
        rows.append(
            {
                "model": model,
                "natural_precision_pct": f"{natural_precision:.4f}",
                "zero_prior_precision_pct": f"{zero_prior_precision:.4f}",
                "precision_drop_points": f"{zero_prior_precision - natural_precision:.4f}",
                "natural_cer_pct": f"{float(natural.get('cer', 0.0)) * 100.0:.4f}",
                "zero_prior_cer_pct": f"{float(zero_prior.get('cer', 0.0)) * 100.0:.4f}",
                "natural_wer_pct": f"{float(natural.get('wer', 0.0)) * 100.0:.4f}",
                "zero_prior_wer_pct": f"{float(zero_prior.get('wer', 0.0)) * 100.0:.4f}",
                "samples_per_condition": EXPECTED_SAMPLES,
            }
        )

    write_summary(rows, args.summary_output)
    print(f"summary saved to: {args.summary_output}")
    for row in rows:
        print(
            f"{row['model']}: natural={row['natural_precision_pct']}%, "
            f"zero-prior={row['zero_prior_precision_pct']}%, "
            f"drop={row['precision_drop_points']} points"
        )


if __name__ == "__main__":
    main()
