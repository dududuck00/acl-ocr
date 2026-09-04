#!/usr/bin/env python3
"""Validate and evaluate PP-OCRv6 cross-architecture rebuttal predictions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "other"
EVALUATOR = REPO_ROOT / "eval" / "eval.py"
MODEL_LABEL = "paddle_ocr_v6"
MODEL_VERSION = "PP-OCRv6"
EXPECTED_SAMPLES = 112
DATASETS = ("from_text", "distort", "replace_swap_5", "replace_shuffle_5", "random")
MODES = ("tiny", "small", "base")


@dataclass(frozen=True)
class Condition:
    dataset: str
    mode: str
    prediction_file: Path
    reference_field: str

    @property
    def evaluation_file(self) -> Path:
        return self.prediction_file.with_name(self.prediction_file.stem + "_eval.json")


def conditions(results_dir: Path) -> list[Condition]:
    selected = []
    for dataset in DATASETS:
        reference = "distorted_text" if dataset == "distort" or dataset.startswith("replace_") else "gt_text"
        for mode in MODES:
            prediction = (
                results_dir
                / f"{MODEL_LABEL}_{dataset}_deepseek_modes"
                / f"{MODEL_LABEL}_{dataset}_{mode}.json"
            )
            selected.append(Condition(dataset, mode, prediction, reference))
    return selected


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(condition: Condition) -> None:
    path = condition.prediction_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    records = [item for item in load_json(path) if isinstance(item, dict) and "overall_metrics" not in item]
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} contains {len(records)} records; expected {EXPECTED_SAMPLES}")

    keys = [item.get("processed_image") or item.get("image") for item in records]
    if None in keys or len(keys) != len(set(keys)):
        raise ValueError(f"{path} has missing or duplicate image identifiers")
    wrong_version = [key for key, item in zip(keys, records) if item.get("model_version") != MODEL_VERSION]
    if wrong_version:
        raise ValueError(f"{path} has stale model versions; first: {wrong_version[0]}")
    api_errors = [(key, item.get("error")) for key, item in zip(keys, records) if item.get("error")]
    if api_errors:
        raise ValueError(f"{path} has {len(api_errors)} API failures; first: {api_errors[0]}")
    unaudited_empty = [
        key
        for key, item in zip(keys, records)
        if not item.get("ocr_text") and item.get("recognition_empty") is not True
    ]
    if unaudited_empty:
        raise ValueError(f"{path} has unaudited empty predictions; first: {unaudited_empty[0]}")
    missing_reference = [key for key, item in zip(keys, records) if not item.get(condition.reference_field)]
    if missing_reference:
        raise ValueError(f"{path} is missing {condition.reference_field}; first: {missing_reference[0]}")


def overall_metrics(path: Path) -> dict:
    overall = [
        item["overall_metrics"]
        for item in load_json(path)
        if isinstance(item, dict) and "overall_metrics" in item
    ]
    if len(overall) != 1 or int(overall[0].get("eval question num", -1)) != EXPECTED_SAMPLES:
        raise ValueError(f"Invalid aggregate metrics in {path}")
    return overall[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PP-OCRv6 5x3 rebuttal grid.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS_DIR / f"{MODEL_LABEL}_cross_arch_evaluation_summary.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = conditions(args.results_dir)
    for condition in selected:
        validate(condition)
        print(f"validated: {condition.prediction_file.relative_to(REPO_ROOT)} ({EXPECTED_SAMPLES}/{EXPECTED_SAMPLES})")
        command = [
            sys.executable,
            str(EVALUATOR),
            "--predict_file",
            str(condition.prediction_file),
            "--output_file",
            str(condition.evaluation_file),
            "--reference-field",
            condition.reference_field,
            "--max_workers",
            str(args.max_workers),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        print("dry run complete; no evaluation files were written")
        return

    rows = []
    for condition in selected:
        metrics = overall_metrics(condition.evaluation_file)
        rows.append(
            {
                "model": MODEL_LABEL,
                "model_version": MODEL_VERSION,
                "dataset": condition.dataset,
                "mode": condition.mode,
                "precision_pct": f"{float(metrics['precision']) * 100.0:.4f}",
                "recall_pct": f"{float(metrics['recall']) * 100.0:.4f}",
                "f_measure_pct": f"{float(metrics['f_measure']) * 100.0:.4f}",
                "cer_pct": f"{float(metrics['cer']) * 100.0:.4f}",
                "wer_pct": f"{float(metrics['wer']) * 100.0:.4f}",
                "samples": EXPECTED_SAMPLES,
                "prediction_file": str(condition.prediction_file.relative_to(REPO_ROOT)),
                "evaluation_file": str(condition.evaluation_file.relative_to(REPO_ROOT)),
            }
        )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary saved to: {args.summary_output}")


if __name__ == "__main__":
    main()
