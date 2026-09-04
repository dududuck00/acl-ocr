#!/usr/bin/env python3
"""Validate and evaluate HunyuanOCR-1.5 paper predictions."""

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
MODEL_VERSION = "HunyuanOCR-1.5"
MODEL_LABEL = "hunyuanocr_1.5"
INFERENCE_REVISION = "transformers-tail-repeat-v1"
EXPECTED_SAMPLES = 112
CROSS_DATASETS = ("from_text", "distort", "replace_swap_5", "replace_shuffle_5", "random")
CROSS_MODES = ("tiny", "small", "base")


@dataclass(frozen=True)
class Condition:
    protocol: str
    dataset: str
    mode: str
    prediction_file: Path
    reference_field: str

    @property
    def evaluation_file(self) -> Path:
        return self.prediction_file.with_name(self.prediction_file.stem + "_eval.json")


def conditions(protocol: str, results_dir: Path) -> list[Condition]:
    selected: list[Condition] = []
    if protocol in {"native", "all"}:
        selected.extend(
            [
                Condition("native", "from_text", "native", results_dir / f"{MODEL_LABEL}_from_text.json", "gt_text"),
                Condition("native", "random", "native", results_dir / f"{MODEL_LABEL}_random_ocr.json", "gt_text"),
            ]
        )
    if protocol in {"cross_arch", "all"}:
        for dataset in CROSS_DATASETS:
            reference = "distorted_text" if dataset == "distort" or dataset.startswith("replace_") else "gt_text"
            for mode in CROSS_MODES:
                path = (
                    results_dir
                    / f"{MODEL_LABEL}_{dataset}_deepseek_modes"
                    / f"{MODEL_LABEL}_{dataset}_{mode}.json"
                )
                selected.append(Condition("cross_arch", dataset, mode, path, reference))
    return selected


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_predictions(condition: Condition) -> None:
    path = condition.prediction_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    records = [item for item in load_json(path) if isinstance(item, dict) and "overall_metrics" not in item]
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} contains {len(records)} records; expected {EXPECTED_SAMPLES}")

    images = [item.get("image") for item in records]
    if None in images or len(images) != len(set(images)):
        raise ValueError(f"{path} has missing or duplicate image identifiers")
    incomplete = [item.get("image") for item in records if not item.get("ocr_text") or item.get("error")]
    if incomplete:
        raise ValueError(f"{path} has {len(incomplete)} incomplete predictions; first: {incomplete[0]}")
    versions = {item.get("model_version") for item in records}
    if versions != {MODEL_VERSION}:
        raise ValueError(f"{path} has unexpected or missing model versions: {sorted(map(str, versions))}")
    revisions = {item.get("inference_revision") for item in records}
    if revisions != {INFERENCE_REVISION}:
        raise ValueError(
            f"{path} contains stale decoding results: {sorted(map(str, revisions))}; "
            "rerun inference with --resume before evaluation"
        )
    missing_reference = [item.get("image") for item in records if condition.reference_field not in item]
    if missing_reference:
        raise ValueError(f"{path} is missing {condition.reference_field}; first: {missing_reference[0]}")


def run_evaluator(condition: Condition, max_workers: int, dry_run: bool) -> None:
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
        str(max_workers),
    ]
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def overall_metrics(path: Path) -> dict:
    overall = [
        item["overall_metrics"]
        for item in load_json(path)
        if isinstance(item, dict) and "overall_metrics" in item
    ]
    if len(overall) != 1:
        raise ValueError(f"{path} must contain exactly one overall_metrics record")
    if int(overall[0].get("eval question num", -1)) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} does not summarize {EXPECTED_SAMPLES} samples")
    return overall[0]


def write_summary(selected: list[Condition], output_file: Path) -> None:
    rows = []
    for condition in selected:
        metrics = overall_metrics(condition.evaluation_file)
        rows.append(
            {
                "model": MODEL_LABEL,
                "model_version": MODEL_VERSION,
                "protocol": condition.protocol,
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary saved to: {output_file}")

    native = {row["dataset"]: row for row in rows if row["protocol"] == "native"}
    if set(native) == {"from_text", "random"}:
        natural = float(native["from_text"]["precision_pct"])
        zero_prior = float(native["random"]["precision_pct"])
        print(
            f"Table 6 HunyuanOCR-1.5: natural={natural:.4f}%, "
            f"zero-prior={zero_prior:.4f}%, drop={zero_prior - natural:.4f} points"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HunyuanOCR-1.5 native and cross-architecture results.")
    parser.add_argument("--protocol", choices=("native", "cross_arch", "all"), default="native")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Defaults to results/other/hunyuanocr_1.5_<protocol>_evaluation_summary.csv.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summary_output is None:
        args.summary_output = RESULTS_DIR / f"{MODEL_LABEL}_{args.protocol}_evaluation_summary.csv"
    selected = conditions(args.protocol, args.results_dir)
    for condition in selected:
        validate_predictions(condition)
        print(f"validated: {condition.prediction_file.relative_to(REPO_ROOT)} ({EXPECTED_SAMPLES}/{EXPECTED_SAMPLES})")
        if not args.skip_eval:
            run_evaluator(condition, args.max_workers, args.dry_run)

    if args.dry_run:
        print("dry run complete; no evaluation or summary files were written")
        return
    write_summary(selected, args.summary_output)


if __name__ == "__main__":
    main()
