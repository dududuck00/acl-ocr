"""Batch CER/WER evaluation for the OCR experiments reported in the paper.

This script reads canonical, merged prediction JSON files under ``results/``.
It never reads retry shards from ``output/`` and never overwrites predictions or
legacy ``*_eval.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from text_error_rates import (
        NORMALIZATION_CHOICES,
        NORMALIZATION_WHITESPACE,
        aggregate_error_rates,
        evaluate_json_records,
    )
except ModuleNotFoundError:  # Supports ``python -m eval.evaluate_paper_cer_wer``.
    from eval.text_error_rates import (
        NORMALIZATION_CHOICES,
        NORMALIZATION_WHITESPACE,
        aggregate_error_rates,
        evaluate_json_records,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "cer_wer"
EXPERIMENT_CHOICES = ("core", "cross_arch", "low_prior", "mask_ablation", "density")
PAPER_CROSS_DATASETS = (
    "from_text",
    "distort",
    "replace_swap_5",
    "replace_shuffle_5",
    "random",
)
PAPER_CROSS_MODES = ("tiny", "small", "base")
PAPER_CROSS_MODELS = {
    "dots_ocr",
    "glmocr",
    "got_ocr",
    "hunyuanocr_1.5",
    "monkeyocr_1.2B",
    "nougat",
    "paddle_ocr_v6",
    "paddleocr_vl_1.6_api",
    "qwen3_vl_8b",
    "qwen3_vl_32b",
    "smoldocling",
}


@dataclass(frozen=True)
class ManifestEntry:
    experiment: str
    model: str
    dataset: str
    mode: str
    input_file: Path
    reference_field: str


def _repo_path(relative_path: str) -> Path:
    return REPOSITORY_ROOT / relative_path


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path)


def _raw_path_from_eval(path_text: str) -> Path:
    path = _repo_path(path_text)
    suffix = "_eval.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"expected an *_eval.json manifest path, got {path_text!r}")
    return path.with_name(path.name[: -len(suffix)] + ".json")


def _reference_field_for_dataset(dataset: str) -> str:
    if dataset == "distort" or dataset.startswith("replace_"):
        return "distorted_text"
    return "gt_text"


def _load_core_manifest() -> list[ManifestEntry]:
    specs: list[tuple[str, str, Mapping[str, str]]] = [
        (
            "from_text",
            "gt_text",
            {
                "tiny": "results/ocr/from_text_tiny.json",
                "small": "results/ocr/from_text_small.json",
                "base": "results/ocr/from_text_1024.json",
            },
        ),
        (
            "distort",
            "distorted_text",
            {
                "tiny": "results/ocr/distort_tiny.json",
                "small": "results/ocr/distort_small.json",
                "base": "results/ocr/distort_1024.json",
            },
        ),
        (
            "replace_swap_5",
            "distorted_text",
            {
                "tiny": "results/replace/swap_5/swap_5_tiny.json",
                "small": "results/replace/swap_5/swap_5_small.json",
                "base": "results/replace/swap_5/swap_5_raw.json",
            },
        ),
        (
            "replace_shuffle_5",
            "distorted_text",
            {
                "tiny": "results/replace/shuffle_5/shuffle_5_tiny.json",
                "small": "results/replace/shuffle_5/shuffle_5_small.json",
                "base": "results/replace/shuffle_5/shuffle_5_raw.json",
            },
        ),
        (
            "replace_swap_10",
            "distorted_text",
            {
                "tiny": "results/replace/swap_10/swap_tiny_10.json",
                "small": "results/replace/swap_10/swap_small_10.json",
                "base": "results/replace/swap_10/swap_raw_10.json",
            },
        ),
        (
            "replace_shuffle_10",
            "distorted_text",
            {
                "tiny": "results/replace/shuffle_10/shuffle_tiny_10.json",
                "small": "results/replace/shuffle_10/shuffle_small_10.json",
                "base": "results/replace/shuffle_10/shuffle_raw_10.json",
            },
        ),
        (
            "random",
            "gt_text",
            {
                "tiny": "results/random/random_tiny.json",
                "small": "results/random/random_small.json",
                "base": "results/random/random_raw.json",
            },
        ),
    ]
    return [
        ManifestEntry("core", "deepseek_ocr", dataset, mode, _repo_path(path), gt_field)
        for dataset, gt_field, paths in specs
        for mode, path in paths.items()
    ]


def _load_cross_arch_manifest() -> list[ManifestEntry]:
    manifest_file = _repo_path("results/other/cross_arch_deepseek_mode_summary.csv")
    entries: list[ManifestEntry] = []
    with manifest_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model not in PAPER_CROSS_MODELS:
                continue
            dataset = row["dataset"]
            entries.append(
                ManifestEntry(
                    "cross_arch",
                    model,
                    dataset,
                    row["mode"],
                    _repo_path(row["path"]),
                    _reference_field_for_dataset(dataset),
                )
            )

    # Updated HunyuanOCR and PP-OCR results are version-isolated and therefore
    # are not present in the legacy precision summary until it is regenerated.
    present_conditions = {(entry.model, entry.dataset, entry.mode) for entry in entries}
    isolated_models = ("hunyuanocr_1.5", "paddle_ocr_v6")
    for model in isolated_models:
        for dataset in PAPER_CROSS_DATASETS:
            for mode in PAPER_CROSS_MODES:
                condition = (model, dataset, mode)
                if condition in present_conditions:
                    continue
                path = _repo_path(
                    f"results/other/{model}_{dataset}_deepseek_modes/"
                    f"{model}_{dataset}_{mode}.json"
                )
                if path.exists():
                    entries.append(
                        ManifestEntry(
                            "cross_arch",
                            model,
                            dataset,
                            mode,
                            path,
                            _reference_field_for_dataset(dataset),
                        )
                    )

    present_conditions = {(entry.model, entry.dataset, entry.mode) for entry in entries}
    expected_conditions = {
        (model, dataset, mode)
        for model in PAPER_CROSS_MODELS
        for dataset in PAPER_CROSS_DATASETS
        for mode in PAPER_CROSS_MODES
    }
    missing_conditions = sorted(expected_conditions - present_conditions)
    if missing_conditions:
        preview = ", ".join("/".join(condition) for condition in missing_conditions[:6])
        suffix = " ..." if len(missing_conditions) > 6 else ""
        raise ValueError(
            f"Cross-architecture evaluation is missing {len(missing_conditions)} required "
            f"version-pinned conditions: {preview}{suffix}"
        )
    return entries


def _load_low_prior_manifest() -> list[ManifestEntry]:
    manifest_file = _repo_path("results/other/deepseek_ocr_low_prior_stress_summary.csv")
    entries: list[ManifestEntry] = []
    with manifest_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entries.append(
                ManifestEntry(
                    "low_prior",
                    "deepseek_ocr",
                    row["type"],
                    row["mode"],
                    _raw_path_from_eval(row["eval_file"]),
                    "gt_text",
                )
            )
    return entries


def _load_mask_manifest() -> list[ManifestEntry]:
    manifest_file = _repo_path("results/other/deepseek_ocr_mask_ablation_summary.csv")
    entries: list[ManifestEntry] = []
    with manifest_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entries.append(
                ManifestEntry(
                    "mask_ablation",
                    "deepseek_ocr",
                    row["dataset"],
                    row["mode"],
                    _raw_path_from_eval(row["eval_file"]),
                    "gt_text",
                )
            )
    return entries


def _load_density_manifest() -> list[ManifestEntry]:
    return [
        ManifestEntry(
            "density",
            "deepseek_ocr",
            "font_density_sweep",
            mode,
            _repo_path(f"results/compress/story_font_density_sweep_{mode}.json"),
            "gt_text",
        )
        for mode in ("tiny", "small", "base", "large")
    ]


def load_manifest(experiments: Sequence[str]) -> list[ManifestEntry]:
    loaders = {
        "core": _load_core_manifest,
        "cross_arch": _load_cross_arch_manifest,
        "low_prior": _load_low_prior_manifest,
        "mask_ablation": _load_mask_manifest,
        "density": _load_density_manifest,
    }
    entries = [entry for experiment in experiments for entry in loaders[experiment]()]
    missing_files = sorted({_relative_path(entry.input_file) for entry in entries if not entry.input_file.is_file()})
    if missing_files:
        formatted = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"manifest references missing prediction files:\n{formatted}")
    return entries


def _with_percentages(metrics: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    for key in ("macro_cer", "micro_cer", "median_cer", "p95_cer", "macro_wer", "micro_wer", "median_wer", "p95_wer"):
        enriched[f"{key}_percent"] = 100.0 * float(enriched[key])
    return enriched


def _evaluate_entry(
    entry: ManifestEntry,
    normalization: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with entry.input_file.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{entry.input_file} must contain a JSON list")

    evaluated, aggregate = evaluate_json_records(
        records,
        reference_field=entry.reference_field,
        prediction_field="ocr_text",
        normalization=normalization,
    )
    source = _relative_path(entry.input_file)
    sample_rows: list[dict[str, Any]] = []
    for metric in evaluated:
        record = records[int(metric["sample_index"])]
        sample_rows.append(
            {
                "experiment": entry.experiment,
                "model": entry.model,
                "dataset": entry.dataset,
                "mode": entry.mode,
                "token_count": record.get("token_count", ""),
                "font_size": record.get("font_size", ""),
                "width": record.get("width", ""),
                "reference_field": entry.reference_field,
                "normalization": normalization,
                "input_file": source,
                **metric,
            }
        )

    summary = {
        "scope": "condition" if entry.experiment != "density" else "density_mode",
        "experiment": entry.experiment,
        "model": entry.model,
        "dataset": entry.dataset,
        "mode": entry.mode,
        "token_count": "",
        "font_size": "",
        "width": "",
        "reference_field": entry.reference_field,
        "normalization": normalization,
        "input_file": source,
        **_with_percentages(aggregate),
    }
    summary["n"] = summary.pop("samples")
    return sample_rows, summary


def _aggregate_sample_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    experiment: str,
    model: str,
    dataset: str,
    mode: str = "all",
    token_count: Any = "",
    font_size: Any = "",
    width: Any = "",
) -> dict[str, Any]:
    aggregate = aggregate_error_rates(rows)
    aggregate["missing_predictions"] = sum(bool(row["prediction_missing"]) for row in rows)
    aggregate["empty_predictions"] = sum(int(row["hypothesis_characters"]) == 0 for row in rows)
    summary = {
        "scope": scope,
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "mode": mode,
        "token_count": token_count,
        "font_size": font_size,
        "width": width,
        "reference_field": rows[0]["reference_field"],
        "normalization": rows[0]["normalization"],
        "input_file": "",
        **_with_percentages(aggregate),
    }
    summary["n"] = summary.pop("samples")
    return summary


def _group_rows(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def build_derived_summaries(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []

    cross_rows = [row for row in sample_rows if row["experiment"] == "cross_arch"]
    for (model, dataset), rows in _group_rows(cross_rows, ("model", "dataset")).items():
        derived.append(
            _aggregate_sample_group(
                rows,
                scope="cross_model_dataset",
                experiment="cross_arch",
                model=str(model),
                dataset=str(dataset),
            )
        )
    for (dataset,), rows in _group_rows(cross_rows, ("dataset",)).items():
        derived.append(
            _aggregate_sample_group(
                rows,
                scope="cross_paper_11_dataset",
                experiment="cross_arch",
                model="paper_11_mean",
                dataset=str(dataset),
            )
        )

    density_rows = [row for row in sample_rows if row["experiment"] == "density"]
    for (mode, token_count, font_size, width), rows in _group_rows(
        density_rows, ("mode", "token_count", "font_size", "width")
    ).items():
        derived.append(
            _aggregate_sample_group(
                rows,
                scope="density_cell",
                experiment="density",
                model="deepseek_ocr",
                dataset="font_density_sweep",
                mode=str(mode),
                token_count=token_count,
                font_size=font_size,
                width=width,
            )
        )
    for (mode, token_count), rows in _group_rows(density_rows, ("mode", "token_count")).items():
        derived.append(
            _aggregate_sample_group(
                rows,
                scope="density_by_token",
                experiment="density",
                model="deepseek_ocr",
                dataset="font_density_sweep",
                mode=str(mode),
                token_count=token_count,
            )
        )
    return derived


SAMPLE_COLUMNS = [
    "experiment",
    "model",
    "dataset",
    "mode",
    "token_count",
    "font_size",
    "width",
    "sample_index",
    "sample_id",
    "image",
    "prediction_missing",
    "cer",
    "wer",
    "character_edits",
    "reference_characters",
    "hypothesis_characters",
    "word_edits",
    "reference_words",
    "hypothesis_words",
    "reference_field",
    "normalization",
    "input_file",
]

SUMMARY_COLUMNS = [
    "scope",
    "experiment",
    "model",
    "dataset",
    "mode",
    "token_count",
    "font_size",
    "width",
    "n",
    "missing_predictions",
    "empty_predictions",
    "macro_cer",
    "micro_cer",
    "median_cer",
    "p95_cer",
    "macro_wer",
    "micro_wer",
    "median_wer",
    "p95_wer",
    "macro_cer_percent",
    "micro_cer_percent",
    "median_cer_percent",
    "p95_cer_percent",
    "macro_wer_percent",
    "micro_wer_percent",
    "median_wer_percent",
    "p95_wer_percent",
    "character_edits",
    "reference_characters",
    "hypothesis_characters",
    "word_edits",
    "reference_words",
    "hypothesis_words",
    "reference_field",
    "normalization",
    "input_file",
]


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_view_files(output_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    mode_order = {"tiny": 0, "small": 1, "base": 2, "large": 3, "all": 4}
    dataset_order = {
        name: index
        for index, name in enumerate(
            (
                "from_text",
                "distort",
                "replace_swap_5",
                "replace_shuffle_5",
                "replace_swap_10",
                "replace_shuffle_10",
                "random",
                "prose",
                "ids",
                "names",
                "code",
                "tables",
                "mask_clean",
                "mask_word_25",
                "mask_word_50",
                "mask_word_75",
                "mask_word_100",
                "mask_char_25",
                "mask_char_50",
                "mask_char_75",
                "mask_char_100",
                "noise_25",
                "noise_50",
                "noise_75",
                "noise_100",
            )
        )
    }
    core_rows = [
        row
        for row in summaries
        if row["scope"] == "condition" and row["experiment"] == "core"
    ]
    core_order = {
        name: index
        for index, name in enumerate(
            (
                "from_text",
                "distort",
                "replace_swap_5",
                "replace_shuffle_5",
                "replace_swap_10",
                "replace_shuffle_10",
                "random",
            )
        )
    }
    indexed_core = {(row["dataset"], row["mode"]): row for row in core_rows}
    core_pivot: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in core_rows}, key=core_order.get):
        pivot: dict[str, Any] = {"dataset": dataset}
        for mode in ("tiny", "small", "base"):
            row = indexed_core[(dataset, mode)]
            for metric in ("macro_cer_percent", "micro_cer_percent", "macro_wer_percent", "micro_wer_percent"):
                pivot[f"{mode}_{metric}"] = row[metric]
        core_pivot.append(pivot)
    core_columns = ["dataset"] + [
        f"{mode}_{metric}"
        for mode in ("tiny", "small", "base")
        for metric in ("macro_cer_percent", "micro_cer_percent", "macro_wer_percent", "micro_wer_percent")
    ]
    _write_csv(output_dir / "deepseek_core_cer_wer_pivot.csv", core_columns, core_pivot)

    compact_columns = [
        "model",
        "dataset",
        "mode",
        "token_count",
        "font_size",
        "width",
        "n",
        "empty_predictions",
        "macro_cer_percent",
        "micro_cer_percent",
        "macro_wer_percent",
        "micro_wer_percent",
    ]
    views = {
        "cross_arch_cer_wer_by_model_dataset.csv": "cross_model_dataset",
        "cross_arch_paper_11_cer_wer_by_dataset.csv": "cross_paper_11_dataset",
        "density_cer_wer_by_token.csv": "density_by_token",
    }
    for filename, scope in views.items():
        rows = [row for row in summaries if row["scope"] == scope]
        if scope == "density_by_token":
            rows.sort(key=lambda row: (mode_order[str(row["mode"])], int(row["token_count"])))
        else:
            rows.sort(
                key=lambda row: (
                    str(row["model"]),
                    dataset_order.get(str(row["dataset"]), 999),
                    str(row["dataset"]),
                )
            )
        _write_csv(output_dir / filename, compact_columns, rows)

    for experiment, filename in (
        ("low_prior", "low_prior_cer_wer.csv"),
        ("mask_ablation", "mask_ablation_cer_wer.csv"),
    ):
        rows = [
            row
            for row in summaries
            if row["scope"] == "condition" and row["experiment"] == experiment
        ]
        rows.sort(
            key=lambda row: (
                dataset_order.get(str(row["dataset"]), 999),
                str(row["dataset"]),
                mode_order[str(row["mode"])],
            )
        )
        _write_csv(output_dir / filename, compact_columns, rows)


def run_batch(
    experiments: Sequence[str],
    output_dir: Path,
    normalization: str,
    workers: int,
) -> None:
    entries = load_manifest(experiments)
    sample_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_entry = {
            executor.submit(_evaluate_entry, entry, normalization): entry for entry in entries
        }
        completed = 0
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                entry_samples, entry_summary = future.result()
            except Exception as error:
                raise RuntimeError(f"failed to evaluate {_relative_path(entry.input_file)}") from error
            sample_rows.extend(entry_samples)
            summaries.append(entry_summary)
            completed += 1
            if completed == len(entries) or completed % 10 == 0:
                print(f"evaluated {completed}/{len(entries)} prediction files")

    sample_rows.sort(
        key=lambda row: (
            str(row["experiment"]),
            str(row["model"]),
            str(row["dataset"]),
            str(row["mode"]),
            int(row["sample_index"]),
        )
    )
    summaries.extend(build_derived_summaries(sample_rows))
    summaries.sort(
        key=lambda row: (
            str(row["scope"]),
            str(row["experiment"]),
            str(row["model"]),
            str(row["dataset"]),
            str(row["mode"]),
            str(row["token_count"]),
            str(row["font_size"]),
            str(row["width"]),
        )
    )

    _write_csv(output_dir / "paper_cer_wer_per_sample.csv", SAMPLE_COLUMNS, sample_rows)
    _write_csv(output_dir / "paper_cer_wer_summary.csv", SUMMARY_COLUMNS, summaries)
    _write_view_files(output_dir, summaries)
    print(f"wrote {len(sample_rows)} sample rows and {len(summaries)} summary rows to {output_dir}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper OCR predictions with CER/WER")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_CHOICES,
        default=list(EXPERIMENT_CHOICES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default=NORMALIZATION_WHITESPACE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    run_batch(args.experiments, args.output_dir, args.normalization, args.workers)


if __name__ == "__main__":
    _main()
