"""Case-sensitive CER/WER utilities for OCR reconstruction evaluation.

The default normalization is deliberately conservative and symmetric: Unicode
NFC is applied and every whitespace run is collapsed to one ASCII space.  Case,
punctuation, digits, and all non-whitespace characters are preserved.

Both CER and WER use the reference length as the denominator, so insertion-heavy
predictions can legitimately score above 1.0 (100%).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence


NORMALIZATION_WHITESPACE = "whitespace"
NORMALIZATION_LINEBREAKS = "linebreaks"
NORMALIZATION_CHOICES = (NORMALIZATION_WHITESPACE, NORMALIZATION_LINEBREAKS)


def normalize_text(text: Any, mode: str = NORMALIZATION_WHITESPACE) -> str:
    """Normalize text symmetrically without changing lexical content.

    ``whitespace`` (the paper default) converts every whitespace run to one
    ASCII space. ``linebreaks`` only canonicalizes line endings, trims trailing
    horizontal whitespace on each line, and removes outer whitespace.
    """

    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"text must be str or None, got {type(text).__name__}")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    if mode == NORMALIZATION_WHITESPACE:
        return " ".join(normalized.split())
    if mode == NORMALIZATION_LINEBREAKS:
        lines = [line.rstrip(" \t\f\v") for line in normalized.split("\n")]
        return "\n".join(lines).strip()
    raise ValueError(f"unknown normalization mode {mode!r}; choose from {NORMALIZATION_CHOICES}")


def levenshtein_distance(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> int:
    """Return exact Levenshtein distance using Myers' bit-vector algorithm.

    The implementation works for strings and arbitrary hashable token
    sequences. It uses Python integers as unbounded bit vectors, which keeps
    long-document OCR evaluation fast without adding a third-party dependency.
    """

    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    # Distance is symmetric. Keeping the shorter sequence in the bit vector
    # reduces its width for strongly length-imbalanced hallucinations.
    if len(reference) <= len(hypothesis):
        pattern = reference
        text = hypothesis
    else:
        pattern = hypothesis
        text = reference

    pattern_length = len(pattern)
    equality_masks: dict[Hashable, int] = {}
    for index, symbol in enumerate(pattern):
        equality_masks[symbol] = equality_masks.get(symbol, 0) | (1 << index)

    bit_mask = (1 << pattern_length) - 1
    highest_bit = 1 << (pattern_length - 1)
    positive_vertical = bit_mask
    negative_vertical = 0
    distance = pattern_length

    for symbol in text:
        equality = equality_masks.get(symbol, 0)
        vertical_or_equality = equality | negative_vertical
        horizontal = (
            (((equality & positive_vertical) + positive_vertical) ^ positive_vertical)
            | equality
        )
        positive_horizontal = negative_vertical | ~(horizontal | positive_vertical)
        negative_horizontal = positive_vertical & horizontal

        if positive_horizontal & highest_bit:
            distance += 1
        elif negative_horizontal & highest_bit:
            distance -= 1

        positive_horizontal = ((positive_horizontal << 1) | 1) & bit_mask
        negative_horizontal = (negative_horizontal << 1) & bit_mask
        positive_vertical = (
            negative_horizontal | ~(vertical_or_equality | positive_horizontal)
        ) & bit_mask
        negative_vertical = positive_horizontal & vertical_or_equality

    return distance


def _error_rate(edits: int, reference_length: int) -> float:
    """Normalize by reference length while defining empty-reference behavior."""

    if reference_length:
        return edits / reference_length
    return 0.0 if edits == 0 else float(edits)


def calculate_error_rates(
    reference: Any,
    hypothesis: Any,
    normalization: str = NORMALIZATION_WHITESPACE,
) -> dict[str, int | float]:
    """Calculate case-sensitive CER and WER plus their sufficient statistics."""

    normalized_reference = normalize_text(reference, normalization)
    normalized_hypothesis = normalize_text(hypothesis, normalization)

    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    character_edits = levenshtein_distance(normalized_reference, normalized_hypothesis)
    word_edits = levenshtein_distance(reference_words, hypothesis_words)

    return {
        "cer": _error_rate(character_edits, len(normalized_reference)),
        "wer": _error_rate(word_edits, len(reference_words)),
        "character_edits": character_edits,
        "reference_characters": len(normalized_reference),
        "hypothesis_characters": len(normalized_hypothesis),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for ``0 <= quantile <= 1``."""

    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def aggregate_error_rates(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
    """Aggregate per-sample metrics into macro, micro, median, and P95 rates."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot aggregate an empty metric collection")

    cer_values = [float(row["cer"]) for row in materialized]
    wer_values = [float(row["wer"]) for row in materialized]
    character_edits = sum(int(row["character_edits"]) for row in materialized)
    reference_characters = sum(int(row["reference_characters"]) for row in materialized)
    hypothesis_characters = sum(int(row["hypothesis_characters"]) for row in materialized)
    word_edits = sum(int(row["word_edits"]) for row in materialized)
    reference_words = sum(int(row["reference_words"]) for row in materialized)
    hypothesis_words = sum(int(row["hypothesis_words"]) for row in materialized)

    return {
        "samples": len(materialized),
        "macro_cer": statistics.fmean(cer_values),
        "micro_cer": _error_rate(character_edits, reference_characters),
        "median_cer": statistics.median(cer_values),
        "p95_cer": percentile(cer_values, 0.95),
        "macro_wer": statistics.fmean(wer_values),
        "micro_wer": _error_rate(word_edits, reference_words),
        "median_wer": statistics.median(wer_values),
        "p95_wer": percentile(wer_values, 0.95),
        "character_edits": character_edits,
        "reference_characters": reference_characters,
        "hypothesis_characters": hypothesis_characters,
        "word_edits": word_edits,
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
    }


def evaluate_json_records(
    records: Sequence[Mapping[str, Any]],
    reference_field: str,
    prediction_field: str = "ocr_text",
    normalization: str = NORMALIZATION_WHITESPACE,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Evaluate JSON records, treating a missing/null prediction as empty text."""

    evaluated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if "overall_metrics" in record and reference_field not in record:
            continue
        if reference_field not in record:
            raise KeyError(f"record {index} is missing reference field {reference_field!r}")
        reference = record[reference_field]
        if not isinstance(reference, str):
            raise TypeError(
                f"record {index} field {reference_field!r} must be str, "
                f"got {type(reference).__name__}"
            )

        prediction_missing = prediction_field not in record or record[prediction_field] is None
        prediction = "" if prediction_missing else record[prediction_field]
        if not isinstance(prediction, str):
            raise TypeError(
                f"record {index} field {prediction_field!r} must be str or null, "
                f"got {type(prediction).__name__}"
            )

        if not normalize_text(reference, normalization):
            raise ValueError(f"record {index} has an empty normalized reference")

        metrics = calculate_error_rates(reference, prediction, normalization)
        evaluated.append(
            {
                "sample_index": index,
                "sample_id": record.get("id", record.get("image", index)),
                "image": record.get("image", ""),
                "prediction_missing": prediction_missing,
                **metrics,
            }
        )

    if not evaluated:
        raise ValueError("input contains no evaluable records")
    aggregate = aggregate_error_rates(evaluated)
    aggregate["missing_predictions"] = sum(
        bool(row["prediction_missing"]) for row in evaluated
    )
    aggregate["empty_predictions"] = sum(
        int(row["hypothesis_characters"]) == 0 for row in evaluated
    )
    return evaluated, aggregate


def _main() -> None:
    parser = argparse.ArgumentParser(description="Calculate case-sensitive OCR CER and WER")
    parser.add_argument(
        "--input",
        "--predict-file",
        "--predict_file",
        dest="input",
        required=True,
        type=Path,
        help="Prediction JSON list",
    )
    parser.add_argument(
        "--reference-field",
        "--gt-field",
        dest="reference_field",
        required=True,
        help="Explicit GT field name",
    )
    parser.add_argument(
        "--prediction-field",
        "--pred-field",
        dest="prediction_field",
        default="ocr_text",
    )
    parser.add_argument(
        "--output",
        "--output-file",
        "--output_file",
        dest="output",
        type=Path,
        help="Optional metrics-only JSON output",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default=NORMALIZATION_WHITESPACE,
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("input JSON must contain a list")

    samples, aggregate = evaluate_json_records(
        records,
        reference_field=args.reference_field,
        prediction_field=args.prediction_field,
        normalization=args.normalization,
    )
    payload = {
        "input": str(args.input),
        "reference_field": args.reference_field,
        "prediction_field": args.prediction_field,
        "normalization": args.normalization,
        "overall_metrics": aggregate,
        "samples": samples,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload["overall_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
