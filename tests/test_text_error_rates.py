import random
import unittest

from eval.text_error_rates import (
    aggregate_error_rates,
    calculate_error_rates,
    evaluate_json_records,
    levenshtein_distance,
)


def dynamic_programming_distance(reference, hypothesis):
    """Small, obviously correct oracle used only by randomized tests."""

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_symbol in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_symbol in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_symbol != hypothesis_symbol),
                )
            )
        previous = current
    return previous[-1]


class LevenshteinDistanceTests(unittest.TestCase):
    def test_known_distances(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance(["a", "b"], ["b", "a"]), 2)

    def test_bit_vector_matches_dynamic_programming_oracle(self):
        generator = random.Random(20260830)
        alphabet = "abcd"
        for _ in range(500):
            reference = "".join(generator.choice(alphabet) for _ in range(generator.randrange(12)))
            hypothesis = "".join(generator.choice(alphabet) for _ in range(generator.randrange(12)))
            self.assertEqual(
                levenshtein_distance(reference, hypothesis),
                dynamic_programming_distance(reference, hypothesis),
                (reference, hypothesis),
            )


class ErrorRateTests(unittest.TestCase):
    def test_exact_match(self):
        metrics = calculate_error_rates("A B", "A B")
        self.assertEqual(metrics["cer"], 0.0)
        self.assertEqual(metrics["wer"], 0.0)

    def test_nfc_and_whitespace_normalization(self):
        metrics = calculate_error_rates("  café\r\nB\t", "cafe\u0301   B")
        self.assertEqual(metrics["cer"], 0.0)
        self.assertEqual(metrics["wer"], 0.0)

    def test_case_and_punctuation_are_preserved(self):
        case_metrics = calculate_error_rates("ID-7", "id-7")
        punctuation_metrics = calculate_error_rates("ID-7", "ID7")
        self.assertEqual(case_metrics["cer"], 0.5)
        self.assertEqual(case_metrics["wer"], 1.0)
        self.assertEqual(punctuation_metrics["cer"], 0.25)
        self.assertEqual(punctuation_metrics["wer"], 1.0)

    def test_empty_prediction_is_full_deletion(self):
        metrics = calculate_error_rates("abc def", "")
        self.assertEqual(metrics["cer"], 1.0)
        self.assertEqual(metrics["wer"], 1.0)

    def test_insertions_can_exceed_one(self):
        metrics = calculate_error_rates("a", "a b c")
        self.assertEqual(metrics["cer"], 4.0)
        self.assertEqual(metrics["wer"], 2.0)

    def test_order_and_duplicates_are_penalized(self):
        reordered = calculate_error_rates("a b", "b a")
        duplicate = calculate_error_rates("a a", "a")
        self.assertAlmostEqual(reordered["cer"], 2 / 3)
        self.assertEqual(reordered["wer"], 1.0)
        self.assertAlmostEqual(duplicate["cer"], 2 / 3)
        self.assertEqual(duplicate["wer"], 0.5)

    def test_macro_and_micro_are_distinct(self):
        rows = [
            calculate_error_rates("a", ""),
            calculate_error_rates("abcdefghij", "xbcdefghij"),
        ]
        aggregate = aggregate_error_rates(rows)
        self.assertAlmostEqual(aggregate["macro_cer"], 0.55)
        self.assertAlmostEqual(aggregate["micro_cer"], 2 / 11)

    def test_reference_field_is_explicit_and_sentinel_is_ignored(self):
        records = [
            {
                "image": "non_numeric_name.png",
                "gt_text": "natural text",
                "distorted_text": "altered text",
                "ocr_text": "altered text",
            },
            {"overall_metrics": {"legacy": True}},
        ]
        _, gt_summary = evaluate_json_records(records, "gt_text")
        samples, distorted_summary = evaluate_json_records(records, "distorted_text")
        self.assertGreater(gt_summary["macro_cer"], 0.0)
        self.assertEqual(distorted_summary["macro_cer"], 0.0)
        self.assertEqual(len(samples), 1)

    def test_missing_prediction_is_scored_as_empty(self):
        samples, summary = evaluate_json_records([{"gt_text": "a b"}], "gt_text")
        self.assertEqual(samples[0]["cer"], 1.0)
        self.assertEqual(samples[0]["wer"], 1.0)
        self.assertEqual(summary["missing_predictions"], 1)

    def test_missing_reference_fails_loudly(self):
        with self.assertRaises(KeyError):
            evaluate_json_records([{"ocr_text": "prediction"}], "gt_text")


if __name__ == "__main__":
    unittest.main()
