"""Synthetic boundary checks; never open the real corpus or generate its poems."""
import json
from pathlib import Path
import tempfile
import unittest

from wordconv_lab.free_generation import (
    collect_spans, extension_rejection_reason, line_rejection_reason,
    load_five_training, score_lines, write_new_json,
)
from wordconv_lab.restoration import BidirNGram


def fixture():
    lines = ["ABCDE", "ABFGH", "IJCDE", "IJFGH",
             "KLMNO", "KLPQR", "STMNO", "STPQR"]
    poem = {"poem_id": "synthetic-0", "split": "train", "length": 5,
            "lines": lines}
    reference = {"characters": {
        char: [{"tone": "P", "group_id": "synthetic-level-1"}]
        for char in set("".join(lines))
    }}
    return [poem], reference, {"issues": []}


class FreeGenerationBoundaryTests(unittest.TestCase):
    def test_loader_skips_held_out_rows_before_decoding(self):
        poems, _, _ = fixture()
        other_length = dict(poems[0], poem_id="synthetic-seven", length=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_bytes(
                b'{"split":"test", BROKEN}\n{"split":"dev", BROKEN}\n'
                + json.dumps(poems[0]).encode() + b'\n'
                + json.dumps(other_length).encode() + b'\n')
            selected, counts = load_five_training(path)
        self.assertEqual(selected, poems)
        self.assertEqual(counts["source_test"], 1)
        self.assertEqual(counts["source_dev"], 1)
        self.assertEqual(counts["train_other_length_skipped"], 1)

    def test_train_only_input_rejects_held_out_records(self):
        for split in ("dev", "test"):
            poems, reference, editorial = fixture()
            poems[0]["split"] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                collect_spans(poems, reference, editorial)

    def test_spans_preserve_every_exact_origin_and_threshold(self):
        poems, reference, editorial = fixture()
        resources = collect_spans(poems, reference, editorial, min_count=2)
        self.assertEqual(set(resources["prefixes"]), {"AB", "IJ", "KL", "ST"})
        self.assertEqual(set(resources["suffixes"]), {"CDE", "FGH", "MNO", "PQR"})
        for kind, start, end in (("prefixes", 0, 2), ("suffixes", 2, 5)):
            for text, record in resources[kind].items():
                expected = [{"poem_id": "synthetic-0", "row": row,
                             "start": start, "end": end}
                            for row, line in enumerate(poems[0]["lines"])
                            if line[start:end] == text]
                self.assertEqual(record["text"], text)
                self.assertEqual(record["count"], len(expected))
                self.assertEqual(record["sources"], expected)
                self.assertEqual(record["tones"], "P" * len(text))
        self.assertEqual(set(resources["train_lines"]), set(poems[0]["lines"]))

        # Leave two fragments at count one; the threshold must not relax.
        poems[0]["lines"][5] = "UVWXY"
        for char in "UVWXY":
            reference["characters"][char] = [
                {"tone": "P", "group_id": "synthetic-level-1"}]
        resources = collect_spans(poems, reference, editorial, min_count=2)
        self.assertNotIn("KL", resources["prefixes"])
        self.assertNotIn("PQR", resources["suffixes"])
        self.assertNotIn("UV", resources["prefixes"])
        self.assertNotIn("WXY", resources["suffixes"])
        self.assertIn("AB", resources["prefixes"])

    def test_unknown_mixed_tone_and_editorial_characters_are_not_resolved(self):
        poems, reference, editorial = fixture()
        del reference["characters"]["A"]
        reference["characters"]["F"].append(
            {"tone": "Z", "group_id": "synthetic-oblique-1"})
        editorial["issues"].append({"character": "I", "issue_id": "synthetic-issue"})
        resources = collect_spans(poems, reference, editorial)
        self.assertNotIn("AB", resources["prefixes"])
        self.assertNotIn("FGH", resources["suffixes"])
        self.assertNotIn("IJ", resources["prefixes"])
        self.assertIn("KL", resources["prefixes"])
        self.assertIn("CDE", resources["suffixes"])

    def test_same_tone_multiple_groups_cannot_supply_unique_rhyme(self):
        poems, reference, editorial = fixture()
        reference["characters"]["E"].append(
            {"tone": "P", "group_id": "synthetic-level-2"})
        resources = collect_spans(poems, reference, editorial)
        self.assertEqual(resources["suffixes"]["CDE"]["tones"], "PPP")
        self.assertIsNone(resources["suffixes"]["CDE"]["rhyme_group"])
        self.assertEqual(resources["suffixes"]["FGH"]["rhyme_group"],
                         "synthetic-level-1")
        reference["characters"]["H"] = [
            {"tone": "Z", "group_id": "synthetic-oblique-1"}]
        resources = collect_spans(poems, reference, editorial)
        self.assertEqual(resources["suffixes"]["FGH"]["tones"], "PPZ")
        self.assertIsNone(resources["suffixes"]["FGH"]["rhyme_group"])

    def test_line_shape_whole_source_copy_and_internal_repetition(self):
        source_lines = {"ABCDE", "FGHIJ"}
        for text in ("ABCD", "ABCDEF", "ABCDE", "ABACA"):
            with self.subTest(text=text):
                self.assertIsNotNone(line_rejection_reason(text, source_lines))
        self.assertIsNone(line_rejection_reason("ABCDX", source_lines))

    def test_extension_rejects_duplicate_lines_and_excessive_repetition(self):
        rhymes = (1, 3, 5, 7)
        self.assertIsNotNone(extension_rejection_reason(("ABCDE",), "ABCDE", 1, rhymes))
        existing = ("ABCDE", "AFGHI")
        self.assertIsNotNone(extension_rejection_reason(existing, "AJKLM", 2, rhymes))
        self.assertIsNone(extension_rejection_reason(existing, "NOPQR", 2, rhymes))

    def test_rhyme_repetition_and_middle_couplet_same_position_are_rejected(self):
        rhymes = (1, 3, 5, 7)
        existing = ("ABCDE", "FGHIJ", "KLMNO")
        # Different line, but the second rhyme would reuse J.
        self.assertIsNotNone(extension_rejection_reason(existing, "PQRST"[:-1] + "J", 3, rhymes))
        # No rhyme duplication; K occurs in the same position as the upper line.
        self.assertIsNotNone(extension_rejection_reason(existing, "KPQRS", 3, rhymes))
        self.assertIsNone(extension_rejection_reason(existing, "PQRST", 3, rhymes))
        # A repeated position in the first couplet is not this middle-couplet rule.
        self.assertIsNone(extension_rejection_reason(("ABCDE",), "AFGHI", 1, rhymes))

    def test_output_write_is_create_only_and_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_new_json(path, {"trial": 1, "selected": ["ABC"]})
            before = path.read_bytes()
            self.assertEqual(json.loads(before), {"trial": 1, "selected": ["ABC"]})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"trial": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_vector_language_scores_match_independent_scalar_sentence_formula(self):
        poems, _, _ = fixture()
        vocabulary = {"<UNK>": 0}
        vocabulary.update({char: index + 1 for index, char in enumerate(
            sorted(set("".join(poems[0]["lines"]))))})
        model = BidirNGram(vocabulary).fit(poems)
        # Covers a seen line, a new recombination, reversed text and UNK.
        texts = ["ABCDE", "ABPQR", "JIHGF", "ABCDX"]
        observed = score_lines(model, texts)
        for text, actual in zip(texts, observed):
            encoded = model.encode(text)
            direction_scores = []
            for direction, sequence in enumerate((encoded, list(reversed(encoded)))):
                a, b = -1, -1
                total = 0.0
                for target in sequence + [model.eos]:
                    total += model._log_probability(direction, a, b, target)
                    a, b = b, target
                direction_scores.append(total / 6)
            expected = .5 * (direction_scores[0] + direction_scores[1])
            with self.subTest(text=text):
                self.assertAlmostEqual(float(actual), expected, places=13)
        self.assertEqual(score_lines(model, []).shape, (0,))
        with self.assertRaises(ValueError):
            score_lines(model, ["ABCD"])


if __name__ == "__main__":
    unittest.main()
