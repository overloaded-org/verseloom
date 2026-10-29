from __future__ import annotations

import copy
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wordconv_lab.audit import assert_same_candidates, audit_records, candidate_set_hash
from wordconv_lab.constraints import check_lines
from wordconv_lab.cli import main as cli_main
from wordconv_lab.convolution import LocalConvScorer, horizontal_convolution
from wordconv_lab.evaluation import import_blind_preferences, paired_cluster_bootstrap, write_blind_review
from wordconv_lab.experiment import run_reranking


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "fixtures/synthetic_couplet.json").read_text())


def constraint_for(candidate, task=None, lexicon=None):
    data = fixture()
    task = data["task"] if task is None else task
    lexicon = data["lexicon"] if lexicon is None else lexicon
    return check_lines([task["upper"], candidate["tokens"]], task["patterns"],
                       [task["upper_readings"], candidate["reading_ids"]], lexicon,
                       task["rhyme_positions"], task["rhyme_group"])


class ConvolutionTests(unittest.TestCase):
    def test_horizontal_boundary_never_reads_next_line(self):
        x = np.array([[[1.], [2.], [3.]], [[100.], [200.], [300.]]])
        result = horizontal_convolution(x, np.ones((3, 1, 1)))
        self.assertEqual(result[0, -1, 0], 5.)
        self.assertEqual(result[1, 0, 0], 300.)

    def test_kernel_preserves_order_not_bag_of_words(self):
        x = np.array([[[1.], [2.], [3.]]])
        kernel = np.array([[[1.]], [[2.]], [[4.]]])
        self.assertEqual(horizontal_convolution(x, kernel)[0, 1, 0], 17.)
        self.assertNotEqual(horizontal_convolution(x, kernel)[0, 1, 0],
                            horizontal_convolution(x[:, ::-1], kernel)[0, 1, 0])

    def test_seed_reproduces_score_and_parts(self):
        x = np.arange(40).reshape(2, 5, 4) / 10
        a, b = LocalConvScorer(4, seed=18), LocalConvScorer(4, seed=18)
        self.assertEqual(a.score_parts(x), b.score_parts(x))
        self.assertFalse(a.trained)

    def test_nonfinite_and_shape_fail_closed(self):
        scorer = LocalConvScorer(4)
        with self.assertRaises(ValueError): scorer.score(np.zeros((4, 5, 4)))
        with self.assertRaises(ValueError): scorer.score(np.zeros((2, 5, 3)))
        with self.assertRaises(ValueError): scorer.score(np.full((2, 5, 4), np.nan))
        with self.assertRaises(ValueError): horizontal_convolution(np.ones((2, 5, 1)), np.ones((2, 1, 1)))


class ConstraintTests(unittest.TestCase):
    def test_valid_fixture_is_not_aesthetic_judgment(self):
        result = constraint_for(fixture()["task"]["candidates"][0])
        self.assertEqual(result.status, "valid")
        self.assertIn("not_evaluated", result.antithesis_status)

    def test_tone_and_rhyme_cannot_be_spliced_across_readings(self):
        candidate = fixture()["task"]["candidates"][3]
        p_result = constraint_for(candidate)
        self.assertEqual(p_result.status, "invalid")
        self.assertTrue(any("rhyme_mismatch" in issue for issue in p_result.violations))
        candidate["reading_ids"][-1] = "poly-z"
        z_result = constraint_for(candidate)
        self.assertEqual(z_result.status, "invalid")
        self.assertTrue(any("tone_mismatch" in issue for issue in z_result.violations))

    def test_ambiguous_without_contextual_choice_is_review(self):
        candidate = fixture()["task"]["candidates"][3]
        candidate["reading_ids"][-1] = None
        result = constraint_for(candidate)
        self.assertEqual(result.status, "needs_review")
        self.assertTrue(any("ambiguous_reading" in issue for issue in result.review_reasons))

    def test_unknown_never_silently_gets_tone(self):
        result = constraint_for(fixture()["task"]["candidates"][4])
        self.assertEqual(result.status, "needs_review")

    def test_unverified_provenance_is_review(self):
        data = fixture()
        data["lexicon"]["A"][0]["status"] = "unreviewed"
        result = constraint_for(data["task"]["candidates"][0], lexicon=data["lexicon"])
        self.assertEqual(result.status, "needs_review")

    def test_one_global_rhyme_group_across_multiple_positions(self):
        lexicon = fixture()["lexicon"]
        result = check_lines([["RA"], ["WRONG"]], ["P", "P"], [["RA-1"], ["WRONG-1"]],
                             lexicon, [(0, 0), (1, 0)], "R1")
        self.assertEqual(result.status, "invalid")
        self.assertEqual(len(result.violations), 1)

    def test_invalid_dimensions_or_rhyme_indices_are_errors(self):
        lexicon = fixture()["lexicon"]
        with self.assertRaises(ValueError):
            check_lines([["RA"]], ["PP"], [["RA-1"]], lexicon, [], "R1")
        with self.assertRaises(ValueError):
            check_lines([["RA"]], ["P"], [["RA-1"]], lexicon, [(1, 0)], "R1")


class ProvenanceTests(unittest.TestCase):
    def records(self):
        return [json.loads(row) for row in (ROOT / "fixtures/audit_pass.jsonl").read_text().splitlines()]

    def test_valid_split_manifest(self):
        self.assertEqual(audit_records(self.records())["status"], "passed")

    def test_whole_poem_leakage_rejected(self):
        records = self.records()
        records[1]["source_poem_id"] = records[0]["source_poem_id"]
        self.assertTrue(any("cross_split_source" in error for error in audit_records(records)["errors"]))

    def test_duplicate_cluster_leakage_rejected(self):
        records = self.records()
        records[1]["duplicate_cluster"] = records[0]["duplicate_cluster"]
        self.assertTrue(any("cross_split_cluster" in error for error in audit_records(records)["errors"]))

    def test_candidate_origin_must_resolve_to_train(self):
        records = self.records()
        records[1]["candidates"][0]["origin"]["source_poem_ids"] = [records[1]["source_poem_id"]]
        self.assertTrue(any("unverified_train_candidate_source" in error for error in audit_records(records)["errors"]))

    def test_heldout_reference_not_smuggled_into_primary_candidates(self):
        records = self.records()
        records[1]["candidates"][0]["origin"]["uses_heldout_reference"] = True
        self.assertTrue(any("heldout_reference_in_primary_pool" in error for error in audit_records(records)["errors"]))

    def test_candidate_hash_changes_with_token_or_constraint(self):
        task = fixture()["task"]
        modified = copy.deepcopy(task)
        modified["candidates"][0]["tokens"][0] = "B"
        self.assertNotEqual(candidate_set_hash(task), candidate_set_hash(modified))
        modified = copy.deepcopy(task)
        modified["rhyme_group"] = "R2"
        self.assertNotEqual(candidate_set_hash(task), candidate_set_hash(modified))

    def test_candidate_order_does_not_change_set_hash(self):
        task = fixture()["task"]
        modified = copy.deepcopy(task)
        modified["candidates"].reverse()
        self.assertEqual(candidate_set_hash(task), candidate_set_hash(modified))


class ExperimentTests(unittest.TestCase):
    def test_shared_candidates_and_filter(self):
        data = fixture()
        a = run_reranking(data["task"], data["lexicon"], model="local_conv")
        b = run_reranking(data["task"], data["lexicon"], model="pointwise")
        assert_same_candidates(a, b)
        self.assertEqual(a["valid_count"], 2)
        self.assertEqual(a["ranking"], run_reranking(data["task"], data["lexicon"])["ranking"])
        self.assertEqual([x["constraint"] for x in a["candidates"]], [x["constraint"] for x in b["candidates"]])
        self.assertFalse(a["trained"])

    def test_no_verified_candidates_abstains(self):
        data = fixture()
        data["task"]["candidates"] = data["task"]["candidates"][2:]
        result = run_reranking(data["task"], data["lexicon"])
        self.assertEqual(result["decision"], "abstain")
        self.assertIsNone(result["selected_candidate_id"])

    def test_empty_set_abstains_without_fallback(self):
        data = fixture()
        data["task"]["candidates"] = []
        self.assertEqual(run_reranking(data["task"], data["lexicon"])["decision"], "abstain")

    def test_mismatched_candidate_comparison_rejected(self):
        with self.assertRaises(ValueError):
            assert_same_candidates({"candidate_set_hash": "a"}, {"candidate_set_hash": "b"})


class EvaluationTests(unittest.TestCase):
    def write_review_rows(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def read_review_rows(self, path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_no_ratings_returns_no_data(self):
        self.assertEqual(paired_cluster_bootstrap([])["status"], "no_evaluation_data")

    def test_source_clusters_have_equal_weight(self):
        rows = [{"source_poem_id": "many", "score_a": 1, "score_b": 0}] * 100
        rows += [{"source_poem_id": "one", "score_a": 0, "score_b": 1}]
        result = paired_cluster_bootstrap(rows, repeats=1000, seed=8)
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result, paired_cluster_bootstrap(rows, repeats=1000, seed=8))

    def test_one_cluster_does_not_claim_inferential_interval(self):
        result = paired_cluster_bootstrap([{"source_poem_id": "one", "score_a": 1, "score_b": 0}])
        self.assertEqual(result["status"], "insufficient_clusters")
        self.assertNotIn("interval", result)

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap([{"source_poem_id": "one", "score_a": float("nan"), "score_b": 0}])

    def test_blank_blind_sheet_has_no_invented_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([{"task_id": "t1", "source_poem_id": "p1", "upper": "upper",
                                 "text_a": "A text", "text_b": "B text"}], sheet, key)
            self.assertEqual(import_blind_preferences(sheet, key), [])
            with sheet.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertNotIn("model", rows[0])
            self.assertEqual(rows[0]["preference"], "")

    def test_ties_import_as_half_and_duplicate_ratings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([{"task_id": "t1", "source_poem_id": "p1", "upper": "upper",
                                 "text_a": "same", "text_b": "same"}], sheet, key)
            with sheet.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
            rows[0]["rater_id"] = "human-1"; rows[0]["preference"] = "tie"
            with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            self.assertEqual(import_blind_preferences(sheet, key)[0]["score_a"], .5)
            with sheet.open("a", encoding="utf-8-sig", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(rows[0])
            with self.assertRaises(ValueError): import_blind_preferences(sheet, key)

    def test_modified_display_content_is_rejected_even_when_unrated(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([{"task_id": "t1", "source_poem_id": "p1", "upper": "upper",
                                 "text_a": "A text", "text_b": "B text"}], sheet, key)
            original = self.read_review_rows(sheet)
            for field in ("upper", "left", "right"):
                with self.subTest(field=field):
                    rows = copy.deepcopy(original)
                    rows[0][field] += " changed"
                    self.write_review_rows(sheet, rows)
                    with self.assertRaisesRegex(ValueError, "displayed content"):
                        import_blind_preferences(sheet, key)

    def test_wrong_comparison_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([{"task_id": "t1", "source_poem_id": "p1", "upper": "upper",
                                 "text_a": "A text", "text_b": "B text"}], sheet, key)
            rows = self.read_review_rows(sheet)
            rows[0]["comparison_id"] = "cmp_wrong"
            self.write_review_rows(sheet, rows)
            with self.assertRaisesRegex(ValueError, "comparison identity"):
                import_blind_preferences(sheet, key)

    def test_old_key_cannot_be_reused_after_candidate_or_model_change(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            old_sheet, old_key = directory / "old.csv", directory / "old.json"
            new_sheet, new_key = directory / "new.csv", directory / "new.json"
            original = {"task_id": "same-task", "source_poem_id": "p1", "upper": "upper",
                        "text_a": "A text", "text_b": "B text",
                        "model_a": "private-model-A-v1", "model_b": "private-model-B-v1",
                        "comparison_id": "same-caller-label"}
            write_blind_review([original], old_sheet, old_key, seed=12)
            for field, replacement in (("text_a", "new candidate"), ("model_a", "private-model-A-v2")):
                with self.subTest(field=field):
                    changed = dict(original, **{field: replacement})
                    write_blind_review([changed], new_sheet, new_key, seed=12)
                    with self.assertRaisesRegex(ValueError, "unknown review id|mismatched"):
                        import_blind_preferences(new_sheet, old_key)
                    self.assertEqual(import_blind_preferences(new_sheet, new_key), [])
            self.assertNotIn("private-model", old_sheet.read_text(encoding="utf-8-sig"))

    def test_tampered_key_orientation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([{"task_id": "t1", "source_poem_id": "p1", "upper": "upper",
                                 "text_a": "A text", "text_b": "B text"}], sheet, key)
            payload = json.loads(key.read_text())
            payload["items"][0]["left_is_a"] = not payload["items"][0]["left_is_a"]
            key.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "orientation is inconsistent"):
                import_blind_preferences(sheet, key)

    def test_legacy_unbound_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([], sheet, key)
            key.write_text(json.dumps({"seed": 0, "items": []}))
            with self.assertRaisesRegex(ValueError, "unbound review key"):
                import_blind_preferences(sheet, key)

    def test_cli_preference_mapping_is_separate_from_general_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            items = [{"task_id": "t1", "source_poem_id": "p1", "upper": "u1", "text_a": "a1", "text_b": "b1"},
                     {"task_id": "t2", "source_poem_id": "p2", "upper": "u2", "text_a": "a2", "text_b": "b2"}]
            write_blind_review(items, sheet, key)
            rows = self.read_review_rows(sheet)
            key_items = json.loads(key.read_text())["items"]
            rows[0]["preference"] = "left" if key_items[0]["left_is_a"] else "right"
            rows[1]["preference"] = "tie"
            for row in rows: row["rater_id"] = "human-1"
            self.write_review_rows(sheet, rows)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(cli_main(["evaluate", "--reviews", str(sheet), "--key", str(key)]), 0)
            report = json.loads(captured.getvalue())
            self.assertEqual(report["mean_difference"], .5)
            self.assertEqual(report["preference_score"], .75)
            self.assertEqual(report["preference_interval"], [(1 + value) / 2 for value in report["interval"]])
            general = paired_cluster_bootstrap(import_blind_preferences(sheet, key))
            self.assertNotIn("preference_score", general)

    def test_cli_empty_review_still_has_no_numeric_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            sheet, key = Path(directory) / "reviews.csv", Path(directory) / "key.json"
            write_blind_review([], sheet, key)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                cli_main(["evaluate", "--reviews", str(sheet), "--key", str(key)])
            report = json.loads(captured.getvalue())
            self.assertEqual(report["status"], "no_evaluation_data")
            self.assertNotIn("preference_score", report)


if __name__ == "__main__":
    unittest.main()
