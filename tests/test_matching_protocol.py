"""Independent v0.5 denominator, selection, mismatch and identity oracles.

These tests use synthetic queries and temporary files only. They never open
the historical test set or any confirmation task/prediction/result file.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wordconv_lab.completion_v05 import complete, selected_checkpoint
from wordconv_lab.matching import ContextSpanMatcher


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "matching_protocol_v05_review", ROOT / "scripts/run_matching_v05.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def task(number, length=5, family=None, target="甲乙"):
    # Different visible queries; identical gold is intentionally permitted.
    return {
        "task_id": f"task_{length}_{number}", "poem_id": f"poem_{length}_{number}",
        "family_id": family or f"family_{length}_{number}", "split": "dev",
        "length": length, "upper_line": (chr(0x4E20 + number) + "甲乙丙丁戊己")[:length],
        "lower_suffix": "辛壬癸甲乙丙丁"[:length - 2],
        "start": 0, "mask_length": 2, "target": target,
    }


class MatchingProtocolIndependentTests(unittest.TestCase):
    def test_context_permutation_is_target_independent_one_to_one_and_order_independent(self):
        tasks = [task(i, length) for length in (5, 7) for i in range(4)]
        actual = protocol.make_context_permutation(tasks, "fixed-toy-salt")
        altered = []
        for original in reversed(tasks):
            row = {k: v for k, v in original.items() if k != "target"}
            row.update(author="changed metadata", title="ignored title")
            altered.append(row)
        self.assertEqual(actual, protocol.make_context_permutation(altered, "fixed-toy-salt"))
        self.assertFalse(actual["uses_targets"])
        self.assertTrue(actual["gold_equality_is_not_a_filter"])
        self.assertEqual({r["task_id"] for r in actual["mapping"]},
                         {r["donor_task_id"] for r in actual["mapping"]})
        index = {t["task_id"]: t for t in tasks}
        for row in actual["mapping"]:
            self.assertEqual(index[row["task_id"]]["length"], index[row["donor_task_id"]]["length"])
            self.assertNotEqual(row["family_id"], row["donor_family_id"])
            self.assertNotEqual(row["query_hash"], row["donor_query_hash"])

    def test_impossible_context_mismatch_fails_instead_of_relaxing_exclusions(self):
        same_family = [task(i, family="same") for i in range(3)]
        with self.assertRaises(ValueError):
            protocol.make_context_permutation(same_family, "fixed")
        same_query = [task(i) for i in range(3)]
        for row in same_query:
            row["upper_line"] = same_query[0]["upper_line"]
        with self.assertRaises(ValueError):
            protocol.make_context_permutation(same_query, "fixed")

    def test_schedule_pairs_arms_and_counts_real_full_updates(self):
        tasks = [task(i, length) for length in (5, 7) for i in range(5)]
        one = protocol.make_schedule(tasks, 7, 3, 1, 20261029)
        two = protocol.make_schedule(list(reversed(tasks)), 7, 3, 1, 20261029)
        self.assertEqual(one, two)
        index = {t["task_id"]: t for t in tasks}
        self.assertEqual([b["update"] for b in one], list(range(1, 8)))
        self.assertEqual(sum(len(b["task_ids"]) for b in one), 21)
        for batch in one:
            self.assertEqual(len(batch["task_ids"]), 3)
            self.assertTrue(all(index[tid]["length"] == batch["length"] for tid in batch["task_ids"]))
        accounting = protocol.schedule_accounting(one, index)
        self.assertEqual(accounting["optimizer_updates"], 7)
        self.assertEqual(accounting["sample_presentations"], 21)
        self.assertEqual(accounting["gradient_accumulation"], 1)

    def test_common_checkpoint_uses_three_seed_mean_not_best_seed_envelope(self):
        tasks = [task(i) for i in range(4)]
        early = [[1] * 4, [32] * 4, [32] * 4]
        steady = [[1, 1, 32, 32]] * 3
        chosen, values = protocol.select_shared_checkpoint(
            tasks, {1200: deepcopy(steady), 400: early, 800: steady})
        self.assertEqual(chosen, 800)
        self.assertEqual(values[400], Fraction(1, 3))
        self.assertEqual(values[800], Fraction(1, 2))
        self.assertEqual(values[800], values[1200])

    def test_checkpoint_selection_equal_weights_families_not_tasks(self):
        tasks = [task(0, family="small")] + [task(i, family="large") for i in range(1, 4)]
        first = [[1, 32, 32, 32]] * 3
        later = [[32, 1, 1, 32]] * 3
        chosen, values = protocol.select_shared_checkpoint(tasks, {400: first, 800: later})
        self.assertEqual(chosen, 400)
        self.assertEqual(values[400], Fraction(1, 2))
        self.assertEqual(values[800], Fraction(1, 3))

    def test_prediction_and_seed_metric_keep_oov_in_full_denominator(self):
        tasks = [task(0, family="f", target="甲乙"), task(1, family="f", target="未知")]
        candidates = tuple(sorted(("甲乙", "丙丁", "青山") +
                                  tuple(chr(0x4E10 + i) + "字" for i in range(32))))
        scores = np.zeros((2, len(candidates)))
        rows = protocol.prediction_rows(tasks, scores, candidates)
        self.assertEqual(rows[0]["top_candidates"], list(candidates[:16]))
        self.assertEqual(rows[0]["target_rank"], candidates.index("甲乙") + 1)
        self.assertIsNone(rows[1]["target_rank"])
        self.assertEqual(len(rows), 2)
        ranks = [[1, None], [2, None], [32, None]]
        self.assertEqual(protocol.exact_seed_family_recall(tasks, ranks), Fraction(1, 3))
        indices = {5: {s: i for i, s in enumerate(candidates)}}
        metrics = protocol.metric_bundle(tasks, ranks, indices, {5: {"甲乙": 2}})
        self.assertEqual(metrics["tasks"], 2)
        self.assertEqual(metrics["seed_count"], 3)
        self.assertEqual(metrics["by_prefix_positive_bucket"]["outside_W"]["tasks"], 1)
        self.assertEqual(metrics["by_prefix_positive_bucket"]["outside_W"]["metrics"]["recall_at_16"]["family_macro"], 0.0)

    def test_old_test_is_skipped_before_json_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poems.jsonl"
            path.write_bytes(b'{"split":"train","id":1}\n'
                             b'{"split":"test",not valid JSON and invalid UTF8 \xff}\n'
                             b'{"split":"dev","id":2}\n')
            data, skipped = protocol.read_train_dev(path, {"train": 1, "dev": 1})
            self.assertEqual(skipped, 1)
            self.assertEqual(data["train"][0]["id"], 1)
            self.assertEqual(data["dev"][0]["id"], 2)

    def test_frozen_guard_rejects_source_or_input_byte_change(self):
        config = json.loads((ROOT / "configs/matching_v0.5.json").read_text())
        config["status"] = "frozen"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.py"; source.write_bytes(b"x=1\n")
            data = base / "input.json"; data.write_bytes(b"{}\n")
            frozen = {"config": config, "config_sha256": protocol.digest(config),
                      "source_sha256": {"source.py": protocol.sha(source)},
                      "input_sha256": {"input.json": protocol.sha(data)}}
            (base / "freeze.json").write_text(json.dumps(frozen))
            with patch.object(protocol, "ROOT", base):
                protocol.check_frozen(base)
                data.write_bytes(b"{ }\n")
                with self.assertRaises(ValueError):
                    protocol.check_frozen(base)
                data.write_bytes(b"{}\n")
                source.write_bytes(b"x=2\n")
                with self.assertRaises(ValueError):
                    protocol.check_frozen(base)

    def test_evidence_publication_refuses_replacing_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            protocol.write_bytes_new(path, b"original\n")
            with self.assertRaises(FileExistsError):
                protocol.write_bytes_new(path, b"replacement\n")
            self.assertEqual(path.read_bytes(), b"original\n")

    def test_completion_rejects_valid_checkpoint_with_wrong_selected_update(self):
        chars = sorted(set("甲乙丙丁戊辛壬癸"))
        vocabulary = {c: i for i, c in enumerate(["<UNK>"] + chars)}
        model = ContextSpanMatcher(vocabulary, {5: ("甲乙", "乙甲")},
                                   embedding_dim=2, representation_dim=3, channels=2, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "models" / "update_0001.npz"
            model.save(checkpoint)  # Internally this is still update zero.
            selection = {"primary_arm": "full_w3", "selection_scope": "synthetic_identity_test",
                         "arms": {"full_w3": {"step": 1, "models": [{"seed": 0,
                             "path": "models/update_0001.npz",
                             "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}}}
            (run / "selection_frozen.json").write_text(json.dumps(selection))
            selected_checkpoint(run, 0)  # Bytes match; internal update must still be checked.
            with self.assertRaisesRegex(ValueError, "optimizer step"):
                complete("甲乙丙丁戊", "辛壬癸", method="matching", run=run)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                selected_checkpoint(run, 0)


if __name__ == "__main__":
    unittest.main()
