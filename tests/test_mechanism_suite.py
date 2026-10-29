import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wordconv_lab.mechanism import (
    DEFAULT_CONFIG, compile_case, exact_audit, generate_cases,
    normalized_regret, paired_summary, run_suite,
)
from wordconv_lab.matrix import organize


def small_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    # Unit fixtures must not expose any of the formal run's task payloads.
    config["master_seed"] = 99887766
    config["split_sizes"] = {"dev": 4, "test": 4}
    config["proposal_budgets"] = [4]
    config["primary_budget"] = 4
    config["bootstrap_repeats"] = 100
    return config


class MechanismSuiteTests(unittest.TestCase):
    def test_case_generation_is_reproducible_and_split_ids_distinct(self):
        a = generate_cases(small_config())
        b = generate_cases(small_config())
        self.assertEqual(a, b)
        self.assertEqual(len({x["case_sha256"] for x in a}), 8)
        self.assertEqual(len({x["problem_id"] for x in a}), 8)
        self.assertEqual({x["family"] for x in a}, set(DEFAULT_CONFIG["families"]))

    def test_case_tampering_rejected(self):
        case = generate_cases(small_config())[0]
        case["unary"][0][0] += 1
        with self.assertRaises(ValueError):
            compile_case(case)

    def test_full_grid_bounded_space_and_feasible_connectivity(self):
        for case in generate_cases(small_config()):
            problem, rule, score = compile_case(case)
            self.assertEqual(problem.rows, 8)
            self.assertIn(problem.columns, (5, 7))
            audit = exact_audit(problem, rule, score)
            self.assertLessEqual(audit["total_states"], 625)
            self.assertTrue(audit["connected_from_initial"])
            self.assertTrue(rule(problem.initial)[0])

    def test_joint_constraint_has_same_auditable_joint_move(self):
        case = next(c for c in generate_cases(small_config()) if c["family"] == "coupled_constraint")
        problem, rule, score = compile_case(case)
        linked = tuple(sorted(case["positions"][:2]))
        self.assertEqual(problem.linked_blocks, (linked,))
        self.assertTrue(any(not rule(candidate)[0] for block, candidate in problem.neighbors(problem.initial) if len(block) == 1))
        self.assertTrue(any(rule(candidate)[0] for block, candidate in problem.neighbors(problem.initial) if len(block) == 2))

    def test_normalized_regret_handles_constant_without_fake_denominator(self):
        self.assertIsNone(normalized_regret(2.0, {"global_max": 2.0, "global_min": 2.0}))
        self.assertAlmostEqual(normalized_regret(4.0, {"global_max": 5.0, "global_min": 1.0}), 0.25)
        with self.assertRaises(ValueError):
            normalized_regret(6.0, {"global_max": 5.0, "global_min": 1.0})

    def test_summary_rejects_unpaired_budgets_or_repeated_problem(self):
        a = {"problem_id": "x", "policy": "fifo", "case_sha256": "a", "problem_hash": "b",
             "budget": 32, "family": "separable", "split": "test", "normalized_regret": 0.5}
        b = {**a, "policy": "priority", "budget": 64, "normalized_regret": 0.2}
        kwargs = dict(bootstrap_seed=1, repeats=100, confidence=.95, relative_gate=.2, family_harm_gate=.05)
        with self.assertRaises(ValueError):
            paired_summary([a, b], **kwargs)
        b["budget"] = 32
        with self.assertRaises(ValueError):
            paired_summary([a, b, a], **kwargs)

    def test_harm_guardrail_can_fail_despite_aggregate_improvement(self):
        rows = []
        for i, family in enumerate(DEFAULT_CONFIG["families"]):
            for j in range(3):
                common = {"problem_id": f"{i}-{j}", "case_sha256": f"c{i}-{j}", "problem_hash": f"p{i}-{j}",
                          "budget": 32, "family": family, "split": "test"}
                rows.append({**common, "policy": "fifo", "normalized_regret": .8 if i < 3 else .1})
                rows.append({**common, "policy": "priority", "normalized_regret": .1 if i < 3 else .3})
        result = paired_summary(rows, bootstrap_seed=1, repeats=100, confidence=.95, relative_gate=.2, family_harm_gate=.05)
        self.assertGreater(result["relative_mean_regret_reduction"], .2)
        self.assertEqual(result["decision"], "DO_NOT_PROMOTE_PRIORITY_AS_DEFAULT")
        self.assertEqual(result["harmed_families"], ["local_trap"])

    def test_manifest_is_frozen_before_solver_and_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite"
            def checked_organize(*args, **kwargs):
                manifest = json.loads((path / "manifest.json").read_text())
                self.assertEqual(manifest["stage"], "frozen_before_first_evaluation")
                self.assertTrue((path / "cases.jsonl").exists())
                self.assertIn("cases_sha256", manifest)
                return organize(*args, **kwargs)
            with patch("wordconv_lab.mechanism.organize", side_effect=checked_organize):
                report = run_suite(path, small_config())
            self.assertEqual(report["total_runs"], 16)
            self.assertEqual(report["replayed_runs"], 16)
            self.assertFalse(report["human_evaluated"])
            with self.assertRaises(FileExistsError):
                run_suite(path, small_config())


if __name__ == "__main__":
    unittest.main()
