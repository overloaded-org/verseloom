import copy
import math
import unittest

from wordconv_lab.matrix import MatrixProblem, exhaustive_oracle, organize, replay, validate_trace


def feasible(state):
    return (state[2] == state[5], () if state[2] == state[5] else ("shared_synthetic_rhyme",))


def score(state):
    bits = [int(x) for x in state]
    # A valley makes retaining alternative frontier branches meaningful.
    return float(sum(bits) + 3 * (sum(bits) == 6) - 2 * (sum(bits) == 1))


class MatrixRiskTests(unittest.TestCase):
    def setUp(self):
        self.problem = MatrixProblem("synthetic-six", 2, 3, (("0", "1"),) * 6,
                                     ("0",) * 6, ((2, 5),))

    def test_exhaustive_oracle_and_both_policies_reach_it(self):
        oracle = exhaustive_oracle(self.problem, feasible, score)
        self.assertEqual(oracle["total_states"], 64)
        self.assertEqual(oracle["feasible_states"], 32)
        for policy in ("fifo", "priority"):
            run = organize(self.problem, feasible, score, scorer_id="valley-v1", strategy=policy,
                           max_proposals=1000, max_score_calls=100)
            self.assertEqual(run.best_score, oracle["best_score"])
            self.assertEqual(run.score_calls, 32)
            self.assertEqual(run.stop_reason, "frontier_exhausted")

    def test_replay_recomputes_every_score(self):
        run = organize(self.problem, feasible, score, scorer_id="valley-v1", max_proposals=25)
        replayed = replay(self.problem, feasible, score, scorer_id="valley-v1", trace=run.trace)
        self.assertEqual(run.trace, replayed.trace)
        with self.assertRaises(ValueError):
            replay(self.problem, feasible, lambda s: score(s) + 1,
                   scorer_id="valley-v1", trace=run.trace)

    def test_tampered_trace_rejected(self):
        run = organize(self.problem, feasible, score, scorer_id="valley-v1")
        altered = copy.deepcopy(run.trace)
        altered[-1]["best_score"] = 999
        with self.assertRaises(ValueError):
            validate_trace(altered)

    def test_budget_charges_duplicates_and_constraint_rejections(self):
        run = organize(self.problem, feasible, score, scorer_id="valley-v1", max_proposals=30)
        attempts = [x for x in run.trace if x["kind"] in {"enqueue", "reject_duplicate", "reject_constraint"}]
        self.assertEqual(len(attempts), run.proposals)
        self.assertEqual(run.proposals, 30)
        self.assertGreater(sum(x["kind"] == "reject_constraint" for x in attempts), 0)
        self.assertGreater(sum(x["kind"] == "reject_duplicate" for x in attempts), 0)
        self.assertTrue(feasible(run.best_state)[0])
        self.assertLessEqual(run.score_calls, 31)

    def test_score_budget_and_zero_proposals_stop(self):
        run = organize(self.problem, feasible, score, scorer_id="valley-v1", max_proposals=0)
        self.assertEqual(run.score_calls, 1)
        self.assertEqual(run.proposals, 0)
        self.assertEqual(run.best_state, self.problem.initial)
        run = organize(self.problem, feasible, score, scorer_id="valley-v1", max_score_calls=3)
        self.assertEqual(run.score_calls, 3)

    def test_budget_cutoff_empty_frontier_is_not_exhaustive_certificate(self):
        problem = MatrixProblem("last-frontier-cutoff", 1, 2, (("0", "1"),) * 2, ("0", "0"))
        rule = lambda s: (s[0] == "0", () if s[0] == "0" else ("first_position_fixed",))
        run = organize(problem, rule, lambda s: float(s[1] == "1"), scorer_id="x", max_proposals=1)
        self.assertEqual(run.stop_reason, "proposal_budget_exhausted")
        self.assertEqual(run.best_score, 0.0)
        self.assertEqual(exhaustive_oracle(problem, rule, lambda s: float(s[1] == "1"))["best_score"], 1.0)

    def test_initial_invalid_and_nonfinite_fail_closed(self):
        bad = MatrixProblem("bad", 2, 3, (("0", "1"),) * 6, ("0", "0", "1", "0", "0", "0"))
        with self.assertRaises(ValueError):
            organize(bad, feasible, score, scorer_id="x")
        with self.assertRaises(ValueError):
            organize(self.problem, feasible, lambda _: math.nan, scorer_id="x")

    def test_fractional_or_boolean_budget_cannot_overspend(self):
        for value in [1.5, True, False, "10", float("inf")]:
            with self.assertRaises(ValueError):
                organize(self.problem, feasible, score, scorer_id="x", max_proposals=value)
            with self.assertRaises(ValueError):
                organize(self.problem, feasible, score, scorer_id="x", max_score_calls=value)

    def test_same_move_set_is_visible_to_both_policies(self):
        outputs = [organize(self.problem, feasible, score, scorer_id="x", strategy=p,
                            max_proposals=7, max_score_calls=100) for p in ("fifo", "priority")]
        for run in outputs:
            states = [tuple(e["state"]) for e in run.trace if e["kind"] in {"enqueue", "reject_constraint"}]
            self.assertEqual(states, [s for _, s in self.problem.neighbors(self.problem.initial)])

    def test_linked_move_allows_reaching_other_rhyme_without_invalid_best(self):
        run = organize(self.problem, feasible, score, scorer_id="x", max_proposals=1000, max_score_calls=100)
        self.assertEqual(run.best_state[2], "1")
        self.assertTrue(all(feasible(s)[0] for s in run.best_path))
        self.assertTrue(any(sum(a != b for a, b in zip(x, y)) == 2
                            for x, y in zip(run.best_path, run.best_path[1:])))

    def test_frontier_exhaustion_is_only_reachable_component(self):
        problem = MatrixProblem("disconnected", 1, 2, (("0", "1"),) * 2, ("0", "0"))
        rule = lambda s: (s[0] == s[1], () if s[0] == s[1] else ("pair_must_match",))
        scorer = lambda s: float(sum(int(x) for x in s))
        result = organize(problem, rule, scorer, scorer_id="disconnected")
        self.assertEqual(result.stop_reason, "frontier_exhausted")
        self.assertEqual(result.best_score, 0.0)
        self.assertEqual(exhaustive_oracle(problem, rule, scorer)["best_score"], 2.0)

    def test_full_eight_line_grid_and_bad_shapes(self):
        for length in (5, 7):
            p = MatrixProblem(f"8x{length}", 8, length, (("A",),) * (8 * length), ("A",) * (8 * length))
            run = organize(p, lambda _: (True, ()), lambda _: 0.0, scorer_id="constant")
            self.assertEqual(run.stop_reason, "frontier_exhausted")
        bad = MatrixProblem("badshape", 8, 7, (("A",),) * 40, ("A",) * 40)
        with self.assertRaises(ValueError):
            bad.validate()


if __name__ == "__main__":
    unittest.main()
