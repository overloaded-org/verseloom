import itertools
import math
import unittest

import numpy as np

from wordconv_lab.factor_dp import convolution_factor, solve_local_chain
from wordconv_lab.learned_conv import LearnedLocalScorer
from wordconv_lab.span_solver import solve_initial_span


VOCAB = {"X": 1, "A": 2, "B": 3, "C": 4, "D": 5}


def encoded_pool(upper, suffix, spans, vocab=VOCAB):
    return np.asarray([[[[vocab.get(c, 0) for c in upper],
                          [vocab.get(c, 0) for c in span + suffix]]
                         for span in spans]], dtype=np.int64)


class InitialSpanSolverTests(unittest.TestCase):
    def test_pool_coupling_blocks_an_illegal_high_scoring_cross_combination(self):
        model = LearnedLocalScorer(6, channels=1, seed=600)
        for array in model.params.values():
            array[:] = 0
        model.params["embedding"][VOCAB["A"], 0] = 1
        model.params["embedding"][VOCAB["D"], 1] = 1
        model.params["horizontal_kernel"][0, 0, 0] = 1
        model.params["horizontal_kernel"][1, 1, 0] = 1
        model.params["horizontal_readout"][0] = 1
        upper, suffix, spans = "XXXXX", "XXX", ("AB", "CD")
        domains = ((("X", "A"), ("X", "C")), (("X", "B"), ("X", "D"))) + ((("X", "X"),),) * 3
        raw = convolution_factor(model.as_local_scorer(), model.feature_factory(VOCAB, 5), 5)
        unconstrained = solve_local_chain(domains, raw)
        self.assertEqual(unconstrained.path[0][1] + unconstrained.path[1][1], "AD")
        constrained = solve_initial_span(model, VOCAB, upper, suffix, spans)
        self.assertEqual(constrained.status, "optimal")
        self.assertIn(constrained.chosen_span, spans)
        self.assertLess(constrained.score, unconstrained.score)
        self.assertAlmostEqual(constrained.score, math.tanh(1) / 10, places=14)
        self.assertEqual(constrained.dp.full_combinations, 4)
        self.assertEqual(constrained.candidate_count, 2)

    def test_trained_model_matches_original_batch_enumeration(self):
        model = LearnedLocalScorer(6, seed=601)
        train_ids = np.random.default_rng(602).integers(0, 6, (2, 4, 2, 5))
        model.train_step(train_ids, np.array([1, 2]))
        for upper, suffix in (("XAXAX", "BCD"), ("XAXAXAX", "BCDXA")):
            for spans in (tuple(a + b for a, b in itertools.product("ABCD", repeat=2)),
                          ("AB", "CD", "DA", "CC")):
                original = model.score_batch(encoded_pool(upper, suffix, spans))[0]
                result = solve_initial_span(model, VOCAB, upper, suffix, spans)
                self.assertAlmostEqual(result.score, float(original.max()), places=13)
                self.assertAlmostEqual(float(original[spans.index(result.chosen_span)]), float(original.max()), places=13)
                self.assertEqual(result.selected_lines, (upper, result.chosen_span + suffix))

    def test_complete_pool_and_task_identities_are_bound(self):
        model = LearnedLocalScorer(6, seed=603)
        a = solve_initial_span(model, VOCAB, "XXXXX", "XXX", ("AB", "CD"))
        b = solve_initial_span(model, VOCAB, "XXXXX", "XXX", ("CD", "AB"))
        c = solve_initial_span(model, VOCAB, "XXXXX", "XXX", ("AB", "DA"))
        d = solve_initial_span(model, VOCAB, "XXXXX", "XXA", ("AB", "CD"))
        self.assertNotEqual(a.candidate_hash, b.candidate_hash)
        self.assertNotEqual(a.candidate_hash, c.candidate_hash)
        self.assertEqual(a.candidate_hash, d.candidate_hash)
        self.assertNotEqual(a.task_hash, d.task_hash)
        self.assertEqual(a.vocabulary_hash, b.vocabulary_hash)
        self.assertEqual(a.to_dict()["candidate_count"], 2)

    def test_unknown_characters_are_explicit_and_never_replaced_or_invented(self):
        model = LearnedLocalScorer(6, seed=604)
        spans = ("AB", "新D")
        result = solve_initial_span(model, VOCAB, "X未XXX", "XXX", spans)
        original = model.score_batch(encoded_pool("X未XXX", "XXX", spans))[0]
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.vocabulary_status, "unknown_characters_scored_as_zero_embedding")
        self.assertEqual(result.unknown_in_context, ("未",))
        self.assertEqual(result.unknown_in_candidates, ("新",))
        self.assertIn(result.chosen_span, spans)
        self.assertAlmostEqual(result.score, float(original.max()), places=13)

    def test_empty_pool_is_infeasible_and_does_not_create_a_span(self):
        result = solve_initial_span(LearnedLocalScorer(6), VOCAB, "XXXXX", "XXX", ())
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.domain_counts[:2], (0, 0))
        self.assertIsNone(result.chosen_span)
        self.assertIsNone(result.selected_lines)

    def test_invalid_shapes_duplicates_and_width_one_are_rejected(self):
        model = LearnedLocalScorer(6)
        for upper, suffix, spans in (("XXXX", "XX", ("AB",)), ("XXXXX", "XX", ("AB",)),
                                     ("XXXXX", "XXX", ("A",)), ("XXXXX", "XXX", ("AB", "AB")),
                                     ("XXXXX", "XXX", ("A ",)), ("XXXXX", "XXX", "AB")):
            with self.assertRaises(ValueError):
                solve_initial_span(model, VOCAB, upper, suffix, spans)
        with self.assertRaisesRegex(ValueError, "width-three"):
            solve_initial_span(LearnedLocalScorer(6, kernel_width=1, channels=20),
                               VOCAB, "XXXXX", "XXX", ("AB",))


if __name__ == "__main__":
    unittest.main()
