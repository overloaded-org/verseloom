from dataclasses import replace
import itertools
import math
import unittest

import numpy as np

from wordconv_lab.convolution import LocalConvScorer
from wordconv_lab.prosody import generate_template
from wordconv_lab.verse_solver import ReadingCandidate, solve_verse


def fixture(length=5):
    template = generate_template(length, "P", False)
    candidates = [[(ReadingCandidate("A", tone, "R1", f"r{row}c{col}", True),)
                   for col, tone in enumerate(line)] for row, line in enumerate(template.core)]
    return template, candidates


def feature(row, col, token):
    return np.array([float(token == "A"), float(token == "B"), row / 7, col / 6])


def direct_score(grid, scorer, feature_fn=feature):
    return sum(scorer.score(np.array([[feature_fn(upper + row, col, c.token)
                                      for col, c in enumerate(grid[upper + row])]
                                     for row in (0, 1)])) for upper in (0, 2, 4, 6))


class FullVerseSolverTests(unittest.TestCase):
    def test_complete_matrix_matches_independent_original_model_enumeration(self):
        template, candidates = fixture()
        for row, col in ((0, 0), (1, 4), (7, 0)):
            candidates[row][col] += (replace(candidates[row][col][0], token="B"),)
        candidates[2][0] += (replace(candidates[2][0][0], token="C", tone="P"),)
        model = LocalConvScorer(4, seed=803)
        scores = []
        for flat in itertools.product(*(domain for row in candidates for domain in row)):
            grid = tuple(tuple(flat[row * 5:(row + 1) * 5]) for row in range(8))
            if (all(c.tone == template.core[row][col] for row in range(8)
                    for col, c in enumerate(grid[row]))
                    and all(grid[row][-1].rhyme == "R1" for row in template.rhyme_rows)):
                scores.append(direct_score(grid, model))
        result = solve_verse(candidates, (template,), ("R1",), model, feature)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.full_combinations, 16)
        self.assertEqual(result.feasible_combinations, len(scores))
        self.assertAlmostEqual(result.score, max(scores), places=12)
        self.assertAlmostEqual(direct_score(result.selected, model), max(scores), places=12)
        self.assertEqual(len(result.lines), 8)
        self.assertTrue(all(len(line) == 5 for line in result.lines))
        self.assertEqual(len(result.configurations[0].couplets), 4)

    def test_seven_character_boundaries_and_all_four_configurations(self):
        template, candidates = fixture(7)
        first_rhymed = generate_template(7, "P", True)
        for col, (a, b) in enumerate(zip(template.core[0], first_rhymed.core[0])):
            if a != b:
                candidates[0][col] += (replace(candidates[0][col][0], token="B", tone=b),)
        result = solve_verse(candidates, (first_rhymed, template), ("R2", "R1"),
                             LocalConvScorer(4, seed=804), feature)
        self.assertEqual(result.status, "optimal")
        self.assertTrue(all(len(line) == 7 for line in result.lines))
        self.assertEqual(len(result.configurations), 4)
        self.assertEqual([b.status for b in result.configurations].count("infeasible"), 2)
        self.assertEqual(result.feasible_combinations, 2)
        chosen = next(t for t in (template, first_rhymed) if t.template_id == result.selected_template)
        self.assertEqual(tuple("".join(c.tone for c in row) for row in result.selected), chosen.core)
        self.assertTrue(all(result.selected[row][-1].rhyme == result.selected_rhyme
                            for row in chosen.rhyme_rows))

    def test_ambiguous_reading_is_not_resolved_by_template_filter(self):
        template, candidates = fixture()
        candidates[0][0] += (replace(candidates[0][0][0], tone="Z", reading_id="other-reading"),)
        result = solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4),
                             lambda *args: self.fail("must review before scoring or filtering"))
        self.assertEqual(result.status, "needs_reading_review")
        self.assertIsNone(result.feasible_combinations)
        self.assertEqual(result.issues[0].reason, "ambiguous_context_reading")
        self.assertEqual(result.configurations, ())

    def test_unknown_or_unverified_unused_candidate_rejects_entire_task(self):
        for field, value in (("tone", "?"), ("rhyme", None), ("reading_id", "unknown"),
                             ("verified", False)):
            with self.subTest(field=field):
                template, candidates = fixture()
                # Z would be filtered out at this P position, but cannot be hidden.
                extra = replace(candidates[0][0][0], token="B", tone="Z", **{field: value}) if field != "tone" else replace(candidates[0][0][0], token="B", tone=value)
                candidates[0][0] += (extra,)
                result = solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4),
                                     lambda *args: self.fail("unreviewed records cannot reach scoring"))
                self.assertEqual(result.status, "needs_reading_review")

    def test_empty_domain_is_explicit_infeasible_not_a_partial_poem(self):
        template, candidates = fixture()
        candidates[6][2] = ()
        result = solve_verse(candidates, (template,), ("R1", "R2"), LocalConvScorer(4), feature)
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.full_combinations, 0)
        self.assertEqual(result.feasible_combinations, 0)
        self.assertIsNone(result.selected)
        self.assertTrue(all((6, 2) in b.missing_positions for b in result.configurations))

    def test_relaxed_or_incomplete_templates_and_grids_are_rejected(self):
        template, candidates = fixture()
        relaxed = replace(template, core=("Z" + template.core[0][1:],) + template.core[1:])
        for templates, grid in (((relaxed,), candidates), ((template,), candidates[:7]),
                                ((replace(template, rhyme_rows=(1, 3)),), candidates),
                                ((template, template), candidates)):
            with self.assertRaises(ValueError):
                solve_verse(grid, templates, ("R1",), LocalConvScorer(4), feature)

    def test_duplicate_reading_or_multicharacter_token_is_rejected(self):
        template, candidates = fixture()
        candidates[0][0] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4), feature)
        candidates[0][0] = (replace(candidates[0][0][0], token="AB"),)
        with self.assertRaisesRegex(ValueError, "code point"):
            solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4), feature)

    def test_parameters_and_all_input_features_are_checked(self):
        template, candidates = fixture()
        for value in (math.nan, math.inf, -math.inf):
            model = LocalConvScorer(4)
            model.pair_readout[0] = value
            with self.assertRaises(ValueError):
                solve_verse(candidates, (template,), ("R1",), model, feature)
        with self.assertRaises(ValueError):
            solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4, kernel_width=5), feature)
        candidates[0][0] += (replace(candidates[0][0][0], token="C", tone="Z"),)
        with self.assertRaisesRegex(ValueError, "finite fixed feature"):
            solve_verse(candidates, (template,), ("R1",), LocalConvScorer(4),
                         lambda row, col, token: np.full(4, math.nan) if token == "C" else feature(row, col, token))

    def test_feature_vectors_and_parameters_are_snapshotted_before_search(self):
        template, candidates = fixture()
        model = LocalConvScorer(4, seed=805)
        original = LocalConvScorer(4, seed=805)
        buffer = np.zeros(4)
        calls = []
        def mutable_feature(row, col, token):
            calls.append((row, col, token))
            model.pair_readout[:] = 100
            buffer[:] = feature(row, col, token)
            return buffer
        result = solve_verse(candidates, (template,), ("R1", "R2"), model, mutable_feature)
        buffer[:] = -999
        self.assertEqual(len(calls), 40)
        self.assertEqual(result.frozen_feature_count, 40)
        self.assertAlmostEqual(result.score, direct_score(result.selected, original), places=12)

    def test_arithmetic_and_whole_verse_overflow_are_not_reported_as_optima(self):
        template, candidates = fixture()
        for sign in (1, -1):
            model = LocalConvScorer(1, channels=1)
            model.horizontal_kernel[:] = 0
            model.horizontal_readout[:] = 0
            model.pair_kernel[:] = 1
            model.pair_readout[:] = sign * 6e307
            with self.assertRaisesRegex(ValueError, "whole-verse.*overflowed"):
                solve_verse(candidates, (template,), ("R1",), model, lambda *args: np.ones(1))
        model.pair_kernel[:] = 1e308
        with self.assertRaisesRegex(ValueError, "non-finite floating-point"):
            solve_verse(candidates, (template,), ("R1",), model, lambda *args: np.full(1, 1e308))

    def test_empty_or_unknown_rhyme_configuration_is_invalid(self):
        template, candidates = fixture()
        for groups in ((), ("?",), ("R1", "R1")):
            with self.assertRaises(ValueError):
                solve_verse(candidates, (template,), groups, LocalConvScorer(4), feature)


if __name__ == "__main__":
    unittest.main()
