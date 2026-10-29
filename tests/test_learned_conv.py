import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wordconv_lab.factor_dp import convolution_factor, path_score, solve_local_chain
from wordconv_lab.learned_conv import LearnedLocalScorer


def example_ids(length=5):
    ids = np.random.default_rng(812).integers(0, 7, (2, 3, 2, length))
    # Repeated occurrences make scatter-add errors visible in gradient checks.
    ids[:, :, 0, 0] = 2
    return ids


class LearnedLocalScorerTests(unittest.TestCase):
    def test_manual_gradients_against_finite_differences(self):
        ids, gold = example_ids(), np.array([0, 2])
        for width in (1, 3):
            model = LearnedLocalScorer(7, kernel_width=width, channels=3, seed=92)
            _, gradients = model.loss_and_gradients(ids, gold)
            coordinates = {
                "embedding": [(2, 0), (1, 4), (6, 7)],
                "horizontal_kernel": [(0, 0, 0), (width - 1, 9, 1), (width // 2, 11, 2)],
                "pair_kernel": [(0, 0, 0), (1, 10, 1)],
                "horizontal_readout": [(0,), (2,)],
                "pair_readout": [(0,), (2,)],
            }
            for name, indices in coordinates.items():
                for index in indices:
                    with self.subTest(width=width, name=name, index=index):
                        original = model.params[name][index]
                        epsilon = 1e-6
                        model.params[name][index] = original + epsilon
                        plus = model.loss_and_gradients(ids, gold)[0]
                        model.params[name][index] = original - epsilon
                        minus = model.loss_and_gradients(ids, gold)[0]
                        model.params[name][index] = original
                        numerical = (plus - minus) / (2 * epsilon)
                        self.assertAlmostEqual(gradients[name][index], numerical, delta=2e-8)

    def test_forward_equals_original_architecture_at_both_line_lengths(self):
        vocabulary = {chr(65 + n): n for n in range(7)}
        for length, width in itertools.product((5, 7), (1, 3)):
            model = LearnedLocalScorer(7, kernel_width=width, channels=4, seed=93)
            ids = example_ids(length)
            scores = model.score_batch(ids, mask_start=2)
            feature = model.feature_factory(vocabulary, length, mask_start=2)
            scalar = model.as_local_scorer()
            for batch, candidate in itertools.product(range(2), range(3)):
                x = np.array([[feature(row, col, chr(65 + int(ids[batch, candidate, row, col])))
                               for col in range(length)] for row in (0, 1)])
                self.assertAlmostEqual(scores[batch, candidate], scalar.score(x), places=13)

    def test_unknown_embedding_and_moments_remain_exact_zero(self):
        model = LearnedLocalScorer(7, seed=94)
        ids, gold = example_ids(), np.array([0, 1])
        _, gradients = model.loss_and_gradients(ids, gold)
        np.testing.assert_array_equal(gradients["embedding"][0], np.zeros(8))
        for _ in range(4):
            model.train_step(ids, gold, weight_decay=0.1)
        for values in (model.params, model.adam_m, model.adam_v):
            np.testing.assert_array_equal(values["embedding"][0], np.zeros(8))
        feature = model.feature_factory({"A": 1}, 5)
        np.testing.assert_array_equal(feature(1, 0, "字")[:8], np.zeros(8))

    def test_training_reduces_actual_listwise_loss(self):
        model = LearnedLocalScorer(7, seed=95)
        ids, gold = example_ids(), np.array([0, 2])
        before = model.loss_and_gradients(ids, gold)[0]
        for _ in range(40):
            receipt = model.train_step(ids, gold)
        after = model.loss_and_gradients(ids, gold)[0]
        self.assertLess(after, before * 0.8)
        self.assertEqual(receipt["step"], 40)
        self.assertTrue(model.as_local_scorer().trained)

    def test_save_restore_preserves_scores_and_next_optimizer_step(self):
        model = LearnedLocalScorer(7, seed=96)
        ids, gold = example_ids(7), np.array([0, 2])
        model.train_step(ids, gold)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            restored = LearnedLocalScorer.load(path)
        np.testing.assert_array_equal(model.score_batch(ids), restored.score_batch(ids))
        self.assertEqual(model.train_step(ids, gold), restored.train_step(ids, gold))
        for name in model.params:
            np.testing.assert_array_equal(model.params[name], restored.params[name])
            np.testing.assert_array_equal(model.adam_m[name], restored.adam_m[name])
            np.testing.assert_array_equal(model.adam_v[name], restored.adam_v[name])

    def test_near_matched_width_one_arm_has_same_initial_embeddings(self):
        local = LearnedLocalScorer(7, kernel_width=3, channels=12, seed=97)
        point = LearnedLocalScorer(7, kernel_width=1, channels=20, seed=97)
        np.testing.assert_array_equal(local.params["embedding"], point.params["embedding"])
        self.assertEqual(local.core_parameter_count(), 744)
        self.assertEqual(point.core_parameter_count(), 760)
        self.assertEqual(local.parameter_count() - local.trainable_parameter_count(), 8)
        self.assertEqual(local.score_batch(example_ids()).shape, point.score_batch(example_ids()).shape)

    def test_trained_width_three_still_has_exact_local_factorization(self):
        model = LearnedLocalScorer(7, seed=98)
        model.train_step(example_ids(), np.array([1, 2]))
        feature = model.feature_factory({"A": 1, "B": 2}, 5)
        scalar = model.as_local_scorer()
        domain = (("A", "A"), ("A", "B"))
        factor = convolution_factor(scalar, feature, 5)
        oracle = -math.inf
        for path in itertools.product(domain, repeat=5):
            x = np.array([[feature(row, col, path[col][row]) for col in range(5)] for row in (0, 1)])
            score = scalar.score(x)
            oracle = max(oracle, score)
            self.assertAlmostEqual(path_score(path, factor), score, places=13)
        result = solve_local_chain((domain,) * 5, factor)
        self.assertAlmostEqual(result.score, oracle, places=13)

    def test_feature_and_scalar_exports_are_frozen_snapshots(self):
        model = LearnedLocalScorer(7, seed=99)
        feature = model.feature_factory({"A": 1}, 5)
        original = feature(1, 0, "A")
        scalar = model.as_local_scorer()
        old_kernel = scalar.horizontal_kernel.copy()
        model.train_step(example_ids(), np.array([1, 2]))
        np.testing.assert_array_equal(original, feature(1, 0, "A"))
        np.testing.assert_array_equal(old_kernel, scalar.horizontal_kernel)

    def test_candidate_permutation_preserves_loss_with_remapped_gold(self):
        model = LearnedLocalScorer(7, seed=100)
        ids, gold = example_ids(), np.array([0, 2])
        permutation = np.array([2, 0, 1])
        remapped = np.array([int(np.flatnonzero(permutation == x)[0]) for x in gold])
        before, before_g = model.loss_and_gradients(ids, gold)
        after, after_g = model.loss_and_gradients(ids[:, permutation], remapped)
        self.assertAlmostEqual(before, after, places=14)
        for name in before_g:
            np.testing.assert_allclose(before_g[name], after_g[name], atol=1e-14, rtol=1e-12)

    def test_invalid_values_and_absent_gold_are_rejected(self):
        model = LearnedLocalScorer(7)
        ids = example_ids()
        for bad in (ids.astype(float), ids.astype(bool), np.full_like(ids, -1), np.full_like(ids, 7)):
            with self.assertRaises(ValueError):
                model.score_batch(bad)
        for gold in (np.array([-1, 0]), np.array([0, 3]), np.array([True, False]), np.array([0.0, 1.0])):
            with self.assertRaises(ValueError):
                model.loss_and_gradients(ids, gold)
        with self.assertRaises(ValueError):
            model.score_batch(ids, mask_start=4)
        model.params["pair_readout"][0] = math.nan
        with self.assertRaises(ValueError):
            model.score_batch(ids)

    def test_failed_optimizer_update_is_atomic(self):
        model = LearnedLocalScorer(7)
        old = {name: value.copy() for name, value in model.params.items()}
        with self.assertRaisesRegex(ValueError, "Adam"):
            model.train_step(example_ids(), np.array([0, 2]), learning_rate=1e308, weight_decay=1e308)
        self.assertEqual(model.step_count, 0)
        for name in old:
            np.testing.assert_array_equal(old[name], model.params[name])
            np.testing.assert_array_equal(model.adam_m[name], np.zeros_like(old[name]))
            np.testing.assert_array_equal(model.adam_v[name], np.zeros_like(old[name]))


if __name__ == "__main__":
    unittest.main()
