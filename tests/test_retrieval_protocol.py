"""Independent denominator, selection and evidence-failure oracles."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retrieval_protocol_v04', ROOT/'scripts/run_retrieval_v04.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def row(family, ranks, length=5):
    return {'family_id': family, 'poem_id': family, 'length': length,
            'methods': {key: {'target_rank': rank, 'candidate_count': 100} for key, rank in ranks.items()}}


class RetrievalProtocolTests(unittest.TestCase):
    def test_selection_is_family_equal_not_task_equal(self):
        rows = [row('small', {'a': 1, 'b': None})]
        rows += [row('large', {'a': None, 'b': 1 if i < 5 else None}) for i in range(8)]
        methods = [{'id': 'a'}, {'id': 'b'}]
        selected, _ = protocol.select_method(rows, methods)
        self.assertEqual(selected['id'], 'a')
        self.assertEqual(protocol.exact_family_recall(rows, 'a'), Fraction(1, 2))
        self.assertEqual(protocol.exact_family_recall(rows, 'b'), Fraction(5, 16))

    def test_exact_tie_keeps_first_grid_entry(self):
        rows = [row('one', {'a': 1, 'b': None}), row('two', {'a': None, 'b': 16})]
        selected, _ = protocol.select_method(rows, [{'id': 'b'}, {'id': 'a'}])
        self.assertEqual(selected['id'], 'b')

    def test_oov_and_outside_k_remain_zero_in_full_denominator(self):
        rows = [row('f1', {'a': 2}), row('f1', {'a': None}), row('f2', {'a': 20})]
        report = protocol.summarize(rows, [{'id': 'a'}])['a']['overall']
        self.assertEqual(report['tasks'], 3)
        self.assertEqual(report['families'], 2)
        self.assertAlmostEqual(report['recall_at_16']['family_macro'], .25)
        self.assertAlmostEqual(report['recall_at_16']['task_micro'], 1/3)
        self.assertAlmostEqual(report['mrr_full_lexicon']['family_macro'], .15)
        self.assertAlmostEqual(report['mrr_full_lexicon']['task_micro'], .55/3)
        self.assertAlmostEqual(report['vocabulary_coverage']['family_macro'], .75)

    def test_narrow_pool_membership_and_exact_lexical_tie(self):
        parts = {'candidates': ('aa', 'ab', 'ac'), 'ngram_scores': np.array([1., 1., 9.]),
                 'alignment_bonus': np.zeros(3), 'boundary_bonus': np.zeros(3),
                 'narrow_mask': np.array([True, True, False])}
        _, _, order = protocol.score_ranking(parts, protocol.narrow_method())
        self.assertEqual(order.tolist(), [0, 1])
        parts['ngram_scores'][0] = float('nan')
        with self.assertRaises(ValueError):
            protocol.score_ranking(parts, protocol.narrow_method())

    def test_completed_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen.json'
            protocol.write_new(path, {'selected': 1})
            with self.assertRaises(ValueError):
                protocol.write_new(path, {'selected': 2})
            self.assertEqual(json.loads(path.read_text()), {'selected': 1})

    def test_frozen_guard_rejects_one_byte_input_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.py'; source.write_text('x=1\n')
            data = root/'input.json'; data.write_text('{}\n')
            config = root/'config.json'; config.write_text('{}\n')
            frozen = {'source_sha256': {'source.py': protocol.sha(source)},
                      'input_sha256': {'input.json': protocol.sha(data)},
                      'config_sha256': protocol.digest({})}
            with patch.object(protocol, 'ROOT', root), patch.object(protocol, 'CONFIG_PATH', config):
                protocol.check_frozen(frozen)
                data.write_text('{ }\n')
                with self.assertRaises(ValueError):
                    protocol.check_frozen(frozen)


if __name__ == '__main__':
    unittest.main()
