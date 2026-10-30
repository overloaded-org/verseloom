"""Regression: never discard source aliases when closing prior-data families."""
from test_bootstrap_quality_data import m, poem
import unittest

class AliasTest(unittest.TestCase):
    def test_duplicate_body_aliases_propagate_quarantine(self):
        a, alias, c = poem(1), poem(1, author='別署', title='別題'), poem(3, author='別署', title='別題')
        rows, _, _ = m.group_candidates([a], [alias, c])
        self.assertTrue(all(p['split'] == 'quarantine_prior_relation' for p in rows))
        self.assertEqual(a['source_refs'][0]['title'], 'p1')
