import unittest

from wordconv_lab.audit import assert_same_candidates


class LexiconIdentityTests(unittest.TestCase):
    def test_same_candidates_with_different_reading_table_is_not_fair(self):
        a = {"candidate_set_hash": "same-candidates", "constraint_lexicon_hash": "dictionary-v1"}
        b = {"candidate_set_hash": "same-candidates", "constraint_lexicon_hash": "dictionary-v2"}
        with self.assertRaises(ValueError):
            assert_same_candidates(a, b)

    def test_dictionary_identity_cannot_be_dropped_on_one_arm(self):
        a = {"candidate_set_hash": "same-candidates", "constraint_lexicon_hash": "dictionary-v1"}
        b = {"candidate_set_hash": "same-candidates"}
        with self.assertRaises(ValueError):
            assert_same_candidates(a, b)
