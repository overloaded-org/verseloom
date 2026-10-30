"""Synthetic boundary tests; no literary-quality certification."""
import importlib.util
import itertools
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bootstrap', ROOT / 'scripts/bootstrap_quality_data.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def poem(number, length=5, author=None, title=None, lines=None):
    if lines is None:
        chars = [chr(0x4e00 + number * 64 + j) for j in range(8 * length)]
        lines = [''.join(chars[i:i + length]) for i in range(0, len(chars), length)]
    pid = 'p' + str(number)
    return dict(poem_id=pid, length=length, lines=lines, author=author or pid,
                title=title or pid, source_refs=[dict(source_id=pid, author=author or pid, title=title or pid)])


class JoinTests(unittest.TestCase):
    def test_prefix_join_matches_brute_force(self):
        rng = random.Random(20261030)
        for _ in range(80):
            sets = [set(rng.sample(range(30), rng.randint(5, 25))) for _ in range(20)]
            # Closely related and identical sets exercise the high threshold.
            sets += [sets[0].copy(), sets[1] | {33}, (sets[2] - {4}) | {34}]
            expected = {(a, b) for a, b in itertools.combinations(range(len(sets)), 2)
                        if 5 * len(sets[a] & sets[b]) >= 4 * len(sets[a] | sets[b])}
            self.assertEqual(set(m.near_pairs(sets)), expected)

    def test_transitive_prior_contamination(self):
        a, b, c = poem(1), poem(2), poem(3)
        b['lines'][0] = a['lines'][0]
        c['lines'][1] = b['lines'][1]
        rows, _, _ = m.group_candidates([a], [b, c])
        self.assertTrue(all(p['split'] == 'quarantine_prior_relation' for p in rows))
        self.assertEqual(rows[0]['family_id'], rows[1]['family_id'])

    def test_near_duplicate_without_shared_line(self):
        a = poem(1, 7)
        b = poem(2, 7, lines=[chr(0x6000 + i) + line[1:] for i, line in enumerate(a['lines'])])
        # One substitution in EACH line has Jaccard below threshold; check test oracle.
        sets = [{line[i:i + 2] for line in p['lines'] for i in range(6)} for p in [a, b]]
        expected = 5 * len(sets[0] & sets[1]) >= 4 * len(sets[0] | sets[1])
        self.assertEqual((0, 1) in set(m.near_pairs(sets)), expected)

    def test_identical_poem_is_quarantined(self):
        a = poem(1)
        rows, _, _ = m.group_candidates([a], [a.copy()])
        self.assertEqual(rows[0]['split'], 'quarantine_prior_relation')

    def test_same_id_different_body_rejected(self):
        a, b = poem(1), poem(2); b['poem_id'] = a['poem_id']
        with self.assertRaises(ValueError): m.group_candidates([a], [b])

    def test_author_title_and_source_id_are_conservative_edges(self):
        a, b, c = poem(1, author='甲', title='乙'), poem(2, author='甲', title='乙'), poem(3)
        c['source_refs'][0]['source_id'] = 'p2'
        rows, _, _ = m.group_candidates([a], [b, c])
        self.assertTrue(all(p['split'] == 'quarantine_prior_relation' for p in rows))

    def test_enumeration_does_not_change_splits(self):
        old, new = [poem(1)], [poem(i) for i in range(2, 60)]
        a = m.group_candidates(old, new)[0]
        b = m.group_candidates(old, list(reversed(new)))[0]
        key = lambda rows: {p['poem_id']: (p['family_id'], p['split']) for p in rows}
        self.assertEqual(key(a), key(b))

    def test_seed_has_only_train_families_and_both_forms(self):
        rows = []
        for n in (5, 7):
            for i in range(10, 20):
                p = poem(i + n * 50, n)
                p.update(family_id=p['poem_id'], split='train_candidate' if i % 2 else 'test_reserve')
                rows.append(p)
        selected = m.select_seed(rows, 3)
        self.assertEqual(len(selected), 6)
        self.assertEqual({p['length'] for p in selected}, {5, 7})
        self.assertTrue(all(p['split'] == 'train_candidate' for p in selected))
        self.assertEqual(len({p['family_id'] for p in selected}), 6)
        with self.assertRaises(ValueError): m.select_seed(rows, 100)

    def test_no_silent_shape_or_glyph_rewriting(self):
        lines = poem(1)['lines']
        valid = {'paragraphs': [lines[i] + '，' + lines[i+1] + '。' for i in range(0, 8, 2)]}
        invalid = {'paragraphs': ['abcd，efgh。']}
        rows, _, counts = m.collect_poems([('fixture', [valid, invalid])])
        self.assertEqual(rows[0]['lines'], lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts['raw_records'], 2)
        self.assertEqual(rows[0]['status'], 'shape_only_not_verified_lushi')

    def test_guangyun_keeps_all_records_and_editorial_markup(self):
        csv = '小韻號,小韻字號,音韻地位,反切,字頭,釋義\n1,1,平,甲乙,還,返回\n2,1,平,丙丁,還,另義\n3,1,上,戊己,［甲］,校勘\n'
        entries, lookup, counts = m.historical_entries(csv.encode())
        self.assertEqual(lookup['還'], ['gy:1', 'gy:2'])
        self.assertNotIn('甲', lookup)
        self.assertEqual(entries[2]['fields']['字頭'], '［甲］')
        self.assertEqual(counts['editorial_or_empty_headword_rows'], 1)
        self.assertTrue(all(e['status'] == 'source_record_not_context_resolution' for e in entries))
        with self.assertRaises(ValueError): m.historical_entries(b'wrong,header\na,b\n')

    def test_tasks_do_not_invent_labels(self):
        p = poem(1); p['family_id'] = 'f1'
        tasks = m.annotation_tasks([p], {}, {'characters': {}})
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]['row_indices_zero_based'], [2, 3])
        self.assertEqual(len(tasks[0]['reading_candidates']), 10)
        for task in tasks:
            m.validate_annotation(task)
            self.assertFalse(task['is_gold'])
            self.assertIsNone(task['annotation']['word_boundaries'])
            self.assertTrue(all(r['selected_entry_id'] is None for r in task['reading_candidates']))

    def test_claimed_gold_or_reading_is_rejected(self):
        p = poem(1); p['family_id'] = 'f1'
        task = m.annotation_tasks([p], {}, {'characters': {}})[0]
        task['is_gold'] = True
        with self.assertRaises(ValueError): m.validate_annotation(task)
        task['is_gold'] = False
        task['reading_candidates'][0]['selected_entry_id'] = 'gy:1'
        with self.assertRaises(ValueError): m.validate_annotation(task)

    def test_review_requires_identity_and_evidence(self):
        p = poem(1); p['family_id'] = 'f1'
        task = m.annotation_tasks([p], {}, {'characters': {}})[0]
        task['annotation']['status'] = 'reviewed'
        with self.assertRaises(ValueError): m.validate_annotation(task)

    def test_output_and_cache_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): m.download(Path(directory))
            with self.assertRaises(ValueError): m.build(ROOT, Path(directory), Path(directory))

    def test_git_identity_is_byte_exact(self):
        self.assertEqual(m.git_blob(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
        self.assertNotEqual(m.git_blob('還'.encode()), m.git_blob('还'.encode()))

    def test_unapproved_download_origin_rejected_before_network(self):
        for url in ['http://raw.githubusercontent.com/a', 'https://evil.example/file', 'https://a:b@api.github.com/']:
            with self.assertRaises(ValueError): m.get(url)

if __name__ == '__main__': unittest.main()
