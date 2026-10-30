"""Synthetic scope/regression fixtures, not independent literary labels."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.quality import QualityReview

class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = QualityReview(ROOT)

    def test_full_parse_required_for_parallel_structure(self):
        r = self.q.pair('清風入竹林', '細雨過江村')
        self.assertEqual(r['sharedSignatures'], ['N:2|V:1|N:2'])
        self.assertEqual(r['status'], 'aligned_under_draft_rules')
        self.assertFalse(r['strictAntithesisCertified'])

    def test_locative_structure_is_not_same_position_inequality(self):
        r = self.q.pair('明月松間照', '清泉石上流')
        self.assertEqual(r['sharedSignatures'], ['N:2|L:2|V:1'])
        self.assertFalse(r['strictAntithesisCertified'])

    def test_noun_piles_are_out_of_parser_scope_not_declared_bad_poetry(self):
        for text in ('青山美少年', '天津御柳古城池'):
            self.assertEqual(self.q.syntax(text)['status'], 'unsupported')
        self.assertEqual(self.q.pair('青山美少年', '日暮春風起')['status'], 'unknown')

    def test_unknown_tokens_never_get_invented_pos_to_make_a_pair_fit(self):
        self.assertEqual(self.q.pair('甲乙丙丁戊', '己庚辛壬癸')['status'], 'unknown')

    def test_multimembership_even_if_same_tone_stays_unresolved(self):
        item = self.q.line('還家越水邊')['readings'][0]
        self.assertEqual(item['tones'], ['P'])
        self.assertGreater(len(item['groups']), 1)
        self.assertEqual(item['status'], 'historical_choice_unresolved')
        self.assertFalse(item['contextVerified'])

    def test_modern_dictionary_hint_does_not_pick_historical_reading(self):
        r = self.q.line('青山美少年')
        hint = next(x for x in r['dictionaryHints'] if x['word']=='少年')
        self.assertEqual(hint['pinyin'], 'shào nián')
        self.assertFalse(hint['historicalMembershipResolved'])
        self.assertEqual(r['readings'][3]['status'], 'historical_choice_unresolved')

    def test_single_reference_is_not_context_certification(self):
        r = self.q.line('白馬入江煙')
        self.assertTrue(all(not x['contextVerified'] for x in r['readings']))
        self.assertFalse(r['contextualPronunciationVerified'])

    def test_bridge_windows_cross_actual_two_plus_three_or_four_plus_three_cut(self):
        for text in ('青山美少年','天津御柳古城池'):
            r = self.q.line(text)
            self.assertEqual(len(r['bridgeEvidence']), 3)
            for item in r['bridgeEvidence']:
                self.assertLess(item['start'], r['cut'])
                self.assertGreater(item['end'], r['cut'])
                self.assertEqual(item['text'], text[item['start']:item['end']])
                self.assertGreaterEqual(item['trainingCount'],0)

    def test_no_corpus_line_crossing_is_counted(self):
        for length in (5,7):
            from collections import Counter
            expected = Counter()
            for poem in self.q.training[length]:
                for line in poem['lines']:
                    for width in (2,3):
                        expected.update(line[i:i+width] for i in range(length-width+1))
            self.assertEqual(expected,self.q.grams[length])

    def test_invalid_lengths_and_nonfinite_scores_fail(self):
        for lines in ([], ['清風入竹林']*7, ['清風入竹林']*7+['七言不是五字句'], [1]*8):
            with self.assertRaises(ValueError): self.q.audit(lines)
        for score in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError): self.q.rank_score({'lines':['清風入竹林']*8,'score':score})

    def test_review_does_not_modify_the_program_output(self):
        poem=['清風入竹林','細雨過江村']*4
        record={'lines':poem,'score':-30}
        old=copy.deepcopy(record)
        score, result=self.q.rank_score(record)
        self.assertEqual(record,old)
        self.assertAlmostEqual(score, result['rawMeanScore']-sum(result['penalties'].values()))
        self.assertFalse(result['audit']['publishAsVerified'])
        self.assertEqual(result['audit']['literaryQuality'],'not_evaluated')

    def test_heldout_rows_are_not_json_decoded_or_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['data/quality_v1/lexicon.json','data/quality_v1/reading_hints.json',
                         'data/reference_v02/pingshui_106_memberships.json','src/wordconv_lab/quality.py']:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            path=root/'data/corpus_v03/poems.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            first=self.q.training[5][0]
            path.write_text(json.dumps(first,ensure_ascii=False)+'\n'+
                            '{"split":"dev", DELIBERATELY NOT JSON}\n'+
                            '{"split":"test", DELIBERATELY NOT JSON}\n')
            q=QualityReview(root)
            self.assertEqual(len(q.training[5]),1)
            self.assertEqual(len(q.training[7]),0)

if __name__=='__main__': unittest.main()
