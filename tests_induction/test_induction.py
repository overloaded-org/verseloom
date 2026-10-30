import importlib.util, itertools, json, math, tempfile, unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('induced',R/'scripts/build_induced_model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixtures():
    return [{'poem_id':str(i),'lines':['甲乙丙丁戊','己庚辛壬癸']*4} for i in range(4)]

class InductionTests(unittest.TestCase):
    def test_discovery_only_extracts_observed_spans(self):
        counts,df,src,p,h=m.discover(fixtures(),dict(m.PARAMS,iterations=2))
        for w in p:self.assertTrue(any(w in line for poem in fixtures() for line in poem['lines']))
        self.assertTrue(all(1<=len(w)<=3 for w in p))
        self.assertTrue(all(df[w]>=3 for w in p if len(w)>1))
    def test_em_objective_does_not_decrease(self):
        *_,h=m.discover(fixtures(),dict(m.PARAMS,iterations=4))
        self.assertTrue(all(b+1e-6>=a for a,b in zip(h,h[1:])))
    def test_segmentation_covers_original_without_prescribed_cut(self):
        *_,p,h=m.discover(fixtures(),dict(m.PARAMS,iterations=2))
        spans=m.segment('甲乙丙丁戊',p,3)
        self.assertEqual(''.join(w for a,b,w in spans),'甲乙丙丁戊')
        self.assertEqual(spans[0][0],0);self.assertEqual(spans[-1][1],5)
        self.assertTrue(all(a==b0 for (a0,b0,w0),(a,b,w) in zip(spans,spans[1:])))
    def test_viterbi_matches_all_segmentations(self):
        p={'甲':.3,'乙':.2,'丙':.1,'甲乙':.15,'乙丙':.2,'甲乙丙':.05};s='甲乙丙'
        def allparts(text):
            if not text:return [()]
            return [(text[:n],)+rest for n in range(1,len(text)+1) if text[:n] in p for rest in allparts(text[n:])]
        expected=max(sum(math.log(p[w]) for w in parts) for parts in allparts(s))
        got=sum(math.log(p[w]) for a,b,w in m.segment(s,p,3));self.assertAlmostEqual(expected,got)
    def test_training_reader_skips_malformed_heldout_before_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'data/corpus_v03').mkdir(parents=True);(r/'boot/run').mkdir(parents=True)
            x={'poem_id':'p','split':'train','length':5,'lines':['甲乙丙丁戊']*8}
            (r/'data/corpus_v03/poems.jsonl').write_bytes(m.packed(x)+b'{"split":"dev", BAD JSON}\n')
            y={**x,'poem_id':'q','split':'train_candidate'};raw=m.packed(y)+b'{"split":"test_reserve", INVALID}\n'
            (r/'boot/run/candidate_poems.jsonl').write_bytes(raw)
            (r/'boot/run/manifest.json').write_bytes(m.packed({'files':{'candidate_poems.jsonl':{'sha256':m.sha(raw)}}}))
            poems,_=m.training_files(r,r/'boot');self.assertEqual([p['poem_id'] for p in poems],['p','q'])
    def test_single_poem_repetition_does_not_inflate_poem_support(self):
        counts,df,*_=m.discover([fixtures()[0]],dict(m.PARAMS,iterations=1))
        self.assertEqual(df['甲乙'],1)
    def test_non_bmp_hash_iterates_unicode_characters(self):
        self.assertNotEqual(m.fnv('𠀀'),m.fnv('一'))
    def test_builder_does_not_import_prescriptive_resources(self):
        code=(R/'scripts/build_induced_model.py').read_text()
        for forbidden in ['lexicon_spec.json','reading_rules','pilot_drafts.json','quality_v2/lexicon','composition_v2/lexicon']:
            self.assertNotIn(forbidden,code)
    def test_runtime_has_no_authored_content_roles(self):
        code=(R/'web/induced-engine.js').read_text()
        for forbidden in ['adjectiveEntities','seasonNames','LOC_SVO','TIME_SVO','Agent–action','reading_rules']:
            self.assertNotIn(forbidden,code)
    def test_records_never_claim_strict_literary_certification(self):
        code=(R/'web/induced-engine.js').read_text()
        for flag in ['literaryQualityCertified:false','contextualPronunciationCertified:false','strictAntithesisCertified:false']:
            self.assertIn(flag,code)
if __name__=='__main__':unittest.main()
