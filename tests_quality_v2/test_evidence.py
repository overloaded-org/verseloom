"""Evidence checks, not literary/interpretive certification."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qv2',ROOT/'scripts/build_quality_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def entry(i,head='甲',gloss='本義',edge='',rhyme='1',tone='端一東平'):
 return {'entry_id':f'gy:{i}','csv_line_end':i+1,'fields':{'字頭':head,'釋義':gloss,'釋義參照':edge,'小韻號':rhyme,'音韻地位':tone,'反切':'甲乙'}}
class EvidenceTests(unittest.TestCase):
 def test_upward_reference(self):
  r=m.GlossGraph([entry(1),entry(2,gloss='上同',edge='上')]).resolve('gy:2');self.assertEqual(r['chain'],['gy:2','gy:1']);self.assertFalse(r['inheritsReading']);self.assertFalse(r['contextCertified'])
 def test_downward_reference(self):
  self.assertEqual(m.GlossGraph([entry(1,gloss='',edge='下'),entry(2)]).resolve('gy:1')['chain'],['gy:1','gy:2'])
 def test_cycle_explicit(self):
  self.assertEqual(m.GlossGraph([entry(1,edge='下'),entry(2,edge='上')]).resolve('gy:1')['status'],'cycle')
 def test_cross_rhyme(self):
  self.assertEqual(m.GlossGraph([entry(1),entry(2,edge='上',rhyme='2')]).resolve('gy:2')['status'],'cross_rhyme_reference')
 def test_missing_or_unsupported_reference(self):
  self.assertEqual(m.GlossGraph([entry(1,edge='上')]).resolve('gy:1')['status'],'missing_neighbour');self.assertEqual(m.GlossGraph([entry(1,edge='猜')]).resolve('gy:1')['status'],'unsupported_reference')
 def test_duplicate_ids(self):
  with self.assertRaises(ValueError):m.GlossGraph([entry(1),entry(1)])
 def test_gloss_reference_cannot_overwrite_reading(self):
  rows=[entry(1,tone='透一東去'),entry(2,edge='上')];g=m.GlossGraph(rows)
  r=m.reading_evidence('甲',{'甲':[rows[1]]},g,{'甲':[{'tone':'P','group_id':'pingshui:upper_level:01'}]},{});self.assertEqual(r['sourceEntryId'],'gy:2');self.assertEqual(r['phonologicalPosition'],'端一東平')
 def test_ambiguous_policy_does_not_pass(self):
  rows=[entry(1),entry(2)];p={'甲':{'entryId':'gy:1','glossContains':'本義','status':'ambiguous_policy'}};self.assertFalse(m.reading_evidence('甲',{'甲':rows},m.GlossGraph(rows),{},p)['admitted'])
 def test_wrong_id_or_gloss(self):
  rows=[entry(1)];g=m.GlossGraph(rows)
  for p in [{'entryId':'gy:2','glossContains':'本義','status':'sense_supported_policy'},{'entryId':'gy:1','glossContains':'不存在','status':'sense_supported_policy'}]:self.assertFalse(m.reading_evidence('甲',{'甲':rows},g,{}, {'甲':p})['admitted'])
 def test_four_tone_category_not_just_ping_ze(self):
  e=entry(1,tone='匣開二麻上');ref={'甲':[{'tone':'Z','group_id':'pingshui:rising:21'},{'tone':'Z','group_id':'pingshui:departing:22'}]};r=m.reading_evidence('甲',{'甲':[e]},m.GlossGraph([e]),ref,{});self.assertTrue(r['admitted']);self.assertEqual(r['group'],'pingshui:rising:21');self.assertFalse(r['contextCertified'])
 def test_multiple_groups_unknown(self):
  e=entry(1);ref={'甲':[{'tone':'P','group_id':'pingshui:upper_level:01'},{'tone':'P','group_id':'pingshui:upper_level:02'}]};self.assertFalse(m.reading_evidence('甲',{'甲':[e]},m.GlossGraph([e]),ref,{})['admitted'])
 def test_heldout_skipped_before_decode(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'c';p.write_text('{"split":"test", invalid}\n'+json.dumps({'split':'train','length':5,'lines':['甲乙丙丁戊']*8,'poem_id':'p'})+'\n');self.assertEqual(len(m.load_training([(p,'train')])),1)
 def test_lattice_unknown(self):
  self.assertTrue(all(a['posHypothesis'] is None for a in m.token_lattice('甲乙丙',[])))
 def test_lattice_alternatives(self):
  w=[{'text':'甲','pos':'N','sense':'a'},{'text':'甲乙','pos':'N','sense':'b'}];self.assertEqual(len([a for a in m.token_lattice('甲乙',w) if a['start']==0]),2)
 def test_no_sentences_in_lexicon(self):
  c=json.loads((ROOT/'data/quality_v2/policy.json').read_bytes());self.assertTrue(all(1<=len(w)<=2 for g in c['wordGroups'] for w in g['texts'].split()));self.assertIn('not_gold',c['status'])
 def test_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.build(Path(d),Path(d),Path(d)/'x.js')
if __name__=='__main__':unittest.main()
