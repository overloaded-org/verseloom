import importlib.util,json,csv,io,tempfile,unittest,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=module('composer_build',ROOT/'scripts/build_composition_v2.py');a=module('composer_audit',ROOT/'scripts/audit_composition_v2.py')
class EvidenceTests(unittest.TestCase):
 def test_up_reference_preserves_original_reading_id(self):
  rows=[{'字頭':'甲','音韻地位':'一平','反切':'甲乙','釋義':'義項','釋義參照':''},{'字頭':'乙','音韻地位':'二上','反切':'丙丁','釋義':'上同','釋義參照':'上'}]
  ref={'乙':[{'tone':'Z','group_id':'pingshui:rising:01','label':'上聲','source_lines':[1]}]}
  r,e=b.character_evidence('乙',rows,{'乙':[1]},ref,{})
  self.assertIsNone(e);self.assertEqual(r['entryId'],'gy:2');self.assertEqual(r['fanqie'],'丙丁');self.assertEqual(r['glossTrace']['path'],['gy:2','gy:1']);self.assertFalse(r['contextuallyCertified'])
 def test_down_reference(self):
  r=b.gloss_trace([{'釋義':'並同','釋義參照':'下'},{'釋義':'實義','釋義參照':''}],0)
  self.assertEqual(r['gloss'],'實義');self.assertEqual(r['path'],['gy:1','gy:2'])
 def test_cycles_are_unknown_not_auto_resolved(self):
  self.assertEqual(b.gloss_trace([{'釋義':'','釋義參照':'下'},{'釋義':'','釋義參照':'上'}],0)['status'],'cycle')
 def test_boundary_and_unknown_reference(self):
  self.assertEqual(b.gloss_trace([{'釋義':'','釋義參照':'上'}],0)['status'],'out_of_bounds')
  self.assertEqual(b.gloss_trace([{'釋義':'','釋義參照':'上二'}],0)['status'],'unknown_reference')
 def test_four_tones_not_coarse_ping_ze_choose_membership(self):
  row={'字頭':'雨','音韻地位':'云虞上','反切':'王矩','釋義':'降水','釋義參照':''}
  refs={'雨':[{'tone':'Z','group_id':'pingshui:rising:07','label':'麌','source_lines':[1]},{'tone':'Z','group_id':'pingshui:departing:07','label':'遇','source_lines':[2]}]}
  r,_=b.character_evidence('雨',[row],{'雨':[0]},refs,{})
  self.assertEqual(r['rhymeGroup'],'pingshui:rising:07')
 def test_same_four_tone_different_rhymes_not_guessed(self):
  row={'字頭':'洗','音韻地位':'心齊上','反切':'先禮','釋義':'洗浴','釋義參照':''}
  refs={'洗':[{'tone':'Z','group_id':'pingshui:rising:08','label':'薺','source_lines':[1]},{'tone':'Z','group_id':'pingshui:rising:16','label':'銑','source_lines':[2]}]}
  r,e=b.character_evidence('洗',[row],{'洗':[0]},refs,{})
  self.assertIsNone(r);self.assertEqual(e,'unresolved_pingshui_mapping')
 def test_multi_reading_without_sense_stays_unknown(self):
  r,e=b.character_evidence('甲',[{},{}],{'甲':[0,1]},{},{})
  self.assertIsNone(r)
 def test_sense_rule_requires_exact_evidence_and_headword(self):
  row={'字頭':'甲','音韻地位':'一平','反切':'甲乙','釋義':'正確義項','釋義參照':''}
  with self.assertRaises(ValueError):b.character_evidence('甲',[row],{'甲':[0]},{},{'甲':{'entry_id':'gy:1','gloss_contains':'不存在','sense':'假'}})
  with self.assertRaises(ValueError):b.character_evidence('乙',[row],{'乙':[]},{},{'乙':{'entry_id':'gy:1','gloss_contains':'正確','sense':'假'}})
 def test_heldout_rows_not_decoded_or_counted(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'data/corpus_v03').mkdir(parents=True);run=root/'run';run.mkdir()
   valid={'poem_id':'p','split':'train','length':5,'lines':['甲乙丙丁戊']*8}
   (root/'data/corpus_v03/poems.jsonl').write_text(json.dumps(valid)+'\n'+ '{"split":"test", INVALID JSON}\n')
   (run/'candidate_poems.jsonl').write_text('{"split":"test_reserve", INVALID JSON}\n{"split":"quarantine_prior_relation", INVALID JSON}\n')
   self.assertEqual(b.read_training(root,run),[valid])
 def test_data_contains_no_stored_output_lines(self):
  data=a.load_data(ROOT/'web/composition-data.js');self.assertEqual(data['meta']['storedOutputLines'],0)
  self.assertNotIn('pools',data);self.assertNotIn('baseline',data)
  self.assertTrue(all(len(w['text'])<=2 for w in data['lexicon']))
 def test_all_admitted_reading_glosses_have_finite_source_paths(self):
  data=a.load_data(ROOT/'web/composition-data.js')
  for r in data['readings'].values():
   self.assertEqual(r['glossTrace']['status'],'resolved');self.assertFalse(r['contextuallyCertified'])
   self.assertEqual(len(r['glossTrace']['path']),len(set(r['glossTrace']['path'])))
 def test_all_lexemes_have_train_only_attestations(self):
  data=a.load_data(ROOT/'web/composition-data.js')
  for w in data['lexicon']:
   self.assertGreater(w['count'],0);self.assertTrue(w['sources']);self.assertTrue(all(s['split'] in ['train','train_candidate'] for s in w['sources']))
 def test_template_variants_are_sourced_narrow_profile(self):
  data=a.load_data(ROOT/'web/composition-data.js')
  for n in [5,7]:
   for x,t in zip(data['templates'][str(n)],b.all_templates(n)):
    self.assertEqual(x['id'],t.template_id)
    self.assertEqual(x['variants'],[list(b.variants(c,'common_prefix')) for c in t.core])
 def test_reproducibility_metadata_does_not_claim_gold(self):
  data=a.load_data(ROOT/'web/composition-data.js');self.assertFalse(data['meta']['independentGold']);self.assertTrue(data['meta']['assistantAuthoredSemantics'])
 def test_build_identity_hash(self):
  data=a.load_data(ROOT/'web/composition-data.js')
  self.assertEqual(data['sourceIdentity']['lexiconSpecSha256'],b.digest((ROOT/'data/composition_v2/lexicon_spec.json').read_bytes()))
if __name__=='__main__':unittest.main()
