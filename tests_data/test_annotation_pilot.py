"""Draft metadata/offset regressions, not literary judgement."""
import copy
import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_check',ROOT/'scripts/check_annotation_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    task={'poem_id':'p','family_id':'f','length':5,'context_lines':['甲乙丙丁戊']*8,
          'source_refs':[{'revision':'fixed'}],'row_indices_zero_based':[2,3],
          'usage':'development_annotation_only','is_gold':False,
          'reading_candidates':[{'row':2,'column':0,'character':'甲','guangyun_entry_ids':['gy:1']}]}
    record={k:copy.deepcopy(task[k]) for k in ('poem_id','family_id','length','context_lines','source_refs','row_indices_zero_based')}
    record.update(task_id='t',status='assistant_draft',is_gold=False,independent_review=False,
                  publish_as_verified=False,usage='development_annotation_only',strict_antithesis='unknown',
                  contextual_pronunciation='unresolved',selected_historical_readings=[],literary_quality_score=None,
                  tokens=[{'row':r,'tokens':[{'start':0,'end':5,'text':'甲乙丙丁戊','pos_candidates':['N']}]} for r in (2,3)],
                  alignment_hypothesis=[{'upper_token':0,'lower_token':0}],alignment_status='proposed_not_certified',
                  reading_notes=[{'row':2,'column':0,'character':'甲','entry_ids':['gy:1']}])
    return {'schema':'verseloom.data-bootstrap.annotation-pilot.v1','records':[record]}, {'t':task}

class PilotTests(unittest.TestCase):
    def test_structural_check_does_not_certify_interpretation(self):
        result=m.validate(*fixture())
        self.assertFalse(result['interpretive_correctness_certified'])
        self.assertEqual(result['independent_gold_labels'],0)

    def test_source_and_span_changes_rejected(self):
        for change in ('source','text','offset'):
            p,t=fixture();r=p['records'][0]
            if change=='source':r['context_lines'][2]='乙甲丙丁戊'
            if change=='text':r['tokens'][0]['tokens'][0]['text']='乙甲丙丁戊'
            if change=='offset':r['tokens'][0]['tokens'][0]['start']=1
            with self.assertRaises(ValueError):m.validate(p,t)

    def test_gold_and_reading_and_quality_claims_rejected(self):
        for field,value in [('is_gold',True),('independent_review',True),('strict_antithesis','pass'),
                            ('selected_historical_readings',['gy:1']),('literary_quality_score',0.9)]:
            p,t=fixture();p['records'][0][field]=value
            with self.assertRaises(ValueError):m.validate(p,t)

    def test_unrelated_dictionary_entries_rejected(self):
        p,t=fixture();p['records'][0]['reading_notes'][0]['entry_ids']=['gy:2']
        with self.assertRaises(ValueError):m.validate(p,t)

    def test_duplicate_and_unmatched_alignment_rejected(self):
        p,t=fixture();p['records'].append(copy.deepcopy(p['records'][0]))
        with self.assertRaises(ValueError):m.validate(p,t)
        p,t=fixture();p['records'][0]['alignment_hypothesis'][0]['lower_token']=1
        with self.assertRaises(ValueError):m.validate(p,t)

if __name__=='__main__':unittest.main()
