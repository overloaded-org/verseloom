#!/usr/bin/env python3
"""Arithmetic audit of already-verified inputs, never authority to run an arm.
python check_counts.py --self-test | python check_counts.py counts.json
For example ballot inputs see synthetic() below. An absent arm is not zero.
"""
import argparse, json
from pathlib import Path
from gate_policy import ballot_gate, decision

def check(x):
    if x.get('verified_originals') != 2:
        return {'status':'LOCKED','human_agreement':None,'deploy':False}
    b=ballot_gate(*(x[k] for k in ('human_human','human_A_locked','human_B_locked')))
    if not b['permission_for_offline_arms']:
        return {'status':'HOLD_SAME_ITEMS','ballots':b,'deploy':False}
    if 'counts' not in x:
        return {'status':'OFFLINE_ONLY_NO_ARM_EVIDENCE','ballots':b,'deploy':False}
    return {'ballots':b,**decision(**x['counts'])}

def synthetic():
    yes={str(i):[True,True] for i in range(1,11)}
    x=dict(verified_originals=2,human_human=yes,human_A_locked=yes,human_B_locked=yes)
    assert check({})['status']=='LOCKED'
    assert check(x)['status']=='OFFLINE_ONLY_NO_ARM_EVIDENCE'
    a={**yes,'1':[True,False],'2':[False,True]};b={**yes,'3':[True,False],'4':[False,True]};hh={**a,**{k:b[k] for k in ('3','4')}}
    mixed=check(dict(x,human_A_locked=a,human_B_locked=b,human_human=hh));assert len(mixed['ballots']['common_editor_groups'])==6
    dissent={**a,'3':[True,False]};r=check(dict(x,human_A_locked=dissent,human_B_locked=dissent));assert r['ballots']['original_pass'] and r['status']=='HOLD_SAME_ITEMS'
    standard=dict(quality=[[5,5,4],[5,5,4]],changes=[10,10,8],executions=[12,11,9],peers=[[10,12],[10,12]],safe=[True,True,True],bodies=5)
    def status(**edit):return check(dict(x,counts={**standard,**edit}))['status']
    assert status()=='REPLICATION_ONLY'
    assert status(quality=[[0,0,0],[0,0,0]],bodies=0)=='INSUFFICIENT'
    assert status(quality=[[6,5,4],[5,5,4]])=='INSUFFICIENT'
    assert status(quality=[[6,6,4],[5,5,4]])=='FAIL'
    assert status(changes=[10,9,8])=='FAIL'
    assert status(changes=[10,10,7])=='FAIL'
    assert status(safe=[True,False,True])=='FAIL'
    return {'status':'PASS_SYNTHETIC_ONLY','real_human_ballots_used':0,'new_arms_executed':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',nargs='?',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if not __debug__:p.error('do not disable assertions')
    if not a.self_test and not a.input:p.error('provide counts.json or --self-test')
    print(json.dumps(synthetic() if a.self_test else check(json.loads(a.input.read_bytes())),ensure_ascii=False,indent=2))
