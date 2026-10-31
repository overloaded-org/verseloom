#!/usr/bin/env python3
"""Source order with identical formal slots: swap couplets 2 and 4, not 2 and 3.
Offline development experiment. No candidate text is authored or deployed.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from train_continuation_ranker import Features, NAMES, keyed, read_partition, require, sha
from train_continuation_order_probe import fit

PERM=(0,1,6,7,4,5,2,3)
def run(a):
    raw=a.model.read_bytes(); corpus=sha(a.corpus.read_bytes()); data=json.loads(raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]); r=json.loads(a.ranker.read_bytes()); old=json.loads(a.order.read_bytes())
    require(not a.out.exists() and old['sourceHashes']==dict(model=sha(raw),ranker=sha(a.ranker.read_bytes()),corpus=corpus),'Output/source mismatch');require(data['sourceIdentity']==r['sourceIdentity'] and data['sourceIdentity']['newCandidateCorpus']==corpus and old['features']==NAMES,'Feature/source mismatch')
    for ts in data['templates'].values():
        for t in ts:
            for i,j in enumerate(PERM):require(set(t['variants'][i])==set(t['variants'][j]) and ((i in t['rhymeRows'])==(j in t['rhymeRows'])),'Not a meter/rhyme preserving swap')
    tr,_=read_partition(a.corpus,{'train_candidate'});dv,_=read_partition(a.corpus,{'dev_reserve'});require(not {p['family_id'] for p in tr}&{p['family_id'] for p in dv},'Family overlap'); f=Features(data);seen=set();train=[]
    for n in (5,7):
        k=0
        for p in sorted((p for p in tr if p['length']==n),key=lambda p:keyed('train:'+p['poem_id'])):
            if p['family_id'] in seen:continue
            seen.add(p['family_id']);train.append(p);k+=1
            if k==1024:break
        require(k==1024,'Insufficient training families')
    def diff(p):
        s=[f.segment(l) for l in p['lines']];v=lambda x:np.asarray([f.extract(x[:j],x[j]) for j in range(1,8)]).ravel();return v(s)-v([s[j] for j in PERM])
    X=np.asarray([diff(p) for p in train]);mask=np.ones(56,dtype=bool);mask[2::8]=False; noalign=np.zeros(56);noalign[mask]=fit(X[:,mask])
    bridge=np.zeros(56);bmask=np.zeros((7,8),bool);bmask[[1,3,5],:]=True;bmask=bmask.ravel();bridge[bmask]=fit(X[:,bmask])
    weights=np.stack([np.tile(r['coefficients'],7),old['weights'][1],fit(X),noalign,bridge]); arms=['general','middle_swap_transfer','same_meter_trained','no_alignment_trained','bridge_only_trained'];obs=[];g=defaultdict(list)
    for p in dv:
        scores=weights@diff(p); credit=np.where(abs(scores)<=1e-12,.5,(scores>0).astype(float));obs.append([p['length'],p['poem_id'],p['family_id'],scores.tolist()]);g[p['length'],p['family_id']].append(credit)
    results={}
    for n in (5,7):
        v=np.array([np.mean(g[k],axis=0) for k in sorted(g) if k[0]==n]);rng=np.random.default_rng(20261031);ix=rng.integers(len(v),size=(2000,len(v)));cell={}
        for name,s in list(zip(arms,v.T))+[('trained_minus_general',v[:,2]-v[:,0]),('trained_minus_transfer',v[:,2]-v[:,1]),('trained_minus_no_alignment',v[:,2]-v[:,3]),('bridge_minus_chance',v[:,4]-.5)]:cell[name]={'mean':s.mean(),'ci95':np.quantile(s[ix].mean(1),[.025,.975]).tolist()}
        results[n]=cell
    result=dict(schema='verseloom.equal-meter-order.v1',order=PERM,templateIds=[t['id'] for ts in data['templates'].values() for t in ts],toneOnlyBothArmsTieByConstruction=True,trainPoems=len(train),devPoems=len(dv),features=NAMES,arms=arms,weights=weights.tolist(),results=results,sourceHashes=old['sourceHashes'],limitations='Post-diagnostic exploratory contrast on the same dev families. Preserved template slots and rhyme rows do not remove positional style cues or establish human aesthetic preference. No generator or Pages change.')
    a.out.mkdir(parents=True);(a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in obs));print(json.dumps(results))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','corpus','ranker','order','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
