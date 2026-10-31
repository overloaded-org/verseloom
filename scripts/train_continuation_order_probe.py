#!/usr/bin/env python3
"""Research only: learn to distinguish two orders of identical intact couplets.
No theme, text features, new corpus, search or parameter tuning on dev.
"""
import argparse, json
from collections import defaultdict
import numpy as np
from pathlib import Path
from train_continuation_ranker import Features, NAMES, keyed, read_partition, require, sha

ORDER = (0,1,4,5,2,3,6,7)

def fit(differences):
    scale=np.maximum(np.sqrt(np.mean(differences**2,axis=0)),1e-6)
    d=differences/scale; w=np.zeros(d.shape[1])
    for _ in range(600):
        z=np.clip(d@w,-40.,40.)
        w-=.15*(-(d.T@(1/(1+np.exp(z))))/len(d)+.02*w)
    return w/scale

def run(a):
    require(not a.out.exists(),'Output exists'); raw=a.model.read_bytes(); data=json.loads(raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]); ranker=json.loads(a.ranker.read_bytes()); require(ranker['sourceIdentity']==data['sourceIdentity'] and ranker['features']==NAMES,'Model mismatch');require(sha(a.corpus.read_bytes())==data['sourceIdentity']['newCandidateCorpus'],'Corpus mismatch')
    train,_=read_partition(a.corpus,{'train_candidate'});dev,_=read_partition(a.corpus,{'dev_reserve'});require(not {p['family_id'] for p in train}&{p['family_id'] for p in dev},'Family overlap')
    selected=[];seen=set()
    for n in (5,7):
        k=0
        for p in sorted((p for p in train if p['length']==n),key=lambda p:keyed('train:'+p['poem_id'])):
            if p['family_id'] in seen:continue
            selected.append(p);seen.add(p['family_id']);k+=1
            if k==1024:break
        require(k==1024,'Training size')
    f=Features(data);vector=lambda lines:np.asarray([f.extract(lines[:i],lines[i]) for i in range(1,8)]).ravel()
    def diff(lines):
        s=[f.segment(line) for line in lines];return vector(s)-vector([s[j] for j in ORDER])
    tr=np.asarray([diff(p['lines']) for p in selected]);coef=fit(tr);general=np.tile(ranker['coefficients'],7)
    # Locality control: separately fit using only the transition into the two
    # middle couplets and the transition out to the ending (2->3,4->5,6->7).
    mask=np.zeros(56,dtype=bool)
    for j in (1,3,5):mask[8*j:8*j+8]=True
    boundary=np.zeros(56);boundary[mask]=fit(tr[:,mask]); weights=np.stack([general,coef,boundary]); bylen={n:[p for p in dev if p['length']==n] for n in (5,7)}; tests=[]
    for n,rows in bylen.items():
        for p in rows:
            offset=int(keyed('context-swap:'+p['poem_id']),16)%len(rows)
            donor=next(q for q in rows[offset:]+rows[:offset] if q['family_id']!=p['family_id'] and set(q['lines']).isdisjoint(p['lines']))
            for typ,lines in [('original',p['lines']),('foreign_bookends',donor['lines'][:2]+p['lines'][2:6]+donor['lines'][6:])]:
                d=diff(lines);scores=weights@d;tests.append([n,p['poem_id'],p['family_id'],typ,donor['poem_id'] if typ!='original' else None,scores.tolist()])
    def interval(values):
        rng=np.random.default_rng(20261031); boot=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
        return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    summary={};arms=['general','order_trained','bridge_trained'];groups=defaultdict(list)
    for n,pid,family,typ,donor,scores in tests:
        score=np.asarray(scores);credit=np.where(abs(score)<=1e-12,.5,(score>0).astype(float));groups[n,typ,family].append(credit)
    for n in (5,7):
        families=sorted({p['family_id'] for p in bylen[n]}); two=[]
        for typ in ('original','foreign_bookends'):
            m=np.asarray([np.mean(groups[n,typ,fid],axis=0) for fid in families]);two.append(m);summary[f'{n}:{typ}']={arm:interval(m[:,j]) for j,arm in enumerate(arms)};summary[f'{n}:{typ}']['order_minus_general']=interval(m[:,1]-m[:,0])
        summary[f'{n}:own_minus_foreign']={arm:interval(two[0][:,j]-two[1][:,j]) for j,arm in enumerate(arms)}
    report=dict(schema='verseloom.order-probe.v1',trainPoems=len(selected),trainFamilies=len(seen),devPoems=len(dev),features=NAMES,rowsOneBased=list(range(2,9)),orderZeroBased=ORDER,arms=arms,summary=summary,parameters={'steps':600,'lr':.15,'l2':.02,'intercept':False},sourceHashes={k:sha(getattr(a,k).read_bytes()) for k in ('model','ranker','corpus')},weights=weights.tolist(),use='development diagnostic, not deployed',limitations='Predicting source order is not literary quality. Corpus graph includes training families. Foreign bookends are length-matched, not meter/rhyme-matched. Family intervals are descriptive, not adjusted for multiple comparisons.')
    a.out.mkdir(parents=True);(a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.out/'predictions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tests));print(json.dumps(summary))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model','ranker','corpus','out'):ap.add_argument('--'+key,type=Path,required=True)
    run(ap.parse_args())
