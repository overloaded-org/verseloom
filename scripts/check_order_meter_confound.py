#!/usr/bin/env python3
"""Can tone templates alone distinguish the middle-couplet swap? Dev-only.
Unknown tones are ignored in both versions. A tie is not proof of equal meter.
"""
import argparse,json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from train_continuation_ranker import read_partition, require, sha

def run(a):
    require(not a.out.exists(),'Output exists'); raw=a.model.read_bytes();data=json.loads(raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]); src=json.loads((a.order/'report.json').read_bytes())['sourceHashes']; require(src['model']==sha(raw) and src['corpus']==sha(a.corpus.read_bytes()),'Source mismatch')
    with (a.order/'predictions.jsonl').open() as f: pred={r[1]:r for r in map(json.loads,f) if r[3]=='original'}
    dev,_=read_partition(a.corpus,{'dev_reserve'}); require(set(pred)=={p['poem_id'] for p in dev},'Prediction set mismatch'); res=[]; stats={}
    @lru_cache(None)
    def tones(line):return ''.join(data['readings'].get(c,{}).get('tone') or '?' for c in line)
    @lru_cache(None)
    def mismatch(actual,variants):return min(sum(t!='?' and t!=v for t,v in zip(actual,variant)) for variant in variants)
    def loss(seq,n):return min(sum(mismatch(t,tuple(templ['variants'][i])) for i,t in enumerate(seq)) for templ in data['templates'][str(n)])
    for p in dev:
        n=p['length']; pid=p['poem_id'];f=p['family_id'];s=list(map(tones,p['lines'])); delta=loss(s[:2]+s[4:6]+s[2:4]+s[6:],n)-loss(s,n);cov=sum(t!='?' for line in s for t in line);res.append([n,pid,f,delta,cov,pred[pid][-1]])
    for n in [5,7]:
        rows=[r for r in res if r[0]==n]; forms={}
        for key,part in [('all',rows),('equal_known_tone_loss',[r for r in rows if r[3]==0])]:
            if not part:forms[key]={'poems':0};continue
            g=defaultdict(list)
            for _,pid,f,delta,cov,scores in part:g[f].append([.5 if abs(x)<=1e-12 else float(x>0) for x in (delta,*scores)])
            v=np.array([np.mean(g[f],axis=0) for f in sorted(g)]);d=v[:,2]-v[:,1];rng=np.random.default_rng(20261031);bs=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
            forms[key]={'poems':len(part),'families':len(g),'toneCoverage':sum(r[4] for r in part)/(len(part)*n*8),'toneOnly':v[:,0].mean(),'general':v[:,1].mean(),'order':v[:,2].mean(),'bridge':v[:,3].mean(),'orderMinusGeneralCI95':np.quantile(bs,[.025,.975]).tolist()}
        stats[n]=forms
    a.out.mkdir(parents=True);(a.out/'report.json').write_text(json.dumps({'schema':'verseloom.order-probe.known-tone-control.v1','results':stats,'sourceHashes':src,'note':'No expert metrical labels. Minimum mismatches across available templates at known positions; unknowns abstain. Equal losses do not mean two legal poems.'},indent=2)+'\n');(a.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in res));print(json.dumps(stats))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('model','corpus','order','out'):ap.add_argument('--'+k,type=Path,required=True)
    run(ap.parse_args())
