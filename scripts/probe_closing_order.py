"""Offline final-line retrieval; train a residual, then intervene on earlier couplets.
Run alongside scripts/train_continuation_ranker.py. Does not generate or deploy poetry.
"""
import argparse, collections, functools, gzip, hashlib, json
from pathlib import Path
import numpy as np
from train_continuation_ranker import Features, read_partition


def sha(raw): return hashlib.sha256(raw).hexdigest()
def key(s): return sha(('verseloom.closing-order.v1/'+s).encode())
def top1(rows):
    s=np.array(rows); return float(not np.any(s>s[0]+1e-12))/np.sum(np.abs(s-s[0])<=1e-12)
def summary(values,families):
    grouped=collections.defaultdict(list)
    for f,v in zip(families,values): grouped[f].append(v)
    a=np.array([np.mean(grouped[f]) for f in sorted(grouped)])
    rng=np.random.default_rng(20261031); samples=np.mean(a[rng.integers(len(a),size=(2000,len(a)))],axis=1)
    return dict(mean=float(a.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),families=len(a))

def run(model,corpus,ranker,predictions,out):
    if out.exists():raise ValueError('Refusing overwrite')
    data=json.loads(model.read_text().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]);w=json.loads(ranker.read_bytes()); beta=np.array(w['coefficients'])
    assert w['sourceIdentity']==data['sourceIdentity'] and sha(corpus.read_bytes())==data['sourceIdentity']['newCandidateCorpus']
    tr,_=read_partition(corpus,{'train_candidate'});dv,_=read_partition(corpus,{'dev_reserve'}); byid={p['poem_id']:p for p in dv}
    assert not {p['family_id'] for p in dv}.intersection(p['family_id'] for p in tr)
    feat=Features(data);seg=functools.lru_cache(None)(feat.segment); table={int(u):{v:s for v,_,s in vs} for u,vs in data['association'].items()}
    def profile(ctx,t):
        vals=[]
        for k in (0,2,4):
            left={u for row in ctx[k:k+2] for u,_,_ in row if u>=0}
            vals.append(sum(min(4,max((table.get(u,{}).get(v,0.) for u in left if u!=v),default=0.)) for v,_,_ in t)/max(1,len(t)))
        x=np.array(vals);return x-x.mean()   # order contrast only, not extra aggregate overlap
    train=[];seen=set()
    for n in (5,7):
        take=0
        for p in sorted((p for p in tr if p['length']==n),key=lambda q:key('train/'+q['poem_id'])):
            if p['family_id'] in seen:continue
            train.append(p);seen.add(p['family_id']);take+=1
            if take==1024:break
        assert take==1024
    def bucket(p):return p['length'],data['readings'].get(p['lines'][7][-1],{}).get('rhymeGroup') or '?'
    banks=collections.defaultdict(list);forms=collections.defaultdict(list)
    for p in train:banks[bucket(p)].append(p);forms[p['length']].append(p)
    offsets=[];diff=[];fallback=0
    for p in train:
        ctx=[seg(l) for l in p['lines'][:7]]; pool=banks[bucket(p)];k=int(key('offset/'+p['poem_id'])[:16],16)%len(pool);opts=[p];fs={p['family_id']};texts=set(p['lines'])
        for j,bank in enumerate((pool[k:]+pool[:k],forms[p['length']])):
            for q in bank:
                if len(opts)==5:break
                if q['family_id'] in fs or q['lines'][7] in texts:continue
                opts.append(q);fs.add(q['family_id']);texts.add(q['lines'][7]);fallback+=j
        assert len(opts)==5
        ts=[seg(q['lines'][7]) for q in opts];a=np.array([np.dot(feat.extract(ctx,t),beta) for t in ts]);z=np.array([profile(ctx,t) for t in ts]);offsets.extend(a[0]-a[1:]);diff.extend(z[0]-z[1:])
    d=np.array(diff);o=np.array(offsets);scale=np.maximum(np.sqrt(np.mean(d*d,axis=0)),1e-6);x=d/scale;v=np.zeros(3)
    for _ in range(400):v-=.15*(-x.T@(1/(1+np.exp(np.clip(o+x@v,-40,40))))/len(x)+.02*v)
    gamma=v/scale;rows=[];parity=0.;invariance=0.;changed=0;total=0
    for raw in predictions.read_text().splitlines():
        rec=json.loads(raw)
        if rec['row']!=7:continue
        assert rec['goldIndex']==0 and len(rec['candidates'])==16
        p=byid[rec['query_id'].rsplit(':',1)[0]]; assert rec['candidates'][0]['poem_id']==p['poem_id']
        ctx=[seg(l) for l in p['lines'][:7]];sw=[ctx[i] for i in (0,1,4,5,2,3,6)];recent=[[] for _ in range(6)]+[ctx[6]]
        arms={i:[] for i in ('general','swapped','lastOnly','ordered','orderedSwapped')}
        for j,c in enumerate(rec['candidates']):
            target=byid[c['poem_id']]['lines'][7];assert c['row']==7 and c['line_sha256']==sha(target.encode());t=seg(target)
            a=feat.extract(ctx,t)@beta;b=feat.extract(sw,t)@beta;e=feat.extract(recent,t)@beta;u=a+profile(ctx,t)@gamma;v=b+profile(sw,t)@gamma
            parity=max(parity,abs(a-rec['learnedScores'][j]));invariance=max(invariance,abs(a-b));changed+=int(abs(u-v)>1e-12);total+=1
            for k,value in zip(arms,(a,b,e,u,v)):arms[k].append(float(value))
        rows.append(dict(poem=p['poem_id'],family=p['family_id'],n=p['length'],candidates=[c['poem_id'] for c in rec['candidates']],scores=arms,r1={k:top1(v) for k,v in arms.items()}))
    assert len(rows)==len(dv)==len({r['poem'] for r in rows}) and parity<1e-10 and invariance<1e-12
    stats={}
    for n in (5,7,'pooled'):
        rs=[r for r in rows if n=='pooled' or r['n']==n];fs=[r['family'] for r in rs];d={}
        for a in arms:d[a]=summary([r['r1'][a] for r in rs],fs)
        for a,b in [('ordered','general'),('ordered','orderedSwapped'),('general','lastOnly')]:d[f'{a}-{b}']=summary([r['r1'][a]-r['r1'][b] for r in rs],fs)
        stats[str(n)]=dict(tasks=len(rs),metrics=d)
    rep=dict(schema='verseloom.closing-order-residual.v1',hashes={k:sha(f.read_bytes()) for k,f in dict(model=model,corpus=corpus,ranker=ranker,predictions=predictions).items()},trainFamilies=len(train),trainPairs=len(offsets),trainFallbackNegatives=fallback,intervention='1234567 -> 1256347; line 7 and sixteen candidates fixed',weights=gamma.tolist(),maxErrorAgainstCI=float(parity),baselineMaxSwapDelta=float(invariance),residualChangedScores=changed,candidateScores=total,stats=stats,developmentOnly=True,deployed=False)
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(rep,indent=2,allow_nan=False)+'\n');
    with gzip.open(out/'trials.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(rep))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('model','corpus','ranker','predictions','out'):ap.add_argument('--'+k,type=Path,required=True)
    run(**vars(ap.parse_args()))
