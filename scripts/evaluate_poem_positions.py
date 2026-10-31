#!/usr/bin/env python3
"""Offline SP-02: frozen-ranker diagnostics on dev_reserve, not generation."""
import argparse, gzip, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from train_continuation_ranker import Features, NAMES, keyed, read_partition, require, sha


def credit(x):
    x = np.asarray(x, float)
    require(len(x) >= 2 and np.isfinite(x).all(), 'Missing or nonfinite scores')
    return float(not np.any(x > x[0]+1e-12)) / int(np.sum(abs(x-x[0]) <= 1e-12))


def candidates(p, row, bank, broad, pair=False):
    target = lambda q: tuple(q['lines'][6:]) if pair else q['lines'][row]
    key = p['poem_id']+':'+('tail-pair' if pair else str(row))
    offset = int(keyed(key+':offset'),16) % len(bank)
    picked=[p]; values={target(p)}; families={p['family_id']}
    for pool, fallback in ((bank[offset:]+bank[:offset],False),(broad,True)):
        for q in pool:
            val=target(q)
            if q['family_id'] in families or val in values: continue
            if any(l in p['lines'] for l in (val if pair else (val,))): continue
            picked.append(q); values.add(val); families.add(q['family_id'])
            if len(picked)==16: return [q['poem_id'] for q in picked],fallback
    raise ValueError('Insufficient distinct families')


def run(a):
    model=a.model.read_bytes(); data=json.loads(model.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0])
    ranker=json.loads(a.ranker.read_bytes()); w=np.array(ranker['coefficients']); fixed=np.array([.35,.25,.4,0.,0.,0.,0.,0.])
    require(ranker['sourceIdentity']==data['sourceIdentity'] and ranker['features']==NAMES and len(w)==8 and np.isfinite(w).all(),'Model mismatch')
    require(sha(a.corpus.read_bytes())==data['sourceIdentity']['newCandidateCorpus'],'Corpus mismatch')
    require(not a.output.exists(),'Output exists'); a.output.mkdir(parents=True)
    F=Features(data); fixtures=json.loads(a.fixtures.read_bytes())
    for v in fixtures: require(np.allclose(F.extract(v['context'],v['target']),v['features'],rtol=0,atol=1e-12),'Feature mismatch')
    with a.predictions.open(encoding='utf8') as f: refs={r['query_id']:r for r in map(json.loads,f)}
    poems,skipped=read_partition(a.corpus,{'dev_reserve'}); banks=defaultdict(list); broad=defaultdict(list)
    def bucket(p,row): return data['readings'].get(p['lines'][row][-1],{}).get('rhymeGroup') or 'unresolved'
    for p in poems:
        for i in range(8): banks[p['length'],i,bucket(p,i)].append(p); broad[p['length'],i].append(p)
    seg={p['poem_id']:[F.segment(l) for l in p['lines']] for p in poems}; seen=set(); groups=defaultdict(lambda:defaultdict(list)); fallbacks=defaultdict(int)
    def feature(ctx,t): return np.asarray(F.extract(ctx,t))
    def emit(sink,p,task,ids,arms,fb=False):
        k=(p['length'],task); groups[k][p['family_id']].append([credit(s) for s in arms]); fallbacks[k]+=int(fb)
        sink.write(json.dumps([p['poem_id'],p['family_id'],p['length'],task,ids,[list(s) for s in arms],bool(fb)],separators=(',',':'))+'\n')
    with gzip.open(a.output/'trials.jsonl.gz','wt',encoding='utf8') as sink:
        for p in sorted(poems,key=lambda p:p['poem_id']):
            pid=p['poem_id']; n=p['length']; P=seg[pid]
            for i in range(8):
                ids,fb=candidates(p,i,banks[n,i,bucket(p,i)],broad[n,i]); X=np.asarray([feature(P[:i],seg[q][i]) if i else feature([seg[q][0]],P[1]) for q in ids]); A=X@fixed; B=X@w
                if i in (1,3,5,7):
                    k=pid+':'+str(i); ref=refs[k]; require(ids==[x['poem_id'] for x in ref['candidates']],'Candidate mismatch')
                    require(np.allclose(A,ref['fixedScores'],atol=1e-10,rtol=0) and np.allclose(B,ref['learnedScores'],atol=1e-10,rtol=0),'Score mismatch');seen.add(k)
                emit(sink,p,'line'+str(i+1),ids,[A,B],fb)
            ids,fb=candidates(p,7,banks[n,7,bucket(p,7)],broad[n,7],True); isolated=[]; attached=[]
            for q in ids:
                u,v=seg[q][6:]; isolated.append(feature([u],v)@w); attached.append((feature(P[:6],u)+feature(P[:6]+[u],v))@w)
            emit(sink,p,'tail_pair',ids,[isolated,attached],fb)
            s=[]
            for perm in (range(8),(0,1,4,5,2,3,6,7)):
                Q=[P[j] for j in perm]; total=sum(feature(Q[:j],Q[j]) for j in range(1,8)); s.append([total@fixed,total@w])
            emit(sink,p,'middle_swap',['source','swap'],np.array(s).T)
    require(seen==set(refs),'Reference set incomplete')
    report={'schema':'verseloom.all-position-probe.v1','devPoems':len(poems),'skippedNonDev':skipped,'families':len({p['family_id'] for p in poems}),'referenceQueries':len(seen),'featureFixtures':len(fixtures),'sources':{k:sha(getattr(a,k).read_bytes()) for k in ('model','ranker','corpus','predictions','fixtures')},'tasks':{},'trained':False,'deployed':False,'independentLiteraryRatings':0}
    for (n,task),families in sorted(groups.items()):
        v=np.array([np.mean(families[k],axis=0) for k in sorted(families)]); d=v[:,1]-v[:,0]; rng=np.random.default_rng(20261031); boot=np.mean(d[rng.integers(len(v),size=(2000,len(v)))],axis=1)
        report['tasks'][f'{n}:{task}']=dict(poems=sum(map(len,families.values())),families=len(v),A=float(v[:,0].mean()),B=float(v[:,1].mean()),delta=float(d.mean()),deltaCI95=np.quantile(boot,[.025,.975]).tolist(),fallbacks=fallbacks[n,task])
    report['arms']={'line1':'Reverse opening retrieval given line 2; not free generation.','line2..8':'Fixed vs general learned raw score; same 16 options and true prefix.','tail_pair':'Isolated intact pair vs attached to true first six; two raw transition scores in attached arm.','middle_swap':'Original vs swapped intact middle couplets; sum over seven transitions. A=fixed, B=learned. Chance 1/2.','limits':'Development only, macro over families. Ties share rank 1. Intervals for paired differences only, not multiple-testing adjusted. Not a literary judgement.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(report))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model','ranker','corpus','predictions','fixtures','output'): ap.add_argument('--'+key,type=Path,required=True)
    run(ap.parse_args())
