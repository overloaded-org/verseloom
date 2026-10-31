"""Recompute all published rates from case receipts without importing the trainer.
Usage: python audit.py INPUTS RUN [REPLAY]. Assertions must be enabled.
"""
import collections,hashlib,io,json,math,sys,tarfile,zipfile
from pathlib import Path
from statistics import mean

def verify(inputs,run,replay=None):
    def read(name):return [json.loads(s) for s in (run/name).read_text().splitlines()]
    rep=json.loads((run/'report.json').read_text()); corpus=zipfile.ZipFile(inputs/'VerseLoom_Quality_Data_v1.zip').read('run/candidate_poems.jsonl');dev={}
    for raw in corpus.splitlines():
        if b'"split":"dev_reserve"' in raw:
            p=json.loads(raw);dev[p['poem_id']]=p
    tar=tarfile.open(fileobj=io.BytesIO(zipfile.ZipFile(inputs/'verseloom-pages-1818d1e.zip').read('artifact.tar')));asset=tar.extractfile('./induced-data.js').read();assert hashlib.sha256(asset).hexdigest()==rep['sourceHashes']['induced_sha256'];tone={k:x['tone'] for k,x in json.loads(asset.decode().split('globalThis.VerseLoomInducedData=')[1].rsplit(';',1)[0])['readings'].items()}
    def macros(rr,col,f):
        g=collections.defaultdict(list)
        for r in rr:g[r[f]].append(r[col]);
        return mean(mean(v) for v in g.values())
    line=read('line-pairs.jsonl');order=read('order-pairs.jsonl');ending=read('ending-pairs.jsonl');rrk=read('rerank.jsonl');supp={}
    assert len(set((r['id'],r['row']) for r in line))==len(line);assert len(set((r['family'],r['n']) for r in order))==len(order)
    for r in line:
        p=dev[r['id']];a,b=r['source'],r['noise'];assert a==p['lines'][r['row']] and p['family_id']==r['family'] and sorted(a)==sorted(b) and a[-1]==b[-1];diff=[i for i in range(len(a)) if a[i]!=b[i]];assert len(diff)==2 and all(tone[a[i]]==tone[b[i]] for i in diff);assert 0<=r['repair']<=1 and math.isclose(r['uniform'],1/r['neighborhood']);assert (float(r['margin']>0) if abs(r['margin'])>1e-10 else .5)==r['trained']
    for n in (5,7):
        groups=set();ll=[r for r in line if r['n']==n];oo=[r for r in order if r['n']==n];ee=[r for r in ending if r['n']==n];res=rep['forms'][str(n)]
        for r in ee:
            p,q=[dev[i] for i in r['ids']];assert p['length']==q['length']==n and p['lines'][-1][-1]==q['lines'][-1][-1]==r['rhyme_char'];assert r['position']==.5;fs={p['family_id'],q['family_id']};assert len(fs)==2 and not fs&groups;groups|=fs
        for name,arr,f in [('line',ll,'family'),('line_hard',[r for r in ll if r['hard']],'family'),('discourse',oo,'family'),('ending_transfer',ee,'pair')]:
            assert len(arr)==res[name]['trained']['n'];assert all(abs(macros(arr,c,f)-res[name][c]['mean'])<1e-12 for c in ('trained','position'))
        amb=[r for r in ll if r['neighborhood']>1];supp[str(n)]={'nontrivialRepairTasks':len(amb),'nontrivialRepairMacro':macros(amb,'repair','family'),'matchedUniformMacro':macros(amb,'uniform','family'),'singleOptionTasks':len(ll)-len(amb)}
    old=zipfile.ZipFile(inputs/'VerseLoom_SP02_Parallel_CI_Evidence.zip'); checked=0
    for r in rrk:
        n=r['n'];group=[]
        for name in ('baseline','outputs'):
            for raw in old.open(f'{n}/{name}.jsonl'):
                o=json.loads(raw);seed=o['experiment']['seed'] if 'experiment' in o else o['options']['seed'];
                if seed==r['seed']:group.append(tuple(o['lines']))
        assert len(group)==5 and set(group)=={tuple(o['lines']) for o in r['options']};checked+=len(group)
        for key,col in [('oldChoice','old'),('newChoice','rankMean')]:assert sorted(r['options'],key=lambda x:(-x[col],x['fingerprint']))[0]['fingerprint']==r[key]
    assert checked==80 and len(rrk)==16
    files=sorted(p.name for p in run.iterdir());matches=None
    if replay:
        assert files==sorted(p.name for p in replay.iterdir());assert all((run/f).read_bytes()==(replay/f).read_bytes() for f in files);matches=len(files)
    return {'status':'PASS','scope':'receipt recomputation, swaps/provenance, family disjoint closing pairs, unchanged generated candidates; not literary review','linePairs':len(line),'orderPairs':len(order),'endingPairs':len(ending),'sourceCandidateEntries':checked,'fileMatchesOnReplay':matches,'supplementary':supp}
if __name__=='__main__':print(json.dumps(verify(*map(Path,sys.argv[1:])),ensure_ascii=False,sort_keys=True,indent=2))
