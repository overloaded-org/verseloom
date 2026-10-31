"""Independent receipt check; no study/scoring code import, no literary grading."""
import argparse, collections as C, hashlib, json, statistics as S
from pathlib import Path

def need(cond,msg):
    if not cond:raise AssertionError(msg)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def check(run,ref,replay):
    metrics=json.loads((run/'metrics.json').read_bytes()); data=rows(run/'pairs.jsonl');prior=rows(ref/'results/line-pairs.jsonl');need(len(data)==len(prior)==10358,'records')
    for x,y in zip(data,prior):
        a,b=x['source'],x['swap'];diff=[i for i in range(len(a)) if a[i]!=b[i]]
        need((x['pid'],x['row'],x['family'],a,b,x['position'],x['sp03'],x['hard'])==(y['id'],y['row'],y['family'],y['source'],y['noise'],y['position'],y['trained'],y['hard']),'archived pair mismatch')
        need(len(diff)==2 and sorted(a)==sorted(b) and a[-1]==b[-1],'not an allowed swap')
        for f,v in x['margins'].items():need(x[f]==(.5 if abs(v)<=1e-10 else float(v>0)),'decision sign')
    for n in (5,7):
        D=metrics['forms'][str(n)]
        for subset in ('all','hard','sym'):
            for fld,obj in D[subset].items():
                fam=C.defaultdict(list);fs=fld.split('-')
                for r in data:
                    if r['n']==n and (subset=='all' or r[subset]):fam[r['family']].append(r[fs[0]]-(r[fs[1]] if len(fs)==2 else 0))
                need(len(fam)==obj['groups'] and sum(map(len,fam.values()))==obj['n'] and abs(S.mean(S.mean(v) for v in fam.values())-obj['mean'])<1e-12,'group metric')
        match=[r for r in rows(run/'matched.jsonl') if r['n']==n];need(len({s for r in match for s in (r['a'],r['b'])})==2*len(match),'matched families')
        need(all(r['lines'][1][-1]==r['lines'][3][-1] for r in match),'matched end');need(abs(S.mean(r['full'] for r in match)-D['matched']['mean'])<1e-12,'matched rate')
    transfer=rows(run/'transfer.jsonl');prior=rows(ref/'results/rerank.jsonl');need(len(transfer)==len(prior)==16,'transfer groups')
    for r,g in zip(transfer,prior):
        opts=g['options'];need(r['before']==g['newChoice'] and r['candidates']==[o['fingerprint'] for o in opts],'candidate pool');i=r['candidates'].index(r['after']);need(r['chosen']==opts[i]['lines'] and hashlib.sha256('/'.join(r['chosen']).encode()).hexdigest()==r['after'],'non-original candidate')
        def rank(v,vs):return sum(x<v for x in vs)+.5*(1+sum(x==v for x in vs))
        a=r['scores'];b=[o['discourse'] for o in opts];winner=min(range(len(a)),key=lambda j:(-rank(a[j],a)-rank(b[j],b),r['candidates'][j]));need(winner==i,'rank winner')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir())};need(set(hashes)=={p.name for p in replay.iterdir()},'replay file set');need(all(hashlib.sha256((replay/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'replay')
    return {'status':'PASS_RECORDS_MACRO_RANKS_REPLAY','pairs':len(data),'matched_pairs':len(rows(run/'matched.jsonl')),'transfer_groups':len(transfer),'file_sha256':hashes,'literary_quality_verified':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('parent',type=Path);p.add_argument('replay',type=Path);a=p.parse_args();print(json.dumps(check(a.run,a.parent,a.replay),indent=2))
