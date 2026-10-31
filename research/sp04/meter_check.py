"""Exploratory stricter transfer, specified after first paired results. No refit."""
from experiment import *

def main(bundle,run,out):
    check(not out.exists() and __debug__, 'fresh output required')
    check(sha((bundle/'study.py').read_bytes())=='57b4bf4f03ead84c08d4e3d8c65d7b5efd1a6405d8cbfedcdf7cc91f41b70e7f','wrong dependency')
    spec=importlib.util.spec_from_file_location('sp03',bundle/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    g,_,data,_=old.inputs(bundle/'inputs',json.loads((bundle/'protocol.json').read_text()));par=json.loads((run/'model.json').read_bytes());role=par['roles'];w=par['full'][9:];tab=[Links() for _ in range(3)];tones={c:r['tone'] for c,r in data['readings'].items() if r.get('tone') in ('P','Z')}
    for p in g['ref']:
        for a,b in zip(p['lines'][::2],p['lines'][1::2]):
            for i,(u,v) in enumerate(zip(a,b)):
                tab[0].add(u,v);tab[1].add(role.get(u,-1),role.get(v,-1))
                for j in (i-1,i+1):
                    if 0<=j<len(b):tab[2].add(role.get(u,-1),role.get(b[j],-1))
    def score(a,b):
        # Truncate both final glyphs, and every feature involving them. Local terms cancel.
        a,b=a[:-1],b[:-1];n=len(a);ca=[role.get(c,-1) for c in a];cb=[role.get(c,-1) for c in b]
        return w[0]*sum(tab[0].score(u,v) for u,v in zip(a,b))/n+(w[1]+w[3])*sum(tab[1].score(u,v) for u,v in zip(ca,cb))/n+w[2]*sum(tab[2].score(ca[i],cb[j]) for i in range(n) for j in (i-1,i+1) if 0<=j<n)/(2*n-2)
    rec=[];summary={}
    for n in (5,7):
        buck=C.defaultdict(list);selected=[p for p in g['dev'] if p['length']==n];eligible=0
        for p in selected:
            k=(3,5)[int(key('row',p['family_id']),16)%2];lower=p['lines'][k];sig=tuple(tones.get(lower[i]) for i in range(1,n-1,2))
            if None not in sig:buck[k,sig].append(p);eligible+=1
        for (k,sig),arr in sorted(buck.items()):
            arr.sort(key=lambda p:key('meter-control',p['family_id']));
            for p,q in zip(arr[::2],arr[1::2]):
                a,b=p['lines'][k-1:k+1];x,y=q['lines'][k-1:k+1];s=score(a,b)+score(x,y)-score(a,y)-score(x,b);rec.append(dict(n=n,row=k,ids=[p['family_id'],q['family_id']],signature=sig,lines=[a,b,x,y],margin=s,credit=old.pt(s)))
        rows=[r for r in rec if r['n']==n];check(len({i for r in rows for i in r['ids']})==2*len(rows),'duplicate family');summary[n]=dict(eligible=eligible,source=len(selected),paired=len(rows),unpaired=eligible-2*len(rows),result=old.macro([('/'.join(r['ids']),r['credit']) for r in rows],f'meter04-{n}'))
    out.mkdir(parents=True);write(out/'pairs.jsonl',rec);put(out/'report.json',summary);print(dump(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.bundle,a.run,a.out)
