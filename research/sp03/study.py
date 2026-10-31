"""VerseLoom SP-03: self-supervised corruption discrimination; not literary gold."""
import argparse, collections as C, functools, hashlib, io, itertools, json, math, re, tarfile, zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
SALT='VL-SP03-20261031-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
def h(*s):return int(sha((':'.join([SALT,*map(str,s)])).encode()),16)
def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def save(p,x):p.write_text(dump(x)+'\n',encoding='utf8')
def rows(p,rr):p.write_text(''.join(dump(x)+'\n' for x in rr),encoding='utf8')
def inputs(root,pr):
    corp=zipfile.ZipFile(root/'VerseLoom_Quality_Data_v1.zip').read('run/candidate_poems.jsonl');assert sha(corp)==pr['source_corpus_sha256']
    groups={'ref':[],'fit':[],'dev':[]};skipped=C.Counter()
    for raw in corp.splitlines():
        spl=re.search(rb'"split":"([^"]+)"',raw)[1].decode()
        if spl not in ('train_candidate','dev_reserve'):skipped[spl]+=1;continue
        p=json.loads(raw);assert p['split']==spl and len(p['lines'])==8 and all(len(l)==p['length'] for l in p['lines'])
        k='dev' if spl=='dev_reserve' else ('fit' if h(p['family_id'])%4==0 else 'ref');groups[k].append(p)
    sets=[{p['family_id'] for p in arr} for arr in groups.values()];assert not any(a&b for a,b in itertools.combinations(sets,2))
    u={}
    for p in sorted(groups['dev'],key=lambda p:h('select',p['poem_id'])):u.setdefault((p['family_id'],p['length']),p)
    groups['dev']=list(u.values())
    tar=tarfile.open(fileobj=io.BytesIO(zipfile.ZipFile(root/'verseloom-pages-1818d1e.zip').read('artifact.tar')));raw=tar.extractfile('./induced-data.js').read();assert sha(raw)==pr['induced_sha256'];data=json.loads(raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]);assert data['sourceIdentity']['newCandidateCorpus']==sha(corp)
    pkg=(root/'VerseLoom_SP02_Parallel_CI_Evidence.zip').read_bytes();assert sha(pkg)==pr['sp02_zip_sha256'];return groups,dict(skipped),data,zipfile.ZipFile(io.BytesIO(pkg))
def fit(d):
    x=np.asarray(d,float);scale=np.maximum(np.sqrt((x*x).mean(0)),1e-8);z=x/scale
    def fun(w):
        m=z@w;return np.logaddexp(0,-m).mean()+.025*(w@w),-z.T@expit(-m)/len(z)+.05*w
    r=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-9,'maxiter':300});assert r.success,r.message;return r.x/scale
def pt(x):return .5 if abs(x)<=1e-10 else float(x>0)
def macro(v,label):
    g=C.defaultdict(list)
    for k,x in v:g[k].append(float(x))
    a=np.asarray([np.mean(g[k]) for k in sorted(g)]);n=len(a)
    if not n:return dict(n=0,groups=0,mean=None,ci95=None)
    rng=np.random.default_rng(h('boot',label)%2**32);e=np.concatenate([a[rng.integers(n,size=(100,n))].mean(1) for _ in range(20)])
    return dict(n=len(v),groups=n,mean=float(a.mean()),ci95=np.quantile(e,[.025,.975]).tolist())
def neighbors(line,tones):
    ret=set()
    for i,j in itertools.combinations(range(len(line)-1),2):
        a,b=line[i],line[j]
        if a==b or a not in tones or b not in tones or tones[a]!=tones[b]:continue
        q=list(line);q[i],q[j]=q[j],q[i];ret.add(''.join(q))
    return sorted(ret)
def mid_end(L):return tuple(L[:2]+L[6:8]+L[4:6]+L[2:4])
def grams(l):return set(zip('^'+l,l+'$'))
class Stats:
    def __init__(self,poems):
        self.bi=C.Counter();self.left=C.Counter();self.right=C.Counter();self.tri=C.Counter();self.pref=C.Counter();self.lp=C.Counter();self.dp=C.Counter();self.char=C.Counter();self.edge=C.Counter();self.ec1=C.Counter();self.ec2=C.Counter();self.etot=0
        for p in poems:
            n=p['length'];L=p['lines']
            for row,l in enumerate(L):
                st='^^'+l+'$'
                for a,b,c in zip(st,st[1:],st[2:]):self.tri[a,b,c]+=1;self.pref[a,b]+=1;self.bi[b,c]+=1;self.left[b]+=1;self.right[c]+=1
                for k,c in enumerate(l):self.lp[n,k,c]+=1
                for c in l[:-1]:self.dp[n,row,c]+=1;self.char[n,c]+=1
            blocks=[set(L[k][:-1]+L[k+1][:-1]) for k in range(0,8,2)]
            for a,b in zip(blocks,blocks[1:]):
                self.edge.update(itertools.product(a,b));self.etot+=len(a)*len(b)
                for c in a:self.ec1[c]+=len(b)
                for c in b:self.ec2[c]+=len(a)
        self.total=sum(self.right.values());self.V=len(self.right)+1
    @functools.lru_cache(maxsize=160000)
    def line(self,l):
        ps=[];ts=[];st='^^'+l+'$';n=len(l)
        for a,b,c in zip(st,st[1:],st[2:]):
            p0=(self.right[c]+.1)/(self.total+.1*self.V);p1=(self.bi[b,c]+8*p0)/(self.left[b]+8);p2=(self.tri[a,b,c]+3*p1)/(self.pref[a,b]+3);ps.append(math.log(p1));ts.append(math.log(p2))
        return np.array([sum(math.log(self.lp[n,k,c]+.2) for k,c in enumerate(l))/n,sum(ps)/len(ps),sum(ts)/len(ts),min(ps)])
    @functools.lru_cache(maxsize=350000)
    def link(self,a,b):
        pr=(self.ec2[b]+1)/(self.etot+len(self.ec2)+1);return max(-4.,min(4.,math.log1p(self.edge[a,b]/(100*pr))-math.log1p(self.ec1[a]/100)))
    @functools.lru_cache(maxsize=100000)
    def edgefeat(self,A,B):
        a,b=sorted(set(A)),sorted(set(B));m=[[self.link(x,y) for x in a] for y in b];rev=[max(self.link(y,x) for x in a) for y in b];vs=[x for r in m for x in r];return np.array([sum(vs)/len(vs),sum(max(r) for r in m)/len(m),sum(rev)/len(rev),sum(x>0 for x in vs)/len(vs)])
    @functools.lru_cache(maxsize=50000)
    def discourse(self,L):
        n=len(L[0]);blocks=[L[k][:-1]+L[k+1][:-1] for k in range(0,8,2)];pos=sum(math.log((self.dp[n,r,c]+.5)/(self.char[n,c]+4)) for r,l in enumerate(L) for c in l[:-1])/(8*(n-1));es=np.array([self.edgefeat(x,y) for x,y in zip(blocks,blocks[1:])]);skip=np.array([self.edgefeat(x,y) for x,y in zip(blocks,blocks[2:])]);return np.r_[pos,es.mean(0),es[-1,:2],skip.mean(0)[:2]]
def cases(p,tones,stats):
    for k,l in enumerate(p['lines']):
        ns=neighbors(l,tones)
        if ns:
            alt=ns[h('local',p['poem_id'],k)%len(ns)];added=grams(alt)-grams(l);yield k,l,alt,bool(added) and all(stats.bi[x]>0 for x in added)
def cross_pairs(dev,n):
    buckets=C.defaultdict(list)
    for p in dev:
        if p['length']==n:buckets[p['lines'][-1][-1]].append(p)
    for key,arr in sorted(buckets.items()):
        arr.sort(key=lambda p:h('cross',p['poem_id']))
        for a,b in zip(arr[::2],arr[1::2]):assert a['family_id']!=b['family_id'];yield a,b

def metrics(rr,group,base,cols,label):
    result={c:macro([(r[group],r[c]) for r in rr],label+c) for c in cols};result['gain_vs_'+base]=macro([(r[group],r['trained']-r[base]) for r in rr],label+'gain');return result
def run(root,out,pr):
    groups,skipped,data,old=inputs(root,pr);tones={k:r['tone'] for k,r in data['readings'].items() if r.get('tone') in ('P','Z')};s=Stats(groups['ref']);train=groups['fit'];dev=groups['dev'];print('reference/fit/dev',*[len(v) for v in groups.values()],flush=True)
    X=np.array([s.line(l)-s.line(alt) for p in train for _,l,alt,_ in cases(p,tones,s)]);w=fit(X);Y=np.array([s.discourse(tuple(p['lines']))-s.discourse(mid_end(p['lines'])) for p in train]);v=fit(Y);vc=fit(Y[:,1:]);wp=fit(X[:,:1]);vp=fit(Y[:,:1]);wb=fit(X[:,1:2]);print('models fitted',len(X),len(Y),flush=True)
    line=[];poem=[];cross=[]
    for p in dev:
        ll=p['lines'];base=dict(id=p['poem_id'],family=p['family_id'],n=p['length']);L=tuple(ll);R=mid_end(ll);d=s.discourse(L)-s.discourse(R);exact=all(a==b or (a in tones and b in tones and tones[a]==tones[b]) for x,y in zip(L,R) for a,b in zip(x,y));poem.append(dict(**base,exact_tone=exact,trained=pt(d@v),position=pt(d[:1]@vp),context=pt(d[1:]@vc),margin=float(d@v)))
        for k,l,alt,hard in cases(p,tones,s):
            a=s.line(l)-s.line(alt);ns=neighbors(alt,tones);assert l in ns and l[-1]==alt[-1] and sorted(l)==sorted(alt) and [tones.get(x) for x in l]==[tones.get(x) for x in alt]
            scores=[float(s.line(t)@w) for t in ns];mx=max(scores);good=[t for t,j in zip(ns,scores) if j>=mx-1e-10];line.append(dict(**base,row=k,noise=alt,source=l,hard=hard,trained=pt(a@w),position=pt(a[:1]@wp),bigram=pt(a[1:2]@wb),repair=(1/len(good) if l in good else 0.),uniform=1/len(ns),neighborhood=len(ns),margin=float(a@w)))
    for n in (5,7):
        for a,b in cross_pairs(dev,n):
            A=tuple(a['lines']);B=tuple(b['lines']);U=A[:6]+B[6:];V=B[:6]+A[6:];d=s.discourse(A)+s.discourse(B)-s.discourse(U)-s.discourse(V);assert abs(d[0])<1e-10
            cross.append(dict(n=n,pair=a['family_id']+'/'+b['family_id'],ids=[a['poem_id'],b['poem_id']],rhyme_char=A[-1][-1],trained=pt(d@v),context=pt(d[1:]@vc),position=pt(d[:1]@vp),margin=float(d@v)))
    report={'counts':{k:{str(n):sum(p['length']==n for p in arr) for n in (5,7)} for k,arr in groups.items()},'notDecoded':skipped,'fitExamples':[len(X),len(Y)],'coefficients':{'local':w.tolist(),'discourse':v.tolist(),'discourse_context_only':vc.tolist(),'local_position':wp.tolist(),'local_bigram':wb.tolist(),'discourse_position':vp.tolist()},'forms':{}}
    for n in (5,7):
        la=[r for r in line if r['n']==n];pa=[r for r in poem if r['n']==n];cr=[r for r in cross if r['n']==n];D={};lab=str(n)
        for key,rr in [('line',la),('line_hard',[r for r in la if r['hard']])]:D[key]=metrics(rr,'family','position',['trained','position','bigram','repair','uniform'],lab+key)
        for key,rr in [('discourse',pa),('discourse_exact_tone',[r for r in pa if r['exact_tone']])]:D[key]=metrics(rr,'family','position',['trained','position','context'],lab+key)
        D['ending_transfer']=metrics(cr,'pair','position',['trained','position','context'],lab+'transfer');D['line_skipped']=8*len(pa)-len(la);D['unpaired_ending_poems']=len(pa)-2*len(cr)
        D['line_gate']=bool(D['line']['trained']['ci95'][0]>.55 and D['line']['gain_vs_position']['ci95'][0]>0 and D['line_hard']['trained']['groups']>=50 and D['line_hard']['trained']['ci95'][0]>.50)
        D['discourse_gate']=bool(D['discourse']['gain_vs_position']['ci95'][0]>0 and len(cr)>=100 and D['ending_transfer']['trained']['ci95'][0]>.50);report['forms'][lab]=D
    report['evaluation']='development; no independent literary ratings';report['production']='STOP';report['sourceHashes']={k:pr[k] for k in ('source_corpus_sha256','induced_sha256','sp02_zip_sha256')};rows(out/'line-pairs.jsonl',line);rows(out/'order-pairs.jsonl',poem);rows(out/'ending-pairs.jsonl',cross);report['transfer']=transfer(old,data,s,w,v,out);save(out/'report.json',report);return report

def fnv(l):
    x=2166136261
    for c in l:x=((x^ord(c))*16777619)&0xffffffff
    return x

def transfer(old,data,s,w,v,out):
    banned=set(data['trainingLineHashes']);res=[];review=[];key=[];mods=[]
    for n in (5,7):
        obs=[json.loads(raw) for name in ['baseline','outputs'] for raw in old.open(f'{n}/{name}.jsonl')];gid=C.defaultdict(list)
        for r in obs:gid[r['experiment']['seed'] if 'experiment' in r else r['options']['seed']].append(r)
        for seed,rr in sorted(gid.items()):
            assert len(rr)==5 and all(r['status']=='complete' and r['templateId']==rr[0]['templateId'] and r['rhymeGroup']==rr[0]['rhymeGroup'] for r in rr);d={}
            for r in rr:
                L=tuple(r['lines']);tpl=next(t for t in data['templates'][str(n)] if t['id']==r['templateId']);R=tpl['rhymeRows'];assert len(L)==len(set(L))==8 and all(len(l)==len(set(l))==n and fnv(l) not in banned for l in L) and max(C.Counter(''.join(L)).values())<=2
                assert all(''.join(data['readings'][c]['tone'] for c in l) in tpl['variants'][i] for i,l in enumerate(L));assert len(set(L[i][-1] for i in R))==len(R) and all(data['readings'][L[i][-1]]['rhymeGroup']==r['rhymeGroup'] for i in R)
                z=d.setdefault(L,dict(lines=L,old=float(r['score']),line=float(np.mean([s.line(l)@w for l in L])),discourse=float(s.discourse(L)@v),arms=[]));z['arms'].append(r.get('experiment',{}).get('arm','baseline'));z['old']=max(z['old'],float(r['score']))
            opts=list(d.values());scores=(rankdata([r['line'] for r in opts])+rankdata([r['discourse'] for r in opts]))/2
            for r,x in zip(opts,scores):r['rankMean']=float(x);r['fingerprint']=sha('/'.join(r['lines']).encode());
            best=lambda name:sorted(opts,key=lambda r:(-r[name],r['fingerprint']))[0];p=best('old');q=best('rankMean');changed=p['lines']!=q['lines'];res.append(dict(n=n,seed=seed,options=opts,oldChoice=p['fingerprint'],newChoice=q['fingerprint'],changed=changed));mods.append((n,changed,len(opts)))
            order=h('blind',n,seed)%2;sides=[p,q][::1 if order==0 else -1];item=sha(f'sp03:{n}:{seed}'.encode())[:12];review.append(dict(item=item,length=n,A=sides[0]['lines'],B=sides[1]['lines']));key.append(dict(item=item,seed=seed,A='old' if order==0 else 'new',sameText=not changed))
    rows(out/'rerank.jsonl',res);rows(out/'blind-cards.jsonl',sorted(review,key=lambda r:r['item']));rows(out/'blind-key.jsonl',key)
    return {str(n):{'initializations':sum(k==n for k,c,t in mods),'changed':sum(k==n and c for k,c,t in mods),'uniqueWithinSets':sum(t for k,c,t in mods if k==n),'originalEntries':40} for n in (5,7)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);pr=json.loads((Path(__file__).parent/'protocol.json').read_text());r=run(a.inputs,a.output,pr);print({k:(v['line_gate'],v['discourse_gate']) for k,v in r['forms'].items()})
