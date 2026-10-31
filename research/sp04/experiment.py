"""SP04: frozen-case research harness. Unpack the SP03 reproduction archive first."""
import argparse, collections as C, functools, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

def sha(b):return hashlib.sha256(b).hexdigest()
def key(*a):return sha('/'.join(map(str,a)).encode())
def check(v,s):
    if not v:raise ValueError(s)
def dump(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False)
def put(p,x):p.write_text(dump(x)+'\n',encoding='utf8')
def write(p,rows):p.write_text(''.join(dump(r)+'\n' for r in rows),encoding='utf8')
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l]

def classes(lines):
    freq=C.Counter(''.join(lines)); ws=sorted(w for w in freq if freq[w]>=6);idx={w:i for i,w in enumerate(ws)}
    cs=sorted(freq,key=lambda w:(-freq[w],w))[:1024];ctx={w:i for i,w in enumerate(cs)};cells=C.Counter()
    for l in lines:
        for i,w in enumerate(l):
            if w not in idx:continue
            for d,off in ((-1,0),(1,len(cs))):
                j=i+d
                if 0<=j<len(l) and l[j] in ctx:cells[idx[w],off+ctx[l[j]]]+=1
    a=sparse.coo_matrix((list(cells.values()),tuple(zip(*cells))),shape=(len(ws),2*len(cs)),dtype=float)
    rows=np.asarray(a.sum(1)).ravel();cols=np.asarray(a.sum(0)).ravel();a.data=np.maximum(0,np.log(a.data*a.data.sum()/(rows[a.row]*cols[a.col])))
    e=TruncatedSVD(n_components=24,random_state=711,n_iter=5).fit_transform(a.tocsr());e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    labels=MiniBatchKMeans(n_clusters=48,random_state=711,n_init=4,batch_size=512,max_iter=100).fit_predict(e)
    return {w:int(v) for w,v in zip(ws,labels)}

class Links:
    def __init__(self):self.p=C.Counter();self.a=C.Counter();self.b=C.Counter();self.n=0
    def add(self,u,v):self.p[u,v]+=1;self.a[u]+=1;self.b[v]+=1;self.n+=1
    @functools.lru_cache(200000)
    def score(self,u,v):
        prior=(self.b[v]+.2)/(self.n+.2*(len(self.b)+1));return math.log1p(self.p[u,v]/(8*prior))-math.log1p(self.a[u]/8)

class Features:
    def __init__(self,poems,old):
        self.old=old;self.role=classes([l for p in poems for l in p['lines']]);self.t=[Links() for _ in range(7)];self.pos=C.Counter()
        for p in poems:
            ll=p['lines'];n=p['length']
            for l in ll:
                c=self.cl(l)
                for i,g in enumerate(c):self.pos[n,i,g]+=1
                for t,seq,d in zip(self.t,[l,l,c,c],[2,3,1,2]):
                    for u,v in zip(seq,seq[d:]):t.add(u,v)
            for a,b in zip(ll[::2],ll[1::2]):
                ca,cb=self.cl(a),self.cl(b)
                for i in range(n):
                    self.t[4].add(a[i],b[i]);self.t[5].add(ca[i],cb[i]);
                    for j in (i-1,i+1):
                        if 0<=j<n:self.t[6].add(ca[i],cb[j])
    def cl(self,l):return [self.role.get(w,-1) for w in l]
    @functools.lru_cache(160000)
    def local(self,l):
        n=len(l);c=self.cl(l);r=list(self.old.line(l))
        for t,seq,d in zip(self.t,[l,l,c,c],[2,3,1,2]):r.append(sum(t.score(u,v) for u,v in zip(seq,seq[d:]))/(n-d))
        return r+[sum(math.log(self.pos[n,i,g]+.2) for i,g in enumerate(c))/n]
    def vec(self,l,m,k):
        a,b=(m,l) if k%2 else (l,m);n=len(a);ca,cb=self.cl(a),self.cl(b)
        lex=sum(self.t[4].score(u,v) for u,v in zip(a,b))/n;role=sum(self.t[5].score(u,v) for u,v in zip(ca,cb))/n
        near=sum(self.t[6].score(ca[i],cb[j]) for i in range(n) for j in (i-1,i+1) if 0<=j<n)/(2*n-2)
        return np.array(self.local(l)+[lex,role,near,role*(2<=k<=5)])

def fit(x,p):
    sc=np.maximum(np.sqrt(p@(x*x)),1e-8);z=x/sc
    def obj(w):
        a=z@w;return p@np.logaddexp(0,-a)+.025*(w@w),-(p*expit(-a))@z+.05*w
    r=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-13,'maxiter':300});check(r.success,r.message);return r.x/sc

def run(bundle,out):
    check(__debug__ and not out.exists(),'need nonoptimized Python and a fresh output directory')
    # Two authenticated SP03 exports differ only in low-level float representation on these input fields.
    versions={'study.py':{'57b4bf4f03ead84c08d4e3d8c65d7b5efd1a6405d8cbfedcdf7cc91f41b70e7f'},'protocol.json':{'a5c25627f3162a9eff892a941570fb84ba305f1f74f7893583144b5f6c039ad5'},'results/line-pairs.jsonl':{'b96dd2fa58c187e7f70e5ca2e4eb6afeb7dc092181a0f6f75d5385ec195b3cc7','7e835a0cca0301361924c76748bd1be16d665784d88d560ed00c7c5628aa81eb'},'results/rerank.jsonl':{'8c3e89404fe3829d01a23b1e94259d5cb2f9cce4d692aae13963e997eaa23eb4','8f7fea21fddf288df6d83d518303acbd46f711706bc3f9790220ac69a000f2af'}}
    for name,allowed in versions.items():check(sha((bundle/name).read_bytes()) in allowed,'unexpected '+name)
    spec=importlib.util.spec_from_file_location('sp03',bundle/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    grp,skipped,data,_=old.inputs(bundle/'inputs',json.loads((bundle/'protocol.json').read_bytes()));ref,cal,dev=[grp[k] for k in ('ref','fit','dev')]
    tones={c:r['tone'] for c,r in data['readings'].items() if r.get('tone') in ('P','Z')};stats=old.Stats(ref);f=Features(ref,stats);X=[];strata=[]
    for p in cal:
        for k,l,z,h in old.cases(p,tones,stats):
            m=p['lines'][k^1];X.append(f.vec(l,m,k)-f.vec(z,m,k));strata.append((len(l),h))
    freq=C.Counter(strata);pw=np.array([1/(len(freq)*freq[t]) for t in strata]);X=np.array(X);wl=fit(X[:,:9],pw);wf=fit(X,pw);out.mkdir(parents=True)
    put(out/'model.json',dict(local=wl.tolist(),full=wf.tolist(),roles=f.role,fitStrata={str(k):v for k,v in freq.items()}));print('fit finished',len(X),flush=True)
    bank={};wrong={};rr=[];cross=[];forms={};transfer=[]
    for n in (5,7):
        a=sorted((p for p in dev if p['length']==n),key=lambda p:key('swap',p['family_id']))
        for i,p in enumerate(a):bank[p['poem_id']]=p;wrong[p['poem_id']]=a[(i+1)%len(a)]
    for r in read(bundle/'results/line-pairs.jsonl'):
        p=bank[r['id']];q=wrong[r['id']];k=r['row'];l,z=r['source'],r['noise'];check(l==p['lines'][k] and p['family_id']!=q['family_id'] and z in old.neighbors(l,tones),'invalid swap')
        m=p['lines'][k^1];v=q['lines'][k^1];d=f.vec(l,m,k)-f.vec(z,m,k);e=f.vec(l,v,k)-f.vec(z,v,k);mar=dict(local=float(d[:9]@wl),full=float(d@wf),wrong=float(e@wf))
        sym=all(stats.bi[g]>0 for t in (l,z) for g in zip(t,t[1:]));rr.append(dict(n=len(l),family=p['family_id'],pid=r['id'],row=k,source=l,swap=z,mate=m,other=v,hard=r['hard'],sym=sym,position=r['position'],sp03=r['trained'],margins=mar,**{name:old.pt(v) for name,v in mar.items()}))
    write(out/'pairs.jsonl',rr)
    for n in (5,7):
        buckets=C.defaultdict(list)
        for p in dev:
            if p['length']==n:
                k=(3,5)[int(key('row',p['family_id']),16)%2];buckets[k,p['lines'][k][-1]].append(p)
        for (k,foot),entries in sorted(buckets.items()):
            a=sorted(entries,key=lambda p:key('pair',p['family_id']))
            for p,q in zip(a[::2],a[1::2]):
                u,v=p['lines'][k-1:k+1];x,y=q['lines'][k-1:k+1];d=f.vec(v,u,k)+f.vec(y,x,k)-f.vec(v,x,k)-f.vec(y,u,k);check(max(abs(d[:9]))<1e-10 and v[-1]==y[-1],'line-only shortcut');g=float(d@wf);cross.append(dict(n=n,k=k,a=p['family_id'],b=q['family_id'],lines=[u,v,x,y],full=old.pt(g),margin=g))
        D={};forms[n]=D
        for name in ('all','hard','sym'):
            a=[r for r in rr if r['n']==n and (name=='all' or r[name])];D[name]={}
            for fld in ('position','sp03','local','full','wrong','full-position','full-sp03','full-local','full-wrong'):
                tags=fld.split('-');D[name][fld]=old.macro([(r['family'],r[tags[0]]-(r[tags[1]] if len(tags)>1 else 0)) for r in a],f'04:{n}:{name}:{fld}')
        cr=[r for r in cross if r['n']==n];check(len({t for r in cr for t in (r['a'],r['b'])})==2*len(cr),'family reuse');D['matched']=old.macro([(r['a']+r['b'],r['full']) for r in cr],f'04:matched:{n}');D['hardGate']=bool(D['sym']['full']['groups']>=50 and D['sym']['full-position']['ci95'][0]>0);D['mateGate']=bool(len(cr)>=100 and D['matched']['ci95'][0]>.5)
    write(out/'matched.jsonl',cross)
    for g in read(bundle/'results/rerank.jsonl'):
        opts=g['options'];ls=[float(np.mean([f.vec(l,L[k^1],k)@wf for k,l in enumerate(L)])) for o in opts for L in [o['lines']]];ranks=rankdata(ls)+rankdata([o['discourse'] for o in opts]);i=min(range(len(opts)),key=lambda j:(-ranks[j],opts[j]['fingerprint']));transfer.append(dict(n=g['n'],seed=g['seed'],before=g['newChoice'],after=opts[i]['fingerprint'],chosen=opts[i]['lines'],scores=ls,candidates=[o['fingerprint'] for o in opts]))
    write(out/'transfer.jsonl',transfer);put(out/'metrics.json',dict(forms=forms,counts={k:len(v) for k,v in grp.items()},skipped=skipped,changed={n:sum(r['n']==n and r['before']!=r['after'] for r in transfer) for n in (5,7)},literaryVotes=0,production=False));print('finished',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.bundle,a.output)
