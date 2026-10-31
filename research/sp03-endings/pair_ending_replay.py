"""Compare ending-selection policies on identical legal pairs. No typed/prefilled verse."""
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import numpy as np
from train_continuation_ranker import Features

def cp_hash(s):
    h=2166136261
    for c in s:h=((h^ord(c))*16777619)&0xffffffff
    return h

def main(model,ranker,residual,bank,refs,out):
    if out.exists():raise ValueError('Use a new output directory')
    data=json.loads(model.read_text().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]);w=json.loads(ranker.read_bytes());rs=json.loads(residual.read_bytes());beta=w['coefficients'];center=w['center'];gamma=rs['weights'];feat=Features(data);lex=data['lexicon'];banned=set(data['trainingLineHashes'])
    assert w['sourceIdentity']==data['sourceIdentity'] and rs['hashes']['model']==hashlib.sha256(model.read_bytes()).hexdigest()
    table={int(u):{v:x for v,_,x in vs} for u,vs in data['association'].items()}
    ref={(r['length'],r['options']['seed']):r for p in refs for r in json.loads(p.read_bytes())};res=[];cards=[];keys=[];rng=random.Random(20261031)
    def spans(ids):
        pos=0;s=[]
        for i in ids:n=len(lex[i]['text']);s.append((i,pos,pos+n));pos+=n
        return s
    def sig(x):return 1/(1+math.exp(-max(-40,min(40,x-center))))
    def check(lines,paths,template,rhyme):
        counts=collections.Counter(''.join(lines));mult=[i for ids in paths for i in ids if len(lex[i]['text'])>1];ends=[lines[j][-1] for j in template['rhymeRows'] if j<len(lines)]
        return (len(set(lines))==len(lines) and max(counts.values())<=2 and len(set(mult))==len(mult) and len(set(ends))==len(ends)
                and all(data['readings'][c]['rhymeGroup']==rhyme for c in ends)
                and all(len(set(l))==len(l) and cp_hash(l) not in banned and ''.join(lex[i]['text'] for i in ids)==l and ''.join(lex[i]['tones'] for i in ids) in template['variants'][row] for row,(l,ids) in enumerate(zip(lines,paths))))
    for b in json.loads(bank.read_bytes()):
        n,k=b['length'],b['seed'];r=ref[n,k];t=b['template'];rh=b['rhyme'];assert r['poolFingerprint']==b['fp'] and t['id']==r['templateId'] and rh==r['rhymeGroup'];p=r['lines'][:6];ids=[[tok['id'] for tok in d['tokens']] for d in r['lineDetails'][:6]];ctx=list(map(spans,ids));counts=collections.Counter(''.join(p));legal=[]
        earlier=[{u for row in ctx[start:start+2] for u,_,_ in row} for start in (0,2,4)]
        prep=[]
        for lo in b['lower']:
            z=[sum(min(4,max((table.get(u,{}).get(v,0.) for u in left if u!=v),default=0.)) for v in lo['tokens'])/len(lo['tokens']) for left in earlier];resid=float((np.array(z)-np.mean(z))@gamma);prep.append((lo,spans(lo['tokens']),resid))
        for up in b['upper']:
            us=spans(up['tokens']);uptext=up['text'];p7=p+[uptext];paths7=ids+[up['tokens']]
            if not check(p7,paths7,t,rh):continue
            score_u=up['localScore']/n-.3*sum(counts[c] for c in uptext)+sig(float(np.dot(feat.extract(ctx,us),beta)));counts7=counts+collections.Counter(uptext)
            for lo,ls,dr in prep:
                lotext=lo['text'];lines=p7+[lotext]
                if not check(lines,paths7+[lo['tokens']],t,rh):continue
                raw=float(np.dot(feat.extract(ctx+[us],ls),beta));fixed=score_u+lo['localScore']/n-.3*sum(counts7[c] for c in lotext);legal.append((uptext,lotext,score_u,fixed+sig(raw),fixed+sig(raw+dr)))
        assert any(list(x[:2])==r['lines'][6:] for x in legal)
        # Greedy selects the best viable penultimate line before comparing closings;
        # joint policies compare exactly the same full legal pair set, with no extra noise.
        gu=min(legal,key=lambda x:(-x[2],x[0]))[0];g=min((x for x in legal if x[0]==gu),key=lambda x:(-x[3],x[:2]));j=min(legal,key=lambda x:(-x[3],x[:2]));s=min(legal,key=lambda x:(-x[4],x[:2]));arms={a:list(x[:2]) for a,x in [('greedy',g),('joint',j),('joint_ordered',s)]}
        pairhash=hashlib.sha256(json.dumps(sorted(x[:2] for x in legal),ensure_ascii=False,separators=(',',':')).encode()).hexdigest();res.append(dict(length=n,seed=k,context=p,template=t['id'],rhyme=rh,pool=b['fp'],legal=len(legal),pairHash=pairhash,arms=arms,original=r['lines'][6:]))
        for arm,a2 in [('greedy','joint'),('joint','joint_ordered')]:
            if arms[arm]==arms[a2]:continue
            ab=[arm,a2];rng.shuffle(ab);cid=hashlib.sha256(f'blind/{n}/{k}/{arm}/{a2}'.encode()).hexdigest()[:12];cards.append(dict(card=cid,form=n,context=p,A=arms[ab[0]],B=arms[ab[1]]));keys.append(dict(card=cid,A=ab[0],B=ab[1],seed=k))
    assert len(res)==len(ref)==16
    rng.shuffle(cards);out.mkdir(parents=True)
    for fn,obj in [('trials.json',res),('blind.json',cards),('key.json',keys)]: (out/fn).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    info={}
    for n in (5,7):
        sel=[s for s in res if s['length']==n];info[str(n)]={'trials':len(sel),'legalTotal':sum(s['legal'] for s in sel),'min':min(s['legal'] for s in sel),'max':max(s['legal'] for s in sel),'greedyToJoint':sum(s['arms']['greedy']!=s['arms']['joint'] for s in sel),'jointToOrdered':sum(s['arms']['joint']!=s['arms']['joint_ordered'] for s in sel)}
    (out/'report.json').write_text(json.dumps(dict(summary=info,blindCards=len(cards),independentRatings=0,liveDefaultChanged=False),indent=2)+'\n');print(info,'blind cards',len(cards))
if __name__=='__main__':
    ap=argparse.ArgumentParser();
    for k in ('model','ranker','residual','bank','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--refs',type=Path,nargs='+',required=True);main(**vars(ap.parse_args()))
