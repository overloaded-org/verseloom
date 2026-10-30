#!/usr/bin/env python3
"""Train a content-neutral continuation ranker; labels are source order, NOT quality.

Only train_candidate fits parameters. dev_reserve is a development diagnostic;
all other partitions are skipped before JSON decoding. Never emit source lines
in the browser model. Candidate generation, meter, corpus and lexicon stay fixed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['poem_association', 'next_line', 'middle_alignment', 'reverse_next_line',
         'same_span_ratio', 'character_overlap', 'multi_segment_overlap', 'next_edge_coverage']
SALT = 'verseloom.continuation.20261030.v1'
PARAMETERS = dict(train_poems_per_form=1024, train_negatives=4, dev_negatives=15,
                  steps=600, learning_rate=0.15, l2=0.02)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def encode(obj): return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
def keyed(text): return sha((SALT+':'+text).encode())
def require(ok, msg):
    if not ok: raise ValueError(msg)


def read_partition(path, allowed):
    rows=[]; skipped=0
    for raw in path.read_bytes().splitlines():
        if not raw.strip(): continue
        match=re.search(rb'"split"\s*:\s*"([^"]+)"', raw)
        require(match is not None, 'Missing partition marker')
        if match[1].decode() not in allowed: skipped+=1; continue
        row=json.loads(raw); require(row['split'] in allowed, 'Conflicting partition')
        require(row['length'] in (5,7) and len(row['lines'])==8, 'Invalid source poem')
        rows.append(row)
    return rows,skipped


class Features:
    def __init__(self, data):
        self.data=data; self.lex=data['lexicon']
        self.bytext={w['text']:(i,math.log(w['prob'])) for i,w in enumerate(self.lex)}
        self.maps={k:{int(u):{v:float(s) for v,c,s in es} for u,es in data[k].items()}
                   for k in ('association','following','alignment')}
        self.cache={}

    def segment(self,text):
        if text in self.cache:return self.cache[text]
        # Same unigram prior, with explicit unknown single characters. No guessed POS.
        paths=[None]*(len(text)+1);paths[0]=(0.0,[])
        for end in range(1,len(text)+1):
            choices=[]
            for width in range(1,min(3,end)+1):
                piece=text[end-width:end]; item=self.bytext.get(piece)
                if not item and width!=1:continue
                base=paths[end-width]
                if base is None:continue
                i,score=item if item else (-1,math.log(1e-12))
                choices.append((base[0]+score,base[1]+[(i,end-width,end)]))
            paths[end]=max(choices,key=lambda t:(t[0],tuple(-x[0] for x in t[1])))
        result=paths[-1][1];self.cache[text]=result;return result

    def maxmean(self,left,right,name):
        table=self.maps[name];total=0.;covered=0
        for v in right:
            best=max((table.get(u,{}).get(v,0.) for u in left if u>=0 and v>=0 and u!=v), default=0.)
            total+=min(4.,best);covered+=best>0
        return total/max(1,len(right)),covered/max(1,len(right))

    def extract(self, context, target):
        # context/target are lists of (id,start,end), including OOV (-1) positions.
        if not context:return [0.]*len(NAMES)
        current=[t[0] for t in target];last=[t[0] for t in context[-1]]
        earlier=sorted(set(t[0] for row in context for t in row))
        a,_=self.maxmean(earlier,current,'association')
        f,cov=self.maxmean(last,current,'following')
        reverse,_=self.maxmean(current,last,'following')
        previous={(start,end):i for i,start,end in context[-1]};same=0;alignment=0.
        for v,start,end in target:
            u=previous.get((start,end)); same+=u is not None
            if u is not None and u>=0 and v>=0:
                alignment+=min(4.,self.maps['alignment'].get(u,{}).get(v,0.))
        den=max(1,len(last),len(current))
        alignment=alignment/den if len(context) in (3,5) else 0.
        leftchars=set(''.join(self.lex[i]['text'] for i in last if i>=0))
        rightchars=set(''.join(self.lex[i]['text'] for i in current if i>=0))
        overlap=len(leftchars&rightchars)/max(1,len(leftchars|rightchars))
        lm={i for i in last if i>=0 and len(self.lex[i]['text'])>1}
        rm={i for i in current if i>=0 and len(self.lex[i]['text'])>1}
        mult=len(lm&rm)/max(1,len(rm))
        return [a,f,alignment,reverse,same/den,overlap,mult,cov]


def select_train(rows):
    selected=[];seen=set()
    for length in (5,7):
        choices=sorted((p for p in rows if p['length']==length),key=lambda p:keyed('train:'+p['poem_id']))
        n=0
        for p in choices:
            if p['family_id'] in seen:continue
            selected.append(p);seen.add(p['family_id']);n+=1
            if n==PARAMETERS['train_poems_per_form']:break
    return selected


def tasks(rows, features, negatives):
    banks=defaultdict(list);general=defaultdict(list)
    def rhyme(text):
        gs=features.data['readings'].get(text[-1],{}).get('rhymeGroup')
        return gs or 'unresolved'
    for p in rows:
        for row in (1,3,5,7):
            rec=(p,row,p['lines'][row]);banks[(p['length'],row,rhyme(rec[2]))].append(rec)
            general[(p['length'],row)].append(rec)
    matrices=[];records=[];fallbacks=0
    for p in sorted(rows,key=lambda p:p['poem_id']):
        for row in (1,3,5,7):
            target=p['lines'][row];qid=p['poem_id']+':'+str(row)
            bank=banks[(p['length'],row,rhyme(target))]
            start=int(keyed(qid+':offset'),16)%max(1,len(bank))
            chosen=[];seen={target};seen_families={p['family_id']}
            for candidates in (bank[start:]+bank[:start],general[(p['length'],row)]):
                for q,j,text in candidates:
                    if q['family_id'] in seen_families or text in seen or text in p['lines']:continue
                    chosen.append((q,j,text));seen.add(text);seen_families.add(q['family_id'])
                    if len(chosen)==negatives:break
                if len(chosen)==negatives:break
                fallbacks+=1
            require(len(chosen)==negatives,'Insufficient distinct distractor families')
            context=[features.segment(line) for line in p['lines'][:row]]
            candidates=[(p,row,target)]+chosen
            matrix=[features.extract(context,features.segment(text)) for q,j,text in candidates]
            matrices.append(matrix)
            records.append(dict(query_id=qid,family=p['family_id'],length=p['length'],row=row,
                candidates=[dict(poem_id=q['poem_id'],family=q['family_id'],row=j,line_sha256=sha(text.encode())) for q,j,text in candidates]))
    return np.array(matrices,dtype=np.float64),records,fallbacks


def fit(x):
    differences=(x[:,0:1,:]-x[:,1:,:]).reshape(-1,len(NAMES))
    scale=np.maximum(np.sqrt(np.mean(differences*differences,axis=0)),1e-6)
    delta=differences/scale;w=np.zeros(len(NAMES),dtype=np.float64)
    # Full-batch pairwise logistic objective, fixed optimization budget.
    for _ in range(PARAMETERS['steps']):
        z=np.clip(delta@w,-40.,40.)
        grad=-(delta.T@(1./(1.+np.exp(z))))/len(delta)+PARAMETERS['l2']*w
        w-=PARAMETERS['learning_rate']*grad
    coefficient=w/scale
    center=float(np.mean(x.reshape(-1,len(NAMES))@coefficient))
    loss=float(np.mean(np.logaddexp(0,-delta@w))+0.5*PARAMETERS['l2']*np.sum(w*w))
    return coefficient,center,loss


def rank_scores(scores):
    # Expected rank metrics under uniform tie-breaking, never privilege gold index 0.
    true=scores[:,0,None];higher=np.sum(scores>true+1e-12,axis=1)
    tied=np.sum(np.abs(scores-true)<=1e-12,axis=1)
    r1=(higher==0)/tied
    mrr=np.array([np.mean(1./np.arange(h+1,h+t+1)) for h,t in zip(higher,tied)])
    return r1,mrr


def summarize(scores, records):
    r1,mrr=rank_scores(scores);out={}
    for length in (5,7):
        groups=defaultdict(list)
        for i,r in enumerate(records):
            if r['length']==length:groups[r['family']].append(i)
        out[str(length)]=dict(tasks=sum(len(v) for v in groups.values()),families=len(groups),
            family_R1=float(np.mean([np.mean(r1[ids]) for ids in groups.values()])),
            family_MRR=float(np.mean([np.mean(mrr[ids]) for ids in groups.values()])))
    return out,r1


def bootstrap_difference(a,b,records):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['family']].append(i)
    values=np.array([np.mean(a[ids]-b[ids]) for ids in groups.values()])
    rng=np.random.default_rng(20261030)
    estimates=np.mean(values[rng.integers(len(values),size=(2000,len(values)))],axis=1)
    return dict(mean=float(values.mean()),ci95=[float(x) for x in np.quantile(estimates,[.025,.975])],families=len(values))


def run(model_path,corpus_path,out):
    require(not out.exists(),'Use a new output directory')
    model_raw=model_path.read_bytes();data=json.loads(model_raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0])
    require(sha(corpus_path.read_bytes())==data['sourceIdentity']['newCandidateCorpus'],'Corpus identity mismatch')
    train_all,_=read_partition(corpus_path,{'train_candidate'});dev,skipped=read_partition(corpus_path,{'dev_reserve'})
    train=select_train(train_all)
    require(not ({p['family_id'] for p in train_all}&{p['family_id'] for p in dev}),'Family contamination')
    f=Features(data)
    print('Building training pairs',len(train),flush=True);tx,tr,tf=tasks(train,f,PARAMETERS['train_negatives'])
    coef,center,loss=fit(tx)
    print('Building development retrieval',len(dev),flush=True);dx,dr,df=tasks(dev,f,PARAMETERS['dev_negatives'])
    fixed=dx[:,:,[0,1,2]]@np.array([.35,.25,.4]);learned=dx@coef
    baseline,bs=summarize(fixed,dr);measured,ls=summarize(learned,dr)
    # Context intervention is run separately without fitting new parameters.
    ranker=dict(schema='verseloom.continuation-ranker.v1',features=NAMES,
        coefficients=[float(x) for x in coef],center=center,sourceIdentity=data['sourceIdentity'],
        training=dict(poems=len(train),contexts=len(tr),pair_differences=len(tr)*PARAMETERS['train_negatives'],parameters=PARAMETERS,loss=loss),
        labels='observed_source_adjacency_not_literary_gold',authoredContentLexemes=0,
        literaryQualityCertified=False,contextualPronunciationCertified=False)
    report=dict(schema='verseloom.continuation-development.v1',modelSha256=sha(model_raw),
        candidateCorpusSha256=sha(corpus_path.read_bytes()),trainerSha256=sha(Path(__file__).read_bytes()),training=ranker['training'],
        devPoems=len(dev),devTasks=len(dr),candidatesPerTask=16,skippedNonDevRows=skipped,
        rhymeBucketFallbacks=dict(train=tf,dev=df),fixed=baseline,learned=measured,
        pairedFamilyDifference=bootstrap_difference(ls,bs,dr),coefficients=ranker['coefficients'],
        independentLiteraryRatings=0,testReserveDecoded=False,
        limitations=['Source adjacency is a self-supervised label, not a literary-quality judgement.',
            'Development set used for this diagnostic, not an independent final claim.',
            'Random negatives may be valid alternative continuations; exact-source retrieval is narrow.',
            'The induction graph includes training poems before ranker fitting; training loss is not held-out evidence.',
            'OOV segments remain unknown; no POS, semantic theme or event script is authored.'])
    out.mkdir(parents=True)
    for name,obj in [('ranker.json',ranker),('report.json',report)]: (out/name).write_bytes(encode(obj))
    with (out/'predictions.jsonl').open('wb') as target:
        for i,row in enumerate(dr):target.write(encode({**row,'fixedScores':fixed[i].tolist(),'learnedScores':learned[i].tolist(),'goldIndex':0}))
    # Fixed parity fixtures contain only IDs/spans, not poem text in the browser asset.
    fixtures=[]
    for p in dev[:16]:
        for row in (1,3,5,7):
            ctx=[f.segment(l) for l in p['lines'][:row]];nxt=f.segment(p['lines'][row])
            fixtures.append(dict(context=ctx,target=nxt,features=f.extract(ctx,nxt)))
    (out/'feature-fixtures.json').write_bytes(encode(fixtures))
    (out/'continuation-data.js').write_bytes(b'/* Self-supervised ranking coefficients; no stored content. */\nglobalThis.VerseLoomContinuationData='+encode(ranker).rstrip()+b';\n')
    np.savez_compressed(out/'development-features.npz',features=dx)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=ROOT/'web/induced-data.js');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.model,a.corpus,a.output)
