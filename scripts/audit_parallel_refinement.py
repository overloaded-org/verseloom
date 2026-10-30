#!/usr/bin/env python3
"""Independent Python arithmetic and hard-constraint audit; not literary review."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset(path, name):
    text=path.read_text(encoding='utf-8')
    return json.loads(text.split('globalThis.'+name+'=',1)[1].rsplit(';',1)[0])


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def fnv(text):
    h=2166136261
    for c in text:
        h=((h ^ ord(c))*16777619)&0xffffffff
    return h


class Auditor:
    def __init__(self, data, model):
        self.data=data; self.model=model; self.lex=data['lexicon']
        need(model['sourceIdentity']==data['sourceIdentity'],'Scorer/source identity')
        self.maps={name:{int(u):{v:(count,score) for v,count,score in rows} for u,rows in data[name].items()}
                   for name in ('next','association','following','alignment')}
        self.positions={k:dict(v) for k,v in data['positions'].items()}
        self.triples={k:dict(v) for k,v in data['triples'].items()}
        total=sum(w['prob'] for w in self.lex)
        self.prior=[w['prob']/total for w in self.lex]
        self.banned=set(data['trainingLineHashes']);self.local_cache={};self.state_cache={}
        self.checked_states=0

    def local(self, ids, n):
        key=(tuple(ids),n)
        if key in self.local_cache:return self.local_cache[key]
        starts=dict(self.data['start'][str(n)]);ends=dict(self.data['end'][str(n)])
        totalstart=sum(starts.values());score=0.;pos=0;prefix=''
        for index,identifier in enumerate(ids):
            w=self.lex[identifier];size=len(w['text']);last=ids[index-1] if index else None
            edges=self.maps['next'].get(last,{})
            count=starts.get(identifier,0) if last is None else edges.get(identifier,(0,0))[0]
            denominator=totalstart if last is None else sum(c for c,s in edges.values())
            pairp=(count+4*self.prior[identifier])/(denominator+4)
            tri=self.triples.get(str(ids[index-2])+':'+str(last),{}) if index>=2 else {}
            tricount=tri.get(identifier,0)
            need(index==0 or count or tricount,'Unsupported segment transition')
            score+=math.log((tricount+2*pairp)/(sum(tri.values())+2))
            score+=.15*math.log1p(self.positions.get(str(n)+':'+str(pos),{}).get(identifier,0))
            if prefix:
                pair=prefix[-1]+w['text'][0]
                score+=.12*math.log((self.data['charPairs'].get(pair,0)+.1)/
                                   (self.data['charCounts'].get(prefix[-1],0)+.1*len(self.data['charCounts'])))
            pos+=size
            if pos==n:
                need(identifier in ends,'No attested end token')
                score+=.25*math.log1p(ends[identifier])
            prefix+=w['text']
        need(pos==n,'Segment total length')
        self.local_cache[key]=score;return score

    def features(self, context, target):
        if not context:return [0.]*8
        current=target;last=context[-1];earlier=sorted(set(i for row in context for i in row))
        def maxmean(left,right,name):
            total=0.;covered=0
            for v in right:
                best=max((self.maps[name].get(u,{}).get(v,(0,0))[1] for u in left if u!=v),default=0.)
                total+=min(4.,best);covered+=best>0
            return total/max(1,len(right)),covered/max(1,len(right))
        association,_=maxmean(earlier,current,'association')
        following,coverage=maxmean(last,current,'following')
        reverse,_=maxmean(current,last,'following')
        spans={};offset=0
        for i in last:
            end=offset+len(self.lex[i]['text']);spans[(offset,end)]=i;offset=end
        same=0;alignment=0.;offset=0
        for v in current:
            end=offset+len(self.lex[v]['text']);u=spans.get((offset,end));offset=end
            if u is not None:
                same+=1;alignment+=min(4.,self.maps['alignment'].get(u,{}).get(v,(0,0))[1])
        denominator=max(1,len(last),len(current))
        alignment=alignment/denominator if len(context) in (3,5) else 0.
        leftchars=set(''.join(self.lex[i]['text'] for i in last))
        rightchars=set(''.join(self.lex[i]['text'] for i in current))
        overlap=len(leftchars&rightchars)/max(1,len(leftchars|rightchars))
        lm={i for i in last if len(self.lex[i]['text'])>1}
        rm={i for i in current if len(self.lex[i]['text'])>1}
        return [association,following,alignment,reverse,same/denominator,overlap,
                len(lm&rm)/max(1,len(rm)),coverage]

    def check(self, chosen, snapshot):
        n=snapshot['length'];template_id=snapshot['templateId'];rhyme=snapshot['rhymeGroup']
        signature=(n,template_id,rhyme,tuple((l['text'],tuple(l['tokens'])) for l in chosen))
        if signature in self.state_cache:return self.state_cache[signature]
        need(len(chosen)==8,'Not eight rows')
        template=next(t for t in self.data['templates'][str(n)] if t['id']==template_id)
        chars=Counter();seen_tokens=set();texts=set();ends=set();context=[];score=0.;receipts=[]
        for row,line in enumerate(chosen):
            text=line['text'];ids=line['tokens']
            need(len(text)==n and len(set(text))==n and text not in texts,'Shape or repeated line');texts.add(text)
            need(fnv(text) not in self.banned,'Copied training line')
            need(''.join(self.lex[i]['text'] for i in ids)==text,'Token/text mismatch')
            tones=''.join(self.data['readings'][c]['tone'] for c in text)
            need(tones==line['tones'] and tones in template['variants'][row],'Tone/template mismatch')
            if row in template['rhymeRows']:
                need(self.data['readings'][text[-1]]['rhymeGroup']==rhyme and text[-1] not in ends,'Rhyme mismatch/reuse')
                ends.add(text[-1])
            for i in ids:
                if len(self.lex[i]['text'])>1:
                    need(i not in seen_tokens,'Repeated multisyllable segment');seen_tokens.add(i)
            local=self.local(ids,n);need(abs(local-line['score'])<1e-9,'Local likelihood receipt')
            features=self.features(context,ids)
            raw=sum(a*b for a,b in zip(features,self.model['coefficients']))-self.model['center']
            learned=1/(1+math.exp(-max(-40,min(40,raw)))) if context else 0.
            repeated=sum(chars[c] for c in text)
            score+=local/n+learned-.3*repeated
            chars.update(text);context.append(ids)
            receipts.append((features,raw,learned))
        need(max(chars.values())<=2,'Global character repetition')
        self.checked_states+=1;result=(score,receipts);self.state_cache[signature]=result;return result


def run(root,out):
    need(not out.exists(),'Refusing overwrite')
    need(not (root/'INCOMPLETE').exists(),'Run is incomplete')
    repo=Path(__file__).resolve().parents[1]
    identity=json.loads((root/'inputs.json').read_bytes())
    for name,expected in identity['files'].items():
        need(sha(repo/name)==expected,'Input changed since experiment: '+name)
    data=asset(repo/'web/induced-data.js','VerseLoomInducedData')
    model=asset(repo/'web/continuation-data.js','VerseLoomContinuationData')
    audit=Auditor(data,model);runs=0;positions=0;trace_blocks=0;accepted=0
    for n in (5,7):
        folder=root/str(n);snapshots={x['seed']:x for x in read_rows(folder/'candidate-pools.jsonl')}
        baselines={x['options']['seed']:x for x in read_rows(folder/'baseline.jsonl')}
        records=read_rows(folder/'outputs.jsonl')+read_rows(folder/'local-outputs.jsonl')
        for r in records:
            e=r['experiment'];s=snapshots[e['seed']];b=baselines[e['seed']]
            need(r['templateId']==b['templateId']==s['templateId'],'Template changed')
            need(r['rhymeGroup']==b['rhymeGroup']==s['rhymeGroup'],'Rhyme changed')
            need(r['sourceIdentity']==s['sourceIdentity']==data['sourceIdentity'],'Source mismatch')
            need(r['candidateDigest']==s['candidateDigest'],'Candidate pool mismatch')
            chosen=[]
            for row,detail in enumerate(r['lineDetails']):
                ids=[w['id'] for w in detail['tokens']]
                need(detail['text']==r['lines'][row],'Line receipt mismatch')
                for w in detail['tokens']:
                    need(w['text']==data['lexicon'][w['id']]['text'] and w['source']==data['lexicon'][w['id']]['source'],'Token provenance mismatch')
                matches=[l for l in s['pools'][row] if l['text']==detail['text'] and l['tokens']==ids]
                need(len(matches)==1,'Output not in frozen row candidates');chosen.append(matches[0])
            score,receipts=audit.check(chosen,s);need(abs(score-r['score'])<1e-8,'Full objective mismatch')
            need(score>=b['score']-1e-8,'Revision decreased objective')
            if 'rows' in e:
                need(all(i in e['rows'] or r['lines'][i]==b['lines'][i] for i in range(8)),'Local intervention escaped its block')
            for detail,(features,raw,learned) in zip(r['lineDetails'],receipts):
                record=detail['relations']['learned']
                need(all(abs(a-b)<1e-10 for a,b in zip(features,record['features'])),'Feature parity')
                need(abs(raw-record['rawScore'])<1e-9 and abs(learned-detail['relations']['value'])<1e-10,'Learned receipt')
            p=r['provenance'];need(p['authoredContentLexemes']==0 and not p['diffusionModel'] and not p['literaryQualityCertified'],'Unsupported claim')
            runs+=1;positions+=8*n
        for t in read_rows(folder/'traces.jsonl'):
            s=snapshots[t['seed']];previous=baselines[t['seed']]['lines'];previous_score=baselines[t['seed']]['score'];count=0
            for move in t['trace']:
                need(move['before']==previous,'Broken revision trace')
                need(all(i in move['rows'] or before==after for i,(before,after) in enumerate(zip(move['before'],move['after']))),'Off-block modification')
                chosen=[]
                for row,text in enumerate(move['after']):
                    matches=[l for l in s['pools'][row] if l['text']==text]
                    need(len(matches)==1,'Ambiguous trace candidate');chosen.append(matches[0])
                score,_=audit.check(chosen,s)
                need(abs(score-move['afterScore'])<1e-8 and abs(previous_score-move['beforeScore'])<1e-8,'Trace score discrepancy')
                need(score>=previous_score-1e-8,'Trace objective decrease')
                need(move['proposals']<=move['expectedProposals'],'Proposal count')
                count+=move['proposals'];need(count<=4096,'Budget overflow')
                previous=move['after'];previous_score=score;trace_blocks+=1;accepted+=int(move['accepted'])
    result={'schema':'verseloom.parallel-refinement.audit.v1','status':'PASS_TEXT_SOURCE_SCORE_AND_TRACE',
            'optimizedAndLocalOutputs':runs,'characterPositions':positions,'uniqueStatesRecomputed':audit.checked_states,
            'traceBlocks':trace_blocks,'acceptedMoves':accepted,'independentLiteraryRatings':0,
            'strictAntithesisCertified':False,'diffusionModel':False,'literaryQualityCertified':False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
