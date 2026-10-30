#!/usr/bin/env python3
"""Separate Python checker for generated text, learned features, and score receipts."""
import argparse,json,math
from pathlib import Path
from collections import Counter
from train_continuation_ranker import Features,encode,require

def fnv(text):
    h=2166136261
    for c in text:h=((h^ord(c))*16777619)&0xffffffff
    return h

def audit(model,ranker,inputs,out):
    require(not out.exists(),'Output exists')
    data=json.loads(model.read_text().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0]);weights=json.loads(ranker.read_bytes())
    f=Features(data);checked=0;complete=0;positions=0;learned_rows=0;banned=set(data['trainingLineHashes'])
    for path in inputs:
        for raw in path.read_text().splitlines():
            r=json.loads(raw);checked+=1
            if r['status']!='complete':require(r['lines']==[],'Failure substituted text');continue
            complete+=1;n=r['length'];lines=r['lines'];ctx=[];score=0.;counts=Counter();word_ids=set()
            require(len(lines)==8 and all(len(l)==n for l in lines),'Shape')
            t=next(t for t in data['templates'][str(n)] if t['id']==r['templateId'])
            require(len({lines[i][-1] for i in t['rhymeRows']})==len(t['rhymeRows']),'Repeated rhyme')
            for row,(text,detail) in enumerate(zip(lines,r['lineDetails'])):
                require(fnv(text) not in banned and len(set(text))==n,'Copied or repeating line')
                require(detail['tones'] in t['variants'][row],'Meter')
                parts=[];spans=[];pos=0
                for token in detail['tokens']:
                    w=data['lexicon'][token['id']];require(token['text']==w['text'],'Changed token')
                    parts.append(w['text']);spans.append((token['id'],pos,pos+len(w['text'])));pos+=len(w['text'])
                    if len(w['text'])>1:require(token['id'] not in word_ids,'Repeated multi segment');word_ids.add(token['id'])
                    for c,reading in zip(w['text'],token['readings']):
                        require(reading['selectedEntryId'] is None and not reading['contextuallyCertified'],'Unsupported pronunciation certification')
                require(''.join(parts)==text,'Token/text discrepancy')
                require(''.join(data['readings'][c]['tone'] for c in text)==detail['tones'],'Tone mismatch')
                if row in t['rhymeRows']:require(data['readings'][text[-1]]['rhymeGroup']==r['rhymeGroup'],'Rhyme mismatch')
                values=f.extract(ctx,spans)
                expected=(.35*values[0]+.25*values[1]+.4*values[2])
                if r['options']['mode']=='learned':
                    learned_rows+=1;rec=detail['relations']['learned'];require(len(rec['features'])==8,'Feature count')
                    require(all(abs(a-b)<1e-12 for a,b in zip(rec['features'],values)),'Feature parity')
                    rawscore=sum(a*b for a,b in zip(values,weights['coefficients']))-weights['center']
                    expected=1/(1+math.exp(-max(-40,min(40,rawscore)))) if ctx else 0
                    require(abs(rawscore-rec['rawScore'])<1e-10,'Logit receipt')
                require(abs(expected-detail['relations']['value'])<1e-10,'Relation score')
                score+=detail['localScore']/n-.3*sum(counts[c] for c in text)+expected-detail['historyPenalty']
                counts.update(text);ctx.append(spans);positions+=n
            require(max(counts.values())<=2,'Character repetition limit');require(abs(score-r['score'])<1e-9,'Whole poem score')
            require(r['provenance']['authoredContentLexemes']==0 and not r['provenance']['literaryQualityCertified'],'Certification claim')
    result=dict(status='PASS_TEXT_CONSTRAINTS_AND_LEARNED_SCORE_RECEIPTS',runs=checked,completed=complete,positions=positions,learnedRows=learned_rows,literaryQualityCertified=False)
    out.write_bytes(encode(result));print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=Path('web/induced-data.js'));p.add_argument('--ranker',type=Path,required=True);p.add_argument('--input',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.model,a.ranker,a.input,a.output)
