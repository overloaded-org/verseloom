#!/usr/bin/env python3
"""Same candidate lists, different-family context intervention; development only."""
import argparse,json
from pathlib import Path
import numpy as np
from train_continuation_ranker import Features,read_partition,summarize,encode,sha,require

def run(model_path,corpus_path,experiment,out):
    require(not out.exists(),'Output exists')
    raw=model_path.read_bytes();data=json.loads(raw.decode().split('globalThis.VerseLoomInducedData=',1)[1].rsplit(';',1)[0])
    report=json.loads((experiment/'report.json').read_bytes());ranker=json.loads((experiment/'ranker.json').read_bytes())
    require(sha(raw)==report['modelSha256'],'Model identity mismatch')
    require(sha(corpus_path.read_bytes())==report['candidateCorpusSha256'],'Corpus identity mismatch')
    dev,_=read_partition(corpus_path,{'dev_reserve'});byid={p['poem_id']:p for p in dev}
    groups={n:sorted((p for p in dev if p['length']==n),key=lambda p:p['poem_id']) for n in (5,7)}
    replacement={}
    for items in groups.values():
        for i,p in enumerate(items):
            q=next(items[(i+j)%len(items)] for j in range(1,len(items)) if items[(i+j)%len(items)]['family_id']!=p['family_id'])
            replacement[p['poem_id']]=q
    records=[json.loads(s) for s in (experiment/'predictions.jsonl').read_text().splitlines()]
    features=Features(data);scores=[];normal=[];weights=np.array(ranker['coefficients'])
    for record in records:
        query,row=record['query_id'].rsplit(':',1);row=int(row);q=replacement[query]
        context=[features.segment(line) for line in q['lines'][:row]]
        values=[]
        for candidate in record['candidates']:
            p=byid[candidate['poem_id']];text=p['lines'][candidate['row']]
            require(sha(text.encode())==candidate['line_sha256'],'Candidate text changed')
            values.append(features.extract(context,features.segment(text)))
        scores.append(np.asarray(values)@weights);normal.append(record['learnedScores'])
    same,_=summarize(np.array(normal),records);mismatch,_=summarize(np.array(scores),records)
    constant,_=summarize(np.zeros((len(records),16)),records)
    result=dict(schema='verseloom.continuation-context-control.v1',tasks=len(records),
        normal=same,mismatched=mismatch,constant=constant,candidatesUnchanged=True,
        labelsUnchanged=True,weightsUnchanged=True,testReserveDecoded=False,
        interpretation='Input-context sensitivity on source-order retrieval, not event-causality or literary quality')
    out.write_bytes(encode(result));print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=Path('web/induced-data.js'));p.add_argument('--corpus',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.model,a.corpus,a.experiment,a.output)
