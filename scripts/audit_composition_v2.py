#!/usr/bin/env python3
"""Independently audit construction receipts. Does not judge literary merit."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.prosody import all_templates, variants

def load_data(path):
    raw=path.read_text();prefix='globalThis.VerseLoomCompositionData='
    return json.loads(raw.split(prefix,1)[1].rsplit(';',1)[0])
def require(ok,msg):
    if not ok:raise ValueError(msg)

def audit(record,data):
    require(record['engineVersion']=='verseloom.phrase-composition.v2','Wrong engine version')
    require(record['provenance']['sourceIdentity']==data['sourceIdentity'],'Wrong sources')
    for flag in ['preselectedLines','storedOutputLines','strictAntithesisCertified','contextualPronunciationCertified','literaryQualityCertified']:
        require(record['provenance'][flag] is False,'Unsupported provenance claim '+flag)
    if record['status']=='infeasible':
        require(record['lines']==[],'Fallback poem on failure');return {'status':'infeasible','checked':True}
    require(record['status']=='complete','Unexpected status')
    n=record['length'];lines=record['lines'];require(n in (5,7) and len(lines)==8 and all(len(x)==n for x in lines),'Shape mismatch')
    templates={t.template_id:t for t in all_templates(n)};require(record['templateId'] in templates,'Unknown template');template=templates[record['templateId']]
    lex={x['id']:x for x in data['lexicon']};counts=collections.Counter(''.join(lines));require(max(counts.values())<=2,'Character overuse')
    used=set();rhymes=[];total=0
    expected_roles={'SVO':['subject','predicate',None], 'LOC_STATE':['setting','subject','predicate'],
       'LOC_SVO':['setting','subject','predicate',None],'TIME_SVO':['time','subject','predicate',None],
       'SUBJ_LOC_VO':['subject','setting','predicate',None],'LOC_STATE7':['setting','subject','aspect','predicate']}
    actual=[]
    for i,detail in enumerate(record['lineDetails']):
        tokens=detail['tokens'];require(''.join(t['text'] for t in tokens)==lines[i]==detail['text'],'Token reconstruction mismatch')
        require(detail['frame'] in expected_roles,'Unknown frame');roles=expected_roles[detail['frame']];require(len(tokens)==len(roles),'Wrong number of slots')
        localtones='';boundaries=[];offset=0
        for j,t in enumerate(tokens):
            require(t['id'] in lex,'Unknown lexical ID');word=lex[t['id']]
            for key in ['text','kind','entity','count','sources']:require(t[key]==word[key],'Lexical evidence changed '+key)
            if roles[j] is not None:require(t['role']==roles[j],'Wrong role')
            if len(t['text'])>1:require(t['text'] not in used,'Repeated word');used.add(t['text'])
            require(not word['seasons'] or record['context']['season'] in word['seasons'],'Season conflict')
            require(not word['times'] or record['context']['time'] in word['times'],'Time conflict')
            require(len(t['readings'])==len(t['text']),'Missing readings')
            for c,r in zip(t['text'],t['readings']):
                require(r==data['readings'][c],'Changed or unrelated reading')
                require(r['contextuallyCertified'] is False,'False context certification')
                require(r['glossTrace']['status']=='resolved','Unresolved lexical source gloss')
                localtones+=r['tone']
            offset+=len(t['text'])
            if offset<n:boundaries.append(lines[i][offset-1:offset+1])
        require(localtones==detail['tones'],'Wrong tone assignment')
        require(localtones in variants(template.core[i],'common_prefix'),'Unsupported metre')
        if i in template.rhyme_rows:
            last=tokens[-1]['readings'][-1];require(last['tone']=='P' and last['rhymeGroup']==record['rhymeGroup'],'Rhyme mismatch');rhymes.append(lines[i][-1])
        require(len(set(lines[i]))==n,'Repeated character in line')
        verbs=[t for t in tokens if t['kind']=='verb']
        subject=next(t for t in tokens if t['role']=='subject')
        if verbs:
            v=lex[verbs[0]['id']];obj=tokens[-1]
            require(subject['entity'] in v['subjects'] and obj['entity'] in v['objects'],'Predicate selectional restrictions')
            require(obj['role']==v['role'],'Wrong object role')
        score=.4*sum(math.log1p(t['count']) for t in tokens)/len(tokens)+.6*sum(math.log1p(data['bigrams'].get(b,0)) for b in boundaries)/max(1,len(boundaries))
        require(abs(score-detail['score'])<1e-9,'Wrong statistical score');total+=score;actual.append(localtones)
    require(len(set(rhymes))==len(rhymes),'Duplicate rhyme character')
    require(abs(total-record['score'])<1e-9,'Wrong total score')
    require(record['actualTones']==actual,'Wrong tone matrix')
    for a,b in [(2,3),(4,5)]:
        up,down=record['lineDetails'][a:b+1];require(up['frame']==down['frame'],'Different syntax frames')
        require(all(x!=y for x,y in zip(lines[a],lines[b])),'Repeated aligned character')
        for u,l in zip(up['tokens'],down['tokens']):
            require(all(u[k]==l[k] for k in ['kind','role','semantic']),'Mismatched alignment')
            require(len(u['text'])==len(l['text']) and u['text']!=l['text'],'Wrong aligned spans')
        us=next(t for t in up['tokens'] if t['role']=='subject');ls=next(t for t in down['tokens'] if t['role']=='subject')
        require(us['entity']!=ls['entity'],'Same declared subject entity in pair')
    require(record['lineDetails'][2]['frame']!=record['lineDetails'][4]['frame'],'Repeated middle frame')
    return {'status':'pass_declared_constraints','checked':True,'length':n,'readingPositions':8*n,'semanticLabelsIndependentlyCertified':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    data=load_data(ROOT/'web/composition-data.js');results=[]
    for line in a.input.read_text().splitlines():results.append(audit(json.loads(line),data))
    report={'schema':'verseloom.composition-audit.v2','records':len(results),'counts':dict(collections.Counter(x['status'] for x in results)),
            'readingPositions':sum(x.get('readingPositions',0) for x in results),'sourceIdentity':data['sourceIdentity'],
            'inputSha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'independentLiteraryRatings':0,'results':results}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['results','sourceIdentity']}))
if __name__=='__main__':main()
