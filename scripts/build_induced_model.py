#!/usr/bin/env python3
"""Train-only segment/relationship induction: no authored words, topics or syntax.

Statistical segments are not certified words; observed adjacency is not causality.
Historical readings are retained as candidate sets, never picked to fit meter.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.prosody import all_templates, variants
VERSION = 'verseloom.induced-data.v1'
PARAMS = dict(maxSpan=3, minPoems=3, iterations=4, vocabularyBudget=10000, neighbors=12)
GY_SHA256 = None  # Git-object pin below identifies the exact external input.
GY_BLOB = '1b3c5c7f36504a5f187f00f8c2f8e8ee2e3b515e'

def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def packed(x): return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
def fnv(s):
    h=2166136261
    for c in s: h=((h ^ ord(c))*16777619)&0xffffffff
    return h

def training_files(root, bootstrap):
    paths=[(root/'data/corpus_v03/poems.jsonl', 'train'), (bootstrap/'run/candidate_poems.jsonl', 'train_candidate')]
    manifest=json.loads((bootstrap/'run/manifest.json').read_bytes())
    if sha(paths[1][0].read_bytes())!=manifest['files']['candidate_poems.jsonl']['sha256']:
        raise ValueError('Candidate corpus identity mismatch')
    poems=[]
    for path, allowed in paths:
        for raw in path.read_bytes().splitlines():
            if not raw.strip(): continue
            m=re.search(rb'"split"\s*:\s*"([^"]+)"', raw)
            if not m: raise ValueError('Missing split')
            if m[1].decode()!=allowed: continue  # held-out content never JSON-decoded
            p=json.loads(raw)
            if p['split']!=allowed or p['length'] not in (5,7) or len(p['lines'])!=8 or any(len(s)!=p['length'] for s in p['lines']):
                raise ValueError('Malformed training poem')
            poems.append(p)
    if len({p['poem_id'] for p in poems})!=len(poems): raise ValueError('Duplicate training identity')
    return sorted(poems, key=lambda p:p['poem_id']), {name:sha(p.read_bytes()) for name,(p,_) in zip(['oldTrainCorpus','newCandidateCorpus'],paths)}

def discover(poems, params):
    counts=collections.Counter(); supports=collections.Counter(); examples={}
    for p in poems:
        observed=set()
        for row,s in enumerate(p['lines']):
            for a in range(len(s)):
                for n in range(1,min(params['maxSpan'],len(s)-a)+1):
                    w=s[a:a+n]; counts[w]+=1; observed.add(w)
                    examples.setdefault(w, [p['poem_id'],row,a,a+n])
        supports.update(observed)
    candidates=sorted(w for w in counts if len(w)==1 or supports[w]>=params['minPoems'])
    weights={w:float(counts[w]) for w in candidates}
    denom=sum(weights.values()); probs={w:v/denom for w,v in weights.items()}
    lines=[s for p in poems for s in p['lines']]
    paths=[]
    for s in lines:
        paths.append([(a,a+n,s[a:a+n]) for a in range(len(s)) for n in range(1,min(params['maxSpan'],len(s)-a)+1) if s[a:a+n] in probs])
    history=[]
    for _ in range(params['iterations']):
        expected=collections.Counter(); ll=0.0
        for s,edges in zip(lines,paths):
            f=[0.0]*(len(s)+1); f[0]=1.0
            for a,b,w in edges: f[b]+=f[a]*probs[w]
            z=f[-1]
            if not z: raise ValueError('Empty segmentation probability')
            back=[0.0]*(len(s)+1); back[-1]=1.0
            for a,b,w in reversed(edges): back[a]+=probs[w]*back[b]
            for a,b,w in edges: expected[w]+=f[a]*probs[w]*back[b]/z
            ll+=math.log(z)
        denom=sum(expected.values())+1e-8*len(probs)
        probs={w:(expected[w]+1e-8)/denom for w in candidates}
        history.append(round(ll,6))
    return counts,supports,examples,probs,history

def segment(s, probs, width):
    best=[None]*(len(s)+1); best[0]=(0.0,())
    for b in range(1,len(s)+1):
        options=[]
        for a in range(max(0,b-width),b):
            w=s[a:b]
            if w in probs and best[a] is not None:
                options.append((best[a][0]+math.log(probs[w]),best[a][1]+((a,b,w),)))
        if options: best[b]=max(options,key=lambda q:(q[0],q[1]))
    return list(best[-1][1]) if best[-1] else []

def reading_sets(chars, root, bootstrap):
    raw=(bootstrap/'cache/raw/guangyun.csv').read_bytes()
    if blob(raw)!=GY_BLOB: raise ValueError('Guangyun pin mismatch')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))); index=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if len(r['字頭'])==1: index[r['字頭']].append((i,r))
    ref=json.loads((root/'data/reference_v02/pingshui_106_memberships.json').read_bytes())['characters']
    editorial=json.loads((root/'data/reference_v02/editorial_issues.json').read_bytes())
    banned={x['character'] for x in editorial['issues'] if x.get('character')}
    accepted={}; rejected=collections.Counter()
    for c in sorted(chars):
        entries=index.get(c,[]); memberships=ref.get(c,[])
        if c in banned: rejected['editorial_warning']+=1; continue
        if not entries or not memberships: rejected['missing_reference']+=1; continue
        four={r['音韻地位'][-1:] for _,r in entries}
        if not four<=set('平上去入'): rejected['unknown_tone']+=1; continue
        gy={'P' if t=='平' else 'Z' for t in four}; ps={r['tone'] for r in memberships}
        if len(gy)!=1 or gy!=ps: rejected['ambiguous_or_conflicting_tone']+=1; continue
        tone=next(iter(gy)); groups=sorted({m['group_id'] for m in memberships if m['tone']==tone})
        accepted[c]={'tone':tone,'rhymeGroup':groups[0] if tone=='P' and len(groups)==1 else None,
            'memberships':memberships,'candidateEntryIds':['gy:'+str(i+1) for i,_ in entries],
            'selectedEntryId':None,'contextuallyCertified':False,'status':'tone_consensus_only_not_sense_resolution'}
    return accepted,dict(rejected),sha(raw)

def make_model(root, bootstrap, params=None):
    params=dict(PARAMS if params is None else params)
    poems,identities=training_files(root,bootstrap)
    counts,df,examples,probs,history=discover(poems,params)
    readings,rejected,gyhash=reading_sets({c for w in probs for c in w},root,bootstrap)
    eligible=[w for w in probs if all(c in readings for c in w)]
    eligible.sort(key=lambda w:(-probs[w],w))
    chosen=set(eligible[:params['vocabularyBudget']]); words=sorted(chosen); ids={w:i for i,w in enumerate(words)}
    lex=[dict(text=w,count=counts[w],poemCount=df[w],prob=round(probs[w],12),
              tones=''.join(readings[c]['tone'] for c in w),rhymeGroup=readings[w[-1]]['rhymeGroup'],source=examples[w]) for w in words]
    nexts=collections.defaultdict(collections.Counter); starts={5:collections.Counter(),7:collections.Counter()}
    peers=collections.defaultdict(collections.Counter); follows=collections.defaultdict(collections.Counter)
    aligned=collections.defaultdict(collections.Counter); charpairs=collections.Counter(); charcounts=collections.Counter()
    ends={5:collections.Counter(),7:collections.Counter()}; positions=collections.defaultdict(collections.Counter); triples=collections.defaultdict(collections.Counter)
    edge_sources={}; boundary_patterns={5:collections.Counter(),7:collections.Counter()}; observed_df=collections.Counter()
    for p in poems:
        byrow=[]; seen=set()
        for row,s in enumerate(p['lines']):
            spans=segment(s,probs,params['maxSpan']); seq=[ids.get(w) for _,_,w in spans]
            valid=[i for i in seq if i is not None]; byrow.append((spans,seq))
            seen.update(valid)
            if seq and seq[0] is not None: starts[p['length']][seq[0]]+=1
            if seq and seq[-1] is not None: ends[p['length']][seq[-1]]+=1
            for (a,b,w),i in zip(spans,seq):
                if i is not None: positions[str(p['length'])+':'+str(a)][i]+=1
            for j in range(len(seq)-2):
                u,v,w=seq[j:j+3]
                if u is not None and v is not None and w is not None: triples[str(u)+':'+str(v)][w]+=1
            if spans: boundary_patterns[p['length']]['-'.join(str(b-a) for a,b,_ in spans)]+=1
            for j in range(len(seq)-1):
                u,v=seq[j:j+2]
                if u is not None and v is not None:
                    nexts[u][v]+=1;edge_sources.setdefault(('next',u,v),[p['poem_id'],row])
            charcounts.update(s);charpairs.update(s[a:a+2] for a in range(len(s)-1))
        observed_df.update(seen)
        for u in sorted(seen):
            for v in sorted(seen-{u}): peers[u][v]+=1
        for row in range(7):
            for u in {i for i in byrow[row][1] if i is not None}:
                for v in {i for i in byrow[row+1][1] if i is not None and i!=u}:
                    follows[u][v]+=1
        for up in (2,4):
            top,si=byrow[up]; bot,sj=byrow[up+1]; lower={(a,b):j for (a,b,w),j in zip(bot,sj) if j is not None}
            for (a,b,w),u in zip(top,si):
                v=lower.get((a,b))
                if u is not None and v is not None and u!=v: aligned[u][v]+=1
    def table(d, metric):
        out={}
        for u,neighbors in sorted(d.items()):
            def strength(v,c):
                if metric=='count': return math.log1p(c)
                ratio=c*len(poems)/max(1,observed_df[u]*observed_df[v])
                return max(0.0,math.log(max(1,ratio)))*c/(c+3)
            selected=sorted(((v,c,strength(v,c)) for v,c in neighbors.items()),key=lambda r:(-r[2],-r[1],r[0]))[:params['neighbors']]
            out[str(u)]=[[v,c,round(s,6)] for v,c,s in selected if s>0]
        return out
    transitions=table(nexts,'count')
    evidence={f'{u}:{v}':edge_sources[('next',int(u),v)] for u,edges in transitions.items() for v,_,_ in edges}
    sourceids={x['source'][0] for x in lex}|{x[0] for x in evidence.values()}
    pindex={p['poem_id']:i for i,p in enumerate(poems)}
    sources={str(pindex[p['poem_id']]):{'poemId':p['poem_id'],'split':p['split'],'length':p['length'],'sourceRefs':p['source_refs']} for p in poems if p['poem_id'] in sourceids}
    for w in lex: w['source']=[pindex[w['source'][0]],*w['source'][1:]]
    evidence={k:[pindex[v[0]],v[1]] for k,v in evidence.items()}
    model={'schema':VERSION,'parameters':params,'lexicon':lex,'readings':readings,
        'start':{str(n):[[i,c] for i,c in sorted(x.items())] for n,x in starts.items()},
        'end':{str(n):[[i,c] for i,c in sorted(x.items())] for n,x in ends.items()},
        'positions':{k:[[i,c] for i,c in sorted(v.items())] for k,v in sorted(positions.items())},
        'triples':{k:[[i,c] for i,c in v.most_common(12)] for k,v in sorted(triples.items()) if sum(v.values())>=2},
        'next':transitions,'association':table(peers,'pmi'),'following':table(follows,'pmi'),'alignment':table(aligned,'count'),
        'nextEvidence':evidence,'sourcePoems':sources,
        'charPairs':dict(sorted(charpairs.items())),'charCounts':dict(sorted(charcounts.items())),
        'trainingLineHashes':sorted({fnv(s) for p in poems for s in p['lines']}),
        'templates':{str(n):[{'id':t.template_id,'core':t.core,'rhymeRows':t.rhyme_rows,
            'variants':[list(variants(p,'common_prefix')) for p in t.core]} for t in all_templates(n)] for n in (5,7)},
        'sourceIdentity':{**identities,'guangyunSha256':gyhash,
            'referenceSha256':sha((root/'data/reference_v02/pingshui_106_memberships.json').read_bytes()),
            'compilerSha256':sha(Path(__file__).read_bytes())},
        'meta':{'trainingPoems':len(poems),'byLength':dict(collections.Counter(p['length'] for p in poems)),
            'discoveredSegments':len(probs),'admittedSegments':len(lex),'spanLengths':dict(collections.Counter(len(w) for w in words)),
            'excludedCharacters':rejected,'segmentationLogLikelihood':history,
            'observedBoundaryPatterns':{str(n):dict(sorted(c.items())) for n,c in boundary_patterns.items()},
            'authoredContentLexemes':0,'authoredSemanticCategories':0,'authoredSentenceFrames':0,'prewrittenOutputLines':0,
            'relationInterpretation':'observed_cooccurrence_and_order_not_events_or_causality',
            'contextualPronunciationCertified':False,'strictAntithesisCertified':False,'literaryQualityCertified':False}}
    report={'schema':'verseloom.induction-build.v1',**model['meta'],'parameters':params,
        'graphEdges':{k:sum(len(x) for x in model[k].values()) for k in ['next','association','following','alignment']},
        'sourceIdentity':model['sourceIdentity']}
    return model,report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'web/induced-data.js');ap.add_argument('--report',type=Path,default=ROOT/'output/induction-build.json');ap.add_argument('--check',action='store_true')
    a=ap.parse_args(); data,report=make_model(ROOT,a.bootstrap)
    content=b'/* Corpus-induced; no authored content lexicon. See induced.html. */\nglobalThis.VerseLoomInducedData='+packed(data).rstrip()+b';\n'
    report['assetSha256']=sha(content);report['assetBytes']=len(content)
    meta_path=a.output.with_name('induced-meta.js')
    meta=b'globalThis.VerseLoomInducedMeta='+packed({'meta':data['meta'],'sourceIdentity':data['sourceIdentity'],'assetBytes':len(content)}).rstrip()+b';\n'
    for path,b in [(a.output,content),(a.report,packed(report)),(meta_path,meta)]:
        if a.check:
            if path.read_bytes()!=b: raise SystemExit('Non-deterministic build: '+str(path))
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
