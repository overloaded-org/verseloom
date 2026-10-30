#!/usr/bin/env python3
"""Compile lexical evidence for constrained phrase composition (no stored lines).

Run with the pinned quality-data artifact's cache/run directory. Only new
train_candidate and old train texts supply statistics. Interpretive tags are
assistant-authored, never independent gold. A phonological candidate is selected
by intended sense BEFORE and independently of a metrical template.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.prosody import all_templates, variants
GY_REV='21585e22c8a730ca2fd175112f4d18e16d5ce578'
GY_BLOB='1b3c5c7f36504a5f187f00f8c2f8e8ee2e3b515e'

def digest(b):return hashlib.sha256(b).hexdigest()
def objbytes(o):return (json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def gloss_trace(rows,index):
    """Resolve documented 上/下 links without transferring the target's reading."""
    path=[]; seen=set(); current=index
    while True:
        if current<0 or current>=len(rows):return {'status':'out_of_bounds','path':path,'gloss':None}
        if current in seen:return {'status':'cycle','path':path,'gloss':None}
        seen.add(current);row=rows[current];path.append('gy:'+str(current+1))
        ref=row.get('釋義參照','')
        if ref=='':return {'status':'resolved','path':path,'gloss':row['釋義']}
        if ref not in ('上','下'):return {'status':'unknown_reference','path':path,'gloss':None}
        current+=-1 if ref=='上' else 1

def character_evidence(char,rows,index,reference,rules):
    candidates=index.get(char,[])
    chosen=None;method=None; intended=None
    if char in rules:
        rule=rules[char];n=int(rule['entry_id'].split(':')[1])-1
        if n not in candidates:raise ValueError('Reading rule headword mismatch '+char)
        trace=gloss_trace(rows,n)
        # Exact source snippet must be present in original or resolved gloss.
        if rule['gloss_contains'] not in (rows[n]['釋義']+' '+(trace['gloss'] or '')):
            raise ValueError('Sense evidence mismatch '+char)
        chosen=n;method='intended_sense_source_rule';intended=rule['sense']
    elif len(candidates)==1:
        chosen=candidates[0];method='single_source_entry_not_context_certification'
    else:
        return None,'ambiguous_or_missing_historical_entry'
    source=rows[chosen];position=source['音韻地位']
    if gloss_trace(rows,chosen)['status']!='resolved':return None,'unresolved_gloss_reference'
    if not position or position[-1] not in '平上去入':return None,'unknown_source_tone'
    tone='P' if position[-1]=='平' else 'Z'
    levels={'平':{'upper_level','lower_level'},'上':{'rising'},'去':{'departing'},'入':{'entering'}}
    memberships=[m for m in reference.get(char,[]) if m['tone']==tone and m['group_id'].split(':')[1] in levels[position[-1]]]
    # Do not guess a Guangyun -> Pingshui crosswalk when same-tone groups compete.
    groups={m['group_id'] for m in memberships}
    if len(groups)!=1:return None,'unresolved_pingshui_mapping'
    m=memberships[0]
    return {'character':char,'entryId':'gy:'+str(chosen+1),'position':position,'fanqie':source['反切'],
            'originalGloss':source['釋義'],'glossTrace':gloss_trace(rows,chosen),
            'sourceRevision':GY_REV,'candidateEntryIds':['gy:'+str(i+1) for i in candidates],
            'method':method,'intendedSense':intended,'tone':tone,'rhymeGroup':m['group_id'],'rhymeLabel':m['label'],
            'pingshuiSourceLines':m['source_lines'],'mapping':'single_compatible_character_and_four_tone_membership',
            'contextuallyCertified':False},None

def read_training(root,run):
    result=[]
    # Held-out rows are skipped before decoding; no statistics are learned from them.
    for path,allowed in [(root/'data/corpus_v03/poems.jsonl',{'train'}),(run/'candidate_poems.jsonl',{'train_candidate'})]:
        for raw in path.read_bytes().splitlines():
            if not raw.strip():continue
            m=re.search(rb'"split"\s*:\s*"([^"]+)"',raw)
            if not m:raise ValueError('Missing split')
            if m[1].decode() not in allowed:continue
            p=json.loads(raw)
            if p['split'] not in allowed:raise ValueError('Split mismatch')
            if p['length'] not in (5,7) or len(p['lines'])!=8:raise ValueError('Invalid poem')
            result.append(p)
    return result

def tags(text,entity):
    seasons=[]; times=[]
    for c,s in [('春','spring'),('夏','summer'),('秋','autumn'),('冬','winter')]:
        if c in text:seasons.append(s)
    if any(w in text for w in ['荷','蓮','蟬']):seasons.append('summer')
    if any(w in text for w in ['桃花','梨花','杏花','東風']):seasons.append('spring')
    if any(w in text for w in ['梅','霜']):seasons.append('winter')
    if any(w in text for w in ['黃葉','紅葉','衰草','蘆花']):seasons.append('autumn')
    if any(w in text for w in ['暮','晚','夕','黃昏','落日','殘陽']):times=['dusk']
    elif any(w in text for w in ['曉','朝','晨','平明']):times=['dawn']
    elif '夜' in text:times=['night']
    elif entity=='moon':times=['night','dusk','dawn']
    elif entity=='sun':times=['day','dawn','dusk']
    return sorted(set(seasons)),times

def build(root,bootstrap):
    spec_path=root/'data/composition_v2/lexicon_spec.json'
    spec=json.loads(spec_path.read_bytes());csvpath=bootstrap/'cache/raw/guangyun.csv';csvbytes=csvpath.read_bytes()
    if blob(csvbytes)!=GY_BLOB:raise ValueError('Pinned Guangyun identity mismatch')
    run=bootstrap/'run';manifest=json.loads((run/'manifest.json').read_bytes())
    for name in ['candidate_poems.jsonl']:
        if digest((run/name).read_bytes())!=manifest['files'][name]['sha256']:raise ValueError('Bootstrap corpus mismatch')
    rows=list(csv.DictReader(io.StringIO(csvbytes.decode('utf-8-sig'))));index=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if len(r['字頭'])==1:index[r['字頭']].append(i)
    refpath=root/'data/reference_v02/pingshui_106_memberships.json';reference=json.loads(refpath.read_bytes())['characters']
    editorial=json.loads((root/'data/reference_v02/editorial_issues.json').read_bytes())
    excluded={x['character'] for x in editorial['issues'] if x.get('character')}
    designs=[]
    for g in spec['groups']:
        designs.extend(dict(text=w,kind=g['kind'],semantic=g['semantic'],entity=g['entity']) for w in g['words'].split())
    designs.extend(dict(v,kind='verb',entity='verb') for v in spec['verbs'])
    chars=sorted(set(''.join(d['text'] for d in designs)));readings={}; rejectedChars={}
    for c in chars:
        r,why=character_evidence(c,rows,index,reference,spec['reading_rules'])
        if c in excluded:r=None;why='existing_editorial_warning'
        if r:readings[c]=r
        else:rejectedChars[c]=why
    poems=read_training(root,run);counts=collections.Counter();examples=collections.defaultdict(list);bi=collections.Counter()
    vocab={d['text'] for d in designs}; banned=set(); traincounts=collections.Counter()
    for p in poems:
        traincounts[str(p['length'])]+=1
        for lineNo,line in enumerate(p['lines']):
            banned.add(line)
            for start in range(len(line)):
                for width in [1,2]:
                    word=line[start:start+width]
                    if len(word)!=width:continue
                    if width==2 and all(c in readings for c in word):bi[word]+=1
                    if word in vocab:
                        counts[word]+=1
                        if len(examples[word])<3:examples[word].append({'poemId':p['poem_id'],'row':lineNo,'start':start,'end':start+width,'split':p['split'],'sourceRefs':p.get('source_refs',[])})
    lex=[]; rejected=[]
    for d in designs:
        text=d['text']; bad=[c for c in text if c not in readings]
        if bad or not counts[text]:
            rejected.append({'text':text,'kind':d['kind'],'reason':'unresolved_reading' if bad else 'no_train_attestation','characters':bad});continue
        season,tm=tags(text,d['entity'])
        if d['kind']=='time':d={**d,'semantic':'season_time' if season else 'clock_time'}
        lex.append({**d,'appliesTo':spec.get('adjectiveEntities',{}).get(text,[]),'id':d['kind']+':'+text,'tones':''.join(readings[c]['tone'] for c in text),
            'rhymeGroup':readings[text[-1]]['rhymeGroup'],'seasons':season,'times':tm,
            'count':counts[text],'sources':examples[text],'semanticStatus':'assistant_intended_construction'})
    if len({l['id'] for l in lex})!=len(lex):raise ValueError('Duplicate lexical IDs')
    templates={str(n):[{'id':t.template_id,'core':t.core,'variants':[list(variants(x,'common_prefix')) for x in t.core],'rhymeRows':t.rhyme_rows} for t in all_templates(n)] for n in (5,7)}
    # Hash exact whole training lines instead of shipping prewritten output lines.
    def fnv(text):
        h=2166136261
        for c in text:h=((h^ord(c))*16777619)&0xffffffff
        return h
    result={'schema':'verseloom.composition-data.v2','lexicon':lex,'readings':readings,'templates':templates,
            'bigrams':dict(sorted(bi.items())),'trainingLineHashes':sorted({fnv(t) for t in banned}),
            'sourceIdentity':{'lexiconSpecSha256':digest(spec_path.read_bytes()),'guangyunSha256':digest(csvbytes),'guangyunRevision':GY_REV,
                'pingshuiSha256':digest(refpath.read_bytes()),'newCorpusSha256':digest((run/'candidate_poems.jsonl').read_bytes()),
                'oldCorpusSha256':digest((root/'data/corpus_v03/poems.jsonl').read_bytes())},
            'meta':{'trainingPoems':len(poems),'trainingByLength':dict(traincounts),'lexemes':len(lex),'readingCharacters':len(readings),
                'rejectedLexemes':len(rejected),'assistantAuthoredSemantics':True,'independentGold':False,'storedOutputLines':0,
                'limitations':['Typed phrase composition is a controlled language, not unrestricted poetry.',
                    'A consistent source-linked reading is not independent contextual pronunciation certification.',
                    'Parallel syntax/semantic slots do not certify allusion, literary merit or absence of semantic redundancy.']}}
    report={'schema':'verseloom.composition-build-report.v2','meta':result['meta'],'admittedByKind':dict(collections.Counter(w['kind'] for w in lex)),
        'rejectedLexemes':rejected,'unresolvedCharacters':rejectedChars,'sourceIdentity':result['sourceIdentity'],
        'glossReferences':dict(collections.Counter(gloss_trace(rows,i)['status'] for i in range(len(rows))))}
    return result,report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'web/composition-data.js');p.add_argument('--report',type=Path,default=ROOT/'docs/COMPOSITION_V2_BUILD.json');p.add_argument('--check',action='store_true');a=p.parse_args()
    data,report=build(ROOT,a.bootstrap);content=b'/* Source-linked lexical data; no stored poems. See composition.html. */\nglobalThis.VerseLoomCompositionData='+objbytes(data).rstrip()+b';\n'
    if a.check:
        if a.output.read_bytes()!=content or a.report.read_bytes()!=objbytes(report):raise SystemExit('Build mismatch')
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(content);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_bytes(objbytes(report))
    print(json.dumps({'status':'matched' if a.check else 'built','bytes':len(content),**report['meta'],'admittedByKind':report['admittedByKind']},ensure_ascii=False))
if __name__=='__main__':main()
