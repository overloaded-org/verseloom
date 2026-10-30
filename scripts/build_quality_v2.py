#!/usr/bin/env python3
"""Compile source-backed word lattices and evidence, never stored output lines.

Only old train and new train_candidate text supplies frequencies. Readings are
chosen by sense policies BEFORE metrical slots exist. These policies are drafts,
not independent literary/phonological gold. Unknowns are not silently accepted.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.prosody import all_templates, variants

VERSION = 'verseloom.word-lattice.v1'
GY_REV = '21585e22c8a730ca2fd175112f4d18e16d5ce578'
SPLIT = re.compile(rb'"split"\s*:\s*"([a-z_]+)"')

def dump(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def digest(b):
    return hashlib.sha256(b).hexdigest()

class GlossGraph:
    """Resolve only the source's documented adjacent-row 上/下 reference edges.

Never inherit the referenced entry's pronunciation, and never guess other tags.
Cycles, cross-small-rhyme links and missing neighbours are explicit failures.
"""
    def __init__(self, rows):
        self.rows = rows
        self.by_id = {r['entry_id']: r for r in rows}
        if len(self.by_id) != len(rows): raise ValueError('Duplicate dictionary IDs')
        self.pos = {r['entry_id']: i for i, r in enumerate(rows)}
        self.cache = {}

    def resolve(self, entry_id):
        if entry_id in self.cache: return self.cache[entry_id]
        visited, pieces, current = [], [], entry_id
        status = 'resolved_source_reference'
        while True:
            if current in visited:
                status = 'cycle'; break
            row = self.by_id.get(current)
            if row is None:
                status = 'missing_entry'; break
            visited.append(current)
            fields = row['fields']
            if fields['釋義']: pieces.append({'entryId': current, 'text': fields['釋義']})
            edge = fields.get('釋義參照', '')
            if not edge: break
            if edge not in ('上', '下'):
                status = 'unsupported_reference'; break
            i = self.pos[current] + (-1 if edge == '上' else 1)
            if i < 0 or i >= len(self.rows):
                status = 'missing_neighbour'; break
            nxt = self.rows[i]
            if nxt['fields']['小韻號'] != fields['小韻號']:
                status = 'cross_rhyme_reference'; break
            current = nxt['entry_id']
        result = {'status': status, 'chain': visited, 'glosses': pieces,
                  'inheritsReading': False, 'contextCertified': False}
        self.cache[entry_id] = result
        return result

def load_training(paths):
    poems = {}
    for path, split in paths:
        for raw in path.read_bytes().splitlines():
            if not raw.strip(): continue
            marker = SPLIT.search(raw)
            if marker is None: raise ValueError('Missing split marker')
            if marker[1].decode() != split: continue
            p = json.loads(raw)
            if p['split'] != split or p['length'] not in (5,7): raise ValueError('Wrong split/shape')
            if len(p['lines']) != 8 or any(len(x) != p['length'] for x in p['lines']): raise ValueError('Malformed poem')
            pid = p['poem_id']
            if pid in poems and poems[pid]['lines'] != p['lines']: raise ValueError('Identity collision')
            poems[pid] = p
    return [poems[k] for k in sorted(poems)]

def reading_evidence(char, rows, graph, ref, policies):
    candidates = rows.get(char, [])
    policy = policies.get(char)
    selected = None; basis = 'unresolved'
    if policy:
        selected = next((r for r in candidates if r['entry_id'] == policy['entryId']), None)
        if selected is None:
            return {'character': char, 'status': 'policy_entry_missing', 'admitted': False}
        resolved = graph.resolve(selected['entry_id'])
        gloss = ''.join(p['text'] for p in resolved['glosses'])
        if policy['glossContains'] not in gloss:
            return {'character': char, 'status': 'policy_gloss_mismatch', 'admitted': False}
        if policy['status'] != 'sense_supported_policy':
            return {'character': char, 'status': 'sense_ambiguity_unresolved', 'admitted': False,
                    'candidateEntryIds':[r['entry_id'] for r in candidates]}
        basis = 'assistant_sense_policy'
    elif len(candidates) == 1:
        selected = candidates[0]; basis = 'single_source_entry'
    if selected is None:
        return {'character':char, 'status':'multiple_or_missing_entries','admitted':False,
                'candidateEntryIds':[r['entry_id'] for r in candidates]}
    resolved = graph.resolve(selected['entry_id'])
    if resolved['status'] != 'resolved_source_reference':
        return {'character':char,'status':resolved['status'],'admitted':False}
    desc = selected['fields']['音韻地位']
    tone = 'P' if desc.endswith('平') else 'Z' if desc.endswith(('上','去','入')) else None
    section={'平':('upper_level','lower_level'),'上':('rising',),'去':('departing',),'入':('entering',)}.get(desc[-1:],())
    matches = [x for x in ref.get(char, []) if x['tone'] == tone and x['group_id'].split(':')[1] in section]
    groups = sorted({x['group_id'] for x in matches})
    if len(groups) != 1:
        return {'character':char,'status':'pingshui_membership_unresolved','admitted':False,
                'candidateGroups':groups,'sourceEntryId':selected['entry_id']}
    return {'character':char,'status':'source_consistent_policy','admitted':True,'tone':tone,
            'group':groups[0], 'sourceEntryId':selected['entry_id'], 'selectionBasis':basis,
            'sensePolicy':policy, 'fanqie':selected['fields']['反切'], 'phonologicalPosition':desc,
            'glossEvidence':resolved,'pingshuiEvidence':matches,
            'contextCertified':False,'reviewStatus':'assistant_policy_not_independent_gold',
            'dictionaryUrl':f'https://github.com/nk2028/tshet-uinh-data/blob/{GY_REV}/韻書/廣韻.csv#L{selected["csv_line_end"]}'}

def token_lattice(line, vocabulary):
    arcs = []
    for i in range(len(line)):
        here = []
        for w in vocabulary:
            if line.startswith(w['text'], i):
                here.append({'start':i,'end':i+len(w['text']),'text':w['text'],
                             'posHypothesis':w['pos'],'senseHypothesis':w['sense'],'status':'machine_hypothesis'})
        arcs.extend(here)
        if not any(a['end'] == i+1 for a in here):
            arcs.append({'start':i,'end':i+1,'text':line[i],'posHypothesis':None,'senseHypothesis':None,'status':'unknown'})
    # No forced best parse. Retain alternatives; unknown arcs are not POS labels.
    return arcs

def build(bootstrap, out, web_output, root=ROOT):
    if out.exists(): raise ValueError('Use a new output directory')
    manifest = json.loads((bootstrap/'manifest.json').read_bytes())
    for name, row in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe source path')
        raw = (bootstrap/path).read_bytes()
        if len(raw) != row['bytes'] or digest(raw) != row['sha256']: raise ValueError('Bootstrap identity changed')
    cfg = json.loads((root/'data/quality_v2/policy.json').read_bytes())
    cfg['words']=[{k:v for k,v in group.items() if k!='texts'} | {'text':text} for group in cfg['wordGroups'] for text in group['texts'].split()]
    ref = json.loads((root/'data/reference_v02/pingshui_106_memberships.json').read_bytes())['characters']
    rows = [json.loads(x) for x in (bootstrap/'guangyun_entries.jsonl').read_bytes().splitlines()]
    if any(r['revision'] != GY_REV for r in rows): raise ValueError('Wrong dictionary revision')
    graph = GlossGraph(rows); by_char = defaultdict(list)
    for r in rows:
        if r['searchable_exact_headword']: by_char[r['fields']['字頭']].append(r)
    chars = sorted({c for w in cfg['words'] for c in w['text']})
    readings = {c:reading_evidence(c,by_char,graph,ref,cfg['readingChoices']) for c in chars}
    poems = load_training([(root/'data/corpus_v03/poems.jsonl','train'), (bootstrap/'candidate_poems.jsonl','train_candidate')])
    counts = Counter(); sources = defaultdict(list); grams = Counter(); training_lines = set()
    wanted = {w['text'] for w in cfg['words']}
    for p in poems:
        for row, text in enumerate(p['lines']):
            training_lines.add(text)
            for i in range(len(text)):
                for n in (1,2,3):
                    token = text[i:i+n]
                    if len(token) != n: continue
                    if n in (2,3): grams[token] += 1
                    if token in wanted:
                        counts[token] += 1
                        if len(sources[token]) < 3:
                            sources[token].append({'poemId':p['poem_id'],'familyId':p.get('family_id'),
                              'split':p['split'],'row':row,'start':i,'end':i+n,'sourceRefs':p['source_refs'], 'line':text})
    words=[]; excluded=[]; seen=set()
    for w in cfg['words']:
        key=(w['text'],w['pos'],w['kind'])
        if key in seen: continue
        seen.add(key)
        reasons=[]
        if not counts[w['text']]: reasons.append('no_training_occurrence')
        reasons += [c+':'+readings[c]['status'] for c in w['text'] if not readings[c]['admitted']]
        if reasons:
            excluded.append({'text':w['text'],'pos':w['pos'],'reasons':reasons});continue
        words.append({**w,'id':len(words),'tones':''.join(readings[c]['tone'] for c in w['text']),
                      'endingGroup':readings[w['text'][-1]]['group'], 'count':counts[w['text']],
                      'sources':sources[w['text']],'readingIds':[readings[c]['sourceEntryId'] for c in w['text']],
                      'annotationStatus':'assistant_policy_not_gold'})
    allowed_chars={c for w in words for c in w['text']}
    # Only export statistics used by the actual vocabulary. No held-out statistics.
    grams={g:n for g,n in sorted(grams.items()) if set(g)<=allowed_chars}
    labels={r['group_id']:r['label'] for rs in ref.values() for r in rs}
    templates={str(n):[{**t.to_dict(),'patterns':[list(variants(x,'common_prefix')) for x in t.core]}
                       for t in all_templates(n)] for n in (5,7)}
    rhyme_groups=sorted({w['endingGroup'] for w in words if w['tones'][-1]=='P' and w['pos'] in ('N','I')})
    sources_hash={'lexicon':digest((root/'data/quality_v2/policy.json').read_bytes()),
                  'reference':digest((root/'data/reference_v02/pingshui_106_memberships.json').read_bytes()),
                  'bootstrapManifest':digest((bootstrap/'manifest.json').read_bytes()),
                  'oldCorpus':digest((root/'data/corpus_v03/poems.jsonl').read_bytes()),
                  'compiler':digest(Path(__file__).read_bytes()),
                  'prosody':digest((root/'src/wordconv_lab/prosody.py').read_bytes())}
    data={'schema':VERSION,'words':words,'readings':{c:readings[c] for c in sorted(allowed_chars)},
          'grams':grams,'scenes':cfg['scenes'],'templates':templates,
          'rhymes':[{'id':g,'label':labels[g]} for g in rhyme_groups],
          'trainingLines':sorted(t for t in training_lines if set(t)<=allowed_chars),
          'sources':sources_hash,'historicalGlossSource':{'revision':GY_REV,'license':'CC0-1.0'},
          'provenance':{'wordPolicies':'assistant_authored','candidateLinesStored':False,'independentGold':False,
                        'strictAntithesisCertified':False,'contextReadingsCertified':False,'literaryQualityCertified':False}}
    encoded=('/* Source-backed word lattice; policies are not independent gold. */\nglobalThis.VerseLoomQualityData='+dump(data).decode().rstrip()+';\n').encode()
    web_output.parent.mkdir(parents=True,exist_ok=True);web_output.write_bytes(encoded)
    tasks=[json.loads(x) for x in (bootstrap/'annotation_tasks_400.jsonl').read_bytes().splitlines()]
    analyses=[]
    for t in tasks:
        point_readings=[]
        for p in t['reading_candidates']:
            candidates=[{'entryId':i,'gloss':graph.resolve(i)} for i in p['guangyun_entry_ids']]
            point_readings.append({'row':p['row'],'column':p['column'],'character':p['character'],
                                   'candidates':candidates,'selectedEntryId':None,'status':'unresolved'})
        analyses.append({'taskId':t['task_id'],'poemId':t['poem_id'],'familyId':t['family_id'],
                         'lines':t['lines'],'contextLines':t['context_lines'],'sourceRefs':t['source_refs'],
                         'tokenLattices':[token_lattice(x,cfg['words']) for x in t['lines']],
                         'readings':point_readings,'status':'machine_analysis_complete_not_annotation_approval',
                         'strictAntithesis':'unknown','reviewed':False,'isGold':False})
    resolution_counts=Counter(graph.resolve(r['entry_id'])['status'] for r in rows)
    report={'schema':VERSION+'.build','trainingPoems':dict(Counter(str(p['length']) for p in poems)),
            'configuredWords':len(cfg['words']),'admittedWordPolicies':len(words),'excludedWordPolicies':len(excluded),
            'wordKinds':dict(Counter(w['kind'] for w in words)),'dictionaryReferenceStatuses':dict(resolution_counts),
            'processedAnnotationTasks':len(analyses),'approvedAnnotations':0,'independentGoldLabels':0,
            'assetBytes':len(encoded),'assetSha256':digest(encoded),'sources':sources_hash}
    out.mkdir(parents=True)
    (out/'annotation_analysis.jsonl').write_bytes(b''.join(dump(x) for x in analyses))
    (out/'excluded_words.json').write_bytes(dump(excluded))
    (out/'dictionary_references.jsonl').write_bytes(b''.join(dump({'entryId':r['entry_id'],**graph.resolve(r['entry_id'])}) for r in rows if r['fields']['釋義參照']))
    (out/'report.json').write_bytes(dump(report))
    (out/'manifest.json').write_bytes(dump({p.name:digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstrap',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--web-output',type=Path,default=ROOT/'web/quality-data.js')
    args=ap.parse_args()
    print(json.dumps(build(args.bootstrap,args.output,args.web_output),ensure_ascii=False,indent=2))
