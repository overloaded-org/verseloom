"""Join training span candidates to unresolved Pingshui source memberships.

This audit never chooses a reading and is NOT used to filter the v0.3 model
candidate pools. Rebuild/check is offline and preserves exact source glyphs.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'data/reading_candidates_v03'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def line_encoded(rows):
    return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for row in rows).encode()


def input_data():
    candidate_path=ROOT/'data/corpus_v03/candidate_spans.json'
    membership_path=ROOT/'data/reference_v02/pingshui_106_memberships.json'
    issue_path=ROOT/'data/reference_v02/editorial_issues.json'
    corpus_manifest=json.loads((ROOT/'data/corpus_v03/build_manifest.json').read_text())
    reference_manifest=json.loads((ROOT/'data/reference_v02/source_manifest.json').read_text())
    if sha(candidate_path)!=corpus_manifest['outputs']['candidate_spans.json']['sha256']:
        raise ValueError('training candidate source differs from its frozen manifest')
    expected={x['filename']:x['sha256'] for x in reference_manifest['derived_files']}
    for path in (membership_path,issue_path):
        if sha(path)!=expected[path.name]:raise ValueError(f'reference source differs: {path.name}')
    candidates=json.loads(candidate_path.read_text())
    reference=json.loads(membership_path.read_text())
    issues=json.loads(issue_path.read_text())
    if candidates['source_split']!='train':raise ValueError('training candidates required')
    train_ids=set()
    with (ROOT/'data/corpus_v03/poems.jsonl').open() as handle:
        for line in handle:
            poem=json.loads(line)
            if poem['split']=='train':train_ids.add(poem['poem_id'])
    for values in candidates['by_length'].values():
        for item in values:
            if len(item['text'])!=2 or not set(item['source_poem_ids'])<=train_ids:
                raise ValueError('non-training origin or invalid span')
    issue_by_char={}
    for item in issues['issues']:
        if item.get('character'):
            issue_by_char.setdefault(item['character'],set()).add(item['issue_id'])
    source_paths=[candidate_path,membership_path,issue_path,
        ROOT/'data/corpus_v03/build_manifest.json',ROOT/'data/reference_v02/source_manifest.json',
        ROOT/'data/corpus_v03/poems.jsonl']
    sources=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in source_paths]
    return candidates,reference,issue_by_char,sources


def build_rows(candidates,reference,issue_by_char):
    rows=[]
    for length,spans in sorted(candidates['by_length'].items()):
        for span in spans:
            characters=[]
            for position,char in enumerate(span['text']):
                memberships=deepcopy(reference['characters'].get(char,[]))
                for member in memberships:
                    if member.get('status')!='source_candidate_needs_context' or member['tone'] not in ('P','Z'):
                        raise ValueError('unexpected reference certainty or tone')
                issue_ids=set(issue_by_char.get(char,set()))
                issue_ids.update(i for m in memberships for i in m.get('source_issue_ids',[]))
                characters.append({'position':position,'character':char,
                    'lookup_status':'source_candidate_needs_context' if memberships else 'unknown',
                    'memberships':memberships,'tone_candidates':sorted({m['tone'] for m in memberships}),
                    'source_issue_ids':sorted(issue_ids),'requires_context_review':True})
            rows.append({'length':int(length),'text':span['text'],'training_count':span['count'],
                'source_split':'train','source_poem_ids':span['source_poem_ids'],
                'character_records':characters,'requires_context_review':True,
                'status':'source_candidate_needs_context',
                'used_for_model_candidate_filtering':False,'used_for_model_features':False})
    return rows


def summarize_rows(rows):
    summary={}
    for length in (5,7):
        selected=[r for r in rows if r['length']==length]
        counter=Counter()
        unknown=Counter()
        issue_chars=Counter()
        distinct=set()
        for row in selected:
            chars=row['character_records']
            distinct.update(c['character'] for c in chars)
            counter['spans']+=1
            counter['both_characters_found']+=all(bool(c['memberships']) for c in chars)
            counter['both_single_membership']+=all(len(c['memberships'])==1 for c in chars)
            counter['both_single_tone_candidate']+=all(len(c['tone_candidates'])==1 for c in chars)
            counter['at_least_one_multiple_membership']+=any(len(c['memberships'])>1 for c in chars)
            counter['at_least_one_cross_tone_character']+=any(len(c['tone_candidates'])>1 for c in chars)
            counter['at_least_one_unknown_character']+=any(not c['memberships'] for c in chars)
            counter['at_least_one_editorial_issue']+=any(c['source_issue_ids'] for c in chars)
            unknown.update(c['character'] for c in chars if not c['memberships'])
            issue_chars.update(c['character'] for c in chars if c['source_issue_ids'])
        summary[str(length)]={**dict(counter),'distinct_characters':len(distinct),
            'unknown_character_occurrences':dict(sorted(unknown.items())),
            'editorial_issue_character_occurrences':dict(sorted(issue_chars.items()))}
    return summary


def independently_verify(candidates,reference,issue_by_char,rows,summary):
    index={(r['length'],r['text']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('duplicate output span identity')
    for length,spans in candidates['by_length'].items():
        counts=Counter()
        for span in spans:
            row=index[(int(length),span['text'])]
            if row['source_poem_ids']!=span['source_poem_ids'] or row['training_count']!=span['count']:
                raise ValueError('training provenance changed')
            if row['requires_context_review'] is not True or row['used_for_model_candidate_filtering'] is not False:
                raise ValueError('invalid audit-only policy')
            raw=[reference['characters'].get(c,[]) for c in span['text']]
            for position,char in enumerate(span['text']):
                record=row['character_records'][position]
                if record['memberships']!=raw[position] or record['character']!=char:
                    raise ValueError('source membership dropped, changed or selected')
                if record['requires_context_review'] is not True:
                    raise ValueError('cannot certify contextual reading')
                expected_issues=set(issue_by_char.get(char,set()))
                expected_issues.update(i for m in raw[position] for i in m.get('source_issue_ids',[]))
                if record['source_issue_ids']!=sorted(expected_issues):raise ValueError('issue provenance mismatch')
            counts['spans']+=1
            counts['both_characters_found']+=len(raw[0])>0 and len(raw[1])>0
            counts['both_single_membership']+=len(raw[0])==len(raw[1])==1
            tone_counts=[len({m['tone'] for m in members}) for members in raw]
            counts['both_single_tone_candidate']+=tone_counts==[1,1]
            counts['at_least_one_multiple_membership']+=max(map(len,raw))>1
            counts['at_least_one_cross_tone_character']+=max(tone_counts)>1
            counts['at_least_one_unknown_character']+=min(map(len,raw))==0
            counts['at_least_one_editorial_issue']+=any(
                issue_by_char.get(c) or any(m.get('source_issue_ids') for m in reference['characters'].get(c,[]))
                for c in span['text'])
        if any(summary[length].get(k)!=v for k,v in counts.items()):
            raise ValueError('independent direct-source counts differ')


def run(output,check=False):
    candidates,reference,issue_by_char,sources=input_data()
    rows=build_rows(candidates,reference,issue_by_char)
    summary=summarize_rows(rows)
    independently_verify(candidates,reference,issue_by_char,rows,summary)
    audit={'schema':'versematrix.training-candidate-reading-audit.v0.3',
        'status':'source_candidates_joined_without_context_disambiguation',
        'source_split':'train','source_reference':reference['source'],
        'normalization':'none; exact original Unicode glyphs',
        'requires_context_review':True,'used_for_model_candidate_filtering':False,
        'used_for_model_features':False,'counts_by_length':summary,
        'category_note':'Categories overlap. Single membership or single tone does not establish a contextual reading.',
        'validation':'Every output membership compared with original index; separate direct-source aggregate counts agree.',
        'limits':['No contextual readings selected and no verified records created.',
                  'Unknown means absent from this transcription, not impossible in historical usage.',
                  'A source issue on an absent character is preserved via the existing editorial issue index.',
                  'This audit supplies no metrical gold and does not change the source-restoration experiment.']}
    payloads={'candidate_readings.jsonl':line_encoded(rows),'audit.json':encoded(audit)}
    manifest={'schema':'versematrix.candidate-reading-build.v0.3','status':'completed',
        'sources':sources,'script':{'path':str(Path(__file__).resolve().relative_to(ROOT)),
                                  'sha256':sha(Path(__file__).resolve())},
        'outputs':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()},
        'test_predictions_accessed':False,'context_disambiguation_performed':False,
        'licensing':['data/reference_v02/LICENSE_NOTICE.md','data/corpus_v03/LICENSE_NOTICE.md']}
    payloads['manifest.json']=encoded(manifest)
    output.mkdir(parents=True,exist_ok=True)
    if not check and any((output/name).exists() for name in payloads):
        raise ValueError('refusing to overwrite audit evidence; use --check')
    for name,data in payloads.items():
        path=output/name
        if check:
            if path.read_bytes()!=data:raise ValueError(f'offline reconstruction differs: {name}')
        else:path.write_bytes(data)
    return {'mode':'check' if check else 'build','ok':True,'spans':len(rows),'counts_by_length':summary}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=OUTPUT)
    args=parser.parse_args()
    print(json.dumps(run(args.out,args.check),ensure_ascii=False,indent=2))
