#!/usr/bin/env python3
"""Check draft provenance and offsets; NOT whether an interpretation is correct."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(pilot, tasks):
    require(pilot.get('schema') == 'verseloom.data-bootstrap.annotation-pilot.v1', 'Unknown draft schema')
    ids = set()
    for record in pilot['records']:
        key = record['task_id']
        require(key not in ids and key in tasks, 'Unknown or duplicate task')
        ids.add(key)
        task = tasks[key]
        require(task['usage'] == 'development_annotation_only' and task['is_gold'] is False, 'Wrong task purpose')
        for name in ('poem_id', 'family_id', 'length', 'context_lines', 'source_refs', 'row_indices_zero_based'):
            require(record[name] == task[name], 'Draft changes source identity: ' + name)
        require(record['status'] == 'assistant_draft' and record['is_gold'] is False and
                record['independent_review'] is False and record['publish_as_verified'] is False,
                'Unreviewed draft promoted to verified or gold')
        require(record['usage'] == 'development_annotation_only' and record['strict_antithesis'] == 'unknown' and
                record['contextual_pronunciation'] == 'unresolved' and not record['selected_historical_readings'] and
                record['literary_quality_score'] is None, 'Draft makes an unsupported certification')
        require([r['row'] for r in record['tokens']] == task['row_indices_zero_based'], 'Wrong annotated rows')
        for row in record['tokens']:
            line = task['context_lines'][row['row']]
            end = 0
            require(bool(row['tokens']), 'Missing tokens')
            for token in row['tokens']:
                a, b = token['start'], token['end']
                require(type(a) is int and type(b) is int and a == end and a < b <= len(line), 'Invalid/overlapping token span')
                require(token['text'] == line[a:b], 'Token is not the original text')
                require(bool(token['pos_candidates']) and all(p in {'N','V','A','ADV','LOC','NUM','MODAL'} for p in token['pos_candidates']), 'Invalid draft tag')
                end = b
            require(end == len(line), 'Incomplete token coverage')
        for alignment in record['alignment_hypothesis']:
            for field, row in [('upper_token', 0), ('lower_token', 1)]:
                n = alignment[field]
                require(type(n) is int and 0 <= n < len(record['tokens'][row]['tokens']), 'Alignment index out of bounds')
        require(record['alignment_status'] == 'proposed_not_certified', 'Alignment wrongly certified')
        readings = {(r['row'],r['column']):r for r in task['reading_candidates']}
        for note in record['reading_notes']:
            point = (note['row'],note['column'])
            require(point in readings and note['character'] == readings[point]['character'], 'Reading note at wrong character')
            require(set(note['entry_ids']) <= set(readings[point]['guangyun_entry_ids']), 'Unrelated historical entry')
    return {'schema':'verseloom.annotation-pilot.check.v1', 'draft_couplets':len(ids),
            'poems':len({r['poem_id'] for r in pilot['records']}),
            'status':'PASS_PROVENANCE_AND_OFFSETS_ONLY','independent_gold_labels':0,
            'contextually_resolved_positions':0,'interpretive_correctness_certified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=ROOT/'data/quality_bootstrap_v1/pilot_drafts.json')
    args=p.parse_args()
    pilot=json.loads(args.pilot.read_text(encoding='utf-8'))
    raw=(args.run/'annotation_tasks_400.jsonl').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==pilot['task_file_sha256'], 'Task bundle identity mismatch')
    tasks={row['task_id']:row for row in map(json.loads,raw.decode('utf-8').splitlines())}
    print(json.dumps(validate(pilot,tasks),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
