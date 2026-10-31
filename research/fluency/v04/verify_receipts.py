"""Audit a v0.4 research pack. No model fitting, generation, Pages writes or third-party dependencies."""
import collections
import hashlib
import json
import sys
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def audit(root):
    root = Path(root)
    records = [json.loads(x) for x in (root/'outputs/run/receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    source = json.loads((root/'inputs/fresh.json').read_bytes())['records']
    old = [r for r in records if r['cohort'] == 'dev']
    new = [r for r in records if r['cohort'] == 'fresh']
    check(len(old) == 64 and len(new) == 96 and len(source) == 12, 'Wrong denominators')
    check([p['seed'] for p in source] == list(range(33101,33113)), 'Unexpected generation seeds')
    check([r['text'] for r in new] == [s for p in source for s in p['lines']], 'Edited output')
    check(len({r['text'] for r in new}) == 96 and not ({r['text'] for r in new} & {r['text'] for r in old}), 'Text overlap')
    cells = {}
    for name, cohort in [('old59',[r for r in old if r['group']=='typed_tiling_impossible']),
                         ('old4',[r for r in old if r['group']=='tiling_but_no_grounded_S']),('old64',old),('new96',new)]:
        cells[name] = {}
        for k in cohort[0]['arms']:
            a = [r['arms'][k] for r in cohort]
            cells[name][k] = [sum(t['tiling'] for t in a), sum(t['sentence'] for t in a), sum(len(t['holes']) for t in a)]
    check(cells['old59']['base']==[0,0,153] and cells['old59']['guarded']==[7,4,125], 'Old lexical arm mismatch')
    check(cells['old4']['base']==[4,0,0] and cells['old4']['grammar']==[4,2,0], 'Old grammar arm mismatch')
    check(cells['new96']['base']==[10,4,234] and cells['new96']['guarded']==[12,4,224], 'Fresh experiment mismatch')
    reviewed = [r for r in new if 'editor' in r]
    check(len(reviewed)==24, 'Missing development readings')
    B = [r for r in reviewed if r['editor']['tag']=='B']
    check(len(B)==6 and all(r['editor']['span'] for r in B), 'Missing annotated span')
    check(all(r['alerts']['base']==r['alerts']['guarded'] for r in reviewed), 'Changed sample alarms')
    check(all(not r['arms']['guarded']['lexical_evidence'] for r in reviewed), 'Changed intervention exposure')
    check(all(a['action']=='retain' for r in records for a in r['alerts'].values()), 'Unexpected filtering')
    return {'cells_tiled_S_holes':cells, 'editor_B_to_guarded':dict(collections.Counter(r['alerts']['guarded']['label'] for r in B)),
            'receipt_sha256':hashlib.sha256((root/'outputs/run/receipts.jsonl').read_bytes()).hexdigest(),
            'scope':'receipt consistency only; no human agreement or literary quality claim'}


if __name__=='__main__':
    print(json.dumps(audit(sys.argv[1]), ensure_ascii=False, indent=2))
