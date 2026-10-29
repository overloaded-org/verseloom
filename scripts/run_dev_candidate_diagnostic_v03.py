"""DEV-ONLY candidate-vocabulary coverage diagnostic after the v0.3 pilot.

No task file, model, prediction or test task is opened. Training/dev poems are
selected from the frozen source stream before JSON decoding. Dev targets are
reconstructed by the existing even-line prefix rule. All four declared source
policies are reported; this is not top-16 retrieval recall or model accuracy.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'results/dev_candidate_diagnostic_v03'
POLICIES = (
    {'id':'even_prefix_min2','rows':'even','spans':'prefix','min_count':2},
    {'id':'even_prefix_min1','rows':'even','spans':'prefix','min_count':1},
    {'id':'even_adjacent_min2','rows':'even','spans':'all_adjacent','min_count':2},
    {'id':'all_adjacent_min2','rows':'all','spans':'all_adjacent','min_count':2},
)
SPLIT_PATTERN = re.compile(rb'"split":"(train|dev|test)"')


def sha(path):
    with path.open('rb') as handle:
        digest=hashlib.sha256()
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def rows_encoded(rows):
    return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for row in rows).encode()


def selected_poems(path):
    selected={'train':[],'dev':[]}
    with path.open('rb') as handle:
        for raw in handle:
            marker=SPLIT_PATTERN.search(raw)
            if marker is None:raise ValueError('missing canonical split marker')
            split=marker.group(1).decode('ascii')
            if split not in selected:
                continue  # Do not decode or interpret test poem contents.
            poem=json.loads(raw)
            if poem['split']!=split or len(poem['lines'])!=8 or poem['length'] not in (5,7):
                raise ValueError('source shape/split mismatch')
            selected[split].append(poem)
    return selected


def counts_for_policy(train,policy):
    counts={5:Counter(),7:Counter()}
    for poem in train:
        row_indices=range(1,8,2) if policy['rows']=='even' else range(8)
        for row in row_indices:
            line=poem['lines'][row]
            starts=(0,) if policy['spans']=='prefix' else range(len(line)-1)
            for start in starts:
                counts[poem['length']][line[start:start+2]]+=1
    return {length:{span:n for span,n in sorted(values.items()) if n>=policy['min_count']}
            for length,values in counts.items()}


def direct_expected_counts(train,policy,length):
    # Separate whole-line collection and flat comprehension for an independent
    # count consistency check; no derived candidate pool or dev target is used.
    lines=[line for p in train if p['length']==length for row,line in enumerate(p['lines'])
           if policy['rows']=='all' or row%2==1]
    if policy['spans']=='prefix':
        counter=Counter(line[:2] for line in lines)
    else:
        counter=Counter(a+b for line in lines for a,b in zip(line,line[1:]))
    return {span:count for span,count in counter.items() if count>=policy['min_count']}


def summarize(dev_rows,policy_id,length=None):
    rows=[r for r in dev_rows if length is None or r['length']==length]
    families=defaultdict(list)
    for row in rows:
        families[row['family_id']].append(int(row['covered_by_policy'][policy_id]))
    covered=sum(row['covered_by_policy'][policy_id] for row in rows)
    means=[sum(values)/len(values) for values in families.values()]
    return {'dev_tasks':len(rows),'covered_dev_tasks':covered,
        'task_weighted_vocabulary_coverage':covered/len(rows) if rows else None,
        'dev_families':len(families),
        'family_equal_weight_vocabulary_coverage':sum(means)/len(means) if means else None}


def run(output,check=False):
    poem_path=ROOT/'data/corpus_v03/poems.jsonl'
    baseline_path=ROOT/'data/corpus_v03/candidate_spans.json'
    manifest_path=ROOT/'data/corpus_v03/build_manifest.json'
    sources=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
             for p in (poem_path,baseline_path,manifest_path)]
    source_manifest=json.loads(manifest_path.read_text())
    for p in (poem_path,baseline_path):
        if sha(p)!=source_manifest['outputs'][p.name]['sha256']:
            raise ValueError(f'frozen corpus identity mismatch: {p.name}')
    config={'schema':'versematrix.dev-candidate-source-diagnostic.v0.3',
        'stage':'post_pilot_development_only','policies':list(POLICIES),
        'candidate_source_split':'train','evaluation_split':'dev','length_pools':[5,7],
        'target':'prefix two characters of each even-numbered line; four tasks per dev poem',
        'expected_dev_tasks':2188,'normalization':'none; source code points retained',
        'cross_line_spans':False,'task_files_opened':False,'models_or_predictions_opened':False,
        'test_tasks_parsed':False,'family_weighting':'mean task coverage within each family, then mean families',
        'scope':'Vocabulary coverage only; not top-16 retrieval recall, model accuracy, literary quality or test improvement.'}
    frozen={'config':config,'sources':sources,
        'script':{'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':sha(Path(__file__).resolve())}}
    output.mkdir(parents=True,exist_ok=True)
    freeze_path=output/'freeze.json'
    if check:
        previous=json.loads(freeze_path.read_text())
        if any(previous[k]!=v for k,v in frozen.items()):raise ValueError('frozen config/source/code differs')
    else:
        if freeze_path.exists():raise ValueError('refusing to overwrite frozen diagnostic; use --check')
        frozen['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        frozen['status']='frozen_before_candidate_counts_or_dev_coverage'
        freeze_path.write_bytes(encoded(frozen))

    # No candidate or dev-coverage calculation happens before the freeze above.
    selected=selected_poems(poem_path)
    pools={p['id']:counts_for_policy(selected['train'],p) for p in POLICIES}
    for policy in POLICIES:
        for length in (5,7):
            if pools[policy['id']][length]!=direct_expected_counts(selected['train'],policy,length):
                raise ValueError('independent training-source counts disagree')
    baseline=json.loads(baseline_path.read_text())
    expected_baseline={int(length):{item['text']:item['count'] for item in values}
                       for length,values in baseline['by_length'].items()}
    if pools['even_prefix_min2']!=expected_baseline:
        raise ValueError('policy 1 differs from the original frozen candidate source')
    rows=[]
    for poem in selected['dev']:
        for couplet in range(4):
            target=poem['lines'][2*couplet+1][:2]
            rows.append({'task_id':poem['poem_id']+f':couplet{couplet}',
                'poem_id':poem['poem_id'],'family_id':poem['family_id'],'split':'dev',
                'length':poem['length'],'observed_source_prefix':target,
                'covered_by_policy':{policy['id']:target in pools[policy['id']][poem['length']]
                                     for policy in POLICIES}})
    if len(rows)!=config['expected_dev_tasks']:raise ValueError('unexpected dev task denominator')
    if len({r['task_id'] for r in rows})!=len(rows):raise ValueError('duplicate dev task')
    results=[]
    for policy in POLICIES:
        candidate_counts={str(length):len(pools[policy['id']][length]) for length in (5,7)}
        results.append({'policy':policy,'candidate_count_by_length':candidate_counts,
            'candidate_count_total_length_tagged':sum(candidate_counts.values()),
            'candidate_count_unique_text_across_lengths':len(set(pools[policy['id']][5])|set(pools[policy['id']][7])),
            'by_length':{str(length):summarize(rows,policy['id'],length) for length in (5,7)},
            'overall':summarize(rows,policy['id'])})
    report={'schema':'versematrix.dev-candidate-source-results.v0.3','status':'completed_dev_only_diagnostic',
        'source_train_poems':len(selected['train']),'source_dev_poems':len(selected['dev']),
        'dev_tasks':len(rows),'dev_families':len({r['family_id'] for r in rows}),
        'all_four_policies_reported':True,'results':results,
        'checks':{'independent_training_counts':'passed','original_baseline_identity':'passed',
                  'dev_task_denominator':'passed'},
        'limits':['This diagnostic was proposed after the pilot; it is exploratory development work.',
                  'No test tasks, models or predictions were opened; test corpus records were skipped before JSON decoding.',
                  'Full source-file SHA covers the frozen mixed-split file bytes without interpreting skipped test records.',
                  'Vocabulary coverage is an upper bound for exact restoration under this vocabulary, not actual ranking accuracy.',
                  'These are not top-16 retrieval-recall measurements and do not establish test improvement.',
                  'Pools remain separate for five/seven characters; total length-tagged count can include the same text twice.',
                  'No original experiment pool, training run, test output or corpus source was changed.']}
    payloads={'report.json':encoded(report),'candidate_counts.json':encoded(pools),
              'dev_coverage.jsonl':rows_encoded(rows)}
    build_manifest={'status':'completed','freeze_sha256':sha(freeze_path),
        'outputs':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()},
        'test_evaluation_performed':False,'original_experiment_changed':False}
    payloads['manifest.json']=encoded(build_manifest)
    for name,data in payloads.items():
        path=output/name
        if check:
            if path.read_bytes()!=data:raise ValueError(f'rebuilt bytes differ: {name}')
        else:path.write_bytes(data)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    print(json.dumps(run(args.out,args.check),ensure_ascii=False,indent=2))
