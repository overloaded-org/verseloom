"""Post-selection engineering audit of immutable v0.5 matching artifacts.

This separate script is not part of the formal training freeze and cannot
change checkpoint selection.  It validates public matrix-factor and deployment
contracts; a separate reviewer owns the independent mathematical forward and
gradient oracle.  Only old dev visible queries are sampled, without their
targets. No test/confirmation tasks, predictions or reports are read.
"""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.matching import ContextSpanMatcher
from wordconv_lab.restoration import VisibleQuery,digest

DEFAULT_EXPERIMENT=ROOT/'results/matching_v05'
DEFAULT_OUTPUT=ROOT/'results/matching_v05_engineering/numeric_audit'
SPLIT_PATTERN=re.compile(rb'"split":"(train|dev|test)"')
SAMPLE_SALT='versematrix-matching-v05-engineering-audit-20261029'
PER_LENGTH=8


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def write_new(path,data):
    if not isinstance(data,bytes):data=encoded(data)
    if path.exists():raise ValueError('refusing to overwrite audit evidence: '+str(path))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as handle:
        handle.write(data);handle.flush();os.fsync(handle.fileno())
    os.link(temporary,path)
    temporary.unlink()


def load_json(path):return json.loads(path.read_text())


def source_names():
    return ('scripts/audit_matching_numeric_v05.py','src/wordconv_lab/matching.py','src/wordconv_lab/restoration.py')


def inputs(experiment):
    return [ROOT/'data/corpus_v03/poems.jsonl',ROOT/'data/corpus_v03/vocabulary.json',
            experiment/'freeze.json',experiment/'candidate_resources.json']


def read_dev_visible_queries():
    rows=[];poem_count=0
    with (ROOT/'data/corpus_v03/poems.jsonl').open('rb') as handle:
        for raw in handle:
            marker=SPLIT_PATTERN.search(raw)
            if marker is None:raise ValueError('missing source split marker')
            if marker.group(1)!=b'dev':continue
            poem=json.loads(raw)
            if poem['split']!='dev' or len(poem['lines'])!=8:raise ValueError('invalid dev source record')
            poem_count+=1
            for couplet in range(4):
                query=VisibleQuery(poem['length'],poem['lines'][2*couplet],poem['lines'][2*couplet+1][2:])
                task_id=poem['poem_id']+f':couplet{couplet}'
                rows.append({'task_id':task_id,'poem_id':poem['poem_id'],'family_id':poem['family_id'],
                             'length':query.length,'query':asdict(query),'query_hash':digest(asdict(query)),
                             'sampling_hash':hashlib.sha256((SAMPLE_SALT+':'+task_id).encode()).hexdigest()})
    if poem_count!=547 or len(rows)!=2188 or len({row['task_id'] for row in rows})!=2188:
        raise ValueError('unexpected old-dev source denominator')
    return rows


def prepare(experiment,output):
    if output.exists():raise ValueError('audit output exists; use the prepared audit or choose a fresh path')
    output.mkdir(parents=True)
    formal=load_json(experiment/'freeze.json')
    core='src/wordconv_lab/matching.py'
    if sha(ROOT/core)!=formal['source_sha256'][core]:raise ValueError('formal core identity changed')
    for name in ('data/corpus_v03/poems.jsonl','data/corpus_v03/vocabulary.json'):
        if sha(ROOT/name)!=formal['input_sha256'][name]:raise ValueError('formal corpus identity changed')
    plan={'schema':'versematrix.matching-artifact-audit-plan.v0.5',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'stage':'separate_engineering_audit_not_formal_experiment_or_model_selection',
          'formal_experiment_directory':str(experiment.relative_to(ROOT)),
          'source_sha256':{name:sha(ROOT/name) for name in source_names()},
          'input_sha256':{str(path.relative_to(ROOT)):sha(path) for path in inputs(experiment)},
          'sampling':{'split':'old_dev','salt':SAMPLE_SALT,'per_length':PER_LENGTH,'lengths':[5,7],
                      'ordering':'SHA256(salt + colon + task_id), then task_id; first eight per length',
                      'targets_retained_or_used':False},
          'models':'all four formal arms times all three seeds, references from selection_frozen.json',
          'checks':{'public_matrix_factor_reconstruction':'exact array equality',
                    'batch_single_scores':'max_abs <= 1e-10 and np.allclose(rtol=1e-12,atol=1e-10)',
                    'batch_single_full_rankings':'exact ordered index equality with lexical ties',
                    'rank_api_top16':'exact ordered candidate equality',
                    'roundtrip_parameters_optimizer_config':'exact equality',
                    'roundtrip_batch_scores_and_full_rankings':'exact equality',
                    'immutable_selected_file_hash_before_after':'exact equality'},
          'selected_metric_reports_opened':False,'test_or_confirmation_files_opened':False,
          'independent_math_oracle':'separate reviewer; this script audits artifact/API algebra and deployment consistency only',
          'training_updates_performed':0}
    write_new(output/'plan_frozen.json',plan)
    all_rows=read_dev_visible_queries()
    sample=[]
    for length in (5,7):
        selected=sorted((row for row in all_rows if row['length']==length),
                        key=lambda row:(row['sampling_hash'],row['task_id']))[:PER_LENGTH]
        if len(selected)!=PER_LENGTH:raise ValueError('insufficient fixed-hash sample')
        sample.extend(selected)
    write_new(output/'sample_queries.json',sample)
    prepared={'status':'prepared_visible_queries_without_targets',
              'plan_sha256':sha(output/'plan_frozen.json'),'sample_sha256':sha(output/'sample_queries.json'),
              'queries':len(sample),'selected_models_opened':False,'formal_experiment_changed':False}
    write_new(output/'prepared.json',prepared)
    return prepared


def ranking(scores,spans):
    return np.lexsort((np.asarray(spans,dtype=str),-scores))


def run(experiment,output):
    if (output/'report.json').exists():raise ValueError('audit already completed; do not overwrite it')
    plan=load_json(output/'plan_frozen.json')
    prepared=load_json(output/'prepared.json')
    if sha(output/'plan_frozen.json')!=prepared['plan_sha256'] or sha(output/'sample_queries.json')!=prepared['sample_sha256']:
        raise ValueError('audit preparation identity changed')
    for category in ('source_sha256','input_sha256'):
        for name,expected in plan[category].items():
            if sha(ROOT/name)!=expected:raise ValueError('audit frozen input/source changed: '+name)
    if str(experiment.relative_to(ROOT))!=plan['formal_experiment_directory']:
        raise ValueError('prepared experiment directory differs')
    selection_path=experiment/'selection_frozen.json'
    selection=load_json(selection_path)
    formal=load_json(experiment/'freeze.json')
    config=formal['config']
    if selection['config_sha256']!=formal['config_sha256']:
        raise ValueError('selected model configuration differs from formal freeze')
    if len(config['arms'])!=4 or config['seeds']!=[0,1,2] or set(selection['arms'])!={arm['id'] for arm in config['arms']}:
        raise ValueError('expected the four fixed arms and three fixed seeds')
    sample=load_json(output/'sample_queries.json')
    vocabulary_json=load_json(ROOT/'data/corpus_v03/vocabulary.json')
    vocabulary={token:i for i,token in enumerate(vocabulary_json['tokens'])}
    resources=load_json(experiment/'candidate_resources.json')
    candidates={int(length):tuple(row['text'] for row in entries) for length,entries in resources['by_length'].items()}
    if {length:len(spans) for length,spans in candidates.items()}!={5:10375,7:1504}:
        raise ValueError('fixed full candidate universe differs')
    selection_hash=sha(selection_path)
    cases=[];models=[]
    for arm in config['arms']:
        arm_id=arm['id'];selected=selection['arms'][arm_id]
        records=selected['models']
        if [record['seed'] for record in records]!=[0,1,2]:raise ValueError('selected seeds differ')
        for record in records:
            seed=record['seed'];path=(experiment/record['path']).resolve()
            if not path.is_relative_to(experiment.resolve()):raise ValueError('checkpoint escapes formal output')
            if sha(path)!=record['sha256']:raise ValueError('immutable selected checkpoint identity changed')
            model=ContextSpanMatcher.load(path)
            expected_config={**config['model'],**{key:arm[key] for key in ('kernel_width','channels','context_mode')},'seed':seed}
            if dict(model.config)!=expected_config or model.step_count!=selected['step']:
                raise ValueError('selected model architecture/update differs from frozen record')
            if dict(model.vocabulary)!=vocabulary or dict(model.candidates_by_length)!=candidates:
                raise ValueError('selected vocabulary/candidate identity differs')
            with tempfile.TemporaryDirectory(prefix='matching-v05-roundtrip-') as temporary:
                saved=model.save(Path(temporary)/'deployment_copy.npz')
                restored=ContextSpanMatcher.load(Path(temporary)/'deployment_copy.npz')
                state_exact=(dict(restored.config)==dict(model.config)
                    and dict(restored.vocabulary)==dict(model.vocabulary)
                    and dict(restored.candidates_by_length)==dict(model.candidates_by_length)
                    and restored.step_count==model.step_count and restored.last_optimizer==model.last_optimizer
                    and all(np.array_equal(getattr(model,collection)[name],getattr(restored,collection)[name])
                            for collection in ('params','adam_m','adam_v') for name in model.params))
                if not state_exact:raise ValueError('checkpoint deployment roundtrip state differs')
                for length in (5,7):
                    subset=[row for row in sample if row['length']==length]
                    queries=[VisibleQuery(**row['query']) for row in subset]
                    spans=candidates[length]
                    factors=model.score_factors(queries)
                    scores=model.score_all(queries)
                    rebuilt=factors['Q']@factors['E'].T+factors['bias']
                    public_factors_exact=(np.array_equal(scores,factors['scores']) and np.array_equal(scores,rebuilt)
                        and np.array_equal(factors['Q'],model.query_embeddings(queries)/model.temperature)
                        and np.array_equal(factors['E'],model.candidate_embeddings(length))
                        and np.array_equal(factors['bias'],np.zeros(len(spans))))
                    singles=np.vstack([model.score_all([query])[0] for query in queries])
                    deployed=restored.score_all(queries)
                    for index,(row,query) in enumerate(zip(subset,queries)):
                        batch_order=ranking(scores[index],spans)
                        single_order=ranking(singles[index],spans)
                        deployed_order=ranking(deployed[index],spans)
                        api=model.rank(query,limit=16)
                        maximum_error=float(np.max(np.abs(scores[index]-singles[index])))
                        top16=[spans[int(i)] for i in batch_order[:16]]
                        result={'arm':arm_id,'seed':seed,'selected_step':model.step_count,
                            'task_id':row['task_id'],'query_hash':row['query_hash'],'length':length,
                            'candidate_count':len(spans),'matrix_factor_reconstruction_exact':bool(public_factors_exact),
                            'max_abs_batch_single_score_error':maximum_error,
                            'batch_single_scores_within_tolerance':bool(maximum_error<=1e-10 and
                                np.allclose(scores[index],singles[index],rtol=1e-12,atol=1e-10)),
                            'batch_single_full_ranking_exact':bool(np.array_equal(batch_order,single_order)),
                            'rank_api_top16_exact':tuple(top16)==tuple(api['candidates']),
                            'roundtrip_scores_exact':bool(np.array_equal(scores[index],deployed[index])),
                            'roundtrip_full_ranking_exact':bool(np.array_equal(batch_order,deployed_order)),
                            'full_rank_sha256':digest(batch_order.tolist()),'top16':top16}
                        result['passed']=all(result[key] for key in (
                            'matrix_factor_reconstruction_exact','batch_single_scores_within_tolerance',
                            'batch_single_full_ranking_exact','rank_api_top16_exact',
                            'roundtrip_scores_exact','roundtrip_full_ranking_exact'))
                        cases.append(result)
                if sha(path)!=record['sha256']:raise ValueError('selected checkpoint changed during audit')
                models.append({'arm':arm_id,'seed':seed,'step':model.step_count,'path':record['path'],
                               'sha256':record['sha256'],'roundtrip_state_exact':bool(state_exact),
                               'roundtrip_file_sha256':saved['sha256'],
                               'roundtrip_serialized_bytes_equal_original':saved['sha256']==record['sha256'],
                               'roundtrip_copy_retention':'temporary copy removed after validation; original selected file unchanged'})
            print(json.dumps({'stage':'selected_artifact_audit','models_completed':len(models),'models_total':12}),flush=True)
    if len(models)!=12 or len(cases)!=192 or sha(selection_path)!=selection_hash:
        raise ValueError('audit model/case count or immutable selection changed')
    passed=all(case['passed'] for case in cases)
    report={'schema':'versematrix.matching-artifact-audit.v0.5','status':'passed' if passed else 'failed',
            'audit_stage':'separate_post_selection_engineering_only',
            'plan_sha256':sha(output/'plan_frozen.json'),'sample_sha256':sha(output/'sample_queries.json'),
            'selection_sha256':selection_hash,'models':len(models),'sampled_dev_queries':len(sample),
            'model_query_cases':len(cases),'full_vocabulary_scores_per_batch_case':{'5':10375,'7':1504},
            'maximum_batch_single_absolute_score_error':max(case['max_abs_batch_single_score_error'] for case in cases),
            'failed_cases':[{'arm':case['arm'],'seed':case['seed'],'task_id':case['task_id']} for case in cases if not case['passed']],
            'training_updates_performed':0,'targets_used':False,'test_or_confirmation_files_opened':False,
            'formal_training_or_selection_changed':False,
            'limits':['This checks public factorization, artifact state and batching/deployment consistency, not learned quality.',
                      'The core author implements this engineering audit; the separate reviewer supplies the independent mathematical oracle.',
                      'Queries are old-dev visible contexts selected by a fixed hash, not performance-selected examples.',
                      'No new checkpoint or candidate choice is promoted by this audit.']}
    write_new(output/'model_receipts.json',models)
    write_new(output/'case_receipts.jsonl',''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n'
                                                for row in cases).encode())
    write_new(output/'report.json',report)
    write_new(output/'manifest.json',{'status':report['status'],
        'files':{name:{'bytes':(output/name).stat().st_size,'sha256':sha(output/name)}
                 for name in ('plan_frozen.json','sample_queries.json','prepared.json','model_receipts.json','case_receipts.jsonl','report.json')},
        'formal_artifacts_unchanged':True})
    if not passed:raise ValueError('artifact audit failed; failure evidence retained without changing any model')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    print(json.dumps((prepare if args.stage=='prepare' else run)(args.experiment.resolve(),args.out.resolve()),
                     ensure_ascii=False,indent=2))
