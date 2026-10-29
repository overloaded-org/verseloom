"""Frozen real-source restoration pilot; no literary-quality gold is inferred.

Run with a fresh output directory. Fitting and checkpoint choice see train/dev;
the final test block is reached once after all six model selections are fixed.
"""
from collections import Counter
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from wordconv_lab.learned_conv import LearnedLocalScorer
from wordconv_lab.restoration import (BidirNGram,VisibleQuery,digest,encode_candidates,
                                     top_candidate,family_metric,paired_family_interval)

CONFIG={
    "version":"0.3.0","task":"observed_source_initial_two_character_restoration",
    "literary_quality_gold":False,"formal_H_C_completed":False,
    "seeds":[0,1,2],"optimizer_updates":400,"batch_size":32,"gradient_accumulation":1,
    "training_candidates":8,"hard_negative_count":3,"proposal_pool_size":16,
    "learning_rate":.003,"weight_decay":.0001,"gradient_clip":5.,
    "optimizer":"Adam_decoupled_weight_decay","beta1":.9,"beta2":.999,"epsilon":1e-8,
    "checkpoint_updates":[100,200,300,400],
    "checkpoint_metric":"all_dev_tasks_equal_family_top1_uncovered_zero_earliest_tie",
    "arms":{"cnn_w3":{"kernel_width":3,"channels":12},
            "aligned_w1":{"kernel_width":1,"channels":20}},
    "embedding_dim":8,"fixed_feature_dim":4,
    "ngram":{"alpha":.1,"weights":[.1,.3,.6],"aligned_weight":.5,"aligned_strength":5.},
    "candidate_origin":"train_only_spans_count_at_least_2",
    "test_target_injection":False,"test_tie_rule":"lexicographic_candidate",
    "bootstrap_repeats":5000,"bootstrap_seed":2026102903,
    "dp_check":"first_24_test_task_hashes_per_length_for_each_selected_cnn_seed",
    "go_rule_proxy_only":{"min_absolute_family_accuracy_gain_over_each_comparator":.02,
                           "paired_ci_lower_gt":0},
    "scope":"fixed_shards_shape_only_poems_ngram_proposed_pool_not_poetry_quality",
}


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def load_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n" for x in rows))


def make_pools(tasks,spans,ngram):
    pools=[]
    start=time.perf_counter()
    for index,task in enumerate(tasks):
        query=VisibleQuery.from_task(task)
        pool=ngram.propose(query,spans[task['length']],CONFIG['proposal_pool_size'])
        pool.update(task_id=task['task_id'],family_id=task['family_id'],length=task['length'])
        pools.append(pool)
        if (index+1)%500==0:
            print(json.dumps({'stage':'candidate_proposals','done':index+1,'total':len(tasks),
                              'seconds':round(time.perf_counter()-start,2)}),flush=True)
    return pools


def arrays_for(tasks,pools,vocab):
    result={}
    for length in (5,7):
        indices=[i for i,t in enumerate(tasks) if t['length']==length]
        if indices:
            ids=np.stack([encode_candidates(VisibleQuery.from_task(tasks[i]),pools[i]['candidates'],vocab) for i in indices])
            result[length]=(np.array(indices),ids)
    return result


def infer(model,tasks,pools,arrays):
    scores=[None]*len(tasks)
    for length,(indices,ids) in arrays.items():
        for start in range(0,len(indices),64):
            batch=model.score_batch(ids[start:start+64])
            for j,values in zip(indices[start:start+64],batch):scores[int(j)]=values
    winners=[top_candidate(pool['candidates'],score) for pool,score in zip(pools,scores)]
    predictions=[pool['candidates'][i] for pool,i in zip(pools,winners)]
    correct=np.array([prediction==task['target'] for prediction,task in zip(predictions,tasks)],dtype=float)
    return correct,predictions,scores


def baseline_values(tasks,pools):
    predictions=[p['candidates'][top_candidate(p['candidates'],p['ngram_scores'])] for p in pools]
    return np.array([p==t['target'] for p,t in zip(predictions,tasks)],dtype=float),predictions


def training_candidates(tasks,pools,spans,seed):
    rng=np.random.default_rng(4100+seed)
    result=[]
    for task,pool in zip(tasks,pools):
        target=task['target']
        hard=[pool['candidates'][i] for i in sorted(range(len(pool['candidates'])),
               key=lambda i:(-pool['ngram_scores'][i],pool['candidates'][i]))
               if pool['candidates'][i]!=target][:CONFIG['hard_negative_count']]
        chosen=[target]+hard
        remaining=[x for x in spans[task['length']] if x not in chosen]
        needed=CONFIG['training_candidates']-len(chosen)
        if len(remaining)<needed:raise ValueError('not enough train-only negative candidates')
        chosen.extend(rng.choice(remaining,needed,replace=False).tolist())
        rng.shuffle(chosen)
        result.append({'task_id':task['task_id'],'candidates':chosen,
                       'gold_index':chosen.index(target),'candidate_set_hash':digest(chosen)})
    return result


def make_schedule(tasks,seed):
    rng=np.random.default_rng(9000+seed)
    groups={length:np.array([i for i,t in enumerate(tasks) if t['length']==length]) for length in (5,7)}
    if any(len(g)==0 for g in groups.values()):raise ValueError('both length strata need training tasks')
    probability=np.array([len(groups[l]) for l in (5,7)],dtype=float);probability/=probability.sum()
    schedule=[]
    for _ in range(CONFIG['optimizer_updates']):
        length=int(rng.choice([5,7],p=probability))
        schedule.append(rng.choice(groups[length],CONFIG['batch_size'],replace=True).tolist())
    return schedule


def train_one(tasks,train_pools,schedule,dev,dev_pools,dev_arrays,vocab,arm,seed,out):
    geometry=CONFIG['arms'][arm]
    model=LearnedLocalScorer(len(vocab),seed=seed,**geometry)
    encoded=[encode_candidates(VisibleQuery.from_task(t),p['candidates'],vocab) for t,p in zip(tasks,train_pools)]
    labels=np.array([p['gold_index'] for p in train_pools])
    history=[];losses=[];best=-1.;chosen=None
    run_dir=out/'models'/f'{arm}_seed{seed}';run_dir.mkdir(parents=True)
    start=time.perf_counter()
    for step,indices in enumerate(schedule,1):
        ids=np.stack([encoded[i] for i in indices])
        receipt=model.train_step(ids,labels[indices],learning_rate=CONFIG['learning_rate'],
                                weight_decay=CONFIG['weight_decay'],max_grad_norm=CONFIG['gradient_clip'])
        if receipt['step']!=step:raise AssertionError('one optimizer update per scheduled batch')
        losses.append(receipt['loss'])
        if step in CONFIG['checkpoint_updates']:
            correct,_,_=infer(model,dev,dev_pools,dev_arrays)
            metric=family_metric(dev,correct)
            row={'update':step,'mean_last_100_train_loss':float(np.mean(losses[-100:])),
                 'dev':metric,'elapsed_seconds':time.perf_counter()-start}
            history.append(row)
            if metric['family_macro']>best:
                best=metric['family_macro'];chosen=step
                model.save(run_dir/'selected.npz')
            print(json.dumps({'stage':'training_checkpoint','arm':arm,'seed':seed,**row}),flush=True)
    model.save(run_dir/'last.npz')
    exposure=Counter(i for batch in schedule for i in batch)
    selected=LearnedLocalScorer.load(run_dir/'selected.npz')
    if selected.step_count!=chosen:raise AssertionError('selected checkpoint identity mismatch')
    summary={'arm':arm,'seed':seed,'geometry':geometry,'core_parameters':model.core_parameter_count(),
             'trainable_parameters':model.trainable_parameter_count(),'optimizer_updates':model.step_count,
             'batch_size':CONFIG['batch_size'],'task_presentations':sum(exposure.values()),
             'unique_tasks_exposed':len(exposure),'eligible_train_tasks':len(tasks),
             'unique_families_exposed':len({tasks[i]['family_id'] for i in exposure}),
             'per_task_exposure':{tasks[i]['task_id']:count for i,count in sorted(exposure.items())},
             'selected_update':chosen,'selected_dev_family_top1':best,'history':history,
             'seconds_including_dev_evaluation_and_saves':time.perf_counter()-start,
             'selected_model_path':str((run_dir/'selected.npz').relative_to(out)),
             'selected_model_sha256':sha(run_dir/'selected.npz'),'last_model_sha256':sha(run_dir/'last.npz')}
    write(run_dir/'training_receipt.json',summary)
    return summary


def run(root,out):
    if out.exists():raise ValueError('refusing to overwrite existing experiment evidence')
    out.mkdir(parents=True)
    corpus=root/'data/corpus_v03'
    sources=[root/'src/wordconv_lab'/x for x in ('corpus.py','restoration.py','learned_conv.py','span_solver.py','factor_dp.py','convolution.py')]+[Path(__file__)]
    inputs=[corpus/name for name in ('poems.jsonl','tasks.jsonl','vocabulary.json','candidate_spans.json','build_manifest.json')]
    manifest={'schema':'versematrix.restoration-freeze.v0.3','stage':'frozen_before_model_evaluation',
              'frozen_at_utc':now(),'config':CONFIG,'config_sha256':digest(CONFIG),
              'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
              'input_sha256':{str(p.relative_to(root)):sha(p) for p in inputs},
              'external_preregistration':False,'test_status':'not_evaluated',
              'python':platform.python_version(),'numpy':np.__version__}
    write(out/'manifest.json',manifest)
    start=time.perf_counter()
    all_tasks=load_rows(corpus/'tasks.jsonl')
    poems=load_rows(corpus/'poems.jsonl')
    vocabulary=json.loads((corpus/'vocabulary.json').read_text())
    vocab={word:i for i,word in enumerate(vocabulary['tokens'])}
    data=json.loads((corpus/'candidate_spans.json').read_text())
    spans={int(length):[x['text'] for x in records] for length,records in data['by_length'].items()}
    train=[t for t in all_tasks if t['split']=='train' and t['target'] in spans[t['length']]]
    dev=[t for t in all_tasks if t['split']=='dev']
    test=[t for t in all_tasks if t['split']=='test']
    ng=BidirNGram(vocab,**CONFIG['ngram']).fit([p for p in poems if p['split']=='train'])
    write(out/'ngram_summary.json',ng.summary())
    print(json.dumps({'stage':'frozen','eligible_train_tasks':len(train),'dev_tasks':len(dev),'test_tasks':len(test)}),flush=True)
    # Proposal API has no labels; even test proposals do not receive target.
    pools={}
    for split,tasks in (('train',train),('dev',dev),('test',test)):
        pools[split]=make_pools(tasks,spans,ng)
        write_rows(out/'pools'/f'{split}.jsonl',pools[split])
        print(json.dumps({'stage':'pool_complete','split':split,'tasks':len(tasks)}),flush=True)
    dev_arrays=arrays_for(dev,pools['dev'],vocab)
    training=[]
    for seed in CONFIG['seeds']:
        candidates=training_candidates(train,pools['train'],spans,seed)
        schedule=make_schedule(train,seed)
        write_rows(out/'training'/f'candidates_seed{seed}.jsonl',candidates)
        write(out/'training'/f'schedule_seed{seed}.json',{'task_ids':[t['task_id'] for t in train],'batches':schedule})
        for arm in CONFIG['arms']:
            receipt=train_one(train,candidates,schedule,dev,pools['dev'],dev_arrays,vocab,arm,seed,out)
            training.append(receipt)
    # This receipt is saved BEFORE any final test model prediction or selection.
    frozen=[{k:r[k] for k in ('arm','seed','selected_update','selected_dev_family_top1',
              'selected_model_path','selected_model_sha256')} for r in training]
    write(out/'selection_frozen.json',{'frozen_at_utc':now(),'test_evaluated':False,'models':frozen})
    print(json.dumps({'stage':'all_models_selected','test_evaluation':'starting_once'}),flush=True)
    test_arrays=arrays_for(test,pools['test'],vocab)
    correctness={};predictions={};scores_by_model={};models={}
    correctness['ngram'],predictions['ngram']=baseline_values(test,pools['test'])
    scores_by_model['ngram']=[np.array(p['ngram_scores']) for p in pools['test']]
    for row in frozen:
        key=f"{row['arm']}_seed{row['seed']}"
        path=out/row['selected_model_path']
        if sha(path)!=row['selected_model_sha256']:raise ValueError('selected model changed')
        model=LearnedLocalScorer.load(path)
        models[key]=model
        correctness[key],predictions[key],scores_by_model[key]=infer(model,test,pools['test'],test_arrays)
    for arm in CONFIG['arms']:
        correctness[arm+'_seed_mean']=np.mean([correctness[f'{arm}_seed{s}'] for s in CONFIG['seeds']],axis=0)
    covered=np.array([t['target'] in p['candidates'] for t,p in zip(test,pools['test'])])
    vocab_covered=np.array([t['target'] in spans[t['length']] for t in test])
    metrics={key:family_metric(test,values) for key,values in correctness.items()}
    for key,values in correctness.items():
        metrics[key]['covered_task_top1']=float(np.mean(values[covered])) if covered.any() else None
        metrics[key]['by_length']={str(length):family_metric([t for t in test if t['length']==length],
            values[[i for i,t in enumerate(test) if t['length']==length]]) for length in (5,7)}
    comparisons={key:paired_family_interval(test,correctness['cnn_w3_seed_mean'],correctness[key],
                   seed=CONFIG['bootstrap_seed'],repeats=CONFIG['bootstrap_repeats'])
                 for key in ('aligned_w1_seed_mean','ngram')}
    gate=all(x['status']=='computed' and x['mean_difference']>=.02 and x['interval'][0]>0 for x in comparisons.values())
    rows=[]
    for i,(task,pool) in enumerate(zip(test,pools['test'])):
        rows.append({'task_id':task['task_id'],'poem_id':task['poem_id'],'family_id':task['family_id'],
                     'length':task['length'],'upper_line':task['upper_line'],'lower_suffix':task['lower_suffix'],
                     'target':task['target'],'candidate_set_hash':pool['candidate_set_hash'],
                     'in_training_candidate_vocabulary':bool(vocab_covered[i]),'covered':bool(covered[i]),
                     'predictions':{key:value[i] for key,value in predictions.items()},
                     'correct':{key:float(value[i]) for key,value in correctness.items()},
                     'scores':{key:value[i].tolist() for key,value in scores_by_model.items()}})
    write_rows(out/'test_predictions.jsonl',rows)
    # Mechanism audit, predeclared by task hash rather than chosen for success.
    from wordconv_lab.span_solver import solve_initial_span
    audit_indices=[]
    for length in (5,7):
        candidates=[i for i,t in enumerate(test) if t['length']==length]
        audit_indices.extend(sorted(candidates,key=lambda i:digest(test[i]['task_id']))[:24])
    dp_checks=[]
    for seed in CONFIG['seeds']:
        key=f'cnn_w3_seed{seed}'
        model=models[key]
        for i in audit_indices:
            task,pool=test[i],pools['test'][i]
            result=solve_initial_span(model,vocab,task['upper_line'],task['lower_suffix'],pool['candidates'])
            maximum=float(np.max(scores_by_model[key][i]))
            chosen_index=pool['candidates'].index(result.chosen_span)
            chosen_original=float(scores_by_model[key][i][chosen_index])
            error=max(abs(result.score-chosen_original),abs(maximum-chosen_original))
            if error>1e-10:raise AssertionError('trained span-constrained DP disagrees with original pool enumeration')
            dp_checks.append({'task_id':task['task_id'],'seed':seed,'length':task['length'],
                              'candidate_set_hash':pool['candidate_set_hash'],'solver_candidate_hash':result.candidate_hash,
                              'chosen_span':result.chosen_span,'candidate_count':len(pool['candidates']),
                              'maximum_score':maximum,'solver_score':result.score,'max_abs_error':error,
                              'in_original_pool':True,'dp_factor_calls':result.dp.factor_calls})
    write(out/'trained_dp_checks.json',{'status':'all_passed','checks':dp_checks,
          'scope':'trained_objective_and_original_span_pool_only_not_poetic_quality'})
    # Fixed task-hash diagnostic sample, not a subjective quality showcase.
    sample=sorted(rows,key=lambda row:digest(row['task_id']))[:12]
    write(out/'fixed_examples.json',{'selection':'first_12_task_hashes_no_score_based_selection','examples':sample})
    summary={'version':'0.3.0','status':'executed_real_source_restoration_pilot',
             'human_evaluated':False,'certified_lushi':False,'test_tasks':len(test),
             'test_poems':len({t['poem_id'] for t in test}),'test_families':len({t['family_id'] for t in test}),
             'vocabulary_coverage':family_metric(test,vocab_covered.astype(float)),
             'proposal_pool_recall':family_metric(test,covered.astype(float)),
             'test_vocab_covered_tasks':int(vocab_covered.sum()),'test_pool_covered_tasks':int(covered.sum()),
             'metrics':metrics,'paired_comparisons':comparisons,
             'proxy_promotion_decision':'PASS_PROXY_ONLY' if gate else 'DO_NOT_PROMOTE_CNN_FROM_THIS_PILOT',
             'formal_literary_quality_gate':'NOT_EVALUATED',
             'training_runs':[{k:v for k,v in receipt.items() if k!='per_task_exposure'} for receipt in training],
             'dp_audit_cases':len(dp_checks),'dp_max_abs_error':max(c['max_abs_error'] for c in dp_checks),
             'elapsed_seconds':time.perf_counter()-start,'config_sha256':digest(CONFIG),
             'limitations':['Observed source exact-match is not a literary-quality gold label.',
                 'Eight-line source shape is not certified regulated verse or contextual prosody.',
                 'All systems rerank the same ngram-proposed pool; this does not test all-vocabulary architecture superiority.',
                 'Training pools inject train targets legitimately; evaluation pools never inject dev/test targets.',
                 'First fixed corpus shards and surface families do not imply author-held-out or all-poetry generalization.',
                 'Three fixed seeds are averaged before family bootstrap; intervals do not estimate unrestricted seed uncertainty.',
                 'Width-one is a horizontal-locality ablation; formal same-receptive-field attention and human review remain unrun.']}
    write(out/'report.json',summary)
    manifest.update(stage='completed',completed_at_utc=now(),test_status='evaluated_now_exposed',
                    selected_models_sha256=sha(out/'selection_frozen.json'))
    write(out/'manifest.json',manifest)
    print(json.dumps({'stage':'completed','decision':summary['proxy_promotion_decision'],
                      'test_tasks':len(test),'dp_checks':len(dp_checks),'elapsed_seconds':summary['elapsed_seconds']}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/restoration_v03'))
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.out)
