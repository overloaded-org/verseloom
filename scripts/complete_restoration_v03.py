"""Complete the frozen v0.3 evaluation after an exactly recovered checkpoint.

No retraining, new model selection, data change or hyperparameter change occurs
here. The evaluation body is copied verbatim from the original frozen runner.
The original source/input hashes and ALL selected checkpoint hashes are required.
"""
from run_restoration_v03 import *


def complete(root,out):
    if (out/'report.json').exists():
        raise ValueError('refusing to overwrite a completed evaluation')
    manifest=json.loads((out/'manifest.json').read_text())
    if digest(CONFIG)!=manifest['config_sha256']:
        raise ValueError('original configuration changed')
    for key in ('source_sha256','input_sha256'):
        for name,expected in manifest[key].items():
            if sha(root/name)!=expected:
                raise ValueError('frozen source/input changed: '+name)
    frozen=json.loads((out/'selection_frozen.json').read_text())['models']
    for row in frozen:
        if sha(out/row['selected_model_path'])!=row['selected_model_sha256']:
            raise ValueError('selected checkpoint identity does not match original freeze')
    recovery=root/'results/restoration_v03_recovery/recovery_receipt.json'
    recovered=json.loads(recovery.read_text())
    if recovered['status']!='exact_frozen_checkpoint_recovered' or recovered['expected_sha256']!=recovered['replayed_sha256']:
        raise ValueError('recovery did not reproduce original selected bytes')
    write(out/'manifest_before_completion.json',manifest)
    completion={'resumed_at_utc':now(),'completion_source_sha256':sha(Path(__file__)),
                'original_manifest_sha256':sha(out/'manifest.json'),
                'selection_frozen_sha256':sha(out/'selection_frozen.json'),
                'recovery_receipt_sha256':sha(recovery),'model_selection_changed':False,
                'config_changed':False,'pool_changed':False,
                'prior_attempt':'checkpoint SHA guard interrupted before complete test report; three model predictions had been computed in memory',
                'evaluation_body':'verbatim original run_restoration_v03.py post-selection body'}
    write(out/'completion_frozen.json',completion)
    manifest['completion']=completion
    training=[json.loads((out/row['selected_model_path']).with_name('training_receipt.json').read_text()) for row in frozen]
    corpus=root/'data/corpus_v03'
    test=[t for t in load_rows(corpus/'tasks.jsonl') if t['split']=='test']
    pools={'test':load_rows(out/'pools/test.jsonl')}
    tokens=json.loads((corpus/'vocabulary.json').read_text())['tokens']
    vocab={x:i for i,x in enumerate(tokens)}
    data=json.loads((corpus/'candidate_spans.json').read_text())
    spans={int(length):[x['text'] for x in records] for length,records in data['by_length'].items()}
    # Wall time since original freeze includes the integrity recovery interruption.
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(manifest['frozen_at_utc'])).total_seconds()
    start=time.perf_counter()-elapsed
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
    complete(Path(__file__).resolve().parents[1],args.out)
