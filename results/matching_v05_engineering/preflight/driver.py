"""Engineering-only throughput check: 20 updates, seed 172905, train only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,platform,re,sys,tempfile,time
import numpy as np

ROOT=Path('/workspace/scratch/c6b13db58303/wordconv-lab')
sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.corpus import make_tasks
from wordconv_lab.matching import ContextSpanMatcher
from wordconv_lab.restoration import VisibleQuery,digest
from wordconv_lab.retrieval import PhraseRetriever


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        handle.write('\n')


out=Path(tempfile.mkdtemp(prefix='matching-v05-preflight-',dir='/workspace/scratch/c6b13db58303/tmp'))
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
config={'purpose':'engineering throughput only; not a formal training run or model selection',
        'seed':172905,'optimizer_updates':20,'batch_size':32,'gradient_accumulation':1,
        'schedule_lengths':[5]*10+[7]*10,
        'architecture':{'embedding_dim':12,'representation_dim':16,'kernel_width':3,'channels':24,
                        'context_mode':'both','temperature':.1,'norm_epsilon':1e-6,'seed':172905},
        'optimizer':{'learning_rate':.003,'weight_decay':.0001,'max_grad_norm':5.},
        'candidate_vocabulary':'fixed train all-adjacent min-count-2, full softmax',
        'split':'train','dev_test_confirm_decoded':False,'model_selection_performed':False}
inputs=['data/corpus_v03/poems.jsonl','data/corpus_v03/vocabulary.json']
sources=['src/wordconv_lab/matching.py','src/wordconv_lab/retrieval.py',
         'src/wordconv_lab/restoration.py','src/wordconv_lab/corpus.py']
save(out/'freeze.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,
    'input_sha256':{name:sha(ROOT/name) for name in inputs},
    'source_sha256':{name:sha(ROOT/name) for name in sources},'driver_sha256':sha(out/'driver.py')})
marker=re.compile(rb'"split":"(train|dev|test)"')
train=[]
with (ROOT/inputs[0]).open('rb') as handle:
    for raw in handle:
        match=marker.search(raw)
        if match is None:raise ValueError('missing split marker')
        if match.group(1)!=b'train':continue
        train.append(json.loads(raw))
if len(train)!=2502:raise ValueError('unexpected train denominator')
vocabulary_json=json.loads((ROOT/inputs[1]).read_text())
vocabulary={token:i for i,token in enumerate(vocabulary_json['tokens'])}
started=time.perf_counter()
retriever=PhraseRetriever(train,vocabulary)
candidate_build_seconds=time.perf_counter()-started
pools=dict(retriever.candidates_by_length)
assert {length:len(spans) for length,spans in pools.items()}=={5:10375,7:1504}
eligible={length:sorted([task for task in make_tasks(train) if task['length']==length and task['target'] in pools[length]],key=lambda task:task['task_id'])
          for length in (5,7)}
candidate_ids={length:{span:i for i,span in enumerate(spans)} for length,spans in pools.items()}
rng=np.random.default_rng(172905)
batches=[]
for length in config['schedule_lengths']:
    batch=[eligible[length][int(i)] for i in rng.integers(0,len(eligible[length]),32)]
    batches.append((length,batch))
save(out/'sampling.json',{'eligible_train_tasks_by_length':{str(k):len(v) for k,v in eligible.items()},
     'candidate_sha256':{str(k):digest(v) for k,v in pools.items()},
     'updates':[{'length':length,'task_ids':[task['task_id'] for task in batch]} for length,batch in batches]})
started=time.perf_counter()
model=ContextSpanMatcher(vocabulary,pools,**config['architecture'])
initialization_seconds=time.perf_counter()-started
updates=[]
started=time.perf_counter()
for length,batch in batches:
    queries=[VisibleQuery.from_task(task) for task in batch]
    targets=np.asarray([candidate_ids[length][task['target']] for task in batch],dtype=np.int64)
    step_start=time.perf_counter()
    row=model.train_step(queries,targets,**config['optimizer'])
    row.update(length=length,seconds=time.perf_counter()-step_start)
    updates.append(row)
training_seconds=time.perf_counter()-started
assert model.step_count==20
checkpoint=model.save(out/'engineering_update20.npz')
restored=ContextSpanMatcher.load(out/'engineering_update20.npz')
for length in (5,7):
    query=[VisibleQuery.from_task(eligible[length][0])]
    factors=model.score_factors(query)
    np.testing.assert_array_equal(factors['scores'],factors['Q']@factors['E'].T+factors['bias'])
    np.testing.assert_array_equal(model.score_all(query),restored.score_all(query))
receipt={'status':'passed','purpose':config['purpose'],'formal_experiment':False,
         'train_poems':len(train),'seed':172905,'optimizer_updates':20,'training_task_presentations':640,
         'unique_training_tasks':len({task['task_id'] for _,batch in batches for task in batch}),
         'candidate_counts':{str(k):len(v) for k,v in pools.items()},
         'eligible_train_tasks_by_length':{str(k):len(v) for k,v in eligible.items()},
         'candidate_build_seconds':candidate_build_seconds,'initialization_seconds':initialization_seconds,
         'training_wall_seconds':training_seconds,'updates':updates,
         'per_length':{str(length):{'updates':10,'mean_seconds':float(np.mean([row['seconds'] for row in updates if row['length']==length])),
                                    'median_seconds':float(np.median([row['seconds'] for row in updates if row['length']==length]))}
                       for length in (5,7)},
         'parameter_summary':model.parameter_summary(),'checkpoint':checkpoint,
         'checks':{'full_softmax':'passed','all_20_updates_finite':'passed','both_lengths_qe_matrix_reconstruction':'exact',
                   'checkpoint_roundtrip_both_lengths':'exact','no_dev_test_confirm_decoded':True},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,
                        'thread_settings':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
         'limitations':['Timing only; no development or confirmation metric was computed.',
                        'This separate seed172905 checkpoint must never be selected as a formal experimental checkpoint.',
                        'Reported losses are engineering numerical diagnostics, not quality evidence.']}
save(out/'receipt.json',receipt)
print(json.dumps({'output_directory':str(out),**{key:receipt[key] for key in ('status','optimizer_updates','training_task_presentations',
    'eligible_train_tasks_by_length','candidate_counts','training_wall_seconds','per_length','checks')},
    'stored_parameters':receipt['parameter_summary']['stored_parameters'],
    'trainable_parameters':receipt['parameter_summary']['trainable_parameters']},ensure_ascii=False))
