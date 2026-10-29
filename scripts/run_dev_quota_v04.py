"""Post-formal DEV-ONLY fixed-budget old/new phrase quota diagnostic.

Every file read is named below. No confirmation task, prediction or report is
opened; old test corpus rows are skipped before JSON decoding. The unchanged
v0.4 retriever reads only train poems and VisibleQuery. Ten predeclared quota
schemes are all reported, with a descriptive dev-only best and no promotion.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wordconv_lab.corpus import make_tasks
from wordconv_lab.restoration import VisibleQuery,digest
from wordconv_lab.retrieval import PhraseRetriever

DEFAULT_OUTPUT=ROOT/'results/dev_quota_v04'
SPLIT_PATTERN=re.compile(rb'"split":"(train|dev|test)"')
QUOTAS=(0,4,8,12,16)
NEW_RANKERS=(
    {'id':'ngram','alignment_weight':0.0,'boundary_weight':0.0},
    {'id':'a025_b025','alignment_weight':0.25,'boundary_weight':0.25},
)
INPUT_NAMES=(
    'data/corpus_v03/poems.jsonl',
    'data/corpus_v03/vocabulary.json',
    'data/corpus_v03/candidate_spans.json',
    'data/corpus_v03/build_manifest.json',
    'results/dev_candidate_diagnostic_v03/candidate_counts.json',
    'results/retrieval_v04/dev_predictions.jsonl',
)
SOURCE_NAMES=(
    'scripts/run_dev_quota_v04.py',
    'src/wordconv_lab/retrieval.py',
    'src/wordconv_lab/restoration.py',
    'src/wordconv_lab/corpus.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def row_bytes(rows):
    return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
                   for row in rows).encode()


def write_new(path,data):
    if path.exists():
        raise ValueError('refusing to overwrite existing diagnostic evidence: '+str(path))
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def grid():
    return [{'id':f'q{quota:02d}_{ranker["id"]}',
             'old_quota':quota,'new_quota':16-quota,'new_ranker':dict(ranker)}
            for quota in QUOTAS for ranker in NEW_RANKERS]


def design():
    return {
        'schema':'versematrix.dev-quota-diagnostic.v0.4',
        'stage':'post_formal_experiment_exploratory_dev_only',
        'previous_dev_use':'This dev set was already used in v0.3 and v0.4 development and selection.',
        'fit_split':'train','evaluate_split':'dev','expected_train_poems':2502,
        'expected_dev_poems':547,'expected_dev_tasks':2188,'output_k':16,
        'old_set_N':'original even-line initial two-character spans, count >=2 in train',
        'wide_set_W':'all eight rows, within-line adjacent two-character spans, count >=2 in train',
        'new_set':'W minus N, disjoint exact source strings within each poem length',
        'old_ranking':'unchanged original BidirNGram, score descending and source-text lexical tie',
        'new_ranking':'unchanged PhraseRetriever, fixed two choices; no new weight tuning',
        'grid_in_tie_order':grid(),
        'pool':'take q from N and 16-q from W minus N, concatenate; this is a set-recall pool, not a joint top1 ranking',
        'insufficient_bucket_policy':'fail; no adaptive quota backfill',
        'descriptive_selection':'largest exact rational equal-family all-task recall@16; first scheme in grid order on ties',
        'oov_policy':'retain every dev task; absent target is a zero',
        'formal_promotion_changed':False,'production_default_changed':False,
        'confirmation_evaluation':False,'confirmation_files_opened':False,
        'old_test_records_decoded':False,'target_injection':False,
        'normalization':'none; exact source code points',
        'checks':['disjoint N and W-minus-N','exactly 16 distinct complete source spans per task/scheme',
                  'requested bucket quotas exactly satisfied','q16 pool and hit exactly replay v0.4 dev narrow baseline',
                  'all candidate origins in train','all ten predeclared schemes reported'],
        'limits':'Development exploration only; no independent improvement claim, literary-quality claim, or equal-computation claim.',
    }


def read_train_dev():
    selected={'train':[],'dev':[]}
    with (ROOT/'data/corpus_v03/poems.jsonl').open('rb') as handle:
        for raw in handle:
            marker=SPLIT_PATTERN.search(raw)
            if marker is None:
                raise ValueError('missing canonical split marker')
            split=marker.group(1).decode('ascii')
            if split not in selected:
                continue
            poem=json.loads(raw)
            if poem['split']!=split:
                raise ValueError('split marker mismatch')
            selected[split].append(poem)
    if len(selected['train'])!=2502 or len(selected['dev'])!=547:
        raise ValueError('unexpected frozen training/development denominator')
    return selected


def check_candidate_resources(model,train):
    resources=model.candidate_resources()
    prior=json.loads((ROOT/'data/corpus_v03/candidate_spans.json').read_text())
    expected_old={length:{item['text']:item['count'] for item in items}
                  for length,items in prior['by_length'].items()}
    actual_old={length:{item['text']:item['narrow_count'] for item in items if item['narrow_eligible']}
                for length,items in resources['by_length'].items()}
    expected_wide=json.loads((ROOT/'results/dev_candidate_diagnostic_v03/candidate_counts.json').read_text())['all_adjacent_min2']
    actual_wide={length:{item['text']:item['count'] for item in items}
                 for length,items in resources['by_length'].items()}
    if actual_old!=expected_old or actual_wide!=expected_wide:
        raise ValueError('training-source candidate identity/counts changed')
    train_ids={p['poem_id'] for p in train}
    if any(not row['source_poem_ids'] or not set(row['source_poem_ids'])<=train_ids
           for items in resources['by_length'].values() for row in items):
        raise ValueError('candidate origin outside train')
    counts={length:{'N':len(actual_old[length]),'W_minus_N':len(actual_wide[length])-len(actual_old[length]),
                    'W':len(actual_wide[length])} for length in ('5','7')}
    if any(n['N']<16 or n['W_minus_N']<16 for n in counts.values()):
        raise ValueError('fixed 16-slot quota is unavailable in a bucket')
    return counts


def baseline_rows(tasks):
    rows={}
    with (ROOT/'results/retrieval_v04/dev_predictions.jsonl').open() as handle:
        for raw in handle:
            row=json.loads(raw)
            if row['task_id'] in rows:
                raise ValueError('duplicate formal dev evidence task')
            rows[row['task_id']]=row
    if set(rows)!={task['task_id'] for task in tasks}:
        raise ValueError('formal dev baseline task set differs')
    for task in tasks:
        row=rows[task['task_id']]
        if any(row[key]!=task[key] for key in ('poem_id','family_id','length','target')):
            raise ValueError('formal dev baseline identity mismatch')
        if row['query_hash']!=digest({key:getattr(VisibleQuery.from_task(task),key)
                                     for key in ('length','upper_line','lower_suffix','start','mask_length')}):
            raise ValueError('formal dev baseline visible query mismatch')
    return rows


def summarize(rows,method_id,length=None):
    rows=[row for row in rows if length is None or row['length']==length]
    groups=defaultdict(lambda:[0,0])
    hits=gains=losses=both=neither=0
    for row in rows:
        hit=int(row['schemes'][method_id]['hit'])
        old=int(row['narrow_reference_hit'])
        hits+=hit
        groups[row['family_id']][0]+=hit
        groups[row['family_id']][1]+=1
        gains+=int(hit and not old)
        losses+=int(old and not hit)
        both+=int(hit and old)
        neither+=int(not hit and not old)
    if not rows:
        raise ValueError('empty development metric slice')
    exact=sum((Fraction(a,b) for a,b in groups.values()),Fraction())/len(groups)
    return {'tasks':len(rows),'families':len(groups),'hits':hits,'task_micro_recall_at_16':hits/len(rows),
            'family_macro_recall_at_16':float(exact),
            'exact_family_recall':{'numerator':exact.numerator,'denominator':exact.denominator},
            'versus_original_narrow':{'gain_tasks':gains,'loss_tasks':losses,'both_hit':both,'both_miss':neither}}


def readme(report):
    lines=['# v0.4 开发集配额诊断','',
           '这是正式实验之后的探索性开发工作。开发集已被多次使用，结果不能作为独立确认；没有读取或评测任何确认集文件，也没有改变正式晋级判断或默认方案。','',
           '## 固定设计','',
           '- 总池大小固定为 16。旧集合 N 是原训练下句首二字候选；新集合是宽词表 W 去除 N 后的互斥剩余片段。',
           '- 旧候选配额依次为 0、4、8、12、16；旧集合始终采用原 ngram 排序。新集合分别采用 ngram 与固定 A=0.25、B=0.25 排序，全部 10 个方案公开。',
           '- 取各自排序前缀后合并为召回候选集合，不能把这个拼接顺序解释成联合 top1 排名。',
           '- 全部 2,188 道开发题保留，包括词表外原文；同一诗族内先平均，再对诗族等权平均。',
           '- 每个候选都是训练来源中的完整二字片段。N 与 W−N 不重叠；每题恰好 16 个不同片段且配额严格一致；q=16 逐题候选池和命中与正式 dev 窄基线完全一致。','',
           '## 全部结果','',
           '| 旧配额 | 新片段排序 | 命中 / 2188 | 题目召回 | 诗族等权召回 | 较旧基线新增 / 丢失 |',
           '|---:|---|---:|---:|---:|---:|']
    for result in report['results']:
        m=result['method'];s=result['overall'];delta=s['versus_original_narrow']
        lines.append(f"| {m['old_quota']} | {m['new_ranker']['id']} | {s['hits']} | {s['task_micro_recall_at_16']:.4%} | {s['family_macro_recall_at_16']:.4%} | {delta['gain_tasks']} / {delta['loss_tasks']} |")
    best=report['descriptive_dev_best']
    lines.extend(['',f"本次开发集描述性最优：`{best['method']['id']}`，族等权召回 {best['overall']['family_macro_recall_at_16']:.4%}。完全同分按照冻结网格顺序选择；这不是正式提升决定。",'',
                  '## 复现','',
                  '在项目根目录运行 `python scripts/run_dev_quota_v04.py --check`，会校验冻结代码与输入身份，然后重建所有确定性结果并逐字节比较。默认非 check 模式拒绝覆盖已有证据。','',
                  '冻结记录包含设计、脚本和输入 SHA。旧语料文件的整文件 SHA 会覆盖混合文件的原始字节，但旧 test 行在 JSON 解码前跳过。正式 dev 预测文件仅用于窄基线回放。未打开任何确认任务、预测或报告，未打开神经模型文件。','',
                  '计算预算只声称输出 K 固定，未声称评分计算量相等。该诊断只描述原文片段召回，未评估文学质量。',''])
    return '\n'.join(lines).encode()


def run(output,check=False):
    config=design()
    frozen={'config':config,
            'source_sha256':{name:sha(ROOT/name) for name in SOURCE_NAMES},
            'input_sha256':{name:sha(ROOT/name) for name in INPUT_NAMES}}
    output.mkdir(parents=True,exist_ok=True)
    freeze_path=output/'freeze.json'
    if check:
        previous=json.loads(freeze_path.read_text())
        if any(previous[key]!=value for key,value in frozen.items()):
            raise ValueError('frozen diagnostic design/code/input changed')
    else:
        frozen['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        frozen['status']='frozen_before_any_quota_ranking_or_dev_hit_calculation'
        write_new(freeze_path,encoded(frozen))

    # No candidate fitting, ranking, or target inspection occurs before freeze.
    build=json.loads((ROOT/'data/corpus_v03/build_manifest.json').read_text())
    for name in ('poems.jsonl','vocabulary.json','candidate_spans.json'):
        if sha(ROOT/'data/corpus_v03'/name)!=build['outputs'][name]['sha256']:
            raise ValueError('old frozen corpus identity differs: '+name)
    data=read_train_dev()
    tasks=sorted(make_tasks(data['dev']),key=lambda task:task['task_id'])
    if len(tasks)!=2188 or len({task['task_id'] for task in tasks})!=2188:
        raise ValueError('unexpected dev task denominator or duplicate IDs')
    vocabulary=json.loads((ROOT/'data/corpus_v03/vocabulary.json').read_text())
    model=PhraseRetriever(data['train'],{token:i for i,token in enumerate(vocabulary['tokens'])})
    candidate_counts=check_candidate_resources(model,data['train'])
    baseline=baseline_rows(tasks)
    methods=grid()
    rows=[]
    source_bucket=Counter()
    for index,task in enumerate(tasks):
        if task['split']!='dev':
            raise ValueError('non-dev query entered diagnostic')
        query=VisibleQuery.from_task(task)
        components=model.score_components(query)
        spans=components['candidates']
        narrow_mask=components['narrow_mask']
        old_set={span for span,old in zip(spans,narrow_mask) if old}
        new_set={span for span,old in zip(spans,narrow_mask) if not old}
        if old_set&new_set or old_set|new_set!=set(spans):
            raise ValueError('candidate buckets are not a disjoint partition')
        old_rank=model.rank_components(components,pool='narrow')
        new_orders={}
        new_score_maps={}
        for ranker in NEW_RANKERS:
            ranked=model.rank_components(components,ranker['alignment_weight'],ranker['boundary_weight'],pool='wide')
            selected=[(span,float(score)) for span,score in zip(ranked['candidates'],ranked['scores']) if span in new_set]
            if len(selected)!=len(new_set):
                raise ValueError('new ranking lost or duplicated a candidate')
            new_orders[ranker['id']]=tuple(span for span,_ in selected)
            new_score_maps[ranker['id']]={span:score for span,score in selected}

        # Only after all complete rankings are fixed is the source target used.
        target=task['target']
        bucket='N' if target in old_set else 'W_minus_N' if target in new_set else 'outside_W'
        source_bucket[bucket]+=1
        old_top=list(old_rank['candidates'][:16])
        reference=baseline[task['task_id']]['methods']['narrow_ngram']
        reference_rank=reference['target_rank']
        reference_hit=reference_rank is not None and reference_rank<=16
        if old_top!=reference['top_candidates'][:16] or (target in old_top)!=reference_hit:
            raise ValueError('q16 differs from formal dev narrow baseline')
        if digest({'query':{'length':query.length,'upper_line':query.upper_line,'lower_suffix':query.lower_suffix,
                            'start':query.start,'mask_length':query.mask_length},'candidates':sorted(old_top)})!=reference['top16_set_hash']:
            raise ValueError('q16 candidate set hash differs from formal dev baseline')
        row={key:task[key] for key in ('task_id','poem_id','family_id','length','target')}
        row.update(split='dev',query_hash=components['query_hash'],target_source_bucket=bucket,
                   narrow_reference_hit=reference_hit,schemes={})
        for method in methods:
            q=method['old_quota'];ranker_id=method['new_ranker']['id']
            old=list(old_rank['candidates'][:q])
            new=list(new_orders[ranker_id][:16-q])
            pool=old+new
            if (len(old)!=q or len(new)!=16-q or len(pool)!=16 or len(set(pool))!=16
                    or not set(old)<=old_set or not set(new)<=new_set
                    or any(len(span)!=2 or span not in spans for span in pool)):
                raise ValueError('quota, source-span integrity, or fixed-budget assertion failed')
            row['schemes'][method['id']]={
                'source_quota':{'N':q,'W_minus_N':16-q},
                'old_candidates':old,'new_candidates':new,'candidate_pool':pool,
                'old_source_scores':[float(value) for value in old_rank['scores'][:q]],
                'new_source_scores':[new_score_maps[ranker_id][span] for span in new],
                'candidate_set_hash':digest({'query_hash':components['query_hash'],'candidates':sorted(pool)}),
                'hit':target in pool,
            }
            if q==16 and (pool!=old_top or row['schemes'][method['id']]['hit']!=reference_hit):
                raise ValueError('q16 replay assertion failed')
        rows.append(row)
        if (index+1)%400==0 or index+1==len(tasks):
            print(json.dumps({'stage':'dev_quota_only','completed':index+1,'total':len(tasks)}),flush=True)
    results=[{'method':method,'overall':summarize(rows,method['id']),
              'by_length':{str(length):summarize(rows,method['id'],length) for length in (5,7)}}
             for method in methods]
    exact_values=[Fraction(result['overall']['exact_family_recall']['numerator'],
                           result['overall']['exact_family_recall']['denominator']) for result in results]
    best_index=max(range(len(results)),key=lambda i:exact_values[i])
    report={
        'schema':'versematrix.dev-quota-results.v0.4','status':'completed_exploratory_dev_only',
        'train_poems':2502,'dev_poems':547,'dev_tasks':2188,'dev_families':len({row['family_id'] for row in rows}),
        'candidate_bucket_counts':candidate_counts,'dev_target_bucket_counts':dict(sorted(source_bucket.items())),
        'grid_results_reported':len(results),'results':results,'descriptive_dev_best':results[best_index],
        'best_tie_rule':'first exact rational maximum in the frozen grid order',
        'checks':{'train_candidate_identity':'passed','disjoint_buckets':'passed','complete_source_spans':'passed',
                  'fixed_16_and_exact_quotas':'passed','all_2188_q16_pools_hits_and_hashes_replay_formal_dev':'passed'},
        'compute_accounting':{'shared_component_query_evaluations':len(rows),
                              'shared_ngram_candidate_evaluations':sum(candidate_counts[str(row['length'])]['W'] for row in rows),
                              'full_rankings_per_query':3,'quota_sets_per_query':10,
                              'fixed_output_k':16,'equal_computation_claim':False},
        'confirmation_tasks_evaluated':0,'confirmation_files_opened':False,
        'formal_promotion_changed':False,'production_default_changed':False,
        'limits':['Post-formal exploratory dev diagnostic; dev was already used for development and selection.',
                  'Every predeclared quota and new-fragment ranker is reported; descriptive best is not independent evidence.',
                  'No confirmation tasks, predictions, reports or neural model files were opened.',
                  'Old test corpus rows were skipped before decoding; whole-file SHA reads the mixed input bytes only.',
                  'Pools are exact two-character source spans, not a verified word segmentation or literary-quality gold.',
                  'Concatenated old/new pool order is not a joint top1 ranking.',
                  'No original corpus, frozen scoring code, formal experiment, promotion or production default was changed.'],
    }
    payloads={'report.json':encoded(report),'dev_quota_rows.jsonl':row_bytes(rows),
              'README.md':readme(report)}
    manifest={'status':'completed','freeze_sha256':sha(freeze_path),
              'outputs':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()},
              'dev_only':True,'confirmation_evaluation_performed':False,
              'original_experiment_changed':False}
    payloads['manifest.json']=encoded(manifest)
    for name,data_bytes in payloads.items():
        path=output/name
        if check:
            if path.read_bytes()!=data_bytes:
                raise ValueError('deterministic rebuilt bytes differ: '+name)
        else:
            write_new(path,data_bytes)
    return {'status':'deterministic_check_passed' if check else report['status'],
            'dev_tasks':report['dev_tasks'],'dev_families':report['dev_families'],
            'candidate_bucket_counts':candidate_counts,'dev_target_bucket_counts':report['dev_target_bucket_counts'],
            'results':[{'id':result['method']['id'],**result['overall']} for result in results],
            'descriptive_dev_best':results[best_index]['method']['id'],
            'confirmation_tasks_evaluated':0,'formal_promotion_changed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    print(json.dumps(run(args.out.resolve(),args.check),ensure_ascii=False,indent=2))
