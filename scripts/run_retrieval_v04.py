"""Development selection followed by one frozen independent retrieval check.

Old test rows are skipped before JSON decoding. All eligible dev/confirmation
tasks remain in denominators. A source prefix is an observed-text target, not
a literary-quality label. No target is passed to the retriever.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from wordconv_lab.corpus import make_tasks
from wordconv_lab.restoration import BidirNGram, VisibleQuery, digest, family_metric
from wordconv_lab.retrieval import PhraseRetriever

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/retrieval_v0.4.json'
SPLIT_PATTERN = re.compile(rb'"split":"(train|dev|test)"')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new(path, value):
    """Create durable evidence once; never rewrite a completed or frozen item."""
    if path.exists():
        raise ValueError('refusing to overwrite evidence: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_rows(path, rows):
    if path.exists():
        raise ValueError('refusing to overwrite rows: ' + str(path))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def grid_methods(config):
    methods = []
    for i, a in enumerate(config['weight_values_in_order']):
        for j, b in enumerate(config['weight_values_in_order']):
            methods.append({'id': f'wide_a{i}_b{j}', 'pool': 'wide',
                            'alignment_weight': a, 'boundary_weight': b})
    return methods


def narrow_method():
    return {'id': 'narrow_ngram', 'pool': 'narrow',
            'alignment_weight': 0.0, 'boundary_weight': 0.0}


def source_paths():
    return [ROOT / name for name in (
        'scripts/run_retrieval_v04.py', 'src/wordconv_lab/retrieval.py',
        'src/wordconv_lab/restoration.py', 'src/wordconv_lab/corpus.py',
        'src/wordconv_lab/confirmation.py', 'scripts/build_confirmation_v04.py',
        'configs/retrieval_v0.4.json')]


def input_paths():
    return [ROOT / name for name in (
        'data/corpus_v03/poems.jsonl', 'data/corpus_v03/vocabulary.json',
        'data/corpus_v03/candidate_spans.json', 'data/corpus_v03/build_manifest.json',
        'results/dev_candidate_diagnostic_v03/candidate_counts.json',
        'data/confirmation_v04/poems.jsonl', 'data/confirmation_v04/tasks.jsonl',
        'data/confirmation_v04/freeze.json', 'data/confirmation_v04/build_manifest.json',
        'data/confirmation_v04/source_manifest.json')]


def identity(paths):
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def check_frozen(freeze):
    for category in ('source_sha256', 'input_sha256'):
        for name, expected in freeze[category].items():
            if sha(ROOT / name) != expected:
                raise ValueError('frozen identity changed: ' + name)
    if digest(read_json(CONFIG_PATH)) != freeze['config_sha256']:
        raise ValueError('frozen configuration changed')


def read_train_dev():
    result = {'train': [], 'dev': []}
    with (ROOT / 'data/corpus_v03/poems.jsonl').open('rb') as handle:
        for raw in handle:
            marker = SPLIT_PATTERN.search(raw)
            if marker is None:
                raise ValueError('missing canonical split marker')
            split = marker.group(1).decode('ascii')
            if split not in result:
                continue
            poem = json.loads(raw)
            if poem['split'] != split:
                raise ValueError('split mismatch')
            result[split].append(poem)
    if len(result['train']) != 2502 or len(result['dev']) != 547:
        raise ValueError('unexpected frozen training/development counts')
    return result


def make_retriever(train):
    data = read_json(ROOT / 'data/corpus_v03/vocabulary.json')
    vocabulary = {word: index for index, word in enumerate(data['tokens'])}
    return PhraseRetriever(train, vocabulary), vocabulary


def audit_candidate_identity(resources, train):
    prior = read_json(ROOT / 'data/corpus_v03/candidate_spans.json')
    diagnostic = read_json(ROOT / 'results/dev_candidate_diagnostic_v03/candidate_counts.json')
    expected_narrow = {length: {item['text']: item['count'] for item in items}
                       for length, items in prior['by_length'].items()}
    actual_narrow = {length: {item['text']: item['narrow_count'] for item in items if item['narrow_eligible']}
                     for length, items in resources['by_length'].items()}
    actual_wide = {length: {item['text']: item['count'] for item in items}
                   for length, items in resources['by_length'].items()}
    if actual_narrow != expected_narrow:
        raise ValueError('narrow candidate identity/frequencies differ from frozen v0.3 baseline')
    if actual_wide != diagnostic['all_adjacent_min2']:
        raise ValueError('wide candidate identity/frequencies differ from independent train-only counts')
    train_ids = {poem['poem_id'] for poem in train}
    if any(not item['source_poem_ids'] or not set(item['source_poem_ids']) <= train_ids
           for items in resources['by_length'].values() for item in items):
        raise ValueError('candidate origin outside training poems')
    return {'status': 'passed', 'narrow_text_and_counts_exact_v03': True,
            'wide_text_and_counts_exact_independent_v03_diagnostic': True,
            'all_candidate_origins_train_only': True,
            'counts': {length: {'narrow': len(actual_narrow[length]), 'wide': len(items)}
                       for length, items in actual_wide.items()}}


def score_ranking(components, method):
    spans = tuple(components['candidates'])
    if spans != tuple(sorted(set(spans))):
        raise ValueError('candidate identities must be unique and lexically ordered')
    scores = (np.asarray(components['ngram_scores'], dtype=float)
              + method['alignment_weight'] * np.asarray(components['alignment_bonus'])
              + method['boundary_weight'] * np.asarray(components['boundary_bonus']))
    if scores.shape != (len(spans),) or not np.isfinite(scores).all():
        raise ValueError('nonfinite or wrong-shaped retrieval score')
    eligible = (np.flatnonzero(components['narrow_mask']) if method['pool'] == 'narrow'
                else np.arange(len(spans)))
    if not len(eligible):
        raise ValueError('empty candidate vocabulary')
    order = eligible[np.argsort(-scores[eligible], kind='stable')]
    return spans, scores, order


def exact_family_recall(rows, method_id, k=16):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        rank = row['methods'][method_id]['target_rank']
        group = groups[row['family_id']]
        group[0] += int(rank is not None and rank <= k)
        group[1] += 1
    if not groups:
        raise ValueError('cannot select on an empty development set')
    return sum((Fraction(a, b) for a, b in groups.values()), Fraction()) / len(groups)


def select_method(rows, methods, k=16):
    # Fraction comparison avoids a different rounded float deciding an exact tie.
    values = [exact_family_recall(rows, method['id'], k) for method in methods]
    chosen = max(range(len(methods)), key=lambda index: values[index])
    return methods[chosen], [
        {'method': method, 'exact_family_recall': {'numerator': value.numerator,
         'denominator': value.denominator}, 'family_recall': float(value)}
        for method, value in zip(methods, values)]


def audit_reference(retriever, vocabulary, train, tasks, config):
    reference = BidirNGram(vocabulary, **config['ngram']).fit(train)
    selected = []
    for length in (5, 7):
        eligible = [task for task in tasks if task['length'] == length]
        selected.extend(sorted(eligible, key=lambda task:
            hashlib.sha256(task['task_id'].encode()).hexdigest())[:8])
    receipts = []
    for task in selected:
        query = VisibleQuery.from_task(task)
        components = retriever.score_components(query)
        spans, _, fast_order = score_ranking(components, narrow_method())
        indices = np.flatnonzero(components['narrow_mask'])
        fast_scores = np.asarray(components['ngram_scores'])[indices]
        original = np.array([reference.score(query, spans[int(i)]) for i in indices])
        original_order = indices[np.argsort(-original, kind='stable')]
        equal = bool(np.array_equal(fast_scores, original))
        rank_equal = bool(np.array_equal(fast_order, original_order))
        receipt = {'task_id': task['task_id'], 'length': task['length'],
                   'candidates': len(indices), 'scores_exact': equal,
                   'full_ranking_exact': rank_equal,
                   'top16_exact': bool(np.array_equal(fast_order[:16], original_order[:16])),
                   'max_absolute_error': float(np.max(np.abs(fast_scores - original)))}
        receipts.append(receipt)
        if not equal or not rank_equal:
            raise ValueError('original baseline numerical audit failed: ' + json.dumps(receipt))
    return {'status': 'passed', 'cases': len(receipts), 'method': config['baseline_audit'],
            'checks': receipts}


def evaluate(tasks, retriever, methods, stage):
    if not tasks or len({task['task_id'] for task in tasks}) != len(tasks):
        raise ValueError('empty or duplicated task set')
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks):
        if task['split'] != stage:
            raise ValueError('unexpected evaluation split')
        query = VisibleQuery.from_task(task)
        components = retriever.score_components(query)
        spans = tuple(components['candidates'])
        # The hidden target is inspected only after the scorer returns all components.
        target_index = {span: i for i, span in enumerate(spans)}.get(task['target'])
        row = {key: task[key] for key in ('task_id', 'poem_id', 'family_id', 'length', 'target')}
        row['query_hash'] = digest(asdict(query))
        row['methods'] = {}
        row['wide_vocabulary_size'] = len(spans)
        row['narrow_vocabulary_size'] = int(np.count_nonzero(components['narrow_mask']))
        for method in methods:
            _, scores, order = score_ranking(components, method)
            matching = np.flatnonzero(order == target_index) if target_index is not None else []
            rank = int(matching[0]) + 1 if len(matching) else None
            keep = 64 if stage == 'confirm' else 16
            top = [spans[int(i)] for i in order[:keep]]
            row['methods'][method['id']] = {
                'target_rank': rank, 'candidate_count': len(order),
                'top_candidates': top,
                'top_scores': [float(scores[int(i)]) for i in order[:keep]],
                'top16_set_hash': digest({'query': asdict(query), 'candidates': sorted(top[:16])})}
        rows.append(row)
        if (index + 1) % 200 == 0 or index + 1 == len(tasks):
            print(json.dumps({'stage': stage, 'completed': index + 1, 'total': len(tasks),
                              'seconds': round(time.perf_counter() - started, 2)}), flush=True)
    return rows, time.perf_counter() - started


def summarize(rows, methods):
    result = {}
    for method in methods:
        method_id = method['id']
        metric = {}
        for length in (None, 5, 7):
            subset = [row for row in rows if length is None or row['length'] == length]
            if not subset:
                metric['overall' if length is None else str(length)] = {'status': 'empty'}
                continue
            ranks = [row['methods'][method_id]['target_rank'] for row in subset]
            values = {'vocabulary_coverage': [int(rank is not None) for rank in ranks],
                      'mrr_full_lexicon': [1.0/rank if rank is not None else 0.0 for rank in ranks]}
            for k in (1, 8, 16, 32, 64):
                values['top1' if k == 1 else f'recall_at_{k}'] = [
                    int(rank is not None and rank <= k) for rank in ranks]
            measurements = {name: family_metric(subset, value) for name, value in values.items()}
            measurements['tasks'] = len(subset)
            measurements['poems'] = len({row['poem_id'] for row in subset})
            measurements['families'] = len({row['family_id'] for row in subset})
            measurements['ranked_candidates_per_task'] = {
                'minimum': min(row['methods'][method_id]['candidate_count'] for row in subset),
                'maximum': max(row['methods'][method_id]['candidate_count'] for row in subset),
                'total': sum(row['methods'][method_id]['candidate_count'] for row in subset)}
            metric['overall' if length is None else str(length)] = measurements
        result[method_id] = {'method': method, **metric}
    return result


def compute_accounting(rows, methods):
    return {
        'shared_component_query_evaluations': len(rows),
        'ngram_candidate_items_in_shared_wide_evaluation': sum(row['wide_vocabulary_size'] for row in rows),
        'association_components': ['alignment_bonus', 'boundary_bonus'],
        'method_rankings': len(rows) * len(methods),
        'ranked_items_by_method': {
            method['id']: sum(row['methods'][method['id']]['candidate_count'] for row in rows)
            for method in methods},
        'interpretation': 'All methods share the wide component calculation in this benchmark; narrow ranked-item counts do not claim separately measured narrow scoring work or timing.',
        'fixed_output_K_not_fixed_computation': True,
    }


def paired_interval(rows, left, right, config):
    groups = defaultdict(list)
    for row in rows:
        ranks = [row['methods'][name]['target_rank'] for name in (left, right)]
        values = [int(rank is not None and rank <= config['primary_k']) for rank in ranks]
        groups[row['family_id']].append(values[0] - values[1])
    means = np.array([np.mean(groups[key]) for key in sorted(groups)])
    if len(means) < 2:
        raise ValueError('not enough confirmation families')
    specification = config['bootstrap']
    rng = np.random.default_rng(specification['seed'])
    draws = np.array([means[rng.integers(0, len(means), len(means))].mean()
                      for _ in range(specification['repeats'])])
    return {'left': left, 'right': right, 'families': len(means),
            'mean_difference': float(means.mean()),
            'interval': np.quantile(draws, [.025, .975], method=specification['percentile_method']).tolist(),
            **specification,
            'estimand': 'paired_equal_family_all_task_recall_at_16_difference_of_frozen_deterministic_retrievers',
            'limitation': 'conditional on selected retriever, source construction and fixed output K; no literary-quality or equal-compute claim'}


def run_dev(out):
    if out.exists():
        raise ValueError('development evidence directory already exists; choose a fresh replay path')
    config = read_json(CONFIG_PATH)
    # Reading confirmation bytes for identity does not parse or score its targets.
    freeze = {'schema': 'versematrix.retrieval-freeze.v0.4', 'frozen_at_utc': now(),
              'config': config, 'config_sha256': digest(config),
              'source_sha256': identity(source_paths()), 'input_sha256': identity(input_paths()),
              'confirmation_content_parsed_by_retrieval_runner': False,
              'stage': 'frozen_before_development_ranking', 'external_preregistration': False,
              'python': platform.python_version(), 'numpy': np.__version__}
    out.mkdir(parents=True)
    write_new(out / 'freeze.json', freeze)
    data = read_train_dev()
    tasks = make_tasks(data['dev'])
    retriever, vocabulary = make_retriever(data['train'])
    resources = retriever.candidate_resources()
    write_new(out / 'candidate_identity_audit.json', audit_candidate_identity(resources, data['train']))
    write_new(out / 'candidate_resources.json', resources)
    write_new(out / 'association_resources.json', retriever.association_resources())
    write_new(out / 'training_statistics_summary.json', retriever.summary())
    audit = audit_reference(retriever, vocabulary, data['train'], tasks, config)
    write_new(out / 'baseline_equivalence_audit.json', audit)
    grid = grid_methods(config)
    methods = [narrow_method()] + grid
    rows, seconds = evaluate(tasks, retriever, methods, 'dev')
    write_rows(out / 'dev_predictions.jsonl', rows)
    selected, selection_table = select_method(rows, grid, config['primary_k'])
    report = {'status': 'completed_development_selection', 'metrics': summarize(rows, methods),
              'all_grid_entries_reported': True, 'selection_table': selection_table,
              'selected_method': selected, 'evaluation_seconds_all_17_methods_shared_components': seconds,
              'compute_accounting': compute_accounting(rows, methods),
              'confirmation_metrics_accessed': False, 'old_test_rows_decoded': False}
    write_new(out / 'dev_report.json', report)
    check_frozen(freeze)
    selection = {'frozen_at_utc': now(), 'selected_method': selected,
                 'selection_metric': config['selection_metric'], 'selection_tie_rule': config['selection_tie_rule'],
                 'freeze_sha256': sha(out / 'freeze.json'),
                 'candidate_resources_sha256': sha(out / 'candidate_resources.json'),
                 'association_resources_sha256': sha(out / 'association_resources.json'),
                 'training_statistics_summary_sha256': sha(out / 'training_statistics_summary.json'),
                 'dev_report_sha256': sha(out / 'dev_report.json'),
                 'dev_predictions_sha256': sha(out / 'dev_predictions.jsonl'),
                 'confirmation_tasks_sha256': sha(ROOT / 'data/confirmation_v04/tasks.jsonl'),
                 'confirmation_evaluated': False}
    write_new(out / 'selection_frozen.json', selection)
    print(json.dumps({'stage': 'selection_frozen', 'selected': selected,
        'dev_family_recall_at_16': report['metrics'][selected['id']]['overall']['recall_at_16']['family_macro'],
        'confirmation_evaluated': False}), flush=True)


def run_confirm(out):
    freeze = read_json(out / 'freeze.json')
    selection = read_json(out / 'selection_frozen.json')
    check_frozen(freeze)
    if sha(out / 'freeze.json') != selection['freeze_sha256']:
        raise ValueError('development freeze changed')
    for name, field in [('candidate_resources.json', 'candidate_resources_sha256'),
                        ('association_resources.json', 'association_resources_sha256'),
                        ('training_statistics_summary.json', 'training_statistics_summary_sha256'),
                        ('dev_report.json', 'dev_report_sha256'),
                        ('dev_predictions.jsonl', 'dev_predictions_sha256')]:
        if sha(out / name) != selection[field]:
            raise ValueError('development evidence changed: ' + name)
    if sha(ROOT / 'data/confirmation_v04/tasks.jsonl') != selection['confirmation_tasks_sha256']:
        raise ValueError('confirmation tasks changed')
    config = freeze['config']
    if selection['selected_method'] not in grid_methods(config):
        raise ValueError('selected method outside frozen grid')
    dev_report = read_json(out / 'dev_report.json')
    dev_rows = [json.loads(raw) for raw in (out / 'dev_predictions.jsonl').read_text().splitlines()]
    recomputed, _ = select_method(dev_rows, grid_methods(config), config['primary_k'])
    if not selection['selected_method'] == dev_report['selected_method'] == recomputed:
        raise ValueError('selected method differs from frozen development evidence')
    write_new(out / 'confirmation_started.json', {
        'started_at_utc': now(), 'selection_sha256': sha(out / 'selection_frozen.json'),
        'confirmation_evaluated_before_this_stage': False,
        'source_sha256': identity(source_paths()), 'input_sha256': identity(input_paths())})
    data = read_train_dev()
    retriever, _ = make_retriever(data['train'])
    if digest(retriever.candidate_resources()) != digest(read_json(out / 'candidate_resources.json')):
        raise ValueError('rebuilt training candidates changed')
    if digest(retriever.summary()) != digest(read_json(out / 'training_statistics_summary.json')):
        raise ValueError('rebuilt training statistics changed')
    if digest(retriever.association_resources()) != digest(read_json(out / 'association_resources.json')):
        raise ValueError('rebuilt training associations changed')
    tasks = [json.loads(raw) for raw in (ROOT / 'data/confirmation_v04/tasks.jsonl').read_text().splitlines()]
    if any(task['split'] != 'confirm' for task in tasks):
        raise ValueError('confirmation split mismatch')
    selected = {**selection['selected_method'], 'id': 'selected'}
    wide = {**grid_methods(config)[0], 'id': 'wide_ngram'}
    methods = [narrow_method(), wide, selected]
    rows, seconds = evaluate(tasks, retriever, methods, 'confirm')
    write_rows(out / 'confirmation_predictions.jsonl', rows)
    comparisons = {
        'selected_minus_narrow': paired_interval(rows, 'selected', 'narrow_ngram', config),
        'wide_minus_narrow': paired_interval(rows, 'wide_ngram', 'narrow_ngram', config),
        'selected_minus_wide': paired_interval(rows, 'selected', 'wide_ngram', config)}
    primary = comparisons['selected_minus_narrow']
    gate = config['promotion_gate']
    passed = primary['mean_difference'] >= gate['minimum_point_gain'] and primary['interval'][0] > gate['ci_lower_strictly_above']
    report = {'schema': 'versematrix.retrieval-confirmation.v0.4',
              'status': 'completed_confirmation_now_exposed', 'completed_at_utc': now(),
              'selected_method': selection['selected_method'], 'metrics': summarize(rows, methods),
              'paired_comparisons': comparisons, 'primary_comparison': 'selected_minus_narrow',
              'secondary_comparisons_descriptive_only': ['wide_minus_narrow', 'selected_minus_wide'],
              'promotion_gate': gate, 'promotion_decision': 'PASS_RETRIEVAL_GATE_ONLY' if passed else 'DO_NOT_PROMOTE_RETRIEVER',
              'evaluation_seconds_three_methods_shared_components': seconds,
              'compute_accounting': compute_accounting(rows, methods),
              'neural_training_updates': 0, 'literary_quality_evaluated': False,
              'contextual_prosody_gold': False, 'old_test_evaluated': False,
              'limits': [
                  'A frozen source-prefix retrieval target is not a preferred-wording or poetry-quality gold label.',
                  'K=16 fixes the output pool size, not retrieval scoring work; wider vocabulary is costlier.',
                  'Source corpus is shape-filtered, family-isolated by specified surface rules, not author-held-out or exhaustively edition-independent.',
                  'Local freeze is not external preregistration. Confirmation is now exposed and cannot validate future adaptive changes.',
                  'Only selected-minus-narrow recall@16 is confirmatory; other K, MRR, top1 and comparisons are descriptive.',
                  'CI lower above zero supports positive gain; it does not show the true gain is at least 2 percentage points.'
              ]}
    check_frozen(freeze)
    write_new(out / 'report.json', report)
    evidence = [path for path in sorted(out.iterdir()) if path.is_file()]
    write_new(out / 'completion_manifest.json', {'status': 'completed', 'completed_at_utc': now(),
        'source_sha256': identity(source_paths()), 'input_sha256': identity(input_paths()),
        'outputs': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in evidence}})
    print(json.dumps({'stage': 'confirmation_complete', 'decision': report['promotion_decision'],
        'family_recall_at_16': {key: value['overall']['recall_at_16']['family_macro'] for key, value in report['metrics'].items()},
        'primary_difference': primary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('dev', 'confirm'))
    parser.add_argument('--out', type=Path, default=ROOT / 'results/retrieval_v04')
    args = parser.parse_args()
    (run_dev if args.stage == 'dev' else run_confirm)(args.out.resolve())
