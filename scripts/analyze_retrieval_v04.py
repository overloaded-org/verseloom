"""Post-confirmation descriptive decomposition of frozen retrieval ranks.

No retraining, candidate change, rescoring or model selection is performed.
All results are paired task counts, not the confirmatory family-macro metric.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/retrieval_v04_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def summarize(rows):
    names = ('narrow_ngram', 'wide_ngram', 'selected')
    result = {'tasks': len(rows), 'poems': len({r['poem_id'] for r in rows}),
              'families': len({r['family_id'] for r in rows})}
    for name in names:
        ranks = [r['methods'][name]['target_rank'] for r in rows]
        covered = sum(rank is not None for rank in ranks)
        hits = sum(rank is not None and rank <= 16 for rank in ranks)
        result[name] = {'vocabulary_covered': covered, 'vocabulary_absent': len(rows)-covered,
                        'hits_at_16': hits, 'covered_but_below_top16': covered-hits,
                        'task_micro_recall_at_16': hits/len(rows),
                        'task_micro_vocabulary_coverage': covered/len(rows),
                        'recall_at_16_conditional_on_vocabulary': hits/covered if covered else None}
    comparisons = {}
    for new, old in [('wide_ngram', 'narrow_ngram'), ('selected', 'narrow_ngram'), ('selected', 'wide_ngram')]:
        counts = defaultdict(int)
        for row in rows:
            nr = row['methods'][new]['target_rank']; old_rank = row['methods'][old]['target_rank']
            nh = nr is not None and nr <= 16; oh = old_rank is not None and old_rank <= 16
            counts['retained_hit' if nh and oh else 'gain' if nh else 'loss' if oh else 'both_miss'] += 1
            if nh and not oh:
                counts['gain_new_vocabulary' if old_rank is None else 'gain_old_vocabulary_reranked'] += 1
            if old_rank is None and nr is not None:
                counts['new_vocabulary_covered'] += 1
        counts = {name: counts[name] for name in ('retained_hit', 'gain', 'loss', 'both_miss',
                    'gain_new_vocabulary', 'gain_old_vocabulary_reranked', 'new_vocabulary_covered')}
        assert counts['retained_hit']+counts['gain']+counts['loss']+counts['both_miss'] == len(rows)
        assert counts['gain_new_vocabulary']+counts['gain_old_vocabulary_reranked'] == counts['gain']
        assert counts['gain']-counts['loss'] == result[new]['hits_at_16']-result[old]['hits_at_16']
        comparisons[new+'_minus_'+old] = counts
    result['paired_task_transitions'] = comparisons
    return result


def run(out, check=False):
    inputs = [ROOT/'results/retrieval_v04'/name for name in ('report.json', 'confirmation_predictions.jsonl')]
    identities = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    frozen = {'stage': 'post_confirmation_descriptive_only', 'input_sha256': identities,
              'script_sha256': sha(Path(__file__)), 'new_scoring': False, 'new_selection': False,
              'primary_result_or_gate_changed': False, 'all_transition_counts_are_task_weighted': True}
    if check:
        previous = json.loads((out/'freeze.json').read_text())
        if any(previous[key] != value for key, value in frozen.items()):
            raise ValueError('diagnostic source/code/config changed')
    else:
        if out.exists():
            raise ValueError('refusing to replace prior diagnostic evidence')
        out.mkdir(parents=True)
        frozen['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        (out/'freeze.json').write_bytes(encoded(frozen))
    rows = [json.loads(line) for line in inputs[1].read_text().splitlines()]
    # With identical scores/ties and a superset, a surviving original candidate
    # cannot improve its rank merely because extra competitors were inserted.
    compared = 0
    for row in rows:
        old = row['methods']['narrow_ngram']['target_rank']
        wide = row['methods']['wide_ngram']['target_rank']
        if old is not None:
            compared += 1
            if wide is None or wide < old:
                raise ValueError('superset ranking monotonicity violated')
    report = {'status': 'completed_descriptive_diagnostic', 'overall': summarize(rows),
              'by_length': {str(length): summarize([row for row in rows if row['length'] == length]) for length in (5,7)},
              'same_score_superset_rank_monotonicity': {'status': 'passed', 'tasks_checked': compared},
              'limits': ['Post-confirmation explanation; not a new confirmatory test or a changed promotion gate.',
                         'Transitions are raw task counts; they are not equal-family weighted gains.',
                         'Only observed source-prefix retrieval is described; literary quality remains unevaluated.']}
    data = encoded(report)
    manifest = encoded({'freeze_sha256': sha(out/'freeze.json'),
                        'report_sha256': hashlib.sha256(data).hexdigest(), 'report_bytes': len(data)})
    for name, payload in [('report.json', data), ('manifest.json', manifest)]:
        if check:
            if (out/name).read_bytes() != payload:
                raise ValueError('diagnostic replay differs: '+name)
        else:
            (out/name).write_bytes(payload)
    print(json.dumps({'status': report['status'], 'overall': report['overall']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    run(args.out, args.check)
