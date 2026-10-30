#!/usr/bin/env python3
"""SP-02 train-only couplet-specific ranker, evaluated on dev source adjacency.

This measures retrieval of attested adjacent lines, not strict antithesis.
No authored content, fixed lines, themes, or templates are introduced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from train_continuation_ranker import (
    Features, PARAMETERS, bootstrap_difference, encode, fit, read_partition,
    require, select_train, summarize, tasks, sha,
)


def select_rows(matrix, records, positions):
    ids = [i for i, r in enumerate(records) if r['row'] in positions]
    return matrix[ids], [records[i] for i in ids]


def evaluate(model_path, corpus_path, out):
    require(not out.exists(), 'Refusing overwrite')
    model_bytes = model_path.read_bytes()
    data = json.loads(model_bytes.decode().split('globalThis.VerseLoomInducedData=', 1)[1].rsplit(';', 1)[0])
    require(sha(corpus_path.read_bytes()) == data['sourceIdentity']['newCandidateCorpus'], 'Corpus identity mismatch')
    train_all, _ = read_partition(corpus_path, {'train_candidate'})
    dev, _ = read_partition(corpus_path, {'dev_reserve'})
    train = select_train(train_all)
    require(not ({p['family_id'] for p in train_all} & {p['family_id'] for p in dev}), 'Family overlap')
    features = Features(data)
    train_matrix, train_records, _ = tasks(train, features, PARAMETERS['train_negatives'])
    dev_matrix, dev_records, _ = tasks(dev, features, PARAMETERS['dev_negatives'])
    # All-position learner is the control; middle couplet learner sees only
    # attested 3rd->4th and 5th->6th transitions (zero-based rows 3 and 5).
    general, _, _ = fit(train_matrix)
    middle_train, middle_train_records = select_rows(train_matrix, train_records, {3, 5})
    middle_dev, middle_dev_records = select_rows(dev_matrix, dev_records, {3, 5})
    middle, _, _ = fit(middle_train)
    # No parameter selection on dev, and no promotion of dev to training.
    general_scores = middle_dev @ general
    middle_scores = middle_dev @ middle
    fixed_scores = middle_dev[:, :, [0, 1, 2]] @ np.array([.35, .25, .4])
    from train_continuation_ranker import rank_scores
    general_r1, _ = rank_scores(general_scores)
    middle_r1, _ = rank_scores(middle_scores)
    fixed_r1, _ = rank_scores(fixed_scores)
    # Diagnose whether a middle-specific learner genuinely uses a different\n    # feature weighting, rather than only renaming the general model.\n    coefficient_delta = [float(x-y) for x,y in zip(middle,general)]\n    report = {
        'schema': 'verseloom.sp02.middle-couplet-retrieval.v1',
        'sourceIdentity': data['sourceIdentity'],
        'modelSha256': sha(model_bytes),
        'corpusSha256': sha(corpus_path.read_bytes()),
        'training': {'trainFamilies': len({r['family'] for r in middle_train_records}),
                     'middleContexts': len(middle_train_records),
                     'negativesPerContext': PARAMETERS['train_negatives']},
        'evaluation': {'devFamilies': len({r['family'] for r in middle_dev_records}),
                       'middleContexts': len(middle_dev_records),
                       'candidatesPerContext': PARAMETERS['dev_negatives'] + 1,
                       'rowsZeroBased': [3, 5]},
        'fixed': summarize(fixed_scores, middle_dev_records)[0],
        'generalLearned': summarize(general_scores, middle_dev_records)[0],
        'coupletLearned': summarize(middle_scores, middle_dev_records)[0],
        'coupletMinusGeneral': bootstrap_difference(middle_r1, general_r1, middle_dev_records),
        'generalMinusFixed': bootstrap_difference(general_r1, fixed_r1, middle_dev_records),
        'coefficientDifference': coefficient_delta,\n        'coefficientL2Difference': float(np.linalg.norm(middle-general)),\n        'coefficients': {'general': [float(x) for x in general],
                         'middle': [float(x) for x in middle]},
        'independentLiteraryRatings': 0,
        'strictAntithesisCertified': False,
        'literaryQualityCertified': False,
        'deployed': False,
        'limitations': [
            'Source adjacency is not grammatical or semantic antithesis.',
            'Only observed source pairs versus cross-poem distractors are tested.',
            'Development corpus is not independent final literary evaluation.',
            'The source graph is corpus-induced; this is not a causal event model.',
            'No learned couplet weights are used in the live generator.'
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('web/induced-data.js'))
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    evaluate(a.model, a.corpus, a.output)
