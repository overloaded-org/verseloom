#!/usr/bin/env python3
"""Audit real outputs and build a blinded review pack; never rewrite poetry."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.quality import QualityReview, sha256
from wordconv_lab.prosody import all_templates


def mechanical(record, reference):
    lines, length = record['lines'], record['lineLength']
    if len(lines) != 8 or length not in (5, 7) or any(len(x) != length for x in lines):
        raise ValueError('Invalid eight-line shape')
    from collections import Counter
    if any(len(set(x)) != length for x in lines) or max(Counter(''.join(lines)).values()) > 2:
        raise ValueError('Repeated characters violate current generator contract')
    spec = next((t for t in all_templates(length) if t.template_id == record['options']['templateId']), None)
    if spec is None or list(spec.core) != record['core'] or list(spec.rhyme_rows) != record['rhymeRows']:
        raise ValueError('Recorded template differs from independent prosody specification')
    endings = []
    for i, text in enumerate(lines):
        pattern = ''
        for char in text:
            tones = {x['tone'] for x in reference.get(char, [])}
            if len(tones) != 1:
                raise ValueError('Reference tone unresolved')
            pattern += next(iter(tones))
        if pattern != record['core'][i]:
            raise ValueError('Tone mismatch')
        if i in record['rhymeRows']:
            endings.append(text[-1])
            groups = {x['group_id'] for x in reference[text[-1]] if x['tone'] == 'P'}
            if groups != {record['rhymeGroup']}:
                raise ValueError('Rhyme mismatch')
        if i in (3, 5) and any(a == b for a, b in zip(text, lines[i-1])):
            raise ValueError('Middle-position repeated character')
    if len(set(endings)) != len(endings):
        raise ValueError('Duplicate rhyme ending')


def run(samples: Path, output: Path):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    review = QualityReview(ROOT)
    tasks = defaultdict(list)
    sources = None
    with samples.open(encoding='utf-8') as handle:
        for raw in handle:
            row = json.loads(raw)
            mechanical(row, review.reference)
            if any(line in review.full_lines[row['lineLength']] for line in row['lines']):
                raise ValueError('Full training line copied')
            if row.get('preselectedLines') is not False or not math.isfinite(row['score']):
                raise ValueError('Invalid source or score')
            if sources is None:
                sources = row['sources']
            if row['sources'] != sources:
                raise ValueError('Mixed engine/data snapshots')
            tasks[row['taskId']].append(row)
    expected = {f'L{length}-{i}' for length in (5, 7) for i in range(32)}
    if set(tasks) != expected:
        raise ValueError('Expected the complete fixed 64-task protocol, not a partial batch')
    for name, expected in sources.items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Sample source differs from runtime: ' + name)
    comparisons, blind, keys = [], [], []
    output.mkdir(parents=True)
    with (output / 'audit.jsonl').open('w', encoding='utf-8') as audits:
        for task, records in sorted(tasks.items()):
            records.sort(key=lambda r: r['candidate'])
            if [r['candidate'] for r in records] != list(range(4)):
                raise ValueError('Each task needs exactly four distinct candidate IDs')
            lengths = {r['lineLength'] for r in records}
            if len(lengths) != 1 or len({r['options']['seed'] for r in records}) != 4:
                raise ValueError('Mixed form or duplicate sample seed')
            scored = [(r, review.rank_score(r)) for r in records]
            chosen, (score, details) = max(scored, key=lambda x: (x[1][0], -x[0]['candidate']))
            base, (_, base_details) = scored[0]
            raw_best, (_, raw_details) = max(scored, key=lambda x: (x[0]['score'], -x[0]['candidate']))
            row = {'taskId':task, 'lineLength':base['lineLength'], 'selectedCandidate':chosen['candidate'],
                   'baseline':{'lines':base['lines'], 'score':base['score'], **base_details['audit']['metrics']},
                   'rawBestOfFour':{'candidate':raw_best['candidate'],'lines':raw_best['lines'],**raw_details['audit']['metrics']},
                   'reviewRanked':{'lines':chosen['lines'], 'score':chosen['score'], **details['audit']['metrics']},
                   'changed':base['lines'] != chosen['lines']}
            comparisons.append(row)
            for record, (_, audit) in scored:
                audits.write(json.dumps({'taskId':task, 'candidate':record['candidate'], **audit},ensure_ascii=False)+'\n')
            order = int(hashlib.sha256(('blind-v1/'+task).encode()).hexdigest()[:2],16)%2
            pair = [raw_best['lines'],chosen['lines']]
            if order: pair.reverse()
            blind.append({'id':task, 'lineLength':base['lineLength'], 'A':pair[0], 'B':pair[1],
                          'ratings':{'fluency':None,'coherence':None,'imagery':None,'antithesis':None},
                          'preference':None,'reason':None,'reviewer':None})
            keys.append({'id':task,'rawBestOfFour':'B' if order else 'A','reviewRanked':'A' if order else 'B',
                         'identical':raw_best['lines']==chosen['lines']})
    summary = {}
    metric_names = ['unsupportedBridges','linesWithSyntaxHypothesis','unambiguousSyntaxLines',
                    'structurallyPossiblePairs','unambiguousAlignedPairs','multiMembershipPositions']
    for length in (5,7):
        group = [r for r in comparisons if r['lineLength']==length]
        if not group: continue
        summary[str(length)] = {'tasks':len(group),'changed':sum(r['changed'] for r in group),
            **{method:{m:statistics.mean(r[method][m] for r in group) for m in metric_names}
               for method in ('baseline','rawBestOfFour','reviewRanked')}}
    report = {'schema':'verseloom.quality-experiment.v1','status':'PROXY_ONLY_NOT_PROMOTED',
              'sampleSha256':sha256(samples),'sources':{**sources,**review.sources,'scripts/run_quality_experiment.py':sha256(Path(__file__))},
              'tasks':len(tasks),'candidateCount':sum(map(len,tasks.values())),
              'independentMechanicalChecks':'passed',
              'protocol':{'taskSeeds':'1000..1031','candidatesPerTask':4,'templates':['L5-Z-R0','L7-P-R0'],
                          'candidateSeedOffsets':'uint32(candidate * 0x9e3779b9)',
                          'qualityWeights':{'unsupportedBridge':0.4,'unparsedLine':0.3,'unmatchedCouplet':0.5,'ambiguousMembership':0.2},
                          'parameterTuningOnThisBatch':False,'independentHumanReviews':0,
                          'blindedPackets':len(blind),'sameComputeBaseline':'rawBestOfFour',
                          'blindedComparison':'rawBestOfFour_vs_reviewRanked',
                          'note':'All proxy metrics overlap the reranking objective. Improvement on them is not independent evidence of literary quality.'},
              'summary':summary,'comparisons':comparisons,
              'claims':{'literaryQualityImproved':False,'strictAntithesisCertified':False,
                        'contextualPronunciationVerified':False,'websiteGeneratorChanged':False}}
    for name, content in [('report.json',report),('blind_review.json',blind),('blind_key.json',keys)]:
        (output/name).write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'tasks':report['tasks'],'candidates':report['candidateCount'],'summary':summary},ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.samples,args.output)
