#!/usr/bin/env python3
"""Build seven-character, train-only 4+3 candidates; never supply a required line.

The v0.6 five-character data and source hashes are deliberately left unchanged.
Seven-character spans use min_count=1 because this frozen split has 299 poems.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.free_generation import SPLIT, Coherence, _top_with_ending_coverage
from wordconv_lab.prosody import all_templates
from wordconv_lab.restoration import BidirNGram
from wordconv_lab.retrieval import _FastBidir


def load_training(path: Path) -> tuple[list[dict], dict]:
    poems, counts = [], Counter()
    with path.open('rb') as source:
        for raw in source:
            if not raw.strip():
                continue
            marker = SPLIT.search(raw)
            if marker is None:
                raise ValueError('Missing train/dev/test marker')
            split = marker.group(1).decode('ascii')
            counts['source_' + split] += 1
            if split != 'train':
                continue  # Never JSON-decode a held-out row.
            item = json.loads(raw)
            if item.get('split') != 'train':
                raise ValueError('Inconsistent split marker')
            if item.get('length') != 7:
                continue
            if len(item.get('lines', [])) != 8 or any(not isinstance(x, str) or len(x) != 7 for x in item['lines']):
                raise ValueError('Seven-character training record has invalid shape')
            poems.append(item)
    if not poems or len({p['poem_id'] for p in poems}) != len(poems):
        raise ValueError('Empty or duplicated training corpus')
    counts['used_seven_training_poems'] = len(poems)
    return sorted(poems, key=lambda p: p['poem_id']), dict(counts)


def collect(poems: list[dict], reference: dict, editorial: dict) -> tuple[dict, set[str]]:
    excluded = {x['character'] for x in editorial['issues'] if x.get('character') is not None}
    raw = {'prefixes': defaultdict(list), 'suffixes': defaultdict(list)}
    train_lines = set()
    for poem in poems:
        if poem.get('split') != 'train' or poem.get('length') != 7:
            raise ValueError('Only seven-character training poems may supply spans')
        for row, line in enumerate(poem['lines']):
            train_lines.add(line)
            for kind, start, stop in [('prefixes', 0, 4), ('suffixes', 4, 7)]:
                raw[kind][line[start:stop]].append([poem['poem_id'], row, start, stop])
    spans = {'prefixes': {}, 'suffixes': {}}
    for kind, records in raw.items():
        for text, sources in sorted(records.items()):
            members = [reference['characters'].get(char, []) for char in text]
            tones = [{m['tone'] for m in entries} for entries in members]
            if any(c in excluded for c in text) or any(len(t) != 1 or not t <= {'P', 'Z'} for t in tones):
                continue
            pattern = ''.join(next(iter(t)) for t in tones)
            groups = {m['group_id'] for m in members[-1] if m['tone'] == 'P'}
            spans[kind][text] = {'count': len(sources), 'sources': sources, 'tones': pattern,
                                  'group': next(iter(groups)) if pattern[-1] == 'P' and len(groups) == 1 else None}
    return spans, train_lines


def scores(model: BidirNGram, texts: list[str], fast: _FastBidir) -> np.ndarray:
    if any(len(t) != 7 for t in texts):
        raise ValueError('Expected seven-character candidates')
    ids = np.asarray([model.encode(t) for t in texts], dtype=np.int64)
    totals = []
    for direction in (0, 1):
        seq = ids if direction == 0 else ids[:, ::-1]
        a = np.full(len(texts), -1, dtype=np.int64)
        b = a.copy()
        total = np.zeros(len(texts), dtype=np.float64)
        for col in range(8):
            target = seq[:, col] if col < 7 else np.full(len(texts), model.eos)
            total += fast.log_probability(direction, a, b, target)
            a, b = b, target
        totals.append(total / 8)
    result = 0.5 * (totals[0] + totals[1])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite candidate score')
    return result


def export(root: Path) -> dict:
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    reference = read('data/reference_v02/pingshui_106_memberships.json')
    poems, loading = load_training(root / 'data/corpus_v03/poems.jsonl')
    spans, train_lines = collect(poems, reference, read('data/reference_v02/editorial_issues.json'))
    tokens = ['<UNK>'] + sorted(set(''.join(line for p in poems for line in p['lines'])))
    model = BidirNGram({c: i for i, c in enumerate(tokens)}).fit(poems)
    fast = _FastBidir(model, {})
    templates = sorted(all_templates(7), key=lambda t: t.template_id)
    patterns = sorted({pattern for t in templates for pattern in t.core})
    pools, counts = {}, Counter()
    for pattern in patterns:
        prefixes = [text for text, entry in spans['prefixes'].items() if entry['tones'] == pattern[:4]]
        suffixes = [text for text, entry in spans['suffixes'].items() if entry['tones'] == pattern[4:]]
        by_ending, pending = defaultdict(list), []
        def flush():
            if not pending:
                return
            values = scores(model, [x['text'] for x in pending], fast)
            for entry, value in zip(pending, values):
                entry['lm_score'] = float(value)
                by_ending[(entry['group'], entry['text'][-1] if entry['group'] != 'nonrhyming' else 'all')].append(entry)
            counts['scoredCandidates'] += len(pending)
            pending.clear()
            # Bounded memory while scoring ALL eligible Cartesian combinations.
            for key, entries in by_ending.items():
                cap = 128 if key[0] == 'nonrhyming' else 12
                entries.sort(key=lambda x: (-x['lm_score'], x['text']))
                del entries[cap:]
        for prefix in prefixes:
            for suffix in suffixes:
                text = prefix + suffix
                counts['toneCompatiblePairs'] += 1
                if len(set(text)) != 7 or text in train_lines:
                    counts['rejectedRepetitionOrTrainingLine'] += 1
                    continue
                group = spans['suffixes'][suffix]['group'] if pattern[-1] == 'P' else 'nonrhyming'
                if group is None:
                    counts['rejectedRhymeAmbiguity'] += 1
                    continue
                pending.append({'text': text, 'group': group})
                if len(pending) >= 8192:
                    flush()
        flush()
        groups = defaultdict(list)
        for (group, _ending), entries in sorted(by_ending.items()):
            groups[group].extend(entries)
        for group, entries in sorted(groups.items()):
            kept = _top_with_ending_coverage(entries, 128) if group != 'nonrhyming' else entries
            pools[pattern + '/' + group] = [[e['text'], e['lm_score'], spans['prefixes'][e['text'][:4]]['count'],
                                            spans['suffixes'][e['text'][4:]]['count']] for e in kept]
    labels = {e['group_id']: e['label'] for members in reference['characters'].values() for e in members}
    rhymed_patterns = [p for p in patterns if p[-1] == 'P']
    rhymes = []
    for group in sorted({key.split('/')[1] for key in pools if not key.endswith('/nonrhyming')}):
        ends = [{row[0][-1] for row in pools.get(p + '/' + group, [])} for p in rhymed_patterns]
        union = set.union(*ends)
        feasible = min(map(len, ends)) >= 3 and len(union) >= 5 and all(p + '/nonrhyming' in pools for p in patterns if p[-1] == 'Z')
        frequency = sum(x['count'] for x in spans['suffixes'].values() if x['group'] == group)
        rhymes.append({'id': group, 'label': labels[group], 'feasible': feasible, 'frequency': frequency,
                       'endingCount': len(union), 'minimumEndingCount': min(map(len, ends))})
    rhymes.sort(key=lambda r: (-int(r['feasible']), -r['minimumEndingCount'], -r['endingCount'], -r['frequency'], r['id']))
    if not any(r['feasible'] for r in rhymes):
        raise ValueError('No seven-character rhyme supports all four templates')
    chars = sorted({c for rows in pools.values() for row in rows for c in row[0]})
    coherence, values = Coherence(poems), bytearray()
    for j, right in enumerate(chars):
        for left in chars[:j]:
            denominator = coherence.df[left] * coherence.df[right]
            value = math.log1p(coherence.n * coherence.joint[(left, right)] / denominator) if denominator else 0.0
            values.extend(struct.pack('<d', value))
    used_prefixes = {row[0][:4] for rows in pools.values() for row in rows}
    used_suffixes = {row[0][4:] for rows in pools.values() for row in rows}
    names = ['data/corpus_v03/poems.jsonl', 'data/reference_v02/pingshui_106_memberships.json',
             'data/reference_v02/editorial_issues.json', 'src/wordconv_lab/prosody.py',
             'src/wordconv_lab/restoration.py', 'src/wordconv_lab/retrieval.py',
             'src/wordconv_lab/free_generation.py', 'scripts/build_web_seven.py']
    return {'schema': 'verseloom.web.v1', 'meta': {
        'lineLength': 7, 'prefixLength': 4, 'minimumSpanCount': 1,
        'trainingPoems': len(poems), 'candidateCount': sum(map(len, pools.values())),
        'feasibleRhymes': sum(r['feasible'] for r in rhymes), 'templateCount': 4, 'characterCount': len(chars),
        'method': 'Train-only seven-character 4+3 span recombination + live unanchored constrained search',
        'candidateScoresPrecomputed': True, 'neuralModelUsed': False, 'loading': loading, 'poolBuild': dict(counts),
        'sources': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
        'referenceSource': reference['source']['source_url'], 'referenceLicense': 'CC BY-SA 4.0; see NOTICE.html',
        'contextualPronunciationVerified': False, 'antithesisCertified': False, 'literaryQualityCertified': False,
        'globalOptimumClaim': False, 'sourcePoolNoveltyScope': 'Excludes complete lines in the seven-character training split only'},
        'defaults': {'rhymeGroup': 'auto', 'templateId': 'auto', 'beamWidth': 32, 'coherenceWeight': 0.5, 'repetitionWeight': 0.8},
        'templates': [{'id': t.template_id, 'label': ('平起' if t.start_second_tone == 'P' else '仄起') + ('首句入韵' if t.first_rhymed else '首句不入韵'), 'core': list(t.core), 'rhymeRows': list(t.rhyme_rows)} for t in templates],
        'rhymes': rhymes, 'pools': pools,
        'spans': {'prefixes': {s: spans['prefixes'][s] for s in sorted(used_prefixes)},
                  'suffixes': {s: spans['suffixes'][s] for s in sorted(used_suffixes)}},
        'coherence': {'characters': ''.join(chars), 'encoding': 'float64-le-lower-triangle-base64', 'values': base64.b64encode(values).decode('ascii')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'web/data7.js')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = export(ROOT)
    content = ('/* Seven-character train-only data. See NOTICE.html. */\nglobalThis.VerseLoomSevenData=' + json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';\n').encode('utf-8')
    if args.check:
        if args.output.read_bytes() != content:
            raise SystemExit('Seven-character export differs from its sources')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(content)
    print(json.dumps({'status': 'matched' if args.check else 'exported', 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'trainingPoems': data['meta']['trainingPoems'], 'candidates': data['meta']['candidateCount'], 'rhymes': data['meta']['feasibleRhymes'], 'poolBuild': data['meta']['poolBuild']}))

if __name__ == '__main__':
    main()
