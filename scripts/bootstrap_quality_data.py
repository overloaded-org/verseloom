#!/usr/bin/env python3
"""Build auditable data candidates and annotation tasks; never update generation.

Public inputs are pinned; prior corpora are exclusion-only, including unselected
confirmation families. Empty annotations are tasks, NOT gold labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wordconv_lab.corpus import collect_poems, cjk_character

VERSION = 'verseloom.data-bootstrap.v1'
SALT = 'verseloom-data-development-20261030-v1'
CP = 'chinese-poetry/chinese-poetry'
CP_REV = 'b8594f81a89752241442f2ce267d6f66f96704ee'
GY = 'nk2028/tshet-uinh-data'
GY_REV = '21585e22c8a730ca2fd175112f4d18e16d5ce578'
GY_BLOB = '1b3c5c7f36504a5f187f00f8c2f8e8ee2e3b515e'
PRIOR = ['data/corpus_v03/poems.jsonl', 'data/confirmation_v04/new_unique_poems.jsonl']
REF = 'data/reference_v02/pingshui_106_memberships.json'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(obj))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as target:
        for row in rows:
            target.write(encoded(row))


def read_rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def get(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'api.github.com', 'raw.githubusercontent.com'}:
        raise ValueError('Only public GitHub HTTPS source downloads are allowed')
    if parsed.username or parsed.password:
        raise ValueError('Credentials are not source data')
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'VerseLoom-data-bootstrap/1', 'Accept': 'application/vnd.github+json'})
            with urlopen(request, timeout=30) as response:
                if urlparse(response.url).hostname != parsed.hostname:
                    raise ValueError('Unexpected source redirect')
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError('Source exceeds the per-file bound')
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError('Unreachable download state')


def raw_url(repo, revision, path):
    return f'https://raw.githubusercontent.com/{repo}/{revision}/' + quote(path)


def download(cache):
    if cache.exists():
        raise ValueError('Refusing to overwrite a source cache; use a new path')
    cache.mkdir(parents=True)
    listing_url = f'https://api.github.com/repos/{CP}/contents/' + quote('全唐诗') + '?ref=' + CP_REV
    listing = json.loads(get(listing_url))
    if not isinstance(listing, list):
        raise ValueError('Source listing is not an array')
    requests = []
    for item in listing:
        match = re.fullmatch(r'poet\.tang\.(\d+)\.json', item.get('name', ''))
        if match and int(match[1]) >= 20000 and item.get('type') == 'file':
            requests.append((CP, CP_REV, item['path'], 'raw/tang/' + item['name'], item['sha'], 'MIT'))
    requests.sort(key=lambda row: int(re.search(r'(\d+)\.json$', row[3])[1]))
    if not requests:
        raise ValueError('No unseen Tang shards found at pinned revision')
    requests += [
        (CP, CP_REV, 'LICENSE', 'licenses/chinese-poetry-MIT.txt', 'cd6a57cd38bd26ccde0a6708b43ffe1935fff03d', 'MIT'),
        (GY, GY_REV, 'LICENSE', 'licenses/tshet-uinh-CC0.txt', '0e259d42c996742e9e3cba14c677129b2c1b6311', 'CC0-1.0'),
        (GY, GY_REV, '韻書/廣韻.csv', 'raw/guangyun.csv', GY_BLOB, 'CC0-1.0'),
    ]
    def one(args):
        repo, revision, source_path, local, expected, license_id = args
        url = raw_url(repo, revision, source_path)
        content = get(url)
        if git_blob(content) != expected:
            raise ValueError('Git object identity mismatch: ' + source_path)
        target = cache / local
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return dict(repository=repo, revision=revision, source_path=source_path, local_path=local,
                    url=url, git_blob=expected, sha256=sha(content), bytes=len(content), license=license_id)
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(one, requests))
    manifest = {'schema': VERSION + '.sources', 'sources': records, 'listing_url': listing_url}
    write_json(cache / 'sources.json', manifest)
    return manifest


def verify_cache(cache):
    manifest = json.loads((cache / 'sources.json').read_text())
    if manifest.get('schema') != VERSION + '.sources':
        raise ValueError('Unknown source schema')
    seen = set()
    fixed = {'raw/guangyun.csv': GY_BLOB,
             'licenses/chinese-poetry-MIT.txt': 'cd6a57cd38bd26ccde0a6708b43ffe1935fff03d',
             'licenses/tshet-uinh-CC0.txt': '0e259d42c996742e9e3cba14c677129b2c1b6311'}
    for item in manifest['sources']:
        path = Path(item['local_path'])
        if path.is_absolute() or '..' in path.parts or str(path) in seen:
            raise ValueError('Unsafe or duplicated cache path')
        seen.add(str(path))
        if str(path) in fixed and item['git_blob'] != fixed[str(path)]:
            raise ValueError('Pinned reference or licence identity changed')
        if (item['repository'], item['revision']) not in {(CP, CP_REV), (GY, GY_REV)}:
            raise ValueError('Unapproved source revision')
        expected_url = raw_url(item['repository'], item['revision'], item['source_path'])
        if item['url'] != expected_url:
            raise ValueError('Source URL does not match pin')
        content = (cache / path).read_bytes()
        if len(content) != item['bytes'] or sha(content) != item['sha256'] or git_blob(content) != item['git_blob']:
            raise ValueError('Source cache identity mismatch: ' + str(path))
    for required in ['licenses/chinese-poetry-MIT.txt', 'licenses/tshet-uinh-CC0.txt', 'raw/guangyun.csv']:
        if required not in seen:
            raise ValueError('Missing source or licence: ' + required)
    return manifest


def normalized_lines(poem):
    # Match key only: never change the preserved body or silently merge variants.
    return tuple(unicodedata.normalize('NFKC', line) for line in poem['lines'])


def near_pairs(sets):
    """Exact >= 4/5 Jaccard join using prefix filtering, verified by set overlap.

For an n-token set the prefix length is n-ceil(4n/5)+1. All sets share a
single token order. Qualifying pairs must share a token in these prefixes.
    """
    frequency = Counter(token for tokens in sets for token in tokens)
    postings = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda j: (len(sets[j]), j)):
        tokens = sets[i]
        ordered = sorted(tokens, key=lambda g: (frequency[g], g))
        prefix = ordered[:len(tokens) - (4 * len(tokens) + 4) // 5 + 1]
        candidates = set(j for gram in prefix for j in postings[gram])
        for j in sorted(candidates):
            if 5 * len(sets[j]) < 4 * len(tokens):
                continue
            intersection = len(tokens & sets[j])
            union = len(tokens | sets[j])
            if union and 5 * intersection >= 4 * union:
                yield min(i, j), max(i, j)
        for gram in prefix:
            postings[gram].append(i)


def group_candidates(old, new):
    index = {}
    for poem in old + new:
        pid = poem['poem_id']
        if pid in index and index[pid]['lines'] != poem['lines']:
            raise ValueError('Poem identity collision')
        if pid not in index:
            index[pid] = {**poem, 'source_refs': []}
        refs = index[pid]['source_refs'] + poem.get('source_refs', [])
        unique_refs = {encoded(ref): ref for ref in refs}
        index[pid]['source_refs'] = [unique_refs[k] for k in sorted(unique_refs)]
    ids = sorted(index)
    poems = [index[pid] for pid in ids]
    prior_ids = {p['poem_id'] for p in old}
    parent = list(range(len(ids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    witnesses = []
    def link(a, b, reason):
        x, y = find(a), find(b)
        if x != y:
            parent[max(x, y)] = min(x, y)
            witnesses.append({'left': ids[a], 'right': ids[b], 'reason': reason})
    by_line, by_work, by_source = {}, {}, {}
    grams = []
    for i, p in enumerate(poems):
        lines = normalized_lines(p)
        for line in lines:
            if line in by_line:
                link(by_line[line], i, 'shared_nfkc_line')
            else:
                by_line[line] = i
        refs = p.get('source_refs', [])
        for ref in refs:
            source_id = ref.get('source_id')
            if source_id:
                if source_id in by_source:
                    link(by_source[source_id], i, 'same_upstream_id')
                else:
                    by_source[source_id] = i
            author, title = ref.get('author'), ref.get('title')
            if author and title:
                key = (unicodedata.normalize('NFKC', author), unicodedata.normalize('NFKC', title))
                if key in by_work:
                    link(by_work[key], i, 'same_author_title_conservative')
                else:
                    by_work[key] = i
        grams.append({line[k:k + 2] for line in lines for k in range(len(line) - 1)})
    for a, b in near_pairs(grams):
        link(a, b, 'nfkc_bigram_jaccard_ge_0.8')
    components = defaultdict(list)
    for i, pid in enumerate(ids):
        components[find(i)].append(pid)
    assigned, component_records = {}, []
    for members in components.values():
        family = 'qf_' + sha(encoded(members))[:24]
        previous = sorted(set(members) & prior_ids)
        bucket = int(sha((SALT + ':' + family).encode()), 16) % 100
        split = 'quarantine_prior_relation' if previous else 'train_candidate' if bucket < 80 else 'dev_reserve' if bucket < 90 else 'test_reserve'
        for pid in members:
            assigned[pid] = (family, split)
        component_records.append({'family_id': family, 'members': members, 'prior_members': previous, 'split': split})
    result = [{**p, 'family_id': assigned[p['poem_id']][0], 'split': assigned[p['poem_id']][1],
               'admission': 'unreviewed_shape_candidate', 'is_gold': False} for p in new]
    return result, sorted(component_records, key=lambda x: x['family_id']), witnesses


def select_seed(poems, per_form=100):
    if type(per_form) is not int or per_form < 1:
        raise ValueError('Positive integer seed size required')
    selected, used_families, author_counts = [], set(), Counter()
    for length in (5, 7):
        choices = sorted((p for p in poems if p['length'] == length and p['split'] == 'train_candidate'),
                         key=lambda p: (sha((SALT + ':annotation:' + p['poem_id']).encode()), p['poem_id']))
        chosen = []
        # Round-robin author cap, 1 then 2 ... 5, retains one poem per family.
        for cap in range(1, 6):
            for p in choices:
                author = p.get('author') or '<unknown>'
                if p['family_id'] in used_families or author_counts[(length, author)] >= cap:
                    continue
                chosen.append(p); used_families.add(p['family_id']); author_counts[(length, author)] += 1
                if len(chosen) == per_form:
                    break
            if len(chosen) == per_form:
                break
        if len(chosen) != per_form:
            raise ValueError(f'Only {len(chosen)} eligible {length}-character seed poems under author/family caps')
        selected.extend(chosen)
    return selected


def historical_entries(content):
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''))
    required = {'小韻號', '小韻字號', '音韻地位', '反切', '字頭', '釋義'}
    if not required <= set(reader.fieldnames or []):
        raise ValueError('Unknown Guangyun CSV schema')
    entries, by_char, counts = [], defaultdict(list), Counter()
    for record_index, row in enumerate(reader, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed Guangyun row')
        entry_id = f'gy:{record_index}'
        head = row['字頭']
        searchable = len(head) == 1 and cjk_character(head)
        entries.append({'entry_id': entry_id, 'source': 'guangyun', 'revision': GY_REV,
                        'csv_record_1based': record_index, 'csv_line_end': reader.line_num,
                        'fields': row, 'searchable_exact_headword': searchable,
                        'status': 'source_record_not_context_resolution'})
        if searchable:
            by_char[head].append(entry_id)
        counts['rows'] += 1
        counts['searchable_rows' if searchable else 'editorial_or_empty_headword_rows'] += 1
    counts['unique_exact_headwords'] = len(by_char)
    return entries, by_char, dict(counts)


def annotation_tasks(seed, by_char, reference):
    tasks = []
    for p in seed:
        for rows in ([2, 3], [4, 5]):
            task = {'schema': VERSION + '.annotation-task', 'task_id': p['poem_id'] + ':rows' + '-'.join(map(str, rows)),
                    'poem_id': p['poem_id'], 'family_id': p['family_id'], 'length': p['length'],
                    'usage': 'development_annotation_only', 'is_gold': False,
                    'author': p.get('author'), 'title': p.get('title'), 'source_refs': p['source_refs'],
                    'context_lines': p['lines'], 'row_indices_zero_based': rows,
                    'lines': [p['lines'][i] for i in rows], 'reading_candidates': [],
                    'annotation': {'status': 'pending', 'word_boundaries': None, 'senses': None,
                                   'syntax_roles': None, 'parallel_alignment': None, 'strict_antithesis': 'unknown',
                                   'contextual_readings': None, 'reviewer': None, 'evidence': []}}
            for i in rows:
                for col, char in enumerate(p['lines'][i]):
                    task['reading_candidates'].append({'row': i, 'column': col, 'character': char,
                        'guangyun_entry_ids': list(by_char.get(char, [])),
                        'pingshui_memberships': reference['characters'].get(char, []),
                        'status': 'unresolved', 'selected_entry_id': None,
                        'guangyun_to_pingshui_mapping': None})
            tasks.append(task)
    return tasks


def validate_annotation(task):
    annotation = task['annotation']
    if task.get('is_gold') is not False or annotation['status'] not in {'pending', 'assistant_draft', 'reviewed'}:
        raise ValueError('Bootstrap annotations cannot be promoted to independent gold')
    if task.get('usage') != 'development_annotation_only':
        raise ValueError('Development annotations cannot become final evaluation')
    for key in ('word_boundaries', 'senses', 'syntax_roles', 'parallel_alignment', 'contextual_readings'):
        if annotation['status'] == 'pending' and annotation[key] is not None:
            raise ValueError('Pending task contains a claimed annotation')
    if annotation['status'] == 'reviewed' and (not annotation.get('reviewer') or not annotation.get('evidence')):
        raise ValueError('Reviewed status requires reviewer identity and evidence')
    for candidate in task['reading_candidates']:
        if candidate['selected_entry_id'] is not None:
            raise ValueError('Automatic import must never select contextual pronunciation')


def build(root, cache, out, per_form=100):
    if out.exists():
        raise ValueError('Output directory already exists; refusing overwrite')
    manifest = verify_cache(cache)
    before = {name: sha((root / name).read_bytes()) for name in PRIOR + [REF]}
    old = [p for name in PRIOR for p in read_rows(root / name)]
    shards = [(item['local_path'], json.loads((cache / item['local_path']).read_bytes()))
              for item in manifest['sources'] if item['local_path'].startswith('raw/tang/')]
    new, duplicates, audit = collect_poems(shards)
    print(json.dumps({'phase': 'shape_screened', 'unique': len(new)}, ensure_ascii=False), flush=True)
    assigned, components, witnesses = group_candidates(old, new)
    seeds = select_seed(assigned, per_form)
    entries, by_char, gy_counts = historical_entries((cache / 'raw/guangyun.csv').read_bytes())
    reference = json.loads((root / REF).read_bytes())
    tasks = annotation_tasks(seeds, by_char, reference)
    for task in tasks:
        validate_annotation(task)
    out.mkdir(parents=True)
    (out / 'INCOMPLETE').write_text('Only a successful final manifest completes this run.\n')
    write_rows(out / 'candidate_poems.jsonl', assigned)
    write_rows(out / 'exact_duplicate_groups.jsonl', duplicates)
    write_rows(out / 'family_components.jsonl', components)
    write_rows(out / 'family_witnesses.jsonl', witnesses)
    write_rows(out / 'annotation_seed_200.jsonl', seeds)
    write_rows(out / 'annotation_tasks_400.jsonl', tasks)
    write_rows(out / 'guangyun_entries.jsonl', entries)
    write_json(out / 'guangyun_index.json', dict(by_char))
    write_json(out / 'sources.json', manifest)
    split_counts = {split: dict(Counter(str(p['length']) for p in assigned if p['split'] == split))
                    for split in sorted({p['split'] for p in assigned})}
    positions = [r for task in tasks for r in task['reading_candidates']]
    report = {'schema': VERSION + '.report', 'source_shards': len(shards), 'source_audit': audit,
              'prior_unique_poems': len({p['poem_id'] for p in old}), 'by_split_and_length': split_counts,
              'family_components': len(components), 'union_witnesses': len(witnesses),
              'seed_poems': len(seeds), 'seed_by_length': dict(Counter(str(p['length']) for p in seeds)),
              'seed_authors_by_length': {str(n): len({p['author'] for p in seeds if p['length'] == n}) for n in (5, 7)},
              'annotation_tasks': len(tasks), 'completed_annotation_tasks': 0, 'independent_gold_labels': 0,
              'guangyun': gy_counts,
              'reading_positions': len(positions), 'positions_with_guangyun_candidates': sum(bool(r['guangyun_entry_ids']) for r in positions),
              'positions_with_multiple_guangyun_candidates': sum(len(r['guangyun_entry_ids']) > 1 for r in positions),
              'contextually_resolved_positions': 0,
              'prior_files': before, 'production_changed': False,
              'split_policy': 'family SHA256 80/10/10; all prior-related families quarantined; new reserve is not a claimed blind benchmark',
              'limitations': ['Eight equal-length lines are not a strict lushi label.',
                  'NFKC is a match key only; simplified/traditional and unrecorded philological variants may remain undetected.',
                  'All old 0..19000 shape poems are exclusion-only, including unselected confirmation rows.',
                  'Author/title matching is deliberately conservative and may over-quarantine distinct poems.',
                  'No model fitted, no new vocabulary admitted, no generator changed.',
                  'Guangyun entries are not Pingshui mappings or contextual reading decisions.',
                  '200 selected poems and 400 annotation tasks are unlabelled development material, not completed gold.']}
    if before != {name: sha((root / name).read_bytes()) for name in before}:
        raise ValueError('A protected input changed')
    write_json(out / 'report.json', report)
    (out / 'PINGSHUI_NOTICE.md').write_bytes(
        (root / 'data/reference_v02/LICENSE_NOTICE.md').read_bytes())
    write_json(out / 'manifest.json', {'schema': VERSION + '.manifest', 'files': {
        p.name: {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'INCOMPLETE'}})
    (out / 'INCOMPLETE').unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download(args.cache)
    report = build(ROOT, args.cache, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
