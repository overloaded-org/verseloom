"""Evidence-aware experimental review, not a literary-quality certifier.

Never writes/replaces poem characters. Missing lexical, semantic or historical
reading evidence remains unknown. Statistics use train lines only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'verseloom.quality-review.v1'
SPLIT = re.compile(rb'"split"\s*:\s*"(train|dev|test)"')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QualityReview:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.reference = json.loads((self.root / 'data/reference_v02/pingshui_106_memberships.json').read_text())['characters']
        self.lexicon = json.loads((self.root / 'data/quality_v1/lexicon.json').read_text())
        self.hints = json.loads((self.root / 'data/quality_v1/reading_hints.json').read_text())['entries']
        self.lex = defaultdict(set)
        for mode in ('characters', 'words'):
            for tag, text in self.lexicon[mode].items():
                for word in (list(text) if mode == 'characters' else text.split()):
                    self.lex[word].add(tuple(tag.split(':', 1)))
        self.grams = {5: Counter(), 7: Counter()}
        self.training = {5: [], 7: []}
        self.full_lines = {5: set(), 7: set()}
        self.poems = {}
        with (self.root / 'data/corpus_v03/poems.jsonl').open('rb') as source:
            for raw in source:
                if not raw.strip():
                    continue
                match = SPLIT.search(raw)
                if not match:
                    raise ValueError('Missing split marker')
                if match[1] != b'train':
                    continue  # No held-out row is JSON-decoded or counted.
                poem = json.loads(raw)
                length = poem.get('length')
                if poem.get('split') != 'train' or length not in (5, 7):
                    raise ValueError('Unsupported training record')
                if len(poem['lines']) != 8 or any(len(x) != length for x in poem['lines']):
                    raise ValueError('Invalid training shape')
                self.training[length].append(poem)
                self.poems[poem['poem_id']] = poem
                for line in poem['lines']:
                    self.full_lines[length].add(line)
                    for width in (2, 3):
                        self.grams[length].update(line[i:i + width] for i in range(length - width + 1))
        paths = ['data/corpus_v03/poems.jsonl', 'data/reference_v02/pingshui_106_memberships.json',
                 'data/quality_v1/lexicon.json', 'data/quality_v1/reading_hints.json',
                 'src/wordconv_lab/quality.py']
        self.sources = {name: sha256(self.root / name) for name in paths}

    def tokens(self, text: str, role: str) -> list[dict]:
        """Enumerate lexical hypotheses without using required meter or a partner."""
        answer = []
        for pos, domain in sorted(self.lex.get(text, ())):
            if pos == role:
                answer.append({'text': text, 'role': role, 'domain': domain, 'structure': 'lexeme'})
        # Narrow compositional NP and location rules. These are hypotheses,
        # explicitly not new dictionary senses or linguistic gold.
        if role == 'N' and len(text) in (2, 3):
            for cut in range(1, len(text)):
                left, right = text[:cut], text[cut:]
                if any(p == 'A' for p, _ in self.lex.get(left, ())):
                    for p, domain in self.lex.get(right, ()):
                        if p == 'N':
                            answer.append({'text': text, 'role': 'N', 'domain': domain, 'structure': f'A{cut}+N{len(right)}'})
        if role == 'L' and len(text) in (2, 3) and any(p == 'L' for p, _ in self.lex.get(text[-1], ())):
            if self.tokens(text[:-1], 'N'):
                answer.append({'text': text, 'role': 'L', 'domain': 'location', 'structure': f'N{len(text)-1}+L1'})
        return list({json.dumps(x, sort_keys=True): x for x in answer}.values())

    @lru_cache(maxsize=16384)
    def syntax(self, text: str) -> dict:
        if not isinstance(text, str) or len(text) not in (5, 7):
            raise ValueError('Only complete five/seven-character lines')
        signatures = {}
        capped = False
        for frame in self.lexicon['frames']:
            def walk(offset: int, index: int, parts: list[dict]):
                nonlocal capped
                if len(signatures) >= 64:
                    capped = True
                    return
                if index == len(frame):
                    if offset == len(text):
                        signature = '|'.join(f"{p['role']}:{len(p['text'])}" for p in parts)
                        signatures.setdefault(signature, parts)
                    return
                remain = len(frame) - index - 1
                for end in range(offset + 1, min(len(text) - remain, offset + 4) + 1):
                    for token in self.tokens(text[offset:end], frame[index]):
                        walk(end, index + 1, parts + [token])
            walk(0, 0, [])
        return {'status': 'unsupported' if not signatures else 'ambiguous' if len(signatures) > 1 or capped else 'single_hypothesis',
                'analyses': [{'signature': key, 'tokens': value} for key, value in sorted(signatures.items())],
                'truncated': capped, 'independentlyReviewed': False}

    @lru_cache(maxsize=16384)
    def line(self, text: str) -> dict:
        length = len(text)
        if length not in (5, 7):
            raise ValueError('Invalid line length')
        cut = length - 3
        bridges = []
        for width in (2, 3):
            for start in range(max(0, cut - width + 1), min(cut, length - width + 1)):
                gram = text[start:start + width]
                bridges.append({'text': gram, 'start': start, 'end': start + width,
                                'trainingCount': self.grams[length][gram]})
        memberships = []
        for col, char in enumerate(text):
            entries = self.reference.get(char, [])
            groups = sorted({x['group_id'] for x in entries})
            tones = sorted({x['tone'] for x in entries})
            memberships.append({'column': col, 'character': char, 'tones': tones, 'groups': groups,
                                'sourceLines': sorted({n for x in entries for n in x.get('source_lines', [])}),
                                'status': 'unknown_character' if not entries else 'historical_choice_unresolved' if len(groups) > 1 else 'reference_only',
                                'contextVerified': False})
        hints = []
        for entry in self.hints:
            start = text.find(entry['word'])
            while start != -1:
                hints.append({**entry, 'start': start, 'end': start + len(entry['word']),
                              'status': 'lexical_hint_not_context_verdict', 'historicalMembershipResolved': False})
                start = text.find(entry['word'], start + 1)
        supported = sum(item['trainingCount'] > 0 for item in bridges)
        return {'text': text, 'cut': cut, 'bridgeEvidence': bridges,
                'unsupportedBridges': len(bridges) - supported, 'syntax': self.syntax(text),
                'readings': memberships, 'dictionaryHints': hints,
                'multiMembershipPositions': sum(len(x['groups']) > 1 for x in memberships),
                'contextualPronunciationVerified': False,
                'trainingLineDuplicate': text in self.full_lines[length]}

    @lru_cache(maxsize=32768)
    def pair(self, upper: str, lower: str) -> dict:
        if len(upper) != len(lower) or len(upper) not in (5, 7):
            raise ValueError('Pair lengths must match')
        a, b = self.syntax(upper), self.syntax(lower)
        shared = sorted({x['signature'] for x in a['analyses']} & {x['signature'] for x in b['analyses']})
        # Do NOT resolve POS ambiguity just because one parse makes a couplet fit.
        unambiguous = a['status'] == b['status'] == 'single_hypothesis'
        status = 'unknown'
        if unambiguous:
            status = 'aligned_under_draft_rules' if shared else 'different_under_draft_rules'
        return {'upper': upper, 'lower': lower, 'status': status,
                'upperAnalyses': len(a['analyses']), 'lowerAnalyses': len(b['analyses']),
                'sharedSignatures': shared, 'structuralPossibility': bool(shared),
                'strictAntithesisCertified': False,
                'note': 'Matching draft POS/span-length signatures does not certify semantic pairing, allusion or non-redundancy.'}

    def audit(self, lines: list[str]) -> dict:
        if not isinstance(lines, list) or len(lines) != 8 or any(not isinstance(x, str) for x in lines):
            raise ValueError('Expected eight strings')
        lengths = {len(x) for x in lines}
        if lengths not in ({5}, {7}):
            raise ValueError('Expected eight equal five/seven-character lines')
        items = [self.line(x) for x in lines]
        pairs = [self.pair(lines[i], lines[i + 1]) for i in (2, 4)]
        metrics = {
            'unsupportedBridges': sum(x['unsupportedBridges'] for x in items),
            'bridgeCount': sum(len(x['bridgeEvidence']) for x in items),
            'linesWithSyntaxHypothesis': sum(bool(x['syntax']['analyses']) for x in items),
            'unambiguousSyntaxLines': sum(x['syntax']['status'] == 'single_hypothesis' for x in items),
            'structurallyPossiblePairs': sum(x['structuralPossibility'] for x in pairs),
            'unambiguousAlignedPairs': sum(x['status'] == 'aligned_under_draft_rules' for x in pairs),
            'multiMembershipPositions': sum(x['multiMembershipPositions'] for x in items),
            'referenceOnlyPositions': sum(len(x['readings']) for x in items),
        }
        return {'schema': VERSION, 'lineLength': len(lines[0]), 'lines': items, 'middleCouplets': pairs, 'metrics': metrics,
                'strictAntithesis': 'unverified', 'contextualPronunciation': 'unverified',
                'literaryQuality': 'not_evaluated', 'publishAsVerified': False,
                'sources': self.sources, 'lexicalAnnotationStatus': self.lexicon['status']}

    def rank_score(self, record: dict) -> tuple[float, dict]:
        """Frozen experimental proxy, deliberately not a literary-quality score."""
        audit = self.audit(record['lines'])
        m, length = audit['metrics'], audit['lineLength']
        raw = float(record['score']) / 8
        if not math.isfinite(raw):
            raise ValueError('Nonfinite base score')
        penalties = {'unsupportedBridge': 0.4 * m['unsupportedBridges'] / m['bridgeCount'],
                     'unparsedLine': 0.3 * (8 - m['linesWithSyntaxHypothesis']) / 8,
                     'unmatchedCouplet': 0.5 * (2 - m['structurallyPossiblePairs']) / 2,
                     'ambiguousMembership': 0.2 * m['multiMembershipPositions'] / (8 * length)}
        score = raw - sum(penalties.values())
        return score, {'rawMeanScore': raw, 'penalties': penalties, 'proxyScore': score, 'audit': audit}
