"""Train-only phrase retrieval for the v0.4 source-restoration experiment.

This module does not read corpora, targets, titles, authors, dev/test labels, or
experiment files.  Its only fitting input is an explicit training-poem snapshot;
its only query input is the allowlisted ``VisibleQuery``.  The new associations
refer to exact source two-character spans, not a linguistic word segmentation.

The fast baseline is an algebraic implementation of the frozen BidirNGram
score.  Sparse count lookups and elementary arithmetic are vectorized, while
the original math.log and the original sentence summation order are retained.
No score rounding or tolerance-based tie rule is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import math
from types import MappingProxyType

import numpy as np

from .restoration import BidirNGram, VisibleQuery, digest


_COMPENSATED_BUILTIN_SUM = sum((1e16,1.0,-1e16)) == 1.0


def _readonly(values, dtype=None):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _math_log(values):
    """Use exactly the scalar libm call made by the original reference model."""
    values = np.asarray(values, dtype=np.float64)
    result = np.fromiter((math.log(x) for x in values.flat),
                         dtype=np.float64, count=values.size)
    return result.reshape(values.shape)


def _builtin_float_sum(terms):
    """Match this runtime's scalar sum, including CPython 3.12 compensation.

    Python 3.12's builtin_sum_impl uses Neumaier compensation for exact Python
    floats.  Our original reference's probability mixture and alignment use
    that builtin, whereas its per-direction sentence total uses plain +=.
    Source: https://github.com/python/cpython/blob/v3.12.14/Python/bltinmodule.c
    Tests compare this operation directly to the running Python builtin.
    """
    terms = np.broadcast_arrays(*[np.asarray(term,dtype=np.float64) for term in terms])
    total = terms[0].copy()
    if not _COMPENSATED_BUILTIN_SUM:
        for term in terms[1:]:
            total = total+term
        return total
    correction = np.zeros(total.shape,dtype=np.float64)
    for term in terms[1:]:
        next_total = total+term
        error = np.where(np.abs(total)>=np.abs(term),
                         (total-next_total)+term,(term-next_total)+total)
        correction = correction+error
        total = next_total
    return np.where((correction!=0)&np.isfinite(correction),total+correction,total)


class _SparseCounts:
    """Sorted integer keys permit vectorized exact lookup of sparse counts."""
    def __init__(self, counts, width, base):
        self.width = width
        self.base = base
        items = []
        for key, value in counts.items():
            coordinates = (key,) if width == 1 else key
            packed = 0
            for coordinate in coordinates:
                packed = packed * base + coordinate + 1
            items.append((packed, value))
        items.sort()
        self.keys = _readonly([item[0] for item in items], np.int64)
        self.values = _readonly([item[1] for item in items], np.float64)

    def lookup(self, *coordinates):
        if len(coordinates) != self.width:
            raise ValueError("sparse key width mismatch")
        coordinates = np.broadcast_arrays(*[np.asarray(c, dtype=np.int64)
                                              for c in coordinates])
        packed = np.zeros(coordinates[0].shape, dtype=np.int64)
        for coordinate in coordinates:
            packed = packed * self.base + coordinate + 1
        if self.keys.size == 0:
            return np.zeros(packed.shape, dtype=np.float64)
        positions = np.searchsorted(self.keys, packed)
        safe = np.minimum(positions, len(self.keys)-1)
        found = (positions < len(self.keys)) & (self.keys[safe] == packed)
        return np.where(found, self.values[safe], 0.0)


class _FastBidir:
    """Frozen count arrays, with the reference's complete scoring semantics."""
    def __init__(self, reference, candidates_by_length):
        self.reference = reference
        self.v = reference.v
        self.base = self.v + 2  # Includes BOS -1, UNK 0 and EOS v.
        if self.base ** 3 > np.iinfo(np.int64).max:
            raise ValueError("vocabulary is too large for collision-free packed keys")
        self.tables = []
        for table in reference.tables:
            self.tables.append({
                "unigram": _readonly([table["unigram"][i] for i in range(self.v+1)], np.float64),
                "context1": _readonly([table["context1"][i] for i in range(-1,self.v+1)], np.float64),
                "bigram": _SparseCounts(table["bigram"], 2, self.base),
                "trigram": _SparseCounts(table["trigram"], 3, self.base),
                "context2": _SparseCounts(table["context2"], 2, self.base),
            })
        self.aligned = _SparseCounts(reference.aligned, 2, self.base)
        self.lower = _readonly([reference.lower[i] for i in range(self.v)], np.float64)
        self.ids = {}
        self.constants = {}
        for length, spans in candidates_by_length.items():
            if not spans:
                continue
            ids = np.asarray([reference.encode(span) for span in spans], dtype=np.int64)
            first, second = ids[:,0], ids[:,1]
            self.ids[length] = _readonly(ids)
            self.constants[length] = {
                "forward_first": _readonly(self.log_probability(0, -1, -1, first)),
                "forward_second": _readonly(self.log_probability(0, -1, first, second)),
                "reverse_eos": _readonly(self.log_probability(1, second, first, reference.eos)),
            }

    def log_probability(self, direction, a, b, target):
        if all(np.ndim(x) == 0 for x in (a,b,target)):
            return self.reference._log_probability(direction, int(a), int(b), int(target))
        a,b,target = np.broadcast_arrays(*[np.asarray(x, dtype=np.int64) for x in (a,b,target)])
        table = self.tables[direction]
        reference = self.reference
        alpha = reference.alpha
        size = self.v+1
        p1 = (table["unigram"][target]+alpha)/(reference.totals[direction]+alpha*size)
        p2 = (table["bigram"].lookup(b,target)+alpha)/(table["context1"][b+1]+alpha*size)
        p3 = (table["trigram"].lookup(a,b,target)+alpha)/(table["context2"].lookup(a,b)+alpha*size)
        probability = _builtin_float_sum((reference.weights[0]*p1,
                                          reference.weights[1]*p2,
                                          reference.weights[2]*p3))
        return _math_log(probability)

    @lru_cache(maxsize=64)
    def aligned_for_upper(self, upper):
        reference = self.reference
        lower_ids = np.arange(self.v, dtype=np.int64)
        prior = (self.lower+reference.alpha)/(reference.lower_total+reference.alpha*self.v)
        probability = ((self.aligned.lookup(upper,lower_ids)+reference.aligned_strength*prior)
                       /(reference.upper[upper]+reference.aligned_strength))
        return _readonly(_math_log(probability))

    def score(self, query):
        reference = self.reference
        ids = self.ids[query.length]
        first, second = ids[:,0], ids[:,1]
        suffix = reference.encode(query.lower_suffix)
        upper = reference.encode(query.upper_line)
        constants = self.constants[query.length]

        # Full forward sentence, including EOS.  After suffix[1], all terms
        # are query constants; add them individually in the original order.
        forward = constants["forward_first"].copy()
        forward = forward + constants["forward_second"]
        a,b = first,second
        for target in suffix+[reference.eos]:
            forward = forward + self.log_probability(0,a,b,target)
            a,b = b,target

        # In reverse, the suffix precedes the two candidate characters.
        reverse_constant = 0.0
        a,b = -1,-1
        for target in reversed(suffix):
            reverse_constant += reference._log_probability(1,a,b,target)
            a,b = b,target
        reverse = reverse_constant + self.log_probability(1,a,b,second)
        reverse = reverse + self.log_probability(1,b,second,first)
        reverse = reverse + constants["reverse_eos"]

        aligned_terms = [self.aligned_for_upper(upper[0])[first],
                         self.aligned_for_upper(upper[1])[second]]
        aligned_terms.extend(reference._aligned_log_probability(u,l) for u,l in zip(upper[2:],suffix))
        aligned = _builtin_float_sum(aligned_terms)/query.length
        directions = _builtin_float_sum((forward/(query.length+1),reverse/(query.length+1)))
        score = .5*directions + reference.aligned_weight*aligned
        if not np.isfinite(score).all():
            raise ValueError("nonfinite baseline score")
        return score


class PhraseRetriever:
    """One train-only snapshot, no refit mutator and no target-bearing queries.

    Wide candidates are every exact within-line adjacent pair occurring at
    least twice in training, counted separately for five/seven-character poems.
    Narrow candidates retain the old even-line-prefix count-at-least-two rule.
    New bonuses use exact text even when the character baseline encodes UNK.
    """
    min_count = 2
    prior_alpha = .1
    association_strength = 5.0

    def __init__(self, train_poems, vocabulary):
        poems = list(train_poems)
        if not poems or any(p.get("split") != "train" for p in poems):
            raise ValueError("retrieval statistics require a nonempty train-only snapshot")
        if len({p.get("poem_id") for p in poems}) != len(poems):
            raise ValueError("training poem IDs must be unique")
        normalized = []
        for poem in poems:
            lines = poem.get("lines")
            if (not isinstance(poem.get("poem_id"),str) or not poem["poem_id"]
                    or not isinstance(lines,(tuple,list)) or len(lines) != 8
                    or not all(isinstance(line,str) for line in lines)):
                raise ValueError("each training record needs a unique ID and eight text lines")
            length = len(lines[0])
            if (length not in (5,7) or any(len(line) != length for line in lines)
                    or poem.get("length",length) != length):
                raise ValueError("training poem must have shape 8 by 5 or 8 by 7")
            normalized.append({"poem_id":poem["poem_id"],"split":"train",
                               "length":length,"lines":tuple(lines)})
        vocabulary = dict(vocabulary)
        if vocabulary.get("<UNK>") != 0:
            raise ValueError("the explicit <UNK> token must have ID zero")
        self.vocabulary = MappingProxyType(vocabulary)
        self.fit_poem_ids = tuple(sorted(p["poem_id"] for p in normalized))
        self._ngram = BidirNGram(vocabulary).fit(normalized)

        all_counts = {length:Counter() for length in (5,7)}
        narrow_counts = {length:Counter() for length in (5,7)}
        origins = {length:defaultdict(set) for length in (5,7)}
        for poem in normalized:
            length = poem["length"]
            for row,line in enumerate(poem["lines"]):
                for start in range(length-1):
                    span = line[start:start+2]
                    all_counts[length][span] += 1
                    origins[length][span].add(poem["poem_id"])
                if row%2 == 1:
                    narrow_counts[length][line[:2]] += 1
        self.candidates_by_length = MappingProxyType({
            length:tuple(sorted(span for span,count in all_counts[length].items()
                                if count >= self.min_count)) for length in (5,7)})
        if not any(self.candidates_by_length.values()):
            raise ValueError("no training span meets the fixed minimum count")
        self._indices = {length:{span:i for i,span in enumerate(spans)}
                         for length,spans in self.candidates_by_length.items()}
        self._counts = {}
        self._narrow_counts = {}
        self._narrow_masks = {}
        self._origins = {}
        self._priors = {}
        for length,spans in self.candidates_by_length.items():
            counts = _readonly([all_counts[length][span] for span in spans],np.int64)
            ncounts = _readonly([narrow_counts[length][span] for span in spans],np.int64)
            self._counts[length] = counts
            self._narrow_counts[length] = ncounts
            self._narrow_masks[length] = _readonly(ncounts >= self.min_count,np.bool_)
            self._origins[length] = tuple(tuple(sorted(origins[length][span])) for span in spans)
            if spans:
                prior = (counts+self.prior_alpha)/(int(counts.sum())+self.prior_alpha*len(spans))
            else:
                prior = np.zeros(0,dtype=np.float64)
            self._priors[length] = _readonly(prior)

        alignment = {length:defaultdict(Counter) for length in (5,7)}
        boundary = {length:defaultdict(Counter) for length in (5,7)}
        for poem in normalized:
            length = poem["length"]
            index = self._indices[length]
            for row in (0,2,4,6):
                upper,lower = poem["lines"][row:row+2]
                for start in range(length-1):
                    span = lower[start:start+2]
                    if span in index:
                        alignment[length][upper[start:start+2]][index[span]] += 1
            for line in poem["lines"]:
                for start in range(length-3):
                    span = line[start:start+2]
                    if span in index:
                        boundary[length][line[start+2:start+4]][index[span]] += 1
        self._alignment = self._freeze_associations(alignment)
        self._boundary = self._freeze_associations(boundary)
        self._fast = _FastBidir(self._ngram,self.candidates_by_length)

    @staticmethod
    def _freeze_associations(tables):
        frozen = {}
        for length,contexts in tables.items():
            frozen[length] = {}
            for context,counts in contexts.items():
                pairs = sorted(counts.items())
                frozen[length][context] = (
                    _readonly([i for i,_ in pairs],np.int64),
                    _readonly([n for _,n in pairs],np.int64))
        return frozen

    def _bonus(self,table,length,context):
        size = len(self.candidates_by_length[length])
        counts = np.zeros(size,dtype=np.int64)
        if context in table[length]:
            indices,values = table[length][context]
            counts[indices] = values
        bonus = np.log1p(counts/(self.association_strength*self._priors[length]))
        return _readonly(bonus),_readonly(counts)

    def score_components(self, query):
        if type(query) is not VisibleQuery:
            raise ValueError("pass the allowlisted VisibleQuery, never a task or target-bearing object")
        spans = self.candidates_by_length[query.length]
        if not spans:
            raise ValueError("no eligible training spans for this query length")
        alignment,alignment_counts = self._bonus(self._alignment,query.length,query.upper_line[:2])
        boundary,boundary_counts = self._bonus(self._boundary,query.length,query.lower_suffix[:2])
        scores = _readonly(self._fast.score(query))
        if not all(np.isfinite(x).all() for x in (scores,alignment,boundary)):
            raise ValueError("nonfinite retrieval component")
        return {
            "candidates":spans,"ngram_scores":scores,
            "alignment_bonus":alignment,"boundary_bonus":boundary,
            "narrow_mask":self._narrow_masks[query.length],
            "alignment_counts":alignment_counts,"boundary_counts":boundary_counts,
            "span_prior":self._priors[query.length],
            "query_hash":digest(asdict(query)),
            "unknown_visible_characters":tuple(sorted({c for c in query.upper_line+query.lower_suffix
                                                         if c not in self.vocabulary})),
        }

    @staticmethod
    def rank_components(components,alignment_weight=0.0,boundary_weight=0.0,
                        *,pool="wide",limit=None):
        """Rank actual source candidates; the score and lexical tie rule are exact."""
        weights = (alignment_weight,boundary_weight)
        if any(isinstance(w,(bool,np.bool_)) or not np.isscalar(w)
               or not np.isfinite(w) or w < 0 for w in weights):
            raise ValueError("association weights must be finite nonnegative numbers")
        if pool not in ("wide","narrow"):
            raise ValueError("pool must be wide or narrow")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        spans = tuple(components["candidates"])
        if (not spans or any(not isinstance(s,str) or len(s)!=2 for s in spans)
                or len(set(spans)) != len(spans)):
            raise ValueError("candidates must be nonempty distinct source two-character spans")
        arrays = [np.asarray(components[name],dtype=np.float64)
                  for name in ("ngram_scores","alignment_bonus","boundary_bonus")]
        if any(a.shape != (len(spans),) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("one finite component value per candidate is required")
        scores = arrays[0] + alignment_weight*arrays[1] + boundary_weight*arrays[2]
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite combined retrieval score")
        mask = np.asarray(components["narrow_mask"])
        if mask.shape != (len(spans),) or mask.dtype.kind != "b":
            raise ValueError("narrow mask must be one Boolean per candidate")
        indices = np.arange(len(spans),dtype=np.int64)
        if pool == "narrow":
            indices = indices[mask]
        if indices.size:
            words = np.asarray(spans,dtype=str)
            order = np.lexsort((words[indices],-scores[indices]))
            indices = indices[order]
        if limit is not None:
            indices = indices[:limit]
        selected = tuple(spans[i] for i in indices)
        query_hash = components.get("query_hash")
        return {"candidates":selected,"indices":_readonly(indices),
                "scores":_readonly(scores[indices]),"pool":pool,
                "alignment_weight":float(alignment_weight),"boundary_weight":float(boundary_weight),
                "query_hash":query_hash,
                "candidate_set_hash":digest({"query_hash":query_hash,"candidates":sorted(selected)})}

    def rank(self,query,alignment_weight=0.0,boundary_weight=0.0,*,pool="wide",limit=None):
        return self.rank_components(self.score_components(query),alignment_weight,boundary_weight,
                                    pool=pool,limit=limit)

    def reference_score(self,query,span):
        """Unoptimized original score, available for independent numerical checks."""
        return self._ngram.score(query,span)

    def candidate_resources(self):
        return {"schema":"versematrix.train-phrase-candidates.v0.4","source_split":"train",
                "min_count":self.min_count,"normalization":"none; exact source code points",
                "wide_definition":"all within-line adjacent two-character spans, all eight rows",
                "narrow_definition":"first two characters of even-numbered rows only",
                "by_length":{str(length):[
                    {"text":span,"count":int(self._counts[length][i]),
                     "source_poem_ids":list(self._origins[length][i]),
                     "narrow_count":int(self._narrow_counts[length][i]),
                     "narrow_eligible":bool(self._narrow_masks[length][i])}
                    for i,span in enumerate(spans)]
                    for length,spans in self.candidates_by_length.items()}}

    def association_resources(self):
        """Deterministic raw counts; caller decides whether to persist this audit."""
        return {name:{str(length):[
                    {"context":context,"span":self.candidates_by_length[length][int(i)],"count":int(n)}
                    for context,(indices,counts) in sorted(table[length].items())
                    for i,n in zip(indices,counts)] for length in (5,7)}
                for name,table in (("aligned_span_pairs",self._alignment),
                                   ("adjacent_span_pairs",self._boundary))}

    def summary(self):
        associations = {}
        for name,table in (("alignment",self._alignment),("boundary",self._boundary)):
            associations[name] = {str(length):{
                "contexts":len(table[length]),
                "unique_context_span_pairs":sum(len(values[0]) for values in table[length].values()),
                "events":sum(int(values[1].sum()) for values in table[length].values())}
                for length in (5,7)}
        return {"schema":"versematrix.phrase-retriever.v0.4",
                "fit_poems":len(self.fit_poem_ids),"fit_poem_ids_sha256":digest(self.fit_poem_ids),
                "baseline":self._ngram.summary(),
                "candidates_by_length":{str(length):{
                    "wide":len(spans),"narrow":int(self._narrow_masks[length].sum()),
                    "wide_candidate_sha256":digest(spans),
                    "candidate_unknown_characters":sorted({c for s in spans for c in s if c not in self.vocabulary})}
                    for length,spans in self.candidates_by_length.items()},
                "associations":associations,"prior_alpha":self.prior_alpha,
                "association_strength":self.association_strength,
                "bonus":"log1p(count(context, span) / (5 * train_span_prior))",
                "association_length_scope":"each poem length separately; eligible wide targets only",
                "baseline_numerics":"original math.log and complete sentence summation order; no score rounding",
                "baseline_builtin_sum_mode":"Neumaier compensation" if _COMPENSATED_BUILTIN_SUM else "plain addition",
                "tie_rule":"descending exact floating score, then source-text lexicographic order",
                "unknown_policy":"baseline character UNK 0; association keys retain exact source text",
                "target_kind":"observed source restoration, not literary quality"}
