"""Exact DP bridge for an explicitly supplied lower-line initial span pool.

The two variable character columns are coupled by a hard first-window factor:
their concatenation MUST be one complete supplied two-character candidate.
Without that factor, taking the Cartesian product of their character domains
would silently create candidate spans outside the frozen comparison pool.

This API has no target argument. It solves the learned source-restoration
objective; it does not infer historical readings or certify literary quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Mapping, Sequence

from .factor_dp import DPResult, convolution_factor, solve_local_chain
from .learned_conv import LearnedLocalScorer


@dataclass(frozen=True)
class SpanResult:
    status: str
    chosen_span: str | None
    score: float | None
    dp: DPResult
    candidate_hash: str
    candidate_count: int
    domain_counts: tuple[int, ...]
    task_hash: str
    vocabulary_hash: str
    unknown_in_context: tuple[str, ...]
    unknown_in_candidates: tuple[str, ...]
    vocabulary_status: str
    selected_lines: tuple[str, str] | None
    scope: str = (
        "Exact in the original finite two-character span pool under the frozen "
        "width-three source-restoration objective; fixed upper line, lower "
        "suffix and mask_start=0. Unknown characters retain their identities "
        "but use the zero unknown embedding. No historical-reading, metrical "
        "or literary-quality certification."
    )

    def to_dict(self):
        return asdict(self)


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _plain_string(value) -> bool:
    return isinstance(value, str) and value.isprintable() and not any(c.isspace() for c in value)


def solve_initial_span(
    model: LearnedLocalScorer,
    vocabulary: Mapping[str, int],
    upper_line: str,
    lower_suffix: str,
    candidate_spans: Sequence[str],
) -> SpanResult:
    """Select a supplied lower-line initial two-character span by exact DP.

    Candidate order is preserved in candidate_hash; ties in DP follow its
    lexicographic interleaved-token path rule. Explicit duplicate candidates
    are rejected rather than silently changing the pool. Every unknown
    character is reported, even if it is not part of the selected span.

    Vocabulary provenance must be audited by the caller. Id 0 or a missing key
    means unknown. Unknowns are scoreable under that explicit embedding rule,
    but are never silently converted into known characters or verified sounds.
    """
    if type(model) is not LearnedLocalScorer or model.kernel_width != 3:
        raise ValueError("this exact bridge requires the declared width-three learned model")
    if not _plain_string(upper_line) or len(upper_line) not in (5, 7):
        raise ValueError("upper_line must contain exactly five or seven non-whitespace characters")
    length = len(upper_line)
    if not _plain_string(lower_suffix) or len(lower_suffix) != length - 2:
        raise ValueError("lower_suffix must contain the fixed remaining length-minus-two characters")
    if isinstance(candidate_spans, (str, bytes)):
        raise ValueError("candidate_spans must be a sequence of complete span strings")
    spans = tuple(candidate_spans)
    if any(not _plain_string(span) or len(span) != 2 for span in spans):
        raise ValueError("each candidate must contain exactly two non-whitespace characters")
    if len(set(spans)) != len(spans):
        raise ValueError("duplicate complete candidates are not allowed")
    vocab = dict(vocabulary)
    if any(not isinstance(token, str) or isinstance(index, bool)
           or not isinstance(index, Integral) or not 0 <= index < model.vocab_size
           for token, index in vocab.items()):
        raise ValueError("invalid character vocabulary mapping")
    vocab = {token: int(index) for token, index in vocab.items()}
    candidate_hash = _digest({"schema": "versematrix.initial-span-pool.v1", "candidate_spans": spans})
    vocabulary_hash = _digest(vocab)
    task_hash = _digest({"schema": "versematrix.initial-span-task.v1", "upper_line": upper_line,
                         "lower_suffix": lower_suffix, "mask_start": 0,
                         "candidate_hash": candidate_hash, "vocabulary_hash": vocabulary_hash})
    unknown_context = tuple(sorted({c for c in upper_line + lower_suffix if vocab.get(c, 0) == 0}))
    unknown_candidates = tuple(sorted({c for span in spans for c in span if vocab.get(c, 0) == 0}))
    vocabulary_status = ("unknown_characters_scored_as_zero_embedding"
                         if unknown_context or unknown_candidates else "all_characters_in_vocabulary")
    first = tuple(sorted({span[0] for span in spans}))
    second = tuple(sorted({span[1] for span in spans}))
    domains = tuple(
        tuple((upper_line[col], lower) for lower in (first if col == 0 else second))
        if col < 2 else ((upper_line[col], lower_suffix[col - 2]),)
        for col in range(length)
    )
    scalar = model.as_local_scorer()
    feature = model.feature_factory(vocab, length, mask_start=0)
    original_factor = convolution_factor(scalar, feature, length)
    allowed = frozenset(spans)

    def constrained_factor(j, left, center, right):
        if j == 0 and center[1] + right[1] not in allowed:
            return -math.inf
        return original_factor(j, left, center, right)

    result = solve_local_chain(domains, constrained_factor)
    chosen = None if result.path is None else result.path[0][1] + result.path[1][1]
    if chosen is not None and chosen not in allowed:
        raise AssertionError("DP escaped the complete supplied span pool")
    selected = None if chosen is None else (upper_line, chosen + lower_suffix)
    return SpanResult(result.status, chosen, result.score, result, candidate_hash, len(spans),
                      tuple(map(len, domains)), task_hash, vocabulary_hash,
                      unknown_context, unknown_candidates, vocabulary_status, selected)
