"""Exact full-verse composition in a small, explicitly declared finite domain.

The caller supplies context-resolved readings, basic templates and rhyme groups.
For each template/rhyme pair the remaining domains are independent by cell; the
objective is the SUM of four fixed-feature LocalConvScorer couplet scores. The
four radius-one chain optima therefore give that configuration's whole-verse
optimum. No arbitrary global semantics, word-span or repetition constraint is
accepted by this API. This is not a certificate of literary merit or of every
historically legitimate metrical variant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Callable

import numpy as np

from .convolution import LocalConvScorer
from .factor_dp import DPResult, convolution_factor, solve_local_chain
from .prosody import Template, generate_template


EXACT_SCOPE = (
    "Complete finite caller-supplied candidate domains, canonical strict basic "
    "templates and caller-supplied rhyme groups only; objective is the sum of "
    "four width-three LocalConvScorer couplet scores on frozen per-cell token "
    "features and weights, evaluated in floating-point arithmetic. No global "
    "semantic, repetition, word-span, common-prefix or aojiu constraints. "
    "Reading verification is asserted by the caller, not inferred from meter; "
    "optimality concerns this objective, not literary quality."
)
_UNKNOWN = {"", "?", "unknown", "unresolved", "unverified", "none", "null"}


@dataclass(frozen=True)
class ReadingCandidate:
    """One character with a reading already resolved for THIS cell's context.

    `verified=True` is the caller's attestation. Rhyme and reading_id are opaque
    known identifiers, not inferred from Mandarin or from the target meter.
    Multiple records for one token at one cell are never resolved by search.
    """

    token: str
    tone: str
    rhyme: str
    reading_id: str
    verified: bool


@dataclass(frozen=True)
class ReadingIssue:
    row: int
    column: int
    token: str
    reason: str


@dataclass(frozen=True)
class ConfigurationResult:
    template_id: str
    rhyme_group: str
    status: str
    filtered_domain_sizes: tuple[tuple[int, ...], ...]
    full_combinations: int
    missing_positions: tuple[tuple[int, int], ...] = ()
    score: float | None = None
    couplets: tuple[DPResult, ...] = ()


@dataclass(frozen=True)
class VerseResult:
    status: str
    length: int
    supplied_templates: tuple[str, ...]
    supplied_rhyme_groups: tuple[str, ...]
    full_combinations: int
    feasible_combinations: int | None
    input_domain_sizes: tuple[tuple[int, ...], ...]
    configurations: tuple[ConfigurationResult, ...] = ()
    selected_template: str | None = None
    selected_rhyme: str | None = None
    selected: tuple[tuple[ReadingCandidate, ...], ...] | None = None
    lines: tuple[str, ...] | None = None
    score: float | None = None
    issues: tuple[ReadingIssue, ...] = ()
    frozen_feature_count: int = 0
    exact_scope: str = EXACT_SCOPE
    literary_quality: str = "not_evaluated"

    def to_dict(self) -> dict:
        return asdict(self)


def _known_id(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() not in _UNKNOWN


def _basic_templates(templates: tuple[Template, ...], length: int) -> tuple[Template, ...]:
    if not templates:
        raise ValueError("supply a nonempty finite set of basic templates")
    seen = set()
    for template in templates:
        if not isinstance(template, Template) or template.length != length:
            raise ValueError("every template must describe this complete eight-line matrix")
        canonical = generate_template(length, template.start_second_tone, template.first_rhymed)
        if (template.template_id, template.core, template.rhyme_rows) != (
            canonical.template_id, canonical.core, canonical.rhyme_rows
        ):
            raise ValueError("only canonical strict basic templates are supported")
        if template.template_id in seen:
            raise ValueError("duplicate template")
        seen.add(template.template_id)
    return tuple(sorted(templates, key=lambda t: t.template_id))


def _snapshot_scorer(scorer: LocalConvScorer) -> LocalConvScorer:
    # A subclass could override the objective while retaining these arrays.
    if type(scorer) is not LocalConvScorer:
        raise ValueError("the declared objective requires the LocalConvScorer architecture")
    if any(type(n) is not int or n <= 0 for n in (scorer.feature_dim, scorer.channels)):
        raise ValueError("scorer dimensions must be positive integers")
    model = LocalConvScorer(scorer.feature_dim, scorer.channels, seed=0)
    shapes = {
        "horizontal_kernel": (3, scorer.feature_dim, scorer.channels),
        "pair_kernel": (2, scorer.feature_dim, scorer.channels),
        "horizontal_readout": (scorer.channels,),
        "pair_readout": (scorer.channels,),
    }
    for name, shape in shapes.items():
        values = np.array(getattr(scorer, name), dtype=np.float64, copy=True)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid finite width-three scorer parameter: {name}")
        values.flags.writeable = False
        setattr(model, name, values)
    return model


def solve_verse(
    candidates: tuple[tuple[tuple[ReadingCandidate, ...], ...], ...],
    templates: tuple[Template, ...],
    rhyme_groups: tuple[str, ...],
    scorer: LocalConvScorer,
    feature: Callable[[int, int, str], np.ndarray],
) -> VerseResult:
    """Optimize the complete 8 x 5/7 matrix in the declared restricted scope.

    Structural/numeric errors raise ValueError. Unknown, unverified or ambiguous
    context readings return needs_reading_review BEFORE any tone/rhyme filtering.
    Empty legal domains are reported for every configuration; none are hidden.
    The feature callback sees GLOBAL row 0..7, column and token. Each declared
    candidate's vector is copied once before search, so later feature/parameter
    mutation cannot change the objective. Fixed task context may condition those
    vectors; features depending on the selected full output are outside this API.

    Configuration ties use sorted template/group identifiers. Within each
    couplet solve_local_chain applies its deterministic interleaved-token tie rule.
    """
    if len(candidates) != 8 or len(candidates[0]) not in (5, 7):
        raise ValueError("expected eight rows of five or seven character domains")
    length = len(candidates[0])
    if any(len(row) != length for row in candidates):
        raise ValueError("all eight rows must have the same length")
    candidates = tuple(tuple(tuple(domain) for domain in row) for row in candidates)
    templates = _basic_templates(tuple(templates), length)
    groups = tuple(rhyme_groups)
    if not groups or any(not _known_id(g) for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("supply distinct known rhyme group identifiers")
    groups = tuple(sorted(groups))
    if not callable(feature):
        raise ValueError("feature must provide fixed per-cell token vectors")
    sizes = tuple(tuple(len(domain) for domain in row) for row in candidates)
    total = math.prod(size for row in sizes for size in row)
    base = dict(length=length, supplied_templates=tuple(t.template_id for t in templates),
                supplied_rhyme_groups=groups, full_combinations=total, input_domain_sizes=sizes)
    issues = []
    for row, domains in enumerate(candidates):
        for col, domain in enumerate(domains):
            records = {}
            for candidate in domain:
                if not isinstance(candidate, ReadingCandidate):
                    raise ValueError("cell domains must contain ReadingCandidate records")
                if not isinstance(candidate.token, str) or len(candidate.token) != 1 or candidate.token.isspace():
                    raise ValueError("each character slot requires one non-whitespace Unicode code point")
                if type(candidate.verified) is not bool:
                    raise ValueError("verified must be an explicit boolean")
                if candidate in records.get(candidate.token, ()):
                    raise ValueError("duplicate candidate reading record")
                records.setdefault(candidate.token, []).append(candidate)
                reasons = []
                if candidate.tone not in ("P", "Z"):
                    reasons.append("unknown_tone")
                if not _known_id(candidate.rhyme):
                    reasons.append("unknown_rhyme")
                if not _known_id(candidate.reading_id):
                    reasons.append("unknown_context_reading")
                if not candidate.verified:
                    reasons.append("unverified_context_reading")
                issues.extend(ReadingIssue(row, col, candidate.token, reason) for reason in reasons)
            issues.extend(ReadingIssue(row, col, token, "ambiguous_context_reading")
                          for token, values in records.items() if len(values) > 1)
    if issues:
        return VerseResult("needs_reading_review", feasible_combinations=None,
                           issues=tuple(issues), **base)

    model = _snapshot_scorer(scorer)
    fixed = {}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for row, domains in enumerate(candidates):
                for col, domain in enumerate(domains):
                    for candidate in domain:
                        vector = np.array(feature(row, col, candidate.token), dtype=np.float64, copy=True)
                        if vector.shape != (model.feature_dim,) or not np.isfinite(vector).all():
                            raise ValueError("all declared candidates need finite fixed feature vectors")
                        vector.flags.writeable = False
                        fixed[(row, col, candidate.token)] = vector

            configurations = []
            best = None
            feasible = 0
            for template, group in itertools.product(templates, groups):
                filtered = tuple(tuple(tuple(c for c in domain
                    if c.tone == template.core[row][col]
                    and (row not in template.rhyme_rows or col != length - 1 or c.rhyme == group))
                    for col, domain in enumerate(domains)) for row, domains in enumerate(candidates))
                counts = tuple(tuple(len(domain) for domain in row) for row in filtered)
                combinations = math.prod(size for row in counts for size in row)
                missing = tuple((row, col) for row in range(8) for col in range(length)
                                if not filtered[row][col])
                if missing:
                    configurations.append(ConfigurationResult(template.template_id, group, "infeasible",
                                                              counts, 0, missing))
                    continue
                couplets = []
                selected = []
                score = 0.0
                for upper in (0, 2, 4, 6):
                    domains = tuple(tuple(itertools.product(
                        (c.token for c in filtered[upper][col]),
                        (c.token for c in filtered[upper + 1][col]))) for col in range(length))
                    local_feature = lambda row, col, token, upper=upper: fixed[(upper + row, col, token)]
                    raw_factor = convolution_factor(model, local_feature, length)

                    def finite_factor(*args):
                        value = float(raw_factor(*args))
                        if not math.isfinite(value):
                            raise ValueError("convolution factors must remain finite")
                        return value

                    result = solve_local_chain(domains, finite_factor)
                    if result.status != "optimal":
                        raise AssertionError("nonempty independent domains with finite scores must be feasible")
                    score += result.score
                    if not math.isfinite(score):
                        raise ValueError("whole-verse score overflowed during accumulation")
                    couplets.append(result)
                    for local_row in (0, 1):
                        selected.append(tuple(next(c for c in filtered[upper + local_row][col]
                                                   if c.token == result.path[col][local_row])
                                              for col in range(length)))
                configurations.append(ConfigurationResult(template.template_id, group, "optimal",
                                                          counts, combinations, score=score,
                                                          couplets=tuple(couplets)))
                feasible += combinations
                if best is None or score > best[0]:
                    best = (score, template.template_id, group, tuple(selected))
    except FloatingPointError as exc:
        raise ValueError("non-finite floating-point arithmetic in the declared objective") from exc
    if feasible > total:
        raise AssertionError("distinct canonical template/rhyme domains should be disjoint")
    if best is None:
        return VerseResult("infeasible", feasible_combinations=0, configurations=tuple(configurations),
                           frozen_feature_count=len(fixed), **base)
    return VerseResult("optimal", feasible_combinations=feasible, configurations=tuple(configurations),
                       selected_template=best[1], selected_rhyme=best[2], selected=best[3],
                       lines=tuple("".join(c.token for c in row) for row in best[3]), score=best[0],
                       frozen_feature_count=len(fixed), **base)
