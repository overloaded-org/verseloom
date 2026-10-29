"""Exact optimization of finite domains with an additive radius-one factor.

The objective is sum_j f_j(s[j-1],s[j],s[j+1]); boundary symbols are None.
Each s[j] may represent a pair of aligned upper/lower-line tokens. A fixed
width-three LocalConvScorer is exactly representable when its input features
depend only on (token, row, column) and fixed task metadata.
Arbitrary global semantic features or hard constraints do not satisfy this API.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
from typing import Callable
import numpy as np

Symbol = tuple[str, str]
Factor = Callable[[int, Symbol | None, Symbol, Symbol | None], float]


def _add_finite(left: float, right: float) -> float:
    result = left + right
    if not math.isfinite(result):
        raise ValueError("finite factor scores overflowed during accumulation")
    return result


@dataclass(frozen=True)
class DPResult:
    status: str
    path: tuple[Symbol, ...] | None
    score: float | None
    factor_calls: int
    transitions: int
    peak_states: int
    full_combinations: int
    scope: str = "exact_for_declared_radius_one_additive_objective_and_domains"


def solve_local_chain(domains: tuple[tuple[Symbol, ...], ...], factor: Factor) -> DPResult:
    if not domains:
        raise ValueError("need at least one position")
    for domain in domains:
        if len(set(domain)) != len(domain):
            raise ValueError("duplicate symbol in domain")
        if any(not isinstance(s, tuple) or len(s) != 2 or any(not isinstance(x, str) for x in s) for s in domain):
            raise ValueError("symbols must be pairs of token strings")
    total = math.prod(map(len, domains))
    if not total:
        return DPResult("infeasible", None, None, 0, 0, 0, 0)
    cache = {}
    transitions = 0

    def value(j, left, current, right):
        key = (j, left, current, right)
        if key not in cache:
            v = float(factor(*key))
            if math.isnan(v) or v == math.inf:
                raise ValueError("factor returned NaN or positive infinity")
            cache[key] = v
        return cache[key]

    def choose(old, new):
        if old is None or new[0] > old[0] or (new[0] == old[0] and new[1] < old[1]):
            return new
        return old

    n = len(domains)
    if n == 1:
        best = None
        for symbol in domains[0]:
            transitions += 1
            v = value(0, None, symbol, None)
            if v != -math.inf:
                best = choose(best, (v, (symbol,)))
        return DPResult("optimal" if best else "infeasible", best[1] if best else None,
                        best[0] if best else None, len(cache), transitions, len(domains[0]), total)
    states = {}
    for first, second in itertools.product(domains[0], domains[1]):
        transitions += 1
        v = value(0, None, first, second)
        if v != -math.inf:
            states[(first, second)] = (v, (first, second))
    peak = len(states)
    for j in range(2, n):
        nxt = {}
        for (left, center), (old_score, old_path) in sorted(states.items()):
            for right in domains[j]:
                transitions += 1
                v = value(j - 1, left, center, right)
                if v == -math.inf:
                    continue
                candidate = (_add_finite(old_score, v), old_path + (right,))
                key = (center, right)
                nxt[key] = choose(nxt.get(key), candidate)
        states = nxt
        peak = max(peak, len(states))
    best = None
    for (left, center), (old_score, path) in sorted(states.items()):
        transitions += 1
        v = value(n - 1, left, center, None)
        if v != -math.inf:
            best = choose(best, (_add_finite(old_score, v), path))
    return DPResult("optimal" if best else "infeasible", best[1] if best else None,
                    best[0] if best else None, len(cache), transitions, peak, total)


def path_score(path: tuple[Symbol, ...], factor: Factor) -> float:
    values = [float(factor(j, path[j-1] if j else None, symbol,
                           path[j+1] if j + 1 < len(path) else None))
              for j, symbol in enumerate(path)]
    if any(math.isnan(v) or v == math.inf for v in values):
        raise ValueError("factor returned NaN or positive infinity")
    if -math.inf in values:
        return -math.inf
    total = 0.0
    for value in values:
        total = _add_finite(total, value)
    return total


def convolution_factor(scorer, feature: Callable[[int, int, str], np.ndarray], length: int) -> Factor:
    """Factorize the existing scorer WITHOUT changing its random/learned weights.

    Feature must be a pure function of the supplied position/token and fixed
    metadata. This is a mathematical precondition, not automatically inferred.
    The returned factors retain horizontal windows and aligned-pair nonlinearities.
    """
    if type(length) is not int or length < 1 or scorer.horizontal_kernel.shape[0] != 3:
        raise ValueError("this factorization supports positive length and width exactly three")
    embeddings = {}

    def emb(row, col, token):
        key = (row, col, token)
        if key not in embeddings:
            array = np.asarray(feature(row, col, token), dtype=np.float64)
            if array.shape != (scorer.feature_dim,) or not np.isfinite(array).all():
                raise ValueError("invalid fixed feature")
            embeddings[key] = array.copy()
        return embeddings[key]

    def factor(j, left, center, right):
        if not 0 <= j < length:
            raise ValueError("factor position outside chain")
        if (j == 0) != (left is None) or (j == length - 1) != (right is None):
            raise ValueError("incorrect boundary symbols")
        horizontal = 0.0
        neighbors = (left, center, right)
        for row in (0, 1):
            activation = np.zeros(scorer.channels)
            for offset, symbol in enumerate(neighbors):
                if symbol is not None:
                    activation += emb(row, j + offset - 1, symbol[row]) @ scorer.horizontal_kernel[offset]
            horizontal += float(np.tanh(activation) @ scorer.horizontal_readout) / (2 * length)
        aligned = sum(emb(row, j, center[row]) @ scorer.pair_kernel[row] for row in (0, 1))
        paired = float(np.tanh(aligned) @ scorer.pair_readout) / length
        return horizontal + paired
    return factor
