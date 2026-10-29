"""Finite, auditable matrix organization; no claim of a new search algorithm.

Both policies operate on full grids, use exactly the same moves and constraints,
and can revisit an earlier accepted frontier state. Every attempted neighbor,
including duplicates and illegal neighbors, consumes proposal budget. Scorers
must be deterministic and independent of the search policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import itertools
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Callable, Iterable

State = tuple[str, ...]
Constraint = Callable[[State], tuple[bool, tuple[str, ...]]]
Scorer = Callable[[State], float]


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatrixProblem:
    problem_id: str
    rows: int
    columns: int
    domains: tuple[tuple[str, ...], ...]
    initial: State
    # In addition to the single-cell moves, all cells in each block change jointly.
    # Both search policies receive precisely this same move set.
    linked_blocks: tuple[tuple[int, ...], ...] = ()

    def validate(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
               for v in (self.rows, self.columns)):
            raise ValueError("Grid dimensions must be positive")
        n = self.rows * self.columns
        if len(self.initial) != n or len(self.domains) != n:
            raise ValueError("Grid, domains and initial state have different sizes")
        for token, domain in zip(self.initial, self.domains):
            if not domain or len(set(domain)) != len(domain):
                raise ValueError("Each domain must be nonempty and duplicate-free")
            if any(not isinstance(x, str) or not x for x in domain):
                raise ValueError("Tokens must be nonempty strings")
            if token not in domain:
                raise ValueError("Initial token outside its declared domain")
        if len(set(self.linked_blocks)) != len(self.linked_blocks):
            raise ValueError("Duplicate linked block")
        for block in self.linked_blocks:
            if len(block) < 2 or tuple(sorted(set(block))) != block:
                raise ValueError("A linked block must contain sorted distinct positions")
            if any(p < 0 or p >= n for p in block):
                raise ValueError("Linked block outside the grid")

    @property
    def problem_hash(self) -> str:
        return digest(asdict(self))

    def neighbors(self, state: State) -> Iterable[tuple[tuple[int, ...], State]]:
        blocks = tuple((p,) for p in range(len(state))) + self.linked_blocks
        for block in blocks:
            alternatives = [tuple(t for t in self.domains[p] if t != state[p]) for p in block]
            for values in itertools.product(*alternatives):
                changed = list(state)
                for p, value in zip(block, values):
                    changed[p] = value
                yield block, tuple(changed)


@dataclass
class MatrixSearchResult:
    strategy: str
    stop_reason: str
    best_state: State
    best_score: float
    proposals: int
    score_calls: int
    unique_legal_states: int
    expanded_states: int
    best_path: list[State]
    trace: list[dict]

    def summary(self) -> dict:
        value = asdict(self)
        value.pop("trace")
        value["trace_sha256"] = self.trace[-1]["event_hash"]
        value["status"] = "engineering_only"
        return value


def _finite_score(scorer: Scorer, state: State) -> float:
    value = float(scorer(state))
    if not math.isfinite(value):
        raise ValueError("Scorer returned a non-finite number")
    return value


def organize(
    problem: MatrixProblem,
    constraint: Constraint,
    scorer: Scorer,
    *,
    scorer_id: str,
    strategy: str = "priority",
    max_proposals: int = 256,
    max_score_calls: int = 64,
) -> MatrixSearchResult:
    """FIFO versus score-prioritized frontier; neither receives an exact oracle.

    This is a concrete constrained-search baseline, not a demonstration that
    poetry cognition is a matrix algorithm. Initialization must already be legal;
    no hidden repair or extra initializer budget is granted to either policy.
    """
    problem.validate()
    if strategy not in {"fifo", "priority"}:
        raise ValueError("strategy must be fifo or priority")
    if any(isinstance(v, bool) or not isinstance(v, Integral)
           for v in (max_proposals, max_score_calls)):
        raise ValueError("Budgets must be non-boolean integers")
    if max_proposals < 0 or max_score_calls < 1:
        raise ValueError("Invalid finite budget")
    valid, reasons = constraint(problem.initial)
    if not valid:
        raise ValueError("Initial state is infeasible: " + ",".join(reasons))
    initial_score = _finite_score(scorer, problem.initial)
    events: list[dict] = []

    def emit(kind: str, **fields: object) -> None:
        event = {"seq": len(events), "kind": kind,
                 "prev_hash": events[-1]["event_hash"] if events else "0" * 64, **fields}
        event["event_hash"] = digest(event)
        events.append(event)

    emit("initialize", problem_hash=problem.problem_hash, scorer_id=scorer_id,
         strategy=strategy, max_proposals=max_proposals, max_score_calls=max_score_calls,
         state=list(problem.initial), score=initial_score)
    fifo: deque[State] = deque([problem.initial])
    heap: list[tuple[float, int, State]] = [(-initial_score, 0, problem.initial)]
    discovered = {problem.initial}
    scores = {problem.initial: initial_score}
    parents: dict[State, State | None] = {problem.initial: None}
    best, best_score = problem.initial, initial_score
    proposals, calls, expanded, serial = 0, 1, 0, 0

    def frontier_nonempty() -> bool:
        return bool(fifo if strategy == "fifo" else heap)

    while frontier_nonempty() and proposals < max_proposals and calls < max_score_calls:
        current = fifo.popleft() if strategy == "fifo" else heapq.heappop(heap)[2]
        expanded += 1
        emit("expand", state=list(current), score=scores[current], proposals=proposals, score_calls=calls)
        for block, candidate in problem.neighbors(current):
            if proposals >= max_proposals or calls >= max_score_calls:
                break
            proposals += 1
            common = {"parent": list(current), "state": list(candidate), "block": list(block),
                      "proposals": proposals, "score_calls_before": calls}
            if candidate in discovered:
                emit("reject_duplicate", **common)
                continue
            discovered.add(candidate)
            allowed, why = constraint(candidate)
            if not allowed:
                emit("reject_constraint", reasons=list(why), **common)
                continue
            value = _finite_score(scorer, candidate)
            calls += 1
            scores[candidate] = value
            parents[candidate] = current
            serial += 1
            if strategy == "fifo":
                fifo.append(candidate)
            else:
                heapq.heappush(heap, (-value, serial, candidate))
            improved = value > best_score or (value == best_score and candidate < best)
            if improved:
                best, best_score = candidate, value
            emit("enqueue", score=value, improves_best=improved, **common)

    # A budget may interrupt expansion of the last queued node. An empty queue
    # in that case does not certify that its unvisited neighbors were exhausted.
    if proposals >= max_proposals:
        reason = "proposal_budget_exhausted"
    elif calls >= max_score_calls:
        reason = "score_budget_exhausted"
    else:
        reason = "frontier_exhausted"
    path = []
    cursor: State | None = best
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    emit("finish", best_state=list(best), best_score=best_score, proposals=proposals,
         score_calls=calls, unique_legal_states=len(scores), expanded_states=expanded,
         stop_reason=reason, best_path=[list(x) for x in path])
    return MatrixSearchResult(strategy, reason, best, best_score, proposals, calls,
                              len(scores), expanded, path, events)


def validate_trace(trace: list[dict]) -> None:
    if not trace or trace[0].get("kind") != "initialize" or trace[-1].get("kind") != "finish":
        raise ValueError("Trace lacks initialization or completion")
    previous = "0" * 64
    for i, original in enumerate(trace):
        event = dict(original)
        got_hash = event.pop("event_hash", None)
        if event.get("seq") != i or event.get("prev_hash") != previous or digest(event) != got_hash:
            raise ValueError(f"Trace integrity failure at event {i}")
        previous = got_hash


def replay(problem: MatrixProblem, constraint: Constraint, scorer: Scorer,
           *, scorer_id: str, trace: list[dict]) -> MatrixSearchResult:
    """Re-execute and compare every decision, including rejects and score values."""
    validate_trace(trace)
    header = trace[0]
    if header["problem_hash"] != problem.problem_hash or header["scorer_id"] != scorer_id:
        raise ValueError("Replay input/scorer identity mismatch")
    result = organize(problem, constraint, scorer, scorer_id=scorer_id,
                      strategy=header["strategy"], max_proposals=header["max_proposals"],
                      max_score_calls=header["max_score_calls"])
    if canonical(result.trace) != canonical(trace):
        raise ValueError("Replay diverged; inputs, scorer, constraints or code differ")
    return result


def exhaustive_oracle(problem: MatrixProblem, constraint: Constraint, scorer: Scorer,
                      *, max_states: int = 100_000) -> dict:
    """Audit only. Never pass this result to the bounded search policy."""
    problem.validate()
    if isinstance(max_states, bool) or not isinstance(max_states, Integral) or max_states < 1:
        raise ValueError("max_states must be a positive non-boolean integer")
    count = math.prod(len(d) for d in problem.domains)
    if count > max_states:
        raise ValueError("Exhaustive audit cap exceeded")
    best: State | None = None
    best_score = -math.inf
    feasible = 0
    for state in itertools.product(*problem.domains):
        allowed, _ = constraint(state)
        if not allowed:
            continue
        feasible += 1
        score = _finite_score(scorer, state)
        if score > best_score or (score == best_score and (best is None or state < best)):
            best, best_score = state, score
    return {"total_states": count, "feasible_states": feasible,
            "best_state": list(best) if best is not None else None,
            "best_score": best_score if best is not None else None,
            "role": "audit_only_not_visible_to_search"}


def write_trace(path: str | Path, trace: list[dict]) -> None:
    validate_trace(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in trace), encoding="utf-8")
