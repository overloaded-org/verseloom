"""Fixed-candidate engineering reranking; no training or quality claims."""
from __future__ import annotations

from typing import Any

import numpy as np

from .audit import candidate_set_hash, canonical_hash
from .constraints import check_lines
from .convolution import LocalConvScorer


def feature_matrix(
    lines: list[list[str]], reading_ids: tuple[tuple[str | None, ...], ...],
    lexicon: dict[str, list[dict[str, Any]]],
) -> np.ndarray:
    """Deterministic synthetic token identity + tone/rhyme + position features.

    There is deliberately no claim that one-hot identity encodes poetic meaning.
    A learned contextual or span representation is a future experiment component.
    """
    if len(lines) != 2 or not lines[0] or len(lines[0]) != len(lines[1]):
        raise ValueError("features require exactly two equal, nonempty lines")
    tokens = sorted(lexicon)
    token_index = {token: index for index, token in enumerate(tokens)}
    rhymes = sorted({record["rhyme"] for records in lexicon.values() for record in records})
    rhyme_index = {rhyme: index for index, rhyme in enumerate(rhymes)}
    dimension = len(tokens) + 2 + len(rhymes) + 2
    array = np.zeros((2, len(lines[0]), dimension))
    for row, line in enumerate(lines):
        for column, token in enumerate(line):
            if token not in token_index:
                raise ValueError("unknown token cannot be encoded")
            selected = reading_ids[row][column]
            matches = [record for record in lexicon[token] if record["id"] == selected]
            if len(matches) != 1:
                raise ValueError("feature encoding needs an explicitly resolved reading")
            record = matches[0]
            array[row, column, token_index[token]] = 1
            array[row, column, len(tokens) + (0 if record["tone"] == "P" else 1)] = 1
            array[row, column, len(tokens) + 2 + rhyme_index[record["rhyme"]]] = 1
            array[row, column, -2] = column / max(1, len(line) - 1)
            array[row, column, -1] = row
    return array


def run_reranking(
    task: dict[str, Any], lexicon: dict[str, list[dict[str, Any]]],
    *, model: str = "local_conv", seed: int = 1729,
) -> dict[str, Any]:
    if model not in {"local_conv", "pointwise"}:
        raise ValueError("model must be local_conv or pointwise")
    scored = []
    scorer = None
    for candidate in task["candidates"]:
        lines = [task["upper"], candidate["tokens"]]
        reading_ids = [task["upper_readings"], candidate["reading_ids"]]
        checked = check_lines(lines, task["patterns"], reading_ids, lexicon,
                              task["rhyme_positions"], task["rhyme_group"])
        item = {"candidate_id": candidate["candidate_id"], "constraint": checked.to_dict(),
                "score": None, "score_parts": None}
        if checked.status == "valid":
            x = feature_matrix(lines, checked.resolved_readings, lexicon)
            if model == "local_conv":
                if scorer is None:
                    scorer = LocalConvScorer(x.shape[2], seed=seed)
                item["score_parts"] = scorer.score_parts(x)
                item["score"] = item["score_parts"]["total"]
            else:
                # Width-one random linear reference. No parameter-budget or
                # scientific baseline equivalence is claimed for this demo.
                weights = np.random.default_rng(seed).normal(0, 1 / np.sqrt(x.shape[2]), x.shape[2])
                item["score"] = float(np.einsum("lpf,f->lp", x, weights).mean())
                item["score_parts"] = {"pointwise": item["score"]}
        scored.append(item)
    valid = [candidate for candidate in scored if candidate["score"] is not None]
    ranked = sorted(valid, key=lambda candidate: (-candidate["score"], candidate["candidate_id"]))
    return {
        "status": "engineering_only",
        "trained": False,
        "model": model,
        "seed": seed,
        "candidate_set_hash": candidate_set_hash(task),
        "constraint_lexicon_hash": canonical_hash(lexicon),
        "candidate_count": len(scored),
        "valid_count": len(valid),
        "decision": "selected" if ranked else "abstain",
        "selected_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "abstain_reason": None if ranked else "no_verified_legal_candidate",
        "ranking": [candidate["candidate_id"] for candidate in ranked],
        "candidates": scored,
        "parameter_count": scorer.parameter_count() if scorer else None,
        "feature_semantics": "synthetic token identity, selected tone/rhyme, position and line role",
        "limitations": [
            "random untrained weights; scores and rankings imply no poetry-quality advantage",
            "no learned semantic representation or word-span representation",
            "pointwise arm is an engineering reference, not the preregistered strong baseline",
            "antithesis and semantic quality are not decided by the legality filter",
        ],
    }
