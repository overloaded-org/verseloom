"""Real human preference import and paired source-poem cluster bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]], *, seed: int = 0, repeats: int = 10000, confidence: float = 0.95
) -> dict[str, Any]:
    """Rows contain source_poem_id, score_a and score_b in [0,1].

    Estimand: equal-weight mean of source-poem mean paired differences.
    Multiple tasks/raters from the same source remain in a single cluster.
    This interval does not model crossed rater effects or training-seed variance.
    """
    if repeats < 100 or not 0 < confidence < 1:
        raise ValueError("need >=100 repeats and confidence in (0,1)")
    if not rows:
        return {"status": "no_evaluation_data", "n_clusters": 0, "n_rows": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        identifier = row.get("source_poem_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("every paired row needs a nonempty source_poem_id")
        a, b = float(row["score_a"]), float(row["score_b"])
        if not np.isfinite([a, b]).all() or not 0 <= a <= 1 or not 0 <= b <= 1:
            raise ValueError("scores must be finite values in [0,1]")
        grouped[identifier].append(a - b)
    means = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    estimate = float(means.mean())
    if len(means) < 2:
        return {"status": "insufficient_clusters", "n_clusters": len(means), "n_rows": len(rows), "mean_difference": estimate}
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(repeats, len(means)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {
        "status": "computed",
        "n_clusters": len(means), "n_rows": len(rows),
        "mean_difference": estimate,
        "confidence": confidence,
        "interval": [float(low), float(high)],
        "seed": seed, "bootstrap_repeats": repeats,
        "estimand": "equal_source_poem_weight_mean_paired_difference",
        "method": "percentile_source_poem_cluster_bootstrap",
        "limitations": "does not account for crossed rater effects or training-seed uncertainty",
    }


_REVIEW_SCHEMA = "versematrix.blind-review.v2"


def _content_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _comparison_identity(item: dict[str, Any]) -> dict[str, Any]:
    for field in ("task_id", "source_poem_id", "upper", "text_a", "text_b"):
        if not isinstance(item.get(field), str):
            raise ValueError(f"review item requires string field: {field}")
    if not item["task_id"] or not item["source_poem_id"]:
        raise ValueError("review task and source identities must not be empty")
    return {
        "schema": _REVIEW_SCHEMA,
        "task_id": item["task_id"],
        "source_poem_id": item["source_poem_id"],
        "upper": item["upper"],
        "context_id": item.get("context_id"),
        "candidate_set_hash": item.get("candidate_set_hash"),
        # Legacy callers may omit model identities; roles are then explicitly
        # unspecified rather than pretending an actual model has been recorded.
        "model_a": item.get("model_a", "unspecified_model_a"),
        "model_b": item.get("model_b", "unspecified_model_b"),
        "candidate_a": {"id": item.get("candidate_id_a"), "text": item["text_a"]},
        "candidate_b": {"id": item.get("candidate_id_b"), "text": item["text_b"]},
        # A caller label is a namespace, not a substitute for content binding.
        "caller_comparison_id": item.get("comparison_id"),
    }


def write_blind_review(
    items: list[dict[str, Any]], output_csv: Path, private_key: Path, seed: int = 0
) -> None:
    """Create an UNFILLED review sheet and a separate analyst-only orientation key.

    Items need task_id, source_poem_id, upper, text_a, text_b. Optional model_a,
    model_b, context_id, candidate_set_hash and candidate_id_a/candidate_id_b
    strengthen provenance. Optional comparison_id is a caller namespace; the
    emitted comparison_id still hashes all identities and candidate contents.
    Model names never enter the reviewer sheet. Keep the key away from raters.
    A key is bound to exact displayed content, including whitespace.
    """
    fields = ["review_id", "comparison_id", "upper", "left", "right", "rater_id", "preference", "comment"]
    rng = np.random.default_rng(seed)
    key_rows = []
    display_rows = []
    identities = [_comparison_identity(item) for item in items]
    if len({identity["task_id"] for identity in identities}) != len(identities):
        raise ValueError("review task ids must be unique")
    for identity in identities:
        comparison_id = "cmp_" + _content_hash(identity)
        left_is_a = bool(rng.integers(2))
        text_a, text_b = identity["candidate_a"]["text"], identity["candidate_b"]["text"]
        left, right = (text_a, text_b) if left_is_a else (text_b, text_a)
        display = {"upper": identity["upper"], "left": left, "right": right}
        display_sha256 = _content_hash(display)
        review_id = _content_hash({"seed": seed, "comparison_id": comparison_id,
                                   "display_sha256": display_sha256})[:24]
        display_rows.append({"review_id": review_id, "comparison_id": comparison_id,
                             **display, "rater_id": "", "preference": "", "comment": ""})
        key_rows.append({"review_id": review_id, "comparison_id": comparison_id,
                         "comparison_identity": identity, "display_sha256": display_sha256,
                         "task_id": identity["task_id"], "source_poem_id": identity["source_poem_id"],
                         "left_is_a": left_is_a, "identical_outputs": text_a == text_b})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    private_key.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(display_rows)
    private_key.write_text(json.dumps({"schema_version": _REVIEW_SCHEMA,
                                      "status": "unfilled_review_template", "seed": seed,
                                      "allowed_preferences": ["left", "right", "tie", "skip"],
                                      "items": key_rows}, ensure_ascii=False, indent=2) + "\n")


def import_blind_preferences(review_csv: Path, private_key: Path) -> list[dict[str, Any]]:
    key = json.loads(private_key.read_text())
    if key.get("schema_version") != _REVIEW_SCHEMA:
        raise ValueError("unsupported or unbound review key; regenerate the review sheet and key together")
    mapping = {}
    for item in key["items"]:
        identity = item.get("comparison_identity")
        if not isinstance(identity, dict) or item.get("comparison_id") != "cmp_" + _content_hash(identity):
            raise ValueError("review key comparison identity is inconsistent")
        if not isinstance(item.get("left_is_a"), bool):
            raise ValueError("review key orientation must be a boolean")
        text_a, text_b = identity["candidate_a"]["text"], identity["candidate_b"]["text"]
        left, right = (text_a, text_b) if item["left_is_a"] else (text_b, text_a)
        display_hash = _content_hash({"upper": identity["upper"], "left": left, "right": right})
        expected_review_id = _content_hash({"seed": key["seed"], "comparison_id": item["comparison_id"],
                                           "display_sha256": display_hash})[:24]
        if item.get("display_sha256") != display_hash or item.get("review_id") != expected_review_id:
            raise ValueError("review key display content or orientation is inconsistent")
        if item.get("task_id") != identity["task_id"] or item.get("source_poem_id") != identity["source_poem_id"]:
            raise ValueError("review key source identity is inconsistent")
        if item.get("identical_outputs") != (text_a == text_b):
            raise ValueError("review key identical-output flag is inconsistent")
        if item["review_id"] in mapping:
            raise ValueError("duplicate review identity in key")
        mapping[item["review_id"]] = item
    output = []
    seen: set[tuple[str, str]] = set()
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"review_id", "comparison_id", "upper", "left", "right", "rater_id", "preference"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("review sheet lacks bound identity or display columns")
        for row in reader:
            if row.get("review_id") not in mapping:
                raise ValueError("unknown review id or mismatched review key")
            item = mapping[row["review_id"]]
            if row.get("comparison_id") != item["comparison_id"]:
                raise ValueError("review comparison identity does not match key")
            display = {field: row.get(field) for field in ("upper", "left", "right")}
            if _content_hash(display) != item["display_sha256"]:
                raise ValueError("displayed content does not match review key")
            preference = row.get("preference", "").strip()
            if not preference or preference == "skip":
                continue
            if preference not in {"left", "right", "tie"}:
                raise ValueError("preference must be left, right, tie or skip")
            rater = row.get("rater_id", "").strip()
            if not rater:
                raise ValueError("completed reviews require a rater_id")
            pair = (row["review_id"], rater)
            if pair in seen:
                raise ValueError("duplicate rating for the same review and rater")
            seen.add(pair)
            if item["identical_outputs"] and preference != "tie":
                raise ValueError("identical displayed outputs must be a tie or skip")
            score_a = 0.5 if preference == "tie" else float((preference == "left") == item["left_is_a"])
            output.append({"source_poem_id": item["source_poem_id"], "task_id": item["task_id"],
                           "comparison_id": item["comparison_id"],
                           "rater_id": rater, "score_a": score_a, "score_b": 1 - score_a})
    return output
