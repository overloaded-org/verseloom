"""Candidate identity, source-poem splits and provenance checks."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_set_hash(task: dict[str, Any]) -> str:
    candidates = task.get("candidates", [])
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    return canonical_hash({
        "task_id": task["task_id"],
        "upper": task["upper"],
        "upper_readings": task["upper_readings"],
        "patterns": task["patterns"],
        "rhyme_positions": task["rhyme_positions"],
        "rhyme_group": task["rhyme_group"],
        "candidates": sorted(candidates, key=lambda candidate: candidate["candidate_id"]),
    })


def assert_same_candidates(run_a: dict[str, Any], run_b: dict[str, Any]) -> None:
    if run_a.get("candidate_set_hash") != run_b.get("candidate_set_hash"):
        raise ValueError("candidate sets or their task context differ")
    if not run_a.get("candidate_set_hash"):
        raise ValueError("candidate set hash missing")
    if "constraint_lexicon_hash" in run_a or "constraint_lexicon_hash" in run_b:
        if (not run_a.get("constraint_lexicon_hash")
                or run_a.get("constraint_lexicon_hash") != run_b.get("constraint_lexicon_hash")):
            raise ValueError("constraint dictionaries differ or lack a fingerprint")


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whole-poem and externally computed near-duplicate cluster isolation.

    This does not itself discover near duplicates. The upstream deduplicator must
    provide stable duplicate_cluster values. Candidate pool provenance must be
    train-only; observed reference answers are excluded from the primary pool.
    """
    errors: list[str] = []
    source_splits: dict[str, set[str]] = {}
    cluster_splits: dict[str, set[str]] = {}
    task_ids: set[str] = set()
    allowed_splits = {"train", "dev", "test"}
    for index, row in enumerate(records):
        missing = [key for key in ("task_id", "source_poem_id", "duplicate_cluster", "split") if not row.get(key)]
        if missing:
            errors.append(f"row:{index}:missing:{','.join(missing)}")
            continue
        if row["task_id"] in task_ids:
            errors.append(f"duplicate_task_id:{row['task_id']}")
        task_ids.add(row["task_id"])
        if row["split"] not in allowed_splits:
            errors.append(f"invalid_split:{row['task_id']}")
        source_splits.setdefault(row["source_poem_id"], set()).add(row["split"])
        cluster_splits.setdefault(row["duplicate_cluster"], set()).add(row["split"])
    for label, mapping in (("source", source_splits), ("cluster", cluster_splits)):
        for identifier, splits in mapping.items():
            if len(splits) > 1:
                errors.append(f"cross_split_{label}:{identifier}:{','.join(sorted(splits))}")
    for row in records:
        for candidate in row.get("candidates", []):
            origin = candidate.get("origin", {})
            task_id = row.get("task_id", "missing")
            candidate_id = candidate.get("candidate_id", "missing")
            if origin.get("pool_split") != "train":
                errors.append(f"non_train_candidate_pool:{task_id}:{candidate_id}")
            source_ids = origin.get("source_poem_ids", [])
            if not source_ids:
                errors.append(f"missing_candidate_sources:{task_id}:{candidate_id}")
            for source_id in source_ids:
                if source_splits.get(source_id) != {"train"}:
                    errors.append(f"unverified_train_candidate_source:{task_id}:{candidate_id}:{source_id}")
            if origin.get("uses_heldout_reference", False):
                errors.append(f"heldout_reference_in_primary_pool:{task_id}:{candidate_id}")
    return {
        "status": "passed" if not errors else "failed",
        "record_count": len(records),
        "errors": errors,
        "limitations": [
            "duplicate_cluster membership must be supplied by an independently audited deduplicator",
            "source version, poem authenticity, phonological annotation and licensing require separate review",
        ],
    }
