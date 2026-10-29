"""Fail-closed symbolic checks for one selected metrical specification.

This is a specification checker, not a complete historical prosody dictionary.
Reading records bind tone and rhyme together. Ambiguous or unverified readings
are routed to review instead of combining attributes from different readings.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    violations: tuple[str, ...]
    review_reasons: tuple[str, ...]
    resolved_readings: tuple[tuple[str | None, ...], ...]
    antithesis_status: str = "not_evaluated_requires_separate_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_lines(
    lines: list[list[str]],
    patterns: list[str],
    reading_ids: list[list[str | None]],
    lexicon: dict[str, list[dict[str, Any]]],
    rhyme_positions: list[tuple[int, int]] | list[list[int]],
    rhyme_group: str,
) -> ConstraintResult:
    """Check positions against explicit tone patterns and one global rhyme group.

    Lexicon reading schema: id, tone (P/Z), rhyme, source_ref, status=verified.
    `source_ref` is provenance, not proof that the source itself is correct.
    Multiple readings always require an explicit reading_id, even if one would
    fit the requested meter; meter is not used to silently disambiguate meaning.
    """
    violations: list[str] = []
    review: list[str] = []
    resolved: list[list[str | None]] = []
    if not lines or len(patterns) != len(lines) or len(reading_ids) != len(lines):
        raise ValueError("lines, patterns and reading_ids must have matching nonzero line counts")
    positions = [tuple(position) for position in rhyme_positions]
    if len(set(positions)) != len(positions):
        raise ValueError("rhyme positions must be unique")
    if positions and not rhyme_group:
        raise ValueError("a single explicit rhyme_group is required")
    for position in positions:
        if len(position) != 2:
            raise ValueError("rhyme position must contain line and column")
        row, col = position
        if not isinstance(row, int) or not isinstance(col, int) or row < 0 or col < 0:
            raise ValueError("rhyme indices must be nonnegative integers")
        if row >= len(lines) or col >= len(lines[row]):
            raise ValueError("rhyme position out of bounds")
    rhyme_set = set(positions)
    for row, (tokens, pattern, choices) in enumerate(zip(lines, patterns, reading_ids)):
        if not tokens or len(tokens) != len(pattern) or len(tokens) != len(choices):
            raise ValueError("line, pattern and reading selection lengths must match")
        if any(tone not in "PZ*" for tone in pattern):
            raise ValueError("patterns accept only P, Z and *")
        row_resolved: list[str | None] = []
        for col, (token, required_tone, choice) in enumerate(zip(tokens, pattern, choices)):
            location = f"{row}:{col}:{token}"
            records = lexicon.get(token, [])
            if not records:
                review.append(f"unknown_reading:{location}")
                row_resolved.append(None)
                continue
            ids = [record.get("id") for record in records]
            if len(ids) != len(set(ids)) or any(not value for value in ids):
                raise ValueError(f"reading ids must be unique and nonempty: {token}")
            if choice is None:
                if len(records) != 1:
                    review.append(f"ambiguous_reading:{location}")
                    row_resolved.append(None)
                    continue
                record = records[0]
            else:
                matches = [record for record in records if record["id"] == choice]
                if not matches:
                    review.append(f"unrecognized_reading_id:{location}:{choice}")
                    row_resolved.append(None)
                    continue
                record = matches[0]
            row_resolved.append(record["id"])
            if record.get("status") != "verified" or not record.get("source_ref"):
                review.append(f"unverified_reading:{location}")
                continue
            if record.get("tone") not in ("P", "Z") or not record.get("rhyme"):
                review.append(f"incomplete_reading:{location}")
                continue
            if required_tone != "*" and record["tone"] != required_tone:
                violations.append(f"tone_mismatch:{location}")
            if (row, col) in rhyme_set and record["rhyme"] != rhyme_group:
                violations.append(f"rhyme_mismatch:{location}")
        resolved.append(row_resolved)
    status = "invalid" if violations else "needs_review" if review else "valid"
    return ConstraintResult(status, tuple(violations), tuple(review), tuple(map(tuple, resolved)))
