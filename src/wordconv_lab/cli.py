from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import assert_same_candidates, audit_records, canonical_hash
from .evaluation import import_blind_preferences, paired_cluster_bootstrap, write_blind_review
from .experiment import run_reranking


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _default_fixture() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "synthetic_couplet.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VerseMatrix engineering-only validation harness")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run an untrained synthetic candidate-reranking demonstration")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--fixture", type=Path, default=_default_fixture())
    demo.add_argument("--seed", type=int, default=1729)
    audit = sub.add_parser("audit", help="Audit JSONL split and primary candidate provenance")
    audit.add_argument("records", type=Path)
    evaluate = sub.add_parser("evaluate", help="Analyze completed real blind reviews; empty forms yield no data")
    evaluate.add_argument("--reviews", type=Path, required=True)
    evaluate.add_argument("--key", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.command == "audit":
        records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
        report = audit_records(records)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 2
    if args.command == "evaluate":
        rows = import_blind_preferences(args.reviews, args.key)
        report = paired_cluster_bootstrap(rows, seed=args.seed)
        # This mapping applies ONLY to complementary preference scores imported
        # above, not arbitrary paired metrics accepted by the bootstrap function.
        if "mean_difference" in report:
            report["preference_score"] = (1 + report["mean_difference"]) / 2
            report["preference_score_estimand"] = "equal_source_poem_weight_A_win_plus_half_tie_rate"
            report["delta_to_preference"] = "delta = 2 * preference_score - 1"
            if "interval" in report:
                report["preference_interval"] = [(1 + value) / 2 for value in report["interval"]]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    fixture = json.loads(args.fixture.read_text())
    if fixture.get("synthetic") is not True:
        raise ValueError("demo accepts only explicitly synthetic fixtures")
    task, lexicon = fixture["task"], fixture["lexicon"]
    args.out.mkdir(parents=True, exist_ok=True)
    conv = run_reranking(task, lexicon, model="local_conv", seed=args.seed)
    pointwise = run_reranking(task, lexicon, model="pointwise", seed=args.seed)
    assert_same_candidates(conv, pointwise)
    _write(args.out / "local_conv.json", conv)
    _write(args.out / "pointwise.json", pointwise)
    # Deliberately empty: synthetic examples must not become invented human ratings.
    write_blind_review([], args.out / "review_template.csv", args.out / "analyst_key.json", seed=args.seed)
    report = {
        "status": "engineering_only", "synthetic": True,
        "fixture_sha256": canonical_hash(fixture),
        "candidate_set_hash": conv["candidate_set_hash"],
        "shared_candidate_set_verified": True,
        "seed": args.seed,
        "trained": False,
        "candidate_count": conv["candidate_count"],
        "verified_legal_candidates": conv["valid_count"],
        "review_rows_created": 0,
        "quality_evidence": "not_evaluated",
        "statistical_evidence": paired_cluster_bootstrap([], seed=args.seed),
        "next_gate": "audited real corpus, frozen model protocol, and independent blind evaluation",
    }
    _write(args.out / "engineering_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
