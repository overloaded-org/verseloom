"""Reproducible 2x2 ENGINEERING check; all scores and tokens are synthetic.

Usage: PYTHONPATH=src python scripts/run_matrix_demo.py --out results/matrix
The four cells do NOT measure human poetry quality or a scientific interaction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from wordconv_lab.convolution import LocalConvScorer
from wordconv_lab.matrix import MatrixProblem, exhaustive_oracle, organize, replay, write_trace


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # A full 8x5 grid, four ordinary editable positions and all four even-line
    # endings. Only eight domains vary, so an independent 256-state oracle fits.
    rhyme_positions = (9, 19, 29, 39)
    editable = (10, 11, 15, 16) + rhyme_positions
    domains = tuple(("A", "B") if p in editable else ("A",) for p in range(40))
    problem = MatrixProblem("abstract-8x5-four-rhyme-positions-v1", 8, 5,
                            domains, ("A",) * 40, (rhyme_positions,))

    def constraint(state):
        valid = len({state[p] for p in rhyme_positions}) == 1
        return valid, () if valid else ("synthetic_shared_rhyme_group",)

    def features(state):
        x = np.zeros((2, 5, 4))
        for p, token in enumerate(state[10:20]):
            row, col = divmod(p, 5)
            x[row, col] = [float(token == "A"), float(token == "B"), row, col / 4]
        return x

    unary_weights = np.array([0.1, 0.5, -0.2, 0.7, 0.3, -0.4, -0.15, 0.25])

    def unary(state):
        return float(np.dot([state[p] == "B" for p in editable], unary_weights))

    conv = LocalConvScorer(feature_dim=4, channels=4, kernel_width=3, seed=17)
    scorers = {"unary_fixture": unary, "conv_random_untrained": lambda state: conv.score(features(state))}
    runs = []
    for name, scorer in scorers.items():
        oracle = exhaustive_oracle(problem, constraint, scorer)
        for strategy in ("fifo", "priority"):
            for label, proposals, calls in (("bounded", 24, 12), ("exhaustive_check", 1000, 100)):
                start = time.perf_counter()
                result = organize(problem, constraint, scorer, scorer_id=name, strategy=strategy,
                                  max_proposals=proposals, max_score_calls=calls)
                elapsed = time.perf_counter() - start
                replay(problem, constraint, scorer, scorer_id=name, trace=result.trace)
                trace_name = f"{name}_{strategy}_{label}.jsonl"
                write_trace(output / trace_name, result.trace)
                item = result.summary()
                item.update({"scorer": name, "budget_label": label, "max_proposals": proposals,
                             "max_score_calls": calls, "search_seconds": elapsed, "replay_passed": True,
                             "oracle": oracle, "oracle_gap": oracle["best_score"] - result.best_score,
                             "trace_file": trace_name})
                if label == "exhaustive_check":
                    assert abs(item["oracle_gap"]) < 1e-12, "exhaustive correctness failure"
                    assert item["unique_legal_states"] == oracle["feasible_states"]
                runs.append(item)
    report = {
        "project": "VerseMatrix Lab", "version": "0.1.0", "status": "engineering_only",
        "implemented": True, "executed": True, "real_data_ready": False,
        "trained": False, "human_evaluated": False,
        "fixture": "Full 8x5 grid, 8 editable A/B positions, four shared synthetic rhyme endings; no poetry labels",
        "shape": [8, 5], "editable_positions": list(editable),
        "rhyme_positions": list(rhyme_positions),
        "comparison": "Identical state graph and feasibility; FIFO vs score-priority frontier",
        "scorer_warning": "UNTRAINED random conv; never compare raw scores between scorer families",
        "interaction_status": "NOT_ESTIMATED_requires_common_external_evaluation",
        "problem_hash": problem.problem_hash, "conv_seed": 17,
        "conv_parameters": conv.parameter_count(), "runs": runs,
    }
    (output / "matrix_engineering_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/matrix"))
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                      "exact_checks_passed": sum(r["budget_label"] == "exhaustive_check" and
                                                  abs(r["oracle_gap"]) < 1e-12 for r in result["runs"]),
                      "human_evaluated": False}, ensure_ascii=False))
