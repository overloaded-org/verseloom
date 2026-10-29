"""Verify a complete eight-line finite-domain solver against a direct oracle.

All tokens/readings and model weights here are synthetic. The original untrained
LocalConvScorer scores every complete input combination; independent tone/rhyme
checks then identify feasible configurations. This tests mathematical plumbing,
not historical reading accuracy, learned semantics or literary quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np

from wordconv_lab.convolution import LocalConvScorer
from wordconv_lab.prosody import generate_template
from wordconv_lab.verse_solver import ReadingCandidate, solve_verse


CONFIG = {
    "schema": "versematrix.full-verse-exact.synthetic.v0.2",
    "lengths": [5, 7],
    "scorer_seeds": {"5": 117, "7": 223},
    "start_second_tones": {"5": "P", "7": "Z"},
    "first_rhymed": [False, True],
    "rhyme_groups": ["R1", "R2"],
    "configurations_per_case": 4,
    "profile": "strict_basic_templates_only",
    "model": {"feature_dim": 6, "channels": 4, "kernel_width": 3, "trained": False},
    "features": "[token==A, token==B, token==C, global_row/7, column/(length-1), 1]",
    "free_same_tone_cells": [[2, 0], [6, "last"]],
    "generator": "base template in every cell; alternate first-line tones; two rhyme groups at all possible rhyme endings; two same-tone binary cells",
    "absolute_tolerance": 1e-12,
    "oracle": "enumerate all raw full-matrix candidates, score all four original couplets, independently check every template/rhyme configuration",
    "quality_claim": "synthetic untrained mechanism verification only",
}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_case(length: int):
    templates = tuple(generate_template(length, CONFIG["start_second_tones"][str(length)], rhymed)
                      for rhymed in CONFIG["first_rhymed"])
    candidates = [[(ReadingCandidate("A", tone, "R1" if tone == "P" else "Z0",
                                    f"synthetic:{length}:{row}:{col}:A", True),)
                   for col, tone in enumerate(line)] for row, line in enumerate(templates[0].core)]
    for col, (original, alternate) in enumerate(zip(templates[0].core[0], templates[1].core[0])):
        if original != alternate:
            base = candidates[0][col][0]
            candidates[0][col] += (replace(base, token="B", tone=alternate,
                                          rhyme="R1" if alternate == "P" else "Z0",
                                          reading_id=base.reading_id[:-1] + "B"),)
    # The extra first-line ending must also cover the second allowed rhyme group.
    first_ping = next(c for c in candidates[0][-1] if c.tone == "P")
    candidates[0][-1] += (replace(first_ping, token="C", rhyme="R2",
                                  reading_id=first_ping.reading_id[:-1] + "C"),)
    for row in (1, 3, 5, 7):
        base = candidates[row][-1][0]
        candidates[row][-1] += (replace(base, token="B", rhyme="R2",
                                        reading_id=base.reading_id[:-1] + "B"),)
    for row, col in ((2, 0), (6, length - 1)):
        base = candidates[row][col][0]
        candidates[row][col] += (replace(base, token="B", reading_id=base.reading_id[:-1] + "B"),)
    return tuple(tuple(row) for row in candidates), templates, tuple(CONFIG["rhyme_groups"])


def fixed_feature(length: int):
    def feature(row: int, col: int, token: str) -> np.ndarray:
        return np.array([float(token == "A"), float(token == "B"), float(token == "C"),
                         row / 7, col / (length - 1), 1.0])
    return feature


def legal(grid, template, rhyme_group) -> bool:
    """Independent complete-grid predicate; does not call solver filtering."""
    return (len(grid) == 8 and all(len(row) == template.length for row in grid)
            and all(grid[row][col].tone == template.core[row][col]
                    for row in range(8) for col in range(template.length))
            and all(grid[row][-1].rhyme == rhyme_group for row in template.rhyme_rows))


def original_score(grid, model, feature) -> float:
    # No convolution_factor, path_score or DP is used in this oracle.
    values = [model.score(np.array([[feature(upper + local, col, candidate.token)
                                     for col, candidate in enumerate(grid[upper + local])]
                                    for local in (0, 1)])) for upper in (0, 2, 4, 6)]
    score = sum(values)
    if not math.isfinite(score):
        raise AssertionError("non-finite original model score")
    return score


def check_case(candidates, templates, groups, length: int) -> tuple[dict, dict]:
    model = LocalConvScorer(**{key: value for key, value in CONFIG["model"].items() if key != "trained"},
                            seed=CONFIG["scorer_seeds"][str(length)])
    feature = fixed_feature(length)
    result = solve_verse(candidates, templates, groups, model, feature)
    if result.status != "optimal":
        raise AssertionError(f"synthetic complete case unexpectedly returned {result.status}")
    branch_oracles = {(template.template_id, group): {"count": 0, "score": -math.inf}
                      for template, group in itertools.product(templates, groups)}
    enumerated = feasible = 0
    oracle_score = -math.inf
    for flat in itertools.product(*(domain for row in candidates for domain in row)):
        grid = tuple(tuple(flat[row * length:(row + 1) * length]) for row in range(8))
        enumerated += 1
        score = original_score(grid, model, feature)
        matches = [(template.template_id, group) for template, group in itertools.product(templates, groups)
                   if legal(grid, template, group)]
        if len(matches) > 1:
            raise AssertionError("the declared canonical configurations should be disjoint")
        if matches:
            feasible += 1
            oracle_score = max(oracle_score, score)
            branch = branch_oracles[matches[0]]
            branch["count"] += 1
            branch["score"] = max(branch["score"], score)
    tolerance = CONFIG["absolute_tolerance"]
    selected_template = next(t for t in templates if t.template_id == result.selected_template)
    if not legal(result.selected, selected_template, result.selected_rhyme):
        raise AssertionError("returned complete matrix violates the independent legality predicate")
    selected_original = original_score(result.selected, model, feature)
    optimality_error = abs(oracle_score - selected_original)
    factorization_error = abs(result.score - selected_original)
    branch_checks = []
    for branch in result.configurations:
        oracle = branch_oracles[(branch.template_id, branch.rhyme_group)]
        error = abs(branch.score - oracle["score"])
        if branch.full_combinations != oracle["count"] or error > tolerance:
            raise AssertionError("configuration counts or exact maxima disagree")
        branch_checks.append({"template_id": branch.template_id, "rhyme_group": branch.rhyme_group,
                              "feasible_combinations": oracle["count"], "oracle_score": oracle["score"],
                              "solver_score": branch.score, "optimality_abs_error": error})
    if (enumerated != result.full_combinations or feasible != result.feasible_combinations
            or max(optimality_error, factorization_error) > tolerance):
        raise AssertionError("whole-verse original-model oracle verification failed")
    summary = {
        "length": length, "rows": 8, "status": "verified_exact_in_declared_scope",
        "full_combinations": enumerated, "feasible_combinations": feasible,
        "outer_configurations": len(branch_checks), "profile": "strict_basic_templates_only",
        "oracle_score": oracle_score, "solver_score": result.score,
        "selected_original_score": selected_original, "optimality_abs_error": optimality_error,
        "factorization_abs_error": factorization_error,
        "selected_template": result.selected_template, "selected_rhyme": result.selected_rhyme,
        "selected_lines": list(result.lines),
        "selected_tones": ["".join(c.tone for c in row) for row in result.selected],
        "selected_rhyme_endings": [{"row": row, "token": result.selected[row][-1].token,
                                    "rhyme": result.selected[row][-1].rhyme}
                                   for row in selected_template.rhyme_rows],
        "oracle_original_model_calls": 4 * enumerated,
        "selected_solution_check_model_calls": 4,
        "dp_local_factor_calls": sum(c.factor_calls for b in result.configurations for c in b.couplets),
        "call_units_warning": "local factor calls and full-couplet model calls are different units; no throughput ratio is claimed",
        "configuration_checks": branch_checks,
    }
    return summary, result.to_dict()


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite a full-verse evidence directory")
    output.mkdir(parents=True)
    sources = [root / "src/wordconv_lab/verse_solver.py", root / "src/wordconv_lab/factor_dp.py",
               root / "src/wordconv_lab/prosody.py", root / "src/wordconv_lab/convolution.py",
               Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(root)): _sha(path) for path in sources}
    manifest = {
        "status": "frozen_before_case_generation_and_execution",
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
        "config_sha256": hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "registration_note": "local before-execution manifest; not an external preregistration",
    }
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    cases = [(length, *synthetic_case(length)) for length in CONFIG["lengths"]]
    input_record = [{"length": length, "synthetic": True,
                     "templates": [t.to_dict() for t in templates], "rhyme_groups": groups,
                     "candidates": [[[asdict(c) for c in domain] for domain in row] for row in candidates]}
                    for length, candidates, templates, groups in cases]
    input_path = output / "cases.json"
    input_path.write_text(_json(input_record), encoding="utf-8")
    summaries, solver_results = [], []
    for length, candidates, templates, groups in cases:
        summary, detail = check_case(candidates, templates, groups, length)
        summaries.append(summary)
        solver_results.append(detail)
    if source_hashes != {str(path.relative_to(root)): _sha(path) for path in sources}:
        raise AssertionError("source identity changed during execution")
    report = {
        "schema": CONFIG["schema"], "status": "all_full_verse_oracle_checks_passed",
        "scorer_trained": False, "synthetic": True, "human_evaluated": False,
        "profile": "strict_basic_templates_only", "configurations_per_case": 4,
        "config_sha256": manifest["config_sha256"], "cases_sha256": _sha(input_path),
        "absolute_tolerance": CONFIG["absolute_tolerance"], "cases": summaries,
        "total_raw_full_matrices_checked": sum(c["full_combinations"] for c in summaries),
        "total_feasible_full_matrices": sum(c["feasible_combinations"] for c in summaries),
        "total_original_couplet_oracle_calls": sum(c["oracle_original_model_calls"] for c in summaries),
        "limits": [
            "Exactly the complete supplied 8x5 and 8x7 finite candidate spaces and four declared outer configurations per case are checked.",
            "The outer rhyme groups are a caller-supplied finite set, not a search of all historical rhyme systems.",
            "Only strict canonical basic templates are optimized; this is not the common_prefix audit profile or a complete account of aojiu.",
            "The objective is the sum of four independent fixed-feature local couplet scores; arbitrary whole-poem semantics or repetition constraints invalidate this decomposition.",
            "Synthetic A/B/C symbols with asserted synthetic readings and untrained random weights do not demonstrate Chinese reading accuracy or literary quality.",
            "Floating-point equivalence is checked at an absolute tolerance of 1e-12; factor-call and whole-model-call counts are not throughput comparisons.",
        ],
    }
    (output / "solver_results.json").write_text(_json(solver_results), encoding="utf-8")
    (output / "report.json").write_text(_json(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/full_verse_v02"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(root, args.out)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "full_matrices": report["total_raw_full_matrices_checked"],
                      "feasible_matrices": report["total_feasible_full_matrices"]}))
