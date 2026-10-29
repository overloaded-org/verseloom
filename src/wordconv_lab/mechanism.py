"""Frozen synthetic mechanism comparison, not poetry-quality evaluation.

The four task families and decision rule are defined before the first run.
There is no training, parameter fitting, dev feedback, or adaptive test selection.
The independent exact audit is never passed into either search policy.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .matrix import MatrixProblem, State, canonical, digest, organize, replay


FAMILIES = ("separable", "local_interaction", "coupled_constraint", "local_trap")
DEFAULT_CONFIG = {
    "schema": "versematrix.synthetic-mechanism.v0.2",
    "master_seed": 2026102902,
    "bootstrap_seed": 44217,
    "bootstrap_repeats": 10000,
    "split_sizes": {"dev": 60, "test": 240},
    "families": list(FAMILIES),
    "grid_columns": [5, 7],
    "variable_counts": [2, 3, 4],
    "domain_sizes": [3, 4, 5],
    "proposal_budgets": [8, 16, 32, 64],
    "primary_budget": 32,
    "score_budget_rule": "proposal_budget_plus_one_including_initial_score",
    "policies": ["fifo", "priority"],
    "relative_regret_reduction_gate": 0.20,
    "family_absolute_regret_harm_gate": 0.05,
    "confidence": 0.95,
    "replay_all_runs": True,
    "interpretation": "synthetic mechanism only; no poetry-quality or convolution claim",
}


def _seed(master: int, split: str, family: str, index: int) -> int:
    value = hashlib.sha256(f"{master}:{split}:{family}:{index}".encode()).digest()
    return int.from_bytes(value[:8], "little")


def _validate_config(config: dict) -> None:
    if config["families"] != list(FAMILIES) or config["policies"] != ["fifo", "priority"]:
        raise ValueError("This version fixes the family list and two policies")
    if any(type(v) is not int or v <= 0 or v % 4 for v in config["split_sizes"].values()):
        raise ValueError("Every split size must be a positive multiple of four")
    if set(config["split_sizes"]) != {"dev", "test"}:
        raise ValueError("Both dev and test are required")
    if not config["proposal_budgets"] or any(type(v) is not int or v <= 0 for v in config["proposal_budgets"]):
        raise ValueError("Budgets must be positive non-boolean integers")
    if config["primary_budget"] not in config["proposal_budgets"]:
        raise ValueError("Primary budget must be one of the frozen budgets")
    if config["variable_counts"] != [2, 3, 4] or config["domain_sizes"] != [3, 4, 5]:
        raise ValueError("This version fixes 2-4 variables and 3-5 values")
    if config["grid_columns"] != [5, 7]:
        raise ValueError("Only full 8x5/8x7 backgrounds are supported here")


def generate_cases(config: dict) -> list[dict]:
    """Generate inputs only: neither a solver nor an oracle is called here."""
    _validate_config(config)
    cases = []
    for split in ("dev", "test"):
        per_family = config["split_sizes"][split] // len(FAMILIES)
        for family_index, family in enumerate(FAMILIES):
            for index in range(per_family):
                seed = _seed(config["master_seed"], split, family, index)
                rng = np.random.default_rng(seed)
                n = int(rng.choice(config["variable_counts"]))
                if family == "coupled_constraint":
                    n = max(n, 3)
                k = int(rng.choice(config["domain_sizes"]))
                columns = config["grid_columns"][(index + family_index) % 2]
                if family == "coupled_constraint":
                    # Two distant line endings are tied; all other variables
                    # remain independently editable. This is synthetic equality.
                    positions = [2 * columns - 1, 4 * columns - 1, 2 * columns + 1, 2 * columns + 2][:n]
                else:
                    positions = [2 * columns, 2 * columns + 1, 3 * columns, 3 * columns + 1][:n]
                initial = rng.integers(0, k, size=n).tolist()
                if family == "coupled_constraint":
                    initial[1] = initial[0]
                if family == "local_trap":
                    initial = [0] * n
                unary = rng.normal(0, 1, size=(n, k)).tolist()
                pair = rng.normal(0, 1, size=(n - 1, k, k)).tolist()
                target = rng.integers(1, k, size=n).tolist()
                case = {
                    "problem_id": f"{split}-{family}-{index:03d}",
                    "split": split, "family": family, "seed": seed,
                    "rows": 8, "columns": columns, "n_variables": n,
                    "domain_size": k, "positions": positions,
                    "initial_values": initial,
                    "unary": unary, "pair": pair, "target": target,
                    "synthetic": True,
                    "construction": {
                        "separable": "mean independent random unary terms",
                        "local_interaction": "0.35 mean unary + 0.65 mean adjacent random pair potentials",
                        "coupled_constraint": "same interaction score, variables 0 and 1 must match; common joint move",
                        "local_trap": "decoy high-score region, negative partial-target valley, planted full-target maximum",
                    }[family],
                }
                case["case_sha256"] = digest(case)
                cases.append(case)
    if len({case["case_sha256"] for case in cases}) != len(cases):
        raise ValueError("Generated duplicate task identities")
    return cases


def compile_case(case: dict):
    """Compile public task payload into common state graph, rule and scorer."""
    body = {key: value for key, value in case.items() if key != "case_sha256"}
    if digest(body) != case["case_sha256"]:
        raise ValueError("Case payload hash mismatch")
    n, k = case["n_variables"], case["domain_size"]
    positions = tuple(case["positions"])
    size = case["rows"] * case["columns"]
    domains = [(".",)] * size
    initial = ["."] * size
    for variable, position in enumerate(positions):
        domains[position] = tuple(str(value) for value in range(k))
        initial[position] = str(case["initial_values"][variable])
    blocks = (tuple(sorted(positions[:2])),) if case["family"] == "coupled_constraint" else ()
    problem = MatrixProblem(case["problem_id"], case["rows"], case["columns"],
                            tuple(domains), tuple(initial), blocks)
    problem.validate()
    unary, pairs = np.asarray(case["unary"]), np.asarray(case["pair"])
    target = tuple(case["target"])
    family = case["family"]

    def values(state: State) -> tuple[int, ...]:
        return tuple(int(state[position]) for position in positions)

    def constraint(state: State):
        v = values(state)
        valid = family != "coupled_constraint" or v[0] == v[1]
        return valid, () if valid else ("synthetic_two_position_equality",)

    def scorer(state: State) -> float:
        v = values(state)
        u = float(sum(unary[i, value] for i, value in enumerate(v)) / n)
        if family == "separable":
            return u
        if family in {"local_interaction", "coupled_constraint"}:
            p = float(sum(pairs[i, v[i], v[i + 1]] for i in range(n - 1)) / (n - 1))
            return 0.35 * u + 0.65 * p
        matched = sum(a == b for a, b in zip(v, target))
        if matched == n:
            return 2.0
        if matched:
            return -1.0 + 0.1 * matched + 0.005 * math.tanh(u)
        changed = sum(value != 0 for value in v)
        return 1.0 - 0.1 * changed + 0.005 * math.tanh(u)

    return problem, constraint, scorer


def exact_audit(problem: MatrixProblem, constraint, scorer) -> dict:
    """Independent full-domain and reachable-component audit, max 625 states."""
    total = math.prod(len(domain) for domain in problem.domains)
    if total > 625:
        raise ValueError("This benchmark caps full assignment space at 625")
    feasible = {}
    for state in itertools.product(*problem.domains):
        if constraint(state)[0]:
            value = float(scorer(state))
            if not math.isfinite(value):
                raise ValueError("Non-finite audit score")
            feasible[state] = value
    if problem.initial not in feasible:
        raise ValueError("Initial state must be feasible")
    seen = {problem.initial}
    queue = deque([problem.initial])
    while queue:
        for _, candidate in problem.neighbors(queue.popleft()):
            if candidate in feasible and candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    scores = list(feasible.values())
    return {
        "total_states": total, "feasible_states": len(feasible), "reachable_states": len(seen),
        "connected_from_initial": len(seen) == len(feasible),
        "global_max": max(scores), "global_min": min(scores),
        "reachable_max": max(feasible[state] for state in seen),
        "reachable_min": min(feasible[state] for state in seen),
        "role": "evaluation_only_never_passed_to_search",
    }


def normalized_regret(best_score: float, oracle: dict) -> float | None:
    scale = oracle["global_max"] - oracle["global_min"]
    if scale <= 1e-12:
        return None
    value = (oracle["global_max"] - best_score) / scale
    if value < -1e-10 or value > 1 + 1e-10:
        raise ValueError("Search best score outside oracle range")
    return float(min(1.0, max(0.0, value)))


def paired_summary(rows: list[dict], *, bootstrap_seed: int, repeats: int,
                   confidence: float, relative_gate: float, family_harm_gate: float) -> dict:
    """One paired row per source problem. Budgets and runs are not replicates."""
    by_problem: dict[str, dict[str, dict]] = {}
    for row in rows:
        policy = row["policy"]
        if policy not in {"fifo", "priority"}:
            raise ValueError("Unexpected policy")
        arms = by_problem.setdefault(row["problem_id"], {})
        if policy in arms:
            raise ValueError("A problem-policy pair appears more than once")
        arms[policy] = row
    paired = []
    omitted = []
    for identifier in sorted(by_problem):
        arms = by_problem[identifier]
        if set(arms) != {"fifo", "priority"}:
            raise ValueError("Every problem needs both policy results")
        a, b = arms["fifo"], arms["priority"]
        for key in ("case_sha256", "problem_hash", "budget", "family", "split"):
            if a[key] != b[key]:
                raise ValueError(f"Unpaired task identity or budget: {key}")
        if a["normalized_regret"] is None or b["normalized_regret"] is None:
            omitted.append(identifier)
            continue
        paired.append((identifier, a["family"], a["normalized_regret"], b["normalized_regret"]))
    if not paired:
        return {"status": "no_nonconstant_tasks", "omitted_constant_tasks": omitted}
    groups = {family: np.asarray([[a, b] for _, f, a, b in paired if f == family], dtype=float)
              for family in FAMILIES if any(f == family for _, f, _, _ in paired)}
    weights = {family: len(values) / len(paired) for family, values in groups.items()}
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.zeros((repeats, 2))
    for family, values in groups.items():
        sampled = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
        boot += weights[family] * sampled
    alpha = (1 - confidence) / 2
    means = np.mean([[a, b] for _, _, a, b in paired], axis=0)
    improvement = float(means[0] - means[1])
    delta_samples = boot[:, 0] - boot[:, 1]
    delta_ci = np.quantile(delta_samples, [alpha, 1 - alpha]).tolist()
    relative = float(improvement / means[0]) if means[0] > 1e-12 else None
    finite = boot[:, 0] > 1e-12
    relative_ci = np.quantile(delta_samples[finite] / boot[finite, 0], [alpha, 1 - alpha]).tolist() if finite.any() else None
    family_rows = {}
    harmed = []
    for family, values in groups.items():
        family_mean = values.mean(axis=0)
        increase = float(family_mean[1] - family_mean[0])
        family_rows[family] = {
            "n_problems": len(values), "fifo_mean_regret": float(family_mean[0]),
            "priority_mean_regret": float(family_mean[1]),
            "absolute_improvement": float(-increase),
            "harm_exceeds_predefined_margin": increase > family_harm_gate,
        }
        if increase > family_harm_gate:
            harmed.append(family)
    gate = relative is not None and relative >= relative_gate and delta_ci[0] > 0 and not harmed
    outcome = "baseline_ceiling" if relative is None else "GO_SYNTHETIC_MECHANISM_ONLY" if gate else "DO_NOT_PROMOTE_PRIORITY_AS_DEFAULT"
    return {
        "status": "computed", "n_paired_problems": len(paired),
        "omitted_constant_tasks": omitted,
        "fifo_mean_regret": float(means[0]), "priority_mean_regret": float(means[1]),
        "absolute_improvement_fifo_minus_priority": improvement,
        "paired_improvement_ci": [float(x) for x in delta_ci],
        "relative_mean_regret_reduction": relative,
        "relative_reduction_ci": [float(x) for x in relative_ci] if relative_ci is not None else None,
        "confidence": confidence, "bootstrap_repeats": repeats, "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "whole_problem_paired_policies_stratified_by_family",
        "family_results": family_rows, "harmed_families": harmed,
        "decision": outcome,
        "go_rule": {"relative_reduction_min": relative_gate, "paired_ci_lower_gt": 0,
                    "no_family_absolute_regret_increase_gt": family_harm_gate},
        "interpretation": "Performance on this fixed synthetic generator only; no real-poetry quality evidence",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_suite(output: Path, config: dict | None = None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    _validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Frozen run directory already exists; use a new output path for an explicitly labeled replay")
    root = Path(__file__).resolve().parents[2]
    code_files = [Path(__file__), Path(__file__).with_name("matrix.py"), root / "scripts/run_mechanism_suite.py"]
    source_sha = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in code_files if path.exists()}
    manifest = {
        "schema": "versematrix.mechanism-freeze.v0.2", "config": config,
        "config_sha256": digest(config), "source_sha256": source_sha,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen_before_input_generation", "external_preregistration": False,
        "test_status": "not_yet_evaluated",
        "python": platform.python_version(), "numpy": np.__version__,
        "history": [],
    }
    _write_json(manifest_path, manifest)
    cases = generate_cases(config)
    case_payload = "".join(canonical(case) + "\n" for case in cases)
    (output / "cases.jsonl").write_text(case_payload, encoding="utf-8")
    manifest.update({"cases_sha256": hashlib.sha256(case_payload.encode()).hexdigest(),
                     "n_cases": len(cases), "stage": "frozen_before_first_evaluation"})
    manifest["history"].append({"event": "inputs_frozen", "at_utc": datetime.now(timezone.utc).isoformat()})
    _write_json(manifest_path, manifest)
    records, audits = [], []
    replay_failures = 0
    started = time.perf_counter()
    with gzip.open(output / "traces.jsonl.gz", "wt", encoding="utf-8") as traces, \
         (output / "runs.jsonl").open("w", encoding="utf-8") as runs_file:
        for case_index, case in enumerate(cases):
            problem, constraint, scorer = compile_case(case)
            # This audit is computed in the evaluator. Only the original scorer,
            # not its extrema or the reachable set, is passed into organize().
            audit = exact_audit(problem, constraint, scorer)
            if not audit["connected_from_initial"]:
                raise ValueError("Frozen task generator produced a disconnected feasible state graph")
            audits.append({"problem_id": case["problem_id"], "case_sha256": case["case_sha256"], **audit})
            for budget in config["proposal_budgets"]:
                for policy in config["policies"]:
                    begin = time.perf_counter()
                    result = organize(problem, constraint, scorer, scorer_id=case["case_sha256"],
                                      strategy=policy, max_proposals=budget, max_score_calls=budget + 1)
                    seconds = time.perf_counter() - begin
                    replay_ok = False
                    if config["replay_all_runs"]:
                        replay(problem, constraint, scorer, scorer_id=case["case_sha256"], trace=result.trace)
                        replay_ok = True
                    if result.proposals > budget or result.score_calls > budget + 1 or not constraint(result.best_state)[0]:
                        raise AssertionError("Search legality or budget failure")
                    first_optimum = None
                    for event in result.trace:
                        if event["kind"] in {"initialize", "enqueue"}:
                            if abs(event["score"] - audit["global_max"]) <= 1e-10:
                                first_optimum = event.get("proposals", 0)
                                break
                    item = {
                        "problem_id": case["problem_id"], "split": case["split"], "family": case["family"],
                        "case_sha256": case["case_sha256"], "problem_hash": problem.problem_hash,
                        "policy": policy, "budget": budget, "max_score_calls": budget + 1,
                        "proposals": result.proposals, "score_calls": result.score_calls,
                        "unique_legal_states": result.unique_legal_states, "expanded_states": result.expanded_states,
                        "best_score": result.best_score, "best_state": list(result.best_state),
                        "normalized_regret": normalized_regret(result.best_score, audit),
                        "reachable_oracle_gap": audit["reachable_max"] - result.best_score,
                        "global_oracle_gap": audit["global_max"] - result.best_score,
                        "reached_optimum": abs(result.best_score - audit["global_max"]) <= 1e-10,
                        "first_optimum_proposal": first_optimum,
                        "wall_seconds_excluding_oracle_and_replay": seconds,
                        "stop_reason": result.stop_reason, "replay_verified": replay_ok,
                        "trace_sha256": result.trace[-1]["event_hash"],
                    }
                    records.append(item)
                    runs_file.write(canonical(item) + "\n")
                    traces.write(canonical({"problem_id": case["problem_id"], "policy": policy,
                                             "budget": budget, "trace": result.trace}) + "\n")
            if (case_index + 1) % 60 == 0:
                print(json.dumps({"stage": "running", "cases_completed": case_index + 1,
                                  "cases_total": len(cases)}, ensure_ascii=False), flush=True)
    _write_json(output / "exact_audits.json", audits)
    summaries = {}
    for split in config["split_sizes"]:
        summaries[split] = {}
        for budget in config["proposal_budgets"]:
            selected = [row for row in records if row["split"] == split and row["budget"] == budget]
            summary = paired_summary(selected, bootstrap_seed=config["bootstrap_seed"] + budget,
                                     repeats=config["bootstrap_repeats"], confidence=config["confidence"],
                                     relative_gate=config["relative_regret_reduction_gate"],
                                     family_harm_gate=config["family_absolute_regret_harm_gate"])
            for policy in config["policies"]:
                arm = [row for row in selected if row["policy"] == policy]
                summary[policy + "_diagnostics"] = {
                    "mean_proposals": float(np.mean([r["proposals"] for r in arm])),
                    "mean_score_calls": float(np.mean([r["score_calls"] for r in arm])),
                    "mean_unique_legal_states": float(np.mean([r["unique_legal_states"] for r in arm])),
                    "optimal_found_rate": float(np.mean([r["reached_optimum"] for r in arm])),
                    "mean_wall_seconds": float(np.mean([r["wall_seconds_excluding_oracle_and_replay"] for r in arm])),
                }
            summaries[split][str(budget)] = summary
    primary = summaries["test"][str(config["primary_budget"])]
    report = {
        "project": "VerseMatrix Lab", "version": "0.2.0",
        "status": "executed_synthetic_mechanism_experiment", "synthetic": True,
        "trained": False, "human_evaluated": False,
        "n_dev_problems": config["split_sizes"]["dev"], "n_test_problems": config["split_sizes"]["test"],
        "total_runs": len(records), "connected_cases": sum(a["connected_from_initial"] for a in audits),
        "replayed_runs": sum(row["replay_verified"] for row in records),
        "config_sha256": manifest["config_sha256"], "cases_sha256": manifest["cases_sha256"],
        "elapsed_seconds_including_audit_replay_and_summary": time.perf_counter() - started,
        "primary_budget": config["primary_budget"], "primary_test_result": primary,
        "all_results": summaries,
        "interpretation": [
            "The test split was not used to change the generator, scoring rules or policy before reporting.",
            "After this run the 240 test tasks are exposed and cannot certify later algorithm tuning.",
            "All task scores are synthetic objectives; these are not poetry quality labels.",
            "The test asks whether one frontier schedule helps on this fixed mixture, not whether matrix notation is inherently superior.",
            "Bootstrap uncertainty is conditional on this generator and fixed family mixture, not a guarantee across all constraint problems.",
            "Runs are replayed outside the measured search budget; the replay is verification, not a search aid.",
        ],
    }
    _write_json(output / "report.json", report)
    manifest.update({"stage": "completed", "test_status": "evaluated_now_exposed",
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    manifest["history"].append({"event": "evaluation_completed", "at_utc": manifest["completed_at_utc"]})
    _write_json(manifest_path, manifest)
    return report
