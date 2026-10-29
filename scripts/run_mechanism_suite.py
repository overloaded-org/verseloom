"""Run the fixed 60-dev/240-test synthetic mechanism suite once.

PYTHONPATH=src python scripts/run_mechanism_suite.py --out results/mechanism_v02
For an explicitly labeled replay, use a different --out directory. Never tune
on these test outputs and call a repeated run independent validation.
"""
import argparse
import json
from pathlib import Path

from wordconv_lab.mechanism import run_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/mechanism_v02"))
    args = parser.parse_args()
    report = run_suite(args.out)
    primary = report["primary_test_result"]
    print(json.dumps({
        "status": report["status"], "runs": report["total_runs"],
        "primary_budget": report["primary_budget"],
        "fifo_mean_regret": primary["fifo_mean_regret"],
        "priority_mean_regret": primary["priority_mean_regret"],
        "relative_regret_reduction": primary["relative_mean_regret_reduction"],
        "paired_improvement_ci": primary["paired_improvement_ci"],
        "harmed_families": primary["harmed_families"],
        "decision": primary["decision"], "real_poetry_quality_evaluated": False,
    }, ensure_ascii=False, indent=2))
