"""Run four predeclared interface examples, never select examples by outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wordconv_lab.completion_v05 import complete, load_training_poems
from wordconv_lab.corpus import make_tasks
from wordconv_lab.restoration import VisibleQuery


def main():
    config_path = ROOT / "configs/completion_examples_v0.5.json"
    freeze = json.loads((ROOT / "results/completion_examples_v05_freeze.json").read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != freeze["config_sha256"]:
        raise ValueError("predeclared example inputs changed")
    config = json.loads(config_path.read_text())
    poems = load_training_poems(ROOT / "data/corpus_v03/poems.jsonl")
    tasks = make_tasks(poems)
    examples = []
    for example in config["examples"]:
        output = complete(example["upper"], example["suffix"], method=config["method"],
                          seed=config["seed"], limit=config["limit"], root=ROOT)
        query = VisibleQuery(len(example["upper"]), example["upper"], example["suffix"])
        matches = [task for task in tasks if VisibleQuery.from_task(task) == query]
        examples.append({"id": example["id"], "kind": example["kind"], "output": output,
            "training_overlap": {
                "upper_line_occurrences": sum(line == example["upper"] for p in poems for line in p["lines"]),
                "exact_visible_training_task_matches": len(matches),
                "matching_training_task_ids": [task["task_id"] for task in matches],
                "interpretation": "interface demonstration; source familiarity or absence here does not establish independent evaluation"}})
    result = {"schema": "versematrix.completion-examples-results.v0.5",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": freeze["config_sha256"], "input_selection": config["selection_policy"],
              "seed": config["seed"], "independent_evaluation": False, "examples": examples}
    output_path = ROOT / "results/completion_examples_v05.json"
    temporary = output_path.with_suffix(".json.building")
    if output_path.exists() or temporary.exists():
        raise FileExistsError("example result paths are create-only")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    import os
    os.link(temporary, output_path)
    temporary.unlink()
    print(json.dumps({"examples": len(examples), "output": str(output_path),
                      "primary_model_seed": config["seed"], "independent_evaluation": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
