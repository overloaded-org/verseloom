"""Fixed-W contextual matching on original training/development splits only.

prepare freezes code/config/input bytes, candidate identities, sampling and
context-mismatch mapping before training. run performs the declared budget and
selects one common checkpoint per arm using all three seeds on development.
No old test or v0.4 confirmation content is decoded or evaluated here.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.corpus import make_tasks
from wordconv_lab.restoration import VisibleQuery, digest, family_metric
from wordconv_lab.retrieval import PhraseRetriever

CONFIG_PATH = ROOT / "configs/matching_v0.5.json"
DEFAULT_OUTPUT = ROOT / "results/matching_v05"
SPLIT_PATTERN = re.compile(rb'"split":"(train|dev|test)"')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def json_ready(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_ready(v) for v in value]
    return value


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode()
    write_bytes_new(path, data)


def write_bytes_new(path, data):
    """Atomic create-only placement; an immutable checkpoint is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def write_rows(path, rows):
    write_bytes_new(path, "".join(json.dumps(json_ready(row), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode())


def read_rows(path):
    return [json.loads(row) for row in path.read_bytes().splitlines()]


@lru_cache(maxsize=8)
def candidate_fingerprint(candidates):
    return digest(candidates)


def read_train_dev(path=None, expected_counts=None):
    path = path or ROOT / "data/corpus_v03/poems.jsonl"
    selected = {"train": [], "dev": []}
    skipped_test = 0
    with path.open("rb") as handle:
        for raw in handle:
            markers = SPLIT_PATTERN.findall(raw)
            if len(markers) != 1:
                raise ValueError("expected one canonical split marker")
            split = markers[0].decode("ascii")
            if split == "test":
                skipped_test += 1
                continue  # No JSON decoding or interpretation of old test rows.
            poem = json.loads(raw)
            if poem["split"] != split:
                raise ValueError("split marker differs from decoded source")
            selected[split].append(poem)
    if expected_counts and any(len(selected[k]) != n for k, n in expected_counts.items()):
        raise ValueError("unexpected training/development poem counts")
    return selected, skipped_test


def candidate_indices(candidates):
    indices = {}
    for length, spans in candidates.items():
        if tuple(spans) != tuple(sorted(set(spans))) or any(len(s) != 2 for s in spans):
            raise ValueError("candidate spans must be distinct and lexical")
        indices[int(length)] = {span: i for i, span in enumerate(spans)}
    return indices


def target_bucket(task, indices, prefix_counts):
    length, target = task["length"], task["target"]
    if target not in indices[length]: return "outside_W"
    return "seen_prefix_positive" if prefix_counts[length].get(target, 0) > 0 else "in_W_never_prefix_positive"


def load_data(config):
    selected, skipped = read_train_dev(expected_counts={
        "train": config["data"]["expected_train_poems"],
        "dev": config["data"]["expected_dev_poems"]})
    old_build = read_json(ROOT / "data/corpus_v03/build_manifest.json")
    for filename in ("poems.jsonl", "vocabulary.json", "candidate_spans.json"):
        path = ROOT / "data/corpus_v03" / filename
        if sha(path) != old_build["outputs"][filename]["sha256"]:
            raise ValueError("original corpus identity changed: " + filename)
    vocabulary = {token: i for i, token in enumerate(read_json(ROOT / "data/corpus_v03/vocabulary.json")["tokens"])}
    retriever = PhraseRetriever(selected["train"], vocabulary)
    resources = retriever.candidate_resources()
    candidates = {int(length): tuple(item["text"] for item in items)
                  for length, items in resources["by_length"].items()}
    if {str(k): len(v) for k, v in candidates.items()} != config["data"]["expected_candidates_by_length"]:
        raise ValueError("fixed candidate W changed")
    expected_wide = read_json(ROOT / "results/dev_candidate_diagnostic_v03/candidate_counts.json")["all_adjacent_min2"]
    if {k: {x["text"]: x["count"] for x in v} for k, v in resources["by_length"].items()} != expected_wide:
        raise ValueError("W/counts differ from the original train-only diagnostic")
    indices = candidate_indices(candidates)
    tasks = {split: sorted(make_tasks(poems), key=lambda t: t["task_id"])
             for split, poems in selected.items()}
    for split in tasks:
        if len(tasks[split]) != config["data"]["expected_" + split + "_tasks"]:
            raise ValueError("task denominator changed")
    prefix_counts = {length: Counter(t["target"] for t in tasks["train"] if t["length"] == length)
                     for length in candidates}
    for length, rows in resources["by_length"].items():
        for row in rows:
            count = prefix_counts[int(length)][row["text"]]
            if count != row["narrow_count"]:
                raise ValueError("candidate prefix-positive count differs from training tasks")
            row["train_prefix_positive_count"] = count
    eligible = [t for t in tasks["train"] if t["target"] in indices[t["length"]]]
    if len(eligible) != config["data"]["expected_eligible_train_tasks"]:
        raise ValueError("eligible training task count changed")
    coverage = {"old_test_rows_skipped_before_json_decode": skipped, "splits": {},
                "candidate_counts": {str(k): len(v) for k, v in candidates.items()},
                "candidate_set_sha256": {str(k): digest(v) for k, v in candidates.items()},
                "vocabulary_size_with_UNK": len(vocabulary),
                "old_test_or_confirmation_content_decoded": False}
    for split, rows in tasks.items():
        coverage["splits"][split] = coverage_summary(rows, indices, prefix_counts, vocabulary)
    return {"poems": selected, "tasks": tasks, "eligible": eligible, "vocabulary": vocabulary,
            "retriever": retriever, "resources": resources, "candidates": candidates,
            "indices": indices, "prefix_counts": prefix_counts, "coverage": coverage}


def coverage_summary(tasks, indices, prefix_counts, vocabulary):
    covered = [int(t["target"] in indices[t["length"]]) for t in tasks]
    buckets = Counter(target_bucket(t, indices, prefix_counts) for t in tasks)
    return {"tasks": len(tasks), "poems": len({t["poem_id"] for t in tasks}),
            "families": len({t["family_id"] for t in tasks}), "covered_tasks": sum(covered),
            "vocabulary_coverage": family_metric(tasks, covered),
            "prefix_positive_buckets": {k: buckets[k] for k in
                ("seen_prefix_positive", "in_W_never_prefix_positive", "outside_W")},
            "unknown_visible_character_tasks": sum(any(c not in vocabulary for c in t["upper_line"] + t["lower_suffix"]) for t in tasks),
            "by_length": {str(length): {"tasks": sum(t["length"] == length for t in tasks),
                "covered": sum(t["length"] == length and value for t, value in zip(tasks, covered))}
                for length in sorted(indices)}}


def make_schedule(tasks, updates, batch_size, seed, salt):
    """Exactly full batches with a declared proportional same-length schedule."""
    if type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer update and batch counts required")
    groups = {length: sorted((t for t in tasks if t["length"] == length), key=lambda t: t["task_id"])
              for length in sorted({t["length"] for t in tasks})}
    if not groups or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("unique nonempty eligible training tasks required")
    total = len(tasks)
    counts = {length: updates * len(rows) // total for length, rows in groups.items()}
    remainder_order = sorted(groups, key=lambda length: (-(updates * len(groups[length]) % total), length))
    for length in remainder_order[:updates - sum(counts.values())]: counts[length] += 1
    lengths = np.array([length for length in groups for _ in range(counts[length])], dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([salt, seed, 0]))
    rng.shuffle(lengths)
    rngs = {length: np.random.default_rng(np.random.SeedSequence([salt, seed, length])) for length in groups}
    orders = {length: np.array([], dtype=np.int64) for length in groups}
    positions = {length: 0 for length in groups}
    cycles = {length: 0 for length in groups}
    result = []
    for update, length in enumerate(lengths.tolist(), 1):
        chosen = []
        while len(chosen) < batch_size:
            if positions[length] == len(orders[length]):
                orders[length] = rngs[length].permutation(len(groups[length]))
                positions[length] = 0
                cycles[length] += 1
            take = min(batch_size - len(chosen), len(orders[length]) - positions[length])
            selected = orders[length][positions[length]:positions[length] + take]
            chosen.extend(groups[length][int(i)]["task_id"] for i in selected)
            positions[length] += take
        result.append({"update": update, "length": length, "task_ids": chosen,
                       "completed_or_active_cycles_by_length": {str(k): v for k, v in cycles.items()}})
    if len(result) != updates or sum(len(row["task_ids"]) for row in result) != updates * batch_size:
        raise AssertionError("sampler exposure budget mismatch")
    return result


def make_context_permutation(tasks, salt):
    """Target-independent one-to-one mismatch, separately for each line length."""
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("unique nonempty development tasks required")
    query_hashes = {t["task_id"]: digest(asdict(VisibleQuery.from_task(t))) for t in tasks}
    rows = []
    offsets = {}
    for length in sorted({t["length"] for t in tasks}):
        ordered = sorted((t for t in tasks if t["length"] == length),
                         key=lambda t: (hashlib.sha256((salt + ":" + t["task_id"]).encode()).hexdigest(), t["task_id"]))
        n = len(ordered)
        offset = next((offset for offset in range(1, n) if all(
            t["family_id"] != ordered[(i + offset) % n]["family_id"] and
            query_hashes[t["task_id"]] != query_hashes[ordered[(i + offset) % n]["task_id"]]
            for i, t in enumerate(ordered))), None)
        if offset is None:
            raise ValueError("no legal family/query-disjoint context permutation")
        offsets[str(length)] = offset
        for i, task in enumerate(ordered):
            donor = ordered[(i + offset) % n]
            rows.append({"task_id": task["task_id"], "donor_task_id": donor["task_id"],
                         "length": length, "family_id": task["family_id"],
                         "donor_family_id": donor["family_id"],
                         "query_hash": query_hashes[task["task_id"]],
                         "donor_query_hash": query_hashes[donor["task_id"]]})
    return {"salt": salt, "offset_by_length": offsets, "uses_targets": False,
            "gold_equality_is_not_a_filter": True,
            "mapping": sorted(rows, key=lambda row: row["task_id"])}


def prediction_rows(tasks, scores, candidates, *, query_hashes=None, donor_ids=None):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(tasks), len(candidates)) or not np.isfinite(scores).all():
        raise ValueError("one finite score per task and complete fixed-W candidate required")
    if tuple(candidates) != tuple(sorted(set(candidates))):
        raise ValueError("candidate lexical tie order changed")
    target_indices = {span: i for i, span in enumerate(candidates)}
    candidate_hash = candidate_fingerprint(tuple(candidates))
    rows = []
    for i, task in enumerate(tasks):
        order = np.argsort(-scores[i], kind="stable")
        target_index = target_indices.get(task["target"])
        positions = np.flatnonzero(order == target_index) if target_index is not None else []
        top = order[:16]
        row = {key: task[key] for key in ("task_id", "poem_id", "family_id", "length", "target")}
        row.update(target_rank=int(positions[0]) + 1 if len(positions) else None,
                   candidate_count=len(candidates),
                   candidate_vocabulary_sha256=candidate_hash,
                   query_hash=query_hashes[i] if query_hashes is not None else digest(asdict(VisibleQuery.from_task(task))),
                   full_ranking_sha256=hashlib.sha256(order.astype("<i4").tobytes()).hexdigest(),
                   full_scores_float64_sha256=hashlib.sha256(scores[i].astype("<f8").tobytes()).hexdigest(),
                   top_candidates=[candidates[int(j)] for j in top],
                   top_scores=[float(scores[i, j]) for j in top])
        if donor_ids is not None: row["donor_task_id"] = donor_ids[i]
        rows.append(row)
    return rows


def exact_seed_family_recall(tasks, ranks_by_seed, k=16):
    if not ranks_by_seed or any(len(ranks) != len(tasks) for ranks in ranks_by_seed):
        raise ValueError("one paired rank vector per seed required")
    groups = defaultdict(list)
    for i, task in enumerate(tasks):
        value = Fraction(sum(rank[i] is not None and rank[i] <= k for rank in ranks_by_seed), len(ranks_by_seed))
        groups[task["family_id"]].append(value)
    if not groups: raise ValueError("empty development set")
    return sum((sum(values, Fraction()) / len(values) for values in groups.values()), Fraction()) / len(groups)


def select_shared_checkpoint(tasks, ranks_by_step, k=16):
    if not ranks_by_step: raise ValueError("no declared checkpoints")
    values = {step: exact_seed_family_recall(tasks, ranks, k) for step, ranks in ranks_by_step.items()}
    chosen = max(sorted(values), key=lambda step: values[step])
    return chosen, values


def metric_bundle(tasks, ranks_by_seed, indices, prefix_counts):
    if not tasks or not ranks_by_seed or any(len(r) != len(tasks) for r in ranks_by_seed):
        raise ValueError("all tasks and seeds must be aligned")
    values = {}
    for name, transform in (
        ("recall_at_16", lambda r: float(r is not None and r <= 16)),
        ("top1", lambda r: float(r == 1)),
        ("mrr", lambda r: 0.0 if r is None else 1.0/r)):
        values[name] = np.mean([[transform(r) for r in ranks] for ranks in ranks_by_seed], axis=0)
    def summarize(positions):
        chosen = [tasks[i] for i in positions]
        if not chosen: return {"tasks": 0, "families": 0, "metrics": None}
        return {"tasks": len(chosen), "families": len({t["family_id"] for t in chosen}),
                "metrics": {name: family_metric(chosen, array[positions].tolist()) for name, array in values.items()}}
    result = summarize(list(range(len(tasks))))
    result["seed_count"] = len(ranks_by_seed)
    result["aggregation"] = "arithmetic_mean_across_seeds_per_task_then_mean_within_family_then_equal_family_mean"
    result["by_length"] = {str(length): summarize([i for i, t in enumerate(tasks) if t["length"] == length]) for length in sorted(indices)}
    result["by_prefix_positive_bucket"] = {bucket: summarize([i for i, t in enumerate(tasks) if target_bucket(t, indices, prefix_counts) == bucket])
        for bucket in ("seen_prefix_positive", "in_W_never_prefix_positive", "outside_W")}
    return result


def evaluate_model(model, tasks, candidates, batch_size, *, mask="none", donors=None):
    rows = []
    for length in sorted(candidates):
        selected = [t for t in tasks if t["length"] == length]
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            contexts = [donors[t["task_id"]] if donors is not None else t for t in batch]
            queries = [VisibleQuery.from_task(t) for t in contexts]
            scores = model.score_all(queries, mask=mask)
            # The scorer above only receives allowlisted visible queries. Gold
            # is inspected here after every candidate has already been scored.
            rows.extend(prediction_rows(batch, scores, candidates[length],
                query_hashes=[digest(asdict(q)) for q in queries],
                donor_ids=[t["task_id"] for t in contexts] if donors is not None else None))
    return sorted(rows, key=lambda r: r["task_id"])


def baseline_rows(data, method_id):
    tasks, candidates = data["tasks"]["dev"], data["candidates"]
    result = []
    if method_id == "strong_statistical":
        for i, task in enumerate(tasks):
            components = data["retriever"].score_components(VisibleQuery.from_task(task))
            scores = (np.asarray(components["ngram_scores"]) + .25*np.asarray(components["alignment_bonus"])
                      + .25*np.asarray(components["boundary_bonus"]))
            result.extend(prediction_rows([task], scores[None, :], candidates[task["length"]]))
            if (i+1) % 500 == 0: print(json.dumps({"stage": "fixed_baseline", "tasks": i+1}), flush=True)
    else:
        for length, spans in candidates.items():
            resources = data["resources"]["by_length"][str(length)]
            count_name = "count" if method_id == "all_position_frequency" else "train_prefix_positive_count"
            if method_id not in ("all_position_frequency", "prefix_positive_frequency"):
                raise ValueError("undeclared baseline")
            scores = np.asarray([row[count_name] for row in resources], dtype=np.float64)
            selected = [t for t in tasks if t["length"] == length]
            for task in selected:
                result.extend(prediction_rows([task], scores[None, :], spans))
    return sorted(result, key=lambda r: r["task_id"])


def source_paths(config_path):
    names = ["scripts/run_matching_v05.py", "src/wordconv_lab/matching.py",
             "src/wordconv_lab/retrieval.py", "src/wordconv_lab/restoration.py", "src/wordconv_lab/corpus.py"]
    return [ROOT / name for name in names] + [config_path]


def input_paths():
    return [ROOT / name for name in (
        "data/corpus_v03/poems.jsonl", "data/corpus_v03/vocabulary.json",
        "data/corpus_v03/candidate_spans.json", "data/corpus_v03/build_manifest.json",
        "results/dev_candidate_diagnostic_v03/candidate_counts.json")]


def identity(paths):
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def validate_config(config):
    if config["status"] not in ("draft", "frozen"):
        raise ValueError("configuration must be draft or frozen")
    if config["seeds"] != [0, 1, 2] or config["primary_arm"] != "full_w3" or config["deployment_seed"] != 0:
        raise ValueError("predeclared seeds/primary/deployment identity changed")
    if [a["id"] for a in config["arms"]] != ["full_w3", "full_w1", "upper_w3", "suffix_w3"]:
        raise ValueError("expected four declared training arms")
    train = config["training"]
    checkpoints = train["checkpoint_updates"]
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != train["optimizer_updates"]:
        raise ValueError("checkpoints must be increasing and end at the full update budget")
    if train["gradient_accumulation"] != 1 or train["batch_size"] != 32:
        raise ValueError("this runner implements exactly one full batch32 per optimizer update")
    if (train["beta1"], train["beta2"], train["epsilon"]) != (.9, .999, 1e-8):
        raise ValueError("optimizer constants differ from matching.py implementation")
    if config["evaluation"]["k"] != 16 or config["evaluation"]["score_mixing"]:
        raise ValueError("fixed K16 and pure neural scoring are required")
    baseline = config["evaluation"]["baselines"][0]
    if (baseline["id"], baseline["alignment_weight"], baseline["boundary_weight"], baseline["pool"]) != ("strong_statistical", .25, .25, "wide"):
        raise ValueError("strong baseline must retain frozen v0.4 weights")


def check_frozen(output):
    freeze = read_json(output / "freeze.json")
    for category in ("source_sha256", "input_sha256"):
        for path, expected in freeze[category].items():
            if sha(ROOT / path) != expected:
                raise ValueError("frozen identity changed: " + path)
    config = freeze["config"]
    if config["status"] != "frozen" or digest(config) != freeze["config_sha256"]:
        raise ValueError("invalid frozen configuration snapshot")
    validate_config(config)
    return config, freeze


def prepare(output, config_path):
    config = read_json(config_path)
    validate_config(config)
    if output.exists(): raise ValueError("prepare requires a new result directory")
    output.mkdir(parents=True)
    if config["status"] == "draft":
        write_new(output / "config_before_freeze.json", config)
        config["status"] = "frozen"
        temporary = config_path.with_name(config_path.name + ".freeze.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, config_path)
    freeze = {"schema": "versematrix.matching-freeze.v0.5", "frozen_at_utc": now(),
              "config": config, "config_sha256": digest(config),
              "source_sha256": identity(source_paths(config_path)), "input_sha256": identity(input_paths()),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
              "status": "frozen_before_formal_training_or_dev_model_evaluation",
              "external_preregistration": False}
    write_new(output / "freeze.json", freeze)
    data = load_data(config)
    write_new(output / "candidate_resources.json", data["resources"])
    write_new(output / "coverage.json", data["coverage"])
    mapping = make_context_permutation(data["tasks"]["dev"], config["diagnostics"]["permutation_salt"])
    write_new(output / "context_permutation.json", mapping)
    schedule_files = []
    for seed in config["seeds"]:
        schedule = make_schedule(data["eligible"], config["training"]["optimizer_updates"],
            config["training"]["batch_size"], seed, config["training"]["schedule_seed_salt"])
        path = output / "schedules" / f"seed{seed}.jsonl"
        write_rows(path, schedule)
        schedule_files.append(path)
    files = [output / name for name in ("candidate_resources.json", "coverage.json", "context_permutation.json")] + schedule_files
    prepared = {"status": "prepared_no_formal_updates", "prepared_at_utc": now(),
                "outputs": {str(path.relative_to(output)): {"sha256": sha(path), "bytes": path.stat().st_size} for path in files},
                "freeze_sha256": sha(output / "freeze.json"),
                "formal_optimizer_updates_performed": 0, "dev_model_evaluation_performed": False}
    write_new(output / "prepared.json", prepared)
    return {"status": prepared["status"], "coverage": data["coverage"],
            "context_permutation_offsets": mapping["offset_by_length"]}


def schedule_accounting(schedule, task_index):
    exposures = [tid for row in schedule for tid in row["task_ids"]]
    tasks = set(exposures)
    return {"optimizer_updates": len(schedule), "micro_steps": len(schedule),
            "gradient_accumulation": 1, "batch_size": len(schedule[0]["task_ids"]) if schedule else 0,
            "sample_presentations": len(exposures), "distinct_tasks_presented": len(tasks),
            "distinct_poems_presented": len({task_index[tid]["poem_id"] for tid in tasks}),
            "distinct_families_presented": len({task_index[tid]["family_id"] for tid in tasks}),
            "sample_presentations_by_length": dict(sorted(Counter(str(task_index[tid]["length"]) for tid in exposures).items())),
            "task_presentation_count_sha256": digest(dict(sorted(Counter(exposures).items())))}


def aligned_ranks(tasks, rows):
    if [r["task_id"] for r in rows] != [t["task_id"] for t in tasks]:
        raise ValueError("prediction/task identity order mismatch")
    return [row["target_rank"] for row in rows]


def run(output):
    from wordconv_lab.matching import ContextSpanMatcher
    config, freeze = check_frozen(output)
    prepared = read_json(output / "prepared.json")
    if sha(output / "freeze.json") != prepared["freeze_sha256"]:
        raise ValueError("preparation freeze identity changed")
    for rel, expected in prepared["outputs"].items():
        if sha(output / rel) != expected["sha256"]:
            raise ValueError("prepared data/schedule identity changed: " + rel)
    if (output / "run_started.json").exists():
        raise ValueError("formal run already started; do not overwrite or silently resume")
    data = load_data(config)
    if data["resources"] != read_json(output / "candidate_resources.json") or data["coverage"] != read_json(output / "coverage.json"):
        raise ValueError("prepared candidate resources/coverage cannot be reproduced")
    mapping = read_json(output / "context_permutation.json")
    if mapping != make_context_permutation(data["tasks"]["dev"], config["diagnostics"]["permutation_salt"]):
        raise ValueError("prepared context permutation changed")
    started = time.perf_counter()
    write_new(output / "run_started.json", {"started_at_utc": now(), "freeze_sha256": sha(output / "freeze.json"),
        "config_sha256": freeze["config_sha256"], "scope": "dev_only"})
    train_index = {t["task_id"]: t for t in data["eligible"]}
    dev = data["tasks"]["dev"]
    baseline_metrics = {}
    baseline_ranks = {}
    for baseline in config["evaluation"]["baselines"]:
        name = baseline["id"]
        rows = baseline_rows(data, name)
        write_rows(output / "baselines" / f"{name}.jsonl", rows)
        ranks = aligned_ranks(dev, rows)
        baseline_ranks[name] = ranks
        baseline_metrics[name] = metric_bundle(dev, [ranks], data["indices"], data["prefix_counts"])
    write_new(output / "baselines" / "metrics.json", baseline_metrics)
    training = config["training"]
    receipts = {}
    checkpoint_ranks = {arm["id"]: defaultdict(list) for arm in config["arms"]}
    checkpoints = {arm["id"]: defaultdict(list) for arm in config["arms"]}
    for seed in config["seeds"]:
        schedule = read_rows(output / "schedules" / f"seed{seed}.jsonl")
        if schedule != make_schedule(data["eligible"], training["optimizer_updates"], training["batch_size"], seed, training["schedule_seed_salt"]):
            raise ValueError("training schedule does not match frozen sampler")
        for arm in config["arms"]:
            name = arm["id"]
            run_name = f"{name}_seed{seed}"
            print(json.dumps({"stage": "train_start", "run": run_name}), flush=True)
            model = ContextSpanMatcher(data["vocabulary"], data["candidates"], seed=seed,
                **config["model"], **{k: arm[k] for k in ("kernel_width", "channels", "context_mode")})
            initial_parameter_summary = model.parameter_summary()
            losses = []
            steps = []
            model_started = time.perf_counter()
            train_seconds = 0.0
            eval_seconds = 0.0
            for batch in schedule:
                tasks = [train_index[tid] for tid in batch["task_ids"]]
                if any(t["length"] != batch["length"] for t in tasks):
                    raise ValueError("same-length batch invariant broken")
                queries = [VisibleQuery.from_task(t) for t in tasks]
                targets = np.asarray([data["indices"][t["length"]][t["target"]] for t in tasks], dtype=np.int64)
                step_started = time.perf_counter()
                step = model.train_step(queries, targets, learning_rate=training["learning_rate"],
                    weight_decay=training["weight_decay"], max_grad_norm=training["max_grad_norm"])
                train_seconds += time.perf_counter() - step_started
                if model.step_count != batch["update"]:
                    raise ValueError("optimizer step count differs from exposed batches")
                losses.append(float(step["loss"]))
                steps.append({"update": batch["update"], "length": batch["length"],
                              "presentations": len(tasks), **json_ready(step)})
                if batch["update"] in training["checkpoint_updates"]:
                    check_frozen(output)
                    path = output / "models" / run_name / f"update_{batch['update']:04d}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(path)
                    model_hash = sha(path)
                    checkpoint_started = time.perf_counter()
                    rows = evaluate_model(model, dev, data["candidates"], config["evaluation"]["batch_size"])
                    eval_seconds += time.perf_counter() - checkpoint_started
                    if sha(path) != model_hash:
                        raise ValueError("immutable checkpoint changed during evaluation")
                    predictions = output / "predictions" / run_name / f"update_{batch['update']:04d}.jsonl"
                    write_rows(predictions, rows)
                    ranks = aligned_ranks(dev, rows)
                    checkpoint_ranks[name][batch["update"]].append(ranks)
                    record = {"seed": seed, "path": str(path.relative_to(output)), "sha256": model_hash,
                        "step": batch["update"], "predictions_path": str(predictions.relative_to(output)),
                        "predictions_sha256": sha(predictions), "parameters": model.parameter_summary(),
                        "accounting": schedule_accounting(schedule[:batch["update"]], train_index),
                        "dev_metrics": metric_bundle(dev, [ranks], data["indices"], data["prefix_counts"])}
                    checkpoints[name][batch["update"]].append(record)
                    write_new(output / "checkpoints" / run_name / f"update_{batch['update']:04d}.json", record)
                    print(json.dumps({"stage": "checkpoint", "run": run_name, "step": batch["update"],
                        "dev_family_recall_at_16": record["dev_metrics"]["metrics"]["recall_at_16"]["family_macro"],
                        "last_100_loss": float(np.mean(losses[-100:]))}), flush=True)
            write_rows(output / "training" / f"{run_name}_steps.jsonl", steps)
            receipt = {"arm": name, "seed": seed, "parameters": model.parameter_summary(),
                "initial_parameters": initial_parameter_summary,
                "accounting": schedule_accounting(schedule, train_index),
                "schedule_sha256": sha(output / "schedules" / f"seed{seed}.jsonl"),
                "loss_first_100_mean": float(np.mean(losses[:100])),
                "loss_last_100_mean": float(np.mean(losses[-100:])),
                "training_seconds": train_seconds, "checkpoint_evaluation_seconds": eval_seconds,
                "run_wall_seconds": time.perf_counter() - model_started,
                "optimizer": json_ready(model.last_optimizer)}
            write_new(output / "training" / f"{run_name}.json", receipt)
            receipts[run_name] = receipt
    selection = {"schema": "versematrix.matching-selection.v0.5", "selected_at_utc": now(),
        "primary_arm": config["primary_arm"], "deployment_seed": config["deployment_seed"],
        "selection_scope": "dev_only_not_independent_confirmation", "config_sha256": freeze["config_sha256"],
        "criterion": config["evaluation"]["checkpoint_selection"], "arms": {}}
    arm_metrics = {}
    for arm in config["arms"]:
        name = arm["id"]
        if any(len(checkpoint_ranks[name][step]) != len(config["seeds"]) for step in training["checkpoint_updates"]):
            raise ValueError("a declared seed/checkpoint is missing")
        chosen, values = select_shared_checkpoint(dev, checkpoint_ranks[name])
        models = checkpoints[name][chosen]
        if [m["seed"] for m in models] != config["seeds"]:
            raise ValueError("seed result order changed")
        for model in models:
            if sha(output / model["path"]) != model["sha256"]:
                raise ValueError("checkpoint changed before common selection")
        selection["arms"][name] = {"step": chosen,
            "exact_family_recall_at_16": {"numerator": values[chosen].numerator, "denominator": values[chosen].denominator},
            "checkpoint_values": [{"step": step, "family_recall_at_16": float(value),
                "exact": {"numerator": value.numerator, "denominator": value.denominator}} for step, value in sorted(values.items())],
            "models": [{k: m[k] for k in ("seed", "path", "sha256", "predictions_path", "predictions_sha256")} for m in models]}
        arm_metrics[name] = {"selected_step": chosen,
            "seed_average": metric_bundle(dev, checkpoint_ranks[name][chosen], data["indices"], data["prefix_counts"]),
            "by_seed": [{"seed": m["seed"], "metrics": m["dev_metrics"], "parameters": m["parameters"], "accounting_at_selected_checkpoint": m["accounting"]} for m in models]}
    write_new(output / "selection_frozen.json", selection)
    primary = selection["arms"][config["primary_arm"]]
    deployment = next(entry for entry in primary["models"] if entry["seed"] == config["deployment_seed"])
    write_new(output / "deployment_candidate.json", {
        "schema": "versematrix.experimental-completion-candidate.v0.5",
        "status": "experimental", "arm": config["primary_arm"],
        "seed": config["deployment_seed"], "step": primary["step"],
        "model_path": deployment["path"], "model_sha256": deployment["sha256"],
        "seed_selection": "fixed_seed_zero_before_training_not_best_seed",
        "selection_path": "selection_frozen.json", "selection_sha256": sha(output / "selection_frozen.json"),
        "candidate_counts": data["coverage"]["candidate_counts"],
        "candidate_set_sha256": data["coverage"]["candidate_set_sha256"],
        "comparison_baseline": {"id": "strong_statistical", "alignment_weight": .25, "boundary_weight": .25},
        "scope": "source_restoration_not_prosody_certification_or_validated_literary_quality",
        "independent_confirmation": "NOT_PERFORMED"})
    dev_index = {t["task_id"]: t for t in dev}
    donors = {row["task_id"]: dev_index[row["donor_task_id"]] for row in mapping["mapping"]}
    diagnostic_metrics = {}
    diagnostic_ranks = {}
    for name, mask, changed_context in (("zero_context", "both", None), ("mismatched_context", "none", donors)):
        rank_vectors = []
        by_seed = []
        for entry in primary["models"]:
            path = output / entry["path"]
            if sha(path) != entry["sha256"]: raise ValueError("selected immutable checkpoint changed")
            model = ContextSpanMatcher.load(path)
            rows = evaluate_model(model, dev, data["candidates"], config["evaluation"]["batch_size"], mask=mask, donors=changed_context)
            write_rows(output / "diagnostics" / f"{name}_seed{entry['seed']}.jsonl", rows)
            ranks = aligned_ranks(dev, rows)
            rank_vectors.append(ranks)
            by_seed.append({"seed": entry["seed"], "metrics": metric_bundle(dev, [ranks], data["indices"], data["prefix_counts"])})
        diagnostic_ranks[name] = rank_vectors
        diagnostic_metrics[name] = {"kind": "frozen_selected_checkpoint_inference_diagnostic_no_training",
            "seed_average": metric_bundle(dev, rank_vectors, data["indices"], data["prefix_counts"]), "by_seed": by_seed}
    base_exact = exact_seed_family_recall(dev, [baseline_ranks["strong_statistical"]])
    full_ranks = checkpoint_ranks[config["primary_arm"]][primary["step"]]
    full_exact = exact_seed_family_recall(dev, full_ranks)
    mismatch_exact = exact_seed_family_recall(dev, diagnostic_ranks["mismatched_context"])
    seed_gains = [exact_seed_family_recall(dev, [ranks]) - base_exact for ranks in full_ranks]
    gate_config = config["resource_gate"]
    gain_ok = full_exact - base_exact >= Fraction(str(gate_config["primary_mean_gain_over_strong_baseline_at_least"]))
    seeds_ok = all(value > 0 for value in seed_gains)
    mismatch_ok = full_exact - mismatch_exact >= Fraction(str(gate_config["primary_mean_minus_mismatched_context_at_least"]))
    gate = {"scope": gate_config["scope"], "primary_mean_gain_over_strong_baseline": float(full_exact-base_exact),
        "seed_gains_over_strong_baseline": [float(value) for value in seed_gains],
        "primary_mean_minus_mismatched_context": float(full_exact-mismatch_exact),
        "checks": {"mean_gain_threshold": gain_ok, "all_three_seed_gains_positive": seeds_ok,
                   "context_mismatch_drop_threshold": mismatch_ok},
        "passed": gain_ok and seeds_ok and mismatch_ok,
        "decision": "ELIGIBLE_FOR_NEW_INDEPENDENT_CONFIRMATION" if gain_ok and seeds_ok and mismatch_ok else "DO_NOT_ALLOCATE_CONFIRMATION_FROM_THIS_DEV_RUN",
        "formal_literary_quality_gate": "NOT_EVALUATED", "independent_confirmation": "NOT_PERFORMED"}
    check_frozen(output)
    report = {"schema": "versematrix.context-span-matching-results.v0.5", "status": "completed_dev_only",
        "completed_at_utc": now(), "config_sha256": freeze["config_sha256"], "coverage": data["coverage"],
        "baselines": baseline_metrics, "arms": arm_metrics, "diagnostics": diagnostic_metrics,
        "resource_gate": gate, "selection_sha256": sha(output / "selection_frozen.json"),
        "training_accounting": {"formal_runs": len(receipts),
            "optimizer_updates": sum(r["accounting"]["optimizer_updates"] for r in receipts.values()),
            "sample_presentations": sum(r["accounting"]["sample_presentations"] for r in receipts.values()),
            "training_seconds": sum(r["training_seconds"] for r in receipts.values()),
            "wall_seconds_including_baselines_and_evaluation": time.perf_counter() - started,
            "train_only_preflight_cost": "reported separately; not an independent experiment or included in these formal counts"},
        "limits": ["Only repeatedly used development data were evaluated; no independent confirmation is claimed.",
            "Checkpoint selection is based on development labels and its observed gains are adaptive.",
            "The primary arm was fixed before training; control-arm performance cannot replace its outcome.",
            "The fixed W and all development tasks, including OOV targets, remain in the declared denominators.",
            "Candidate phrases are exact adjacent source spans; linguistic word segmentation is not assumed.",
            "Shape-only source poems are not certified strict lushi; restoration agreement is not literary quality.",
            "Upper-only and suffix-only are independently trained controls; zero/mismatched contexts are frozen-model inference diagnostics.",
            "Immutable checkpoints and source hashes support replay of complete rankings; no selected checkpoint was copied over another file."]}
    write_new(output / "report.json", report)
    return {"status": report["status"], "resource_gate": gate,
            "selected_steps": {name: value["step"] for name, value in selection["arms"].items()},
            "training_accounting": report["training_accounting"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.out.resolve(), args.config.resolve()) if args.action == "prepare" else run(args.out.resolve())
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
