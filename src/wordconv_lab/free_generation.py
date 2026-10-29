"""Unprompted, train-only 2+3 span recombination for one five-character poem.

This new statistical baseline is not the v0.5 convolution model. Reference
memberships constrain candidate tones; no contextual reading or literary gold
is invented. A bounded beam is a heuristic, not a global optimality proof.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from .prosody import FIVE_LINE, all_templates, audit_tones
from .restoration import BidirNGram
from .retrieval import _FastBidir


SCHEMA = "versematrix.free-generation.statistical-trial.v0.6"
SPLIT = re.compile(rb'"split"\s*:\s*"(train|dev|test)"')


def write_new_json(path: Path, value: dict) -> None:
    """Exclusive file creation: never overwrite a previous observation."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_five_training(path: Path) -> tuple[list[dict], dict]:
    poems, counts = [], Counter()
    with Path(path).open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            match = SPLIT.search(raw)
            if match is None:
                raise ValueError("source row lacks a recognized split")
            split = match.group(1).decode("ascii")
            counts["source_" + split] += 1
            if split != "train":
                continue  # Held-out rows are not JSON-decoded.
            poem = json.loads(raw)
            if poem.get("split") != "train":
                raise ValueError("split marker does not match parsed row")
            if poem.get("length") == 5:
                poems.append(poem)
            else:
                counts["train_other_length_skipped"] += 1
    counts["used_five_training_poems"] = len(poems)
    return poems, dict(counts)


def collect_spans(poems: list[dict], reference: dict, editorial: dict,
                  min_count: int = 2) -> dict:
    if type(min_count) is not int or min_count < 2:
        raise ValueError("minimum span count must be an integer at least two")
    if not poems or any(p.get("split") != "train" for p in poems):
        raise ValueError("supply nonempty training-only poems")
    if len({p.get("poem_id") for p in poems}) != len(poems):
        raise ValueError("duplicate training poem IDs")
    reference = reference["characters"]
    excluded = {item["character"] for item in editorial["issues"]
                if item.get("character") is not None}
    source_spans = {"prefixes": defaultdict(list), "suffixes": defaultdict(list)}
    train_lines = set()
    for poem in sorted(poems, key=lambda p: p["poem_id"]):
        lines = poem.get("lines", [])
        if (poem.get("length") != 5 or len(lines) != 8
                or any(not isinstance(line, str) or len(line) != 5 for line in lines)):
            raise ValueError("requires eight five-character lines per training poem")
        for row, line in enumerate(lines):
            train_lines.add(line)
            for kind, start, end in (("prefixes", 0, 2), ("suffixes", 2, 5)):
                source_spans[kind][line[start:end]].append(
                    {"poem_id": poem["poem_id"], "row": row,
                     "start": start, "end": end})
    result = {"prefixes": {}, "suffixes": {}, "train_lines": sorted(train_lines),
              "rejections": {}, "excluded_editorial_characters": sorted(excluded),
              "contextual_reading_status": "not_verified",
              "source_policy": "exact code points; no spelling normalization"}
    for kind, texts in source_spans.items():
        rejected = Counter()
        for text, sources in sorted(texts.items()):
            if len(sources) < min_count:
                rejected["below_minimum_count"] += 1
                continue
            if any(c in excluded for c in text):
                rejected["editorial_issue"] += 1
                continue
            memberships = [reference.get(c, []) for c in text]
            if any(not m for m in memberships):
                rejected["unknown_character"] += 1
                continue
            tones = [{m["tone"] for m in entries} for entries in memberships]
            if any(len(t) != 1 or not t <= {"P", "Z"} for t in tones):
                rejected["ambiguous_or_invalid_tone"] += 1
                continue
            tone_string = "".join(next(iter(t)) for t in tones)
            final_groups = {entry["group_id"] for entry in memberships[-1]
                            if entry["tone"] == "P"}
            rhyme_group = (next(iter(final_groups))
                           if tone_string[-1] == "P" and len(final_groups) == 1 else None)
            result[kind][text] = {
                "text": text, "count": len(sources), "sources": sources,
                "tones": tone_string, "rhyme_group": rhyme_group,
                "character_memberships": memberships,
                "contextual_reading_verified": False,
            }
        result["rejections"][kind] = dict(rejected)
        result["rejections"][kind]["eligible"] = len(result[kind])
        result["rejections"][kind]["raw_unique"] = len(texts)
    return result


def line_rejection_reason(text: str, train_lines: set[str]) -> str | None:
    if len(text) != 5:
        return "wrong_length"
    if text in train_lines:
        return "complete_training_line"
    if len(set(text)) != 5:
        return "within_line_repeated_character"
    return None


def extension_rejection_reason(lines: tuple[str, ...], text: str, row: int,
                               rhyme_rows: tuple[int, ...],
                               max_character_uses: int = 2) -> str | None:
    if row != len(lines):
        raise ValueError("row must follow the existing lines")
    if text in lines:
        return "repeated_line"
    counts = Counter("".join(lines) + text)
    if any(n > max_character_uses for n in counts.values()):
        return "character_use_limit"
    if row in rhyme_rows and any(lines[r][-1] == text[-1]
                                for r in rhyme_rows if r < row):
        return "repeated_rhyme_character"
    if row in (3, 5) and any(a == b for a, b in zip(lines[-1], text)):
        return "middle_couplet_same_position_character"
    return None


def score_lines(model: BidirNGram, texts: list[str]) -> np.ndarray:
    """Exactly the existing interpolated forward/reverse LM, without alignment."""
    if not texts:
        return np.zeros(0, dtype=np.float64)
    if any(len(text) != 5 for text in texts):
        raise ValueError("only five-character candidates supported")
    fast = _FastBidir(model, {})
    ids = np.asarray([model.encode(text) for text in texts], dtype=np.int64)
    totals = []
    for direction in (0, 1):
        seq = ids if direction == 0 else ids[:, ::-1]
        a = np.full(len(texts), -1, dtype=np.int64)
        b = a.copy()
        total = np.zeros(len(texts), dtype=np.float64)
        for col in range(6):
            target = seq[:, col] if col < 5 else np.full(len(texts), model.eos)
            total = total + fast.log_probability(direction, a, b, target)
            a, b = b, target
        totals.append(total / 6)
    result = .5 * (totals[0] + totals[1])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite line language score")
    return result


def _top_with_ending_coverage(candidates: list[dict], limit: int) -> list[dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["text"][-1]].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda x: (-x["lm_score"], x["text"]))
    endings = sorted(grouped, key=lambda c: (-grouped[c][0]["lm_score"], c))
    result = []
    for depth in range(max((len(v) for v in grouped.values()), default=0)):
        for ending in endings:
            if depth < len(grouped[ending]):
                result.append(grouped[ending][depth])
                if len(result) == limit:
                    return sorted(result, key=lambda x: (-x["lm_score"], x["text"]))
    return sorted(result, key=lambda x: (-x["lm_score"], x["text"]))


def build_line_pools(resources: dict, model: BidirNGram, config: dict) -> tuple[dict, dict]:
    prefixes, suffixes = resources["prefixes"], resources["suffixes"]
    train_lines = set(resources["train_lines"])
    prefixes_by_tone, suffixes_by_tone = defaultdict(list), defaultdict(list)
    for text, record in prefixes.items():
        prefixes_by_tone[record["tones"]].append(text)
    for text, record in suffixes.items():
        suffixes_by_tone[record["tones"]].append(text)
    retained, counts = defaultdict(list), Counter()
    # A finite per-ending shortlist permits rhyme diversity without retaining
    # every Cartesian combination. All eligible combinations are LM-scored.
    raw_total = len(prefixes) * len(suffixes)
    counts["cartesian_prefix_suffix_pairs"] = raw_total
    compatible_total = 0
    for pattern in sorted(set(FIVE_LINE.values())):
        pending = []
        per_ending = defaultdict(list)
        prefix_list = sorted(prefixes_by_tone[pattern[:2]])
        suffix_list = sorted(suffixes_by_tone[pattern[2:]])
        compatible_total += len(prefix_list) * len(suffix_list)

        def flush():
            if not pending:
                return
            scores = score_lines(model, [item["text"] for item in pending])
            for item, score in zip(pending, scores):
                item["lm_score"] = float(score)
                group = item["rhyme_group"] or "nonrhyming"
                bucket = (group, item["text"][-1] if pattern[-1] == "P" else "all")
                per_ending[bucket].append(item)
            counts["lm_scored_candidates"] += len(pending)
            pending.clear()

        for prefix in prefix_list:
            for suffix in suffix_list:
                text = prefix + suffix
                rejection = line_rejection_reason(text, train_lines)
                if rejection:
                    counts[rejection] += 1
                    continue
                group = suffixes[suffix]["rhyme_group"]
                if pattern[-1] == "P" and group is None:
                    counts["rhyme_endpoint_not_unique_level_group"] += 1
                    continue
                pending.append({"text": text, "prefix": prefix, "suffix": suffix,
                                "tones": pattern,
                                "rhyme_group": group if pattern[-1] == "P" else None})
                if len(pending) == config["score_batch_size"]:
                    flush()
        flush()
        for (group, ending), values in sorted(per_ending.items()):
            values.sort(key=lambda x: (-x["lm_score"], x["text"]))
            limit = (config["candidates_per_ending"] if group != "nonrhyming"
                     else config["line_pool_size"])
            counts["per_ending_or_nonrhyme_shortlist_pruned"] += max(0, len(values) - limit)
            retained[(pattern, group)].extend(values[:limit])
    counts["tone_pattern_mismatch"] = raw_total - compatible_total
    pools = {}
    for (pattern, group), candidates in sorted(retained.items()):
        selected = (_top_with_ending_coverage(candidates, config["line_pool_size"])
                    if group != "nonrhyming" else candidates)
        counts["rhymed_pool_cap_pruned"] += len(candidates) - len(selected)
        pools[pattern + "/" + group] = selected
    counts["retained_pool_candidates"] = sum(map(len, pools.values()))
    if (counts["lm_scored_candidates"] != counts["retained_pool_candidates"]
            + counts["per_ending_or_nonrhyme_shortlist_pruned"]
            + counts["rhymed_pool_cap_pruned"]):
        raise AssertionError("candidate accounting mismatch")
    return pools, dict(counts)


def choose_rhyme(pools: dict, resources: dict) -> tuple[str | None, list[dict]]:
    groups = sorted({key.split("/", 1)[1] for key in pools
                     if not key.endswith("/nonrhyming")})
    records = []
    for group in groups:
        a = {x["text"][-1] for x in pools.get("PPZZP/" + group, [])}
        b = {x["text"][-1] for x in pools.get("ZZZPP/" + group, [])}
        # This stronger check supports ALL FOUR templates, including a first
        # rhyming line: either type may need three distinct endings.
        feasible = len(a) >= 3 and len(b) >= 3 and len(a | b) >= 5
        frequency = sum(r["count"] for r in resources["suffixes"].values()
                        if r["rhyme_group"] == group)
        records.append({"group": group, "PPZZP_endings": sorted(a),
                        "ZZZPP_endings": sorted(b), "frequency": frequency,
                        "minimum_type_ending_coverage": min(len(a), len(b)),
                        "union_ending_coverage": len(a | b), "feasible": feasible})
    records.sort(key=lambda x: (-int(x["feasible"]),
                               -x["minimum_type_ending_coverage"],
                               -x["union_ending_coverage"], -x["frequency"], x["group"]))
    return (records[0]["group"] if records and records[0]["feasible"] else None), records


class Coherence:
    """Automatic training-poem character co-occurrence, no hand-picked themes."""
    def __init__(self, poems: list[dict]):
        self.n = len(poems)
        self.df, self.joint = Counter(), Counter()
        for poem in poems:
            characters = sorted(set("".join(poem["lines"])))
            self.df.update(characters)
            for i, a in enumerate(characters):
                for b in characters[i+1:]:
                    self.joint[(a, b)] += 1

    @lru_cache(maxsize=200000)
    def pair(self, left: str, right: str) -> float:
        values = []
        for a in left:
            for b in right:
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                denominator = self.df[a] * self.df[b]
                values.append(math.log1p(self.n * self.joint[key] / denominator)
                              if denominator else 0.0)
        return sum(values) / len(values) if values else 0.0


def search_poem(pools: dict, group: str, poems: list[dict], config: dict) -> dict:
    coherence = Coherence(poems)
    outcomes = []
    for template in sorted(all_templates(5), key=lambda t: t.template_id):
        states = [{"lines": (), "selected": (), "score": 0.0, "terms": ()}]
        row_receipts = []
        for row, pattern in enumerate(template.core):
            pool_group = group if row in template.rhyme_rows else "nonrhyming"
            candidates = pools.get(pattern + "/" + pool_group, [])
            counts = Counter()
            extensions = []
            for state in states:
                used = Counter("".join(state["lines"]))
                for candidate in candidates:
                    counts["attempted_extensions"] += 1
                    text = candidate["text"]
                    reason = extension_rejection_reason(state["lines"], text, row,
                                template.rhyme_rows, config["max_character_uses"])
                    if reason:
                        counts[reason] += 1
                        continue
                    raw_coherence = (sum(coherence.pair(old, text) for old in state["lines"])
                                     / len(state["lines"]) if state["lines"] else 0.0)
                    repeated = sum(used[c] for c in text)
                    coherence_term = config["coherence_weight"] * raw_coherence
                    repetition_penalty = config["repetition_weight"] * repeated
                    increment = candidate["lm_score"] + coherence_term - repetition_penalty
                    term = {"lm": candidate["lm_score"],
                            "coherence_raw": raw_coherence,
                            "coherence_weighted": coherence_term,
                            "previous_character_occurrences": repeated,
                            "repetition_penalty": repetition_penalty,
                            "increment": increment}
                    extensions.append({"lines": state["lines"] + (text,),
                        "selected": state["selected"] + (candidate,),
                        "score": state["score"] + increment,
                        "terms": state["terms"] + (term,)})
            extensions.sort(key=lambda x: (-x["score"], x["lines"]))
            counts["legal_extensions"] = len(extensions)
            counts["beam_pruned"] = max(0, len(extensions) - config["beam_width"])
            states = extensions[:config["beam_width"]]
            counts["retained_states"] = len(states)
            row_receipts.append({"row": row, "candidate_pool_size": len(candidates),
                                 "counts": dict(counts)})
            if not states:
                break
        outcomes.append({"template_id": template.template_id,
                         "rhyme_rows": template.rhyme_rows,
                         "core": template.core,
                         "status": "complete" if states and len(states[0]["lines"]) == 8 else "infeasible",
                         "best": states[0] if states and len(states[0]["lines"]) == 8 else None,
                         "rows": row_receipts})
    complete = [x for x in outcomes if x["status"] == "complete"]
    complete.sort(key=lambda x: (-x["best"]["score"], x["template_id"], x["best"]["lines"]))
    return {"selected": complete[0] if complete else None,
            "template_searches": outcomes,
            "scope": "finite candidate pools and bounded beam; no global optimality claim"}


def run(root: Path, output: Path, config_path: Path) -> dict:
    root, output, config_path = root.resolve(), output.resolve(), config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema", "minimum_span_count", "line_pool_size", "candidates_per_ending",
                "score_batch_size", "beam_width", "max_character_uses",
                "coherence_weight", "repetition_weight", "randomness", "seed"}
    if set(config) != required or config["schema"] != SCHEMA:
        raise ValueError("unsupported or unrecognized generation config")
    for name in ("minimum_span_count", "line_pool_size", "candidates_per_ending",
                 "score_batch_size", "beam_width", "max_character_uses"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("invalid positive integer config: " + name)
    if config["minimum_span_count"] < 2 or config["max_character_uses"] != 2:
        raise ValueError("this protocol requires min_count>=2 and maximum character uses 2")
    if config["seed"] != 0 or config["randomness"] != "none_deterministic_lexical_ties":
        raise ValueError("this trial is deterministic; seed zero is a run label only")
    for name in ("coherence_weight", "repetition_weight"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError("invalid score weight")
    input_names = ["data/corpus_v03/poems.jsonl",
                   "data/reference_v02/pingshui_106_memberships.json",
                   "data/reference_v02/editorial_issues.json"]
    code_names = ["src/wordconv_lab/free_generation.py", "src/wordconv_lab/restoration.py",
                  "src/wordconv_lab/retrieval.py", "src/wordconv_lab/prosody.py",
                  "scripts/generate_free_poem_v06.py"]
    identities = {name: {"sha256": sha256_file(root / name),
                         "size_bytes": (root / name).stat().st_size}
                  for name in input_names + code_names}
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    freeze = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_sha256": sha256_file(config_path),
              "files": identities, "python": platform.python_version(),
              "numpy": np.__version__, "inputs_filled_by_human": [],
              "formal_run_number": 1,
              "generation_method": "new statistical 2+3 span recombination; not v0.5 neural generation",
              "policy": "freeze before reading corpus records or constructing/scoring candidates; first output retained without manual edits"}
    write_new_json(output / "freeze.json", freeze)
    try:
        poems, loading = load_five_training(root / input_names[0])
        reference = json.loads((root / input_names[1]).read_text(encoding="utf-8"))
        editorial = json.loads((root / input_names[2]).read_text(encoding="utf-8"))
        resources = collect_spans(poems, reference, editorial, config["minimum_span_count"])
        write_new_json(output / "resources.json", resources)
        tokens = ["<UNK>"] + sorted({c for poem in poems for line in poem["lines"] for c in line})
        model = BidirNGram({token: i for i, token in enumerate(tokens)}).fit(poems)
        pools, pool_counts = build_line_pools(resources, model, config)
        write_new_json(output / "candidate_pools.json", {"pools": pools, "counts": pool_counts})
        group, rhyme_records = choose_rhyme(pools, resources)
        write_new_json(output / "rhyme_selection.json", {"selected": group, "ranking": rhyme_records})
        search = (search_poem(pools, group, poems, config) if group is not None
                  else {"selected": None, "template_searches": [], "scope": "no feasible rhyme group"})
        write_new_json(output / "search.json", search)
        selected = search["selected"]
        result = {"schema": SCHEMA, "status": "completed" if selected else "infeasible",
                  "method": freeze["generation_method"], "inputs_filled_by_human": [],
                  "generation_scope": "first deterministic run; all forty characters selected by code",
                  "no_manual_output_editing": True, "loading": loading,
                  "language_model": model.summary(), "pool_counts": pool_counts,
                  "rhyme_group": group, "template_id": selected["template_id"] if selected else None,
                  "lines": list(selected["best"]["lines"]) if selected else [],
                  "score": selected["best"]["score"] if selected else None,
                  "line_details": [], "contextual_reading_verified": False,
                  "antithesis_verified": False, "literary_quality_evaluated": False,
                  "whole_line_novelty_scope": "not an exact complete line in the used five-character training corpus",
                  "global_optimality_claim": False,
                  "seconds": time.time() - started}
        if selected:
            for row, (candidate, terms) in enumerate(zip(selected["best"]["selected"],
                                                        selected["best"]["terms"])):
                result["line_details"].append({"row": row, **candidate, "score_terms": terms,
                    "prefix_provenance": resources["prefixes"][candidate["prefix"]],
                    "suffix_provenance": resources["suffixes"][candidate["suffix"]]})
            result["candidate_tone_matrix_check"] = audit_tones(
                [x["tones"] for x in result["line_details"]], profile="strict_core")
            # This audit is explicitly conditional on reference candidates,
            # not a conversion into contextually verified pronunciation.
            result["candidate_tone_matrix_check"]["conditional_on_unverified_reference_memberships"] = True
        if sha256_file(config_path) != freeze["config_sha256"]:
            raise ValueError("config changed during formal generation")
        if any(sha256_file(root / name) != identity["sha256"] for name, identity in identities.items()):
            raise ValueError("input or implementation changed during formal generation")
        result["frozen_files_unchanged"] = True
        write_new_json(output / "result.json", result)
        return result
    except Exception as error:
        write_new_json(output / "failure.json", {"exception": type(error).__name__,
            "message": str(error), "status": "failed_no_parameter_retry_authorized",
            "seconds": time.time() - started})
        raise
