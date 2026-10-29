"""Independently verify a completed unprompted poem trial; never generate a poem.

Rebuilds source spans, scalar language probabilities and selected-path scores
directly from train records. Does not call the generator or replay its beam.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_snapshot(root):
    poems, loading = [], Counter()
    marker = re.compile(rb'"split"\s*:\s*"(train|dev|test)"')
    with (root / "data/corpus_v03/poems.jsonl").open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            found = marker.search(raw)
            require(found is not None, "unknown source split")
            split = found.group(1).decode()
            loading["source_" + split] += 1
            if split != "train":
                continue
            poem = json.loads(raw)
            require(poem["split"] == "train", "split mismatch")
            if poem["length"] != 5:
                loading["train_other_length_skipped"] += 1
                continue
            require(len(poem["lines"]) == 8 and all(len(x) == 5 for x in poem["lines"]),
                    "source shape")
            poems.append(poem)
    loading["used_five_training_poems"] = len(poems)
    return sorted(poems, key=lambda p: p["poem_id"]), dict(loading)


def reconstruct_resources(poems, reference, excluded, minimum):
    raw = {"prefixes": defaultdict(list), "suffixes": defaultdict(list)}
    lines = set()
    for poem in poems:
        for row, line in enumerate(poem["lines"]):
            lines.add(line)
            raw["prefixes"][line[:2]].append(dict(poem_id=poem["poem_id"], row=row, start=0, end=2))
            raw["suffixes"][line[2:]].append(dict(poem_id=poem["poem_id"], row=row, start=2, end=5))
    eligible, rejections = {}, {}
    for kind in raw:
        retained, reasons = {}, Counter()
        for text, origins in sorted(raw[kind].items()):
            if len(origins) < minimum:
                reasons["below_minimum_count"] += 1
                continue
            if set(text) & excluded:
                reasons["editorial_issue"] += 1
                continue
            memberships = [reference.get(char, []) for char in text]
            if any(not entries for entries in memberships):
                reasons["unknown_character"] += 1
                continue
            alternatives = [{entry["tone"] for entry in entries} for entries in memberships]
            if any(len(tones) != 1 or not tones <= {"P", "Z"} for tones in alternatives):
                reasons["ambiguous_or_invalid_tone"] += 1
                continue
            tones = "".join(sorted(values)[0] for values in alternatives)
            groups = {entry["group_id"] for entry in memberships[-1] if entry["tone"] == "P"}
            group = sorted(groups)[0] if tones[-1] == "P" and len(groups) == 1 else None
            retained[text] = dict(text=text, count=len(origins), sources=origins,
                tones=tones, rhyme_group=group, character_memberships=memberships,
                contextual_reading_verified=False)
        reasons["eligible"], reasons["raw_unique"] = len(retained), len(raw[kind])
        eligible[kind], rejections[kind] = retained, dict(reasons)
    return eligible, rejections, lines


class ScalarLanguageOracle:
    def __init__(self, poems):
        characters = sorted(set("".join(line for poem in poems for line in poem["lines"])))
        self.ids = {char: i + 1 for i, char in enumerate(characters)}
        self.vocabulary_size = len(characters) + 1
        self.eos = self.vocabulary_size
        self.tables = []
        for reverse in (False, True):
            uni, bi, tri, context1, context2 = (Counter() for _ in range(5))
            for poem in poems:
                for line in poem["lines"]:
                    sequence = [self.ids[char] for char in line]
                    if reverse:
                        sequence.reverse()
                    a = b = -1
                    for target in sequence + [self.eos]:
                        uni[target] += 1
                        bi[(b, target)] += 1
                        tri[(a, b, target)] += 1
                        context1[b] += 1
                        context2[(a, b)] += 1
                        a, b = b, target
            self.tables.append((uni, bi, tri, context1, context2, sum(uni.values())))

    @lru_cache(maxsize=500000)
    def log_probability(self, direction, a, b, target):
        uni, bi, tri, context1, context2, total = self.tables[direction]
        smoothing = .1 * (self.vocabulary_size + 1)
        probabilities = ((uni[target] + .1) / (total + smoothing),
                         (bi[(b, target)] + .1) / (context1[b] + smoothing),
                         (tri[(a, b, target)] + .1) / (context2[(a, b)] + smoothing))
        return math.log(sum(w * p for w, p in zip((.1, .3, .6), probabilities)))

    @lru_cache(maxsize=50000)
    def score(self, text):
        directions = []
        for direction in (0, 1):
            sequence = [self.ids.get(char, 0) for char in text]
            if direction:
                sequence.reverse()
            a = b = -1
            total = 0.0
            for target in sequence + [self.eos]:
                total += self.log_probability(direction, a, b, target)
                a, b = b, target
            directions.append(total / 6)
        return .5 * sum(directions)


class ScalarCoherenceOracle:
    def __init__(self, poems):
        self.n = len(poems)
        self.df, self.joint = Counter(), Counter()
        for poem in poems:
            chars = sorted(set("".join(poem["lines"])))
            self.df.update(chars)
            for i, first in enumerate(chars):
                for second in chars[i + 1:]:
                    self.joint[(first, second)] += 1

    @lru_cache(maxsize=50000)
    def pair(self, first, second):
        terms = []
        for a in first:
            for b in second:
                if a == b:
                    continue
                denominator = self.df[a] * self.df[b]
                numerator = self.n * self.joint[tuple(sorted((a, b)))]
                terms.append(math.log1p(numerator / denominator) if denominator else 0.0)
        return sum(terms) / len(terms) if terms else 0.0


def template(identity):
    found = re.fullmatch(r"L5-([PZ])-R([01])", identity)
    require(found is not None, "unsupported template identity")
    start, first_rhyme = found.group(1), found.group(2) == "1"
    opposite = "Z" if start == "P" else "P"
    seconds = [start, opposite, opposite, start, start, opposite, opposite, start]
    rhyme_rows = [0, 1, 3, 5, 7] if first_rhyme else [1, 3, 5, 7]
    forms = {("P", "P"): "PPZZP", ("P", "Z"): "PPPZZ",
             ("Z", "P"): "ZZZPP", ("Z", "Z"): "ZZPPZ"}
    return [forms[(tone, "P" if row in rhyme_rows else "Z")]
            for row, tone in enumerate(seconds)], rhyme_rows


def audit(root, run):
    freeze = read_json(run / "freeze.json")
    result = read_json(run / "result.json")
    stored = read_json(run / "resources.json")
    pool_file = read_json(run / "candidate_pools.json")
    pools = pool_file["pools"]
    rhyme = read_json(run / "rhyme_selection.json")
    search = read_json(run / "search.json")
    config = freeze["config"]
    require(result["status"] == "completed", "trial did not complete")
    require(freeze["inputs_filled_by_human"] == result["inputs_filled_by_human"] == [], "content input")
    require(config["seed"] == 0 and config["randomness"] == "none_deterministic_lexical_ties", "seed policy")
    require(freeze["formal_run_number"] == 1, "formal run label")
    require(result["no_manual_output_editing"] is True, "editing declaration")
    require(not (run / "failure.json").exists(), "failure record in completed run")
    for name, identity in freeze["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "frozen path escapes project")
        require(checksum(path) == identity["sha256"], "changed frozen file " + name)
        require(path.stat().st_size == identity["size_bytes"], "frozen size " + name)
    config_path = root / "configs/free_generation_v0.6.json"
    require(checksum(config_path) == freeze["config_sha256"], "config digest")
    require(read_json(config_path) == config, "config content")
    for flag in ("contextual_reading_verified", "antithesis_verified", "literary_quality_evaluated",
                 "global_optimality_claim"):
        require(result[flag] is False, "unsupported certification " + flag)

    poems, loading = source_snapshot(root)
    require(loading == result["loading"], "loading counts")
    reference = read_json(root / "data/reference_v02/pingshui_106_memberships.json")["characters"]
    excluded = {x["character"] for x in read_json(root / "data/reference_v02/editorial_issues.json")["issues"]
                if x.get("character") is not None}
    resources, rejected, train_lines = reconstruct_resources(poems, reference, excluded, config["minimum_span_count"])
    for kind in ("prefixes", "suffixes"):
        require(resources[kind] == stored[kind], "source resources differ: " + kind)
    require(rejected == stored["rejections"], "resource rejection counts")
    require(sorted(train_lines) == stored["train_lines"], "complete training line index")
    require(sorted(excluded) == stored["excluded_editorial_characters"], "editorial exclusion")
    language, coherence = ScalarLanguageOracle(poems), ScalarCoherenceOracle(poems)
    require(result["language_model"]["fit_poems"] == len(poems), "LM training population")
    require(result["language_model"]["fit_poem_ids_sha256"] == canonical_digest(tuple(p["poem_id"] for p in poems)), "LM source IDs")
    require(result["language_model"]["vocabulary_size_with_unk"] == language.vocabulary_size, "LM vocabulary")
    errors, measures = [], 0

    def close(actual, expected, label):
        nonlocal measures
        require(isinstance(actual, (int, float)) and math.isfinite(actual), "nonfinite " + label)
        error = abs(actual - expected)
        errors.append(error)
        measures += 1
        require(error <= 1e-12 * max(1., abs(expected)), "scalar mismatch " + label)

    def verify_candidate(candidate, pattern, group):
        text = candidate["text"]
        require(len(text) == 5 and len(set(text)) == 5, "line shape/repeated character")
        require(text not in train_lines, "complete training line copied")
        require(candidate["prefix"] + candidate["suffix"] == text, "fragment assembly")
        p, s = resources["prefixes"][candidate["prefix"]], resources["suffixes"][candidate["suffix"]]
        require(p["tones"] + s["tones"] == pattern == candidate["tones"], "candidate tones")
        if group != "nonrhyming":
            require(s["rhyme_group"] == group == candidate["rhyme_group"], "candidate rhyme")
        else:
            require(pattern[-1] == "Z" and candidate["rhyme_group"] is None, "nonrhyme candidate")
        close(candidate["lm_score"], language.score(text), "candidate LM")

    retained = 0
    for key, candidates in pools.items():
        pattern, group = key.split("/", 1)
        require(len(candidates) <= config["line_pool_size"], "pool cap")
        require(len({x["text"] for x in candidates}) == len(candidates), "duplicate pool member")
        require(candidates == sorted(candidates, key=lambda x: (-x["lm_score"], x["text"])), "pool order")
        for candidate in candidates:
            verify_candidate(candidate, pattern, group)
        retained += len(candidates)
    counts = pool_file["counts"]
    require(counts == result["pool_counts"], "pool counts copied into result")
    require(counts["retained_pool_candidates"] == retained, "retained pool count")
    require(counts["cartesian_prefix_suffix_pairs"] == len(resources["prefixes"]) * len(resources["suffixes"]), "raw Cartesian count")
    require(counts["lm_scored_candidates"] == retained + counts["per_ending_or_nonrhyme_shortlist_pruned"] + counts["rhymed_pool_cap_pruned"], "pruning accounting")

    # Reconstruct the deterministic choice of rhyme from the fixed pools.
    group_names = sorted({key.split("/", 1)[1] for key in pools if not key.endswith("/nonrhyming")})
    ranking = []
    for group in group_names:
        a = {x["text"][-1] for x in pools.get("PPZZP/" + group, [])}
        b = {x["text"][-1] for x in pools.get("ZZZPP/" + group, [])}
        ranking.append(dict(group=group, PPZZP_endings=sorted(a), ZZZPP_endings=sorted(b),
            frequency=sum(x["count"] for x in resources["suffixes"].values() if x["rhyme_group"] == group),
            minimum_type_ending_coverage=min(len(a), len(b)), union_ending_coverage=len(a | b),
            feasible=len(a) >= 3 and len(b) >= 3 and len(a | b) >= 5))
    ranking.sort(key=lambda x: (-int(x["feasible"]), -x["minimum_type_ending_coverage"],
                               -x["union_ending_coverage"], -x["frequency"], x["group"]))
    require(ranking == rhyme["ranking"], "rhyme ranking")
    require(ranking and ranking[0]["feasible"], "no eligible rhyme")
    group = ranking[0]["group"]
    require(group == rhyme["selected"] == result["rhyme_group"], "selected rhyme")

    template_ids = ["L5-P-R0", "L5-P-R1", "L5-Z-R0", "L5-Z-R1"]
    require([x["template_id"] for x in search["template_searches"]] == template_ids, "template enumeration")
    paths_verified = 0
    for outcome in search["template_searches"]:
        core, rhyme_rows = template(outcome["template_id"])
        require(core == outcome["core"] and rhyme_rows == outcome["rhyme_rows"], "template grammar")
        previous_states = 1
        for row, receipt in enumerate(outcome["rows"]):
            require(row == receipt["row"], "beam row index")
            pool_group = group if row in rhyme_rows else "nonrhyming"
            pool_size = len(pools.get(core[row] + "/" + pool_group, []))
            require(receipt["candidate_pool_size"] == pool_size, "beam pool size")
            c = receipt["counts"]
            attempted = c.get("attempted_extensions", 0)
            require(attempted == previous_states * pool_size, "attempt count")
            rejected_count = sum(c.get(reason, 0) for reason in ("repeated_line", "character_use_limit",
                "repeated_rhyme_character", "middle_couplet_same_position_character"))
            require(attempted == c["legal_extensions"] + rejected_count, "extension accounting")
            require(c["retained_states"] == min(c["legal_extensions"], config["beam_width"]), "beam retained")
            require(c["beam_pruned"] == c["legal_extensions"] - c["retained_states"], "beam pruning")
            previous_states = c["retained_states"]
        if outcome["status"] != "complete":
            require(outcome["best"] is None and previous_states == 0, "infeasible status")
            continue
        require(len(outcome["rows"]) == 8 and previous_states > 0, "complete status")
        best = outcome["best"]
        require(len(best["lines"]) == len(best["selected"]) == len(best["terms"]) == 8, "path shape")
        previous, total = [], 0.0
        for row, (text, candidate, terms) in enumerate(zip(best["lines"], best["selected"], best["terms"])):
            require(candidate["text"] == text, "path candidate text")
            pool_group = group if row in rhyme_rows else "nonrhyming"
            require(candidate in pools[core[row] + "/" + pool_group], "selected candidate absent from pool")
            require(text not in previous, "repeated selected line")
            used = Counter("".join(previous))
            require(max(Counter("".join(previous) + text).values()) <= config["max_character_uses"], "selected character limit")
            if row in rhyme_rows:
                require(text[-1] not in {previous[i][-1] for i in rhyme_rows if i < row}, "selected repeated rhyme")
            if row in (3, 5):
                require(all(a != b for a, b in zip(previous[-1], text)), "middle same position")
            lm = language.score(text)
            raw_coherence = sum(coherence.pair(old, text) for old in previous) / len(previous) if previous else 0.0
            repeated = sum(used[char] for char in text)
            expected = dict(lm=lm, coherence_raw=raw_coherence,
                coherence_weighted=config["coherence_weight"] * raw_coherence,
                previous_character_occurrences=repeated,
                repetition_penalty=config["repetition_weight"] * repeated,
                increment=lm + config["coherence_weight"] * raw_coherence - config["repetition_weight"] * repeated)
            require(set(terms) == set(expected), "score term keys")
            for key in expected:
                close(terms[key], expected[key], "selected " + key)
            total += expected["increment"]
            previous.append(text)
        close(best["score"], total, "complete path score")
        paths_verified += 1
    complete = [x for x in search["template_searches"] if x["status"] == "complete"]
    complete.sort(key=lambda x: (-x["best"]["score"], x["template_id"], x["best"]["lines"]))
    require(complete and complete[0] == search["selected"], "selected template outcome")
    selected = complete[0]
    require(result["template_id"] == selected["template_id"], "result template")
    require(result["lines"] == selected["best"]["lines"], "result lines differ from trace")
    close(result["score"], selected["best"]["score"], "result score")
    require(len(result["line_details"]) == 8, "line provenance count")
    for row, detail in enumerate(result["line_details"]):
        candidate = selected["best"]["selected"][row]
        expected = dict(row=row, **candidate, score_terms=selected["best"]["terms"][row],
            prefix_provenance=resources["prefixes"][candidate["prefix"]],
            suffix_provenance=resources["suffixes"][candidate["suffix"]])
        require(detail == expected, "complete selected source provenance")
    audit_record = result["candidate_tone_matrix_check"]
    require(audit_record["status"] == "matches_supported_profile" and result["template_id"] in audit_record["matches"], "candidate tone check")
    require(audit_record["conditional_on_unverified_reference_memberships"] is True, "unverified reading boundary")
    return dict(status="PASS_INDEPENDENT_SOURCE_AND_SELECTED_PATH_AUDIT",
        audited_utc=datetime.now(timezone.utc).isoformat(),
        formal_generation_executed_by_auditor=False, real_output_altered=False,
        used_five_training_poems=len(poems), source_span_records=sum(map(len, resources.values())),
        source_span_occurrences=sum(record["count"] for items in resources.values() for record in items.values()),
        retained_candidate_scores_recomputed=retained,
        complete_template_paths_recomputed=paths_verified, numerical_measures=measures,
        max_absolute_numeric_difference=max(errors, default=0.),
        selected_poem_sha256=canonical_digest(result["lines"]),
        scope="all source spans and provenance, every retained line score, four template trace accounting, completed selected-path constraints/scores, and final outcome selection",
        limits=["Does not independently replay every discarded beam state or certify global optimality.",
                "Reference membership uniqueness does not certify contextual historical pronunciation.",
                "No antithesis or literary-quality certification.",
                "Recorded freeze chronology and first-run label cannot prove absence of unrecorded activity."],
        files={name: checksum(run / name) for name in ("freeze.json", "result.json", "resources.json", "candidate_pools.json", "rhyme_selection.json", "search.json")},
        auditor_sha256=checksum(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.project_dir.resolve(), args.run.resolve())
    destination = args.output or args.run / "independent_audit.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
