"""Source-restoration corpus construction, never a lushi/quality gold label.

Original code points are retained. Exact text is deduplicated before grouping;
shared complete lines and >= 4/5 within-line bigram-set Jaccard edges join
families before hash splitting. Model-visible tasks expose only the upper line,
masked lower line and suffix; original targets remain separate evaluation data.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import re
import unicodedata

SPLIT_SALT = "versematrix-source-restoration-v03-20261029"
MASK = "□□"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def cjk_character(character):
    name = unicodedata.name(character, "")
    return name.startswith("CJK UNIFIED IDEOGRAPH-") or name.startswith("CJK COMPATIBILITY IDEOGRAPH-")


def source_lines(record):
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs or not all(isinstance(p, str) for p in paragraphs):
        return None, "invalid_paragraphs"
    lines = tuple(part for part in re.split("[，。！？；]", "".join(paragraphs)) if part)
    if len(lines) != 8 or len(lines[0]) not in (5, 7) or any(len(line) != len(lines[0]) for line in lines):
        return None, "not_8_by_5_or_7"
    if not all(cjk_character(c) for line in lines for c in line):
        return None, "non_cjk_body"
    return lines, None


def collect_poems(shards):
    """shards: iterable of (project-relative source path, loaded records)."""
    by_text = {}
    rejected = Counter()
    total = 0
    for source_path, records in shards:
        for source_index, record in enumerate(records):
            total += 1
            if not isinstance(record, dict):
                rejected["invalid_record"] += 1
                continue
            lines, error = source_lines(record)
            if error:
                rejected[error] += 1
                continue
            reference = {"source_path": source_path, "source_index": source_index,
                         "source_id": record.get("id"), "author": record.get("author"),
                         "title": record.get("title")}
            if lines not in by_text:
                by_text[lines] = {"poem_id": "poem_" + digest(lines)[:24],
                    "author": record.get("author"), "title": record.get("title"),
                    "length": len(lines[0]), "lines": list(lines), "source_refs": [],
                    "status": "shape_only_not_verified_lushi"}
            by_text[lines]["source_refs"].append(reference)
    poems = sorted(by_text.values(), key=lambda p: p["poem_id"])
    if len({p["poem_id"] for p in poems}) != len(poems):
        raise ValueError("poem identity collision")
    exact_groups = [{"poem_id": p["poem_id"], "source_refs": p["source_refs"]}
                    for p in poems if len(p["source_refs"]) > 1]
    counts = {"raw_records": total, "rejected": dict(sorted(rejected.items())),
              "accepted_source_records": sum(len(p["source_refs"]) for p in poems),
              "unique_text_poems": len(poems), "exact_duplicate_groups": len(exact_groups),
              "exact_duplicate_excess_records": sum(len(p["source_refs"])-1 for p in poems)}
    return poems, exact_groups, counts


def bigram_set(lines):
    return {line[j:j+2] for line in lines for j in range(len(line)-1)}


def family_edges(poems):
    """Return every qualifying distinct-poem pair with all matching relations."""
    edges = {}
    by_line = defaultdict(list)
    for i, poem in enumerate(poems):
        for line in set(poem["lines"]):
            by_line[line].append(i)
    for line, indices in sorted(by_line.items()):
        for a, b in itertools.combinations(indices, 2):
            entry = edges.setdefault((a, b), {"shared_lines": [], "bigram_jaccard": None})
            entry["shared_lines"].append(line)
    bigrams = [bigram_set(p["lines"]) for p in poems]
    postings = defaultdict(list)
    for i, grams in enumerate(bigrams):
        overlaps = Counter(j for g in grams for j in postings[g])
        for j, intersection in overlaps.items():
            union = len(grams) + len(bigrams[j]) - intersection
            if union and 5 * intersection >= 4 * union:
                entry = edges.setdefault((j, i), {"shared_lines": [], "bigram_jaccard": None})
                entry["bigram_jaccard"] = {"intersection": intersection, "union": union,
                                            "value": intersection/union}
        for gram in grams:
            postings[gram].append(i)
    return [{"left": poems[a]["poem_id"], "right": poems[b]["poem_id"], **details}
            for (a, b), details in sorted(edges.items())]


def assign_families_and_splits(poems, edges):
    parent = {p["poem_id"]: p["poem_id"] for p in poems}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for edge in edges:
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for p in poems:
        members[find(p["poem_id"])].append(p["poem_id"])
    group_info = {}
    for root, ids in members.items():
        fid = "family_" + digest(sorted(ids))[:24]
        bucket = int(hashlib.sha256((SPLIT_SALT + ":" + fid).encode()).hexdigest(), 16) % 10000
        split = "train" if bucket < 7000 else "dev" if bucket < 8500 else "test"
        group_info[root] = (fid, split)
    return [{**p, "family_id": group_info[find(p["poem_id"])][0],
             "split": group_info[find(p["poem_id"])][1]} for p in poems]


def assert_split_integrity(poems, edges=None):
    """Recompute evidence when edges are omitted; do not trust provided family IDs."""
    index = {p["poem_id"]: p for p in poems}
    if len(index) != len(poems):
        raise ValueError("duplicate poem id")
    families = defaultdict(set)
    contents = defaultdict(set)
    for p in poems:
        if p["split"] not in ("train", "dev", "test"):
            raise ValueError("invalid split")
        families[p["family_id"]].add(p["split"])
        contents[tuple(p["lines"])].add(p["split"])
    if any(len(v) > 1 for v in families.values()) or any(len(v) > 1 for v in contents.values()):
        raise ValueError("exact text or family crosses split")
    for edge in family_edges(poems) if edges is None else edges:
        a, b = index[edge["left"]], index[edge["right"]]
        if a["family_id"] != b["family_id"] or a["split"] != b["split"]:
            raise ValueError("shared-line or near-duplicate relation crosses family/split")


def make_tasks(poems):
    tasks = []
    for poem in poems:
        for couplet in range(4):
            upper, lower = poem["lines"][2*couplet:2*couplet+2]
            tasks.append({"task_id": poem["poem_id"] + f":couplet{couplet}",
                "poem_id": poem["poem_id"], "family_id": poem["family_id"],
                "split": poem["split"], "length": poem["length"], "couplet_index": couplet,
                "upper_line": upper, "masked_lower_line": MASK + lower[2:],
                "lower_suffix": lower[2:], "start": 0, "mask_length": 2,
                "target": lower[:2], "author": poem["author"], "title": poem["title"],
                "source_refs": poem["source_refs"],
                "status": "observed_source_restoration_not_quality_gold"})
    return tasks


def model_view(task):
    """Explicit allow-list: targets, poem identities and audit metadata stay out."""
    return {name: task[name] for name in ("length", "upper_line", "masked_lower_line",
                                         "lower_suffix", "start", "mask_length")}


def train_resources(poems, tasks, min_count=2):
    if type(min_count) is not int or min_count < 1:
        raise ValueError("positive integer min_count required")
    characters = Counter(c for p in poems if p["split"] == "train" for line in p["lines"] for c in line)
    spans = {5: Counter(), 7: Counter()}
    origins = {5: defaultdict(set), 7: defaultdict(set)}
    for task in tasks:
        if task["split"] == "train":
            spans[task["length"]][task["target"]] += 1
            origins[task["length"]][task["target"]].add(task["poem_id"])
    included = sorted(c for c, n in characters.items() if n >= min_count)
    vocabulary = {"source_split": "train", "min_count": min_count,
        "characters": included, "tokens": ["<UNK>"] + included,
        "character_to_id": {c: i+1 for i,c in enumerate(included)},
        "unknown_token": "<UNK>", "unknown_id": 0,
        "character_counts": dict(sorted(characters.items())),
        "unknown_policy": "characters below min_count or absent from train use explicit UNK"}
    candidates = {"source_split": "train", "min_count": min_count,
        "span_definition": "two-character prefix of an even numbered line, not linguistic word segmentation",
        "by_length": {str(length): [{"text": text, "count": count,
                                      "source_poem_ids": sorted(origins[length][text])}
                                     for text, count in sorted(spans[length].items()) if count >= min_count]
                      for length in (5, 7)}}
    if any(c not in included for values in candidates["by_length"].values() for item in values for c in item["text"]):
        raise ValueError("candidate contains a character outside the training vocabulary")
    return vocabulary, candidates


def validate_tasks_and_resources(poems, tasks, vocabulary, candidates):
    index = {p["poem_id"]: p for p in poems}
    if len({t["task_id"] for t in tasks}) != len(tasks) or len(tasks) != 4*len(poems):
        raise ValueError("missing or duplicate restoration tasks")
    for task in tasks:
        p = index[task["poem_id"]]
        q = task["couplet_index"]
        lower = p["lines"][2*q+1]
        expected = {"family_id": p["family_id"], "split": p["split"], "length": p["length"],
                    "upper_line": p["lines"][2*q], "masked_lower_line": MASK+lower[2:],
                    "lower_suffix": lower[2:], "target": lower[:2], "start": 0, "mask_length": 2}
        if any(task.get(k) != v for k, v in expected.items()):
            raise ValueError("task text/mask/split identity mismatch")
        if any(k in task for k in ("lower_line", "full_lower_line", "lines")):
            raise ValueError("complete lower line leaked into task")
    expected_vocab, expected_candidates = train_resources(poems, tasks, vocabulary["min_count"])
    if vocabulary != expected_vocab or candidates != expected_candidates:
        raise ValueError("vocabulary/candidates are not the declared train-only resources")


def corpus_audit(poems, tasks, edges, counts, vocabulary, candidates):
    assert_split_integrity(poems, edges)
    validate_tasks_and_resources(poems, tasks, vocabulary, candidates)
    pools = {int(length): {item["text"] for item in values} for length, values in candidates["by_length"].items()}
    slices = {}
    for split in ("train", "dev", "test"):
        selected = [p for p in poems if p["split"] == split]
        selected_tasks = [t for t in tasks if t["split"] == split]
        covered = sum(t["target"] in pools[t["length"]] for t in selected_tasks)
        slices[split] = {"poems": len(selected), "families": len({p["family_id"] for p in selected}),
            "poems_by_length": dict(Counter(str(p["length"]) for p in selected)),
            "tasks": len(selected_tasks), "target_in_train_candidate_vocabulary": covered,
            "target_absent_from_train_candidate_vocabulary": len(selected_tasks)-covered,
            "candidate_vocabulary_coverage": covered/len(selected_tasks) if selected_tasks else None,
            "authors": len({p["author"] for p in selected})}
    family_sizes = Counter(p["family_id"] for p in poems)
    return {"schema": "versematrix.corpus-audit.v0.3", "status": "source_restoration_data_built",
        "interpretation": "Not verified lushi, contextual reading gold, literary-quality gold or author-held-out generalization.",
        **counts, "families": len(family_sizes), "largest_family_poems": max(family_sizes.values(), default=0),
        "family_size_histogram": dict(sorted(Counter(family_sizes.values()).items())),
        "relation_pairs": len(edges), "shared_line_pairs": sum(bool(e["shared_lines"]) for e in edges),
        "bigram_jaccard_pairs": sum(e["bigram_jaccard"] is not None for e in edges),
        "split": slices, "character_vocabulary_size": len(vocabulary["characters"]),
        "candidate_vocabulary_size": {k: len(v) for k,v in candidates["by_length"].items()},
        "checks": {"split_integrity": "passed", "masked_task_identity": "passed", "train_only_resources": "passed"},
        "limits": ["Exact source code points only; simplified/traditional or variant glyph duplicates may remain.",
                   "Poem families use specified surface relations, not exhaustive philological edition families.",
                   "Corpus is the first ten fixed repository shards, not a random sample of all poetry.",
                   "OOV tasks are retained; candidate vocabulary coverage is not retrieval recall or model accuracy.",
                   "Source restoration measures agreement with an observed text, not superiority over alternative wording."]}
