"""Fresh source confirmation families with explicit exclusion of prior text.

The grouping relation is the unchanged v0.3 surface-text relation. This module
never fits a scorer, constructs a candidate vocabulary, or computes target hits.
All old train/dev/test poems form one exclusion set. A new poem connected to it
through any number of other new poems is excluded before family selection.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib

from .corpus import digest, family_edges, make_tasks


CONFIRMATION_SALT = "versematrix-confirmation-v04-20261029"
MAX_FAMILIES = 512


def _validate_poems(poems, old=False):
    index = {}
    contents = {}
    for poem in poems:
        pid = poem["poem_id"]
        lines = tuple(poem["lines"])
        if pid in index:
            raise ValueError("duplicate poem identity in input")
        if len(lines) != 8 or poem["length"] not in (5, 7) or any(len(s) != poem["length"] for s in lines):
            raise ValueError("confirmation requires eight equal-length five/seven-character lines")
        if lines in contents:
            raise ValueError("input text was not deduplicated")
        if old and poem.get("split") not in ("train", "dev", "test"):
            raise ValueError("prior exclusion set must include the existing train/dev/test records")
        index[pid] = poem
        contents[lines] = pid
    return index, contents


def select_confirmation(old_poems, new_poems, max_families=MAX_FAMILIES,
                        salt=CONFIRMATION_SALT):
    """Return selected poems/tasks and complete exclusion/selection evidence.

    Exact source identity is checked across inputs. Shared-line and Jaccard
    edges are computed on the union, then closed transitively before selection.
    Selection uses SHA256(salt + ':' + family_id), with family_id as the tie
    breaker; every member of a selected eligible family is retained.
    """
    if type(max_families) is not int or max_families < 1:
        raise ValueError("max_families must be a positive integer")
    if not isinstance(salt, str) or not salt:
        raise ValueError("selection salt must be a nonempty string")
    old_index, old_texts = _validate_poems(old_poems, old=True)
    new_index, _ = _validate_poems(new_poems)
    if not old_index:
        raise ValueError("the prior exclusion set must not be empty")
    # Exact duplicate text must have the same stable identity in both corpora.
    for pid, poem in new_index.items():
        lines = tuple(poem["lines"])
        if pid in old_index and old_index[pid]["lines"] != poem["lines"]:
            raise ValueError("same poem identity has different source text")
        if lines in old_texts and old_texts[lines] != pid:
            raise ValueError("exact duplicate text has inconsistent stable identity")
    union_index = {**old_index, **new_index}
    union_poems = [union_index[pid] for pid in sorted(union_index)]
    edges = family_edges(union_poems)
    parent = {pid: pid for pid in union_index}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    for edge in edges:
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for pid in sorted(union_index):
        groups[find(pid)].append(pid)
    components = []
    by_pid = {}
    for members in groups.values():
        old_members = sorted(set(members) & old_index.keys())
        new_members = sorted(set(members) & new_index.keys())
        fid = "family_" + digest(members)[:24]
        eligible = bool(new_members) and not old_members
        selection_hash = hashlib.sha256((salt + ":" + fid).encode()).hexdigest() if eligible else None
        record = {"family_id": fid, "member_poem_ids": members,
                  "old_poem_ids": old_members, "new_poem_ids": new_members,
                  "eligible": eligible, "selection_hash": selection_hash,
                  "selection_rank": None, "selected": False,
                  "decision": "exclude_prior_connected_component" if old_members and new_members
                              else "old_only_component" if not new_members else "eligible_not_selected"}
        components.append(record)
        for pid in members:
            by_pid[pid] = record
    eligible = sorted((c for c in components if c["eligible"]),
                      key=lambda c: (c["selection_hash"], c["family_id"]))
    for rank, component in enumerate(eligible, 1):
        component["selection_rank"] = rank
        if rank <= max_families:
            component["selected"] = True
            component["decision"] = "selected_confirmation_family"
    selected = []
    decisions = []
    for pid in sorted(new_index):
        poem = new_index[pid]
        c = by_pid[pid]
        decisions.append({"poem_id": pid, "family_id": c["family_id"],
                          "length": poem["length"], "decision": c["decision"],
                          "exact_text_seen_in_prior": pid in old_index,
                          "prior_component_poem_ids": c["old_poem_ids"],
                          "new_component_poem_ids": c["new_poem_ids"],
                          "selection_hash": c["selection_hash"],
                          "selection_rank": c["selection_rank"]})
        if c["selected"]:
            selected.append({**deepcopy(poem), "family_id": c["family_id"],
                             "split": "confirm",
                             "status": "shape_only_not_verified_lushi"})
    selected_ids = {p["poem_id"] for p in selected}
    selected_families = {p["family_id"] for p in selected}
    if selected_ids & old_index.keys():
        raise AssertionError("selected text overlaps prior text")
    for c in components:
        if c["family_id"] in selected_families:
            if c["old_poem_ids"] or set(c["new_poem_ids"]) - selected_ids:
                raise AssertionError("selected family is contaminated or incomplete")
    for edge in edges:
        if edge["left"] in selected_ids or edge["right"] in selected_ids:
            if edge["left"] not in selected_ids or edge["right"] not in selected_ids:
                raise AssertionError("selected surface-family edge crosses confirmation boundary")
    tasks = make_tasks(selected)
    if len(tasks) != 4 * len(selected) or any(t["split"] != "confirm" for t in tasks):
        raise AssertionError("confirmation tasks do not match source poems")
    counts = Counter(d["decision"] for d in decisions)
    audit = {
        "schema": "versematrix.confirmation-selection.v0.4",
        "prior_exclusion_poems": len(old_poems),
        "prior_exclusion_by_split": dict(sorted(Counter(p["split"] for p in old_poems).items())),
        "new_unique_text_poems": len(new_poems),
        "new_exact_text_already_in_prior": len(old_index.keys() & new_index.keys()),
        "joint_unique_text_poems": len(union_poems),
        "joint_relation_pairs": len(edges),
        "joint_shared_line_pairs": sum(bool(e["shared_lines"]) for e in edges),
        "joint_bigram_jaccard_pairs": sum(e["bigram_jaccard"] is not None for e in edges),
        "joint_components": len(components),
        "new_poems_excluded_by_prior_component": counts["exclude_prior_connected_component"],
        "new_components_excluded_by_prior": sum(c["decision"] == "exclude_prior_connected_component" for c in components),
        "eligible_families": len(eligible),
        "eligible_poems": sum(len(c["new_poem_ids"]) for c in eligible),
        "family_cap": max_families,
        "selected_families": len(selected_families),
        "selected_poems": len(selected),
        "selected_tasks": len(tasks),
        "selected_poems_by_length": dict(sorted(Counter(str(p["length"]) for p in selected).items())),
        "selected_tasks_by_length": dict(sorted(Counter(str(t["length"]) for t in tasks).items())),
        "eligible_unselected_poems": counts["eligible_not_selected"],
        "selected_authors_also_present_in_prior": len({p.get("author") for p in selected} & {p.get("author") for p in old_poems}),
        "checks": {"all_old_splits_excluded": "passed", "transitive_family_exclusion": "passed",
                   "whole_family_selection": "passed", "selected_edge_boundary": "passed",
                   "four_tasks_per_selected_poem": "passed"},
        "scorers_or_retrieval_used": False,
        "candidate_vocabulary_built": False,
        "source_restoration_hits_computed": False,
        "limits": [
            "Eight equal-length lines are a shape filter, not a strict-lushi certificate.",
            "Exact Unicode code points only; variant glyph and edition-level relations can remain undetected.",
            "Families use the declared surface relation, not exhaustive philological identity.",
            "Authors can appear in both old data and confirmation; this is not author-held-out evaluation.",
            "Fixed consecutive repository shards are not a random population sample of poetry.",
            "Observed original text is a restoration target, not a literary-quality gold label.",
            "Unselected new families are retained as exclusion/selection evidence and are not used for development.",
        ],
    }
    return {"poems": selected, "tasks": tasks, "selection_records": decisions,
            "components": sorted(components, key=lambda c: c["family_id"]),
            "joint_family_edges": edges, "audit": audit}
