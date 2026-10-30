#!/usr/bin/env python3
"""Export frozen candidate pools and train-only statistics for the browser demo.

This does not generate a poem, retrain a model, or change a frozen experiment.
The browser reruns only constrained beam search over these pre-scored pools.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.free_generation import Coherence, load_five_training  # noqa: E402
from wordconv_lab.prosody import all_templates  # noqa: E402


def export(root: Path) -> dict:
    run = root / "results/free_generation_v06_run001"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    freeze = read(run / "freeze.json")
    for name, identity in freeze["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("Frozen source identity changed: " + name)
    pools = read(run / "candidate_pools.json")
    resources = read(run / "resources.json")
    ranking = read(run / "rhyme_selection.json")
    baseline = read(run / "result.json")
    reference = read(root / "data/reference_v02/pingshui_106_memberships.json")
    poems, loading = load_five_training(root / "data/corpus_v03/poems.jsonl")
    if len(poems) != 2203 or sum(map(len, pools["pools"].values())) != 2392:
        raise ValueError("Unexpected source corpus or pool")
    coherence = Coherence(poems)
    chars = sorted({char for pool in pools["pools"].values() for line in pool for char in line["text"]})
    values = bytearray()
    # Lower triangle, row-major: index = j*(j-1)//2 + i, where i<j.
    # Float64 preserves the exact Python log1p values; no Float32 quantization.
    for j, right in enumerate(chars):
        for left in chars[:j]:
            denominator = coherence.df[left] * coherence.df[right]
            value = math.log1p(coherence.n * coherence.joint[(left, right)] / denominator) if denominator else 0.0
            values.extend(struct.pack("<d", value))
    labels = {entry["group_id"]: entry["label"]
              for entries in reference["characters"].values() for entry in entries}
    compact_pools = {}
    train_lines = set(resources["train_lines"])
    for key, pool in sorted(pools["pools"].items()):
        compact_pools[key] = []
        for candidate in pool:
            text = candidate["text"]
            if len(text) != 5 or len(set(text)) != 5 or text in train_lines:
                raise ValueError("Frozen candidate failed the source-pool filter")
            compact_pools[key].append([
                text, candidate["lm_score"],
                resources["prefixes"][candidate["prefix"]]["count"],
                resources["suffixes"][candidate["suffix"]]["count"],
            ])
    source_names = [
        "data/corpus_v03/poems.jsonl", "data/reference_v02/pingshui_106_memberships.json",
        "results/free_generation_v06_run001/candidate_pools.json",
        "results/free_generation_v06_run001/resources.json",
        "results/free_generation_v06_run001/rhyme_selection.json",
        "results/free_generation_v06_run001/result.json",
        "src/wordconv_lab/free_generation.py", "src/wordconv_lab/prosody.py",
    ]
    return {
        "schema": "verseloom.web.v1",
        "meta": {
            "trainingPoems": len(poems), "candidateCount": 2392,
            "feasibleRhymes": sum(row["feasible"] for row in ranking["ranking"]),
            "templateCount": 4, "characterCount": len(chars),
            "method": "Frozen scored candidate pools + live browser constrained beam search",
            "candidateScoresPrecomputed": True, "neuralModelUsed": False,
            "loading": loading,
            "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
            "referenceSource": reference["source"]["source_url"],
            "referenceLicense": "CC BY-SA 4.0; see NOTICE.html and the repository source notices",
            "contextualPronunciationVerified": False, "antithesisCertified": False,
            "literaryQualityCertified": False, "globalOptimumClaim": False,
            "sourcePoolNoveltyScope": "Excludes complete lines in the 2,203 five-character training poems only",
        },
        "defaults": {"rhymeGroup": "auto", "templateId": "auto", "beamWidth": 32,
                     "coherenceWeight": 0.5, "repetitionWeight": 0.8},
        "templates": [{"id": t.template_id,
                       "label": ("平起" if t.start_second_tone == "P" else "仄起") + ("首句入韵" if t.first_rhymed else "首句不入韵"),
                       "core": list(t.core), "rhymeRows": list(t.rhyme_rows)}
                      for t in sorted(all_templates(5), key=lambda t: t.template_id)],
        "rhymes": [{"id": row["group"], "label": labels[row["group"]], "feasible": row["feasible"],
                    "frequency": row["frequency"], "endingCount": row["union_ending_coverage"]}
                   for row in ranking["ranking"]],
        "baseline": {"lines": baseline["lines"], "templateId": baseline["template_id"],
                     "rhymeGroup": baseline["rhyme_group"], "rhymeLabel": labels[baseline["rhyme_group"]],
                     "score": baseline["score"]},
        "pools": compact_pools,
        "coherence": {"characters": "".join(chars), "encoding": "float64-le-lower-triangle-base64",
                      "values": base64.b64encode(values).decode("ascii")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "web/data.js")
    parser.add_argument("--check", action="store_true", help="Check the existing export without rewriting it")
    args = parser.parse_args()
    data = export(ROOT)
    content = "/* Derived from frozen VerseLoom records. See NOTICE.html for sources and licenses. */\n"
    content += "globalThis.VerseLoomData=" + json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n"
    encoded = content.encode("utf-8")
    if args.check:
        if args.output.read_bytes() != encoded:
            raise SystemExit("Browser data differs from the deterministic source export")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(json.dumps({"status": "matched" if args.check else "exported", "bytes": len(encoded),
                      "sha256": hashlib.sha256(encoded).hexdigest(), "candidates": data["meta"]["candidateCount"],
                      "rhymes": data["meta"]["feasibleRhymes"]}))


if __name__ == "__main__":
    main()
