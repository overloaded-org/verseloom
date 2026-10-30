#!/usr/bin/env python3
"""SP-02: falsifiable couplet/closure diagnostics on unmodified generation receipts.

This is a measurement, not a strict antithesis or literary-quality oracle.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def analyze(records):
    pairs = defaultdict(dict)
    for row in records:
        o = row["options"]
        require(o["mode"] in ("relations", "learned"), "Unexpected arm")
        key = (row["length"], o["seed"])
        require(o["mode"] not in pairs[key], "Duplicate arm")
        pairs[key][o["mode"]] = row
    result = []
    for (length, seed), arms in sorted(pairs.items()):
        require(set(arms) == {"relations", "learned"}, "Incomplete comparison")
        a, b = arms["relations"], arms["learned"]
        require(a["poolFingerprint"] == b["poolFingerprint"], "Candidate pools differ")
        require(a["options"]["beamWidth"] == b["options"]["beamWidth"], "Search budgets differ")
        item = {"length": length, "seed": seed, "status": [a["status"], b["status"]], "poolIdentical": True}
        if a["status"] == b["status"] == "complete":
            changed = [i for i in range(8) if a["lines"][i] != b["lines"][i]]
            def measures(r):
                d = r["lineDetails"]
                # This is evidence of statistical span alignment, NOT grammatical antithesis.
                middle = [d[i]["relations"].get("alignment", 0) for i in (3, 5)]
                # Count multi-character segments appearing in both members of a couplet.
                shared = []
                for i in (2, 4):
                    left = {t["text"] for t in d[i]["tokens"] if len(t["text"]) > 1}
                    right = {t["text"] for t in d[i+1]["tokens"] if len(t["text"]) > 1}
                    shared.append(len(left & right))
                return {"middleAlignment": middle, "repeatedMiddleSegments": shared,
                        "closureFollowing": d[7]["relations"].get("following", 0),
                        "closureScore": d[7]["relations"]["value"]}
            item.update(changedPositions=changed, entirePoemChanged=len(changed) == 8,
                        sameOpening=a["lines"][0] == b["lines"][0],
                        sameFirstSix=a["lines"][:6] == b["lines"][:6],
                        relationMeasures=measures(a), learnedMeasures=measures(b))
        result.append(item)
    return {"schema": "verseloom.sp02.couplet-diagnostic.v1",
            "scope": "observational matched-seed diagnostics, not causal proof of antithesis",
            "comparisons": result,
            "pairs": len(result),
            "completePairs": sum(x["status"] == ["complete", "complete"] for x in result),
            "entirePoemChanges": sum(x.get("entirePoemChanged", False) for x in result),
            "isolatedCoupletImprovementProven": False,
            "strictAntithesisCertified": False,
            "literaryQualityCertified": False,
            "nextIntervention": "Freeze first six lines and candidate pool; compare learned closure and joint middle-couplet search with equal budgets."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "Refusing to overwrite evidence")
    rows = [json.loads(line) for f in a.input for line in f.read_text().splitlines() if line.strip()]
    report = analyze(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k!="comparisons"}))


if __name__ == "__main__":
    main()
