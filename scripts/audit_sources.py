"""Reproduce the bundled source snapshot audit; this is NOT poem classification."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def audit(root: Path) -> dict:
    directory = root / "data" / "audit_sources"
    manifest = json.loads((directory / "manifest.json").read_text())
    checked = []
    for item in manifest:
        path = root / item["local_path"]
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != item["sha256"] or len(data) != item["bytes"]:
            raise ValueError(f"Snapshot identity mismatch: {item['local_path']}")
        checked.append(item["local_path"])
    poems = json.loads((directory / "tang_0.json").read_text())
    rhyme = json.loads((directory / "pingshui.json").read_text())
    memberships = defaultdict(set)
    for section, groups in rhyme.items():
        for group, characters in groups.items():
            for char in characters:
                memberships[char].add((section, group))
    normalized = Counter()
    shape_counts = {5: 0, 7: 0}
    shape_tokens = []
    for poem in poems:
        text = "".join(poem["paragraphs"])
        lines = [x for x in re.split("[，。！？；]", text) if x]
        normalized["".join(lines)] += 1
        for length in (5, 7):
            if len(lines) == 8 and all(len(line) == length for line in lines):
                shape_counts[length] += 1
                shape_tokens.extend("".join(lines))
    results = {
        "status": "audit_only_not_benchmark", "manifest_files_verified": len(checked),
        "poetry": {"records": len(poems), "shape_only_8x5": shape_counts[5],
                   "shape_only_8x7": shape_counts[7], "exact_normalized_unique": len(normalized),
                   "duplicate_groups": sum(n > 1 for n in normalized.values())},
        "rhyme": {"group_counts": {k: len(v) for k, v in rhyme.items()},
                  "group_count_total": sum(map(len, rhyme.values())),
                  "has_shang_sanjiang": "三讲" in rhyme.get("上声部", {}),
                  "distinct_characters": len(memberships),
                  "characters_with_multiple_memberships": sum(len(v) > 1 for v in memberships.values())},
        "direct_lookup_diagnostic": {"shape_only_tokens": len(shape_tokens),
                                     "directly_found_tokens": sum(c in memberships for c in shape_tokens)},
        "limitations": ["This shard is not a random sample of all poetry.",
                        "8x5/8x7 shape is not proof of regulated verse.",
                        "No contextual reading disambiguation or traditional/simplified conversion performed.",
                        "Source licence is not a text-critical edition or phonology quality certificate."],
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/source_audit.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
