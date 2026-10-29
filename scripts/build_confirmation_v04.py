"""Freeze, fetch, build and independently byte-check v0.4 confirmation data.

Run --freeze before --fetch. Downloading does not evaluate any restoration
model. A normal build and --check operate entirely on frozen local source bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.confirmation import CONFIRMATION_SALT, MAX_FAMILIES, select_confirmation
from wordconv_lab.corpus import collect_poems

COMMIT = "b8594f81a89752241442f2ce267d6f66f96704ee"
OFFSETS = list(range(10000, 20000, 1000))
OLD_POEMS = ROOT / "data/corpus_v03/poems.jsonl"
OLD_BUILD = ROOT / "data/corpus_v03/build_manifest.json"
OLD_LICENSE = ROOT / "data/audit_sources/poetry_LICENSE"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def rows_encoded(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")) + "\n" for row in rows).encode()


def atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def source_plan(output):
    rows = []
    for offset in OFFSETS:
        source_path = f"全唐诗/poet.tang.{offset}.json"
        rows.append({"repository": "chinese-poetry/chinese-poetry", "commit": COMMIT,
                     "offset": offset, "source_path": source_path,
                     "local_path": str((output / "raw" / f"poet.tang.{offset}.json").relative_to(ROOT)),
                     "download_url": f"https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/{COMMIT}/"
                                     + urllib.parse.quote(source_path)})
    return rows


def freeze_payload(output):
    old_bytes = OLD_POEMS.read_bytes()
    old_build = json.loads(OLD_BUILD.read_bytes())
    if sha(old_bytes) != old_build["outputs"]["poems.jsonl"]["sha256"]:
        raise ValueError("old corpus bytes differ from their completed build manifest")
    old = [json.loads(line) for line in old_bytes.splitlines()]
    if len(old) != 3579 or set(p["split"] for p in old) != {"train", "dev", "test"}:
        raise ValueError("unexpected prior exclusion corpus")
    config = {
        "schema": "versematrix.confirmation-policy.v0.4",
        "source_commit": COMMIT, "shard_offsets": OFFSETS,
        "shape": [[8, 5], [8, 7]],
        "normalization": "none; exact original Unicode code points",
        "characters": "CJK unified/compatibility ideographs only",
        "families": ["exact text deduplication", "shared complete line", "within-line bigram-set Jaccard >= 4/5"],
        "exclusion": "Build connected components jointly over ALL old train/dev/test poems and all new unique poems; exclude every new poem in any component containing an old poem, including indirect new-poem bridges.",
        "selection": "Eligible joint component family_id = family_ + first 24 hex digits of digest(sorted unique poem IDs)); sort by SHA256(salt + ':' + family_id), then family_id; retain all poems from the first max_families eligible components.",
        "selection_salt": CONFIRMATION_SALT, "max_families": MAX_FAMILIES,
        "insufficient_families": "retain all eligible families; do not change sources or supplement",
        "selected_split": "confirm",
        "task": "four even-line initial-two-character masks per unique selected poem",
        "no_candidate_or_model_processing": True,
        "unselected_new_data": "retained as selection evidence only; not used for train/dev",
        "scope": "Source-text restoration confirmation; not certified lushi, contextual reading gold, literary-quality gold or author-held-out evaluation.",
    }
    code_paths = [ROOT / "src/wordconv_lab/corpus.py", ROOT / "src/wordconv_lab/confirmation.py",
                  Path(__file__).resolve(), ROOT / "tests/test_confirmation.py"]
    return {"config": config, "source_plan": source_plan(output),
            "prior_input_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes())
                                   for p in [OLD_POEMS, OLD_BUILD, OLD_LICENSE]},
            "code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in code_paths}}


def verify_freeze(output):
    path = output / "freeze.json"
    recorded = json.loads(path.read_bytes())
    expected = freeze_payload(output)
    if any(recorded.get(k) != value for k, value in expected.items()):
        raise ValueError("frozen confirmation policy, old inputs, source plan or code changed")
    return recorded


def freeze(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "freeze.json").exists():
        raise ValueError("confirmation policy already frozen; do not overwrite")
    if (output / "raw").exists() and any((output / "raw").iterdir()):
        raise ValueError("fresh source bytes must not predate the policy freeze")
    payload = freeze_payload(output)
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "frozen_before_new_source_download_and_processing"
    atomic_write(output / "freeze.json", encoded(payload))
    return {"status": "frozen", "source_shards": len(OFFSETS), "max_families": MAX_FAMILIES,
            "prior_exclusion_poems": 3579}


def fetch(output):
    frozen = verify_freeze(output)
    (output / "raw").mkdir(exist_ok=True)
    path = output / "source_manifest.json"
    previous = json.loads(path.read_bytes()) if path.exists() else None
    records = {r["offset"]: r for r in previous["sources"]} if previous else {}
    if previous and previous["freeze_sha256"] != sha((output / "freeze.json").read_bytes()):
        raise ValueError("download manifest belongs to another freeze")
    expected_offsets = set(OFFSETS)
    if set(records) - expected_offsets:
        raise ValueError("undeclared source in download manifest")
    for plan in frozen["source_plan"]:
        target = ROOT / plan["local_path"]
        old_record = records.get(plan["offset"])
        if old_record:
            if any(old_record.get(k) != value for k, value in plan.items()):
                raise ValueError("source download record differs from frozen plan")
            if target.exists():
                if sha(target.read_bytes()) != old_record["sha256"]:
                    raise ValueError("previously downloaded source changed")
                continue
        if target.exists():
            raise ValueError("unregistered source file already exists")
        request = urllib.request.Request(plan["download_url"], headers={"User-Agent": "VerseMatrix-source-audit/0.4"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
            retrieved_url = response.geturl()
        parsed = json.loads(data)
        if not isinstance(parsed, list):
            raise ValueError("fixed source is not a JSON record list")
        if old_record and sha(data) != old_record["sha256"]:
            raise ValueError("resumed source download hash differs")
        record = old_record or {**plan, "bytes": len(data), "sha256": sha(data),
                               "records": len(parsed), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                               "retrieved_url": retrieved_url,
                               "license_path": str(OLD_LICENSE.relative_to(ROOT)),
                               "license_sha256": sha(OLD_LICENSE.read_bytes())}
        records[plan["offset"]] = record
        manifest = {"schema": "versematrix.confirmation-sources.v0.4",
                    "freeze_sha256": sha((output / "freeze.json").read_bytes()),
                    "status": "downloaded" if set(records) == expected_offsets else "partial",
                    "sources": [records[offset] for offset in sorted(records)]}
        # A receipt is durable before final local placement; a missing file on
        # resume is re-fetched and must match the recorded digest.
        atomic_write(path, encoded(manifest))
        atomic_write(target, data)
        print(json.dumps({"downloaded_offset": plan["offset"], "bytes": len(data), "records": len(parsed)}), flush=True)
    return {"status": "downloaded", "shards": len(records),
            "source_bytes": sum(r["bytes"] for r in records.values())}


def build(output, check=False):
    verify_freeze(output)
    source_manifest = json.loads((output / "source_manifest.json").read_bytes())
    if source_manifest["status"] != "downloaded" or [r["offset"] for r in source_manifest["sources"]] != OFFSETS:
        raise ValueError("all declared source shards are required before building")
    if source_manifest["freeze_sha256"] != sha((output / "freeze.json").read_bytes()):
        raise ValueError("source manifest belongs to another policy freeze")
    shards = []
    for plan, source in zip(source_plan(output), source_manifest["sources"]):
        if any(source.get(k) != value for k, value in plan.items()):
            raise ValueError("source metadata differs from frozen plan")
        data = (ROOT / source["local_path"]).read_bytes()
        if sha(data) != source["sha256"] or len(data) != source["bytes"]:
            raise ValueError("new source digest/size mismatch")
        records = json.loads(data)
        if len(records) != source["records"]:
            raise ValueError("source record count mismatch")
        shards.append((source["local_path"], records))
    new_poems, exact, source_counts = collect_poems(shards)
    old_poems = [json.loads(line) for line in OLD_POEMS.read_bytes().splitlines()]
    result = select_confirmation(old_poems, new_poems)
    audit = {**result["audit"], "source_record_audit": source_counts}
    payloads = {"new_unique_poems.jsonl": rows_encoded(new_poems),
                "new_exact_duplicate_groups.jsonl": rows_encoded(exact),
                "audit.json": encoded(audit)}
    for name in ["poems", "tasks", "selection_records", "components", "joint_family_edges"]:
        payloads[name + ".jsonl"] = rows_encoded(result[name])
    manifest = {"schema": "versematrix.confirmation-build.v0.4", "status": "completed",
                "freeze_sha256": sha((output / "freeze.json").read_bytes()),
                "source_manifest_sha256": sha((output / "source_manifest.json").read_bytes()),
                "outputs": {name: {"bytes": len(data), "sha256": sha(data)} for name, data in payloads.items()},
                "scorer_trained_here": False, "model_scores_accessed": False,
                "restoration_hits_evaluated": False, "literary_quality_evaluated": False}
    payloads["build_manifest.json"] = encoded(manifest)
    if not check and (output / "build_manifest.json").exists():
        raise ValueError("completed confirmation corpus exists; use --check")
    for name, data in payloads.items():
        path = output / name
        if check:
            if path.read_bytes() != data:
                raise ValueError(f"derived confirmation bytes differ: {name}")
        else:
            atomic_write(path, data)
    return {"status": "check_passed" if check else "built", "audit": audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "data/confirmation_v04")
    args = parser.parse_args()
    output = args.out.resolve()
    result = freeze(output) if args.freeze else fetch(output) if args.fetch else build(output, args.check)
    print(json.dumps(result, ensure_ascii=False, indent=2))
