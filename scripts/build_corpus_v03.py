"""Offline reconstruction/check of the frozen v0.3 source-restoration corpus."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.corpus import (SPLIT_SALT, assign_families_and_splits, collect_poems,
    corpus_audit, family_edges, make_tasks, train_resources)

COMMIT = "b8594f81a89752241442f2ce267d6f66f96704ee"
HASHES = {
0:"4f88f429938082523cdc72a1612bd7a7c3e3f7a33f8192c61ae51340efc3c92d",
1000:"6e703b8d8dcedbd6110f6d48d043665ca2131db2035a5bfcef5e59a19aeb6d75",
2000:"ec28efa7e2ae9cdbbba09f0ed186e32ccae47a325ad1e5816b7ad56071c1665a",
3000:"9e4ba2323752b6b9fbdba875cdbc6a26636fe0cabc85fd066370d4915952001f",
4000:"b8a7946c879b9658a6c667a4c55f776fae7e5e00a17789ab06b274c1429afc66",
5000:"a0306e6962cff01639ad9a00b042a201b8dec1f30fa5d768a7b112589887861e",
6000:"982a21d2b15922b406911f1c825785054314c686da86b2155d7aebb362d22771",
7000:"a6fb69d7b4343c904dc56ae5aae34facc2478d196b51c37ceba9d1e2f171b342",
8000:"ac3848e95b02736168003863795de4474838c2d6a71a24ca2a4a280ea5177773",
9000:"b99d6596e3a1fe1a661652383083f92d196b8baf4e461d087081d5e1e1b7a51f"}


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()


def jsonlines(values):
    return "".join(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for value in values).encode()


def build(output, check=False):
    output.mkdir(parents=True,exist_ok=True)
    source_manifest=[]
    shards=[]
    for offset, expected in HASHES.items():
        relative=Path("data/audit_sources/tang_0.json") if offset==0 else Path("data/corpus_v03/raw")/f"poet.tang.{offset}.json"
        path=ROOT/relative
        data=path.read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        if actual!=expected:raise ValueError(f"source SHA mismatch: {relative}")
        records=json.loads(data)
        source_path=f"全唐诗/poet.tang.{offset}.json"
        source_manifest.append({"repository":"chinese-poetry/chinese-poetry","commit":COMMIT,
            "source_path":source_path,"local_path":str(relative),"bytes":len(data),"sha256":actual,
            "download_url":f"https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/{COMMIT}/"+urllib.parse.quote(source_path),
            "records":len(records),"license_path":"data/audit_sources/poetry_LICENSE",
            "license_sha256":hashlib.sha256((ROOT/'data/audit_sources/poetry_LICENSE').read_bytes()).hexdigest()})
        shards.append((str(relative),records))
    config={"source_commit":COMMIT,"shard_offsets":list(HASHES),"shape":[[8,5],[8,7]],
        "characters":"CJK unified/compatibility ideographs only, exact Unicode code points",
        "split_salt":SPLIT_SALT,"split_hash_buckets":{"train":[0,6999],"dev":[7000,8499],"test":[8500,9999]},
        "families":["exact text deduplication","shared complete line","within-line bigram-set Jaccard >= 4/5"],
        "task":"four even-line two-character-prefix masks per unique poem", "candidate_min_count":2,
        "status":"source_restoration_not_verified_lushi_or_quality_gold"}
    code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT/'src/wordconv_lab/corpus.py',Path(__file__).resolve())}
    frozen={"config":config,"source_sha256":{s['local_path']:s['sha256'] for s in source_manifest},"code_sha256":code_hashes}
    freeze_path=output/'freeze.json'
    if check:
        existing=json.loads(freeze_path.read_text())
        if any(existing[k]!=v for k,v in frozen.items()):raise ValueError("frozen source/config/code changed")
    else:
        if freeze_path.exists():raise ValueError("freeze exists; use --check rather than overwrite evidence")
        frozen['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        frozen['status']='frozen_before_processing'
        freeze_path.write_bytes(encoded(frozen))
    poems,exact,counts=collect_poems(shards)
    edges=family_edges(poems)
    poems=assign_families_and_splits(poems,edges)
    tasks=make_tasks(poems)
    vocab,candidates=train_resources(poems,tasks,min_count=2)
    audit=corpus_audit(poems,tasks,edges,counts,vocab,candidates)
    payloads={"source_manifest.json":encoded(source_manifest),"poems.jsonl":jsonlines(poems),
        "exact_duplicate_groups.jsonl":jsonlines(exact),"family_edges.jsonl":jsonlines(edges),
        "tasks.jsonl":jsonlines(tasks),"vocabulary.json":encoded(vocab),
        "candidate_spans.json":encoded(candidates),"audit.json":encoded(audit)}
    manifest={"schema":"versematrix.corpus-build.v0.3","status":"completed",
        "freeze_sha256":hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "outputs":{name:{"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()} for name,data in payloads.items()},
        "scorer_trained_here":False,"model_scores_accessed":False,"literary_quality_evaluated":False}
    payloads['build_manifest.json']=encoded(manifest)
    for name,data in payloads.items():
        path=output/name
        if check:
            if path.read_bytes()!=data:raise ValueError(f"derived bytes differ: {name}")
        else:
            tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);tmp.replace(path)
    return {"mode":"check" if check else "build","ok":True,"counts":audit}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'data/corpus_v03')
    args=parser.parse_args()
    print(json.dumps(build(args.out,args.check),ensure_ascii=False,indent=2))
