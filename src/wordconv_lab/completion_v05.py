"""User-facing two-character completion using frozen source-span candidates.

The returned lines are suggestions. Retrieval does not certify contextual
pronunciation, regulated-verse legality, or literary quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .restoration import VisibleQuery, digest
from .retrieval import PhraseRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATTERN = re.compile(rb'"split"\s*:\s*"(train|dev|test)"')


def build_query(upper: str, suffix: str) -> VisibleQuery:
    """Strip surrounding whitespace, preserving exact source characters."""
    if not isinstance(upper, str) or not isinstance(suffix, str):
        raise ValueError("上句和后缀必须是文字")
    upper, suffix = upper.strip(), suffix.strip()
    if len(upper) not in (5, 7):
        raise ValueError("上句须为五字或七字，不带标点")
    if len(suffix) != len(upper) - 2:
        raise ValueError("下句后缀须比上句少二字：五言填三字，七言填五字")
    # Source code points are retained; no simplified/traditional guess is made.
    def cjk(character):
        code = ord(character)
        return (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
                or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF)
    if not all(cjk(c) for c in upper + suffix):
        raise ValueError("请只输入汉字；标点和缺字占位符无需填写")
    return VisibleQuery(len(upper), upper, suffix)


def load_training_poems(path: Path) -> list[dict]:
    result = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            split = SPLIT_PATTERN.search(line)
            if split is None:
                raise ValueError("source record has no recognized split")
            if split.group(1) != b"train":
                continue  # Do not decode held-out source records.
            poem = json.loads(line)
            if poem.get("split") != "train":
                raise ValueError("training split marker does not match record")
            result.append(poem)
    if not result:
        raise ValueError("no training poems available")
    return result


def build_statistical_retriever(root: Path = PROJECT_ROOT) -> PhraseRetriever:
    raw = json.loads((root / "data/corpus_v03/vocabulary.json").read_text(encoding="utf-8"))
    vocabulary = {token: index for index, token in enumerate(raw["tokens"])}
    model = PhraseRetriever(load_training_poems(root / "data/corpus_v03/poems.jsonl"), vocabulary)
    resources = json.loads((root / "results/retrieval_v04/candidate_resources.json").read_text(encoding="utf-8"))
    if model.candidate_resources() != resources:
        raise ValueError("training candidates differ from the frozen v0.4 vocabulary")
    return model


def selected_checkpoint(run: Path, seed: int = 0) -> tuple[Path, dict]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("seed must be 0, 1 or 2; the default 0 is fixed before evaluation")
    run = run.resolve()
    selection = json.loads((run / "selection_frozen.json").read_text(encoding="utf-8"))
    primary = selection["primary_arm"]
    if primary != "full_w3":
        raise ValueError("the declared primary architecture changed")
    arm = selection["arms"][primary]
    records = [x for x in arm["models"] if x["seed"] == seed]
    if len(records) != 1:
        raise ValueError("one exact checkpoint reference is required")
    record = records[0]
    path = (run / record["path"]).resolve()
    if not path.is_relative_to(run):
        raise ValueError("checkpoint reference escapes the experiment directory")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != record["sha256"]:
        raise ValueError("checkpoint checksum differs from the frozen selection")
    return path, {"arm": primary, "seed": seed, "step": arm["step"],
                  "sha256": observed, "selection_scope": selection["selection_scope"]}


def _suggestions(query: VisibleQuery, ranked: dict) -> list[dict]:
    return [{"rank": index + 1, "span": span,
             "completed_lower_line": span + query.lower_suffix, "score": float(score)}
            for index, (span, score) in enumerate(zip(ranked["candidates"], ranked["scores"]))]


def complete(upper: str, suffix: str, *, method: str = "compare", seed: int = 0,
             limit: int = 16, root: Path = PROJECT_ROOT, run: Path | None = None) -> dict:
    query = build_query(upper, suffix)
    if method not in ("compare", "statistical", "matching"):
        raise ValueError("unknown completion method")
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("limit must be an integer from 1 to 16")
    result = {
        "schema": "versematrix.completion.v0.5", "query": asdict(query),
        "query_hash": digest(asdict(query)), "requested_candidates": limit,
        "task": "initial_two_character_source_span_completion",
        "interpretation": "候选补句；分数衡量语料原文匹配，不是格律认证或文学质量评分。",
        "orthography": "exact_codepoints_no_simplified_traditional_conversion",
        "methods": {},
    }
    vocabulary = None
    if method in ("compare", "statistical"):
        baseline = build_statistical_retriever(root)
        ranked = baseline.rank(query, alignment_weight=.25, boundary_weight=.25,
                               pool="wide", limit=limit)
        result["methods"]["statistical"] = {
            "label": "固定统计基线", "candidate_count": len(baseline.candidates_by_length[query.length]),
            "candidate_vocabulary_hash": digest(baseline.candidates_by_length[query.length]),
            "suggestions": _suggestions(query, ranked),
        }
        vocabulary = baseline.vocabulary
    if method in ("compare", "matching"):
        from .matching import ContextSpanMatcher
        checkpoint, identity = selected_checkpoint(run or root / "results/matching_v05", seed)
        learned = ContextSpanMatcher.load(checkpoint)
        if learned.step_count != identity["step"]:
            raise ValueError("checkpoint optimizer step differs from the frozen selection")
        if (learned.seed != seed or learned.kernel_width != 3
                or learned.context_mode != "both"):
            raise ValueError("checkpoint architecture or seed differs from the selected primary model")
        ranked = learned.rank(query, limit=limit)
        candidates = learned.candidates_by_length[query.length]
        candidate_hash = digest(candidates)
        if "statistical" in result["methods"]:
            if candidate_hash != result["methods"]["statistical"]["candidate_vocabulary_hash"]:
                raise ValueError("learned and statistical methods have different candidate vocabularies")
        result["methods"]["matching"] = {
            "label": "上下文矩阵匹配（实验模型）", "candidate_count": len(candidates),
            "candidate_vocabulary_hash": candidate_hash, "checkpoint": identity,
            "seed_policy": "default seed 0 fixed before evaluation; not a selected best seed or ensemble",
            "suggestions": _suggestions(query, ranked),
        }
        vocabulary = learned.vocabulary
    result["unknown_visible_characters"] = sorted({c for c in upper.strip() + suffix.strip()
                                                    if c not in vocabulary})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="输入上句和下句后缀，返回至多16个二字补句候选。")
    parser.add_argument("--upper", required=True, help="上句：5字或7字，不带标点")
    parser.add_argument("--suffix", required=True, help="下句已知后缀：3字或5字")
    parser.add_argument("--method", choices=("compare", "statistical", "matching"), default="compare")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT,
                        help="含data和results的项目目录；脚本或可编辑安装会自动定位")
    parser.add_argument("--run", type=Path, help="实验结果目录；默认随包的results/matching_v05")
    parser.add_argument("--json", action="store_true", help="输出可复用JSON记录")
    args = parser.parse_args(argv)
    try:
        result = complete(args.upper, args.suffix, method=args.method, seed=args.seed,
                          limit=args.limit, run=args.run, root=args.project_dir.resolve())
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(result["query"]["upper_line"] + "，□□" + result["query"]["lower_suffix"] + "。")
        for model in result["methods"].values():
            print("\n" + model["label"])
            for suggestion in model["suggestions"]:
                print(f"{suggestion['rank']:2d}. {suggestion['completed_lower_line']}  ({suggestion['score']:.6f})")
        print("\n" + result["interpretation"])
        if result["unknown_visible_characters"]:
            print("训练字表未覆盖：" + "、".join(result["unknown_visible_characters"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
