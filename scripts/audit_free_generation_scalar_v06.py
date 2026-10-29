"""Persist the seven synthetic scalar LM checks used before the free-poem run.

This is an engineering replay of synthetic checks, not another poem generation.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wordconv_lab.free_generation import score_lines
from wordconv_lab.restoration import BidirNGram
from audit_free_poem_v06 import ScalarLanguageOracle


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    poems = [
        {"poem_id": "a", "split": "train", "lines": [
            "ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXY", "ZABCD", "EFGHI", "JKLMN"]},
        {"poem_id": "b", "split": "train", "lines": [
            "ABCDQ", "FGHIS", "KLMNT", "PQRSA", "UVWXY", "ZACDE", "EFGIJ", "JKLMO"]},
    ]
    characters = sorted(set("".join(line for poem in poems for line in poem["lines"])))
    vocabulary = {char: index for index, char in enumerate(["<UNK>"] + characters)}
    model = BidirNGram(vocabulary).fit(poems)
    oracle = ScalarLanguageOracle(poems)
    cases = ["ABCDE", "ABCDQ", "ACFIK", "ZZZZZ", "?????", "A?C?E", "GFEDC"]
    actual = score_lines(model, cases)
    expected = np.asarray([oracle.score(text) for text in cases])
    difference = np.abs(actual - expected)
    if float(np.max(difference)) >= 1e-12:
        raise AssertionError("independent synthetic scalar score mismatch")
    receipt = {
        "status": "PASS_INDEPENDENT_SYNTHETIC_SCALAR_LM",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_generation_receipt_replay_of_same_seven_pre_generation_synthetic_checks",
        "cases": [{"text": text, "vectorized": float(a), "independent_scalar": float(b),
                   "absolute_error": float(error)}
                  for text, a, b, error in zip(cases, actual, expected, difference)],
        "max_absolute_error": float(np.max(difference)),
        "vocabulary_size": len(vocabulary),
        "real_corpus_read": False,
        "real_poem_generated": False,
        "synthetic_source_poems": poems,
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "scripts/audit_free_poem_v06.py",
            ROOT / "src/wordconv_lab/free_generation.py",
            ROOT / "src/wordconv_lab/restoration.py", ROOT / "src/wordconv_lab/retrieval.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "cases": len(cases),
        "max_absolute_error": receipt["max_absolute_error"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
