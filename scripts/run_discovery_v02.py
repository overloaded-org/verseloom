"""Document the basic prosody grammar and verify exact local-factor optimization.

Every scoring benchmark here uses UNTRAINED random convolution weights. The
benchmark verifies transformation/optimization, never poetic or learned quality.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from wordconv_lab.convolution import LocalConvScorer
from wordconv_lab.factor_dp import convolution_factor, path_score, solve_local_chain
from wordconv_lab.prosody import all_templates, audit_tones


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite a discovery evidence directory")
    output.mkdir(parents=True)
    sources = [root / "src/wordconv_lab/factor_dp.py", root / "src/wordconv_lab/prosody.py",
               root / "src/wordconv_lab/convolution.py", Path(__file__)]
    manifest = {"status": "frozen_before_execution", "seeds": [0, 1], "lengths": [5, 7],
                "domain": "two binary token choices in each line at each column; four superstates",
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    templates = [t.to_dict() for length in (5, 7) for t in all_templates(length)]
    (output / "basic_templates.json").write_text(json.dumps(templates,ensure_ascii=False,indent=2)+"\n")
    examples = json.loads((root / "data/worked_examples_v02.json").read_text())
    # The source file may use a wrapper; keep its own provenance unchanged.
    if isinstance(examples, dict):
        example_list = examples.get("poems", examples.get("examples", []))
    else:
        example_list = examples
    annotations=[]
    for item in example_list:
        tones=item.get("published_tones") or item.get("pz") or item.get("tones_pz") or item.get("tone_patterns")
        if tones:
            annotations.append({"title":item["title"], "status":"source_annotation_worked_example_not_blind_test",
                                "audit":audit_tones(tones)})
    results=[]
    for length in (5,7):
        for seed in (0,1):
            model=LocalConvScorer(4,seed=seed)
            def feature(row,col,token):
                return np.array([float(token=="A"),float(token=="B"),row,col/(length-1)])
            domain=tuple(itertools.product(("A","B"),repeat=2))
            domains=(domain,)*length
            factor=convolution_factor(model,feature,length)
            start=time.perf_counter()
            result=solve_local_chain(domains,factor)
            dp_time=time.perf_counter()-start
            best_score=-float("inf")
            best_path=None
            max_factor_error=0.0
            start=time.perf_counter()
            for path in itertools.product(*domains):
                matrix=np.array([[feature(row,col,path[col][row]) for col in range(length)] for row in (0,1)])
                # Independent oracle directly calls the original model, not the factored representation.
                score=model.score(matrix)
                if score>best_score:
                    best_score,best_path=score,path
            oracle_time=time.perf_counter()-start
            chosen=np.array([[feature(row,col,result.path[col][row]) for col in range(length)] for row in (0,1)])
            direct_score=model.score(chosen)
            factor_error=abs(direct_score-result.score)
            optimum_error=abs(best_score-direct_score)
            if max(factor_error,optimum_error)>1e-12:
                raise AssertionError("factorization or exact optimum validation failed")
            item=asdict(result)
            item.update({"length":length,"seed":seed,"trained":False,
                         "direct_original_score":direct_score,"oracle_score":best_score,
                         "factorization_abs_error":factor_error,"optimality_abs_error":optimum_error,
                         "dp_seconds":dp_time,"oracle_seconds":oracle_time,
                         "oracle_original_model_calls":4**length,
                         "call_units_warning":"local factor calls and whole-model calls are different units; do not divide as throughput"})
            results.append(item)
    report={"version":"0.2.0","status":"exact_algorithm_verified_in_declared_scope",
            "grammar_templates":len(templates),"scorer_trained":False,"human_evaluated":False,
            "worked_examples":annotations,"dp_checks":results,
            "limits":["Finite declared domains, additive radius-one factors and fixed positional token features only.",
                      "Arbitrary global semantic scores or unresolved global constraints invalidate this small DP state.",
                      "Global optimum is of the internal mathematical objective, not literary quality.",
                      "Common-prefix grammar is deliberately incomplete; unsupported verse is not declared invalid."]}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results/discovery_v02"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=run(root,args.out)
    print(json.dumps({"status":report["status"],"templates":report["grammar_templates"],
                      "dp_cases":len(report["dp_checks"]),"human_evaluated":False}))
