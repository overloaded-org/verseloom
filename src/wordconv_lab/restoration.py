"""Train-only retrieval and paired metrics for source-text restoration.

Targets are observed source strings, NOT labels of literary quality. Public
query objects cannot carry the target, poem title, author or complete lower line.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import math

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(",", ":"),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VisibleQuery:
    length: int
    upper_line: str
    lower_suffix: str
    start: int = 0
    mask_length: int = 2

    def __post_init__(self):
        if self.length not in (5,7) or self.start != 0 or self.mask_length != 2:
            raise ValueError("supports initial two-character mask in five/seven character lines")
        if len(self.upper_line)!=self.length or len(self.lower_suffix)!=self.length-2:
            raise ValueError("visible line shape mismatch")

    @classmethod
    def from_task(cls,task):
        return cls(**{k:task[k] for k in ("length","upper_line","lower_suffix","start","mask_length")})


class BidirNGram:
    """Fixed bidirectional interpolated trigram plus aligned-character counts.

    Weights and smoothing are design choices frozen before model evaluation.
    This baseline also proposes the main candidate pool; architectural claims
    must therefore remain conditional on this statistical proposal mechanism.
    """
    def __init__(self,vocabulary,alpha=.1,weights=(.1,.3,.6),aligned_weight=.5,aligned_strength=5.):
        self.vocabulary=dict(vocabulary)
        if sorted(self.vocabulary.values())!=list(range(len(self.vocabulary))):
            raise ValueError("vocabulary IDs must be consecutive from zero")
        self.v=len(vocabulary)
        self.eos=self.v
        self.alpha=alpha
        self.weights=weights
        self.aligned_weight=aligned_weight
        self.aligned_strength=aligned_strength
        self.tables=[dict(unigram=Counter(),bigram=Counter(),trigram=Counter(),
                          context1=Counter(),context2=Counter()) for _ in range(2)]
        self.aligned=Counter()
        self.upper=Counter()
        self.lower=Counter()
        self.fit_poem_ids=()
        self.fitted=False

    def encode(self,text):
        return [self.vocabulary.get(x,0) for x in text]

    def fit(self,poems):
        if self.fitted:
            raise ValueError("model already fit; create a new immutable training snapshot")
        if not poems or any(p["split"]!="train" for p in poems):
            raise ValueError("language statistics may read training poems only")
        for poem in poems:
            if len(poem["lines"])!=8:
                raise ValueError("requires full eight-line source records")
            encoded=[self.encode(line) for line in poem["lines"]]
            for line in encoded:
                for direction,seq in enumerate((line,list(reversed(line)))):
                    table=self.tables[direction]
                    a,b=-1,-1
                    for target in seq+[self.eos]:
                        table["unigram"][target]+=1
                        table["bigram"][(b,target)]+=1
                        table["trigram"][(a,b,target)]+=1
                        table["context1"][b]+=1
                        table["context2"][(a,b)]+=1
                        a,b=b,target
            for row in (0,2,4,6):
                for u,l in zip(encoded[row],encoded[row+1]):
                    self.aligned[(u,l)]+=1
                    self.upper[u]+=1
                    self.lower[l]+=1
        self.fit_poem_ids=tuple(sorted(p["poem_id"] for p in poems))
        self.totals=[sum(t["unigram"].values()) for t in self.tables]
        self.lower_total=sum(self.lower.values())
        self.fitted=True
        return self

    @lru_cache(maxsize=300000)
    def _log_probability(self,direction,a,b,target):
        table=self.tables[direction]
        alpha=self.alpha
        size=self.v+1
        p1=(table["unigram"][target]+alpha)/(self.totals[direction]+alpha*size)
        p2=(table["bigram"][(b,target)]+alpha)/(table["context1"][b]+alpha*size)
        p3=(table["trigram"][(a,b,target)]+alpha)/(table["context2"][(a,b)]+alpha*size)
        return math.log(sum(w*p for w,p in zip(self.weights,(p1,p2,p3))))

    @lru_cache(maxsize=100000)
    def _aligned_log_probability(self,upper,lower):
        prior=(self.lower[lower]+self.alpha)/(self.lower_total+self.alpha*self.v)
        probability=(self.aligned[(upper,lower)]+self.aligned_strength*prior)/(self.upper[upper]+self.aligned_strength)
        return math.log(probability)

    def score(self,query:VisibleQuery,span:str)->float:
        if not self.fitted or type(query) is not VisibleQuery:
            raise ValueError("fit model and pass an allowlisted VisibleQuery")
        if not isinstance(span,str) or len(span)!=2:
            raise ValueError("candidate must be one two-character source span")
        lower=self.encode(span+query.lower_suffix)
        upper=self.encode(query.upper_line)
        scores=[]
        for direction,line in enumerate((lower,list(reversed(lower)))):
            a,b=-1,-1
            total=0.
            for target in line+[self.eos]:
                total+=self._log_probability(direction,a,b,target)
                a,b=b,target
            scores.append(total/(query.length+1))
        aligned=sum(self._aligned_log_probability(u,l) for u,l in zip(upper,lower))/query.length
        return .5*sum(scores)+self.aligned_weight*aligned

    def propose(self,query:VisibleQuery,candidate_spans,limit=16):
        if type(limit) is not int or limit<1:
            raise ValueError("candidate limit must be positive")
        spans=tuple(candidate_spans)
        if not spans or len(set(spans))!=len(spans):
            raise ValueError("need nonempty distinct training-source spans")
        scored=[(self.score(query,span),span) for span in spans]
        selected=sorted(scored,key=lambda p:(-p[0],p[1]))[:limit]
        # Lexical order gives every scorer the same independent tie policy.
        selected.sort(key=lambda p:p[1])
        chosen=[span for _,span in selected]
        identity={"query":asdict(query),"candidates":chosen}
        return {"candidates":chosen,"ngram_scores":[score for score,_ in selected],
                "candidate_set_hash":digest(identity),"query_hash":digest(asdict(query))}

    def summary(self):
        return {"fit_poems":len(self.fit_poem_ids),"fit_poem_ids_sha256":digest(self.fit_poem_ids),
                "vocabulary_size_with_unk":self.v,"alpha":self.alpha,"weights":self.weights,
                "aligned_weight":self.aligned_weight,"aligned_strength":self.aligned_strength,
                "direction_unique_counts":[{k:len(v) for k,v in table.items()} for table in self.tables],
                "aligned_unique_pairs":len(self.aligned),"target_kind":"observed_source_restoration"}


def encode_candidates(query,candidates,vocabulary):
    if type(query) is not VisibleQuery:
        raise ValueError("only allowlisted visible query accepted")
    if not candidates or any(len(c)!=2 for c in candidates):
        raise ValueError("need two-character candidates")
    return np.asarray([[[vocabulary.get(c,0) for c in query.upper_line],
                        [vocabulary.get(c,0) for c in span+query.lower_suffix]]
                       for span in candidates],dtype=np.int64)


def top_candidate(candidates,scores):
    values=np.asarray(scores,dtype=float)
    if values.shape!=(len(candidates),) or not np.isfinite(values).all():
        raise ValueError("invalid candidate scores")
    return min(range(len(candidates)),key=lambda i:(-values[i],candidates[i]))


def family_metric(tasks,values):
    if len(tasks)!=len(values) or not tasks:
        raise ValueError("one value per nonempty task")
    grouped=defaultdict(list)
    for task,value in zip(tasks,values):
        if not np.isfinite(value) or not 0<=value<=1:
            raise ValueError("metric values must be finite and in [0,1]")
        grouped[task["family_id"]].append(float(value))
    return {"family_macro":float(np.mean([np.mean(x) for x in grouped.values()])),
            "task_micro":float(np.mean(values)),"tasks":len(tasks),"families":len(grouped)}


def paired_family_interval(tasks,a,b,seed=2026102903,repeats=5000):
    if len(a)!=len(tasks) or len(b)!=len(tasks):
        raise ValueError("paired values must cover identical tasks")
    grouped=defaultdict(list)
    for task,x,y in zip(tasks,a,b):
        grouped[task["family_id"]].append(float(x)-float(y))
    means=np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    if len(means)<2:
        return {"status":"insufficient_families","families":len(means)}
    rng=np.random.default_rng(seed)
    draws=np.array([means[rng.integers(0,len(means),len(means))].mean() for _ in range(repeats)])
    return {"status":"computed","families":len(means),"mean_difference":float(means.mean()),
            "confidence":.95,"interval":np.quantile(draws,[.025,.975]).tolist(),
            "repeats":repeats,"seed":seed,
            "estimand":"equal_family_mean_of_seed_averaged_paired_source_restoration_accuracy",
            "limitation":"conditional on the three fixed seeds and proposal mechanism; not literary preference"}
