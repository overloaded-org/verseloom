"""Independent scalar mathematical oracle for the v0.5 matching model.

Run this script from any directory to replay the synthetic gradient audit.
It prints its receipt and does not write or modify model/data artifacts.
The factors function can also independently rebuild a saved parameter snapshot.
"""
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))

def independent_scalar_factors(parameters, configuration, vocabulary, candidates, queries, mask="none"):
    import math
    import numpy as np
    e = configuration["embedding_dim"]
    c = configuration["channels"]
    d = configuration["representation_dim"]
    width = configuration["kernel_width"]
    radius = width // 2
    mode = configuration["context_mode"]
    epsilon = configuration["norm_epsilon"]
    temperature = configuration["temperature"]
    p = parameters
    def normalized(raw):
        denominator = math.sqrt(math.fsum(float(x)*float(x) for x in raw) + epsilon*epsilon)
        return [float(x)/denominator for x in raw]
    candidate_vectors = []
    for span in candidates:
        joined = [float(x) for ch in span for x in p["embedding"][vocabulary[ch]]]
        raw = [math.tanh(math.fsum(joined[j]*float(p["span_projection"][j,k])
                                  for j in range(2*e)) + float(p["span_projection_bias"][k]))
               for k in range(d)]
        candidate_vectors.append(normalized(raw))
    query_vectors = []
    for query in queries:
        tokens = [list(query.upper_line) + [None]*(7-query.length),
                  [None,None] + list(query.lower_suffix) + [None]*(7-query.length)]
        active = [[tokens[row][pos] is not None for pos in range(7)] for row in range(2)]
        if mode == "suffix" or mask in ("upper","both"):
            active[0] = [False]*7
        if mode == "upper" or mask in ("suffix","both"):
            active[1] = [False]*7
        hidden = [[[0.0]*c for pos in range(7)] for row in range(2)]
        for row in range(2):
            for pos in range(7):
                if not active[row][pos]:
                    continue
                for channel in range(c):
                    terms = [float(p["context_bias"][channel])]
                    for offset in range(width):
                        source_pos = pos + offset - radius
                        if 0 <= source_pos < 7 and active[row][source_pos]:
                            token_id = vocabulary.get(tokens[row][source_pos],0)
                            terms.extend(float(p["embedding"][token_id,j])*float(p["context_kernel"][offset,j,channel])
                                         for j in range(e))
                    hidden[row][pos][channel] = math.tanh(math.fsum(terms))
        joined = [x for position in hidden[0] + hidden[1][2:7] for x in position]
        raw = [math.tanh(math.fsum(joined[j]*float(p["query_projection"][j,k]) for j in range(12*c))
                         + float(p["query_bias"][k])) for k in range(d)]
        query_vectors.append([x/temperature for x in normalized(raw)])
    scores = [[math.fsum(q[k]*v[k] for k in range(d)) for v in candidate_vectors] for q in query_vectors]
    return np.asarray(query_vectors),np.asarray(candidate_vectors),np.asarray(scores)

def independent_scalar_loss(parameters,configuration,vocabulary,candidates,queries,targets):
    import math
    scores = independent_scalar_factors(parameters,configuration,vocabulary,candidates,queries)[2]
    losses = []
    for row,target in zip(scores,targets):
        maximum = max(row)
        losses.append(math.log(math.fsum(math.exp(float(x)-float(maximum)) for x in row))
                      + float(maximum) - float(row[target]))
    return math.fsum(losses)/len(losses)


def run_numeric_check():
    from pathlib import Path
    import hashlib,json
    import numpy as np
    from wordconv_lab.matching import ContextSpanMatcher
    from wordconv_lab.restoration import VisibleQuery
    source=PROJECT_ROOT/'src/wordconv_lab/matching.py'
    initial_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    queries_by_length={
      5:[VisibleQuery(5,"甲甲乙丙丁","乙甲甲"),VisibleQuery(5,"乙甲乙未知","丙丁甲")],
      7:[VisibleQuery(7,"甲甲乙丙丁甲乙","乙甲甲丙丁"),VisibleQuery(7,"乙甲乙未知甲丙","丙丁甲乙甲")],
    }
    vocabulary={s:i for i,s in enumerate(["<UNK>"]+sorted(set("甲乙丙丁")))}
    candidates=tuple(sorted(["甲乙","乙甲","甲甲","丙丁"]))
    cases=[]
    for width in (1,3):
        for length in (5,7):
            model=ContextSpanMatcher(vocabulary,{length:candidates},embedding_dim=3,representation_dim=4,
                                    kernel_width=width,channels=3,seed=17+length+width)
            qs=queries_by_length[length];targets=np.array([0,2])
            actual,gradients=model.loss_and_gradients(qs,targets)
            q,e,s=independent_scalar_factors(model.params,model.config,vocabulary,candidates,qs)
            expected=independent_scalar_loss(model.params,model.config,vocabulary,candidates,qs,targets)
            error=float(np.max(np.abs(s-model.score_all(qs))))
            assert abs(actual-expected)<1e-12 and error<1e-12
            for row,query in enumerate(qs):
                ranked=model.rank(query,limit=4)['candidates']
                order=sorted(range(len(candidates)),key=lambda i:(-s[row,i],candidates[i]))
                assert ranked==tuple(candidates[i] for i in order)
            samples=[]
            for name,values in model.params.items():
                eligible=np.flatnonzero(model._trainable_masks[name])
                positions=sorted(set([int(eligible[0]),int(eligible[len(eligible)//3]),
                                      int(eligible[2*len(eligible)//3]),int(eligible[-1])]))
                for flat in positions:
                    index=np.unravel_index(flat,values.shape);old=float(values[index]);epsilon=1e-6
                    try:
                        values[index]=old+epsilon
                        plus=independent_scalar_loss(model.params,model.config,vocabulary,candidates,qs,targets)
                        values[index]=old-epsilon
                        minus=independent_scalar_loss(model.params,model.config,vocabulary,candidates,qs,targets)
                    finally:
                        values[index]=old
                    numerical=(plus-minus)/(2*epsilon);analytical=float(gradients[name][index])
                    difference=abs(numerical-analytical)
                    scaled=difference/max(1.,abs(numerical),abs(analytical))
                    assert scaled<3e-5,(width,length,name,index,numerical,analytical,scaled)
                    samples.append({'parameter':name,'index':[int(i) for i in index],
                                    'analytic':analytical,'finite_difference':numerical,
                                    'absolute_error':difference,'scaled_error':scaled})
            assert np.array_equal(gradients['embedding'][0],np.zeros(model.embedding_dim))
            for mask in ('upper','suffix','both'):
                manual=independent_scalar_factors(model.params,model.config,vocabulary,candidates,qs,mask)[2]
                assert np.max(np.abs(manual-model.score_all(qs,mask=mask)))<1e-12
            cases.append({'kernel_width':width,'length':length,'loss_error':abs(actual-expected),
                          'maximum_score_error':error,'gradient_coordinates':len(samples),
                          'max_scaled_gradient_error':max(r['scaled_error'] for r in samples),'samples':samples})
    assert hashlib.sha256(source.read_bytes()).hexdigest()==initial_hash
    return {'status':'PASS_INDEPENDENT_SCALAR_MATH','source_sha256':initial_hash,
            'method':'Explicit scalar convolution, ordered span encoder, smooth normalization, fixed temperature and stable full-softmax loss; center differences of that separate scalar implementation.',
            'new_core_forward_used_for_oracle':False,'epsilon':1e-6,'scaled_gradient_tolerance':3e-5,
            'cases':cases,'gradient_coordinates':sum(r['gradient_coordinates'] for r in cases),
            'maximum_scaled_gradient_error':max(r['max_scaled_gradient_error'] for r in cases),
            'maximum_forward_score_error':max(r['maximum_score_error'] for r in cases),
            'scope':'Synthetic 5/7 cases, widths 1/3, repeated shared characters and visible UNK; no real dev/test/confirmation scores used.'}


if __name__ == '__main__':
    import json
    print(json.dumps(run_numeric_check(),ensure_ascii=False,indent=2))
