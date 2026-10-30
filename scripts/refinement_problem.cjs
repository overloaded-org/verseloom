'use strict';
// Adapts the unchanged corpus lattice and trained continuation weights to full
// eight-line rescoring. No hand-authored semantic filters or POS assumptions.
const assert=require('node:assert/strict'), crypto=require('node:crypto');
const digest=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const ids=l=>l.tokens.join(',');
const key=l=>l.text+'|'+ids(l);
const hash=s=>{let h=2166136261;for(const c of s)h=Math.imul(h^c.codePointAt(0),16777619)>>>0;return h;};
const order=(a,b)=>a<b?-1:a>b?1:0;
function makeProblem(data, model, rankerAPI, lattice, baseline, limit=24) {
  assert.equal(baseline.status,'complete');
  assert.equal(lattice.schema,'verseloom.research-lattice.v1');
  assert.deepEqual(lattice.sourceIdentity,data.sourceIdentity);
  assert.deepEqual(baseline.provenance.sourceIdentity,data.sourceIdentity);
  assert.deepEqual(model.sourceIdentity,data.sourceIdentity);
  assert.equal(baseline.poolFingerprint,lattice.poolFingerprint);
  assert.equal(baseline.engineVersion,lattice.engineVersion);
  assert.deepEqual(baseline.options,lattice.options);
  assert.equal(baseline.options.mode,'learned');
  assert.deepEqual(baseline.options.recentPoems,[],'Recency is intentionally excluded from this experiment');
  assert(Number.isInteger(limit)&&limit>=2&&limit<=128);
  const n=baseline.length,template=data.templates[String(n)].find(t=>t.id===baseline.templateId);
  assert(template);const rhyme=baseline.rhymeGroup;
  const lex=data.lexicon.map((w,id)=>({...w,id,size:Array.from(w.text).length}));
  const maps={};for(const name of ['association','following','alignment'])
    maps[name]=new Map(Object.entries(data[name]).map(([u,edges])=>[+u,new Map(edges.map(([v,count,score])=>[v,{count,score}]))]));
  const learned=rankerAPI.create(data,model,{lex,maps});
  const byPool=new Map(lattice.pools),initial=[],pools=[];
  for(let row=0;row<8;row++){
    const group=template.rhymeRows.includes(row)?rhyme:'nonrhyming';
    const all=template.variants[row].flatMap(t=>byPool.get(t+'/'+group)||[]);
    const ref=baseline.lineDetails[row], tokenIds=ref.tokens.map(t=>t.id).join(',');
    const original=all.find(l=>l.text===baseline.lines[row]&&ids(l)===tokenIds);
    assert(original,'Initial line is absent from the identical runtime lattice');
    assert(Math.abs(original.score-ref.localScore)<1e-12,'Baseline local score mismatch');
    const ranked=[...new Map(all.map(l=>[key(l),l])).values()]
      .sort((a,b)=>b.rank-a.rank||order(key(a),key(b)));
    const chosen=ranked.filter(l=>key(l)!==key(original)).slice(0,limit-1).concat(original)
      .sort((a,b)=>order(key(a),key(b)));
    for(const l of chosen){Object.freeze(l.tokens);Object.freeze(l);}
    pools.push(Object.freeze(chosen));initial.push(original);
  }
  const membership=pools.map(xs=>new Set(xs)),banned=new Set(data.trainingLineHashes);
  const cached=new Map(),continuations=new Map();let cacheMisses=0;
  function evaluate(lines){
    assert(Array.isArray(lines)&&lines.length===8);
    assert(lines.every((l,i)=>membership[i].has(l)),'Unknown candidate injection');
    const signature=lines.map(key).join('/');
    if(cached.has(signature))return cached.get(signature);
    cacheMisses++;
    const chars=new Map(),ends=new Set(),tokens=new Set(),texts=new Set();
    const rejected=()=>{cached.set(signature,null);return null;};
    let score=0,localTotal=0,relationTotal=0,repeatTotal=0;const detail=[];
    for(let row=0;row<8;row++){
      const line=lines[row],text=line.text,cs=Array.from(text);
      if(cs.length!==n||new Set(cs).size!==n||texts.has(text)||banned.has(hash(text)))return rejected();
      if(!template.variants[row].includes(line.tones))return rejected();
      if(line.tokens.map(i=>lex[i].text).join('')!==text)return rejected();
      if(cs.map(c=>data.readings[c].tone).join('')!==line.tones)return rejected();
      if(template.rhymeRows.includes(row)){
        const end=cs.at(-1);if(line.rhymeGroup!==rhyme||data.readings[end].rhymeGroup!==rhyme||ends.has(end))return rejected();
        ends.add(end);
      }
      for(const id of line.tokens)if(lex[id].size>1){if(tokens.has(id))return rejected();tokens.add(id);}
      let repeated=0;for(const c of cs){const count=chars.get(c)||0;if(count>=2)return rejected();repeated+=count;chars.set(c,count+1);}
      texts.add(text);
      // Re-score every downstream line after any edit, including tail changes.
      const prefix=lines.slice(0,row+1).map(key).join('/');
      let rel=continuations.get(prefix);
      if(!rel){rel=learned(lines.slice(0,row),line);continuations.set(prefix,rel);}
      const local=line.score/n,penalty=.3*repeated,increment=local+rel.value-penalty;
      localTotal+=local;relationTotal+=rel.value;repeatTotal+=penalty;score+=increment;
      detail.push({row,text,localScore:line.score,relation:rel,repetitionPenalty:penalty,increment});
    }
    const result={score,localTotal,relationTotal,repeatTotal,detail};cached.set(signature,result);return result;
  }
  const value=evaluate(initial);assert(value,'Initial poem violates hard constraints');
  assert(Math.abs(value.score-baseline.score)<1e-9,'Full-score adapter changes baseline semantics');
  const candidateDigest=digest(pools.map(xs=>xs.map(l=>({text:l.text,tokens:l.tokens,score:l.score,tones:l.tones,rhymeGroup:l.rhymeGroup}))));
  function serialize(state,value){return {
    schema:'verseloom.refinement-output.v1',status:'complete',length:n,
    lines:state.map(l=>l.text),templateId:template.id,rhymeGroup:rhyme,rhymeRows:template.rhymeRows,
    score:value.score,objectiveComponents:{local:value.localTotal,learned:value.relationTotal,repetition:value.repeatTotal},
    lineDetails:state.map((l,i)=>({text:l.text,tones:l.tones,localScore:l.score,
      tokens:l.tokens.map(id=>({id,text:lex[id].text,source:lex[id].source})),transitions:l.edges,
      relations:value.detail[i].relation,historyPenalty:0})),
    sourceIdentity:data.sourceIdentity,candidateDigest,poolFingerprint:lattice.poolFingerprint,
    provenance:{authoredContentLexemes:0,authoredSentenceFrames:0,preselectedLines:false,
      generatedInitialization:true,formalProfileFixedForComparison:true,
      diffusionModel:false,trainedDenoisingModel:false,
      strictAntithesisCertified:false,contextualPronunciationCertified:false,literaryQualityCertified:false}
  };}
  return {initial:Object.freeze(initial),pools:Object.freeze(pools),evaluate,serialize,
    candidateDigest,baselineScore:value.score,templateId:template.id,rhymeGroup:rhyme,
    snapshot:()=>({candidateDigest,length:n,templateId:template.id,rhymeGroup:rhyme,
      initial:initial.map(key),pools:pools.map(xs=>xs.map(l=>({text:l.text,tokens:l.tokens,score:l.score,tones:l.tones,rhymeGroup:l.rhymeGroup})))}),
    counts:()=>({fullScoreCacheMisses:cacheMisses,uniquePrefixesScored:continuations.size})};
}
module.exports={makeProblem,digest};
