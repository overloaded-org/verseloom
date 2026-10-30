'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
require('../web/induced-data.js');require('../web/continuation-data.js');require('../web/continuation-ranker.js');require('../web/induced-engine.js');
const d=VerseLoomInducedData,m=VerseLoomContinuationData,ranker=VerseLoomContinuation,eng=VerseLoomInduced;
const runtime={lex:d.lexicon.map(w=>({...w,size:Array.from(w.text).length})),maps:{}};
for(const k of ['association','following','alignment'])runtime.maps[k]=new Map(Object.entries(d[k]).map(([u,es])=>[+u,new Map(es.map(([v,count,score])=>[v,{count,score}]))]));
const strip=({elapsedMs,...v})=>v;
test('Python and JS features agree, including unknown segments',()=>{
 const file=process.env.CONTINUATION_FIXTURES;assert(file,'CONTINUATION_FIXTURES required');
 const fixtures=JSON.parse(fs.readFileSync(file));assert(fixtures.length>=32);
 for(const row of fixtures){const actual=ranker.extractSpans(row.context,row.target,runtime.lex,runtime.maps);
  actual.forEach((x,i)=>assert(Math.abs(x-row.features[i])<1e-12));}
});
test('wrong source identity and nonfinite weights are rejected',()=>{
 const wrong=structuredClone(m);wrong.sourceIdentity.newCandidateCorpus='different';assert.throws(()=>ranker.create(d,wrong,runtime));
 for(const bad of [NaN,Infinity]){const wrong=structuredClone(m);wrong.coefficients[0]=bad;assert.throws(()=>ranker.create(d,wrong,runtime));}
});
test('same pools and deterministic replay under learned scoring',async()=>{
 const o={length:5,seed:42,beamWidth:8};
 const a=await eng.generate(d,{...o,mode:'relations'}),b=await eng.generate(d,{...o,mode:'learned'});
 assert.equal(a.poolFingerprint,b.poolFingerprint);assert.equal(b.status,'complete');assert(b.checks.every(x=>x.passed));
 assert.deepEqual(strip(b),strip(await eng.generate(d,b.options)));
 assert(b.lineDetails.slice(1).every(l=>l.relations.learned.features.length===8));
 assert.equal(b.provenance.authoredContentLexemes,0);assert.equal(b.provenance.literaryQualityCertified,false);
});
test('missing learned model fails closed without output substitution',async()=>{
 const saved=globalThis.VerseLoomContinuationData;try{delete globalThis.VerseLoomContinuationData;
  await assert.rejects(eng.generate(d,{mode:'learned'}),/资源缺失/);
 }finally{globalThis.VerseLoomContinuationData=saved;}
});
test('no theme, story, fixed-line or custom content control is introduced',async()=>{
 for(const key of ['theme','story','fixedLines','opening','semanticRoles'])await assert.rejects(eng.generate(d,{mode:'learned',[key]:'test'}));
 assert.equal(m.authoredContentLexemes,0);assert.equal(m.labels,'observed_source_adjacency_not_literary_gold');
});
