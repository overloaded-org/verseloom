'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
require('../web/induced-data.js');require('../web/continuation-data.js');require('../web/continuation-ranker.js');
const engine=require('../web/induced-engine.js');
const {makeProblem}=require('../scripts/refinement_problem.cjs');
const options={length:5,seed:0,variation:.8,beamWidth:16,mode:'learned',recentPoems:[]};
const strip=({elapsedMs,...r})=>r;
let fixture;
async function build(){if(!fixture){const baseline=await engine.generate(VerseLoomInducedData,options);
 const lattice=await engine.researchLattice(VerseLoomInducedData,options);
 fixture={baseline,lattice};}return fixture;}
test('CommonJS research access does not expand the browser content API',()=>{
 assert.equal(VerseLoomInduced.researchLattice,undefined);assert.equal(typeof engine.researchLattice,'function');
});
test('research hook rejects fixed lines, topics and content controls',async()=>{
 for(const k of ['fixedLines','opening','story','theme','season','sentenceFrames'])
  await assert.rejects(engine.researchLattice(VerseLoomInducedData,{[k]:'unused'}));
});
test('Node-only hook does not alter production output under the same code',async()=>{
 const a=await engine.generate(VerseLoomInducedData,options);
 const c=vm.createContext({setTimeout,performance,VerseLoomInducedData,VerseLoomContinuationData,VerseLoomContinuation});
 vm.runInContext(fs.readFileSync(require.resolve('../web/induced-engine.js'),'utf8'),c);
 const b=await c.VerseLoomInduced.generate(VerseLoomInducedData,options);
 assert.equal(JSON.stringify(strip(b)),JSON.stringify(strip(a)));
});
test('lattice snapshot is detached from cached production candidates',async()=>{
 const {baseline}=await build();const x=await engine.researchLattice(VerseLoomInducedData,options);
 x.pools[0][1][0].text='tampered';
 const y=await engine.researchLattice(VerseLoomInducedData,options);
 assert.notEqual(y.pools[0][1][0].text,'tampered');
 assert.deepEqual(strip(await engine.generate(VerseLoomInducedData,options)),strip(baseline));
});
test('full-poem objective exactly reproduces baseline scoring and constraints',async()=>{
 const {baseline,lattice}=await build();const p=makeProblem(VerseLoomInducedData,VerseLoomContinuationData,VerseLoomContinuation,lattice,baseline);
 assert(Math.abs(p.evaluate(p.initial).score-baseline.score)<1e-9);
 assert.equal(p.pools.length,8);assert(p.pools.every((row,i)=>row.includes(p.initial[i])&&row.length<=24));
 assert.throws(()=>p.evaluate([{...p.initial[0]},...p.initial.slice(1)]),/Unknown candidate/);
});
test('stale sources, invalid seed receipts and unexpected histories fail closed',async()=>{
 const {baseline,lattice}=await build();
 const call=(l,b)=>makeProblem(VerseLoomInducedData,VerseLoomContinuationData,VerseLoomContinuation,l,b);
 assert.throws(()=>call({...lattice,poolFingerprint:0},baseline));
 assert.throws(()=>call({...lattice,sourceIdentity:{}},baseline));
 assert.throws(()=>call(lattice,{...baseline,options:{...baseline.options,seed:999}}));
 assert.throws(()=>call(lattice,{...baseline,score:baseline.score+10}),/baseline semantics/);
});
test('production hook content contains no semantic tables or copied poem fixtures',()=>{
 const s=fs.readFileSync(require.resolve('../web/induced-engine.js'),'utf8').split('// Node-only, detached research lattice.')[1];
 for(const x of ['lexicon_spec','subjectEntities','美少年','虎眠'])assert(!s.includes(x));
});
