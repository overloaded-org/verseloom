'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'..');require(path.join(root,'web/composition-data.js'));require(path.join(root,'web/composition-engine.js'));
const data=globalThis.VerseLoomCompositionData,engine=globalThis.VerseLoomComposition;
const options={length:5,seed:42,season:'spring',time:'dusk',templateId:'L5-Z-R0',rhymeGroup:'pingshui:lower_level:09'};
const clean=({elapsedMs,...r})=>r;
let first;
test('typed generation constructs all eight lines from lexical IDs, with no anchors',async()=>{
 first=await engine.generate(data,options);assert.equal(first.status,'complete');assert.equal(first.lines.length,8);
 assert.equal(first.provenance.preselectedLines,false);assert.equal(first.provenance.storedOutputLines,false);
 assert(first.lineDetails.every(l=>l.tokens.map(t=>t.text).join('')===l.text));assert(first.checks.every(c=>c.passed));
 assert.equal(first.provenance.selectionStage,'after-eight-lines');
});
test('same seed, version, data and inputs reproduce every decision',async()=>{
 const second=await engine.generate(data,options);assert.deepEqual(clean(first),clean(second));
});
test('manual seed zero and uint32 endpoint are valid; no ambient Math.random is used',async()=>{
 for(const seed of [0,4294967295]){const r=await engine.generate(data,{...options,seed});assert.equal(r.options.seed,seed);assert.equal(r.provenance.contextualPronunciationCertified,false);}
 assert(!fs.readFileSync(path.join(root,'web/composition-engine.js'),'utf8').includes('Math.random'));
});
test('different seeds change all sentence positions in the frozen diagnostic fixture',()=>{
 const file=process.env.VERSELOOM_COMPOSITION_REPORT;
 assert(file,'Run evaluate_composition_v2.cjs and set VERSELOOM_COMPOSITION_REPORT before this suite');
 const report=JSON.parse(fs.readFileSync(file));for(const s of report.summaries){assert(s.uniquePoems>1);assert(s.uniqueLinesByPosition.every(n=>n>1));}
});
test('lexical enumeration order does not choose a different poem',async()=>{
 const altered=structuredClone(data);altered.lexicon.reverse();const r=await engine.generate(altered,options);assert.deepEqual(clean(first),clean(r));
});
test('line locks, invalid parameters and mixed-form histories are rejected',async()=>{
 for(const change of [{fixedLines:['日暮春風起']},{firstLine:'固定首句'},{seed:-1},{seed:2**32},{seed:'42'},{seed:NaN},{variation:Infinity},{variation:-1},{length:6},{length:'7'},{beamWidth:0},{recentPoems:[new Array(8)]},{recentPoems:Array(9).fill(Array(8).fill('甲乙丙丁戊'))},{recentPoems:[Array(8).fill('甲乙丙丁戊己庚')]},{templateId:'L7-P-R0'},{season:'rain'},{time:'unknown'}])await assert.rejects(engine.generate(data,{...options,...change}));
});
test('parallel relation checks roles and semantic classes, not merely unequal words',()=>{
 const make=(kind,role,semantic,text)=>({kind,role,semantic,text});
 const a={frame:'SVO',subject:{entity:'wind'},tokens:[make('noun','subject','weather','清風'),make('verb','predicate','effect','拂'),make('noun','patient','plant','翠竹')]};
 const b={frame:'SVO',subject:{entity:'rain'},tokens:[make('noun','subject','weather','細雨'),make('verb','predicate','effect','潤'),make('noun','patient','plant','紅花')]};
 assert(engine.parallel(a,b));
 for(const key of ['role','kind','semantic']){const c=structuredClone(b);c.tokens[1][key]='incompatible';assert(!engine.parallel(a,c));}
 const c=structuredClone(b);c.subject.entity='wind';assert(!engine.parallel(a,c));
});
test('empty legal vocabulary fails without a stored-poem fallback',async()=>{
 const empty={...data,lexicon:[],baseline:{lines:Array(8).fill('不可使用此句')}};
 const r=await engine.generate(empty,{seed:42});assert.equal(r.status,'infeasible');assert.deepEqual(r.lines,[]);
});
test('cancellation does not consume state needed for replay',async()=>{
 let seen=0;await assert.rejects(engine.generate(data,options,()=>{seen++;throw new Error('cancelled');}));assert.equal(seen,1);
 const r=await engine.generate(data,options);assert.deepEqual(clean(first),clean(r));
});
test('history is copied, excludes recent first lines, and is replayable',async()=>{
 const h=[first.lines.slice()];const promise=engine.generate(data,{...options,seed:3,recentPoems:h});h[0][0]='變更外部歷史';
 const r=await promise;assert.deepEqual(r.options.recentPoems,[first.lines]);
 if(r.status==='complete')assert.notEqual(r.lines[0],first.lines[0]);
 const replay=await engine.generate(data,r.options);assert.deepEqual(clean(r),clean(replay));
});
test('worker runs the same engine and preserves request identity',async()=>{
 const messages=[],scope={setTimeout,performance,console};scope.self=scope;scope.postMessage=m=>messages.push(m);
 const ctx=vm.createContext(scope);scope.importScripts=(...names)=>names.forEach(n=>vm.runInContext(fs.readFileSync(path.join(root,'web',n),'utf8'),ctx));
 vm.runInContext(fs.readFileSync(path.join(root,'web/composition-worker.js'),'utf8'),ctx);
 await scope.onmessage({data:{id:'worker-42',options}});const response=messages.find(m=>m.type==='result');assert(response);assert.equal(response.id,'worker-42');assert.deepEqual(clean(JSON.parse(JSON.stringify(response.result))),clean(first));
});
