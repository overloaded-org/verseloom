'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {optimize}=require('../scripts/refinement_core.cjs');
function fixture(){
 const a={text:'a'},b={text:'b'},x={text:'x'},y={text:'y'};
 const pools=[[a,x],[b,y]],initial=[a,b];
 const evaluate=s=>s[0]===a&&s[1]===b?{score:0}:s[0]===x&&s[1]===y?{score:10}:null;
 return {pools,initial,evaluate};
}
test('atomic joint moves cross a barrier that legal single moves cannot',()=>{
 const f=fixture();const single=optimize({...f,blocks:[[0],[1]],maxSweeps:4}),pair=optimize({...f,blocks:[[0,1]],maxSweeps:1});
 assert.equal(single.score,0);assert.equal(pair.score,10);assert.deepEqual(pair.trace[0].changedRows,[0,1]);
 assert.equal(pair.trace[0].proposals,4);assert(pair.trace[0].completeBlock);
});
test('pair selection agrees with exhaustive finite-grid oracle',()=>{
 for(let seed=0;seed<20;seed++){
  const pools=Array.from({length:2},(_,i)=>Array.from({length:5},(_,j)=>({text:`${i}:${j}`,id:j})));
  const evaluate=s=>({score:Math.sin(seed+s[0].id*3+s[1].id*7)});
  const initial=pools.map(xs=>xs[0]),r=optimize({initial,pools,evaluate,blocks:[[0,1]],maxSweeps:1});
  const oracle=Math.max(...pools[0].flatMap(a=>pools[1].map(b=>evaluate([a,b]).score)));
  assert(Math.abs(r.score-oracle)<1e-12);
 }
});
test('hard proposal budget counts rejected moves too',()=>{
 const f=fixture();const r=optimize({...f,blocks:[[0,1]],budget:3,maxSweeps:4});
 assert.equal(r.proposals,3);assert.equal(r.score,0);assert.equal(r.reason,'budget');
 assert.equal(r.trace[0].completeBlock,false);
});
test('iterative sweep revisits the first position rather than locking it',()=>{
 const pools=[[{text:'a',id:0},{text:'x',id:1}],[{text:'b',id:0},{text:'y',id:1}]];
 const initial=pools.map(x=>x[0]);
 const evaluate=s=>({score:[[0,1],[-1,4]][s[0].id][s[1].id]});
 const first=optimize({initial,pools,evaluate,blocks:[[0],[1]],maxSweeps:1});
 const again=optimize({initial,pools,evaluate,blocks:[[0],[1]],maxSweeps:4});
 assert.equal(first.score,1);assert.equal(again.score,4);
 assert(again.trace.some(t=>t.sweep>0&&t.changedRows.includes(0)));
});
test('local diagnostic holds only nonintervened algorithm context fixed',()=>{
 const f=fixture(),z={text:'z'};f.pools.push([z]);f.initial.push(z);
 const r=optimize({...f,blocks:[[0,1]],maxSweeps:2});
 assert.equal(r.state[2],z);assert(r.trace.every(t=>!t.changedRows.includes(2)));
});
test('accepted trace is monotone, replayable and starts at the same state',()=>{
 const f=fixture(),opts={...f,blocks:[[0,1]],maxSweeps:4};
 const a=optimize(opts),b=optimize(opts);assert.deepEqual(a,b);
 assert(a.trace.every(t=>t.afterScore>=t.beforeScore));assert.deepEqual(f.initial.map(x=>x.text),['a','b']);
});
test('score ties cannot create revision cycles',()=>{
 const f=fixture();f.evaluate=()=>({score:1});
 const r=optimize({...f,blocks:[[0,1]],maxSweeps:4});
 assert.equal(r.acceptedMoves,0);assert.equal(r.completedSweeps,1);assert.equal(r.reason,'coordinate-stationary');
});
test('an illegal initial state fails rather than supplying a fallback',()=>{
 assert.throws(()=>optimize({...fixture(),evaluate:()=>null,blocks:[[0,1]]}),/initial state/);
});
test('invalid budgets, schedules and out-of-domain anchors fail closed',()=>{
 for(const budget of [0,-1,1.2,NaN,Infinity,100001])assert.throws(()=>optimize({...fixture(),blocks:[[0]],budget}));
 for(const blocks of [[],[[2]],[[0,0]],[[0,1,2]]])assert.throws(()=>optimize({...fixture(),blocks}));
 const f=fixture();assert.throws(()=>optimize({...f,initial:[{text:'not-in-pool'},f.initial[1]],blocks:[[0]]}));
});
test('non-finite objective values are rejected',()=>{
 const f=fixture();let first=true;f.evaluate=()=>({score:first?(first=false,0):Infinity});
 assert.throws(()=>optimize({...f,blocks:[[0,1]]}),/Non-finite/);
});
