'use strict';
// Search policies only. This module contains no poetic words, topic/POS tables,
// fixed sentences, model training, noise process, or claim of diffusion.
const assert = require('node:assert/strict');
const same = (a,b) => a.length === b.length && a.every((x,i) => x === b[i]);

function optimize({initial, pools, evaluate, blocks, budget=4096, maxSweeps=4, tolerance=1e-12}) {
  assert(Array.isArray(initial) && initial.length > 0);
  assert(Array.isArray(pools) && pools.length === initial.length);
  assert(pools.every((xs,i) => Array.isArray(xs) && xs.length && xs.includes(initial[i])));
  assert.equal(typeof evaluate, 'function');
  assert(Number.isInteger(budget) && budget >= 1 && budget <= 100000);
  assert(Number.isInteger(maxSweeps) && maxSweeps >= 1 && maxSweeps <= 64);
  assert(Number.isFinite(tolerance) && tolerance > 0);
  assert(Array.isArray(blocks) && blocks.length > 0);
  assert(blocks.every(b => Array.isArray(b) && b.length >= 1 && b.length <= 2 &&
    new Set(b).size === b.length && b.every(i => Number.isInteger(i) && i >= 0 && i < initial.length)));
  // Defensive snapshots: callers cannot change a search in flight.
  const domains = pools.map(xs => xs.slice()), schedule = blocks.map(b => b.slice());
  let state = initial.slice(), start = evaluate(state), value = start;
  assert(value && Number.isFinite(value.score), 'The algorithm-produced initial state must be legal');
  const initialScore=value.score, trace=[], visitedRows=new Set();
  let proposals=0, legal=0, accepted=0, completedSweeps=0, reason='sweep-limit';
  for (let sweep=0; sweep<maxSweeps; sweep++) {
    let improved=false, fullSweep=true;
    for (let blockIndex=0; blockIndex<schedule.length; blockIndex++) {
      if (proposals === budget) {fullSweep=false;break;}
      const rows=schedule[blockIndex], before=state.slice(), beforeScore=value.score;
      let best=state.slice(), bestValue=value, visits=0, admissible=0;
      const expected=rows.reduce((n,i)=>n*domains[i].length,1);
      const left=domains[rows[0]], right=rows.length===2?domains[rows[1]]:[null];
      // No upper-line pruning: every visited two-line replacement is atomic.
      outer: for (const a of left) for (const b of right) {
        if (proposals === budget) break outer;
        const candidate=before.slice();candidate[rows[0]]=a;
        if (rows.length===2) candidate[rows[1]]=b;
        proposals++;visits++;rows.forEach(i=>visitedRows.add(i));
        const score=evaluate(candidate);
        if (!score) continue;
        assert(Number.isFinite(score.score), 'Non-finite score');legal++;admissible++;
        if (score.score > bestValue.score+tolerance) {best=candidate;bestValue=score;}
      }
      const changed=bestValue.score > beforeScore+tolerance;
      if (changed) {state=best;value=bestValue;accepted++;improved=true;}
      const changedRows=before.map((x,i)=>x===state[i]?null:i).filter(i=>i!==null);
      assert(changedRows.every(i=>rows.includes(i)), 'Move changed a frozen context position');
      assert(value.score >= beforeScore-tolerance, 'Accepted move lowered the full objective');
      trace.push({sweep,blockIndex,rows:rows.slice(),proposals:visits,legalProposals:admissible,
        expectedProposals:expected,completeBlock:visits===expected,accepted:changed,changedRows,
        before:before.map(x=>x.text),after:state.map(x=>x.text),
        beforeScore,afterScore:value.score,delta:value.score-beforeScore});
      if (visits < expected) {fullSweep=false;break;}
    }
    if (fullSweep) completedSweeps++;
    if (!fullSweep || proposals===budget) {reason='budget';break;}
    if (!improved) {reason='coordinate-stationary';break;}
  }
  return {state,value,initialScore,score:value.score,gain:value.score-initialScore,
    proposals,legalProposals:legal,acceptedMoves:accepted,completedSweeps,
    attemptedRows:[...visitedRows].sort((a,b)=>a-b),reason,trace,
    budget,maxSweeps,blocks:schedule};
}
module.exports={optimize,same};
