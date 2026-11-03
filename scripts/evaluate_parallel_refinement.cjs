'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..');
require('../web/induced-data.js');require('../web/continuation-data.js');require('../web/continuation-ranker.js');
const engine=require('../web/induced-engine.js');
const {makeProblem,digest}=require('./refinement_problem.cjs'),{optimize}=require('./refinement_core.cjs');
const protocol=JSON.parse(fs.readFileSync(path.join(ROOT,'research/parallel-refinement-v1.json')));
const out=path.resolve(process.argv[2]),length=Number(process.argv[3]);assert(protocol.lengths.includes(length));
assert(!fs.existsSync(out),'Refusing overwrite');fs.mkdirSync(out,{recursive:true});
const write=(name,value)=>fs.appendFileSync(path.join(out,name),JSON.stringify(value)+'\n');
const singles=Array.from({length:8},(_,i)=>[i]),coupled=[[0],[1],[2,3],[4,5],[6],[7]];
const policies={single_pass:{blocks:singles,maxSweeps:1},joint_pass:{blocks:coupled,maxSweeps:1},
 single_iterative:{blocks:singles,maxSweeps:protocol.maxSweeps},joint_iterative:{blocks:coupled,maxSweeps:protocol.maxSweeps}};
(async()=>{
 const summaries=[],locals=[],failures=[];
 for(const seed of protocol.seeds){
  const options={...protocol.baselineOptions,length,seed};
  const baseline=await engine.generate(VerseLoomInducedData,options);write('baseline.jsonl',baseline);
  if(baseline.status!=='complete'){failures.push({seed,length,status:baseline.status});continue;}
  const lattice=await engine.researchLattice(VerseLoomInducedData,options);
  const problem=makeProblem(VerseLoomInducedData,VerseLoomContinuationData,VerseLoomContinuation,lattice,baseline,protocol.candidateLimitPerPosition);
  write('candidate-pools.jsonl',{seed,length,...problem.snapshot(),sourceIdentity:VerseLoomInducedData.sourceIdentity});
  const initialFingerprint=digest(problem.initial.map(l=>({text:l.text,tokens:l.tokens})));
  for(const [arm,policy] of Object.entries(policies)){
   const result=optimize({initial:problem.initial,pools:problem.pools,evaluate:problem.evaluate,
    blocks:policy.blocks,budget:protocol.proposalBudget,maxSweeps:policy.maxSweeps});
   const serialized=problem.serialize(result.state,result.value);
   const changed=serialized.lines.map((l,i)=>l===baseline.lines[i]?null:i).filter(i=>i!==null);
   const row={arm,seed,length,status:serialized.status,initialScore:result.initialScore,score:result.score,gain:result.gain,
    proposals:result.proposals,legalProposals:result.legalProposals,budget:result.budget,
    acceptedMoves:result.acceptedMoves,completedSweeps:result.completedSweeps,reason:result.reason,changedPositions:changed,
    attemptedRows:result.attemptedRows,templateFixed:serialized.templateId===baseline.templateId,rhymeFixed:serialized.rhymeGroup===baseline.rhymeGroup,
    candidateDigest:problem.candidateDigest,initialFingerprint};
   summaries.push(row);write('outputs.jsonl',{...serialized,experiment:row});
   write('traces.jsonl',{arm,seed,length,candidateDigest:problem.candidateDigest,trace:result.trace});
   assert.equal(initialFingerprint,digest(problem.initial.map(l=>({text:l.text,tokens:l.tokens}))));
   if(seed===protocol.seeds[0]){
    const replay=optimize({initial:problem.initial,pools:problem.pools,evaluate:problem.evaluate,
     blocks:policy.blocks,budget:protocol.proposalBudget,maxSweeps:policy.maxSweeps});
    assert.deepEqual(result,replay,'Revision/replay diverged');
   }
  }
  // Frozen context is a diagnostic control, never a user-supplied output prompt.
  for(const block of protocol.localInterventions)for(const policy of ['sequential','joint']){
   const run=optimize({initial:problem.initial,pools:problem.pools,evaluate:problem.evaluate,
    blocks:policy==='joint'?[block]:block.map(i=>[i]),budget:protocol.proposalBudget,
    maxSweeps:policy==='joint'?1:protocol.maxSweeps});
   const value=problem.serialize(run.state,run.value);
   assert(value.lines.every((l,i)=>block.includes(i)||l===baseline.lines[i]));
   const record={seed,length,rows:block,policy,score:run.score,gain:run.gain,
    proposals:run.proposals,legalProposals:run.legalProposals,acceptedMoves:run.acceptedMoves,
    allPairsExamined:policy==='joint'&&run.trace.every(t=>t.completeBlock),candidateDigest:problem.candidateDigest};
   locals.push(record);write('local-outputs.jsonl',{...value,experiment:record});
  }
  const task=summaries.filter(x=>x.seed===seed);
  console.log(JSON.stringify({length,seed,gains:Object.fromEntries(task.map(x=>[x.arm,+x.gain.toFixed(5)])),
   proposals:Object.fromEntries(task.map(x=>[x.arm,x.proposals]))}));
 }
 const mean=xs=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
 const byArm=Object.fromEntries(Object.keys(policies).map(arm=>{
  const xs=summaries.filter(x=>x.arm===arm);return [arm,{completed:xs.length,changed:xs.filter(x=>x.changedPositions.length).length,
   meanGain:mean(xs.map(x=>x.gain)),meanProposals:mean(xs.map(x=>x.proposals)),
   meanLegalProposals:mean(xs.map(x=>x.legalProposals)),acceptedMoves:xs.reduce((a,b)=>a+b.acceptedMoves,0),
   changedByPosition:Array.from({length:8},(_,i)=>xs.filter(x=>x.changedPositions.includes(i)).length),
   budgetExhausted:xs.filter(x=>x.reason==='budget').length,allPositionsConsidered:xs.every(x=>x.attemptedRows.length===8)}];}));
 const localComparisons=[];
 for(const seed of protocol.seeds)for(const rows of protocol.localInterventions){
  const xs=locals.filter(x=>x.seed===seed&&x.rows.join(',')===rows.join(','));if(xs.length!==2)continue;
  const seq=xs.find(x=>x.policy==='sequential'),joint=xs.find(x=>x.policy==='joint');
  localComparisons.push({seed,rows,extraGain:joint.score-seq.score,jointProposals:joint.proposals,sequentialProposals:seq.proposals,
   allPairsExamined:joint.allPairsExamined,candidatePoolIdentical:joint.candidateDigest===seq.candidateDigest});
 }
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({schema:'verseloom.parallel-refinement.form.v1',length,
  seeds:protocol.seeds,baselineRuns:protocol.seeds.length,baselineFailures:failures,
  optimizedRuns:summaries.length,byArm,localComparisons,localRuns:locals.length,paired:summaries,
  engineVersion:engine.version,sourceIdentity:VerseLoomInducedData.sourceIdentity,
  replayedSeed:protocol.seeds[0],independentLiteraryRatings:0,
  warnings:['Same score is optimized and measured; gains are not independent quality evidence.',
   'Equal proposal ceilings, not equal actual work. Single-coordinate search often stops earlier.',
   'Candidate row shortlists remain finite. Whole-line coordinate descent is not trained diffusion.',
   'Previously examined development seeds; no held-out literary assessment.']},null,2)+'\n');
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
