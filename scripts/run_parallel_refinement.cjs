'use strict';
// Launch both forms concurrently and await both; no background delivery.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createHash}=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),out=process.argv[2];
assert(out,'Usage: node scripts/run_parallel_refinement.cjs NEW_OUTPUT_DIRECTORY');
const target=path.resolve(out);assert(!fs.existsSync(target),'Refusing to overwrite evidence');
fs.mkdirSync(target,{recursive:true});fs.writeFileSync(path.join(target,'INCOMPLETE'),'Research run not complete.\n');
const protocolPath=path.join(ROOT,'research/parallel-refinement-v1.json');
fs.copyFileSync(protocolPath,path.join(target,'protocol.json'));
const digest=raw=>createHash('sha256').update(raw).digest('hex');
const paths=['web/induced-data.js','web/continuation-data.js','web/induced-engine.js','web/continuation-ranker.js',
 'scripts/refinement_core.cjs','scripts/refinement_problem.cjs','scripts/evaluate_parallel_refinement.cjs','research/parallel-refinement-v1.json'];
fs.writeFileSync(path.join(target,'inputs.json'),JSON.stringify({schema:'verseloom.parallel-refinement.inputs.v1',
 files:Object.fromEntries(paths.map(p=>[p,digest(fs.readFileSync(path.join(ROOT,p)))]))},null,2)+'\n');
function run(length){return new Promise((resolve,reject)=>{
 const child=spawn(process.execPath,[path.join(__dirname,'evaluate_parallel_refinement.cjs'),path.join(target,String(length)),String(length)],{stdio:'inherit'});
 child.once('error',reject);child.once('exit',(code,signal)=>code===0?resolve():reject(Error(`Form ${length} failed: ${code}/${signal}`)));
});}
(async()=>{
 const done=await Promise.allSettled([run(5),run(7)]);
 if(done.some(x=>x.status==='rejected'))throw Error(done.filter(x=>x.status==='rejected').map(x=>x.reason).join('\n'));
 const reports=[5,7].map(n=>JSON.parse(fs.readFileSync(path.join(target,String(n),'report.json'))));
 fs.writeFileSync(path.join(target,'report.json'),JSON.stringify({schema:'verseloom.parallel-refinement.report.v1',
 status:'COMPUTED_NOT_LITERARILY_CERTIFIED',reports,independentLiteraryRatings:0,
 deployed:false,diffusionModel:false,budgetInterpretation:'Identical proposal ceilings and candidate sets, NOT identical elapsed time or actual evaluation counts.'},null,2)+'\n');
 fs.unlinkSync(path.join(target,'INCOMPLETE'));console.log('BOTH FORMS COMPLETED',target);
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
