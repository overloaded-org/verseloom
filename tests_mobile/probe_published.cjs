'use strict';
// Observational repetition probe + paired relation intervention, no content forcing.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const site=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);
if(fs.existsSync(out))throw Error('Refusing to overwrite evidence');fs.mkdirSync(out,{recursive:true});
for(const name of ['data.js','data7.js','engine.js','induced-data.js','induced-engine.js'])require(path.join(site,name));
(async()=>{
 const repetition=[];
 for(const length of [5,7]){
  const data=length===5?VerseLoomData:VerseLoomSevenData,rows=[],counts={'美少年':0,'虎眠':0},rhymes=new Set();
  for(let seed=0;seed<64;seed++){
   const r=await VerseLoomEngine.generate(data,{seed,variation:0.8,beamWidth:32});
   rows.push({seed,status:r.status,lines:r.lines,rhyme:r.rhymeGroup});
   if(r.status!=='complete')continue;
   assert(r.checks.every(c=>c.passed));rhymes.add(r.rhymeGroup);
   for(const phrase of Object.keys(counts))if(r.lines.some(l=>l.includes(phrase)))counts[phrase]++;
  }
  repetition.push({length,runs:64,complete:rows.filter(r=>r.status==='complete').length,rhymes:rhymes.size,phraseDocumentCounts:counts,rows});
  console.error('repetition',length,counts);
 }
 const records=[],pairs=[];
 for(const length of [5,7])for(let seed=0;seed<8;seed++){
  const opts={length,seed,variation:0.8,beamWidth:16,recentPoems:[]};
  const a=await VerseLoomInduced.generate(VerseLoomInducedData,{...opts,mode:'local'}),b=await VerseLoomInduced.generate(VerseLoomInducedData,{...opts,mode:'relations'});
  for(const r of [a,b]){if(r.status==='complete')assert(r.checks.every(c=>c.passed));assert.equal(r.provenance.authoredContentLexemes,0);assert.equal(r.provenance.authoredSentenceFrames,0);}
  assert.equal(a.poolFingerprint,b.poolFingerprint,'Intervention changed candidate construction');
  records.push(a,b);pairs.push({length,seed,local:a.status,relations:b.status,sameCandidatePool:true,changed:JSON.stringify(a.lines)!==JSON.stringify(b.lines)});
 }
 fs.writeFileSync(path.join(out,'repetition.json'),JSON.stringify(repetition,null,2)+'\n');
 fs.writeFileSync(path.join(out,'induced-outputs.jsonl'),records.map(x=>JSON.stringify(x)).join('\n')+'\n');
 const report={schema:'verseloom.published-probe.v1',sourceCommit:JSON.parse(fs.readFileSync(path.join(site,'release.json'))).commit,
  baselineEngine:VerseLoomEngine.version,inducedEngine:VerseLoomInduced.version,
  repetition:repetition.map(({rows,...r})=>r),induction:{seeds:'0..7',runs:32,completed:records.filter(r=>r.status==='complete').length,pairs,changedPairs:pairs.filter(p=>p.changed).length},
  independentLiteraryRatings:0,scope:'Development diagnostic; no literary improvement or unlimited diversity guarantee'};
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
