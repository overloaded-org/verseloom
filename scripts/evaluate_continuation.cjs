'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
require('../web/induced-data.js');require('../web/continuation-data.js');require('../web/continuation-ranker.js');require('../web/induced-engine.js');
const out=process.argv[2],length=Number(process.argv[3]);assert(out&&[5,7].includes(length));const resume=process.argv.includes('--resume');assert(!fs.existsSync(out)||resume);fs.mkdirSync(out,{recursive:true});
(async()=>{const file=path.join(out,'outputs.jsonl');const existing=resume&&fs.existsSync(file)?fs.readFileSync(file,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
 assert(existing.every(r=>r.engineVersion===VerseLoomInduced.version&&JSON.stringify(r.provenance.sourceIdentity)===JSON.stringify(VerseLoomInducedData.sourceIdentity)&&r.options.beamWidth===16&&r.options.variation===.8&&r.options.recentPoems.length===0&&r.length===length&&[0,1,2,3,42,43,44,45].includes(r.options.seed)&&['relations','learned'].includes(r.options.mode)));
 assert.equal(new Set(existing.map(r=>r.options.seed+'/'+r.options.mode)).size,existing.length);
 const records=[],comparisons=[],identity=({elapsedMs,...x})=>x;
 for(const seed of [0,1,2,3,42,43,44,45]){let pair=[];
  for(const mode of ['relations','learned']){const o={length,seed,variation:.8,beamWidth:16,mode};
   const saved=existing.find(r=>r.options.seed===seed&&r.options.mode===mode);
   const r=saved||await VerseLoomInduced.generate(VerseLoomInducedData,o);records.push(r);pair.push(r);
   if(!saved)fs.appendFileSync(path.join(out,'outputs.jsonl'),JSON.stringify(r)+'\n');
   if(r.status==='complete')assert(r.checks.every(x=>x.passed));
  }
  assert.equal(pair[0].poolFingerprint,pair[1].poolFingerprint);
  comparisons.push({seed,length,poolIdentical:true,statuses:pair.map(r=>r.status),changed:JSON.stringify(pair[0].lines)!==JSON.stringify(pair[1].lines)});
 }
 const last=records.at(-1),replay=await VerseLoomInduced.generate(VerseLoomInducedData,last.options);assert.deepEqual(identity(last),identity(replay));
 const changedRows=comparisons.map((c,i)=>{const a=records[2*i],b=records[2*i+1];const indices=Array.from({length:8},(_,j)=>j).filter(j=>a.lines[j]!==b.lines[j]);return {seed:c.seed,length,changedLinePositions:indices,changedLineCount:indices.length,changedMiddleCoupletLines:indices.filter(j=>j>=2&&j<=5).length};});
 const report={schema:'verseloom.continuation-generation.v1',length,runs:records.length,completed:records.filter(x=>x.status==='complete').length,
 comparisons,changedRows,changedLinePositionsHistogram:Object.fromEntries(Array.from({length:8},(_,i)=>[String(i),changedRows.filter(x=>x.changedLinePositions.includes(i)).length])),replay:true,resumedRecords:existing.length,independentLiteraryRatings:0,
 uniqueByMode:Object.fromEntries(['relations','learned'].map(mode=>[mode,new Set(records.filter(x=>x.options.mode===mode&&x.status==='complete').map(r=>r.lines.join('/'))).size]))};
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exitCode=1;});
