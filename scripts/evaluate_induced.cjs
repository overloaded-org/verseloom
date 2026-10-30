'use strict';
const fs=require('node:fs'),path=require('node:path'),root=path.resolve(__dirname,'..');
require(path.join(root,'web/induced-data.js'));require(path.join(root,'web/induced-engine.js'));
const out=process.argv[2];if(!out||fs.existsSync(out))throw Error('Provide a new output directory');fs.mkdirSync(out,{recursive:true});
const seeds=[0,1,42,20261030],records=[],summary=[];const data=VerseLoomInducedData;
(async()=>{for(const length of [5,7])for(const seed of seeds){const pair=[];for(const mode of ['local','relations']){const r=await VerseLoomInduced.generate(data,{length,seed,mode,variation:0.8,beamWidth:16});records.push(r);pair.push(r);fs.appendFileSync(path.join(out,'outputs.jsonl'),JSON.stringify(r)+'\n');console.log(length,seed,mode,r.status,r.elapsedMs.toFixed(0));}
summary.push({length,seed,sameCandidatePool:pair[0].poolFingerprint===pair[1].poolFingerprint,changed:pair[0].lines.join('/')!==pair[1].lines.join('/'),localStatus:pair[0].status,relationStatus:pair[1].status});}
const report={schema:'verseloom.induction-development-pilot.v1',seeds,forms:[5,7],modes:['local','relations'],runs:records.length,comparisons:summary,
byForm:[5,7].map(length=>({length,byMode:['local','relations'].map(mode=>{const rows=records.filter(r=>r.length===length&&r.options.mode===mode),ok=rows.filter(r=>r.status==='complete');return {mode,attempted:rows.length,completed:ok.length,uniquePoems:new Set(ok.map(r=>r.lines.join('/'))).size,uniqueByPosition:Array.from({length:8},(_,i)=>new Set(ok.map(r=>r.lines[i])).size),distinctSegments:new Set(ok.flatMap(r=>r.lineDetails.flatMap(l=>l.tokens.map(t=>t.text)))).size};})})),
selectionPolicy:'All declared seeds, both forms, both modes, including failures; first seed is the display example.',independentLiteraryRatings:0,causalInterpretation:'Intervention tests score dependence, not literary quality or learned real-world causality',meta:data.meta,sourceIdentity:data.sourceIdentity};
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');})().catch(e=>{console.error(e);process.exitCode=1;});
