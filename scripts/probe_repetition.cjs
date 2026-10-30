'use strict';
const fs=require('node:fs'),path=require('node:path');
const dir=process.argv[2],out=process.argv[3],n=Number(process.argv[4]||32);
if(!dir||!out||!Number.isInteger(n)||n<1||n>1024)throw Error('Usage: node scripts/probe_repetition.cjs SITE OUTPUT [1..1024]');
if(fs.existsSync(out))throw Error('Refusing overwrite');
require(path.resolve(dir,'data.js'));require(path.resolve(dir,'data7.js'));require(path.resolve(dir,'engine.js'));
(async()=>{const records=[],summary=[];
for(const length of [5,7]){
 const d=length===5?globalThis.VerseLoomData:globalThis.VerseLoomSevenData;const rows=[];
 for(let seed=0;seed<n;seed++){
 const r=await globalThis.VerseLoomEngine.generate(d,{seed,variation:0.8,beamWidth:32});
 rows.push({length,seed,status:r.status,lines:r.lines,rhyme:r.rhymeGroup,checks:r.checks,score:r.score,options:r.options});
 }
 const complete=rows.filter(r=>r.status==='complete');
 const counts=new Map();for(const r of complete)for(const g of new Set(r.lines.flatMap(s=>{const a=Array.from(s),o=[];for(let i=0;i<a.length-1;i++)o.push(a.slice(i,i+2).join(''));return o;}))) counts.set(g,(counts.get(g)||0)+1);
 summary.push({length,attempts:n,complete:complete.length,rhymes:new Set(complete.map(r=>r.rhyme)).size,uniquePoems:new Set(complete.map(r=>r.lines.join('/'))).size,phrases:Object.fromEntries(['美少年','虎眠'].map(s=>[s,complete.filter(r=>r.lines.some(l=>l.includes(s))).length])),topBigrams:[...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,10)});
 records.push(...rows);console.log(summary.at(-1));
}
fs.writeFileSync(out,JSON.stringify({protocol:'seeds 0..N-1; both lengths; auto rhyme/template; width32; variation0.8; no history; target phrases diagnostic only',summary,records},null,2));})().catch(e=>{console.error(e);process.exitCode=1;});
