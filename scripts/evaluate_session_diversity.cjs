'use strict';
// Fixed development probes; no content keywords enter engine options.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),dir=path.resolve(process.argv[2]),n=Number(process.argv[3]||8);
if(!Number.isInteger(n)||n<1||n>1024)throw Error('Invalid sample size');
if(fs.existsSync(dir))throw Error('Refusing overwrite');fs.mkdirSync(dir,{recursive:true});
require(path.join(root,'web/data.js'));require(path.join(root,'web/data7.js'));require(path.join(root,'web/engine.js'));
require(path.join(root,'web/induced-data.js'));require(path.join(root,'web/induced-engine.js'));
function spans(lines){const s=new Set();for(const line of lines){const a=[...line];for(const k of [2,3])for(let i=0;i+k<=a.length;i++)s.add(a.slice(i,i+k).join(''));}return s;}
function overlap(lines,history){const a=spans(lines),b=spans(history.flat());return a.size?[...a].filter(x=>b.has(x)).length/a.size:0;}
const identity=({elapsedMs,...r})=>r;
(async()=>{const rows=[],summary=[];for(const engine of ['baseline','induced'])for(const length of [5,7]){
 if(process.argv[4]&&process.argv[4]!==engine+'-'+length)continue;
 const generate=engine==='baseline'?opts=>VerseLoomEngine.generate(length===5?VerseLoomData:VerseLoomSevenData,opts):opts=>VerseLoomInduced.generate(VerseLoomInducedData,opts);
 const histories={off:[],on:[]},group=[];
 for(let i=0;i<n;i++){
  const seed=1000+i,base={seed,variation:0.8,beamWidth:16,...(engine==='induced'?{length,mode:'relations'}:{})};
  const no=await generate({...base,recentPoems:[]}),yes=await generate({...base,recentPoems:histories.on});
  for(const [name,r] of [['off',no],['on',yes]]){
   const row={engine,length,seed,recency:name,status:r.status,overlap:r.status==='complete'?overlap(r.lines,histories[name]):null,lines:r.lines,checks:r.checks,options:r.options,poolFingerprint:r.poolFingerprint,historyPenalties:r.lineDetails?.map(x=>x.historyPenalty),sourceIdentity:r.provenance?.sourceIdentity};
   group.push(row);rows.push(row);
   if(r.status==='complete'){assert(r.checks.every(c=>c.passed));histories[name]=histories[name].concat([r.lines]).slice(-8);}
  }
  if(engine==='induced')assert.equal(no.poolFingerprint,yes.poolFingerprint,'History must not change vocabulary/line pools');
  if(i===n-1)assert.deepEqual(identity(yes),identity(await generate(yes.options)),'Replay identity');
 }
 const s={engine,length,seeds:[1000,1000+n-1],attemptsPerArm:n};
 for(const name of ['off','on']){const rs=group.filter(x=>x.recency===name),complete=rs.filter(x=>x.status==='complete');s[name]={complete:complete.length,uniquePoems:new Set(complete.map(x=>x.lines.join('/'))).size,meanRecentSpanOverlap:complete.reduce((x,r)=>x+r.overlap,0)/Math.max(1,complete.length),maximumLineFrequency:Math.max(0,...Object.values(complete.flatMap(r=>r.lines).reduce((a,l)=>(a[l]=(a[l]||0)+1,a),{})))};}
 summary.push(s);fs.writeFileSync(path.join(dir,'outputs.jsonl'),rows.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(dir,'report.json'),JSON.stringify({protocol:'Fixed development seeds; both forms; same data/model; no authored content; failures counted; overlap is an optimized diagnostic, NOT literature quality',independentLiteraryRatings:0,summary},null,2));console.log(JSON.stringify(s));
} })().catch(e=>{console.error(e);process.exitCode=1;});
