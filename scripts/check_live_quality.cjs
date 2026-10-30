'use strict';
// Published bytes are bound to this build's trusted manifest hash before evaluation.
const assert=require('node:assert/strict'),crypto=require('node:crypto'),vm=require('node:vm');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 const base=new URL(process.env.SITE_URL);assert.equal(base.protocol,'https:');assert(!base.username&&!base.password);
 const manifest=Buffer.from(await (await fetch(new URL('build-manifest.json',base),{redirect:'error'})).arrayBuffer());assert.equal(hash(manifest),process.env.MANIFEST_SHA);
 const entries=JSON.parse(manifest).files,assets={};
 for(const name of ['quality.html','quality-data.js','quality-engine.js','quality-app.js','quality-worker.js','quality-offline.html','quality-NOTICE.html']){
  const entry=entries.find(e=>e.path===name);assert(entry,'Missing quality asset '+name);
  const response=await fetch(new URL(name,base),{redirect:'error'});assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer());assert.equal(bytes.length,entry.bytes);assert.equal(hash(bytes),entry.sha256);assets[name]=bytes.toString('utf8');
 }
 assert(assets['quality.html'].includes('id="q-length"'));assert(!assets['quality.html'].includes('日暮春風起'));assert(!/<script[^>]+src=/.test(assets['quality-offline.html']));
 const context=vm.createContext({setTimeout,Date,console});vm.runInContext(assets['quality-data.js'],context);vm.runInContext(assets['quality-engine.js'],context);
 const d=context.VerseLoomQualityData,e=context.VerseLoomQuality,results=[];
 for(const n of [5,7]){
  const r=await e.generate(d,{lineLength:n,seed:42,scene:'night',templateId:n===5?'L5-P-R0':'L7-Z-R0',rhymeGroup:'pingshui:lower_level:01'});
  assert.equal(r.status,'complete');assert(e.verify(d,r).passed);assert.equal(r.provenance.preselectedLines,false);assert.equal(r.provenance.storedSentencePool,false);
  const repeat=await e.generate(d,r.options);assert.equal(JSON.stringify(r.lines),JSON.stringify(repeat.lines));
  results.push({lineLength:n,lines:Array.from(r.lines),matchedMiddlePairs:r.pairChecks.filter(p=>p.matched).length,replay:true});
 }
 console.log(JSON.stringify({status:'PASS_LIVE_QUALITY_WORDS',engine:e.version,assetCount:7,manifestSha256:process.env.MANIFEST_SHA,results,independentLiteraryCertification:false,browserUIExercised:false},null,2));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1});
