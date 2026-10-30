'use strict';
const assert=require('node:assert/strict'),crypto=require('node:crypto'),vm=require('node:vm');
const base=process.env.SITE_URL,expected=process.env.MANIFEST_SHA;
assert(base&&expected,'SITE_URL and MANIFEST_SHA are required');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{const manifestBytes=Buffer.from(await(await fetch(new URL('build-manifest.json',base))).arrayBuffer());assert.equal(sha(manifestBytes),expected);const manifest=JSON.parse(manifestBytes);
const bodies={};for(const name of ['induced.html','induced-app.js','induced-worker.js','induced-engine.js','induced-data.js','induced-meta.js']){const record=manifest.files.find(r=>r.path===name);assert(record,name);const response=await fetch(new URL(name,base));assert(response.ok,name);const bytes=Buffer.from(await response.arrayBuffer());assert.equal(sha(bytes),record.sha256);assert.equal(bytes.length,record.bytes);bodies[name]=bytes.toString('utf8');}
const context=vm.createContext({setTimeout,performance,console});vm.runInContext(bodies['induced-data.js'],context);vm.runInContext(bodies['induced-engine.js'],context);
const samples=[];for(const length of [5,7]){const options={length,seed:0,variation:0.8,beamWidth:16};const a=await context.VerseLoomInduced.generate(context.VerseLoomInducedData,options),b=await context.VerseLoomInduced.generate(context.VerseLoomInducedData,options);assert.equal(a.status,'complete');const strip=({elapsedMs,...r})=>r;assert.deepEqual(strip(a),strip(b));assert(a.checks.every(x=>x.passed));samples.push({length,seed:0,lines:a.lines,replay:true});}
const continuationCheck=await require('./check_live_continuation.cjs')();
console.log(JSON.stringify({status:'PASS_LIVE_CORPUS_INDUCTION',url:new URL('induced.html',base).href,manifestSha256:expected,verifiedAssets:6,engine:context.VerseLoomInduced.version,samples,continuationCheck,browserUIExercised:false,literaryQualityCertified:false},null,2));})().catch(e=>{console.error(e);process.exitCode=1;});
