'use strict';
const assert=require('node:assert/strict'),crypto=require('node:crypto'),vm=require('node:vm');
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
async function verify(){
 const base=new URL(process.env.SITE_URL),expected=process.env.MANIFEST_SHA;
 assert(base.protocol==='https:'&&!base.username&&!base.password&&!base.search&&!base.hash&&base.pathname.endsWith('/'));
 assert(/^[a-f0-9]{64}$/.test(expected));
 async function get(name){const r=await fetch(new URL(name,base),{redirect:'error',signal:AbortSignal.timeout(30000)});assert.equal(r.status,200,name);return Buffer.from(await r.arrayBuffer());}
 const bytes=await get('build-manifest.json');assert.equal(digest(bytes),expected);const manifest=JSON.parse(bytes),bodies={};
 const names=['induced.html','induced-worker.js','induced-engine.js','induced-data.js','continuation-ranker.js','continuation-data.js'];
 for(const name of names){const entry=manifest.files.find(r=>r.path===name);assert(entry,name);const body=await get(name);assert.equal(body.length,entry.bytes);assert.equal(digest(body),entry.sha256);bodies[name]=body.toString('utf8');}
 assert(bodies['induced.html'].includes('value="learned"'));assert(bodies['induced.html'].includes('verseloom-mobile-v1'));
 const context=vm.createContext({setTimeout,performance,console});
 for(const name of ['induced-data.js','continuation-data.js','continuation-ranker.js','induced-engine.js'])vm.runInContext(bodies[name],context);
 const samples=[],strip=({elapsedMs,...r})=>r;
 for(const length of [5,7]){const o={length,seed:42,beamWidth:8,mode:'learned'};const a=await context.VerseLoomInduced.generate(context.VerseLoomInducedData,o),b=await context.VerseLoomInduced.generate(context.VerseLoomInducedData,o);
  assert.equal(a.status,'complete');assert(a.checks.every(c=>c.passed));assert.deepEqual(strip(a),strip(b));assert.equal(a.provenance.authoredContentLexemes,0);
  samples.push({length,lines:a.lines,replay:true,hasLearnedReceipts:a.lineDetails.slice(1).every(l=>l.relations.learned)});
 }
 return {status:'PASS_LIVE_CONTINUATION_RANKER',url:new URL('induced.html',base).href,engine:context.VerseLoomInduced.version,verifiedAssets:names.length,samples,browserUIExercised:false,literaryQualityCertified:false};
}
module.exports=verify;
if(require.main===module)verify().then(r=>console.log(JSON.stringify(r,null,2))).catch(e=>{console.error(e);process.exitCode=1;});
