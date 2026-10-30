'use strict';
/** Verify same-build public bytes, then execute those trusted bytes in Node. */
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const vm = require('node:vm');
const digest = b => createHash('sha256').update(b).digest('hex');
const required = ['composition.html','composition.css','composition-data.js',
 'composition-engine.js','composition-app.js','composition-worker.js',
 'composition-offline.html','licenses/tshet-uinh-CC0.txt','index.html','release.json'];
async function verify(url, expected, commit, fetchImpl=globalThis.fetch) {
 const base=new URL(url);
 assert(base.protocol==='https:' && !base.username && !base.password && !base.search && !base.hash && base.pathname.endsWith('/'));
 assert(/^[a-f0-9]{64}$/.test(expected||''));
 assert(/^[a-f0-9]{40}$/.test(commit||''));
 async function read(name, sha, size) {
  for(let attempt=0;attempt<5;attempt++) {
   try {
    const r=await fetchImpl(new URL(name,base),{redirect:'error',signal:AbortSignal.timeout(15000),headers:{'Cache-Control':'no-cache'}});
    assert.equal(r.status,200,name);
    const bytes=Buffer.from(await r.arrayBuffer());assert(bytes.length<=10*1024*1024);
    if(size!==undefined)assert.equal(bytes.length,size,name);
    assert.equal(digest(bytes),sha,name);
    const mime=r.headers.get('content-type')||'';
    if(name.endsWith('.js'))assert(/javascript|ecmascript/i.test(mime),name);
    if(name.endsWith('.html'))assert(/text\/html/i.test(mime),name);
    return bytes;
   }catch(e){if(attempt===4)throw e;await new Promise(r=>setTimeout(r,2000));}
  }
 }
 const manifest=JSON.parse((await read('build-manifest.json',expected)).toString('utf8'));
 assert.equal(manifest.schema,'verseloom.browser-demo.build.v1');
 const entries=new Map(manifest.files.map(e=>[e.path,e])),assets=new Map();
 for(const name of required){const entry=entries.get(name);assert(entry,'Missing '+name);assets.set(name,await read(name,entry.sha256,entry.bytes));}
 const release=JSON.parse(assets.get('release.json').toString());
 assert.equal(release.commit,commit);assert.equal(release.entry,'composition.html');
 assert.equal(release.engineVersion,'verseloom.phrase-composition.v2');
 assert(assets.get('index.html').toString().includes('id="composition-v2-entry"'));
 const html=assets.get('composition.html').toString();
 for(const id of ['length','season','time','seed','auto','generate','replay','poem'])assert(html.includes('id="'+id+'"'));
 assert(!/<script\b[^>]*src=/i.test(assets.get('composition-offline.html').toString()));
 const ctx=vm.createContext({setTimeout,performance,console});
 for(const name of ['composition-data.js','composition-engine.js'])vm.runInContext(assets.get(name).toString('utf8'),ctx,{filename:name,timeout:3000});
 assert.equal(ctx.VerseLoomComposition.version,release.engineVersion);
 assert.equal(ctx.VerseLoomCompositionData.meta.storedOutputLines,0);
 const samples=[];
 for(const length of [5,7]){
  const options={length,seed:42,season:'spring',time:'dusk',templateId:length===5?'L5-Z-R0':'L7-Z-R1',rhymeGroup:length===5?'pingshui:lower_level:09':'pingshui:lower_level:06'};
  const r=await ctx.VerseLoomComposition.generate(ctx.VerseLoomCompositionData,options);
  assert.equal(r.status,'complete');assert.equal(r.lines.length,8);assert(r.lines.every(l=>[...l].length===length));assert(r.checks.every(c=>c.passed));
  assert.equal(r.provenance.preselectedLines,false);assert.equal(r.provenance.storedOutputLines,false);
  const repeated=await ctx.VerseLoomComposition.generate(ctx.VerseLoomCompositionData,r.options);
  const identity=({elapsedMs,...x})=>JSON.stringify(x);assert.equal(identity(r),identity(repeated));
  samples.push({length,seed:42,lines:Array.from(r.lines),seedReplay:true});
 }
 return {status:'PASS_LIVE_COMPOSITION_V2',url:new URL('composition.html',base).href,commit,manifestSha256:expected,
  engineVersion:release.engineVersion,verifiedAssets:required.length,samples,browserUIExercised:false};
}
module.exports={verify,digest};
if(require.main===module)verify(process.env.SITE_URL,process.env.MANIFEST_SHA,process.env.GITHUB_SHA)
 .then(r=>console.log(JSON.stringify(r,null,2))).catch(e=>{console.error(e.stack||String(e));process.exitCode=1;});
