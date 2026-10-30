'use strict';
// Published bytes are matched to THIS build before executing downloaded code.
const assert=require('node:assert/strict'), {createHash}=require('node:crypto'), vm=require('node:vm');
const hash=b=>createHash('sha256').update(b).digest('hex');
async function main(){
  const base=new URL(process.env.SITE_URL), expected=process.env.MANIFEST_SHA;
  assert.equal(base.protocol,'https:');assert(/^[a-f0-9]{64}$/.test(expected||''));
  async function get(name){
    for(let n=0;n<5;n++){
      const r=await fetch(new URL(name,base),{redirect:'error',signal:AbortSignal.timeout(15000),headers:{'Cache-Control':'no-cache'}});
      if(r.status===200)return Buffer.from(await r.arrayBuffer());
      if(n===4)throw new Error(`HTTP ${r.status}: ${name}`);
      await new Promise(resolve=>setTimeout(resolve,2000));
    }
  }
  const bytes=await get('build-manifest.json');assert.equal(hash(bytes),expected);
  const manifest=JSON.parse(bytes), files=new Map(manifest.files.map(e=>[e.path,e]));
  const downloaded={};
  for(const name of ['index.html','data7.js','engine.js','app.js','worker.js','seven.css','standalone.html']){
    const entry=files.get(name);assert(entry,`Missing ${name}`);
    const b=await get(name);assert.equal(b.length,entry.bytes);assert.equal(hash(b),entry.sha256);downloaded[name]=b.toString('utf8');
  }
  assert(downloaded['index.html'].includes('id="form-select"'));
  assert(downloaded['index.html'].includes('value="7"'));
  assert(downloaded['standalone.html'].includes('VerseLoomSevenData='));
  assert(!/<script\b[^>]*src=/.test(downloaded['standalone.html']));
  const scope=vm.createContext({setTimeout,performance,atob});
  vm.runInContext(downloaded['data7.js'],scope,{timeout:3000});
  vm.runInContext(downloaded['engine.js'],scope,{timeout:3000});
  const data=scope.VerseLoomSevenData, engine=scope.VerseLoomEngine;
  const generated=[];
  for(const seed of [0,1,42]){
    const r=await engine.generate(data,{seed,variation:.8,templateId:'L7-P-R0'});
    assert.equal(r.status,'complete');assert.equal(r.lineLength,7);assert.equal(r.lines.length,8);
    assert(r.lines.every(x=>Array.from(x).length===7));assert(r.checks.every(x=>x.passed));
    assert.equal(r.provenance.preselectedLines,false);assert.equal(r.selection.stage,'after-eight-lines');
    generated.push(r);
  }
  assert(new Set(generated.map(r=>r.lines.join('/'))).size>1);
  const replay=await engine.generate(data,generated[2].options);
  const identity=({elapsedMs,...r})=>JSON.stringify(r);assert.equal(identity(replay),identity(generated[2]));
  console.log(JSON.stringify({status:'PASS_LIVE_SEVEN',url:base.href,manifestSha256:expected,lineLength:7,
    candidates:data.meta.candidateCount,rhymes:data.meta.feasibleRhymes,seedReplay:true,preselectedLines:false,
    samples:generated.map(r=>({seed:r.options.seed,lines:r.lines})),browserUIExercised:false},null,2));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
