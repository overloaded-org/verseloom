// Diagnostic only: expose the frozen engine's private lattice to read, not change, candidates.
const fs=require('node:fs'),assert=require('node:assert/strict'),vm=require('node:vm');
const [engine,model,files,out]=process.argv.slice(2);assert(!fs.existsSync(out));
const s=fs.readFileSync(engine,'utf8');const anchor='global.VerseLoomInduced=Object.freeze({generate,version:VERSION});';assert.equal(s.split(anchor).length,2);
const injected=s.replace(anchor,`global.getEndPools=async function(data,reference){
 const o={...reference.options,mode:'relations'};const {pools}=await linePools(data,prepare(data),options(o),()=>{});
 const fp=hash([...pools].map(([k,rs])=>k+':'+rs.map(l=>l.text).join(',')).join('|'));
 const t=data.templates[String(o.length)].find(t=>t.id===reference.templateId);
 const get=row=>t.variants[row].flatMap(p=>pools.get(p+'/'+(t.rhymeRows.includes(row)?reference.rhymeGroup:'nonrhyming'))||[]).map(l=>({text:l.text,tokens:l.tokens,tones:l.tones,localScore:l.score}));
 return {fp,template:t,rhyme:reference.rhymeGroup,length:o.length,seed:o.seed,upper:get(6),lower:get(7)};};`+anchor);
// The ordinary Node realm is used for native performance; these are verified project sources.
vm.runInThisContext(fs.readFileSync(model,'utf8'));vm.runInThisContext(injected);
(async()=>{const records=[];for(const f of files.split(','))for(const ref of JSON.parse(fs.readFileSync(f))){const r=await globalThis.getEndPools(VerseLoomInducedData,ref);assert.equal(r.fp,ref.poolFingerprint);for(const [j,pool] of [[6,r.upper],[7,r.lower]])assert(pool.some(c=>c.text===ref.lines[j]&&Math.abs(c.localScore-ref.lineDetails[j].localScore)<1e-10&&JSON.stringify(c.tokens)===JSON.stringify(ref.lineDetails[j].tokens.map(t=>t.id))));records.push(r);}fs.writeFileSync(out,JSON.stringify(records));console.log('Matched all frozen pool fingerprints, original endings and token paths:',records.length)})().catch(e=>{console.error(e);process.exitCode=1;});
