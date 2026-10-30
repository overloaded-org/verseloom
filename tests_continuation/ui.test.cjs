'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('web/induced.html','utf8'),app=fs.readFileSync('web/induced-app.js','utf8');
function setup(){
 const elements=new Map(),jobs=[];const make=()=>({value:'',checked:false,hidden:false,disabled:false,textContent:'',setAttribute(){},replaceChildren(){},appendChild(){}});
 for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(x=>x[1]))elements.set(id,make());
 elements.get('i-length').value='5';elements.get('i-seed').value='42';elements.get('i-mode').value='learned';
 const sandbox={document:{getElementById:id=>{assert(elements.has(id),id);return elements.get(id);},baseURI:'https://example.test/verseloom/',createElement:make},
 URL,Uint32Array,Worker:class{postMessage(m){jobs.push(m);}terminate(){}},addEventListener(){},
 VerseLoomInducedMeta:{meta:{trainingPoems:13810,admittedSegments:10000},assetBytes:18136797}};
 vm.runInNewContext(app,sandbox);return {elements,jobs};
}
test('mobile research control submits learned mode without a content prompt',()=>{
 assert(html.includes('value="learned"'));assert(html.includes('width=device-width,initial-scale=1'));
 const {elements,jobs}=setup();elements.get('i-form').onsubmit({preventDefault(){}});
 assert.equal(jobs.length,1);assert.equal(jobs[0].options.mode,'learned');assert.equal(jobs[0].options.seed,42);
 assert.deepEqual(Object.keys(jobs[0].options).sort(),['length','mode','seed','variation','beamWidth','recentPoems'].sort());
});
test('changing scoring does not silently generate or relabel an existing result',()=>{
 const {elements,jobs}=setup();elements.get('i-mode').onchange();assert.equal(jobs.length,0);assert.match(elements.get('i-status').textContent,/点击生成/);
});
