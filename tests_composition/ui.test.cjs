'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'..');
function app(real=false){
 class E{constructor(){this.children=[];this.listeners={};this.value='';this.disabled=false;this.hidden=false;this.checked=false;this.textContent='';this.attributes={};}addEventListener(n,f){this.listeners[n]=f;}setAttribute(n,v){this.attributes[n]=v;}appendChild(c){this.children.push(c);return c;}replaceChildren(...c){this.children=c;}remove(){}click(){this.listeners.click?.({});}}
 const html=fs.readFileSync(path.join(root,'web/composition.html'),'utf8'),elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new E()]));
 const document={readyState:'complete',baseURI:'https://example.test/verseloom/composition.html',getElementById:id=>{assert(elements[id],'Missing '+id);return elements[id];},createElement:()=>new E(),body:new E()};
 for(const[k,v]of Object.entries({length:'5',season:'spring',time:'dusk',seed:'42',variation:'0.8',beam:'32'}))elements[k].value=v;elements.auto.checked=true;
 let random=0,pending=null;const requests=[];const context=vm.createContext({document,location:{protocol:'file:'},navigator:{},performance,console,Uint32Array,setTimeout,clearTimeout,addEventListener(){},crypto:{getRandomValues:a=>{a[0]=++random*101;return a;}}});context.window=context;
 vm.runInContext(fs.readFileSync(path.join(root,'web/composition-data.js'),'utf8'),context);
 if(real)vm.runInContext(fs.readFileSync(path.join(root,'web/composition-engine.js'),'utf8'),context);
 const engine=context.VerseLoomComposition;
 context.VerseLoomComposition={generate:(_data,o,p)=>{requests.push(JSON.parse(JSON.stringify(o)));if(real){pending=engine.generate(_data,o,p);return pending;}
  return Promise.resolve({status:'complete',length:o.length,lines:Array.from({length:8},(_,i)=>(o.length===5?'甲乙丙丁戊':'甲乙丙丁戊己庚')),options:JSON.parse(JSON.stringify(o)),contextLabel:'春 · 薄暮',rhymeLabel:'测试',templateId:'test',elapsedMs:1,lineDetails:[],checks:[]});}};
 vm.runInContext(fs.readFileSync(path.join(root,'web/composition-app.js'),'utf8'),context);
 async function act(id,event){assert.equal(elements[id].disabled,false);elements[id].listeners[event]({preventDefault(){}});await new Promise(r=>setImmediate(r));if(pending)await pending;await new Promise(r=>setImmediate(r));}
 return{elements,requests,act,random:()=>random,html};
}
test('initial page has no stored poem and export/replay are disabled',()=>{const x=app();assert.equal(x.elements.copy.disabled,true);assert.equal(x.elements.export.disabled,true);assert.equal(x.elements.replay.disabled,true);assert(!x.html.includes('日暮春風起'));});
test('auto generation changes visible seed and explicit replay keeps input snapshot',async()=>{const x=app();await x.act('form','submit');await x.act('form','submit');assert.deepEqual(x.requests.map(r=>r.seed),[101,202]);assert.equal(x.elements.seed.value,'202');await x.act('replay','click');assert.deepEqual(x.requests[2],x.requests[1]);assert.equal(x.random(),2);});
test('switching poetic form clears output and keeps separate histories',async()=>{const x=app();await x.act('form','submit');x.elements.length.value='7';await x.act('length','change');assert.equal(x.elements.poem.children.length,0);assert(x.elements.replay.disabled);await x.act('form','submit');assert.equal(x.requests[1].length,7);assert.deepEqual(x.requests[1].recentPoems,[]);x.elements.length.value='5';await x.act('length','change');await x.act('form','submit');assert.equal(x.requests[2].recentPoems.length,1);});
test('manual seed and deterministic mode do not silently use session history',async()=>{const x=app();await x.act('form','submit');x.elements.auto.checked=false;await x.act('form','submit');assert.deepEqual(x.requests[1].recentPoems,[]);assert.equal(x.random(),1);});
test('invalid visible seed starts no work',async()=>{for(const value of ['', '-1','1.5','4294967296','1e3']){const x=app();x.elements.seed.value=value;await x.act('form','submit');assert.equal(x.requests.length,0);assert.equal(x.elements.error.hidden,false);}});
test('real UI event invokes both composers and renders correct line lengths',async()=>{const x=app(true);x.elements.auto.checked=false;for(const n of [5,7]){x.elements.length.value=String(n);await x.act('length','change');await x.act('form','submit');assert.equal(x.elements.poem.children.length,8);assert(x.elements.poem.children.every(e=>Array.from(e.textContent).length===n+1));}const before=x.elements.poem.children.map(e=>e.textContent);await x.act('replay','click');assert.deepEqual(x.elements.poem.children.map(e=>e.textContent),before);});
