'use strict';
// DOM contract, not a browser-rendering test.
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'..');
function application() {
  class Element {
    constructor(){this.children=[];this.value='';this.hidden=false;this.disabled=false;this.checked=false;this.style={};this.attributes={};this.listeners={};this.classList={toggle(){}};}
    addEventListener(n,fn){this.listeners[n]=fn;} setAttribute(k,v){this.attributes[k]=v;}
    appendChild(c){this.children.push(c);return c;} append(...c){this.children.push(...c);} replaceChildren(f){this.children=f.children||[];}
    querySelectorAll(){return this.children;} remove(){} focus(){} select(){}
  }
  const html=fs.readFileSync(path.join(root,'web/index.html'),'utf8');
  const el=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  const document={readyState:'complete',baseURI:'https://example.test/verseloom/',getElementById(id){assert(el[id],id);return el[id];},createElement:()=>new Element(),createDocumentFragment:()=>new Element(),body:new Element()};
  el['seed-input'].value='0';el['auto-seed'].checked=true;el['avoid-recent'].checked=true;el['variation-select'].value='.8';
  let count=0,pending;const requests=[];
  const scope={document,location:{protocol:'file:'},navigator:{},performance,atob,Uint32Array,console,addEventListener(){},clearTimeout,
    setTimeout(fn,ms){return ms>=1000?0:setTimeout(fn,ms);},crypto:{getRandomValues(a){a[0]=++count*101;return a;}}};
  scope.window=scope;const context=vm.createContext(scope);
  for(const f of ['data.js','data7.js','engine.js'])vm.runInContext(fs.readFileSync(path.join(root,'web',f),'utf8'),context);
  const actual=scope.VerseLoomEngine;
  scope.VerseLoomEngine={generate(data,options,cb){requests.push({length:data.meta.lineLength??5,options:JSON.parse(JSON.stringify(options))});pending=actual.generate(data,options,cb);return pending;}};
  vm.runInContext(fs.readFileSync(path.join(root,'web/app.js'),'utf8'),context);
  async function act(id,event){assert.equal(el[id].disabled,false,id);el[id].listeners[event]({preventDefault(){}});await new Promise(setImmediate);if(pending)await pending;await new Promise(setImmediate);}
  const visible=()=>el['poem-lines'].children.map(line=>line.children.filter(c=>c.className?.startsWith('character')).map(c=>c.children[0].textContent).join(''));
  return {el,act,requests,visible,html};
}

test('switch 5 → 7 → 5 updates data, templates, metadata, history, glyphs and replay',async()=>{
  const app=application();assert.deepEqual(app.visible(),[]);assert(!app.html.includes('日暮春風起'));
  await app.act('generation-form','submit');assert(app.visible().every(x=>x.length===5));
  const fiveFirst=app.visible()[0];
  app.el['form-select'].value='7';await app.act('form-select','change');
  assert.deepEqual(app.visible(),[]);assert.equal(app.el['replay-button'].disabled,true);
  assert.equal(app.el['training-count'].textContent,'299');assert.equal(app.el['candidate-count'].textContent,'3,892');
  assert.equal(app.el['rhyme-select'].children.length,19);
  assert(app.el['template-select'].children.slice(1).every(x=>x.value.startsWith('L7-')));
  await app.act('generation-form','submit');const seven=app.visible();assert.equal(seven.length,8);assert(seven.every(x=>x.length===7));
  assert.equal(app.requests[1].length,7);assert.deepEqual(app.requests[1].options.openingHistory,[]);
  assert.equal(app.el['prefix-label'].textContent,'前四字');assert.equal(app.el['prefix-text'].textContent.length,4);
  const original=JSON.parse(JSON.stringify(app.requests[1]));
  app.el['seed-input'].value='999';app.el['variation-select'].value='1.5';await app.act('replay-button','click');
  assert.deepEqual(app.visible(),seven);assert.deepEqual(app.requests[2],original);
  await app.act('generation-form','submit');assert.deepEqual(app.requests[3].options.openingHistory,[seven[0]]);
  assert.notEqual(app.visible()[0].slice(0,2),seven[0].slice(0,2));
  app.el['form-select'].value='5';await app.act('form-select','change');
  assert.equal(app.el['training-count'].textContent,'2,203');assert.equal(app.el['rhyme-select'].children.length,13);
  await app.act('generation-form','submit');assert(app.visible().every(x=>x.length===5));
  assert.deepEqual(app.requests[4].options.openingHistory,[fiveFirst]);
});

test('seven manual seed, reset and invalid form do not lock or silently switch a sentence',async()=>{
  const app=application();app.el['form-select'].value='7';await app.act('form-select','change');
  app.el['auto-seed'].checked=false;app.el['seed-input'].value='42';
  await app.act('generation-form','submit');const lines=app.visible();await app.act('generation-form','submit');
  assert.deepEqual(app.visible(),lines);assert.deepEqual(app.requests[1],app.requests[0]);
  await app.act('reset-button','click');assert.equal(app.el['form-select'].value,'7');assert.equal(app.el['auto-seed'].checked,true);
  assert.equal(app.el['variation-select'].value,'0.8');await app.act('generation-form','submit');
  assert.deepEqual(app.requests[2].options.openingHistory,[]);assert.equal(app.requests[2].length,7);
  app.el['form-select'].value='6';await app.act('form-select','change');assert.equal(app.el['form-select'].value,'7');assert.equal(app.el['error-region'].hidden,false);
});
