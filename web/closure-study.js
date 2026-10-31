/* Exhaustive ending-pair search over the current generated lattice, never a line bank on disk. */
(function(g){
'use strict';
const policy='verseloom.closure-pairs.v1';
// Fitted on train families only: scripts/probe_closing_order.py; refuse mismatched statistics.
const weights=Object.freeze([0.043852069671080074,-0.2883127359675583,0.2463232889115275]);
const pinned={compilerSha256:'57506aa5f94b2fcf7acebf337ca5ba6377d61b92bbfcc27dfbcb9b058aed0ad5',newCandidateCorpus:'a87ae4b81709d6662fccc03f1e983c1dee1936aba4a36cc512c16fabea12ca5c',oldTrainCorpus:'5011658d63415772495ff645f92cf90e075998e096e672204f4a71234737360c',referenceSha256:'6481c560885544fed72bc426c7461f2cea01dde8c299eab1e041d23a18364537',guangyunSha256:'f2b66197355d4fbff0776ab34e4aece817363b00446fbcf08e2f1677a7ac0c5f'};
const baseCoefficients=[2.6430792130297385,3.4468543432935115,4.677392649015701,0.7565911724278884,0.4130565333320451,-2.8503341771239676,2.659080565401457,4.083922534113606];
const demand=(v,m)=>{if(!v)throw Error(m);};
function fnv(s){let h=2166136261;for(const c of s)h=Math.imul(h^c.codePointAt(0),16777619)>>>0;return h;}
const sigmoid=x=>1/(1+Math.exp(-Math.min(40,Math.max(-40,x))));
const later=(a,b)=>{const x=[...a],y=[...b];for(let k=0;k<Math.min(x.length,y.length);k++){let d=x[k].codePointAt(0)-y[k].codePointAt(0);if(d)return d;}return x.length-y.length;};
function win(a,b,field){return !b||a[field]>b[field]||(a[field]===b[field]&&(later(a.u.text,b.u.text)<0||(a.u.text===b.u.text&&later(a.l.text,b.l.text)<0)));}
function strata(context,line,association){
  return [0,2,4].map(start=>{
    const earlier=new Set(context.slice(start,start+2).flatMap(r=>r.tokens));let total=0;
    for(const v of line.tokens){let best=0;for(const u of earlier)if(u!==v)best=Math.max(best,association.get(u)?.get(v)?.score||0);total+=Math.min(4,best);}
    return total/line.tokens.length;
  });
}
function residual(z){const mean=(z[0]+z[1]+z[2])/3;return z.reduce((a,x,i)=>a+(x-mean)*weights[i],0);}
async function generate(data,options,progress=()=>{}){
  demand(options&&Object.keys(options).every(k=>['length','seed'].includes(k))&&[5,7].includes(options.length)&&Number.isInteger(options.seed)&&options.seed>=0&&options.seed<=4294967295,'只接受五／七言和整数种子，不接收题材、前缀或候选诗句。');
  demand(Object.entries(pinned).every(([k,v])=>data.sourceIdentity?.[k]===v)&&JSON.stringify(g.VerseLoomContinuationData.coefficients)===JSON.stringify(baseCoefficients)&&g.VerseLoomContinuationData.center===0.8638766050595278,'模型不匹配；没有回退到存档。');
  const n=options.length,base=await g.VerseLoomInduced.generate(data,{length:n,seed:options.seed,mode:'learned',variation:.8,beamWidth:16,recentPoems:[]},progress);
  const header={policy,options:{length:n,seed:options.seed},sourceIdentity:data.sourceIdentity,engineVersion:base.engineVersion,baselineOptions:base.options,poolFingerprint:base.poolFingerprint};
  if(base.status!=='complete')return {...header,status:'infeasible',lines:[],reason:'no_complete_baseline'};
  demand(base.checks.every(c=>c.passed),'原始诗稿机械检查未通过。');
  const {upper,lower,template,score,association,banned}=g.VerseLoomContinuation.closingBank(data,base);
  const lex=data.lexicon,ctx=base.lineDetails.slice(0,6).map(d=>({text:d.text,tokens:d.tokens.map(t=>t.id)})),prefix=ctx.map(r=>r.text);
  const counts=new Map(),words=new Set(),ends=new Set(),seen=new Set(prefix);
  for(const [i,p] of ctx.entries()){for(const c of p.text)counts.set(c,(counts.get(c)||0)+1);for(const v of p.tokens)if([...lex[v].text].length>1)words.add(v);if(template.rhymeRows.includes(i))ends.add([...p.text].at(-1));}
  function prepare(line,row){
    const chars=[...line.text],multi=line.tokens.filter(id=>[...lex[id].text].length>1),tones=line.tokens.map(id=>lex[id].tones).join('');
    demand(line.tokens.map(id=>lex[id].text).join('')===line.text&&tones===line.tones&&chars.every((c,i)=>data.readings[c].tone===tones[i]),'候选身份检查失败');
    if(chars.length!==n||new Set(chars).size!==n||banned.has(fnv(line.text))||!template.variants[row].includes(tones)||seen.has(line.text)||multi.some(id=>words.has(id))||chars.some(c=>(counts.get(c)||0)>=2))return null;
    if(template.rhymeRows.includes(row)&&(ends.has(chars.at(-1))||data.readings[chars.at(-1)].rhymeGroup!==base.rhymeGroup))return null;
    return {line,chars,multi,local:line.score/n-.3*chars.reduce((a,c)=>a+(counts.get(c)||0),0)};
  }
  const uppers=upper.map(x=>prepare(x,6)).filter(Boolean),lowers=lower.map(x=>prepare(x,7)).filter(Boolean);
  for(const l of lowers){l.z=strata(ctx,l.line,association);l.dr=residual(l.z);}
  let greedy=null,joint=null,ordered=null,original=null,legal=0;const legalKeys=[];
  for(const [idx,U] of uppers.entries()){
    if(idx%12===0){progress({phase:'endings',message:'联合比较尾联 '+idx+' / '+uppers.length});await new Promise(r=>setTimeout(r,0));}
    const u=U.line,context=ctx.concat(u),ru=score(ctx,u).value,us=U.local+ru;
    for(const L of lowers){const l=L.line;
      if(u.text===l.text||L.multi.some(id=>U.multi.includes(id))||L.chars.some(c=>U.chars.includes(c)&&(counts.get(c)||0)>=1))continue;
      if(template.rhymeRows.includes(6)&&template.rhymeRows.includes(7)&&U.chars.at(-1)===L.chars.at(-1))continue;
      const penalty=.3*L.chars.filter(c=>U.chars.includes(c)).length,raw=score(context,l).learned.rawScore;
      const fixed=us+L.local-penalty,js=fixed+sigmoid(raw),os=fixed+sigmoid(raw+L.dr);
      const a={u,l,us,js,os,raw,z:L.z,dr:L.dr};legal++;legalKeys.push([u.text,l.text]);
      if(!greedy||us>greedy.us||(us===greedy.us&&(later(u.text,greedy.u.text)<0||(u.text===greedy.u.text&&win(a,greedy,'js')))))greedy=a;
      if(win(a,joint,'js'))joint=a;if(win(a,ordered,'os'))ordered=a;
      if(u.text===base.lines[6]&&l.text===base.lines[7])original=a;
    }
  }
  demand(legal&&original,'原尾联不在共同合法候选中；已停止，不以存档替代。');
  legalKeys.sort((a,b)=>later(a[0],b[0])||later(a[1],b[1]));
  const bytes=new TextEncoder().encode(JSON.stringify(legalKeys)),ab=await crypto.subtle.digest('SHA-256',bytes),pairHash=Array.from(new Uint8Array(ab),x=>x.toString(16).padStart(2,'0')).join('');
  function pack(a){return {ending:[a.u.text,a.l.text],tokenIds:[a.u.tokens,a.l.tokens],tones:[a.u.tones,a.l.tones],statisticalScore:a.js,orderedScore:a.os,closingAssociations:a.z,orderLogitAdjustment:a.dr,rawClosingLogit:a.raw};}
  return {...header,status:'complete',prefix,lines:prefix.concat(ordered.u.text,ordered.l.text),rhymeGroup:base.rhymeGroup,templateId:template.id,legalPairs:legal,legalPairSha256:pairHash,original:pack(original),greedy:pack(greedy),joint:pack(joint),ordered:pack(ordered),modelWeights:weights,defaultGeneratorChanged:false,literaryQualityCertified:false};
}
g.VerseLoomClosure=Object.freeze({generate,version:policy});
})(globalThis);
