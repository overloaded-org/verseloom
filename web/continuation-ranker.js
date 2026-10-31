/* Learned source-adjacency ranking; no content vocabulary, themes or sentence frames. */
(function(global){
'use strict';
const NAMES=Object.freeze(['poem_association','next_line','middle_alignment','reverse_next_line',
  'same_span_ratio','character_overlap','multi_segment_overlap','next_edge_coverage']);
const check=(ok,message)=>{if(!ok)throw new Error(message);};
const sessions=new WeakMap();
function extractSpans(context,target,lex,maps){
  if(!context.length)return Array(NAMES.length).fill(0);
  const current=target.map(t=>t[0]),last=context.at(-1).map(t=>t[0]);
  const earlier=[...new Set(context.flatMap(row=>row.map(t=>t[0])))].sort((a,b)=>a-b);
  const lookup=(table,u,v)=>table.get(u)?.get(v)?.score||0;
  function maxmean(left,right,name){let total=0,covered=0;
    for(const v of right){let best=0;for(const u of left)if(u>=0&&v>=0&&u!==v)best=Math.max(best,lookup(maps[name],u,v));
      total+=Math.min(4,best);if(best>0)covered++;}
    return [total/Math.max(1,right.length),covered/Math.max(1,right.length)];
  }
  const association=maxmean(earlier,current,'association')[0];
  const [following,coverage]=maxmean(last,current,'following');
  const reverse=maxmean(current,last,'following')[0];
  const previous=new Map(context.at(-1).map(([i,start,end])=>[start+':'+end,i]));
  let same=0,alignment=0;
  for(const [v,start,end] of target){const u=previous.get(start+':'+end);if(u!==undefined){same++;
    if(u>=0&&v>=0)alignment+=Math.min(4,lookup(maps.alignment,u,v));}}
  const den=Math.max(1,last.length,current.length);
  alignment=[3,5].includes(context.length)?alignment/den:0;
  const chars=ids=>new Set(ids.filter(i=>i>=0).flatMap(i=>Array.from(lex[i].text)));
  const a=chars(last),b=chars(current),intersection=[...a].filter(c=>b.has(c)).length;
  const overlap=intersection/Math.max(1,new Set([...a,...b]).size);
  const multis=ids=>new Set(ids.filter(i=>i>=0&&Array.from(lex[i].text).length>1).map(i=>i));
  const lm=multis(last),rm=multis(current);
  const multi=[...lm].filter(i=>rm.has(i)).length/Math.max(1,rm.size);
  return [association,following,alignment,reverse,same/den,overlap,multi,coverage];
}
function create(data,model,runtime){
  check(model?.schema==='verseloom.continuation-ranker.v1','接续模型未加载或版本不匹配');
  check(JSON.stringify(model.features)===JSON.stringify(NAMES),'接续特征定义不匹配');
  check(Array.isArray(model.coefficients)&&model.coefficients.length===NAMES.length&&
    model.coefficients.every(Number.isFinite)&&Number.isFinite(model.center),'接续参数无效');
  const keys=Object.keys(data.sourceIdentity).sort();
  check(JSON.stringify(keys)===JSON.stringify(Object.keys(model.sourceIdentity||{}).sort())&&
    keys.every(k=>model.sourceIdentity[k]===data.sourceIdentity[k]),'接续模型与当前语料不一致');
  check(model.authoredContentLexemes===0&&model.literaryQualityCertified===false,'错误的模型声明');
  const cache=new WeakMap();
  function spans(line){if(cache.has(line))return cache.get(line);let pos=0;
    const out=line.tokens.map(id=>{const start=pos;pos+=runtime.lex[id].size;return [id,start,pos];});
    cache.set(line,out);return out;}
  const score=(previous,line)=>{
    const f=extractSpans(previous.map(spans),spans(line),runtime.lex,runtime.maps);
    const contributions=f.map((x,i)=>x*model.coefficients[i]);
    const raw=contributions.reduce((a,b)=>a+b,0)-model.center;
    const value=previous.length?1/(1+Math.exp(-Math.max(-40,Math.min(40,raw)))):0;
    return {value,association:f[0],following:f[1],alignment:f[2],witnesses:[],
      learned:{schema:model.schema,features:f,contributions,rawScore:raw,
        interpretation:'bounded_source_order_preference_not_literary_probability'}};
  };
  sessions.set(data,{runtime,score});return score;
}
// Read the just-completed search in the same Worker. Never use an archived pool.
// Selecting a live seed/length is the only user-facing input in the closure tool.
function closingBank(data,result){
  const s=sessions.get(data),o=result.options;
  check(s&&result.status==='complete'&&o.mode==='learned'&&!o.recentPoems.length,'Missing matching learned search');
  const lattice=s.runtime.poolCache.get([o.length,o.seed,o.variation].join('/'));
  check(lattice,'The search lattice is no longer available');
  const template=data.templates[String(o.length)].find(t=>t.id===result.templateId);
  const get=i=>template.variants[i].flatMap(pattern=>lattice.pools.get(pattern+'/'+(template.rhymeRows.includes(i)?result.rhymeGroup:'nonrhyming'))||[]);
  const upper=get(6),lower=get(7);
  for(const [i,bank] of [[6,upper],[7,lower]])check(bank.some(l=>l.text===result.lines[i]&&l.score===result.lineDetails[i].localScore&&JSON.stringify(l.tokens)===JSON.stringify(result.lineDetails[i].tokens.map(t=>t.id))),'Ending candidate identity mismatch');
  return {upper,lower,template,score:s.score,association:s.runtime.maps.association,banned:s.runtime.banned};
}
global.VerseLoomContinuation=Object.freeze({create,extractSpans,features:NAMES,closingBank});
})(globalThis);
