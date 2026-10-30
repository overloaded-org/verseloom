/* Corpus-derived segment lattice and relation search. No authored content words. */
(function(global){
'use strict';
const VERSION='verseloom.corpus-lattice.v1.2', prepared=new WeakMap();
const check=(v,m)=>{if(!v)throw new Error(m);};
const pause=()=>new Promise(r=>setTimeout(r,0));
const order=(a,b)=>a<b?-1:a>b?1:0, lastChar=s=>Array.from(s).at(-1);
function hash(s,seed=0){let h=(2166136261^seed)>>>0;for(const c of s)h=Math.imul(h^c.codePointAt(0),16777619)>>>0;return h;}
function uniform(s,seed){let h=hash(s,seed);h=Math.imul(h^(h>>>16),0x7feb352d)>>>0;h=Math.imul(h^(h>>>15),0x846ca68b)>>>0;return (((h^(h>>>16))>>>0)+0.5)/4294967296;}
const gumbel=(s,seed)=>-Math.log(-Math.log(uniform(s,seed)));
function options(input){
 check(input&&typeof input==='object'&&!Array.isArray(input),'参数必须为对象');
 const names=new Set(['length','seed','variation','beamWidth','mode','recentPoems']);
 check(Object.keys(input).every(k=>names.has(k)),'不接受主题、预填句、词类或故事路线');
 const o={length:5,seed:0,variation:0.8,beamWidth:32,mode:'relations',recentPoems:[],...input};
 check([5,7].includes(o.length),'只支持五言或七言');
 check(Number.isInteger(o.seed)&&o.seed>=0&&o.seed<=4294967295,'种子必须为uint32整数');
 check(typeof o.variation==='number'&&Number.isFinite(o.variation)&&o.variation>=0&&o.variation<=2,'探索强度须为0至2');
 check(Number.isInteger(o.beamWidth)&&o.beamWidth>=8&&o.beamWidth<=64,'搜索宽度须为8至64');
 check(['local','relations','learned'].includes(o.mode),'未知对照模式');
 check(Array.isArray(o.recentPoems)&&o.recentPoems.length<=8&&Array.from(o.recentPoems).every(p=>Array.isArray(p)&&p.length===8&&Array.from(p).every(s=>typeof s==='string'&&Array.from(s).length===o.length)),'近期诗稿格式无效');
 o.recentPoems=o.recentPoems.map(p=>p.slice());return o;
}
function prepare(data){
 if(prepared.has(data))return prepared.get(data);
 check(data?.schema==='verseloom.induced-data.v1','数据版本错误');
 const lex=data.lexicon.map((w,id)=>({...w,id,size:Array.from(w.text).length}));
 check(lex.length>0&&lex.every(w=>w.size>=1&&w.size<=3&&w.tones.length===w.size&&/^[PZ]+$/.test(w.tones)&&Number.isFinite(w.prob)&&w.prob>0),'词段数据错误');
 const total=lex.reduce((a,w)=>a+w.prob,0);lex.forEach(w=>w.prior=w.prob/total);
 const maps={};for(const k of ['next','association','following','alignment']){
  maps[k]=new Map(Object.entries(data[k]).map(([u,es])=>[+u,new Map(es.map(([v,c,s])=>[v,{count:c,score:s}]))]));
 }
 const nextTotals=new Map([...maps.next].map(([u,es])=>[u,[...es.values()].reduce((a,x)=>a+x.count,0)]));
 const triples=new Map(Object.entries(data.triples).map(([k,es])=>[k,new Map(es)]));
 const positions=new Map(Object.entries(data.positions).map(([k,es])=>[k,new Map(es)]));
 const r={lex,maps,nextTotals,triples,positions,poolCache:new Map(),charV:Object.keys(data.charCounts).length,banned:new Set(data.trainingLineHashes),byTone:new Map()};
 for(const w of lex){if(!r.byTone.has(w.tones))r.byTone.set(w.tones,[]);r.byTone.get(w.tones).push(w);}
 prepared.set(data,r);return r;
}
const ranked=(a,b)=>b.rank-a.rank||order(a.text,b.text);
function keep(states,width){
 states.sort(ranked);const seen=new Set(),buckets=new Map();
 for(const s of states){if(seen.has(s.text))continue;seen.add(s.text);const key=Array.from(s.lines?.[0]?.text||s.text).slice(0,2).join('');if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(s);}
 const out=[];for(let depth=0;out.length<width;depth++){let found=false;for(const b of buckets.values()){if(b[depth]){found=true;out.push(b[depth]);if(out.length===width)break;}}if(!found)break;}return out;
}
async function linePools(data,r,o,progress){
 const cacheKey=[o.length,o.seed,o.variation].join('/');
 if(r.poolCache.has(cacheKey)){progress({phase:'lattice',message:'复用同输入的词段搜索，不复用整诗'});await pause();return r.poolCache.get(cacheKey);}
 const patterns=[...new Set(data.templates[String(o.length)].flatMap(t=>t.variants.flat()))].sort();
 const starts=new Map(data.start[String(o.length)]),ends=new Map(data.end[String(o.length)]);const startTotal=[...starts.values()].reduce((a,c)=>a+c,0);
 const pools=new Map();let attempts=0,copied=0;
 for(let pi=0;pi<patterns.length;pi++){
  const pattern=patterns[pi], beams=Array.from({length:o.length+1},()=>[]);beams[0]=[{text:'',tokens:[],score:0,rank:0,edges:[]}];
  const compatible=Array.from({length:o.length},(_,pos)=>r.lex.filter(w=>pos+w.size<=o.length&&pattern.slice(pos,pos+w.size)===w.tones&& (pos+w.size<o.length||pattern.at(-1)==='Z'||w.rhymeGroup)));
  const proposals=compatible.map((ws,pos)=>ws.map(w=>({w,key:Math.log(pos===0?((starts.get(w.id)||0)+1)*w.prior**0.15:w.prior)+o.variation*gumbel('proposal/'+pattern+'/'+pos+'/'+w.id,o.seed)})).sort((a,b)=>b.key-a.key||a.w.id-b.w.id).slice(0,pos===0?256:64).map(x=>x.w));
  const compatibleIds=compatible.map(xs=>new Set(xs.map(w=>w.id)));
  for(let pos=0;pos<o.length;pos++){
   const states=pos===0?beams[0]:keep(beams[pos],64);
   for(const s of states){
    const last=s.tokens.at(-1), edges=last===undefined?new Map():r.maps.next.get(last)||new Map();
    const nextIds=new Set(proposals[pos].map(w=>w.id));
    for(const id of edges.keys())if(compatibleIds[pos].has(id))nextIds.add(id);
    const tri=s.tokens.length>=2?r.triples.get(s.tokens.at(-2)+':'+last):null, tritotal=tri?[...tri.values()].reduce((a,c)=>a+c,0):0;
    if(tri)for(const id of tri.keys())if(compatibleIds[pos].has(id))nextIds.add(id);
    for(const id of [...nextIds].sort((a,b)=>a-b)){
     const w=r.lex[id];attempts++;const text=s.text+w.text, size=pos+w.size;
     if(new Set(text).size!==Array.from(text).length)continue;
     const count=last===undefined?(starts.get(id)||0):(edges.get(id)?.count||0);
     const denominator=last===undefined?startTotal:r.nextTotals.get(last)||0;
     const pairp=(count+4*w.prior)/(denominator+4);
     const tricount=tri?.get(id)||0;
     const logp=Math.log((tricount+2*pairp)/(tritotal+2));
     const positionCount=r.positions.get(o.length+':'+pos)?.get(id)||0;
     if(pos>0&&!count&&!tricount)continue;
     if(size===o.length&&!ends.has(id))continue;
     let boundary=0;if(s.text){const c=lastChar(s.text);boundary=Math.log(((data.charPairs[c+Array.from(w.text)[0]]||0)+0.1)/((data.charCounts[c]||0)+0.1*r.charV));}
     const endpoint=size===o.length?0.25*Math.log1p(ends.get(id)||0):0;
     const score=s.score+logp+0.12*boundary+0.15*Math.log1p(positionCount)+endpoint;
     const rank=score+o.variation*1.6*gumbel('path/'+pattern+'/'+text,o.seed);
     const st={text,tokens:s.tokens.concat(id),score,rank,edges:s.edges.concat({from:last??null,to:id,count,source:last===undefined?null:data.nextEvidence[last+':'+id]||null})};
     if(size===o.length){
      if(r.banned.has(hash(text))){copied++;continue;}
      const key=pattern+'/'+(pattern.at(-1)==='P'?w.rhymeGroup:'nonrhyming');
      if(!pools.has(key))pools.set(key,[]);pools.get(key).push({...st,tones:pattern,rhymeGroup:w.rhymeGroup});
     }else beams[size].push(st);
    }
   }
  }
  progress({phase:'lattice',message:'词段自行接续 '+(pi+1)+'/'+patterns.length,attempts});await pause();
 }
 for(const [k,rows] of pools){pools.set(k,keep(rows,48));}
 const result={pools,attempts,copied};if(r.poolCache.size>=2)r.poolCache.delete(r.poolCache.keys().next().value);r.poolCache.set(cacheKey,result);return result;
}
function relational(previous,line,r,middle){
 if(!previous.length)return {value:0,association:0,following:0,alignment:0,witnesses:[]};
 const current=line.tokens, earlier=[...new Set(previous.flatMap(l=>l.tokens))],last=previous.at(-1).tokens;
 const receipts=[];
 function averageMax(a,b,table,kind){let total=0;for(const v of b){let best=0,pick=null;for(const u of a){if(u===v)continue;const edge=table.get(u)?.get(v);if(edge&&edge.score>best){best=edge.score;pick={kind,from:u,to:v,count:edge.count,score:edge.score};}}total+=Math.min(4,best);if(pick)receipts.push(pick);}return total/Math.max(1,b.length);}
 const association=averageMax(earlier,current,r.maps.association,'poem_cooccurrence');
 const following=averageMax(last,current,r.maps.following,'next_line');
 let alignment=0;
 if(middle){let a=0;const spans=new Map();for(const u of last){const n=r.lex[u].size;spans.set(a+':'+(a+n),u);a+=n;}
  let b=0,matched=0;for(const v of current){const n=r.lex[v].size,u=spans.get(b+':'+(b+n));b+=n;const e=r.maps.alignment.get(u)?.get(v);if(e){alignment+=Math.min(4,e.score);matched++;receipts.push({kind:'same_span_middle_pair',from:u,to:v,count:e.count,score:e.score});}}
  alignment/=Math.max(last.length,current.length);
 }
 return {value:0.35*association+0.25*following+0.4*alignment,association,following,alignment,witnesses:receipts};
}

  // Content-neutral recency: count each 2/3-character span once per prior poem.
  // This is a soft search preference, never a blacklist or a literary score.
  function recentSpanCounts(poems) {
    const counts = new Map();
    for (const poem of poems) {
      const spans = new Set();
      for (const line of poem) {
        const chars = Array.from(line);
        for (const width of [2, 3]) for (let i = 0; i + width <= chars.length; i++)
          spans.add(chars.slice(i, i + width).join(""));
      }
      for (const span of spans) counts.set(span, (counts.get(span) || 0) + 1);
    }
    return counts;
  }
  function spanPenalty(text, counts) {
    const chars = Array.from(text), terms = [];
    for (const width of [2, 3]) for (let i = 0; i + width <= chars.length; i++)
      terms.push(Math.log1p(counts.get(chars.slice(i, i + width).join("")) || 0));
    return terms.reduce((a, b) => a + b, 0) / Math.max(1, terms.length);
  }

async function generate(data,input={},onProgress=()=>{}){
 check(typeof onProgress==='function','进度回调无效');const o=options(input),r=prepare(data),start=performance.now();
 let learned=null;
 if(o.mode==='learned'){
  check(global.VerseLoomContinuation&&global.VerseLoomContinuationData,'接续学习资源缺失；未回退到其他模式');
  learned=global.VerseLoomContinuation.create(data,global.VerseLoomContinuationData,r);
 }
 const {pools,attempts,copied}=await linePools(data,r,o,onProgress);
 const rhymes=[...new Set([...pools.keys()].filter(k=>!k.endsWith('/nonrhyming')).map(k=>k.slice(k.indexOf('/')+1)))];
 const coverage=g=>[...pools].filter(([k])=>k.endsWith('/'+g)).reduce((a,[,v])=>a+new Set(v.map(l=>lastChar(l.text))).size,0);
 rhymes.sort((a,b)=>(Math.log(coverage(b)+1)+o.variation*gumbel(b,o.seed))-(Math.log(coverage(a)+1)+o.variation*gumbel(a,o.seed))||order(a,b));
 const recency=recentSpanCounts(o.recentPoems), penaltyCache=new Map();
 const penaltyFor=line=>{if(o.variation===0||!recency.size)return 0;if(!penaltyCache.has(line.text))penaltyCache.set(line.text,0.5*spanPenalty(line.text,recency));return penaltyCache.get(line.text);};
 const finalists=[],diagnostics=[];let considered=0;
 for(const rhyme of rhymes.slice(0,4))for(const template of data.templates[String(o.length)]){
  let states=[{lines:[],score:0,rank:0,text:'',relation:[],historyPenalties:[]}];const stage=[];
  for(let row=0;row<8;row++){
   const group=template.rhymeRows.includes(row)?rhyme:'nonrhyming';
   const pool=template.variants[row].flatMap(t=>pools.get(t+'/'+group)||[]);const extensions=[];
   for(const s of states){
    const chars=new Map();for(const old of s.lines)for(const c of old.text)chars.set(c,(chars.get(c)||0)+1);
    const ends=new Set(template.rhymeRows.filter(i=>i<row).map(i=>lastChar(s.lines[i].text)));
    const words=new Set(s.lines.flatMap(l=>l.tokens).filter(i=>r.lex[i].size>1));
    for(const line of pool){considered++;
     if(s.lines.some(l=>l.text===line.text)||[...line.text].some(c=>(chars.get(c)||0)>=2))continue;
     if(template.rhymeRows.includes(row)&&ends.has(lastChar(line.text)))continue;
     if(line.tokens.some(i=>r.lex[i].size>1&&words.has(i)))continue;
     if(row===0&&o.recentPoems.some(p=>p[0]===line.text))continue;
     const rel=learned?learned(s.lines,line):o.mode==='relations'?relational(s.lines,line,r,row===3||row===5):{value:0,association:0,following:0,alignment:0,witnesses:[]};
     const repeated=[...line.text].reduce((a,c)=>a+(chars.get(c)||0),0);
     const historyPenalty=penaltyFor(line);
     const score=s.score+line.score/o.length-0.3*repeated+(o.mode!=='local'?rel.value:0)-historyPenalty;
     const text=s.text+'/'+line.text;
     const rank=score+o.variation*1.2*gumbel('poem/'+template.id+'/'+text,o.seed);
     extensions.push({text,lines:s.lines.concat(line),score,rank,relation:s.relation.concat(rel),historyPenalties:s.historyPenalties.concat(historyPenalty)});
    }
   }
   states=keep(extensions,o.beamWidth);stage.push({row,pool:pool.length,states:states.length,distinctOpenings:new Set(states.map(s=>s.lines[0].text)).size});
   if(!states.length)break;
  }
  for(const s of states)if(s.lines.length===8&&!o.recentPoems.some(p=>p.join('/')===s.lines.map(l=>l.text).join('/')))finalists.push({...s,rhyme,template});
  diagnostics.push({templateId:template.id,rhyme,stage});onProgress({phase:'whole-poem',message:'整首比较：'+template.id+' / '+rhyme,completed:finalists.length});await pause();
 }
 finalists.sort((a,b)=>b.rank-a.rank||order(a.text,b.text));
 const chosen=finalists[0],provenance={authoredContentLexemes:0,authoredSemanticCategories:0,authoredSentenceFrames:0,preselectedLines:false,
  storedOutputLines:0,selectionStage:'after-eight-lines',sourceIdentity:data.sourceIdentity,
  historyPolicy:'all-line-2-3gram-soft-penalty-v1',continuationRanker:learned?{schema:global.VerseLoomContinuationData.schema,training:global.VerseLoomContinuationData.training}:null,graphMeaning:'statistical_association_not_event_causality',contextualPronunciationCertified:false,strictAntithesisCertified:false,literaryQualityCertified:false};
 const base={schema:'verseloom.induced-result.v1',engineVersion:VERSION,options:o,length:o.length,provenance,attempts,copiedTrainingLinesRejected:copied,considered,diagnostics,
  completeCandidates:finalists.length,poolFingerprint:hash([...pools].map(([k,rs])=>k+':'+rs.map(l=>l.text).join(',')).join('|')),elapsedMs:performance.now()-start};
 if(!chosen)return {...base,status:'infeasible',lines:[]};
 const lines=chosen.lines.map(l=>l.text);
 return {...base,status:'complete',lines,score:chosen.score,templateId:chosen.template.id,rhymeGroup:chosen.rhyme,
  rhymeRows:chosen.template.rhymeRows,actualTones:chosen.lines.map(l=>l.tones),
  lineDetails:chosen.lines.map((l,i)=>({text:l.text,tones:l.tones,localScore:l.score,boundaries:l.tokens.map(id=>r.lex[id].size),
   tokens:l.tokens.map(id=>({...data.lexicon[id],id,sourcePoem:data.sourcePoems[data.lexicon[id].source[0]],readings:[...r.lex[id].text].map(c=>({character:c,...data.readings[c]}))})),
   historyPenalty:chosen.historyPenalties[i],transitions:l.edges,relations:chosen.relation[i]})),
  checks:[{id:'shape',passed:lines.length===8&&lines.every(l=>[...l].length===o.length)},
   {id:'meter',passed:chosen.lines.every((l,i)=>chosen.template.variants[i].includes(l.tones))},
   {id:'rhyme',passed:chosen.template.rhymeRows.every(i=>chosen.lines[i].rhymeGroup===chosen.rhyme)},
   {id:'no_training_line',passed:lines.every(l=>!r.banned.has(hash(l)))},
   {id:'no_authored_content',passed:true,scope:'No content lexicon, theme, event script or sentence frame read by this engine'}]};
}
global.VerseLoomInduced=Object.freeze({generate,version:VERSION});
})(globalThis);
