/* Word-level, source-backed experimental generator. No fixed or stored sentences. */
(function (global) {
  'use strict';
  const VERSION = 'verseloom.quality-words.v1';
  const charMasks=new WeakMap(), candidateCaches=new WeakMap();
  function maskFor(data,text) {
    if(!charMasks.has(data))charMasks.set(data,new Map(Object.keys(data.readings).sort(textCompare).map((c,i)=>[c,1n<<BigInt(i)])));
    return [...text].reduce((m,c)=>m|charMasks.get(data).get(c),0n);
  }
  const allowed = new Set(['lineLength','seed','variation','beamWidth','templateId','rhymeGroup','scene','openingHistory']);
  const requireValue = (ok, message) => { if (!ok) throw new Error(message); };
  const textCompare = (a,b) => a < b ? -1 : a > b ? 1 : 0;
  function noise(seed, key) {
    let h = (2166136261 ^ seed) >>> 0;
    for (const c of key) h = Math.imul(h ^ c.codePointAt(0), 16777619) >>> 0;
    h = Math.imul(h ^ h >>> 16, 0x7feb352d) >>> 0;
    h = Math.imul(h ^ h >>> 15, 0x846ca68b) >>> 0;
    return ((h ^ h >>> 16) >>> 0) / 4294967296;
  }
  function normalize(data, input) {
    requireValue(data?.schema === 'verseloom.word-lattice.v1', '词组数据版本不符。');
    requireValue(input && typeof input === 'object' && !Array.isArray(input), '参数必须是对象。');
    requireValue(Object.keys(input).every(k => allowed.has(k)), '不支持指定或固定任何诗句。');
    const o = Object.assign({lineLength:5,seed:0,variation:0.8,beamWidth:32,templateId:'auto',rhymeGroup:'auto',scene:'auto',openingHistory:[]},input);
    requireValue(o.lineLength === 5 || o.lineLength === 7, '只支持五言或七言。');
    requireValue(Number.isInteger(o.seed) && o.seed >= 0 && o.seed <= 4294967295, '种子须为 uint32 整数。');
    requireValue(typeof o.variation === 'number' && Number.isFinite(o.variation) && o.variation >= 0 && o.variation <= 2, '探索强度范围为 0–2。');
    requireValue(Number.isInteger(o.beamWidth) && o.beamWidth >= 1 && o.beamWidth <= 64, '搜索宽度范围为 1–64。');
    requireValue(o.templateId === 'auto' || data.templates[o.lineLength].some(t => t.template_id === o.templateId), '起式与诗体不匹配。');
    requireValue(o.rhymeGroup === 'auto' || data.rhymes.some(r => r.id === o.rhymeGroup), '未知韵部。');
    requireValue(o.scene === 'auto' || data.scenes.some(s => s.id === o.scene), '未知场景。');
    requireValue(Array.isArray(o.openingHistory) && o.openingHistory.length <= 8 && Array.from(o.openingHistory).every(x => typeof x === 'string' && Array.from(x).length === o.lineLength), '近期首句记录不合法。');
    o.openingHistory=o.openingHistory.slice();
    return o;
  }
  function compatibleWord(w, scene) {
    if (w.season && scene.season && w.season !== scene.season) return false;
    if (w.light === 'night' && scene.light === 'day') return false;
    if (w.light === 'day' && scene.light === 'night') return false;
    return true;
  }
  function lineScore(data, text, tokens) {
    let score = 0;
    for (let i=0;i<text.length-1;i++) score += Math.log1p(data.grams[text.slice(i,i+2)] || 0);
    for (let i=0;i<text.length-2;i++) score += 0.6*Math.log1p(data.grams[text.slice(i,i+3)] || 0);
    score /= (text.length-1)+0.6*(text.length-2);
    return score+0.08*tokens.reduce((s,w)=>s+Math.log1p(w.count),0)/tokens.length;
  }
  function grammarAccept(tokens, frame) {
    const shapes={SVO:['N2','V1','N2'],LSVO:['N2|L2','N2','V1','N2'],LNDI:['L2','N1','D1','I1'],TLNDI:['T2','L2','N1','D1','I1']};
    const shape=shapes[frame];
    if(!shape || tokens.length!==shape.length || tokens.some((w,i)=>!shape[i].split('|').includes(w.pos+w.text.length)))return false;
    const off=frame==='LSVO'?1:0;
    if (frame==='SVO'||frame==='LSVO') {
      const [s,v,o]=tokens.slice(off);
      if(frame==='LSVO' && tokens[0].pos==='N') {
        const setting=tokens[0];
        if(setting.kind==='building' && !['light','sound','human','mind','weather'].includes(s.kind))return false;
        if(setting.kind==='water' && !['light','sound','human','mind','weather','bird','boat'].includes(s.kind))return false;
      }
      if(v.kind==='enter') {
        const domains={human:['building','home','land'],mind:['home','land'],water:['water','land'],boat:['water'],
          bird:['land','sky','building'],weather:['land','building','water','home'],light:['water','land','building','home'],sound:['water','land','building','home']};
        if(!domains[s.kind]?.includes(o.kind))return false;
      }
      return v.subjects.includes(s.kind) && v.objects.includes(o.kind) && s.text!==o.text;
    }
    const [s,d,v]=tokens.slice(-3);
    if (!v.subjects.includes(s.kind)) return false;
    // Restrict modal/aspect interpretation rather than using arbitrary character fits.
    if (d.text==='欲' && ['靜','清','寒','寂'].includes(v.text)) return false;
    return true;
  }
  function signature(tokens,frame) {
    return frame+'/'+tokens.map(w=>w.pos+':'+w.text.length+':'+(w.pos==='N'?w.kind:'')).join('|');
  }
  function lineCandidates(data, o, patterns, scene, rhyme, stage, stats) {
    const words=data.words.filter(w=>compatibleWord(w,scene)).slice().sort((a,b)=>textCompare(a.text,b.text)||textCompare(a.pos,b.pos)||textCompare(a.kind,b.kind));
    const np=words.filter(w=>w.pos==='N'&&w.text.length===2);
    const loc=words.filter(w=>w.pos==='L');
    const locN=np.filter(w=>['water','land','building'].includes(w.kind)).map(w=>({...w,role:'setting'}));
    const human=np.filter(w=>['human','mind'].includes(w.kind));
    const nature=np.filter(w=>!['human','mind','home'].includes(w.kind));
    const sub=stage==='closing'?human.concat(nature):nature;
    const objects=stage==='closing'?np:nature;
    const verbs=words.filter(w=>w.pos==='V');
    const n1=words.filter(w=>w.pos==='N'&&w.text.length===1);
    const adverbs=words.filter(w=>w.pos==='D');
    const iv=words.filter(w=>w.pos==='I');
    const time=words.filter(w=>w.pos==='T');
    const frames=o.lineLength===5?
      [{id:'SVO',slots:[sub,verbs,objects]},...(stage==='closing'?[]:[{id:'LNDI',slots:[loc,n1,adverbs,iv]}])]:
      [{id:'LSVO',slots:[loc,sub,verbs,objects]},...(stage==='closing'?[]:[{id:'TLNDI',slots:[time,loc,n1,adverbs,iv]}])];
    const training=new Set(data.trainingLines), seen=new Map();
    for (const pattern of patterns) for (const frame of frames) {
      function walk(i, offset, tokens, used) {
        if (i===frame.slots.length) {
          stats.wordCombinations++;
          if (offset!==o.lineLength || !grammarAccept(tokens,frame.id)) {stats.semanticRejected++;return;}
          const text=tokens.map(w=>w.text).join('');
          if (training.has(text)) {stats.trainingLineRejected++;return;}
          const last=tokens[tokens.length-1];
          if (rhyme && last.endingGroup!==rhyme) return;
          const sig=signature(tokens,frame.id);
          const entry={text,tokens,frame:frame.id,signature:sig,tones:pattern,
                       endingGroup:last.endingGroup,mask:maskFor(data,text),score:lineScore(data,text,tokens)+(stage==='closing'&&tokens.some(w=>w.kind==='human'||w.kind==='mind')?0.7:0)+(stage==='closing'&&tokens.at(-1).kind==='home'?0.4:0)};
          const key=text+'/'+sig;
          if(!seen.has(key)) seen.set(key,entry);
          return;
        }
        for (const w of frame.slots[i]) {
          if (!pattern.startsWith(w.tones,offset)) continue;
          if (i===frame.slots.length-1 && rhyme && w.endingGroup!==rhyme) continue;
          if ([...w.text].some(c=>used.has(c)) || new Set(w.text).size!==w.text.length) continue;
          // Subject/verb incompatibility can be pruned before choosing an object.
          if (w.pos==='V' && tokens.length && !w.subjects.includes(tokens.at(-1).kind)) continue;
          if (w.pos==='N' && tokens.at(-1)?.pos==='V' && !tokens.at(-1).objects.includes(w.kind)) continue;
          walk(i+1,offset+w.text.length,tokens.concat(w),new Set([...used,...w.text]));
        }
      }
      walk(0,0,[],new Set());
    }
    return [...seen.values()];
  }
  function diversify(rows,limit,key,score) {
    const sorted=rows.slice().sort((a,b)=>score(b)-score(a)||textCompare(key(a)+'/'+a.text,key(b)+'/'+b.text));
    const groups=new Map();
    for (const r of sorted) {const k=key(r);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r);}
    const buckets=[...groups.values()], kept=[];
    for(let depth=0;kept.length<limit;depth++) {
      let added=false;
      for(const b of buckets) if(depth<b.length) {kept.push(b[depth]);added=true;if(kept.length===limit)break;}
      if(!added)break;
    }
    return kept;
  }
  function aligned(a,b) {
    if(a.signature!==b.signature || a.text.length!==b.text.length) return false;
    if(a.tokens.some((w,i)=>w.text===b.tokens[i].text))return false;
    return [...a.text].every((c,i)=>c!==[...b.text][i]);
  }
  function scoreUnit(unit,o,row,template) {
    return unit.reduce((s,l,i)=>s+l.score+o.variation*(2*noise(o.seed,template+'/'+(row+i)+'/'+l.text)-1),0);
  }
  const pause=()=>new Promise(resolve=>setTimeout(resolve,0));
  async function generate(data,input={},onProgress=()=>{}) {
    const started=Date.now(), o=normalize(data,input);
    requireValue(typeof onProgress==='function','进度回调无效。');
    const scenes=data.scenes.filter(s=>o.scene==='auto'||s.id===o.scene)
      .slice().sort((a,b)=>noise(o.seed,'scene/'+a.id)-noise(o.seed,'scene/'+b.id)||textCompare(a.id,b.id));
    const scene={...scenes[0]};
    if(!scene.season) {scene.season=noise(o.seed,'season')<0.5?'spring':'autumn';scene.label=(scene.season==='spring'?'春夜':'秋夜')+'山居';}
    const templates=data.templates[o.lineLength].filter(t=>o.templateId==='auto'||t.template_id===o.templateId);
    // Rhyme ordering is chosen from vocabulary coverage, never from a fixed poem.
    const endingCounts=new Map();
    for(const w of data.words.filter(w=>['N','I'].includes(w.pos)&&compatibleWord(w,scene)&&w.tones.endsWith('P'))) {
      if(!endingCounts.has(w.endingGroup))endingCounts.set(w.endingGroup,new Set());
      endingCounts.get(w.endingGroup).add(w.text.at(-1));
    }
    let rhymes=data.rhymes.filter(r=>o.rhymeGroup==='auto'? (endingCounts.get(r.id)?.size||0)>=5:r.id===o.rhymeGroup);
    rhymes=rhymes.slice().sort((a,b)=>noise(o.seed,'rhyme/'+a.id)-noise(o.seed,'rhyme/'+b.id)||textCompare(a.id,b.id));
    const stats={wordCombinations:0,semanticRejected:0,trainingLineRejected:0,attemptedStates:0,legalStates:0};
    if(!candidateCaches.has(data))candidateCaches.set(data,new Map());
    const cache=candidateCaches.get(data), charged=new Set(), finalists=[], attempts=[];
    const blocked=new Set(o.openingHistory), prefixes=new Set(o.openingHistory.slice(-4).map(x=>x.slice(0,2)));
    let step=0;
    for(const rhyme of rhymes) {
      for(const template of templates) {
        const rows=template.patterns.map((patterns,row)=> {
          const rhymeId=template.rhyme_rows.includes(row)?rhyme.id:null;
          const stage=row>=6?'closing':'landscape';
          const key=o.lineLength+'/'+scene.id+'/'+scene.season+'/'+patterns.join(',')+'/'+(rhymeId||'-')+'/'+stage;
          if(!cache.has(key)) {
            const localStats={wordCombinations:0,semanticRejected:0,trainingLineRejected:0};
            cache.set(key,{lines:lineCandidates(data,o,patterns,scene,rhymeId,stage,localStats),stats:localStats});
          }
          if(!charged.has(key)){charged.add(key);for(const [k,n] of Object.entries(cache.get(key).stats))stats[k]+=n;}
          const pool=cache.get(key).lines.filter(l=>row!==0||(!blocked.has(l.text)&&!prefixes.has(l.text.slice(0,2))));
          const scorer=l=>l.score+o.variation*(2*noise(o.seed,template.template_id+'/'+row+'/'+l.text)-1);
          if(rhymeId) {
            const byEnd=new Map();for(const l of pool){const k=l.text.at(-1);if(!byEnd.has(k))byEnd.set(k,[]);byEnd.get(k).push(l);}
            const fair=[];for(const [end,items] of [...byEnd].sort(([a],[b])=>textCompare(a,b))) {
              const k=Math.max(1,Math.floor(256/byEnd.size));
              fair.push(...diversify(items,k,l=>l.signature,scorer));
            }
            return fair;
          }
          return diversify(pool,256,l=>l.signature,scorer);
        });
        const unitsByStart=[];
        for(const start of [0,1,2,4,6,7]) {
          const isPair=start===2||start===4;let units=[];
          if(isPair) {
            const lower=new Map();
            for(const l of rows[start+1]){if(!lower.has(l.signature))lower.set(l.signature,[]);lower.get(l.signature).push(l);}
            for(const a of rows[start])for(const b of lower.get(a.signature)||[])if(aligned(a,b))units.push([a,b]);
          } else units=rows[start].map(l=>[l]);
          const ranked=units.map(u=>({unit:u,text:u.map(l=>l.text).join('/'),value:scoreUnit(u,o,start,template.template_id)}));
          const selected=diversify(ranked,384,r=>r.unit[0].signature+'/'+r.unit.at(-1).text.at(-1),r=>r.value);
          unitsByStart.push({start,isPair,units:selected,total:units.length});
        }
        // Most constrained joint unit first, rather than fixing any initial line.
        unitsByStart.sort((a,b)=>a.units.length-b.units.length||a.start-b.start);
        let states=[{lines:Array(8).fill(null),score:0,selectionScore:0,chars:new Map(),endings:new Set(),saturated:0n}];
        const trace=[];const assigned=new Set();
        for(const plan of unitsByStart) {
          const {start,isPair}=plan;
          assigned.add(start);if(isPair)assigned.add(start+1);
          const extensions=[];
          for(const state of states)for(const choice of plan.units) {
            stats.attemptedStates++;
            const chars=new Map(state.chars), ends=new Set(state.endings);let ok=true,penalty=0,saturated=state.saturated;
            for(let k=0;k<choice.unit.length;k++) {
              const l=choice.unit[k],row=start+k;
              if(state.lines.some(x=>x?.text===l.text)){ok=false;break;}
              if(template.rhyme_rows.includes(row)){
                const ending=l.text.at(-1);if(ends.has(ending)){ok=false;break;}ends.add(ending);
              }
              for(const c of l.text){const count=chars.get(c)||0;if(count>=2){ok=false;break;}penalty+=count*0.28;chars.set(c,count+1);if(count===1)saturated|=maskFor(data,c);}
              if(!ok)break;
            }
            if(!ok)continue;
            const remaining=[];
            for(let next=0;next<8;next++)if(!assigned.has(next)) {
              const rhymeRow=template.rhyme_rows.includes(next);let any=false;const possibleEnds=new Set();
              for(const l of rows[next]) {
                if((l.mask & saturated)!==0n || (rhymeRow && ends.has(l.text.at(-1))))continue;
                any=true;if(!rhymeRow)break;possibleEnds.add(l.text.at(-1));
              }
              if(!any){ok=false;break;}
              if(rhymeRow)remaining.push(possibleEnds);
            }
            if(!ok)continue;
            function assignEnds(i,used) {
              if(i===remaining.length)return true;
              for(const c of remaining[i])if(!used.has(c)&&assignEnds(i+1,new Set([...used,c])))return true;
              return false;
            }
            if(!assignEnds(0,new Set()))continue;
            stats.legalStates++;
            const lines=state.lines.slice();choice.unit.forEach((l,k)=>{lines[start+k]=l;});
            extensions.push({lines,text:lines.map(l=>l?.text||'').join('/'),chars,endings:ends,saturated,
              score:state.score+choice.unit.reduce((s,l)=>s+l.score,0)-penalty,
              selectionScore:state.selectionScore+choice.value-penalty});
          }
          states=diversify(extensions,o.beamWidth,s=>(s.lines[0]||s.lines.find(Boolean)).text.slice(0,2),s=>s.selectionScore);
          trace.push({rows:isPair?[start,start+1]:[start],jointUnits:plan.total,retained:states.length,
                      distinctOpenings:new Set(states.map(s=>s.lines[0]?.text).filter(Boolean)).size});
          onProgress({message:scene.label+' · 联合组织第'+(start+1)+(isPair?'、'+(start+2):'')+'句',
                      step:++step,rhyme:rhyme.id,template:template.template_id});
          await pause();
          if(!states.length)break;
        }
        attempts.push({rhyme:rhyme.id,template:template.template_id,trace,complete:states.filter(s=>s.lines.every(Boolean)).length});
        for(const state of states)if(state.lines.every(Boolean))finalists.push({state,template,rhyme});
      }
      // Auto rhyme selects the first fully feasible rhyme in seeded order, recorded above.
      if(finalists.length)break;
    }
    let chosen=null;
    if(finalists.length) {
      const entries=finalists.slice().sort((a,b)=>textCompare(a.state.text,b.state.text)||textCompare(a.template.template_id,b.template.template_id));
      if(o.variation===0)chosen=entries.sort((a,b)=>b.state.score-a.state.score||textCompare(a.state.text,b.state.text))[0];
      else {
        const groups=new Map();for(const e of entries){const k=e.state.lines[0].text.slice(0,2);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(e);}
        function draw(list,key){const max=Math.max(...list.map(e=>e.state.selectionScore/8));
          const weights=list.map(e=>Math.exp((e.state.selectionScore/8-max)/o.variation));let ticket=noise(o.seed,'final/'+key)*weights.reduce((a,b)=>a+b,0);
          for(let i=0;i<list.length;i++){ticket-=weights[i];if(ticket<0)return list[i];}return list.at(-1);}
        const reps=[...groups].sort(([a],[b])=>textCompare(a,b)).map(([k,g])=>({...g.slice().sort((a,b)=>b.state.selectionScore-a.state.selectionScore)[0],group:k}));
        const rep=draw(reps,'group');chosen=draw(groups.get(rep.group),'path/'+rep.group);
      }
    }
    const lines=chosen?chosen.state.lines.map(l=>l.text):[];
    const report={schema:'verseloom.quality.result.v1',engineVersion:VERSION,status:chosen?'complete':'infeasible',
      lineLength:o.lineLength,lines,options:o,scene,templateId:chosen?.template.template_id||null,
      rhyme:chosen?.rhyme||null,score:chosen?.state.score??null,selectionStage:'after-eight-lines',
      lineDetails:chosen?chosen.state.lines.map(l=>({text:l.text,frame:l.frame,signature:l.signature,tones:l.tones,
        tokens:l.tokens.map(w=>({text:w.text,pos:w.pos,kind:w.kind,sense:w.sense,readingIds:w.readingIds,sources:w.sources})),score:l.score})):[],
      pairChecks:chosen?[2,4].map(i=>({rows:[i,i+1],matched:aligned(chosen.state.lines[i],chosen.state.lines[i+1]),
        signature:chosen.state.lines[i].signature,scope:'declared_word_boundaries_POS_roles_semantic_classes',strictAntithesisCertified:false})):[],
      provenance:{sources:data.sources,preselectedLines:false,storedSentencePool:false,neuralModelUsed:false,
        historicalReadingBasis:'source_entries_plus_assistant_sense_policies',contextualPronunciationCertified:false,
        literaryQualityCertified:false,independentReview:false},stats,attempts,elapsedMs:Date.now()-started};
    if(chosen) {
      const audit=verify(data,report);requireValue(audit.passed,'生成内部核验失败：'+audit.failures.join('、'));
      report.checks=audit;
    }
    return report;
  }
  function verify(data,r) {
    const failures=[];const fail=(ok,s)=>{if(!ok)failures.push(s);};
    const t=data.templates[r.lineLength]?.find(t=>t.template_id===r.templateId);
    fail(Boolean(t),'template');fail(r.lines?.length===8,'eight-lines');
    if(!t||!r.lineDetails||r.lineDetails.length!==8)return {passed:false,failures:failures.concat('details')};
    const counts=new Map(),ends=[];const originals=new Set(data.trainingLines);
    for(let i=0;i<8;i++) {
      const d=r.lineDetails[i], text=r.lines[i];
      fail([...text].length===r.lineLength,'length:'+i);fail(new Set(text).size===r.lineLength,'within-line-repeat:'+i);
      fail(!originals.has(text),'training-line:'+i);
      fail(d.tokens.map(w=>w.text).join('')===text,'tokens:'+i);
      const tokens=d.tokens.map(w=>data.words.find(e=>e.text===w.text&&e.pos===w.pos&&e.kind===w.kind));
      fail(tokens.every(Boolean),'word-source:'+i);
      if(tokens.some(x=>!x))continue;
      fail(grammarAccept(tokens,d.frame),'semantics:'+i);
      fail(signature(tokens,d.frame)===d.signature,'signature:'+i);
      fail(tokens.every((w,j)=>JSON.stringify(w.readingIds)===JSON.stringify(d.tokens[j].readingIds)),'reading-identity:'+i);
      fail(tokens.every(w=>compatibleWord(w,r.scene)),'scene:'+i);
      const tones=[...text].map(c=>data.readings[c]?.tone||'?').join('');
      fail(t.patterns[i].includes(tones),'tones:'+i);
      fail([...text].every(c=>data.readings[c]?.admitted),'reading:'+i);
      for(const c of text)counts.set(c,(counts.get(c)||0)+1);
      if(t.rhyme_rows.includes(i)) {fail(data.readings[text.at(-1)].group===r.rhyme.id,'rhyme:'+i);ends.push(text.at(-1));}
    }
    fail(new Set(ends).size===ends.length,'rhyme-repetition');fail([...counts.values()].every(n=>n<=2),'poem-repetition');
    for(const i of [2,4])fail(aligned(r.lineDetails[i],r.lineDetails[i+1]),'couplet:'+i);
    return {passed:!failures.length,failures,scope:'declared_operational_constraints_not_literary_certification'};
  }
  global.VerseLoomQuality=Object.freeze({generate,verify,version:VERSION});
})(globalThis);
