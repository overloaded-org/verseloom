/* VerseLoom typed lexical composition. No stored sentence pool or line anchors. */
(function (global) {
  "use strict";
  const VERSION = "verseloom.phrase-composition.v2";
  const reject = (ok, message) => { if (!ok) throw new Error(message); };
  const textOrder = (a,b) => a < b ? -1 : a > b ? 1 : 0;
  const sum = xs => xs.reduce((a,b)=>a+b,0);
  const delay = () => new Promise(r=>setTimeout(r,0));
  function hash(text,seed=0) { let h=(2166136261^seed)>>>0;for(const c of text)h=Math.imul(h^c.codePointAt(0),16777619)>>>0;return h; }
  function noise(text,seed) {let h=hash(text,seed);h=Math.imul(h^(h>>>16),0x7feb352d)>>>0;h=Math.imul(h^(h>>>15),0x846ca68b)>>>0;return ((h^(h>>>16))>>>0)/4294967296;}
  const seasonNames={spring:"春",summer:"夏",autumn:"秋",winter:"冬"};
  const timeNames={dawn:"清晓",day:"日间",dusk:"薄暮",night:"夜间"};
  const adjectiveEntities={
    "清":"water sound scent light", "寒":"water wind rain moon glow sound building", "暖":"wind sun glow", "靜":"water tree flower grass building vessel",
    "寂":"building sound land", "幽":"land building tree flower", "碧":"water grass tree leaf land", "翠":"tree grass leaf land", "青":"tree grass leaf land",
    "紅":"glow flower leaf", "白":"flower bird", "綠":"water tree grass leaf", "新":"flower grass tree leaf", "老":"tree traveler", "殘":"flower leaf glow",
    "疏":"tree flower leaf", "密":"tree grass leaf", "長":"water land", "短":"grass", "高":"building tree land", "低":"glow bird vessel", "明":"moon glow sun",
    "暗":"moon glow building", "遠":"land bird vessel", "近":"land building", "孤":"traveler vessel bird building", "斜":"sun glow", "濃":"scent tree",
    "香":"flower grass tree scent", "晚":"flower", "早":"flower"
  };
  const adjectiveClass={清:"sensory",寒:"temperature",暖:"temperature",靜:"activity",寂:"activity",幽:"space",碧:"color",翠:"color",青:"color",紅:"color",白:"color",綠:"color",新:"age",老:"age",殘:"age",疏:"density",密:"density",長:"extent",短:"extent",高:"height",低:"height",明:"brightness",暗:"brightness",遠:"distance",近:"distance",孤:"number",斜:"orientation",濃:"density",香:"sensory",晚:"time",早:"time"};
  function normalize(input) {
    reject(input && typeof input==="object" && !Array.isArray(input),"参数必须是对象。");
    const allowed=new Set(["length","seed","variation","beamWidth","season","time","rhymeGroup","templateId","recentPoems"]);
    reject(Object.keys(input).every(k=>allowed.has(k)),"不接受固定句或未知参数。");
    const o={length:5,seed:0,variation:0.8,beamWidth:32,season:"auto",time:"auto",rhymeGroup:"auto",templateId:"auto",recentPoems:[],...input};
    reject([5,7].includes(o.length),"诗体必须是五言或七言。");
    reject(Number.isInteger(o.seed)&&o.seed>=0&&o.seed<=4294967295,"种子须为32位非负整数。");
    reject(typeof o.variation==="number"&&Number.isFinite(o.variation)&&o.variation>=0&&o.variation<=2,"探索强度必须在0至2之间。");
    reject(Number.isInteger(o.beamWidth)&&o.beamWidth>=8&&o.beamWidth<=64,"搜索宽度须为8至64。");
    reject(o.season==="auto"||o.season==="unrestricted"||Object.hasOwn(seasonNames,o.season),"未知季节。");
    reject(o.time==="auto"||o.time==="unrestricted"||Object.hasOwn(timeNames,o.time),"未知时段。");
    reject(o.templateId==="auto"||new RegExp('^L'+o.length+'-[PZ]-R[01]$').test(o.templateId),"起式与诗体不匹配。");
    reject(typeof o.rhymeGroup==="string","韵部必须是字符串。");
    reject(Array.isArray(o.recentPoems)&&o.recentPoems.length<=8&&Array.from(o.recentPoems).every(p=>Array.isArray(p)&&p.length===8&&Array.from(p).every(l=>typeof l==="string"&&Array.from(l).length===o.length)),"近期诗稿必须为相同诗体的最多8首完整诗。");
    o.recentPoems=o.recentPoems.map(p=>p.slice());return o;
  }
  function parallel(a,b) {
    if(a.frame!==b.frame||a.tokens.length!==b.tokens.length)return false;
    // Exact role, span and declared semantic class matching, not just different characters.
    for(let i=0;i<a.tokens.length;i++) {
      const x=a.tokens[i],y=b.tokens[i];
      if(x.kind!==y.kind||x.role!==y.role||x.semantic!==y.semantic||x.text.length!==y.text.length||x.text===y.text)return false;
    }
    // Guard a narrow form of same-scene paraphrase; not a universal 合掌 detector.
    if(a.subject.entity===b.subject.entity)return false;
    return true;
  }
  const roles = (tokens,names) => tokens.map((t,i)=>({...t,role:names[i]}));
  function candidate(tokens,frame,subject,object,options,data) {
    const text=tokens.map(t=>t.text).join("");
    const chars=Array.from(text);
    if(new Set(chars).size!==chars.length)return null;
    // Exact train-line hash denial is conservative on a hash collision (never approves a copied line).
    if(data.banned.has(hash(text)))return null;
    const tones=tokens.map(t=>t.tones).join("");
    if(!data.patterns.has(tones))return null;
    const boundaries=[]; let offset=0;
    for(const t of tokens){offset+=t.text.length;if(offset<text.length)boundaries.push(text.slice(offset-1,offset+1));}
    const support=sum(boundaries.map(t=>Math.log1p(data.source.bigrams[t]||0)))/Math.max(1,boundaries.length);
    const frequency=sum(tokens.map(t=>Math.log1p(t.count)))/tokens.length;
    const raw=0.4*frequency+0.6*support;
    const sampling=options.variation*(2*noise(frame+"/"+text,options.seed)-1);
    return {text,tones,frame,tokens,subject,object,rhymeGroup:tokens.at(-1).rhymeGroup,
      score:raw,selectionScore:raw+sampling,samplingBias:sampling,boundarySupport:support};
  }
  async function compileCandidates(source,o,context,progress) {
    const patterns=new Set(source.templates[String(o.length)].flatMap(t=>t.variants.flat()));
    const data={source,banned:new Set(source.trainingLineHashes),patterns};
    const fits=t=>(context.season==="unrestricted"||!t.seasons.length||t.seasons.includes(context.season))&&(context.time==="unrestricted"||!t.times.length||t.times.includes(context.time));
    const words=source.lexicon.filter(fits); const nouns=words.filter(t=>t.kind==="noun"),verbs=words.filter(t=>t.kind==="verb"),locations=words.filter(t=>t.kind==="location"),times=words.filter(t=>t.kind==="time"),adjectives=words.filter(t=>t.kind==="adjective");
    const pools=new Map();let constructed=0,matched=0;
    function add(tokens,frame,s,obj) {
      constructed++;
      const c=candidate(tokens,frame,s,obj,o,data);if(!c)return;matched++;
      const key=c.tones+"/"+(c.tones.endsWith('P')?c.rhymeGroup:'nonrhyming');
      if(!pools.has(key))pools.set(key,[]);pools.get(key).push(c);
    }
    let tick=0;
    for(const v of verbs) {
      for(const s of nouns.filter(n=>v.subjects.includes(n.entity)))for(const obj of nouns.filter(n=>v.objects.includes(n.entity))) {
        if(s.text===obj.text)continue;
        if(v.text==='入'&&['wind','sound','scent'].includes(s.entity)&&obj.entity!=='building')continue;
        if(v.text==='入'&&s.entity==='water'&&!/[江河溪湖池塘澗灣]/.test(obj.text))continue;
        if(s.entity==='tears'&&obj.entity!=='clothing')continue;
        if(v.text==='滋'&&['leaf','flower'].includes(obj.entity)&&/[黃紅落殘秋]/.test(obj.text))continue;
        const core=roles([s,v,obj],["subject","predicate",v.role]);
        if(o.length===5)add(core,"SVO",s,obj);
        else {
          for(const loc of locations){add(roles([loc,s,v,obj],["setting","subject","predicate",v.role]),"LOC_SVO",s,obj);
            add(roles([s,loc,v,obj],["subject","setting","predicate",v.role]),"SUBJ_LOC_VO",s,obj);}
          for(const time of times)add(roles([time,s,v,obj],["time","subject","predicate",v.role]),"TIME_SVO",s,obj);
        }
        if(++tick%1000===0){progress({phase:"lexical",message:"正在按词义、角色和平仄组合新句",constructed});await delay();}
      }
    }
    // Alternative frame; the adjective slot carries a finer semantic class.
    if(o.length===5)for(const loc of locations)for(const n of nouns)for(const adj of adjectives){
      if(!(adjectiveEntities[adj.text]||"").split(' ').includes(n.entity))continue;
      if(adjectiveClass[adj.text]==='color'&&(/[青碧翠綠紅白黃]/.test(n.text)&&!n.text.includes(adj.text)))continue;
      if(adj.text==='紅'&&/[竹荷蓮柳]/.test(n.text))continue;
      add(roles([loc,n,{...adj,semantic:adjectiveClass[adj.text]||'state'}],["setting","subject","predicate"]),"LOC_STATE",n,null);
    }

    if(o.length===7)for(const loc of locations)for(const n of nouns)for(const adv of words.filter(t=>t.kind==='adverb'))for(const adj of words.filter(t=>t.kind==='adjective2')) {
      if(!adj.appliesTo.includes(n.entity))continue;
      add(roles([loc,n,adv,adj],["setting","subject","aspect","predicate"]),"LOC_STATE7",n,null);
    }
    // Keep representatives from every structural/semantic opening family, not just high-frequency words.
    for(const [key,rows] of pools){
      const unique=new Map();for(const c of rows)unique.set(c.frame+'/'+c.text,c);
      const sorted=[...unique.values()].sort((a,b)=>b.selectionScore-a.selectionScore||textOrder(a.frame+'/'+a.text,b.frame+'/'+b.text));
      const buckets=new Map();for(const c of sorted){const k=c.frame+'/'+c.text.at(-1)+'/'+c.subject.entity+'/'+(c.object?.semantic||'state');if(!buckets.has(k))buckets.set(k,[]);buckets.get(k).push(c);}
      const kept=[];const bs=[...buckets.values()];for(let depth=0;kept.length<256;depth++){let found=false;for(const b of bs){if(b[depth]){found=true;kept.push(b[depth]);if(kept.length===256)break;}}if(!found)break;}
      pools.set(key,kept.sort((a,b)=>b.selectionScore-a.selectionScore||textOrder(a.text,b.text)));
    }
    progress({phase:"lexical",message:"词组组合完成，开始联合选择四联",constructed,matched});await delay();
    return {pools,constructed,matched};
  }
  function retain(states,width) {
    states.sort((a,b)=>b.selectionScore-a.selectionScore||textOrder(a.key,b.key));
    const groups=new Map();for(const s of states){const key=s.lines.find(Boolean).text.slice(0,2);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(s);}
    const kept=[];for(let depth=0;kept.length<width;depth++){let found=false;for(const g of groups.values()){if(g[depth]){found=true;kept.push(g[depth]);if(kept.length===width)break;}}if(!found)break;}return kept;
  }
  function place(state,line,rhymeRows,row) {
    const lines=state.lines.slice(),counts=new Map(),ends=[],seen=new Set(),words=new Set(),events=new Set();
    lines[row]=line;
    for(let i=0;i<lines.length;i++) {
      if(!lines[i])continue;
      const current=lines[i],text=current.text;
      for(const token of current.tokens){if(token.text.length>1){if(words.has(token.text))return null;words.add(token.text);}}
      const verb=current.tokens.find(t=>t.kind==='verb');
      const event=current.subject.entity+'/'+(verb?.text||current.tokens.at(-1).text)+'/'+(current.object?.semantic||'state');
      if(events.has(event))return null;events.add(event);
      if(seen.has(text))return null;seen.add(text);
      for(const c of text){counts.set(c,(counts.get(c)||0)+1);if(counts.get(c)>2)return null;}
      if(rhymeRows.includes(i)){const last=text.at(-1);if(ends.includes(last))return null;ends.push(last);}
    }
    if(lines[2]&&lines[4]&&lines[2].frame===lines[4].frame)return null;
    const score=state.score+line.score;
    const selectionScore=state.selectionScore+line.selectionScore;
    return {lines,score,selectionScore,key:lines.map(l=>l?l.text:'').join('/')};
  }
  function extension(state,pair,rhymeRows,row) {
    const partial=place(state,pair[0],rhymeRows,row);
    return partial?place(partial,pair[1],rhymeRows,row+1):null;
  }
  function drawComplete(states,o){
    states.sort((a,b)=>textOrder(a.key,b.key));
    if(o.variation===0)return states.slice().sort((a,b)=>b.score-a.score||textOrder(a.key,b.key))[0];
    const best=Math.max(...states.map(s=>s.selectionScore/8));
    const weights=states.map(s=>Math.exp((s.selectionScore/8-best)/o.variation));
    let ticket=noise('completed-poem',o.seed)*sum(weights);
    for(let i=0;i<states.length;i++){ticket-=weights[i];if(ticket<0)return states[i];}return states.at(-1);
  }
  async function generate(source,input={},onProgress=()=>{}) {
    reject(source?.schema==="verseloom.composition-data.v2","词组数据版本不匹配。");reject(typeof onProgress==='function','进度回调无效。');
    const o=normalize(input), started=performance.now();
    const context={season:o.season==='auto'?Object.keys(seasonNames)[Math.floor(noise('season',o.seed)*4)]:o.season,
      time:o.time==='auto'?Object.keys(timeNames)[Math.floor(noise('time',o.seed)*4)]:o.time};
    const {pools,constructed,matched}=await compileCandidates(source,o,context,onProgress);
    const templates=source.templates[String(o.length)].filter(t=>o.templateId==='auto'||t.id===o.templateId);
    const rhymes=[...new Set([...pools.keys()].filter(k=>!k.endsWith('nonrhyming')).map(k=>k.slice(k.indexOf('/')+1)))];
    reject(o.rhymeGroup==='auto'||rhymes.includes(o.rhymeGroup),'当前词组、季节和时段不支持该韵部。');
    rhymes.sort((a,b)=>{
      const coverage=g=>Math.min(...[...new Set(templates.flatMap(t=>t.core).filter(p=>p.endsWith('P')))].map(p=>new Set((pools.get(p+'/'+g)||[]).map(l=>l.text.at(-1))).size));
      return coverage(b)-coverage(a)||textOrder(a,b);
    });
    const selectedRhymes=o.rhymeGroup==='auto'?rhymes.slice(0,10):[o.rhymeGroup];
    let finalists=[],comparisons=0;const pairCache=new Map();const histories=new Set(o.recentPoems.map(p=>p.join('/')));const diagnostics=[];
    for(const rhyme of selectedRhymes)for(const t of templates) {
      let states=[{lines:Array(8).fill(null),score:0,selectionScore:0,key:''}]; const stageCounts=[];
      for(const couplet of [1,2,0,3]) {
        const i=2*couplet;
        const a=t.variants[i].flatMap(p=>pools.get(p+'/'+(t.rhymeRows.includes(i)?rhyme:'nonrhyming'))||[]);
        const b=t.variants[i+1].flatMap(p=>pools.get(p+'/'+rhyme)||[]);
        if(couplet===0||couplet===3) {
          // The outer couplets need no syntactic pairing: apply legality before pruning.
          // This prevents a globally high-ranked but incompatible shortlist from hiding legal completions.
          const startCount=states.length;
          for(const [row,pool] of [[i,a],[i+1,b]]) {
            const next=[];
            for(const state of states)for(const line of pool){comparisons++;const e=place(state,line,t.rhymeRows,row);if(e)next.push(e);}
            states=retain(next,o.beamWidth);
            onProgress({phase:'poem',message:t.id+' · 组织第'+(row+1)+'句',templateId:t.id,rhyme,stage:couplet});await delay();
            if(!states.length)break;
          }
          stageCounts.push({couplet,pairs:null,states:states.length,startingStates:startCount,legalityBeforePruning:true,openings:new Set(states.map(s=>s.lines.find(Boolean).text)).size});
          if(!states.length)break;
          continue;
        }
        const cacheKey=t.variants[i].join(',')+'|'+t.variants[i+1].join(',')+'|'+rhyme;
        let cached=pairCache.get(cacheKey);
        if(!cached) {
        const pairs=[];
        for(const u of a)for(const l of b){
          comparisons++;
          if(u.text===l.text)continue;
          if((couplet===1||couplet===2)&&(!parallel(u,l)||Array.from(u.text).some((c,k)=>c===Array.from(l.text)[k])))continue;
          // Two state descriptions do not license arbitrary emotion jumps; all words already share scene tags.
          const quality=u.selectionScore+l.selectionScore;
          const coda=couplet===3&&[u.subject,l.subject].some(s=>['heart','traveler','tears'].includes(s.entity))?0.5:0;
          pairs.push({pair:[u,l],score:quality+coda,key:u.frame+'/'+u.text+'/'+l.text});
        }
        pairs.sort((a,b)=>b.score-a.score||textOrder(a.key,b.key));
        // Balanced upper-line alternatives, jointly scored with their compatible lower lines.
        const groups=new Map();for(const p of pairs){const key=p.pair[0].text;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(p);}
        const choice=[];for(let k=0;choice.length<1024;k++){let found=false;for(const rows of groups.values()){if(rows[k]){found=true;choice.push(rows[k]);if(choice.length===1024)break;}}if(!found)break;}
        cached={choice,pairCount:pairs.length};pairCache.set(cacheKey,cached);
        }
        const {choice,pairCount}=cached;
        const next=[];for(const s of states)for(const p of choice){const e=extension(s,p.pair,t.rhymeRows,i);if(e)next.push(e);}
        states=retain(next,o.beamWidth);stageCounts.push({couplet,pairs:pairCount,retainedPairs:choice.length,states:states.length,openings:new Set(states.map(s=>s.lines.find(Boolean).text)).size});
        onProgress({phase:'poem',message:t.id+' · 联合组织第'+(couplet+1)+'联',templateId:t.id,rhyme,stage:couplet});await delay();
        if(!states.length)break;
      }
      const done=states.filter(s=>s.lines.every(Boolean)&&!histories.has(s.key)&&!o.recentPoems.some(p=>p[0]===s.lines[0].text));
      for(const state of done)finalists.push({...state,template:t,rhyme});
      diagnostics.push({templateId:t.id,rhyme,completed:done.length,stages:stageCounts});
    }
    const chosen=finalists.length?drawComplete(finalists,o):null;
    const provenance={preselectedLines:false,storedOutputLines:false,usesExternalModel:false,selectionStage:'after-eight-lines',
      grammar:'typed-lexical-role-construction',semanticLabels:'assistant-authored-intended-senses',contextualPronunciationCertified:false,
      strictAntithesisCertified:false,literaryQualityCertified:false,sourceIdentity:source.sourceIdentity};
    if(!chosen)return {status:'infeasible',schema:'verseloom.composition-result.v2',engineVersion:VERSION,lines:[],options:o,context,provenance,diagnostics,constructed,matched,comparisons,elapsedMs:performance.now()-started};
    const lines=chosen.lines.map(l=>l.text);const rlabel=source.readings[lines[1].at(-1)].rhymeLabel;
    return {status:'complete',schema:'verseloom.composition-result.v2',engineVersion:VERSION,length:o.length,lines,
      templateId:chosen.template.id,rhymeGroup:chosen.rhyme,rhymeLabel:rlabel,core:chosen.template.core,actualTones:chosen.lines.map(l=>l.tones),metricalProfile:'common_prefix',rhymeRows:chosen.template.rhymeRows,
      score:chosen.score,selectionScore:chosen.selectionScore,options:o,context,contextLabel:(seasonNames[context.season]||"不限定季节")+' · '+(timeNames[context.time]||"不限定时段"),
      lineDetails:chosen.lines.map(l=>({text:l.text,frame:l.frame,tones:l.tones,score:l.score,samplingBias:l.samplingBias,
        tokens:l.tokens.map(t=>({id:t.id,text:t.text,kind:t.kind,semantic:t.semantic,role:t.role,entity:t.entity,count:t.count,sources:t.sources,
          readings:Array.from(t.text).map(c=>source.readings[c])}))})),
      checks:[{id:'shape',passed:lines.every(l=>Array.from(l).length===o.length)&&lines.length===8},
        {id:'meter',passed:chosen.lines.every((l,i)=>chosen.template.variants[i].includes(l.tones))},
        {id:'rhyme',passed:chosen.template.rhymeRows.every(i=>chosen.lines[i].rhymeGroup===chosen.rhyme)},
        {id:'parallel_roles',passed:parallel(chosen.lines[2],chosen.lines[3])&&parallel(chosen.lines[4],chosen.lines[5]),scope:'declared roles and semantic slots, not literary certification'},
        {id:'season_time',passed:true,scope:'declared lexical restrictions'},
        {id:'source_linked_readings',passed:true,scope:'intended sense or single-entry evidence; not independent contextual certification'}],
      provenance,diagnostics,constructed,matched,comparisons,completedPaths:finalists.length,elapsedMs:performance.now()-started};
  }
  global.VerseLoomComposition=Object.freeze({generate,parallel,version:VERSION});
})(globalThis);
