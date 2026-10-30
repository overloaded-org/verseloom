(function(){'use strict';
function init(){
 const $=id=>document.getElementById(id), data=globalThis.VerseLoomCompositionData, engine=globalThis.VerseLoomComposition;
 let active=null,serial=0,result=null,history={5:[],7:[]};
 const labels={shape:'八句字数',meter:'声明声律规格',rhyme:'同部押韵',parallel_roles:'中联角色与词类',season_time:'季节时段约束',source_linked_readings:'读音证据关联'};
 function busy(b){for(const id of ['length','season','time','seed','auto','variation','beam','generate'])$(id).disabled=b;$('stop').hidden=!b;$('sheet').setAttribute('aria-busy',String(b));$('replay').disabled=b||!result;$('copy').disabled=b||!result;$('export').disabled=b||!result;}
 function error(message){$('error').textContent=message;$('error').hidden=false;}
 function fresh(previous){const a=new Uint32Array(1);if(globalThis.crypto?.getRandomValues){crypto.getRandomValues(a);return a[0]===previous?(previous+1)>>>0:a[0];}return(previous+1)>>>0;}
 function inspect(index){$('inspection').hidden=false;$('tokens').replaceChildren();for(const token of result.lineDetails[index].tokens){
  const box=document.createElement('div');box.className='token';const heading=document.createElement('h3');heading.textContent=token.text+' · '+token.role+' / '+token.semantic;box.appendChild(heading);
  const p=document.createElement('p');p.className='small';p.textContent='训练语料中出现 '+token.count+' 次；标签是预定词义与结构解释，不是独立金标。';box.appendChild(p);
  for(const r of token.readings){const d=document.createElement('details');d.className='reading';const summary=document.createElement('summary');summary.textContent=r.character+' · '+r.position+' · '+r.rhymeLabel+' · '+r.entryId;d.appendChild(summary);const text=document.createElement('p');text.textContent='反切：'+r.fanqie+'。原释义：'+r.originalGloss+'。释义参照：'+r.glossTrace.path.join(' → ')+'。方法：'+r.method+(r.intendedSense?'；预定义项：'+r.intendedSense:'')+'。语境独立认证：未完成。';d.appendChild(text);box.appendChild(d);}
  const src=document.createElement('p');src.className='small';src.textContent=token.sources.map(s=>s.poemId+' 第'+(s.row+1)+'句，字位'+s.start+'–'+s.end+'，'+s.split).join('；');box.appendChild(src);$('tokens').appendChild(box);
 }}
 function show(value){result=value;$('poem').replaceChildren();value.lines.forEach((line,i)=>{const b=document.createElement('button');b.type='button';b.className='poem-line';b.textContent=line+(i%2?'。':'，');b.setAttribute('aria-label','第'+(i+1)+'句，查看词组依据');b.addEventListener('click',()=>inspect(i));$('poem').appendChild(b);});
 $('title').textContent=(value.length===5?'五言':'七言')+' · 本次新稿';$('state').textContent='本机生成 · 非存档';$('context').textContent=value.contextLabel;$('meta').textContent=value.rhymeLabel+' / '+value.templateId+' / 种子 '+value.options.seed+' / '+(value.elapsedMs/1000).toFixed(2)+'秒';$('checks').replaceChildren();for(const c of value.checks){const s=document.createElement('span');s.textContent=(c.passed?'✓ ':'待核 ')+(labels[c.id]||c.id);$('checks').appendChild(s);}$('inspection').hidden=true;
 }
 function progress(job,p){if(job.cancelled||active!==job){const e=new Error('已取消');e.name='AbortError';throw e;}$('progress').textContent=p.message||'正在组织词组…';}
 function finish(job){if(active!==job)return false;if(job.worker)job.worker.terminate();active=null;busy(false);return true;}
 function accept(job,value){if(active!==job||job.cancelled)return;if(value.status!=='complete'){finish(job);error('本次没有找到满足全部声明约束的完整诗稿。可换种子、搜索宽度或约束条件重试；没有使用存档或放宽约束。');$('state').textContent=result?'保留上次结果':'未找到完整解';return;}
  if(value.length!==job.options.length||value.lines.length!==8||value.lines.some(l=>Array.from(l).length!==value.length)){fail(job,new Error('结果诗体不匹配'));return;}
  if(!job.replay)history[value.length]=history[value.length].concat([value.lines.slice()]).slice(-8);show(value);finish(job);$('progress').textContent='生成完成。点击任一句查看词组与历史读音依据。';
 }
 function fail(job,e){if(active!==job||job.cancelled)return;finish(job);error(e.message||String(e));}
 function fallback(job){if(job.fallback||active!==job)return;job.fallback=true;if(job.worker){job.worker.terminate();job.worker=null;}Promise.resolve().then(()=>engine.generate(data,job.options,p=>progress(job,p))).then(r=>accept(job,r)).catch(e=>fail(job,e));}
 function start(replay=false){if(active)return;let options;try{
   if(replay){if(!result)return;options=JSON.parse(JSON.stringify(result.options));}
   else{const seed=$('seed').value.trim();if(!/^\d{1,10}$/.test(seed)||Number(seed)>4294967295)throw new Error('种子须为0至4294967295的整数。');options={length:Number($('length').value),seed:Number(seed),variation:Number($('variation').value),beamWidth:Number($('beam').value),season:$('season').value,time:$('time').value};
    if($('auto').checked&&options.variation>0){options.seed=fresh(options.seed);options.recentPoems=history[options.length].map(p=>p.slice());}else options.recentPoems=[];
   }
 }catch(e){error(e.message);return;}
 $('seed').value=String(options.seed);if(replay){$('length').value=String(options.length);$('season').value=options.season;$('time').value=options.time;$('variation').value=String(options.variation);$('beam').value=String(options.beamWidth);}
 const job={id:++serial,options,replay,worker:null,cancelled:false};active=job;$('error').hidden=true;busy(true);$('state').textContent='本机计算中';$('progress').textContent='准备词组、音义与声律约束…';
 if(globalThis.Worker&&location.protocol!=='file:'&&!globalThis.VERSELOOM_COMPOSITION_STANDALONE){try{const worker=new Worker(new URL('composition-worker.js',document.baseURI));job.worker=worker;worker.onmessage=({data:m})=>{if(active!==job||m.id!==job.id)return;if(m.type==='progress')progress(job,m.progress);else if(m.type==='result')accept(job,m.result);else if(m.type==='error')fail(job,new Error(m.message));};worker.onerror=e=>{e.preventDefault();fallback(job);};worker.postMessage({id:job.id,options});}catch(e){fallback(job);}}else fallback(job);
 }
 $('form').addEventListener('submit',e=>{e.preventDefault();start();});$('replay').addEventListener('click',()=>start(true));$('stop').addEventListener('click',()=>{if(!active)return;const job=active;job.cancelled=true;finish(job);$('progress').textContent='已停止；没有填入任何备用诗稿。';$('state').textContent=result?'保留上次结果':'尚未生成';});
 $('length').addEventListener('change',()=>{if(active)return;result=null;$('poem').replaceChildren();$('title').textContent='一张空白诗笺';$('checks').replaceChildren();$('meta').textContent='';$('context').textContent='所有句位开放';$('state').textContent='尚未运行';$('progress').textContent='';$('inspection').hidden=true;busy(false);});
 $('copy').addEventListener('click',async()=>{if(!result)return;const text=result.lines.map((l,i)=>l+(i%2?'。':'，')).join('\n');try{await navigator.clipboard.writeText(text);$('progress').textContent='诗文已复制。';}catch(e){$('progress').textContent='浏览器未允许复制，请选中诗文复制。';}});
 $('export').addEventListener('click',()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='verseloom-phrases-'+result.length+'-'+result.options.seed+'.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 if(!data||!engine){busy(true);error('词组资源未加载，请使用完整站点或单文件版本。');return;}$('data-meta').textContent='本版本：'+data.meta.trainingPoems.toLocaleString('zh-CN')+'首训练来源，'+data.meta.lexemes+'个准入词项；整句输出库为0。';busy(false);
 window.addEventListener('pagehide',()=>{if(active){active.cancelled=true;active.worker?.terminate();}});
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();})();
