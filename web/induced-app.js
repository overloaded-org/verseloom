/* Controller only. Candidate construction, defaults, model and search are unchanged. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const histories={5:[],7:[]}, meta=globalThis.VerseLoomInducedMeta;
  const ready=Boolean(meta && typeof Worker==='function');
  let worker=null, active=null, result=null, serial=0;
  function status(s){$('i-status').textContent=s;}
  function mobileScroll(node){if(globalThis.matchMedia?.('(max-width: 760px)')?.matches) node.scrollIntoView?.({block:'start',behavior:'auto'});}
  function busy(on){
    for(const id of ['i-length','i-mode','i-seed','i-auto','i-generate','i-history','i-reset-history']) $(id).disabled=on||!ready;
    $('i-history').disabled=on||!ready||!$('i-auto').checked;
    $('i-cancel').hidden=!on; $('i-sheet').setAttribute('aria-busy',String(on));
    for(const id of ['i-again','i-replay','i-copy','i-export']) $(id).disabled=on||!ready||!result;
    $('i-generate').textContent=on?'正在生成…':ready?'开始生成':'生成器未就绪';
    $('i-again').textContent=$('i-auto').checked?'再生成一首':'按当前设置生成';
  }
  function stop(){worker?.terminate();worker=null;active=null;busy(false);}
  function fail(s){stop();$('i-error').textContent=s;$('i-error').hidden=false;status('本次未完成。');}
  function resetEvidence(){ $('i-inspection').hidden=true; $('i-detail').textContent=''; }
  function show(r){
    result=r; const poem=$('i-poem');poem.replaceChildren();resetEvidence();
    r.lines.forEach((line,i)=>{const b=document.createElement('button');b.type='button';b.className='poem-line';b.textContent=line+(i%2?'。':'，');
      b.setAttribute('aria-label','第'+(i+1)+'句：'+line+'，查看依据');b.setAttribute('aria-controls','i-inspection');b.setAttribute('aria-pressed','false');
      b.onclick=()=>{for(const x of poem.children)x.setAttribute('aria-pressed',String(x===b));const pane=$('i-inspection');pane.hidden=false;
        const d=r.lineDetails[i];pane.querySelector('h2').textContent='第'+(i+1)+'句 · '+line;
        $('i-detail').textContent=JSON.stringify(d,null,2);mobileScroll(pane);};poem.appendChild(b);
    });
    $('i-summary').textContent=r.templateId+' / '+r.rhymeGroup+' / 种子 '+r.options.seed+' / '+({local:'句内对照',relations:'固定关系评分',learned:'学得接续评分'}[r.options.mode]||r.options.mode)+' / '+r.engineVersion;
    status('生成完成。点击诗句可查看依据。');busy(false);mobileScroll($('i-sheet'));
  }
  function ensureWorker(){if(worker)return;const w=new Worker(new URL('induced-worker.js',document.baseURI));worker=w;
    w.onerror=e=>{if(worker!==w)return;e.preventDefault();fail('统计数据或计算线程加载失败，请重试。没有使用备用诗稿。');};
    w.onmessage=({data:m})=>{if(worker!==w||!active||m.id!==active.id)return;
      if(m.type==='progress'){status(m.progress.message);return;}if(m.type==='error'){fail(m.message);return;}if(m.type!=='result')return;
      const job=active,v=m.result;active=null;
      if(v?.status!=='complete'){busy(false);status('未找到完整结果。未放宽约束，可换种子重试。');return;}
      if(v.length!==job.options.length||!Array.isArray(v.lines)||v.lines.length!==8||v.lines.some(l=>typeof l!=='string'||Array.from(l).length!==v.length)){fail('结果诗体不匹配。');return;}
      if(!job.replay)histories[v.length]=histories[v.length].concat([v.lines.slice()]).slice(-8);show(v);
    };
  }
  function start(replay){if(active||!ready)return;$('i-error').hidden=true;
    try{let options;
      if(replay){if(!result)return;options=JSON.parse(JSON.stringify(result.options));}
      else{const raw=$('i-seed').value.trim();if(!/^\d{1,10}$/.test(raw)||Number(raw)>4294967295)throw Error('种子必须为0至4294967295的整数');let seed=Number(raw);
        if($('i-auto').checked){const random=new Uint32Array(1);crypto.getRandomValues(random);seed=random[0]===seed?(seed+1)>>>0:random[0];}
        const length=Number($('i-length').value);options={length,mode:$('i-mode').value,seed,variation:0.8,beamWidth:16,recentPoems:$('i-auto').checked&&$('i-history').checked?histories[length].map(p=>p.slice()):[]};}
      $('i-seed').value=String(options.seed);$('i-length').value=String(options.length);$('i-mode').value=options.mode;
      ensureWorker();active={id:++serial,options,replay};busy(true);status('正在加载语料并生成；可随时停止。');worker.postMessage({id:active.id,options});
    }catch(e){fail(e.message);}
  }
  $('i-form').onsubmit=e=>{e.preventDefault();start(false);if(active&&e.submitter?.id==='i-again')mobileScroll($('i-cancel'));};
  $('i-replay').onclick=()=>{start(true);if(active)mobileScroll($('i-cancel'));};
  $('i-auto').onchange=()=>busy(Boolean(active));
  $('i-mode').onchange=()=>{if(!active)status(result?'设置已改变；点击生成才产生新稿。':'设置已改变；点击生成才会出现诗稿。');};
  $('i-cancel').onclick=()=>{stop();status('已停止，没有填入备用诗稿。');};
  $('i-reset-history').onclick=()=>{if(!active){histories[Number($('i-length').value)]=[];status('已清空当前诗体的近期记录。');}};
  $('i-length').onchange=()=>{if(active)return;result=null;$('i-poem').replaceChildren();$('i-summary').textContent='';resetEvidence();status('诗体已切换。点击生成，写一首新诗。');busy(false);};
  $('i-copy').onclick=async()=>{if(!result)return;try{await navigator.clipboard.writeText(result.lines.map((s,i)=>s+(i%2?'。':'，')).join('\n'));status('已复制。');}catch{status('复制权限未开，请选中诗文复制。');}};
  $('i-export').onclick=()=>{if(!result)return;const u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='verseloom-induced-'+result.options.seed+'.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1000);};
  if(meta)$('i-data').textContent=meta.meta.trainingPoems.toLocaleString()+'首训练文本 / '+meta.meta.admittedSegments.toLocaleString()+'个自动准入词段。首次加载约'+(meta.assetBytes/1048576).toFixed(1)+' MiB统计数据。';
  if(!ready)fail('语料元信息或浏览器计算线程不可用。请刷新或使用支持 Worker 的浏览器。');
  addEventListener('pagehide',stop);busy(false);
})();
