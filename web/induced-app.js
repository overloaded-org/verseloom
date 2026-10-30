/* Session history affects search preferences, never supplies required content. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  let worker = null, active = null, result = null, serial = 0;
  const histories = {5: [], 7: []};
  function status(text) { $('i-status').textContent = text; }
  function busy(value) {
    for (const id of ['i-length', 'i-mode', 'i-seed', 'i-auto', 'i-generate', 'i-history', 'i-reset-history']) $(id).disabled = value;
    $('i-history').disabled = value || !$('i-auto').checked;
    $('i-cancel').hidden = !value;
    $('i-sheet').setAttribute('aria-busy', String(value));
    for (const id of ['i-replay', 'i-copy', 'i-export']) $(id).disabled = value || !result;
  }
  function stop() { worker?.terminate(); worker = null; active = null; busy(false); }
  function fail(message) { stop(); $('i-error').textContent = message; $('i-error').hidden = false; }
  function show(value) {
    result = value; $('i-poem').replaceChildren(); $('i-inspection').hidden = true;
    for (const [index, text] of value.lines.entries()) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'poem-line';
      button.textContent = text + (index % 2 ? '。' : '，');
      button.setAttribute('aria-label', '第' + (index + 1) + '句，查看来源与算法依据');
      button.onclick = () => {
        $('i-inspection').hidden = false;
        $('i-detail').textContent = JSON.stringify(value.lineDetails[index], null, 2);
      };
      $('i-poem').appendChild(button);
    }
    $('i-summary').textContent = value.templateId + ' / ' + value.rhymeGroup + ' / 种子 ' + value.options.seed + ' / ' + ({local:'句内对照',relations:'固定关系评分',learned:'学得接续评分'}[value.options.mode]||value.options.mode) + ' / ' + value.engineVersion;
    status('生成完成。未人工改字；点击任一句查看统计依据。'); busy(false);
    if (globalThis.matchMedia?.('(max-width: 760px)')?.matches)
      $('i-sheet').scrollIntoView?.({block: 'start', behavior: 'auto'});
  }
  function ensureWorker() {
    if (worker) return;
    worker = new Worker(new URL('induced-worker.js', document.baseURI));
    worker.onerror = event => { event.preventDefault(); fail('统计数据或计算线程加载失败。没有使用备用诗稿。'); };
    worker.onmessage = ({data: message}) => {
      if (!active || message.id !== active.id) return;
      if (message.type === 'progress') status(message.progress.message);
      else if (message.type === 'error') fail(message.message);
      else if (message.type === 'result') {
        const job = active, value = message.result;
        if (value.status === 'complete' && (value.length !== job.options.length || value.lines.length !== 8 || value.lines.some(line => Array.from(line).length !== value.length))) {
          fail('结果诗体不匹配。'); return;
        }
        active = null;
        if (value.status === 'complete') {
          if (!job.replay) histories[value.length] = histories[value.length].concat([value.lines.slice()]).slice(-8);
          show(value);
        } else { busy(false); status('未找到完整结果。没有放宽声律约束或填入存档；可换种子重试。'); }
      }
    };
  }
  function start(replay) {
    if (active) return;
    $('i-error').hidden = true;
    try {
      let options;
      if (replay) {
        if (!result) return;
        options = JSON.parse(JSON.stringify(result.options));
      } else {
        const raw = $('i-seed').value.trim();
        if (!/^\d{1,10}$/.test(raw) || Number(raw) > 4294967295) throw new Error('种子必须为0至4294967295的整数');
        let seed = Number(raw);
        if ($('i-auto').checked) {
          const random = new Uint32Array(1); crypto.getRandomValues(random);
          seed = random[0] === seed ? (seed + 1) >>> 0 : random[0];
        }
        const length = Number($('i-length').value);
        options = {length, mode: $('i-mode').value, seed, variation: 0.8, beamWidth: 16,
          recentPoems: $('i-auto').checked && $('i-history').checked ? histories[length].map(poem => poem.slice()) : []};
      }
      if (typeof Worker === 'undefined') throw new Error('这个实验需要浏览器 Worker；未启用时不模拟生成。');
      $('i-seed').value = String(options.seed); $('i-length').value = String(options.length); $('i-mode').value = options.mode;
      ensureWorker(); active = {id: ++serial, options, replay}; busy(true);
      status('加载训练统计并构造候选；可随时停止。'); worker.postMessage({id: active.id, options});
    } catch (error) { fail(error.message); }
  }
  $('i-form').onsubmit = event => { event.preventDefault(); start(false); };
  $('i-replay').onclick = () => start(true);
  $('i-auto').onchange = () => busy(Boolean(active));
  $('i-mode').onchange = () => { if (!active) status(result ? '模式设置已改变；当前诗稿仍是上一次结果，点击生成才运行新模式。' : '模式设置已改变；点击生成后才会出现算法输出。'); };
  $('i-cancel').onclick = () => { stop(); status('已停止；没有填入备用结果。'); };
  $('i-reset-history').onclick = () => {
    if (active) return;
    histories[Number($('i-length').value)] = []; status('已清空当前诗体的近期记录；没有指定任何词句。');
  };
  $('i-length').onchange = () => {
    if (active) return;
    result = null; $('i-poem').replaceChildren(); $('i-summary').textContent = ''; $('i-inspection').hidden = true;
    status('切换诗体后保持空白；五言与七言的近期记录分别保存。'); busy(false);
  };
  $('i-copy').onclick = async () => {
    if (!result) return;
    try { await navigator.clipboard.writeText(result.lines.map((line, i) => line + (i % 2 ? '。' : '，')).join('\n')); status('已复制。'); }
    catch { status('复制权限未开启，请选中诗文复制。'); }
  };
  $('i-export').onclick = () => {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'verseloom-induced-' + result.options.seed + '.json';
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const meta = globalThis.VerseLoomInducedMeta;
  if (meta) $('i-data').textContent = meta.meta.trainingPoems.toLocaleString() + '首训练文本 / ' + meta.meta.admittedSegments.toLocaleString() + '个自动准入词段。首次加载约' + (meta.assetBytes / 1048576).toFixed(1) + ' MiB统计数据。';
  else fail('来源元数据没有加载。');
  addEventListener('pagehide', stop); busy(false);
})();
