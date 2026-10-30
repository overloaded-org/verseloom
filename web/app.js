(function () {
  "use strict";

  function initialize() {
    const byId = (id) => document.getElementById(id);
    const ui = {
      form: byId("generation-form"), rhyme: byId("rhyme-select"), template: byId("template-select"),
      seed: byId("seed-input"), autoSeed: byId("auto-seed"), variation: byId("variation-select"),
      replay: byId("replay-button"), seedDisplay: byId("seed-display"),
      avoidRecent: byId("avoid-recent"), openingDisplay: byId("opening-display"),
      beam: byId("beam-select"), beamSummary: byId("beam-summary"), generate: byId("generate-button"),
      generateLabel: byId("generate-label"), nextRhyme: byId("next-rhyme-button"), cancel: byId("cancel-button"),
      reset: byId("reset-button"), sheet: byId("poem-sheet"), kind: byId("result-kind"), heading: byId("poem-heading"),
      badge: byId("result-badge"), lines: byId("poem-lines"), lineHint: byId("line-hint"),
      rhymeDisplay: byId("rhyme-display"), templateDisplay: byId("template-display"), time: byId("time-display"),
      score: byId("score-display"), tone: byId("tone-toggle"), toneLabel: byId("tone-toggle-label"),
      toneLegend: byId("tone-legend"), copy: byId("copy-button"), download: byId("download-button"),
      progressRegion: byId("progress-region"), progressMessage: byId("progress-message"), progressPercent: byId("progress-percent"),
      progressBar: byId("search-progress"), progressFill: byId("progress-fill"), errorRegion: byId("error-region"),
      errorHeading: byId("error-heading"), errorMessage: byId("error-message"), checks: byId("checks-list"),
      checksSummary: byId("checks-summary"), inspector: byId("inspection-panel"), live: byId("live-status"), toast: byId("toast")
    };
    const data = globalThis.VerseLoomData;
    const engine = globalThis.VerseLoomEngine;
    const ordinal = ["一", "二", "三", "四", "五", "六", "七", "八"];
    let result = null;
    let displayedLines = [];
    let activeRun = null;
    let nextRunId = 0;
    let completedRuns = 0;
    let showTones = false;
    let toastTimer;
    // Only this tab's displayed successes; nothing is uploaded or persisted.
    // Avoid the visible archived opening on the first exploratory click too.
    let recentOpenings = data?.baseline?.lines?.length ? [data.baseline.lines[0]] : [];

    function announce(message) {
      ui.live.textContent = message;
    }

    function notify(message) {
      clearTimeout(toastTimer);
      ui.toast.textContent = message;
      ui.toast.hidden = false;
      toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 2800);
    }

    function displayError(title, message) {
      ui.errorHeading.textContent = title;
      ui.errorMessage.textContent = message;
      ui.errorRegion.hidden = false;
      announce(title + "。" + message);
    }

    function finiteNumber(value, digits) {
      return Number.isFinite(value) ? value.toFixed(digits) : "—";
    }

    function labelForTemplate(id) {
      const template = data.templates.find((entry) => entry.id === id);
      return template ? template.label : id || "—";
    }

    function labelForRhyme(id) {
      const rhyme = data.rhymes.find((entry) => entry.id === id);
      return rhyme ? rhyme.label : id || "—";
    }

    function setBusy(busy) {
      [ui.rhyme, ui.template, ui.beam, ui.generate, ui.nextRhyme, ui.reset, ui.seed, ui.autoSeed, ui.variation, ui.avoidRecent].forEach((element) => { element.disabled = busy; });
      ui.avoidRecent.disabled = busy || !ui.autoSeed.checked || Number(ui.variation.value) === 0;
      ui.cancel.hidden = !busy;
      ui.generate.classList.toggle("is-busy", busy);
      ui.generateLabel.textContent = busy ? "正在成篇…" : !completedRuns ? "开始生成" :
        ui.autoSeed.checked && Number(ui.variation.value) > 0 ? "换个种子，再写一首" : "按当前设置生成";
      ui.replay.disabled = busy || !result;
      ui.sheet.setAttribute("aria-busy", String(busy));
      ui.progressRegion.hidden = !busy;
      ui.download.disabled = busy || !result;
      ui.copy.disabled = busy || !displayedLines.length;
      ui.badge.classList.toggle("is-busy", busy);
      if (busy) ui.badge.textContent = "正在本机计算";
    }

    function restoreBadge() {
      ui.badge.classList.toggle("is-generated", Boolean(result));
      ui.badge.textContent = result ? "本机生成 · 第 " + completedRuns + " 次" : "示例 · 尚未运行";
    }

    function progressFor(job, progress) {
      if (job.cancelled) {
        const error = new Error("Generation cancelled");
        error.name = "AbortError";
        throw error;
      }
      if (activeRun !== job) return;
      const supplied = progress && typeof progress === "object" ? progress : {};
      let percent = Number(supplied.percent);
      if (!Number.isFinite(percent) && supplied.templateCount) {
        percent = 100 * (Number(supplied.templateIndex || 0) + (Number(supplied.row || 0) + 1) / Number(supplied.rowCount || 8)) / Number(supplied.templateCount);
      }
      percent = Number.isFinite(percent) ? Math.min(100, Math.max(0, Math.round(percent))) : 0;
      ui.progressMessage.textContent = supplied.message || "正在比较候选诗句…";
      ui.progressPercent.textContent = percent + "%";
      ui.progressBar.setAttribute("aria-valuenow", String(percent));
      ui.progressFill.style.width = percent + "%";
    }

    function finishJob(job) {
      if (job.worker) { job.worker.terminate(); job.worker = null; }
      if (activeRun !== job) return false;
      activeRun = null;
      setBusy(false);
      return true;
    }

    function selectLine(index) {
      if (!result || !Array.isArray(result.lineDetails)) return;
      const detail = result.lineDetails[index];
      if (!detail) return;
      ui.lines.querySelectorAll(".poem-line").forEach((element, row) => {
        const selected = row === index;
        element.classList.toggle("is-selected", selected);
        element.setAttribute("aria-pressed", String(selected));
      });
      byId("inspection-row").textContent = "第" + ordinal[index] + "句";
      byId("prefix-text").textContent = detail.prefix || "";
      byId("suffix-text").textContent = detail.suffix || "";
      byId("prefix-count").textContent = Number.isFinite(detail.prefixCount) ? detail.prefixCount.toLocaleString("zh-CN") : "—";
      byId("suffix-count").textContent = Number.isFinite(detail.suffixCount) ? detail.suffixCount.toLocaleString("zh-CN") : "—";
      byId("line-lm").textContent = finiteNumber(detail.lmScore, 3);
      const coherenceWeight = result.options && Number.isFinite(result.options.coherenceWeight) ? result.options.coherenceWeight : .5;
      const coherenceTerm = Number.isFinite(detail.coherenceWeighted) ? detail.coherenceWeighted : detail.coherenceRaw * coherenceWeight;
      byId("line-coherence").textContent = Number.isFinite(coherenceTerm) ? "+" + coherenceTerm.toFixed(3) : "—";
      byId("line-repetition").textContent = Number.isFinite(detail.repetitionPenalty) ? "−" + detail.repetitionPenalty.toFixed(3) : "—";
      byId("line-increment").textContent = finiteNumber(detail.increment, 3);
      byId("line-sampling").textContent = finiteNumber(detail.samplingBias ?? 0, 3);
      ui.inspector.hidden = false;
    }

    function renderPoem(value, generated) {
      displayedLines = value.lines.slice();
      const template = data.templates.find((entry) => entry.id === value.templateId);
      const core = value.core || (template && template.core) || [];
      const rhymeRows = value.rhymeRows || (template && template.rhymeRows) || [1, 3, 5, 7];
      const rows = document.createDocumentFragment();
      displayedLines.forEach((line, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "poem-line";
        button.disabled = !generated;
        button.setAttribute("aria-label", "第" + ordinal[index] + "句：" + line + (generated ? "，查看词块来源" : "，存档示例"));
        if (generated) button.setAttribute("aria-pressed", "false");
        const number = document.createElement("span");
        number.className = "line-number";
        number.textContent = ordinal[index];
        number.setAttribute("aria-hidden", "true");
        button.appendChild(number);
        const tones = value.lineDetails && value.lineDetails[index] ? value.lineDetails[index].tones : core[index];
        Array.from(line).forEach((character, column) => {
          const tone = tones ? tones[column] : "";
          const cell = document.createElement("span");
          cell.className = "character" + (tone ? " tone-" + tone : "") + (column === 4 && rhymeRows.includes(index) ? " is-rhyme" : "");
          const glyph = document.createElement("span");
          glyph.className = "character-glyph";
          glyph.textContent = character;
          const toneLabel = document.createElement("span");
          toneLabel.className = "character-tone";
          toneLabel.textContent = tone === "P" ? "平" : tone === "Z" ? "仄" : "—";
          toneLabel.setAttribute("aria-hidden", "true");
          cell.append(glyph, toneLabel);
          button.appendChild(cell);
        });
        const punctuation = document.createElement("span");
        punctuation.className = "line-punctuation";
        punctuation.textContent = index % 2 === 0 ? "，" : "。";
        punctuation.setAttribute("aria-hidden", "true");
        button.appendChild(punctuation);
        button.addEventListener("click", () => selectLine(index));
        rows.appendChild(button);
      });
      ui.lines.replaceChildren(rows);
      ui.lines.classList.toggle("show-tones", showTones);
      ui.openingDisplay.textContent = generated ? value.opening?.policy === "prefix-balanced-softmax-v1" ?
        "词块分组抽样" + (value.options.openingHistory.length ? " · 近期去重" : "") : "原版择优" : "尚未运行";
      ui.kind.textContent = generated ? "本次生成" : "首轮存档";
      ui.heading.textContent = generated ? "五律 · 新稿" : "五律 · 初试";
      ui.rhymeDisplay.textContent = value.rhymeLabel || labelForRhyme(value.rhymeGroup);
      ui.templateDisplay.textContent = value.templateLabel || labelForTemplate(value.templateId);
      ui.time.textContent = generated && Number.isFinite(value.elapsedMs) ? value.elapsedMs < 1000 ? Math.round(value.elapsedMs) + " ms" : (value.elapsedMs / 1000).toFixed(2) + " s" : "尚未运行";
      ui.score.textContent = generated ? finiteNumber(value.score, 3) : "—";
      ui.seedDisplay.textContent = generated ? String(value.options.seed) +
        (value.options.variation === 0 ? "（随机关闭）" : " · 强度 " + value.options.variation) : "尚未运行";
      ui.lineHint.textContent = generated ? "点选一句，查看词块来源与分数" : "生成后，点一句查看词块来源";
      ui.download.disabled = !generated;
      ui.copy.disabled = !displayedLines.length;
      if (!generated) ui.inspector.hidden = true;
    }

    function renderChecks(checks) {
      const list = Array.isArray(checks) ? checks : [];
      const fragment = document.createDocumentFragment();
      list.forEach((check) => {
        const item = document.createElement("li");
        item.className = check.passed ? "check-pass" : "check-fail";
        if (check.note) item.title = check.note;
        const mark = document.createElement("span");
        mark.textContent = check.passed ? "✓" : "!";
        mark.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.textContent = (check.label || check.id) + (check.passed ? "" : " · 待复核");
        item.append(mark, label);
        fragment.appendChild(item);
      });
      ui.checks.replaceChildren(fragment);
      const passed = list.filter((check) => check.passed).length;
      ui.checksSummary.textContent = list.length ? passed + " / " + list.length + " 项通过" : "本次无核验记录";
    }

    function receiveResult(job, value) {
      if (activeRun !== job || job.cancelled) return;
      if (!value || value.status !== "complete") {
        finishJob(job);
        restoreBadge();
        displayError("当前设置没有找到完整诗稿", "当前首句与格律约束下未找到完整路径。请换韵部／起式、增加搜索宽度，或取消近期首句去重。没有偷偷放宽约束，下方仍保留上一份诗稿。");
        return;
      }
      if (!Array.isArray(value.lines) || value.lines.length !== 8 || value.lines.some((line) => typeof line !== "string" || Array.from(line).length !== 5)) {
        receiveError(job, new Error("生成结果未通过八句五言结构检查。"));
        return;
      }
      const repeatedOutput = result && result.lines.join("\n") === value.lines.join("\n");
      finishJob(job);
      completedRuns += 1;
      result = value;
      if (!job.replay) recentOpenings = recentOpenings.concat(value.lines[0]).slice(-8);
      renderPoem(result, true);
      renderChecks(result.checks);
      restoreBadge();
      setBusy(false);
      selectLine(0);
      if (repeatedOutput && !job.replay) notify("本次搜索仍选中相同诗稿；换种子不保证唯一，可提高探索强度。");
      announce("生成完成，" + (result.rhymeLabel || labelForRhyme(result.rhymeGroup)) + "，八句五言。" + ui.checksSummary.textContent + "。");
    }

    function receiveError(job, error) {
      if (job.cancelled || activeRun !== job || (error && error.name === "AbortError")) return;
      finishJob(job);
      restoreBadge();
      const message = error && error.message ? String(error.message) : "请稍后重试，或恢复默认设置。";
      displayError("本次搜索未完成", message);
    }

    function runOnMainThread(job) {
      if (job.cancelled || activeRun !== job || job.mainStarted) return;
      job.mainStarted = true;
      if (job.worker) { job.worker.terminate(); job.worker = null; }
      Promise.resolve().then(() => engine.generate(data, job.options, (progress) => progressFor(job, progress)))
        .then((value) => receiveResult(job, value)).catch((error) => receiveError(job, error));
    }

    function readSeed() {
      const text = ui.seed.value.trim();
      if (!/^\d{1,10}$/.test(text) || Number(text) > 4294967295) {
        throw new Error("种子须为 0 至 4294967295 的整数。");
      }
      return Number(text);
    }

    function freshSeed(previous) {
      // Only choose the visible run seed here. The engine itself is deterministic.
      if (globalThis.crypto && typeof globalThis.crypto.getRandomValues === "function") {
        const value = globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
        return value === previous ? (previous + 1) >>> 0 : value;
      }
      // Offline/older environments can still explore with a recorded new seed.
      return (previous + 1) >>> 0;
    }

    function currentOptions() {
      const defaults = data.defaults || {};
      return {
        seed: readSeed(),
        variation: Number(ui.variation.value),
        rhymeGroup: ui.rhyme.value,
        templateId: ui.template.value,
        beamWidth: Number(ui.beam.value),
        coherenceWeight: defaults.coherenceWeight ?? defaults.coherence_weight ?? .5,
        repetitionWeight: defaults.repetitionWeight ?? defaults.repetition_weight ?? .8
      };
    }

    function beginGeneration(replayOptions = null) {
      if (activeRun) return;
      let options;
      try {
        options = replayOptions ? {...replayOptions} : currentOptions();
        if (!replayOptions) {
          options.openingHistory = [];
          if (ui.autoSeed.checked && options.variation > 0) {
            options.seed = freshSeed(options.seed);
            if (ui.avoidRecent.checked) options.openingHistory = recentOpenings.slice();
          }
        } else {
          options.openingHistory = (replayOptions.openingHistory || []).slice();
        }
      } catch (error) {
        displayError("生成参数无效", error.message);
        return;
      }
      ui.seed.value = String(options.seed);
      if (replayOptions) {
        ui.rhyme.value = options.rhymeGroup;
        ui.template.value = options.templateId;
        ui.beam.value = String(options.beamWidth);
        ui.variation.value = String(options.variation);
        ui.beamSummary.textContent = "宽度 " + ui.beam.value;
      }
      const job = { id: ++nextRunId, options, replay: Boolean(replayOptions), worker: null, cancelled: false, mainStarted: false };
      activeRun = job;
      ui.errorRegion.hidden = true;
      setBusy(true);
      progressFor(job, { percent: 0, message: "正在准备候选诗句…" });
      announce("正在浏览器中搜索诗句，请稍候。可以随时停止计算。");
      if (!globalThis.Worker || location.protocol === "file:" || globalThis.VERSELOOM_STANDALONE === true) {
        runOnMainThread(job);
        return;
      }
      try {
        const worker = new Worker(new URL("worker.js", document.baseURI));
        job.worker = worker;
        worker.onmessage = (event) => {
          const message = event.data;
          if (!message || message.id !== job.id || activeRun !== job || job.cancelled) return;
          if (message.type === "progress") progressFor(job, message.progress);
          else if (message.type === "result") receiveResult(job, message.result);
          else if (message.type === "error") receiveError(job, new Error(message.message || "生成任务未完成。"));
        };
        worker.onerror = (event) => {
          event.preventDefault();
          runOnMainThread(job);
        };
        worker.postMessage({ id: job.id, options: job.options });
      } catch (error) {
        runOnMainThread(job);
      }
    }

    function cancelGeneration() {
      const job = activeRun;
      if (!job) return;
      job.cancelled = true;
      finishJob(job);
      restoreBadge();
      notify("已停止本次计算，保留上一份诗稿");
      announce("本次计算已停止。");
    }

    function restoreDefaults() {
      if (activeRun) return;
      ui.rhyme.value = "auto";
      ui.template.value = "auto";
      ui.beam.value = String(data.defaults?.beamWidth ?? data.defaults?.beam_width ?? 32);
      ui.beamSummary.textContent = "宽度 " + ui.beam.value;
      ui.errorRegion.hidden = true;
      ui.seed.value = "0";
      ui.variation.value = "0";
      ui.autoSeed.checked = false;
      recentOpenings = [];
      setBusy(false);
      notify("已恢复原始配置并关闭随机，点击生成可复现首作");
    }

    function nextRhyme() {
      if (activeRun) return;
      const feasible = data.rhymes.filter((entry) => entry.feasible);
      if (!feasible.length) { displayError("暂时没有可用韵部", "请重新加载完整的网页数据。"); return; }
      const current = ui.rhyme.value !== "auto" ? ui.rhyme.value : result?.rhymeGroup || data.baseline.rhymeGroup;
      const currentIndex = feasible.findIndex((entry) => entry.id === current);
      ui.rhyme.value = feasible[(currentIndex + 1) % feasible.length].id;
      beginGeneration();
    }

    function formattedPoem() {
      const couplets = [];
      for (let index = 0; index < displayedLines.length; index += 2) {
        couplets.push(displayedLines[index] + "，" + displayedLines[index + 1] + "。");
      }
      return couplets.join("\n");
    }

    async function copyPoem() {
      if (!displayedLines.length) return;
      const text = formattedPoem();
      try {
        if (!navigator.clipboard || !globalThis.isSecureContext) throw new Error("Clipboard fallback");
        await navigator.clipboard.writeText(text);
      } catch (error) {
        const previouslyFocused = document.activeElement;
        const temporary = document.createElement("textarea");
        temporary.value = text;
        temporary.setAttribute("readonly", "");
        temporary.style.cssText = "position:fixed;opacity:0;left:-9999px;top:0";
        document.body.appendChild(temporary);
        temporary.select();
        let copied = false;
        try { copied = document.execCommand("copy"); } catch (clipboardError) { copied = false; }
        temporary.remove();
        if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
        if (!copied) { notify("浏览器未允许复制，请直接选中诗文复制"); return; }
      }
      notify(result ? "诗文已复制" : "首轮存档示例已复制");
    }

    function downloadResult() {
      if (!result || activeRun) return;
      const exported = {
        ...result,
        export: {
          application: "VerseLoom · 律织网页演示",
          exportedAt: new Date().toISOString(),
          note: "Browser search over a frozen, pre-scored candidate pool; not a new neural training run or literary-quality certification."
        }
      };
      const blob = new Blob([JSON.stringify(exported, null, 2) + "\n"], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "verseloom-seed-" + result.options.seed + "-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify("已下载本次结果与参数");
    }

    if (!data || !engine || typeof engine.generate !== "function" || !Array.isArray(data.templates) || !Array.isArray(data.rhymes) || !data.baseline) {
      [ui.generate, ui.nextRhyme, ui.reset, ui.rhyme, ui.template, ui.beam, ui.copy, ui.tone, ui.seed, ui.autoSeed, ui.variation, ui.replay, ui.avoidRecent].forEach((element) => { element.disabled = true; });
      displayError("生成资源尚未加载", "请刷新重试。离线使用时，请使用完整网页包或单文件版本；当前诗稿仍是首轮存档示例。");
      return;
    }

    data.rhymes.filter((entry) => entry.feasible).forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      ui.rhyme.appendChild(option);
    });
    data.templates.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      ui.template.appendChild(option);
    });
    if (data.meta) {
      [["training-count", "trainingPoems"], ["candidate-count", "candidateCount"], ["template-count", "templateCount"]].forEach(([id, key]) => {
        if (Number.isFinite(data.meta[key])) byId(id).textContent = data.meta[key].toLocaleString("zh-CN");
      });
    }
    ui.beam.value = String(data.defaults?.beamWidth ?? data.defaults?.beam_width ?? 32);
    ui.beamSummary.textContent = "宽度 " + ui.beam.value;
    renderPoem(data.baseline, false);
    setBusy(false);
    ui.tone.disabled = false;
    ui.form.addEventListener("submit", (event) => { event.preventDefault(); beginGeneration(); });
    ui.nextRhyme.addEventListener("click", nextRhyme);
    ui.replay.addEventListener("click", () => { if (result) beginGeneration(result.options); });
    ui.autoSeed.addEventListener("change", () => setBusy(Boolean(activeRun)));
    ui.variation.addEventListener("change", () => setBusy(Boolean(activeRun)));
    ui.cancel.addEventListener("click", cancelGeneration);
    ui.reset.addEventListener("click", restoreDefaults);
    ui.beam.addEventListener("change", () => { ui.beamSummary.textContent = "宽度 " + ui.beam.value; });
    ui.tone.addEventListener("click", () => {
      showTones = !showTones;
      ui.lines.classList.toggle("show-tones", showTones);
      ui.tone.setAttribute("aria-pressed", String(showTones));
      ui.toneLabel.textContent = showTones ? "隐藏平仄" : "显示平仄";
      ui.toneLegend.hidden = !showTones;
    });
    ui.copy.addEventListener("click", copyPoem);
    ui.download.addEventListener("click", downloadResult);
    window.addEventListener("pagehide", () => {
      if (activeRun) { activeRun.cancelled = true; if (activeRun.worker) activeRun.worker.terminate(); }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
}());
