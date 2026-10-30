/* VerseLoom browser search. No model API, training, or prewritten output selection. */
(function (global) {
  "use strict";

  const ENGINE_VERSION = "verseloom.joint-lines.v4"; // Same unanchored search, now length-aware.
  const prepared = new WeakMap();
  const allowedOptions = new Set(["rhymeGroup", "templateId", "beamWidth", "coherenceWeight", "repetitionWeight", "seed", "variation", "openingHistory", "recentPoems"]);
  const now = () => global.performance ? global.performance.now() : Date.now();
  const yieldTurn = () => new Promise(resolve => setTimeout(resolve, 0));

  function requireValue(condition, message) {
    if (!condition) throw new Error(message);
  }

  function choiceNoise(seed, templateId, row, text) {
    let hash = (2166136261 ^ seed) >>> 0;
    for (const char of templateId + "/" + row + "/" + text) {
      hash = Math.imul(hash ^ char.codePointAt(0), 16777619) >>> 0;
    }
    hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d) >>> 0;
    hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b) >>> 0;
    hash = (hash ^ (hash >>> 16)) >>> 0;
    return 2 * ((hash + 0.5) / 4294967296) - 1;
  }

  function lineLength(data) {
    const length = data.meta.lineLength ?? 5;
    requireValue(length === 5 || length === 7, "仅支持五言与七言数据。");
    return length;
  }

  // Compare code points, matching Python string order rather than locale order.
  function compareText(left, right) {
    const a = Array.from(left), b = Array.from(right);
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const difference = a[i].codePointAt(0) - b[i].codePointAt(0);
      if (difference) return difference;
    }
    return a.length - b.length;
  }

  function compareLines(left, right) {
    for (let i = 0; i < Math.min(left.length, right.length); i++) {
      const difference = compareText(left[i], right[i]);
      if (difference) return difference;
    }
    return left.length - right.length;
  }

  // Compensated float summation follows Python 3.12's higher-accuracy sum.
  function preciseSum(values) {
    let high = 0, low = 0;
    for (const value of values) {
      const total = high + value;
      low += Math.abs(high) >= Math.abs(value) ? (high - total) + value : (value - total) + high;
      high = total;
    }
    return high + low;
  }

  function prepare(data) {
    if (prepared.has(data)) return prepared.get(data);
    requireValue(data && data.schema === "verseloom.web.v1", "浏览器数据版本不受支持。");
    const length = lineLength(data), prefixLength = length - 3;
    const patterns = length === 5 ? ["PPZZP", "PPPZZ", "ZZZPP", "ZZPPZ"] : ["PPZZZPP", "PPZZPPZ", "ZZPPZZP", "ZZPPPZZ"];
    const characters = Array.from(data.coherence.characters);
    const charIds = new Map(characters.map((char, index) => [char, index]));
    requireValue(charIds.size === characters.length, "字符表包含重复项。");
    requireValue(data.coherence.encoding === "float64-le-lower-triangle-base64", "共现数据编码不受支持。");
    const binary = global.atob(data.coherence.values);
    const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
    const valueCount = characters.length * (characters.length - 1) / 2;
    requireValue(bytes.length === valueCount * 8, "共现数据长度不正确。");
    const view = new DataView(bytes.buffer);
    const values = new Float64Array(valueCount);
    for (let i = 0; i < values.length; i++) {
      values[i] = view.getFloat64(i * 8, true);
      requireValue(Number.isFinite(values[i]) && values[i] >= 0, "共现数据包含无效数值。");
    }
    const pools = Object.create(null);
    let candidateCount = 0;
    for (const [key, rows] of Object.entries(data.pools)) {
      const slash = key.indexOf("/");
      const tones = key.slice(0, slash), group = key.slice(slash + 1);
      requireValue(patterns.includes(tones), "候选平仄格式不正确。");
      pools[key] = rows.map(row => {
        const chars = Array.from(row[0]);
        requireValue(chars.length === length && new Set(chars).size === length && Number.isFinite(row[1]), "候选句格式不正确。");
        requireValue(chars.every(char => charIds.has(char)) && row[2] >= (data.meta.minimumSpanCount ?? 2) && row[3] >= (data.meta.minimumSpanCount ?? 2), "候选来源不完整。");
        return {id: candidateCount++, text: row[0], chars, charIds: chars.map(char => charIds.get(char)),
          tones, rhymeGroup: group === "nonrhyming" ? null : group, lmScore: row[1],
          prefix: chars.slice(0, prefixLength).join(""), suffix: chars.slice(prefixLength).join(""), prefixCount: row[2], suffixCount: row[3]};
      });
    }
    requireValue(candidateCount === data.meta.candidateCount, "候选数量与来源记录不符。");
    const result = {pools, values, candidateCount};
    prepared.set(data, result);
    return result;
  }

  function normalizeOptions(data, options) {
    requireValue(options && typeof options === "object" && !Array.isArray(options), "生成参数应为对象。");
    requireValue(Object.keys(options).every(key => allowedOptions.has(key)), "含有不受支持的生成参数。");
    const length = lineLength(data);
    const result = Object.assign({seed: 0, variation: 0, openingHistory: [], recentPoems: []}, data.defaults, options);
    requireValue(Array.isArray(result.openingHistory) && result.openingHistory.length <= 8 &&
      Array.from(result.openingHistory).every(line => typeof line === "string" && Array.from(line).length === length),
      "首句历史须为最多 8 个与当前句长一致的句子。");
    result.openingHistory = result.openingHistory.slice();
    requireValue(Array.isArray(result.recentPoems) && result.recentPoems.length <= 8 &&
      Array.from(result.recentPoems).every(poem => Array.isArray(poem) && poem.length === 8 &&
        Array.from(poem).every(line => typeof line === "string" && Array.from(line).length === length)),
      "近期诗稿须为最多8首相同诗体的完整八句。");
    result.recentPoems = result.recentPoems.map(poem => poem.slice());
    requireValue(Number.isInteger(result.seed) && result.seed >= 0 && result.seed <= 4294967295,
      "种子须为 0 至 4294967295 的整数。");
    requireValue(typeof result.variation === "number" && Number.isFinite(result.variation) && result.variation >= 0 && result.variation <= 2,
      "探索强度须为 0 至 2 之间的有限数值。");
    requireValue(Number.isInteger(result.beamWidth) && result.beamWidth >= 1 && result.beamWidth <= 128, "搜索宽度须为 1 至 128 的整数。");
    for (const name of ["coherenceWeight", "repetitionWeight"]) {
      requireValue(typeof result[name] === "number" && Number.isFinite(result[name]) && result[name] >= 0 && result[name] <= 2,
        "评分权重须为 0 至 2 之间的有限数值。");
    }
    requireValue(result.templateId === "auto" || data.templates.some(row => row.id === result.templateId), "未知的当前句长模板。");
    requireValue(result.rhymeGroup === "auto" || data.rhymes.some(row => row.id === result.rhymeGroup && row.feasible), "请选择可用韵部。");
    return result;
  }

  const compareStates = (a, b) => b.selectionScore - a.selectionScore || compareLines(a.lines, b.lines);
  // Group by first TWO characters for both forms, not by the seven-character 4+3 cut.
  const firstPrefix = state => state.selected[0].chars.slice(0, 2).join("");

  function retainBeam(extensions, options) {
    extensions.sort(compareStates);
    if (options.variation === 0) return extensions.slice(0, options.beamWidth);
    const groups = new Map();
    for (const state of extensions) {
      const key = firstPrefix(state);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(state);
    }
    const buckets = [...groups.values()], kept = [];
    for (let depth = 0; kept.length < options.beamWidth; depth++) {
      let found = false;
      for (const bucket of buckets) {
        if (depth >= bucket.length) continue;
        found = true;
        kept.push(bucket[depth]);
        if (kept.length === options.beamWidth) break;
      }
      if (!found) break;
    }
    return kept.sort(compareStates);
  }

  // Choose only after all eight lines exist. No first/middle/final line anchors.
  function chooseComplete(outcomes, options) {
    const complete = outcomes.filter(row => row.status === "complete");
    if (!complete.length) return {selected: null, receipt: {stage: "after-eight-lines", candidates: 0, prefixes: 0}};
    const candidates = complete.flatMap(outcome => outcome.finalists.map(state => ({outcome, state})))
      .sort((a, b) => compareText(a.outcome.templateId, b.outcome.templateId) || compareLines(a.state.lines, b.state.lines));
    const groups = new Map();
    for (const candidate of candidates) {
      const key = firstPrefix(candidate.state);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(candidate);
    }
    let chosen;
    if (options.variation === 0) {
      const original = complete.slice().sort((a, b) => b.best.score - a.best.score ||
        compareText(a.templateId, b.templateId) || compareLines(a.best.lines, b.best.lines))[0];
      chosen = {outcome: original, state: original.best};
    } else {
      function draw(items, scores, key) {
        const top = Math.max(...scores);
        const weights = scores.map(score => Math.exp((score - top) / options.variation));
        let ticket = (choiceNoise(options.seed, "complete-poem", 8, key) + 1) / 2 * preciseSum(weights);
        for (let i = 0; i < items.length; i++) {
          ticket -= weights[i];
          if (ticket < 0) return items[i];
        }
        return items[items.length - 1];
      }
      const entries = [...groups.entries()].sort((a, b) => compareText(a[0], b[0]));
      const group = draw(entries, entries.map(([, rows]) => Math.max(...rows.map(row => row.state.selectionScore / 8))), "group");
      chosen = draw(group[1], group[1].map(row => row.state.selectionScore / 8), "path/" + group[0]);
    }
    return {selected: {...chosen.outcome, best: chosen.state}, receipt: {
      stage: "after-eight-lines", policy: options.variation > 0 ? "completed-path-prefix-balanced-softmax" : "score-only",
      candidates: candidates.length, prefixes: groups.size, preselectedLines: false,
      temperature: options.variation, scoreScale: "mean-per-line",
    }};
  }

  function checkResult(selected, rhymeGroup, length) {
    const lines = selected.best.lines, details = selected.best.selected;
    const counts = new Map();
    for (const line of lines) for (const char of line) counts.set(char, (counts.get(char) || 0) + 1);
    const endings = selected.rhymeRows.map(row => details[row].chars[length - 1]);
    const middle = [3, 5].every(row => details[row].chars.every((char, col) => char !== details[row - 1].chars[col]));
    return [
      {id: "shape", label: length === 7 ? "八句七言" : "八句五言", passed: lines.length === 8 && lines.every(line => Array.from(line).length === length)},
      {id: "tones", label: "基本平仄模板", passed: details.every((line, row) => line.tones === selected.core[row]), note: "依据韵表候选读音；未确认语境读音。"},
      {id: "rhyme", label: "同部平声押韵", passed: selected.rhymeRows.every(row => details[row].rhymeGroup === rhymeGroup)},
      {id: "distinct_rhymes", label: "韵脚不重", passed: new Set(endings).size === endings.length},
      {id: "line_repeats", label: "句内不重字", passed: details.every(line => new Set(line.chars).size === length)},
      {id: "character_limit", label: "全诗每字至多两次", passed: [...counts.values()].every(count => count <= 2)},
      {id: "middle_positions", label: "中间两联对应位置异字", passed: middle, note: "这项机械约束不等于对仗认证。"},
      {id: "candidate_source", label: "使用已过滤候选", passed: true, note: "候选池已排除对应句长训练集完整原句；不代表全诗库原创性认证。"},
    ];
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

  async function generate(data, inputOptions = {}, onProgress = () => {}) {
    const started = now();
    const runtime = prepare(data);
    const length = lineLength(data);
    const options = normalizeOptions(data, inputOptions);
    requireValue(typeof onProgress === "function", "进度回调无效。");
    const feasible = data.rhymes.filter(row => row.feasible);
    const exploringRhyme = options.rhymeGroup === "auto" && options.variation > 0;
    const rhyme = options.rhymeGroup !== "auto" ? data.rhymes.find(row => row.id === options.rhymeGroup) :
      exploringRhyme ? feasible.slice().sort((a, b) =>
        choiceNoise(options.seed, "auto-rhyme", length, b.id) - choiceNoise(options.seed, "auto-rhyme", length, a.id) || compareText(a.id, b.id))[0] : feasible[0];
    const recency = recentSpanCounts(options.recentPoems);
    requireValue(rhyme, "没有可用韵部。");
    const templates = data.templates.filter(row => options.templateId === "auto" || row.id === options.templateId)
      .slice().sort((a, b) => compareText(a.id, b.id));
    const pairCache = new Map();
    const pair = (left, right) => {
      const key = left.id * runtime.candidateCount + right.id;
      if (pairCache.has(key)) return pairCache.get(key);
      const terms = [];
      for (const a of left.charIds) for (const b of right.charIds) {
        if (a === b) continue;
        const low = Math.min(a, b), high = Math.max(a, b);
        terms.push(runtime.values[high * (high - 1) / 2 + low]);
      }
      const value = terms.length ? preciseSum(terms) / terms.length : 0;
      if (pairCache.size >= 200000) pairCache.delete(pairCache.keys().next().value);
      pairCache.set(key, value);
      return value;
    };
    const blocked = new Set(options.openingHistory);
    const blockedPrefixes = new Set(options.openingHistory.slice(-4).map(line => Array.from(line).slice(0, 2).join("")));
    const outcomes = [];
    let candidatesConsidered = 0;
    for (let templateIndex = 0; templateIndex < templates.length; templateIndex++) {
      const template = templates[templateIndex], rowReceipts = [];
      let states = [{lines: [], selected: [], score: 0, selectionScore: 0, terms: []}];
      for (let row = 0; row < 8; row++) {
        const rhymeRow = template.rhymeRows.includes(row);
        const originalPool = runtime.pools[template.core[row] + "/" + (rhymeRow ? rhyme.id : "nonrhyming")] || [];
        const pool = row === 0 ? originalPool.filter(line => !blocked.has(line.text) && !blockedPrefixes.has(line.chars.slice(0, 2).join(""))) : originalPool;
        const biases = new Map(pool.map(candidate => [candidate.text, options.variation === 0 ? 0 :
          options.variation * choiceNoise(options.seed, template.id, row, candidate.text)]));
        const penalties = new Map(pool.map(candidate => [candidate.text,
          options.variation > 0 ? 0.5 * spanPenalty(candidate.text, recency) : 0]));
        const counts = {attempted_extensions: 0}, extensions = [];
        const reject = reason => { counts[reason] = (counts[reason] || 0) + 1; };
        for (const state of states) {
          const used = new Map();
          for (const line of state.selected) for (const char of line.chars) used.set(char, (used.get(char) || 0) + 1);
          for (const candidate of pool) {
            counts.attempted_extensions++; candidatesConsidered++;
            if (state.lines.includes(candidate.text)) { reject("repeated_line"); continue; }
            if (candidate.chars.some(char => (used.get(char) || 0) >= 2)) { reject("character_use_limit"); continue; }
            if (rhymeRow && template.rhymeRows.some(previous => previous < row && state.selected[previous].chars[length - 1] === candidate.chars[length - 1])) {
              reject("repeated_rhyme_character"); continue;
            }
            if ((row === 3 || row === 5) && candidate.chars.some((char, col) => char === state.selected[row - 1].chars[col])) {
              reject("middle_couplet_same_position_character"); continue;
            }
            const coherenceRaw = state.selected.length ? preciseSum(state.selected.map(old => pair(old, candidate))) / state.selected.length : 0;
            const repeated = candidate.chars.reduce((sum, char) => sum + (used.get(char) || 0), 0);
            const coherenceWeighted = options.coherenceWeight * coherenceRaw;
            const repetitionPenalty = options.repetitionWeight * repeated;
            const increment = candidate.lmScore + coherenceWeighted - repetitionPenalty;
            const samplingBias = biases.get(candidate.text);
            const score = state.score + increment;
            const historyPenalty = penalties.get(candidate.text);
            const selectionScore = state.selectionScore + increment + samplingBias - historyPenalty;
            requireValue(Number.isFinite(score) && Number.isFinite(selectionScore), "累计分数溢出。");
            extensions.push({lines: state.lines.concat(candidate.text), selected: state.selected.concat(candidate),
              score, selectionScore, terms: state.terms.concat({coherenceRaw, coherenceWeighted, repetitionPenalty,
                previousCharacterOccurrences: repeated, increment, samplingBias, historyPenalty})});
          }
        }
        counts.legal_extensions = extensions.length;
        counts.beam_pruned = Math.max(0, extensions.length - options.beamWidth);
        states = retainBeam(extensions, options);
        counts.retained_states = states.length;
        rowReceipts.push({row, originalCandidatePoolSize: originalPool.length, candidatePoolSize: pool.length, counts,
          distinctOpenings: new Set(states.map(state => state.lines[0])).size,
          distinctOpeningPrefixes: new Set(states.map(firstPrefix)).size});
        onProgress({templateId: template.id, templateIndex, templateCount: templates.length, row, rowCount: 8,
          percent: Math.round((templateIndex * 8 + row + 1) / (templates.length * 8) * 100),
          message: template.label + " · 正在组织第 " + (row + 1) + " 句"});
        await yieldTurn();
        if (!states.length) break;
      }
      outcomes.push({templateId: template.id, label: template.label, core: template.core, rhymeRows: template.rhymeRows,
        status: states.length && states[0].lines.length === 8 ? "complete" : "infeasible",
        best: states.length && states[0].lines.length === 8 ? states[0] : null,
        finalists: states.length && states[0].lines.length === 8 ? states : [], rows: rowReceipts});
    }
    const {selected, receipt: selection} = chooseComplete(outcomes, options);
    const result = {schema: "verseloom.web.result.v1", engineVersion: ENGINE_VERSION, lineLength: length, status: selected ? "complete" : "infeasible",
      lines: selected ? selected.best.lines : [], templateId: selected ? selected.templateId : null,
      templateLabel: selected ? selected.label : null, rhymeGroup: rhyme.id, rhymeLabel: rhyme.label,
      core: selected ? selected.core : [], rhymeRows: selected ? selected.rhymeRows : [],
      score: selected ? selected.best.score : null, selectionScore: selected ? selected.best.selectionScore : null,
      rhymeSelection: {policy: exploringRhyme ? "seeded-feasible-rhyme" : options.rhymeGroup === "auto" ? "historical-score-only" : "explicit",
        feasibleCount: feasible.length, selected: rhyme.id},
      selection, opening: null, elapsedMs: now() - started, candidatesConsidered,
      options, checks: selected ? checkResult(selected, rhyme.id, length) : [],
      lineDetails: selected ? selected.best.selected.map((line, row) => ({text: line.text, prefix: line.prefix, suffix: line.suffix,
        tones: line.tones, rhymeGroup: line.rhymeGroup, lmScore: line.lmScore, prefixCount: line.prefixCount,
        suffixCount: line.suffixCount, ...selected.best.terms[row]})) : [],
      templateSummaries: outcomes.map(row => ({templateId: row.templateId, label: row.label, status: row.status,
        score: row.best ? row.best.score : null, selectionScore: row.best ? row.best.selectionScore : null,
        lines: row.best ? row.best.lines : [], rows: row.rows, completedPaths: row.finalists.length})),
      provenance: {method: data.meta.method, lineLength: length, preselectedLines: false, randomization: {
        algorithm: "joint-diverse-beam-seeded-rhyme-v4", enabled: options.variation > 0,
        seed: options.seed, variation: options.variation, openingHistory: options.openingHistory, recentPoems: options.recentPoems,
        historyPolicy: "all-line-2-3gram-soft-penalty-v1",
        changesHardConstraints: false, uniqueOutputGuaranteed: false,
      }, candidateScoresPrecomputed: true, neuralModelUsed: false,
        sources: data.meta.sources, contextualPronunciationVerified: false, antithesisCertified: false,
        literaryQualityCertified: false, globalOptimumClaim: false},
    };
    return result;
  }

  global.VerseLoomEngine = Object.freeze({generate, version: ENGINE_VERSION, supportedLineLengths: Object.freeze([5, 7])});
})(globalThis);
