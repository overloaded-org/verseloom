/* VerseLoom browser search. No model API, training, or prewritten output selection. */
(function (global) {
  "use strict";

  const ENGINE_VERSION = "verseloom.seeded-openings.v2";
  const prepared = new WeakMap();
  const allowedOptions = new Set(["rhymeGroup", "templateId", "beamWidth", "coherenceWeight", "repetitionWeight", "seed", "variation", "openingHistory"]);
  const now = () => global.performance ? global.performance.now() : Date.now();
  const yieldTurn = () => new Promise(resolve => setTimeout(resolve, 0));

  function requireValue(condition, message) {
    if (!condition) throw new Error(message);
  }

  // Stateless, keyed 32-bit mixer. No ambient RNG, clock, candidate IDs or
  // iteration order participates. The same choice gets the same bounded bias
  // in every beam prefix. This is exploratory ranking, not a probability model
  // or a cryptographic random-number generator.
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
      requireValue(["PPZZP", "PPPZZ", "ZZZPP", "ZZPPZ"].includes(tones), "候选平仄格式不正确。");
      pools[key] = rows.map(row => {
        const chars = Array.from(row[0]);
        requireValue(chars.length === 5 && new Set(chars).size === 5 && Number.isFinite(row[1]), "候选句格式不正确。");
        requireValue(chars.every(char => charIds.has(char)) && row[2] >= 2 && row[3] >= 2, "候选来源不完整。");
        return {id: candidateCount++, text: row[0], chars, charIds: chars.map(char => charIds.get(char)),
          tones, rhymeGroup: group === "nonrhyming" ? null : group, lmScore: row[1],
          prefix: chars.slice(0, 2).join(""), suffix: chars.slice(2).join(""), prefixCount: row[2], suffixCount: row[3]};
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
    const result = Object.assign({seed: 0, variation: 0, openingHistory: []}, data.defaults, options);
    requireValue(Array.isArray(result.openingHistory) && result.openingHistory.length <= 8 &&
      Array.from(result.openingHistory).every(line => typeof line === "string" && Array.from(line).length === 5),
      "首句历史须为最多 8 个五字句的数组。");
    // Snapshot caller-owned history: later UI changes must not change an active run.
    result.openingHistory = result.openingHistory.slice();
    requireValue(Number.isInteger(result.seed) && result.seed >= 0 && result.seed <= 4294967295,
      "种子须为 0 至 4294967295 的整数。");
    requireValue(typeof result.variation === "number" && Number.isFinite(result.variation) && result.variation >= 0 && result.variation <= 2,
      "探索强度须为 0 至 2 之间的有限数值。");
    requireValue(Number.isInteger(result.beamWidth) && result.beamWidth >= 1 && result.beamWidth <= 128, "搜索宽度须为 1 至 128 的整数。");
    for (const name of ["coherenceWeight", "repetitionWeight"]) {
      requireValue(typeof result[name] === "number" && Number.isFinite(result[name]) && result[name] >= 0 && result[name] <= 2,
        "评分权重须为 0 至 2 之间的有限数值。");
    }
    requireValue(result.templateId === "auto" || data.templates.some(row => row.id === result.templateId), "未知的五律模板。");
    requireValue(result.rhymeGroup === "auto" || data.rhymes.some(row => row.id === result.rhymeGroup && row.feasible), "请选择可用韵部。");
    return result;
  }

  // Sample an opening before completing the poem. A frequent prefix gets ONE
  // group weight, not one lottery ticket per suffix. Shortlist by original LM
  // score (a statistical proxy, not certified literary quality), then draw a
  // prefix and one of its top three lines. Seed keys are independent of traversal.
  function selectOpening(pool, templateId, options, failedAnchors = []) {
    const blocked = new Set(options.openingHistory);
    const prefixes = new Set(options.openingHistory.slice(-4).map(line => Array.from(line).slice(0, 2).join("")));
    const eligible = pool.filter(line => !blocked.has(line.text) && !prefixes.has(line.prefix) && !failedAnchors.includes(line.text))
      .slice().sort((a, b) => b.lmScore - a.lmScore || compareText(a.text, b.text));
    const receipt = {policy: options.variation > 0 ? "prefix-balanced-softmax-v1" : "score-only",
      originalCandidates: pool.length, eligibleCandidates: eligible.length,
      blockedLines: [...blocked], blockedPrefixes: [...prefixes], failedAnchors: failedAnchors.slice(),
      selected: null, groupCount: 0, candidateCount: 0};
    if (!eligible.length || options.variation === 0) return {pool: eligible, receipt};
    const best = eligible[0].lmScore;
    const groups = new Map();
    // Cap the statistical-score sacrifice relative to the best eligible opening.
    // Do not silently recycle history if it leaves no continuation: fail visibly.
    for (const line of eligible) {
      if (best - line.lmScore > 0.8) continue;
      if (!groups.has(line.prefix)) {
        if (groups.size >= 16) continue;
        groups.set(line.prefix, []);
      }
      if (groups.get(line.prefix).length < 3) groups.get(line.prefix).push(line);
    }
    const temperature = Math.max(0.2, options.variation);
    function draw(items, scores, key) {
      const top = Math.max(...scores);
      const weights = scores.map(score => Math.exp((score - top) / temperature));
      const total = preciseSum(weights);
      let ticket = (choiceNoise(options.seed, templateId + "/" + key, 0, "opening") + 1) / 2 * total;
      for (let index = 0; index < items.length; index++) {
        ticket -= weights[index];
        if (ticket < 0) return items[index];
      }
      return items[items.length - 1];
    }
    const entries = [...groups].sort((a, b) => compareText(a[0], b[0]));
    const chosenGroup = draw(entries, entries.map(([, lines]) => lines[0].lmScore), "prefix");
    const lines = chosenGroup[1].slice().sort((a, b) => compareText(a.text, b.text));
    const chosen = draw(lines, lines.map(line => line.lmScore), "line/" + chosenGroup[0]);
    Object.assign(receipt, {selected: chosen.text, groupCount: groups.size,
      candidateCount: entries.reduce((sum, [, items]) => sum + items.length, 0),
      scoreGap: best - chosen.lmScore, temperature});
    return {pool: [chosen], receipt};
  }

  function checkResult(selected, rhymeGroup) {
    const lines = selected.best.lines, details = selected.best.selected;
    const counts = new Map();
    for (const line of lines) for (const char of line) counts.set(char, (counts.get(char) || 0) + 1);
    const endings = selected.rhymeRows.map(row => details[row].chars[4]);
    const middle = [3, 5].every(row => details[row].chars.every((char, col) => char !== details[row - 1].chars[col]));
    return [
      {id: "shape", label: "八句五言", passed: lines.length === 8 && lines.every(line => Array.from(line).length === 5)},
      {id: "tones", label: "基本平仄模板", passed: details.every((line, row) => line.tones === selected.core[row]), note: "依据韵表候选读音；未确认语境读音。"},
      {id: "rhyme", label: "同部平声押韵", passed: selected.rhymeRows.every(row => details[row].rhymeGroup === rhymeGroup)},
      {id: "distinct_rhymes", label: "韵脚不重", passed: new Set(endings).size === endings.length},
      {id: "line_repeats", label: "句内不重字", passed: details.every(line => new Set(line.chars).size === 5)},
      {id: "character_limit", label: "全诗每字至多两次", passed: [...counts.values()].every(count => count <= 2)},
      {id: "middle_positions", label: "中间两联对应位置异字", passed: middle, note: "这项机械约束不等于对仗认证。"},
      {id: "candidate_source", label: "使用已过滤候选", passed: true, note: "冻结候选池已排除五言训练集完整原句；不代表全诗库原创性认证。"},
    ];
  }

  async function generate(data, inputOptions = {}, onProgress = () => {}) {
    const started = now();
    const runtime = prepare(data);
    const options = normalizeOptions(data, inputOptions);
    requireValue(typeof onProgress === "function", "进度回调无效。");
    const rhyme = options.rhymeGroup === "auto" ? data.rhymes.find(row => row.feasible) : data.rhymes.find(row => row.id === options.rhymeGroup);
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
    const outcomes = [];
    let candidatesConsidered = 0, progressPercent = 0;
    for (let templateIndex = 0; templateIndex < templates.length; templateIndex++) {
      const template = templates[templateIndex], openingAttempts = [], failedAnchors = [];
      let openingReceipt = null, rowReceipts = [], states = [];
      const maxAttempts = options.variation > 0 ? 8 : 1;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        rowReceipts = [];
        states = [{lines: [], selected: [], score: 0, selectionScore: 0, terms: []}];
        for (let row = 0; row < 8; row++) {
          const rhymeRow = template.rhymeRows.includes(row);
          let pool = runtime.pools[template.core[row] + "/" + (rhymeRow ? rhyme.id : "nonrhyming")] || [];
          if (row === 0 && (options.variation > 0 || options.openingHistory.length)) {
            const opening = selectOpening(pool, template.id, options, failedAnchors);
            pool = opening.pool;
            openingReceipt = opening.receipt;
          }
          // Precompute by logical choice, not by traversal, for stable replay.
          const biases = new Map(pool.map(candidate => [candidate.text, options.variation === 0 ? 0 :
            options.variation * choiceNoise(options.seed, template.id, row, candidate.text)]));
          const counts = {attempted_extensions: 0}, extensions = [];
          const reject = reason => { counts[reason] = (counts[reason] || 0) + 1; };
          for (const state of states) {
            const used = new Map();
            for (const line of state.selected) for (const char of line.chars) used.set(char, (used.get(char) || 0) + 1);
            for (const candidate of pool) {
              counts.attempted_extensions++; candidatesConsidered++;
              if (state.lines.includes(candidate.text)) { reject("repeated_line"); continue; }
              if (candidate.chars.some(char => (used.get(char) || 0) >= 2)) { reject("character_use_limit"); continue; }
              if (rhymeRow && template.rhymeRows.some(previous => previous < row && state.selected[previous].chars[4] === candidate.chars[4])) {
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
              extensions.push({lines: state.lines.concat(candidate.text), selected: state.selected.concat(candidate),
                score: state.score + increment, selectionScore: state.selectionScore + increment + samplingBias,
                terms: state.terms.concat({coherenceRaw, coherenceWeighted, repetitionPenalty,
                  previousCharacterOccurrences: repeated, increment, samplingBias})});
            }
          }
          extensions.sort((a, b) => b.selectionScore - a.selectionScore || compareLines(a.lines, b.lines));
          counts.legal_extensions = extensions.length;
          counts.beam_pruned = Math.max(0, extensions.length - options.beamWidth);
          states = extensions.slice(0, options.beamWidth);
          counts.retained_states = states.length;
          rowReceipts.push({row, candidatePoolSize: pool.length, counts});
          progressPercent = Math.max(progressPercent, Math.round((templateIndex * 8 + row + 1) / (templates.length * 8) * 100));
          onProgress({templateId: template.id, templateIndex, templateCount: templates.length, row, rowCount: 8,
            percent: progressPercent, openingAttempt: attempt + 1,
            message: template.label + (attempt ? " · 更换首句第 " + (attempt + 1) + " 次" : "") + " · 正在组织第 " + (row + 1) + " 句"});
          await yieldTurn();
          if (!states.length) break;
        }
        const completed = Boolean(states.length && states[0].lines.length === 8);
        if (openingReceipt) openingAttempts.push({opening: openingReceipt, status: completed ? "complete" : "infeasible", rows: rowReceipts});
        if (completed || !openingReceipt?.selected) break;
        failedAnchors.push(openingReceipt.selected);
      }
      outcomes.push({templateId: template.id, label: template.label, core: template.core, rhymeRows: template.rhymeRows,
        status: states.length && states[0].lines.length === 8 ? "complete" : "infeasible",
        best: states.length && states[0].lines.length === 8 ? states[0] : null, rows: rowReceipts, opening: openingReceipt, openingAttempts});
    }
    const complete = outcomes.filter(row => row.status === "complete").sort((a, b) => b.best.selectionScore - a.best.selectionScore ||
      compareText(a.templateId, b.templateId) || compareLines(a.best.lines, b.best.lines));
    const selected = complete[0];
    const result = {schema: "verseloom.web.result.v1", engineVersion: ENGINE_VERSION, status: selected ? "complete" : "infeasible",
      lines: selected ? selected.best.lines : [], templateId: selected ? selected.templateId : null,
      templateLabel: selected ? selected.label : null, rhymeGroup: rhyme.id, rhymeLabel: rhyme.label,
      core: selected ? selected.core : [], rhymeRows: selected ? selected.rhymeRows : [],
      score: selected ? selected.best.score : null, selectionScore: selected ? selected.best.selectionScore : null, elapsedMs: now() - started, candidatesConsidered,
      opening: selected ? selected.opening : null,
      options, checks: selected ? checkResult(selected, rhyme.id) : [],
      lineDetails: selected ? selected.best.selected.map((line, row) => ({text: line.text, prefix: line.prefix, suffix: line.suffix,
        tones: line.tones, rhymeGroup: line.rhymeGroup, lmScore: line.lmScore, prefixCount: line.prefixCount,
        suffixCount: line.suffixCount, ...selected.best.terms[row]})) : [],
      templateSummaries: outcomes.map(row => ({templateId: row.templateId, label: row.label, status: row.status,
        score: row.best ? row.best.score : null, selectionScore: row.best ? row.best.selectionScore : null, lines: row.best ? row.best.lines : [], rows: row.rows, opening: row.opening, openingAttempts: row.openingAttempts})),
      provenance: {method: data.meta.method, randomization: {
        algorithm: "prefix-balanced-softmax-and-keyed-bias-v2", enabled: options.variation > 0, seed: options.seed,
        variation: options.variation, openingHistory: options.openingHistory, changesHardConstraints: false, uniqueOutputGuaranteed: false,
      }, candidateScoresPrecomputed: true, neuralModelUsed: false,
        sources: data.meta.sources, contextualPronunciationVerified: false, antithesisCertified: false,
        literaryQualityCertified: false, globalOptimumClaim: false},
    };
    return result;
  }

  global.VerseLoomEngine = Object.freeze({generate, version: ENGINE_VERSION});
})(globalThis);
