"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");

const root = path.resolve(__dirname, "..");
const json = name => JSON.parse(fs.readFileSync(path.join(root, name), "utf8"));
const frozen = "results/free_generation_v06_run001/";
const originalResult = json(frozen + "result.json");
const originalSearch = json(frozen + "search.json");
const originalResources = json(frozen + "resources.json");
const reference = json("data/reference_v02/pingshui_106_memberships.json");
require(path.join(root, "web/data.js"));
require(path.join(root, "web/engine.js"));
const data = globalThis.VerseLoomData;
const engine = globalThis.VerseLoomEngine;
let reproduced;

function near(actual, expected, message) {
  assert.ok(Number.isFinite(actual), message + " must be finite");
  assert.ok(Math.abs(actual - expected) <= 1e-12, message + ": " + actual + " != " + expected);
}

function independentlyCheckConstraints(result) {
  assert.equal(result.status, "complete");
  assert.equal(result.lines.length, 8);
  const template = data.templates.find(row => row.id === result.templateId);
  const counts = new Map();
  const endings = [];
  for (const [row, text] of result.lines.entries()) {
    const chars = Array.from(text);
    assert.equal(chars.length, 5);
    assert.equal(new Set(chars).size, 5);
    assert.ok(!originalResources.train_lines.includes(text), "Complete training line was copied");
    const tones = chars.map(char => {
      const values = new Set(reference.characters[char].map(item => item.tone));
      assert.equal(values.size, 1);
      return [...values][0];
    }).join("");
    assert.equal(tones, template.core[row]);
    if (template.rhymeRows.includes(row)) {
      const groups = new Set(reference.characters[chars[4]].filter(item => item.tone === "P").map(item => item.group_id));
      assert.deepEqual([...groups], [result.rhymeGroup]);
      endings.push(chars[4]);
    }
    for (const char of chars) counts.set(char, (counts.get(char) || 0) + 1);
    if (row === 3 || row === 5) {
      const previous = Array.from(result.lines[row - 1]);
      assert.ok(chars.every((char, column) => char !== previous[column]));
    }
  }
  assert.equal(new Set(endings).size, endings.length);
  assert.ok([...counts.values()].every(value => value <= 2));
  assert.ok(result.checks.every(check => check.passed));
}

test("default browser search reproduces the frozen first poem and all four template paths", async () => {
  const progress = [];
  reproduced = await engine.generate(data, {}, value => progress.push(value));
  assert.deepEqual(reproduced.lines, originalResult.lines);
  assert.equal(reproduced.templateId, originalResult.template_id);
  assert.equal(reproduced.rhymeGroup, originalResult.rhyme_group);
  near(reproduced.score, originalResult.score, "selected score");
  assert.equal(reproduced.templateSummaries.length, 4);
  for (const outcome of reproduced.templateSummaries) {
    const expected = originalSearch.template_searches.find(row => row.template_id === outcome.templateId);
    assert.equal(outcome.status, expected.status);
    assert.deepEqual(outcome.lines, expected.best.lines, outcome.templateId + " path");
    near(outcome.score, expected.best.score, outcome.templateId + " score");
    assert.deepEqual(outcome.rows.map(row => ({row: row.row, candidate_pool_size: row.candidatePoolSize, counts: row.counts})), expected.rows);
  }
  assert.equal(progress.length, 32);
  assert.equal(progress.at(-1).percent, 100);
  assert.ok(progress.every((row, index) => index === 0 || row.percent >= progress[index - 1].percent));
});

test("generated output independently passes tone, rhyme, repetition, and training-line checks", () => {
  independentlyCheckConstraints(reproduced);
  for (let row = 0; row < 8; row++) {
    const detail = reproduced.lineDetails[row];
    const expected = originalResult.line_details[row];
    assert.equal(detail.text, expected.text);
    assert.equal(detail.prefix + detail.suffix, detail.text);
    assert.equal(detail.prefixCount, originalResources.prefixes[detail.prefix].count);
    assert.equal(detail.suffixCount, originalResources.suffixes[detail.suffix].count);
    near(detail.lmScore, expected.lm_score, "LM score");
    near(detail.coherenceRaw, expected.score_terms.coherence_raw, "coherence");
    near(detail.increment, expected.score_terms.increment, "increment");
  }
});

test("all twelve available rhymes produce valid poems and a changed rhyme changes the output", async () => {
  const alternatives = data.rhymes.filter(row => row.feasible && row.id !== reproduced.rhymeGroup);
  assert.equal(alternatives.length, 11);
  for (const alternate of alternatives) {
    const result = await engine.generate(data, {rhymeGroup: alternate.id});
    assert.equal(result.rhymeGroup, alternate.id);
    assert.notDeepEqual(result.lines, reproduced.lines);
    independentlyCheckConstraints(result);
  }
});

test("an explicit template is honored and matches its frozen baseline path", async () => {
  const result = await engine.generate(data, {templateId: "L5-P-R1"});
  const expected = originalSearch.template_searches.find(row => row.template_id === "L5-P-R1");
  assert.equal(result.templateId, "L5-P-R1");
  assert.equal(result.templateSummaries.length, 1);
  assert.deepEqual(result.lines, expected.best.lines);
  near(result.score, expected.best.score, "explicit template score");
  independentlyCheckConstraints(result);
});

test("supported score and beam controls remain genuine constrained search", async () => {
  const options = {beamWidth: 16, coherenceWeight: 0.75, repetitionWeight: 1.1, templateId: "L5-Z-R0"};
  const result = await engine.generate(data, options);
  assert.equal(result.options.beamWidth, 16);
  assert.equal(result.options.coherenceWeight, 0.75);
  assert.equal(result.options.repetitionWeight, 1.1);
  independentlyCheckConstraints(result);
});

test("invalid and unsupported inputs are rejected instead of silently coerced", async () => {
  const infeasible = data.rhymes.find(row => !row.feasible);
  for (const options of [
    null, [], {beamWidth: 0}, {beamWidth: 129}, {beamWidth: 1.5}, {beamWidth: "32"},
    {coherenceWeight: NaN}, {coherenceWeight: Infinity}, {coherenceWeight: -1}, {coherenceWeight: 3},
    {repetitionWeight: "0.8"}, {templateId: "L7-Z-R0"}, {rhymeGroup: "missing"},
    {rhymeGroup: infeasible.id}, {unsupportedOption: 42},
  ]) {
    await assert.rejects(engine.generate(data, options));
  }
  await assert.rejects(engine.generate({schema: "unknown"}, {}));
  await assert.rejects(engine.generate(data, {}, "not a callback"));
});

test("a progress callback can cancel main-thread fallback without later rows", async () => {
  let calls = 0;
  await assert.rejects(engine.generate(data, {}, () => {
    calls++;
    const error = new Error("cancelled");
    error.name = "AbortError";
    throw error;
  }), {name: "AbortError"});
  assert.equal(calls, 1);
});

test("the classic worker protocol executes the real engine and preserves request IDs", async () => {
  const messages = [];
  const scope = {setTimeout, clearTimeout, performance, atob, console};
  scope.self = scope;
  scope.postMessage = message => messages.push(message);
  const context = vm.createContext(scope);
  scope.importScripts = (...names) => {
    for (const name of names) {
      assert.ok(["engine.js", "data.js"].includes(name));
      vm.runInContext(fs.readFileSync(path.join(root, "web", name), "utf8"), context, {filename: name});
    }
  };
  vm.runInContext(fs.readFileSync(path.join(root, "web/worker.js"), "utf8"), context, {filename: "worker.js"});
  await scope.onmessage({data: {id: "worker-test", options: {templateId: "L5-Z-R0"}}});
  const result = messages.find(message => message.type === "result");
  assert.ok(result);
  assert.equal(result.id, "worker-test");
  assert.equal(result.result.status, "complete");
  assert.deepEqual(JSON.parse(JSON.stringify(result.result.lines)), originalResult.lines);
  assert.equal(messages.filter(message => message.type === "progress").length, 8);
  assert.equal(messages.filter(message => message.type === "error").length, 0);
  await scope.onmessage({data: {id: "invalid-worker-input", options: null}});
  const rejected = messages.find(message => message.id === "invalid-worker-input");
  assert.equal(rejected.type, "error");
});

test("derived data remains traceable to unchanged frozen source files", () => {
  assert.equal(data.meta.trainingPoems, 2203);
  assert.equal(data.meta.candidateCount, 2392);
  assert.equal(data.rhymes.filter(row => row.feasible).length, 12);
  assert.equal(data.meta.neuralModelUsed, false);
  assert.equal(data.meta.candidateScoresPrecomputed, true);
  for (const [name, expectedHash] of Object.entries(data.meta.sources)) {
    const actualHash = crypto.createHash("sha256").update(fs.readFileSync(path.join(root, name))).digest("hex");
    assert.equal(actualHash, expectedHash, name);
  }
});

function replayIdentity(result) {
  const {elapsedMs, ...identity} = result;
  return identity;
}

test("seeded exploration replays exactly, including uint32 endpoints and all score terms", async () => {
  for (const seed of [0, 42, 4294967295]) {
    const options = {seed, variation: 0.8, templateId: "L5-Z-R0"};
    const first = await engine.generate(data, options);
    await engine.generate(data, {...options, seed: (seed + 7) >>> 0});
    const second = await engine.generate(data, options);
    assert.deepEqual(replayIdentity(first), replayIdentity(second));
    independentlyCheckConstraints(first);
    assert.equal(first.engineVersion, "verseloom.seeded-openings.v2");
    assert.equal(first.provenance.randomization.seed, seed);
    let raw = 0, selected = 0;
    for (const line of first.lineDetails) {
      assert.ok(Math.abs(line.samplingBias) < options.variation);
      raw += line.increment;
      selected = selected + line.increment + line.samplingBias;
    }
    near(first.score, raw, "unperturbed objective");
    near(first.selectionScore, selected, "exploratory objective");
  }
});

test("zero exploration preserves the exact historical search independently of seed", async () => {
  for (const seed of [0, 42, 4294967295]) {
    const result = await engine.generate(data, {seed, variation: 0});
    assert.deepEqual(result.lines, originalResult.lines);
    near(result.score, originalResult.score, "legacy score");
    assert.equal(result.selectionScore, result.score);
    assert.ok(result.lineDetails.every(line => line.samplingBias === 0));
    assert.equal(result.provenance.randomization.enabled, false);
    for (const outcome of result.templateSummaries) {
      const old = originalSearch.template_searches.find(row => row.template_id === outcome.templateId);
      assert.deepEqual(outcome.lines, old.best.lines);
      near(outcome.score, old.best.score, "legacy template score");
    }
  }
});

test("seed has a causal effect with BOTH rhyme and template fixed, without relaxing constraints", async () => {
  const poems = new Set();
  for (let seed = 0; seed < 16; seed++) {
    const result = await engine.generate(data, {seed, variation: 0.8, templateId: "L5-Z-R0", rhymeGroup: originalResult.rhyme_group});
    independentlyCheckConstraints(result);
    assert.equal(result.templateId, "L5-Z-R0");
    assert.equal(result.rhymeGroup, originalResult.rhyme_group);
    poems.add(result.lines.join("/"));
  }
  assert.ok(poems.size > 1, "A seed field that never changes selection is not exploration");
});

test("seeded runs cover all twelve rhymes and all four explicitly selected templates", async () => {
  for (const rhyme of data.rhymes.filter(row => row.feasible)) {
    const result = await engine.generate(data, {seed: 20261030, variation: 0.8, rhymeGroup: rhyme.id});
    independentlyCheckConstraints(result);
    assert.equal(result.rhymeGroup, rhyme.id);
  }
  for (const template of data.templates) {
    const result = await engine.generate(data, {seed: 17, variation: 1.5, templateId: template.id});
    independentlyCheckConstraints(result);
    assert.equal(result.templateId, template.id);
  }
});

test("seeded choice is independent of candidate enumeration and concurrent invocation order", async () => {
  const permuted = JSON.parse(JSON.stringify(data));
  for (const rows of Object.values(permuted.pools)) rows.reverse();
  const options = {seed: 42, variation: 0.8, templateId: "L5-Z-R0"};
  const [a, b, c] = await Promise.all([
    engine.generate(data, options), engine.generate(permuted, options), engine.generate(data, options),
  ]);
  assert.deepEqual(replayIdentity(a), replayIdentity(b));
  assert.deepEqual(replayIdentity(a), replayIdentity(c));
});

test("invalid seeds and exploration strengths fail rather than truncating or wrapping", async () => {
  for (const seed of [-1, 4294967296, 1.5, NaN, Infinity, "42", true, false, null, undefined]) {
    await assert.rejects(engine.generate(data, {seed, variation: 0.8}));
  }
  for (const variation of [-0.1, 2.01, NaN, Infinity, "0.8", true, null, undefined]) {
    await assert.rejects(engine.generate(data, {seed: 42, variation}));
  }
});

test("seeded worker result equals main-thread result, and cancellation does not consume future randomness", async () => {
  const options = {seed: 42, variation: 0.8, templateId: "L5-Z-R0"};
  const referenceResult = await engine.generate(data, options);
  await assert.rejects(engine.generate(data, options, () => { throw new Error("cancel test"); }));
  const messages = [];
  const scope = {setTimeout, clearTimeout, performance, atob, console};
  scope.self = scope;
  scope.postMessage = value => messages.push(value);
  const context = vm.createContext(scope);
  scope.importScripts = (...names) => names.forEach(name => vm.runInContext(fs.readFileSync(path.join(root, "web", name), "utf8"), context));
  vm.runInContext(fs.readFileSync(path.join(root, "web/worker.js"), "utf8"), context);
  await scope.onmessage({data: {id: "seeded-worker", options}});
  const response = messages.find(item => item.type === "result");
  assert.ok(response);
  assert.equal(response.id, "seeded-worker");
  assert.deepEqual(replayIdentity(JSON.parse(JSON.stringify(response.result))), replayIdentity(referenceResult));
});

require("./seed-ui.test.cjs");

test("opening diversity is measured separately from whole-poem diversity over 64 fixed seeds", async () => {
  const firstLines = new Set(), counts = new Map();
  for (let seed = 0; seed < 64; seed++) {
    const result = await engine.generate(data, {seed, variation: 0.8, templateId: "L5-Z-R0"});
    independentlyCheckConstraints(result);
    firstLines.add(result.lines[0]);
    const prefix = result.lines[0].slice(0, 2);
    counts.set(prefix, (counts.get(prefix) || 0) + 1);
    assert.equal(result.opening.selected, result.lines[0]);
    assert.ok(result.opening.groupCount <= 16);
    assert.ok(result.opening.candidateCount <= 48);
    assert.ok(result.opening.scoreGap <= 0.8);
  }
  // Regression threshold for THIS frozen fixture, not a population guarantee.
  assert.ok(firstLines.size >= 30);
  assert.ok(counts.size >= 12);
  assert.ok(Math.max(...counts.values()) <= 16, "One opening prefix dominates over a quarter of the fixture");
});

test("recent eight openings and four prefixes are excluded during 64 consecutive runs", async () => {
  let history = [];
  for (let seed = 0; seed < 64; seed++) {
    const options = {seed, variation: 0.8, templateId: "L5-Z-R0", openingHistory: history};
    const result = await engine.generate(data, options);
    independentlyCheckConstraints(result);
    assert.ok(!history.includes(result.lines[0]));
    assert.ok(!history.slice(-4).some(line => line.slice(0, 2) === result.lines[0].slice(0, 2)));
    assert.deepEqual(result.options.openingHistory, history);
    if (seed % 16 === 0) {
      const replay = await engine.generate(data, result.options);
      assert.deepEqual(replayIdentity(result), replayIdentity(replay));
    }
    history = history.concat(result.lines[0]).slice(-8);
  }
});

test("history filtering retains all four templates and twelve rhyme choices", async () => {
  for (const template of data.templates) for (const rhyme of data.rhymes.filter(row => row.feasible)) {
    const options = {seed: 20261030, variation: 0.8, templateId: template.id, rhymeGroup: rhyme.id};
    const first = await engine.generate(data, options);
    independentlyCheckConstraints(first);
    const next = await engine.generate(data, {...options, openingHistory: [first.lines[0]]});
    independentlyCheckConstraints(next);
    assert.notEqual(next.lines[0].slice(0, 2), first.lines[0].slice(0, 2));
  }
});

test("no eligible opening returns infeasible instead of silently dropping history", async () => {
  const limited = JSON.parse(JSON.stringify(data));
  const key = "ZZPPZ/nonrhyming";
  limited.meta.candidateCount -= limited.pools[key].length - 1;
  limited.pools[key] = [limited.pools[key][0]];
  const result = await engine.generate(limited, {seed: 42, variation: 0.8, templateId: "L5-Z-R0",
    openingHistory: [limited.pools[key][0][0]]});
  assert.equal(result.status, "infeasible");
  assert.equal(result.lines.length, 0);
  assert.equal(result.templateSummaries[0].opening.eligibleCandidates, 0);
  assert.deepEqual(result.options.openingHistory, [limited.pools[key][0][0]]);
});

test("history is validated, copied at run start, and passed identically through a worker", async () => {
  for (const openingHistory of [null, "日暮春風起", [1], ["日暮"], ["123456"], new Array(1), Array(9).fill("日暮春風起")]) {
    await assert.rejects(engine.generate(data, {openingHistory}));
  }
  const history = [originalResult.lines[0]];
  const options = {seed: 42, variation: 0.8, templateId: "L5-Z-R0", openingHistory: history};
  const running = engine.generate(data, options);
  history.push("江南歸路遠");
  const result = await running;
  assert.deepEqual(result.options.openingHistory, [originalResult.lines[0]]);
  const messages = [], scope = {setTimeout, performance, atob};
  scope.self = scope;
  scope.postMessage = message => messages.push(message);
  const context = vm.createContext(scope);
  scope.importScripts = (...names) => names.forEach(name => vm.runInContext(fs.readFileSync(path.join(root, "web", name), "utf8"), context));
  vm.runInContext(fs.readFileSync(path.join(root, "web/worker.js"), "utf8"), context);
  await scope.onmessage({data: {id: "opening-history", options: result.options}});
  const reply = messages.find(item => item.type === "result");
  assert.ok(reply);
  assert.deepEqual(replayIdentity(JSON.parse(JSON.stringify(reply.result))), replayIdentity(result));
});
