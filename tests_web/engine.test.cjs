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
    {rhymeGroup: infeasible.id}, {fixedFirstLine: "日暮春風起"},
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

require("./joint-lines.test.cjs");
require("./seed-ui.test.cjs");
