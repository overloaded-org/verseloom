"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
require(path.join(root, "web/data.js"));
require(path.join(root, "web/engine.js"));
const data = globalThis.VerseLoomData, engine = globalThis.VerseLoomEngine;
const reference = JSON.parse(fs.readFileSync(path.join(root, "data/reference_v02/pingshui_106_memberships.json")));
const resources = JSON.parse(fs.readFileSync(path.join(root, "results/free_generation_v06_run001/resources.json")));
const identity = ({elapsedMs, ...result}) => result;
function check(result) {
  assert.equal(result.status, "complete");
  assert.equal(result.lines.length, 8);
  assert.equal(result.opening, null);
  assert.equal(result.selection.stage, "after-eight-lines");
  assert.equal(result.provenance.preselectedLines, false);
  const template = data.templates.find(t => t.id === result.templateId), counts = new Map(), rhymes = [];
  for (const [i, text] of result.lines.entries()) {
    const chars = Array.from(text);
    assert.equal(chars.length, 5); assert.equal(new Set(chars).size, 5);
    assert(!resources.train_lines.includes(text));
    assert.equal(chars.map(c => {
      const tones = new Set(reference.characters[c].map(r => r.tone));
      assert.equal(tones.size, 1); return [...tones][0];
    }).join(""), template.core[i]);
    for (const c of chars) counts.set(c, (counts.get(c) || 0) + 1);
    if (template.rhymeRows.includes(i)) {
      const groups = [...new Set(reference.characters[chars[4]].filter(r => r.tone === "P").map(r => r.group_id))];
      assert.deepEqual(groups, [result.rhymeGroup]); rhymes.push(chars[4]);
    }
    if (i === 3 || i === 5) assert(chars.every((c, j) => c !== Array.from(result.lines[i - 1])[j]));
  }
  assert([...counts.values()].every(n => n <= 2)); assert.equal(new Set(rhymes).size, rhymes.length);
  assert(result.checks.every(c => c.passed));
}

test("all eligible first lines compete and multiple openings survive until complete poems exist", async () => {
  const result = await engine.generate(data, {seed: 42, variation: 0.8, templateId: "L5-Z-R0"});
  check(result);
  const rows = result.templateSummaries[0].rows;
  assert.equal(rows.length, 8);
  assert.equal(rows[0].candidatePoolSize, data.pools["ZZPPZ/nonrhyming"].length);
  assert.equal(rows[0].counts.attempted_extensions, rows[0].candidatePoolSize);
  assert(rows.every(row => row.distinctOpenings > 1));
  assert(result.selection.candidates > 1); assert(result.selection.prefixes > 1);
});

test("same seed and inputs replay all decisions and scores, including uint32 endpoints", async () => {
  for (const seed of [0, 42, 4294967295]) {
    const opts = {seed, variation: 0.8, templateId: "L5-Z-R0"};
    const a = await engine.generate(data, opts), b = await engine.generate(data, opts);
    check(a); assert.deepEqual(identity(a), identity(b));
    assert.equal(a.engineVersion, "verseloom.joint-lines.v3");
    let raw = 0, selection = 0;
    for (const line of a.lineDetails) {
      assert(Math.abs(line.samplingBias) < opts.variation);
      raw += line.increment; selection = selection + line.increment + line.samplingBias;
    }
    assert(Math.abs(a.score - raw) < 1e-12); assert(Math.abs(a.selectionScore - selection) < 1e-12);
  }
});

test("no position is fixed across 64 predefined seeds with rhyme AND template fixed", async () => {
  const positions = Array.from({length: 8}, () => new Set()), poems = new Set(), prefixes = new Map();
  const records = [];
  for (let seed = 0; seed < 64; seed++) {
    const r = await engine.generate(data, {seed, variation: 0.8, templateId: "L5-Z-R0", rhymeGroup: data.baseline.rhymeGroup});
    check(r); r.lines.forEach((line, row) => positions[row].add(line)); poems.add(r.lines.join("/"));
    const prefix = r.lines[0].slice(0, 2); prefixes.set(prefix, (prefixes.get(prefix) || 0) + 1);
    records.push({seed, lines: r.lines, score: r.score, completedCandidates: r.selection.candidates});
  }
  assert(positions.every(rows => rows.size > 1), "A position has become a constant");
  assert(poems.size > 1); assert(prefixes.size > 1);
  // Descriptive fixture results, not population guarantees or literary scores.
  const report = {schema: "verseloom.joint-lines.seed-diagnostic.v1", seeds: "0..63", seedCount: 64,
    template: "L5-Z-R0", rhymeGroup: data.baseline.rhymeGroup, variation: 0.8,
    uniquePoems: poems.size, uniqueLinesByPosition: positions.map(rows => rows.size),
    openingPrefixCount: prefixes.size, maximumPrefixCount: Math.max(...prefixes.values()), records};
  if (process.env.VERSELOOM_DIAGNOSTIC_OUTPUT) fs.writeFileSync(process.env.VERSELOOM_DIAGNOSTIC_OUTPUT, JSON.stringify(report, null, 2) + "\n");
});

test("zero exploration remains historical optimization, never reads the archived poem", async () => {
  const altered = structuredClone(data); altered.baseline = {lines: Array(8).fill("不可使用此句")};
  for (const seed of [0, 42, 4294967295]) {
    const result = await engine.generate(altered, {seed, variation: 0});
    check(result); assert.deepEqual(result.lines, data.baseline.lines);
    assert.equal(result.score, result.selectionScore);
    assert(result.lineDetails.every(line => line.samplingBias === 0));
  }
});

test("enumeration order and concurrency do not change seeded selection", async () => {
  const altered = structuredClone(data);
  for (const pool of Object.values(altered.pools)) pool.reverse();
  const opts = {seed: 42, variation: 0.8, templateId: "L5-Z-R0"};
  const [a, b, c] = await Promise.all([engine.generate(data, opts), engine.generate(altered, opts), engine.generate(data, opts)]);
  assert.deepEqual(identity(a), identity(b)); assert.deepEqual(identity(a), identity(c));
});

test("all 48 template and rhyme combinations preserve constraints with and without recent history", async () => {
  for (const template of data.templates) for (const rhyme of data.rhymes.filter(r => r.feasible)) {
    const opts = {seed: 20261030, variation: 0.8, templateId: template.id, rhymeGroup: rhyme.id};
    const a = await engine.generate(data, opts); check(a);
    const b = await engine.generate(data, {...opts, openingHistory: [a.lines[0]]}); check(b);
    assert.notEqual(a.lines[0].slice(0, 2), b.lines[0].slice(0, 2));
    assert.equal(b.templateId, template.id); assert.equal(b.rhymeGroup, rhyme.id);
  }
});

test("recent history excludes repeats but never specifies a required line", async () => {
  let history = [];
  for (let seed = 0; seed < 24; seed++) {
    const result = await engine.generate(data, {seed, variation: 0.8, templateId: "L5-Z-R0", openingHistory: history});
    check(result);
    assert(!history.includes(result.lines[0]));
    assert(!history.slice(-4).some(line => line.slice(0, 2) === result.lines[0].slice(0, 2)));
    if (seed % 8 === 0) assert.deepEqual(identity(result), identity(await engine.generate(data, result.options)));
    history = history.concat(result.lines[0]).slice(-8);
  }
});

test("exhausted candidates fail without recycling history or returning a canned fallback", async () => {
  const limited = structuredClone(data), key = "ZZPPZ/nonrhyming";
  limited.meta.candidateCount -= limited.pools[key].length - 1; limited.pools[key] = [limited.pools[key][0]];
  const r = await engine.generate(limited, {seed: 42, variation: 0.8, templateId: "L5-Z-R0", openingHistory: [limited.pools[key][0][0]]});
  assert.equal(r.status, "infeasible"); assert.deepEqual(r.lines, []);
  assert.equal(r.templateSummaries[0].rows[0].candidatePoolSize, 0);
});

test("invalid seeds, histories, temperatures and line-lock options are rejected", async () => {
  for (const seed of [-1, 4294967296, 1.5, NaN, Infinity, "42", true, null, undefined]) await assert.rejects(engine.generate(data, {seed}));
  for (const variation of [-1, 2.1, NaN, Infinity, "0.8", true, null, undefined]) await assert.rejects(engine.generate(data, {variation}));
  for (const openingHistory of [null, "日暮春風起", [1], ["短句"], new Array(1), Array(9).fill("日暮春風起")]) await assert.rejects(engine.generate(data, {openingHistory}));
  for (const opts of [{fixedLines: ["日暮春風起"]}, {opening: "日暮春風起"}, {fixedFirstLine: "日暮春風起"}]) await assert.rejects(engine.generate(data, opts));
});

test("history is snapshotted; cancellation and worker execution preserve replay identity", async () => {
  const history = [data.baseline.lines[0]], opts = {seed: 42, variation: 0.8, templateId: "L5-Z-R0", openingHistory: history};
  const running = engine.generate(data, opts); history.push("江南歸路遠");
  const r = await running; check(r); assert.deepEqual(r.options.openingHistory, [data.baseline.lines[0]]);
  await assert.rejects(engine.generate(data, r.options, () => {throw new Error("cancel");}));
  assert.deepEqual(identity(r), identity(await engine.generate(data, r.options)));
  const messages = [], scope = {setTimeout, performance, atob}; scope.self = scope;
  scope.postMessage = message => messages.push(message);
  const context = vm.createContext(scope);
  scope.importScripts = (...names) => names.forEach(name => vm.runInContext(fs.readFileSync(path.join(root, "web", name), "utf8"), context));
  vm.runInContext(fs.readFileSync(path.join(root, "web/worker.js"), "utf8"), context);
  await scope.onmessage({data: {id: "joint-lines", options: r.options}});
  const reply = messages.find(x => x.type === "result"); assert(reply);
  assert.equal(reply.id, "joint-lines"); assert.deepEqual(identity(JSON.parse(JSON.stringify(reply.result))), identity(r));
});
