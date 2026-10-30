"use strict";
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm'), crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
require('../web/data.js'); require('../web/data7.js'); require('../web/engine.js');
const data = globalThis.VerseLoomSevenData, five = globalThis.VerseLoomData, engine = globalThis.VerseLoomEngine;
const read = file => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const reference = read('data/reference_v02/pingshui_106_memberships.json');
const poems = fs.readFileSync(path.join(root, 'data/corpus_v03/poems.jsonl'), 'utf8').trim().split('\n').filter(line => /"split"\s*:\s*"train"/.test(line)).map(JSON.parse).filter(p => p.length === 7);
const trainLines = new Set(poems.flatMap(p => p.lines));
const poemById = new Map(poems.map(p => [p.poem_id, p]));
const identity = ({elapsedMs, ...result}) => result;
function check(r) {
  assert.equal(r.status, 'complete'); assert.equal(r.lineLength, 7); assert.equal(r.lines.length, 8);
  assert.equal(r.opening, null); assert.equal(r.provenance.preselectedLines, false);
  assert.equal(r.selection.stage, 'after-eight-lines');
  const template = data.templates.find(t => t.id === r.templateId), counts = new Map(), ends = [];
  for (const [i, line] of r.lines.entries()) {
    const chars = Array.from(line); assert.equal(chars.length, 7); assert.equal(new Set(chars).size, 7);
    assert(!trainLines.has(line));
    const tones = chars.map(c => {const t = new Set(reference.characters[c].map(x => x.tone)); assert.equal(t.size, 1); return [...t][0];}).join('');
    assert.equal(tones, template.core[i]);
    if (template.rhymeRows.includes(i)) {
      assert.deepEqual([...new Set(reference.characters[chars[6]].filter(x => x.tone === 'P').map(x => x.group_id))], [r.rhymeGroup]);
      ends.push(chars[6]);
    }
    if (i === 3 || i === 5) assert(chars.every((c, j) => c !== Array.from(r.lines[i - 1])[j]));
    for (const c of chars) counts.set(c, (counts.get(c) || 0) + 1);
    assert.equal(r.lineDetails[i].prefix.length, 4); assert.equal(r.lineDetails[i].suffix.length, 3);
    assert.equal(r.lineDetails[i].prefix + r.lineDetails[i].suffix, line);
  }
  assert.equal(new Set(ends).size, ends.length); assert([...counts.values()].every(x => x <= 2));
  assert.equal(r.checks.length, 8); assert(r.checks.every(x => x.passed));
}

test('seven data is train-only, traceable and built from actual 4+3 spans', () => {
  assert.equal(data.meta.lineLength, 7); assert.equal(data.meta.trainingPoems, poems.length); assert.equal(poems.length, 299);
  assert.equal(data.meta.candidateCount, 3892); assert.equal(data.meta.minimumSpanCount, 1);
  for (const [file, sha] of Object.entries(data.meta.sources)) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex'), sha);
  for (const kind of ['prefixes', 'suffixes']) for (const [text, span] of Object.entries(data.spans[kind])) {
    assert.equal(span.count, span.sources.length);
    for (const [id, row, start, end] of span.sources) {
      const poem = poemById.get(id); assert(poem); assert.equal(poem.split, 'train');
      assert.equal(poem.lines[row].slice(start, end), text);
      assert.equal(text.length, kind === 'prefixes' ? 4 : 3);
    }
  }
  for (const [key, rows] of Object.entries(data.pools)) for (const [text, score, pc, sc] of rows) {
    assert.equal(text.length, 7); assert.equal(new Set(text).size, 7); assert(!trainLines.has(text)); assert(Number.isFinite(score));
    assert.equal(pc, data.spans.prefixes[text.slice(0, 4)].count); assert.equal(sc, data.spans.suffixes[text.slice(4)].count);
    assert.equal(data.spans.prefixes[text.slice(0, 4)].tones + data.spans.suffixes[text.slice(4)].tones, key.split('/')[0]);
  }
});

test('seven standard patterns and rhyme rows are checked independently of generator implementation', () => {
  const patterns = {
    'L7-P-R0': ['PPZZPPZ', 'ZZPPZZP', 'ZZPPPZZ', 'PPZZZPP', 'PPZZPPZ', 'ZZPPZZP', 'ZZPPPZZ', 'PPZZZPP'],
    'L7-Z-R0': ['ZZPPPZZ', 'PPZZZPP', 'PPZZPPZ', 'ZZPPZZP', 'ZZPPPZZ', 'PPZZZPP', 'PPZZPPZ', 'ZZPPZZP'],
  };
  for (const t of data.templates) {
    const expected = patterns[t.id.replace('R1', 'R0')].slice();
    if (t.id.endsWith('R1')) expected[0] = t.id.includes('-P-') ? 'PPZZZPP' : 'ZZPPZZP';
    assert.deepEqual(t.core, expected); assert.deepEqual(t.rhymeRows, t.id.endsWith('R1') ? [0,1,3,5,7] : [1,3,5,7]);
  }
});

test('all 72 seven-character form/rhyme combinations work, also with recent history', async () => {
  assert.equal(data.rhymes.filter(r => r.feasible).length, 18);
  for (const template of data.templates) for (const rhyme of data.rhymes.filter(r => r.feasible)) {
    const opts = {seed: 20261030, variation: .8, templateId: template.id, rhymeGroup: rhyme.id};
    const a = await engine.generate(data, opts); check(a);
    const b = await engine.generate(data, {...opts, openingHistory: [a.lines[0]]}); check(b);
    assert.notEqual(a.lines[0].slice(0, 2), b.lines[0].slice(0, 2));
    assert.equal(b.templateId, template.id); assert.equal(b.rhymeGroup, rhyme.id);
  }
});

test('seven generation does not pin any line; deterministic seeds replay and all positions vary', async () => {
  const records = [], positions = Array.from({length:8}, () => new Set()), all = new Set();
  for (let seed = 0; seed < 64; seed++) {
    const r = await engine.generate(data, {seed, variation:.8, templateId:'L7-P-R0'}); check(r);
    r.lines.forEach((line,i) => positions[i].add(line)); all.add(r.lines.join('/'));
    records.push({seed, lines:r.lines, score:r.score, rhymeGroup:r.rhymeGroup});
    assert(r.templateSummaries[0].rows.every(row => row.distinctOpenings > 1));
    if (seed % 16 === 0) assert.deepEqual(identity(r), identity(await engine.generate(data, r.options)));
  }
  assert(positions.every(s => s.size > 1)); assert(all.size > 1);
  const report = {seeds:'0..63', lineLength:7, template:'L7-P-R0', variation:.8, uniquePoems:all.size, uniqueLinesByPosition:positions.map(s=>s.size), records};
  if (process.env.VERSELOOM_SEVEN_REPORT) fs.writeFileSync(process.env.VERSELOOM_SEVEN_REPORT, JSON.stringify(report,null,2)+'\n');
});

test('seven history, invalid inputs, cancellation, ordering and uint32 endpoints', async () => {
  for (const opts of [{openingHistory:['日暮春風起']},{templateId:'L5-Z-R0'},{fixedLines:['不可預先固定句']},{seed:-1}]) await assert.rejects(engine.generate(data,opts));
  const changed = structuredClone(data); for(const pool of Object.values(changed.pools)) pool.reverse();
  for(const seed of [0,4294967295]) {
    const opts = {seed,variation:.8,templateId:'L7-Z-R1'};
    const a=await engine.generate(data,opts), b=await engine.generate(changed,opts); check(a); assert.deepEqual(identity(a),identity(b));
    await assert.rejects(engine.generate(data,opts,()=>{throw new Error('cancel');}));
    assert.deepEqual(identity(a),identity(await engine.generate(data,opts)));
  }
});

test('worker supports both forms and rejects an unsupported form without a five-character fallback',async()=>{
  const messages=[], scope={setTimeout,performance,atob}; scope.self=scope; scope.postMessage=m=>messages.push(m);
  const context=vm.createContext(scope);
  scope.importScripts=(...names)=>names.forEach(n=>vm.runInContext(fs.readFileSync(path.join(root,'web',n),'utf8'),context));
  vm.runInContext(fs.readFileSync(path.join(root,'web/worker.js'),'utf8'),context);
  for(const length of [7,5]) {
    const options={seed:42,variation:.8}; await scope.onmessage({data:{id:length,lineLength:length,options}});
    const reply=messages.find(m=>m.type==='result'&&m.id===length); assert(reply);
    assert.deepEqual(identity(JSON.parse(JSON.stringify(reply.result))),identity(await engine.generate(length===7?data:five,options)));
  }
  await scope.onmessage({data:{id:'invalid',lineLength:6,options:{}}});
  assert.equal(messages.find(m=>m.id==='invalid').type,'error');
});

test('five-character frozen data and deterministic result remain unchanged',async()=>{
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'web/data.js'))).digest('hex'),'29f380e5b37ed3fc3ed6202dad40453e68ed3ac76b9243dc755b4f562b8759b6');
  const r=await engine.generate(five); assert.deepEqual(r.lines,five.baseline.lines); assert(Math.abs(r.score-five.baseline.score)<1e-12);
});
