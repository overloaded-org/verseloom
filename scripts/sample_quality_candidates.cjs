"use strict";
// Produce actual engine outputs, never authored replacement lines.
// Fixed experimental protocol: first candidate is baseline; all four are
// available to the experimental reranker. This is not a same-compute comparison.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
require(path.join(root, 'web/data.js'));
require(path.join(root, 'web/data7.js'));
require(path.join(root, 'web/engine.js'));
const output = process.argv[2];
if (!output) throw Error('Usage: node scripts/sample_quality_candidates.cjs NEW_OUTPUT.jsonl');
const identity = name => crypto.createHash('sha256').update(fs.readFileSync(path.join(root, name))).digest('hex');
const sources = Object.fromEntries(['web/engine.js','web/data.js','web/data7.js','scripts/sample_quality_candidates.cjs'].map(n => [n,identity(n)]));
const fd = fs.openSync(output, 'wx');
(async () => {
  let count = 0;
  for (const length of [5, 7]) {
    const data = length === 5 ? VerseLoomData : VerseLoomSevenData;
    const rhyme = data.rhymes.find(r => r.feasible).id;
    const template = length === 5 ? 'L5-Z-R0' : 'L7-P-R0';
    for (let task = 0; task < 32; task++) for (let candidate = 0; candidate < 4; candidate++) {
      const seed = (1000 + task + Math.imul(candidate, 0x9e3779b9)) >>> 0;
      const options = {seed, variation: 0.8, beamWidth: 32, templateId: template, rhymeGroup: rhyme, openingHistory: []};
      const result = await VerseLoomEngine.generate(data, options);
      if (result.status !== 'complete' || !result.checks.every(x => x.passed)) throw Error('Sample generation failed');
      const row = {schema:'verseloom.quality-samples.v1', taskId:`L${length}-${task}`, candidate, lineLength:length,
        lines:result.lines, score:result.score, options:result.options,
        core:result.core, rhymeRows:result.rhymeRows, rhymeGroup:result.rhymeGroup,
        engineVersion:result.engineVersion, preselectedLines:result.provenance.preselectedLines,
        resultSha256:crypto.createHash('sha256').update(JSON.stringify((({elapsedMs,...x}) => x)(result))).digest('hex'), sources};
      fs.writeSync(fd, JSON.stringify(row)+'\n');count++;
    }
  }
  fs.fsyncSync(fd);fs.closeSync(fd);
  console.log(JSON.stringify({status:'SAMPLED',records:count,tasks:64,candidatesPerTask:4,sourceCommit:process.env.GITHUB_SHA || null,output}));
})().catch(error => {try {fs.closeSync(fd);} catch {} console.error(error);process.exitCode=1;});
