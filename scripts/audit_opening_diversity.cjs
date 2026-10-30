"use strict";
// Deterministic engineering fixture, not a human-rated poetry-quality study.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {createHash} = require("node:crypto");
const root = path.resolve(__dirname, "..");
require(path.join(root, "web/data.js"));
require(path.join(root, "web/engine.js"));
const data = globalThis.VerseLoomData, engine = globalThis.VerseLoomEngine;
const hashFile = file => createHash("sha256").update(fs.readFileSync(path.join(root, file))).digest("hex");
function distribution(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return Object.fromEntries([...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)));
}
async function audit() {
  const arms = {};
  for (const mode of ["baseline", "seeded", "session"]) {
    const rows = [];
    let history = [data.baseline.lines[0]], lineCollisions = 0, prefixCollisions = 0;
    for (let seed = 0; seed < 64; seed++) {
      const result = await engine.generate(data, {seed, variation: mode === "baseline" ? 0 : 0.8,
        templateId: "L5-Z-R0", rhymeGroup: data.baseline.rhymeGroup,
        openingHistory: mode === "session" ? history : []});
      assert.equal(result.status, "complete");
      assert(result.checks.every(check => check.passed));
      lineCollisions += Number(history.includes(result.lines[0]));
      prefixCollisions += Number(history.slice(-4).some(line => line.slice(0, 2) === result.lines[0].slice(0, 2)));
      rows.push({seed, lines: result.lines, score: result.score, options: result.options});
      history = history.concat(result.lines[0]).slice(-8);
    }
    const firstLines = distribution(rows.map(row => row.lines[0]));
    const prefixes = distribution(rows.map(row => row.lines[0].slice(0, 2)));
    arms[mode] = {summary: {
      runs: rows.length, complete: rows.length, mechanicalChecksPassed: rows.length,
      uniquePoems: new Set(rows.map(row => row.lines.join("/"))).size,
      uniqueFirstLines: Object.keys(firstLines).length, uniqueOpeningPrefixes: Object.keys(prefixes).length,
      firstLineCollisionWithin8: lineCollisions, openingPrefixCollisionWithin4: prefixCollisions,
      pairwiseFirstLineCollisionRate: Object.values(firstLines).reduce((sum, n) => sum + n * (n - 1), 0) / (64 * 63),
      meanStatisticalScore: rows.reduce((sum, row) => sum + row.score, 0) / rows.length,
      mostCommonPrefixShare: Math.max(...Object.values(prefixes)) / rows.length,
      firstLines, prefixes,
    }, rows};
  }
  assert.equal(arms.session.summary.firstLineCollisionWithin8, 0);
  assert.equal(arms.session.summary.openingPrefixCollisionWithin4, 0);
  return {schema: "verseloom.opening-diversity.audit.v1", engineVersion: engine.version,
    design: "Seeds 0..63; fixed L5-Z-R0, frozen default rhyme, beam 32; session begins with visible archived opening; no sample selection.",
    scope: "Finite engineering fixture. Not an estimate of population diversity or literary quality. Repeats outside the recent window remain possible.",
    sourceSha256: Object.fromEntries(["web/data.js", "web/engine.js", "scripts/audit_opening_diversity.cjs"].map(name => [name, hashFile(name)])), arms};
}
if (require.main === module) {
  audit().then(report => {
    const output = process.argv[2];
    if (output) fs.writeFileSync(output, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
    console.log(JSON.stringify(Object.fromEntries(Object.entries(report.arms).map(([name, arm]) => [name, arm.summary])), null, 2));
  }).catch(error => {console.error(error); process.exitCode = 1;});
}
module.exports = {audit};
