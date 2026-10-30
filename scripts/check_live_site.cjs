"use strict";

// Verify public HTTP assets against the manifest produced by this same build.
// Execute the downloaded data/engine in Node; this is not a browser UI test.
const assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const vm = require("node:vm");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const required = ["index.html", "styles.css", "data.js", "engine.js", "app.js", "worker.js", "standalone.html", "NOTICE.html"];

async function verifySite(siteUrl, manifestSha, settings = {}) {
  const base = new URL(siteUrl);
  assert(base.protocol === "https:" && !base.username && !base.password && !base.search && !base.hash && base.pathname.endsWith("/"), "Expected a clean HTTPS site URL ending in /");
  assert(/^[a-f0-9]{64}$/.test(manifestSha || ""), "Expected build manifest SHA-256");
  const fetchImpl = settings.fetchImpl || globalThis.fetch;
  const attempts = settings.attempts ?? 5;
  const delayMs = settings.delayMs ?? 3000;
  assert(Number.isInteger(attempts) && attempts > 0 && attempts <= 10, "Invalid retry count");
  assert(Number.isFinite(delayMs) && delayMs >= 0 && delayMs <= 10000, "Invalid retry delay");
  async function read(path, sha, size) {
    const url = new URL(path, base).href;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        const response = await fetchImpl(url, {redirect: "error", signal: AbortSignal.timeout(15000), headers: {"Cache-Control": "no-cache"}});
        assert.equal(response.status, 200, `HTTP ${response.status}: ${path || "/"}`);
        const mime = response.headers.get("content-type") || "";
        const type = path.endsWith(".js") ? /(?:javascript|ecmascript)/i : path.endsWith(".css") ? /text\/css/i : path.endsWith(".json") ? /application\/json/i : (!path || path.endsWith(".html")) ? /text\/html/i : null;
        if (type) assert.match(mime, type, `Unexpected content type: ${path}`);
        const bytes = Buffer.from(await response.arrayBuffer());
        assert(bytes.length <= 10 * 1024 * 1024, `Asset too large: ${path}`);
        if (size !== undefined) assert.equal(bytes.length, size, `Byte length mismatch: ${path}`);
        assert.equal(digest(bytes), sha, `SHA-256 mismatch: ${path || "/"}`);
        return bytes;
      } catch (error) {
        if (attempt === attempts) throw error;
        await pause(delayMs);
      }
    }
  }
  const manifestBytes = await read("build-manifest.json", manifestSha);
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  assert.equal(manifest.schema, "verseloom.browser-demo.build.v1");
  assert(Array.isArray(manifest.files) && manifest.files.length <= 100, "Invalid manifest files");
  const entries = new Map();
  for (const item of manifest.files) {
    assert(item && typeof item.path === "string" && /^[A-Za-z0-9_./-]+$/.test(item.path), "Invalid asset path");
    assert(!item.path.startsWith("/") && item.path.split("/").every(part => part && part !== "." && part !== ".."), "Unsafe asset path");
    assert(!entries.has(item.path), "Duplicate asset path");
    assert(/^[a-f0-9]{64}$/.test(item.sha256) && Number.isInteger(item.bytes) && item.bytes >= 0 && item.bytes <= 10 * 1024 * 1024, "Invalid asset identity");
    entries.set(item.path, item);
  }
  for (const path of required) assert(entries.has(path), `Missing required asset: ${path}`);
  const assets = new Map();
  for (const item of entries.values()) assets.set(item.path, await read(item.path, item.sha256, item.bytes));
  const index = entries.get("index.html");
  const homepage = (await read("", index.sha256, index.bytes)).toString("utf8");
  for (const name of ["data.js", "engine.js", "app.js"]) assert(homepage.includes(name), `Homepage missing script: ${name}`);
  const context = vm.createContext({atob, setTimeout, performance});
  // These bytes have been matched against this run's trusted build hash above.
  vm.runInContext(assets.get("data.js").toString("utf8"), context, {filename: "published-data.js", timeout: 2000});
  vm.runInContext(assets.get("engine.js").toString("utf8"), context, {filename: "published-engine.js", timeout: 2000});
  const data = context.VerseLoomData;
  assert(data && context.VerseLoomEngine, "Missing exported browser engine/data");
  function check(result) {
    assert.equal(result.status, "complete");
    assert.equal(result.lines.length, 8);
    assert(result.lines.every(line => Array.from(line).length === 5));
    assert.equal(result.checks.length, 8);
    assert(result.checks.every(check => check.passed === true), "Failed mechanical constraint");
    assert.equal(result.provenance.neuralModelUsed, false);
    assert.equal(result.provenance.preselectedLines, false);
    assert.equal(result.opening, null);
    assert.equal(result.selection.stage, "after-eight-lines");
  }
  const first = await context.VerseLoomEngine.generate(data);
  check(first);
  assert.equal(JSON.stringify(first.lines), JSON.stringify(data.baseline.lines), "Default poem differs from frozen baseline");
  assert(Math.abs(first.score - data.baseline.score) <= 1e-12, "Default score differs from frozen baseline");
  const alternate = data.rhymes.find(row => row.feasible && row.id !== first.rhymeGroup);
  assert(alternate, "No alternate rhyme for live check");
  const second = await context.VerseLoomEngine.generate(data, {rhymeGroup: alternate.id});
  check(second);
  assert.equal(second.rhymeGroup, alternate.id);
  assert.notEqual(JSON.stringify(first.lines), JSON.stringify(second.lines), "Changing rhyme did not change the poem");
  assert.equal(context.VerseLoomEngine.version, "verseloom.joint-lines.v3");
  assert(!homepage.includes("日暮春風起"), "The live page must not prefill an archived poem");
  for (const id of ["seed-input", "auto-seed", "avoid-recent", "replay-button"])
    assert(homepage.includes('id="' + id + '"'), "Missing seed/history UI: " + id);
  let history = [], last;
  const seededOpenings = [];
  for (let seed = 42; seed < 48; seed++) {
    last = await context.VerseLoomEngine.generate(data, {seed, variation: 0.8,
      templateId: "L5-Z-R0", rhymeGroup: first.rhymeGroup, openingHistory: history});
    check(last);
    assert(last.templateSummaries[0].rows.every(row => row.distinctOpenings > 1), "Search prematurely locked a single opening");
    assert(!history.includes(last.lines[0]), "Recent first line repeated");
    assert(!history.slice(-4).some(line => line.slice(0, 2) === last.lines[0].slice(0, 2)), "Recent opening prefix repeated");
    seededOpenings.push(last.lines[0]);
    history = history.concat(last.lines[0]).slice(-8);
  }
  const replay = await context.VerseLoomEngine.generate(data, last.options);
  const withoutTime = result => { const {elapsedMs, ...identity} = result; return JSON.stringify(identity); };
  assert.equal(withoutTime(replay), withoutTime(last), "Seed/history replay mismatch");
  return {status: "PASS_LIVE_HTTP_ASSETS_AND_NODE_ENGINE", url: base.href, manifestSha256: manifestSha,
    assetCount: assets.size, homepageMatched: true, defaultPoem: Array.from(first.lines),
    changedRhyme: second.rhymeLabel, changedPoem: Array.from(second.lines),
    engineVersion: context.VerseLoomEngine.version, seededOpenings, seedAndHistoryReplay: true,
    preselectedLines: false, selectionStage: "after-eight-lines", homepagePrefilled: false,
    recentFirstLineCollisions: 0, recentPrefixCollisions: 0,
    browserUIExercised: false, checkedAt: new Date().toISOString()};
}

module.exports = {verifySite, digest};
if (require.main === module) {
  verifySite(process.env.SITE_URL, process.env.MANIFEST_SHA)
    .then(result => console.log(JSON.stringify(result, null, 2)))
    .catch(error => {console.error(error.stack || String(error)); process.exitCode = 1;});
}
