"use strict";
// DOM-contract tests, not real-browser rendering/compatibility tests.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
function application(realEngine = false) {
  class Element {
    constructor(tag = "div") {
      this.tagName = tag; this.children = []; this.value = ""; this.hidden = false; this.disabled = false;
      this.checked = false; this.style = {}; this.attributes = {}; this.listeners = {};
      this.classList = {toggle() {}};
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { this.children.push(...children); }
    replaceChildren(fragment) { this.children = fragment.children || []; }
    querySelectorAll() { return this.children; }
    remove() {} focus() {} select() {}
  }
  const html = fs.readFileSync(path.join(root, "web/index.html"), "utf8");
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const document = {
    readyState: "complete", baseURI: "https://example.test/verseloom/",
    getElementById(id) { assert.ok(elements[id], "HTML is missing " + id); return elements[id]; },
    createElement: tag => new Element(tag), createDocumentFragment: () => new Element(), body: new Element(),
  };
  elements["rhyme-select"].value = "auto"; elements["template-select"].value = "auto";
  elements["seed-input"].value = "0"; elements["auto-seed"].checked = true;
  elements["avoid-recent"].checked = true; elements["variation-select"].value = "0.8";
  const requests = []; let randomCalls = 0, pending = null;
  const context = vm.createContext({
    document, location: {protocol: "file:"}, navigator: {}, console, Uint32Array, atob, performance,
    setTimeout(fn, ms) { if (ms >= 1000) return 0; return setTimeout(fn, ms); }, clearTimeout,
    crypto: {getRandomValues(array) { array[0] = ++randomCalls * 101; return array; }}, addEventListener() {},
  });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(root, "web/data.js"), "utf8"), context);
  if (realEngine) vm.runInContext(fs.readFileSync(path.join(root, "web/engine.js"), "utf8"), context);
  const actualEngine = context.VerseLoomEngine;
  context.VerseLoomEngine = {generate: async (_data, options, progress) => {
    requests.push(JSON.parse(JSON.stringify(options)));
    if (realEngine) { pending = actualEngine.generate(_data, options, progress); return await pending; }
    return {status: "complete", lines: [...context.VerseLoomData.baseline.lines], options: {...options},
      templateId: "L5-Z-R0", rhymeGroup: "test", score: -1, checks: [], lineDetails: [], elapsedMs: 1};
  }};
  vm.runInContext(fs.readFileSync(path.join(root, "web/app.js"), "utf8"), context);
  async function act(id, event) {
    assert.equal(elements[id].disabled, false, id + " should be enabled");
    elements[id].listeners[event]({preventDefault() {}});
    await new Promise(resolve => setImmediate(resolve));
    if (pending) await pending;
    await new Promise(resolve => setImmediate(resolve));
  }
  return {elements, requests, act, html, randomCalls: () => randomCalls};
}

test("the page starts blank, not with a prewritten poem or a primed history", async () => {
  const app = application();
  assert.equal(app.elements["poem-lines"].children.length, 0);
  assert.equal(app.elements["copy-button"].disabled, true);
  assert.equal(app.elements["download-button"].disabled, true);
  assert.equal(app.elements["replay-button"].disabled, true);
  assert(!app.html.includes("日暮春風起"));
  await app.act("generation-form", "submit");
  assert.deepEqual(app.requests[0].openingHistory, []);
});

test("automatic seed changes are visible; replay reuses the original seed AND settings", async () => {
  const app = application();
  await app.act("generation-form", "submit"); await app.act("generation-form", "submit");
  assert.deepEqual(app.requests.map(row => row.seed), [101, 202]);
  assert.equal(app.elements["seed-input"].value, "202");
  assert(app.elements["seed-display"].textContent.startsWith("202"));
  app.elements["seed-input"].value = "999"; app.elements["beam-select"].value = "16";
  app.elements["variation-select"].value = "1.5";
  await app.act("replay-button", "click");
  assert.deepEqual(app.requests[2], app.requests[1]); assert.equal(app.elements["seed-input"].value, "202");
  assert.equal(app.elements["beam-select"].value, "32"); assert.equal(app.randomCalls(), 2);
});

test("manual seed zero is valid and does not consume random numbers", async () => {
  const app = application(); app.elements["auto-seed"].checked = false;
  await app.act("generation-form", "submit"); await app.act("generation-form", "submit");
  assert.equal(app.requests[0].seed, 0); assert.deepEqual(app.requests[0], app.requests[1]); assert.equal(app.randomCalls(), 0);
});

test("reset restores free exploration rather than turning the page into a fixed poem", async () => {
  const app = application(); await app.act("generation-form", "submit");
  app.elements["variation-select"].value = "0"; app.elements["auto-seed"].checked = false;
  await app.act("reset-button", "click"); await app.act("generation-form", "submit");
  assert.equal(app.requests[1].seed, 202); assert.equal(app.requests[1].variation, 0.8);
  assert.equal(app.requests[1].rhymeGroup, "auto"); assert.equal(app.requests[1].templateId, "auto");
  assert.deepEqual(app.requests[1].openingHistory, []); assert.equal(app.elements["auto-seed"].checked, true);
});

test("malformed seeds do not get coerced and do not start a job", async () => {
  for (const value of ["", "-1", "1.5", "1e3", "4294967296", "not-a-seed"]) {
    const app = application(); app.elements["auto-seed"].checked = false; app.elements["seed-input"].value = value;
    await app.act("generation-form", "submit");
    assert.equal(app.requests.length, 0); assert.equal(app.elements["error-region"].hidden, false); assert.equal(app.randomCalls(), 0);
  }
});

test("only actual successes enter history and replay does not mutate it", async () => {
  const app = application(); await app.act("generation-form", "submit");
  assert.equal(app.requests[0].openingHistory.length, 0);
  await app.act("generation-form", "submit"); assert.equal(app.requests[1].openingHistory.length, 1);
  const saved = structuredClone(app.requests[1]);
  await app.act("replay-button", "click"); assert.deepEqual(app.requests[2], saved);
  await app.act("generation-form", "submit"); assert.equal(app.requests[3].openingHistory.length, 2);
});

test("manual, deterministic and opted-out runs do not silently apply session history", async () => {
  const app = application(); await app.act("generation-form", "submit");
  app.elements["auto-seed"].checked = false; await app.act("generation-form", "submit");
  assert.deepEqual(app.requests[1].openingHistory, []);
  app.elements["auto-seed"].checked = true; app.elements["avoid-recent"].checked = false;
  await app.act("generation-form", "submit"); assert.deepEqual(app.requests[2].openingHistory, []);
  app.elements["avoid-recent"].checked = true; app.elements["variation-select"].value = "0";
  await app.act("generation-form", "submit"); assert.deepEqual(app.requests[3].openingHistory, []);
});

test("actual UI event handlers invoke the joint engine and replay the displayed poem", async () => {
  const app = application(true); app.elements["template-select"].value = "L5-Z-R0";
  const visible = () => app.elements["poem-lines"].children.map(line => line.children
    .filter(child => child.className?.startsWith("character")).map(char => char.children[0].textContent).join(""));
  const openings = [];
  for (let i = 0; i < 6; i++) {
    await app.act("generation-form", "submit"); const lines = visible();
    assert.equal(lines.length, 8); assert(lines.every(line => line.length === 5));
    assert(!openings.slice(-4).some(line => line.slice(0, 2) === lines[0].slice(0, 2))); openings.push(lines[0]);
    assert.equal(app.elements["opening-display"].textContent, "八句共同搜索 · 无预定句");
  }
  const last = visible(); await app.act("replay-button", "click");
  assert.deepEqual(visible(), last); assert.deepEqual(app.requests.at(-1), app.requests.at(-2));
});
