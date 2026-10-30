'use strict';
/** Development coverage probe: retain infeasibility rather than recycling poems. */
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
require(path.join(root, 'web/composition-data.js'));
require(path.join(root, 'web/composition-engine.js'));

async function main(output) {
  if (!output) throw new Error('Usage: node scripts/probe_composition_contexts_v2.cjs NEW_OUTPUT.json');
  if (fs.existsSync(output)) throw new Error('Refusing to overwrite an existing report');
  const records = [];
  for (const length of [5, 7]) {
    for (const season of ['spring', 'summer', 'autumn', 'winter']) {
      for (const time of ['dawn', 'day', 'dusk', 'night']) {
        const options = {length, season, time, seed: 20261030,
          templateId: length === 5 ? 'L5-Z-R0' : 'L7-Z-R1',
          rhymeGroup: length === 5 ? 'pingshui:lower_level:09' : 'pingshui:lower_level:06'};
        try {
          const r = await VerseLoomComposition.generate(VerseLoomCompositionData, options);
          records.push({options, status: r.status, lines: r.lines, elapsedMs: r.elapsedMs});
        } catch (error) {
          records.push({options, status: 'rejected', error: error.message});
        }
        console.log(JSON.stringify(records.at(-1)));
      }
    }
  }
  const report = {scope: 'Fixed templates/rhymes, one declared seed; tests failure transparency, not universal coverage', records};
  fs.mkdirSync(path.dirname(path.resolve(output)), {recursive: true});
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
  if (records.some(r => r.status === 'rejected')) process.exitCode = 1;
}
main(process.argv[2]).catch(error => {console.error(error.stack || String(error)); process.exitCode = 1;});
