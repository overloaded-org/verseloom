'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
require('../web/continuation-data.js');
test('published coefficients agree with a fresh train-only rebuild',()=>{
 const file=process.env.CONTINUATION_FIXTURES;assert(file);
 const rebuilt=JSON.parse(fs.readFileSync(path.join(path.dirname(file),'ranker.json'))),fixed=VerseLoomContinuationData;
 assert.deepEqual(rebuilt.sourceIdentity,fixed.sourceIdentity);assert.deepEqual(rebuilt.features,fixed.features);
 assert.equal(rebuilt.training.pair_differences,32768);
 rebuilt.coefficients.forEach((x,i)=>assert(Math.abs(x-fixed.coefficients[i])<1e-9));assert(Math.abs(rebuilt.center-fixed.center)<1e-9);
});
