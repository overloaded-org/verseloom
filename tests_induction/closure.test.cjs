'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{createHash}=require('node:crypto');
for(const f of ['induced-data','continuation-data','continuation-ranker','induced-engine','closure-study'])require('../web/'+f+'.js');
const api=VerseLoomClosure,d=VerseLoomInducedData,sha=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
// Digests cover all six prefix lines, template, rhyme, legal count, sorted pair SHA256,
// original ending and three unedited choices from the separately run Python research reference.
const expected={5:['1318fd463f4596ae1a94dab06f5d1bc985b6db7f05a4066c46703edc4fa3e971','79f41b0753431263d58946355f3e62173dc3195a11d03c9929125b0e7452e3e6','0dd4814ca9cca9bca7eec38ca07d0e30e5c5c66dbeb75dc86902c299e1b5cc3a','e2af0981cf650a85f833c7a106f3f7a5cf7d6f581b21a930b7f7c85b7abbce04','651bef52437c8c4cae108be44d9d302d51dda9c08e30b638b5e991d35aae454b','56d19f77449a3bc5213b517e2a31281120d8cbfc185e28e68f766f899321e505','4a0de796d5071d0315d7f1284034a6dd96d1fb55b2a128260dc9953f49d6cddd','78ce72316abaec37555ff5953398f13f8746238058c7c4645dd2379abc09309a'],7:['e06876a0735ce56181c8a9fbced3eef56b2599dd7acfef9b9327d1dbc535c17c','2e769e57ffd242e7032d67598b94db6f4cb04c54d8d5933a2f44522b517fafa5','600c3ff2146e8d00366681ee58b7fec0205b3c6ef5749b4c667a959c04ed29c9','afa879c491aaedbf72391325821b60f79666ec34379138c10b6d28f1d09a9e73','cf4ccc9436dfb2e1972cb96f36ecd21942305d905f6a34486eb5dde83f442f84','3763e3fdcc9ed6c65a58e937dd40af0dfeac3029a8ac055beeb4925f759609f6','12e58b2e70b6e64ba8433438b51b1232d3de11cfacf41a3b85e016830bc179ba','7017cbe8b518c7399511f56ab9594de97060a423e4c9774bdc169768c9d9547b']};
const seeds=[0,1,2,3,42,43,44,45];
for(const n of [5,7])test('all frozen Python ending choices and pool hashes: '+n,async()=>{let legalTotal=0;for(const [i,seed] of seeds.entries()){
 const x=await api.generate(d,{length:n,seed});assert.equal(x.status,'complete');assert.equal(x.prefix.length,6);
 const actual=sha([x.prefix,x.templateId,x.rhymeGroup,x.legalPairs,x.legalPairSha256,x.original.ending,x.greedy.ending,x.joint.ending,x.ordered.ending]);assert.equal(actual,expected[n][i],JSON.stringify({n,seed,legal:x.legalPairs,pairHash:x.legalPairSha256,arms:[x.original.ending,x.greedy.ending,x.joint.ending,x.ordered.ending]}));
 assert(x.joint.statisticalScore+1e-12>=x.original.statisticalScore);assert(x.ordered.orderedScore+1e-12>=x.joint.orderedScore);legalTotal+=x.legalPairs;
 }assert.equal(legalTotal,n===5?20584:59521);});
test('no user-driven prefixes or theme and no mismatched model',async()=>{for(const k of ['theme','prefix','recentPoems','ending','lines','candidate'])await assert.rejects(api.generate(d,{seed:0,length:7,[k]:[]}));await assert.rejects(api.generate({...d,sourceIdentity:{}},{seed:0,length:5}));});
