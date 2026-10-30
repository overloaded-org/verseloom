"use strict";
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');require(path.join(root,'web/composition-data.js'));require(path.join(root,'web/composition-engine.js'));
const out=process.argv[2];if(!out||fs.existsSync(out))throw new Error('Specify a new output directory');fs.mkdirSync(out,{recursive:true});
const count=Number(process.argv[3]||16);if(!Number.isInteger(count)||count<1||count>128)throw new Error('Invalid seed count');
(async()=>{
 const summaries=[];
 for(const length of [5,7]){
  const unique=new Set(),positions=Array.from({length:8},()=>new Set());let completed=0;
  for(let seed=0;seed<count;seed++){
   const options={length,seed,season:'spring',time:'dusk',rhymeGroup:length===5?'pingshui:lower_level:09':'pingshui:lower_level:06',templateId:length===5?'L5-Z-R0':'L7-Z-R1'};
   const r=await VerseLoomComposition.generate(VerseLoomCompositionData,options);
   if(r.status==='complete'){completed++;unique.add(r.lines.join('/'));r.lines.forEach((l,i)=>positions[i].add(l));}
   fs.appendFileSync(path.join(out,'outputs.jsonl'),JSON.stringify(r)+'\n');console.log(JSON.stringify({length,seed,status:r.status,ms:r.elapsedMs}));
  }
  summaries.push({length,attempts:count,completed,uniquePoems:unique.size,uniqueLinesByPosition:positions.map(s=>s.size)});
 }
 const report={schema:'verseloom.composition-diagnostic.v2',engineVersion:VerseLoomComposition.version,sourceIdentity:VerseLoomCompositionData.sourceIdentity,
  fixtures:'seeds 0..'+(count-1)+'; spring/dusk; L5-Z-R0/lower09 and L7-Z-R1/lower06',summaries,
  independentLiteraryRatings:0,strictAntithesisCertification:false,contextualPronunciationCertification:false,
  note:'Descriptive engineering/development diagnostics. No claim of literary improvement or universal feasibility.'};
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1});
