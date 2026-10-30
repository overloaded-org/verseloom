"use strict";
importScripts('composition-data.js','composition-engine.js');
self.onmessage=async({data:m})=>{try{const result=await VerseLoomComposition.generate(VerseLoomCompositionData,m.options,p=>postMessage({id:m.id,type:'progress',progress:p}));postMessage({id:m.id,type:'result',result});}catch(e){postMessage({id:m.id,type:'error',message:e.message||String(e)});}};
