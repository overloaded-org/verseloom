'use strict';
importScripts('induced-data.js','continuation-data.js','continuation-ranker.js','induced-engine.js','closure-study.js');
self.onmessage=async({data:m})=>{try{const result=await VerseLoomClosure.generate(VerseLoomInducedData,m.options,p=>self.postMessage({id:m.id,type:'progress',message:p.message}));self.postMessage({id:m.id,type:'result',result});}catch(e){self.postMessage({id:m.id,type:'error',message:e.message});}};
