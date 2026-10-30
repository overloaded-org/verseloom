'use strict';
importScripts('quality-data.js','quality-engine.js');
self.onmessage=async event=>{
 const {id,options}=event.data||{};
 try {const result=await VerseLoomQuality.generate(VerseLoomQualityData,options,p=>self.postMessage({id,type:'progress',progress:p}));self.postMessage({id,type:'result',result});}
 catch(error){self.postMessage({id,type:'error',message:String(error.message||error)});}
};
