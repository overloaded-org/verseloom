"use strict";
importScripts("engine.js", "data.js");
let busy = false;
self.onmessage = async ({data}) => {
  if (!data || typeof data !== "object") return;
  const {id, options} = data;
  if (busy) {
    self.postMessage({id, type: "error", message: "上一次生成尚未结束。"});
    return;
  }
  busy = true;
  try {
    const result = await VerseLoomEngine.generate(VerseLoomData, options === undefined ? {} : options, progress => {
      self.postMessage({id, type: "progress", progress});
    });
    self.postMessage({id, type: "result", result});
  } catch (error) {
    self.postMessage({id, type: "error", message: error instanceof Error ? error.message : "生成失败，请重新运行。"});
  } finally {
    busy = false;
  }
};
