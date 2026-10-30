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
    const length = data.lineLength === undefined ? 5 : data.lineLength;
    if (length !== 5 && length !== 7) throw new Error("仅支持五言与七言。");
    if (length === 7 && !self.VerseLoomSevenData) importScripts("data7.js");
    const dataset = length === 7 ? self.VerseLoomSevenData : VerseLoomData;
    const result = await VerseLoomEngine.generate(dataset, options === undefined ? {} : options, progress => {
      self.postMessage({id, type: "progress", progress});
    });
    self.postMessage({id, type: "result", result});
  } catch (error) {
    self.postMessage({id, type: "error", message: error instanceof Error ? error.message : "生成失败，请重新运行。"});
  } finally {
    busy = false;
  }
};
