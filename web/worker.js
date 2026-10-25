// CAMP worker: audio -> features -> predictions, off the main thread.
//   in:  { type: "bundle", bundle }  once per subject
//        { type: "run", id, samples }  Float32Array, mono, 22,050 Hz, unprepared
//   out: { type: "stage", id, stage }  "features" | "predict"
//        { type: "done", id, features, prediction, ms }
//        { type: "error", id, message }

import { prepare, extract, SR } from "./features.js";
import { predict } from "./model.js";

let bundle = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "bundle") { bundle = msg.bundle; return; }
  if (msg.type !== "run") return;
  const { id, samples } = msg;
  try {
    if (samples.length < 3 * SR) throw new Error("The audio is shorter than 3 seconds. Use a longer clip.");
    const t0 = performance.now();
    self.postMessage({ type: "stage", id, stage: "features" });
    const features = extract(prepare(samples));
    self.postMessage({ type: "stage", id, stage: "predict" });
    const prediction = predict(features, bundle);
    self.postMessage({ type: "done", id, features, prediction, ms: performance.now() - t0 },
                     [prediction.vertex.buffer, prediction.region.buffer, prediction.voxel.buffer]);
  } catch (err) {
    self.postMessage({ type: "error", id, message: err.message });
  }
};
