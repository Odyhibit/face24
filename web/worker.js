// Runs the int8 codecs off the main thread.
import { loadModel, encode, decode } from "./face24.js";
import { stripAvif } from "./avif.js";

// AVIF encoder (libavif + libaom compiled to wasm, ~3.5 MB). The page preloads it at startup
// ("avifLoad"); a failed load is forgotten so the next request retries.
const AVIF_ENCODER = "https://cdn.jsdelivr.net/npm/@jsquash/avif@2.1.1/codec/enc/avif_enc.js";
// Same settings as tools/avif_eval.py uses with Pillow: speed 4, 4:0:0 (subsample 0), 8-bit.
const AVIF_OPTIONS = { quality: 50, qualityAlpha: -1, denoiseLevel: 0, tileColsLog2: 0, tileRowsLog2: 0, speed: 4,
  subsample: 0, chromaDeltaQ: false, sharpness: 0, tune: 0, enableSharpYUV: false, bitDepth: 8, lossless: false };
let avifModule;
function avifEncoder() {
  avifModule ??= import(AVIF_ENCODER).then(m => m.default({ noInitialRun: true }))
    .catch(e => { avifModule = undefined; throw e; });
  return avifModule;
}
async function avifFrame(pixels, quality) {
  const module = await avifEncoder();
  const rgba = new Uint8Array(pixels.length * 4);
  pixels.forEach((v, i) => { rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = v; rgba[4 * i + 3] = 255; });
  const out = module.encode(rgba, 64, 80, { ...AVIF_OPTIONS, quality });
  if (!out) throw new Error("AVIF encoding failed");
  return stripAvif(new Uint8Array(out));
}
// For each budget, the smallest frame that is at least that many bytes (ties: higher quality),
// so AVIF never gets fewer bytes than the codec. Sizes aren't strictly monotonic in quality, so
// scan upwards and stop once frames are well past the largest budget.
async function avifForBudgets(pixels, budgets) {
  const found = {}, stop = 2 * Math.max(...budgets);
  for (let quality = 0; quality <= 100; quality++) {
    const payload = await avifFrame(pixels, quality);
    for (const b of budgets)
      if (payload.length >= b && (!found[b] || payload.length <= found[b].payload.length)) found[b] = { quality, payload };
    if (payload.length > stop) break;
  }
  return found;
}

const SIZES = [64, 128, 256];
const models = {};
const ready = (async () => {
  for (const size of SIZES) {
    models[size] = await loadModel(new URL(`./models/face24_${size}.bin`, import.meta.url));
    postMessage({ type: "progress", size, bytes: models[size].fileBytes });
  }
})();

self.onmessage = async ({ data }) => {
  const { id, op } = data;
  try {
    // Preload doesn't need the codec models, so it runs in parallel with them.
    if (op === "avifLoad") { await avifEncoder(); return postMessage({ id, result: true }); }
    await ready;
    if (op === "ready") return postMessage({ id, result: true });
    if (op === "encodeAll") {
      const results = {};
      for (const size of SIZES) {
        let t = performance.now();
        const payload = encode(models[size], data.pixels);
        const encodeMs = performance.now() - t;
        t = performance.now();
        const recon = decode(models[size], payload);
        results[size] = { payload, recon, encodeMs, decodeMs: performance.now() - t };
      }
      return postMessage({ id, result: results });
    }
    if (op === "avifAll") {
      const t = performance.now();
      const found = await avifForBudgets(data.pixels, SIZES);
      return postMessage({ id, result: { found, ms: performance.now() - t } });
    }
    if (op === "avifSelftest") {
      const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
      const rows = [];
      for (const c of data.vectors.cases) {
        const pixels = b64(data.vectors.inputs[c.input]), want = b64(c.payload);
        const got = await avifFrame(pixels, c.quality);
        rows.push({ input: c.input, budget: c.budget, quality: c.quality, bytes: got.length, want: want.length,
          identical: got.length === want.length && got.every((x, i) => x === want[i]) });
      }
      return postMessage({ id, result: rows });
    }
    if (op === "decode") {
      const model = models[data.payload.length];
      if (!model) throw new Error(`Payload has ${data.payload.length} bytes; expected 64, 128 or 256.`);
      const t = performance.now();
      const recon = decode(model, data.payload);
      return postMessage({ id, result: { recon, decodeMs: performance.now() - t } });
    }
    if (op === "selftest") {
      const v = data.vectors, rows = [];
      const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
      for (const size of SIZES) {
        v.inputs.forEach((input, n) => {
          const payload = encode(models[size], b64(input));
          const want = b64(v.models[size].payloads[n]);
          const same = payload.reduce((a, x, i) => a + (x === want[i]), 0);
          const recon = decode(models[size], want);
          const ref = new Float32Array(b64(v.models[size].recon[n]).buffer);
          let max = 0;
          for (let i = 0; i < ref.length; i++) max = Math.max(max, Math.abs(recon[i] - ref[i]));
          rows.push({ size, n, same, total: payload.length, max });
        });
      }
      return postMessage({ id, result: rows });
    }
    throw new Error("Unknown op " + op);
  } catch (error) {
    console.error("[face24 worker]", op, error);
    postMessage({ id, error: String(error && error.message || error) });
  }
};
