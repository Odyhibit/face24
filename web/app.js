// Face24 browser test page: photo -> MediaPipe landmarks -> Face24 crop -> int8 codecs -> raw-byte QR,
// and QR (upload or camera) -> bytes -> face. Everything runs locally; nothing is uploaded.
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs";
import { readBarcodes, writeBarcode } from "https://cdn.jsdelivr.net/npm/zxing-wasm@3.1.6/dist/es/full/index.js";
import { alignFace } from "./align.js";
import { toBytes } from "./face24.js";
import { decodeAvif, rebuildAvif } from "./avif.js";
import { LABEL_PRINTER_URL, createLabelGridUrl, symbolToModules } from "./label-transfer.js";

const $ = id => document.getElementById(id);
const LANDMARKER = "../weights/mediapipe/face_landmarker.task";
const PHASE = { 64: "64 bytes", 128: "128 bytes", 256: "256 bytes (RVQ)" };

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let nextId = 0;
const pending = new Map();
let modelBytes = 0;
worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    modelBytes += data.bytes;
    setStatus("models", `Codecs: loaded ${data.size}-byte model (${(modelBytes / 1e6).toFixed(1)} MB so far)`);
    return;
  }
  const p = pending.get(data.id);
  pending.delete(data.id);
  data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
};
const call = (op, payload = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  worker.postMessage({ id, op, ...payload });
});

function setStatus(key, text, state = "") {
  const el = $(`status-${key}`);
  el.textContent = text;
  el.dataset.state = state;
}

// ---------------------------------------------------------------- startup
let landmarker;
const startup = performance.now();
call("ready").then(() => setStatus("models", `Codecs ready · ${(modelBytes / 1e6).toFixed(1)} MB int8 weights`, "ok"))
  .catch(e => setStatus("models", "Codecs failed: " + e.message, "error"));
(async () => {
  try {
    const files = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm");
    landmarker = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: LANDMARKER },
      runningMode: "IMAGE", numFaces: 8, minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
    });
    setStatus("mediapipe", `Face landmarks ready (${((performance.now() - startup) / 1000).toFixed(1)} s)`, "ok");
    $("photo").disabled = false;
  } catch (e) {
    setStatus("mediapipe", "Face landmarker failed: " + e.message, "error");
  }
})();

// ---------------------------------------------------------------- drawing helpers
function drawGray(canvas, values) { // values: Uint8Array or Float32Array(0..1), 64x80
  const bytes = values instanceof Uint8Array ? values : toBytes(values);
  canvas.width = 64; canvas.height = 80;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(64, 80);
  bytes.forEach((v, i) => img.data.set([v, v, v, 255], 4 * i));
  ctx.putImageData(img, 0, 0);
}

// Header-stripped AVIF at each budget, shown next to the codec output.
let avifRun = 0;
async function showAvif(pixels) {
  const run = ++avifRun, sizes = [64, 128, 256];
  for (const size of sizes) { $(`avif-meta-${size}`).textContent = "encoding…"; $(`avif-${size}`).getContext("2d").clearRect(0, 0, 64, 80); }
  if (!$("status-avif").dataset.state) setStatus("avif", "AVIF encoder: downloading (about 3.5 MB)…");
  try {
    const { found, ms } = await call("avifAll", { pixels });
    if (run !== avifRun) return;
    setStatus("avif", `AVIF encoder ready · last search ${(ms / 1000).toFixed(1)} s`, "ok");
    for (const size of sizes) {
      const f = found[size];
      if (!f) { $(`avif-meta-${size}`).textContent = "no size found"; continue; }
      const grey = await decodeAvif(f.payload);
      drawGray($(`avif-${size}`), grey);
      $(`avif-meta-${size}`).textContent = `${f.payload.length} B`;
    }
  } catch (e) {
    console.error("[face24] AVIF failed:", e);
    if (run !== avifRun) return;
    setStatus("avif", "AVIF failed: " + e.message, "error");
    for (const size of sizes) $(`avif-meta-${size}`).textContent = "failed";
  }
}

async function makeQr(payload) {
  const out = await writeBarcode(payload, { format: "QRCode", ecLevel: "M", scale: 6 });
  if (out.error) throw new Error(out.error);
  // Verify like the Python app does: scan the generated pixels before showing them.
  const [check] = await readBarcodes(out.image, { formats: ["QRCode"] });
  if (!check || check.bytes.length !== payload.length || check.bytes.some((b, i) => b !== payload[i]))
    throw new Error("Generated QR did not scan back to the same bytes");
  // The module grid (one pixel per module, 0 = dark) goes to the label printer as-is.
  return { url: URL.createObjectURL(out.image), modules: symbolToModules(out.symbol) };
}

// ---------------------------------------------------------------- print a label
const qrModules = {};
for (const size of [64, 128, 256]) {
  $(`print-${size}`).addEventListener("click", () => {
    const msg = $("encode-message");
    try {
      const url = createLabelGridUrl(LABEL_PRINTER_URL, qrModules[size], `Face24 ${size} B`);
      window.open(url, "_blank", "noopener");
    } catch (e) {
      msg.dataset.state = "error";
      msg.textContent = "Could not open the label editor: " + e.message;
    }
  });
}

// ---------------------------------------------------------------- encode a photo
$("photo").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  const msg = $("encode-message");
  $("results").hidden = true;
  for (const size of [64, 128, 256]) { delete qrModules[size]; $(`print-${size}`).disabled = true; }
  msg.textContent = "Detecting face…"; msg.dataset.state = "";
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none" });
    const canvas = $("original");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;

    let t = performance.now();
    const result = landmarker.detect(canvas);
    const detectMs = performance.now() - t;
    const count = result.faceLandmarks.length;
    if (count === 0) throw new Error("No face was detected. Try a clear, well-lit image with the face visible.");
    if (count > 1) throw new Error(`Detected ${count} faces. Please use an image containing exactly one face.`);
    const mesh = result.faceLandmarks[0].map(p => [p.x, p.y]);
    t = performance.now();
    const { pixels } = alignFace(rgba, bitmap.width, bitmap.height, mesh);
    const alignMs = performance.now() - t;
    drawGray($("aligned"), pixels);

    msg.textContent = "Encoding with all three codecs…";
    t = performance.now();
    const results = await call("encodeAll", { pixels });
    const codecMs = performance.now() - t;
    for (const size of [64, 128, 256]) {
      const r = results[size];
      drawGray($(`recon-${size}`), r.recon);
      $(`psnr-${size}`).textContent = `${size} B`;
      const qr = await makeQr(r.payload);
      $(`qr-${size}`).src = qr.url;
      $(`dl-${size}`).href = qr.url;
      $(`dl-${size}`).download = `face24_${size}.png`;
      qrModules[size] = qr.modules;
      $(`print-${size}`).disabled = false;
      $(`meta-${size}`).textContent = `${PHASE[size]} · encode ${r.encodeMs.toFixed(0)} ms · decode ${r.decodeMs.toFixed(0)} ms`;
    }
    $("results").hidden = false;
    msg.dataset.state = "ok";
    msg.textContent = `Done · landmarks ${detectMs.toFixed(0)} ms · alignment ${alignMs.toFixed(0)} ms · ` +
      `3 codecs ${codecMs.toFixed(0)} ms (in a background worker).`;
    showAvif(pixels);
  } catch (e) {
    msg.dataset.state = "error";
    msg.textContent = e.message;
  }
});

// ---------------------------------------------------------------- decode a QR
async function decodeQr(source, label) {
  const msg = $("decode-message");
  const [code] = await readBarcodes(source, { formats: ["QRCode"] });
  if (!code) return false;
  const payload = new Uint8Array(code.bytes);
  try {
    const { recon, decodeMs } = await call("decode", { payload });
    drawGray($("decoded"), recon);
    $("decoded-card").hidden = false;
    msg.dataset.state = "ok";
    msg.textContent = `${label}: ${payload.length} raw bytes → ${PHASE[payload.length]} decoder · ${decodeMs.toFixed(0)} ms`;
  } catch (e) {
    msg.dataset.state = "error";
    msg.textContent = e.message;
  }
  return true;
}

$("qr-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (!(await decodeQr(file, "Uploaded QR"))) {
    $("decode-message").dataset.state = "error";
    $("decode-message").textContent = "No QR code found in that image.";
  }
});

let stream;
$("camera").addEventListener("click", async () => {
  const video = $("video"), button = $("camera");
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; video.hidden = true; button.textContent = "Scan with camera"; return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  } catch (e) {
    $("decode-message").dataset.state = "error";
    $("decode-message").textContent = "Camera unavailable: " + e.message;
    return;
  }
  video.srcObject = stream; video.hidden = false; await video.play();
  button.textContent = "Stop camera";
  const grab = document.createElement("canvas"), ctx = grab.getContext("2d", { willReadFrequently: true });
  const tick = async () => {
    if (!stream) return;
    grab.width = video.videoWidth; grab.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const found = await decodeQr(ctx.getImageData(0, 0, grab.width, grab.height), "Camera");
    if (found) { button.click(); return; }
    setTimeout(tick, 150);
  };
  tick();
});

// ---------------------------------------------------------------- self-test
async function avifSelftest(lines) {
  const v = await (await fetch("./avif_vectors.json")).json();
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  let ok = true;
  lines.push("", "AVIF decoding (this browser's decoder vs Pillow):");
  for (const c of v.cases) {
    const payload = b64(c.payload), rebuilt = rebuildAvif(payload), want = b64(c.rebuilt);
    const same = rebuilt.length === want.length && rebuilt.every((x, i) => x === want[i]);
    let max = NaN;
    try {
      const grey = await decodeAvif(payload), ref = b64(c.decoded);
      max = grey.reduce((m, x, i) => Math.max(m, Math.abs(x - ref[i])), 0);
    } catch (e) { lines.push(`  decode error: ${e.message}`); }
    const pass = same && max <= 1;
    ok &&= pass;
    lines.push(`  ${pass ? "ok  " : "FAIL"} input ${c.input}, ${c.budget} B budget: ${payload.length} B frame · ` +
      `file ${same ? "identical" : "DIFFERS"} · max pixel diff ${max}`);
  }
  lines.push("", "AVIF encoding (wasm libavif 1.0.1 vs Pillow's libavif 1.3.0, same quality setting):");
  try {
    const rows = await call("avifSelftest", { vectors: v });
    rows.forEach(r => lines.push(`  input ${r.input}, q${r.quality}: ${r.bytes} B vs Pillow ${r.want} B${r.identical ? " · identical" : ""}`));
    const identical = rows.filter(r => r.identical).length;
    lines.push(`  ok   template check passed; ${identical}/${rows.length} frames byte-identical to Pillow (differences are fine)`);
  } catch (e) {
    ok = false;
    lines.push("  FAIL " + e.message);
  }
  return ok;
}

$("selftest").addEventListener("click", async () => {
  const out = $("selftest-result");
  out.textContent = "Running…";
  try {
    const vectors = await (await fetch("./testvectors.json")).json();
    const t = performance.now();
    const rows = await call("selftest", { vectors });
    const codecOk = rows.every(r => r.same >= r.total - 1 && r.max < 1e-4);
    const lines = [`Face24 codecs: ${codecOk ? "PASS" : "FAIL"} · ${((performance.now() - t) / 1000).toFixed(1)} s`, ...rows.map(r =>
      `  ${r.size} B, input ${r.n}: payload ${r.same}/${r.total} bytes match PyTorch · decoder max diff ${r.max.toExponential(1)}`)];
    out.textContent = lines.join("\n") + "\n\nAVIF: running (first run downloads the 3.5 MB encoder)…";
    const avifOk = await avifSelftest(lines);
    const ok = codecOk && avifOk;
    out.dataset.state = ok ? "ok" : "error";
    out.textContent = (ok ? "PASS" : "FAIL") + "\n\n" + lines.join("\n");
  } catch (e) {
    out.dataset.state = "error";
    out.textContent = e.message;
  }
});
