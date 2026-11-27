// Compare the JavaScript engine (web/face24.js, web/align.js) with the PyTorch/Python
// reference outputs in web/testvectors.json.   Run: node tools/check_web.mjs
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseModel, encode, decode } from "../web/face24.js";
import { alignFace } from "../web/align.js";
import { stripAvif, rebuildAvif } from "../web/avif.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const vectors = JSON.parse(readFileSync(join(root, "web/testvectors.json"), "utf8"));
const b64 = s => new Uint8Array(Buffer.from(s, "base64"));
const f32 = s => { const b = b64(s); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
let failures = 0;

for (const size of [64, 128, 256]) {
  const file = readFileSync(join(root, `web/models/face24_${size}.bin`));
  const model = parseModel(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
  const ref = vectors.models[size];
  vectors.inputs.forEach((input, n) => {
    const pixels = b64(input);
    let t = performance.now();
    const payload = encode(model, pixels);
    const encodeMs = performance.now() - t;
    const expected = b64(ref.payloads[n]);
    const same = payload.reduce((a, v, i) => a + (v === expected[i]), 0);
    t = performance.now();
    const recon = decode(model, expected); // decode the *reference* payload to isolate the decoder
    const decodeMs = performance.now() - t;
    const want = f32(ref.recon[n]);
    let maxDiff = 0;
    for (let i = 0; i < want.length; i++) maxDiff = Math.max(maxDiff, Math.abs(recon[i] - want[i]));
    const ok = same >= payload.length - 1 && maxDiff < 1e-4;
    if (!ok) failures++;
    console.log(`${ok ? "PASS" : "FAIL"} ${size}B input${n}: payload ${same}/${payload.length} bytes match, ` +
      `decoder max |diff| ${maxDiff.toExponential(2)}, encode ${encodeMs.toFixed(0)} ms, decode ${decodeMs.toFixed(0)} ms`);
  });
}

const localPath = join(root, "web/testvectors_local.json");
const local = existsSync(localPath) ? JSON.parse(readFileSync(localPath, "utf8")) : {};
if (!local.alignment) console.log("SKIP alignment: run tools/export_web.py --photo <photo> to create web/testvectors_local.json");
if (local.alignment) {
  const a = local.alignment, rgb = b64(a.rgb);
  const rgba = new Uint8ClampedArray(a.width * a.height * 4);
  for (let i = 0; i < a.width * a.height; i++) { rgba.set(rgb.subarray(3 * i, 3 * i + 3), 4 * i); rgba[4 * i + 3] = 255; }
  const { pixels } = alignFace(rgba, a.width, a.height, a.landmarks);
  const want = b64(a.aligned);
  let maxDiff = 0, exact = 0;
  for (let i = 0; i < want.length; i++) { const d = Math.abs(pixels[i] - want[i]); maxDiff = Math.max(maxDiff, d); exact += d === 0; }
  const ok = maxDiff <= 1;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} alignment: ${exact}/${want.length} pixels identical, max diff ${maxDiff} grey levels`);
}
// Header-stripped AVIF: rebuilding from the frame payload must give Pillow's file byte for byte.
const avif = JSON.parse(readFileSync(join(root, "web/avif_vectors.json"), "utf8"));
for (const c of avif.cases) {
  const payload = b64(c.payload), rebuilt = rebuildAvif(payload), want = b64(c.rebuilt), back = stripAvif(rebuilt);
  const ok = rebuilt.length === want.length && rebuilt.every((v, i) => v === want[i]) &&
    back.length === payload.length && back.every((v, i) => v === payload[i]);
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} AVIF input${c.input} ${c.budget}B budget: ${payload.length}-byte frame rebuilds to Pillow's file`);
}
process.exit(failures ? 1 : 0);
