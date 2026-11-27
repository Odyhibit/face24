// Header-stripped AVIF: the QR payload is only the AV1 frame data. Everything else in the
// file (container boxes, temporal delimiter, sequence header) is this fixed template, made by
// tools/avif_eval.py with Pillow (quality any, speed 4, 4:0:0, 64x80) and identical for every face.
const TEMPLATE_B64 =
  "AAAAHGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZgAAAOltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABwaWN0AAAAAAAAAAAAAAAAAAAAAA5waXRtAAAAAAABAAAAHmlsb2MAAAAARAAAAQABAAAAAQAAAQ0AAAATAAAAKGlpbmYAAAAAAAEAAAAaaW5mZQIAAAAAAQAAYXYwMUNvbG9yAAAAAGhpcHJwAAAASWlwY28AAAAUaXNwZQAAAAAAAABAAAAAUAAAAA5waXhpAAAAAAEIAAAADGF2MUOBABwAAAAAE2NvbHJuY2x4AAEADQAGgAAAABdpcG1hAAAAAAAAAAEAAQQBAoMEAAAAG21kYXQSAAoGGBW/nsqgMgcUAAAAiBDA";
const ILOC_LENGTH = 115; // 2 bytes, big-endian: length of the mdat contents

const TEMPLATE = Uint8Array.from(atob(TEMPLATE_B64), c => c.charCodeAt(0));
const T = split(TEMPLATE);
const FIXED_OBUS = concat(T.obus[0].header, T.obus[1].header, T.obus[1].body); // temporal delimiter + sequence header
const FRAME_HEADER = T.obus[2].header[0];
export const SEQUENCE_HEADER = hex(T.obus[1].body);
// Sequence headers whose frames decode identically under the template's header. The browser
// encoder (jsquash, libavif 1.0.1) also writes colour labels (primaries 2 / transfer 2 /
// matrix 6). They don't affect a full-range 4:0:0 image or how frame data is parsed; every
// other field (profile, still picture, 64x80, coding tools, monochrome, 8-bit, full range,
// no film grain) is bit-for-bit the same.
const EQUIVALENT_HEADERS = new Set([SEQUENCE_HEADER, "1815bf9ecb020206a0"]);

function hex(bytes) { return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join(""); }
function concat(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
function leb128(value) {
  const out = [];
  do { let b = value & 0x7f; value >>>= 7; out.push(b | (value ? 0x80 : 0)); } while (value);
  return Uint8Array.from(out);
}

function findMdat(avif) {
  let p = 0;
  while (p + 8 <= avif.length) {
    const size = new DataView(avif.buffer, avif.byteOffset + p).getUint32(0);
    const kind = String.fromCharCode(...avif.subarray(p + 4, p + 8));
    if (kind === "mdat") return p;
    if (size < 8) break;
    p += size;
  }
  throw new Error("AVIF file has no mdat box");
}

function split(avif) {
  const start = findMdat(avif), data = avif.subarray(start + 8), obus = [];
  let p = 0;
  while (p < data.length) {
    const h0 = p, type = (data[p] >> 3) & 15;
    if (data[p] & 0x04) throw new Error("Unexpected OBU extension header");
    if (!(data[p] & 0x02)) throw new Error("OBU without size field");
    p++;
    let size = 0, shift = 0, byte;
    do { byte = data[p++]; size += (byte & 0x7f) * 2 ** shift; shift += 7; } while (byte & 0x80);
    obus.push({ type, header: data.subarray(h0, p), body: data.subarray(p, p + size) });
    p += size;
  }
  return { container: avif.subarray(0, start), obus };
}

/** Complete AVIF file -> frame payload. Throws if the file doesn't fit the shared template. */
export function stripAvif(avif) {
  const { obus } = split(avif);
  const types = obus.map(o => o.type).join(",");
  if (types !== "2,1,6") throw new Error(`Unexpected AV1 layout (OBU types ${types}; file ${avif.length} B: ${hex(avif.subarray(0, 48))}…)`);
  if (!EQUIVALENT_HEADERS.has(hex(obus[1].body)))
    throw new Error(`AV1 sequence header ${hex(obus[1].body)} differs from the template's ${SEQUENCE_HEADER}`);
  if (obus[2].header[0] !== FRAME_HEADER) throw new Error("Unexpected frame OBU header");
  return obus[2].body.slice();
}

/** Frame payload -> complete AVIF file. */
export function rebuildAvif(payload) {
  const mdat = concat(FIXED_OBUS, [FRAME_HEADER], leb128(payload.length), payload);
  const head = T.container.slice();
  head[ILOC_LENGTH] = mdat.length >> 8; head[ILOC_LENGTH + 1] = mdat.length & 255;
  const box = new Uint8Array(8);
  new DataView(box.buffer).setUint32(0, mdat.length + 8);
  box.set([0x6d, 0x64, 0x61, 0x74], 4); // "mdat"
  return concat(head, box, mdat);
}

/** Frame payload -> Uint8Array(64*80) grey pixels, using the browser's own AVIF decoder. */
export async function decodeAvif(payload) {
  const blob = new Blob([rebuildAvif(payload)], { type: "image/avif" });
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
  if (bitmap.width !== 64 || bitmap.height !== 80) throw new Error("Decoded AVIF has the wrong size");
  const canvas = new OffscreenCanvas(64, 80), ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const rgba = ctx.getImageData(0, 0, 64, 80).data, grey = new Uint8Array(64 * 80);
  for (let i = 0; i < grey.length; i++) grey[i] = rgba[4 * i]; // 4:0:0, so R = G = B
  return grey;
}
