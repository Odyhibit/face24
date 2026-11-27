// Face24 alignment in JavaScript, mirroring demo/codec.py + face24/alignment.py:
// MediaPipe mesh -> five points -> similarity transform (24 px interocular, eyes at y=23)
// -> 4x supersampled OpenCV-style bicubic warp -> Pillow-style Lanczos to 64x80 grayscale.

const BACKGROUND = 128 / 255;
const SS = 4, OUT_W = 64, OUT_H = 80;

/** mesh: [[x,y], ...] normalized to [0,1] (MediaPipe face landmarks). Returns pixel-space five points. */
export function fivePointsFromMesh(mesh, width, height) {
  if (!mesh || mesh.length < 468) throw new Error("Face landmarks are missing or invalid; try a clearer, front-facing photo.");
  const p = mesh.map(([x, y]) => [x * width, y * height]);
  const mean = (a, b) => [(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2];
  const eyes = [mean(33, 133), mean(362, 263)].sort((a, b) => a[0] - b[0]);
  const mouth = [p[61], p[291]].sort((a, b) => a[0] - b[0]);
  const points = [eyes[0], eyes[1], p[1], mouth[0], mouth[1]];
  const iod = Math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]);
  if (points[1][0] <= points[0][0] || !(iod >= 2 && iod <= Math.min(width, height) * 0.8))
    throw new Error("Detected eye landmarks have implausible geometry; try a clearer, front-facing photo.");
  return { five: points, dense: p };
}

/** Similarity transform as a 2x3 matrix [[a,b,c],[d,e,f]] (source -> 64x80 output). */
export function similarity(five, interocular = 24, eyeY = 23) {
  const [l, r] = five;
  const eye = [(l[0] + r[0]) / 2, (l[1] + r[1]) / 2];
  const v = [r[0] - l[0], r[1] - l[1]], d = Math.hypot(v[0], v[1]);
  if (d <= 1 || v[0] <= 0) throw new Error("Invalid interocular geometry");
  const u = [v[0] / d, v[1] / d], s = interocular / d;
  const R = [[s * u[0], s * u[1]], [-s * u[1], s * u[0]]];
  return [[R[0][0], R[0][1], 31.5 - (R[0][0] * eye[0] + R[0][1] * eye[1])],
          [R[1][0], R[1][1], eyeY - (R[1][0] * eye[0] + R[1][1] * eye[1])]];
}

const apply = (m, [x, y]) => [m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2]];

/** Pillow "L" conversion: ITU-R 601-2 luma with integer rounding. rgba: Uint8ClampedArray. */
export function toGray(rgba, width, height) {
  const out = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const L = (rgba[4 * i] * 19595 + rgba[4 * i + 1] * 38470 + rgba[4 * i + 2] * 7471 + 0x8000) >> 16;
    out[i] = L / 255;
  }
  return out;
}

function cubicWeights(x) {
  const A = -0.75;
  const w0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
  const w1 = ((A + 2) * x - (A + 3)) * x * x + 1;
  const w2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
  return [w0, w1, w2, 1 - w0 - w1 - w2];
}

/** cv2.warpAffine(INTER_CUBIC, BORDER_CONSTANT) for a float32 plane; coordinates snap to 1/32 px like OpenCV. */
function warpCubic(src, sw, sh, m, dw, dh, border) {
  const [[a, b, c], [d, e, f]] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  const ic = -(ia * c + ib * f), iff = -(id * c + ie * f);
  const out = new Float32Array(dw * dh);
  const tabs = Array.from({ length: 32 }, (_, k) => cubicWeights(k / 32));
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const X = Math.round((ia * x + ib * y + ic) * 32), Y = Math.round((id * x + ie * y + iff) * 32);
      const x0 = (X >> 5) - 1, y0 = (Y >> 5) - 1, wx = tabs[X & 31], wy = tabs[Y & 31];
      let sum = 0;
      for (let j = 0; j < 4; j++) {
        const yy = y0 + j;
        let row = 0;
        for (let i = 0; i < 4; i++) {
          const xx = x0 + i;
          const v = (xx >= 0 && xx < sw && yy >= 0 && yy < sh) ? src[yy * sw + xx] : border;
          row += v * wx[i];
        }
        sum += row * wy[j];
      }
      out[y * dw + x] = sum;
    }
  }
  return out;
}

function lanczosCoeffs(inSize, outSize) {
  const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = 3 * filterscale;
  const sinc = x => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));
  const lanczos = x => (-3 < x && x < 3 ? sinc(x) * sinc(x / 3) : 0);
  const result = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = [];
    let total = 0;
    for (let x = 0; x < xmax; x++) { const w = lanczos((x + xmin - center + 0.5) / filterscale); k.push(w); total += w; }
    result.push({ xmin, k: k.map(w => (total !== 0 ? w / total : w)) });
  }
  return result;
}

/** Pillow F-mode LANCZOS resize: horizontal pass, then vertical. */
function lanczosResize(src, sw, sh, dw, dh) {
  const hc = lanczosCoeffs(sw, dw), vc = lanczosCoeffs(sh, dh);
  const tmp = new Float32Array(dw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) {
    const { xmin, k } = hc[x]; let s = 0;
    for (let i = 0; i < k.length; i++) s += src[y * sw + xmin + i] * k[i];
    tmp[y * dw + x] = s;
  }
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
    const { xmin, k } = vc[y]; let s = 0;
    for (let i = 0; i < k.length; i++) s += tmp[(xmin + i) * dw + x] * k[i];
    out[y * dw + x] = s;
  }
  return out;
}

const roundHalfEven = v => { const r = Math.round(v); return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; };

/**
 * rgba: Uint8ClampedArray of the photo; mesh: normalized MediaPipe landmarks.
 * Returns {pixels: Uint8Array(64*80), five}. Throws a readable error if the face does not fit.
 */
export function alignFace(rgba, width, height, mesh) {
  const { five, dense } = fivePointsFromMesh(mesh, width, height);
  const m = similarity(five);
  const important = [...five, ...[70, 105, 336, 300, 152].map(i => dense[i])];
  for (const pt of important) {
    const [x, y] = apply(m, pt);
    if (!(x >= -0.5 && x <= 63.5 && y >= -0.5 && y <= 79.5))
      throw new Error("This face's critical landmarks do not fit the Face24 crop. Try a more frontal or less tightly framed photo.");
  }
  const high = [[m[0][0] * SS, m[0][1] * SS, m[0][2] * SS + (SS - 1) / 2],
                [m[1][0] * SS, m[1][1] * SS, m[1][2] * SS + (SS - 1) / 2]];
  const gray = toGray(rgba, width, height);
  const hi = warpCubic(gray, width, height, high, OUT_W * SS, OUT_H * SS, BACKGROUND);
  const small = lanczosResize(hi, OUT_W * SS, OUT_H * SS, OUT_W, OUT_H);
  const pixels = Uint8Array.from(small, v => roundHalfEven(Math.min(1, Math.max(0, v)) * 255));
  return { pixels, five };
}
