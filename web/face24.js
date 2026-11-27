// Face24 int8 codecs in plain JavaScript (browser or Node). No dependencies.
// Weights are int8 with per-output-channel scales; activations are float32.
// Tensors are single images in CHW layout: {c, h, w, data: Float32Array}.

export async function loadModel(source) {
  const buffer = source instanceof ArrayBuffer ? source : await (await fetch(source)).arrayBuffer();
  return parseModel(buffer);
}

export function parseModel(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "F24W") throw new Error("Not a Face24 web model file");
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const tensors = {};
  for (const [name, t] of Object.entries(header.tensors)) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "float32") {
      tensors[name] = { shape: t.shape, data: new Float32Array(buffer, base + t.offset, count) };
    } else {
      // Dequantize once at load time; float math afterwards keeps the engine simple.
      const q = new Int8Array(buffer, base + t.offset, count);
      const scales = new Float32Array(buffer, base + t.scale_offset, t.shape[t.scale_axis]);
      const data = new Float32Array(count);
      const inner = t.shape.slice(t.scale_axis + 1).reduce((a, b) => a * b, 1);
      const axisLen = t.shape[t.scale_axis];
      for (let i = 0; i < count; i++) data[i] = q[i] * scales[Math.floor(i / inner) % axisLen];
      tensors[name] = { shape: t.shape, data };
    }
  }
  return { ...header, tensors, fileBytes: buffer.byteLength };
}

// ------------------------------------------------------------------ layers
function conv(x, weight, bias, stride, pad) {
  const [O, I, KH, KW] = weight.shape;
  const [sh, sw] = stride, [ph, pw] = pad;
  const H = Math.floor((x.h + 2 * ph - KH) / sh) + 1;
  const W = Math.floor((x.w + 2 * pw - KW) / sw) + 1;
  const out = new Float32Array(O * H * W);
  const w = weight.data, inp = x.data, plane = x.h * x.w;
  // Valid output ranges per kernel offset, so the inner loop has no bounds checks.
  const xr = [], yr = [];
  for (let kx = 0; kx < KW; kx++) xr.push([Math.max(0, Math.ceil((pw - kx) / sw)), Math.min(W, Math.floor((x.w - 1 + pw - kx) / sw) + 1)]);
  for (let ky = 0; ky < KH; ky++) yr.push([Math.max(0, Math.ceil((ph - ky) / sh)), Math.min(H, Math.floor((x.h - 1 + ph - ky) / sh) + 1)]);
  for (let o = 0; o < O; o++) {
    const base = o * H * W;
    out.fill(bias.data[o], base, base + H * W);
    for (let i = 0; i < I; i++) {
      const src = i * plane;
      for (let ky = 0; ky < KH; ky++) {
        const [y0, y1] = yr[ky];
        for (let kx = 0; kx < KW; kx++) {
          const k = w[((o * I + i) * KH + ky) * KW + kx];
          const [x0, x1] = xr[kx];
          for (let y = y0; y < y1; y++) {
            let ip = src + (y * sh + ky - ph) * x.w + x0 * sw + kx - pw;
            const end = base + y * W + x1;
            for (let op = base + y * W + x0; op < end; op++, ip += sw) out[op] += k * inp[ip];
          }
        }
      }
    }
  }
  return { c: O, h: H, w: W, data: out };
}

function convTranspose(x, weight, bias) { // stride 1, no padding
  const [I, O, KH, KW] = weight.shape;
  const H = x.h + KH - 1, W = x.w + KW - 1;
  const out = new Float32Array(O * H * W);
  for (let o = 0; o < O; o++) out.fill(bias.data[o], o * H * W, (o + 1) * H * W);
  const w = weight.data, inp = x.data;
  for (let i = 0; i < I; i++) {
    for (let o = 0; o < O; o++) {
      for (let ky = 0; ky < KH; ky++) {
        for (let kx = 0; kx < KW; kx++) {
          const k = w[((i * O + o) * KH + ky) * KW + kx];
          for (let y = 0; y < x.h; y++) {
            const orow = o * H * W + (y + ky) * W + kx, irow = i * x.h * x.w + y * x.w;
            for (let xx = 0; xx < x.w; xx++) out[orow + xx] += k * inp[irow + xx];
          }
        }
      }
    }
  }
  return { c: O, h: H, w: W, data: out };
}

// PyTorch bilinear, align_corners=False.
function upsample(x, H, W) {
  const out = new Float32Array(x.c * H * W);
  const sy = x.h / H, sx = x.w / W;
  const ys = [], xs = [];
  for (let y = 0; y < H; y++) {
    const s = Math.max((y + 0.5) * sy - 0.5, 0), y0 = Math.floor(s);
    const y1 = Math.min(y0 + 1, x.h - 1); ys.push([y0, y1, s - y0]);
  }
  for (let xx = 0; xx < W; xx++) {
    const s = Math.max((xx + 0.5) * sx - 0.5, 0), x0 = Math.floor(s);
    const x1 = Math.min(x0 + 1, x.w - 1); xs.push([x0, x1, s - x0]);
  }
  for (let c = 0; c < x.c; c++) {
    const src = c * x.h * x.w;
    for (let y = 0; y < H; y++) {
      const [y0, y1, fy] = ys[y];
      for (let xx = 0; xx < W; xx++) {
        const [x0, x1, fx] = xs[xx], d = x.data;
        const top = d[src + y0 * x.w + x0] * (1 - fx) + d[src + y0 * x.w + x1] * fx;
        const bottom = d[src + y1 * x.w + x0] * (1 - fx) + d[src + y1 * x.w + x1] * fx;
        out[(c * H + y) * W + xx] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return { c: x.c, h: H, w: W, data: out };
}

function run(ops, x, t) {
  for (const op of ops) {
    if (op.op === "conv") x = conv(x, t[op.w], t[op.b], op.stride, op.pad);
    else if (op.op === "convT") x = convTranspose(x, t[op.w], t[op.b]);
    else if (op.op === "relu") x = { ...x, data: x.data.map(v => (v > 0 ? v : 0)) };
    else if (op.op === "sigmoid") x = { ...x, data: x.data.map(v => 1 / (1 + Math.exp(-v))) };
    else if (op.op === "upsample") x = op.size ? upsample(x, op.size[0], op.size[1]) : upsample(x, x.h * 2, x.w * 2);
    else if (op.op === "residual") {
      const y = run(op.body, x, t);
      const sum = new Float32Array(x.data.length);
      for (let i = 0; i < sum.length; i++) sum[i] = x.data[i] + y.data[i];
      x = { ...x, data: sum };
    } else throw new Error("Unknown op " + op.op);
  }
  return x;
}

// ------------------------------------------------------------------ codec
function nearest(z, codebook, residualFrom) {
  // z: CHW features (64 x gh x gw). Returns indices (row-major) and the chosen vectors.
  const [entries, dim] = codebook.shape, cb = codebook.data, cells = z.h * z.w;
  const norms = new Float32Array(entries);
  for (let e = 0; e < entries; e++) { let s = 0; for (let d = 0; d < dim; d++) s += cb[e * dim + d] ** 2; norms[e] = s; }
  const indices = new Uint8Array(cells);
  for (let p = 0; p < cells; p++) {
    let best = 0, bestScore = Infinity;
    for (let e = 0; e < entries; e++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) {
        const v = z.data[d * cells + p] - (residualFrom ? residualFrom[d * cells + p] : 0);
        dot += v * cb[e * dim + d];
      }
      const score = norms[e] - 2 * dot; // |v|^2 is constant per cell
      if (score < bestScore) { bestScore = score; best = e; }
    }
    indices[p] = best;
  }
  return indices;
}

function lookup(indices, codebook, gh, gw, into) {
  const dim = codebook.shape[1], cells = gh * gw, out = into || new Float32Array(dim * cells);
  for (let p = 0; p < cells; p++) for (let d = 0; d < dim; d++) out[d * cells + p] += codebook.data[indices[p] * dim + d];
  return out;
}

/** pixels: Uint8Array(80*64) grayscale -> Uint8Array payload (64/128/256 bytes). */
export function encode(model, pixels) {
  const t = model.tensors, [gh, gw] = model.grid;
  const input = { c: 1, h: 80, w: 64, data: Float32Array.from(pixels, v => v / 255) };
  const z = run(model.encoder, input, t);
  if (model.kind === "rvq") {
    const i1 = nearest(z, t.codebook1);
    const q1 = lookup(i1, t.codebook1, gh, gw);
    const i2 = nearest(z, t.codebook2, q1);
    const payload = new Uint8Array(2 * gh * gw);
    payload.set(i1, 0); payload.set(i2, gh * gw);
    return payload;
  }
  return nearest(z, t.codebook);
}

/** payload -> Float32Array(80*64) in [0,1]. */
export function decode(model, payload) {
  if (payload.length !== model.payload_bytes) throw new Error(`Expected ${model.payload_bytes} bytes, got ${payload.length}`);
  const t = model.tensors, [gh, gw] = model.grid, cells = gh * gw;
  let q;
  if (model.kind === "rvq") {
    q = lookup(payload.subarray(0, cells), t.codebook1, gh, gw);
    lookup(payload.subarray(cells), t.codebook2, gh, gw, q);
  } else q = lookup(payload, t.codebook, gh, gw);
  return run(model.decoder, { c: 64, h: gh, w: gw, data: q }, t).data;
}

export function toBytes(values) {
  return Uint8Array.from(values, v => Math.round(Math.min(1, Math.max(0, v)) * 255));
}
