// Compact QR transfer to Little Label Studio (https://github.com/Odyhibit/niimbot_labels).
// Packs the QR module grid into the label app's Custom QR link, so the printed code
// is the exact symbol shown here (no decoding or re-encoding).
// Keep packQrGrid in sync with niimbot_labels/image-transfer.js.

export const LABEL_PRINTER_URL = "https://odyhibit.github.io/niimbot_labels/";

/** zxing-wasm BarcodeSymbol (one pixel per module, 0 = dark) -> boolean rows. */
export function symbolToModules(symbol) {
  const { data, width, height } = symbol || {};
  if (!data || width !== height) throw new Error("QR symbol must be square.");
  const rows = [];
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) row.push(data[y * width + x] === 0);
    rows.push(row);
  }
  return rows;
}

function packQrGrid(modules) {
  const size = modules?.length;
  if (!Number.isInteger(size) || size < 21 || size > 177 || (size - 21) % 4) throw new Error("Unsupported QR grid width.");
  const bytes = new Uint8Array(Math.ceil(size * size / 8));
  modules.forEach((row, y) => {
    if (!Array.isArray(row) || row.length !== size) throw new Error("QR grid must be square.");
    row.forEach((dark, x) => {
      if (![true, false, 0, 1].includes(dark)) throw new Error("QR modules must be booleans or 0/1.");
      if (dark) { const i = y * size + x; bytes[i >> 3] |= 1 << (7 - i % 8); }
    });
  });
  return { size, data: btoa(String.fromCharCode(...bytes)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "") };
}

export function createLabelGridUrl(printerUrl, modules, name = "Custom QR") {
  const grid = packQrGrid(modules), url = new URL(printerUrl);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("Use an HTTP or HTTPS label app URL.");
  url.hash = new URLSearchParams({ grid: String(grid.size), data: grid.data, name: String(name).slice(0, 120) }).toString();
  return url.href;
}
