// Baixa os 3 modelos ONNX do homr 0.7.0 (os mesmos que `homr --init` baixa) para public/omr-models/,
// de onde a página Clavis OMR (omr.html) os carrega. Não vão para o git (~157 MB).
// O GitHub Releases não manda cabeçalhos CORS, por isso os modelos são servidos pelo próprio site.
// Uso: npm run omr-models
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const BASE = "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/";
const MODELS = [
  "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f",
  "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903",
  "decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903",
];
const out = path.resolve(import.meta.dirname, "../public/omr-models");
fs.mkdirSync(out, { recursive: true });

/** Extrai uma entrada de um .zip (lendo o diretório central; só "stored" e "deflate"). */
function unzipEntry(zip, wanted) {
  let eocd = zip.length - 22;
  while (eocd >= 0 && zip.readUInt32LE(eocd) !== 0x06054b50) eocd--;
  if (eocd < 0) throw new Error("zip inválido");
  let p = zip.readUInt32LE(eocd + 16);
  const count = zip.readUInt16LE(eocd + 10);
  for (let i = 0; i < count; i++) {
    const method = zip.readUInt16LE(p + 10);
    const size = zip.readUInt32LE(p + 20);
    const nameLen = zip.readUInt16LE(p + 28);
    const extraLen = zip.readUInt16LE(p + 30);
    const commentLen = zip.readUInt16LE(p + 32);
    const local = zip.readUInt32LE(p + 42);
    const name = zip.toString("utf8", p + 46, p + 46 + nameLen);
    p += 46 + nameLen + extraLen + commentLen;
    if (path.basename(name) !== wanted) continue;
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const data = zip.subarray(start, start + size);
    return method === 0 ? data : zlib.inflateRawSync(data);
  }
  throw new Error(`${wanted} não está no zip`);
}

for (const m of MODELS) {
  const file = path.join(out, m + ".onnx");
  if (fs.existsSync(file)) {
    console.log("já existe:", path.basename(file));
    continue;
  }
  console.log("baixando", m + ".zip");
  const res = await fetch(BASE + m + ".zip");
  if (!res.ok) throw new Error(`HTTP ${res.status} em ${m}.zip`);
  const onnx = unzipEntry(Buffer.from(await res.arrayBuffer()), m + ".onnx");
  fs.writeFileSync(file, onnx);
  console.log("ok:", path.basename(file), (onnx.length / 2 ** 20).toFixed(1), "MB");
}
