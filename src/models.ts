// Clavis OMR — os modelos ONNX do homr: download, Cache API e versões do onnxruntime-web.
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).

export const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Mesmos arquivos que o homr 0.7.0 baixa (scripts/omr-models.mjs os põe em public/omr-models/).
export const MODELS = [
  "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
  "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  "decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
];
const MODEL_CACHE = "clavis-omr-modelos-v1";

export type Kind = "segnet" | "encoder" | "decoder";
export const kindOf = (model: string): Kind =>
  model.startsWith("segnet") ? "segnet" : model.startsWith("encoder") ? "encoder" : "decoder";

/** Baixa um modelo uma vez e guarda na Cache API; depois lê do cache. Com `keep = false` só garante o cache. */
export async function modelBytes(
  base: string,
  name: string,
  onProgress?: (done: number, total: number) => void,
  keep = true,
): Promise<Uint8Array | null> {
  const url = new URL(name, base).href;
  // Sem Cache API (ou com erro nela) baixa toda vez, mas funciona.
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const hit = await cache?.match(url);
  if (hit) return keep ? new Uint8Array(await hit.arrayBuffer()) : null;
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`modelo ${name}: HTTP ${res.status}`);
  const total = +(res.headers.get("content-length") ?? 0);
  const chunks: Uint8Array[] = [];
  let done = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done: end, value } = await reader.read();
    if (end) break;
    chunks.push(value);
    done += value.length;
    onProgress?.(done, total);
  }
  const blob = new Blob(chunks as BlobPart[]);
  const cached = await cache
    ?.put(url, new Response(blob, { headers: { "content-type": "application/octet-stream" } }))
    .then(() => true, () => false);
  return keep || !cached ? new Uint8Array(await blob.arrayBuffer()) : null;
}
