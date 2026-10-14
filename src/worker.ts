// Clavis OMR — Web Worker: roda o homr (Python) no Pyodide e as redes no onnxruntime-web.
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import * as ort from "onnxruntime-web/webgpu";
import type { FromWorker, ToWorker } from "./protocol";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// O homr e o musicxml vêm do PyPI sem alteração (wheels conferidas pelo sha256).
const WHEELS = [
  {
    url: "https://files.pythonhosted.org/packages/bd/cc/c313f3309b541cf6a3816666a7d9f4e2fdc1d34627281205e126597b0fe5/homr-0.7.0-py3-none-any.whl",
    sha256: "84c6028df06e60b1aff158c9049d1876b22137144e319b61bacc41518842c4a9",
  },
  {
    url: "https://files.pythonhosted.org/packages/50/35/1e0f741f6484815a8ce28f7d1268b48ab33d954e596b8e292493c6e0f539/musicxml-1.4-py3-none-any.whl",
    sha256: "d055293ba6aeee2714e2c1ae21f0a5fb18e01109a15d3605ac5222b4cea48803",
  },
];
// Mesmos arquivos que o homr 0.7.0 baixa (scripts/omr-models.mjs os põe em public/omr-models/).
const MODELS = [
  "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
  "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  "decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
];
const MODEL_CACHE = "clavis-omr-modelos-v1";
const PY_FILES = import.meta.glob("./py/**/*.py", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

type Kind = "segnet" | "encoder" | "decoder";
const kindOf = (model: string): Kind =>
  model.startsWith("segnet") ? "segnet" : model.startsWith("encoder") ? "encoder" : "decoder";

/* eslint-disable @typescript-eslint/no-explicit-any */
let py: any;
let recognize: any;
let modelBase = "";
let useWebGPU = false;
let segnetBatch = 8;
let backend: Record<Kind, string> = { segnet: "wasm", encoder: "wasm", decoder: "wasm" };
const sessions = new Map<string, Promise<ort.InferenceSession>>();
const times: Record<Kind, { ms: number; runs: number; createMs: number }> = {
  segnet: { ms: 0, runs: 0, createMs: 0 },
  encoder: { ms: 0, runs: 0, createMs: 0 },
  decoder: { ms: 0, runs: 0, createMs: 0 },
};

const post = (m: FromWorker) => self.postMessage(m);
const status = (text: string) => post({ type: "status", text });

async function sha256(buf: ArrayBuffer) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return [...h].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Baixa um modelo uma vez e guarda na Cache API; depois lê do cache. */
async function modelBytes(name: string, onProgress?: (done: number, total: number) => void, keep = true): Promise<Uint8Array | null> {
  const url = new URL(name, modelBase).href;
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

function session(model: string): Promise<ort.InferenceSession> {
  let s = sessions.get(model);
  if (!s) {
    const kind = kindOf(model);
    s = (async () => {
      const bytes = (await modelBytes(model))!;
      const t0 = performance.now();
      const eps: ort.InferenceSession.ExecutionProviderConfig[] =
        kind === "segnet" && useWebGPU ? ["webgpu", "wasm"] : ["wasm"];
      const sess = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
      times[kind].createMs += performance.now() - t0;
      backend[kind] = String(eps[0]);
      return sess;
    })();
    sessions.set(model, s);
  }
  return s;
}

async function releaseSegnet() {
  for (const [model, s] of sessions)
    if (kindOf(model) === "segnet") {
      sessions.delete(model);
      await (await s).release();
    }
}

const TYPES = new Map<unknown, ort.Tensor.Type>([
  [Float32Array, "float32"],
  [BigInt64Array, "int64"],
  [Int32Array, "int32"],
  [Uint8Array, "uint8"],
]);

/** numpy (PyProxy) → ort.Tensor, copiando para fora da memória do Pyodide. Tensores do JS passam direto. */
function toTensor(v: any): ort.Tensor {
  if (v instanceof ort.Tensor) return v;
  const buf = v.getBuffer();
  try {
    let data = buf.data;
    let type = TYPES.get(data.constructor);
    if (data instanceof Float64Array) [data, type] = [Float32Array.from(data), "float32"];
    if (!type) throw new Error(`tipo de array não suportado: ${data.constructor.name}`);
    return new ort.Tensor(type, data.slice(), buf.shape);
  } finally {
    buf.release();
    v.destroy();
  }
}

/** Chamado pela fachada `onnxruntime` (Python). Os argumentos são copiados antes do primeiro await. */
const bridge = {
  run(model: string, names: any, values: any, outputs: any): Promise<ort.Tensor[]> {
    const ns: string[] = names.toJs();
    const outs: string[] = outputs.toJs();
    const feeds: Record<string, ort.Tensor> = {};
    let i = 0;
    for (const v of values) feeds[ns[i++]] = toTensor(v);
    return runModel(model, feeds, outs);
  },
};

async function runModel(model: string, feeds: Record<string, ort.Tensor>, outs: string[]) {
  const kind = kindOf(model);
  // A segmentação acabou quando o transformer começa: libera a sessão dela (~200 MB) até a próxima página.
  if (kind !== "segnet") await releaseSegnet();
  const s = await session(model);
  const t0 = performance.now();
  const r = await s.run(feeds, outs);
  const t = times[kind];
  t.ms += performance.now() - t0;
  t.runs++;
  if (kind !== "decoder" || t.runs % 25 === 0) post({ type: "run", kind, runs: t.runs });
  return outs.map((n) => r[n]);
}

async function init(base: string, webgpu: boolean, batch: number) {
  modelBase = base;
  segnetBatch = batch;
  useWebGPU = webgpu && "gpu" in navigator;
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.logLevel = "error";

  status("Baixando os modelos (só na primeira vez, ~157 MB)…");
  const sizes: number[] = MODELS.map(() => 0);
  const totals: number[] = MODELS.map(() => 0);
  await Promise.all(
    MODELS.map((m, i) =>
      modelBytes(m, (d, t) => {
        sizes[i] = d;
        totals[i] = t;
        post({ type: "download", done: sizes.reduce((a, b) => a + b), total: totals.reduce((a, b) => a + b) });
      }, false),
    ),
  );

  status("Carregando o Python (Pyodide)…");
  const { loadPyodide } = await import(/* @vite-ignore */ PYODIDE + "pyodide.mjs");
  py = await loadPyodide({ indexURL: PYODIDE });
  status("Carregando numpy, OpenCV e Pillow…");
  await py.loadPackage(["numpy", "opencv-python", "pillow", "typing-extensions"]);
  const site: string = py.runPython("import site; site.getsitepackages()[0]");

  status("Instalando o homr…");
  for (const w of WHEELS) {
    const buf = await (await fetch(w.url)).arrayBuffer();
    if ((await sha256(buf)) !== w.sha256) throw new Error(`wheel com sha256 diferente: ${w.url}`);
    py.unpackArchive(buf, "wheel", { extractDir: site });
  }
  const dir = "/home/pyodide/clavis";
  for (const [path, text] of Object.entries(PY_FILES)) {
    const dest = `${dir}/${path.replace(/^\.\/py\//, "")}`;
    py.FS.mkdirTree(dest.slice(0, dest.lastIndexOf("/")));
    py.FS.writeFile(dest, text);
  }
  py.runPython(`import sys; sys.path.insert(0, "${dir}")`);
  py.pyimport("onnxruntime").set_bridge(bridge);
  const glue = py.pyimport("clavis_omr");
  glue.patch_homr(site);
  status("Importando o homr…");
  py.runPython("import homr.main, homr.staff_parsing, homr.music_xml_generator");
  recognize = glue.recognize;
  post({ type: "ready", webgpu: useWebGPU });
}

function snapshot() {
  return JSON.parse(JSON.stringify(times)) as typeof times;
}

async function page(index: number, bytes: ArrayBuffer, ext: string) {
  const before = snapshot();
  const t0 = performance.now();
  const path = `/tmp/pagina-${index + 1}.${ext}`;
  py.FS.writeFile(path, new Uint8Array(bytes));
  const extra: Record<string, unknown> = {};
  const report = (what: string, value: any) => {
    const v = value?.toJs ? value.toJs({ dict_converter: Object.fromEntries }) : value;
    value?.destroy?.();
    if (what === "staffs") post({ type: "staffs", count: v as number });
    else extra[what] = v;
  };
  const xml: string = await recognize(path, report, segnetBatch);
  py.FS.unlink(path);
  py.runPython("import gc; gc.collect()");
  const after = snapshot();
  const delta = (k: Kind) => ({
    ms: after[k].ms - before[k].ms,
    runs: after[k].runs - before[k].runs,
    createMs: after[k].createMs - before[k].createMs,
  });
  post({
    type: "result",
    index,
    xml,
    stats: {
      totalMs: performance.now() - t0,
      segnet: delta("segnet"),
      encoder: delta("encoder"),
      decoder: delta("decoder"),
      python: extra.python_times as Record<string, number>,
      pyHeapMB: Math.round(py._module.HEAP8.byteLength / 2 ** 20),
      backend: { ...backend },
    },
  });
}

let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  queue = queue
    .then(() => (m.type === "init" ? init(m.modelBase, m.webgpu, m.segnetBatch) : page(m.index, m.bytes, m.ext)))
    .catch((err: Error) => post({ type: "error", message: String(err?.message ?? err), index: m.type === "page" ? m.index : -1 }));
};
