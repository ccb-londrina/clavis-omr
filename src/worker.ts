// Clavis OMR — Web Worker: roda o homr (Python) no Pyodide. As redes rodam no onnxruntime-web, em outros workers
// (um por modelo, veja ort-serve.ts), para que a memória delas volte ao sistema quando o modelo sai de uso.
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import { type Kind, kindOf, MODELS, modelBytes } from "./models";
import type { FromOrt, FromWorker, TensorRef, ToOrt, ToWorker, WireTensor } from "./protocol";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
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
const PY_FILES = import.meta.glob("./py/**/*.py", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

/* eslint-disable @typescript-eslint/no-explicit-any */
let py: any;
let recognize: any;
let modelBase = "";
let useWebGPU = false;
let segnetBatch = 8;
let backend: Record<Kind, string> = { segnet: "wasm", encoder: "wasm", decoder: "wasm" };
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

/** Um worker do onnxruntime-web, que vive enquanto o seu modelo está em uso. */
class OrtWorker {
  private worker: Worker;
  private pending = new Map<number, { resolve: (m: Extract<FromOrt, { type: "result" }>) => void; reject: (e: Error) => void }>();
  private nextId = 1;

  constructor(gpu: boolean) {
    // Dois arquivos porque o pacote com WebGPU traz um WebAssembly duas vezes maior.
    this.worker = gpu
      ? new Worker(new URL("./ort-worker-gpu.ts", import.meta.url), { type: "module" })
      : new Worker(new URL("./ort-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<FromOrt>) => {
      const m = e.data;
      const p = this.pending.get(m.id);
      this.pending.delete(m.id);
      if (m.type === "result") p?.resolve(m);
      else p?.reject(new Error(m.message));
    };
    this.worker.onerror = (e) => this.fail(new Error(`worker do onnxruntime: ${e.message || "falhou (falta de memória?)"}`));
    this.send({ type: "init", modelBase, webgpu: gpu });
  }

  private send(m: ToOrt, transfer: Transferable[] = []) {
    this.worker.postMessage(m, { transfer });
  }

  private fail(err: Error) {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  run(model: string, feeds: Record<string, WireTensor | TensorRef>, outs: string[], keep: string[], transfer: Transferable[]) {
    const id = this.nextId++;
    return new Promise<Extract<FromOrt, { type: "result" }>>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ type: "run", id, model, feeds, outs, keep }, transfer);
    });
  }

  terminate() {
    this.worker.terminate();
    this.fail(new Error("worker do onnxruntime encerrado"));
  }
}

/**
 * Só um modelo carregado por vez, cada um no seu worker: ao trocar de modelo (segmentação → encoder → decoder →
 * encoder da pauta seguinte…), o worker anterior é encerrado e o heap WebAssembly dele (centenas de MB) volta ao
 * sistema. Criar o worker e a sessão de novo custa ~1 s, contra ~10–30 s de cada pauta.
 */
let current: { kind: Kind; worker: OrtWorker } | null = null;

function endModel() {
  current?.worker.terminate();
  current = null;
}

const TYPES = new Map<unknown, string>([
  [Float32Array, "float32"],
  [BigInt64Array, "int64"],
  [Int32Array, "int32"],
  [Uint8Array, "uint8"],
]);

/** numpy (PyProxy) → tensor para o worker do onnxruntime, copiado para fora da memória do Pyodide. */
function toWire(v: any, transfer: Transferable[]): WireTensor {
  const buf = v.getBuffer();
  try {
    let data = buf.data;
    let type = TYPES.get(data.constructor);
    if (data instanceof Float64Array) [data, type] = [Float32Array.from(data), "float32"];
    else data = data.slice();
    if (!type) throw new Error(`tipo de array não suportado: ${data.constructor.name}`);
    transfer.push(data.buffer);
    return { type, dims: buf.shape, data };
  } finally {
    buf.release();
    v.destroy();
  }
}

/** Chamado pela fachada `onnxruntime` (Python). Os argumentos são copiados antes do primeiro await. */
const bridge = {
  run(model: string, names: any, values: any, outputs: any): Promise<(WireTensor | TensorRef)[]> {
    const ns: string[] = names.toJs();
    const outs: string[] = outputs.toJs();
    const feeds: Record<string, WireTensor | TensorRef> = {};
    const transfer: Transferable[] = [];
    let i = 0;
    // Arrays do numpy chegam como PyProxy; saídas anteriores (objetos do JS) voltam como estão.
    for (const v of values) feeds[ns[i++]] = v instanceof py.ffi.PyProxy ? toWire(v, transfer) : v;
    return runModel(model, feeds, outs, transfer);
  },
};

async function runModel(model: string, feeds: Record<string, WireTensor | TensorRef>, outs: string[], transfer: Transferable[]) {
  const kind = kindOf(model);
  if (current?.kind !== kind) {
    endModel();
    current = { kind, worker: new OrtWorker(kind === "segnet" && useWebGPU) };
  }
  // O cache do decoder (32 tensores por passo) fica no worker do onnxruntime: só serve para o passo seguinte.
  const keep = outs.filter((n) => n.startsWith("cache_out"));
  const r = await current.worker.run(model, feeds, outs, keep, transfer);
  const t = times[kind];
  t.ms += r.ms;
  t.createMs += r.createMs;
  t.runs++;
  backend[kind] = r.backend;
  if (kind !== "decoder" || t.runs % 25 === 0) post({ type: "run", kind, runs: t.runs });
  return r.outputs;
}

async function init(base: string, webgpu: boolean, batch: number) {
  modelBase = base;
  segnetBatch = batch;
  useWebGPU = webgpu && "gpu" in navigator;

  status("Baixando os modelos (só na primeira vez, ~157 MB)…");
  const sizes: number[] = MODELS.map(() => 0);
  const totals: number[] = MODELS.map(() => 0);
  await Promise.all(
    MODELS.map((m, i) =>
      modelBytes(modelBase, m, (d, t) => {
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
  const xml: string = await recognize(path, report, segnetBatch).finally(endModel);
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
