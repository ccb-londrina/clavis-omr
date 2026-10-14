// Clavis OMR — worker do onnxruntime-web: roda um modelo (segmentação, encoder ou decoder) para o worker do Pyodide.
// Fica num worker próprio para que o heap WebAssembly dele, que só cresce, suma quando o worker é encerrado.
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import type * as Ort from "onnxruntime-web";
import { modelBytes, ORT_WASM } from "./models";
import type { FromOrt, TensorRef, ToOrt, WireTensor } from "./protocol";

export function serveOrt(ort: typeof Ort) {
  let modelBase = "";
  let useWebGPU = false;
  const sessions = new Map<string, Promise<{ sess: Ort.InferenceSession; createMs: number; backend: string }>>();
  // Saídas guardadas para a próxima execução (o cache do decoder vai de um passo ao seguinte).
  let held = new Map<number, Ort.Tensor>();
  let nextRef = 1;

  function session(model: string) {
    let s = sessions.get(model);
    if (!s) {
      s = (async () => {
        const bytes = (await modelBytes(modelBase, model))!;
        const t0 = performance.now();
        const eps: Ort.InferenceSession.ExecutionProviderConfig[] = useWebGPU ? ["webgpu", "wasm"] : ["wasm"];
        const sess = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
        return { sess, createMs: performance.now() - t0, backend: String(eps[0]) };
      })();
      sessions.set(model, s);
    }
    return s;
  }

  /** Uma cópia própria do tensor, que pode ser transferida (a saída pode ser uma vista de outra memória). */
  const ownData = (d: WireTensor["data"]) =>
    d.byteOffset === 0 && d.byteLength === d.buffer.byteLength ? d : (d.slice() as WireTensor["data"]);

  async function run(m: Extract<ToOrt, { type: "run" }>) {
    const created = !sessions.has(m.model);
    const { sess, createMs, backend } = await session(m.model);
    const feeds: Record<string, Ort.Tensor> = {};
    for (const [name, v] of Object.entries(m.feeds)) {
      if ("ref" in v) {
        const t = held.get(v.ref);
        if (!t) throw new Error(`tensor ${v.ref} não está mais no worker do onnxruntime`);
        feeds[name] = t;
      } else feeds[name] = new ort.Tensor(v.type as Ort.Tensor.Type, v.data, v.dims);
    }
    const t0 = performance.now();
    const r = await sess.run(feeds, m.outs);
    const ms = performance.now() - t0;
    // As saídas guardadas valem só para a execução seguinte.
    for (const t of held.values()) t.dispose();
    held = new Map();
    const transfer: ArrayBuffer[] = [];
    const outputs = m.outs.map((n): WireTensor | TensorRef => {
      const t = r[n];
      if (m.keep.includes(n)) {
        const ref = nextRef++;
        held.set(ref, t);
        return { ref, type: t.type, dims: t.dims };
      }
      const data = ownData(t.data as WireTensor["data"]);
      transfer.push(data.buffer as ArrayBuffer);
      return { type: t.type, dims: t.dims, data };
    });
    post({ type: "result", id: m.id, outputs, ms, createMs: created ? createMs : 0, backend }, transfer);
  }

  const post = (m: FromOrt, transfer: Transferable[] = []) => self.postMessage(m, { transfer });

  let queue: Promise<void> = Promise.resolve();
  self.onmessage = (e: MessageEvent<ToOrt>) => {
    const m = e.data;
    if (m.type === "init") {
      modelBase = m.modelBase;
      useWebGPU = m.webgpu;
      ort.env.wasm.wasmPaths = ORT_WASM;
      ort.env.logLevel = "error";
      return;
    }
    queue = queue.then(() => run(m)).catch((err: Error) => post({ type: "error", id: m.id, message: String(err?.message ?? err) }));
  };
}
