// Mensagens entre a página do Clavis OMR e o worker.
export type Kind = "segnet" | "encoder" | "decoder";
export interface ModelTime {
  ms: number;
  runs: number;
  createMs: number;
}
export interface PageStats {
  totalMs: number;
  segnet: ModelTime;
  encoder: ModelTime;
  decoder: ModelTime;
  python?: Record<string, number>;
  pyHeapMB: number;
  backend: Record<Kind, string>;
}
export type ToWorker =
  | { type: "init"; modelBase: string; webgpu: boolean; segnetBatch: number }
  | { type: "page"; index: number; bytes: ArrayBuffer; ext: string };
export type FromWorker =
  | { type: "status"; text: string }
  | { type: "download"; done: number; total: number }
  | { type: "ready"; webgpu: boolean }
  | { type: "run"; kind: Kind; runs: number }
  | { type: "staffs"; count: number }
  | { type: "result"; index: number; xml: string; stats: PageStats }
  | { type: "error"; message: string; index: number };

// Entre o worker do Pyodide e os workers do onnxruntime (src/ort-*.ts): um por modelo, só um de cada vez. Cada um é
// encerrado quando o worker do Pyodide passa para outro modelo, e a memória WebAssembly dele volta ao sistema.
/** Tensor copiado entre workers. */
export interface WireTensor {
  type: string;
  dims: readonly number[];
  data: Float32Array | Uint8Array | Int32Array | BigInt64Array | Uint16Array;
}
/** Tensor que ficou no worker do onnxruntime (o cache do decoder, que só serve para o passo seguinte). */
export interface TensorRef {
  ref: number;
  type: string;
  dims: readonly number[];
}
export type ToOrt =
  | { type: "init"; modelBase: string; webgpu: boolean }
  | { type: "run"; id: number; model: string; feeds: Record<string, WireTensor | TensorRef>; outs: string[]; keep: string[] };
export type FromOrt =
  | { type: "result"; id: number; outputs: (WireTensor | TensorRef)[]; ms: number; createMs: number; backend: string }
  | { type: "error"; id: number; message: string };
