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
