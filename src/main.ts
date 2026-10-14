// Clavis OMR (protótipo): fotos de partitura → MusicXML, tudo no navegador. Abre o resultado no Clavis.
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import { mergePages } from "./merge";
import type { FromWorker, PageStats, ToWorker } from "./protocol";

/** Chave do localStorage que o Clavis (src/main.ts do Clavis) lê ao abrir com #omr. */
export const OMR_STORAGE_KEY = "clavis.omr";
/** Onde está o Clavis: a página é publicada em <clavis>/omr/, na mesma origem (o localStorage é compartilhado). */
const CLAVIS_URL = import.meta.env.VITE_CLAVIS_URL ?? "../";
/** Código-fonte desta versão exata (AGPL-3.0, seção 13). */
const SOURCE_URL = `https://github.com/ccb-londrina/clavis-omr/tree/v${__OMR_VERSION__}`;

interface Page {
  file: File;
  thumb: string;
  width: number;
  height: number;
  xml?: string;
  stats?: PageStats;
  state: string;
  error?: boolean;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  camera: $<HTMLInputElement>("camera"),
  files: $<HTMLInputElement>("files"),
  pages: $<HTMLOListElement>("pages"),
  noPages: $("noPages"),
  title: $<HTMLInputElement>("title"),
  webgpu: $<HTMLInputElement>("webgpu"),
  run: $<HTMLButtonElement>("run"),
  stop: $<HTMLButtonElement>("stop"),
  progressBox: $("progressBox"),
  progress: $<HTMLProgressElement>("progress"),
  status: $("status"),
  resultBox: $("resultBox"),
  open: $<HTMLButtonElement>("open"),
  download: $<HTMLButtonElement>("download"),
  log: $("log"),
};

const pages: Page[] = [];
let worker: Worker | null = null;
let workerGpu: boolean | null = null;
let result = "";
let running = false;

const log = (s: string) => {
  ui.log.textContent += s + "\n";
  console.log("[omr]", s);
};
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

async function addFiles(list: FileList | null) {
  for (const file of list ?? []) {
    const bmp = await createImageBitmap(file).catch(() => null);
    if (!bmp) {
      alert(`Não consegui abrir ${file.name} como imagem.`);
      continue;
    }
    pages.push({ file, thumb: URL.createObjectURL(file), width: bmp.width, height: bmp.height, state: "" });
    bmp.close();
  }
  if (!ui.title.value && pages.length) ui.title.value = pages[0].file.name.replace(/(-p\d+)?\.\w+$/, "");
  render();
}

function render() {
  ui.pages.innerHTML = pages
    .map(
      (p, i) => `<li>
        <img src="${p.thumb}" alt="" />
        <span class="name">${i + 1}. ${esc(p.file.name)}</span>
        <span class="state${p.error ? " err" : ""}">${esc(p.state)}</span>
        <button data-act="up" data-i="${i}" title="Subir" ${i === 0 || running ? "disabled" : ""}>↑</button>
        <button data-act="del" data-i="${i}" title="Remover" ${running ? "disabled" : ""}>✕</button>
      </li>`,
    )
    .join("");
  ui.noPages.hidden = pages.length > 0;
  ui.run.disabled = running || pages.length === 0;
}

ui.pages.onclick = (e) => {
  const b = (e.target as HTMLElement).closest("button");
  if (!b || running) return;
  const i = +b.dataset.i!;
  if (b.dataset.act === "up" && i > 0) [pages[i - 1], pages[i]] = [pages[i], pages[i - 1]];
  if (b.dataset.act === "del") URL.revokeObjectURL(pages.splice(i, 1)[0].thumb);
  render();
};
ui.camera.onchange = () => void addFiles(ui.camera.files).then(() => (ui.camera.value = ""));
ui.files.onchange = () => void addFiles(ui.files.files).then(() => (ui.files.value = ""));

/** Recortes de 320×320 por lote na segmentação (o homr usa 8; ?lote=N na URL muda, para medir a memória). */
const SEGNET_BATCH = Math.max(1, +(new URLSearchParams(location.search).get("lote") ?? 8) || 8);

/** Lotes da segmentação, estimados pelo tamanho da página que o homr usa (1920 px de largura). */
const segBatches = (p: Page) => Math.ceil((Math.ceil((1920 * p.height) / p.width / 320) * 6) / SEGNET_BATCH);

function startWorker(webgpu: boolean): Promise<void> {
  if (worker && workerGpu === webgpu) return Promise.resolve();
  worker?.terminate();
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  workerGpu = webgpu;
  const w = worker;
  return new Promise((resolve, reject) => {
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "status") setStatus(m.text);
      else if (m.type === "download") {
        ui.progress.value = m.total ? m.done / m.total : 0;
        setStatus(`Baixando os modelos: ${(m.done / 2 ** 20).toFixed(0)} de ${(m.total / 2 ** 20).toFixed(0)} MB…`);
      } else if (m.type === "ready") {
        log(`pronto (WebGPU na segmentação: ${m.webgpu ? "sim" : "não"}; lote ${SEGNET_BATCH})`);
        resolve();
      } else if (m.type === "error") reject(new Error(m.message));
    };
    w.postMessage({ type: "init", modelBase: new URL("omr-models/", document.baseURI).href, webgpu, segnetBatch: SEGNET_BATCH } satisfies ToWorker);
  });
}

function setStatus(text: string) {
  ui.status.textContent = text;
}

function recognizePage(i: number): Promise<void> {
  const p = pages[i];
  const total = segBatches(p);
  let staffs = 0;
  let encoders = 0;
  const show = (frac: number, text: string) => {
    ui.progress.value = (i + Math.min(frac, 1)) / pages.length;
    p.state = text;
    setStatus(`Página ${i + 1} de ${pages.length}: ${text}`);
    render();
  };
  show(0, "preparando…");
  return new Promise((resolve, reject) => {
    worker!.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "run" && m.kind === "segnet") show((0.5 * m.runs) / total, `lendo símbolos ${Math.min(m.runs, total)}/${total}`);
      else if (m.type === "staffs") {
        staffs = m.count;
        show(0.5, `${staffs} pautas encontradas`);
      } else if (m.type === "run" && m.kind === "encoder") {
        encoders++;
        show(0.5 + (0.5 * (encoders - 1)) / Math.max(staffs, 1), `lendo a pauta ${encoders} de ${staffs}`);
      } else if (m.type === "result") {
        p.xml = m.xml;
        p.stats = m.stats;
        p.state = `✓ ${(m.stats.totalMs / 1000).toFixed(0)} s`;
        render();
        resolve();
      } else if (m.type === "error") {
        p.state = "erro";
        p.error = true;
        render();
        reject(new Error(m.message));
      }
    };
    void p.file.arrayBuffer().then((bytes) => {
      const ext = (p.file.name.match(/\.(\w+)$/)?.[1] ?? "jpg").toLowerCase();
      worker!.postMessage({ type: "page", index: i, bytes, ext } satisfies ToWorker, [bytes]);
    });
  });
}

function describe(i: number, s: PageStats) {
  const sec = (ms: number) => (ms / 1000).toFixed(1) + " s";
  const py = s.python ?? {};
  log(
    `página ${i + 1}: total ${sec(s.totalMs)} | segmentação ${sec(s.segnet.ms)} (${s.segnet.runs} lotes, ${s.backend.segnet}) ` +
      `| encoder ${sec(s.encoder.ms)} (${s.encoder.runs}) | decoder ${sec(s.decoder.ms)} (${s.decoder.runs} passos) ` +
      `| criar sessões ${sec(s.segnet.createMs + s.encoder.createMs + s.decoder.createMs)} ` +
      `| Python: detecção ${sec(1000 * (py.detect_s ?? 0))}, pautas ${sec(1000 * (py.transformer_s ?? 0))} | heap do Pyodide ${s.pyHeapMB} MB`,
  );
}

ui.run.onclick = async () => {
  running = true;
  result = "";
  ui.resultBox.hidden = true;
  ui.progressBox.hidden = false;
  ui.run.disabled = true;
  ui.stop.hidden = false;
  for (const p of pages) [p.state, p.error, p.xml] = ["", false, undefined];
  render();
  const t0 = performance.now();
  try {
    await startWorker(ui.webgpu.checked);
    log(`preparo: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const t1 = performance.now();
    for (let i = 0; i < pages.length; i++) {
      await recognizePage(i);
      describe(i, pages[i].stats!);
    }
    result = mergePages(
      pages.map((p) => p.xml!),
      ui.title.value.trim(),
    );
    const secs = (performance.now() - t1) / 1000;
    log(`${pages.length} página(s) em ${secs.toFixed(0)} s (${(secs / pages.length).toFixed(0)} s por página)`);
    ui.progress.value = 1;
    setStatus(`Pronto: ${pages.length} página(s) em ${Math.round(secs / 60)} min ${Math.round(secs % 60)} s.`);
    ui.resultBox.hidden = false;
    (window as unknown as { omrResult: unknown }).omrResult = { xml: result, pages: pages.map((p) => p.stats), secs };
  } catch (err) {
    setStatus("Erro: " + (err as Error).message);
    log("erro: " + (err as Error).message);
    (window as unknown as { omrResult: unknown }).omrResult = { error: (err as Error).message };
  } finally {
    running = false;
    ui.stop.hidden = true;
    render();
  }
};

ui.stop.onclick = () => {
  worker?.terminate();
  worker = null;
  setStatus("Parado.");
  log("parado pelo usuário");
  running = false;
  ui.stop.hidden = true;
  for (const p of pages) if (!p.xml) p.state = "";
  render();
};

const fileName = () => (ui.title.value.trim() || "partitura").replace(/[\\/:*?"<>|]+/g, "_") + ".musicxml";

ui.download.onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([result], { type: "application/vnd.recordare.musicxml+xml" }));
  a.download = fileName();
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

ui.open.onclick = () => {
  try {
    localStorage.setItem(OMR_STORAGE_KEY, JSON.stringify({ title: ui.title.value.trim() || "Partitura reconhecida", xml: result }));
  } catch {
    alert("Não deu para passar a partitura ao Clavis (armazenamento cheio). Use “Baixar .musicxml”.");
    return;
  }
  location.href = CLAVIS_URL + "#omr";
};

$<HTMLAnchorElement>("source").href = SOURCE_URL;
$<HTMLAnchorElement>("clavis").href = CLAVIS_URL;
render();
