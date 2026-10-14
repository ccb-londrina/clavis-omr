import { readFileSync } from "node:fs";
import { defaultClientConditions, defineConfig } from "vite";

const { version } = JSON.parse(readFileSync("package.json", "utf8")) as { version: string };

export default defineConfig({
  // Publicado numa subpasta do Clavis (ex.: <clavis>/omr/): caminhos relativos.
  base: "./",
  define: { __OMR_VERSION__: JSON.stringify(version) },
  // onnxruntime-web sem o .wasm embutido no pacote: o worker o carrega do CDN (ort.env.wasm.wasmPaths).
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  // O worker é módulo ES (importa o Pyodide do CDN).
  worker: { format: "es" },
});
