// Clavis OMR — worker do onnxruntime-web só com WebAssembly (o transformer, e a segmentação sem GPU).
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import * as ort from "onnxruntime-web/wasm";
import { serveOrt } from "./ort-serve";

serveOrt(ort);
