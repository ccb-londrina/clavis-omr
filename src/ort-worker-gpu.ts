// Clavis OMR — worker do onnxruntime-web com WebGPU (a segmentação, quando o navegador tem GPU).
// Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md e LICENSE).
import * as ort from "onnxruntime-web/webgpu";
import { serveOrt } from "./ort-serve";

serveOrt(ort);
