/// <reference types="vite/client" />

/** Versão do package.json: o link "código-fonte" aponta para a tag v<versão>. */
declare const __OMR_VERSION__: string;

interface ImportMetaEnv {
  /** Endereço do Clavis, relativo à página (padrão "../": a página fica em <clavis>/omr/). */
  readonly VITE_CLAVIS_URL?: string;
}
