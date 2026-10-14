# Clavis OMR

Reconhece fotos de partitura (OMR) **no navegador**, sem servidor, e abre o resultado no
[Clavis](https://clavis.ccblondrina.org.br/), o professor virtual de instrumentos da CCB Londrina.
Tudo roda no aparelho de quem usa: nenhuma imagem é enviada.

Por baixo, é o [homr](https://github.com/liebharc/homr) 0.7.0 (Python) rodando no
[Pyodide](https://pyodide.org), com as redes neurais no [ONNX Runtime Web](https://onnxruntime.ai)
(WebAssembly; WebGPU opcional na segmentação). O resultado é o mesmo do homr nativo (~97% das alturas no
Hinário nº 5), em ~3–4 min por página num desktop (~2,5 min com WebGPU).

> **Protótipo.** Funciona no desktop. No celular ainda não foi medido, e a aba usa ~1,2–1,4 GB de memória.

## Como funciona

| Arquivo | O que faz |
|---|---|
| `index.html`, `src/main.ts`, `src/omr.css` | a página: escolher/fotografar as páginas, progresso, juntar as páginas (`src/merge.ts`), abrir no Clavis ou baixar o `.musicxml` |
| `src/worker.ts` | Web Worker: carrega o Pyodide (numpy, OpenCV, Pillow) do CDN, instala as **wheels do homr 0.7.0 e do musicxml 1.4 direto do PyPI** (conferidas pelo sha256) e roda as redes no onnxruntime-web |
| `src/py/onnxruntime/` | fachada Python do pacote `onnxruntime` que delega ao onnxruntime-web |
| `src/py/clavis_omr.py` | aplica o patch no homr instalado (as chamadas ao onnxruntime viram `await`) e reconhece uma página |
| `scripts/omr-models.mjs` | `npm run omr-models`: baixa os 3 modelos ONNX do homr (~150 MB) para `public/omr-models/` |

**Integração com o Clavis.** A página é publicada numa subpasta do Clavis (`<clavis>/omr/`), na mesma origem.
"Abrir no Clavis" grava `{ title, xml }` no `localStorage` (chave `clavis.omr`) e navega para `../#omr`; o Clavis
abre a partitura como um arquivo local. Outro endereço do Clavis: `VITE_CLAVIS_URL=... npm run build`.

**Modelos.** O GitHub Releases do homr não manda cabeçalhos CORS, então os modelos são servidos pelo mesmo site
(`omr-models/`). O worker os guarda na Cache API do navegador na primeira vez.

## Desenvolver

```bash
npm ci
npm run omr-models
npm run dev
```

Para testar o "Abrir no Clavis", publique na pasta `public/omr/` do Clavis (mesma origem):

```bash
npm run build -- --outDir ../clavis/public/omr --emptyOutDir
```

## Publicar

O `Dockerfile` do Clavis baixa uma tag deste repositório, roda `npm ci`, `npm run omr-models` e `npm run build`,
e copia o `dist/` para `/omr/`. Para lançar uma versão: mudar `version` no `package.json`, criar a tag
`v<versão>` e atualizar `OMR_VERSION` no `Dockerfile` do Clavis. O link "código-fonte" da página aponta para
essa tag.

## Licença

**GNU Affero General Public License v3.0 ou posterior** ([LICENSE](LICENSE)), como o homr, que este projeto
adapta e distribui ao navegador. Modificações sobre o homr 0.7.0 (feitas em 2026-10 para rodar no navegador):
veja a lista `PATCHES` em `src/py/clavis_omr.py`.

Componentes de terceiros carregados pela página: [homr](https://github.com/liebharc/homr) (AGPL-3.0),
[musicxml](https://pypi.org/project/musicxml/) (MIT), [Pyodide](https://pyodide.org) (MPL-2.0) com numpy,
OpenCV e Pillow, [ONNX Runtime Web](https://onnxruntime.ai) (MIT).

O Clavis, que só lê o MusicXML gerado por esta página, é um programa separado (licença ISC).
