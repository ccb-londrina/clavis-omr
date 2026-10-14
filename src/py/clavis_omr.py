"""Clavis OMR: roda o homr 0.7.0 no Pyodide, com as redes no onnxruntime-web.

O homr vem do PyPI sem alteração (o worker instala a wheel). Antes de importá-lo, `patch_homr`
aplica as mudanças abaixo nos arquivos instalados, para que as chamadas ao onnxruntime (que no
navegador são assíncronas) virem `await` e a cadeia de funções até elas vire `async def`.
Cada trecho precisa aparecer exatamente o número de vezes indicado: com outra versão do homr o
patch falha em vez de rodar algo pela metade.

Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md na raiz do repositório). Modificações sobre o homr 0.7.0
(https://github.com/liebharc/homr, AGPL-3.0), feitas em 2026-10 para rodar no navegador:
- inferência assíncrona (lista PATCHES);
- título (OCR com rapidocr) desligado: não há threads nem rapidocr no Pyodide;
- lote da segmentação ajustável (SEGNET_BATCH_SIZE).
"""

import os
import sys
import types
from time import perf_counter

# (arquivo, trecho original, trecho novo, nº de ocorrências)
PATCHES: list[tuple[str, str, str, int]] = [
    # --- segmentação
    ("segmentation/inference_segnet.py",
     "    def run(self, input_data: NDArray) -> NDArray:",
     "    async def run(self, input_data: NDArray) -> NDArray:", 1),
    ("segmentation/inference_segnet.py",
     "        self.model.run_with_iobinding(self.io_binding)",
     "        await self.model.run_with_iobinding(self.io_binding)", 1),
    ("segmentation/inference_segnet.py", "\ndef inference(", "\nasync def inference(", 1),
    ("segmentation/inference_segnet.py",
     "batch_out = model.run(", "batch_out = await model.run(", 2),
    ("segmentation/inference_segnet.py", "\ndef extract(", "\nasync def extract(", 1),
    ("segmentation/inference_segnet.py",
     "clefs_keys = inference(\n", "clefs_keys = await inference(\n", 1),
    # --- main: segmentação → detecção das pautas
    ("main.py", "\ndef get_predictions(", "\nasync def get_predictions(", 1),
    ("main.py", "    result = extract(\n", "    result = await extract(\n", 1),
    ("main.py", "\ndef load_and_preprocess_predictions(",
     "\nasync def load_and_preprocess_predictions(", 1),
    ("main.py", "predictions = get_predictions(", "predictions = await get_predictions(", 1),
    ("main.py", "\ndef detect_staffs_in_image(", "\nasync def detect_staffs_in_image(", 1),
    ("main.py", "predictions, debug = load_and_preprocess_predictions(",
     "predictions, debug = await load_and_preprocess_predictions(", 1),
    ("main.py", "title_future = detect_title(debug, staffs[0])", "title_future = None", 1),
    # lote da segmentação ajustável (o homr usa 8; lotes menores gastam menos memória)
    ("main.py", 'os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"\n',
     'os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"\nSEGNET_BATCH_SIZE = 8\n', 1),
    ("main.py", "        use_gpu_inference=segnet_use_gpu,\n    )",
     "        use_gpu_inference=segnet_use_gpu,\n        batch_size=SEGNET_BATCH_SIZE,\n    )", 1),
    # --- transformer: pauta → encoder + decoder
    ("staff_parsing.py", "\ndef parse_staff_image(", "\nasync def parse_staff_image(", 1),
    ("staff_parsing.py", "result = parse_staff_tromr(", "result = await parse_staff_tromr(", 1),
    ("staff_parsing.py", "\ndef parse_staffs(", "\nasync def parse_staffs(", 1),
    ("staff_parsing.py", "result_staff = parse_staff_image(",
     "result_staff = await parse_staff_image(", 1),
    ("staff_parsing_tromr.py", "\ndef parse_staff_tromr(", "\nasync def parse_staff_tromr(", 1),
    ("staff_parsing_tromr.py", "return predict_best(", "return await predict_best(", 1),
    ("staff_parsing_tromr.py", "\ndef predict_best(", "\nasync def predict_best(", 1),
    ("staff_parsing_tromr.py", "result = inference.predict(", "result = await inference.predict(", 1),
    ("transformer/staff2score.py", "    def predict(self", "    async def predict(self", 1),
    ("transformer/staff2score.py", "context = self.encoder.generate(x)",
     "context = await self.encoder.generate(x)", 1),
    ("transformer/staff2score.py", "out = self.decoder.generate(", "out = await self.decoder.generate(", 1),
    ("transformer/encoder_inference.py", "    def generate(self", "    async def generate(self", 1),
    ("transformer/encoder_inference.py", "        self.encoder.run_with_iobinding(",
     "        await self.encoder.run_with_iobinding(", 1),
    ("transformer/decoder_inference.py", "    def generate(\n", "    async def generate(\n", 1),
    ("transformer/decoder_inference.py", "            self.net.run_with_iobinding(",
     "            await self.net.run_with_iobinding(", 1),
]


def _stub_modules() -> None:
    """rapidocr (título) e requests (download dos modelos) não são usados no navegador."""
    rapidocr = types.ModuleType("rapidocr")

    class RapidOCR:  # noqa: D401
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OCR do título desligado no navegador")

    rapidocr.RapidOCR = RapidOCR
    requests = types.ModuleType("requests")
    requests.exceptions = types.SimpleNamespace(RequestException=Exception)

    def get(*args, **kwargs):
        raise RuntimeError("sem requests no navegador: os modelos vêm do worker")

    requests.get = get
    sys.modules.setdefault("rapidocr", rapidocr)
    sys.modules.setdefault("requests", requests)


def patch_homr(site_packages: str) -> None:
    root = os.path.join(site_packages, "homr")
    by_file: dict[str, list[tuple[str, str, int]]] = {}
    for file, old, new, count in PATCHES:
        by_file.setdefault(file, []).append((old, new, count))
    for file, reps in by_file.items():
        path = os.path.join(root, file)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for old, new, count in reps:
            found = text.count(old)
            if found != count:
                raise RuntimeError(f"patch do homr: {file}: esperava {count}× {old!r}, achei {found}")
            text = text.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    _stub_modules()


async def recognize(image_path: str, report, segnet_batch: int = 8) -> str:
    """Reconhece uma página (imagem) e devolve o MusicXML. `report(etapa, valor)` informa o progresso."""
    import homr.main
    from homr.main import ProcessingConfig, detect_staffs_in_image
    from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
    from homr.staff_parsing import _ensure_same_number_of_staffs, parse_staffs
    from homr.transformer.configs import Config

    homr.main.SEGNET_BATCH_SIZE = segnet_batch
    t0 = perf_counter()
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
    )
    multi_staffs, image, debug, _title = await detect_staffs_in_image(image_path, config)
    t1 = perf_counter()
    systems = _ensure_same_number_of_staffs(multi_staffs, image)
    report("staffs", len(systems) * len(systems[0].staffs))
    transformer_config = Config()
    transformer_config.use_gpu_inference = False  # modelos fp32 (os mesmos do homr na CPU)
    transformer_config.use_coreml_encoder = False
    result_staffs = await parse_staffs(debug, multi_staffs, image, config=transformer_config)
    t2 = perf_counter()
    xml = generate_xml(XmlGeneratorArguments(False, None, None), result_staffs, "")
    out = os.path.splitext(image_path)[0] + ".musicxml"
    xml.write(out)
    with open(out, encoding="utf-8") as f:
        text = f.read()
    report("python_times", {"detect_s": t1 - t0, "transformer_s": t2 - t1, "xml_s": perf_counter() - t2})
    return text
