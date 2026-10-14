"""Fachada do pacote `onnxruntime` para o Pyodide: as redes rodam no onnxruntime-web (JS).

Implementa só o que o homr 0.7.0 usa (InferenceSession + io_binding + OrtValue). O `run` do
onnxruntime-web devolve uma Promise, então `run_with_iobinding` aqui é uma corrotina: o patch em
`clavis_omr.py` troca as chamadas do homr por `await ...run_with_iobinding(...)`.

As redes rodam em outro worker (um por modelo). As saídas chegam como objetos do JS e só viram numpy
quando alguém pede `.numpy()`; o cache do decoder (32 tensores por passo) nem sai do worker do
onnxruntime: volta para ele como referência no passo seguinte.

Parte do Clavis OMR — AGPL-3.0-or-later (veja README.md na raiz do repositório).
"""

import os

import numpy as np

# Objeto JS com run(model, nomes_entrada, valores_entrada, nomes_saida) -> Promise<tensor[]>.
# Definido pelo worker (src/worker.ts) antes de importar o homr.
_bridge = None

_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def set_bridge(bridge) -> None:
    global _bridge  # noqa: PLW0603
    _bridge = bridge


def get_available_providers() -> list[str]:
    return ["CPUExecutionProvider"]


def set_default_logger_severity(level: int) -> None:
    pass


def preload_dlls(*args, **kwargs) -> None:
    pass


class NodeArg:
    def __init__(self, name: str) -> None:
        self.name = name


class OrtValue:
    """Um valor do onnxruntime: um array do numpy (entrada) ou um tensor do JS (saída)."""

    def __init__(self, array=None, tensor=None) -> None:
        self._array = array
        self._tensor = tensor

    @staticmethod
    def ortvalue_from_numpy(array, device: str = "cpu", device_id: int = 0) -> "OrtValue":
        return OrtValue(array=array)

    def numpy(self):
        if self._array is None:
            t = self._tensor
            if getattr(t, "data", None) is None:
                raise RuntimeError("tensor guardado no worker do onnxruntime: não dá para ler no Python")
            out = np.empty(tuple(int(d) for d in t.dims), dtype=_DTYPES[t.type])
            if out.size:
                t.data.assign_to(out)
            self._array = out
            self._tensor = None
        return self._array

    def _feed(self):
        return self._tensor if self._tensor is not None else np.ascontiguousarray(self._array)


class IOBinding:
    def __init__(self) -> None:
        self._inputs: dict[str, object] = {}
        self._outputs: dict[str, None] = {}
        self._results: list[OrtValue] = []

    def bind_cpu_input(self, name: str, array) -> None:
        self._inputs[name] = np.ascontiguousarray(array)

    def bind_ortvalue_input(self, name: str, value: OrtValue) -> None:
        self._inputs[name] = value._feed()

    def bind_output(self, name: str, device_type: str = "cpu", device_id: int = 0, *args) -> None:
        self._outputs[name] = None

    def get_outputs(self) -> list[OrtValue]:
        return self._results

    def clear_binding_inputs(self) -> None:
        self._inputs.clear()

    def clear_binding_outputs(self) -> None:
        self._outputs.clear()


class InferenceSession:
    """A sessão de verdade é criada no JS na primeira execução, a partir do nome do arquivo."""

    def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs) -> None:
        self._model = os.path.basename(str(path_or_bytes))

    def get_inputs(self) -> list[NodeArg]:
        return [NodeArg("input")]

    def get_outputs(self) -> list[NodeArg]:
        return [NodeArg("output")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def io_binding(self) -> IOBinding:
        return IOBinding()

    async def run_with_iobinding(self, iobinding: IOBinding, run_options=None) -> None:
        names = list(iobinding._inputs)
        values = list(iobinding._inputs.values())
        outputs = list(iobinding._outputs)
        result = await _bridge.run(self._model, names, values, outputs)
        iobinding._results = [OrtValue(tensor=t) for t in result]
