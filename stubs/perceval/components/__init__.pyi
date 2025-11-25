from perceval.components.abstract_processor import AProcessor
from perceval.components.experiment import Experiment
from perceval.components.linear_circuit import ACircuit, Circuit
from perceval.components.processor import Processor
from perceval.components.sources import Source
from perceval.utils.matrix import Matrix

class Unitary:
    def __init__(self, U: Matrix, name: str | None, use_polarization: bool = False) -> None: ...  # noqa: N803

class PS:
    def __init__(self, phase: float) -> None: ...

class BS:
    def __init__(self, theta: float, phi: float) -> None: ...

class PERM:
    def __init__(self, permutation: list[int]) -> None: ...

class Catalog:
    def __init__(self, path: str) -> None: ...
    def __getitem__(self, item_name: str) -> CatalogItem: ...

class CatalogItem:
    def build_processor(self) -> Processor: ...

catalog = Catalog("perceval.components.core_catalog")
