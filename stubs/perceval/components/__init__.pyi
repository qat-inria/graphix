from perceval.processor import Processor
from perceval.utils.matrix import Matrix

from .sources import Source  # noqa: F401, TID252

class Unitary:
    def __init__(self, matrix: Matrix) -> None: ...

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

class Circuit:
    def __init__(self, m: int, name: str | None) -> None: ...

catalog = Catalog("perceval.components.core_catalog")
