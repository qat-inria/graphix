from perceval.backends._abstract_backends import ABackend, AStrongSimulationBackend
from perceval.backends._slos import SLOSBackend

class BackendFactory:
    def __init__(self) -> None: ...
    def get_backend(self, backend_name: str, **kwargs) -> ABackend: ...  # type: ignore[no-untyped-def]
