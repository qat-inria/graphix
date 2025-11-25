from perceval.backends._abstract_backends import AStrongSimulationBackend  # noqa: PLC2701

class SLOSBackend(AStrongSimulationBackend):
    def __init__(self, mask: str | list[str] | None, use_symbolic: bool = False) -> None: ...
