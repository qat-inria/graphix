from perceval.components.linear_circuit import ACircuit
from perceval.utils.noise_model import NoiseModel

class Experiment:
    def __init__(self, m_circuit: int | ACircuit | None, noise: NoiseModel | None, name: str = "Experiment") -> None: ...
