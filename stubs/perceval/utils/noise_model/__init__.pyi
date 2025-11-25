class NoiseModel:
    def __init__(self,
                 rightness: float | None,
                 indistinguishability: float | None,
                 g2: float | None,
                 g2_distinguishable: bool | None,
                 transmittance: float | None,
                 phase_imprecision: float | None,
                 phase_error: float | None) -> None: ...
