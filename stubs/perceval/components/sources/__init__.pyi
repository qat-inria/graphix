from perceval.distributions import SVDistribution
from perceval.utils.states import FockState

class Source:
    def __init__(self, *args, **kwargs) -> None: ...  # type: ignore[no-untyped-def]
    def generate_distribution(self, expected_input: FockState, prob_threshold: float = 0) -> SVDistribution: ...
