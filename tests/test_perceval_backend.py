from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from perceval import Source  # type: ignore  # noqa: PGH003

from graphix.fundamentals import Plane
from graphix.sim.perceval import PercevalBackend, PercevalState
from graphix.states import PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.pattern import Pattern

SOURCES: list[Source] = [Source(emission_probability=1,
                multiphoton_component=0,
                indistinguishability=1),
            Source(emission_probability=0.9,
                multiphoton_component=0,
                indistinguishability=1),
            Source(emission_probability=1.0,
                multiphoton_component=0,
                indistinguishability=0.9),
            Source(emission_probability=1.0,
                multiphoton_component=0.1,
                indistinguishability=1.0),
            Source(emission_probability=0.9,
                multiphoton_component=0.1,
                indistinguishability=0.9)]


@pytest.mark.parametrize("source", SOURCES)
def test_init_success(hadamardpattern: Pattern, source: Source) -> None:
    """Test that the PercevalBackend can be initialized successfully."""
    backend = PercevalBackend()
    backend.add_nodes(hadamardpattern.input_nodes)
    backend.set_source(source)
    state = PercevalState(source=source)
    assert np.allclose(state.state, backend.state.state)


@pytest.mark.parametrize("source", SOURCES)
def test_init_fail(hadamardpattern: Pattern, source: Source, fx_rng: Generator) -> None:
    rand_angle: np.ndarray = fx_rng.random(2) * 2 * np.pi
    rand_plane: np.ndarray = fx_rng.choice(np.array(Plane), 2)
    state: PlanarState = PlanarState(rand_plane[0], rand_angle[0])
    state2: PlanarState = PlanarState(rand_plane[1], rand_angle[1])
    backend: PercevalBackend = PercevalBackend()
    backend.set_source(source)
    with pytest.raises(ValueError):
        backend.add_nodes(hadamardpattern.input_nodes, data=[state, state2])
