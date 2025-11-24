from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from perceval import Source  # type: ignore  # noqa: PGH003

from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.pauli import Pauli
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
def test_init_success(hadamardpattern: Pattern, source: Source, fx_rng: Generator) -> None:
    """Test that the PercevalBackend can be initialized successfully."""
    backend = PercevalBackend(source=source)
    backend.add_nodes(hadamardpattern.input_nodes)
    vec = PercevalState(source=source)
    assert np.allclose(vec.state, backend.state.state)

@pytest.mark.parametrize("source", SOURCES)
def test_init_fail(hadamardpattern, source: Source, fx_rng: Generator) -> None:
    rand_angle = fx_rng.random(2) * 2 * np.pi
    rand_plane = fx_rng.choice(np.array(Plane), 2)
    state = PlanarState(rand_plane[0], rand_angle[0])
    state2 = PlanarState(rand_plane[1], rand_angle[1])
    with pytest.raises(ValueError):
        PercevalBackend(source).add_nodes(hadamardpattern.input_nodes, data=[state, state2])

@pytest.mark.parametrize("source", SOURCES)
def test_deterministic_measure_one(fx_rng: Generator):
    # plus state & zero state (default), but with tossed coins
    for _ in range(10):
        backend = PercevalBackend(source)
        coins = [fx_rng.choice([0, 1]), fx_rng.choice([0, 1])]
        expected_result = sum(coins) % 2
        states = [
            Pauli.X.eigenstate(coins[0]),
            Pauli.Z.eigenstate(coins[1]),
        ]
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)
        backend.entangle_nodes(edge=(nodes[0], nodes[1]))
        measurement = Measurement(0, Plane.XY)
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement)
        assert result == expected_result

@pytest.mark.parametrize("source", SOURCES)
def test_deterministic_measure(source: Source):
    """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
    for _ in range(10):
        # plus state (default)
        backend = PercevalBackend(source)
        n_neighbors = 10
        states = [Pauli.X.eigenstate()] + [Pauli.Z.eigenstate() for i in range(n_neighbors)]
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)
        for i in range(1, n_neighbors + 1):
            backend.entangle_nodes(edge=(nodes[0], i))
        measurement = Measurement(0, Plane.XY)
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement)
        assert result == 0
        assert list(backend.node_index) == list(range(1, n_neighbors + 1))

@pytest.mark.parametrize("source", SOURCES)
def test_deterministic_measure_many(source: Source):
    """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
    for _ in range(10):
        # plus state (default)
        backend = PercevalBackend(source)
        n_traps = 5
        n_neighbors = 5
        n_whatever = 5
        traps = [Pauli.X.eigenstate() for _ in range(n_traps)]
        dummies = [Pauli.Z.eigenstate() for _ in range(n_neighbors)]
        others = [Pauli.I.eigenstate() for _ in range(n_whatever)]
        states = traps + dummies + others
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)
        for dummy in nodes[n_traps : n_traps + n_neighbors]:
            for trap in nodes[:n_traps]:
                backend.entangle_nodes(edge=(trap, dummy))
            for other in nodes[n_traps + n_neighbors :]:
                backend.entangle_nodes(edge=(other, dummy))
        # Same measurement for all traps
        measurement = Measurement(0, Plane.XY)
        for trap in nodes[:n_traps]:
            node_to_measure = trap
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == 0
        assert list(backend.node_index) == list(range(n_traps, n_neighbors + n_traps + n_whatever))

@pytest.mark.parametrize("source", SOURCES)
def test_deterministic_measure_with_coin(source: Source, fx_rng: Generator):
    """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.

    We add coin toss to that.
    """
    for _ in range(10):
        # plus state (default)
        backend = PercevalBackend(source)
        n_neighbors = 10
        coins = [fx_rng.choice([0, 1])] + [fx_rng.choice([0, 1]) for _ in range(n_neighbors)]
        expected_result = sum(coins) % 2
        states = [Pauli.X.eigenstate(coins[0])] + [Pauli.Z.eigenstate(coins[i + 1]) for i in range(n_neighbors)]
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)
        for i in range(1, n_neighbors + 1):
            backend.entangle_nodes(edge=(nodes[0], i))
        measurement = Measurement(0, Plane.XY)
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement)
        assert result == expected_result
        assert list(backend.node_index) == list(range(1, n_neighbors + 1))
