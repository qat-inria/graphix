from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.fundamentals import Plane
from graphix.generator import _gflow2pattern
from graphix.gflow_ import _get_subadj_matrices, find_gflow
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import NodeIndex
from graphix.states import PlanarState
from tests.conftest import fx_rng

if TYPE_CHECKING:
    from numpy.random import Generator


class OpenGraphTestCase(NamedTuple):
    og: OpenGraph
    has_gflow: bool


def get_og_from_rndcircuit(depth: int, nqubits: int) -> OpenGraph:
    """Return an open graph from a random circuit.

    Parameters
    ----------
    depth : int
        Circuit depth of the random circuits for generating open graphs.
    nqubits : int
        Number of qubits in the random circuits for generating open graphs. It controls the number of outputs.

    Returns
    -------
    OpenGraph
        Open graph with causal and gflow.
    """
    circuit = rand_circuit(nqubits, depth, fx_rng._fixture_function())
    pattern = circuit.transpile().pattern
    _, edges = pattern.get_graph()
    graph: nx.Graph[int] = nx.Graph(edges)

    angles = pattern.get_angles()
    planes = pattern.get_meas_plane()
    meas = {node: Measurement(angle, planes[node]) for node, angle in angles.items()}

    return OpenGraph(
        inside=graph,
        inputs=pattern.input_nodes,
        outputs=pattern.output_nodes,
        measurements=meas,
    )


def prepare_test_og() -> list[OpenGraphTestCase]:
    test_cases: list[OpenGraphTestCase] = []

    # Open graph with gflow
    def get_og_1() -> OpenGraph:
        """Return an open graph with gflow.

        Open graph reproduced from Fig. 2 in Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 3), (3, 4), (3, 5), (2, 4), (2, 5), (0, 5)])
        inputs = [0, 3, 2]
        outputs = [1, 4, 5]
        meas = {i: Measurement(0.1, Plane.XY) for i in [0, 2, 3]}
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_1(),
            has_gflow=True,
        )
    )

    # Open graph with extended gflow
    def get_og_2() -> OpenGraph:
        r"""Return an open graph with extended gflow.

        The returned graph has the following structure:

          [0]-[1]
          /|   |
        (4)|   |
          \|   |
           2 -(5)-3
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)])
        inputs = [0, 1]
        outputs = [4, 5]
        meas = {
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.3, Plane.XZ),
            3: Measurement(0.4, Plane.YZ),
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_2(),
            has_gflow=True,
        )
    )

    # Open graph without measurements
    def get_og_3() -> OpenGraph:
        r"""Return an open graph with extended gflow.

        The returned graph has the following structure:

        [(0)] - [(1)] - [(2)] - [(3)] - [(4)] - [(5)]
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        inputs = [0, 1, 2, 3, 4, 5]
        outputs = [0, 1, 2, 3, 4, 5]
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements={})

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_3(),
            has_gflow=True,
        )
    )

    # Open graph without gflow, inputs on XY
    def get_og_4() -> OpenGraph:
        """Return an open graph with that has Pauli flow but no gflow and equal number of outputs and inputs.

        The returned open graph has the following structure:

        [0]-2-5-(8)
            | |
            3-6
            | |
        [1]-4-7-(9)

        Adapted from Fig. 7 in D. E. Browne et al 2007 New J. Phys. 9 250.
        """
        edges = [
            (0, 2),
            (1, 4),
            (2, 3),
            (3, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 6),
            (6, 7),
            (5, 8),
            (7, 9),
        ]

        graph = nx.Graph(edges)

        inputs = [0, 1]
        outputs = [8, 9]
        meas = {i: Measurement(0, Plane.XY) for i in range(8)}
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_4(),
            has_gflow=False,
        )
    )

    # Open graph without gflow, inputs on XZ
    def get_og_5() -> OpenGraph:
        r"""Return an open graph without gflow.

        The returned graph has the following structure:

          [0]-[1]
          /|   |
        (4)|   |
          \|   |
           2 -(5)-3

        This graph is analogous to example 2, but it does not have gflow because some of the input nodes are measured on a plane that is not XY.
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)])
        inputs = [0, 1]
        outputs = [4, 5]
        meas = {
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XZ),
            2: Measurement(0.3, Plane.XZ),
            3: Measurement(0.4, Plane.YZ),
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_5(),
            has_gflow=False,
        )
    )

    # Open graph without gflow, inputs on YZ
    def get_og_6() -> OpenGraph:
        r"""Return an open graph without gflow.

        The returned graph has the following structure:

        [0]-(1)

        This graph has a trivial structure, but it does not have gflow because some of the input nodes are measured on a plane that is not XY.
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1)])
        inputs = [0]
        outputs = [1]
        meas = {
            0: Measurement(0.1, Plane.YZ),
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_6(),
            has_gflow=False,
        )
    )

    # Open graphs from random circuits
    test_cases.extend(
        OpenGraphTestCase(
            og=get_og_from_rndcircuit(depth=1, nqubits=2),
            has_gflow=True,
        )
        for _ in range(3)
    )

    return test_cases


class TestGflow:
    def test_find_get_a_matrix(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 3), (3, 4), (3, 5), (2, 4), (2, 5), (0, 5), (0, 3)])
        non_output_mapping = NodeIndex()
        non_output_mapping.extend([0, 3, 2])
        corr_candidates_mapping = NodeIndex()
        corr_candidates_mapping.extend([1, 4, 5])

        a_matrix, b_matrix = _get_subadj_matrices(graph, non_output_mapping, corr_candidates_mapping)
        a_matrix_ref = MatGF2([[1, 0, 1], [1, 1, 1], [0, 1, 1]])
        b_matrix_ref = MatGF2([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        assert a_matrix == a_matrix_ref
        assert b_matrix == b_matrix_ref

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_gflow_determinism(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        gflow = find_gflow(og)

        if not test_case.has_gflow:
            assert gflow is None
        else:
            assert gflow is not None

            pattern = _gflow2pattern(
                graph=og.inside,
                inputs=set(og.inputs),
                meas_planes={i: m.plane for i, m in og.measurements.items()},
                angles={i: m.angle for i, m in og.measurements.items()},
                g=gflow[0],
                l_k=gflow[1],
            )
            pattern.reorder_output_nodes(og.outputs)

            alpha = 2 * np.pi * fx_rng.random()
            for plane in {Plane.XY, Plane.XZ, Plane.YZ}:  # ensure no trivial input
                state_ref = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))

                n_shots = 5
                results = []
                for _ in range(n_shots):
                    state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
                    results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

                avg = sum(results) / n_shots

                assert avg == pytest.approx(1)
