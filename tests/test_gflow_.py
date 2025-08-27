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
from graphix.sim.base_backend import NodeIndex
from graphix.states import PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator


class OpenGraphTestCase(NamedTuple):
    og: OpenGraph
    has_gflow: bool


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

    # Open graph without gflow
    def get_og_3() -> OpenGraph:
        """Return an open graph without Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
        inputs = [1, 0]
        outputs = [6, 7]
        meas = {
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XZ),
            2: Measurement(0.45, Plane.XZ),
            3: Measurement(0.35, Plane.YZ),
            4: Measurement(0.75, Plane.YZ),
            5: Measurement(0.1, Plane.YZ),
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            og=get_og_3(),
            has_gflow=False,
        )
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
            state_ref = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))

            n_shots = 5
            results = []
            for _ in range(n_shots):
                state = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

            avg = sum(results) / n_shots

            assert avg == pytest.approx(1)
