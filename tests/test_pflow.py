from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np
import pytest
from numpy.random import Generator

from graphix.fundamentals import Plane
from graphix.generator import _pflow2pattern
from graphix.gflow import find_pauliflow
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pflow import OpenGraphIndex, _get_pflow_matrices, _get_reduced_adj, find_pflow
from graphix.states import PlanarState


class OpenGraphTestCase(NamedTuple):
    ogi: OpenGraphIndex
    radj: MatGF2
    flow_demand_mat: MatGF2
    order_demand_mat: MatGF2
    has_pflow: bool


def prepare_test_og() -> list[OpenGraphTestCase]:
    test_cases: list[OpenGraphTestCase] = []

    # Trivial open graph with pflow and nI = nO
    def get_og_1() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-1-(2)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 2)])
        inputs = [0]
        outputs = [2]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.5, Plane.YZ),  # Y
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_1()),
            radj=MatGF2([[1, 0], [0, 1]]),
            flow_demand_mat=MatGF2([[1, 0], [1, 1]]),
            order_demand_mat=MatGF2([[0, 0], [0, 0]]),
            has_pflow=True,
        )
    )

    # Non-trivial open graph without pflow and nI = nO
    def get_og_2() -> OpenGraph:
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
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.XZ),  # X
            3: Measurement(0.5, Plane.YZ),  # Y
            4: Measurement(0.5, Plane.YZ),  # Y
            5: Measurement(0.1, Plane.YZ),  # YZ
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_2()),
            radj=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            flow_demand_mat=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
            order_demand_mat=MatGF2(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            has_pflow=False,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_3() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs.

        Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        graph: nx.Graph[int] = nx.Graph(
            [(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]
        )
        inputs = [0]
        outputs = [5, 6]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.YZ),  # Y
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0, Plane.XZ),  # Z
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_3()),
            radj=MatGF2(
                [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
            ),
            flow_demand_mat=MatGF2(
                [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
            ),
            order_demand_mat=MatGF2(
                [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            ),
            has_pflow=True,
        )
    )
    return test_cases


class TestPflow:
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_reduced_adj(self, test_case: OpenGraphTestCase) -> None:
        ogi = test_case.ogi
        radj = _get_reduced_adj(ogi)

        assert radj == test_case.radj

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_pflow_matrices(self, test_case: OpenGraphTestCase) -> None:
        ogi = test_case.ogi
        flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

        assert flow_demand_matrix == test_case.flow_demand_mat
        assert order_demand_matrix == test_case.order_demand_mat

    # This test compares against the existing function for calculating the Pauli flow.
    # Eventually, we should make the test independent of other flow-finding functions
    @pytest.mark.skip(reason="Bug in `graphix.gflow.find_pauliflow`")
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.ogi.og
        # TODO: Refactor to take open graph as input
        graph = og.inside
        inputs = set(og.inputs)
        outputs = set(og.outputs)
        meas_planes = {i: m.plane for i, m in og.measurements.items()}
        meas_angles = {i: m.angle for i, m in og.measurements.items()}

        if len(outputs) > len(inputs):
            pass  # Not implented yet
        else:
            pflow = find_pflow(og)
            pflow_ref = find_pauliflow(
                graph=og.inside, iset=inputs, oset=outputs, meas_planes=meas_planes, meas_angles=meas_angles
            )

            if pflow is None:
                assert pflow_ref[0] is None
            else:
                pattern = _pflow2pattern(
                    graph=graph, angles=meas_angles, inputs=inputs, meas_planes=meas_planes, p=pflow[0], l_k=pflow[1]
                )
                pattern.reorder_output_nodes(outputs)

                # The method og.to_pattern() will first try to construct the pattern from a causal flow, then the general flow and finally the Pauli flow. In general the corresponding patterns might represent different unitaries, therefore we calculate the reference pattern with _pflow2pattern instead.

                pattern_ref = _pflow2pattern(
                    graph=graph,
                    angles=meas_angles,
                    inputs=inputs,
                    meas_planes=meas_planes,
                    p=pflow_ref[0],
                    l_k=pflow_ref[1],
                )
                pattern_ref.reorder_output_nodes(outputs)

                alpha = 2 * np.pi * fx_rng.random()
                state = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                state_ref = pattern_ref.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))

                assert np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow_determinism(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.ogi.og

        if len(og.outputs) > len(og.inputs):
            pass  # Not implemented yet
        else:
            pflow = find_pflow(og)

            if not test_case.has_pflow:
                assert pflow is None
            else:
                assert pflow is not None

                pattern = _pflow2pattern(
                    graph=og.inside,
                    inputs=set(og.inputs),
                    meas_planes={i: m.plane for i, m in og.measurements.items()},
                    angles={i: m.angle for i, m in og.measurements.items()},
                    p=pflow[0],
                    l_k=pflow[1],
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
