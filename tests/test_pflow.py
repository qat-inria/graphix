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
from graphix.pflow import _get_pflow_matrices, _get_reduced_adj, find_pflow
from graphix.states import PlanarState


class OpenGraphTestCase(NamedTuple):
    og: OpenGraph
    non_input_idx: dict[int, int]
    non_output_idx: dict[int, int]
    radj: MatGF2
    flow_demand_mat: MatGF2
    order_demand_mat: MatGF2
    has_plow: bool


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

    test_cases.extend(
        (
            OpenGraphTestCase(
                og=get_og_1(),
                non_input_idx={1: 0, 2: 1},
                non_output_idx={0: 0, 1: 1},
                radj=MatGF2([[1, 0], [0, 1]]),
                flow_demand_mat=MatGF2([[1, 0], [1, 1]]),
                order_demand_mat=MatGF2([[0, 0], [0, 0]]),
                has_plow=True,
            ),
            # Same as case 1 but with permuted columns
            OpenGraphTestCase(
                og=get_og_1(),
                non_input_idx={1: 1, 2: 0},
                non_output_idx={0: 0, 1: 1},
                radj=MatGF2([[0, 1], [1, 0]]),
                flow_demand_mat=MatGF2([[0, 1], [1, 1]]),
                order_demand_mat=MatGF2([[0, 0], [0, 0]]),
                has_plow=True,
            ),
            # Same as case 1 but with permuted rows
            OpenGraphTestCase(
                og=get_og_1(),
                non_input_idx={1: 0, 2: 1},
                non_output_idx={1: 0, 0: 1},
                radj=MatGF2([[0, 1], [1, 0]]),
                flow_demand_mat=MatGF2([[1, 1], [1, 0]]),
                order_demand_mat=MatGF2([[0, 0], [0, 0]]),
                has_plow=True,
            ),
        )
    )

    # Non-trivial open graph with pflow and nI = nO
    def get_og_2() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

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
            og=get_og_2(),
            non_input_idx=dict(zip(range(2, 8), range(6))),
            non_output_idx=dict(zip(range(6), range(6))),
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
            has_plow=True,
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
            og=get_og_3(),
            non_input_idx=dict(zip([1, 2, 3, 4, 5, 6], range(6))),
            non_output_idx=dict(zip([0, 1, 2, 3, 4], range(5))),
            radj=MatGF2(
                [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
            ),
            flow_demand_mat=MatGF2(
                [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
            ),
            order_demand_mat=MatGF2(
                [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            ),
            has_plow=True,
        )
    )
    return test_cases


class TestPflow:
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_reduced_adj(self, test_case: OpenGraphTestCase) -> None:
        og = test_case.og
        radj = _get_reduced_adj(og, row_idx=test_case.non_output_idx, col_idx=test_case.non_input_idx)

        assert radj == test_case.radj

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_pflow_matrices(self, test_case: OpenGraphTestCase) -> None:
        og = test_case.og
        flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(
            og, row_idx=test_case.non_output_idx, col_idx=test_case.non_input_idx
        )

        assert flow_demand_matrix == test_case.flow_demand_mat
        assert order_demand_matrix == test_case.order_demand_mat

    # This test compares against the existing function for calculating the Pauli flow.
    # Eventually, we should make the test independent of other flow-finding functions
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.og
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
