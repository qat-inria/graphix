from __future__ import annotations

import networkx as nx
from numpy.random import Generator

from graphix.fundamentals import Plane
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pflow import _get_pflow_matrices, _get_reduced_adj

from typing import NamedTuple

import pytest


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
    def test_get_plfow_matrices(self, test_case: OpenGraphTestCase) -> None:
        og = test_case.og
        flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(og, row_idx=test_case.non_output_idx, col_idx=test_case.non_input_idx)

        assert flow_demand_matrix == test_case.flow_demand_mat
        assert order_demand_matrix == test_case.order_demand_mat

