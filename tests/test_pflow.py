from __future__ import annotations
import enum

import networkx as nx
from numpy.random import Generator

from graphix.fundamentals import Plane
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pflow import _get_pflow_matrices, _get_reduced_adj


class TestPflow:
    # MEMO: this code is copy pasted from pflow pattern
    def get_graph_pflow_equal_io(self, fx_rng: Generator) -> OpenGraph:
        """Create a graph which has pflow but no gflow with equal number of inputs and outputs.

        Parameters
        ----------
        fx_rng : :class:`numpy.random.Generator`
            See graphix.tests.conftest.py

        Returns
        -------
        OpenGraph: :class:`graphix.opengraph.OpenGraph`
        """
        graph: nx.Graph[int] = nx.Graph(
            [(0, 2), (1, 4), (2, 3), (3, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (5, 8), (7, 9)]
        )
        inputs = [1, 0]
        outputs = [9, 8]

        # Heuristic mixture of Pauli and non-Pauli angles ensuring there's no gflow but there's pflow.
        meas_angles: dict[int, float] = {
            **dict.fromkeys(range(4), 0),
            **dict(zip(range(4, 8), (2 * fx_rng.random(4)).tolist())),
        }
        meas_planes = dict.fromkeys(range(8), Plane.XY)
        meas = {i: Measurement(angle, plane) for (i, angle), plane in zip(meas_angles.items(), meas_planes.values())}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    def get_graph_pflow_unequal_io(self) -> OpenGraph:
        """Create a graph which has pflow with unequal number of inputs and outputs. Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).

        Returns
        -------
        OpenGraph: :class:`graphix.opengraph.OpenGraph`
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)])
        inputs = [0]
        outputs = [5, 6]
        meas = {0: Measurement(0.1, Plane.XY), 1: Measurement(0.1, Plane.XZ), 2: Measurement(0.5, Plane.YZ), 3: Measurement(0.1, Plane.XY), 4: Measurement(0, Plane.XZ)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    def test_get_reduced_adj(self) -> None:
        og = self.get_graph_pflow_unequal_io()
        radj, row_idx, col_idx = _get_reduced_adj(og)

        radj_ref = MatGF2([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]])

        row_idx_ref = dict(zip([0, 1, 2, 3, 4], range(5)))
        col_idx_ref = dict(zip([1, 2, 3, 4, 5, 6], range(6)))

        for node_row, i_ref in row_idx_ref.items():
            for node_col, j_ref in col_idx_ref.items():
                i = row_idx[node_row]
                j = col_idx[node_col]

                assert radj.data[i, j] == radj_ref.data[i_ref, j_ref]

    def test_get_plfow_matrices_1(self) -> None:
        og = self.get_graph_pflow_unequal_io()

        flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(og)

        flow_demand_matrix_ref = MatGF2([[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]])
        order_demand_matrix_ref = MatGF2([[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

        assert flow_demand_matrix == flow_demand_matrix_ref
        assert order_demand_matrix == order_demand_matrix_ref
