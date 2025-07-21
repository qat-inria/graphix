from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pflow import _get_pflow_matrices

if TYPE_CHECKING:
    from numpy.random import Generator


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
    

    def get_graph_pflow_unequal_io(self, fx_rng: Generator) -> OpenGraph: