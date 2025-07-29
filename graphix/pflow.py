from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from graphix.fundamentals import Axis, Plane
from graphix.gflow import get_pauli_nodes
from graphix.linalg import MatGF2
from graphix.sim.base_backend import NodeIndex

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphix.opengraph import OpenGraph


class OpenGraphIndex:
    """A class for managing the mapping between node numbers of a given open graph and matrix indices in the Pauli flow finding algorithm.

    It reuses the class :class:`graphix.sim.base_backend.NodeIndex` introduced for managing the mapping between node numbers and qubit indices in the internal state of the backend.

    Attributes
    ----------
        og (OpenGraph)
        non_inputs (NodeIndex) : Mapping between matrix indices and non-input nodes (labelled with integers).
        non_outputs (NodeIndex) : Mapping between matrix indices and non-output nodes (labelled with integers).
    """

    def __init__(self, og: OpenGraph) -> None:
        self.og = og
        nodes = set(og.inside.nodes)

        # Nodes don't need to be sorted. We do it for debugging purposes, so we can check the matrices in intermediate steps of the algorithm.

        nodes_non_input = sorted(nodes - set(og.inputs))
        nodes_non_output = sorted(nodes - set(og.outputs))

        self.non_inputs = NodeIndex()
        self.non_inputs.extend(nodes_non_input)

        self.non_outputs = NodeIndex()
        self.non_outputs.extend(nodes_non_output)


def _get_reduced_adj(ogi: OpenGraphIndex) -> MatGF2:
    r"""Return reduced adjacency matrix (RAdj) of the input open graph.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose RAdj is to be computed.

    Returns
    -------
    adj_red: MatGF2
        Reduced adjacency matrix.

    Notes
    -----
    The adjacency matrix of a graph :math:`Adj_G` is a :math:`n \times n` matrix

    The RAdj matrix of an open graph OG is an :math:`(n - n_O) \times (n - n_I)` submatrix of :math:`Adj_G` constructed by removing the output rows and input columns of of :math:`Adj_G`.

    See Definition 3.3 in Mitosek and Backens, 2024 (arXiv:2410.23439)
    """
    graph = ogi.og.inside
    row_tags = ogi.non_outputs
    col_tags = ogi.non_inputs

    adj_red = MatGF2(np.zeros((len(row_tags), len(col_tags)), dtype=np.int64))

    for n1, n2 in graph.edges:
        if n1 in row_tags and n2 in col_tags:
            i, j = row_tags.index(n1), col_tags.index(n2)
            adj_red.data[i, j] = 1
        if n2 in row_tags and n1 in col_tags:
            i, j = row_tags.index(n2), col_tags.index(n1)
            adj_red.data[i, j] = 1

    return adj_red


def _get_pflow_matrices(ogi: OpenGraphIndex) -> tuple[MatGF2, MatGF2]:
    r"""Construct flow-demand and order-demand matrices.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose flow-demand and order-demand matrices are to be computed.

    Returns
    -------
    flow_demand_matrix: MatGF2
    order_demand_matrix: MatGF2

    Notes
    -----
    See Definitions 3.4 and 3.5, and Algorithm 1 in Mitosek and Backens, 2024 (arXiv:2410.23439)
    """
    flow_demand_matrix = _get_reduced_adj(ogi)
    order_demand_matrix = flow_demand_matrix.copy()

    inputs_set = set(ogi.og.inputs)
    meas = ogi.og.measurements

    row_tags = ogi.non_outputs
    col_tags = ogi.non_inputs

    # TODO: integrate pauli measurements in open graphs
    meas_planes = {i: m.plane for i, m in meas.items()}
    meas_angles = {i: m.angle for i, m in meas.items()}
    lx, ly, lz = get_pauli_nodes(meas_planes, meas_angles)
    meas_plane_axis: dict[int, Plane | Axis] = {}
    for i, plane in meas_planes.items():
        if i in lx:
            meas_plane_axis[i] = Axis.X
        elif i in ly:
            meas_plane_axis[i] = Axis.Y
        elif i in lz:
            meas_plane_axis[i] = Axis.Z
        else:
            meas_plane_axis[i] = plane

    for v in row_tags:  # v is a node tag
        i = row_tags.index(v)
        plane_axis_v = meas_plane_axis[v]

        if plane_axis_v in {Plane.YZ, Plane.XZ, Axis.Z}:
            flow_demand_matrix.data[i, :] *= 0  # Set row corresponding to node v to 0
        if plane_axis_v in {Plane.YZ, Plane.XZ, Axis.Y, Axis.Z} and v not in inputs_set:
            j = col_tags.index(v)
            flow_demand_matrix.data[i, j] = 1  # Set element (v, v) = 0
        if plane_axis_v in {Plane.XY, Axis.X, Axis.Y, Axis.Z}:
            order_demand_matrix.data[i, :] *= 0  # Set row corresponding to node v to 0
        if plane_axis_v in {Plane.XY, Plane.XZ} and v not in inputs_set:
            j = col_tags.index(v)
            order_demand_matrix.data[i, j] = 1  # Set element (v, v) = 1

    return flow_demand_matrix, order_demand_matrix


def _find_pflow_simple(ogi: OpenGraphIndex) -> tuple[MatGF2, MatGF2, nx.DiGraph[int]] | None:
    r"""Construct correction-function matrix :math:`C` and product of the order-demand matrix :math:`N` and the correction-function matrix, :math:`NC`.

    Parameters
    ----------
    og : OpenGraph
        Open graph for which :math:`C` and :math:`NC` are to be computed.

    Returns
    -------
    correction_matrix: MatGF2
        Matrix encoding the correction function.
    ordering_matrix: MatGF2
        Matrix enconding the partial ordering between nodes.
    dag: nx.DiGraph[int]
        Directed acyclical graph represented by the ordering matrix.

    or `None`
        if the input open graph does not have flow.

    Notes
    -----
    - The correction matrix :math:`C` is an :math:`(n - n_I) \times (n - n_O)` matrix related to the correction function :math:`c(v) = \{u \in \overline{I}|C_{u,v} = 1\}`, where :math:`\overline{I}` are the non-input nodes of `og`.

    - The Pauli flow's ordering :math:`<_c` is the transitive closure of :math:`\lhd_c`, where the latter is related to :math:`NC` as :math:`v \lhd_c w \Leftrightarrow (NC)_{w,v} = 1`, for :math:`v, w, \in \overline{O}` two non-output nodes of `og`.

    - The function returns `None` when:
        - The flow-demand matrix of `og` is not invertible.
        - The matrix :math:`NC` is not a DAG.
     then, `og` does not have Pauli flow.

    See Definitions 3.4, 3.5 and 3.6, Lemma 3.12, Theorem 3.1, and Algorithm 2 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

    correction_matrix = flow_demand_matrix.right_inverse()  # C matrix

    if correction_matrix is None:
        return None  # The flow-demand matrix is not invertible, therefore there's no flow.

    ordering_matrix = order_demand_matrix @ correction_matrix  # NC matrix

    # NetworkX uses the convention that a non-zero A(i,j) element represents a link i -> j.
    # We use the opposite convention, hence the transpose.
    dag = nx.from_numpy_array(ordering_matrix.data.T, create_using=nx.DiGraph)

    if not nx.is_directed_acyclic_graph(dag):
        return None  # The NC matrix is not a DAG, therefore there's no flow.

    return correction_matrix, ordering_matrix, dag


def _find_pflow_general(ogi: OpenGraphIndex) -> tuple[MatGF2, MatGF2, nx.DiGraph[int]] | None:
    pass


def _algebraic2pflow(
    ogi: OpenGraphIndex,
    correction_matrix: MatGF2,
    dag: nx.DiGraph[int],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    r"""Transform a Pauli flow in its algebraic form (correction matrix and DAG) into a Pauli flow in its standard form (correction function and partial order).

    Parameters
    ----------
    og: OpenGraph
        Open graph whose Pauli flow is being calculated.
    correction_matrix: MatGF2
        Matrix encoding the correction function.
    dag: nx.DiGraph[int]
        Directed acyclical graph represented by the ordering matrix.

    Returns
    -------
    pf: dict[int, set[int]]
        Pauli flow correction function. pf[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    Notes
    -----
    - The correction matrix :math:`C` is an :math:`(n - n_I) \times (n - n_O)` matrix related to the correction function :math:`c(v) = \{u \in \overline{I}|C_{u,v} = 1\}`, where :math:`\overline{I}` are the non-input nodes of `og`. In other words, the column :math:`v` of :math:`C` encodes the correction set of :math:`v`, :math:`c(v)`.

    - The Pauli flow's ordering :math:`<_c` is the transitive closure of :math:`\lhd_c`, where the latter is related to the adjacency matrix of `dag` :math:`NC` as :math:`v \lhd_c w \Leftrightarrow (NC)_{w,v} = 1`, for :math:`v, w, \in \overline{O}` two non-output nodes of `og`.

    See Definition 3.6, Lemma 3.12, and Theorem 3.1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    row_tags = ogi.non_inputs
    col_tags = ogi.non_outputs

    pf: dict[int, set[int]] = {}
    for node in col_tags:
        i = col_tags.index(node)
        correction_set = {row_tags[j] for j in correction_matrix.data[:, i].nonzero()[0]}
        pf[node] = correction_set

    # Output nodes are always in layer 0
    l_k = dict.fromkeys(ogi.og.outputs, 0)

    # If m >_c n, with >_c the flow order for two nodes m, n, then layer(n) > layer(m).
    # Therefore, we iterate the topological sort of the graph in _reverse_ order to obtain the order of measurements.

    for layer, idx in enumerate(reversed(list(nx.topological_generations(dag))), start=1):
        l_k.update({col_tags[i]: layer for i in idx})

    return pf, l_k


def find_pflow(og: OpenGraph) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    ni = len(og.inputs)
    no = len(og.outputs)

    if ni > no:
        return None

    ogi = OpenGraphIndex(og)
    if (pflow_algebraic := _find_pflow_simple(ogi) if ni == no else _find_pflow_general(ogi)) is None:
        return None

    correction_matrix, _, dag = pflow_algebraic

    return _algebraic2pflow(ogi, correction_matrix, dag)


def is_pflow_valid(og: OpenGraph, pf: Mapping[int, set[int]], l_k: Mapping[int, int]) -> bool:
    """Verify if a given Pauli flow is correct by checking the Pauli flow conditions (P1 - P9).

    See Definition 5 in Browne et al., NJP 9, 250 (2007).

    Returns
    -------
    pf: dict[int, set[int]]
        Pauli flow correction function. pf[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    """

    return False
