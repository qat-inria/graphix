import networkx as nx
import numpy as np

from graphix.fundamentals import Plane
from graphix.gflow import get_pauli_nodes
from graphix.linalg import MatGF2
from graphix.opengraph import OpenGraph


def _get_reduced_adj(og: OpenGraph) -> tuple[MatGF2, list[int], list[int]]:
    r"""Return reduced adjacency matrix (RAdj) of the input open graph.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose RAdj is to be computed.

    Returns
    -------
    adj_red: MatGF2
        Reduced adjacency matrix.
    row_nodes: list[int]
        An ordered list of the non-output nodes according to the rows of `adj_red`.
    col_nodes: list[int]
        An ordered list of the non-input nodes according to the columns of `adj_red`.

    Notes
    -----
    The adjacency matrix of a graph :math:`Adj_G` is a :math:`n \times n` matrix

    The RAdj matrix of an open graph OG is a :math:`(n - n_O) \times (n - n_I)` submatrix of :math:`Adj_G` constructed by removing the output rows and input columns of of :math:`Adj_G`.

    See Definition 3.3 in Mitosek and Backens, 2024 (arXiv:2410.23439)
    """
    graph = og.inside
    inputs = og.inputs
    outputs = og.outputs
    nodes = list(graph.nodes)
    row_nodes = nodes.copy()
    col_nodes = nodes.copy()

    adj = nx.adjacency_matrix(graph, nodelist=nodes, dtype=np.int64)
    adj_red = MatGF2(adj.toarray())

    for i in inputs:
        idx = nodes.index(i)
        col_nodes.pop(idx)
        adj_red.remove_col(idx)

    for o in outputs:
        idx = nodes.index(o)
        row_nodes.pop(idx)
        adj_red.remove_row(idx)

    return adj_red, row_nodes, col_nodes


def _get_pflow_matrices(og: OpenGraph) -> tuple[MatGF2, MatGF2]:
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
    flow_demand_matrix, row_nodes, col_nodes = _get_reduced_adj(og)
    order_demand_matrix = flow_demand_matrix.copy()

    inputs_set = set(og.inputs)
    meas = og.measurements

    # TODO: integrate pauli measurements in open graphs
    meas_planes = {i: m.plane for i, m in meas.items()}
    meas_angles = {i: m.angle for i, m in meas.items()}
    lx, ly, lz = get_pauli_nodes(meas_planes, meas_angles)

    for i, v in enumerate(row_nodes):
        if v in lz or meas_planes[v] in {Plane.YZ, Plane.XZ}:
            # Set row corresponding to node v to 0
            flow_demand_matrix.remove_row(row=i)
            flow_demand_matrix.add_row(row=i)
        if (v in {*ly, *lz} or meas_planes[v] in {Plane.YZ, Plane.XZ}) and i not in inputs_set:
            # Set M_{v, v} = 0
            j = col_nodes.index(v)
            flow_demand_matrix.data[i, j] = 1
        if v in {*lx, *ly, *lz} or meas_planes[v] == Plane.XY:
            # Set row corresponding to node v to 0
            order_demand_matrix.remove_row(row=i)
            order_demand_matrix.add_row(row=i)
        if meas_planes[v] in {Plane.XY, Plane.XZ} and i not in inputs_set:
            # Set N_{v, v} = 0
            j = col_nodes.index(v)
            order_demand_matrix.data[i, j] = 1

    return flow_demand_matrix, order_demand_matrix
