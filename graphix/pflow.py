from __future__ import annotations

import networkx as nx
import numpy as np

from graphix.fundamentals import Plane
from graphix.gflow import get_pauli_nodes
from graphix.linalg import MatGF2
from graphix.opengraph import OpenGraph


def _get_reduced_adj(og: OpenGraph) -> tuple[MatGF2, dict[int, int], dict[int, int]]:
    r"""Return reduced adjacency matrix (RAdj) of the input open graph.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose RAdj is to be computed.

    Returns
    -------
    adj_red: MatGF2
        Reduced adjacency matrix.
    row_idx: dict[int, int]
        Mapping between the non-output nodes (keys) and the rows of `adj_red` (values).
    col_idx: dict[int, int]
        Mapping between the non-input nodes (keys) and the columns of `adj_red` (values).

    Notes
    -----
    The adjacency matrix of a graph :math:`Adj_G` is a :math:`n \times n` matrix

    The RAdj matrix of an open graph OG is a :math:`(n - n_O) \times (n - n_I)` submatrix of :math:`Adj_G` constructed by removing the output rows and input columns of of :math:`Adj_G`.

    See Definition 3.3 in Mitosek and Backens, 2024 (arXiv:2410.23439)
    """
    graph = og.inside
    inputs = set(og.inputs)
    outputs = set(og.outputs)
    nodes = set(graph.nodes)

    row_idx = {node: i for i, node in enumerate(nodes - outputs)}
    col_idx = {node: j for j, node in enumerate(nodes - inputs)}

    adj_red = MatGF2(np.zeros((len(row_idx), len(col_idx)), dtype=np.int64))

    for n1, n2 in graph.edges:
        if n1 in row_idx and n2 in col_idx:
            i, j = row_idx[n1], col_idx[n2]
            adj_red.data[i, j] = 1
        if n2 in row_idx and n1 in col_idx:
            i, j = row_idx[n2], col_idx[n1]
            adj_red.data[i, j] = 1

    # Maybe interesing to add new attribute to MatGF2 for labelling rows and cols ?
    return adj_red, row_idx, col_idx


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
