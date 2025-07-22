from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from sympy import Matrix

from graphix.fundamentals import Axis, Plane
from graphix.gflow import get_pauli_nodes
from graphix.linalg import MatGF2

if TYPE_CHECKING:
    from graphix.opengraph import OpenGraph

# NOT WORKING
# def _is_dag(mat: MatGF2) -> bool:
#     """Check if matrix represents a DAG."""

#     m, n = mat.data.shape
#     if m != n:
#         return False

#     all_nodes = set(range(m))

#     while all_nodes:
#         visited: set[int] = set()
#         queue = {all_nodes.pop()}
#         while queue:
#             n = queue.pop()
#             if n in visited:
#                 return False
#             visited |= {n}
#             neigh = set(mat.data[:, n].nonzero()[0])
#             queue |= neigh
#             all_nodes -= {n}

#     return True


def _get_reduced_adj(og: OpenGraph, row_idx: dict[int, int], col_idx: dict[int, int]) -> MatGF2:
    r"""Return reduced adjacency matrix (RAdj) of the input open graph.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose RAdj is to be computed.
    row_idx: dict[int, int]
        Mapping between the non-output nodes (keys) and the rows of `adj_red` (values).
    col_idx: dict[int, int]
        Mapping between the non-input nodes (keys) and the columns of `adj_red` (values).

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
    graph = og.inside

    adj_red = MatGF2(np.zeros((len(row_idx), len(col_idx)), dtype=np.int64))

    for n1, n2 in graph.edges:
        if n1 in row_idx and n2 in col_idx:
            i, j = row_idx[n1], col_idx[n2]
            adj_red.data[i, j] = 1
        if n2 in row_idx and n1 in col_idx:
            i, j = row_idx[n2], col_idx[n1]
            adj_red.data[i, j] = 1

    return adj_red


def _get_pflow_matrices(og: OpenGraph, row_idx: dict[int, int], col_idx: dict[int, int]) -> tuple[MatGF2, MatGF2]:
    r"""Construct flow-demand and order-demand matrices.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose flow-demand and order-demand matrices are to be computed.
    row_idx: dict[int, int]
        Mapping between the non-output nodes (keys) and the rows of the flow-demand and order-demand matrices (values).
    col_idx: dict[int, int]
        Mapping between the non-input nodes (keys) and the columns of the flow-demand and order-demand matrices (values).

    Returns
    -------
    flow_demand_matrix: MatGF2
    order_demand_matrix: MatGF2

    Notes
    -----
    See Definitions 3.4 and 3.5, and Algorithm 1 in Mitosek and Backens, 2024 (arXiv:2410.23439)
    """

    flow_demand_matrix = _get_reduced_adj(og, row_idx, col_idx)
    order_demand_matrix = flow_demand_matrix.copy()

    inputs_set = set(og.inputs)
    meas = og.measurements

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

    for v, i in row_idx.items():
        plane_axis_v = meas_plane_axis[v]

        if plane_axis_v in {Plane.YZ, Plane.XZ, Axis.Z}:
            # Set row corresponding to node v to 0
            flow_demand_matrix.remove_row(row=i)
            flow_demand_matrix.add_row(row=i)
        if plane_axis_v in {Plane.YZ, Plane.XZ, Axis.Y, Axis.Z} and i not in inputs_set:
            # Set element (v, v) = 0
            j = col_idx[v]
            flow_demand_matrix.data[i, j] = 1
        if plane_axis_v in {Plane.XY, Axis.X, Axis.Y, Axis.Z}:
            # Set row corresponding to node v to 0
            order_demand_matrix.remove_row(row=i)
            order_demand_matrix.add_row(row=i)
        if plane_axis_v in {Plane.XY, Plane.XZ} and i not in inputs_set:
            # Set element (v, v) = 0
            j = col_idx[v]
            order_demand_matrix.data[i, j] = 1

    return flow_demand_matrix, order_demand_matrix


def _find_pflow_simple(og: OpenGraph) -> tuple[MatGF2, MatGF2, nx.DiGraph[int], dict[int, int], dict[int, int]] | None:
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
    non_input_idx: dict[int, int]
        Mapping between the non-input nodes (keys) and the indices of the matrices involved in the calculation (values).
    non_output_idx: dict[int, int]
        Mapping between the non-output nodes (keys) and the indices of the matrices involved in the calculation (values).   
    
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

    # TODO: It would be interesting to add new attribute to MatGF2 for labelling rows and cols
    nodes = set(og.inside.nodes)
    non_inputs_idx = {node: j for j, node in enumerate(nodes - set(og.inputs))}
    non_outputs_idx = {node: i for i, node in enumerate(nodes - set(og.outputs))}

    flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(og, row_idx=non_outputs_idx, col_idx=non_inputs_idx)

    correction_matrix = flow_demand_matrix.right_inverse()  # C matrix

    if correction_matrix is None:
        return None  # The flow-demand matrix is not invertible, therefore there's no flow.

    ordering_matrix = order_demand_matrix @ correction_matrix  # NC matrix

    # NetworkX uses the convention that a non-zero A(i,j) element represents a link i -> j.
    # We use the opposite convention, hence the transpose.   
    dag = nx.from_numpy_array(ordering_matrix.data.T, create_using=nx.DiGraph)
    
    if not nx.is_directed_acyclic_graph(dag):
        return None  # The NC matrix is not a DAG, therefore there's no flow.

    return correction_matrix, ordering_matrix, dag, non_inputs_idx, non_outputs_idx


def _find_pflow_general(og: OpenGraph) -> tuple[MatGF2, MatGF2, nx.DiGraph[int], dict[int, int], dict[int, int]] | None:
    pass


def _algebraic2pflow(correction_matrix: MatGF2, dag: nx.DiGraph[int], row_idx: dict[int, int], col_idx: dict[int, int]) -> tuple[dict[int, set[int]], dict[int, int]]:
    r"""Transform a Pauli flow in its algebraic form (correction matrix and DAG) into a Pauli flow in its standard form (correction function and partial order).
    
    Parameters
    ----------
    correction_matrix: MatGF2
        Matrix encoding the correction function.
    dag: nx.DiGraph[int]
        Directed acyclical graph represented by the ordering matrix.
    row_idx: dict[int, int]
        Mapping between the non-input nodes (keys) and the rows of the correction matrix (values).
    col_idx: dict[int, int]
        Mapping between the non-output nodes (keys) and the columns of the correction matrix (values).  

    Returns
    -------
    pf: dict[int, set[int]]
        Pauli flow correction function. pf[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict[int, int]
        Partial order between corrected qubits. l_k[d] is a node set of depth d.

    Notes
    -----
    - The correction matrix :math:`C` is an :math:`(n - n_I) \times (n - n_O)` matrix related to the correction function :math:`c(v) = \{u \in \overline{I}|C_{u,v} = 1\}`, where :math:`\overline{I}` are the non-input nodes of `og`. In other words, the column :math:`v` of :math:`C` encodes the correction set of :math:`v`, :math:`c(v)`.

    - The Pauli flow's ordering :math:`<_c` is the transitive closure of :math:`\lhd_c`, where the latter is related to the adjacency matrix of `dag` :math:`NC` as :math:`v \lhd_c w \Leftrightarrow (NC)_{w,v} = 1`, for :math:`v, w, \in \overline{O}` two non-output nodes of `og`.

    See Definition 3.6, Lemma 3.12, and Theorem 3.1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """

    pf: dict[int, set[int]]= {}
    for node, i in col_idx.items():
        correction_set = {row_idx[j] for j in correction_matrix.data[:, i].nonzero()[0]}
        pf[node] = correction_set
    
    

    return (pf, )


def find_pflow(og: OpenGraph) -> tuple[dict[int, set[int]], dict[int, int]] | None:

    ni = len(og.inputs)
    no = len(og.outputs)

    if ni > no:
        return None
    if ni == no:
        pflow_algebraic = _find_pflow_simple(og)
    else:
        pflow_algebraic = _find_pflow_general(og)
    
    if pflow_algebraic is None:
        return None
    
    correction_matrix, _, dag, non_inputs_idx, non_outputs_idx = pflow_algebraic

    return _algebraic2pflow(correction_matrix, dag, row_idx=non_inputs_idx, col_idx=non_outputs_idx)



