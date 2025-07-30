from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from graphix.fundamentals import Axis, Plane
from graphix.gflow import get_pauli_nodes
from graphix.linalg import MatGF2
from graphix.sim.base_backend import NodeIndex

if TYPE_CHECKING:
    from graphix.opengraph import OpenGraph


class OpenGraphIndex:
    """A class for managing the mapping between node numbers of a given open graph and matrix indices in the Pauli flow finding algorithm.

    It reuses the class `:class: graphix.sim.base_backend.NodeIndex` introduced for managing the mapping between node numbers and qubit indices in the internal state of the backend.

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
    ogi : OpenGraphIndex
        Open graph whose RAdj is to be computed.

    Returns
    -------
    adj_red: MatGF2
        Reduced adjacency matrix.

    Notes
    -----
    The adjacency matrix of a graph :math:`Adj_G` is a :math:`n \times n` matrix

    The RAdj matrix of an open graph OG is an :math:`(n - n_O) \times (n - n_I)` submatrix of :math:`Adj_G` constructed by removing the output rows and input columns of of :math:`Adj_G`.

    See Definition 3.3 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    graph = ogi.og.inside
    row_tags = ogi.non_outputs
    col_tags = ogi.non_inputs

    adj_red = MatGF2(np.zeros((len(row_tags), len(col_tags)), dtype=np.int_))

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
    ogi : OpenGraphIndex
        Open graph whose flow-demand and order-demand matrices are to be computed.

    Returns
    -------
    flow_demand_matrix: MatGF2
    order_demand_matrix: MatGF2

    Notes
    -----
    See Definitions 3.4 and 3.5, and Algorithm 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
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


def _find_pflow_simple(ogi: OpenGraphIndex) -> tuple[MatGF2, MatGF2] | None:
    r"""Construct the correction matrix :math:`C` and the ordering matrix, :math:`NC`, defined as the product of the order-demand matrix :math:`N` and the correction matrix.

    Parameters
    ----------
    ogi : OpenGraphIndex
        Open graph for which :math:`C` and :math:`NC` are to be computed.

    Returns
    -------
    correction_matrix: MatGF2
        Matrix encoding the correction function.
    ordering_matrix: MatGF2
        Matrix encoding the partial ordering between nodes.

    or `None`
        if the input open graph does not have Pauli flow.

    Notes
    -----
    - The correction matrix :math:`C` is an :math:`(n - n_I) \times (n - n_O)` matrix related to the correction function :math:`c(v) = \{u \in \overline{I}|C_{u,v} = 1\}`, where :math:`\overline{I}` are the non-input nodes of `ogi`.

    - The Pauli flow's ordering :math:`<_c` is the transitive closure of :math:`\lhd_c`, where the latter is related to :math:`NC` as :math:`v \lhd_c w \Leftrightarrow (NC)_{w,v} = 1`, for :math:`v, w, \in \overline{O}` two non-output nodes of `ogi`.

    - The function only returns `None` when the flow-demand matrix is not invertible (meaning that `ogi` does not have Pauli flow). The condition that the ordering matrix :math:`NC` must encode a directed acyclic graph (DAG) is verified in a subsequent step by `:func: _get_topological_order`.

    See Definitions 3.4, 3.5 and 3.6, Lemma 3.12, Theorem 3.1, and Algorithm 2 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

    correction_matrix = flow_demand_matrix.right_inverse()  # C matrix

    if correction_matrix is None:
        return None  # The flow-demand matrix is not invertible, therefore there's no flow.

    ordering_matrix = order_demand_matrix @ correction_matrix  # NC matrix

    return correction_matrix, ordering_matrix


def _get_p_matrix(ogi: OpenGraphIndex, nb_matrix: MatGF2) -> MatGF2 | None:
    #  Steps 8 - 12

    n_cols_p = len(ogi.non_outputs)
    n_rows_p = len(ogi.og.outputs) - len(ogi.og.inputs)

    # Steps 8, 9 and 10
    kils_matrix = MatGF2(nb_matrix.data[:, n_cols_p:])  # N_R matrix
    kils_matrix.concatenate(MatGF2(nb_matrix.data[:, :n_cols_p]), axis=1)  # Concatenate N_L matrix
    kils_matrix.concatenate(MatGF2(np.eye(n_cols_p, dtype=np.int_)), axis=1)  # Concatenate identity matrix

    kls_matrix = kils_matrix.gauss_elimination(ncols=n_rows_p, copy=True)  # RREF form is not needed, only REF.

    # Step 11
    p_matrix = MatGF2(np.zeros((n_rows_p, n_cols_p), dtype=np.int_))
    solved_nodes: set[int] = set()
    non_outputs_set = set(ogi.non_outputs)

    # Step 12

    def get_solvable_nodes() -> set[int]:
        """Return the set nodes whose associated linear system is solvable.

        A node is solvable if:
            - It has not been solved yet.
            - Its column in the second block of :math:`K_{LS}` (which determines the constants in each equation) has only zeros where it intersects rows for which all the coefficients in the first block are 0s.

        See Theorem 4.4, step 12.a in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        solvable_nodes: set[int] = set()

        row_idxs = np.flatnonzero(
            ~kls_matrix.data[:, :n_rows_p].any(axis=1)
        )  # Row indices of the 0-rows in the first block of K_{LS}
        if row_idxs.size:
            for v in non_outputs_set - solved_nodes:
                j = n_rows_p + ogi.non_outputs.index(
                    v
                )  # `n_rows_p` is the column offset from the first block of K_{LS}
                if not kls_matrix.data[row_idxs, j].any():
                    solvable_nodes.add(v)

        return solvable_nodes

    def update_p_matrix() -> None:
        """Update `p_matrix`.

        The solution of the linear system associated with node :math:`v` in `solvable_nodes` corresponds to the column of `p_matrix` associated with node :math:`v`.

        See Theorem 4.4, steps 12.b and 12.c in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        for v in solvable_nodes:
            j = ogi.non_outputs.index(v)
            j_shift = n_rows_p + j  # n_rows_p is the column offset from the first block of K_{LS}
            mat = MatGF2(kls_matrix.data[:, :n_rows_p])  # first block of kls, in row echelon form.
            b = MatGF2(kls_matrix.data[:, j_shift])
            x = _back_substitute(mat, b)
            p_matrix.data[:, j] = x.data

    def update_kls_matrix() -> None:
        """Update `kls_matrix`.

        Bring the linear system encoded in :math:`K_{LS}` to the row-echelon form (REF) that would be achieved by Gaussian elimination if the row and column vectors corresponding to vertices in `solvable_nodes` where not included in the starting matrix.

        See Theorem 4.4, step 12.d in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        shift = n_rows_p + n_cols_p  # `n_rows_p` + `n_cols_p` is the column offset from the first two blocks of K_{LS}
        row_permutation: list[int]

        def reorder(old_pos: int, new_pos: int) -> None:  # Used in step 12.d.vi
            """Reorder elemetns of `row_permutation`.

            The element at `old_pos` is placed on the right of the element at `new_pos`.
            Example:
            ```
            row_permutation = [0, 1, 2, 3, 4]
            reorder(1, 3) -> [0, 2, 3, 1, 4]
            reorder(2, -1) -> [2, 0, 1, 3, 4]
            ```
            """
            val = row_permutation.pop(old_pos)
            row_permutation.insert(new_pos + (new_pos < old_pos), val)

        for v in solvable_nodes:
            # Step 12.d.ii
            j = ogi.non_outputs.index(v)
            j_shift = shift + j
            row_idxs = np.flatnonzero(
                kls_matrix.data[:, j_shift]
            ).tolist()  # Row indices with 1s in column of node v in third block.
            k = row_idxs.pop()  # TODO: Could `row_idxs` be empty ?

            # Step 12.d.iii
            kls_matrix.data[row_idxs, :] += kls_matrix.data[k, :]  # Adding a row to previous rows preserves REF.

            # Step 12.d.iv
            kls_matrix.data[k, :] += kils_matrix.data[j, :]  # Row `k` may now break REF.

            # Step 12.d.v
            pivots = []  # Store pivots for next step.
            for i, row in enumerate(kls_matrix.data):
                col_idxs = np.flatnonzero(row[:n_rows_p])  # Column indices with 1s in first block.
                if i == k:
                    pivots.append(None if col_idxs.size == 0 else col_idxs[0])
                    continue
                if (
                    col_idxs.size == 0
                ):  # Row `i` has all zeros in the first block. Only row `k` can break REF, so rows below have all zeros in the first block too.
                    break
                p = col_idxs[0]
                pivots.append(p)
                if kls_matrix.data[k, p]:  # Row `k` has a 1 in the column corresponding to the leading 1 of row `i`.
                    kls_matrix.data[k] += kls_matrix.data[i, :]

            # Step 12.d.vi
            row_permutation = list(range(n_cols_p))  # Row indices of `kls_matrix`.
            n_pivots = len(pivots)

            if n_pivots >= k:  # Row `k` is among non-zero rows
                if (p0 := pivots[k]) is None:  # Row `k` has all zeros in first block.
                    reorder(k, n_pivots)  # Move row `k` to the top of the zeros block.
                else:
                    new_pos = np.argmax(np.array(pivots) > p0) - 1
                    reorder(k, int(new_pos))
            else:  # Row `k` is among zero rows.
                col_idxs = np.flatnonzero(kls_matrix.data[k, :n_rows_p])
                if col_idxs.size:  # Row `k` is non-zero.
                    p0 = col_idxs[0]  # Leading 1 of row `k`.
                    new_pos = np.argmax(np.array(pivots) > p0) - 1
                    reorder(k, int(new_pos))

            kls_matrix.permute_row(row_permutation)

    while solved_nodes != non_outputs_set:
        solvable_nodes = get_solvable_nodes()  # Step 12.a
        if not solvable_nodes:
            return None

        update_p_matrix()  # Steps 12.b, 12.c
        update_kls_matrix()  # Step 12.d
        solved_nodes.update(solvable_nodes)

    return p_matrix


def _back_substitute(mat: MatGF2, b: MatGF2) -> MatGF2:
    r"""Solve the linear system (LS) `mat @ x == b`.

    Parameters
    ----------
    A: MatGF2
        Matrix with shape `(m, n)` containing the LS coefficients in row echelon form (REF).
    b: MatGF2
        Matrix with shape `(m,)` containing the constants column vector.

    Returns
    -------
    x: MatGF2
        Matrix with shape `(n,)` containing the solutions of the LS.

    Notes
    -----
    This function is not integrated in `:class: graphix.linalg.MatGF2` because it does not perform any checks on the form of `mat` to ensure that it is in REF or that the system is solvable.
    """
    m, n = mat.data.shape
    x = MatGF2(np.zeros(n, dtype=np.int_))

    for i in range(m - 1, -1, -1):
        row = mat.data[i]
        col_idxs = np.flatnonzero(row)  # Column indices with 1s
        if col_idxs.size == 0:
            continue  # Skip if row is all zeros

        j = col_idxs[0]
        # x_j = b_i + sum_{k = j+1}^{n-1} A_{i,k} x_k = b_i + sum_{k} x_k because A in REF and x_j = 0
        x.data[j] = b.data[i] ^ np.bitwise_xor.reduce(x.data[col_idxs])

    return x


def _find_pflow_general(ogi: OpenGraphIndex) -> tuple[MatGF2, MatGF2] | None:
    # Steps 1 and 2
    flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

    # Steps 3 and 4
    correction_matrix_0 = flow_demand_matrix.right_inverse()  # C0 matrix
    if correction_matrix_0 is None:
        return None  # The flow-demand matrix is not invertible, therefore there's no flow.

    # Steps 5, 6 and 7
    ker_flow_demand_matrix = flow_demand_matrix.null_space().transpose()  # F matrix
    c_prime_matrix = correction_matrix_0.copy()
    c_prime_matrix.concatenate(ker_flow_demand_matrix, axis=1)
    nb_matrix = order_demand_matrix @ c_prime_matrix

    # Steps 8 - 12
    if (p_matrix := _get_p_matrix(ogi, nb_matrix)) is None:
        return None

    # Step 13
    cb_matrix = MatGF2(np.eye(len(ogi.non_outputs), dtype=np.int_))
    cb_matrix.concatenate(p_matrix, axis=0)

    correction_matrix = c_prime_matrix @ cb_matrix
    ordering_matrix = nb_matrix @ cb_matrix

    return correction_matrix, ordering_matrix


def _get_topological_order(ordering_matrix: MatGF2) -> list[list[int]] | None:
    """Stratify the directed acyclic graph (DAG) represented by the ordering matrix into generations.

    Parameters
    ----------
    ordering_matrix: MatGF2
        Matrix encoding the partial ordering between nodes intepreted as the adjacency matrix of a directed graph.

    Returns
    -------
    list[list[int]]
        Topological generations. Integers represent the indices of the matrix `ordering_matrix`, not the labelling of the nodes.

    or `None`
        if `ordering_matrix` is not a DAG.
    """
    # NetworkX uses the convention that a non-zero A(i,j) element represents a link i -> j.
    # We use the opposite convention, hence the transpose.
    dag = nx.from_numpy_array(ordering_matrix.data.T, create_using=nx.DiGraph)

    topo_gen = nx.topological_generations(dag)
    try:
        return list(topo_gen)
    except nx.NetworkXUnfeasible:
        return None


def _algebraic2pflow(
    ogi: OpenGraphIndex,
    correction_matrix: MatGF2,
    ordering_matrix: MatGF2,
) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    r"""Transform the correction and ordering matrices into a Pauli flow in its standard form (correction function and partial order).

    Parameters
    ----------
    ogi: OpenGraphIndex
        Open graph whose Pauli flow is being calculated.
    correction_matrix: MatGF2
        Matrix encoding the correction function.
    ordering_matrix: MatGF2
        Matrix encoding the partial ordering between nodes (DAG).

    Returns
    -------
    pf: dict[int, set[int]]
        Pauli flow correction function. pf[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    or `None`
        if the ordering matrix is not a DAG, in which case the input open graph does not have Pauli flow.

    Notes
    -----
    - The correction matrix :math:`C` is an :math:`(n - n_I) \times (n - n_O)` matrix related to the correction function :math:`c(v) = \{u \in \overline{I}|C_{u,v} = 1\}`, where :math:`\overline{I}` are the non-input nodes of `ogi`. In other words, the column :math:`v` of :math:`C` encodes the correction set of :math:`v`, :math:`c(v)`.

    - The Pauli flow's ordering :math:`<_c` is the transitive closure of :math:`\lhd_c`, where the latter is related to the ordering matrix :math:`NC` as :math:`v \lhd_c w \Leftrightarrow (NC)_{w,v} = 1`, for :math:`v, w, \in \overline{O}` two non-output nodes of `ogi`.

    See Definition 3.6, Lemma 3.12, and Theorem 3.1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    row_tags = ogi.non_inputs
    col_tags = ogi.non_outputs

    # Calculation of the partial ordering

    if (topo_gen := _get_topological_order(ordering_matrix)) is None:
        return None  # The NC matrix is not a DAG, therefore there's no flow.

    l_k = dict.fromkeys(ogi.og.outputs, 0)  # Output nodes are always in layer 0

    # If m >_c n, with >_c the flow order for two nodes m, n, then layer(n) > layer(m).
    # Therefore, we iterate the topological sort of the graph in _reverse_ order to obtain the order of measurements.
    for layer, idx in enumerate(reversed(topo_gen), start=1):
        l_k.update({col_tags[i]: layer for i in idx})

    # Calculation of the correction function

    pf: dict[int, set[int]] = {}
    for node in col_tags:
        i = col_tags.index(node)
        correction_set = {row_tags[j] for j in correction_matrix.data[:, i].nonzero()[0]}
        pf[node] = correction_set

    return pf, l_k


def find_pflow(og: OpenGraph) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    """Return Pauli flow."""
    ni = len(og.inputs)
    no = len(og.outputs)

    if ni > no:
        return None

    ogi = OpenGraphIndex(og)
    if (pflow_algebraic := _find_pflow_simple(ogi) if ni == no else _find_pflow_general(ogi)) is None:
        return None
    if (pflow := _algebraic2pflow(ogi, *pflow_algebraic)) is None:
        return None

    pf, l_k = pflow

    return pf, l_k


# def is_pflow_valid(og: OpenGraph, pf: Mapping[int, set[int]], l_k: Mapping[int, int]) -> bool:
#     """Verify if a given Pauli flow is correct by checking the Pauli flow conditions (P1 - P9).

#     See Definition 5 in Browne et al., NJP 9, 250 (2007).

#     Returns
#     -------
#     pf: dict[int, set[int]]
#         Pauli flow correction function. pf[i] is the set of qubits to be corrected for the measurement of qubit i.
#     l_k: dict[int, int]
#         Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

#     """

#     return False
