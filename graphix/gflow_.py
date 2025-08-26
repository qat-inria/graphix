r"""Extended general flow (gflow) finding algorithm.

This module implements the algorithm presented in [1]. For a given labelled open graph :math:`(G, I, O, \lambda)`, this algorithm finds a gflow in polynomial time with the number of nodes, :math:`O(N^4)`.
The algorithm in [1] is a generalization of the algorithm [2] supporting measurements in all three measurement planes.

References
----------
[1] Backens et al., Quantum 5, 421 (2021).
[2] Mhalla and Pedrix, Proc. of 35th ICALP (2008), 857-868, (arXiv:0709.2670).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix.fundamentals import Plane
from graphix.linalg import MatGF2
from graphix.opengraph import OpenGraph
from graphix.pflow import _back_substitute  # this could be moved to a `common` module or to `graphix.linalg`
from graphix.sim.base_backend import NodeIndex

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.opengraph import OpenGraph


def find_gflow(og: OpenGraph) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    """Return a maximally delayed general flow (gflow) of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph
        Open graph whose Pauli flow is calculated.

    Returns
    -------
    gf : dict[int, set[int]]
        gflow correction function. `gf[i]` is the set of qubits to be corrected for the measurement of qubit `i`.
    l_k : dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    or `None`
        if the input open graph does not have gflow.

    Notes
    -----
    See Definition 2.36 and Algorithm 1 in Backens et al., Quantum 5, 421 (2021).
    """
    l_k = {}
    gf: dict[int, set[int]] = {}
    for node in og.inside.nodes:
        l_k[node] = 0

    # TODO: Current implementation assumes that `og` doesn't have Pauli measurements (verified by type-checking).
    meas_planes = {i: m.plane for i, m in og.measurements.items()}

    return _gflowaux(og.inside, set(og.inputs), set(og.outputs), meas_planes, 1, l_k, gf)


def _gflowaux(
    graph: nx.Graph,
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_planes: Mapping[int, Plane],
    k: int,
    l_k: dict[int, int],
    gf: dict[int, set[int]],
) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    """Find one layer of the gflow.

    Parameters
    ----------
    graph: networkx.Graph
        Full underlying graph of the input open graph.
    iset: set
        Set of input nodes.
    oset: set
        Set of corrected nodes.
    meas_planes: dict
        Measurement planes for each qubit. `meas_planes[i]` is the measurement plane for qubit `i`.
    k: int
        Current layer number.
    gf : dict[int, set[int]]
        gflow correction function. `gf[i]` is the set of qubits to be corrected for the measurement of qubit `i`.
    l_k : dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    Returns
    -------
    gf : dict[int, set[int]]
        gflow correction function. gf[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k : dict[int, int]
        Partial order between corrected qubits, such that the pair (`key`, `value`) corresponds to (node, depth).

    Notes
    -----
    The variable `oset` contains the open graph's outputs in the first step (`k = 1`). The corrected nodes are added to `oset` in subsequent steps.
    """
    nodes = set(graph.nodes)
    if oset == nodes:
        return gf, l_k

    corrected_nodes = set()

    non_output = nodes - oset
    correction_candidates = oset - iset
    row_tags = NodeIndex()
    row_tags.extend(non_output)
    col_tags = NodeIndex()
    col_tags.extend(correction_candidates)

    a_matrix = _get_a_matrix(graph, row_tags, col_tags)
    n_cols = a_matrix.data.shape[1]
    # a_matrix.gauss_elimination()  # bring `a_matrix` into REF for solving the LS of equations.
    # row_zero_idxs = np.flatnonzero(~a_matrix.data.any(axis=1))  # Row indices of the 0-rows.
    # row_zero_idx = (
    #     row_zero_idxs[0] if row_zero_idxs.size else a_matrix.data.shape[0]
    # )  # Index of first 0-row or number of rows if there are none.

    for node in non_output:
        a_matrix_red = a_matrix.copy()
        b = MatGF2(np.zeros((a_matrix.data.shape[0], 1), dtype=np.int_))
        # solvable = True  # LS is always solvable for YZ measurement (`b` is a 0-vector)

        if meas_planes[node] in {Plane.XY, Plane.XZ}:
            index = row_tags.index(node)
            b[index] = 1

        a_matrix_red.concatenate(b, axis=1)
        a_matrix_red.gauss_elimination(ncols=n_cols)

        # The LS is solvable iff `b` has 0s in rows of `a_matrix` which are 0.
        row_zero_idxs = np.flatnonzero(~a_matrix.data[:, :n_cols].any(axis=1))  # Row indices of the 0-rows.

        # The LS is solvable iff `b` has 0s in rows of `a_matrix` which are 0.

        k_prime_set = set()
        if not a_matrix_red.data[row_zero_idxs, -1].any():
            x = _back_substitute(
                a_matrix_red[:, :n_cols], a_matrix_red[:, -1]
            )  # assumes `a_matrix` is in REF and LS is solvable.
            k_prime_set |= {col_tags[idx] for idx in np.flatnonzero(x.data)}
        if (
            meas_planes[node] in {Plane.XZ, Plane.YZ} and node not in iset
        ):  # ensure that the correcting node is not an input. This is not verified in the original algorithm (see reference in docstring).
            k_prime_set |= {node}
        if k_prime_set:
            corrected_nodes |= {node}
            gf[node] = k_prime_set
            l_k[node] = k

    if len(corrected_nodes) == 0:
        # The condition O == V in the reference's algorithm is verified at the beggining of the function.
        return None

    return _gflowaux(
        graph,
        iset,
        oset | corrected_nodes,
        meas_planes,
        k + 1,
        l_k,
        gf,
    )


def _get_a_matrix(graph: nx.Graph, row_tags: NodeIndex, col_tags: NodeIndex) -> MatGF2:
    r"""Return a submatrix of the adjacency matrix of the input open graph.

    Parameters
    ----------
    graph: networkx.Graph
        Full underlying graph of the input open graph.
    row_tags: NodeIndex
        Mapping between row indices and node labels included in the returned submatrix of the adjacency matrix.
    col_tags: NodeIndex
        Mapping between column indices and node labels included in the returned submatrix of the adjacency matrix.

    Returns
    -------
    adj_red : MatGF2
        Submatrix of the adjacency matrix.
    """
    adj_red = MatGF2(np.zeros((len(row_tags), len(col_tags)), dtype=np.int_))

    for n1, n2 in graph.edges:
        for u, v in ((n1, n2), (n2, n1)):
            if u in row_tags and v in col_tags:
                i, j = row_tags.index(u), col_tags.index(v)
                adj_red[i, j] = 1

    return adj_red
