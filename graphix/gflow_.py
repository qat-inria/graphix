r"""Extended general flow (gflow) finding algorithm.

This module implements the algorithm presented in [1]. For a given labelled open graph :math:`(G, I, O, \lambda)`, this algorithm finds a gflow in polynomial time with the number of nodes, :math:`O(N^4)`.
Algorithm 1 in [1] is a generalization of Algorithm 2 in [2] supporting measurements in all three measurement planes.

References
----------
[1] Backens et al., Quantum 5, 421 (2021).
[2] Mhalla and Pedrix, Proc. of 35th ICALP (2008), 857-868, (arXiv:0709.2670).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix.fundamentals import Plane
from graphix.linalg import MatGF2, back_substitute
from graphix.opengraph import OpenGraph
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
    gf: dict[int, set[int]] = {}
    l_k = {}
    for node in og.inside.nodes:
        l_k[node] = 0

    # Current implementation assumes that `og` doesn't have Pauli measurements (verified by type-checking).
    meas_planes = {i: m.plane for i, m in og.measurements.items()}

    return _gflowaux(og.inside, set(og.inputs), set(og.outputs), meas_planes, 1, gf, l_k)


def _gflowaux(
    graph: nx.Graph,
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_planes: Mapping[int, Plane],
    k: int,
    gf: dict[int, set[int]],
    l_k: dict[int, int],
) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    r"""Find one layer of the gflow.

    Parameters
    ----------
    graph : networkx.Graph
        Full underlying graph of the input open graph.
    iset : set[int]
        Set of input nodes.
    oset : set[int]
        Set of output nodes.
    meas_planes : dict[int, Plane]
        Measurement planes for each qubit. `meas_planes[i]` is the measurement plane for qubit `i`.
    k : int
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
    - The variable `oset` contains the open graph's outputs in the first step (`k = 1`). The corrected nodes are added to `oset` in subsequent steps.

    - Let us define :math:`O' = O\setminus I`, where :math:`O` is the set of nodes `oset` and :math:`I` is the set of nodes `iset`. In each function call, for each node :math:`u \in \overline{O}` (the set of non-outputs, nodes to be corrected), we find the set of correcting nodes :math:`K' \subseteq O' such that:
        - :math:`Odd(K') \cap \overline{O} = \{u\}` if `u` is measured in plane XY,
        - :math:`Odd(K' \cup \{u\}) \cap \overline{O} = \{u\}` if `u` is measured in plane XZ,
        - :math:`Odd(K' \cup \{u\}) \cap \overline{O} = \empty` if `u` is measured in plane YZ.
    This amounts to solving the linear system :math:`A x = b` over :math:`GF(2)`, where:
        - :math:`A` is a submatrix of the adjancency of the open graph with rows and columns respectively corresponding to non-output and correcting-candidates nodes,
        - :math:`x` is a column vector with :math:`|O'|` lenght whose non-zero elements correspond to nodes in :math:`K`,
        - :math:`b` is a column vector with :math:`|\overline{O}|` lenght which depends on the measuring plane of the node we are attempting to correct (:math:`u`). In particular, it is :
            - a zero-vector with 1 at the `u`-entry if `u` is measured in plane XY,
            - a zero-vector with 1 at the `u`-entry plus the `u` column of the graph adjacency matrix if `u` is measured in plane XZ,
            - the `u` column of the graph adjacency matrix if `u` is measured in plane YZ.

    See Algorithm 1 in Backens et al., Quantum 5, 421 (2021) and Algorithm 2 in  Mhalla and Pedrix, Proc. of 35th ICALP (2008), 857-868, (arXiv:0709.2670).
    """
    nodes = set(graph.nodes)
    if oset == nodes:
        return gf, l_k

    non_output = nodes - oset
    correcting_candidates = oset - iset
    non_output_mapping = NodeIndex()
    non_output_mapping.extend(non_output)
    corr_candidates_mapping = NodeIndex()
    corr_candidates_mapping.extend(correcting_candidates)

    a_matrix, b_matrix = _get_subadj_matrices(graph, non_output_mapping, corr_candidates_mapping)

    b_vectors = _get_b_vectors(non_output, meas_planes, b_matrix, non_output_mapping)

    n_corr = len(correcting_candidates)
    a_matrix.concatenate(b_vectors, axis=1)
    a_matrix.gauss_elimination(ncols=n_corr)
    row_zero_idxs = np.flatnonzero(~a_matrix.data[:, :n_corr].any(axis=1))  # Row indices of the 0-rows.

    corrected_nodes = set()

    for node in non_output:
        # :func:`graphix.linalg.back_substitute` assumes that matrix is in REF form and LS is solvable, so we must check solvability first.
        # The LS is solvable iff the "constants" column of the gaussian-reduced system has 0s in rows which are 0 in the "coefficients" block.
        j = n_corr + non_output_mapping.index(node)
        if not a_matrix.data[row_zero_idxs, j].any():
            # LS is solvable (equivalent to K' exists in the algorithm of the reference).
            k_prime_set = set()
            x = back_substitute(a_matrix[:, :n_corr], a_matrix[:, j])
            k_prime_set |= {corr_candidates_mapping[idx] for idx in np.flatnonzero(x.data)}
            if meas_planes[node] in {Plane.XZ, Plane.YZ} and node not in iset:
                # Ensure that the correcting node is not an input. This is not verified in the original algorithm (see reference in docstring).
                k_prime_set |= {node}
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
        gf,
        l_k,
    )


def _get_subadj_matrices(
    graph: nx.Graph, non_output_mapping: NodeIndex, corr_candidates_mapping: NodeIndex
) -> tuple[MatGF2, MatGF2]:
    r"""Return two different submatrices of the adjacency matrix of the input open graph.

    Parameters
    ----------
    graph : networkx.Graph
        Full underlying graph of the input open graph.
    non_output_mapping : NodeIndex
        Mapping between matrix indices and non-output node labels.
    corr_candidates_mapping : NodeIndex
        Mapping between matrix indices and correcting-candidates node labels.

    Returns
    -------
    adj_red_a : MatGF2
        Submatrix of the adjacency matrix. Rows and columns respectively correspond to non-output and correcting-candidates nodes.
    adj_red_b : MatGF2
        Submatrix of the adjacency matrix. Rows and columns correspond to non-output nodes.
    """
    n_non_output = len(non_output_mapping)
    n_corr = len(corr_candidates_mapping)
    adj_red_a = MatGF2(np.zeros((n_non_output, n_corr), dtype=np.int_))
    adj_red_b = MatGF2(np.zeros((n_non_output, n_non_output), dtype=np.int_))

    for n1, n2 in graph.edges:
        for u, v in ((n1, n2), (n2, n1)):
            if u in non_output_mapping and v in corr_candidates_mapping:
                i, j = non_output_mapping.index(u), corr_candidates_mapping.index(v)
                adj_red_a[i, j] = 1
            if {u, v}.issubset(non_output_mapping):
                i, j = non_output_mapping.index(u), non_output_mapping.index(v)
                adj_red_b[i, j] = 1

    return adj_red_a, adj_red_b


def _get_b_vectors(
    non_output: AbstractSet[int], meas_planes: Mapping[int, Plane], adj_red_b: MatGF2, non_output_mapping: NodeIndex
) -> MatGF2:
    """Return "constants" vectors of the linear systems of equations.

    Parameters
    ----------
    non_output : set[int]

    meas_planes : dict[int, Plane]
        Measurement planes for each qubit. `meas_planes[i]` is the measurement plane for qubit `i`.
    adj_red_b : MatGF2
        Submatrix of the adjacency matrix. Rows and columns correspond to non-output nodes.
    non_output_mapping : NodeIndex
        Mapping between matrix indices and non-output node labels.

    Returns
    -------
    b_vectors : MatGF2
        Matrix containing the "constants" vectors of the linear systems of equations. Each column corresponds to a node of the non-output set and realizes a linear system with `len(non_output)` unkwnowns.
    """
    n_nodes = len(non_output)
    b_vectors = MatGF2(np.zeros((n_nodes, n_nodes), dtype=np.int_))

    for node in non_output:
        idx = non_output_mapping.index(node)
        if meas_planes[node] in {Plane.XY, Plane.XZ}:
            b_vectors[idx, idx] = 1
        if meas_planes[node] in {Plane.XZ, Plane.YZ}:
            b_vectors[:, idx] += adj_red_b[:, idx]

    return b_vectors
