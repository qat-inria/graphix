"""
MBQC Pattern Generator.

This module provides functions to generate and manipulate measurement-based quantum computing (MBQC) patterns. It includes tools for constructing various quantum states, applying measurements, and simulating the outcomes of quantum computations.

Key functions:
- generate_pattern: Generates a specified MBQC pattern based on input parameters.
- simulate_measurements: Simulates measurements on a given MBQC pattern to produce results.

Usage:
    Import the module and use the provided functions to create and analyze MBQC patterns.

Examples:
    >>> from mbqc_generator import generate_pattern
    >>> pattern = generate_pattern(params)

    >>> from mbqc_generator import simulate_measurements
    >>> results = simulate_measurements(pattern)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import graphix.gflow
from graphix.command import E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.pattern import Pattern

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.parameter import ExpressionOrFloat


def generate_from_graph(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    outputs: Iterable[int],
    meas_planes: Mapping[int, Plane] | None = None,
) -> Pattern:
    """
    Generate the measurement pattern from an open graph and measurement angles.

    This function takes an open graph \( G = (nodes, edges, input, outputs) \),
    specified by :class:`networkx.Graph`, along with two lists specifying input and output nodes.
    Currently, only XY-plane measurements are supported.

    It first searches for flow in the open graph using :func:`graphix.gflow.find_flow`.
    If found, it constructs the measurement pattern according to Theorem 1 of
    [NJP 9, 250 (2007)].

    If no flow is found, it then searches for gflow using :func:`graphix.gflow.find_gflow`,
    from which a measurement pattern can be constructed based on Theorem 2 of
    [NJP 9, 250 (2007)].

    If no gflow is found, it searches for Pauli flow using :func:`graphix.gflow.find_pauliflow`,
    from which a measurement pattern can be constructed according to Theorem 4 of
    [NJP 9, 250 (2007)].

    The constructed measurement pattern deterministically realizes the unitary embedding

    .. math::

        U = \left( \prod_i \langle +_{\alpha_i} |_i \right) E_G N_{I^C},

    where the measurements (bras) are always in the \(\langle +| \) basis
    determined by the measurement angles \(\alpha_i\) that are applied to the measuring nodes,
    effectively eliminating randomness by the added byproduct commands.

    See also
    --------
    :func:`graphix.gflow.find_flow`
    :func:`graphix.gflow.find_gflow`
    :func:`graphix.gflow.find_pauliflow`
    :class:`graphix.pattern.Pattern`

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        Graph on which MBQC should be performed.
    angles : Mapping[int, ExpressionOrFloat]
        Measurement angles for each node on the graph (in units of pi),
        except for output nodes.
    inputs : Iterable[int]
        List of node indices for input nodes.
    outputs : Iterable[int]
        List of node indices for output nodes.
    meas_planes : Mapping[int, Plane] | None, optional
        Measurement planes for each node on the graph, except for output nodes.

    Returns
    -------
    pattern : :class:`graphix.pattern.Pattern`
        Constructed measurement pattern.
    """
    inputs_set = set(inputs)
    outputs_set = set(outputs)

    measuring_nodes = set(graph.nodes) - outputs_set

    meas_planes = dict.fromkeys(measuring_nodes, Plane.XY) if not meas_planes else dict(meas_planes)

    # search for flow first
    f, l_k = graphix.gflow.find_flow(graph, inputs_set, outputs_set, meas_planes=meas_planes)
    if f is not None and l_k is not None:
        # flow found
        pattern = _flow2pattern(graph, angles, inputs, f, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    # no flow found - we try gflow
    g, l_k = graphix.gflow.find_gflow(graph, inputs_set, outputs_set, meas_planes=meas_planes)
    if g is not None and l_k is not None:
        # gflow found
        pattern = _gflow2pattern(graph, angles, inputs, meas_planes, g, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    # no flow or gflow found - we try pflow
    p, l_k = graphix.gflow.find_pauliflow(graph, inputs_set, outputs_set, meas_planes=meas_planes, meas_angles=angles)
    if p is not None and l_k is not None:
        # pflow found
        pattern = _pflow2pattern(graph, angles, inputs, meas_planes, p, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    raise ValueError("no flow or gflow or pflow found")


def _flow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    f: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """
    Construct a measurement pattern from a causal flow.

    This function constructs a measurement pattern based on theorem 1 from
    the paper "NJP 9, 250 (2007)". It utilizes the provided causal flow
    information from the input graph and other parameters to generate
    the corresponding measurement pattern.

    Parameters
    ----------
    graph : nx.Graph[int]
        The directed graph representing the causal flow, where nodes are
        associated with integer identifiers.

    angles : Mapping[int, ExpressionOrFloat]
        A mapping of node identifiers to their associated measurement angles,
        which can be either float values or symbolic expressions.

    inputs : Iterable[int]
        An iterable containing the identifiers of the input nodes for the
        measurement pattern.

    f : Mapping[int, AbstractSet[int]]
        A mapping where each key is a node identifier, and the value is a
        set of identifiers representing the nodes that are causally related
        or influenced by the corresponding node.

    l_k : Mapping[int, int]
        A mapping where each key is a node identifier and the value is an
        integer that represents a specific property or attribute related to
        the node in the context of the measurement pattern.

    Returns
    -------
    Pattern
        The constructed measurement pattern based on the provided causal flow
        and parameters.

    Notes
    -----
    Ensure that the input graph is well-formed and that all mappings and
    iterables contain valid identifiers that correspond to the nodes in the
    graph. The output pattern will be in a form suitable for further
    processing or analysis.
    """
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    measured: list[int] = []
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            measured.append(j)
            pattern.add(M(node=j, angle=angles[j]))
            neighbors: set[int] = set()
            for k in f[j]:
                neighbors |= set(graph.neighbors(k))
            for k in neighbors - {j}:
                # if k not in measured:
                pattern.add(Z(node=k, domain={j}))
            (fj,) = f[j]
            pattern.add(X(node=fj, domain={j}))
    return pattern


def _gflow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    meas_planes: Mapping[int, Plane],
    g: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """
    Construct a measurement pattern from a generalized flow according to Theorem 2 of
    [NJP 9, 250 (2007)].

    Parameters
    ----------
    graph : nx.Graph[int]
        The graph representing the structure of the system.
    angles : Mapping[int, ExpressionOrFloat]
        A mapping of node indices to their corresponding measurement angles (can be
        expressions or floats).
    inputs : Iterable[int]
        A sequence of input node indices that will be measured.
    meas_planes : Mapping[int, Plane]
        A mapping of node indices to their corresponding measurement planes.
    g : Mapping[int, AbstractSet[int]]
        A mapping that associates each node index with a set of related node indices,
        representing generalized flows.
    l_k : Mapping[int, int]
        A mapping that indicates the relationship between node indices and their
        corresponding labels.

    Returns
    -------
    Pattern
        The constructed measurement pattern based on the provided parameters.
    """
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            pattern.add(M(node=j, plane=meas_planes[j], angle=angles[j]))
            odd_neighbors = graphix.gflow.find_odd_neighbor(graph, g[j])
            for k in odd_neighbors - {j}:
                pattern.add(Z(node=k, domain={j}))
            for k in g[j] - {j}:
                pattern.add(X(node=k, domain={j}))
    return pattern


def _pflow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    meas_planes: Mapping[int, Plane],
    p: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """
    Construct a measurement pattern from a Pauli flow according to Theorem 4 of
    [NJP 9, 250 (2007)].

    Parameters
    ----------
    graph : nx.Graph[int]
        The graph representation of the system, where nodes represent qubits
        and edges represent quantum operations.

    angles : Mapping[int, ExpressionOrFloat]
        A mapping from qubit indices to their corresponding rotation angles.

    inputs : Iterable[int]
        A collection of input qubit indices that are used as the starting point
        for constructing the measurement pattern.

    meas_planes : Mapping[int, Plane]
        A mapping from qubit indices to measurement planes associated with
        each qubit.

    p : Mapping[int, AbstractSet[int]]
        A mapping from measurement indices to sets of qubit indices that
        are measured together in a specific operation.

    l_k : Mapping[int, int]
        A mapping from qubit indices to integers defining the measurement
        configurations.

    Returns
    -------
    Pattern
        The constructed measurement pattern that embodies the specified
        Pauli flow.

    Notes
    -----
    This function implements the procedure outlined in the referenced
    paper to generate a pattern suitable for quantum measurements
    based on the given Pauli flow.
    """
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            pattern.add(M(node=j, plane=meas_planes[j], angle=angles[j]))
            odd_neighbors = graphix.gflow.find_odd_neighbor(graph, p[j])
            future_nodes: set[int] = set.union(
                *(nodes for (layer, nodes) in layers.items() if layer < i)
            )  # {k | k > j}, with "j" last corrected node and ">" the Pauli flow ordering
            for k in odd_neighbors & future_nodes:
                pattern.add(Z(node=k, domain={j}))
            for k in p[j] & future_nodes:
                pattern.add(X(node=k, domain={j}))
    return pattern
