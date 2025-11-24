"""
Provides a class for creating and manipulating open graphs.

An open graph is a data structure that allows for the representation
of networks, where nodes can be added, removed, or modified, and
edges can connect any pair of nodes.

Attributes
----------
None

Methods
-------
- add_node(node): Add a node to the graph.
- remove_node(node): Remove a node from the graph.
- add_edge(node1, node2): Add an edge between two nodes.
- remove_edge(node1, node2): Remove the edge between two nodes.
- get_neighbors(node): Retrieve a list of neighbors for a given node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

import graphix.generator
from graphix.measurements import Measurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from graphix.pattern import Pattern


@dataclass(frozen=True)
class OpenGraph:
    """
    OpenGraph contains the graph, measurement, and input and output nodes.

    This is the graph we wish to implement deterministically.

    Parameters
    ----------
    inside : networkx.Graph
        The underlying graph state.
    measurements : dict
        A dictionary whose keys are the IDs of nodes, and the values are
        the measurements at those nodes.
    inputs : list of int
        An ordered list of node IDs that are inputs to the graph.
    outputs : list of int
        An ordered list of node IDs that are outputs of the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from graphix.fundamentals import Plane
    >>> from graphix.opengraph import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
    >>> inputs = [0]
    >>> outputs = [2]
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    """

    inside: nx.Graph[int]
    measurements: dict[int, Measurement]
    inputs: list[int]  # Inputs are ordered
    outputs: list[int]  # Outputs are ordered

    def __post_init__(self) -> None:
        """
        Validate the open graph.

        This method is called automatically after the object has been initialized.
        It checks the integrity and correctness of the Open Graph properties to ensure
        that the data conforms to the Open Graph protocol.

        Raises
        ------
        ValueError
            If any required fields are missing or if the data is invalid.

        Notes
        -----
        This method ensures that the Open Graph metadata is correctly set up
        before it is used.
        """
        if not all(node in self.inside.nodes for node in self.measurements):
            raise ValueError("All measured nodes must be part of the graph's nodes.")
        if not all(node in self.inside.nodes for node in self.inputs):
            raise ValueError("All input nodes must be part of the graph's nodes.")
        if not all(node in self.inside.nodes for node in self.outputs):
            raise ValueError("All output nodes must be part of the graph's nodes.")
        if any(node in self.outputs for node in self.measurements):
            raise ValueError("Output node cannot be measured.")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Input nodes contain duplicates.")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Output nodes contain duplicates.")

    def isclose(self, other: OpenGraph, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """
        Determine if two open graphs implement approximately the same unitary operator.

        This function checks if the structure of the graphs is the same and if all
        measurement angles are sufficiently close, within the specified relative and
        absolute tolerances. Note that this method does not verify if the graphs are
        equal up to an isomorphism.

        Parameters
        ----------
        other : OpenGraph
            The other open graph to compare against.
        rel_tol : float, optional
            The relative tolerance, which determines how close the values
            need to be relative to the size of the values (default is 1e-09).
        abs_tol : float, optional
            The absolute tolerance, which specifies the minimum absolute difference
            needed to consider the values as close (default is 0.0).

        Returns
        -------
        bool
            Returns `True` if the two open graphs are approximately equal,
            `False` otherwise.
        """
        if not nx.utils.graphs_equal(self.inside, other.inside):
            return False

        if self.inputs != other.inputs or self.outputs != other.outputs:
            return False

        if set(self.measurements.keys()) != set(other.measurements.keys()):
            return False

        return all(
            m.isclose(other.measurements[node], rel_tol=rel_tol, abs_tol=abs_tol)
            for node, m in self.measurements.items()
        )

    @staticmethod
    def from_pattern(pattern: Pattern) -> OpenGraph:
        """
        Construct an `OpenGraph` object based on the resource-state graph associated with a given measurement pattern.

        Parameters
        ----------
        pattern : Pattern
            The measurement pattern used to initialize the `OpenGraph`.

        Returns
        -------
        OpenGraph
            An instance of the `OpenGraph` initialized according to the specified measurement pattern.

        Examples
        --------
        >>> pattern = Pattern(...)  # create an instance of Pattern
        >>> graph = OpenGraph.from_pattern(pattern)
        """
        graph = pattern.extract_graph()

        inputs = pattern.input_nodes
        outputs = pattern.output_nodes

        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        meas = {node: Measurement(meas_angles[node], meas_planes[node]) for node in meas_angles}

        return OpenGraph(graph, meas, inputs, outputs)

    def to_pattern(self) -> Pattern:
        """
        Convert the `OpenGraph` into a `Pattern`.

        This method converts the current instance of the `OpenGraph` into a `Pattern`.

        Raises
        ------
        Exception
            If the open graph does not have flow, gflow, or Pauli flow.

        Notes
        -----
        The pattern will be generated using maximally-delayed flow.
        """
        g = self.inside.copy()
        inputs = self.inputs
        outputs = self.outputs
        meas = self.measurements

        angles = {node: m.angle for node, m in meas.items()}
        planes = {node: m.plane for node, m in meas.items()}

        return graphix.generator.generate_from_graph(g, angles, inputs, outputs, planes)

    def compose(self, other: OpenGraph, mapping: Mapping[int, int]) -> tuple[OpenGraph, dict[int, int]]:
        """
        Compose two open graphs by merging subsets of nodes from `self` and `other`, and relabeling the nodes of `other` that were not merged.

        Parameters
        ----------
        other : OpenGraph
            Open graph to be composed with `self`.
        mapping : dict[int, int]
            Partial relabelling of the nodes in `other`, where the keys represent the old node labels and the values represent the new node labels.

        Returns
        -------
        og : OpenGraph
            Composed open graph resulting from the combination of `self` and `other`.
        mapping_complete : dict[int, int]
            Complete relabelling of the nodes in `other`, with keys and values indicating the old and new node labels, respectively.

        Notes
        -----
        Let :math:`\{G(V_1, E_1), I_1, O_1\}` be the open graph `self`, and :math:`\{G(V_2, E_2), I_2, O_2\}` be the open graph `other`. The resulting open graph will be denoted as :math:`\{G(V, E), I, O\}` and `{v:u}` an element of `mapping`.

        Define :math:`V` and :math:`U` as the sets of nodes represented by `mapping.keys()` and `mapping.values()`, respectively, and :math:`M = U \cap V_1` as the set of merged nodes.

        The open graph composition requires that:
        - :math:`V \subseteq V_2`.
        - If both `v` and `u` are measured, the corresponding measurements must have the same plane and angle.

        The conventions for the returned open graph are as follows:
        - :math:`I = (I_1 \cup I_2) \setminus M \cup (I_1 \cap I_2 \cap M)`,
        - :math:`O = (O_1 \cup O_2) \setminus M \cup (O_1 \cap O_2 \cap M)`,
        - If only one node of the pair `{v:u}` is measured, the measure is assigned to :math:`u \in V` in the resulting open graph.
        - Input (and output) nodes in the returned open graph maintain the order of open graph `self` followed by those of open graph `other`. Merged nodes are removed unless they are input (or output) nodes in both open graphs, in which case, they appear in the order they originally had in the graph `self`.
        """
        if not (mapping.keys() <= other.inside.nodes):
            raise ValueError("Keys of mapping must be correspond to nodes of other.")
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("Values in mapping contain duplicates.")
        for v, u in mapping.items():
            if (
                (vm := other.measurements.get(v)) is not None
                and (um := self.measurements.get(u)) is not None
                and not vm.isclose(um)
            ):
                raise ValueError(f"Attempted to merge nodes {v}:{u} but have different measurements")

        shift = max(*self.inside.nodes, *mapping.values()) + 1

        mapping_sequential = {
            node: i for i, node in enumerate(sorted(other.inside.nodes - mapping.keys()), start=shift)
        }  # assigns new labels to nodes in other not specified in mapping

        mapping_complete = {**mapping, **mapping_sequential}

        g2_shifted = nx.relabel_nodes(other.inside, mapping_complete)
        g = nx.compose(self.inside, g2_shifted)

        merged = set(mapping_complete.values()) & self.inside.nodes

        def merge_ports(p1: Iterable[int], p2: Iterable[int]) -> list[int]:
            p2_mapped = [mapping_complete[node] for node in p2]
            p2_set = set(p2_mapped)
            part1 = [node for node in p1 if node not in merged or node in p2_set]
            part2 = [node for node in p2_mapped if node not in merged]
            return part1 + part2

        inputs = merge_ports(self.inputs, other.inputs)
        outputs = merge_ports(self.outputs, other.outputs)

        measurements_shifted = {mapping_complete[i]: meas for i, meas in other.measurements.items()}
        measurements = {**self.measurements, **measurements_shifted}

        return OpenGraph(g, measurements, inputs, outputs), mapping_complete
