"""
Graph simulator.

This module provides functionality for simulating and analyzing graphs.
It includes various algorithms and tools for graph operations such as
traversal, searching, and manipulation of graph structures.

Modules:
--------
- Graph: Contains the implementation of the graph data structure.
- Algorithms: Includes functions for common graph algorithms such as
  DFS, BFS, Dijkstra's algorithm and more.
- Utilities: Helper functions for graph visualization and other utilities.

Usage:
------
To use this module, import the required classes or functions and
instantiate the graph as needed.

Example:
--------
```python
from graph_simulator import Graph

g = Graph()
g.add_edge(1, 2)
g.add_edge(2, 3)
print(g.bfs(1))
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import networkx as nx
import typing_extensions

from graphix import utils
from graphix.clifford import Clifford
from graphix.measurements import outcome
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

if TYPE_CHECKING:
    import functools
    from collections.abc import Iterable, Mapping

    from graphix.measurements import Outcome


if TYPE_CHECKING:
    Graph = nx.Graph[int]
else:
    Graph = nx.Graph


class MBQCGraphNode(TypedDict):
    """
    Attributes of a Measurement-Based Quantum Computing (MBQC) graph node.

    This class represents a node in a graph used for
    Measurement-Based Quantum Computing (MBQC) which includes
    attributes defining the node's state and connections
    to other nodes in the graph.

    Attributes
    ----------
    id : int
        Unique identifier for the node.
    state : str
        The quantum state associated with the node.
    neighbors : list of MBQCGraphNode
        A list of nodes that are directly connected to this node.
    measurement_result : bool or None
        The result of the measurement performed on this node,
        if applicable. Defaults to None if no measurement has
        been performed.

    Methods
    -------
    add_neighbor(node):
        Adds a neighboring node to this node's list of neighbors.
    __str__():
        Returns a string representation of the node.
    __repr__():
        Returns a formal string representation of the node.
    """

    sign: bool
    loop: bool
    hollow: bool


class GraphState(Graph):
    """
    Graph state simulator implemented with :mod:`networkx`.

    This class performs Pauli measurements on graph states.

    References
    ----------
    M. Elliot, B. Eastin & C. Caves, J. Phys. A 43, 025301 (2010) and
    PRA 77, 042307 (2008).

    Attributes
    ----------
    hollow : bool
        True if the node is hollow (has a local H operator).
    sign : bool
        True if the node has a negative sign (local Z operator).
    loop : bool
        True if the node has a loop (local S operator).
    """

    nodes: functools.cached_property[Mapping[int, MBQCGraphNode]]  # type: ignore[assignment]

    def __init__(
        self,
        nodes: Iterable[int] | None = None,
        edges: Iterable[tuple[int, int]] | None = None,
        vops: Mapping[int, Clifford] | None = None,
    ) -> None:
        """
        Instantiate a graph simulator.

        Parameters
        ----------
        nodes : Iterable[int], optional
            A container of nodes. If None, the graph will be initialized with no nodes.
        edges : Iterable[tuple[int, int]], optional
            A list of tuples (i, j) representing pairs of nodes to be entangled. If None, no edges will be created.
        vops : Mapping[int, Clifford], optional
            A dictionary of local Clifford gates, where the keys are node indices and the values are the corresponding Clifford operations. If None, no local operations will be assigned.
        """
        super().__init__()
        if nodes is not None:
            self.add_nodes_from(nodes)
        if edges is not None:
            self.add_edges_from(edges)
        if vops is not None:
            self.apply_vops(vops)

    @typing_extensions.override
    def add_nodes_from(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        nodes_for_adding: Iterable[int | tuple[int, MBQCGraphNode]],  # type: ignore[override]
        **attr: Any,
    ) -> None:
        """
        Add nodes to the graph.

        This method wraps the `networkx.Graph.add_nodes_from` function to initialize
        attributes of `MBQCGraphNode`.

        Parameters
        ----------
        nodes_for_adding : iterable of int or tuple of (int, MBQCGraphNode)
            An iterable of nodes to be added. Each node can either be an integer
            representing the node identifier, or a tuple consisting of an integer
            and an instance of `MBQCGraphNode`.

        **attr : Any
            Additional attributes to initialize for the nodes being added.

        Returns
        -------
        None
            This method does not return any value, it modifies the graph state
            in place.
        """
        nodes_for_adding = list(nodes_for_adding)
        super().add_nodes_from(nodes_for_adding, **attr)  # type: ignore[arg-type]
        for data in nodes_for_adding:
            u, mp = data if isinstance(data, tuple) else (data, MBQCGraphNode(sign=False, hollow=False, loop=False))
            for k, v_ in mp.items():
                dst = self.nodes[u]
                v = bool(v_)
                # Need to use literal inside brackets
                if k == "sign":
                    dst["sign"] = v
                elif k == "hollow":
                    dst["hollow"] = v
                elif k == "loop":
                    dst["loop"] = v
                else:
                    msg = "Invalid node attribute."
                    raise ValueError(msg)

    @typing_extensions.override
    def add_node(
        self,
        node_for_adding: int,
        **attr: Any,
    ) -> None:
        """
        Add a node to the graph, wrapping the `networkx.Graph.add_node` method.

        This method initializes attributes for the node as specified by
        keyword arguments, which can include any additional properties
        needed for the `MBQCGraphNode`.

        Parameters
        ----------
        node_for_adding : int
            The identifier for the node to be added to the graph.
        **attr : Any
            Additional attributes to initialize for the MBQCGraphNode.

        Returns
        -------
        None
        """
        self.add_nodes_from((node_for_adding,), **attr)

    def local_complement(self, node: int) -> None:
        """
        Perform local complementation of a graph.

        Parameters
        ----------
        node : int
            The index of the node on which to perform the local complementation.

        Returns
        -------
        None
            This method modifies the graph in place and does not return a value.

        Notes
        -----
        Local complementation at a node involves taking the induced subgraph formed by the neighbors of the node,
        and replacing it with its complement.
        """
        g = self.subgraph(self.neighbors(node))
        g_new: nx.Graph[int] = nx.complement(g)
        self.remove_edges_from(g.edges)
        self.add_edges_from(g_new.edges)

    def apply_vops(self, vops: Mapping[int, Clifford]) -> None:
        """
        Apply local Clifford operators to the graph state from a dictionary.

        Parameters
        ----------
        vops : Mapping[int, Clifford]
            A dictionary containing node indices as keys and local Clifford operators
            as values.

        Returns
        -------
        None
        """
        for node, vop in vops.items():
            for lc in reversed(vop.hsz):
                if lc == Clifford.Z:
                    self.z(node)
                elif lc == Clifford.H:
                    self.h(node)
                elif lc == Clifford.S:
                    self.s(node)
                else:
                    raise RuntimeError

    def get_vops(self) -> dict[int, Clifford]:
        """
        Apply local Clifford operators to the graph state from a dictionary.

        Returns
        -------
        vops : dict[int, Clifford]
            A dictionary containing node indices as keys and local Clifford operators as values.
        """
        vops: dict[int, Clifford] = {}
        for i in self.nodes:
            vop = Clifford.I
            if self.nodes[i]["sign"]:
                vop = Clifford.Z @ vop
            if self.nodes[i]["loop"]:
                vop = Clifford.S @ vop
            if self.nodes[i]["hollow"]:
                vop = Clifford.H @ vop
            vops[i] = vop
        return vops

    def flip_fill(self, node: int) -> None:
        """
        Flips the fill (local Hamiltonian) of a specified node in the graph.

        Parameters
        ----------
        node : int
            The graph node for which the fill is to be flipped.

        Returns
        -------
        None
        """
        self.nodes[node]["hollow"] = not self.nodes[node]["hollow"]

    def flip_sign(self, node: int) -> None:
        """
        Flip the sign (local Z) of a node.

        This method flips the sign of the specified node in the graph state.
        Note that the application of the Z gate is different from `flip_sign`
        if there exists an edge from the node.

        Parameters
        ----------
        node : int
            The graph node for which to flip the sign.

        Returns
        -------
        None
        """
        self.nodes[node]["sign"] = not self.nodes[node]["sign"]

    def advance(self, node: int) -> None:
        """
        Flip the loop (local S) of a specified node in the graph.

        This method modifies the state of the loop associated with the given node.
        If the loop already exists, the sign is flipped, reflecting the relation
        SS = Z. Note that the application of the S gate differs from `advance`
        if there is an edge connected to the node.

        Parameters
        ----------
        node : int
            The graph node for which to advance the loop.

        Returns
        -------
        None
        """
        if self.nodes[node]["loop"]:
            self.nodes[node]["loop"] = False
            self.flip_sign(node)
        else:
            self.nodes[node]["loop"] = True

    def h(self, node: int) -> None:
        """
        Apply the H gate to a specified qubit (node).

        Parameters
        ----------
        node : int
            The index of the graph node to which the H gate will be applied.

        Returns
        -------
        None
        """
        self.flip_fill(node)

    def s(self, node: int) -> None:
        """
        Apply the S gate to a specified qubit (node).

        Parameters
        ----------
        node : int
            The index of the graph node to which the S gate will be applied.

        Returns
        -------
        None
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.flip_fill(node)
                self.nodes[node]["loop"] = False
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
            else:
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
                if self.nodes[node]["sign"]:
                    for i in self.neighbors(node):
                        self.flip_sign(i)
        else:  # solid
            self.advance(node)

    def z(self, node: int) -> None:
        """
        Apply the Z gate to a qubit (node).

        Parameters
        ----------
        node : int
            The graph node to which the Z gate will be applied.

        Returns
        -------
        None
        """
        if self.nodes[node]["hollow"]:
            for i in self.neighbors(node):
                self.flip_sign(i)
            if self.nodes[node]["loop"]:
                self.flip_sign(node)
        else:  # solid
            self.flip_sign(node)

    def equivalent_graph_e1(self, node: int) -> None:
        """
        Transform a graph state to a different graph state representing the same stabilizer state.

        This transformation is applicable only to a node that has a loop.

        Parameters
        ----------
        node : int
            A graph node with a loop to which rule E1 will be applied.

        Returns
        -------
        None
        """
        if not self.nodes[node]["loop"]:
            raise ValueError("node must have loop")
        self.flip_fill(node)
        self.local_complement(node)
        for i in self.neighbors(node):
            self.advance(i)
        self.flip_sign(node)
        if self.nodes[node]["sign"]:
            for i in self.neighbors(node):
                self.flip_sign(i)

    def equivalent_graph_e2(self, node1: int, node2: int) -> None:
        """
        Transform a graph state to a different graph state representing the same stabilizer state.

        This transformation applies only to two connected nodes without a loop.

        Parameters
        ----------
        node1 : int
            The first connected graph node to apply rule E2.
        node2 : int
            The second connected graph node to apply rule E2.

        Returns
        -------
        None
        """
        if (node1, node2) not in self.edges and (node2, node1) not in self.edges:
            raise ValueError("nodes must be connected by an edge")
        if self.nodes[node1]["loop"] or self.nodes[node2]["loop"]:
            raise ValueError("nodes must not have loop")
        sg1 = self.nodes[node1]["sign"]
        sg2 = self.nodes[node2]["sign"]
        self.flip_fill(node1)
        self.flip_fill(node2)
        # local complement along edge between node1, node2
        self.local_complement(node1)
        self.local_complement(node2)
        self.local_complement(node1)
        for i in iter(set(self.neighbors(node1)) & set(self.neighbors(node2))):
            self.flip_sign(i)
        if sg1:
            self.flip_sign(node1)
            for i in self.neighbors(node1):
                self.flip_sign(i)
        if sg2:
            self.flip_sign(node2)
            for i in self.neighbors(node2):
                self.flip_sign(i)

    def equivalent_fill_node(self, node: int) -> int:
        """
        Fill the chosen node by applying graph transformation rules E1 and E2.

        If the selected node is hollow and isolated, it cannot be filled,
        and a warning is raised.

        Parameters
        ----------
        node : int
            The index of the node to fill.

        Returns
        -------
        result : int
            - 1 if the selected node is hollow and isolated.
            - 2 if the node is filled and isolated.
            - 0 otherwise.
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.equivalent_graph_e1(node)
                return 0
            # node = hollow and loopless
            if utils.iter_empty(self.neighbors(node)):
                return 1
            for i in self.neighbors(node):
                if not self.nodes[i]["loop"]:
                    self.equivalent_graph_e2(node, i)
                    return 0
            # if all neighbor has loop, pick one and apply E1, then E1 to the node.
            i = next(self.neighbors(node))
            self.equivalent_graph_e1(i)  # this gives loop to node.
            self.equivalent_graph_e1(node)
            return 0
        if utils.iter_empty(self.neighbors(node)):
            return 2
        return 0

    def measure_x(self, node: int, choice: Outcome = 0) -> Outcome:
        """
        Perform measurement in the X basis.

        According to the original paper, X measurement is realized by
        applying the Hadamard (H) gate to the measured node before
        performing a Z measurement.

        Parameters
        ----------
        node : int
            The index of the qubit to be measured.
        choice : Outcome, optional
            The choice of measurement outcome. Observes (-1) ** choice.
            The default is 0.

        Returns
        -------
        Outcome
            The measurement outcome, which is either 0 or 1.
        """
        if choice not in {0, 1}:
            raise ValueError("choice must be 0 or 1")
        # check if isolated
        if utils.iter_empty(self.neighbors(node)):
            if self.nodes[node]["hollow"] or self.nodes[node]["loop"]:
                choice_ = choice
            elif self.nodes[node]["sign"]:  # isolated and state is |->
                choice_ = 1
            else:  # isolated and state is |+>
                choice_ = 0
            self.remove_node(node)
            return choice_
        self.h(node)
        return self.measure_z(node, choice=choice)

    def measure_y(self, node: int, choice: Outcome = 0) -> Outcome:
        """
        Perform measurement in the Y basis.

        According to the original paper, we realize Y measurement by
        applying S, Z, and H gates to the measured node before performing
        a Z measurement.

        Parameters
        ----------
        node : int
            The index of the qubit to be measured.
        choice : Outcome, optional
            The choice of measurement outcome. The observable is
            (-1) ** choice. Default is 0.

        Returns
        -------
        Outcome
            The measurement outcome, which will be either 0 or 1.
        """
        if choice not in {0, 1}:
            raise ValueError("choice must be 0 or 1")
        self.s(node)
        self.z(node)
        self.h(node)
        return self.measure_z(node, choice=choice)

    def measure_z(self, node: int, choice: Outcome = 0) -> Outcome:
        """
        Perform measurement in the Z basis.

        This method realizes a simple Z measurement on an undecorated graph state
        by filling the measured node and removing the local Hadamard gate.

        Parameters
        ----------
        node : int
            The index of the qubit to be measured.
        choice : Outcome, optional
            The choice of measurement outcome, where the observed outcome is
            (-1) ** choice. The default is 0.

        Returns
        -------
        result : Outcome
            The measurement outcome, which will be either 0 or 1.
        """
        if choice not in {0, 1}:
            raise ValueError("choice must be 0 or 1")
        isolated = self.equivalent_fill_node(node)
        if choice:
            for i in self.neighbors(node):
                self.flip_sign(i)
        result = choice if not isolated else outcome(self.nodes[node]["sign"])
        self.remove_node(node)
        return result

    def draw(self, fill_color: str = "C0", **kwargs: dict[str, Any]) -> None:
        """
        Draw a decorated graph state.

        Negative nodes are indicated by a negative sign on the node labels.

        Parameters
        ----------
        fill_color : str, optional
            The fill color of the nodes. Default is "C0".

        kwargs : keyword arguments, optional
            Additional arguments to be passed to `networkx.draw()`.
        """
        nqubit = len(self.nodes)
        nodes = list(self.nodes)
        edges: list[tuple[int, int]] = list(self.edges)
        labels = {i: i for i in iter(self.nodes)}
        colors = [fill_color for _ in range(nqubit)]
        for i in range(nqubit):
            if self.nodes[nodes[i]]["loop"]:
                edges.append((nodes[i], nodes[i]))
            if self.nodes[nodes[i]]["hollow"]:
                colors[i] = "white"
            if self.nodes[nodes[i]]["sign"]:
                labels[nodes[i]] = -1 * labels[nodes[i]]
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        nx.draw(g, labels=labels, node_color=colors, edgecolors="k", **kwargs)

    def to_statevector(self) -> Statevec:
        """
        Convert the graph state into a state vector.

        Returns
        -------
        Statevec
            The state vector representation of the graph state.
        """
        node_list = list(self.nodes)
        nqubit = len(self.nodes)
        gstate = Statevec(nqubit=nqubit)
        # map graph node indices into 0 - (nqubit-1) for qubit indexing in statevec
        imapping = {node_list[i]: i for i in range(nqubit)}
        mapping = [node_list[i] for i in range(nqubit)]
        for i, j in self.edges:
            gstate.entangle((imapping[i], imapping[j]))
        for i in range(nqubit):
            if self.nodes[mapping[i]]["sign"]:
                gstate.evolve_single(Ops.Z, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["loop"]:
                gstate.evolve_single(Ops.S, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["hollow"]:
                gstate.evolve_single(Ops.H, i)
        return gstate

    def get_isolates(self) -> list[int]:
        """
        Returns a list of isolated nodes in the graph.

        An isolated node is defined as a node that has no edges connected to it.

        Returns
        -------
        list[int]
            A list of the identifiers of isolated nodes.
        """
        return list(nx.isolates(self))
