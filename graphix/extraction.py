"Functions to extract the fusion network from a given graph state."

from __future__ import annotations

import copy
import dataclasses
import operator
from enum import Enum

import networkx as nx
import numpy as np

from graphix.graphsim import GraphState


class ResourceType(Enum):
    """
    Represents a type of resource.

    Attributes
    ----------
    name : str
        The name of the resource type.
    description : str
        A brief description of the resource type.
    capacity : int
        The maximum capacity of the resource type.

    Methods
    -------
    __init__(name: str, description: str, capacity: int) -> None:
        Initializes a new instance of the ResourceType class.
    __str__() -> str:
        Returns a string representation of the ResourceType instance.
    """

    GHZ = "GHZ"
    LINEAR = "LINEAR"
    NONE = None

    def __str__(self) -> str:
        """
        Returns the name of the resource type.

        Returns
        -------
        str
            The name of the resource type.
        """
        return self.name


@dataclasses.dataclass
class ResourceGraph:
    """
    Resource graph state object.

    Parameters
    ----------
    cltype : ResourceType
        Type of the cluster.
    graph : graphix.graphsim.GraphState
        Graph state of the cluster.
    """

    cltype: ResourceType
    graph: GraphState

    def __eq__(self, other: object) -> bool:
        """
        Determine if two resource graphs are equal.

        This method checks if the current resource graph is structurally
        and functionally identical to another resource graph.

        Parameters
        ----------
        other : object
            The object to compare against. This is typically another instance
            of `ResourceGraph`.

        Returns
        -------
        bool
            `True` if the two resource graphs are equal, `False` otherwise.
        """
        if not isinstance(other, ResourceGraph):
            raise TypeError("cannot compare ResourceGraph with other object")

        return self.cltype == other.cltype and nx.utils.graphs_equal(self.graph, other.graph)


def get_fusion_network_from_graph(
    graph: GraphState,
    max_ghz: float = np.inf,
    max_lin: float = np.inf,
) -> list[ResourceGraph]:
    """
    Extract GHZ and linear cluster graph state decomposition of the desired resource state :class:`~graphix.graphsim.GraphState`.

    The extraction algorithm is based on the work by Zilk et al. (2022) [1].

    References
    ----------
    [1] Zilk et al., A compiler for universal photonic quantum computers, 2022.
        `arXiv:2210.09251 <https://arxiv.org/abs/2210.09251>`_

    Parameters
    ----------
    graph : :class:`~graphix.graphsim.GraphState`
        The graph state to be decomposed.
    max_ghz : float, optional
        Maximum size of GHZ clusters (default is np.inf).
    max_lin : float, optional
        Maximum size of linear clusters (default is np.inf).

    Returns
    -------
    list[ResourceGraph]
        A list of :class:`ResourceGraph` objects representing the decomposed clusters.
    """
    adjdict = {k: dict(copy.deepcopy(v)) for k, v in graph.adjacency()}

    number_of_edges = graph.number_of_edges()
    resource_list = []
    neighbors_list = []

    # Prepare a list sorted by number of neighbors to get the largest GHZ clusters first.
    for v, va in adjdict.items():
        if len(va) > 2:
            neighbors_list.append((v, len(va)))
        # If there is an isolated node, add it to the list.
        if len(va) == 0:
            resource_list.append(create_resource_graph([v], root=v))

    # Find GHZ graphs in the graph and remove their edges from the graph.
    # All nodes that have more than 2 edges become the roots of the GHZ clusters.
    for v, _ in sorted(neighbors_list, key=operator.itemgetter(1), reverse=True):
        if len(adjdict[v]) > 2:
            nodes = [v]
            while len(adjdict[v]) > 0 and len(nodes) < max_ghz:
                n, _ = adjdict[v].popitem()
                nodes.append(n)
                del adjdict[n][v]
                number_of_edges -= 1
            resource_list.append(create_resource_graph(nodes, root=v))

    # Find Linear clusters in the remaining graph and remove their edges from the graph.
    while number_of_edges != 0:
        for v, va in adjdict.items():
            if len(va) == 1:
                n = v
                nodes = [n]
                while len(adjdict[n]) > 0 and len(nodes) < max_lin:
                    n2, _ = adjdict[n].popitem()
                    nodes.append(n2)
                    del adjdict[n2][n]
                    number_of_edges -= 1
                    n = n2

                # We define any cluster whose size is smaller than 4, a GHZ cluster
                if len(nodes) == 3:
                    resource_list.append(create_resource_graph([nodes[1], nodes[0], nodes[2]], root=nodes[1]))
                elif len(nodes) == 2:
                    resource_list.append(create_resource_graph(nodes, root=nodes[0]))
                else:
                    resource_list.append(create_resource_graph(nodes))

        # If a cycle exists in the graph, extract one 3-qubit ghz cluster from the cycle.
        for v, va in adjdict.items():
            if len(va) == 2:
                neighbors = list(va.keys())
                nodes = [v, *neighbors]
                del adjdict[neighbors[0]][v]
                del adjdict[neighbors[1]][v]
                del va[neighbors[0]]
                del va[neighbors[1]]
                number_of_edges -= 2

                resource_list.append(create_resource_graph(nodes, root=v))
                break
    return resource_list


def create_resource_graph(node_ids: list[int], root: int | None = None) -> ResourceGraph:
    """
    Create a resource graph state (GHZ or linear) from node ids.

    Parameters
    ----------
    node_ids : list[int]
        List of node ids.
    root : int, optional
        Root of the GHZ cluster. If None, a linear cluster is created. Defaults to None.

    Returns
    -------
    ResourceGraph
        A `ResourceGraph` object representing the constructed resource graph.
    """
    cluster_type = None
    edges = []
    if root is not None:
        edges = [(root, i) for i in node_ids if i != root]
        cluster_type = ResourceType.GHZ
    else:
        edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids)) if i + 1 < len(node_ids)]
        cluster_type = ResourceType.LINEAR
    tmp_graph = GraphState()
    tmp_graph.add_nodes_from(node_ids)
    tmp_graph.add_edges_from(edges)
    return ResourceGraph(cltype=cluster_type, graph=tmp_graph)


def get_fusion_nodes(c1: ResourceGraph, c2: ResourceGraph) -> list[int]:
    """
    Get the nodes that are fused between two resource states. Currently, we consider only type-I fusion.

    References
    ----------
    [1] Daniel E. Browne and Terry Rudolph. Resource-efficient linear optical quantum computation.
        Physical Review Letters, 95(1):010501, 2005.

    Parameters
    ----------
    c1 : ResourceGraph
        First resource state to be fused.
    c2 : ResourceGraph
        Second resource state to be fused.

    Returns
    -------
    list[int]
        List of nodes that are fused between the two clusters.
    """
    if not isinstance(c1, ResourceGraph) or not isinstance(c2, ResourceGraph):
        raise TypeError("c1 and c2 must be Cluster objects")

    if c1 == c2:
        return []
    return [n for n in c1.graph.nodes if n in c2.graph.nodes]
