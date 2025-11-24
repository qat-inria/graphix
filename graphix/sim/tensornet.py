"""
Tensor Network Simulator for Measurement-Based Quantum Computing (MBQC).

This module provides tools for simulating quantum circuits using tensor network methods,
specifically tailored for the measurement-based model of quantum computation.
It allows for the representation and manipulation of quantum states as tensor networks,
facilitating the analysis and execution of MBQC protocols.

Usage
-----
To use this module, import the necessary classes and functions, and create a tensor network
representation of your quantum state. You can then perform measurements and simulate
quantum gates within a measurement-based framework.

Functions and Classes
---------------------
- [List the functions and classes available in this module, if applicable]

Examples
--------
1. Import the module:
   ```python
   from tensor_network_simulator import MBQC
   ```

2. Create and manipulate a tensor network:
   ```python
   network = MBQC.TensorNetwork(...)
   ```

3. Simulate measurements:
   ```python
   result = network.measure(...)
   ```

This module is intended for researchers and practitioners in quantum computing
seeking to explore the capabilities of MBQC through tensor network methods.
"""

from __future__ import annotations

import string
import sys
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex

import numpy as np
import numpy.typing as npt
import quimb.tensor as qtn
from quimb.tensor import Tensor, TensorNetwork

# TypeAlias introduced in Python 3.10
# override introduced in Python 3.12
from typing_extensions import TypeAlias, override

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.ops import Ops
from graphix.parameter import Expression
from graphix.sim.base_backend import Backend, BackendState
from graphix.states import BasicStates, PlanarState

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cotengra.oe import PathOptimizer
    from numpy.random import Generator

    from graphix import Pattern
    from graphix.clifford import Clifford
    from graphix.measurements import Measurement, Outcome
    from graphix.sim import Data
    from graphix.simulator import MeasureMethod

if sys.version_info >= (3, 10):
    PrepareState: TypeAlias = str | npt.NDArray[np.complex128]
else:
    from typing import Union

    PrepareState: TypeAlias = Union[str, npt.NDArray[np.complex128]]


class MBQCTensorNet(BackendState, TensorNetwork):
    """
    Tensor Network Simulator interface for Measurement-Based Quantum Computation (MBQC) patterns.

    This class utilizes the `quimb.tensor.core.TensorNetwork` to facilitate
    the simulation of quantum circuits and operations relevant to MBQC.

    Attributes
    ----------
    tensor_network : TensorNetwork
        The underlying tensor network representing the quantum state.
    qubits : int
        Number of qubits in the quantum circuit.
    measurements : list of tuples
        A list specifying the measurement outcomes and their corresponding qubit indices.

    Methods
    -------
    add_qubit():
        Adds a qubit to the tensor network.

    apply_gate(gate, qubit_indices):
        Applies a quantum gate to the specified qubits in the tensor network.

    measure(qubit_index):
        Performs a measurement on the specified qubit and updates the tensor network.

    simulate():
        Runs the simulation of the quantum circuit based on the current state of the tensor network.
    """

    _dangling: dict[str, str]

    def __init__(
        self,
        branch_selector: BranchSelector,
        graph_nodes: Iterable[int] | None = None,
        graph_edges: Iterable[tuple[int, int]] | None = None,
        default_output_nodes: Iterable[int] | None = None,
        ts: list[TensorNetwork] | TensorNetwork | None = None,
        virtual: bool = False,
    ) -> None:
        """
        Initialize MBQCTensorNet.

        Parameters
        ----------
        branch_selector : BranchSelector
            Selector for branches in the MBQC.
        graph_nodes : Iterable[int] or None, optional
            List of integer node indices of the graph state.
        graph_edges : Iterable[tuple[int, int]] or None, optional
            List of tuples representing edge indices of the graph state.
        default_output_nodes : Iterable[int] or None, optional
            Output node indices at the end of MBQC operations, if known in advance.
        ts : list[TensorNetwork] or TensorNetwork or None, optional
            Optional initial state(s) of the tensor network. Can be a single tensor network or a list of them.
        virtual : bool, optional
            Flag indicating if the network operates in a virtual mode. Default is False.
        """
        if ts is None:
            ts = []
        super().__init__(ts=ts, virtual=virtual)
        self._dangling = ts._dangling if isinstance(ts, MBQCTensorNet) else {}
        self.default_output_nodes = None if default_output_nodes is None else list(default_output_nodes)
        # prepare the graph state if graph_nodes and graph_edges are given
        if graph_nodes is not None and graph_edges is not None:
            self.set_graph_state(graph_nodes, graph_edges)
        self.__branch_selector = branch_selector

    def get_open_tensor_from_index(self, index: int | str) -> npt.NDArray[np.complex128]:
        """
        Get tensor specified by node index. The tensor has a dangling edge.

        Parameters
        ----------
        index : int or str
            Node index.

        Returns
        -------
        numpy.ndarray
            Specified tensor with complex data type.
        """
        if isinstance(index, int):
            index = str(index)
        assert isinstance(index, str)
        tags = [index, "Open"]
        tid = next(iter(self._get_tids_from_tags(tags, which="all")))
        tensor = self.tensor_map[tid]
        return tensor.data.astype(dtype=np.complex128)

    def add_qubit(self, index: int, state: PrepareState = "plus") -> None:
        """
        Add a single qubit to the network.

        Parameters
        ----------
        index : int
            Index of the new qubit.
        state : PrepareState, optional
            Initial state of the new qubit. Can be one of the following:
            - "plus"
            - "minus"
            - "zero"
            - "one"
            - "iplus"
            - "iminus"
            - or a 1x2 numpy.ndarray representing an arbitrary state.

            The default is "plus".
        """
        ind = gen_str()
        tag = str(index)
        if state == "plus":
            vec = BasicStates.PLUS.get_statevector()
        elif state == "minus":
            vec = BasicStates.MINUS.get_statevector()
        elif state == "zero":
            vec = BasicStates.ZERO.get_statevector()
        elif state == "one":
            vec = BasicStates.ONE.get_statevector()
        elif state == "iplus":
            vec = BasicStates.PLUS_I.get_statevector()
        elif state == "iminus":
            vec = BasicStates.MINUS_I.get_statevector()
        else:
            if isinstance(state, str):
                raise TypeError(f"Unknown state: {state}")
            if state.shape != (2,):
                raise ValueError("state must be 2-element np.ndarray")
            if not np.isclose(np.linalg.norm(state), 1):
                raise ValueError("state must be normalized")
            vec = state
        tsr = Tensor(vec, [ind], [tag, "Open"])
        self.add_tensor(tsr)
        self._dangling[tag] = ind

    def evolve_single(self, index: int, arr: npt.NDArray[np.complex128], label: str = "U") -> None:
        """
        Apply a single-qubit operator to a qubit at the specified index.

        Parameters
        ----------
        index : int
            The index of the qubit to which the operator will be applied.
        arr : npt.NDArray[np.complex128]
            A 2x2 numpy array representing the single-qubit operator.
        label : str, optional
            A label for the gate, defaults to "U".

        Returns
        -------
        None
        """
        old_ind = self._dangling[str(index)]
        tid = list(self._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]

        new_ind = gen_str()
        tensor.retag({"Open": "Close"}, inplace=True)

        node_ts = Tensor(
            arr,
            [new_ind, old_ind],
            [str(index), label, "Open"],
        )
        self._dangling[str(index)] = new_ind
        self.add_tensor(node_ts)

    def add_qubits(self, indices: Sequence[int], states: PrepareState | Iterable[PrepareState] = "plus") -> None:
        """
        Add qubits to the network.

        Parameters
        ----------
        indices : Sequence[int]
            Indices of the new qubits.
        states : PrepareState or Iterable[PrepareState], optional
            Initial state or list of initial states of the new qubits.
            Defaults to "plus".
        """
        if isinstance(states, str):
            states_iter: list[PrepareState] = [states] * len(indices)
        else:
            states_list = list(states)
            # `states` is of type `PrepareState`, a type alias for
            # `str | npt.NDArray[np.complex128]`. To distinguish
            # between the two cases, we just need to check whether
            # an element is a character or a complex number.
            if len(states_list) == 0 or isinstance(states_list[0], SupportsComplex):
                states_iter = [np.array(states_list)] * len(indices)
            else:

                def get_prepare_state(item: PrepareState | SupportsComplex) -> PrepareState:
                    if isinstance(item, SupportsComplex):
                        raise TypeError("Unexpected complex")
                    return item

                states_iter = [get_prepare_state(item) for item in states_list]
        for ind, state in zip(indices, states_iter):
            self.add_qubit(ind, state)

    def measure_single(
        self,
        index: int,
        basis: str | npt.NDArray[np.complex128] = "Z",
        bypass_probability_calculation: bool = True,
        outcome: Outcome | None = None,
        rng: Generator | None = None,
    ) -> Outcome:
        """
        Measure a node in a specified basis. Note that this does not perform the partial trace.

        Parameters
        ----------
        index : int
            Index of the node to be measured.
        basis : str or np.ndarray, optional
            Measurement basis, which can be "Z", "X", or "Y" for Pauli basis measurements,
            or a 1x2 numpy.ndarray for arbitrary measurement bases. Default is "Z".
        bypass_probability_calculation : bool, optional
            If True (default), skips the calculation of the probability of the measurement
            result and uses equal probability for each outcome. If False, calculates the
            probability of the measurement result from the state.
        outcome : Outcome or None, optional
            User-chosen measurement result, specifying the outcome of (-1)^{outcome}.
            Default is None.
        rng : Generator or None, optional
            Random number generator for stochastic measurements. Default is None.

        Returns
        -------
        Outcome
            The measurement result.
        """
        if bypass_probability_calculation:
            result = outcome if outcome is not None else self.__branch_selector.measure(index, lambda: 0.5, rng=rng)
            # Basis state to be projected
            if isinstance(basis, np.ndarray):
                if outcome is not None:
                    raise Warning("Measurement outcome is chosen but the basis state was given.")
                proj_vec = basis
            elif basis == "Z" and result == 0:
                proj_vec = BasicStates.ZERO.get_statevector()
            elif basis == "Z" and result == 1:
                proj_vec = BasicStates.ONE.get_statevector()
            elif basis == "X" and result == 0:
                proj_vec = BasicStates.PLUS.get_statevector()
            elif basis == "X" and result == 1:
                proj_vec = BasicStates.MINUS.get_statevector()
            elif basis == "Y" and result == 0:
                proj_vec = BasicStates.PLUS_I.get_statevector()
            elif basis == "Y" and result == 1:
                proj_vec = BasicStates.MINUS_I.get_statevector()
            else:
                raise ValueError("Invalid measurement basis.")
        else:
            raise NotImplementedError("Measurement probability calculation not implemented.")
        old_ind = self._dangling[str(index)]
        proj_ts = Tensor(proj_vec, [old_ind], [str(index), "M", "Close", "ancilla"]).H
        # add the tensor to the network
        tid = list(self._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]
        tensor.retag({"Open": "Close"}, inplace=True)
        self.add_tensor(proj_ts)
        return result

    def set_graph_state(self, nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> None:
        """
        Prepare the graph state without directly applying CZ gates.

        Parameters
        ----------
        nodes : iterable of int
            A set of nodes in the graph.
        edges : iterable of tuple of int
            A set of edges represented as tuples, where each tuple contains two integers
            indicating the nodes connected by the edge.

        See Also
        --------
        :meth:`~graphix.sim.tensornet.TensorNetworkBackend.__init__()`
        """
        ind_dict: dict[int, list[str]] = {}
        vec_dict: dict[int, list[bool]] = {}
        for edge in edges:
            for node in edge:
                if node not in ind_dict:
                    ind = gen_str()
                    self._dangling[str(node)] = ind
                    ind_dict[node] = [ind]
                    vec_dict[node] = []
            greater = edge[0] > edge[1]  # true for 1/0, false for +/-
            vec_dict[edge[0]].append(greater)
            vec_dict[edge[1]].append(not greater)

            ind = gen_str()
            ind_dict[edge[0]].append(ind)
            ind_dict[edge[1]].append(ind)

        for node in nodes:
            if node not in ind_dict:
                ind = gen_str()
                self._dangling[str(node)] = ind
                self.add_tensor(Tensor(BasicStates.PLUS.get_statevector(), [ind], [str(node), "Open"]))
                continue
            dim_tensor = len(vec_dict[node])
            tensor = np.array(
                [
                    outer_product(
                        [BasicStates.VEC[0 + 2 * vec_dict[node][i]].get_statevector() for i in range(dim_tensor)]
                    ),
                    outer_product(
                        [BasicStates.VEC[1 + 2 * vec_dict[node][i]].get_statevector() for i in range(dim_tensor)]
                    ),
                ]
            ) * 2 ** (dim_tensor / 4 - 1.0 / 2)
            self.add_tensor(Tensor(tensor, ind_dict[node], [str(node), "Open"]))

    def _require_default_output_nodes(self) -> list[int]:
        if self.default_output_nodes is None:
            raise ValueError("output_nodes is not set.")
        return self.default_output_nodes

    def get_basis_coefficient(
        self, basis: int | str, normalize: bool = True, indices: Sequence[int] | None = None
    ) -> complex:
        """
        Calculate the coefficient of a given computational basis.

        Parameters
        ----------
        basis : int or str
            Computational basis expressed in binary (str) or integer, e.g., '101' or 5.
        normalize : bool, optional
            If True, normalize the coefficient by the norm of the entire state. Default is True.
        indices : Sequence[int], optional
            Target qubit indices to compute the coefficients. Default is the MBQC output nodes
            (self.default_output_nodes).

        Returns
        -------
        coef : complex
            The coefficient associated with the specified basis.
        """
        if indices is None:
            indices = self._require_default_output_nodes()
        if isinstance(basis, str):
            basis = int(basis, 2)
        tn = self.copy()
        # prepare projected state
        for i in range(len(indices)):
            node = str(indices[i])
            exp = len(indices) - i - 1
            if (basis // 2**exp) == 1:
                state_out = BasicStates.ONE.get_statevector()  # project onto |1>
                basis -= 2**exp
            else:
                state_out = BasicStates.ZERO.get_statevector()  # project onto |0>
            tensor = Tensor(state_out, [tn._dangling[node]], [node, f"qubit {i}", "Close"])
            # retag
            old_ind = tn._dangling[node]
            tid = next(iter(tn._get_tids_from_inds(old_ind)))
            tn.tensor_map[tid].retag({"Open": "Close"})
            tn.add_tensor(tensor)

        # contraction
        tn_simplified = tn.full_simplify("ADCR")
        coef = tn_simplified.contract(output_inds=[])
        if normalize:
            norm = self.get_norm()
            return coef / norm
        return coef

    def get_basis_amplitude(self, basis: str | int) -> float:
        """Calculate the probability amplitude of the specified computational basis state.

        Parameters
        ----------
        basis : int or str
            computational basis expressed in binary (str) or integer, e.g. 101 or 5.

        Returns
        -------
        float :
            the probability amplitude of the specified state.
        """
        if isinstance(basis, str):
            basis = int(basis, 2)
        coef = self.get_basis_coefficient(basis)
        return abs(coef) ** 2

    def to_statevector(self, indices: Sequence[int] | None = None) -> npt.NDArray[np.complex128]:
        """
        Retrieve the statevector from the tensor network.

        This method tends to be slow; however, there are plans to parallelize its execution.

        Parameters
        ----------
        indices : Sequence[int], optional
            List of target qubit indices. Default is the MBQC output nodes (self.default_output_nodes).

        Returns
        -------
        npt.NDArray[np.complex128]
            The statevector obtained from the tensor network.
        """
        n_qubit = len(self._require_default_output_nodes()) if indices is None else len(indices)
        statevec: npt.NDArray[np.complex128] = np.zeros(2**n_qubit, np.complex128)
        for i in range(len(statevec)):
            statevec[i] = self.get_basis_coefficient(i, normalize=False, indices=indices)
        return statevec / np.linalg.norm(statevec)

    def flatten(self) -> npt.NDArray[np.complex128]:
        """
        Return a flattened state vector.

        The state vector is transformed into a one-dimensional array while retaining
        its complex number format. This is useful for various calculations and
        manipulations where a flat representation of the state is needed.

        Returns
        -------
        npt.NDArray[np.complex128]
            A one-dimensional array containing the flattened state vector elements.
        """
        return self.to_statevector().flatten()

    def get_norm(self, optimize: str | PathOptimizer | None = None) -> float:
        """
        Calculate the norm of the state.

        Parameters
        ----------
        optimize : str or PathOptimizer, optional
            The optimization method to use. If None, no optimization is performed.

        Returns
        -------
        float
            The norm of the state.
        """
        tn_cp1 = self.copy()
        tn_cp2 = tn_cp1.conj()
        tn = TensorNetwork([tn_cp1, tn_cp2])
        tn_simplified = tn.full_simplify("ADCR")
        contraction = tn_simplified.contract(output_inds=[], optimize=optimize)
        return float(abs(contraction) ** 0.5)

    def expectation_value(
        self,
        op: npt.NDArray[np.complex128],
        qubit_indices: Sequence[int],
        output_node_indices: Iterable[int] | None = None,
        optimize: str | PathOptimizer | None = None,
    ) -> float:
        """
        Calculate the expectation value of the given operator.

        Parameters
        ----------
        op : numpy.ndarray
            Single- or multi-qubit Hermitian operator.
        qubit_indices : Sequence[int]
            Indices of the logical qubits where the operator is applied.
        output_node_indices : Iterable[int] or None, optional
            Indices of nodes in the entire tensor network that remain unmeasured after MBQC operations.
            Defaults to the output nodes specified in the measurement pattern (self.default_output_nodes).
        optimize : str, PathOptimizer or None, optional
            Optimization method to be used. Defaults to None.

        Returns
        -------
        float
            The expectation value of the operator.
        """
        out_inds = self._require_default_output_nodes() if output_node_indices is None else list(output_node_indices)
        target_nodes = [out_inds[ind] for ind in qubit_indices]
        op_dim = len(qubit_indices)
        op = op.reshape([2 for _ in range(2 * op_dim)])
        new_ind_left = [gen_str() for _ in range(op_dim)]
        new_ind_right = [gen_str() for _ in range(op_dim)]
        tn_cp_left = self.copy()
        op_ts = Tensor(op, new_ind_right + new_ind_left, ["Expectation Op.", "Close"])
        tn_cp_right = tn_cp_left.conj()

        # reindex & retag
        for node in out_inds:
            old_ind = tn_cp_left._dangling[str(node)]
            tid_left = next(iter(tn_cp_left._get_tids_from_inds(old_ind)))
            tid_right = next(iter(tn_cp_right._get_tids_from_inds(old_ind)))
            if node in target_nodes:
                tn_cp_left.tensor_map[tid_left].reindex({old_ind: new_ind_left[target_nodes.index(node)]}, inplace=True)
                tn_cp_right.tensor_map[tid_right].reindex(
                    {old_ind: new_ind_right[target_nodes.index(node)]}, inplace=True
                )
            tn_cp_left.tensor_map[tid_left].retag({"Open": "Close"})
            tn_cp_right.tensor_map[tid_right].retag({"Open": "Close"})
        tn_cp_left.add([op_ts, tn_cp_right])

        # contraction
        tn_cp_left = tn_cp_left.full_simplify("ADCR")
        exp_val = tn_cp_left.contract(output_inds=[], optimize=optimize)
        norm = self.get_norm(optimize=optimize)
        return exp_val / norm**2

    def evolve(self, operator: npt.NDArray[np.complex128], qubit_indices: list[int], decompose: bool = True) -> None:
        """
        Apply an arbitrary operator to the quantum state.

        Parameters
        ----------
        operator : numpy.ndarray, shape (N, N)
            The operator to be applied to the quantum state. It is assumed to be a square matrix with complex entries.
        qubit_indices : list of int
            The positions of the logical qubits to which the operator will be applied.
        decompose : bool, optional
            Whether to decompose the given operator into a Matrix Product Operator (MPO). Default is True.

        Notes
        -----
        If `decompose` is set to True, the operator will be decomposed. Otherwise, it will be applied directly to the state.
        """
        if len(operator.shape) != len(qubit_indices) * 2:
            shape = [2 for _ in range(2 * len(qubit_indices))]
            operator = operator.reshape(shape)

        # operator indices
        default_output_nodes = self._require_default_output_nodes()
        node_indices = [default_output_nodes[index] for index in qubit_indices]
        old_ind_list = [self._dangling[str(index)] for index in node_indices]
        new_ind_list = [gen_str() for _ in range(len(node_indices))]
        for i in range(len(node_indices)):
            self._dangling[str(node_indices[i])] = new_ind_list[i]

        ts: Tensor | TensorNetwork = Tensor(
            operator,
            new_ind_list + old_ind_list,
            [str(index) for index in node_indices],
        )
        if decompose:  # decompose tensor into Matrix Product Operator(MPO)
            tensors: list[Tensor | TensorNetwork] = []
            bond_inds: dict[int, str | None] = {0: None}
            for i in range(len(node_indices) - 1):
                bond_inds[i + 1] = gen_str()
                left_inds: list[str] = [new_ind_list[i], old_ind_list[i]]
                bond_ind = bond_inds[i]
                if bond_ind is not None:
                    left_inds.append(bond_ind)
                unit_tensor, ts = ts.split(left_inds=left_inds, bond_ind=bond_inds[i + 1])
                tensors.append(unit_tensor)
            tensors.append(ts)
            ts = TensorNetwork(tensors)
        self.add(ts)

    @override
    def copy(self, virtual: bool = False, deep: bool = False) -> MBQCTensorNet:
        """
        Return a copy of this object.

        Parameters
        ----------
        virtual : bool, optional
            Defaults to False.
            Whether to create a virtual copy (shared data) or not.

        deep : bool, optional
            Defaults to False.
            Whether to copy the underlying data as well.

        Returns
        -------
        MBQCTensorNet
            A duplicated object of the current instance.
        """
        if deep:
            return deepcopy(self)
        return self.__class__(branch_selector=self.__branch_selector, ts=self)


def _get_decomposed_cz() -> list[npt.NDArray[np.complex128]]:
    """
    Return the decomposed CZ tensors.

    This is an internal method.

    The CZ gate can be decomposed into two 3-rank tensors (Schmidt rank = 2).
    Decomposing into low-rank tensors is an important preprocessing step for
    optimal contraction path searching. Therefore, in this backend, the
    DECOMPOSED_CZ gate is applied instead of the original CZ gate.

    The decomposition of the CZ gate is illustrated as follows:

            output            output
            |    |           |      |
           --------   SVD   ---    ---
           |  CZ  |   -->   |L|----|R|
           --------         ---    ---
            |    |           |      |
            input             input

    Returns
    -------
    list[npt.NDArray[np.complex128]]
        A list containing the decomposed CZ tensors.
    """
    cz_ts = Tensor(
        Ops.CZ.reshape((2, 2, 2, 2)).astype(np.complex128),
        ["O1", "O2", "I1", "I2"],
        ["CZ"],
    )
    decomposed_cz = cz_ts.split(left_inds=["O1", "I1"], right_inds=["O2", "I2"], max_bond=4)
    return [
        decomposed_cz.tensors[0].data.astype(np.complex128),
        decomposed_cz.tensors[1].data.astype(np.complex128),
    ]


@dataclass(frozen=True)
class _AbstractTensorNetworkBackend(Backend[MBQCTensorNet], ABC):
    state: MBQCTensorNet
    pattern: Pattern
    graph_prep: str
    input_state: Data
    branch_selector: BranchSelector
    output_nodes: list[int]
    results: dict[int, Outcome]
    _decomposed_cz: list[npt.NDArray[np.complex128]]
    _isolated_nodes: set[int]


@dataclass(frozen=True)
class TensorNetworkBackend(_AbstractTensorNetworkBackend):
    """
    Tensor Network Simulator for MBQC.

    Executes the measurement pattern using Tensor Network (TN) expressions of graph states.

    Parameters
    ----------
    pattern : graphix.Pattern
        The measurement pattern to be executed.
    graph_prep : str
        The method for preparing a graph state. Options include:
        - 'parallel':
            A faster method for preparing a graph state. The expression of a graph state can be obtained from the graph geometry.
            Refer to https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.052315 for detailed calculations.
            Note that 'N' and 'E' commands in the measurement pattern are ignored.
        - 'sequential':
            Executes 'N' and 'E' commands sequentially, strictly following the measurement pattern. All 'N' and 'E' commands are executed in this strategy.
        - 'auto' (default):
            Automatically selects a preparation strategy based on the maximum degree of the graph.
    input_state : preparation for input states
        Only BasicStates.PLUS is currently supported for tensor networks.
    branch_selector : graphix.branch_selector.BranchSelector, optional
        A branch selector to be used for measurements.
    """

    def __init__(
        self,
        pattern: Pattern,
        graph_prep: str = "auto",
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
    ) -> None:
        """
        Construct a tensor network backend.

        Parameters
        ----------
        pattern : Pattern
            The pattern that defines the structure of the tensor network.

        graph_prep : str, optional
            The method for preparing the tensor network graph.
            Defaults to "auto".

        input_state : Data, optional
            The initial input state to be used in the tensor network.
            If None, a default initial state will be used.
            Defaults to None.

        branch_selector : BranchSelector, optional
            An optional selector for branches in the network.
            If None, a default branch selector will be used.
            Defaults to None.
        """
        if input_state is None:
            input_state = BasicStates.PLUS
        elif input_state != BasicStates.PLUS:
            msg = "TensorNetworkBackend currently only supports BasicStates.PLUS as input state."
            raise NotImplementedError(msg)
        if branch_selector is None:
            branch_selector = RandomBranchSelector()
        if graph_prep in {"parallel", "sequential"}:
            pass
        elif graph_prep == "opt":
            graph_prep = "parallel"
            warnings.warn(
                f"graph preparation strategy '{graph_prep}' is deprecated and will be replaced by 'parallel'",
                stacklevel=1,
            )
        elif graph_prep == "auto":
            max_degree = pattern.compute_max_degree()
            # "parallel" does not support non standard pattern
            graph_prep = "sequential" if max_degree > 5 or not pattern.is_standard() else "parallel"
        else:
            raise ValueError(f"Invalid graph preparation strategy: {graph_prep}")
        results = deepcopy(pattern.results)
        if graph_prep == "parallel":
            if not pattern.is_standard():
                raise ValueError("parallel preparation strategy does not support not-standardized pattern")
            graph = pattern.extract_graph()
            state = MBQCTensorNet(
                graph_nodes=graph.nodes,
                graph_edges=graph.edges,
                default_output_nodes=pattern.output_nodes,
                branch_selector=branch_selector,
            )
            decomposed_cz = []
        else:  # graph_prep == "sequential":
            state = MBQCTensorNet(default_output_nodes=pattern.output_nodes, branch_selector=branch_selector)
            decomposed_cz = _get_decomposed_cz()
        isolated_nodes = pattern.extract_isolated_nodes()
        super().__init__(
            state,
            pattern,
            graph_prep,
            input_state,
            branch_selector,
            pattern.output_nodes,
            results,
            decomposed_cz,
            isolated_nodes,
        )

    @override
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        """
        Add new nodes (qubits) to the network and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be distinct from all previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes. This parameter can be
            either a single basic state or a list of basic states.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of `nodes`,
              and each node is initialized with its corresponding state.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        if data != BasicStates.PLUS:
            raise NotImplementedError(
                "TensorNetworkBackend currently only supports |+> input state (see https://github.com/TeamGraphix/graphix/issues/167)."
            )
        if self.graph_prep == "sequential":
            self.state.add_qubits(nodes)
        elif self.graph_prep == "opt":
            pass

    @override
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """
        Make entanglement between the nodes specified by the given edge.

        Parameters
        ----------
        edge : tuple of int
            A tuple specifying the two target nodes of the CZ gate.
        """
        if self.graph_prep == "sequential":
            old_inds = [self.state._dangling[str(node)] for node in edge]
            tids = self.state._get_tids_from_inds(old_inds, which="any")
            tensors = [self.state.tensor_map[tid] for tid in tids]
            new_inds = [gen_str() for _ in range(3)]

            # retag dummy indices
            for i in range(2):
                tensors[i].retag({"Open": "Close"}, inplace=True)
                self.state._dangling[str(edge[i])] = new_inds[i]
            cz_tn = TensorNetwork(
                [
                    qtn.Tensor(
                        self._decomposed_cz[0],
                        [new_inds[0], old_inds[0], new_inds[2]],
                        [str(edge[0]), "CZ", "Open"],
                    ),
                    qtn.Tensor(
                        self._decomposed_cz[1],
                        [new_inds[2], new_inds[1], old_inds[1]],
                        [str(edge[1]), "CZ", "Open"],
                    ),
                ]
            )
            self.state.add_tensor_network(cz_tn)
        elif self.graph_prep == "opt":
            pass

    @override
    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Outcome:
        """
        Perform measurement of a specified node in the tensor network.

        In the context of tensor networks, performing a measurement involves
        applying a measurement operator to the tensor, which is then directly
        contracted with the projected state.

        Parameters
        ----------
        node : int
            Index of the node to measure.
        measurement : Measurement
            The measurement object that defines the measurement plane and angle.
        rng : Generator, optional
            A random number generator for stochastic processes. If None, a default generator is used.

        Returns
        -------
        Outcome
            The outcome of the measurement process, encapsulating the results of the measurement.
        """
        if node in self._isolated_nodes:
            vector: npt.NDArray[np.complex128] = self.state.get_open_tensor_from_index(node)
            probs = (np.abs(vector) ** 2).astype(np.float64)
            probs /= np.sum(probs)
            result: Outcome = self.branch_selector.measure(node, lambda: probs[0], rng=rng)
            self.results[node] = result
            buffer = 1 / probs[result] ** 0.5
        else:
            result = self.branch_selector.measure(node, lambda: 0.5, rng=rng)
            self.results[node] = result
            buffer = 2**0.5
        if isinstance(measurement.angle, Expression):
            raise TypeError("Parameterized pattern unsupported.")
        vec = PlanarState(measurement.plane, measurement.angle).get_statevector()
        if result:
            vec = measurement.plane.orth.matrix @ vec
        proj_vec = vec * buffer
        self.state.measure_single(node, basis=proj_vec, rng=rng)
        return result

    @override
    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """
        Perform byproduct correction.

        Parameters
        ----------
        cmd : command.X | command.Z
            Byproduct command, which can be either 'X' or 'Z', along with the corresponding
            node and signal domain.
        measure_method : MeasureMethod
            The measurement method to use for correction.
        """
        if sum(measure_method.get_measure_result(j) for j in cmd.domain) % 2 == 1:
            op = Ops.X if isinstance(cmd, command.X) else Ops.Z
            self.state.evolve_single(cmd.node, op, str(cmd.kind))

    @override
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """
        Apply a single-qubit Clifford gate to the specified node.

        Parameters
        ----------
        node : int
            The index of the node to which the Clifford gate will be applied.
        clifford : Clifford
            The Clifford gate to be applied. For details on the Clifford gates,
            see https://arxiv.org/pdf/2212.11975.pdf.
        """
        self.state.evolve_single(node, clifford.matrix)

    @override
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """
        Finalize the tensor network backend.

        Parameters
        ----------
        output_nodes : iterable of int
            A collection of output node indices to be processed.

        Returns
        -------
        None

        Notes
        -----
        This method currently does not perform any actions.
        """


def gen_str() -> str:
    """
    Generate a dummy string for einsum.

    Returns
    -------
    str
        A dummy string representation suitable for einsum operations.
    """
    return qtn.rand_uuid()


def outer_product(vectors: Sequence[npt.NDArray[np.complex128]]) -> npt.NDArray[np.complex128]:
    """
    Return the outer product of the given vectors.

    Parameters
    ----------
    vectors : Sequence[npt.NDArray[np.complex128]]
        A sequence of vectors for which the outer product is to be computed.

    Returns
    -------
    npt.NDArray[np.complex128]
        The resulting outer product as a tensor object.
    """
    subscripts = string.ascii_letters[: len(vectors)]
    subscripts = ",".join(subscripts) + "->" + subscripts
    return np.array(np.einsum(subscripts, *vectors), dtype=np.complex128)
