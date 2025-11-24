"""
Abstract base class for simulation backends.

This class serves as the foundation for all simulation backend implementations.
Derived classes should provide concrete implementations of the required methods
to facilitate specific simulation behavior.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, SupportsFloat, TypeVar

import numpy as np
import numpy.typing as npt

# TypeAlias introduced in Python 3.10
# override introduced in Python 3.12
from typing_extensions import TypeAlias, override

from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.ops import Ops
from graphix.parameter import check_expression_or_complex
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.random import Generator

    from graphix import command
    from graphix.fundamentals import Plane
    from graphix.measurements import Measurement, Outcome
    from graphix.noise_models.noise_model import Noise
    from graphix.parameter import ExpressionOrComplex, ExpressionOrFloat
    from graphix.sim.data import Data
    from graphix.simulator import MeasureMethod


if sys.version_info >= (3, 10):
    Matrix: TypeAlias = npt.NDArray[np.object_ | np.complex128]
else:
    from typing import Union

    Matrix: TypeAlias = npt.NDArray[Union[np.object_, np.complex128]]


def tensordot(op: Matrix, psi: Matrix, axes: tuple[int | Sequence[int], int | Sequence[int]]) -> Matrix:
    """
    Tensor dot product that preserves the type of `psi`.

    This wrapper around `np.tensordot` ensures static type checking
    for both numeric (`complex128`) and symbolic (`object`) arrays.
    Even though the runtime behavior is the same, NumPy's static types don't
    support `Matrix` directly.

    If `psi` and `op` are numeric, the result is numeric. If either `psi`
    or `op` is symbolic, the other is converted to symbolic if needed,
    and the result is also symbolic.

    Parameters
    ----------
    op : Matrix
        Operator tensor, either symbolic or numeric.
    psi : Matrix
        State tensor, either symbolic or numeric.
    axes : tuple[int | Sequence[int], int | Sequence[int]]
        Axes along which to contract `op` and `psi`.

    Returns
    -------
    Matrix
        The result of the tensor contraction with the same type as `psi`.
    """
    if psi.dtype == np.complex128 and op.dtype == np.complex128:
        psi_c = psi.astype(np.complex128, copy=False)
        op_c = op.astype(np.complex128, copy=False)
        return np.tensordot(op_c, psi_c, axes).astype(np.complex128)
    psi_o = psi.astype(np.object_, copy=False)
    op_o = op.astype(np.object_, copy=False)
    return np.tensordot(op_o, psi_o, axes)


def kron(a: Matrix, b: Matrix) -> Matrix:
    """
    Compute the Kronecker product of two matrices with type-safe handling of symbolic and numeric matrices.

    The two matrices must have the same type (both symbolic or both numeric).

    Parameters
    ----------
    a : Matrix
        The left operand (symbolic or numeric).
    b : Matrix
        The right operand (symbolic or numeric).

    Returns
    -------
    Matrix
        The Kronecker product of `a` and `b`.

    Raises
    ------
    TypeError
        If `a` and `b` do not have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return np.kron(a_c, b_c).astype(np.complex128)

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return np.kron(a_o, b_o)

    raise TypeError("Operands should have the same type.")


def outer(a: Matrix, b: Matrix) -> Matrix:
    """
    Outer product with type-safe handling of symbolic and numeric vectors.

    Computes the outer product of two matrices, ensuring that both matrices
    are of the same type (either symbolic or numeric).

    Parameters
    ----------
    a : Matrix
        The left operand (symbolic or numeric) for the outer product.
    b : Matrix
        The right operand (symbolic or numeric) for the outer product.

    Returns
    -------
    Matrix
        The outer product of `a` and `b`.

    Raises
    ------
    TypeError
        If `a` and `b` do not have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return np.outer(a_c, b_c).astype(np.complex128)

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return np.outer(a_o, b_o)

    raise TypeError("Operands should have the same type.")


def vdot(a: Matrix, b: Matrix) -> ExpressionOrComplex:
    """
    Conjugate dot product ⟨a|b⟩ with type-safe handling of symbolic and numeric vectors.

    The two matrices must have the same type to compute the dot product.

    Parameters
    ----------
    a : Matrix
        The left operand (either symbolic or numeric).
    b : Matrix
        The right operand (either symbolic or numeric).

    Returns
    -------
    ExpressionOrComplex
        The conjugate dot product of the two matrices.

    Raises
    ------
    TypeError
        If `a` and `b` are not of the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return complex(np.vdot(a_c, b_c))

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return check_expression_or_complex(np.vdot(a_o, b_o))

    raise TypeError("Operands should have the same type.")


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """
    Compute the matrix product of two matrices `a` and `b` with type-safe handling of symbolic and numeric vectors.

    The two matrices must have the same type for the operation to proceed.

    Parameters
    ----------
    a : Matrix
        The left operand, which can be either symbolic or numeric.
    b : Matrix
        The right operand, which can also be either symbolic or numeric.

    Returns
    -------
    Matrix
        The resulting matrix product of `a` and `b`.

    Raises
    ------
    TypeError
        If `a` and `b` are not of the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return a_c @ b_c

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return a_o @ b_o  # type: ignore[no-any-return]

    raise TypeError("Operands should have the same type.")


class NodeIndex:
    """
    A class for managing the mapping between node numbers and qubit indices in the internal state of the backend.

    This allows for efficient access and manipulation of qubit orderings throughout the execution of a pattern.

    Attributes
    ----------
    __list : list
        A private list of the current active nodes (labeled with integers).
    __dict : dict
        A private dictionary mapping current node labels (integers) to their corresponding qubit indices
        in the backend's internal quantum state.
    """

    __dict: dict[int, int]
    __list: list[int]

    def __init__(self) -> None:
        """
        Initialize an empty mapping between nodes and qubit indices.

        This constructor sets up a new instance of the NodeIndex class, initializing
        an empty data structure to store the mapping of nodes to their corresponding
        qubit indices.
        """
        self.__dict = {}
        self.__list = []

    def __getitem__(self, index: int) -> int:
        """
        Return the qubit node associated with the specified index.

        Parameters
        ----------
        index : int
            Position in the internal list.

        Returns
        -------
        int
            Node label corresponding to the specified index.
        """
        return self.__list[index]

    def index(self, node: int) -> int:
        """
        Return the qubit index associated with the specified node label.

        Parameters
        ----------
        node : int
            Node label to look up.

        Returns
        -------
        int
            Position of the specified ``node`` in the internal ordering.
        """
        return self.__dict[node]

    def __iter__(self) -> Iterator[int]:
        """
        Return an iterator over the node labels in their current order.

        Yields
        ------
        int
            The node labels in their current order.
        """
        return iter(self.__list)

    def __len__(self) -> int:
        """
        Return the number of currently active nodes.

        Returns
        -------
        int
            The count of active nodes in the index.
        """
        return len(self.__list)

    def extend(self, nodes: Iterable[int]) -> None:
        """
        Extend the mapping with additional nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            Node labels to append.

        Returns
        -------
        None
        """
        base = len(self)
        self.__list.extend(nodes)
        # The following loop iterates over `self.__list[base:]` instead of `nodes`
        # because the iterable `nodes` can be transient and consumed by the
        # `self.__list.extend` on the line just above.
        for index, node in enumerate(self.__list[base:]):
            self.__dict[node] = base + index

    def remove(self, node: int) -> None:
        """
        Remove a node and reassign indices of the remaining nodes.

        Parameters
        ----------
        node : int
            The label of the node to remove.
        """
        index = self.__dict[node]
        del self.__list[index]
        del self.__dict[node]
        for new_index, u in enumerate(self.__list[index:], start=index):
            self.__dict[u] = new_index

    def swap(self, i: int, j: int) -> None:
        """
        Swap two nodes given their indices.

        Parameters
        ----------
        i : int
            Index of the first node in the current ordering.
        j : int
            Index of the second node in the current ordering.

        Returns
        -------
        None
        """
        node_i = self.__list[i]
        node_j = self.__list[j]
        self.__list[i] = node_j
        self.__list[j] = node_i
        self.__dict[node_i] = j
        self.__dict[node_j] = i


class NoiseNotSupportedError(Exception):
    """
    Exception raised when `apply_channel` is called on a backend that does not support noise.

    Attributes
    ----------
    message : str
        Explanation of the error.

    Parameters
    ----------
    message : str, optional
        Custom error message (default is "Noise not supported by this backend.")

    Examples
    --------
    >>> raise NoiseNotSupportedError("This backend cannot apply noise.")
    """

    def __str__(self) -> str:
        """
        Return a string representation of the error message.

        Returns
        -------
        str
            The error message indicating that noise is not supported.
        """
        return "This backend does not support noise."


class BackendState(ABC):
    """
    Abstract base class for representing the quantum state of a backend.

    `BackendState` defines the interface for quantum state representations used by
    various backend implementations. It provides a common foundation for different
    simulation strategies, such as dense linear algebra or tensor network contraction.

    Concrete subclasses must implement the storage and manipulation logic appropriate
    for a specific backend and representation strategy.

    Notes
    -----
    This class is abstract and cannot be instantiated directly.

    Examples of concrete subclasses include:
    - :class:`Statevec`: for pure states represented as state vectors.
    - :class:`DensityMatrix`: for mixed states represented as density matrices.
    - :class:`MBQCTensorNet`: for compressed representations using tensor networks.

    See Also
    --------
    :class:`DenseState`, :class:`MBQCTensorNet`, :class:`Statevec`, :class:`DensityMatrix`
    """

    @abstractmethod
    def flatten(self) -> Matrix:
        """
        Return the flattened representation of the state.

        Returns
        -------
        Matrix
            A matrix representing the flattened state.
        """


class DenseState(BackendState):
    """
    Abstract base class for quantum states with full dense representations.

    `DenseState` defines the shared interface and behavior for state representations
    that explicitly store the entire quantum state in memory as a dense array.
    This includes both state vectors (for pure states) and density matrices (for
    mixed states).

    This class serves as a common parent for :class:`Statevec` and :class:`DensityMatrix`, which
    implement the concrete representations of dense quantum states. It is used in
    simulation backends that operate using standard linear algebra on the full
    state, such as :class:`StatevecBackend` and :class:`DensityMatrixBackend`.

    Notes
    -----
    This class is abstract and cannot be instantiated directly.

    Not all :class:`BackendState` subclasses are dense. For example, :class:`MBQCTensorNet` is a
    `BackendState` that represents the quantum state using a tensor network, rather than
    a single dense array.

    See Also
    --------
    :class:`Statevec`, :class:`DensityMatrix`
    """

    # Note that `@property` must appear before `@abstractmethod` for pyright
    @property
    @abstractmethod
    def nqubit(self) -> int:
        """
        Get the number of qubits in the quantum state.

        Returns
        -------
        int
            The number of qubits.
        """

    @abstractmethod
    def add_nodes(self, nqubit: int, data: Data) -> None:
        """
        Add nodes (qubits) to the state and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state.

        data : Data
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation.

        See Also
        --------
        Backend.add_nodes : For further details on state initialization and supported formats.
        """

    @abstractmethod
    def entangle(self, edge: tuple[int, int]) -> None:
        """
        Entangle graph nodes.

        Parameters
        ----------
        edge : tuple of int
            A tuple containing the indices of the control and target qubits, respectively.
        """

    @abstractmethod
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """
        Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            A square matrix of shape (2^n, 2^n) representing the operation to be applied.
        qargs : Sequence[int]
            A sequence of integers representing the indices of the target qubits.
        """

    @abstractmethod
    def evolve_single(self, op: Matrix, i: int) -> None:
        """
        Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            A 2x2 matrix representing the single-qubit operation.
        i : int
            The index of the qubit on which the operation is to be applied.
        """

    @abstractmethod
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """
        Return the expectation value of a single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            A 2x2 operator representing the single-qubit operation.
        loc : int
            The index of the target qubit.

        Returns
        -------
        complex
            The expectation value of the operator for the specified qubit.
        """

    @abstractmethod
    def remove_qubit(self, qarg: int) -> None:
        """
        Remove a separable qubit from the quantum state.

        Parameters
        ----------
        qarg : int
            The index of the qubit to be removed from the system. The index should
            correspond to a qubit in the current state representation.

        Raises
        ------
        IndexError
            If the provided index is out of bounds or does not correspond to an
            existing qubit in the system.

        Notes
        -----
        This method modifies the current state of the system by removing the
        specified qubit, resulting in a new quantum state that is one qubit
        smaller.
        """

    @abstractmethod
    def swap(self, qubits: tuple[int, int]) -> None:
        """
        Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            A tuple containing the indices of the qubits to be swapped. The first element
            is the index of the control qubit and the second element is the index of the
            target qubit.
        """

    def apply_noise(self, qubits: Sequence[int], noise: Noise) -> None:  # noqa: ARG002,PLR6301
        """
        Apply noise to the specified qubits.

        This method applies a noise operation to the given qubits.
        The default implementation raises a `NoiseNotSupportedError`,
        indicating that the backend does not support noise. Backends
        that support noise (e.g., `DensityMatrixBackend`) should
        override this method to implement the desired noise effects.

        Parameters
        ----------
        qubits : Sequence[int]
            List of target qubit indices to which the noise will be applied.
        noise : Noise
            The noise model that defines the type and characteristics of noise to apply.

        Notes
        -----
        Ensure that the backend and noise model are compatible. Proper
        error handling or checks should be implemented in derived classes
        that implement noise functionalities.
        """
        raise NoiseNotSupportedError


def _op_mat_from_result(
    vec: tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat], result: Outcome, symbolic: bool = False
) -> Matrix:
    """
    Return the operator :math:`\tfrac{1}{2}(I + (-1)^r \vec{v}\cdot\vec{\sigma})`.

    Parameters
    ----------
    vec : tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat]
        Cartesian components of a unit vector.
    result : Outcome
        Measurement result ``r``.
    symbolic : bool, optional
        If ``True``, return an array of ``object`` dtype. Default is ``False``.

    Returns
    -------
    Matrix
        2x2 operator acting on the measured qubit.
    """
    sign = (-1) ** result
    if symbolic:
        op_mat_symbolic: npt.NDArray[np.object_] = np.eye(2, dtype=np.object_) / 2
        for i, t in enumerate(vec):
            op_mat_symbolic += sign * t * Clifford(i + 1).matrix / 2
        return op_mat_symbolic
    op_mat_complex: npt.NDArray[np.complex128] = np.eye(2, dtype=np.complex128) / 2
    x, y, z = vec
    # mypy requires each of x, y, and z to be tested explicitly for it to infer
    # that they are instances of `SupportsFloat`.
    # In particular, using a loop or comprehension like
    # `not all(isinstance(v, SupportsFloat) for v in (x, y, z))` is not supported.
    if not isinstance(x, SupportsFloat) or not isinstance(y, SupportsFloat) or not isinstance(z, SupportsFloat):
        raise TypeError("Vector of float expected with symbolic = False")
    float_vec = [x, y, z]
    for i, t in enumerate(float_vec):
        op_mat_complex += sign * t * Clifford(i + 1).matrix / 2
    return op_mat_complex


def perform_measure(
    qubit_node: int,
    qubit_loc: int,
    plane: Plane,
    angle: ExpressionOrFloat,
    state: DenseState,
    branch_selector: BranchSelector,
    rng: Generator | None = None,
    symbolic: bool = False,
) -> Outcome:
    """
    Perform measurement of a qubit.

    Parameters
    ----------
    qubit_node : int
        The index of the qubit node to be measured.
    qubit_loc : int
        The location of the qubit within the specified node.
    plane : Plane
        The plane in which the measurement is to be performed.
    angle : ExpressionOrFloat
        The angle at which the measurement is made, can be a symbolic expression or a float.
    state : DenseState
        The current state of the qubit to be measured.
    branch_selector : BranchSelector
        Selector for choosing branches in the measurement process.
    rng : Generator, optional
        Random number generator for stochastic processes (default is None).
    symbolic : bool, optional
        Flag to indicate if symbolic computation should be used (default is False).

    Returns
    -------
    Outcome
        The outcome of the measurement, which can include the result and any relevant metadata.

    Notes
    -----
    This function performs the measurement of a qubit within the quantum system, taking into account various parameters and configurations as specified in the arguments.
    """
    vec = plane.polar(angle)
    # op_mat0 may contain the matrix operator associated with the outcome 0,
    # but the value is computed lazily, i.e., only if needed.
    op_mat0 = None

    def get_op_mat0() -> Matrix:
        nonlocal op_mat0
        if op_mat0 is None:
            op_mat0 = _op_mat_from_result(vec, 0, symbolic=symbolic)
        return op_mat0

    def f_expectation0() -> float:
        exp_val = state.expectation_single(get_op_mat0(), qubit_loc)
        assert math.isclose(exp_val.imag, 0, abs_tol=1e-10)
        return exp_val.real

    result = branch_selector.measure(qubit_node, f_expectation0, rng)
    op_mat = _op_mat_from_result(vec, 1, symbolic=symbolic) if result else get_op_mat0()
    state.evolve_single(op_mat, qubit_loc)
    return result


_StateT_co = TypeVar("_StateT_co", bound="BackendState", covariant=True)


@dataclass(frozen=True)
class Backend(Generic[_StateT_co]):
    """
    Abstract base class for all quantum backends.

    A backend is responsible for managing a quantum system, including the set of active
    qubits (nodes), their initialization, evolution, and measurement. It defines the
    interface through which high-level quantum programs interact with the underlying
    simulation or hardware model.

    Concrete subclasses implement specific state representations and simulation strategies,
    such as dense state vectors, density matrices, or tensor networks.

    Responsibilities of a backend typically include:
    - Managing a dynamic set of qubits (nodes) and their state.
    - Applying quantum gates or operations.
    - Performing measurements and returning classical outcomes.
    - Tracking and exposing the underlying quantum state.

    Examples of concrete subclasses include:
    - `StatevecBackend` (pure states via state vectors).
    - `DensityMatrixBackend` (mixed states via density matrices).
    - `TensorNetworkBackend` (compressed states via tensor networks).

    Parameters
    ----------
    state : BackendState
        Internal state of the backend: instance of :class:`Statevec`, :class:`DensityMatrix`, or :class:`MBQCTensorNet`.

    Notes
    -----
    This class is abstract and should not be instantiated directly.

    The class hierarchy of states mirrors the class hierarchy of backends:
    - `DenseStateBackend` and `TensorNetworkBackend` are subclasses of `Backend`,
      and `DenseState` and `MBQCTensorNet` are subclasses of `BackendState`.
    - `StatevecBackend` and `DensityMatrixBackend` are subclasses of `DenseStateBackend`,
      and `Statevec` and `DensityMatrix` are subclasses of `DenseState`.

    The type variable `_StateT_co` specifies the type of the `state` field, so that subclasses
    provide a precise type for this field:
    - `StatevecBackend` is a subtype of ``Backend[Statevec]``.
    - `DensityMatrixBackend` is a subtype of ``Backend[DensityMatrix]``.
    - `TensorNetworkBackend` is a subtype of ``Backend[MBQCTensorNet]``.

    The type variables `_StateT_co` and `_DenseStateT_co` are declared as covariant.
    That is, ``Backend[T1]`` is a subtype of ``Backend[T2]`` if ``T1`` is a subtype of ``T2``.
    This means that `StatevecBackend`, `DensityMatrixBackend`, and `TensorNetworkBackend` are
    all subtypes of ``Backend[BackendState]``. This covariance is sound because backends are frozen
    dataclasses; thus, the type of `state` cannot be changed after instantiation.

    The interface expected from a backend includes the following methods:
    - `add_nodes`: executes `N` commands.
    - `apply_channel`: used for noisy simulations. The class `Backend` provides a default implementation that
      raises `NoiseNotSupportedError`, indicating that the backend does not support noise. Backends that support
      noise (e.g., `DensityMatrixBackend`) override this method to implement the effect of noise.
    - `apply_clifford`: executes `C` commands.
    - `correct_byproduct`: executes `X` and `Z` commands.
    - `entangle_nodes`: executes `E` commands.
    - `finalize`: called at the end of pattern simulation to convey the order of output nodes.
    - `measure`: executes `M` commands.

    See Also
    --------
    :class:`BackendState`, :class:`DenseStateBackend`, :class:`StatevecBackend`, :class:`DensityMatrixBackend`, :class:`TensorNetworkBackend`
    """

    # `init=False` is required because `state` cannot appear in a contravariant position
    # (specifically, as a parameter of `__init__`) since `_StateT_co` is covariant.
    state: _StateT_co = dataclasses.field(init=False)

    @abstractmethod
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        r"""
        Add new nodes (qubits) to the backend and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be fresh: each index must be distinct from all
            previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation.

            All backends must support the basic predefined states in ``BasicStates``.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.

            Some backends support other forms of state specification.

            - ``StatevecBackend`` supports arbitrary state vectors:
                - A single-qubit state vector will be broadcast to all nodes.
                - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`,
                  initializes the new nodes jointly.

            - ``DensityMatrixBackend`` supports both state vectors and density matrices:
                - State vectors are handled as in ``StatevecBackend``, and converted to
                  density matrices.
                - A density matrix must have shape :math:`2^n \times 2^n`, where :math:`n = \mathrm{len}(nodes)`,
                  and is used to jointly initialize the new nodes.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """

    def apply_noise(self, nodes: Sequence[int], noise: Noise) -> None:  # noqa: ARG002,PLR6301
        """
        Apply noise to the specified nodes.

        This method is intended to be overridden by backends that support noise. The default
        implementation raises a `NoiseNotSupportedError`, indicating that the backend does not
        support noise. Backends such as `DensityMatrixBackend` should implement the effect of
        noise in this method.

        Parameters
        ----------
        nodes : Sequence[int]
            Target qubits to which the noise is to be applied.
        noise : Noise
            The noise model to apply to the specified nodes.

        Raises
        ------
        NoiseNotSupportedError
            If the backend does not support noise.
        """
        raise NoiseNotSupportedError

    @abstractmethod
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """
        Apply a single-qubit Clifford gate to a specified node.

        This method applies a Clifford gate, specified by its corresponding
        index in `graphix.clifford.CLIFFORD`, to the given node in the quantum circuit.

        Parameters
        ----------
        node : int
            The index of the node (qubit) to which the Clifford gate will be applied.

        clifford : Clifford
            The Clifford gate to be applied, represented as an instance of the `Clifford` class.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a derived class.
        """

    @abstractmethod
    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """
        Corrects byproduct errors for the specified X or Z byproduct operators.

        Parameters
        ----------
        cmd : command.X | command.Z
            The byproduct operator to correct, which can be either
            an X or Z gate.
        measure_method : MeasureMethod
            The measurement method used for the correction process.
            This determines how the correction is applied based on
            the measurement results.

        Returns
        -------
        None
            This method modifies the state of the system in-place
            and does not return a value.
        """

    @abstractmethod
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """
        Apply the CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple of int
            A pair of node indices (i, j) that represent the connected nodes to be entangled.
        """

    @abstractmethod
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """
        Finalize the processing of the output nodes after pattern simulation.

        Parameters
        ----------
        output_nodes : Iterable[int]
            A collection of integers representing the indices of the output nodes
            to be processed at the end of the simulation.

        Notes
        -----
        This method is intended to be called after the completion of the pattern
        simulation to ensure that the specified output nodes are processed in the
        correct order.
        """

    @abstractmethod
    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Outcome:
        """
        Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node : int
            The index of the node to measure.

        measurement : Measurement
            The measurement to be applied to the specified node.

        rng : Generator, optional
            Random-number generator for measurements.
            This generator is used only in cases of random branch selection
            (see :class:`RandomBranchSelector`).

        Returns
        -------
        Outcome
            The outcome of the measurement.
        """


_DenseStateT_co = TypeVar("_DenseStateT_co", bound="DenseState", covariant=True)


@dataclass(frozen=True)
class DenseStateBackend(Backend[_DenseStateT_co], Generic[_DenseStateT_co]):
    """
    Abstract base class for backends that represent quantum states explicitly in memory.

    This class defines common functionality for backends that store the entire quantum
    state as a dense array—either as a state vector (pure state) or a density matrix
    (mixed state)—and perform quantum operations using standard linear algebra. It is
    designed to be the shared base class of `StatevecBackend` and `DensityMatrixBackend`.

    In contrast to :class:`TensorNetworkBackend`, which uses structured and compressed
    representations (e.g., matrix product states) to scale to larger systems,
    `DenseStateBackend` subclasses simulate quantum systems by maintaining the full
    state in memory. This approach enables straightforward implementation of gates,
    measurements, and noise models, but scales exponentially with the number of qubits.

    This class is not meant to be instantiated directly.

    Parameters
    ----------
    node_index : NodeIndex, optional
        Mapping between node numbers and qubit indices in the internal state of the backend.
    branch_selector : :class:`graphix.branch_selector.BranchSelector`, optional
        Branch selector used for measurements. Default is :class:`RandomBranchSelector`.
    symbolic : bool, optional
        If True, support arbitrary objects (typically, symbolic expressions) in matrices.

    See Also
    --------
    :class:`StatevecBackend`, :class:`DensityMatrixBackend`, :class:`TensorNetworkBackend`
    """

    node_index: NodeIndex = dataclasses.field(default_factory=NodeIndex)
    branch_selector: BranchSelector = dataclasses.field(default_factory=RandomBranchSelector)
    symbolic: bool = False

    @override
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        """
        Add new nodes (qubits) to the backend and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be unique: each index must be distinct from all
            previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation. The default
            is ``BasicStates.PLUS``.

        Notes
        -----
        See :meth:`Backend.add_nodes` for further details.
        """
        self.state.add_nodes(nqubit=len(nodes), data=data)
        self.node_index.extend(nodes)

    @override
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """
        Apply a controlled-Z (CZ) gate to two connected nodes.

        Parameters
        ----------
        edge : tuple of int
            A pair of node indices (i, j) representing the connected nodes to be entangled.
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    @override
    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Outcome:
        """
        Perform measurement of a node and trace out the corresponding qubit.

        Parameters
        ----------
        node : int
            The index of the node to be measured.
        measurement : Measurement
            The measurement operator to apply to the node.
        rng : Generator, optional
            An optional random number generator for stochastic measurements.

        Returns
        -------
        Outcome
            The result of the measurement on the specified node.
        """
        loc = self.node_index.index(node)
        result = perform_measure(
            node,
            loc,
            measurement.plane,
            measurement.angle,
            self.state,
            self.branch_selector,
            rng=rng,
            symbolic=self.symbolic,
        )
        self.node_index.remove(node)
        self.state.remove_qubit(loc)
        return result

    @override
    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """
        Corrects for the X or Z byproduct operators by applying the corresponding X or Z gate.

        Parameters
        ----------
        cmd : command.X | command.Z
            The command representing the byproduct operator to be corrected.
        measure_method : MeasureMethod
            The measurement method used during the correction process.

        Returns
        -------
        None
            This method modifies the state of the backend in place and does not return a value.
        """
        if np.mod(sum(measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
            op = Ops.X if cmd.kind == CommandKind.X else Ops.Z
            self.apply_single(node=cmd.node, op=op)

    @override
    def apply_noise(self, nodes: Sequence[int], noise: Noise) -> None:
        """
        Apply noise to the specified nodes.

        Parameters
        ----------
        nodes : sequence of int
            The target qubits to which noise will be applied.
        noise : Noise
            The noise object that defines the noise to be applied.
        """
        indices = [self.node_index.index(i) for i in nodes]
        self.state.apply_noise(indices, noise)

    def apply_single(self, node: int, op: Matrix) -> None:
        """
        Apply a single gate operation to the state.

        Parameters
        ----------
        node : int
            The index of the node to which the operation will be applied.
        op : Matrix
            The matrix representing the gate operation to be applied.

        Returns
        -------
        None
            This method modifies the state in place and does not return a value.
        """
        index = self.node_index.index(node)
        self.state.evolve_single(op=op, i=index)

    @override
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """
        Apply a single-qubit Clifford gate to the specified qubit.

        Parameters
        ----------
        node : int
            The index of the qubit to which the Clifford gate will be applied.
        clifford : Clifford
            The Clifford gate to be applied, as specified by the @vop index in
            graphix.clifford.CLIFFORD.

        Returns
        -------
        None
            This method modifies the state of the qubit in place and does not return a value.

        Notes
        -----
        This method is intended to be used within the context of a quantum state backend that
        supports the application of single-qubit Clifford gates.
        """
        loc = self.node_index.index(node)
        self.state.evolve_single(clifford.matrix, loc)

    def sort_qubits(self, output_nodes: Iterable[int]) -> None:
        """
        Sort the qubit order in the internal statevector.

        Parameters
        ----------
        output_nodes : Iterable[int]
            The order of qubits to sort the internal statevector.

        Returns
        -------
        None
            This method modifies the internal statevector in place and does not return a value.
        """
        for i, ind in enumerate(output_nodes):
            if self.node_index.index(ind) != i:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index.swap(i, move_from)

    @override
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """
        Finalize the pattern simulation.

        This method is called at the end of the pattern simulation to perform any
        necessary cleanup or final processing related to the output nodes.

        Parameters
        ----------
        output_nodes : Iterable[int]
            A collection of indices representing the output nodes that were
            involved in the simulation.

        Returns
        -------
        None
        """
        self.sort_qubits(output_nodes)

    @property
    def nqubit(self) -> int:
        """
        Returns the number of qubits in the current state.

        Returns
        -------
        int
            The number of qubits of the current state.
        """
        return self.state.nqubit
