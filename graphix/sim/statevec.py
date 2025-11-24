"""
MBQC State Vector Backend
==========================

This module provides functionality for simulating measurement-based quantum computation (MBQC) using a state vector representation.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from graphix import parameter, states
from graphix.parameter import Expression, ExpressionOrSupportsComplex, check_expression_or_float
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix, kron, tensordot
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix.parameter import ExpressionOrFloat, ExpressionOrSupportsFloat, Parameter
    from graphix.sim.data import Data


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


class Statevec(DenseState):
    """
    Statevector object.

    The Statevec class represents a quantum state in the form of a vector.
    It provides methods to manipulate and perform calculations on the
    statevector, enabling quantum mechanical simulations.

    Attributes
    ----------
    vector : numpy.ndarray
        A complex-valued array representing the quantum state.

    Methods
    -------
    normalize():
        Normalizes the statevector to ensure it is a valid quantum state.

    tensor_product(other):
        Computes the tensor product of the current statevector with another.

    measure(basis):
        Measures the statevector in the specified basis and returns the outcome.

    __str__():
        Returns a string representation of the statevector.

    __repr__():
        Returns a detailed string representation of the statevector for debugging.
    """

    psi: Matrix

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ) -> None:
        """
        Initialize statevector objects.

        The `data` parameter can be one of the following:
        - A single :class:`graphix.states.State` (classical description of a quantum state).
        - An iterable of :class:`graphix.states.State` objects.
        - An iterable of scalars (a 2**n numerical statevector).
        - A *graphix.statevec.Statevec* object.

        If *nqubit* is not provided, the number of qubits is inferred from *data* and checked for consistency.
        If only one :class:`graphix.states.State` is provided and *nqubit* is a valid integer, the statevector is initialized
        in the tensor product state. If both *nqubit* and *data* are provided, consistency of the dimensions is checked.
        If a *graphix.statevec.Statevec* is passed, a copy of it is returned.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. Can be a classical description or a numerical input. Defaults to
            :class:`graphix.states.BasicStates.PLUS`.
        nqubit : int, optional
            Number of qubits to prepare. Defaults to None.
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        if isinstance(data, Statevec):
            # assert nqubit is None or len(state.flatten()) == 2**nqubit
            if nqubit is not None and len(data.flatten()) != 2**nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
                )
            self.psi = data.psi.copy()
            return

        # The type
        # list[states.State] | list[ExpressionOrSupportsComplex] | list[Iterable[ExpressionOrSupportsComplex]]
        # would be more precise, but given a value X of type Iterable[A] | Iterable[B],
        # mypy infers that list(X) has type list[A | B] instead of list[A] | list[B].
        input_list: list[states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex]]
        if isinstance(data, states.State):
            if nqubit is None:
                nqubit = 1
            input_list = [data] * nqubit
        elif isinstance(data, Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")

            self.psi = np.array(1, dtype=np.complex128)

        elif isinstance(input_list[0], states.State):
            if nqubit is None:
                nqubit = len(input_list)
            elif nqubit != len(input_list):
                raise ValueError("Mismatch between nqubit and length of input state.")

            def get_statevector(
                s: states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex],
            ) -> npt.NDArray[np.complex128]:
                if not isinstance(s, states.State):
                    raise TypeError("Data should be an homogeneous sequence of states.")
                return s.get_statevector()

            list_of_sv = [get_statevector(s) for s in input_list]

            tmp_psi = functools.reduce(lambda m0, m1: np.kron(m0, m1).astype(np.complex128), list_of_sv)
            # reshape
            self.psi = tmp_psi.reshape((2,) * nqubit)
        # `SupportsFloat` is needed because `numpy.float64` is not an instance of `SupportsComplex`!
        elif isinstance(input_list[0], (Expression, SupportsComplex, SupportsFloat)):
            if nqubit is None:
                length = len(input_list)
                if length & (length - 1):
                    raise ValueError("Length is not a power of two")
                nqubit = length.bit_length() - 1
            elif nqubit != len(input_list).bit_length() - 1:
                raise ValueError("Mismatch between nqubit and length of input state")
            psi = np.array(input_list)
            # check only if the matrix is not symbolic
            if psi.dtype != "O" and not np.allclose(np.sqrt(np.sum(np.abs(psi) ** 2)), 1):
                raise ValueError("Input state is not normalized")
            self.psi = psi.reshape((2,) * nqubit)
        else:
            raise TypeError(f"First element of data has type {type(input_list[0])} whereas Number or State is expected")

    def __str__(self) -> str:
        """
        Return a string representation of the Statevec instance.

        Returns
        -------
        str
            A description of the Statevec instance.
        """
        return f"Statevec object with statevector {self.psi} and length {self.dims()}."

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        """
        Add nodes (qubits) to the state vector and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data, optional
            The state in which to initialize the newly added nodes. It can take the following forms:

            - A single basic state, in which case all new nodes are initialized to that state.
            - A list of basic states, which must match the length of `nodes`, where each node is initialized
              with its corresponding state.
            - A single-qubit state vector, which will be broadcast to all new nodes.
            - A multi-qubit state vector with dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`,
              which initializes the new nodes jointly.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        sv_to_add = Statevec(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """
        Apply a single-qubit operation to the specified qubit index.

        Parameters
        ----------
        op : numpy.ndarray
            A 2x2 matrix representing the single-qubit operation to be applied.
        i : int
            The index of the qubit on which the operation will be performed.
        """
        psi = tensordot(op, self.psi, (1, i))
        self.psi = np.moveaxis(psi, 0, i)

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """
        Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            A 2^n x 2^n matrix representing the operation to be applied.
        qargs : Sequence[int]
            A sequence of integers representing the indices of the target qubits.
        """
        op_dim = int(np.log2(len(op)))
        # TODO shape = (2,)* 2 * op_dim
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        psi = tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), qargs),
        )
        self.psi = np.moveaxis(psi, range(len(qargs)), qargs)

    def dims(self) -> tuple[int, ...]:
        """
        Returns the dimensions of the state vector.

        Returns
        -------
        tuple[int, ...]
            A tuple representing the dimensions of the state vector.
        """
        return self.psi.shape

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """
        Get the number of qubits in the quantum state.

        Returns
        -------
        int
            The number of qubits represented by the state vector.
        """
        return self.psi.size - 1

    @override
    def remove_qubit(self, qarg: int) -> None:
        """
        Remove a separable qubit from the system and assemble a statevector for the remaining qubits.

        This method produces a result equivalent to performing a partial trace if the specified qubit
        (*qarg*) is separable from the rest of the qubits.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with the sum taken over
        :math:`i \in [0 \dots 00,\ 0\dots 01,\ \dots,\ 1 \dots 11]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
            \end{align}

        (after normalization), where :math:`k` is equal to *qarg*. If the :math:`k` th qubit is in the
        state :math:`\ket{1}`, the above will yield zero amplitudes. In this case, the returned
        state will be the one above with :math:`0_{\mathrm{k}}` replaced by :math:`1_{\mathrm{k}}`.

        .. warning::
            This method assumes that the qubit with index *qarg* is separable from the other qubits
            and is designed to be a significantly faster alternative to the partial trace used after
            single-qubit measurements. Care should be taken when using this method. Checks for
            separability will be implemented as an option in the future.

        .. seealso::
            :meth:`graphix.sim.statevec.Statevec.ptrace` and the associated warnings.

        Parameters
        ----------
        qarg : int
            The index of the qubit to be removed.
        """
        norm = _get_statevec_norm(self.psi)
        if isinstance(norm, SupportsFloat):
            assert not np.isclose(norm, 0)
        index: list[slice[int] | int] = [slice(None)] * self.psi.ndim
        index[qarg] = 0
        psi = self.psi[tuple(index)]
        norm = _get_statevec_norm(psi)
        if isinstance(norm, SupportsFloat) and math.isclose(norm, 0):
            index[qarg] = 1
            psi = self.psi[tuple(index)]
        self.psi = psi
        self.normalize()

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """
        Connect graph nodes by creating an entangled state between the specified qubits.

        Parameters
        ----------
        edge : tuple of int
            A tuple containing two integers representing the indices of the control and target qubits, respectively. The first element of the tuple is the index of the control qubit, and the second element is the index of the target qubit.

        Returns
        -------
        None
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(CZ_TENSOR, self.psi, ((2, 3), edge))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), edge)

    def tensor(self, other: Statevec) -> None:
        """
        Compute the tensor product of the current state with another state.

        The result is stored in the current state as :math:`self \otimes other`.

        Parameters
        ----------
        other : Statevec
            The statevector to be tensored with the current state.
        """
        psi_self = self.psi.flatten()
        psi_other = other.psi.flatten()

        total_num = len(self.dims()) + len(other.dims())
        self.psi = kron(psi_self, psi_other).reshape((2,) * total_num)

    def cnot(self, qubits: tuple[int, int]) -> None:
        """
        Apply the CNOT (Controlled-NOT) gate to the state vector.

        Parameters
        ----------
        qubits : tuple of int
            A tuple containing the indices of the control and target qubits, respectively.
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """
        Swap the specified qubits.

        Parameters
        ----------
        qubits : tuple of int
            A tuple containing the indices of the qubits to be swapped.
            The first element is the index of the control qubit, and the
            second element is the index of the target qubit.
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    def normalize(self) -> None:
        """
        Normalize the state vector in-place.

        This method modifies the state vector of the instance to ensure that it has a unit norm.
        The normalization is performed by dividing the state vector by its norm.

        Returns
        -------
        None
        """
        # Note that the following calls to `astype` are guaranteed to
        # return the original NumPy array itself, since `copy=False` and
        # the `dtype` matches. This is important because the array is
        # then modified in place.
        if self.psi.dtype == np.object_:
            psi_o = self.psi.astype(np.object_, copy=False)
            norm_o = _get_statevec_norm_symbolic(psi_o)
            psi_o /= norm_o
        else:
            psi_c = self.psi.astype(np.complex128, copy=False)
            norm_c = _get_statevec_norm_numeric(psi_c)
            psi_c /= norm_c

    def flatten(self) -> Matrix:
        """
        Return the flattened state vector.

        This method transforms the current state vector into a one-dimensional
        representation, making it easier to work with in various applications
        such as computations or visualizations.

        Returns
        -------
        Matrix
            A one-dimensional array that represents the flattened state vector.

        Notes
        -----
        The flattening is done by reshaping the original state vector into a
        single row or column, depending on the internal representation of
        the state vector.
        """
        return self.psi.flatten()

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """
        Return the expectation value of a single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            A 2x2 operator representing the quantum operation on a single qubit.
        loc : int
            The index of the target qubit.

        Returns
        -------
        complex
            The expectation value of the operator for the specified qubit.
        """
        st1 = copy.copy(self)
        st1.normalize()
        st2 = copy.copy(st1)
        st1.evolve_single(op, loc)
        return complex(np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten()))

    def expectation_value(self, op: Matrix, qargs: Sequence[int]) -> complex:
        """
        Return the expectation value of a multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            A 2^n x 2^n operator representing the multi-qubit operator.
        qargs : Sequence[int]
            A sequence of integers representing the target qubit indices.

        Returns
        -------
        complex
            The expectation value of the operator on the specified qubits.
        """
        st2 = copy.copy(self)
        st2.normalize()
        st1 = copy.copy(st2)
        st1.evolve(op, qargs)
        return complex(np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten()))

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Statevec:
        """
        Substitute occurrences of a variable in measurement angles with a given value.

        This method returns a new instance of the state vector in which all occurrences
        of the specified variable are replaced by the provided substitute value in the
        measurement angles.

        Parameters
        ----------
        variable : Parameter
            The variable to substitute in the measurement angles.
        substitute : ExpressionOrSupportsFloat
            The value to substitute for the variable.

        Returns
        -------
        Statevec
            A new instance of the state vector with the substitutions applied.

        Notes
        -----
        The original state vector remains unchanged.
        """
        result = Statevec()
        result.psi = np.vectorize(lambda value: parameter.subs(value, variable, substitute))(self.psi)
        return result

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Statevec:
        """
        Return a copy of the state vector with substitutions applied to measurement angles.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A mapping of parameters (keys) to their corresponding values (expressions or floats)
            that will replace the occurrences in the measurement angles of the state vector.

        Returns
        -------
        Statevec
            A new state vector with all occurrences of the given keys in measurement angles
            substituted by the provided values, computed in parallel.
        """
        result = Statevec()
        result.psi = np.vectorize(lambda value: parameter.xreplace(value, assignment))(self.psi)
        return result


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """
    MBQC Simulator using the statevector method.

    This class implements a simulator for measurement-based quantum computation (MBQC)
    using a statevector approach. It provides functionalities for initializing quantum states,
    performing measurements, and simulating quantum operations in a measurement-based framework.

    Attributes
    ----------
    quantum_state : Statevector
        The current quantum state represented as a statevector.
    measurements : list
        A list to keep track of the measurement outcomes.

    Methods
    -------
    initialize_state(state: Statevector) -> None
        Initializes the quantum state with the provided statevector.

    apply_gate(gate: str, qubits: list) -> None
        Applies a quantum gate to specified qubits in the current state.

    measure(qubit: int) -> bool
        Performs a measurement on the specified qubit and updates the state accordingly.

    reset() -> None
        Resets the quantum state and measurement history.
    """

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))


def _get_statevec_norm_symbolic(psi: npt.NDArray[np.object_]) -> ExpressionOrFloat:
    """
    Calculate the norm of a given state vector.

    Parameters
    ----------
    psi : npt.NDArray[np.object_]
        A state vector represented as a NumPy array of objects.

    Returns
    -------
    ExpressionOrFloat
        The norm of the state vector, which can be either a symbolic expression or a float value.
    """
    flat = psi.flatten()
    return check_expression_or_float(np.sqrt(np.sum(flat.conj() * flat)))


def _get_statevec_norm_numeric(psi: npt.NDArray[np.complex128]) -> float:
    flat = psi.flatten()
    norm_sq = np.sum(flat.conj() * flat)
    assert math.isclose(norm_sq.imag, 0, abs_tol=1e-15)
    return math.sqrt(norm_sq.real)


def _get_statevec_norm(psi: Matrix) -> ExpressionOrFloat:
    """
    Calculate the norm of the state vector.

    Parameters
    ----------
    psi : Matrix
        The state vector for which the norm is to be calculated.

    Returns
    -------
    ExpressionOrFloat
        The norm of the state vector.
    """
    # Narrow psi to concrete dtype
    if psi.dtype == np.object_:
        return _get_statevec_norm_symbolic(psi.astype(np.object_, copy=False))
    return _get_statevec_norm_numeric(psi.astype(np.complex128, copy=False))
