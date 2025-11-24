"""
Density matrix simulator.

This module simulates measurement-based quantum computation (MBQC) using a density matrix representation. It provides functionalities for creating, manipulating, and simulating density matrices in the context of quantum computation.

Key Features
------------
- Simulation of density matrices for quantum states.
- Support for various quantum operations and measurements.
- Tools for the analysis of measurement-based quantum computation.

Usage
-----
To use this module, import it and create a density matrix for your quantum state. Then, apply quantum gates and perform measurements as needed.

Example
-------
```python
import density_matrix_simulator as dms

# Create a density matrix
dm = dms.create_density_matrix(state_vector)

# Apply a quantum operation
dm = dms.apply_gate(dm, gate)

# Perform a measurement
result = dms.measure(dm)
```
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex

import numpy as np
from typing_extensions import override

from graphix import linalg_validations as lv
from graphix import parameter
from graphix.channels import KrausChannel
from graphix.parameter import Expression, ExpressionOrFloat, ExpressionOrSupportsComplex
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix, kron, matmul, outer, tensordot, vdot
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix.noise_models.noise_model import Noise
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.sim.data import Data


class DensityMatrix(DenseState):
    """
    A class to represent a Density Matrix.

    Attributes
    ----------
    matrix : ndarray
        The density matrix represented as a 2D numpy array.

    Methods
    -------
    __init__(matrix: np.ndarray) -> None
        Initializes the DensityMatrix with a given matrix.

    to_density_operator() -> DensityOperator
        Converts the density matrix to a density operator.

    trace() -> float
        Computes and returns the trace of the density matrix.

    is_valid() -> bool
        Checks if the density matrix is valid (i.e., Hermitian and positive semi-definite).
    """

    rho: Matrix

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ) -> None:
        """
        Initialize density matrix objects.

        The behaviour builds on that of *graphix.statevec.Statevec*. The `data` parameter can be one of the following:
        - A single :class:`graphix.states.State` (classical description of a quantum state)
        - An iterable of :class:`graphix.states.State` objects
        - An iterable of iterables of scalars (a *2**n x 2**n* numerical density matrix)
        - A *graphix.statevec.DensityMatrix* object
        - A *graphix.statevec.Statevector* object

        If `nqubit` is not provided, the number of qubits is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and `nqubit` is a valid integer, the statevector is initialized in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a *graphix.statevec.Statevec* or *graphix.statevec.DensityMatrix* is passed, a copy is returned.

        Parameters
        ----------
        data : Data
            Input data to prepare the state. Can be a classical description or a numerical input. Defaults to graphix.states.BasicStates.PLUS.
        nqubit : int, optional
            Number of qubits to prepare. Defaults to *None*.

        Returns
        -------
        None
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        def check_size_consistency(mat: Matrix) -> None:
            if nqubit is not None and mat.shape != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {mat.shape}."
                )

        if isinstance(data, DensityMatrix):
            check_size_consistency(data.rho)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = data.rho.copy()
            return
        if isinstance(data, Iterable):
            input_list = list(data)
            if len(input_list) != 0 and isinstance(input_list[0], Iterable):

                def get_row(
                    item: Iterable[ExpressionOrSupportsComplex] | State | Expression | SupportsComplex,
                ) -> list[ExpressionOrSupportsComplex]:
                    if isinstance(item, Iterable):
                        return list(item)
                    raise TypeError("Every row of a matrix should be iterable.")

                input_matrix: list[list[ExpressionOrSupportsComplex]] = [get_row(item) for item in input_list]
                self.rho = np.array(input_matrix)
                if not lv.is_qubitop(self.rho):
                    raise ValueError("Cannot interpret the provided density matrix as a qubit operator.")
                check_size_consistency(self.rho)
                if self.rho.dtype != np.object_:
                    if not lv.is_unit_trace(self.rho):
                        raise ValueError("Density matrix must have unit trace.")
                    if not lv.is_psd(self.rho):
                        raise ValueError("Density matrix must be positive semi-definite.")
                return
        statevec = Statevec(data, nqubit)
        # NOTE this works since np.outer flattens the inputs!
        self.rho = outer(statevec.psi, statevec.psi.conj())

    @property
    def nqubit(self) -> int:
        """
        Returns the number of qubits.

        Returns
        -------
        int
            The number of qubits represented by the density matrix.
        """
        # Circumvent typing bug with numpy>=2.3
        # `shape` field is typed `tuple[Any, ...]` instead of `tuple[int, ...]`
        # See https://github.com/numpy/numpy/issues/29830
        nqubit: int = self.rho.shape[0].bit_length() - 1
        return nqubit

    def __str__(self) -> str:
        """
        Return a string representation of the DensityMatrix.

        Returns
        -------
        str
            A string description of the DensityMatrix instance, providing
            relevant information about its contents.
        """
        return f"DensityMatrix object, with density matrix {self.rho} and shape {self.dims()}."

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        """
        Add nodes (qubits) to the density matrix and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the density matrix.

        data : Data
            The state in which to initialize the newly added nodes. This can take several forms:
            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of the existing nodes,
              and each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all new nodes.
            - A multi-qubit state vector of dimension :math:`2^n` initializes the new nodes jointly.
            - A density matrix must have shape :math:`2^n \times 2^n`, and is used
              to jointly initialize the new nodes.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        dm_to_add = DensityMatrix(nqubit=nqubit, data=data)
        self.tensor(dm_to_add)

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """
        Evolves a single qubit by applying a specified operation.

        This method applies a given single-qubit operation to the qubit at the specified index
        in the DensityMatrix.

        Parameters
        ----------
        op : np.ndarray
            A 2x2 matrix representing the operation to apply.
        i : int
            The index of the qubit to which the operator is applied.

        Returns
        -------
        None
        """
        assert i >= 0
        assert i < self.nqubit
        if op.shape != (2, 2):
            raise ValueError("op must be 2*2 matrix.")

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)
        rho_tensor = tensordot(tensordot(op, rho_tensor, axes=(1, i)), op.conj().T, axes=(i + self.nqubit, 0))
        rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + self.nqubit))
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """
        Evolve the density matrix with a multi-qubit operation.

        Parameters
        ----------
        op : np.ndarray
            A 2^n x 2^n matrix representing the multi-qubit operation.
        qargs : sequence of int
            The indices of the target qubits to which the operation will be applied.

        Notes
        -----
        This method updates the density matrix by applying the specified multi-qubit
        operation to the qubits indicated by `qargs`.
        """
        d = op.shape
        # check it is a matrix.
        if len(d) == 2:
            # check it is square
            if d[0] == d[1]:
                pass
            else:
                raise ValueError(f"The provided operator has shape {op.shape} and is not a square matrix.")
        else:
            raise ValueError(f"The provided data has incorrect shape {op.shape}.")

        nqb_op = np.log2(len(op))
        if not np.isclose(nqb_op, int(nqb_op)):
            raise ValueError("Incorrect operator dimension: not consistent with qubits.")
        nqb_op = int(nqb_op)

        if nqb_op != len(qargs):
            raise ValueError("The dimension of the operator doesn't match the number of targets.")

        if not all(0 <= i < self.nqubit for i in qargs):
            raise ValueError("Incorrect target indices.")
        if len(set(qargs)) != nqb_op:
            raise ValueError("A repeated target qubit index is not possible.")

        op_tensor = op.reshape((2,) * 2 * nqb_op)

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)

        rho_tensor = tensordot(
            tensordot(op_tensor, rho_tensor, axes=(tuple(nqb_op + i for i in range(len(qargs))), tuple(qargs))),
            op.conj().T.reshape((2,) * 2 * nqb_op),
            axes=(tuple(i + self.nqubit for i in qargs), tuple(i for i in range(len(qargs)))),
        )
        rho_tensor = np.moveaxis(
            rho_tensor,
            list(range(len(qargs))) + [-i for i in range(1, len(qargs) + 1)],
            list(qargs) + [i + self.nqubit for i in reversed(list(qargs))],
        )
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """
        Return the expectation value of a single-qubit operator.

        Parameters
        ----------
        op : np.ndarray
            A 2x2 Hermitian operator.
        loc : int
            The index of the qubit on which to apply the operator.

        Returns
        -------
        complex
            The expectation value (real for Hermitian operators).
        """
        if not (0 <= loc < self.nqubit):
            raise ValueError(f"Wrong target qubit {loc}. Must between 0 and {self.nqubit - 1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        st1 = copy.copy(self)
        st1.normalize()

        nqubit = self.nqubit
        rho_tensor: Matrix = st1.rho.reshape((2,) * nqubit * 2)
        rho_tensor = tensordot(op, rho_tensor, axes=((1,), (loc,)))
        rho_tensor = np.moveaxis(rho_tensor, 0, loc)

        # complex() needed with mypy strict mode (no-any-return)
        return complex(np.trace(rho_tensor.reshape((2**nqubit, 2**nqubit))))

    def dims(self) -> tuple[int, ...]:
        """
        Return the dimensions of the density matrix.

        Returns
        -------
        tuple[int, ...]
            A tuple representing the dimensions of the density matrix.
        """
        return self.rho.shape

    def tensor(self, other: DensityMatrix) -> None:
        """
        Tensor product with another density matrix.

        Updates the current density matrix to be the tensor product
        of itself and another density matrix.

        Parameters
        ----------
        other : DensityMatrix
            DensityMatrix object to be tensored with the current instance.

        Notes
        -----
        This operation modifies the current density matrix in place.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self.rho = kron(self.rho, other.rho)

    def cnot(self, edge: tuple[int, int]) -> None:
        """
        Apply the CNOT gate to the density matrix.

        Parameters
        ----------
        edge : tuple of int
            A tuple representing the control and target qubits (e.g., (control, target)) on which to apply the CNOT gate.

        Notes
        -----
        The control qubit flips the target qubit if and only if the control qubit is in the |1⟩ state.
        """
        self.evolve(CNOT_TENSOR.reshape(4, 4), edge)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """
        Swap qubits.

        Parameters
        ----------
        qubits : tuple[int, int]
            A tuple containing the indices of the control and target qubits to be swapped.
        """
        self.evolve(SWAP_TENSOR.reshape(4, 4), qubits)

    def entangle(self, edge: tuple[int, int]) -> None:
        """
        Entangle qubits in the density matrix.

        Parameters
        ----------
        edge : tuple[int, int]
            A tuple representing the (control, target) qubit indices to be entangled.

        Notes
        -----
        This method modifies the density matrix to create entanglement between the
        specified qubit indices.
        """
        self.evolve(CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self) -> None:
        """
        Normalize the density matrix.

        This method rescales the density matrix such that its trace equals one.
        It modifies the current instance of the DensityMatrix class in-place.

        Returns
        -------
        None
        """
        # Note that the following calls to `astype` are guaranteed to
        # return the original NumPy array itself, since `copy=False` and
        # the `dtype` matches. This is important because the array is
        # then modified in place.
        if self.rho.dtype == np.object_:
            rho_o = self.rho.astype(np.object_, copy=False)
            rho_o /= np.trace(rho_o)
        else:
            rho_c = self.rho.astype(np.complex128, copy=False)
            rho_c /= np.trace(rho_c)

    @override
    def remove_qubit(self, qarg: int) -> None:
        """
        Remove a qubit from the density matrix.

        Parameters
        ----------
        qarg : int
            The index of the qubit to be removed.

        Returns
        -------
        None

        Notes
        -----
        This method modifies the density matrix by removing the specified qubit,
        which can affect the state representation.
        """
        self.ptrace(qarg)
        self.normalize()

    def ptrace(self, qargs: Collection[int] | int) -> None:
        """
        Perform the partial trace over specified qubits.

        Parameters
        ----------
        qargs : int or collection of int
            The indices of the qubits to be traced out. This can be a single integer
            corresponding to the index of a qubit or a collection of integers representing
            multiple qubits.
        """
        n = int(np.log2(self.rho.shape[0]))
        if isinstance(qargs, int):
            qargs = [qargs]
        assert isinstance(qargs, (list, tuple))
        qargs_num = len(qargs)
        nqubit_after = n - qargs_num
        assert n > 0
        assert all(qarg >= 0 and qarg < n for qarg in qargs)

        rho_res = self.rho.reshape((2,) * n * 2)
        # ket, bra indices to trace out
        trace_axes = list(qargs) + [n + qarg for qarg in qargs]
        op: Matrix = np.eye(2**qargs_num).reshape((2,) * qargs_num * 2).astype(np.complex128)
        rho_res = tensordot(op, rho_res, axes=(range(2 * qargs_num), trace_axes))

        self.rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))

    def fidelity(self, statevec: Statevec) -> ExpressionOrFloat:
        """
        Calculate the fidelity against a reference state vector.

        Parameters
        ----------
        statevec : numpy.ndarray
            The state vector (flattened numpy array) to compare with.

        Returns
        -------
        ExpressionOrFloat
            The calculated fidelity between the current density matrix and the provided state vector.
        """
        result = vdot(statevec.psi, matmul(self.rho, statevec.psi))
        if isinstance(result, Expression):
            return result
        assert math.isclose(result.imag, 0)
        return result.real

    def flatten(self) -> Matrix:
        """
        Returns a flattened density matrix.

        This method takes the current density matrix and converts it into a one-dimensional array,
        effectively flattening it. This can be useful for various applications in quantum mechanics
        and other fields where a one-dimensional representation of the matrix is needed.

        Returns
        -------
        Matrix
            A one-dimensional representation of the density matrix.
        """
        return self.rho.flatten()

    def apply_channel(self, channel: KrausChannel, qargs: Sequence[int]) -> None:
        """
        Apply a channel to a density matrix.

        Parameters
        ----------
        channel : KrausChannel
            The KrausChannel to be applied to the density matrix.
        qargs : Sequence[int]
            The target qubit indices where the channel will be applied.

        Returns
        -------
        None
            This function modifies the density matrix in place and does not return a value.

        Raises
        ------
        ValueError
            If the final density matrix is not normalized after the application of the channel.
            This shouldn't happen since KrausChannel objects are normalized by construction.
        """
        result_array = np.zeros((2**self.nqubit, 2**self.nqubit), dtype=np.complex128)

        if not isinstance(channel, KrausChannel):
            raise TypeError("Can't apply a channel that is not a Channel object.")

        for k_op in channel:
            dm = copy.copy(self)
            dm.evolve(k_op.operator, qargs)
            result_array += k_op.coef * np.conj(k_op.coef) * dm.rho
            # reinitialize to input density matrix

        if not np.allclose(result_array.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")

        self.rho = result_array

    @override
    def apply_noise(self, qubits: Sequence[int], noise: Noise) -> None:
        """
        Apply noise to the specified qubits.

        Parameters
        ----------
        qubits : sequence of int
            The target qubits to which the noise will be applied.
        noise : Noise
            The noise process to be applied to the specified qubits.
        """
        channel = noise.to_kraus_channel()
        self.apply_channel(channel, qubits)

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> DensityMatrix:
        """
        Return a copy of the density matrix with all occurrences of a specified variable in measurement angles replaced by a provided value.

        Parameters
        ----------
        variable : Parameter
            The variable to be substituted in the density matrix.
        substitute : ExpressionOrSupportsFloat
            The value or expression that will replace the specified variable.

        Returns
        -------
        DensityMatrix
            A new instance of `DensityMatrix` with the substitutions applied.
        """
        result = copy.copy(self)
        result.rho = np.vectorize(lambda value: parameter.subs(value, variable, substitute))(self.rho)
        return result

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> DensityMatrix:
        """
        Return a copy of the density matrix with all occurrences of the specified keys
        in measurement angles substituted by the corresponding values in parallel.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A mapping of parameters to their replacement values. The keys represent
            the parameters whose occurrences in the density matrix will be replaced,
            and the values are the new values to substitute.

        Returns
        -------
        DensityMatrix
            A new DensityMatrix instance with the substitutions applied.
        """
        result = copy.copy(self)
        result.rho = np.vectorize(lambda value: parameter.xreplace(value, assignment))(self.rho)
        return result


@dataclass(frozen=True)
class DensityMatrixBackend(DenseStateBackend[DensityMatrix]):
    """
    A class representing a Measurement-Based Quantum Computation (MBQC) simulator
    using the density matrix method.

    Attributes
    ----------
    density_matrix : numpy.ndarray
        The density matrix representing the state of the quantum system.

    Methods
    -------
    apply_gate(gate, qubits)
        Applies a quantum gate to specified qubits in the density matrix.

    measure(qubit)
        Measures a specified qubit and updates the density matrix according to the measurement outcome.

    reset()
        Resets the density matrix to the initial state.
    """

    state: DensityMatrix = dataclasses.field(init=False, default_factory=lambda: DensityMatrix(nqubit=0))
