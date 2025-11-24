"""
Quantum states and operators.

This module provides functionalities related to quantum mechanics,
specifically focusing on the representation and manipulation of quantum
states and operators. It includes definitions, operations, and common
mathematical techniques used in quantum theory.
"""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np
import numpy.typing as npt

from graphix import utils
from graphix.parameter import Expression, cos_sin, exp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.parameter import ExpressionOrComplex, ExpressionOrFloat


class Ops:
    """
    Basic single- and two-qubit operators.

    This class provides various methods to create and manipulate
    single and two-qubit operators commonly used in quantum computing.

    Attributes
    ----------
    None

    Methods
    -------
    single_qubit_operator(name):
        Returns a single-qubit operator based on the specified name.

    two_qubit_operator(name):
        Returns a two-qubit operator based on the specified name.

    apply_operator(state, operator):
        Applies the given operator to the specified quantum state.

    example_method():
        An example method to demonstrate functionality.
    """

    I: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, 1]]))
    X: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[0, 1], [1, 0]]))
    Y: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[0, -1j], [1j, 0]]))
    Z: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, -1]]))
    S: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, 1j]]))
    SDG: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, -1j]]))
    H: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 1], [1, -1]]) / np.sqrt(2))
    CZ: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
        )
    )
    CNOT: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
        )
    )
    SWAP: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        )
    )
    CCX: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
        )
    )

    @overload
    @staticmethod
    def _cast_array(array: Iterable[Iterable[complex]], theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def _cast_array(
        array: Iterable[Iterable[ExpressionOrComplex]], theta: ExpressionOrFloat
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]: ...

    @staticmethod
    def _cast_array(
        array: Iterable[Iterable[ExpressionOrComplex]], theta: ExpressionOrFloat
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        if isinstance(theta, Expression):
            return np.asarray(array, dtype=np.object_)
        return np.asarray(array, dtype=np.complex128)

    @overload
    @staticmethod
    def rx(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rx(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rx(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """
        X rotation operator.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        operator : ndarray, shape (2, 2)
            The 2x2 X rotation operator as a NumPy array.
        """
        cos, sin = cos_sin(theta / 2)
        return Ops._cast_array(
            [[cos, -1j * sin], [-1j * sin, cos]],
            theta,
        )

    @overload
    @staticmethod
    def ry(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def ry(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def ry(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """
        Y rotation operator.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        operator : numpy.ndarray, shape (2, 2)
            The resulting Y rotation operator as a 2x2 numpy array.
        """
        cos, sin = cos_sin(theta / 2)
        return Ops._cast_array([[cos, -sin], [sin, cos]], theta)

    @overload
    @staticmethod
    def rz(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rz(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """
        Z rotation operator.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        operator : ndarray
            A 2x2 numpy array representing the Z rotation operator, which is of type
            complex128 or object.
        """
        return Ops._cast_array([[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]], theta)

    @overload
    @staticmethod
    def rzz(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rzz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rzz(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """
        zz-rotation operator.

        The zz-rotation operator is equivalent to the sequence of operations:
        1. CNOT(control, target)
        2. RZ(target, angle)
        3. CNOT(control, target)

        Parameters
        ----------
        theta : ExpressionOrFloat
            Rotation angle in radians.

        Returns
        -------
        operator : npt.NDArray[np.complex128] | npt.NDArray[np.object_]
            A 4x4 numpy array representing the zz-rotation operator.
        """
        return Ops._cast_array(Ops.CNOT @ np.kron(Ops.I, Ops.rz(theta)) @ Ops.CNOT, theta)

    @staticmethod
    def build_tensor_pauli_ops(n_qubits: int) -> npt.NDArray[np.complex128]:
        """
        Build all the 4^n tensor Pauli operators {I, X, Y, Z}^{\otimes n}.

        Parameters
        ----------
        n_qubits : int
            The number of copies (qubits) to consider.

        Returns
        -------
        np.ndarray
            An array of shape (2^n, 2^n) containing the 4^n operators.
        """
        if isinstance(n_qubits, int):
            if not n_qubits >= 1:
                raise ValueError(f"The number of qubits must be an integer <= 1 and not {n_qubits}.")
        else:
            raise TypeError(f"The number of qubits must be an integer and not {n_qubits}.")

        def _reducer(lhs: npt.NDArray[np.complex128], rhs: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
            return np.kron(lhs, rhs).astype(np.complex128, copy=False)

        return np.array([reduce(_reducer, i) for i in product((Ops.I, Ops.X, Ops.Y, Ops.Z), repeat=n_qubits)])
