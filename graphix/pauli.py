"""
Pauli Gates
------------

This module provides implementations of the Pauli gates, which include
the set {I, X, Y, Z} combined with the scaling factors ±{1, j}.

The Pauli gates are fundamental quantum gates used in quantum computing
that represent basic quantum operations.

Attributes
----------
I : matrix
    The identity gate.
X : matrix
    The Pauli-X gate (also known as NOT gate).
Y : matrix
    The Pauli-Y gate.
Z : matrix
    The Pauli-Z gate.

Usage
-----
These gates can be applied to qubits in a quantum circuit to perform quantum
operations.

Examples
--------
To apply a Pauli-X gate on a qubit:

    |0⟩ → X|0⟩ = |1⟩

To apply a Pauli-Z gate with a phase factor:

    Z = j * Z
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import typing_extensions

from graphix.fundamentals import IXYZ, Axis, ComplexUnit, SupportsComplexCtor
from graphix.ops import Ops
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from graphix.states import PlanarState


class _PauliMeta(type):
    def __iter__(cls) -> Iterator[Pauli]:
        """
        Iterate over all Pauli gates, including the unit.

        Yields
        -------
        Pauli
            An iterator over all possible Pauli gates, including the identity gate.

        Notes
        -----
        The Pauli gates include: I (identity), X, Y, Z.
        This method provides an iterable interface to access all of them in sequence.
        """
        return Pauli.iterate()


@dataclasses.dataclass(frozen=True)
class Pauli(metaclass=_PauliMeta):
    """
    Pauli gate: ``u * {I, X, Y, Z}``, where u is a complex unit.

    This class represents the Pauli gates, which can be combined with other
    Pauli gates using the matrix multiplication operator (``@``), and with
    complex units and unit constants using the multiplication operator (``*``).
    Additionally, Pauli gates can be negated.

    Attributes
    ----------
    - None

    Methods
    -------
    - __matmul__(other): Multiplies this Pauli gate by another.
    - __mul__(other): Scales this Pauli gate by a complex unit.
    - __neg__(): Negates this Pauli gate.
    """

    symbol: IXYZ = IXYZ.I
    unit: ComplexUnit = ComplexUnit.ONE
    I: ClassVar[Pauli]
    X: ClassVar[Pauli]
    Y: ClassVar[Pauli]
    Z: ClassVar[Pauli]

    @staticmethod
    def from_axis(axis: Axis) -> Pauli:
        """
        Create a Pauli object from the specified axis.

        Parameters
        ----------
        axis : Axis
            The axis associated with the desired Pauli operation.

        Returns
        -------
        Pauli
            The Pauli object corresponding to the provided axis.
        """
        return Pauli(IXYZ[axis.name])

    @property
    def axis(self) -> Axis:
        """
        Return the axis associated with the Pauli operator.

        Raises
        ------
        ValueError
            If the Pauli operator is the identity.
        """
        if self.symbol == IXYZ.I:
            raise ValueError("I is not an axis.")
        return Axis[self.symbol.name]

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """
        Return the matrix representation of the Pauli gate.

        Returns
        -------
        npt.NDArray[np.complex128]
            A 2x2 complex numpy array representing the matrix of the Pauli gate.
        """
        co = complex(self.unit)
        if self.symbol == IXYZ.I:
            return co * Ops.I
        if self.symbol == IXYZ.X:
            return co * Ops.X
        if self.symbol == IXYZ.Y:
            return co * Ops.Y
        if self.symbol == IXYZ.Z:
            return co * Ops.Z
        typing_extensions.assert_never(self.symbol)

    def eigenstate(self, binary: int = 0) -> PlanarState:
        """
        Return the eigenstate of the Pauli operator.

        Parameters
        ----------
        binary : int, optional
            The binary representation of the eigenstate, default is 0.

        Returns
        -------
        PlanarState
            The eigenstate corresponding to the specified binary input.
        """
        if binary not in {0, 1}:
            raise ValueError("b must be 0 or 1.")
        if self.symbol == IXYZ.X:
            return BasicStates.PLUS if binary == 0 else BasicStates.MINUS
        if self.symbol == IXYZ.Y:
            return BasicStates.PLUS_I if binary == 0 else BasicStates.MINUS_I
        if self.symbol == IXYZ.Z:
            return BasicStates.ZERO if binary == 0 else BasicStates.ONE
        # Any state is eigenstate of the identity
        if self.symbol == IXYZ.I:
            return BasicStates.PLUS
        typing_extensions.assert_never(self.symbol)

    def _repr_impl(self, prefix: str | None) -> str:
        """
        Return the string representation of the Pauli operator with an optional prefix.

        Parameters
        ----------
        prefix : str or None
            An optional string to prepend to the representation. If None, no prefix is added.

        Returns
        -------
        str
            The string representation of the Pauli operator, optionally prefixed.
        """
        sym = self.symbol.name
        if prefix is not None:
            sym = f"{prefix}.{sym}"
        if self.unit == ComplexUnit.ONE:
            return sym
        if self.unit == ComplexUnit.MINUS_ONE:
            return f"-{sym}"
        if self.unit == ComplexUnit.J:
            return f"1j * {sym}"
        if self.unit == ComplexUnit.MINUS_J:
            return f"-1j * {sym}"
        typing_extensions.assert_never(self.unit)

    def __repr__(self) -> str:
        """
        Return a string representation of the Pauli.

        Returns
        -------
        str
            A string representing the Pauli object.
        """
        return self._repr_impl(self.__class__.__name__)

    def __str__(self) -> str:
        """
        Return a simplified string representation of the Pauli operators.

        Returns
        -------
        str
            A string representation of the Pauli operator.
        """
        return self._repr_impl(None)

    @staticmethod
    def _matmul_impl(lhs: IXYZ, rhs: IXYZ) -> Pauli:
        """Return the product of ``lhs`` and ``rhs`` ignoring units."""
        if lhs == IXYZ.I:
            return Pauli(rhs)
        if rhs == IXYZ.I:
            return Pauli(lhs)
        if lhs == rhs:
            return Pauli()
        lr = (lhs, rhs)
        if lr == (IXYZ.X, IXYZ.Y):
            return Pauli(IXYZ.Z, ComplexUnit.J)
        if lr == (IXYZ.Y, IXYZ.X):
            return Pauli(IXYZ.Z, ComplexUnit.MINUS_J)
        if lr == (IXYZ.Y, IXYZ.Z):
            return Pauli(IXYZ.X, ComplexUnit.J)
        if lr == (IXYZ.Z, IXYZ.Y):
            return Pauli(IXYZ.X, ComplexUnit.MINUS_J)
        if lr == (IXYZ.Z, IXYZ.X):
            return Pauli(IXYZ.Y, ComplexUnit.J)
        if lr == (IXYZ.X, IXYZ.Z):
            return Pauli(IXYZ.Y, ComplexUnit.MINUS_J)
        raise RuntimeError("Unreachable.")  # pragma: no cover

    def __matmul__(self, other: Pauli) -> Pauli:
        """
        Compute the matrix product of two Pauli operators.

        Parameters
        ----------
        other : Pauli
            The Pauli operator to multiply with the current instance.

        Returns
        -------
        Pauli
            A new Pauli operator that is the result of the matrix multiplication.
        """
        if isinstance(other, Pauli):
            return self._matmul_impl(self.symbol, other.symbol) * (self.unit * other.unit)
        return NotImplemented

    def __mul__(self, other: ComplexUnit | SupportsComplexCtor) -> Pauli:
        """
        Return the product of two Paulis.

        Parameters
        ----------
        other : ComplexUnit | SupportsComplexCtor
            The Pauli or complex number to multiply with the current Pauli object.

        Returns
        -------
        Pauli
            The resulting Pauli object from the multiplication.
        """
        if u := ComplexUnit.try_from(other):
            return dataclasses.replace(self, unit=self.unit * u)
        return NotImplemented

    def __rmul__(self, other: ComplexUnit | SupportsComplexCtor) -> Pauli:
        """
        Return the product of a scalar and a Pauli operator.

        This method is invoked when the left operand is a scalar and the right
        operand is an instance of the Pauli class. The result is a new Pauli
        operator that represents the scalar multiplication of the original Pauli.

        Parameters
        ----------
        other : ComplexUnit | SupportsComplexCtor
            A scalar value (complex or compatible type) to multiply with this Pauli operator.

        Returns
        -------
        Pauli
            A new Pauli operator that is the result of the scalar multiplication.
        """
        return self.__mul__(other)

    def __neg__(self) -> Pauli:
        """
        Return the negation of the Pauli operator.

        This method returns a new Pauli operator that represents the opposite
        of the current operator.

        Returns
        -------
        Pauli
            A new Pauli instance that is the negation of the current instance.
        """
        return dataclasses.replace(self, unit=-self.unit)

    @staticmethod
    def iterate(symbol_only: bool = False) -> Iterator[Pauli]:
        """
        Iterate over all Pauli gates.

        Parameters
        ----------
        symbol_only : bool, optional
            If True, exclude the unit in the iteration. Default is False.

        Yields
        ------
        Pauli
            An iterator over all Pauli gates, potentially omitting the unit gate based on the `symbol_only` parameter.
        """
        us = (ComplexUnit.ONE,) if symbol_only else tuple(ComplexUnit)
        for symbol in IXYZ:
            for unit in us:
                yield Pauli(symbol, unit)


Pauli.I = Pauli(IXYZ.I)
Pauli.X = Pauli(IXYZ.X)
Pauli.Y = Pauli(IXYZ.Y)
Pauli.Z = Pauli(IXYZ.Z)
