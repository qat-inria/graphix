"""
24 Unique single-qubit Clifford gates and their multiplications, conjugations,
and Pauli conjugations.

This module provides functionalities for defining and manipulating the 24 unique
single-qubit Clifford gates. It includes operations such as multiplying gates,
performing conjugations, and applying Pauli conjugations. Each gate is represented
in a standard form and can be combined with other gates to form complex quantum
operations.
"""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import typing_extensions

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_LABEL,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_TO_QASM3,
)
from graphix.fundamentals import IXYZ, ComplexUnit
from graphix.measurements import Domains
from graphix.pauli import Pauli

if TYPE_CHECKING:
    import numpy.typing as npt


class Clifford(Enum):
    """
    Clifford Gate Class.

    The Clifford class represents a quantum gate that is a member of the
    Clifford group. This group consists of gates that preserve the
    structure of quantum computations and can be efficiently simulated
    on a classical computer.

    Attributes
    ----------
    name : str
        The name of the Clifford gate.
    matrix : numpy.ndarray
        The unitary matrix representation of the Clifford gate.

    Methods
    -------
    apply(state):
        Applies the Clifford gate to the given quantum state.

    inverse():
        Returns the inverse of the Clifford gate.
    """

    # MEMO: Cannot use ClassVar here
    I: Clifford
    X: Clifford
    Y: Clifford
    Z: Clifford
    S: Clifford
    SDG: Clifford
    H: Clifford

    _0 = 0
    _1 = 1
    _2 = 2
    _3 = 3
    _4 = 4
    _5 = 5
    _6 = 6
    _7 = 7
    _8 = 8
    _9 = 9
    _10 = 10
    _11 = 11
    _12 = 12
    _13 = 13
    _14 = 14
    _15 = 15
    _16 = 16
    _17 = 17
    _18 = 18
    _19 = 19
    _20 = 20
    _21 = 21
    _22 = 22
    _23 = 23

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """
        Return the matrix representation of the Clifford gate.

        Returns
        -------
        npt.NDArray[np.complex128]
            A complex-valued numpy array representing the matrix of the Clifford gate.
        """
        return CLIFFORD[self.value]

    @staticmethod
    def try_from_matrix(mat: npt.NDArray[Any]) -> Clifford | None:
        """
        Try to construct a Clifford gate from a given matrix.

        Parameters
        ----------
        mat : npt.NDArray[Any]
            The input matrix that represents a potential Clifford gate.

        Returns
        -------
        Clifford or None
            The corresponding Clifford gate if found, otherwise None.

        Notes
        -----
        The global phase is ignored when determining the Clifford gate.
        """
        if mat.shape != (2, 2):
            return None
        for ci in Clifford:
            mi = ci.matrix
            for piv, piv_ in zip(mat.flat, mi.flat):
                if math.isclose(abs(piv), 0):
                    continue
                if math.isclose(abs(piv_), 0):
                    continue
                if np.allclose(mat / piv, mi / piv_):
                    return ci
        return None

    def __repr__(self) -> str:
        """
        Return a string representation of the Clifford expression in the form of
        HSZ decomposition.

        Returns
        -------
        str
            A string that represents the Clifford expression.
        """
        formula = " @ ".join([f"Clifford.{gate}" for gate in self.hsz])
        if len(self.hsz) == 1:
            return formula
        return f"({formula})"

    def __str__(self) -> str:
        """
        Return the string representation of the Clifford gate.

        This method retrieves the name of the specific Clifford gate
        represented by the instance.

        Returns
        -------
        str
            The name of the Clifford gate.
        """
        return CLIFFORD_LABEL[self.value]

    @property
    def conj(self) -> Clifford:
        """
        Return the conjugate of the Clifford gate.

        A Clifford gate is a type of quantum gate that is important in quantum computing.
        The conjugate of a Clifford gate is obtained by applying the conjugate operation
        to the gate representation.

        Returns
        -------
        Clifford
            A new Clifford gate that represents the conjugate of the original gate.
        """
        return Clifford(CLIFFORD_CONJ[self.value])

    @property
    def hsz(self) -> list[Clifford]:
        """
        Return a decomposition of the Clifford gate using the gates 'H', 'S', and 'Z'.

        The decomposition provides a representation of the Clifford gate
        in terms of the Hadamard ('H'), Phase ('S'), and Pauli-Z ('Z') gates.

        Returns
        -------
        list[Clifford]
            A list containing the decomposition of the Clifford gate into
            the specified gate operations.
        """
        return [Clifford(i) for i in CLIFFORD_HSZ_DECOMPOSITION[self.value]]

    @property
    def qasm3(self) -> tuple[str, ...]:
        """
        Return a decomposition of the Clifford gate as qasm3 gates.

        Returns
        -------
        tuple[str, ...]
            A tuple containing the qasm3 representation of the
            Clifford gate decomposition.

        Notes
        -----
        The qasm3 format is used for representing quantum circuits in a
        standardized way for various quantum programming frameworks.
        """
        return CLIFFORD_TO_QASM3[self.value]

    def __matmul__(self, other: Clifford) -> Clifford:
        """
        Perform matrix multiplication within the Clifford group.

        Parameters
        ----------
        other : Clifford
            The Clifford object to multiply with the current instance.

        Returns
        -------
        Clifford
            A new Clifford object resulting from the multiplication of the two Clifford instances,
            computed modulo a unit factor.

        Notes
        -----
        This operation follows the rules of multiplication specific to the Clifford group, ensuring
        that the result remains within the group.

        Examples
        --------
        Here is an example of how to use the __matmul__ method:

            >>> c1 = Clifford(...)
            >>> c2 = Clifford(...)
            >>> result = c1 @ c2
        """
        if isinstance(other, Clifford):
            return Clifford(CLIFFORD_MUL[self.value][other.value])
        return NotImplemented

    def measure(self, pauli: Pauli) -> Pauli:
        """
        Compute the measurement of a Pauli operator using Clifford operations.

        Parameters
        ----------
        pauli : Pauli
            The Pauli operator to be measured.

        Returns
        -------
        Pauli
            The result of the measurement operation, which is the transformed Pauli operator
            after applying the Clifford operations.

        Notes
        -----
        This method computes the result of the operation \( C^\dagger P C \), where \( C \)
        is the Clifford operation and \( P \) is the given Pauli operator.
        """
        if pauli.symbol == IXYZ.I:
            return copy.deepcopy(pauli)
        table = CLIFFORD_MEASURE[self.value]
        if pauli.symbol == IXYZ.X:
            symbol, sign = table.x
        elif pauli.symbol == IXYZ.Y:
            symbol, sign = table.y
        elif pauli.symbol == IXYZ.Z:
            symbol, sign = table.z
        else:
            typing_extensions.assert_never(pauli.symbol)
        return pauli.unit * Pauli(symbol, ComplexUnit.from_properties(sign=sign))

    def commute_domains(self, domains: Domains) -> Domains:
        """
        Commute `X^sZ^t` with `C`.

        Given the operator `X^sZ^t`, this method returns the operator `X^s'Z^t'` such that the equality
        `X^sZ^t C = C X^s'Z^t'` holds.

        Parameters
        ----------
        domains : Domains
            The domains object containing the representation of the operators involved in the commutation.

        Returns
        -------
        Domains
            A new domains object representing the commuted operator `X^s'Z^t'`.

        Notes
        -----
        Applying this method to `self.conj` computes the reverse commutation:
        indeed, `C†X^sZ^t = (X^sZ^tC)† = (CX^s'Z^t')† = X^s'Z^t'C†`.
        """
        s_domain = domains.s_domain.copy()
        t_domain = domains.t_domain.copy()
        for gate in self.hsz:
            if gate == Clifford.I:
                pass
            elif gate == Clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == Clifford.S:
                t_domain ^= s_domain
            elif gate == Clifford.Z:
                pass
            else:  # pragma: no cover
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        return Domains(s_domain, t_domain)


Clifford.I = Clifford(0)
Clifford.X = Clifford(1)
Clifford.Y = Clifford(2)
Clifford.Z = Clifford(3)
Clifford.S = Clifford(4)
Clifford.SDG = Clifford(5)
Clifford.H = Clifford(6)
