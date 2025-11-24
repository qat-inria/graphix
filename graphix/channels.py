"""
Quantum channels and noise models.

This module provides implementations and utilities for quantum channels
and various noise models used in quantum computing. It allows users to
simulate the effects of noise on quantum states and operations, enabling
the analysis and design of robust quantum algorithms and protocols.

Key functionalities include:
- Definition and representation of different quantum channels.
- Simulation of noise effects on quantum operations.
- Tools for analyzing the performance of quantum algorithms in the
  presence of noise.
"""

from __future__ import annotations

import copy
import typing
from typing import TYPE_CHECKING, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from graphix import linalg_validations as lv
from graphix.ops import Ops

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_T = TypeVar("_T", bound=np.generic)


def _ilog2(n: int) -> int:
    """
    Return the integer base-2 logarithm of `n`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The integer part of the base-2 logarithm of `n`, specifically
        `floor(log2(n))` for `n > 0`.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    return (n - 1).bit_length()


class KrausData:
    """
    Kraus operator data.

    Attributes
    ----------
    coef : complex
        Scalar prefactor of the operator.

    operator : npt.NDArray[np.complex128]
        Operator.
    """

    __coef: complex
    __operator: npt.NDArray[np.complex128]

    def __init__(self, coef: complex, operator: npt.NDArray[_T]) -> None:
        if not lv.is_square(operator):
            raise ValueError("Operator must be a square matrix.")
        if not lv.is_qubitop(operator):
            raise ValueError("Operator must be a qubit operator.")
        self.__coef = coef
        self.__operator = operator.astype(np.complex128, copy=True)
        self.__operator.flags.writeable = False

    @property
    def coef(self) -> complex:
        """
        Return the scalar prefactor.

        Returns
        -------
        complex
            The scalar prefactor associated with the Kraus operator.
        """
        return self.__coef

    @property
    def operator(self) -> npt.NDArray[np.complex128]:
        """
        Return the Kraus operator.

        Returns
        -------
        npt.NDArray[np.complex128]
            The Kraus operator associated with this instance of `KrausData`.
        """
        return self.__operator.view()

    @property
    def nqubit(self) -> int:
        """
        Get the number of qubits in the Kraus data.

        This property calculates and returns the number of qubits
        based on the shape of the associated Kraus operators.

        Returns
        -------
        int
            The number of qubits represented by the Kraus operators.

        Raises
        ------
        ValueError
            If the dimensions of the Kraus operators are inconsistent
            with a valid qubit representation.
        """
        size, _ = self.__operator.shape
        return _ilog2(size)


class KrausChannel:
    """
    Quantum channel class in the Kraus representation.

    Defined by Kraus operators :math:`K_i` with scalar prefactors
    :code:`coef` :math:`c_i`, where the channel acts on the density matrix
    as :math:`\rho' = \sum_i K_i^\dagger \rho K_i`. The data should satisfy
    :math:`\sum_i K_i^\dagger K_i = I`.
    """

    __nqubit: int
    __data: list[KrausData]

    @staticmethod
    def _nqubit(kraus_data: Iterable[KrausData]) -> int:
        """
        Return the number of qubits acted on by the given Kraus operators.

        Parameters
        ----------
        kraus_data : Iterable[KrausData]
            An iterable of KrausData objects representing the Kraus operators.

        Returns
        -------
        int
            The number of qubits that the provided Kraus operators act on.
        """
        # MEMO: ``kraus_data`` is not empty.
        it = iter(kraus_data)
        nqubit = next(it).nqubit

        if any(data.nqubit != nqubit for data in it):
            raise ValueError("All operators must have the same shape.")

        return nqubit

    def __init__(self, kraus_data: Iterable[KrausData]) -> None:
        """
        Initialize a KrausChannel given a Kraus operator.

        Parameters
        ----------
        kraus_data : Iterable[KrausData]
            Iterable of Kraus operator data.

        Raises
        ------
        ValueError
            If kraus_data is empty.
        """
        kraus_data = [copy.deepcopy(kdata) for kdata in kraus_data]

        if not kraus_data:
            raise ValueError("Cannot instantiate the channel with empty data.")

        self.__nqubit = self._nqubit(kraus_data)
        self.__data = kraus_data

        if len(self.__data) > 4**self.__nqubit:
            raise ValueError("len(kraus_data) cannot exceed 4**nqubit.")

        # Check that the channel is properly normalized, i.e., \sum_K_i^\dagger K_i = Identity.
        data = next(iter(self.__data))
        work = np.zeros_like(data.operator, dtype=np.complex128)
        for data in self.__data:
            m = data.coef * data.operator
            work += m.conj().T @ m
        if not np.allclose(work, np.eye(2**self.__nqubit)):
            raise ValueError("The specified channel is not normalized.")

    @typing.overload
    def __getitem__(self, index: SupportsIndex, /) -> KrausData: ...

    @typing.overload
    def __getitem__(self, index: slice, /) -> list[KrausData]: ...

    def __getitem__(self, index: SupportsIndex | slice, /) -> KrausData | list[KrausData]:
        """
        Return the Kraus operator(s) at the specified index or indices.

        Parameters
        ----------
        index : SupportsIndex | slice
            The index or slice to access the Kraus operator(s).
            This can be a single index to retrieve one Kraus operator,
            or a slice to retrieve multiple operators.

        Returns
        -------
        KrausData | list[KrausData]
            The Kraus operator at the given index if a single index is provided,
            or a list of Kraus operators if a slice is provided.

        Notes
        -----
        - Ensure that the index is within the valid range of available Kraus operators.
        """
        return self.__data[index]

    def __len__(self) -> int:
        """
        Return the number of Kraus operators.

        Returns
        -------
        int
            The number of Kraus operators associated with the channel.
        """
        return len(self.__data)

    def __iter__(self) -> Iterator[KrausData]:
        """
        Iterate over the Kraus operators.

        Yields
        ------
        KrausData
            Each Kraus operator in the channel.
        """
        return iter(self.__data)

    @property
    def nqubit(self) -> int:
        """
        Get the number of qubits in the Kraus channel.

        Returns
        -------
        int
            The number of qubits.
        """
        return self.__nqubit


def dephasing_channel(prob: float) -> KrausChannel:
    """
    Single-qubit dephasing channel.

    The dephasing channel is defined by the equation:
    :math:`(1-p) \rho + p Z \rho Z`, where :math:`Z` is the Pauli-Z operator.

    Parameters
    ----------
    prob : float
        The probability associated with the dephasing channel,
        where :math:`p` is the probability of applying the Z operation.

    Returns
    -------
    :class:`graphix.channels.KrausChannel`
        An object containing the corresponding Kraus operators for
        the dephasing channel.
    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob), Ops.Z),
        ]
    )


def depolarising_channel(prob: float) -> KrausChannel:
    """
    Single-qubit depolarizing channel.

    The depolarizing channel is defined mathematically as:

    .. math::
        (1-p) \rho + \frac{p}{3} (X \rho X + Y \rho Y + Z \rho Z) = (1 - 4 \frac{p}{3}) \rho + 4 \frac{p}{3} \text{id}

    where \( p \) is the probability of depolarization and \( \rho \) is the density matrix of the qubit.

    Parameters
    ----------
    prob : float
        The probability associated with the channel (0 ≤ prob ≤ 1).

    Returns
    -------
    KrausChannel
        A Kraus channel representing the depolarizing channel with the given probability.

    Raises
    ------
    ValueError
        If `prob` is not in the range [0, 1].
    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob / 3.0), Ops.X),
            KrausData(np.sqrt(prob / 3.0), Ops.Y),
            KrausData(np.sqrt(prob / 3.0), Ops.Z),
        ]
    )


def pauli_channel(px: float, py: float, pz: float) -> KrausChannel:
    """
    Single-qubit Pauli channel.

    This function represents a single-qubit Pauli channel that models the
    effects of noise on a quantum state. The channel consists of a mixture
    of the identity operator and the Pauli operators (X, Y, Z) applied to
    the quantum state with certain probabilities.

    The mathematical representation of the Pauli channel is given by:

    .. math::
        (1-p_X-p_Y-p_Z) \rho + p_X X \rho X + p_Y Y \rho Y + p_Z Z \rho Z

    Parameters
    ----------
    px : float
        Probability of applying the Pauli-X operator.
    py : float
        Probability of applying the Pauli-Y operator.
    pz : float
        Probability of applying the Pauli-Z operator.

    Returns
    -------
    KrausChannel
        The Kraus representation of the Pauli channel.

    Raises
    ------
    ValueError
        If the sum of probabilities exceeds 1.

    Notes
    -----
    Ensure that the inputs satisfy the condition:
    `px + py + pz <= 1`. This guarantees that the probabilities
    are valid within the context of a quantum channel.
    """
    if px + py + pz > 1:
        raise ValueError("The sum of probabilities must not exceed 1.")
    p_i = 1 - px - py - pz
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - p_i), Ops.I),
            KrausData(np.sqrt(px / 3.0), Ops.X),
            KrausData(np.sqrt(py / 3.0), Ops.Y),
            KrausData(np.sqrt(pz / 3.0), Ops.Z),
        ]
    )


def two_qubit_depolarising_channel(prob: float) -> KrausChannel:
    """
    Two-qubit depolarising channel.

    The depolarising channel introduces errors to a two-qubit quantum system based on a given
    probability. The channel can be mathematically represented as:

    .. math::
        \mathcal{E} (\rho) = (1-p) \rho + \frac{p}{15}  \sum_{P_i \in \{id, X, Y ,Z\}^{\otimes 2}/(id \otimes id)} P_i \rho P_i

    Parameters
    ----------
    prob : float
        The probability of depolarisation. Must be in the range [0, 1].

    Returns
    -------
    :class:`graphix.channels.KrausChannel`
        An object containing the corresponding Kraus operators for the channel.
    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), np.kron(Ops.I, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.X)),
        ]
    )


def two_qubit_depolarising_tensor_channel(prob: float) -> KrausChannel:
    """
    Two-qubit tensor channel of single-qubit depolarising channels with the same probability.

    The Kraus operators for this channel are defined as follows:

    .. math::
        \Big\{ \sqrt{(1-p)} \, \text{id}, \sqrt{\frac{p}{3}} \, X, \sqrt{\frac{p}{3}} \, Y, \sqrt{\frac{p}{3}} \, Z \Big\} \otimes \Big\{ \sqrt{(1-p)} \, \text{id}, \sqrt{\frac{p}{3}} \, X, \sqrt{\frac{p}{3}} \, Y, \sqrt{\frac{p}{3}} \, Z \Big\}

    Parameters
    ----------
    prob : float
        The probability associated with the depolarising channel, where `0 <= prob <= 1`.

    Returns
    -------
    KrausChannel
        An instance of :class:`graphix.channels.KrausChannel` containing the corresponding Kraus operators.
    """
    return KrausChannel(
        [
            KrausData(1 - prob, np.kron(Ops.I, Ops.I)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Y)),
        ]
    )
