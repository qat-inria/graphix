"""
Quantum states and operators.

This module provides definitions and functionalities for quantum states
and operators, crucial components in the study of quantum mechanics and
quantum computing. It includes representations, manipulations, and
operations associated with quantum systems.
"""

from __future__ import annotations

import abc
import dataclasses
from abc import ABC
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import typing_extensions

from graphix.fundamentals import Plane


# generic class State for all States
# FIXME: Name conflict
class State(ABC):
    """
    Abstract base class for single qubit state objects.

    The only requirement for concrete classes is to implement
    the `get_statevector()` method, which returns the state vector
    representation of the state.
    """

    @abc.abstractmethod
    def get_statevector(self) -> npt.NDArray[np.complex128]:
        """
        Get the state vector of the quantum state.

        Returns
        -------
        npt.NDArray[np.complex128]
            The complex-valued state vector representing the quantum state.
        """

    def get_densitymatrix(self) -> npt.NDArray[np.complex128]:
        """
        Return the density matrix of the quantum state.

        Returns
        -------
        numpy.ndarray
            A complex-valued 2D array representing the density matrix of the state,
            where each entry corresponds to the probability amplitudes for the quantum state.
        """
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.get_statevector(), self.get_statevector().conj()).astype(np.complex128, copy=False)


@dataclasses.dataclass
class PlanarState(State):
    """
    Light object used to instantiate backends.

    This class does not cover all possible states; this is addressed
    in :class:`graphix.sim.statevec.Statevec` and
    :class:`graphix.sim.densitymatrix.DensityMatrix` constructors.

    Parameters
    ----------
    plane : :class:`graphix.pauli.Plane`
        One of the three planes (XY, XZ, YZ).
    angle : complex
        Angle in radians.

    Returns
    -------
    : class:`graphix.states.State`
        A State object.
    """

    plane: Plane
    angle: float

    def __repr__(self) -> str:
        """
        Return a string representation of the PlanarState object.

        Returns
        -------
        str
            A string that represents the current state of the PlanarState instance.
        """
        return f"graphix.states.PlanarState({self.plane}, {self.angle})"

    def __str__(self) -> str:
        """
        Return a string representation of the planar state.

        This method provides a textual description of the current state
        of the PlanarState instance, suitable for output and logging.

        Returns
        -------
        str
            A string that describes the planar state.
        """
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def get_statevector(self) -> npt.NDArray[np.complex128]:
        """
        Return the state vector of the PlanarState instance.

        Returns
        -------
        statevector : ndarray, shape (n,)
            The state vector represented as a complex ndarray, where
            n is the dimensionality of the state.
        """
        if self.plane == Plane.XY:
            return np.asarray([1 / np.sqrt(2), np.exp(1j * self.angle) / np.sqrt(2)], dtype=np.complex128)

        if self.plane == Plane.YZ:
            return np.asarray([np.cos(self.angle / 2), 1j * np.sin(self.angle / 2)], dtype=np.complex128)

        if self.plane == Plane.XZ:
            return np.asarray([np.cos(self.angle / 2), np.sin(self.angle / 2)], dtype=np.complex128)
        # other case never happens since exhaustive
        typing_extensions.assert_never(self.plane)


# States namespace for input initialization.
class BasicStates:
    """
    Basic states.

    This class represents a collection of basic states used in the application.

    Parameters
    ----------
    None

    Attributes
    ----------
    states : list
        A list of basic state representations.

    Methods
    -------
    add_state(state)
        Adds a new state to the list of basic states.

    remove_state(state)
        Removes a state from the list of basic states if it exists.

    get_states()
        Returns a list of all the current states.
    """

    ZERO: ClassVar[PlanarState] = PlanarState(Plane.XZ, 0)
    ONE: ClassVar[PlanarState] = PlanarState(Plane.XZ, np.pi)
    PLUS: ClassVar[PlanarState] = PlanarState(Plane.XY, 0)
    MINUS: ClassVar[PlanarState] = PlanarState(Plane.XY, np.pi)
    PLUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, np.pi / 2)
    MINUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, -np.pi / 2)
    # remove that in the end
    # need in TN backend
    VEC: ClassVar[list[PlanarState]] = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]
