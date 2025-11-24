"Data structure for single-qubit measurements in measurement-based quantum computation (MBQC)."

from __future__ import annotations

import dataclasses
import math
from typing import Literal, NamedTuple, SupportsInt

from typing_extensions import TypeAlias  # TypeAlias introduced in Python 3.10

from graphix import utils
from graphix.fundamentals import Axis, Plane, Sign

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001

Outcome: TypeAlias = Literal[0, 1]


def outcome(b: bool) -> Outcome:
    """
    Returns the corresponding integer value for a boolean input.

    Parameters
    ----------
    b : bool
        A boolean value to convert. If True, returns 1; if False, returns 0.

    Returns
    -------
    Outcome
        An integer representation of the boolean input: 1 for True and 0 for False.
    """
    return 1 if b else 0


def toggle_outcome(outcome: Outcome) -> Outcome:
    """
    Toggle the given outcome.

    Parameters
    ----------
    outcome : Outcome
        The outcome to be toggled.

    Returns
    -------
    Outcome
        The toggled outcome.
    """
    return 1 if outcome == 0 else 0


@dataclasses.dataclass
class Domains:
    """
    Represents the mathematical expression `X^s Z^t`, where `s` and `t` are the
    XOR of results derived from specified sets of indices.

    Attributes
    ----------
    s : int
        The result of the XOR operation on the specified set of indices
        for the first component, X.
    t : int
        The result of the XOR operation on the specified set of indices
        for the second component, Z.
    indices_X : list of int
        Indices used to compute the XOR for the first component, X.
    indices_Z : list of int
        Indices used to compute the XOR for the second component, Z.

    Methods
    -------
    calculate_xor(indices)
        Computes the XOR of the elements at the given indices.
    """

    s_domain: set[int]
    t_domain: set[int]


class Measurement(NamedTuple):
    """
    An MBQC measurement.

    Parameters
    ----------
    angle : float
        The angle of the measurement. Should be between [0, 2).
    plane : str
        The measurement plane.
    """

    angle: ExpressionOrFloat
    plane: Plane

    def isclose(self, other: Measurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """
        Compare if two measurements are in the same plane and their angles are close.

        Parameters
        ----------
        other : Measurement
            The measurement to compare against.
        rel_tol : float, optional
            The relative tolerance parameter (default is 1e-09). This parameter is used to determine
            whether two angles are close with respect to their magnitudes.
        abs_tol : float, optional
            The absolute tolerance parameter (default is 0.0). This parameter is used to determine
            whether two angles are close regardless of their magnitudes.

        Returns
        -------
        bool
            True if the measurements are in the same plane and their angles are close
            within the specified tolerances, False otherwise.

        Examples
        --------
        >>> from graphix.opengraph import Measurement
        >>> from graphix.fundamentals import Plane
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        True
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.YZ))
        False
        >>> Measurement(0.1, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        False
        """
        return (
            math.isclose(self.angle, other.angle, rel_tol=rel_tol, abs_tol=abs_tol)
            if isinstance(self.angle, float) and isinstance(other.angle, float)
            else self.angle == other.angle
        ) and self.plane == other.plane


class PauliMeasurement(NamedTuple):
    """
    Class for performing Pauli measurements on quantum states.

    This class provides methods to represent and apply Pauli measurements
    on quantum states in a given basis. It encapsulates the behavior of
    measuring quantum bits (qubits) under the Pauli operator basis
    (I, X, Y, Z).

    Parameters
    ----------
    measurement_type : str
        The type of Pauli measurement to be performed. Accepted values are
        'I', 'X', 'Y', or 'Z'.

    Attributes
    ----------
    measurement_type : str
        The type of Pauli measurement being performed.

    Methods
    -------
    apply(measurement_state)
        Apply the Pauli measurement to a given quantum state.

    get_measurement_outcome(measurement_state)
        Retrieve the outcome of the measurement on the specified quantum state.

    Notes
    -----
    The measurement is probabilistic, and the outcome may vary based on
    the quantum state being measured.

    Examples
    --------
    >>> measurement = PauliMeasurement('X')
    >>> state = [1, 0]  # Example quantum state
    >>> result = measurement.apply(state)
    >>> outcome = measurement.get_measurement_outcome(state)
    """

    axis: Axis
    sign: Sign

    @staticmethod
    def try_from(plane: Plane, angle: ExpressionOrFloat) -> PauliMeasurement | None:
        """
        Attempt to create a Pauli measurement description from the given parameters.

        Parameters
        ----------
        plane : Plane
            The plane in which the Pauli measurement is defined.
        angle : ExpressionOrFloat
            The angle associated with the Pauli measurement, which can be either
            an expression or a floating-point number.

        Returns
        -------
        PauliMeasurement or None
            The corresponding Pauli measurement description if the parameters
            are valid for a Pauli measurement; otherwise, returns None.
        """
        angle_double = 2 * angle
        if not isinstance(angle_double, SupportsInt) or not utils.is_integer(angle_double):
            return None
        angle_double_mod_4 = int(angle_double) % 4
        axis = plane.cos if angle_double_mod_4 % 2 == 0 else plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return PauliMeasurement(axis, sign)
