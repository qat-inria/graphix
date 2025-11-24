"""
Fundamental components related to quantum mechanics.

This module provides essential classes and functions that are
used in the study and application of quantum mechanics.
"""

from __future__ import annotations

import enum
import sys
import typing
from enum import Enum
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, SupportsIndex, overload

import typing_extensions

from graphix.ops import Ops
from graphix.parameter import cos_sin
from graphix.repr_mixins import EnumReprMixin

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from graphix.parameter import Expression, ExpressionOrFloat


if sys.version_info >= (3, 10):
    SupportsComplexCtor = SupportsComplex | SupportsFloat | SupportsIndex | complex
else:  # pragma: no cover
    from typing import Union

    SupportsComplexCtor = Union[SupportsComplex, SupportsFloat, SupportsIndex, complex]


class Sign(EnumReprMixin, Enum):
    """
    Sign, plus or minus.

    Attributes
    ----------
    value : str
        A string that represents the sign, either '+' or '-'.

    Methods
    -------
    __init__(self, value: str)
        Initializes the Sign instance with a given sign value.

    __str__(self) -> str
        Returns the string representation of the Sign instance.

    __eq__(self, other) -> bool
        Compares two Sign instances for equality.

    __ne__(self, other) -> bool
        Compares two Sign instances for inequality.

    is_positive(self) -> bool
        Returns True if the sign is positive, otherwise False.

    is_negative(self) -> bool
        Returns True if the sign is negative, otherwise False.
    """

    PLUS = 1
    MINUS = -1

    def __str__(self) -> str:
        """
        Return a string representation of the Sign.

        The method returns a string that indicates the sign value,
        which can either be '+' or '-'.

        Returns
        -------
        str
            A string representing the sign, either '+' or '-'.
        """
        if self == Sign.PLUS:
            return "+"
        return "-"

    @staticmethod
    def plus_if(b: bool) -> Sign:
        """
        Return a Sign object representing '+' if the input is True,
        and '-' if the input is False.

        Parameters
        ----------
        b : bool
            A boolean value that determines the Sign output.

        Returns
        -------
        Sign
            A Sign object representing either '+' or '-'
            based on the value of the input boolean.
        """
        if b:
            return Sign.PLUS
        return Sign.MINUS

    @staticmethod
    def minus_if(b: bool) -> Sign:
        """
        Return a Sign indicating the opposite based on the input boolean.

        Parameters
        ----------
        b : bool
            A boolean value that determines which Sign to return.

        Returns
        -------
        Sign
            Returns a Sign object representing '-' if `b` is True,
            and '+' if `b` is False.
        """
        if b:
            return Sign.MINUS
        return Sign.PLUS

    def __neg__(self) -> Sign:
        """
        Return the negation of the sign.

        The __neg__ method swaps the sign of the current instance.

        Returns
        -------
        Sign
            A new instance of Sign with the opposite sign.
        """
        return Sign.minus_if(self == Sign.PLUS)

    @typing.overload
    def __mul__(self, other: Sign) -> Sign: ...

    @typing.overload
    def __mul__(self, other: int) -> int: ...

    @typing.overload
    def __mul__(self, other: float) -> float: ...

    @typing.overload
    def __mul__(self, other: complex) -> complex: ...

    def __mul__(self, other: Sign | complex) -> Sign | int | float | complex:
        """
        Multiply the sign with another sign or a number.

        Parameters
        ----------
        other : Sign or complex
            The sign or complex number to multiply with.

        Returns
        -------
        Sign or int or float or complex
            The result of the multiplication, which can be a Sign,
            int, float, or a complex number depending on the type
            of the operand.

        Notes
        -----
        If `other` is an instance of `Sign`, the result will be a new
        instance of `Sign`. If `other` is a numeric type (int, float,
        or complex), the multiplication will be performed according
        to standard arithmetic rules.
        """
        if isinstance(other, Sign):
            return Sign.plus_if(self == other)
        if isinstance(other, int):
            return int(self) * other
        if isinstance(other, float):
            return float(self) * other
        if isinstance(other, complex):
            return complex(self) * other
        return NotImplemented

    @typing.overload
    def __rmul__(self, other: int) -> int: ...

    @typing.overload
    def __rmul__(self, other: float) -> float: ...

    @typing.overload
    def __rmul__(self, other: complex) -> complex: ...

    def __rmul__(self, other: complex) -> int | float | complex:
        """
        Multiply the sign with a number.

        Parameters
        ----------
        other : complex
            A complex number to be multiplied with the sign.

        Returns
        -------
        int, float, complex
            The result of the multiplication between the sign and the provided number.
        """
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __int__(self) -> int:
        """
        Converts the Sign instance to an integer representation.

        Returns:
            int:
                Returns `1` if the sign is positive (`+`),
                and `-1` if the sign is negative (`-`).
        """
        # mypy does not infer the return type correctly
        return self.value  # type: ignore[no-any-return]

    def __float__(self) -> float:
        """
        Returns a float representation of the sign.

        The method returns `1.0` when the sign is positive (`+`) and `-1.0` when the sign is negative (`-`).

        Returns
        -------
        float
            `1.0` for a positive sign and `-1.0` for a negative sign.
        """
        return float(self.value)

    def __complex__(self) -> complex:
        """
        Convert the `Sign` instance to a complex number.

        Returns
        -------
        complex
            Returns `1.0 + 0j` for a positive sign and `-1.0 + 0j` for a negative sign.
        """
        return complex(self.value)


class ComplexUnit(EnumReprMixin, Enum):
    """
    A class representing complex units: 1, -1, j, -j.

    The complex units can be multiplied with other complex units
    as well as with Python constants 1, -1, 1j, and -1j.
    Additionally, complex units can be negated.

    Attributes
    ----------
    value : complex
        The value of the complex unit, which can be one of the
        complex units defined in this class.

    Methods
    -------
    __mul__(other):
        Multiplies the complex unit by another complex unit or
        a Python constant.

    __neg__():
        Negates the complex unit.
    """

    # HACK: complex(u) == (1j) ** u.value for all u in ComplexUnit.

    ONE = 0
    J = 1
    MINUS_ONE = 2
    MINUS_J = 3

    @staticmethod
    def try_from(value: ComplexUnit | SupportsComplexCtor) -> ComplexUnit | None:
        """
        Returns a ComplexUnit instance if the given value is compatible.

        Parameters
        ----------
        value : ComplexUnit or SupportsComplexCtor
            The value to be converted into a ComplexUnit instance.

        Returns
        -------
        ComplexUnit or None
            A ComplexUnit instance if the value is compatible; otherwise, None.
        """
        if isinstance(value, ComplexUnit):
            return value
        value = complex(value)
        if value == 1:
            return ComplexUnit.ONE
        if value == -1:
            return ComplexUnit.MINUS_ONE
        if value == 1j:
            return ComplexUnit.J
        if value == -1j:
            return ComplexUnit.MINUS_J
        return None

    @staticmethod
    def from_properties(*, sign: Sign = Sign.PLUS, is_imag: bool = False) -> ComplexUnit:
        """
        Construct a `ComplexUnit` from its properties.

        Parameters
        ----------
        sign : Sign, optional
            The sign of the complex unit. Default is `Sign.PLUS`.
        is_imag : bool, optional
            Indicates whether the complex unit represents an imaginary unit.
            Default is `False`.

        Returns
        -------
        ComplexUnit
            An instance of `ComplexUnit` constructed based on the provided properties.
        """
        osign = 0 if sign == Sign.PLUS else 2
        oimag = 1 if is_imag else 0
        return ComplexUnit(osign + oimag)

    @property
    def sign(self) -> Sign:
        """
        Return the sign of the complex unit.

        Returns
        -------
        Sign
            The sign of the complex unit, indicating its direction
            in the complex plane.
        """
        return Sign.plus_if(self.value < 2)

    @property
    def is_imag(self) -> bool:
        """
        Determine if the complex number is purely imaginary.

        Returns
        -------
        bool
            Returns True if the complex number is in the form of *j* or *-j*;
            otherwise, returns False.
        """
        return bool(self.value % 2)

    def __complex__(self) -> complex:
        """
        Return the unit as a complex number.

        Returns
        -------
        complex
            The complex representation of the unit.
        """
        ret: complex = 1j**self.value
        return ret

    def __str__(self) -> str:
        """
        Return a human-readable representation of the complex unit.

        Returns
        -------
        str
            A string representation of the complex unit.
        """
        result = "1j" if self.is_imag else "1"
        if self.sign == Sign.MINUS:
            result = "-" + result
        return result

    def __mul__(self, other: ComplexUnit | SupportsComplexCtor) -> ComplexUnit:
        """
        Multiply the complex unit by another complex unit or a compatible numeric type.

        Parameters
        ----------
        other : ComplexUnit or SupportsComplexCtor
            The complex unit or numeric type to multiply with.

        Returns
        -------
        ComplexUnit
            A new ComplexUnit instance representing the product of the two complex units.

        Examples
        --------
        >>> cu1 = ComplexUnit(1, 2)
        >>> cu2 = ComplexUnit(3, 4)
        >>> result = cu1 * cu2
        >>> print(result)
        ComplexUnit(3, 10)

        >>> result = cu1 * 2
        >>> print(result)
        ComplexUnit(2, 4)
        """
        if isinstance(other, ComplexUnit):
            return ComplexUnit((self.value + other.value) % 4)
        if isinstance(
            other,
            (SupportsComplex, SupportsFloat, SupportsIndex, complex),
        ) and (other_ := ComplexUnit.try_from(other)):
            return self.__mul__(other_)
        return NotImplemented

    def __rmul__(self, other: SupportsComplexCtor) -> ComplexUnit:
        """
        Perform right multiplication of a complex unit with a number.

        Parameters
        ----------
        other : SupportsComplexCtor
            A number (e.g., int, float) with which to multiply the complex unit.

        Returns
        -------
        ComplexUnit
            A new `ComplexUnit` instance that is the result of the multiplication.

        Notes
        -----
        The multiplication is performed in such a way that the complex unit behaves
        according to the rules of complex arithmetic.
        """
        return self.__mul__(other)

    def __neg__(self) -> ComplexUnit:
        """
        Return the opposite of the complex unit.

        This method implements the unary negation operator for
        instances of the ComplexUnit class, returning a new
        ComplexUnit object representing the negation of the
        current instance.

        Returns
        -------
        ComplexUnit
            A new ComplexUnit object that is the negation of the
            current instance.
        """
        return ComplexUnit((self.value + 2) % 4)


class IXYZ(Enum):
    """
    Representation of the I, X, Y, or Z types.

    This class serves as a symbolic representation for the types I, X, Y, and Z.
    It can be utilized in various contexts where these identifiers are needed.

    Attributes
    ----------
    identifier : str
        A string that holds the identifier, which can be either 'I', 'X', 'Y', or 'Z'.

    Methods
    -------
    get_identifier() -> str:
        Returns the current identifier.

    set_identifier(identifier: str) -> None:
        Sets the identifier to the provided value, must be one of 'I', 'X', 'Y', or 'Z'.

    Example
    -------
    >>> xyz = IXYZ()
    >>> xyz.set_identifier('X')
    >>> print(xyz.get_identifier())
    'X'
    """

    I = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """
        Return the matrix representation.

        Returns
        -------
        npt.NDArray[np.complex128]
            The matrix representation of the object in complex128 format.
        """
        if self == IXYZ.I:
            return Ops.I
        if self == IXYZ.X:
            return Ops.X
        if self == IXYZ.Y:
            return Ops.Y
        if self == IXYZ.Z:
            return Ops.Z
        typing_extensions.assert_never(self)


class Axis(EnumReprMixin, Enum):
    """
    Represents an axis in a 3D space.

    Attributes
    ----------
    axis : str
        The axis can be one of 'X', 'Y', or 'Z'.

    Methods
    -------
    __init__(axis: str)
        Initializes the Axis object with the specified axis.
    """

    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """
        Get the matrix representation of the Axis.

        Returns
        -------
        npt.NDArray[np.complex128]
            The matrix representation as a NumPy ndarray with complex128 data type.
        """
        if self == Axis.X:
            return Ops.X
        if self == Axis.Y:
            return Ops.Y
        if self == Axis.Z:
            return Ops.Z
        typing_extensions.assert_never(self)


class Plane(EnumReprMixin, Enum):
    """
    Represents a geometric plane in a three-dimensional space.

    The plane can be defined as either the XY, YZ, or XZ plane.

    Attributes
    ----------
    type : str
        The type of the plane, which can be either 'XY', 'YZ', or 'XZ'.

    Methods
    -------
    __init__(plane_type: str):
        Initializes the plane with the specified type.

    __str__():
        Returns a string representation of the plane type.
    """

    XY = enum.auto()
    YZ = enum.auto()
    XZ = enum.auto()

    @property
    def axes(self) -> tuple[Axis, Axis]:
        """
        Return the pair of axes that define the plane.

        Returns
        -------
        tuple[Axis, Axis]
            A tuple containing two axes that represent the plane.
        """
        if self == Plane.XY:
            return (Axis.X, Axis.Y)
        if self == Plane.YZ:
            return (Axis.Y, Axis.Z)
        if self == Plane.XZ:
            return (Axis.X, Axis.Z)
        typing_extensions.assert_never(self)

    @property
    def orth(self) -> Axis:
        """
        Returns the axis orthogonal to the plane.

        Returns
        -------
        Axis
            The axis that is orthogonal to the plane represented by this instance.
        """
        if self == Plane.XY:
            return Axis.Z
        if self == Plane.YZ:
            return Axis.X
        if self == Plane.XZ:
            return Axis.Y
        typing_extensions.assert_never(self)

    @property
    def cos(self) -> Axis:
        """
        Return the axis of the plane that conventionally carries the cosine function.

        Returns
        -------
        Axis
            The axis associated with the cosine representation in the plane.
        """
        if self == Plane.XY:
            return Axis.X
        if self == Plane.YZ:
            return Axis.Z  # former convention was Y
        if self == Plane.XZ:
            return Axis.Z  # former convention was X
        typing_extensions.assert_never(self)

    @property
    def sin(self) -> Axis:
        """
        Returns the axis of the plane that conventionally carries the sine.

        This property retrieves the axis associated with the sine function in the context
        of the plane representation.

        Returns
        -------
        Axis
            The axis of the plane corresponding to the sine function.
        """
        if self == Plane.XY:
            return Axis.Y
        if self == Plane.YZ:
            return Axis.Y  # former convention was Z
        if self == Plane.XZ:
            return Axis.X  # former convention was Z
        typing_extensions.assert_never(self)

    @overload
    def polar(self, angle: float) -> tuple[float, float, float]: ...

    @overload
    def polar(self, angle: Expression) -> tuple[Expression, Expression, Expression]: ...

    def polar(
        self, angle: ExpressionOrFloat
    ) -> tuple[float, float, float] | tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat]:
        """
        Convert a polar coordinate to Cartesian coordinates.

        This method returns the Cartesian coordinates (x, y, z) of the point
        with a magnitude of 1 at the specified angle, following the conventional
        orientation for cosine and sine.

        Parameters
        ----------
        angle : ExpressionOrFloat
            The angle in radians at which the point is located.

        Returns
        -------
        tuple[float, float, float] | tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat]
            A tuple containing the Cartesian coordinates (x, y, z) of the
            point on the unit circle at the specified angle. The coordinates may
            be either floats or expressions, depending on the input type.
        """
        pp = (self.cos, self.sin)
        cos, sin = cos_sin(angle)
        if pp == (Axis.X, Axis.Y):
            return (cos, sin, 0)
        if pp == (Axis.Z, Axis.Y):
            return (0, sin, cos)
        if pp == (Axis.Z, Axis.X):
            return (sin, 0, cos)
        raise RuntimeError("Unreachable.")  # pragma: no cover

    @staticmethod
    def from_axes(a: Axis, b: Axis) -> Plane:
        """
        Create a Plane from two given axes.

        Parameters
        ----------
        a : Axis
            The first axis that defines the plane.
        b : Axis
            The second axis that defines the plane.

        Returns
        -------
        Plane
            A Plane object defined by the specified axes.
        """
        ab = {a, b}
        if ab == {Axis.X, Axis.Y}:
            return Plane.XY
        if ab == {Axis.Y, Axis.Z}:
            return Plane.YZ
        if ab == {Axis.X, Axis.Z}:
            return Plane.XZ
        assert a == b
        raise ValueError(f"Cannot make a plane giving the same axis {a} twice.")
