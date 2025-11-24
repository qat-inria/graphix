"""
Parameter class.

The Parameter object acts as a placeholder for measurement angles and
allows manipulation of the measurement pattern without specific
value assignment.
"""

from __future__ import annotations

import cmath
import math
import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Mapping


class Expression(ABC):
    """
    Represents a mathematical expression with parameters.

    Attributes
    ----------
    parameters : dict
        A dictionary containing parameter names and their corresponding values.

    Methods
    -------
    evaluate():
        Evaluates the expression using the current parameter values.
    """

    @abstractmethod
    def __mul__(self, other: object) -> ExpressionOrFloat:
        """
        Perform multiplication with another object.

        This method implements the multiplication operator (*), allowing for
        the expression to be multiplied by another object.

        Parameters
        ----------
        other : object
            The object to be multiplied with this expression.

        Returns
        -------
        ExpressionOrFloat
            The product of this expression and the other object.
        """

    @abstractmethod
    def __rmul__(self, other: object) -> ExpressionOrFloat:
        """
        Return the product of `other` with this expression.

        This special method is called to implement the multiplication operator (*)
        when the left operand does not support multiplication with this type.
        Typically, `other` can be a number.

        Parameters
        ----------
        other : object
            The left operand to be multiplied with the expression.

        Returns
        -------
        ExpressionOrFloat
            The product of `other` and this expression.
        """

    @abstractmethod
    def __add__(self, other: object) -> ExpressionOrFloat:
        """
        Add two expressions or an expression and a scalar.

        Parameters
        ----------
        other : object
            The object to be added to this expression. This could be another
            expression or a float.

        Returns
        -------
        ExpressionOrFloat
            The result of the addition, which may be an expression or a float
            depending on the type of `other`.

        Notes
        -----
        This method implements the addition operator (+) by defining
        the behavior of the `+` operator for instances of this class.
        """

    @abstractmethod
    def __radd__(self, other: object) -> ExpressionOrFloat:
        """
        Return the sum of `other` with this expression.

        This special method is called to implement the addition operator (+)
        when the left operand does not support addition with this type.
        Typically, `other` can be a number.

        Parameters
        ----------
        other : object
            The object to be added to this expression.

        Returns
        -------
        ExpressionOrFloat
            The result of adding `other` to this expression.
        """

    @abstractmethod
    def __sub__(self, other: object) -> ExpressionOrFloat:
        """
        Subtract another object from this expression.

        This special method is called to implement the subtraction operator (-).

        Parameters
        ----------
        other : object
            The object to subtract from this expression.

        Returns
        -------
        ExpressionOrFloat
            The result of the subtraction, which may be an Expression or a float.
        """

    @abstractmethod
    def __rsub__(self, other: object) -> ExpressionOrFloat:
        """
        Return the difference of `other` with this expression.

        This special method is called to implement the subtraction operator (-)
        when the left operand does not support subtraction with this type.
        Typically, `other` can be a number.

        Parameters
        ----------
        other : object
            The value to subtract from this expression.

        Returns
        -------
        ExpressionOrFloat
            The result of the subtraction.
        """

    @abstractmethod
    def __neg__(self) -> ExpressionOrFloat:
        """
        Return the negation of this expression.

        This method implements the unary negation operator (-) for the expression.

        Returns
        -------
        ExpressionOrFloat
            The negated expression.
        """

    @abstractmethod
    def __truediv__(self, other: object) -> ExpressionOrFloat:
        """
        Return the quotient of this expression with another object.

        This special method is called to implement the division operator (/).

        Parameters
        ----------
        other : object
            The object to divide this expression by. This can be another
            `Expression` or any object that is compatible with division.

        Returns
        -------
        ExpressionOrFloat
            The result of the division, which can be an `Expression` or a
            floating-point number, depending on the implementation and the type
            of `other`.
        """

    @abstractmethod
    def subs(self, variable: Parameter, value: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """
        Substitute occurrences of a variable in the expression.

        Parameters
        ----------
        variable : Parameter
            The variable to be substituted in the expression.
        value : ExpressionOrSupportsFloat
            The value that will replace occurrences of the variable.
            This can be an expression or a float-like value.

        Returns
        -------
        ExpressionOrComplex
            A new expression with every occurrence of `variable` replaced with `value`.

        Notes
        -----
        This method is intended to be implemented by subclasses.
        """

    @abstractmethod
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """
        Replace occurrences in the expression.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A mapping where keys are parameters to be replaced in the expression,
            and values are the corresponding expressions or values to use as replacements.

        Returns
        -------
        ExpressionOrComplex
            A new expression resulting from replacing every occurrence of keys
            from `assignment` with their corresponding values. The substitutions
            are performed in parallel, meaning that once an occurrence has been
            replaced by a value, this value is not subject to any further
            replacement, even if another occurrence of a key appears in this value.
        """


class ExpressionWithTrigonometry(Expression, ABC):
    """
    Expression that supports trigonometric functions.

    This class allows for the evaluation and manipulation of mathematical
    expressions that include trigonometric functions such as sine, cosine,
    and tangent.

    Attributes
    ----------
    expression : str
        A string representation of the mathematical expression.

    Methods
    -------
    evaluate(x):
        Evaluates the expression at a given point x.

    derivative():
        Computes the derivative of the expression with respect to x.

    simplify():
        Simplifies the expression to its most reduced form.

    trigonometric_identity():
        Returns a trigonometric identity if applicable to the expression.
    """

    @abstractmethod
    def cos(self) -> ExpressionWithTrigonometry:
        """
        Return the cosine of the expression.

        Returns
        -------
        ExpressionWithTrigonometry
            The cosine of the current expression.
        """

    @abstractmethod
    def sin(self) -> ExpressionWithTrigonometry:
        """
        Compute the sine of the expression.

        Returns
        -------
        ExpressionWithTrigonometry
            The sine of the current expression.
        """

    @abstractmethod
    def exp(self) -> ExpressionWithTrigonometry:
        """
        Compute the exponential of the expression.

        Returns
        -------
        ExpressionWithTrigonometry
            The exponential of the current expression instance.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses of
        `ExpressionWithTrigonometry`.
        """


class Parameter(Expression):
    """
    Abstract class for a substitutable parameter.

    This class serves as a base for creating different parameter types
    that can be substituted in various contexts. It provides a common
    interface for all subclasses.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """


class PlaceholderOperationError(ValueError):
    """
    Exception raised for unsupported operations on a placeholder.

    This error is raised when an operation is attempted on a placeholder
    that does not support that operation.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """

    def __init__(self) -> None:
        """
        Initialize a PlaceholderOperationError.

        This error is raised when a placeholder operation is encountered in the
        context where an operation is expected but not defined.

        Parameters
        ----------
        None
        """
        super().__init__(
            "Placeholder angles do not support any form of computation before substitution except affine operation. You may use `subs` with an actual value before the computation."
        )


@dataclass
class AffineExpression(Expression):
    """
    AffineExpression class.

    An affine expression is of the form *a*x + b* where *a* and *b* are numbers and *x* is a parameter.

    Attributes
    ----------
    a : float
        Coefficient of the parameter x.
    b : float
        Constant term in the expression.
    x : float
        The parameter in the affine expression.

    Methods
    -------
    evaluate(x)
        Evaluates the affine expression at a given value of x.
    """

    a: float
    x: Parameter
    b: float

    def offset(self, d: float) -> AffineExpression:
        """
        Add a constant offset to the affine expression.

        Parameters
        ----------
        d : float
            The value to be added to the expression.

        Returns
        -------
        AffineExpression
            A new AffineExpression instance with the offset applied.
        """
        return AffineExpression(a=self.a, x=self.x, b=self.b + d)

    def _scale_non_null(self, k: float) -> AffineExpression:
        """
        Scale the current affine expression by a non-zero factor.

        Parameters
        ----------
        k : float
            The scaling factor, which must be non-zero.

        Returns
        -------
        AffineExpression
            A new affine expression that is the result of scaling the current expression by `k`.
        """
        return AffineExpression(a=k * self.a, x=self.x, b=k * self.b)

    def scale(self, k: float) -> ExpressionOrFloat:
        """
        Scale the expression by a given factor.

        Parameters
        ----------
        k : float
            The factor by which to scale the expression.

        Returns
        -------
        ExpressionOrFloat
            The scaled expression or a float value, depending on the context of the expression.
        """
        if k == 0:
            return 0
        return self._scale_non_null(k)

    def __mul__(self, other: object) -> ExpressionOrFloat:
        """
        Multiply the current affine expression by another object.

        This method allows for the multiplication of an AffineExpression instance
        with another expression or a numerical value, returning the resulting
        expression or float.

        Parameters
        ----------
        other : object
            The object to multiply with. This can be another affine expression,
            a numerical value, or any object that supports multiplication with
            an affine expression.

        Returns
        -------
        ExpressionOrFloat
            The result of the multiplication, which can be either an expression or
            a float, depending on the type of `other`.

        Notes
        -----
        Refer to the documentation in the parent class for more detailed behavior
        and any additional context related to this operation.

        See Also
        --------
        __add__ : Method for adding another expression or value.
        __sub__ : Method for subtracting another expression or value.
        """
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: object) -> ExpressionOrFloat:
        """
        Perform right multiplication of an affine expression with another object.

        Parameters
        ----------
        other : object
            The object to multiply with this affine expression. This can be a scalar or another expression.

        Returns
        -------
        ExpressionOrFloat
            The result of the multiplication, which may be an affine expression or a float, depending on the type of the `other` object.

        Notes
        -----
        Refer to the documentation in the parent class for further details on the multiplication behavior.
        """
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __add__(self, other: object) -> ExpressionOrFloat:
        """
        Add another expression or a float to this affine expression.

        Parameters
        ----------
        other : object
            The expression or float to be added to this affine expression.
            If `other` is an instance of AffineExpression or another compatible
            expression type, the addition will be performed accordingly.

        Returns
        -------
        ExpressionOrFloat
            The result of the addition, which can be an instance of
            Expression or a float, depending on the type of `other`.

        Notes
        -----
        Refer to the documentation of the parent class for additional details
        regarding operation behaviors and constraints.

        Examples
        --------
        >>> expr1 = AffineExpression(...)
        >>> expr2 = AffineExpression(...)
        >>> result = expr1 + expr2
        >>> result = expr1 + 5.0

        See Also
        --------
        AffineExpression.__sub__: Subtracts another expression or float from
        this affine expression.
        """
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        if isinstance(other, AffineExpression):
            if other.x != self.x:
                raise PlaceholderOperationError
            a = self.a + other.a
            if a == 0:
                return 0
            return AffineExpression(a=a, x=self.x, b=self.b + other.b)
        return NotImplemented

    def __radd__(self, other: object) -> ExpressionOrFloat:
        """
        Compute the right addition of the AffineExpression with another object.

        This method is called when the addition operation is performed with
        the AffineExpression instance on the right. It delegates the addition
        operation to the corresponding implementation within the class.

        Parameters
        ----------
        other : object
            The object to be added to the AffineExpression. This can be
            an instance of Expression, numeric type or any other compatible
            type that supports addition.

        Returns
        -------
        ExpressionOrFloat
            The result of adding the AffineExpression to the other object.
            This can either be an Expression instance or a float,
            depending on the implementation and types involved in the addition.

        Notes
        -----
        Refer to the documentation in the parent class for detailed
        information about the addition behavior and any specific
        constraints associated with the other operand.
        """
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        return NotImplemented

    def __sub__(self, other: object) -> ExpressionOrFloat:
        """
        Subtracts another object from the current AffineExpression instance.

        Parameters
        ----------
        other : object
            The object to be subtracted from the current instance. This can be
            another AffineExpression or a compatible numerical type.

        Returns
        -------
        ExpressionOrFloat
            The result of the subtraction operation, which could either be
            another AffineExpression or a float, depending on the type of
            the 'other' parameter.

        Notes
        -----
        Refer to the documentation in the parent class for additional
        details on the subtraction behavior and compatibility with different
        types.
        """
        if isinstance(other, AffineExpression):
            return self + -other
        if isinstance(other, SupportsFloat):
            return self + -float(other)
        return NotImplemented

    def __rsub__(self, other: object) -> ExpressionOrFloat:
        """
        Perform right subtraction of an AffineExpression from another object.

        This method allows for the subtraction operation to be
        performed with the AffineExpression class on the right side
        of the operation. It is called when the object on the left
        of the subtraction does not have a corresponding
        `__sub__` method to handle the operation.

        Parameters
        ----------
        other : object
            The object from which the AffineExpression will be subtracted.

        Returns
        -------
        ExpressionOrFloat
            The result of the right subtraction, which can be either an
            Expression or a float, depending on the types involved.

        Notes
        -----
        Refer to the documentation in the parent class for more details
        regarding the behavior and properties of the AffineExpression class.
        """
        if isinstance(other, SupportsFloat):
            return self._scale_non_null(-1).offset(float(other))
        return NotImplemented

    def __neg__(self) -> ExpressionOrFloat:
        """
        Return the negation of the AffineExpression.

        This method overrides the unary negation operator (-) for the AffineExpression class.
        It effectively creates a new AffineExpression that represents the negation of the current instance.

        Returns
        -------
        ExpressionOrFloat
            A new AffineExpression object that is the negation of the original instance,
            or a scalar float if appropriate.

        Notes
        -----
        Refer to the documentation of the parent class for further details on the behavior
        and properties of the AffineExpression.
        """
        return self._scale_non_null(-1)

    def __truediv__(self, other: object) -> ExpressionOrFloat:
        """
        Defines the behavior of the division operator for AffineExpression instances.

        Parameters
        ----------
        other : object
            The object to divide the AffineExpression by. This can be another
            AffineExpression or a numeric value.

        Returns
        -------
        ExpressionOrFloat
            The result of the division. If `other` is a numeric type,
            the result will be a numeric value. If `other` is an AffineExpression,
            the result will be an instance of Expression.

        Notes
        -----
        This method inherits behavior from the parent class. For specific details
        on the division behavior and handling of different types, refer to the
        documentation of the parent class.
        """
        if isinstance(other, SupportsFloat):
            return self.scale(1 / float(other))
        return NotImplemented

    def __str__(self) -> str:
        """
        Return a string representation of the affine expression.

        This method provides a human-readable format of the affine expression, which can
        include constants, coefficients, and variables involved in the expression,
        formatted appropriately.
        """
        return f"{self.a} * {self.x} + {self.b}"

    def __eq__(self, other: object) -> bool:
        """
        Check if two expressions are equal.

        Parameters
        ----------
        other : object
            The other expression to compare with.

        Returns
        -------
        bool
            True if the expressions are equal, False otherwise.
        """
        if isinstance(other, AffineExpression):
            return self.a == other.a and self.x == other.x and self.b == other.b
        return False

    def evaluate(self, value: ExpressionOrSupportsFloat) -> ExpressionOrFloat:
        """
        Evaluate the expression at a given value.

        Parameters
        ----------
        value : ExpressionOrSupportsFloat
            The input value at which to evaluate the expression.

        Returns
        -------
        ExpressionOrFloat
            The result of evaluating the expression at the specified value.
        """
        if isinstance(value, SupportsFloat):
            return self.a * float(value) + self.b
        return self.a * value + self.b

    def subs(self, variable: Parameter, value: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """
        Substitute a given variable with a specific value in the affine expression.

        Parameters
        ----------
        variable : Parameter
            The variable to be substituted in the expression.
        value : ExpressionOrSupportsFloat
            The value that will replace the variable in the expression.
            This can be a numerical value or another expression that supports floating-point operations.

        Returns
        -------
        ExpressionOrComplex
            A new expression with the specified variable replaced by the given value.

        Notes
        -----
        This method overrides the substitution behavior defined in the parent class.
        Please refer to the parent class documentation for more details on the substitution mechanism.
        """
        if variable == self.x:
            return self.evaluate(value)
        return self

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """
        Replace variables in the expression with corresponding values from an assignment.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A mapping of parameters to their corresponding expressions or float values.
            The keys are the parameters to be replaced, and the values are the expressions
            or numeric values that will replace them in the current expression.

        Returns
        -------
        ExpressionOrComplex
            A new expression with the specified variables replaced by their assigned values.
            The result may be of type Expression or a complex number depending on the
            replacements made.

        Notes
        -----
        Refer to the documentation in the parent class for additional context.
        """
        value = assignment.get(self.x)
        # `value` can be 0, so checking with `is not None` is mandatory here.
        if value is not None:
            return self.evaluate(value)
        return self


class Placeholder(AffineExpression, Parameter):
    """
    Placeholder for measurement angles.

    This class serves as a placeholder for measurement angles that may appear
    in affine expressions. Placeholders and affine expressions can be used as
    angles in rotation gates of the :class:`Circuit` class or for the measurement
    angle of measurement commands. Pattern optimizations, such as
    standardization, signal shifting, and Pauli preprocessing, can be applied
    to patterns that include placeholders.

    It is important to note that these placeholders and affine expressions do
    not support arbitrary computation and are not suitable for simulation.
    You may use :func:`Circuit.subs` or :func:`Pattern.subs` with an actual
    value prior to computation.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a new instance of the :class:`Placeholder` object.

        Parameters
        ----------
        name : str
            The name of the parameter, used for binding values.
        """
        self.__name = name
        super().__init__(a=1, x=self, b=0)

    @property
    def name(self) -> str:
        "str: Return the name of the placeholder."
        return self.__name

    def __repr__(self) -> str:
        """
        Return a string representation of the Placeholder instance.

        Returns
        -------
        str
            A string that represents the Placeholder instance.
        """
        return f"Placeholder({self.__name!r})"

    def __str__(self) -> str:
        """
        Return a string representation of the placeholder.

        Returns
        -------
        str
            The name of the placeholder.
        """
        return self.__name

    def __eq__(self, other: object) -> bool:
        """
        Compare two Placeholder instances for equality.

        Parameters
        ----------
        other : object
            The object to compare with the current instance.

        Returns
        -------
        bool
            True if the two Placeholder instances are identical, False otherwise.
        """
        if isinstance(other, Parameter):
            return self is other
        return super().__eq__(other)

    def __hash__(self) -> int:
        """
        Return a hash value for the placeholder.

        Returns
        -------
        int
            An integer representing the hash value of the placeholder.
        """
        return id(self)


if sys.version_info >= (3, 10):
    ExpressionOrFloat = Expression | float

    ExpressionOrComplex = Expression | complex

    ExpressionOrSupportsFloat = Expression | SupportsFloat

    ExpressionOrSupportsComplex = Expression | SupportsComplex
else:
    ExpressionOrFloat = typing.Union[Expression, float]

    ExpressionOrComplex = typing.Union[Expression, complex]

    ExpressionOrSupportsFloat = typing.Union[Expression, SupportsFloat]

    ExpressionOrSupportsComplex = typing.Union[Expression, SupportsComplex]


T = TypeVar("T")


def check_expression_or_complex(value: object) -> ExpressionOrComplex:
    """
    Check if the given object is of type ExpressionOrComplex.

    Parameters
    ----------
    value : object
        The object to be checked.

    Returns
    -------
    ExpressionOrComplex
        The input object if it is of type ExpressionOrComplex.

    Raises
    ------
    TypeError
        If the input object is not of type ExpressionOrComplex.
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, SupportsComplex):
        return complex(value)
    msg = f"ExpressionOrComplex expected, but {type(value)} found."
    raise TypeError(msg)


def check_expression_or_float(value: object) -> ExpressionOrFloat:
    """
    Check if the given object is of type ExpressionOrFloat.

    Parameters
    ----------
    value : object
        The object to be checked.

    Returns
    -------
    ExpressionOrFloat
        The input object if it is of type ExpressionOrFloat.

    Raises
    ------
    TypeError
        If the input object is not of type ExpressionOrFloat.

    Notes
    -----
    This function is useful for validating input types in contexts where
    either an expression or a float is expected.
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, SupportsFloat):
        return float(value)
    msg = f"ExpressionOrFloat expected, but {type(value)} found."
    raise TypeError(msg)


@overload
def subs(value: ExpressionOrFloat, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> ExpressionOrFloat: ...


@overload
def subs(value: T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> T | complex: ...


# The return type could be `T | complex` since `subs` returns `Expression` only
# if `T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def subs(value: T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> T | Expression | complex:
    """
    Substitute in `value`.

    Parameters
    ----------
    value : T
        The value in which the substitution will be applied. It can be an instance
        of :class:`Expression` or another type that does not implement the `subs`
        method.

    variable : Parameter
        The variable to be substituted in the `value`.

    substitute : ExpressionOrSupportsFloat
        The value that will replace the variable in the `value`.

    Returns
    -------
    T | Expression | complex
        The modified `value` after applying the substitution. If `value` is an
        instance of :class:`Expression`, the method `value.subs(variable, substitute)`
        is called. The result is coerced to a complex or a float if it is a number.
        If `value` does not implement `subs`, the original `value` is returned unchanged.

    Notes
    -----
    This function is used to apply substitution to collections where some elements
    are instances of `Expression` and others are plain numbers.
    """
    if not isinstance(value, Expression):
        return value
    new_value = value.subs(variable, substitute)
    # On Python<=3.10, complex is not a subtype of SupportsComplex
    if isinstance(new_value, (complex, SupportsComplex)):
        c = complex(new_value)
        if c.imag == 0.0:
            return c.real
        return c
    return new_value


@overload
def xreplace(
    value: ExpressionOrFloat, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
) -> ExpressionOrFloat: ...


@overload
def xreplace(value: T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> T | Expression | complex: ...


# The return type could be `T | Expression | complex` since `subs` returns `Expression` only
# if `T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def xreplace(value: T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> T | Expression | complex:
    """
    Substitute in parallel in `value`.

    Parameters
    ----------
    value : T
        The input value which may be an instance of :class:`Expression` or other types.

    assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
        A mapping of parameters to expressions or values for substitution.

    Returns
    -------
    T | Expression | complex
        The result after substitution. If `value` is an instance of :class:`Expression`,
        the method `value.xreplace(assignment)` is called, and the result is coerced into
        a complex number if it returns a numeric value. If `value` does not implement
        `xreplace`, it is returned unchanged.

    Notes
    -----
    This function is used to apply parallel substitutions to collections where some
    elements are of type `Expression` and others are plain numbers.
    """
    if not isinstance(value, Expression):
        return value
    new_value = value.xreplace(assignment)
    # On Python<=3.10, complex is not a subtype of SupportsComplex
    if isinstance(new_value, (complex, SupportsComplex)):
        c = complex(new_value)
        if c.imag == 0.0:
            return c.real
        return c
    return new_value


def cos_sin(angle: ExpressionOrFloat) -> tuple[ExpressionOrFloat, ExpressionOrFloat]:
    """
    Calculate the cosine and sine of a given angle.

    Parameters
    ----------
    angle : ExpressionOrFloat
        The angle in radians for which to compute the cosine and sine.
        This can be a float or an expression.

    Returns
    -------
    tuple[ExpressionOrFloat, ExpressionOrFloat]
        A tuple containing the cosine and sine of the angle, in that order.

    Notes
    -----
    The function uses the mathematical definitions of cosine and sine
    to compute the values and can handle both scalar and symbolic inputs.
    """
    if isinstance(angle, Expression):
        if isinstance(angle, ExpressionWithTrigonometry):
            cos: ExpressionOrFloat = angle.cos()
            sin: ExpressionOrFloat = angle.sin()
        else:
            raise PlaceholderOperationError
    else:
        cos = math.cos(angle)
        sin = math.sin(angle)
    return cos, sin


def exp(z: ExpressionOrComplex) -> ExpressionOrComplex:
    """
    Compute the exponential of a number or an expression.

    Parameters
    ----------
    z : ExpressionOrComplex
        A number or an expression to which the exponential function will be applied.

    Returns
    -------
    ExpressionOrComplex
        The exponential of the input value.

    Notes
    -----
    This function computes `e^z`, where `e` is Euler's number (approximately 2.71828).
    The input can either be a numerical value or an expression that evaluates to a number.

    Examples
    --------
    >>> exp(1)
    2.718281828459045
    >>> exp(Expression('x'))
    Expression('e^x')
    """
    if isinstance(z, Expression):
        if isinstance(z, ExpressionWithTrigonometry):
            return z.exp()
        raise PlaceholderOperationError
    return cmath.exp(z)
