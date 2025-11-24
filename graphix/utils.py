"""
Utilities for various common operations and functions.

This module contains general utility functions that can be used across
different parts of the application. It provides tools for string manipulation,
data processing, and other helpful operations.

Functions:
----------
- function_name_1(arg1, arg2): Brief description of what function does.
- function_name_2(arg1): Brief description of what function does.

Examples:
---------
Examples of how to use the utilities can be added here.
"""

from __future__ import annotations

import inspect
import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, SupportsInt, TypeVar, overload

import numpy as np
import numpy.typing as npt

# Self introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import Self, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_T = TypeVar("_T")


def check_list_elements(l: Iterable[_T], ty: type[_T]) -> None:
    """
    Check that every element of the iterable has the given type.

    Parameters
    ----------
    l : Iterable[_T]
        The iterable to check.
    ty : type[_T]
        The type that each element in the iterable is expected to have.

    Raises
    ------
    TypeError
        If any element in the iterable is not of the specified type.

    Notes
    -----
    This function checks each element in the provided iterable `l`.
    If an element does not match the expected type `ty`, a TypeError
    is raised with a message indicating the incorrect element and its type.
    """
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")


def check_kind(cls: type, scope: dict[str, Any]) -> None:
    """
    Check that the class has a 'kind' attribute.

    Parameters
    ----------
    cls : type
        The class to check for the 'kind' attribute.
    scope : dict[str, Any]
        A dictionary representing the scope in which the class is defined.

    Raises
    ------
    AttributeError
        If the class does not have a 'kind' attribute.
    """
    if not hasattr(cls, "kind"):
        msg = f"{cls.__name__} must have a tag attribute named kind."
        raise TypeError(msg)
    if sys.version_info < (3, 10):
        # MEMO: `inspect.get_annotations` unavailable
        return

    # Type annotation to work around a regression in mypy 1.17, see https://github.com/python/mypy/issues/19458
    ann: Any | None = inspect.get_annotations(cls, eval_str=True, locals=scope).get("kind")
    if ann is None:
        msg = "kind must be annotated."
        raise TypeError(msg)
    if typing.get_origin(ann) is not ClassVar:
        msg = "Tag attribute must be a class variable."
        raise TypeError(msg)
    (ann,) = typing.get_args(ann)
    if typing.get_origin(ann) is not Literal:
        msg = "Tag attribute must be a literal."
        raise TypeError(msg)


def is_integer(value: SupportsInt) -> bool:
    """
    Determine if a given value is an integer.

    Parameters
    ----------
    value : SupportsInt
        The value to be checked.

    Returns
    -------
    bool
        `True` if the value is an integer, `False` otherwise.
    """
    return value == int(value)


G = TypeVar("G", bound=np.generic)


@typing.overload
def lock(data: npt.NDArray[Any]) -> npt.NDArray[np.complex128]: ...


@typing.overload
def lock(data: npt.NDArray[Any], dtype: type[G]) -> npt.NDArray[G]: ...


def lock(data: npt.NDArray[Any], dtype: type = np.complex128) -> npt.NDArray[Any]:
    """
    Create a true immutable view of the given data.

    Parameters
    ----------
    data : numpy.ndarray
        The input array data. It must not have aliasing references; otherwise, users can still
        enable the writable flag on the array, which would violate the immutability.

    dtype : type, optional
        The desired data type for the resulting immutable view. Default is `np.complex128`.

    Returns
    -------
    numpy.ndarray
        An immutable view of the input data array.
    """
    m: npt.NDArray[Any] = data.astype(dtype)
    m.flags.writeable = False
    v = m.view()
    assert not v.flags.writeable
    return v


def iter_empty(it: Iterator[_T]) -> bool:
    """
    Check if an iterable is empty.

    Parameters
    ----------
    it : Iterator[_T]
        An iterator to be checked for emptiness.

    Returns
    -------
    bool
        True if the iterator is empty, False otherwise.

    Notes
    -----
    This function consumes the iterator.
    """
    return all(False for _ in it)


_ValueT = TypeVar("_ValueT")


class Validator(ABC, Generic[_ValueT]):
    """
    Descriptor to validate values.

    This descriptor is designed to enforce specific validation rules on a
    value before it is set to an attribute of a class.

    For more information on descriptors, see the Python documentation at:
    https://docs.python.org/3/howto/descriptor.html#custom-validators
    """

    def __set_name__(self, owner: object, name: str) -> None:
        """
        Set the name of the attribute being managed.

        This method is called when the owning class is being defined,
        allowing the descriptor to set its name and maintain reference
        to the owner class.

        Parameters
        ----------
        owner : object
            The class that owns this descriptor.
        name : str
            The name of the attribute in the owner class.
        """
        self.private_name = "_" + name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> Self:  # access on class
        ...

    @overload
    def __get__(self, obj: object, objtype: type | None = None) -> _ValueT:  # access on instance
        ...

    def __get__(self, obj: object, objtype: object = None) -> _ValueT | Self:
        """
        Retrieve the validated value.

        This method is called to get the validated value from the private
        field of the Validator. It should typically be used in the context
        of a descriptor.

        Parameters
        ----------
        obj : object
            The instance from which the value is being retrieved.
        objtype : object, optional
            The type of the object (class of the instance) if applicable.

        Returns
        -------
        _ValueT | Self
            The validated value from the private field or the Validator instance itself.
        """
        if obj is None:  # access on the class, not an instance
            return self
        result: _ValueT = getattr(obj, self.private_name)
        return result

    def __set__(self, obj: object, value: _ValueT) -> None:
        """
        Validate and set the value in the private field.

        Parameters
        ----------
        obj : object
            The instance of the class where the value will be set.
        value : _ValueT
            The value to be validated and assigned to the private field.

        Raises
        ------
        ValueError
            If the value does not meet the validation criteria.
        """
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value: _ValueT) -> None:
        """
        Validate the assigned value.

        Parameters
        ----------
        value : _ValueT
            The value to be validated.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses
        to provide specific validation logic.
        """


@dataclass
class BoundedFloat(Validator[float]):
    """
    Descriptor to validate floating-point numbers within given bounds.

    This class is used to ensure that a floating-point value falls within a specified
    minimum and maximum range. It raises a ValueError if the assigned value is not
    within the defined bounds.

    Parameters
    ----------
    min_value : float
        The minimum allowable value for the float.
    max_value : float
        The maximum allowable value for the float.

    Raises
    ------
    ValueError
        If the assigned value is less than min_value or greater than max_value.

    Examples
    --------
    >>> class MyClass:
    ...     my_float = BoundedFloat(0.0, 10.0)
    ...
    >>> obj = MyClass()
    >>> obj.my_float = 5.0  # Valid
    >>> obj.my_float = -1.0  # Raises ValueError
    >>> obj.my_float = 11.0  # Raises ValueError
    """

    minvalue: float | None = None
    maxvalue: float | None = None

    @override
    def validate(self, value: float) -> None:
        """
        Validate the assigned value.

        Parameters
        ----------
        value : float
            The value to be validated.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the value is outside the allowed bounds.

        Notes
        -----
        This method checks if the provided value falls within the defined bounds
        of the BoundedFloat instance. If the value is valid, the method completes
        without raising an error; otherwise, it raises a ValueError.
        """
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Expected {value!r} to be at least {self.minvalue!r}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(f"Expected {value!r} to be no more than {self.maxvalue!r}")


class Probability(BoundedFloat):
    """
    Descriptor for probability value, constrained between 0 and 1.

    This class ensures that any value assigned to it is a valid
    probability, meaning it must be within the range [0, 1]. If
    an attempted assignment is outside this range, a ValueError
    will be raised.

    Parameters
    ----------
    value : float
        The initial value of the probability. Must be between 0 and 1.

    Methods
    -------
    __set__(instance, value)
        Sets the value of the probability, ensuring it is within the valid range.

    __get__(instance, owner)
        Gets the value of the probability.

    __delete__(instance)
        Deletes the probability value from the instance.

    Raises
    ------
    ValueError
        If the assigned value is not between 0 and 1.
    """

    def __init__(self) -> None:
        super().__init__(minvalue=0, maxvalue=1)
