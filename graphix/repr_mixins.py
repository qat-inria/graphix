"""
Mixins for evaluation-friendly `repr` for dataclasses and Enum members.

This module provides mixin classes that enhance the representation of
dataclass instances and Enum members, making their string representation
more suitable for evaluation and debugging purposes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # these live only in the stub package, not at runtime
    from _typeshed import DataclassInstance


class DataclassReprMixin:
    """
    Mixin for a concise, eval-friendly `repr` of dataclasses.

    Compared to the default dataclass `repr`, this mixin:
        - Omits class variables (as `dataclasses.fields` only returns actual fields).
        - Omits fields whose values equal their defaults.
        - Displays field names only when preceding fields have been omitted, ensuring positional listings when possible.

    To use this mixin, apply `@dataclass(repr=False)` on the target class.
    """

    def __repr__(self: DataclassInstance) -> str:
        """
        Return a string representation of the dataclass instance.

        The representation string includes the name of the dataclass and its fields,
        allowing for easy identification and debugging of the instance.

        Returns
        -------
        str
            A string representation of the dataclass instance.
        """
        cls_name = type(self).__name__
        arguments = []
        saw_omitted = False
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.default is not MISSING or field.default_factory is not MISSING:
                default = field.default_factory() if field.default_factory is not MISSING else field.default
                if value == default:
                    saw_omitted = True
                    continue
            custom_repr = field.metadata.get("repr")
            value_str = custom_repr(value) if custom_repr else repr(value)
            if saw_omitted:
                arguments.append(f"{field.name}={value_str}")
            else:
                arguments.append(value_str)
        arguments_str = ", ".join(arguments)
        return f"{cls_name}({arguments_str})"


class EnumReprMixin:
    """
    Mixin to provide a concise, eval-friendly repr for Enum members.

    Compared to the default representation `<ClassName.MEMBER_NAME: value>`, this mixin
    alters the `__repr__` method to return `ClassName.MEMBER_NAME`. This representation can
    be evaluated in Python (assuming the enum class is in scope) to retrieve the same Enum
    member.
    """

    def __repr__(self) -> str:
        """
        Return a string representation of an Enum member.

        Returns
        -------
        str
            A string in the format `ClassName.MEMBER_NAME`, where `ClassName` is the name of the class
            and `MEMBER_NAME` is the name of the Enum member.
        """
        # Equivalently (as of Python 3.12), `str(value)` also produces
        # "ClassName.MEMBER_NAME", but we build it explicitly here for
        # clarity.
        if not isinstance(self, Enum):
            msg = "EnumMixin can only be used with Enum classes."
            raise TypeError(msg)
        return f"{self.__class__.__name__}.{self.name}"
