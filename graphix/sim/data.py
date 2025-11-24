"""
Type `Data` for initializing nodes in backends.

This module declares the type `Data` to support type-checking in the
`base_backend`. The definition of `Data` necessitates importing the
`statevec` and `density_matrix` modules, both of which have dependencies
on `base_backend`. To avoid circular imports, the `data` module is
imported solely within the type-checking block of `base_backend`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from typing_extensions import TypeAlias  # TypeAlias introduced in Python 3.10

from graphix.parameter import ExpressionOrSupportsComplex
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.states import State

if sys.version_info >= (3, 10):
    Data: TypeAlias = (
        State
        | DensityMatrix
        | Statevec
        | Iterable[State]
        | Iterable[ExpressionOrSupportsComplex]
        | Iterable[Iterable[ExpressionOrSupportsComplex]]
    )
else:
    from typing import Union

    Data: TypeAlias = Union[
        State,
        DensityMatrix,
        Statevec,
        Iterable[State],
        Iterable[ExpressionOrSupportsComplex],
        Iterable[Iterable[ExpressionOrSupportsComplex]],
    ]
