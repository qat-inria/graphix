"""
Provide a default random number generator if `None` is provided.

This module offers functionality to generate random numbers using a default
random number generator. If no specific generator is supplied, the default
is utilized to ensure consistent random number generation.

Parameters
----------
None

Returns
-------
RandomNumberGenerator
    An instance of the default random number generator.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

_rng_local = threading.local()


def ensure_rng(rng: Generator | None = None) -> Generator:
    """
    Ensure a random number generator is returned.

    This function returns a default random number generator if the
    provided generator is None. If a generator is provided, it returns
    that generator.

    Parameters
    ----------
    rng : Generator | None, optional
        A random number generator. If None, a default generator will
        be created and returned.

    Returns
    -------
    Generator
        A random number generator instance.
    """
    if rng is not None:
        return rng
    stored: Generator | None = getattr(_rng_local, "rng", None)
    if stored is not None:
        return stored
    rng = np.random.default_rng()
    # MEMO: Cannot perform type check
    setattr(_rng_local, "rng", rng)  # noqa: B010
    return rng
