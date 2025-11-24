"""
Branch Selector Module.

This module contains branch selectors that determine the computation
branch explored during a simulation, specifically influencing the
choice of measurement outcomes. Branch selection can either be
random (see :class:`RandomBranchSelector`) or deterministic
(see :class:`ConstBranchSelector`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import override

from graphix.measurements import Outcome, outcome
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator


class BranchSelector(ABC):
    """
    Abstract class for branch selectors.

    A branch selector provides the method `measure`, which returns the
    measurement outcome (0 or 1) for a given qubit.
    """

    @abstractmethod
    def measure(self, qubit: int, f_expectation0: Callable[[], float], rng: Generator | None = None) -> Outcome:
        """
        Return the measurement outcome of a specified qubit.

        Parameters
        ----------
        qubit : int
            Index of the qubit to measure.

        f_expectation0 : Callable[[], float]
            A callable that retrieves the expected probability of outcome 0.
            The probability is computed only if this function is called,
            enabling lazy computation and preventing unnecessary computational cost.

        rng : Generator, optional
            Random-number generator for measurements. This generator is used
            only in cases of random branch selection (see :class:`RandomBranchSelector`).
            If `None`, a default random-number generator is used. The default is `None`.
        """


@dataclass
class RandomBranchSelector(BranchSelector):
    """
    Random branch selector.

    Parameters
    ----------
    pr_calc : bool, optional
        Whether to compute the probability distribution before selecting the measurement result.
        If ``False``, measurements yield 0/1 with equal probability (50% each).
        Default is ``True``.
    """

    pr_calc: bool = True

    @override
    def measure(self, qubit: int, f_expectation0: Callable[[], float], rng: Generator | None = None) -> Outcome:
        """
        Measure the outcome of a specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the qubit to measure.
        f_expectation0 : Callable[[], float]
            A callable that computes the expectation value for outcome 0.
        rng : Generator | None, optional
            An optional random number generator for generating random outcomes.
            If None, the default generator will be used.

        Returns
        -------
        Outcome
            The measurement outcome of the specified qubit. The result is determined
            based on the computed probability of outcome 0 if `pr_calc` is True.
            Otherwise, the result is randomly chosen with a 50% chance for either
            outcome.

        Notes
        -----
        If `pr_calc` is False, the measurement outcome is decided randomly, while
        if True, it relies on the computation from `f_expectation0`.
        """
        rng = ensure_rng(rng)
        if self.pr_calc:
            prob_0 = f_expectation0()
            return outcome(rng.random() > prob_0)
        result: Outcome = rng.choice([0, 1])
        return result


_T = TypeVar("_T", bound=Mapping[int, Outcome])


@dataclass
class FixedBranchSelector(BranchSelector, Generic[_T]):
    """
    Branch selector with predefined measurement outcomes.

    The mapping is fixed in `results`. By default, an error is raised if
    a qubit is measured without a predefined outcome. However, another
    branch selector can be specified in `default` to handle such cases.

    Parameters
    ----------
    results : Mapping[int, bool]
        A dictionary mapping qubit indices to their measurement outcomes.
        If a qubit is not present in this mapping, the `default` branch
        selector is used.
    default : BranchSelector | None, optional
        Branch selector to use for qubits not present in `results`.
        If `None`, an error is raised when an unmapped qubit is measured.
        Default is `None`.
    """

    results: _T
    default: BranchSelector | None = None

    @override
    def measure(self, qubit: int, f_expectation0: Callable[[], float], rng: Generator | None = None) -> Outcome:
        """
        Measure the predefined outcome of a specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the qubit to be measured.
        f_expectation0 : Callable[[], float]
            A callable that returns the expectation value of the measurement for the qubit.
        rng : Generator | None, optional
            An optional random number generator to use for stochastic processes. If None, the default generator is used.

        Returns
        -------
        Outcome
            The measurement outcome of the specified qubit. If the qubit is not available in the results,
            the default branch selector is used. If no default is provided, an error is raised.

        Raises
        ------
        ValueError
            If the qubit is not present and no default branch selector is available.
        """
        result = self.results.get(qubit)
        if result is None:
            if self.default is None:
                raise ValueError(f"Unexpected measurement of qubit {qubit}.")
            return self.default.measure(qubit, f_expectation0)
        return result


@dataclass
class ConstBranchSelector(BranchSelector):
    """
    Branch selector with a constant measurement outcome.

    The value `result` is returned for every qubit.

    Parameters
    ----------
    result : Outcome
        The fixed measurement outcome for all qubits.
    """

    result: Outcome

    @override
    def measure(self, qubit: int, f_expectation0: Callable[[], float], rng: Generator | None = None) -> Outcome:
        """
        Return the constant measurement outcome ``result`` for any qubit.

        Parameters
        ----------
        qubit : int
            The index of the qubit to measure.
        f_expectation0 : Callable[[], float]
            A callable that returns the expected value for the measurement.
        rng : Generator, optional
            A random number generator for stochastic processes. If None, a default generator will be used.

        Returns
        -------
        Outcome
            The constant measurement outcome corresponding to the specified qubit.
        """
        return self.result
