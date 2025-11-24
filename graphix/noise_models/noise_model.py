"""
Abstract interface for noise models.

This module defines the :class:`NoiseModel`, which serves as the base class
for :class:`graphix.simulator.PatternSimulator` when executing noisy simulations.
Child classes are expected to implement specific noise processes by overriding
the abstract methods defined within this interface.
"""

from __future__ import annotations

import dataclasses
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

# override introduced in Python 3.12
from typing_extensions import override

from graphix.command import BaseM, Command, CommandKind, Node, _KindChecker

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.random import Generator

    from graphix.channels import KrausChannel
    from graphix.measurements import Outcome


class Noise(ABC):
    """
    Abstract base class for noise.

    This class serves as a blueprint for implementing various types of noise
    generators. Subclasses must provide specific implementations of noise
    generation methods.

    Attributes
    ----------
    None

    Methods
    -------
    generate_noise():
        Generate noise based on the specific implementation in the subclass.
    """

    @property
    @abstractmethod
    def nqubits(self) -> int:
        """
        Get the number of qubits targeted by the noise.

        Returns
        -------
        int
            The number of qubits that the noise affects.
        """

    @abstractmethod
    def to_kraus_channel(self) -> KrausChannel:
        """
        Convert the noise model to a Kraus channel representation.

        Returns
        -------
        KrausChannel
            A Kraus channel that describes the noise behavior.

        Notes
        -----
        This method should be implemented by subclasses to provide the specific
        Kraus operators associated with the noise model.
        """


@dataclass
class ApplyNoise(_KindChecker):
    """
    Apply noise to an input signal.

    This class is used to inject noise into a given signal to simulate
    real-world conditions or to evaluate the robustness of processing
    algorithms.

    Parameters
    ----------
    noise_level : float
        The standard deviation of the noise to be added to the signal.
    noise_type : str, optional
        The type of noise to apply. Can be 'gaussian', 'salt_and_pepper',
        or 'uniform'. Default is 'gaussian'.

    Examples
    --------
    >>> noise_applier = ApplyNoise(noise_level=0.1, noise_type='gaussian')
    >>> noisy_signal = noise_applier.apply(signal)

    Methods
    -------
    apply(signal)
        Applies the configured noise to the provided signal.
    """

    kind: ClassVar[Literal[CommandKind.ApplyNoise]] = dataclasses.field(default=CommandKind.ApplyNoise, init=False)
    noise: Noise
    nodes: list[Node]


if sys.version_info >= (3, 10):
    CommandOrNoise = Command | ApplyNoise
else:
    from typing import Union

    CommandOrNoise = Union[Command, ApplyNoise]


class NoiseModel(ABC):
    """
    Abstract base class for all noise models.

    This class defines the interface for noise modeling and serves
    as a base for all specific implementations of noise models.
    Subclasses should implement the necessary methods to describe
    the inherent noise characteristics of particular systems.

    Parameters
    ----------
    None

    Methods
    -------
    add_noise(data):
        Applies the noise model to the given data.

    remove_noise(data):
        Attempts to reverse the noise applied to the given data.

    noise_level():
        Returns the current noise level of the model.
    """

    @abstractmethod
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to input nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            An iterable of node indices for which noise is to be applied.
        rng : Generator or None, optional
            A random number generator instance. If None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of noise commands corresponding to the input nodes.
        """

    @abstractmethod
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to the command.

        Parameters
        ----------
        cmd : CommandOrNoise
            The command to which noise will be applied.
        rng : Generator, optional
            A random number generator to use for generating noise. If None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise objects that represent the noise to be applied to the input command.
        """

    @abstractmethod
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """
        Return a possibly flipped measurement outcome.

        Parameters
        ----------
        cmd : BaseM
            The measurement command that produced the given outcome.

        result : Outcome
            Ideal measurement result.

        rng : Generator, optional
            Random number generator, if not provided, a default generator will be used.

        Returns
        -------
        Outcome
            Possibly corrupted result.
        """

    def transpile(self, sequence: Iterable[CommandOrNoise], rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Apply noise to a sequence of commands and return the resulting sequence.

        Parameters
        ----------
        sequence : Iterable[CommandOrNoise]
            A sequence of commands or noise to which the noise model will be applied.

        rng : Generator, optional
            A random number generator to use for stochastic noise applications.
            If None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise after the noise model has been applied.
        """
        return [n_cmd for cmd in sequence for n_cmd in self.command(cmd, rng=rng)]


class NoiselessNoiseModel(NoiseModel):
    """
    Noiseless noise model.

    This model represents a noise process that performs no operation on the input data.
    It can be used in scenarios where no noise is desired or as a baseline for comparison
    with other noise models.

    Parameters
    ----------
    None

    Methods
    -------
    apply(input_data):
        Returns the input data unchanged.
    """

    @override
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to input nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            A collection of input node indices for which noise is to be applied.
        rng : Generator | None, optional
            A random number generator for generating noise. If None, a default generator is used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise objects to apply to the specified input nodes.
        """
        return []

    @override
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to the given command.

        Parameters
        ----------
        cmd : CommandOrNoise
            The command to which noise should be applied.
        rng : Generator, optional
            A random number generator for noise generation. If None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands with applied noise based on the input command.
        """
        return [cmd]

    @override
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """
        Assign a wrong measurement result based on the specified command.

        Parameters
        ----------
        cmd : BaseM
            The command that determines the measurement operation.
        result : Outcome
            The actual outcome of the measurement that needs to be confused.
        rng : Generator, optional
            A random number generator for introducing randomness in the confusion process.
            If None, a default random generator will be used.

        Returns
        -------
        Outcome
            The confused measurement result, which may not match the actual outcome based on the noise model implemented.
        """
        return result


@dataclass(frozen=True)
class ComposeNoiseModel(NoiseModel):
    """
    Compose noise models.

    This class allows the combination of multiple noise models into a single
    composite noise model. It facilitates the simulation and analysis of
    complex noise scenarios by enabling users to stack and configure various
    noise models together.

    Attributes
    ----------
    noise_models : list
        A list of noise models to be combined.

    Methods
    -------
    add_model(model)
        Adds a new noise model to the composite noise model.

    remove_model(model)
        Removes a specified noise model from the composite noise model.

    evaluate(inputs)
        Evaluates the combined noise effect on the given inputs.

    clear_models()
        Removes all noise models from the composite noise model.
    """

    models: list[NoiseModel]

    @override
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to input nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            An iterable of node indices to which the noise will be applied.
        rng : Generator | None, optional
            A random number generator to use for noise generation. If None,
            the default random generator is used. Default is None.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise objects that will be applied to the
            specified input nodes.
        """
        return [n_cmd for m in self.models for n_cmd in m.input_nodes(nodes)]

    @override
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to the command.

        Parameters
        ----------
        cmd : CommandOrNoise
            The command to which noise will be applied.
        rng : Generator, optional
            Random number generator for stochastic operations (default is None).

        Returns
        -------
        list[CommandOrNoise]
            A list of commands with noise applied to the original command.
        """
        sequence = [cmd]
        for model in self.models:
            sequence = model.transpile(sequence)
        return sequence

    @override
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """
        Assign a wrong measurement result to the given outcome.

        Parameters
        ----------
        cmd : BaseM
            The command that generated the outcome.
        result : Outcome
            The original outcome to be confused.
        rng : Generator, optional
            A random number generator for sampling. If None, a default random generator will be used.

        Returns
        -------
        Outcome
            The modified outcome with a confounded measurement result.
        """
        for m in self.models:
            result = m.confuse_result(cmd, result)
        return result
