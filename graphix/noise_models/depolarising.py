"""
Depolarising noise model.

This module implements the depolarising noise model, which is a common
mathematical representation of noise in quantum systems. The model
describes how a quantum state can become mixed due to interactions
with the environment, resulting in a loss of coherence.

The depolarising noise can be characterized by a single parameter
that quantifies the strength of the noise. It transforms a pure state
into a mixed state, effectively simulating the randomization of
quantum states.

Attributes
----------
None

Functions
---------
- apply_depolarising_noise: Applies depolarising noise to a given quantum state.
- fidelity: Computes the fidelity between two quantum states after noise
  has been applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, depolarising_channel, two_qubit_depolarising_channel
from graphix.command import BaseM, CommandKind
from graphix.measurements import toggle_outcome
from graphix.noise_models.noise_model import ApplyNoise, CommandOrNoise, Noise, NoiseModel
from graphix.rng import ensure_rng
from graphix.utils import Probability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.random import Generator

    from graphix.measurements import Outcome


class DepolarisingNoise(Noise):
    """
    One-qubit depolarising noise.

    This class implements depolarising noise with a given probability.

    Parameters
    ----------
    prob : float
        The probability of applying depolarising noise. Must be in the range [0, 1].

    Attributes
    ----------
    prob : float
        The probability of depolarising noise.
    """

    prob = Probability()

    def __init__(self, prob: float) -> None:
        """
        Initialize one-qubit depolarizing noise.

        Parameters
        ----------
        prob : float
            Probability parameter of the noise, between 0 and 1. The closer the
            value is to 1, the stronger the depolarizing effect.
        """
        self.prob = prob

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """
        Return the number of qubits targeted by the noise element.

        Returns
        -------
        int
            The number of qubits affected by the depolarising noise.
        """
        return 1

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """
        Return the Kraus channel describing the noise element.

        Returns
        -------
        KrausChannel
            The Kraus channel that represents the depolarising noise.
        """
        return depolarising_channel(self.prob)


class TwoQubitDepolarisingNoise(Noise):
    """
    Two-qubit depolarising noise characterized by a probability of application.

    Parameters
    ----------
    prob : float
        The probability of depolarising noise occurring for each qubit.

    Description
    -----------
    This class implements a two-qubit depolarising noise model, which introduces noise into
    the qubits by randomly applying a set of operations with a given probability. The noise
    affects the qubits by transforming the pure states into mixed states, according to the
    depolarising noise process.

    The general form of the depolarising channel for two qubits is given by:

    .. math::
        \rho' = (1 - p) \rho + \frac{p}{15} \sum_{i=0}^{15} E_i \rho E_i^\dagger

    where :math:`\rho` is the density matrix of the two-qubit state, :math:`p` is the
    probability of depolarization, and :math:`E_i` are the corresponding noise operators.

    The noise operations include the identity and the application of Pauli operators.

    Notes
    -----
    Ensure that the probability ``prob`` is within the range [0, 1].
    """

    prob = Probability()

    def __init__(self, prob: float) -> None:
        """
        Initialize two-qubit depolarizing noise.

        Parameters
        ----------
        prob : float
            Probability parameter of the noise, between 0 and 1.

        Raises
        ------
        ValueError
            If prob is not between 0 and 1.
        """
        self.prob = prob

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """
        Return the number of qubits targeted by the noise element.

        Returns
        -------
        int
            The number of qubits affected by the depolarising noise process.
        """
        return 2

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """
        Returns the Kraus channel describing the noise element.

        This method computes and returns the Kraus representation of the depolarizing noise channel
        for a two-qubit system. The resulting channel can be used to model quantum noise in
        quantum information processes.

        Returns
        -------
        KrausChannel
            A KrausChannel object representing the defined two-qubit depolarizing noise.

        Notes
        -----
        The depolarizing channel affects the qubits by introducing noise, described through a
        set of Kraus operators. The mathematical formulation and details about the noise
        parameters can be found in relevant quantum information theory literature.

        Example
        -------
        >>> noise = TwoQubitDepolarisingNoise(...)
        >>> kraus_channel = noise.to_kraus_channel()
        """
        return two_qubit_depolarising_channel(self.prob)


class DepolarisingNoiseModel(NoiseModel):
    """
    Depolarising noise model.

    This model represents the depolarising noise that can affect quantum states,
    causing them to lose their coherence. The noise acts by mixing the quantum state
    with a completely mixed state, with a specified probability.

    Parameters
    ----------
    noise_strength : float
        The strength of the depolarising noise, ranging from 0 to 1. A value of 0
        indicates no noise, while a value of 1 results in maximum noise.

    Attributes
    ----------
    noise_strength : float
        The current strength of the depolarising noise model.

    Methods
    -------
    apply_noise(state):
        Applies the depolarising noise to the given quantum state.
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

    @typing_extensions.override
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to input nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            The list of input node indices to which the noise will be applied.
        rng : Generator | None, optional
            A random number generator for noise application. If None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise operations to be applied to the specified input nodes.
        """
        return [ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[node]) for node in nodes]

    @typing_extensions.override
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """
        Return the noise to apply to the command.

        Parameters
        ----------
        cmd : CommandOrNoise
            The command or noise operation to which the noise will be applied.
        rng : Generator | None, optional
            A random number generator; if None, a default generator will be used.

        Returns
        -------
        list[CommandOrNoise]
            A list of commands or noise operations that represent the noise to be applied.
        """
        if cmd.kind == CommandKind.N:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.E:
            return [
                cmd,
                ApplyNoise(noise=TwoQubitDepolarisingNoise(self.entanglement_error_prob), nodes=list(cmd.nodes)),
            ]
        if cmd.kind == CommandKind.M:
            return [ApplyNoise(noise=DepolarisingNoise(self.measure_channel_prob), nodes=[cmd.node]), cmd]
        if cmd.kind == CommandKind.X:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.x_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.Z:
            return [cmd, ApplyNoise(noise=DepolarisingNoise(self.z_error_prob), nodes=[cmd.node])]
        # Use of `==` here for mypy
        if cmd.kind == CommandKind.C or cmd.kind == CommandKind.T or cmd.kind == CommandKind.ApplyNoise:  # noqa: PLR1714
            return [cmd]
        if cmd.kind == CommandKind.S:
            raise ValueError("Unexpected signal!")
        typing_extensions.assert_never(cmd.kind)

    @typing_extensions.override
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """
        Assign a wrong measurement result based on the provided command.

        Parameters
        ----------
        cmd : BaseM
            The command that is being measured.
        result : Outcome
            The original measurement outcome.
        rng : Generator, optional
            A random number generator to control the randomness of the result assignment.
            If None, a default random number generator will be used.

        Returns
        -------
        Outcome
            The modified measurement outcome, which may be different from the input result.
        """
        if self.rng.uniform() < self.measure_error_prob:
            return toggle_outcome(result)
        return result
