from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.random_objects as randobj
from graphix.channels import (
    KrausChannel,
    KrausData,
    dephasing_channel,
    depolarising_channel,
    two_qubit_depolarising_channel,
    two_qubit_depolarising_tensor_channel,
)
from graphix.ops import Ops

if TYPE_CHECKING:
    from numpy.random import Generator


class TestChannel:
    """
    Tests for the Channel class.

    This class contains unit tests to verify the functionality of the
    Channel class. Each test method prefixed with 'test_' will be run
    automatically by the testing framework.

    Methods
    -------
    test_initial_state()
        Tests the initial state of a Channel instance.

    test_send_message()
        Tests the functionality of sending a message through the Channel.

    test_receive_message()
        Tests the functionality of receiving a message from the Channel.

    test_close_channel()
        Tests the behavior of closing the Channel.

    test_channel_state_after_close()
        Tests that the Channel behaves as expected after it has been closed.
    """

    def test_init_with_data_success(self, fx_rng: Generator) -> None:
        """
        Test the successful initialization of the Channel class with data.

        This method verifies that the Channel class can be correctly initialized
        when provided with valid data. It ensures that all attributes are set
        appropriately and that the object behaves as expected.

        Parameters
        ----------
        fx_rng : Generator
            A random number generator used to produce test data for the
            initialization.

        Notes
        -----
        This test case is part of the unit tests for the Channel class and
        should be run in the appropriate test environment.
        """
        prob = fx_rng.uniform()
        mychannel = KrausChannel(
            [
                KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
            ],
        )
        assert isinstance(mychannel.nqubit, int)
        assert mychannel.nqubit == 1
        assert len(mychannel) == 2

    def test_init_with_data_fail(self, fx_rng: Generator) -> None:
        """
        Test for unsuccessful initialization of the Channel class.

        This test verifies that the Channel class does not initialize successfully
        when provided with invalid data. It checks for proper exception handling
        and ensures that the object is not created when expected to fail.

        Parameters
        ----------
        fx_rng : Generator
            A random number generator used for creating test data.

        Raises
        ------
        Exception
            If the initialization is successful when it should fail.
        """
        prob = fx_rng.uniform()

        # empty data
        with pytest.raises(ValueError):
            _ = KrausChannel([])

        # incorrect operator dimension
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), np.array([1.0, 0.0])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
                ],
            )

        # incorrect operator dimension: square but not qubits
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), fx_rng.uniform(size=(3, 3))),
                    KrausData(np.sqrt(prob), fx_rng.uniform(size=(3, 3))),
                ],
            )

        # doesn't square to 1. Not normalized. Parameter.
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(2 * np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 0.0], [0.0, -1.0]])),
                ],
            )

        # doesn't square to 1. Not normalized. Operator.
        with pytest.raises(ValueError):
            _ = KrausChannel(
                [
                    KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
                    KrausData(np.sqrt(prob), np.array([[1.0, 3.0], [0.0, -1.0]])),
                ],
            )

        # incorrect rank (number of kraus_operators)
        # use a random channel to do that.
        with pytest.raises(ValueError):
            randobj.rand_channel_kraus(dim=2**2, rank=20, rng=fx_rng)

    def test_dephasing_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), np.array([[1.0, 0.0], [0.0, 1.0]])),
            KrausData(np.sqrt(prob), Ops.Z),
        ]
        dephase_channel = dephasing_channel(prob)
        assert isinstance(dephase_channel, KrausChannel)
        assert dephase_channel.nqubit == 1
        assert len(dephase_channel) == 2

        for i in range(len(dephase_channel)):
            assert np.allclose(dephase_channel[i].coef, data[i].coef)
            assert np.allclose(dephase_channel[i].operator, data[i].operator)

    def test_depolarising_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob / 3.0), Ops.X),
            KrausData(np.sqrt(prob / 3.0), Ops.Y),
            KrausData(np.sqrt(prob / 3.0), Ops.Z),
        ]

        depol_channel = depolarising_channel(prob)

        assert isinstance(depol_channel, KrausChannel)
        assert depol_channel.nqubit == 1
        assert len(depol_channel) == 4

        for i in range(len(depol_channel)):
            assert np.allclose(depol_channel[i].coef, data[i].coef)
            assert np.allclose(depol_channel[i].operator, data[i].operator)

    def test_2_qubit_depolarising_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(np.sqrt(1 - prob), np.kron(Ops.I, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.X)),
        ]

        depol_channel_2_qubit = two_qubit_depolarising_channel(prob)

        assert isinstance(depol_channel_2_qubit, KrausChannel)
        assert depol_channel_2_qubit.nqubit == 2
        assert len(depol_channel_2_qubit) == 16

        for i in range(len(depol_channel_2_qubit)):
            assert np.allclose(depol_channel_2_qubit[i].coef, data[i].coef)
            assert np.allclose(depol_channel_2_qubit[i].operator, data[i].operator)

    def test_2_qubit_depolarising_tensor_channel(self, fx_rng: Generator) -> None:
        prob = fx_rng.uniform()
        data = [
            KrausData(1 - prob, np.kron(Ops.I, Ops.I)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Y)),
        ]

        depol_tensor_channel_2_qubit = two_qubit_depolarising_tensor_channel(prob)

        assert isinstance(depol_tensor_channel_2_qubit, KrausChannel)
        assert depol_tensor_channel_2_qubit.nqubit == 2
        assert len(depol_tensor_channel_2_qubit) == 16

        for i in range(len(depol_tensor_channel_2_qubit)):
            assert np.allclose(depol_tensor_channel_2_qubit[i].coef, data[i].coef)
            assert np.allclose(depol_tensor_channel_2_qubit[i].operator, data[i].operator)
