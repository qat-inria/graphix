"""Perceval simulator.

Simulate MBQC with the Perceval open source framework for programming photonic quantum computers.

Ref: https://github.com/Quandela/Perceval.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import perceval as pcvl
from perceval.backends import BackendFactory
from perceval.backends._slos import SLOSBackend
from perceval.components import Catalog, Circuit, Processor, Source  # type: ignore[reportAttributeAccessIssue]
from perceval.simulators import Simulator
from perceval.utils.states import BasicState, StateVector, SVDistribution

from graphix.command import CommandKind
from graphix.fundamentals import Plane
from graphix.sim.base_backend import NodeIndex
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    import numpy.typing as npt
    from numpy.random import Generator

    from graphix import command
    from graphix.clifford import Clifford
    from graphix.measurements import Measurement, Outcome
    from graphix.sim.data import Data
    from graphix.simulator import MeasureMethod


class PercevalState:
    """Perceval state wrapper for Graphix."""

    source: Source
    state: StateVector
    sim: Simulator

    def __init__(self, source: Source, perceval_state: StateVector) -> None:
        self.state = perceval_state
        self.source = source
        # Statevec.__init__(Statevec(nqubit=0))
        self.sim: Simulator = Simulator(SLOSBackend)
        self.sim.set_min_detected_photons_filter(0)
        self.sim.keep_heralds(False)

    @property
    def nqubit(self) -> int:
        """Return the number of qubits of the current state."""
        return int(self.state.m / 2)  # type: ignore[attr-defined]

    def entangle(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = edge[0]
        ctrl = edge[1]
        cz_input_modes = [2 * ctrl, 2 * ctrl + 1, 2 * target, 2 * target + 1]
        # We construct the circuit via the Processor class since this class applies
        # the correct permutation before and after to place CZ at correct modes
        # (while the Circuit class does not and returns an error if the modes are not contiguous)
        ent_proc: Processor = Processor(SLOSBackend, 2 * self.nqubit, noise=None, name="Local processor")  # type: ignore[arg-type]
        catalog = Catalog("perceval.components.core_catalog")
        ent_proc.add(cz_input_modes, catalog["heralded cz"].build_processor())
        ent_circ: Circuit = ent_proc.linear_circuit()
        self.sim.set_circuit(ent_circ)

        # the first 2n modes store the state, the last modes are heralds (1 photon in 2 modes for each CZ gate)
        heralds: dict[int, int] = dict.fromkeys(list(range(2 * self.nqubit, ent_proc.circuit_size)), 1)
        self.sim.set_heralds(heralds)
        herald_state: SVDistribution = self.source.generate_distribution(BasicState([1, 1]))  # type: ignore[arg-type]
        # Here we again explicitely choose not to deal with mixed states
        sampled_herald_state = herald_state.sample(1)[0]
        self.state = self.sim.evolve(self.state * sampled_herald_state)
        self.sim.clear_heralds()

    def evolve(self, op: Circuit) -> None:
        """Evolve the state with a circuit.

        Parameters
        ----------
        op : pcvl.Circuit
            circuit to apply to the state
        """
        self.sim.set_circuit(op)
        self.state = self.sim.evolve(self.state)

    def set_source(self, source: Source) -> None:
        """Set the Perceval source."""
        self.source = source


@dataclass(frozen=True)
class PercevalBackend(PercevalState):
    """Perceval simulator backend for Graphix.

    Ref: https://github.com/Quandela/Perceval

    TODO
    - fix issue with perceval_state (see init)
    - choice of PNR or threshold detectors (currently only threshold is implemented)
    - other state generation strategies: with fusions, with RUS gates (would probably required standardised pattern)
    - add option to keep track and return success probability (adapt add_nodes, entangle_nodes and measure to work with mixed states)
    - add a mode which does not perform the simulation but only constructs the circuit (with the proper feed-forward), to run on QPU

    Args:
        source: pcvl.Source
        perceval_state: pcvl.State
    """

    state: PercevalState = dataclasses.field(
        init=False,
        default_factory=lambda: PercevalState(
            source=pcvl.Source(emission_probability=1, multiphoton_component=0, indistinguishability=1),
            perceval_state=BasicState(),
        ),
    )
    node_index: NodeIndex = dataclasses.field(default_factory=NodeIndex)

    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        """Add new nodes to the backend and initialize them in a specified state.

        Args:
            nodes: A list of node indices to add to the backend.
            data: The state in which to initialize the newly added nodes. The supported forms of state specification depend on the backend implementation.
        """
        zero_mixed_state: SVDistribution = self.state.source.generate_distribution(BasicState([1, 0]))  # type: ignore[attr-defined]
        # path-encoded |0> state

        # Here we explicitely choose not to deal with mixed states by sampling a single input state from the distribution above.
        # This means we cannot compute the fraction of successful runs and may run into post-selection problems (see measure function).
        # This is done because perceval does not (for now) handle measurements on SVDistribution (mixed states).
        # Those cannot be translated to DensityMatrix objects (which can be measured) since DensityMatrix does not handle distinguishable photons (yet).
        # One solution would be to manually implement measurements on SVDistributions.
        init_qubit: StateVector = zero_mixed_state.sample(1)[0]

        # recover amplitudes of input state
        statevector: npt.NDArray[np.complex128] = data.get_statevector()
        alpha: np.complex128 = statevector[0]
        beta: np.complex128 = statevector[1]

        if np.abs(beta) != 0:  # if beta = 0, the input is already |0>, no need to do anything
            # construct unitary matrix taking |0> to the state psi
            gamma: float = np.abs(beta)
            delta: complex = -np.conjugate(alpha) * gamma / np.conjugate(beta)
            matrix: pcvl.MatrixN = pcvl.MatrixN(np.asarray([[alpha, gamma], [beta, delta]]))
            init_circ: pcvl.Circuit = pcvl.Circuit(2)
            init_circ.add(0, pcvl.components.Unitary(U=matrix))
            self.state.sim.set_circuit(init_circ)
            init_qubit = self.state.sim.evolve(init_qubit)

        self.state.state *= init_qubit
        self.node_index.extend(nodes)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        # get optical modes corresponding to edge qubits
        index_0 = self.node_index.index(edge[0])
        index_1 = self.node_index.index(edge[1])
        ctrl = min(index_0, index_1)
        target = max(index_0, index_1)
        self.state.entangle((target, ctrl))

    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Outcome:  # noqa: ARG002
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node: int
        measurement: Measurement
        """
        index: int = self.node_index.index(node)
        meas_circ: pcvl.Circuit = pcvl.Circuit(2 * self.nqubit)
        if measurement.angle is float:
            angle: pcvl.Parameter | float = measurement.angle
        else:
            raise NotImplementedError("Parametrised angles not implemented for PercevalBackend.")
        # YZ and XZ not properly tested, only used XY plane measurements
        if measurement.plane == Plane.XY:
            # rotation around Z axis by -angle
            meas_circ.add(2 * index + 1, pcvl.components.PS(-angle))
            # transformation from X basis to Z basis
            meas_circ.add(2 * index, pcvl.components.BS.H())
        elif measurement.plane == Plane.YZ:
            # rotation around X axis by -angle
            meas_circ.add(2 * index, pcvl.components.BS.H())
            meas_circ.add(2 * index + 1, pcvl.components.PS(-angle))
            meas_circ.add(2 * index, pcvl.components.BS.H())
            # transformation from Y basis to Z basis
            meas_circ.add(2 * index + 1, pcvl.components.PS(-np.pi / 2))
            meas_circ.add(2 * index, pcvl.components.BS.H())
        elif measurement.plane == Plane.XZ:
            # rotation around Y axis by -angle
            meas_circ.add(2 * index + 1, pcvl.components.PS(-np.pi / 2))
            meas_circ.add(2 * index, pcvl.components.BS.H())
            meas_circ.add(2 * index + 1, pcvl.components.PS(-angle))
            meas_circ.add(2 * index, pcvl.components.BS.H())
            meas_circ.add(2 * index + 1, pcvl.components.PS(np.pi / 2))
            # transformation from X basis to Z basis
            meas_circ.add(2 * index, pcvl.components.BS.H())
        # applies operation on measured qubit before performing a Z basis measurement
        self.state.evolve(meas_circ)

        # the measure function returns a dictionary where the keys are the possible measurement outcomes (as pcvl.BasicState s)
        # and the values are the results (the first is the probability of obtaining that outcome, the second is the remaining state, after collapse)
        # in order to sample from these outcomes, we construct a pcvl.BSDistribution
        all_possible_meas_outcomes: dict[pcvl.FockState, tuple[float, pcvl.StateVector]] = self.state.state.measure(
            [2 * index, 2 * index + 1]
        )
        outcome_dist: dict[float, pcvl.StateVector] = {}
        for outcome, res in all_possible_meas_outcomes.items():
            outcome_dist[outcome] = res[0]
        outcomes: pcvl.BSDistribution = pcvl.BSDistribution(outcome_dist)

        # we then post-select the distribution above on having a qubit encoding and sample from it
        # the post-selection may fail (because we don't simulate the full distribution, see comment in add_nodes)
        # need to decide how to catch that and what to do with it
        # one possibility would be to retry the full computation and count the number of retries
        # if we're simulating the full distribution instead, we can at this stage recover the success probability
        ps: pcvl.PostSelect = pcvl.PostSelect("([0] > 0 & [1] == 0) | ([0] == 0 & [1] > 0)")
        ps_outcomes: pcvl.BSDistribution = pcvl.utils.postselect.post_select_distribution(outcomes, ps)[0]
        meas_result: pcvl.BSSamples = ps_outcomes.sample(1)[0]
        result: Literal[0, 1] = meas_result[0] == 0

        # we then set the state to the reduced state that corresponds to the sampled outcome
        self.state.state = all_possible_meas_outcomes[meas_result][1]
        self.node_index.remove(node)
        return result

    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""
        if np.mod(sum(measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
            index: int = self.node_index.index(cmd.node)
            correct_circ: pcvl.Circuit = pcvl.Circuit(2 * self.nqubit)
            if cmd.kind == CommandKind.X:
                correct_circ.add(2 * index, pcvl.components.PERM([1, 0]))
            elif cmd.kind == CommandKind.Z:
                correct_circ.add(2 * index + 1, pcvl.components.PS(np.pi))
            self.state.evolve(correct_circ)

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""
        index: int = self.node_index.index(node)
        # use unitary defining the clifford to initialise the perceval circuit
        clifford_circ = pcvl.Circuit(2 * self.nqubit).add(
            2 * index, pcvl.components.Unitary(U=pcvl.MatrixN(clifford.matrix))
        )
        self.state.evolve(clifford_circ)

    def sort_qubits(self, output_nodes: Iterable[int]) -> None:
        """Sort the qubit order in internal statevector."""
        # not tested, checked code only on classical outputs
        if self.nqubit > 0:
            perm_circ = pcvl.Circuit(2 * self.nqubit)
            for i, ind in enumerate(output_nodes):
                if self.node_index.index(ind) != i:
                    move_from: int = self.node_index.index(ind)
                    self.node_index.swap(i, move_from)

                    low: int = min(i, move_from)
                    high: int = max(i, move_from)
                    perm_circ.add(
                        2 * low,
                        pcvl.components.PERM(
                            [2 * (high - low), 2 * (high - low) + 1, *list(range(2, 2 * (high - low))), 0, 1]
                        ),
                    )
            self.state.evolve(perm_circ)

    def set_source(self, source: pcvl.Source) -> None:
        """Set the Perceval source."""
        self.state.set_source(source)

    def finalize(self, output_nodes: Iterable[int]) -> None:
        """To be run at the end of pattern simulation."""
        self.sort_qubits(output_nodes)

    @property
    def nqubit(self) -> int:
        """Return the number of qubits of the current state."""
        return self.state.nqubit
