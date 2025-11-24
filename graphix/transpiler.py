"""
Gate-to-MBQC transpiler.

This module accepts desired gate operations and transpiles them into measurement-based quantum computation (MBQC) measurement patterns.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, SupportsFloat

import numpy as np
from typing_extensions import assert_never

from graphix import command, instruction, parameter
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.command import E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.instruction import Instruction, InstructionKind
from graphix.ops import Ops
from graphix.parameter import ExpressionOrFloat, Parameter
from graphix.pattern import Pattern
from graphix.sim import Data, Statevec, base_backend

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from numpy.random import Generator

    from graphix.command import Command


@dataclasses.dataclass
class TranspileResult:
    """
    The result of a transpilation.

    Parameters
    ----------
    pattern : :class:`graphix.pattern.Pattern`
        The pattern object representing the transpiled circuit.
    classical_outputs : tuple of int
        A tuple containing the indices of nodes measured with *M* gates.
    """

    pattern: Pattern
    classical_outputs: tuple[int, ...]


@dataclasses.dataclass
class SimulateResult:
    """
    The result of a simulation.

    Parameters
    ----------
    statevec : graphix.sim.statevec.Statevec
        The state vector after the simulation.
    classical_measures : tuple of int
        A tuple containing the classical measures obtained from the simulation.
    """

    statevec: Statevec
    classical_measures: tuple[int, ...]


Angle = ExpressionOrFloat


def _check_target(out: Sequence[int | None], index: int) -> int:
    target = out[index]
    if target is None:
        msg = f"Qubit {index} has already been measured."
        raise ValueError(msg)
    return target


class Circuit:
    """
    Gate-to-MBQC transpiler.

    Holds gate operations and translates them into MBQC measurement patterns.

    Attributes
    ----------
    width : int
        Number of logical qubits for the gate network.
    instruction : list
        List containing the sequence of gates applied.
    """

    instruction: list[Instruction]

    def __init__(self, width: int, instr: Iterable[Instruction] | None = None) -> None:
        """
        Construct a circuit.

        Parameters
        ----------
        width : int
            The number of logical qubits for the gate network.
        instr : Iterable[Instruction], optional
            A list of initial instructions. If None, no initial instructions are provided. Default is None.
        """
        self.width = width
        self.instruction = []
        self.active_qubits = set(range(width))
        if instr is not None:
            self.extend(instr)

    def add(self, instr: Instruction) -> None:
        """
        Add an instruction to the circuit.

        Parameters
        ----------
        instr : Instruction
            The instruction to be added to the circuit.

        Returns
        -------
        None
        """
        if instr.kind == InstructionKind.CCX:
            self.ccx(instr.controls[0], instr.controls[1], instr.target)
        elif instr.kind == InstructionKind.RZZ:
            self.rzz(instr.control, instr.target, instr.angle)
        elif instr.kind == InstructionKind.CNOT:
            self.cnot(instr.control, instr.target)
        elif instr.kind == InstructionKind.SWAP:
            self.swap(instr.targets[0], instr.targets[1])
        elif instr.kind == InstructionKind.H:
            self.h(instr.target)
        elif instr.kind == InstructionKind.S:
            self.s(instr.target)
        elif instr.kind == InstructionKind.X:
            self.x(instr.target)
        elif instr.kind == InstructionKind.Y:
            self.y(instr.target)
        elif instr.kind == InstructionKind.Z:
            self.z(instr.target)
        elif instr.kind == InstructionKind.I:
            self.i(instr.target)
        elif instr.kind == InstructionKind.M:
            self.m(instr.target, instr.plane, instr.angle)
        elif instr.kind == InstructionKind.RX:
            self.rx(instr.target, instr.angle)
        elif instr.kind == InstructionKind.RY:
            self.ry(instr.target, instr.angle)
        elif instr.kind == InstructionKind.RZ:
            self.rz(instr.target, instr.angle)
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind._XC or instr.kind == InstructionKind._ZC:  # noqa: PLR1714
            raise ValueError(f"Unsupported instruction: {instr}")
        else:
            assert_never(instr.kind)

    def extend(self, instrs: Iterable[Instruction]) -> None:
        """
        Extend the circuit by adding a sequence of instructions.

        Parameters
        ----------
        instrs : Iterable[Instruction]
            An iterable collection of `Instruction` objects to be added to the circuit.

        Returns
        -------
        None
            This method modifies the circuit in place and does not return a value.

        Notes
        -----
        This method allows for the addition of multiple instructions in one call,
        which can be useful for building complex circuits more efficiently.
        """
        for instr in instrs:
            self.add(instr)

    def __repr__(self) -> str:
        """
        Return a string representation of the Circuit.

        This method provides a concise summary of the Circuit object that
        can be useful for debugging and logging purposes. The output
        format may include key attributes and their values to give an
        overview of the Circuit's state.

        Returns
        -------
        str
            A string representing the Circuit.
        """
        return f"Circuit(width={self.width}, instr={self.instruction})"

    def cnot(self, control: int, target: int) -> None:
        """
        Apply a CNOT gate.

        Parameters
        ----------
        control : int
            The index of the control qubit.
        target : int
            The index of the target qubit.

        Returns
        -------
        None
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        assert control != target
        self.instruction.append(instruction.CNOT(control=control, target=target))

    def swap(self, qubit1: int, qubit2: int) -> None:
        """
        Apply a SWAP gate between two qubits.

        Parameters
        ----------
        qubit1 : int
            The index of the first qubit to be swapped.
        qubit2 : int
            The index of the second qubit to be swapped.

        Returns
        -------
        None
        """
        assert qubit1 in self.active_qubits
        assert qubit2 in self.active_qubits
        assert qubit1 != qubit2
        self.instruction.append(instruction.SWAP(targets=(qubit1, qubit2)))

    def h(self, qubit: int) -> None:
        """
        Apply a Hadamard gate to the specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the target qubit on which the Hadamard gate will be applied.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.H(target=qubit))

    def s(self, qubit: int) -> None:
        """
        Apply an S gate to the specified qubit.

        Parameters
        ----------
        qubit : int
            The target qubit on which to apply the S gate.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.S(target=qubit))

    def x(self, qubit: int) -> None:
        """
        Apply a Pauli X gate to the specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the target qubit on which the Pauli X gate will be applied.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.X(target=qubit))

    def y(self, qubit: int) -> None:
        """
        Apply a Pauli Y gate.

        Parameters
        ----------
        qubit : int
            The index of the target qubit.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Y(target=qubit))

    def z(self, qubit: int) -> None:
        """
        Apply a Pauli Z gate to the specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the target qubit on which the Pauli Z gate will be applied.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Z(target=qubit))

    def rx(self, qubit: int, angle: Angle) -> None:
        """
        Apply an X rotation gate.

        Parameters
        ----------
        qubit : int
            The target qubit.
        angle : Angle
            The rotation angle in radians.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RX(target=qubit, angle=angle))

    def ry(self, qubit: int, angle: Angle) -> None:
        """
        Apply a Y rotation gate.

        Parameters
        ----------
        qubit : int
            The index of the target qubit.
        angle : Angle
            The rotation angle in radians.

        Returns
        -------
        None
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RY(target=qubit, angle=angle))

    def rz(self, qubit: int, angle: Angle) -> None:
        """
        Apply a Z rotation gate.

        Parameters
        ----------
        qubit : int
            Target qubit.
        angle : Angle
            Rotation angle in radians.

        Notes
        -----
        The Z rotation gate applies a phase shift to the state of the target qubit,
        which is represented by the specified angle in radians. The operation can be
        expressed mathematically as:

        .. math:: R_z(\theta) = e^{-i\theta/2} |0\rangle\langle0| + e^{i\theta/2} |1\rangle\langle1|
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RZ(target=qubit, angle=angle))

    def rzz(self, control: int, target: int, angle: Angle) -> None:
        """
        Apply a ZZ-rotation gate.

        The ZZ-rotation gate is equivalent to the following sequence:
        1. CNOT(control, target),
        2. Rz(target, angle),
        3. CNOT(control, target).

        This gate realizes a rotation expressed by the equation:
        :math:`e^{-i \frac{\theta}{2} Z_c Z_t}`.

        Parameters
        ----------
        control : int
            The index of the control qubit.
        target : int
            The index of the target qubit.
        angle : Angle
            The rotation angle in radians.

        Returns
        -------
        None
            This method does not return a value but applies the gate to the circuit.
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        self.instruction.append(instruction.RZZ(control=control, target=target, angle=angle))

    def ccx(self, control1: int, control2: int, target: int) -> None:
        """
        Apply a CCX (Toffoli) gate.

        Parameters
        ----------
        control1 : int
            First control qubit.
        control2 : int
            Second control qubit.
        target : int
            Target qubit.

        Returns
        -------
        None
        """
        assert control1 in self.active_qubits
        assert control2 in self.active_qubits
        assert target in self.active_qubits
        assert control1 != control2
        assert control1 != target
        assert control2 != target
        self.instruction.append(instruction.CCX(controls=(control1, control2), target=target))

    def i(self, qubit: int) -> None:
        """
        Apply an identity (teleportation) gate to the specified qubit.

        Parameters
        ----------
        qubit : int
            The index of the target qubit on which the identity gate will be applied.
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.I(target=qubit))

    def m(self, qubit: int, plane: Plane, angle: Angle) -> None:
        """
        Measure a quantum qubit.

        The measured qubit cannot be used afterwards.

        Parameters
        ----------
        qubit : int
            The index of the target qubit to be measured.
        plane : Plane
            The measurement plane in which the qubit is to be measured.
        angle : Angle
            The angle of measurement with respect to the chosen plane.

        Returns
        -------
        None
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.M(target=qubit, plane=plane, angle=angle))
        self.active_qubits.remove(qubit)

    def transpile(self) -> TranspileResult:
        """
        Transpile the circuit to a specific pattern.

        Returns
        -------
        result : TranspileResult
            An object containing the results of the transpilation process.
        """
        n_node = self.width
        out: list[int | None] = list(range(self.width))
        pattern = Pattern(input_nodes=list(range(self.width)))
        classical_outputs = []
        for instr in _transpile_rzz(self.instruction):
            if instr.kind == instruction.InstructionKind.CNOT:
                ancilla = [n_node, n_node + 1]
                control = _check_target(out, instr.control)
                target = _check_target(out, instr.target)
                out[instr.control], out[instr.target], seq = self._cnot_command(control, target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.SWAP:
                target0 = _check_target(out, instr.targets[0])
                target1 = _check_target(out, instr.targets[1])
                out[instr.targets[0]], out[instr.targets[1]] = (
                    target1,
                    target0,
                )
            elif instr.kind == instruction.InstructionKind.I:
                pass
            elif instr.kind == instruction.InstructionKind.H:
                single_ancilla = n_node
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._h_command(target, single_ancilla)
                pattern.extend(seq)
                n_node += 1
            elif instr.kind == instruction.InstructionKind.S:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._s_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.X:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._x_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.Y:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._y_command(target, ancilla)
                pattern.extend(seq)
                n_node += 4
            elif instr.kind == instruction.InstructionKind.Z:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._z_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.RX:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._rx_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.RY:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._ry_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 4
            elif instr.kind == instruction.InstructionKind.RZ:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._rz_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.CCX:
                ancilla = [n_node + i for i in range(18)]
                control0 = _check_target(out, instr.controls[0])
                control1 = _check_target(out, instr.controls[1])
                target = _check_target(out, instr.target)
                (
                    out[instr.controls[0]],
                    out[instr.controls[1]],
                    out[instr.target],
                    seq,
                ) = self._ccx_command(
                    control0,
                    control1,
                    target,
                    ancilla,
                )
                pattern.extend(seq)
                n_node += 18
            elif instr.kind == instruction.InstructionKind.M:
                target = _check_target(out, instr.target)
                seq = self._m_command(target, instr.plane, instr.angle)
                pattern.extend(seq)
                classical_outputs.append(target)
                out[instr.target] = None
            else:
                raise ValueError("Unknown instruction, commands not added")
        output_nodes = [node for node in out if node is not None]
        pattern.reorder_output_nodes(output_nodes)
        return TranspileResult(pattern, tuple(classical_outputs))

    @classmethod
    def _cnot_command(
        cls, control_node: int, target_node: int, ancilla: Sequence[int]
    ) -> tuple[int, int, list[command.Command]]:
        """
        Generate MBQC commands for the CNOT gate.

        Parameters
        ----------
        control_node : int
            Index of the control node on the graph.
        target_node : int
            Index of the target node on the graph.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.

        Returns
        -------
        tuple[int, int, list[command.Command]]
            A tuple containing:
                - control_out : int
                    Index of the control node after the gate operation.
                - target_out : int
                    Index of the target node after the gate operation.
                - commands : list[command.Command]
                    List of MBQC commands generated for the operation.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(target_node, ancilla[0])),
                E(nodes=(control_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=target_node),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={target_node}),
                Z(node=control_node, domain={target_node}),
            )
        )
        return control_node, ancilla[1], seq

    @classmethod
    def _m_command(cls, input_node: int, plane: Plane, angle: Angle) -> list[Command]:
        """
        MBQC commands for measuring a qubit.

        Parameters
        ----------
        input_node : int
            Target node on the graph.
        plane : Plane
            Plane of the measurement.
        angle : Angle
            Angle of the measurement (unit: π radians).

        Returns
        -------
        commands : list of Command
            List of MBQC commands.
        """
        return [M(node=input_node, plane=plane, angle=angle)]

    @classmethod
    def _h_command(cls, input_node: int, ancilla: int) -> tuple[int, list[Command]]:
        """
        MBQC commands for the Hadamard gate.

        Parameters
        ----------
        input_node : int
            The target node on the graph.
        ancilla : int
            The index of the ancilla node to be added.

        Returns
        -------
        out_node : int
            The control node on the graph after the gate.
        commands : list
            A list of MBQC commands.
        """
        seq: list[Command] = [N(node=ancilla)]
        seq.extend((E(nodes=(input_node, ancilla)), M(node=input_node), X(node=ancilla, domain={input_node})))
        return ancilla, seq

    @classmethod
    def _s_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """
        Generates MBQC commands for the S gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.

        Returns
        -------
        out_node : int
            Index of the control node on the graph after the gate.
        commands : list[command.Command]
            List of MBQC commands generated for the S gate.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), command.N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-0.5),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _x_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for the Pauli X gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Sequence of two integers representing the indices of the ancilla nodes to be added to the graph.

        Returns
        -------
        out_node : int
            Index of the control node on the graph after the gate.
        commands : list[command.Command]
            List of MBQC commands.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node),
                M(node=ancilla[0], angle=-1),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _y_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for the Pauli Y gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.

        Returns
        -------
        out_node : int
            Index of the control node on the graph after the gate.
        commands : list[command.Command]
            List of MBQC commands.
        """
        assert len(ancilla) == 4
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=1.0, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={input_node}),
                M(node=ancilla[2]),
                X(node=ancilla[3], domain={ancilla[0], ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _z_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for the Pauli Z gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.

        Returns
        -------
        out_node : int
            Index of the control node on the graph after the gate.
        commands : list
            List of MBQC commands.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-1),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _rx_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for X rotation gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.
        angle : Angle
            Measurement angle in radians.

        Returns
        -------
        out_node : int
            Control node on the graph after the gate.
        commands : list[command.Command]
            List of MBQC commands.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node),
                M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _ry_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for the Y rotation gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.
        angle : Angle
            Rotation angle in radians.

        Returns
        -------
        out_node : int
            Control node on the graph after the gate operation.
        commands : list[command.Command]
            List of MBQC commands generated for the Y rotation.
        """
        assert len(ancilla) == 4
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={input_node}),
                M(node=ancilla[2]),
                X(node=ancilla[3], domain={ancilla[0], ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _rz_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """
        MBQC commands for the Z rotation gate.

        Parameters
        ----------
        input_node : int
            Index of the input node.
        ancilla : Sequence[int]
            Indices of the ancilla nodes to be added to the graph.
        angle : Angle
            Measurement angle in radians.

        Returns
        -------
        out_node : int
            Index of the node on the graph after the gate.
        commands : list
            List of MBQC commands.
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]  # assign new qubit labels
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-angle / np.pi),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _ccx_command(
        cls,
        control_node1: int,
        control_node2: int,
        target_node: int,
        ancilla: Sequence[int],
    ) -> tuple[int, int, int, list[command.Command]]:
        """
        MBQC commands for the CCX gate.

        Parameters
        ----------
        control_node1 : int
            First control node on the graph.
        control_node2 : int
            Second control node on the graph.
        target_node : int
            Target node on the graph.
        ancilla : Sequence[int]
            Ancilla node indices to be added to the graph.

        Returns
        -------
        tuple
            A tuple containing:
                control_out1 : int
                    First control node on the graph after the gate.
                control_out2 : int
                    Second control node on the graph after the gate.
                target_out : int
                    Target node on the graph after the gate.
                commands : list[command.Command]
                    List of MBQC commands.
        """
        assert len(ancilla) == 18
        seq: list[Command] = [N(node=ancilla[i]) for i in range(18)]  # assign new qubit labels
        seq.extend(
            (
                E(nodes=(target_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[1], control_node2)),
                E(nodes=(control_node1, ancilla[14])),
                E(nodes=(ancilla[2], ancilla[3])),
                E(nodes=(ancilla[14], ancilla[4])),
                E(nodes=(ancilla[3], ancilla[5])),
                E(nodes=(ancilla[3], ancilla[4])),
                E(nodes=(ancilla[5], ancilla[6])),
                E(nodes=(control_node2, ancilla[6])),
                E(nodes=(control_node2, ancilla[9])),
                E(nodes=(ancilla[6], ancilla[7])),
                E(nodes=(ancilla[9], ancilla[4])),
                E(nodes=(ancilla[9], ancilla[10])),
                E(nodes=(ancilla[7], ancilla[8])),
                E(nodes=(ancilla[10], ancilla[11])),
                E(nodes=(ancilla[4], ancilla[8])),
                E(nodes=(ancilla[4], ancilla[11])),
                E(nodes=(ancilla[4], ancilla[16])),
                E(nodes=(ancilla[8], ancilla[12])),
                E(nodes=(ancilla[11], ancilla[15])),
                E(nodes=(ancilla[12], ancilla[13])),
                E(nodes=(ancilla[16], ancilla[17])),
                M(node=target_node),
                M(node=ancilla[0], s_domain={target_node}),
                M(node=ancilla[1], s_domain={ancilla[0]}),
                M(node=control_node1),
                M(node=ancilla[2], angle=-1.75, s_domain={ancilla[1], target_node}),
                M(node=ancilla[14], s_domain={control_node1}),
                M(node=ancilla[3], s_domain={ancilla[2], ancilla[0]}),
                M(node=ancilla[5], angle=-0.25, s_domain={ancilla[3], ancilla[1], ancilla[14], target_node}),
                M(node=control_node2, angle=-0.25),
                M(node=ancilla[6], s_domain={ancilla[5], ancilla[2], ancilla[0]}),
                M(node=ancilla[9], s_domain={control_node2, ancilla[5], ancilla[2]}),
                M(
                    node=ancilla[7],
                    angle=-1.75,
                    s_domain={ancilla[6], ancilla[3], ancilla[1], ancilla[14], target_node},
                ),
                M(node=ancilla[10], angle=-1.75, s_domain={ancilla[9], ancilla[14]}),
                M(node=ancilla[4], angle=-0.25, s_domain={ancilla[14]}),
                M(node=ancilla[8], s_domain={ancilla[7], ancilla[5], ancilla[2], ancilla[0]}),
                M(node=ancilla[11], s_domain={ancilla[10], control_node2, ancilla[5], ancilla[2]}),
                M(
                    node=ancilla[12],
                    angle=-0.25,
                    s_domain={ancilla[8], ancilla[6], ancilla[3], ancilla[1], target_node},
                ),
                M(
                    node=ancilla[16],
                    s_domain={
                        ancilla[4],
                        control_node1,
                        ancilla[2],
                        control_node2,
                        ancilla[7],
                        ancilla[10],
                        ancilla[2],
                        control_node2,
                        ancilla[5],
                    },
                ),
                X(node=ancilla[17], domain={ancilla[14], ancilla[16]}),
                X(node=ancilla[15], domain={ancilla[9], ancilla[11]}),
                X(node=ancilla[13], domain={ancilla[0], ancilla[2], ancilla[5], ancilla[7], ancilla[12]}),
                Z(node=ancilla[17], domain={ancilla[4], ancilla[5], ancilla[7], ancilla[10], control_node1}),
                Z(node=ancilla[15], domain={control_node2, ancilla[2], ancilla[5], ancilla[10]}),
                Z(node=ancilla[13], domain={ancilla[1], ancilla[3], ancilla[6], ancilla[8], target_node}),
            )
        )
        return ancilla[17], ancilla[15], ancilla[13], seq

    def simulate_statevector(
        self,
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
    ) -> SimulateResult:
        """
        Run statevector simulation of the gate sequence.

        Parameters
        ----------
        input_state : Data | None
            The initial state for the simulation. If None, a default state will be used.
        branch_selector : BranchSelector | None
            Branch selector for measurements (default is :class:`RandomBranchSelector`).
        rng : Generator | None, optional
            Random-number generator for measurements. This generator is used only in
            case of random branch selection (see :class:`RandomBranchSelector`).

        Returns
        -------
        result : SimulateResult
            The output state of the statevector simulation and results of classical measurements.
        """
        symbolic = self.is_parameterized()
        if branch_selector is None:
            branch_selector = RandomBranchSelector()

        state = Statevec(nqubit=self.width) if input_state is None else Statevec(nqubit=self.width, data=input_state)

        classical_measures = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]
            if instr.kind == instruction.InstructionKind.CNOT:
                state.cnot((instr.control, instr.target))
            elif instr.kind == instruction.InstructionKind.SWAP:
                state.swap(instr.targets)
            elif instr.kind == instruction.InstructionKind.I:
                pass
            elif instr.kind == instruction.InstructionKind.S:
                state.evolve_single(Ops.S, instr.target)
            elif instr.kind == instruction.InstructionKind.H:
                state.evolve_single(Ops.H, instr.target)
            elif instr.kind == instruction.InstructionKind.X:
                state.evolve_single(Ops.X, instr.target)
            elif instr.kind == instruction.InstructionKind.Y:
                state.evolve_single(Ops.Y, instr.target)
            elif instr.kind == instruction.InstructionKind.Z:
                state.evolve_single(Ops.Z, instr.target)
            elif instr.kind == instruction.InstructionKind.RX:
                state.evolve_single(Ops.rx(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RY:
                state.evolve_single(Ops.ry(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RZ:
                state.evolve_single(Ops.rz(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RZZ:
                state.evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
            elif instr.kind == instruction.InstructionKind.CCX:
                state.evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
            elif instr.kind == instruction.InstructionKind.M:
                result = base_backend.perform_measure(
                    instr.target,
                    instr.target,
                    instr.plane,
                    instr.angle * np.pi,
                    state,
                    branch_selector,
                    rng=rng,
                    symbolic=symbolic,
                )
                classical_measures.append(result)
            else:
                raise ValueError(f"Unknown instruction: {instr}")
        return SimulateResult(state, tuple(classical_measures))

    def map_angle(self, f: Callable[[Angle], Angle]) -> Circuit:
        """
        Apply a function to all angles in the circuit.

        Parameters
        ----------
        f : Callable[[Angle], Angle]
            A function that takes an Angle as input and returns an Angle.

        Returns
        -------
        Circuit
            A new Circuit instance with all angles transformed by the function `f`.
        """
        result = Circuit(self.width)
        for instr in self.instruction:
            # Use == for mypy
            if (
                instr.kind == InstructionKind.RZZ  # noqa: PLR1714
                or instr.kind == InstructionKind.M
                or instr.kind == InstructionKind.RX
                or instr.kind == InstructionKind.RY
                or instr.kind == InstructionKind.RZ
            ):
                new_instr = dataclasses.replace(instr, angle=f(instr.angle))
                result.instruction.append(new_instr)
            else:
                result.instruction.append(instr)
        return result

    def is_parameterized(self) -> bool:
        """
        Determine if the circuit is parameterized.

        A circuit is considered parameterized if there is at least
        one measurement angle that is not an instance of `SupportsFloat`.
        This typically indicates the presence of a parameterized
        expression, such as an instance of `sympy.Expr`, although
        the use of `sympy` is not enforced.

        Returns
        -------
        bool
            `True` if the circuit is parameterized, `False` otherwise.
        """
        # Use of `==` here for mypy
        return any(
            not isinstance(instr.angle, SupportsFloat)
            for instr in self.instruction
            if instr.kind == InstructionKind.RZZ  # noqa: PLR1714
            or instr.kind == InstructionKind.M
            or instr.kind == InstructionKind.RX
            or instr.kind == InstructionKind.RY
            or instr.kind == InstructionKind.RZ
        )

    def subs(self, variable: Parameter, substitute: ExpressionOrFloat) -> Circuit:
        """
        Return a copy of the circuit with all occurrences of the specified variable
        in measurement angles substituted with the given value.

        Parameters
        ----------
        variable : Parameter
            The variable to be substituted in the measurement angles.
        substitute : ExpressionOrFloat
            The value to replace the occurrences of the variable.

        Returns
        -------
        Circuit
            A new circuit with the substitutions applied.

        Notes
        -----
        This method does not modify the original circuit and creates a copy
        with the necessary substitutions.
        """
        return self.map_angle(lambda angle: parameter.subs(angle, variable, substitute))

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrFloat]) -> Circuit:
        """
        Return a copy of the circuit with all occurrences of the specified keys in measurement angles substituted by the provided values.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrFloat]
            A mapping of parameters to their corresponding replacement values.

        Returns
        -------
        Circuit
            A new circuit with the specified substitutions made in the measurement angles.

        Notes
        -----
        This method performs the substitutions in parallel, ensuring that all occurrences are replaced throughout the circuit.
        """
        return self.map_angle(lambda angle: parameter.xreplace(angle, assignment))


def _extend_domain(measure: M, domain: set[int]) -> None:
    """
    Extend the correction domain of `measure` by `domain`.

    Parameters
    ----------
    measure : M
        Measurement command to modify.
    domain : set[int]
        Set of nodes to XOR into the appropriate domain of `measure`.

    Returns
    -------
    None
    """
    if measure.plane == Plane.XY:
        measure.s_domain ^= domain
    else:
        measure.t_domain ^= domain


def _transpile_rzz(instructions: Iterable[Instruction]) -> Iterator[Instruction]:
    for instr in instructions:
        if instr.kind == InstructionKind.RZZ:
            yield instruction.CNOT(control=instr.control, target=instr.target)
            yield instruction.RZ(target=instr.target, angle=instr.angle)
            yield instruction.CNOT(control=instr.control, target=instr.target)
        else:
            yield instr
