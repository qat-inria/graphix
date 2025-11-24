"""
Exporter to OpenQASM3.

This module provides functionality to export quantum circuit representations
to the OpenQASM3 format, allowing for interoperability with various quantum
computing platforms and tools that support OpenQASM3.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

# assert_never added in Python 3.11
from typing_extensions import assert_never

from graphix.fundamentals import Axis, Sign
from graphix.instruction import Instruction, InstructionKind
from graphix.measurements import PauliMeasurement
from graphix.pretty_print import OutputFormat, angle_to_str

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from graphix import Circuit
    from graphix.parameter import ExpressionOrFloat


def circuit_to_qasm3(circuit: Circuit) -> str:
    """
    Export circuit instructions to OpenQASM 3.0 representation.

    Parameters
    ----------
    circuit : Circuit
        The circuit containing the instructions to be exported.

    Returns
    -------
    str
        The OpenQASM 3.0 string representation of the circuit.
    """
    return "\n".join(circuit_to_qasm3_lines(circuit))


def circuit_to_qasm3_lines(circuit: Circuit) -> Iterator[str]:
    """
    Export circuit instructions to a line-by-line OpenQASM 3.0 representation.

    Parameters
    ----------
    circuit : Circuit
        The circuit to be exported.

    Returns
    -------
    Iterator[str]
        An iterator over the OpenQASM 3.0 lines that represent the circuit.
    """
    yield "OPENQASM 3;"
    yield 'include "stdgates.inc";'
    yield f"qubit[{circuit.width}] q;"
    if any(instr.kind == InstructionKind.M for instr in circuit.instruction):
        yield f"bit[{circuit.width}] b;"
    for instr in circuit.instruction:
        yield f"{instruction_to_qasm3(instr)};"


def qasm3_qubit(index: int) -> str:
    """
    Return the name of the indexed qubit.

    Parameters
    ----------
    index : int
        The index of the qubit, which should be a non-negative integer.

    Returns
    -------
    str
        The name of the qubit in QASM 3 format, represented as 'q[<index>]'.

    Raises
    ------
    ValueError
        If the index is negative.
    """
    return f"q[{index}]"


def qasm3_gate_call(gate: str, operands: Iterable[str], args: Iterable[str] | None = None) -> str:
    """
    Return the OpenQASM3 gate call.

    Parameters
    ----------
    gate : str
        The name of the quantum gate to be invoked.
    operands : Iterable[str]
        The list of qubit operands on which the gate operates.
    args : Iterable[str] | None, optional
        Additional arguments for the gate, if applicable. If no additional
        arguments are needed, this can be set to None. The default is None.

    Returns
    -------
    str
        The formatted OpenQASM3 gate call as a string.

    Examples
    --------
    >>> qasm3_gate_call('cx', ['q[0]', 'q[1]'])
    'cx q[0], q[1];'

    >>> qasm3_gate_call('rz', ['q[0]'], args=['1.57'])
    'rz(1.57) q[0];'
    """
    operands_str = ", ".join(operands)
    if args is None:
        return f"{gate} {operands_str}"
    args_str = ", ".join(args)
    return f"{gate}({args_str}) {operands_str}"


def angle_to_qasm3(angle: ExpressionOrFloat) -> str:
    """
    Convert an angle to its OpenQASM3 string representation.

    Parameters
    ----------
    angle : ExpressionOrFloat
        The angle to be converted, which can be of type Expression or a float value.

    Returns
    -------
    str
        The OpenQASM3 representation of the given angle.

    Examples
    --------
    >>> angle_to_qasm3(1.5708)
    '1.5708'

    >>> angle_to_qasm3('pi/2')
    'pi/2'
    """
    if not isinstance(angle, float):
        raise TypeError("QASM export of symbolic pattern is not supported")
    rad_over_pi = angle / pi
    return angle_to_str(rad_over_pi, output=OutputFormat.ASCII, multiplication_sign=True)


def instruction_to_qasm3(instruction: Instruction) -> str:
    """
    Convert a single circuit instruction to its OpenQASM3 representation.

    Parameters
    ----------
    instruction : Instruction
        The circuit instruction to be converted.

    Returns
    -------
    str
        The OpenQASM3 representation of the given instruction.
    """
    if instruction.kind == InstructionKind.M:
        if PauliMeasurement.try_from(instruction.plane, instruction.angle) != PauliMeasurement(Axis.Z, Sign.PLUS):
            raise ValueError("OpenQASM3 only supports measurements in Z axis.")
        return f"b[{instruction.target}] = measure q[{instruction.target}]"
    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.RX  # noqa: PLR1714
        or instruction.kind == InstructionKind.RY
        or instruction.kind == InstructionKind.RZ
    ):
        angle = angle_to_qasm3(instruction.angle)
        return qasm3_gate_call(instruction.kind.name.lower(), args=[angle], operands=[qasm3_qubit(instruction.target)])

    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.H  # noqa: PLR1714
        or instruction.kind == InstructionKind.S
        or instruction.kind == InstructionKind.X
        or instruction.kind == InstructionKind.Y
        or instruction.kind == InstructionKind.Z
    ):
        return qasm3_gate_call(instruction.kind.name.lower(), [qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.I:
        return qasm3_gate_call("id", [qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.CNOT:
        return qasm3_gate_call("cx", [qasm3_qubit(instruction.control), qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.SWAP:
        return qasm3_gate_call("swap", [qasm3_qubit(instruction.targets[i]) for i in (0, 1)])
    if instruction.kind == InstructionKind.RZZ:
        angle = angle_to_qasm3(instruction.angle)
        return qasm3_gate_call(
            "crz", args=[angle], operands=[qasm3_qubit(instruction.control), qasm3_qubit(instruction.target)]
        )
    if instruction.kind == InstructionKind.CCX:
        return qasm3_gate_call(
            "ccx",
            [
                qasm3_qubit(instruction.controls[0]),
                qasm3_qubit(instruction.controls[1]),
                qasm3_qubit(instruction.target),
            ],
        )
    # Use of `==` here for mypy
    if instruction.kind == InstructionKind._XC or instruction.kind == InstructionKind._ZC:  # noqa: PLR1714
        raise ValueError("Internal instruction should not appear")
    assert_never(instruction.kind)
