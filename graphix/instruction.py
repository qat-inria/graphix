"Instruction classes for defining and managing various instructions within the system."

from __future__ import annotations

import enum
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal, SupportsFloat, Union

from graphix import utils
from graphix.fundamentals import Plane

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001
from graphix.pretty_print import OutputFormat, angle_to_str
from graphix.repr_mixins import DataclassReprMixin


def repr_angle(angle: ExpressionOrFloat) -> str:
    """
    Return the representation string of an angle in radians.

    This function is used for pretty-printing instructions that include
    `angle` parameters. The representation delegates to
    :func:`pretty_print.angle_to_str`.

    Parameters
    ----------
    angle : ExpressionOrFloat
        The angle in radians to be represented as a string.

    Returns
    -------
    str
        A string representation of the angle in radians.
    """
    # Non-float-supporting objects are returned as-is
    if not isinstance(angle, SupportsFloat):
        return str(angle)

    # Convert to float, express in π units, and format in ASCII/plain mode
    pi_units = float(angle) / math.pi
    return angle_to_str(pi_units, OutputFormat.ASCII)


class InstructionKind(Enum):
    """
    Tag for instruction kind.

    This class serves as a marker for different types of instructions
    that can be utilized within a specific context, allowing for better
    organization and handling of various instruction types.

    Attributes
    ----------
    kind : str
        A string representing the type of instruction.

    Methods
    -------
    __init__(kind: str)
        Initializes the InstructionKind with the specified kind.

    __repr__() -> str
        Returns a string representation of the InstructionKind instance.
    """

    def __init__(self, kind: str):
        """Initializes the InstructionKind with the specified kind.

        Parameters
        ----------
        kind : str
            A string representing the type of instruction.
        """
        self.kind = kind

    def __repr__(self) -> str:
        """Returns a string representation of the InstructionKind instance.

        Returns
        -------
        str
            A string that represents the InstructionKind.
    """

    CCX = enum.auto()
    RZZ = enum.auto()
    CNOT = enum.auto()
    SWAP = enum.auto()
    H = enum.auto()
    S = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    I = enum.auto()
    M = enum.auto()
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    # The two following instructions are used internally by the transpiler
    _XC = enum.auto()
    _ZC = enum.auto()


class _KindChecker:
    """
    Enforce tag field declaration.

    This class is used to ensure that the required tag fields are declared
    and followed during instantiation and usage. It helps to maintain the
    integrity of data structures by enforcing type checks and field
    specifications.

    Attributes
    ----------
    tag_fields : list
        A list of required tag fields that must be declared.

    Methods
    -------
    check_tags(declared_tags):
        Validates the declared tags against the required tag fields.

    add_tag_field(tag):
        Adds a new tag field to the list of required fields.

    is_valid_tag(tag):
        Checks if the provided tag is a valid required tag.
    """

    def __init_subclass__(cls) -> None:
        """
        Validate that subclasses define the ``kind`` attribute.

        This method is automatically called when a class is subclassed. It checks
        if the subclass has defined the required `kind` attribute. If the
        attribute is not defined, it raises an AttributeError.

        Parameters
        ----------
        cls : type
            The class that is being initialized as a subclass.

        Raises
        ------
        AttributeError
            If the subclass does not have the `kind` attribute defined.
        """
        super().__init_subclass__()
        utils.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


@dataclass(repr=False)
class CCX(_KindChecker, DataclassReprMixin):
    """
    Toffoli circuit instruction.

    The CCX instruction implements a Toffoli gate, which is a controlled-controlled-NOT operation.
    It affects three qubits: if the first two qubits (controls) are in the state |1⟩,
    it flips the state of the third qubit (target).

    Parameters
    ----------
    control1 : int
        The index of the first control qubit.
    control2 : int
        The index of the second control qubit.
    target : int
        The index of the target qubit.

    Attributes
    ----------
    control1 : int
        The index of the first control qubit.
    control2 : int
        The index of the second control qubit.
    target : int
        The index of the target qubit.

    Methods
    -------
    apply(circuit):
        Applies the CCX instruction to the specified circuit.
    """

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = field(default=InstructionKind.CCX, init=False)


@dataclass(repr=False)
class RZZ(_KindChecker, DataclassReprMixin):
    """
    RZZ circuit instruction.

    This class represents the RZZ quantum gate, which is a two-qubit gate that implements a phase shift
    based on the control and target qubit states.

    Attributes
    ----------
    theta : float
        The angle parameter that defines the phase shift.

    Methods
    -------
    apply(qubits)
        Applies the RZZ gate to the specified qubits.

    to_matrix()
        Returns the matrix representation of the RZZ gate.

    __str__()
        Returns a string representation of the RZZ gate.
    """

    target: int
    control: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    # FIXME: Remove `| None` from `meas_index`
    # - `None` makes codes messy/type-unsafe
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZZ]] = field(default=InstructionKind.RZZ, init=False)


@dataclass(repr=False)
class CNOT(_KindChecker, DataclassReprMixin):
    """
    CNOT circuit instruction.

    The CNOT (Controlled NOT) gate is a two-qubit gate that flips the state of
    the second qubit (target) if the first qubit (control) is in the state |1⟩.

    Parameters
    ----------
    control : int
        The index of the control qubit.
    target : int
        The index of the target qubit.

    Attributes
    ----------
    control : int
        The index of the control qubit.
    target : int
        The index of the target qubit.

    Methods
    -------
    apply(state)
        Applies the CNOT gate to the given quantum state.
    """

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = field(default=InstructionKind.CNOT, init=False)


@dataclass(repr=False)
class SWAP(_KindChecker, DataclassReprMixin):
    """
    SWAP circuit instruction.

    The SWAP gate is a two-qubit quantum gate that swaps the states of two qubits.
    It can be represented by the following operation:

    .. math::
        \text{SWAP} |ab\rangle = |ba\rangle

    where |a⟩ and |b⟩ represent the states of the two qubits.

    Attributes
    ----------
    qubits : tuple
        A tuple containing the indices of the two qubits to be swapped.

    Methods
    -------
    apply(circuit):
        Applies the SWAP gate to the specified circuit.
    """

    def __init__(self, qubits):
        """
        Parameters
        ----------
        qubits : tuple
            A tuple of two integers representing the indices of the qubits to swap.
        """
        self.qubits = qubits

    def apply(self, circuit):
        """
        Applies the SWAP gate to the specified circuit.

        Parameters
        ----------
        circuit : Circuit
            The circuit to which the SWAP gate should be applied.
    """

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = field(default=InstructionKind.SWAP, init=False)


@dataclass(repr=False)
class H(_KindChecker, DataclassReprMixin):
    """
    H circuit instruction.

    This class represents the Hadamard gate, which is a fundamental
    quantum gate used to create superposition states.

    Attributes
    ----------
    name : str
        The name of the gate.
    num_qubits : int
        The number of qubits the gate acts upon (always 1 for the Hadamard gate).

    Methods
    -------
    apply(qubit_state):
        Applies the Hadamard gate to the given qubit state.

    Examples
    --------
    >>> h_gate = H()
    >>> new_state = h_gate.apply([1, 0])  # Apply H to |0>
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = field(default=InstructionKind.H, init=False)


@dataclass(repr=False)
class S(_KindChecker, DataclassReprMixin):
    """
    S circuit instruction.

    The S gate is a single-qubit gate that applies a phase shift of π/2
    to the state of the qubit. It is also known as the phase gate or
    the rotation gate.

    Parameters
    ----------
    qubit : int
        The index of the qubit to which the S gate will be applied.

    Attributes
    ----------
    qubit : int
        The index of the qubit.

    Methods
    -------
    apply(circuit):
        Applies the S gate to the specified qubit in the given circuit.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = field(default=InstructionKind.S, init=False)


@dataclass(repr=False)
class X(_KindChecker, DataclassReprMixin):
    """
    X circuit instruction.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    None

    Notes
    -----
    This class represents the X gate instruction in quantum circuits, which is used to flip the state of a qubit.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = field(default=InstructionKind.X, init=False)


@dataclass(repr=False)
class Y(_KindChecker, DataclassReprMixin):
    """
    Y circuit instruction.

    The Y class implements the Y gate, which is a fundamental quantum gate
    that performs a rotation around the Y-axis of the Bloch sphere.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    __init__()
        Initializes the Y gate.

    apply(qubit)
        Applies the Y gate to the specified qubit.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = field(default=InstructionKind.Y, init=False)


@dataclass(repr=False)
class Z(_KindChecker, DataclassReprMixin):
    """
    Z circuit instruction.

    The Z gate, also known as the phase flip gate, is a single-qubit gate that
    performs a phase flip. It leaves the computational basis states |0⟩ unchanged
    while introducing a phase of π (180 degrees) to the |1⟩ state.

    Parameters
    ----------
    None

    Methods
    -------
    apply(qubit_state: np.ndarray) -> np.ndarray
        Applies the Z gate to the given qubit state.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = field(default=InstructionKind.Z, init=False)


@dataclass(repr=False)
class I(_KindChecker, DataclassReprMixin):
    """
    Represents an I (identity) circuit instruction.

    This instruction performs no operation on the qubit it acts upon, effectively leaving
    the qubit in its current state. It can be useful for circuits where a specific structure
    is required without modifying the qubit state.

    Attributes
    ----------
    name : str
        The name of the instruction, which is 'I'.

    Methods
    -------
    __call__(self, qubit):
        Applies the I instruction to the specified qubit.
    """

    def __init__(self):
        self.name = 'I'

    def __call__(self, qubit):
        """Applies the I instruction to the specified qubit.

        Parameters
        ----------
        qubit : Qubit
            The qubit to which the instruction is applied.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = field(default=InstructionKind.I, init=False)


@dataclass(repr=False)
class M(_KindChecker, DataclassReprMixin):
    """
    M circuit instruction.

    This class represents an M instruction in quantum circuits, which is
    typically used to denote measurement operations.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    target: int
    plane: Plane
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.M]] = field(default=InstructionKind.M, init=False)


@dataclass(repr=False)
class RX(_KindChecker, DataclassReprMixin):
    """
    X Rotation Circuit Instruction.

    The RX gate represents rotation around the X-axis of the Bloch sphere.
    This gate is commonly used in quantum computing to manipulate qubits.

    Parameters
    ----------
    theta : float
        The angle of rotation in radians.

    Attributes
    ----------
    theta : float
        The angle of rotation in radians for the RX gate.

    Methods
    -------
    matrix() -> numpy.ndarray
        Returns the matrix representation of the RX gate.
    """

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RX]] = field(default=InstructionKind.RX, init=False)


@dataclass(repr=False)
class RY(_KindChecker, DataclassReprMixin):
    """
    Y rotation circuit instruction.

    The RY instruction represents a rotation around the Y axis of the Bloch sphere.

    Parameters
    ----------
    angle : float
        The angle in radians by which to rotate around the Y axis.

    Attributes
    ----------
    angle : float
        The rotation angle in radians.

    Methods
    -------
    __call__(qubit)
        Applies the RY rotation to the specified qubit.
    """

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RY]] = field(default=InstructionKind.RY, init=False)


@dataclass(repr=False)
class RZ(_KindChecker, DataclassReprMixin):
    """
    Z rotation circuit instruction.

    This class represents a rotation about the Z-axis in a quantum circuit.

    Parameters
    ----------
    angle : float
        The angle by which to rotate the qubit around the Z-axis, in radians.

    Attributes
    ----------
    angle : float
        The rotation angle.

    Methods
    -------
    apply(circuit):
        Apply the RZ gate to the given circuit.
    """

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZ]] = field(default=InstructionKind.RZ, init=False)


@dataclass
class _XC(_KindChecker):
    """
    X correction circuit instruction.

    This class is used internally by the transpiler to represent
    the X correction circuit instruction.
    """

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._XC]] = field(default=InstructionKind._XC, init=False)


@dataclass
class _ZC(_KindChecker):
    """
    Z correction circuit instruction.

    This class is used internally by the transpiler to represent a
    Z correction circuit instruction.
    """

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._ZC]] = field(default=InstructionKind._ZC, init=False)


if sys.version_info >= (3, 10):
    Instruction = CCX | RZZ | CNOT | SWAP | H | S | X | Y | Z | I | M | RX | RY | RZ | _XC | _ZC
else:
    Instruction = Union[CCX, RZZ, CNOT, SWAP, H, S, X, Y, Z, I, M, RX, RY, RZ, _XC, _ZC]
