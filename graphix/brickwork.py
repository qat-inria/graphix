"""Brickwork classes.

accepts desired gate operations and transpile into brickwork MBQC measurement patterns.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_never

from graphix import Pattern, command
from graphix.command import E, M, N, X, Z
from graphix.instruction import InstructionKind
from graphix.parameter import ExpressionOrFloat, check_expression_or_float

if TYPE_CHECKING:
    from graphix import instruction
    from graphix.transpiler import Circuit

Angle = ExpressionOrFloat


class Brick(ABC):
    """
    Abstract base class for bricks in a brickwork state (currently CNOT and single-qubit pairs).

    Methods
    -------
    measures() -> list[list[float]]:
        Returns the measurement angles for the brick. The sublists correspond to the top and bottom qubits in the brick.
    """

    @abstractmethod
    def measures(self) -> list[list[Angle]]: ...
    """Returns the measurement angles for the brick. The sublists correspond to the top and bottom qubits in the brick."""
    # @abstractmethod
    # def to_circuit(self, circuit: Circuit, nqubit_top: int) -> None: ...


@dataclass
class CNOT(Brick):
    """
    Represents a CNOT gate in a brickwork state.

    Attributes
    ----------
    target_above : bool
        Indicates if the target qubit is above the control qubit (true) or not (false).

    Methods
    -------
    measures() -> list[list[float]]:
        Returns the measurement angles for the CNOT gate.
    """

    target_above: bool

    def measures(self) -> list[list[Angle]]:
        """Return the measurement angles for the CNOT gate."""
        if self.target_above:
            return [[0, np.pi / 2, 0, -np.pi / 2], [0, 0, np.pi / 2, 0]]
        return [[0, 0, np.pi / 2, 0], [0, np.pi / 2, 0, -np.pi / 2]]

    # def to_circuit(self, circuit: Circuit, nqubit_top: int) -> None:
    #     if self.target_above:
    #         control = nqubit_top + 1
    #         target = nqubit_top
    #     else:
    #         control = nqubit_top
    #         target = nqubit_top + 1
    #     circuit.cnot(control, target)


class XZ(Enum):
    """Tag for the axis of rotation of a single-qubit gate brick."""

    X = enum.auto()
    Z = enum.auto()


def value_or_zero(v: Angle | None) -> Angle:
    """Return the Angle object if it is not None, otherwise return 0."""
    if v is None:
        return 0
    return v


@dataclass
class SingleQubit:
    """
    Represents a single-qubit gate in a brickwork state, which can be decomposed into Rz-Rx-Rz rotations.

    Attributes
    ----------
    rz0 : float | None
        The angle of the first Rz rotation. None if not set.
    rx : float | None
        The angle of the Rx rotation. None if not set.
    rz1 : float | None
        The angle of the second Rz rotation. None if not set.

    Methods
    -------
    measures() -> list[float]:
        Returns the measurement angles for the single-qubit gate.

    is_identity() -> bool:
        Checks if the gate is an identity operation (no rotations).

    add(axis: XZ, angle: float) -> bool:
        Attempts to add a rotation to the gate. Returns True if successful, False otherwise.
    """

    rz0: Angle | None = None
    rx: Angle | None = None
    rz1: Angle | None = None

    def measures(self) -> list[Angle]:
        """Return the measurement angles for the single-qubit gate."""
        return [
            -value_or_zero(self.rz0),
            -value_or_zero(self.rx),
            -value_or_zero(self.rz1),
            0,
        ]

    # def to_circuit(self, circuit: Circuit, nqubit: int) -> None:
    #     if self.rz0 is not None:
    #         circuit.rz(nqubit, self.rz0)
    #     if self.rx is not None:
    #         circuit.rx(nqubit, self.rx)
    #     if self.rz1 is not None:
    #         circuit.rz(nqubit, self.rz1)

    def is_identity(self) -> bool:
        """Check if the gate is an identity operation (no rotations)."""
        return self.rz0 is None and self.rx is None and self.rz1 is None

    def add(self, axis: XZ, angle: Angle) -> bool:
        """Add a rotation to the brick. Returns True if successful, False otherwise. Includes typing assertion to ensure all cases are covered."""
        if axis == XZ.X:
            if self.rx is None and self.rz1 is None:
                self.rx = angle
                return True
            return False
        if axis == XZ.Z:
            if self.rz0 is None and self.rx is None:
                self.rz0 = angle
                return True
            if self.rz1 is None:
                self.rz1 = angle
                return True
            return False
        assert_never(axis)  # Checks that all cases are covered


@dataclass
class SingleQubitPair(Brick):
    """
    Represents operations on a pair of single-qubits in a brickwork state.

    Attributes
    ----------
    top : SingleQubit
        The single-qubit on the top of the brick.
    bottom : SingleQubit
        The single-qubit on the bottom of the brick.

    Methods
    -------
    get(position: bool) -> SingleQubit:
        Returns the SingleQubit at the specified position (True for bottom, False for top).

    measures() -> list[list[float]]:
        Returns the measurement angles for both single-qubits in the brick.
    """

    top: SingleQubit
    bottom: SingleQubit

    def get(self, position: bool) -> SingleQubit:
        """Return the SingleQubit at the specified position (True for bottom, False for top)."""
        if position:
            return self.bottom
        return self.top

    def measures(self) -> list[list[Angle]]:
        """Return the measurement angles for both single-qubits in the brick."""
        return [self.top.measures(), self.bottom.measures()]

    # def to_circuit(self, circuit: Circuit, nqubit_top: int) -> None:
    #     self.top.to_circuit(circuit, nqubit_top)
    #     self.bottom.to_circuit(circuit, nqubit_top + 1)


def identity() -> SingleQubitPair:
    """Create a SingleQubitPair representing identity operations on both qubits."""
    return SingleQubitPair(SingleQubit(), SingleQubit())


@dataclass
class Layer:
    """
    Represents a layer in a brickwork state.

    In MBQC notation the layer is a vertical stack of bricks, where even bricks are entangled on column 3 mod 8 and odd bricks on 5 mod 8.
    In the case of an odd number of qubits, the layer does not include the half-brick.

    ref: A. Broadbent, J. Fitzsimons, and E. Kashefi, 2009 50th annual IEEE symposium on foundations of computer science. IEEE. (2009).

    Attributes
    ----------
    odd : bool
        Indicates if the layer is odd (True) or even (False) in the brickwork pattern.
    bricks : list[Brick]
        The list of bricks in the layer.

    Methods
    -------
    get(qubit: int) -> tuple[Brick, bool]:
        Returns the brick and position in the brick (True for bottom, False for top) for the specified qubit.
    """

    odd: bool
    bricks: list[Brick]

    def get(self, qubit: int) -> tuple[Brick, bool]:
        """Return the brick and position in the brick (True for bottom, False for top) for the specified qubit."""
        index = (qubit - int(self.odd)) // 2
        return (self.bricks[index], bool(qubit % 2) != self.odd)


def __get_layer(width: int, layers: list[Layer], depth: int) -> Layer:
    """Initialize next required layer(s) as identity operators.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    layers : list[Layer]
        list of current layers in brickwork
    depth : int
        starting depth of the layer(s) to create and the layer to return

    Returns
    -------
    Layer
        the layer at the specified depth
    """
    for i in range(len(layers), depth + 1):
        odd = bool(i % 2)
        layer_size = (width - 1) // 2 if odd else max(width // 2, 1)
        layers.append(
            Layer(
                odd,
                [identity() for _ in range(layer_size)],
            )
        )
    return layers[depth]


def __insert_identity(
    width: int,
    layers: list[Layer],
    depth: list[int],
    instr: instruction.I,
) -> None:
    """
    Insert an identity gate into the brickwork layers.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    layers : list[Layer]
        list of current layers in brickwork
    depth : list[int]
        list of how many bricks deep each qubit is
    instr : instruction.I
        the identity instruction to insert
    """
    target_depth = depth[instr.target]
    if target_depth > 0:
        previous_layer = layers[target_depth - 1]
        brick, position = previous_layer.get(instr.target)
        if isinstance(brick, SingleQubitPair):
            return
        assert isinstance(brick, CNOT)
    if (instr.target == 0 and target_depth % 2) or (
        width >= 2 and instr.target == width - 1 and target_depth % 2 != width % 2
    ):
        target_depth += 1
    layer = __get_layer(width, layers, target_depth)
    brick, position = layer.get(instr.target)
    assert isinstance(brick, SingleQubitPair)
    gate = brick.get(position)
    assert gate.is_identity()
    depth[instr.target] = target_depth + 1


def __insert_rotation(
    width: int,
    layers: list[Layer],
    depth: list[int],
    instr: instruction.RX | instruction.RZ,
) -> None:
    """
    Insert a rotation into the brickwork.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    layers : list[Layer]
        list of current layers in brickwork
    depth : list[int]
        list of how many bricks deep each qubit is
    instr : instruction.RX | instruction.RZ
        the rotation instruction to insert
    """
    axis = XZ.X if instr.kind == InstructionKind.RX else XZ.Z  # TODO What is the point of this?
    target_depth = depth[instr.target]
    if target_depth > 0:
        previous_layer = layers[target_depth - 1]
        brick, position = previous_layer.get(instr.target)
        if isinstance(brick, SingleQubitPair):
            gate = brick.get(position)
            assert isinstance(instr.angle, Angle)
            if gate.add(axis, instr.angle):
                return
        else:
            assert isinstance(brick, CNOT)
    if (instr.target == 0 and target_depth % 2) or (
        width >= 2 and instr.target == width - 1 and target_depth % 2 != width % 2
    ):
        target_depth += 1
    layer = __get_layer(width, layers, target_depth)
    brick, position = layer.get(instr.target)
    assert isinstance(brick, SingleQubitPair)
    gate = brick.get(position)
    assert gate.is_identity()
    added = gate.add(axis, check_expression_or_float(instr.angle))
    assert added
    depth[instr.target] = target_depth + 1


def __insert_cnot(
    width: int,
    layers: list[Layer],
    depth: list[int],
    instr: instruction.CNOT,
) -> None:
    """
    Insert a CNOT as a brick into the brickwork layers.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    layers : list[Layer]
        list of current layers in brickwork
    depth : list[int]
        list of how many bricks deep each qubit is
    instr : instruction.CNOT
        the CNOT instruction to insert
    """
    if abs(instr.control - instr.target) != 1:
        raise ValueError(
            "Unsupported CNOT: control and target qubits should be consecutive"
        )
    top_qubit_index = min(instr.control, instr.target)  # Chooses top qubit for indexation purposes
    bottom_qubit_index = max(instr.control, instr.target)
    min_depth = max(depth[top_qubit_index], depth[bottom_qubit_index])  # Ensures the operation is late enough (e.g. top qubit could be 2 deep already)
    target_depth = min_depth if top_qubit_index % 2 == min_depth % 2 else min_depth + 1
    layer = __get_layer(width, layers, target_depth)
    index = top_qubit_index // 2
    layer.bricks[index] = CNOT(top_qubit_index == instr.target)
    depth[top_qubit_index] = target_depth + 1
    depth[bottom_qubit_index] = target_depth + 1


def transpile_to_layers(circuit: Circuit) -> list[Layer]:
    """
    Transpile a circuit into a list of brickwork layers.

    Parameters
    ----------
    circuit : Circuit
        the circuit to transpile, which should contain only CNOT, RX and RZ gates

    Returns
    -------
    list[Layer]
        the list of layers representing the brickwork state
    """
    layers: list[Layer] = []
    depth = [0 for _ in range(circuit.width)]
    for instr in circuit.instruction:
        # Use of `if` instead of `match` here for mypy
        if instr.kind == InstructionKind.CNOT:
            __insert_cnot(circuit.width, layers, depth, instr)
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind.RX or instr.kind == InstructionKind.RZ:  # noqa: PLR1714
            __insert_rotation(circuit.width, layers, depth, instr)
        elif instr.kind == InstructionKind.I:
            __insert_identity(circuit.width, layers, depth, instr)
        else:
            raise ValueError(
                "Unsupported gate: circuits should contain only CNOT, RX and RZ"
            )
    return layers


@dataclass
class NodeGenerator:
    """Helper class to generate the next unusued node index for building a pattern."""

    from_index: int

    def fresh_command(self) -> tuple[int, command.Command]:
        """Return the next unused node index and a command to prepare it in the |+> state."""
        index = self.from_index
        self.from_index += 1
        return index, N(node=index)

    # def fresh(self, pattern: Pattern) -> int:
    #     """
    #     Add a new command to prepare the next unused node index in the |+> state to the pattern and return the index.

    #     FUNCTION CURRENTLY UNUSED IN THE CODEBASE.

    #     Parameters
    #     ----------
    #     pattern : Pattern
    #         the pattern to which the new command will be added

    #     Returns
    #     -------
    #     int
    #         the index of the newly added node
    #     """
    #     index, command = self.fresh_command()
    #     pattern.add(command)
    #     return index


def j_commands(
    node_generator: NodeGenerator, node: int, angle: Angle) -> tuple[int, list[command.Command]]:
    """Generate the commands for a J(theta) operation on a given node.

    Parameters
    ----------
    node_generator : NodeGenerator
        the NodeGenerator to use for generating new nodes
    node : int
        the node on which to perform the J(theta) operation
    angle : float
        the angle theta for the J(theta) operation

    Returns
    -------
    tuple[int, list[command.Command]]
        a tuple containing the index of the new node and the list of commands to perform the J(theta) operation
    """
    next_node, command_n = node_generator.fresh_command()
    commands: list[command.Command] = [
        command_n,
        E(nodes=(node, next_node)),
        M(node=node, angle=angle / np.pi),
        X(node=next_node, domain={node}),
    ]
    return next_node, commands


class ConstructionOrder(Enum):
    """Enumeration of construction orders for building MBQC measurement patterns from a measurement table.

    Values are used by `typer` in the command-line interface.
    """

    Canonical = "canonical"
    Deviant = "deviant"
    DeviantRight = "deviant-right"


def _nqubits_from_layers(layers: list[Layer]) -> int:
    """Return the number of qubits represented by the given layers.

    Parameters
    ----------
    layers : list[Layer]
        the list of layers representing the brickwork state

    Returns
    -------
    int
        the number of qubits represented by the layers
    """
    if len(layers) == 0:
        raise ValueError("Layer list should not be empty")
    if len(layers) == 1:
        return 2 * len(layers[0].bricks)
    even_brick_count = len(layers[0].bricks)
    odd_brick_count = len(layers[1].bricks)
    return even_brick_count * 2 + int(even_brick_count == odd_brick_count)


def layers_to_measurement_table(layers: list[Layer]) -> list[list[Angle]]:
    """Convert layers of bricks into a measurement table.

    Goes left to right in measurement order, .

    Parameters
    ----------
    layers : list[Layer]
        the list of layers representing the brickwork state

    Returns
    -------
    list[list[Angle]]

    """
    nqubits = _nqubits_from_layers(layers)
    table: list[list[Angle]] = []
    for layer_index, layer in enumerate(layers):
        all_brick_measures = [brick.measures() for brick in layer.bricks]  # Should be a list of two lists of four floats, one for each qubit (top, bottom) in the brick
        for column_index in range(4):
            column: list[Angle] = []
            if layer.odd:
                column.append(0)
            column.extend(
                measures[i][column_index]
                for measures in all_brick_measures
                for i in (0, 1)
            )
            if layer_index % 2 != nqubits % 2:
                column.append(0)
            table.append(column)
    return table


def measurement_table_to_pattern(width: int, table: list[list[Angle]], order: ConstructionOrder) -> Pattern:
    """
    Convert a measurement table into a MBQC measurement pattern.

    This function constructs the measurement pattern according to the specified construction order.
    The construction order determines how entanglement operations are ordered relative to measurement commands.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    table : list[list[Angle]]
        the measurement table, where each sublist represents a column of measurement angles
    order : ConstructionOrder
        the construction order to use for building the pattern

    Returns
    -------
    Pattern
        the resulting MBQC measurement pattern
    """
    input_nodes = list(range(width))
    pattern = Pattern(input_nodes)
    nodes = input_nodes
    node_generator = NodeGenerator(width)
    for time, column in enumerate(table):
        postponed = None  # for deviant order
        for qubit, angle in enumerate(column):
            next_node, commands = j_commands(node_generator, nodes[qubit], angle)
            if (time % 4 in {2, 0} and time > 0) and order != ConstructionOrder.Deviant:
                brick_layer = (time - 1) // 4
                if order == ConstructionOrder.Canonical:
                    if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                        pattern.add(
                            E(nodes=(nodes[qubit], nodes[qubit + 1]))
                        )
                    pattern.extend(commands)
                elif order == ConstructionOrder.DeviantRight:
                    if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                        pattern.extend(commands[:2])
                        postponed = (nodes[qubit], commands[2:])
                    elif postponed is None:
                        pattern.extend(commands)
                    else:
                        pattern.extend(commands[:2])
                        previous_qubit, previous_commands = postponed
                        postponed = None
                        pattern.add(E(nodes=(previous_qubit, nodes[qubit])))
                        pattern.extend(previous_commands)
                        pattern.extend(commands[2:])
            elif time % 4 in {1, 3} and order == ConstructionOrder.Deviant:
                brick_layer = time // 4
                if qubit % 2 == brick_layer % 2 and qubit != width - 1:
                    pattern.add(commands[0])
                    postponed = (nodes[qubit], commands[1:])
                elif postponed is None:
                    pattern.extend(commands)
                else:
                    pattern.add(commands[0])
                    previous_qubit, previous_commands = postponed
                    postponed = None
                    pattern.add(E(nodes=(nodes[qubit - 1], next_node)))
                    pattern.extend(previous_commands)
                    pattern.extend(commands[1:])
                    pattern.extend(
                        [
                            Z(node=nodes[qubit - 1], domain={nodes[qubit]}),
                            Z(node=next_node, domain={previous_qubit}),
                        ]
                    )
            else:
                pattern.extend(commands)
            nodes[qubit] = next_node
    if order != ConstructionOrder.Deviant:
        last_brick_layer = (len(table) - 1) // 4
        for qubit in range(last_brick_layer % 2, width - 1, 2):
            pattern.add(E(nodes=(nodes[qubit], nodes[qubit + 1])))
    return pattern


def layers_to_pattern(width: int, layers: list[Layer], order: ConstructionOrder = ConstructionOrder.Canonical) -> Pattern:
    """
    Convert layers of bricks into a MBQC measurement pattern.

    This is a convenience function that combines `layers_to_measurement_table` and `measurement_table_to_pattern`.

    Parameters
    ----------
    width : int
        number of qubits in the circuit
    layers : list[Layer]
        the list of layers representing the brickwork state
    order : ConstructionOrder, optional
        the construction order to use for building the pattern, by default ConstructionOrder.Canonical

    Returns
    -------
    Pattern
        the resulting MBQC measurement pattern
    """
    table = layers_to_measurement_table(layers)
    return measurement_table_to_pattern(width, table, order)
