"""
Command pattern implementation for undo/redo in the layout editor.

Provides:
- Command ABC for all layout operations
- CommandManager for undo/redo stack management
- Concrete commands: Place, Delete, Move, Rotate, SetZOffset
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from .data_model import DungeonLayout, PlacedPrimitive, CellCoord, PrimitiveFootprint


class Command(ABC):
    """Abstract base class for undoable commands."""

    @abstractmethod
    def execute(self, layout: 'DungeonLayout') -> bool:
        """
        Execute the command.

        Returns:
            True if execution succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def undo(self, layout: 'DungeonLayout') -> bool:
        """
        Undo the command.

        Returns:
            True if undo succeeded, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this command."""
        pass


@dataclass
class PlacePrimitiveCommand(Command):
    """Command to place a new primitive on the grid."""

    primitive_type: str
    origin_x: int
    origin_y: int
    rotation: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    z_offset: float = 0.0
    footprint: Optional['PrimitiveFootprint'] = None

    # Set after execution
    _placed_id: Optional[str] = field(default=None, repr=False)
    _auto_connections: List[Any] = field(default_factory=list, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        from .data_model import PlacedPrimitive, CellCoord

        origin = CellCoord(self.origin_x, self.origin_y)

        # Check if placement is valid
        if self.footprint and not layout.can_place_at(self.footprint, origin, self.rotation):
            return False

        # Create the primitive
        prim = PlacedPrimitive.create(
            primitive_type=self.primitive_type,
            origin=origin,
            rotation=self.rotation,
            parameters=self.parameters,
            z_offset=self.z_offset,
        )

        if self.footprint:
            prim.set_footprint(self.footprint)

        # Add to layout
        if not layout.add_primitive(prim):
            return False

        self._placed_id = prim.id

        # Auto-connect and save connections for undo
        connections_before = len(layout.connections)
        layout.auto_connect(prim.id)
        self._auto_connections = layout.connections[connections_before:]

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self._placed_id is None:
            return False

        # Remove the primitive (also removes connections)
        return layout.remove_primitive(self._placed_id)

    @property
    def description(self) -> str:
        return f"Place {self.primitive_type} at ({self.origin_x}, {self.origin_y})"

    @property
    def placed_id(self) -> Optional[str]:
        """Get the ID of the placed primitive (available after execute)."""
        return self._placed_id


@dataclass
class DeletePrimitiveCommand(Command):
    """Command to delete a primitive from the grid."""

    primitive_id: str

    # Saved state for undo
    _saved_primitive: Optional['PlacedPrimitive'] = field(default=None, repr=False)
    _saved_connections: List[Any] = field(default_factory=list, repr=False)
    _saved_footprint: Optional['PrimitiveFootprint'] = field(default=None, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        # Save the primitive for undo
        prim = layout.primitives[self.primitive_id]
        self._saved_primitive = copy.deepcopy(prim)
        self._saved_footprint = prim.footprint

        # Save connections involving this primitive
        self._saved_connections = [
            copy.deepcopy(c) for c in layout.connections
            if c.primitive_a_id == self.primitive_id or c.primitive_b_id == self.primitive_id
        ]

        # Remove the primitive
        return layout.remove_primitive(self.primitive_id)

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self._saved_primitive is None:
            return False

        # Restore the primitive
        prim = copy.deepcopy(self._saved_primitive)
        if self._saved_footprint:
            prim.set_footprint(self._saved_footprint)

        layout.primitives[prim.id] = prim

        # Restore connections
        for conn in self._saved_connections:
            layout.connections.append(copy.deepcopy(conn))

        return True

    @property
    def description(self) -> str:
        if self._saved_primitive:
            return f"Delete {self._saved_primitive.primitive_type}"
        return f"Delete primitive {self.primitive_id[:8]}..."


@dataclass
class MovePrimitiveCommand(Command):
    """Command to move a primitive to a new position."""

    primitive_id: str
    new_origin_x: int
    new_origin_y: int

    # Saved state for undo
    _old_origin_x: Optional[int] = field(default=None, repr=False)
    _old_origin_y: Optional[int] = field(default=None, repr=False)
    _old_connections: List[Any] = field(default_factory=list, repr=False)
    _new_connections: List[Any] = field(default_factory=list, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        from .data_model import CellCoord

        if self.primitive_id not in layout.primitives:
            return False

        prim = layout.primitives[self.primitive_id]
        footprint = prim.footprint

        # Save old position
        self._old_origin_x = prim.origin_cell.x
        self._old_origin_y = prim.origin_cell.y

        # Check if new position is valid (excluding self)
        new_origin = CellCoord(self.new_origin_x, self.new_origin_y)

        if footprint:
            # Temporarily remove primitive to check placement
            new_cells = set(footprint.occupied_cells(new_origin, prim.rotation))
            for other_id, other_prim in layout.primitives.items():
                if other_id == self.primitive_id:
                    continue
                other_cells = set(other_prim.occupied_cells())
                if new_cells & other_cells:
                    return False  # Collision

        # Save old connections
        self._old_connections = [
            copy.deepcopy(c) for c in layout.connections
            if c.primitive_a_id == self.primitive_id or c.primitive_b_id == self.primitive_id
        ]

        # Remove old connections
        layout.connections = [
            c for c in layout.connections
            if c.primitive_a_id != self.primitive_id and c.primitive_b_id != self.primitive_id
        ]

        # Move the primitive
        prim.origin_cell = new_origin

        # Re-establish connections at new position
        connections_before = len(layout.connections)
        layout.auto_connect(self.primitive_id)
        self._new_connections = layout.connections[connections_before:]

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        from .data_model import CellCoord

        if self.primitive_id not in layout.primitives:
            return False

        if self._old_origin_x is None or self._old_origin_y is None:
            return False

        prim = layout.primitives[self.primitive_id]

        # Remove new connections
        layout.connections = [
            c for c in layout.connections
            if c.primitive_a_id != self.primitive_id and c.primitive_b_id != self.primitive_id
        ]

        # Restore position
        prim.origin_cell = CellCoord(self._old_origin_x, self._old_origin_y)

        # Restore old connections
        for conn in self._old_connections:
            layout.connections.append(copy.deepcopy(conn))

        return True

    @property
    def description(self) -> str:
        if self._old_origin_x is not None:
            return f"Move from ({self._old_origin_x}, {self._old_origin_y}) to ({self.new_origin_x}, {self.new_origin_y})"
        return f"Move to ({self.new_origin_x}, {self.new_origin_y})"


@dataclass
class RotatePrimitiveCommand(Command):
    """Command to rotate a primitive."""

    primitive_id: str
    new_rotation: int  # 0, 90, 180, 270

    # Saved state for undo
    _old_rotation: Optional[int] = field(default=None, repr=False)
    _old_connections: List[Any] = field(default_factory=list, repr=False)
    _new_connections: List[Any] = field(default_factory=list, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        prim = layout.primitives[self.primitive_id]
        footprint = prim.footprint

        # Save old rotation
        self._old_rotation = prim.rotation

        # Check if rotated footprint fits
        if footprint:
            # Check for collisions at new rotation
            new_cells = set(footprint.occupied_cells(prim.origin_cell, self.new_rotation))
            for other_id, other_prim in layout.primitives.items():
                if other_id == self.primitive_id:
                    continue
                other_cells = set(other_prim.occupied_cells())
                if new_cells & other_cells:
                    return False  # Collision

        # Save old connections
        self._old_connections = [
            copy.deepcopy(c) for c in layout.connections
            if c.primitive_a_id == self.primitive_id or c.primitive_b_id == self.primitive_id
        ]

        # Remove old connections
        layout.connections = [
            c for c in layout.connections
            if c.primitive_a_id != self.primitive_id and c.primitive_b_id != self.primitive_id
        ]

        # Apply rotation
        prim.rotation = self.new_rotation

        # Re-establish connections
        connections_before = len(layout.connections)
        layout.auto_connect(self.primitive_id)
        self._new_connections = layout.connections[connections_before:]

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        if self._old_rotation is None:
            return False

        prim = layout.primitives[self.primitive_id]

        # Remove new connections
        layout.connections = [
            c for c in layout.connections
            if c.primitive_a_id != self.primitive_id and c.primitive_b_id != self.primitive_id
        ]

        # Restore rotation
        prim.rotation = self._old_rotation

        # Restore old connections
        for conn in self._old_connections:
            layout.connections.append(copy.deepcopy(conn))

        return True

    @property
    def description(self) -> str:
        if self._old_rotation is not None:
            return f"Rotate from {self._old_rotation}° to {self.new_rotation}°"
        return f"Rotate to {self.new_rotation}°"


@dataclass
class SetZOffsetCommand(Command):
    """Command to change a primitive's Z offset."""

    primitive_id: str
    new_z_offset: float

    # Saved state for undo
    _old_z_offset: Optional[float] = field(default=None, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        prim = layout.primitives[self.primitive_id]
        self._old_z_offset = prim.z_offset
        prim.z_offset = self.new_z_offset

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        if self._old_z_offset is None:
            return False

        prim = layout.primitives[self.primitive_id]
        prim.z_offset = self._old_z_offset

        return True

    @property
    def description(self) -> str:
        if self._old_z_offset is not None:
            return f"Change Z offset from {self._old_z_offset} to {self.new_z_offset}"
        return f"Set Z offset to {self.new_z_offset}"


@dataclass
class DuplicatePrimitiveCommand(Command):
    """Command to duplicate a primitive with an offset."""

    source_primitive_id: str
    offset_x: int = 1  # Offset in cells from source
    offset_y: int = 0

    # Set after execution
    _duplicated_id: Optional[str] = field(default=None, repr=False)

    def execute(self, layout: 'DungeonLayout') -> bool:
        from .data_model import PlacedPrimitive, CellCoord

        if self.source_primitive_id not in layout.primitives:
            return False

        source = layout.primitives[self.source_primitive_id]

        # Calculate new position
        new_origin = CellCoord(
            source.origin_cell.x + self.offset_x,
            source.origin_cell.y + self.offset_y
        )

        # Check if placement is valid
        footprint = source.footprint
        if footprint and not layout.can_place_at(footprint, new_origin, source.rotation):
            return False

        # Create duplicate
        prim = PlacedPrimitive.create(
            primitive_type=source.primitive_type,
            origin=new_origin,
            rotation=source.rotation,
            parameters=copy.deepcopy(source.parameters),
            z_offset=source.z_offset,
        )

        if footprint:
            prim.set_footprint(footprint)

        # Add to layout
        if not layout.add_primitive(prim):
            return False

        self._duplicated_id = prim.id
        layout.auto_connect(prim.id)

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self._duplicated_id is None:
            return False

        return layout.remove_primitive(self._duplicated_id)

    @property
    def description(self) -> str:
        # Include source type if we can get it from the layout
        return f"Duplicate primitive at offset ({self.offset_x}, {self.offset_y})"

    @property
    def duplicated_id(self) -> Optional[str]:
        """Get the ID of the duplicated primitive (available after execute)."""
        return self._duplicated_id


@dataclass
class SetPrimitiveParameterCommand(Command):
    """Command to change a parameter value on a placed primitive.

    Enables full parameter editing in Layout Mode with undo/redo support.
    User parameter values take precedence over computed defaults.
    """

    primitive_id: str
    param_key: str
    new_value: Any

    # Saved state for undo
    _old_value: Optional[Any] = field(default=None, repr=False)
    _had_value: bool = field(default=False, repr=False)  # Track if key existed before

    def execute(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        prim = layout.primitives[self.primitive_id]

        # Save old state for undo
        self._had_value = self.param_key in prim.parameters
        self._old_value = prim.parameters.get(self.param_key)

        # Set the new value
        prim.parameters[self.param_key] = self.new_value

        return True

    def undo(self, layout: 'DungeonLayout') -> bool:
        if self.primitive_id not in layout.primitives:
            return False

        prim = layout.primitives[self.primitive_id]

        if self._had_value:
            # Restore the old value
            prim.parameters[self.param_key] = self._old_value
        else:
            # Remove the key if it didn't exist before
            prim.parameters.pop(self.param_key, None)

        return True

    @property
    def description(self) -> str:
        return f"Set {self.param_key} to {self.new_value}"


class CommandManager:
    """
    Manages command execution with undo/redo stacks.

    Usage:
        manager = CommandManager()
        manager.execute(PlacePrimitiveCommand(...), layout)
        manager.undo(layout)  # Undo the place
        manager.redo(layout)  # Redo the place
    """

    def __init__(self, max_undo_depth: int = 100):
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._max_depth = max_undo_depth

    def execute(self, command: Command, layout: 'DungeonLayout') -> bool:
        """
        Execute a command and add it to the undo stack.

        Returns:
            True if command executed successfully.
        """
        if command.execute(layout):
            self._undo_stack.append(command)
            self._redo_stack.clear()  # Clear redo stack on new command

            # Limit stack size
            if len(self._undo_stack) > self._max_depth:
                self._undo_stack.pop(0)

            return True
        return False

    def undo(self, layout: 'DungeonLayout') -> Optional[str]:
        """
        Undo the last command.

        Returns:
            Description of undone command, or None if nothing to undo.
        """
        if not self._undo_stack:
            return None

        command = self._undo_stack.pop()
        if command.undo(layout):
            self._redo_stack.append(command)
            return command.description
        else:
            # Undo failed, put command back
            self._undo_stack.append(command)
            return None

    def redo(self, layout: 'DungeonLayout') -> Optional[str]:
        """
        Redo the last undone command.

        Returns:
            Description of redone command, or None if nothing to redo.
        """
        if not self._redo_stack:
            return None

        command = self._redo_stack.pop()
        if command.execute(layout):
            self._undo_stack.append(command)
            return command.description
        else:
            # Redo failed, put command back
            self._redo_stack.append(command)
            return None

    @property
    def can_undo(self) -> bool:
        """Check if there are commands to undo."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Check if there are commands to redo."""
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> Optional[str]:
        """Get description of command that would be undone."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return None

    @property
    def redo_description(self) -> Optional[str]:
        """Get description of command that would be redone."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return None

    def clear(self):
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def undo_count(self) -> int:
        """Number of commands in undo stack."""
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        """Number of commands in redo stack."""
        return len(self._redo_stack)
