"""
Node-based layout editor for dungeon design.

Provides a grid-based canvas for placing and connecting primitives
to create complex dungeon layouts.
"""

from .data_model import (
    CellCoord,
    PortalDirection,
    Portal,
    PlacedPrimitive,
    DungeonLayout,
    PrimitiveFootprint,
)
from .grid_canvas import GridCanvas
from .cell_item import CellItem
from .palette_widget import PaletteWidget, get_footprint, get_category
from .layout_editor_widget import LayoutEditorWidget
from .validation import ValidationResult, ValidationIssue, validate_layout
from .validation_panel import ValidationPanel
from .layout_generator import LayoutGenerator, GenerationResult, generate_from_layout
from .commands import (
    Command,
    CommandManager,
    PlacePrimitiveCommand,
    DeletePrimitiveCommand,
    MovePrimitiveCommand,
    RotatePrimitiveCommand,
    SetZOffsetCommand,
    DuplicatePrimitiveCommand,
)

__all__ = [
    'CellCoord',
    'PortalDirection',
    'Portal',
    'PlacedPrimitive',
    'DungeonLayout',
    'PrimitiveFootprint',
    'GridCanvas',
    'CellItem',
    'PaletteWidget',
    'get_footprint',
    'get_category',
    'LayoutEditorWidget',
    'ValidationResult',
    'ValidationIssue',
    'validate_layout',
    'ValidationPanel',
    'LayoutGenerator',
    'GenerationResult',
    'generate_from_layout',
    'Command',
    'CommandManager',
    'PlacePrimitiveCommand',
    'DeletePrimitiveCommand',
    'MovePrimitiveCommand',
    'RotatePrimitiveCommand',
    'SetZOffsetCommand',
    'DuplicatePrimitiveCommand',
]
