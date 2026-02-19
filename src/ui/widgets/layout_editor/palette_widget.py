"""
Palette widget for selecting primitives to place in the layout editor.

Displays available primitives organized by category with visual previews.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QToolButton, QSizePolicy,
    QGroupBox, QLayout,
)
from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox

# Alias for compatibility - click to cycle, no dropdown
QComboBox = SafeComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QIcon, QColor, QPainter, QPixmap


class FlowLayout(QLayout):
    """A layout that arranges widgets in a flowing grid, wrapping to next row as needed.

    This ensures all module buttons remain visible regardless of panel width.
    """

    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing if spacing >= 0 else 4
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                      margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(margins.left(), margins.top(),
                                       -margins.right(), -margins.bottom())
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue

            space_x = self._spacing
            space_y = self._spacing

            item_width = item.sizeHint().width()
            item_height = item.sizeHint().height()

            next_x = x + item_width + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + margins.bottom()

from typing import Dict, List, Optional, Callable


# Floor level presets for multi-level dungeons
# These represent the Z-offset in idTech units for each floor
# Separation of 160 units = 128 (room height) + 16 (floor extension) + 16 (ceiling extension)
# This ensures sealed geometry doesn't collide between floors
FLOOR_LEVELS: Dict[str, int] = {
    'Basement': -160,
    'Ground': 0,
    'Upper': +160,
    'Tower': +320,
}

# Floor level colors for visualization
FLOOR_LEVEL_COLORS: Dict[int, QColor] = {
    -160: QColor(100, 100, 200),   # Basement: blue tint
    0: QColor(150, 150, 150),      # Ground: gray
    160: QColor(200, 200, 100),    # Upper: yellow tint
    320: QColor(200, 150, 100),    # Tower: orange tint
}

from .data_model import PrimitiveFootprint, Portal, PortalDirection, CellCoord
from .corridor_shapes import get_corridor_cells
from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.ui.style_constants import set_accessible


# Footprint definitions for all primitives
# These define the grid size and portal positions for each primitive type
PRIMITIVE_FOOTPRINTS: Dict[str, PrimitiveFootprint] = {
    # ==========================================================================
    # HALLS - Corridor segments that connect rooms
    # ==========================================================================
    'StraightHall': PrimitiveFootprint(
        width_cells=1,
        depth_cells=2,
        portals=[
            Portal(id='front', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 1)),
            Portal(id='back', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'SecretHall': PrimitiveFootprint(
        width_cells=1,
        depth_cells=2,
        portals=[
            Portal(id='front', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 1)),
            Portal(id='back', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'TJunction': PrimitiveFootprint(
        width_cells=3,
        depth_cells=2,
        portals=[
            # True T-shape: crossbar WEST-EAST at row 0, stem NORTH at row 1
            Portal(id='west', direction=PortalDirection.WEST, cell_offset=CellCoord(0, 0)),
            Portal(id='east', direction=PortalDirection.EAST, cell_offset=CellCoord(2, 0)),
            Portal(id='north', direction=PortalDirection.NORTH, cell_offset=CellCoord(1, 1)),
        ]
    ),
    'Crossroads': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='north', direction=PortalDirection.NORTH, cell_offset=CellCoord(1, 2)),
            Portal(id='south', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
            Portal(id='east', direction=PortalDirection.EAST, cell_offset=CellCoord(2, 1)),
            Portal(id='west', direction=PortalDirection.WEST, cell_offset=CellCoord(0, 1)),
        ]
    ),
    'SquareCorner': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[
            Portal(id='a', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
            Portal(id='b', direction=PortalDirection.EAST, cell_offset=CellCoord(1, 1)),
        ]
    ),
    'VerticalStairHall': PrimitiveFootprint(
        width_cells=1,  # Geometry is 128 units wide (1 cell), not 2
        depth_cells=4,
        portals=[
            # Bottom portal at lower floor level (z_level=0 relative to primitive)
            Portal(
                id='bottom',
                direction=PortalDirection.SOUTH,
                cell_offset=CellCoord(0, 0),
                z_level=0,  # At primitive's z_offset (lower floor)
            ),
            # Top portal at upper floor level (z_level=160 = one floor up)
            # Must match floor separation to ensure portal Z alignment
            Portal(
                id='top',
                direction=PortalDirection.NORTH,
                cell_offset=CellCoord(0, 3),
                z_level=160,  # At primitive's z_offset + 160 (upper floor)
            ),
        ]
    ),

    # ==========================================================================
    # ROOMS - Enclosed spaces with entrances
    # ==========================================================================
    'Sanctuary': PrimitiveFootprint(
        width_cells=3,
        depth_cells=4,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Tomb': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
            Portal(id='exit', direction=PortalDirection.NORTH, cell_offset=CellCoord(1, 2)),
        ]
    ),
    'Tower': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Chamber': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Storage': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'GreatHall': PrimitiveFootprint(
        width_cells=4,
        depth_cells=6,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
            Portal(id='side', direction=PortalDirection.EAST, cell_offset=CellCoord(3, 3)),
        ]
    ),
    'Prison': PrimitiveFootprint(
        width_cells=4,
        depth_cells=4,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Armory': PrimitiveFootprint(
        width_cells=3,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Cistern': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Stronghold': PrimitiveFootprint(
        width_cells=5,
        depth_cells=5,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(2, 0)),
        ]
    ),
    'Courtyard': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Arena': PrimitiveFootprint(
        width_cells=4,
        depth_cells=4,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Laboratory': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Vault': PrimitiveFootprint(
        width_cells=3,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Barracks': PrimitiveFootprint(
        width_cells=3,
        depth_cells=4,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),
    'Shrine': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'Pit': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'Antechamber': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
            Portal(id='exit', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 1)),
            Portal(id='side_east', direction=PortalDirection.EAST, cell_offset=CellCoord(1, 0)),
            Portal(id='side_west', direction=PortalDirection.WEST, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'SecretChamber': PrimitiveFootprint(
        width_cells=3,
        depth_cells=3,
        portals=[
            Portal(id='entrance', direction=PortalDirection.SOUTH, cell_offset=CellCoord(1, 0)),
        ]
    ),

    # ==========================================================================
    # STRUCTURAL - Interior decorations (not shown in layout palette)
    # ==========================================================================
    'StraightStaircase': PrimitiveFootprint(
        width_cells=1,
        depth_cells=2,
        portals=[
            Portal(id='top', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 1)),
            Portal(id='bottom', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'Arch': PrimitiveFootprint(
        width_cells=1,
        depth_cells=1,
        portals=[
            Portal(id='front', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 0)),
            Portal(id='back', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
        ]
    ),
    'Pillar': PrimitiveFootprint(
        width_cells=1,
        depth_cells=1,
        portals=[]
    ),

    # ==========================================================================
    # CONNECTIVE - Bridges, platforms (not shown in layout palette)
    # ==========================================================================
    'Bridge': PrimitiveFootprint(
        width_cells=1,
        depth_cells=3,
        portals=[
            Portal(id='start', direction=PortalDirection.SOUTH, cell_offset=CellCoord(0, 0)),
            Portal(id='end', direction=PortalDirection.NORTH, cell_offset=CellCoord(0, 2)),
        ]
    ),
    # Note: Platform, Rampart, Gallery are Module-Mode-only primitives.
    # They have footprints here for completeness per CLAUDE.md §8 requirement.
    'Platform': PrimitiveFootprint(
        width_cells=2,
        depth_cells=2,
        portals=[]  # Open platform, no portals
    ),
    'Rampart': PrimitiveFootprint(
        width_cells=1,
        depth_cells=3,
        portals=[]  # Defensive walkway, no portals
    ),
    'Gallery': PrimitiveFootprint(
        width_cells=1,
        depth_cells=4,
        portals=[]  # Elevated walkway, no portals
    ),
    # Buttress, Battlement are Module-Mode-only structural primitives.
    'Buttress': PrimitiveFootprint(
        width_cells=1,
        depth_cells=1,
        portals=[]  # Decorative structural element
    ),
    'Battlement': PrimitiveFootprint(
        width_cells=1,
        depth_cells=1,
        portals=[]  # Defensive crenellation
    ),
}


# Category definitions for layout mode
# Only include primitives that can be connected via portals
PRIMITIVE_CATEGORIES = {
    'Halls': [
        'StraightHall', 'SquareCorner', 'TJunction', 'Crossroads', 'SecretHall',
    ],
    'Rooms': [
        'Sanctuary', 'Tomb', 'Tower', 'Chamber', 'Storage',
        'GreatHall', 'Prison', 'Armory', 'Cistern', 'Stronghold', 'Courtyard',
        'Arena', 'Laboratory', 'Vault', 'Barracks', 'Shrine', 'Pit', 'Antechamber',
        'SecretChamber'
    ],
}

CATEGORY_COLORS = {
    'Halls': QColor(70, 130, 180),
    'Structural': QColor(139, 119, 101),
    'Rooms': QColor(148, 103, 189),
    'Connective': QColor(44, 160, 44),
}


def get_footprint(primitive_type: str) -> Optional[PrimitiveFootprint]:
    """Get the footprint for a primitive type."""
    return PRIMITIVE_FOOTPRINTS.get(primitive_type)


def get_category(primitive_type: str) -> str:
    """Get the category for a primitive type."""
    for cat, prims in PRIMITIVE_CATEGORIES.items():
        if primitive_type in prims:
            return cat
    return 'default'


class PrimitiveButton(QToolButton):
    """Button representing a placeable primitive.

    IMPORTANT: Does NOT use setCheckable(True) to avoid macOS accessibility crash.
    Instead, manages selection state manually via _is_selected.
    """

    clicked_primitive = pyqtSignal(str, PrimitiveFootprint, str)  # type, footprint, category

    def __init__(self, primitive_type: str, footprint: PrimitiveFootprint,
                 category: str, parent=None):
        super().__init__(parent)

        self._primitive_type = primitive_type
        self._footprint = footprint
        self._category = category
        self._is_selected = False  # Manual state tracking - NO setCheckable!

        self._setup_ui()

    def _setup_ui(self):
        """Configure button appearance."""
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setText(self._primitive_type)
        self.setIcon(self._create_icon())
        self.setIconSize(QSize(48, 48))
        self.setFixedSize(80, 80)
        # NO setCheckable! We manage state manually to avoid macOS accessibility crash

        # Add tooltip with footprint dimensions
        portal_count = len(self._footprint.portals)
        portal_text = f"{portal_count} portal{'s' if portal_count != 1 else ''}"
        self.setToolTip(
            f"{self._primitive_type}\n"
            f"Size: {self._footprint.width_cells}×{self._footprint.depth_cells} cells\n"
            f"{portal_text}"
        )

        # Apply initial style
        self._update_style()

        self.clicked.connect(self._on_clicked)

    def _get_style(self) -> str:
        """Get the stylesheet based on current selection state."""
        color = CATEGORY_COLORS.get(self._category, QColor(128, 128, 128))
        if self._is_selected:
            return f"""
                QToolButton {{
                    border: 2px solid {color.name()};
                    border-radius: 4px;
                    background: {color.darker(150).name()};
                    color: #e0e0e0;
                    font-size: 10pt;
                    padding: 4px;
                }}
                QToolButton:hover {{
                    background: {color.darker(130).name()};
                }}
                QToolButton:focus {{
                    outline: 2px solid {color.name()};
                    outline-offset: 2px;
                }}
            """
        else:
            return f"""
                QToolButton {{
                    border: 1px solid #444;
                    border-radius: 4px;
                    background: #2d2d2d;
                    color: #e0e0e0;
                    font-size: 10pt;
                    padding: 4px;
                }}
                QToolButton:hover {{
                    background: #3d3d3d;
                    border-color: {color.name()};
                }}
                QToolButton:focus {{
                    outline: 2px solid {color.name()};
                    outline-offset: 2px;
                }}
            """

    def _update_style(self):
        """Update the stylesheet based on selection state."""
        self.setStyleSheet(self._get_style())

        # Accessibility labels
        portal_count = len(self._footprint.portals)
        set_accessible(
            self,
            self._primitive_type,
            f"{self._footprint.width_cells} by {self._footprint.depth_cells} cells, "
            f"{portal_count} portal{'s' if portal_count != 1 else ''}"
        )

    def _create_icon(self) -> QIcon:
        """Create an icon showing the actual corridor/room shape."""
        size = 48
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get color for category
        color = CATEGORY_COLORS.get(self._category, QColor(128, 128, 128))
        wall_color = color.darker(180)  # Dark color for wall/unused cells
        corridor_color = color  # Bright color for corridor cells

        # Calculate cell size to fit footprint
        max_cells = max(self._footprint.width_cells, self._footprint.depth_cells)
        cell_px = (size - 8) // max_cells

        # Center the footprint
        total_w = self._footprint.width_cells * cell_px
        total_h = self._footprint.depth_cells * cell_px
        offset_x = (size - total_w) // 2
        offset_y = (size - total_h) // 2

        # Get the corridor cells for this primitive type (cells that are "open")
        corridor_cells = self._get_corridor_cells()

        # Draw cells - corridors bright, walls dark
        for cx in range(self._footprint.width_cells):
            for cy in range(self._footprint.depth_cells):
                # Convert to screen Y (flip Y axis)
                screen_cy = self._footprint.depth_cells - 1 - cy
                cell_x = offset_x + cx * cell_px
                cell_y = offset_y + screen_cy * cell_px

                if (cx, cy) in corridor_cells:
                    painter.setBrush(corridor_color)
                else:
                    painter.setBrush(wall_color)
                painter.setPen(Qt.NoPen)
                painter.drawRect(cell_x, cell_y, cell_px, cell_px)

        # Draw grid lines
        painter.setPen(color.darker(130))
        for i in range(self._footprint.width_cells + 1):
            x = offset_x + i * cell_px
            painter.drawLine(x, offset_y, x, offset_y + total_h)
        for j in range(self._footprint.depth_cells + 1):
            y = offset_y + j * cell_px
            painter.drawLine(offset_x, y, offset_x + total_w, y)

        # Draw portal indicators on cell edges
        portal_color = QColor(76, 175, 80)
        painter.setBrush(portal_color)
        painter.setPen(Qt.NoPen)

        for portal in self._footprint.portals:
            # Calculate cell position
            cx = portal.cell_offset.x
            cy = portal.cell_offset.y
            screen_cy = self._footprint.depth_cells - 1 - cy
            cell_x = offset_x + cx * cell_px
            cell_y = offset_y + screen_cy * cell_px

            # Draw portal indicator on the appropriate edge
            portal_size = max(4, cell_px // 4)
            if portal.direction == PortalDirection.NORTH:
                # Top edge of cell
                px = cell_x + cell_px // 2 - portal_size // 2
                py = cell_y - portal_size // 2
                painter.drawRect(px, py, portal_size, portal_size)
            elif portal.direction == PortalDirection.SOUTH:
                # Bottom edge of cell
                px = cell_x + cell_px // 2 - portal_size // 2
                py = cell_y + cell_px - portal_size // 2
                painter.drawRect(px, py, portal_size, portal_size)
            elif portal.direction == PortalDirection.EAST:
                # Right edge of cell
                px = cell_x + cell_px - portal_size // 2
                py = cell_y + cell_px // 2 - portal_size // 2
                painter.drawRect(px, py, portal_size, portal_size)
            elif portal.direction == PortalDirection.WEST:
                # Left edge of cell
                px = cell_x - portal_size // 2
                py = cell_y + cell_px // 2 - portal_size // 2
                painter.drawRect(px, py, portal_size, portal_size)

        painter.end()
        return QIcon(pixmap)

    def _get_corridor_cells(self) -> set:
        """Get the set of (x, y) cell coordinates that are corridor (not walls).

        For halls, this traces the actual corridor path between portals.
        For rooms, all cells are part of the room interior.
        """
        return get_corridor_cells(
            self._primitive_type,
            self._footprint.width_cells,
            self._footprint.depth_cells
        )

    def _on_clicked(self):
        """Emit signal when clicked."""
        # Select this button when clicked
        self._is_selected = True
        self._update_style()
        self.clicked_primitive.emit(
            self._primitive_type, self._footprint, self._category
        )

    def setChecked(self, checked: bool):
        """Set selection state manually (compatibility with Qt API)."""
        if self._is_selected != checked:
            self._is_selected = checked
            self._update_style()

    def isChecked(self) -> bool:
        """Get selection state (compatibility with Qt API)."""
        return self._is_selected


class PaletteWidget(QWidget):
    """Widget for selecting primitives to place."""

    primitive_selected = pyqtSignal(str, PrimitiveFootprint, str)  # type, footprint, category
    selection_cleared = pyqtSignal()
    floor_level_changed = pyqtSignal(int)  # Emitted with z_offset when floor level changes

    def __init__(self, parent=None):
        super().__init__(parent)

        self._buttons: Dict[str, PrimitiveButton] = {}
        self._current_selection: Optional[str] = None
        self._current_floor_level: int = 0  # Default to Ground level

        self._setup_ui()

    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Note: Floor level selection removed - users set Z-offset via property editor
        # and use 3D view to visualize vertical relationships

        # Title
        title = QLabel("Modules")
        title.setStyleSheet("font-weight: bold; font-size: 11pt; color: #ccc;")
        layout.addWidget(title)

        # Scroll area for categories
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Add categories
        for category, primitives in PRIMITIVE_CATEGORIES.items():
            self._add_category(scroll_layout, category, primitives)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Clear selection button
        clear_btn = QPushButton("Clear Selection")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 6px 12px;
                background: #444;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background: #555;
            }}
            QPushButton:focus {{
                outline: 2px solid {sc.FOCUS_COLOR};
                outline-offset: 2px;
            }}
        """)
        clear_btn.clicked.connect(self._on_clear)
        set_accessible(clear_btn, "Clear Selection",
                      "Deselect the current primitive")
        layout.addWidget(clear_btn)

    def _add_category(self, parent_layout: QVBoxLayout, category: str,
                      primitives: List[str]):
        """Add a category section."""
        # Category header - use lighter color variant for better contrast
        header = QLabel(category)
        color = CATEGORY_COLORS.get(category, QColor(128, 128, 128))
        # Lighten the color for better contrast on dark background
        lighter_color = color.lighter(130)
        header.setStyleSheet(f"""
            font-weight: bold;
            font-size: 11pt;
            color: {lighter_color.name()};
            padding: 4px 4px;
            border-bottom: 1px solid {color.name()};
        """)
        parent_layout.addWidget(header)

        # Flow layout for primitive buttons - wraps to next row when panel is narrow
        flow_container = QWidget()
        flow_layout = FlowLayout(flow_container, margin=0, spacing=4)

        for prim_type in primitives:
            footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
            if footprint is None:
                continue

            btn = PrimitiveButton(prim_type, footprint, category)
            btn.clicked_primitive.connect(self._on_primitive_clicked)
            self._buttons[prim_type] = btn
            flow_layout.addWidget(btn)

        parent_layout.addWidget(flow_container)

    def _on_floor_level_changed(self, level_name: str):
        """Handle floor level change (legacy - floor UI removed)."""
        # Floor level UI removed - users set Z-offset via property editor
        z_offset = FLOOR_LEVELS.get(level_name, 0)
        self._current_floor_level = z_offset
        self.floor_level_changed.emit(z_offset)

    def _on_primitive_clicked(self, prim_type: str, footprint: PrimitiveFootprint,
                              category: str):
        """Handle primitive button click."""
        # Deselect all other buttons (manual exclusive selection)
        for btn_type, btn in self._buttons.items():
            if btn_type != prim_type:
                btn.setChecked(False)

        self._current_selection = prim_type
        self.primitive_selected.emit(prim_type, footprint, category)

    def _on_clear(self):
        """Clear selection."""
        self._current_selection = None

        # Uncheck all buttons
        for btn in self._buttons.values():
            btn.setChecked(False)

        self.selection_cleared.emit()

    def get_selected(self) -> Optional[str]:
        """Get currently selected primitive type."""
        return self._current_selection

    def select_primitive(self, prim_type: str):
        """Programmatically select a primitive."""
        if prim_type in self._buttons:
            self._buttons[prim_type].setChecked(True)
            self._buttons[prim_type]._on_clicked()

    def clear_selection(self):
        """Clear selection programmatically."""
        self._on_clear()

    def get_current_floor_level(self) -> int:
        """Get the currently selected floor level Z offset.

        Note: Floor level UI removed. Always returns 0 (Ground level).
        Users set Z-offset via property editor after placement.
        """
        return 0

    def get_current_floor_level_name(self) -> str:
        """Get the name of the currently selected floor level.

        Note: Floor level UI removed. Always returns 'Ground'.
        """
        return 'Ground'

    def set_floor_level(self, level_name: str):
        """Set the floor level by name (no-op, floor UI removed)."""
        pass
