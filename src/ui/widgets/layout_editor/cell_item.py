"""
QGraphicsItem for placed primitives in the layout editor.

Renders primitives as colored rectangles with portal indicators,
handles selection and hover states.
"""

from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QStyleOptionGraphicsItem, QWidget,
)
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont

from .data_model import PlacedPrimitive, PrimitiveFootprint, PortalDirection, Portal
from .corridor_shapes import get_corridor_cells


# Color palette for different primitive categories
CATEGORY_COLORS = {
    'Halls': QColor(70, 130, 180),      # Steel blue
    'Structural': QColor(139, 119, 101),  # Tan/brown
    'Rooms': QColor(148, 103, 189),      # Purple
    'Connective': QColor(44, 160, 44),   # Green
    'default': QColor(128, 128, 128),    # Gray
}

# Floor level color tints for multi-level visualization
FLOOR_LEVEL_TINTS = {
    -128: QColor(100, 100, 200),   # Basement: blue tint
    0: None,                        # Ground: no tint (use category color)
    128: QColor(200, 200, 100),    # Upper: yellow tint
    256: QColor(200, 150, 100),    # Tower: orange tint
}

PORTAL_COLORS = {
    'enabled': QColor(76, 175, 80),     # Green
    'disabled': QColor(158, 158, 158),  # Gray
    'connected': QColor(33, 150, 243),  # Blue
}


def _blend_with_floor_tint(base_color: QColor, z_offset: float) -> QColor:
    """Blend base color with floor level tint for multi-level visualization.

    Args:
        base_color: The category base color
        z_offset: The primitive's z_offset value

    Returns:
        Color blended with floor level tint (30% tint, 70% base)
    """
    z_int = int(z_offset)
    tint = FLOOR_LEVEL_TINTS.get(z_int)

    if tint is None:
        # No tint for ground level or unknown levels
        return base_color

    # Blend: 70% base + 30% tint
    r = int(base_color.red() * 0.7 + tint.red() * 0.3)
    g = int(base_color.green() * 0.7 + tint.green() * 0.3)
    b = int(base_color.blue() * 0.7 + tint.blue() * 0.3)

    return QColor(r, g, b)


class CellItem(QGraphicsItem):
    """Graphics item representing a placed primitive on the grid."""

    def __init__(self, primitive: PlacedPrimitive, footprint: PrimitiveFootprint,
                 cell_size: float, category: str = 'default'):
        super().__init__()

        self._primitive = primitive
        self._footprint = footprint
        self._cell_size = cell_size
        self._category = category
        self._selected = False
        self._hovered = False
        self._dragging = False  # True when item is being dragged

        # Set flags
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        # Position in scene coordinates
        self._update_position()

    def _update_position(self):
        """Update position based on origin cell and cell size."""
        x = self._primitive.origin_cell.x * self._cell_size
        y = self._primitive.origin_cell.y * self._cell_size
        # Note: QGraphicsView has Y increasing downward, so we may need to flip
        # For now, keep Y positive upward to match idTech coordinates
        self.setPos(x, -y - self._size_pixels()[1])

    def _size_pixels(self):
        """Get size in pixels after rotation."""
        w, d = self._footprint.rotated_size(self._primitive.rotation)
        return (w * self._cell_size, d * self._cell_size)

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle for the item."""
        w, h = self._size_pixels()
        # Add margin for selection border
        margin = 2
        return QRectF(-margin, -margin, w + margin * 2, h + margin * 2)

    def shape(self):
        """Return shape for hit testing."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        w, h = self._size_pixels()
        path.addRect(0, 0, w, h)
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget = None):
        """Paint the cell item."""
        w, h = self._size_pixels()

        # Get base color from category
        base_color = CATEGORY_COLORS.get(self._category, CATEGORY_COLORS['default'])

        # Apply floor level tint for multi-level visualization
        base_color = _blend_with_floor_tint(base_color, self._primitive.z_offset)

        # Adjust color based on state
        if self._dragging:
            # Dimmed ghost appearance when being dragged
            fill_color = QColor(base_color)
            fill_color.setAlpha(80)
            border_color = QColor(128, 128, 128, 150)
            border_width = 2
            # Apply opacity to painter for entire item
            painter.setOpacity(0.4)
        elif self._selected:
            fill_color = base_color.lighter(130)
            border_color = QColor(255, 193, 7)  # Yellow/amber
            border_width = 3
        elif self._hovered:
            fill_color = base_color.lighter(115)
            border_color = QColor(255, 255, 255)
            border_width = 2
        else:
            fill_color = base_color
            border_color = base_color.darker(130)
            border_width = 1

        # Get rotated footprint dimensions
        fw, fd = self._footprint.rotated_size(self._primitive.rotation)

        # Get corridor cells and rotate them to match primitive rotation
        corridor_cells = self._get_rotated_corridor_cells()
        wall_color = base_color.darker(180)  # Dark for walls
        corridor_color = fill_color  # Bright for corridors

        # Draw individual cells - corridors bright, walls dark
        for cx in range(fw):
            for cy in range(fd):
                # Convert to screen Y (flip Y axis for display)
                screen_cy = fd - 1 - cy
                cell_x = cx * self._cell_size
                cell_y = screen_cy * self._cell_size

                if (cx, cy) in corridor_cells:
                    painter.setBrush(QBrush(corridor_color))
                else:
                    painter.setBrush(QBrush(wall_color))

                painter.setPen(Qt.NoPen)
                painter.drawRect(QRectF(cell_x, cell_y, self._cell_size, self._cell_size))

        # Draw outer border
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(0, 0, w, h))

        # Draw grid lines within the footprint
        grid_pen = QPen(base_color.darker(110), 1, Qt.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, fw):
            x = i * self._cell_size
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
        for j in range(1, fd):
            y = j * self._cell_size
            painter.drawLine(QPointF(0, y), QPointF(w, y))

        # Draw portals
        self._draw_portals(painter, w, h)

        # Draw primitive name
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        # Abbreviate name if needed
        name = self._primitive.primitive_type
        if len(name) > 12:
            name = name[:10] + "..."

        text_rect = QRectF(4, 4, w - 8, h - 8)
        painter.drawText(text_rect, Qt.AlignTop | Qt.AlignLeft, name)

        # Draw rotation indicator
        if self._primitive.rotation != 0:
            rot_text = f"{self._primitive.rotation}°"
            painter.drawText(text_rect, Qt.AlignBottom | Qt.AlignRight, rot_text)

        # Draw Z-offset badge (top-right corner for visibility)
        if self._primitive.z_offset != 0:
            z_text = f"Z:{int(self._primitive.z_offset)}"
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)

            # Use floor level colors for consistency
            z_int = int(self._primitive.z_offset)
            if z_int > 0:
                z_color = FLOOR_LEVEL_TINTS.get(z_int, QColor(100, 180, 255))
                if z_color is None:
                    z_color = QColor(100, 180, 255)  # Default upper color
            else:
                z_color = FLOOR_LEVEL_TINTS.get(z_int, QColor(255, 150, 100))
                if z_color is None:
                    z_color = QColor(255, 150, 100)  # Default basement color

            # Position in top-right corner
            badge_width = 40
            badge_height = 14
            z_rect = QRectF(w - badge_width - 2, 2, badge_width, badge_height)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(z_color))
            painter.drawRoundedRect(z_rect, 2, 2)

            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(z_rect, Qt.AlignCenter, z_text)

        # Draw selection corner markers for enhanced visibility
        if self._selected:
            self._draw_selection_corners(painter, w, h)

    def _draw_portals(self, painter: QPainter, w: float, h: float):
        """Draw portal indicators on edges."""
        portal_size = 8
        portal_length = min(self._cell_size * 0.4, 40)

        for portal in self._footprint.portals:
            # Get rotated direction and cell offset
            direction = portal.rotated_direction(self._primitive.rotation)
            offset = Portal._rotate_offset(
                portal.cell_offset, self._primitive.rotation,
                self._footprint.width_cells, self._footprint.depth_cells
            )

            # Calculate portal position on edge
            ox = offset.x * self._cell_size
            oy = offset.y * self._cell_size

            # Note: Y is flipped in screen coordinates
            cell_center_x = ox + self._cell_size / 2
            cell_center_y = (self._footprint.rotated_size(self._primitive.rotation)[1] - 1 - offset.y) * self._cell_size + self._cell_size / 2

            # Determine portal rectangle based on direction
            if direction == PortalDirection.NORTH:
                px = cell_center_x - portal_length / 2
                py = 0
                pw, ph = portal_length, portal_size
            elif direction == PortalDirection.SOUTH:
                px = cell_center_x - portal_length / 2
                py = h - portal_size
                pw, ph = portal_length, portal_size
            elif direction == PortalDirection.EAST:
                px = w - portal_size
                py = cell_center_y - portal_length / 2
                pw, ph = portal_size, portal_length
            else:  # WEST
                px = 0
                py = cell_center_y - portal_length / 2
                pw, ph = portal_size, portal_length

            # Color based on portal state
            if portal.enabled:
                color = PORTAL_COLORS['enabled']
            else:
                color = PORTAL_COLORS['disabled']

            painter.setPen(QPen(color.darker(120), 1))
            painter.setBrush(QBrush(color))
            painter.drawRect(QRectF(px, py, pw, ph))

            # Draw direction arrow when selected (shows portal direction clearly)
            if self._selected:
                self._draw_portal_arrow(painter, px, py, pw, ph, direction, color)

    def _draw_portal_arrow(self, painter: QPainter, px: float, py: float,
                           pw: float, ph: float, direction: PortalDirection,
                           color: QColor):
        """Draw a directional arrow indicating portal direction.

        Arrow points outward from the portal (the direction it faces).
        """
        from PyQt5.QtGui import QPolygonF

        arrow_size = 6
        cx = px + pw / 2
        cy = py + ph / 2

        # Create arrow polygon pointing outward
        if direction == PortalDirection.NORTH:
            # Arrow points up (out of top edge)
            points = [
                QPointF(cx, py - arrow_size),  # Tip
                QPointF(cx - arrow_size, py + 2),
                QPointF(cx + arrow_size, py + 2),
            ]
        elif direction == PortalDirection.SOUTH:
            # Arrow points down (out of bottom edge)
            points = [
                QPointF(cx, py + ph + arrow_size),  # Tip
                QPointF(cx - arrow_size, py + ph - 2),
                QPointF(cx + arrow_size, py + ph - 2),
            ]
        elif direction == PortalDirection.EAST:
            # Arrow points right (out of right edge)
            points = [
                QPointF(px + pw + arrow_size, cy),  # Tip
                QPointF(px + pw - 2, cy - arrow_size),
                QPointF(px + pw - 2, cy + arrow_size),
            ]
        else:  # WEST
            # Arrow points left (out of left edge)
            points = [
                QPointF(px - arrow_size, cy),  # Tip
                QPointF(px + 2, cy - arrow_size),
                QPointF(px + 2, cy + arrow_size),
            ]

        painter.setPen(QPen(color.lighter(150), 1))
        painter.setBrush(QBrush(color.lighter(130)))
        painter.drawPolygon(QPolygonF(points))

    def _draw_selection_corners(self, painter: QPainter, w: float, h: float):
        """Draw corner markers when selected for enhanced visibility."""
        marker_size = 10
        marker_color = QColor(255, 193, 7)  # Amber

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(marker_color))

        # Top-left corner
        painter.drawRect(QRectF(0, 0, marker_size, 3))
        painter.drawRect(QRectF(0, 0, 3, marker_size))

        # Top-right corner
        painter.drawRect(QRectF(w - marker_size, 0, marker_size, 3))
        painter.drawRect(QRectF(w - 3, 0, 3, marker_size))

        # Bottom-left corner
        painter.drawRect(QRectF(0, h - 3, marker_size, 3))
        painter.drawRect(QRectF(0, h - marker_size, 3, marker_size))

        # Bottom-right corner
        painter.drawRect(QRectF(w - marker_size, h - 3, marker_size, 3))
        painter.drawRect(QRectF(w - 3, h - marker_size, 3, marker_size))

    def _get_rotated_corridor_cells(self) -> set:
        """Get corridor cells rotated to match primitive rotation.

        Returns cell coordinates in the rotated footprint coordinate system.
        """
        # Get base corridor cells (unrotated)
        base_cells = get_corridor_cells(
            self._primitive.primitive_type,
            self._footprint.width_cells,
            self._footprint.depth_cells
        )

        rotation = self._primitive.rotation
        if rotation == 0:
            return base_cells

        # Rotate each cell coordinate
        orig_w = self._footprint.width_cells
        orig_d = self._footprint.depth_cells
        rotated_cells = set()

        for (x, y) in base_cells:
            if rotation == 90:
                # 90° CW: (x, y) -> (y, orig_w - 1 - x)
                new_x = y
                new_y = orig_w - 1 - x
            elif rotation == 180:
                # 180°: (x, y) -> (orig_w - 1 - x, orig_d - 1 - y)
                new_x = orig_w - 1 - x
                new_y = orig_d - 1 - y
            elif rotation == 270:
                # 270° CW (90° CCW): (x, y) -> (orig_d - 1 - y, x)
                new_x = orig_d - 1 - y
                new_y = x
            else:
                new_x, new_y = x, y

            rotated_cells.add((new_x, new_y))

        return rotated_cells

    def set_selected(self, selected: bool):
        """Set selection state."""
        self._selected = selected
        self.update()

    def set_dragging(self, dragging: bool):
        """Set dragging state for visual feedback."""
        self._dragging = dragging
        self.update()

    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    @property
    def primitive(self) -> PlacedPrimitive:
        """Get the underlying primitive."""
        return self._primitive

    @property
    def primitive_id(self) -> str:
        """Get the primitive ID."""
        return self._primitive.id

    def update_from_primitive(self):
        """Update display from primitive data."""
        self._update_position()
        self.update()


class GhostCellItem(QGraphicsItem):
    """Semi-transparent preview item for placement preview."""

    def __init__(self, footprint: PrimitiveFootprint, cell_size: float,
                 category: str = 'default'):
        super().__init__()

        self._footprint = footprint
        self._cell_size = cell_size
        self._category = category
        self._rotation = 0
        self._valid = True

        self.setZValue(-1)  # Behind other items

    def set_rotation(self, rotation: int):
        """Set preview rotation."""
        self._rotation = rotation
        self.update()

    def set_valid(self, valid: bool):
        """Set whether placement is valid."""
        self._valid = valid
        self.update()

    def _size_pixels(self):
        """Get size in pixels after rotation."""
        w, d = self._footprint.rotated_size(self._rotation)
        return (w * self._cell_size, d * self._cell_size)

    def boundingRect(self) -> QRectF:
        w, h = self._size_pixels()
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget = None):
        w, h = self._size_pixels()

        # Get base color
        base_color = CATEGORY_COLORS.get(self._category, CATEGORY_COLORS['default'])

        # Set transparency and validity coloring
        if self._valid:
            fill_color = QColor(base_color)
            fill_color.setAlpha(100)
            border_color = QColor(76, 175, 80, 200)  # Green
        else:
            fill_color = QColor(244, 67, 54, 100)  # Red tint
            border_color = QColor(244, 67, 54, 200)

        painter.setPen(QPen(border_color, 2, Qt.DashLine))
        painter.setBrush(QBrush(fill_color))
        painter.drawRect(QRectF(0, 0, w, h))

        # Draw grid overlay
        grid_pen = QPen(QColor(255, 255, 255, 80), 1, Qt.DotLine)
        painter.setPen(grid_pen)
        fw, fd = self._footprint.rotated_size(self._rotation)
        for i in range(1, fw):
            x = i * self._cell_size
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
        for j in range(1, fd):
            y = j * self._cell_size
            painter.drawLine(QPointF(0, y), QPointF(w, y))
