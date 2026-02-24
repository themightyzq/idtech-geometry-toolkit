"""
QPainter-based layout flow visualization for Front (XZ) and Side (YZ) views.

Shows modules as colored rectangles at their world positions with Z-height,
and connections as lines between portals. Helps users visualize vertical
layout of multi-floor dungeons without requiring OpenGL.

Interactive: click to select modules, drag to reposition (horizontal cell + vertical Z-offset).
"""

from enum import Enum
from typing import Optional, Dict, Tuple, List

from PyQt5.QtWidgets import QWidget, QMenu, QAction
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QWheelEvent,
    QMouseEvent, QPainterPath, QCursor,
)

from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.generators.primitives.portal_system import PORTAL_HEIGHT
from quake_levelgenerator.src.ui.widgets.layout_editor.data_model import PortalDirection, CellCoord


class FlowViewAxis(Enum):
    """Which plane this flow view shows."""
    FRONT = "front"   # XZ plane
    SIDE = "side"     # YZ plane


# Category colors matching cell_item.py
_CATEGORY_COLORS = {
    'Halls': QColor(70, 130, 180),
    'Structural': QColor(139, 119, 101),
    'Rooms': QColor(148, 103, 189),
    'Multi-Floor Rooms': QColor(200, 120, 180),
    'Connective': QColor(44, 160, 44),
    'default': QColor(128, 128, 128),
}

# Connection type colors — must match grid_canvas.py
CONN_SECRET = QColor(220, 20, 60, 220)        # Crimson red
CONN_HORIZONTAL = QColor(33, 150, 243, 180)   # Blue
CONN_VERTICAL = QColor(156, 39, 176, 180)     # Purple
CONN_MISMATCH = QColor(255, 152, 0, 200)      # Orange

# Portal colors — must match cell_item.py PORTAL_COLORS
PORTAL_ENABLED = QColor(76, 175, 80)          # Green
PORTAL_DISABLED = QColor(158, 158, 158)       # Gray
PORTAL_UPPER_LEVEL = QColor(255, 152, 0)      # Orange

# Map primitive types to categories (built lazily)
_TYPE_TO_CATEGORY: Optional[Dict[str, str]] = None


def _get_category(primitive_type: str) -> str:
    """Get the category for a primitive type."""
    global _TYPE_TO_CATEGORY
    if _TYPE_TO_CATEGORY is None:
        from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import (
            PRIMITIVE_CATEGORIES,
        )
        _TYPE_TO_CATEGORY = {}
        for cat, prims in PRIMITIVE_CATEGORIES.items():
            for p in prims:
                _TYPE_TO_CATEGORY[p] = cat
    return _TYPE_TO_CATEGORY.get(primitive_type, 'default')


def _get_color(primitive_type: str) -> QColor:
    """Get the color for a primitive type based on its category."""
    cat = _get_category(primitive_type)
    return _CATEGORY_COLORS.get(cat, _CATEGORY_COLORS['default'])


# Default heights per module type (from get_parameter_schema() defaults)
_MODULE_DEFAULT_HEIGHTS = {
    # Halls — all 128
    'StraightHall': 128, 'TJunction': 128, 'Crossroads': 128,
    'SquareCorner': 128, 'VerticalStairHall': 128, 'SecretHall': 128,
    # Standard rooms
    'Sanctuary': 192, 'Tomb': 96, 'Tower': 384, 'Chamber': 128,
    'Storage': 112, 'GreatHall': 192, 'Prison': 96, 'Armory': 128,
    'Cistern': 128, 'Stronghold': 384, 'Courtyard': 192, 'Arena': 128,
    'Laboratory': 128, 'Vault': 112, 'Barracks': 112, 'Shrine': 112,
    'Pit': 96, 'Antechamber': 112, 'SecretChamber': 128,
    # Multi-floor rooms
    'Amphitheater': 192, 'CatwalkChamber': 160, 'BalconyRoom': 160,
    'SunkenChamber': 128, 'LibraryArchive': 320, 'Grotto': 160,
    'RadialShrine': 192, 'Forge': 192,
}

# Maps module type → parameter key for height in prim.parameters
_HEIGHT_PARAM_KEYS = {
    'Sanctuary': 'nave_height',
    'Tomb': 'tomb_height',
    'Tower': 'tower_height',
    'LibraryArchive': 'room_height',
    'Storage': 'ceiling_height',  # preset: "low"=80, "normal"=112, "tall"=144
}
# Halls use 'hall_height'; everything else uses 'height'

_STORAGE_HEIGHT_PRESETS = {"low": 80, "normal": 112, "tall": 144}

# Fallback for unknown types
_DEFAULT_MODULE_HEIGHT = 128

_HALL_TYPES = frozenset((
    'StraightHall', 'TJunction', 'Crossroads',
    'SquareCorner', 'VerticalStairHall', 'SecretHall',
))


def _get_module_height(prim) -> float:
    """Get the visual height for a placed primitive.

    Checks user parameter overrides first, then falls back to default lookup.
    """
    ptype = prim.primitive_type
    params = prim.parameters

    # Check user overrides via the correct parameter key
    if ptype in _HEIGHT_PARAM_KEYS:
        key = _HEIGHT_PARAM_KEYS[ptype]
        if key in params:
            val = params[key]
            if ptype == 'Storage':
                return _STORAGE_HEIGHT_PRESETS.get(
                    val, _MODULE_DEFAULT_HEIGHTS.get(ptype, _DEFAULT_MODULE_HEIGHT))
            return float(val)
    elif ptype in _HALL_TYPES:
        if 'hall_height' in params:
            return float(params['hall_height'])
    elif ptype == 'Stronghold':
        if 'levels' in params or 'level_height' in params:
            levels = params.get('levels', 3)
            level_h = params.get('level_height', 128)
            return float(levels * level_h)
    else:
        if 'height' in params:
            return float(params['height'])

    return _MODULE_DEFAULT_HEIGHTS.get(ptype, _DEFAULT_MODULE_HEIGHT)

# Minimum drag distance (pixels) to start a Z-drag
_DRAG_THRESHOLD = 5

# Z-offset snap increment (world units)
_Z_SNAP = 16.0


class LayoutFlowView(QWidget):
    """Schematic flow view showing module positions from Front (XZ) or Side (YZ).

    Draws modules as colored rectangles at their world X/Y + Z positions,
    with connection lines between portals. No OpenGL required.

    Interactive:
    - Left-click to select a module
    - Left-drag to reposition (horizontal = cell move, vertical = Z-offset)
    - Middle/right-drag to pan, scroll to zoom
    - F key to fit view to layout
    """

    view_updated = pyqtSignal()
    primitive_selected = pyqtSignal(str)  # primitive_id
    selection_cleared = pyqtSignal()
    command_requested = pyqtSignal(object)  # Command (SetZOffsetCommand or MovePrimitiveCommand)
    layout_changed = pyqtSignal()  # Emitted when layout is modified (e.g. toggle secret)

    def __init__(self, axis: FlowViewAxis, parent=None):
        super().__init__(parent)
        self._axis = axis
        self._layout = None  # DungeonLayout

        # View transform: world -> screen
        self._zoom = 0.5   # pixels per world unit
        self._pan_x = 0.0  # world-space center X
        self._pan_z = 0.0  # world-space center Z

        # Mouse state
        self._last_mouse_pos = None
        self._mouse_button = None

        # Selection state
        self._selected_prim_id: Optional[str] = None
        self._hover_prim_id: Optional[str] = None

        # Drag state for Z-offset and cell-position editing
        self._dragging = False
        self._drag_start_pos = None  # Screen position at drag start
        self._drag_start_z: float = 0.0  # Z-offset at drag start
        self._drag_start_cell_x: int = 0  # Cell X at drag start
        self._drag_start_cell_y: int = 0  # Cell Y at drag start
        self._drag_prim_id: Optional[str] = None  # Primitive being dragged

        # Right-click tracking (distinguish click from drag for context menu)
        self._right_dragged = False

        # Grid
        self._grid_spacing = 128.0  # world units

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # Dark background
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(40, 40, 40))
        self.setPalette(pal)

    def set_layout(self, layout):
        """Set the DungeonLayout to visualize."""
        self._layout = layout
        self._selected_prim_id = None
        self._hover_prim_id = None
        self._dragging = False
        self.update()

    def set_selected(self, prim_id: Optional[str]):
        """Set the selected primitive (for external sync)."""
        if self._selected_prim_id != prim_id:
            self._selected_prim_id = prim_id
            self.update()

    def fit_to_layout(self):
        """Auto-fit the view to show all modules."""
        if not self._layout or not self._layout.primitives:
            return

        # Compute bounding box of all modules in this view's axes
        min_h, max_h = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        grid_size = self._layout.grid_size

        for prim in self._layout.primitives.values():
            fp = prim.footprint
            if fp is None:
                from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import (
                    PRIMITIVE_FOOTPRINTS,
                )
                fp = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)

            rw, rd = (1, 1)
            if fp:
                rw, rd = fp.rotated_size(prim.rotation)

            # Horizontal position depends on view axis
            if self._axis == FlowViewAxis.FRONT:
                h_start = prim.origin_cell.x * grid_size
                h_end = h_start + rw * grid_size
            else:
                h_start = prim.origin_cell.y * grid_size
                h_end = h_start + rd * grid_size

            z_start = prim.z_offset
            z_end = z_start + _get_module_height(prim)

            # Extend for portal z_levels + PORTAL_HEIGHT
            for portal in prim.get_portals():
                portal_top_z = prim.z_offset + prim.get_portal_z_level(portal.id) + PORTAL_HEIGHT
                z_end = max(z_end, portal_top_z)

            min_h = min(min_h, h_start)
            max_h = max(max_h, h_end)
            min_z = min(min_z, z_start)
            max_z = max(max_z, z_end)

        if min_h == float('inf'):
            return

        # Add margin
        margin_h = max((max_h - min_h) * 0.1, grid_size)
        margin_z = max((max_z - min_z) * 0.1, 64)

        world_w = (max_h - min_h) + 2 * margin_h
        world_h = (max_z - min_z) + 2 * margin_z

        if world_w <= 0 or world_h <= 0:
            return

        # Compute zoom to fit
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        zoom_h = w / world_w
        zoom_z = h / world_h
        self._zoom = min(zoom_h, zoom_z)

        # Center on the layout
        self._pan_x = (min_h + max_h) / 2
        self._pan_z = (min_z + max_z) / 2

        self.update()

    def focus_on_selected(self):
        """Center and zoom the view on the selected primitive, or fit all if none selected."""
        if not self._selected_prim_id or not self._layout:
            self.fit_to_layout()
            return

        prim = self._layout.primitives.get(self._selected_prim_id)
        if not prim:
            self.fit_to_layout()
            return

        grid_size = self._layout.grid_size
        fp = prim.footprint
        if fp is None:
            from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import PRIMITIVE_FOOTPRINTS
            fp = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)

        rw, rd = (1, 1)
        if fp:
            rw, rd = fp.rotated_size(prim.rotation)

        # Compute world bounds for this primitive
        if self._axis == FlowViewAxis.FRONT:
            h_start = prim.origin_cell.x * grid_size
            h_end = h_start + rw * grid_size
        else:
            h_start = prim.origin_cell.y * grid_size
            h_end = h_start + rd * grid_size

        z_start = prim.z_offset
        z_end = z_start + _get_module_height(prim)

        # Extend for portal z_levels
        for portal in prim.get_portals():
            portal_top_z = prim.z_offset + prim.get_portal_z_level(portal.id) + PORTAL_HEIGHT
            z_end = max(z_end, portal_top_z)

        # Add margin
        margin_h = max((h_end - h_start) * 0.3, grid_size)
        margin_z = max((z_end - z_start) * 0.3, 64)

        world_w = (h_end - h_start) + 2 * margin_h
        world_h = (z_end - z_start) + 2 * margin_z

        if world_w <= 0 or world_h <= 0:
            return

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        zoom_h = w / world_w
        zoom_z = h / world_h
        self._zoom = min(zoom_h, zoom_z)

        self._pan_x = (h_start + h_end) / 2
        self._pan_z = (z_start + z_end) / 2

        self.update()

    # --- Hit testing ---

    def _get_prim_rect(self, prim) -> QRectF:
        """Get the screen rectangle for a primitive."""
        if not self._layout:
            return QRectF()

        grid_size = self._layout.grid_size
        fp = prim.footprint
        if fp is None:
            from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import (
                PRIMITIVE_FOOTPRINTS,
            )
            fp = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)

        rw, rd = (1, 1)
        if fp:
            rw, rd = fp.rotated_size(prim.rotation)

        if self._axis == FlowViewAxis.FRONT:
            h_start = prim.origin_cell.x * grid_size
            h_size = rw * grid_size
        else:
            h_start = prim.origin_cell.y * grid_size
            h_size = rd * grid_size

        z_start = prim.z_offset
        z_size = _get_module_height(prim)

        # Extend to encompass portal z_levels + PORTAL_HEIGHT
        for portal in prim.get_portals():
            portal_z = prim.get_portal_z_level(portal.id)
            portal_top = portal_z + PORTAL_HEIGHT
            if portal_top > z_size:
                z_size = portal_top

        top_left = self._world_to_screen(h_start, z_start + z_size)
        bottom_right = self._world_to_screen(h_start + h_size, z_start)
        return QRectF(top_left, bottom_right)

    def _hit_test(self, screen_x: float, screen_y: float) -> Optional[str]:
        """Find the primitive at screen position, return its ID or None.

        Returns the topmost (last drawn) primitive if overlapping.
        """
        if not self._layout or not self._layout.primitives:
            return None

        hit_id = None
        for prim in self._layout.primitives.values():
            rect = self._get_prim_rect(prim)
            if rect.contains(QPointF(screen_x, screen_y)):
                hit_id = prim.id  # Last match wins (topmost)
        return hit_id

    # --- Coordinate transforms ---

    def _world_to_screen(self, world_h: float, world_z: float) -> QPointF:
        """Convert world coordinates to screen coordinates."""
        w, h = self.width(), self.height()
        sx = w / 2 + (world_h - self._pan_x) * self._zoom
        sy = h / 2 - (world_z - self._pan_z) * self._zoom  # Z up = screen up
        return QPointF(sx, sy)

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        w, h = self.width(), self.height()
        world_h = (sx - w / 2) / self._zoom + self._pan_x
        world_z = -(sy - h / 2) / self._zoom + self._pan_z
        return (world_h, world_z)

    # --- Painting ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self._draw_grid(painter)
        self._draw_axes(painter)

        if self._layout and self._layout.primitives:
            self._draw_connections(painter)
            self._draw_portal_alignment_lines(painter)
            self._draw_modules(painter)
            self._draw_portals(painter)
        self._draw_label(painter)

        painter.end()

    def _draw_grid(self, painter: QPainter):
        """Draw reference grid lines."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        # Calculate visible world range
        world_left, world_top = self._screen_to_world(0, 0)
        world_right, world_bottom = self._screen_to_world(w, h)

        # Adjust spacing based on zoom
        spacing = self._grid_spacing
        pixels_per_grid = spacing * self._zoom
        while pixels_per_grid < 20:
            spacing *= 2
            pixels_per_grid = spacing * self._zoom
        while pixels_per_grid > 100 and spacing > 32:
            spacing /= 2
            pixels_per_grid = spacing * self._zoom

        # Minor grid
        painter.setPen(QPen(QColor(55, 55, 55), 1))
        self._draw_grid_lines(painter, spacing, world_left, world_right, world_bottom, world_top)

        # Major grid (every 4 minor)
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        self._draw_grid_lines(painter, spacing * 4, world_left, world_right, world_bottom, world_top)

    def _draw_grid_lines(self, painter: QPainter, spacing: float,
                         h_min: float, h_max: float, z_min: float, z_max: float):
        """Draw grid lines at given spacing."""
        import math
        w, h = self.width(), self.height()

        # Vertical lines (horizontal axis)
        start_h = math.floor(h_min / spacing) * spacing
        hval = start_h
        while hval <= h_max:
            pt = self._world_to_screen(hval, 0)
            painter.drawLine(int(pt.x()), 0, int(pt.x()), h)
            hval += spacing

        # Horizontal lines (Z axis)
        start_z = math.floor(z_min / spacing) * spacing
        zval = start_z
        while zval <= z_max:
            pt = self._world_to_screen(0, zval)
            painter.drawLine(0, int(pt.y()), w, int(pt.y()))
            zval += spacing

    def _draw_axes(self, painter: QPainter):
        """Draw origin axes."""
        w, h = self.width(), self.height()

        # Z=0 line (ground level)
        origin_z = self._world_to_screen(0, 0)
        if 0 <= origin_z.y() <= h:
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawLine(0, int(origin_z.y()), w, int(origin_z.y()))

        # Horizontal origin line
        if 0 <= origin_z.x() <= w:
            painter.setPen(QPen(QColor(80, 80, 80), 2))
            painter.drawLine(int(origin_z.x()), 0, int(origin_z.x()), h)

    def _draw_modules(self, painter: QPainter):
        """Draw each module as a colored rectangle."""
        if not self._layout:
            return

        grid_size = self._layout.grid_size
        font = QFont("Menlo", 8)
        painter.setFont(font)

        for prim in self._layout.primitives.values():
            rect = self._get_prim_rect(prim)
            is_selected = (prim.id == self._selected_prim_id)
            is_hovered = (prim.id == self._hover_prim_id)

            # Draw filled rectangle
            color = _get_color(prim.primitive_type)
            fill_color = QColor(color)

            if is_selected:
                fill_color.setAlpha(220)
            elif is_hovered:
                fill_color.setAlpha(190)
            else:
                fill_color.setAlpha(160)

            painter.setBrush(QBrush(fill_color))

            if is_selected:
                # Selected: bright highlight border
                painter.setPen(QPen(QColor(255, 220, 50), 3))
            elif is_hovered:
                # Hovered: lighter border
                painter.setPen(QPen(color.lighter(140), 2))
            else:
                painter.setPen(QPen(color, 2))

            painter.drawRect(rect)

            # Draw label if rectangle is large enough
            if rect.width() > 30 and rect.height() > 14:
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                # Truncate long names
                label = prim.primitive_type
                if len(label) > 12:
                    label = label[:10] + ".."
                painter.drawText(rect, Qt.AlignCenter, label)

            # Draw Z-offset label below
            if rect.width() > 20:
                z_label = f"Z:{int(prim.z_offset)}"
                painter.setPen(QPen(QColor(200, 200, 200, 180), 1))
                small_font = QFont("Menlo", 7)
                painter.setFont(small_font)
                z_rect = QRectF(rect.left(), rect.bottom() + 1, rect.width(), 12)
                painter.drawText(z_rect, Qt.AlignCenter, z_label)
                painter.setFont(font)

    def _draw_connections(self, painter: QPainter):
        """Draw connection lines between portals.

        Color-codes connections to match GridCanvas:
        - Red dash-dot: Secret connections (CLIP wall)
        - Blue dashed: Horizontal same-level connections
        - Purple dashed: Vertical connections (via VerticalStairHall)
        - Orange dashed: Mismatched Z-level connections (warning)
        """
        if not self._layout or not self._layout.connections:
            return

        grid_size = self._layout.grid_size

        for conn in self._layout.connections:
            prim_a = self._layout.primitives.get(conn.primitive_a_id)
            prim_b = self._layout.primitives.get(conn.primitive_b_id)
            if not prim_a or not prim_b:
                continue

            # Get portal screen positions
            pos_a = self._get_portal_screen_pos(prim_a, conn.portal_a_id, grid_size)
            pos_b = self._get_portal_screen_pos(prim_b, conn.portal_b_id, grid_size)
            if pos_a is None or pos_b is None:
                continue

            # Determine connection color — same logic as grid_canvas.py
            if conn.is_secret:
                color = CONN_SECRET
                conn_pen = QPen(color, 3, Qt.DashDotLine)
            else:
                # Use actual portal Z positions, not primitive z_offsets
                z_a = prim_a.get_absolute_portal_z(conn.portal_a_id)
                z_b = prim_b.get_absolute_portal_z(conn.portal_b_id)
                z_diff = abs(z_a - z_b)

                is_vertical_connector = (
                    prim_a.primitive_type == 'VerticalStairHall' or
                    prim_b.primitive_type == 'VerticalStairHall' or
                    any(p.z_level > 0 for p in prim_a.get_portals()) or
                    any(p.z_level > 0 for p in prim_b.get_portals())
                )

                if z_diff < 2:
                    color = CONN_HORIZONTAL
                elif is_vertical_connector:
                    color = CONN_VERTICAL
                else:
                    color = CONN_MISMATCH

                conn_pen = QPen(color, 2, Qt.DashLine)

            painter.setPen(conn_pen)
            painter.drawLine(pos_a, pos_b)

            # Draw small circles at endpoints using connection color
            painter.setPen(QPen(color.darker(110), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pos_a, 3, 3)
            painter.drawEllipse(pos_b, 3, 3)

    def _draw_portal_alignment_lines(self, painter: QPainter):
        """Draw thin horizontal reference lines at connected portal Z-levels.

        For aligned portals (z_diff < 2): a subtle green dashed line spanning
        between the two module horizontal positions.
        For misaligned portals: short orange lines at each portal's Z-level.
        """
        if not self._layout or not self._layout.connections:
            return

        grid_size = self._layout.grid_size
        align_color = QColor(76, 175, 80, 60)    # Green alpha 60
        mismatch_color = QColor(255, 152, 0, 80)  # Orange alpha 80

        for conn in self._layout.connections:
            prim_a = self._layout.primitives.get(conn.primitive_a_id)
            prim_b = self._layout.primitives.get(conn.primitive_b_id)
            if not prim_a or not prim_b:
                continue

            z_a = prim_a.get_absolute_portal_z(conn.portal_a_id)
            z_b = prim_b.get_absolute_portal_z(conn.portal_b_id)

            # Get horizontal positions for spanning
            pos_a = self._get_portal_screen_pos(prim_a, conn.portal_a_id, grid_size)
            pos_b = self._get_portal_screen_pos(prim_b, conn.portal_b_id, grid_size)
            if pos_a is None or pos_b is None:
                continue

            left_x = min(pos_a.x(), pos_b.x())
            right_x = max(pos_a.x(), pos_b.x())

            if abs(z_a - z_b) < 2:
                # Aligned — draw a single horizontal dashed line at their shared Z
                screen_z = self._world_to_screen(0, z_a)
                pen = QPen(align_color, 1, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(left_x, screen_z.y()),
                    QPointF(right_x, screen_z.y()),
                )
            else:
                # Misaligned — draw short lines at each portal's Z
                line_extend = 20  # pixels beyond module edges
                pen = QPen(mismatch_color, 1, Qt.DashLine)
                painter.setPen(pen)

                screen_za = self._world_to_screen(0, z_a)
                painter.drawLine(
                    QPointF(left_x - line_extend, screen_za.y()),
                    QPointF(right_x + line_extend, screen_za.y()),
                )

                screen_zb = self._world_to_screen(0, z_b)
                painter.drawLine(
                    QPointF(left_x - line_extend, screen_zb.y()),
                    QPointF(right_x + line_extend, screen_zb.y()),
                )

    def _get_portal_screen_pos(self, prim, portal_id: str, grid_size: int) -> Optional[QPointF]:
        """Get screen position for a portal on a primitive."""
        for portal in prim.get_portals():
            if portal.id == portal_id:
                cell = prim.get_portal_world_cell(portal)
                z_level = prim.get_portal_z_level(portal.id)

                if self._axis == FlowViewAxis.FRONT:
                    world_h = (cell.x + 0.5) * grid_size
                else:
                    world_h = (cell.y + 0.5) * grid_size

                world_z = prim.z_offset + z_level
                return self._world_to_screen(world_h, world_z)
        return None

    def _draw_portals(self, painter: QPainter):
        """Draw portal indicators on module edges.

        Shows world-scaled colored rectangles on the correct edges of module rects
        based on the portal's rotated direction:
        - Green: enabled portal at z_level=0
        - Orange: upper-level portal (z_level > 0)
        - Gray: disabled portal

        Portal height is scaled to PORTAL_HEIGHT in world units (minimum 6px).
        Portals facing into/out of the screen are shown as small diamond markers.
        When module is selected, also shows z_level labels for non-zero portals.
        """
        if not self._layout:
            return

        grid_size = self._layout.grid_size
        INDICATOR_WIDTH = 8  # Fixed pixel width for edge markers

        for prim in self._layout.primitives.values():
            portals = prim.get_portals()
            if not portals:
                continue

            is_selected = (prim.id == self._selected_prim_id)
            module_rect = self._get_prim_rect(prim)

            for portal in portals:
                # Determine portal color
                z_level = prim.get_portal_z_level(portal.id)
                if not portal.enabled:
                    color = PORTAL_DISABLED
                elif z_level > 0:
                    color = PORTAL_UPPER_LEVEL
                else:
                    color = PORTAL_ENABLED

                # Get portal screen position
                cell = prim.get_portal_world_cell(portal)
                if self._axis == FlowViewAxis.FRONT:
                    world_h = (cell.x + 0.5) * grid_size
                else:
                    world_h = (cell.y + 0.5) * grid_size

                portal_floor_z = prim.z_offset + z_level
                portal_top_z = portal_floor_z + PORTAL_HEIGHT

                # Compute world-scaled portal height in screen space
                floor_screen = self._world_to_screen(world_h, portal_floor_z)
                top_screen = self._world_to_screen(world_h, portal_top_z)
                portal_screen_height = max(abs(floor_screen.y() - top_screen.y()), 6)

                # Determine edge placement based on rotated direction
                rotated_dir = portal.rotated_direction(prim.rotation)

                if self._axis == FlowViewAxis.FRONT:
                    # Front view: X horizontal, Z vertical
                    if rotated_dir == PortalDirection.EAST:
                        edge_x = module_rect.right() - INDICATOR_WIDTH / 2
                        draw_on_edge = True
                    elif rotated_dir == PortalDirection.WEST:
                        edge_x = module_rect.left() - INDICATOR_WIDTH / 2
                        draw_on_edge = True
                    else:
                        # NORTH/SOUTH — perpendicular to view
                        edge_x = floor_screen.x() - INDICATOR_WIDTH / 2
                        draw_on_edge = False
                else:
                    # Side view: Y horizontal, Z vertical
                    if rotated_dir == PortalDirection.NORTH:
                        edge_x = module_rect.right() - INDICATOR_WIDTH / 2
                        draw_on_edge = True
                    elif rotated_dir == PortalDirection.SOUTH:
                        edge_x = module_rect.left() - INDICATOR_WIDTH / 2
                        draw_on_edge = True
                    else:
                        # EAST/WEST — perpendicular to view
                        edge_x = floor_screen.x() - INDICATOR_WIDTH / 2
                        draw_on_edge = False

                if draw_on_edge:
                    # Draw world-scaled portal opening rectangle on module edge
                    portal_rect = QRectF(
                        edge_x,
                        top_screen.y(),
                        INDICATOR_WIDTH,
                        portal_screen_height,
                    )
                    painter.setPen(QPen(color.darker(120), 1))
                    painter.setBrush(QBrush(color))
                    painter.drawRect(portal_rect)
                else:
                    # Perpendicular portal — draw diamond marker
                    mid_y = (floor_screen.y() + top_screen.y()) / 2
                    cx = edge_x + INDICATOR_WIDTH / 2
                    size = max(portal_screen_height * 0.4, 4)
                    path = QPainterPath()
                    path.moveTo(cx, mid_y - size)
                    path.lineTo(cx + size, mid_y)
                    path.lineTo(cx, mid_y + size)
                    path.lineTo(cx - size, mid_y)
                    path.closeSubpath()
                    painter.setPen(QPen(color.darker(120), 1))
                    painter.setBrush(QBrush(color))
                    painter.drawPath(path)
                    portal_rect = QRectF(cx - size, mid_y - size, size * 2, size * 2)

                # Show z_level label for non-zero portals when selected
                if is_selected and z_level > 0:
                    painter.setPen(QPen(PORTAL_UPPER_LEVEL, 1))
                    small_font = QFont("Menlo", 7)
                    painter.setFont(small_font)
                    label_rect = QRectF(portal_rect.right() + 2, portal_rect.top(), 40, 12)
                    painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter,
                                     f"z:{int(z_level)}")

    def _draw_label(self, painter: QPainter):
        """Draw view label in top-left corner."""
        if self._axis == FlowViewAxis.FRONT:
            label = "Front (XZ)"
        else:
            label = "Side (YZ)"

        painter.setPen(QPen(QColor(80, 80, 80), 1))
        font = QFont("Menlo", 11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 20, label)

        # Draw drag hint if a module is selected
        if self._selected_prim_id and self._layout:
            prim = self._layout.primitives.get(self._selected_prim_id)
            if prim:
                hint_font = QFont("Menlo", 9)
                painter.setFont(hint_font)
                painter.setPen(QPen(QColor(150, 150, 150), 1))
                cell = prim.origin_cell
                painter.drawText(10, 36,
                    f"Selected: {prim.primitive_type} ({cell.x},{cell.y}) Z:{int(prim.z_offset)} — drag to move")

    # --- Mouse handling ---

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.pos()
        self._right_dragged = False

        if event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier:
            # Alt+Left = pan (emulate middle button)
            self._mouse_button = Qt.MiddleButton
            event.accept()
            return

        self._mouse_button = event.button()

        if event.button() == Qt.LeftButton:
            # Hit test for selection
            hit_id = self._hit_test(event.x(), event.y())

            if hit_id:
                # Select the primitive
                if self._selected_prim_id != hit_id:
                    self._selected_prim_id = hit_id
                    self.primitive_selected.emit(hit_id)
                    self.update()

                # Prepare for potential drag (cell move + Z-offset)
                self._drag_start_pos = event.pos()
                prim = self._layout.primitives.get(hit_id)
                if prim:
                    self._drag_start_z = prim.z_offset
                    self._drag_start_cell_x = prim.origin_cell.x
                    self._drag_start_cell_y = prim.origin_cell.y
                    self._drag_prim_id = hit_id
            else:
                # Click on empty space = clear selection
                if self._selected_prim_id is not None:
                    self._selected_prim_id = None
                    self.selection_cleared.emit()
                    self.update()
                self._drag_prim_id = None

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging and self._drag_prim_id and self._layout:
            prim = self._layout.primitives.get(self._drag_prim_id)
            if prim:
                from quake_levelgenerator.src.ui.widgets.layout_editor.commands import (
                    SetZOffsetCommand, MovePrimitiveCommand,
                )
                final_z = prim.z_offset
                final_cell_x = prim.origin_cell.x
                final_cell_y = prim.origin_cell.y

                # Revert to original state before issuing commands
                prim.z_offset = self._drag_start_z
                prim.origin_cell = CellCoord(self._drag_start_cell_x, self._drag_start_cell_y)

                # Emit move command if cell changed
                cell_changed = (final_cell_x != self._drag_start_cell_x or
                                final_cell_y != self._drag_start_cell_y)
                if cell_changed:
                    move_cmd = MovePrimitiveCommand(
                        primitive_id=self._drag_prim_id,
                        new_origin_x=final_cell_x,
                        new_origin_y=final_cell_y,
                    )
                    self.command_requested.emit(move_cmd)

                # Emit Z command if Z changed
                if final_z != self._drag_start_z:
                    z_cmd = SetZOffsetCommand(
                        primitive_id=self._drag_prim_id,
                        new_z_offset=final_z,
                    )
                    self.command_requested.emit(z_cmd)

        self._dragging = False
        self._drag_start_pos = None
        self._drag_prim_id = None
        self._last_mouse_pos = None
        self._mouse_button = None
        self.setCursor(QCursor(Qt.ArrowCursor))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse_pos is None:
            # Just hovering - update hover state
            old_hover = self._hover_prim_id
            self._hover_prim_id = self._hit_test(event.x(), event.y())
            if self._hover_prim_id != old_hover:
                if self._hover_prim_id:
                    self.setCursor(QCursor(Qt.PointingHandCursor))
                else:
                    self.setCursor(QCursor(Qt.ArrowCursor))
                self.update()
            return

        dx = event.x() - self._last_mouse_pos.x()
        dy = event.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.pos()

        if self._mouse_button in (Qt.MiddleButton, Qt.RightButton):
            # Pan: move world center opposite to mouse movement
            self._pan_x -= dx / self._zoom
            self._pan_z += dy / self._zoom  # Y inverted
            self._right_dragged = True
            self.update()
            event.accept()
            return

        if self._mouse_button == Qt.LeftButton and self._drag_prim_id:
            # Check if we've exceeded drag threshold on either axis
            if not self._dragging and self._drag_start_pos:
                total_dx = abs(event.x() - self._drag_start_pos.x())
                total_dy = abs(event.y() - self._drag_start_pos.y())
                if max(total_dx, total_dy) >= _DRAG_THRESHOLD:
                    self._dragging = True
                    self.setCursor(QCursor(Qt.SizeAllCursor))

            if self._dragging and self._layout:
                grid_size = self._layout.grid_size

                # Get world coordinates at current and start positions
                new_world_h, new_world_z = self._screen_to_world(event.x(), event.y())
                start_world_h, start_world_z = self._screen_to_world(
                    self._drag_start_pos.x(), self._drag_start_pos.y()
                )

                # Horizontal: snap to cell positions
                delta_h = new_world_h - start_world_h
                delta_cells = round(delta_h / grid_size)

                if self._axis == FlowViewAxis.FRONT:
                    new_cell_x = self._drag_start_cell_x + delta_cells
                    new_cell_y = self._drag_start_cell_y
                else:
                    new_cell_x = self._drag_start_cell_x
                    new_cell_y = self._drag_start_cell_y + delta_cells

                # Vertical: Z-offset with snap
                delta_z = new_world_z - start_world_z
                new_z = self._drag_start_z + delta_z
                new_z = round(new_z / _Z_SNAP) * _Z_SNAP

                # Apply both for live preview
                prim = self._layout.primitives.get(self._drag_prim_id)
                if prim:
                    prim.origin_cell = CellCoord(new_cell_x, new_cell_y)
                    prim.z_offset = new_z
                    self.update()

        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        if self._mouse_button == Qt.MiddleButton:
            event.accept()
            return

        delta = event.angleDelta().y()
        if abs(delta) < 10:
            event.accept()
            return

        # Zoom toward cursor
        old_world = self._screen_to_world(event.x(), event.y())

        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom = max(0.01, min(10.0, self._zoom * factor))

        new_world = self._screen_to_world(event.x(), event.y())

        # Adjust pan so cursor stays over same world point
        self._pan_x -= (new_world[0] - old_world[0])
        self._pan_z -= (new_world[1] - old_world[1])

        self.update()
        event.accept()

    def focusOutEvent(self, event):
        """Clear mouse state when focus is lost to prevent stuck drag."""
        self._cancel_drag()
        super().focusOutEvent(event)

    def leaveEvent(self, event):
        """Clear mouse state when cursor leaves widget."""
        self._hover_prim_id = None
        if not self._dragging:
            self._last_mouse_pos = None
            self._mouse_button = None
        self.update()
        super().leaveEvent(event)

    def _cancel_drag(self):
        """Cancel an in-progress drag, reverting cell position and Z-offset."""
        if self._dragging and self._drag_prim_id and self._layout:
            prim = self._layout.primitives.get(self._drag_prim_id)
            if prim:
                prim.z_offset = self._drag_start_z
                prim.origin_cell = CellCoord(self._drag_start_cell_x, self._drag_start_cell_y)
        self._dragging = False
        self._drag_start_pos = None
        self._drag_prim_id = None
        self._last_mouse_pos = None
        self._mouse_button = None
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.update()

    # --- Connection interaction ---

    def contextMenuEvent(self, event):
        """Show context menu on right-click near a connection line."""
        if not self._layout or not self._layout.connections:
            super().contextMenuEvent(event)
            return

        # Only show context menu if this was a stationary click (not after panning)
        if self._right_dragged:
            self._right_dragged = False
            event.accept()
            return

        conn = self._find_connection_at(QPointF(event.x(), event.y()))
        if conn:
            self._show_connection_context_menu(event.globalPos(), conn)
            event.accept()
        else:
            super().contextMenuEvent(event)

    def _find_connection_at(self, screen_pos: QPointF):
        """Find a connection line near the given screen position.

        Returns the Connection object if one is within click tolerance, else None.
        """
        if not self._layout or not self._layout.connections:
            return None

        CLICK_TOLERANCE = 8.0  # pixels
        grid_size = self._layout.grid_size

        for conn in self._layout.connections:
            prim_a = self._layout.primitives.get(conn.primitive_a_id)
            prim_b = self._layout.primitives.get(conn.primitive_b_id)
            if not prim_a or not prim_b:
                continue

            pos_a = self._get_portal_screen_pos(prim_a, conn.portal_a_id, grid_size)
            pos_b = self._get_portal_screen_pos(prim_b, conn.portal_b_id, grid_size)
            if pos_a is None or pos_b is None:
                continue

            dist = self._point_to_line_distance(
                screen_pos.x(), screen_pos.y(),
                pos_a.x(), pos_a.y(),
                pos_b.x(), pos_b.y()
            )

            if dist < CLICK_TOLERANCE:
                return conn

        return None

    def _point_to_line_distance(self, px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """Calculate shortest distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def _show_connection_context_menu(self, global_pos, conn):
        """Show context menu for a connection."""
        menu = QMenu(self)

        if conn.is_secret:
            action_text = "Remove Secret (Open Portal)"
        else:
            action_text = "Make Secret (CLIP Wall)"

        toggle_action = QAction(action_text, self)
        toggle_action.triggered.connect(lambda: self._toggle_connection_secret(conn))
        menu.addAction(toggle_action)

        menu.exec_(global_pos)

    def _toggle_connection_secret(self, conn):
        """Toggle the is_secret flag on a connection."""
        if not self._layout:
            return

        from quake_levelgenerator.src.ui.widgets.layout_editor.data_model import Connection

        for i, c in enumerate(self._layout.connections):
            if (c.primitive_a_id == conn.primitive_a_id and
                c.portal_a_id == conn.portal_a_id and
                c.primitive_b_id == conn.primitive_b_id and
                c.portal_b_id == conn.portal_b_id):
                new_conn = Connection(
                    primitive_a_id=c.primitive_a_id,
                    portal_a_id=c.portal_a_id,
                    primitive_b_id=c.primitive_b_id,
                    portal_b_id=c.portal_b_id,
                    is_secret=not c.is_secret
                )
                self._layout.connections[i] = new_conn
                break

        self.layout_changed.emit()
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.fit_to_layout()
        elif event.key() == Qt.Key_G:
            self.focus_on_selected()
        elif event.key() == Qt.Key_Escape:
            if self._dragging:
                self._cancel_drag()
            elif self._selected_prim_id:
                self._selected_prim_id = None
                self.selection_cleared.emit()
                self.update()
        else:
            super().keyPressEvent(event)
