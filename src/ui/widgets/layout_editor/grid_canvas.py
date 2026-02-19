"""
QGraphicsView-based grid canvas for the layout editor.

Provides:
- Infinite grid with zoom/pan
- Click-to-place primitives with ghost preview
- Selection and drag-to-move operations
- Rotation support (R key) for both placement preview and placed primitives
- Undo/redo via command pattern
"""

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QApplication, QMenu, QAction,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QWheelEvent,
    QMouseEvent, QKeyEvent, QTransform, QFont,
)
from typing import Optional, Dict, Any, TYPE_CHECKING

from .data_model import (
    DungeonLayout, PlacedPrimitive, CellCoord, PrimitiveFootprint,
)
from .cell_item import CellItem, GhostCellItem

if TYPE_CHECKING:
    from .commands import Command


class GridCanvas(QGraphicsView):
    """Grid-based canvas for placing dungeon primitives."""

    # Signals
    primitive_placed = pyqtSignal(PlacedPrimitive)  # Emitted when a primitive is placed
    primitive_selected = pyqtSignal(str)  # Emitted with primitive ID when selected
    selection_cleared = pyqtSignal()  # Emitted when selection is cleared
    cell_hovered = pyqtSignal(int, int)  # Emitted with cell coordinates on hover
    command_requested = pyqtSignal(object)  # Emitted with Command to execute
    primitive_deleted = pyqtSignal(str)  # Emitted with primitive ID when deleted
    status_message = pyqtSignal(str)  # Emitted with status messages for user feedback
    mode_changed = pyqtSignal(bool, str)  # Emitted when mode changes (is_placing, primitive_type)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Data
        self._layout = DungeonLayout()
        self._cell_items: Dict[str, CellItem] = {}

        # Grid settings
        self._cell_size = 40.0  # Pixels per cell at zoom 1.0
        self._grid_color = QColor(60, 60, 60)
        self._grid_color_major = QColor(80, 80, 80)
        self._background_color = QColor(30, 30, 30)

        # View settings
        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 5.0
        self._pan_active = False
        self._pan_start = QPointF()

        # Placement mode
        self._placement_footprint: Optional[PrimitiveFootprint] = None
        self._placement_type: Optional[str] = None
        self._placement_category: str = 'default'
        self._placement_rotation: int = 0
        self._placement_z_offset: float = 0.0  # Floor level for placement
        self._ghost_item: Optional[GhostCellItem] = None

        # Selection
        self._selected_id: Optional[str] = None

        # Drag-to-move state
        self._drag_item: Optional[CellItem] = None
        self._drag_start_cell: Optional[CellCoord] = None
        self._drag_ghost: Optional[GhostCellItem] = None
        self._drag_cell_offset: Optional[CellCoord] = None  # Offset from click to origin
        self._drag_start_pos: Optional[QPointF] = None  # Start position in view coords
        self._drag_threshold = 8  # Minimum pixels before drag starts

        # Flow visualization
        self._show_flow = False
        self._flow_path: list = []  # List of primitive IDs in path order

        # Setup view
        self._setup_view()

    def _setup_view(self):
        """Configure view settings."""
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        # Enable keyboard focus for rotation (R key) and other shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

        # Set background
        self.setBackgroundBrush(QBrush(self._background_color))

        # Large scene rect for "infinite" canvas
        self._scene.setSceneRect(-10000, -10000, 20000, 20000)

        # Initial view centered on origin
        self.centerOn(0, 0)

    # ---------------------------------------------------------------
    # Layout management
    # ---------------------------------------------------------------

    def set_layout(self, layout: DungeonLayout):
        """Set the dungeon layout to display."""
        self._layout = layout
        self._rebuild_items()

    def get_layout(self) -> DungeonLayout:
        """Get the current layout."""
        return self._layout

    def clear_layout(self):
        """Clear all primitives."""
        self._layout = DungeonLayout()
        self._rebuild_items()
        self._clear_selection()

    def _rebuild_items(self):
        """Rebuild all cell items from layout."""
        # Remove existing items
        for item in self._cell_items.values():
            self._scene.removeItem(item)
        self._cell_items.clear()

        # Create items for all primitives
        for prim_id, prim in self._layout.primitives.items():
            self._add_cell_item(prim)

    def _add_cell_item(self, prim: PlacedPrimitive):
        """Add a cell item for a primitive."""
        footprint = prim.footprint
        if footprint is None:
            # Default 1x1 footprint
            footprint = PrimitiveFootprint(width_cells=1, depth_cells=1)

        # Determine category (would come from primitive catalog)
        category = self._get_primitive_category(prim.primitive_type)

        item = CellItem(prim, footprint, self._cell_size, category)
        self._scene.addItem(item)
        self._cell_items[prim.id] = item

    def _get_primitive_category(self, prim_type: str) -> str:
        """Get category for a primitive type."""
        # Map primitive types to categories
        halls = ['StraightHall', 'SquareCorner', 'TJunction', 'Crossroads']
        structural = ['StraightStaircase', 'Arch', 'Pillar']
        rooms = ['Chapel', 'Crypt', 'Tower']
        connective = ['Bridge', 'Balcony']

        if prim_type in halls:
            return 'Halls'
        elif prim_type in structural:
            return 'Structural'
        elif prim_type in rooms:
            return 'Rooms'
        elif prim_type in connective:
            return 'Connective'
        return 'default'

    # ---------------------------------------------------------------
    # Placement mode
    # ---------------------------------------------------------------

    def start_placement(self, primitive_type: str, footprint: PrimitiveFootprint,
                        category: str = 'default', z_offset: float = 0.0):
        """Enter placement mode for a primitive type."""
        self._placement_type = primitive_type
        self._placement_footprint = footprint
        self._placement_category = category
        self._placement_rotation = 0
        self._placement_z_offset = z_offset

        # Grab keyboard focus for rotation (R key)
        self.setFocus()

        # Create ghost item
        if self._ghost_item:
            self._scene.removeItem(self._ghost_item)
        self._ghost_item = GhostCellItem(footprint, self._cell_size, category)
        self._ghost_item.setVisible(False)
        self._scene.addItem(self._ghost_item)

        self.setCursor(Qt.CrossCursor)
        self.mode_changed.emit(True, primitive_type)

    def cancel_placement(self):
        """Exit placement mode."""
        self._placement_type = None
        self._placement_footprint = None

        if self._ghost_item:
            self._scene.removeItem(self._ghost_item)
            self._ghost_item = None

        self.setCursor(Qt.ArrowCursor)
        self.mode_changed.emit(False, "")

    def rotate_placement(self):
        """Rotate the current placement by 90 degrees."""
        if self._placement_footprint is None:
            return

        self._placement_rotation = (self._placement_rotation + 90) % 360
        if self._ghost_item:
            self._ghost_item.set_rotation(self._placement_rotation)
        self.status_message.emit(f"Rotation: {self._placement_rotation}°")

    @property
    def is_placing(self) -> bool:
        """Check if in placement mode."""
        return self._placement_type is not None

    @property
    def placement_type(self) -> Optional[str]:
        """Get the current placement primitive type."""
        return self._placement_type

    @property
    def placement_rotation(self) -> int:
        """Get the current placement rotation."""
        return self._placement_rotation

    @property
    def placement_z_offset(self) -> float:
        """Get the current placement Z offset."""
        return self._placement_z_offset

    def set_placement_z_offset(self, z_offset: float):
        """Set the Z offset for placement mode."""
        self._placement_z_offset = z_offset

    def get_placement_blocker(self, cell: CellCoord, footprint: PrimitiveFootprint,
                              rotation: int) -> Optional[str]:
        """Return human-readable reason why placement would fail, or None if valid."""
        cells = set(footprint.occupied_cells(cell, rotation))
        for other_id, other in self._layout.primitives.items():
            other_cells = set(other.occupied_cells())
            if cells & other_cells:
                return f"Blocked by {other.primitive_type}"
        return None

    # ---------------------------------------------------------------
    # Selection
    # ---------------------------------------------------------------

    def _select_item(self, prim_id: str):
        """Select a primitive."""
        # Deselect previous
        if self._selected_id and self._selected_id in self._cell_items:
            self._cell_items[self._selected_id].set_selected(False)

        self._selected_id = prim_id
        self._layout.select(prim_id)

        if prim_id in self._cell_items:
            self._cell_items[prim_id].set_selected(True)

        self.primitive_selected.emit(prim_id)

    def _clear_selection(self):
        """Clear selection."""
        if self._selected_id and self._selected_id in self._cell_items:
            self._cell_items[self._selected_id].set_selected(False)

        self._selected_id = None
        self._layout.clear_selection()
        self.selection_cleared.emit()

    def delete_selected(self):
        """Delete the currently selected primitive."""
        if self._selected_id is None:
            return

        prim_id = self._selected_id

        # Emit command for deletion
        # Note: primitive_deleted signal is emitted by LayoutEditorWidget
        # after command execution succeeds
        from .commands import DeletePrimitiveCommand
        cmd = DeletePrimitiveCommand(primitive_id=prim_id)
        self.command_requested.emit(cmd)

    def _rotate_selected(self):
        """Rotate the currently selected primitive by 90 degrees."""
        if self._selected_id is None:
            return

        prim = self._layout.primitives.get(self._selected_id)
        if not prim:
            return

        # Calculate new rotation (90 degrees clockwise)
        new_rotation = (prim.rotation + 90) % 360

        # Emit command for rotation
        from .commands import RotatePrimitiveCommand
        cmd = RotatePrimitiveCommand(
            primitive_id=self._selected_id,
            new_rotation=new_rotation
        )
        self.command_requested.emit(cmd)

    # ---------------------------------------------------------------
    # Coordinate conversion
    # ---------------------------------------------------------------

    def _scene_to_cell(self, scene_pos: QPointF) -> CellCoord:
        """Convert scene position to cell coordinate."""
        # Note: Scene Y increases downward, cells Y increases upward
        x = int(scene_pos.x() // self._cell_size)
        y = int(-scene_pos.y() // self._cell_size)
        return CellCoord(x, y)

    def _cell_to_scene(self, cell: CellCoord) -> QPointF:
        """Convert cell coordinate to scene position (top-left of cell in scene coords)."""
        return QPointF(
            cell.x * self._cell_size,
            -(cell.y + 1) * self._cell_size  # +1 because Y is flipped
        )

    def _cell_to_scene_for_footprint(self, cell: CellCoord, footprint: PrimitiveFootprint,
                                      rotation: int = 0) -> QPointF:
        """Convert cell coordinate to scene position for a footprint (matching CellItem positioning)."""
        _, depth = footprint.rotated_size(rotation)
        return QPointF(
            cell.x * self._cell_size,
            -(cell.y * self._cell_size + depth * self._cell_size)
        )

    # ---------------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw the background grid."""
        super().drawBackground(painter, rect)

        # Calculate visible grid range
        left = int(rect.left() // self._cell_size) - 1
        right = int(rect.right() // self._cell_size) + 1
        top = int(rect.top() // self._cell_size) - 1
        bottom = int(rect.bottom() // self._cell_size) + 1

        # Draw grid lines
        minor_pen = QPen(self._grid_color, 1)
        major_pen = QPen(self._grid_color_major, 2)

        # Vertical lines
        for i in range(left, right + 1):
            x = i * self._cell_size
            if i % 5 == 0:
                painter.setPen(major_pen)
            else:
                painter.setPen(minor_pen)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        # Horizontal lines
        for j in range(top, bottom + 1):
            y = j * self._cell_size
            if j % 5 == 0:
                painter.setPen(major_pen)
            else:
                painter.setPen(minor_pen)
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # Draw origin axes
        axis_pen = QPen(QColor(100, 100, 100), 2)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
        painter.drawLine(QPointF(0, rect.top()), QPointF(0, rect.bottom()))

    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Draw foreground overlays (flow visualization, connections, dead-ends)."""
        super().drawForeground(painter, rect)

        # Always draw connections - they're essential for understanding the layout
        self._draw_connections(painter)

        # Flow path and dead-end markers are optional
        if self._show_flow:
            self._draw_flow_path(painter)
            self._draw_dead_end_markers(painter)

    def _draw_connections(self, painter: QPainter):
        """Draw connection lines between connected portals.

        Color-codes connections based on type:
        - Red dashed: Secret connections (CLIP wall)
        - Blue: Horizontal same-level connections
        - Purple: Vertical connections (via VerticalStairHall)
        - Orange: Mismatched Z-level connections (potential issues)
        """
        # Define connection type colors
        CONN_SECRET = QColor(220, 20, 60, 220)        # Crimson red for secrets
        CONN_HORIZONTAL = QColor(33, 150, 243, 180)   # Blue
        CONN_VERTICAL = QColor(156, 39, 176, 180)     # Purple
        CONN_MISMATCH = QColor(255, 152, 0, 200)      # Orange (warning)

        for conn in self._layout.connections:
            prim_a = self._layout.primitives.get(conn.primitive_a_id)
            prim_b = self._layout.primitives.get(conn.primitive_b_id)

            if not prim_a or not prim_b:
                continue

            # Secret connections always use red color
            if conn.is_secret:
                color = CONN_SECRET
                conn_pen = QPen(color, 3, Qt.DashDotLine)  # Thicker dash-dot for secrets
            else:
                # Determine connection type based on Z-offsets
                z_diff = abs(prim_a.z_offset - prim_b.z_offset)

                # Check if either primitive is a VerticalStairHall (proper vertical connector)
                is_vertical_stair = (
                    prim_a.primitive_type == 'VerticalStairHall' or
                    prim_b.primitive_type == 'VerticalStairHall'
                )

                if z_diff < 2:
                    # Same level - blue
                    color = CONN_HORIZONTAL
                elif is_vertical_stair:
                    # Proper vertical connection via stair hall - purple
                    color = CONN_VERTICAL
                else:
                    # Z-level mismatch without proper connector - orange warning
                    color = CONN_MISMATCH

                conn_pen = QPen(color, 2, Qt.DashLine)

            painter.setPen(conn_pen)

            # Get center points of each primitive
            center_a = self._get_primitive_center(prim_a)
            center_b = self._get_primitive_center(prim_b)

            if center_a and center_b:
                painter.drawLine(center_a, center_b)

    def _draw_flow_path(self, painter: QPainter):
        """Draw the critical path through the layout."""
        if len(self._flow_path) < 2:
            return

        # Draw thick colored line along the path
        path_pen = QPen(QColor(255, 193, 7, 200), 4)  # Amber/yellow
        painter.setPen(path_pen)

        for i in range(len(self._flow_path) - 1):
            prim_a = self._layout.primitives.get(self._flow_path[i])
            prim_b = self._layout.primitives.get(self._flow_path[i + 1])

            if not prim_a or not prim_b:
                continue

            center_a = self._get_primitive_center(prim_a)
            center_b = self._get_primitive_center(prim_b)

            if center_a and center_b:
                painter.drawLine(center_a, center_b)

        # Draw path order numbers
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        for i, prim_id in enumerate(self._flow_path):
            prim = self._layout.primitives.get(prim_id)
            if prim:
                center = self._get_primitive_center(prim)
                if center:
                    # Draw number badge
                    badge_rect = QRectF(center.x() - 10, center.y() - 10, 20, 20)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(255, 193, 7)))
                    painter.drawEllipse(badge_rect)
                    painter.setPen(QPen(QColor(0, 0, 0)))
                    painter.drawText(badge_rect, Qt.AlignCenter, str(i + 1))

    def _draw_dead_end_markers(self, painter: QPainter):
        """Draw warning markers on primitives with only 1 or 0 connections (dead ends)."""
        # Build connection count for each primitive
        conn_count: Dict[str, int] = {pid: 0 for pid in self._layout.primitives}
        for conn in self._layout.connections:
            if conn.primitive_a_id in conn_count:
                conn_count[conn.primitive_a_id] += 1
            if conn.primitive_b_id in conn_count:
                conn_count[conn.primitive_b_id] += 1

        # Draw markers on dead ends
        for prim_id, count in conn_count.items():
            if count <= 1:
                prim = self._layout.primitives.get(prim_id)
                if prim:
                    center = self._get_primitive_center(prim)
                    if center:
                        # Draw orange warning triangle in top-right
                        warn_color = QColor(255, 152, 0)  # Orange
                        painter.setPen(QPen(warn_color.darker(120), 2))
                        painter.setBrush(QBrush(warn_color))

                        # Calculate position offset from center
                        footprint = prim.footprint
                        if footprint:
                            w, d = footprint.rotated_size(prim.rotation)
                            offset_x = (w / 2 - 0.5) * self._cell_size
                            offset_y = -(d / 2 - 0.5) * self._cell_size
                        else:
                            offset_x = self._cell_size * 0.3
                            offset_y = -self._cell_size * 0.3

                        # Draw warning badge
                        badge_x = center.x() + offset_x
                        badge_y = center.y() + offset_y
                        badge_rect = QRectF(badge_x - 8, badge_y - 8, 16, 16)
                        painter.drawEllipse(badge_rect)

                        # Draw exclamation mark
                        painter.setPen(QPen(QColor(0, 0, 0), 2))
                        font = QFont()
                        font.setPointSize(10)
                        font.setBold(True)
                        painter.setFont(font)
                        painter.drawText(badge_rect, Qt.AlignCenter, "!")

    def _get_primitive_center(self, prim: PlacedPrimitive) -> Optional[QPointF]:
        """Get the scene center point of a primitive."""
        footprint = prim.footprint
        if not footprint:
            cell = prim.origin_cell
            return QPointF(
                (cell.x + 0.5) * self._cell_size,
                -(cell.y + 0.5) * self._cell_size
            )

        w, d = footprint.rotated_size(prim.rotation)
        return QPointF(
            (prim.origin_cell.x + w / 2) * self._cell_size,
            -(prim.origin_cell.y + d / 2) * self._cell_size
        )

    def set_show_flow(self, show: bool):
        """Toggle flow visualization."""
        self._show_flow = show
        if show:
            self._calculate_flow_path()
        self.viewport().update()

    @property
    def show_flow(self) -> bool:
        """Check if flow visualization is enabled."""
        return self._show_flow

    def _calculate_flow_path(self):
        """Calculate the critical path through the layout using BFS."""
        from collections import deque

        self._flow_path = []

        if not self._layout.primitives:
            return

        # Build adjacency graph
        adj: Dict[str, list] = {pid: [] for pid in self._layout.primitives}
        for conn in self._layout.connections:
            if conn.primitive_a_id in adj and conn.primitive_b_id in adj:
                adj[conn.primitive_a_id].append(conn.primitive_b_id)
                adj[conn.primitive_b_id].append(conn.primitive_a_id)

        # Find the longest path from any starting point (simple BFS approach)
        # This finds the path from the first primitive through all reachable nodes
        start_id = next(iter(self._layout.primitives))

        visited = set()
        parent: Dict[str, Optional[str]] = {start_id: None}
        queue = deque([start_id])
        last_visited = start_id

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            last_visited = current

            for neighbor in adj[current]:
                if neighbor not in visited and neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        # Build path from start to furthest point
        path = []
        node = last_visited
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()

        self._flow_path = path

    def get_flow_metrics(self) -> Dict[str, Any]:
        """Get flow metrics for the current layout."""
        metrics = {
            'path_length': len(self._flow_path),
            'total_primitives': len(self._layout.primitives),
            'total_connections': len(self._layout.connections),
            'room_count': 0,
            'hall_count': 0,
            'dead_ends': 0,
        }

        # Count by category
        rooms = ['Chapel', 'Crypt', 'Tower']
        halls = ['StraightHall', 'SquareCorner', 'TJunction', 'Crossroads']

        # Build connection count for dead-end detection
        conn_count: Dict[str, int] = {pid: 0 for pid in self._layout.primitives}
        for conn in self._layout.connections:
            if conn.primitive_a_id in conn_count:
                conn_count[conn.primitive_a_id] += 1
            if conn.primitive_b_id in conn_count:
                conn_count[conn.primitive_b_id] += 1

        for prim in self._layout.primitives.values():
            if prim.primitive_type in rooms:
                metrics['room_count'] += 1
            elif prim.primitive_type in halls:
                metrics['hall_count'] += 1

            # Dead end has only one connection
            if conn_count.get(prim.id, 0) <= 1:
                metrics['dead_ends'] += 1

        return metrics

    # ---------------------------------------------------------------
    # Mouse events
    # ---------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        # Pan with middle-button, right-button, or Alt/Option+Left (TrenchBroom-style)
        if (event.button() == Qt.MiddleButton or
            event.button() == Qt.RightButton or
            (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier)):
            # Start pan
            self._pan_active = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            cell = self._scene_to_cell(scene_pos)

            if self.is_placing:
                # Place primitive
                self._place_at_cell(cell)
                event.accept()
                return
            else:
                # Check for item at click position
                item = self._scene.itemAt(scene_pos, QTransform())
                if isinstance(item, CellItem):
                    self._select_item(item.primitive_id)
                    # Start potential drag operation
                    self._drag_item = item
                    self._drag_start_cell = item.primitive.origin_cell
                    self._drag_start_pos = QPointF(event.pos())  # Store view coords for threshold
                    # Calculate offset from click position to primitive origin
                    # This ensures the primitive stays "grabbed" at the click point
                    click_cell = self._scene_to_cell(scene_pos)
                    origin = item.primitive.origin_cell
                    self._drag_cell_offset = CellCoord(
                        click_cell.x - origin.x,
                        click_cell.y - origin.y
                    )
                else:
                    self._clear_selection()
                    self._drag_item = None
                    self._drag_start_cell = None
                    self._drag_cell_offset = None
                    self._drag_start_pos = None

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        # End pan (middle-button, right-button, or Alt+Left)
        if self._pan_active and event.button() in (
            Qt.MiddleButton, Qt.RightButton, Qt.LeftButton
        ):
            self._pan_active = False
            self.setCursor(Qt.CrossCursor if self.is_placing else Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._drag_ghost:
            # Complete drag-to-move operation
            scene_pos = self.mapToScene(event.pos())
            mouse_cell = self._scene_to_cell(scene_pos)

            # Apply offset to get the actual target origin cell
            # (where we dropped minus where we grabbed relative to origin)
            if self._drag_cell_offset:
                target_cell = CellCoord(
                    mouse_cell.x - self._drag_cell_offset.x,
                    mouse_cell.y - self._drag_cell_offset.y
                )
            else:
                target_cell = mouse_cell

            if self._drag_item and self._drag_start_cell:
                # Only emit move command if position actually changed
                if target_cell != self._drag_start_cell:
                    from .commands import MovePrimitiveCommand
                    cmd = MovePrimitiveCommand(
                        primitive_id=self._drag_item.primitive_id,
                        new_origin_x=target_cell.x,
                        new_origin_y=target_cell.y
                    )
                    self.command_requested.emit(cmd)

            # Clean up drag state
            self._end_drag()
            event.accept()
            return

        # Clean up any incomplete drag
        self._drag_item = None
        self._drag_start_cell = None
        self._drag_cell_offset = None
        self._drag_start_pos = None

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self._pan_active:
            # Pan view
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())
        cell = self._scene_to_cell(scene_pos)

        # Handle drag-to-move
        if self._drag_item and self._drag_start_cell and not self.is_placing:
            # Check pixel distance threshold before starting drag
            if self._drag_start_pos and not self._drag_ghost:
                delta = event.pos() - self._drag_start_pos
                distance_sq = delta.x() * delta.x() + delta.y() * delta.y()
                if distance_sq < self._drag_threshold * self._drag_threshold:
                    # Not far enough to start drag
                    event.accept()
                    return

            # Calculate target cell (where the origin would be placed)
            if self._drag_cell_offset:
                target_cell = CellCoord(
                    cell.x - self._drag_cell_offset.x,
                    cell.y - self._drag_cell_offset.y
                )
            else:
                target_cell = cell

            # Check if we've moved enough to start dragging (target origin changed)
            if target_cell != self._drag_start_cell or self._drag_ghost:
                self._update_drag_ghost(cell)
                self.cell_hovered.emit(target_cell.x, target_cell.y)
                event.accept()
                return

        # Update ghost item position for placement mode
        if self._ghost_item and self.is_placing:
            # Position ghost at cell (using footprint-aware positioning to match CellItem)
            ghost_pos = self._cell_to_scene_for_footprint(
                cell, self._placement_footprint, self._placement_rotation
            )
            self._ghost_item.setPos(ghost_pos)
            self._ghost_item.setVisible(True)

            # Check validity
            valid = self._layout.can_place_at(
                self._placement_footprint, cell, self._placement_rotation
            )
            self._ghost_item.set_valid(valid)

            # Emit hover signal
            self.cell_hovered.emit(cell.x, cell.y)

        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with scroll wheel."""
        # Ignore wheel events while panning (fixes trackpad two-finger click zoom spike)
        if self._pan_active:
            event.accept()
            return

        delta = event.angleDelta().y()

        # Ignore tiny deltas (trackpad noise)
        if abs(delta) < 10:
            event.accept()
            return

        # Scale zoom factor proportionally to scroll amount (smoother trackpad experience)
        # Standard mouse wheel gives ~120 per notch, trackpads give smaller values
        # Base factor of 1.0015 per unit, clamped to reasonable range
        base_factor = 1.0 + (abs(delta) / 120.0) * 0.15  # Max ~15% per full notch
        base_factor = min(base_factor, 1.25)  # Clamp to prevent huge jumps

        zoom_factor = base_factor if delta > 0 else 1.0 / base_factor

        new_zoom = self._zoom * zoom_factor
        if self._min_zoom <= new_zoom <= self._max_zoom:
            self._zoom = new_zoom
            self.scale(zoom_factor, zoom_factor)

        event.accept()

    # ---------------------------------------------------------------
    # Keyboard events
    # ---------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard input."""
        key = event.key()

        if key == Qt.Key_R:
            if self.is_placing:
                # Rotate placement preview
                self.rotate_placement()
            elif self._selected_id:
                # Rotate selected primitive by 90 degrees
                self._rotate_selected()
            event.accept()
            return

        if key == Qt.Key_Escape:
            if self.is_placing:
                self.cancel_placement()
            else:
                self._clear_selection()
            event.accept()
            return

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            event.accept()
            return

        super().keyPressEvent(event)

    # ---------------------------------------------------------------
    # Drag-to-move
    # ---------------------------------------------------------------

    def _update_drag_ghost(self, mouse_cell: CellCoord):
        """Update or create the drag ghost at the given cell."""
        if not self._drag_item:
            return

        prim = self._drag_item.primitive
        footprint = prim.footprint

        # Create ghost if needed
        if self._drag_ghost is None and footprint:
            category = self._get_primitive_category(prim.primitive_type)
            self._drag_ghost = GhostCellItem(footprint, self._cell_size, category)
            self._drag_ghost.set_rotation(prim.rotation)
            self._scene.addItem(self._drag_ghost)
            self.setCursor(Qt.SizeAllCursor)
            # Dim the source item to show it's being moved
            self._drag_item.set_dragging(True)

        if self._drag_ghost and footprint:
            # Apply offset to get the actual target origin cell
            if self._drag_cell_offset:
                target_cell = CellCoord(
                    mouse_cell.x - self._drag_cell_offset.x,
                    mouse_cell.y - self._drag_cell_offset.y
                )
            else:
                target_cell = mouse_cell

            # Position ghost at target origin cell (using footprint-aware positioning)
            ghost_pos = self._cell_to_scene_for_footprint(target_cell, footprint, prim.rotation)
            self._drag_ghost.setPos(ghost_pos)
            self._drag_ghost.setVisible(True)

            # Check validity (excluding the dragged item itself)
            valid = self._can_move_to(prim.id, footprint, target_cell, prim.rotation)
            self._drag_ghost.set_valid(valid)

    def _can_move_to(self, prim_id: str, footprint: PrimitiveFootprint,
                     cell: CellCoord, rotation: int) -> bool:
        """Check if a primitive can be moved to a new position."""
        new_cells = set(footprint.occupied_cells(cell, rotation))
        for other_id, other_prim in self._layout.primitives.items():
            if other_id == prim_id:
                continue
            other_cells = set(other_prim.occupied_cells())
            if new_cells & other_cells:
                return False
        return True

    def _end_drag(self):
        """End the drag operation and clean up."""
        if self._drag_ghost:
            self._scene.removeItem(self._drag_ghost)
            self._drag_ghost = None
        # Reset dragging state on the source item
        if self._drag_item:
            self._drag_item.set_dragging(False)
        self._drag_item = None
        self._drag_start_cell = None
        self._drag_cell_offset = None
        self._drag_start_pos = None
        self.setCursor(Qt.ArrowCursor)

    # ---------------------------------------------------------------
    # Placement
    # ---------------------------------------------------------------

    def _place_at_cell(self, cell: CellCoord):
        """Attempt to place the current primitive at a cell."""
        if not self.is_placing or self._placement_footprint is None:
            return

        # Check validity
        if not self._layout.can_place_at(
            self._placement_footprint, cell, self._placement_rotation
        ):
            # Emit detailed failure message
            blocker = self.get_placement_blocker(
                cell, self._placement_footprint, self._placement_rotation
            )
            if blocker:
                self.status_message.emit(f"Cannot place: {blocker}")
            else:
                self.status_message.emit("Cannot place at this location")
            return

        # Emit command for placement
        from .commands import PlacePrimitiveCommand
        cmd = PlacePrimitiveCommand(
            primitive_type=self._placement_type,
            origin_x=cell.x,
            origin_y=cell.y,
            rotation=self._placement_rotation,
            z_offset=self._placement_z_offset,
            footprint=self._placement_footprint,
        )
        self.command_requested.emit(cmd)

        # If shift is NOT held, cancel placement mode
        # The actual selection happens after command execution
        if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
            self.cancel_placement()

    # ---------------------------------------------------------------
    # View control
    # ---------------------------------------------------------------

    def zoom_in(self):
        """Zoom in."""
        if self._zoom < self._max_zoom:
            self._zoom *= 1.15
            self.scale(1.15, 1.15)

    def zoom_out(self):
        """Zoom out."""
        if self._zoom > self._min_zoom:
            self._zoom /= 1.15
            self.scale(1 / 1.15, 1 / 1.15)

    def reset_view(self):
        """Reset zoom and center on origin."""
        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(0, 0)

    def fit_to_content(self):
        """Fit view to show all content."""
        if not self._cell_items:
            self.reset_view()
            return

        # Calculate bounding rect of all items
        items_rect = QRectF()
        for item in self._cell_items.values():
            items_rect = items_rect.united(
                item.mapRectToScene(item.boundingRect())
            )

        if items_rect.isValid():
            # Add padding
            padding = self._cell_size * 2
            items_rect.adjust(-padding, -padding, padding, padding)
            self.fitInView(items_rect, Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        if self._ghost_item:
            self._ghost_item.setVisible(False)
        super().leaveEvent(event)

    def refresh_from_layout(self, select_id: Optional[str] = None):
        """
        Refresh the visual items from the current layout.
        Called after command execution to sync visuals with data.
        """
        self._rebuild_items()
        if select_id and select_id in self._layout.primitives:
            self._select_item(select_id)
        elif self._selected_id and self._selected_id not in self._layout.primitives:
            self._clear_selection()

        # Recalculate flow path if visualization is active
        if self._show_flow:
            self._calculate_flow_path()
            self.viewport().update()

    def select_primitive(self, prim_id: str):
        """
        Public method to select a primitive by ID.

        Args:
            prim_id: The ID of the primitive to select.
        """
        if prim_id in self._layout.primitives:
            self._select_item(prim_id)

    # ---------------------------------------------------------------
    # Context Menu for Connections
    # ---------------------------------------------------------------

    def contextMenuEvent(self, event):
        """Handle right-click context menu for connections."""
        scene_pos = self.mapToScene(event.pos())

        # Check if click is near a connection line
        conn = self._find_connection_at(scene_pos)
        if conn:
            self._show_connection_context_menu(event.globalPos(), conn)
            event.accept()
            return

        super().contextMenuEvent(event)

    def _find_connection_at(self, scene_pos: QPointF):
        """Find a connection near the given scene position.

        Args:
            scene_pos: Position in scene coordinates

        Returns:
            Connection object if found, None otherwise
        """
        CLICK_TOLERANCE = 15  # Pixels tolerance for clicking on a line

        for conn in self._layout.connections:
            prim_a = self._layout.primitives.get(conn.primitive_a_id)
            prim_b = self._layout.primitives.get(conn.primitive_b_id)

            if not prim_a or not prim_b:
                continue

            center_a = self._get_primitive_center(prim_a)
            center_b = self._get_primitive_center(prim_b)

            if not center_a or not center_b:
                continue

            # Calculate distance from point to line segment
            dist = self._point_to_line_distance(
                scene_pos.x(), scene_pos.y(),
                center_a.x(), center_a.y(),
                center_b.x(), center_b.y()
            )

            if dist < CLICK_TOLERANCE:
                return conn

        return None

    def _point_to_line_distance(self, px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """Calculate shortest distance from point to line segment."""
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1

        # Handle degenerate case
        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Calculate projection parameter
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def _show_connection_context_menu(self, global_pos, conn):
        """Show context menu for a connection."""
        menu = QMenu(self)

        # Toggle secret action
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
        # Find and update the connection in the layout
        for i, c in enumerate(self._layout.connections):
            if (c.primitive_a_id == conn.primitive_a_id and
                c.portal_a_id == conn.portal_a_id and
                c.primitive_b_id == conn.primitive_b_id and
                c.portal_b_id == conn.portal_b_id):
                # Create new connection with toggled secret flag
                from .data_model import Connection
                new_conn = Connection(
                    primitive_a_id=c.primitive_a_id,
                    portal_a_id=c.portal_a_id,
                    primitive_b_id=c.primitive_b_id,
                    portal_b_id=c.portal_b_id,
                    is_secret=not c.is_secret
                )
                self._layout.connections[i] = new_conn
                break

        # Emit layout changed signal
        self.layout_changed.emit()
        # Trigger repaint
        self.viewport().update()
