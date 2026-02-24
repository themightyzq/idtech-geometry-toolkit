"""
Quad-view widget for multi-floor visualization.

Provides a 2x2 viewport layout similar to Blender/TrenchBroom:
- Top-Left: Top orthographic (XY plane) OR interactive GridCanvas (Layout mode)
- Top-Right: Front orthographic (XZ plane) OR LayoutFlowView (Layout mode)
- Bottom-Left: Side orthographic (YZ plane) OR LayoutFlowView (Layout mode)
- Bottom-Right: 3D Perspective view (always shows generated geometry)

In Module mode: all four panes show mesh-based views.
In Layout mode: top-left = interactive GridCanvas, front/side = layout flow
schematics showing Z-levels, bottom-right = 3D generated geometry.
"""

from typing import Optional, List, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QKeySequence
import numpy as np

from .orthographic_camera import ViewAxis
from .ortho_view_widget import OrthoViewWidget
from .preview_widget import GLWidget
from .mesh_builder import RenderMesh, SurfaceMeshes
from .layout_flow_view import LayoutFlowView, FlowViewAxis
from quake_levelgenerator.src.ui import style_constants as sc


class ViewSynchronizer(QObject):
    """Synchronizes view state across multiple orthographic views.

    Handles:
    - Zoom synchronization (all ortho views share same zoom)
    - Pan synchronization (shared axes sync between views)
    - Selection propagation (all views highlight same primitive)
    """

    selection_changed = pyqtSignal(str)  # primitive_id

    def __init__(self, top_view: OrthoViewWidget, front_view: OrthoViewWidget,
                 side_view: OrthoViewWidget):
        super().__init__()

        self._top = top_view
        self._front = front_view
        self._side = side_view

        self._updating = False  # Prevent circular updates

        # Connect signals
        self._top.zoom_changed.connect(lambda z: self._sync_zoom(self._top, z))
        self._front.zoom_changed.connect(lambda z: self._sync_zoom(self._front, z))
        self._side.zoom_changed.connect(lambda z: self._sync_zoom(self._side, z))

        self._top.pan_changed.connect(lambda a, v: self._sync_pan(self._top, a, v))
        self._front.pan_changed.connect(lambda a, v: self._sync_pan(self._front, a, v))
        self._side.pan_changed.connect(lambda a, v: self._sync_pan(self._side, a, v))

    def _sync_zoom(self, source: OrthoViewWidget, zoom: float):
        """Synchronize zoom level across all ortho views."""
        if self._updating:
            return

        self._updating = True
        try:
            for view in [self._top, self._front, self._side]:
                if view is not source:
                    view.set_zoom(zoom)
        finally:
            self._updating = False

    def _sync_pan(self, source: OrthoViewWidget, axis: str, value: float):
        """Synchronize pan on a specific axis to views that share it.

        Axis sharing:
        - X axis: TOP and FRONT views
        - Y axis: TOP and SIDE views
        - Z axis: FRONT and SIDE views
        """
        if self._updating:
            return

        self._updating = True
        try:
            if axis == 'x':
                # X is shared by TOP and FRONT
                for view in [self._top, self._front]:
                    if view is not source:
                        view.set_pan_axis(axis, value)
            elif axis == 'y':
                # Y is shared by TOP and SIDE
                for view in [self._top, self._side]:
                    if view is not source:
                        view.set_pan_axis(axis, value)
            elif axis == 'z':
                # Z is shared by FRONT and SIDE
                for view in [self._front, self._side]:
                    if view is not source:
                        view.set_pan_axis(axis, value)
        finally:
            self._updating = False

    def sync_selection(self, primitive_id: str):
        """Propagate selection to all views."""
        self.selection_changed.emit(primitive_id)
        # Future: views would highlight the primitive

    def fit_all_to_bounds(self):
        """Fit all views to current bounds."""
        for view in [self._top, self._front, self._side]:
            view.fit_to_bounds()


class QuadViewWidget(QWidget):
    """Container widget with 2x2 viewport layout.

    Provides Blender/TrenchBroom-style quad view for multi-floor visualization.

    Module mode: All four panes show mesh-based ortho/perspective views.
    Layout mode: Top-left = interactive GridCanvas with palette bar,
                 Front/Side = schematic layout flow views showing Z-levels,
                 Bottom-right = 3D perspective with generated geometry.

    Layout:
    +------------------+------------------+
    |   Top (XY) /     |   Front (XZ) /   |
    |   GridCanvas     |   FlowView       |
    +------------------+------------------+
    |   Side (YZ) /    |   3D Perspective  |
    |   FlowView       |                  |
    +------------------+------------------+

    Signals:
        maximized_changed: Emitted when a pane is maximized/restored (str pane_name or None)
        command_requested: Emitted when quad GridCanvas requests a command
        regen_requested: Emitted after debounce to request geometry regeneration
        primitive_selected: Emitted when a primitive is selected in quad canvas
        selection_cleared: Emitted when selection is cleared in quad canvas
    """

    maximized_changed = pyqtSignal(object)  # str or None
    command_requested = pyqtSignal(object)  # Command from quad GridCanvas
    regen_requested = pyqtSignal()  # Debounced geometry regen request
    primitive_selected = pyqtSignal(str)  # primitive_id from quad canvas
    selection_cleared = pyqtSignal()  # selection cleared in quad canvas

    # Pane identifiers
    PANE_TOP = "top"
    PANE_FRONT = "front"
    PANE_SIDE = "side"
    PANE_3D = "3d"

    def __init__(self, parent=None):
        super().__init__(parent)

        self._maximized_pane: Optional[str] = None
        self._mesh: Optional[RenderMesh] = None
        self._wireframe_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._surface_meshes: Optional[SurfaceMeshes] = None

        # Interactive canvas state (created lazily)
        self._grid_canvas = None
        self._interactive_mode = False

        # Layout flow views (created lazily)
        self._front_flow_view = None
        self._side_flow_view = None

        # Debounce timer for geometry regeneration
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(300)
        self._regen_timer.timeout.connect(self._on_regen_timeout)

        self._init_ui()
        self._setup_synchronizer()

    def _init_ui(self):
        """Initialize the UI with nested splitters."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create view widgets
        self._top_view = OrthoViewWidget(ViewAxis.TOP)
        self._front_view = OrthoViewWidget(ViewAxis.FRONT)
        self._side_view = OrthoViewWidget(ViewAxis.SIDE)
        self._perspective_view = GLWidget()

        # --- Top-left pane: QStackedWidget ---
        # No background paint — background painting over a QOpenGLWidget child
        # causes blank rendering on macOS due to Core Animation layer conflicts.
        self._top_pane_stack = QStackedWidget()
        self._top_pane_stack.setStyleSheet(f"""
            QStackedWidget {{
                border: 1px solid {sc.BORDER_MEDIUM};
            }}
        """)
        self._top_pane_stack.addWidget(self._top_view)  # Index 0: OrthoView
        # Index 1: Interactive container (created lazily)
        self._top_frame = self._top_pane_stack

        # --- Front pane: QStackedWidget ---
        self._front_pane_stack = QStackedWidget()
        self._front_pane_stack.setStyleSheet(f"""
            QStackedWidget {{
                border: 1px solid {sc.BORDER_MEDIUM};
            }}
        """)
        self._front_pane_stack.addWidget(self._front_view)  # Index 0: OrthoView
        # Index 1: LayoutFlowView (created lazily)
        self._front_frame = self._front_pane_stack

        # --- Side pane: QStackedWidget ---
        self._side_pane_stack = QStackedWidget()
        self._side_pane_stack.setStyleSheet(f"""
            QStackedWidget {{
                border: 1px solid {sc.BORDER_MEDIUM};
            }}
        """)
        self._side_pane_stack.addWidget(self._side_view)  # Index 0: OrthoView
        # Index 1: LayoutFlowView (created lazily)
        self._side_frame = self._side_pane_stack

        # --- 3D Perspective pane ---
        # GL-safe frame: no background paint to avoid occluding QOpenGLWidget on macOS.
        self._perspective_frame = QStackedWidget()
        self._perspective_frame.setStyleSheet(f"""
            QStackedWidget {{
                border: 1px solid {sc.BORDER_MEDIUM};
            }}
        """)
        self._perspective_frame.addWidget(self._perspective_view)

        # Create nested splitters for 2x2 layout
        # Top row
        self._top_splitter = QSplitter(Qt.Horizontal)
        self._top_splitter.addWidget(self._top_frame)
        self._top_splitter.addWidget(self._front_frame)
        self._top_splitter.setSizes([500, 500])

        # Bottom row
        self._bottom_splitter = QSplitter(Qt.Horizontal)
        self._bottom_splitter.addWidget(self._side_frame)
        self._bottom_splitter.addWidget(self._perspective_frame)
        self._bottom_splitter.setSizes([500, 500])

        # Main vertical splitter
        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(self._top_splitter)
        self._main_splitter.addWidget(self._bottom_splitter)
        self._main_splitter.setSizes([400, 400])

        # Style splitter handles
        splitter_style = """
            QSplitter::handle {
                background: #555;
            }
            QSplitter::handle:hover {
                background: #777;
            }
        """
        self._main_splitter.setStyleSheet(splitter_style)
        self._top_splitter.setStyleSheet(splitter_style)
        self._bottom_splitter.setStyleSheet(splitter_style)

        self._main_splitter.setHandleWidth(4)
        self._top_splitter.setHandleWidth(4)
        self._bottom_splitter.setHandleWidth(4)

        # Stacked widget for normal view vs maximized view
        self._stack = QStackedWidget()
        self._stack.addWidget(self._main_splitter)

        # Single-pane container (for maximized state)
        self._maximized_container = QWidget()
        self._maximized_layout = QVBoxLayout(self._maximized_container)
        self._maximized_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._maximized_container)

        layout.addWidget(self._stack)

    def _ensure_interactive_widgets(self):
        """Lazily create the GridCanvas on first use.

        The quad view top-left pane is a clean GridCanvas — no extra controls.
        Module placement uses the main layout editor's palette; the quad view
        is for viewing, selecting, moving, and adjusting Z-offsets.
        """
        if self._grid_canvas is not None:
            return

        from quake_levelgenerator.src.ui.widgets.layout_editor.grid_canvas import GridCanvas

        self._grid_canvas = GridCanvas()

        # Add directly to top pane stack (index 1)
        self._top_pane_stack.addWidget(self._grid_canvas)

        # Forward GridCanvas signals through QuadViewWidget
        self._grid_canvas.command_requested.connect(self.command_requested.emit)
        self._grid_canvas.primitive_selected.connect(self.primitive_selected.emit)
        self._grid_canvas.selection_cleared.connect(self.selection_cleared.emit)
        self._grid_canvas.layout_changed.connect(self._on_canvas_layout_changed)

        # Sync grid canvas selection to flow views
        self._grid_canvas.primitive_selected.connect(self._on_grid_canvas_selection)
        self._grid_canvas.selection_cleared.connect(self._on_grid_canvas_selection_cleared)

    def _ensure_flow_views(self):
        """Lazily create LayoutFlowView widgets for front and side panes."""
        if self._front_flow_view is not None:
            return

        self._front_flow_view = LayoutFlowView(FlowViewAxis.FRONT)
        self._front_pane_stack.addWidget(self._front_flow_view)  # Index 1

        self._side_flow_view = LayoutFlowView(FlowViewAxis.SIDE)
        self._side_pane_stack.addWidget(self._side_flow_view)  # Index 1

        # Forward flow view signals through QuadViewWidget
        for fv in (self._front_flow_view, self._side_flow_view):
            fv.primitive_selected.connect(self._on_flow_view_selection)
            fv.selection_cleared.connect(self._on_flow_view_selection_cleared)
            fv.command_requested.connect(self._on_flow_view_command)
            fv.layout_changed.connect(self._on_canvas_layout_changed)

    def _on_canvas_layout_changed(self):
        """Handle direct layout changes from canvas."""
        # Update flow views when layout changes
        self._refresh_flow_views()
        self.request_regen()

    def _on_flow_view_selection(self, prim_id: str):
        """Handle primitive selection from a flow view - sync to all views."""
        # Sync selection to the other flow view and grid canvas
        self._sync_selection(prim_id)
        self.primitive_selected.emit(prim_id)

    def _on_flow_view_selection_cleared(self):
        """Handle selection cleared from a flow view."""
        self._sync_selection(None)
        self.selection_cleared.emit()

    def _on_flow_view_command(self, command):
        """Handle command from a flow view (e.g. SetZOffsetCommand)."""
        self.command_requested.emit(command)

    def _on_grid_canvas_selection(self, prim_id: str):
        """Handle selection from grid canvas - sync to flow views."""
        if self._front_flow_view:
            self._front_flow_view.set_selected(prim_id)
        if self._side_flow_view:
            self._side_flow_view.set_selected(prim_id)

    def _on_grid_canvas_selection_cleared(self):
        """Handle selection cleared from grid canvas - sync to flow views."""
        if self._front_flow_view:
            self._front_flow_view.set_selected(None)
        if self._side_flow_view:
            self._side_flow_view.set_selected(None)

    def _sync_selection(self, prim_id: Optional[str]):
        """Sync selection state across all interactive views."""
        if self._front_flow_view:
            self._front_flow_view.set_selected(prim_id)
        if self._side_flow_view:
            self._side_flow_view.set_selected(prim_id)
        # Sync to grid canvas if available
        if self._grid_canvas and prim_id:
            self._grid_canvas.select_primitive(prim_id)
        elif self._grid_canvas and prim_id is None:
            self._grid_canvas._clear_selection()

    def _refresh_flow_views(self):
        """Refresh the flow views with current layout data."""
        if self._front_flow_view:
            self._front_flow_view.update()
        if self._side_flow_view:
            self._side_flow_view.update()

    def _setup_synchronizer(self):
        """Set up view synchronization."""
        self._sync = ViewSynchronizer(
            self._top_view,
            self._front_view,
            self._side_view
        )

    # --- Interactive mode ---

    def set_interactive_mode(self, layout_mode: bool):
        """Switch panes between mesh views (Module mode) and layout flow views.

        When layout_mode=True:
        - Top-left: Interactive GridCanvas + palette bar
        - Front: LayoutFlowView (XZ schematic showing Z-levels)
        - Side: LayoutFlowView (YZ schematic showing Z-levels)
        - 3D: Perspective view (unchanged, shows generated geometry)

        When layout_mode=False:
        - All panes revert to OrthoViewWidget / GLWidget (mesh rendering)

        Args:
            layout_mode: True for layout interactive mode, False for mesh views.
        """
        self._interactive_mode = layout_mode

        if layout_mode:
            self._ensure_interactive_widgets()
            self._ensure_flow_views()
            self._top_pane_stack.setCurrentIndex(1)    # GridCanvas
            self._front_pane_stack.setCurrentIndex(1)  # Front flow view
            self._side_pane_stack.setCurrentIndex(1)   # Side flow view
        else:
            self._top_pane_stack.setCurrentIndex(0)    # OrthoView (Top)
            self._front_pane_stack.setCurrentIndex(0)  # OrthoView (Front)
            self._side_pane_stack.setCurrentIndex(0)   # OrthoView (Side)

    def set_layout(self, layout):
        """Set the DungeonLayout on the interactive canvas and flow views.

        Should only be called when interactive mode is active (or about to be).
        Automatically fits all views to show the layout content.

        Args:
            layout: DungeonLayout instance (shared with LayoutEditorWidget).
        """
        if not self._interactive_mode:
            return
        self._ensure_interactive_widgets()
        self._ensure_flow_views()
        self._grid_canvas.set_layout(layout)
        # Center the grid canvas on the layout content so modules are visible
        self._grid_canvas.fit_to_content()
        self._front_flow_view.set_layout(layout)
        self._side_flow_view.set_layout(layout)
        self._front_flow_view.fit_to_layout()
        self._side_flow_view.fit_to_layout()

    def refresh_layout_canvas(self, select_id=None):
        """Refresh the interactive GridCanvas and flow views after external changes.

        Args:
            select_id: Optional primitive ID to select after refresh.
        """
        if self._grid_canvas:
            self._grid_canvas.refresh_from_layout(select_id)
        self._refresh_flow_views()

    def fit_flow_views(self):
        """Fit the layout flow views to show all modules."""
        if self._front_flow_view:
            self._front_flow_view.fit_to_layout()
        if self._side_flow_view:
            self._side_flow_view.fit_to_layout()

    def focus_on_selected(self):
        """Focus all interactive panels on the selected module."""
        if self._grid_canvas:
            self._grid_canvas.focus_on_selected()
        if self._front_flow_view:
            self._front_flow_view.focus_on_selected()
        if self._side_flow_view:
            self._side_flow_view.focus_on_selected()

    def get_grid_canvas(self):
        """Get the interactive GridCanvas, or None if not created."""
        return self._grid_canvas

    def is_interactive(self) -> bool:
        """Check if the top-left pane is in interactive mode."""
        return self._interactive_mode

    # --- Debounced regeneration ---

    def request_regen(self):
        """Request geometry regeneration with debounce."""
        self._regen_timer.start()

    def _on_regen_timeout(self):
        """Debounce timer fired - emit regen request."""
        self.regen_requested.emit()

    # --- Mesh data ---

    def set_mesh(self, mesh: RenderMesh,
                 wireframe_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 surface_meshes: Optional[SurfaceMeshes] = None):
        """Set mesh data for views.

        In interactive mode, mesh data goes only to the 3D perspective view
        (the ortho views are hidden behind GridCanvas/FlowViews).
        In module mode, mesh data goes to all views.

        Args:
            mesh: Combined mesh for rendering
            wireframe_data: Optional (vertices, indices) for wireframe
            surface_meshes: Optional SurfaceMeshes for per-surface texturing
        """
        self._mesh = mesh
        self._wireframe_data = wireframe_data
        self._surface_meshes = surface_meshes

        if not self._interactive_mode:
            # Module mode: update all ortho views
            self._top_view.set_mesh(mesh, wireframe_data)
            self._front_view.set_mesh(mesh, wireframe_data)
            self._side_view.set_mesh(mesh, wireframe_data)

        # Always update 3D view
        self._perspective_view.set_mesh(mesh, wireframe_data, surface_meshes)

    def fit_all_to_bounds(self):
        """Fit all visible views to current bounds."""
        if self._interactive_mode:
            # In layout mode: fit 3D view and flow views
            self._perspective_view.fit_to_bounds()
            self.fit_flow_views()
        else:
            # In module mode: fit all ortho views and 3D
            self._top_view.fit_to_bounds()
            self._front_view.fit_to_bounds()
            self._side_view.fit_to_bounds()
            self._perspective_view.fit_to_bounds()

    def clear(self):
        """Clear all views by setting empty mesh data."""
        empty_mesh = RenderMesh(
            vertices=np.array([], dtype=np.float32).reshape(0, 11),
            indices=np.array([], dtype=np.uint32).reshape(0, 3),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, 0.0, 0.0)
        )
        self.set_mesh(empty_mesh, None, None)

    def maximize_pane(self, pane: str):
        """Maximize a single pane to fill the entire widget.

        Args:
            pane: Pane identifier (PANE_TOP, PANE_FRONT, PANE_SIDE, PANE_3D)
        """
        if self._maximized_pane == pane:
            return

        # Get the view to maximize
        view_map = {
            self.PANE_TOP: self._top_frame,
            self.PANE_FRONT: self._front_frame,
            self.PANE_SIDE: self._side_frame,
            self.PANE_3D: self._perspective_frame,
        }

        frame = view_map.get(pane)
        if not frame:
            return

        # Remove from splitter and add to maximized container
        self._maximized_layout.addWidget(frame)
        self._stack.setCurrentIndex(1)
        self._maximized_pane = pane

        self.maximized_changed.emit(pane)

    def restore_quad_view(self):
        """Restore quad-view layout from maximized state."""
        if self._maximized_pane is None:
            return

        # Restore frames to their original splitters
        self._top_splitter.insertWidget(0, self._top_frame)
        self._top_splitter.insertWidget(1, self._front_frame)
        self._bottom_splitter.insertWidget(0, self._side_frame)
        self._bottom_splitter.insertWidget(1, self._perspective_frame)

        # Reset splitter sizes
        self._top_splitter.setSizes([500, 500])
        self._bottom_splitter.setSizes([500, 500])
        self._main_splitter.setSizes([400, 400])

        self._stack.setCurrentIndex(0)
        self._maximized_pane = None

        self.maximized_changed.emit(None)

    def toggle_maximize(self, pane: str):
        """Toggle between maximized and quad-view for a pane."""
        if self._maximized_pane == pane:
            self.restore_quad_view()
        else:
            self.maximize_pane(pane)

    def is_maximized(self) -> bool:
        """Check if a pane is currently maximized."""
        return self._maximized_pane is not None

    def get_maximized_pane(self) -> Optional[str]:
        """Get the currently maximized pane, or None."""
        return self._maximized_pane

    # --- Delegate methods to perspective view ---

    def set_render_mode(self, mode):
        """Set render mode on 3D perspective view."""
        self._perspective_view.set_render_mode(mode)

    def set_texture(self, texture_path: Optional[str]):
        """Set texture on 3D perspective view."""
        self._perspective_view.set_texture(texture_path)

    def set_surface_textures(self, textures):
        """Set surface textures on 3D perspective view."""
        self._perspective_view.set_surface_textures(textures)

    # --- Settings persistence ---

    def save_splitter_state(self, settings: QSettings):
        """Save splitter positions to settings."""
        settings.setValue("quad_main_splitter", self._main_splitter.saveState())
        settings.setValue("quad_top_splitter", self._top_splitter.saveState())
        settings.setValue("quad_bottom_splitter", self._bottom_splitter.saveState())

    def restore_splitter_state(self, settings: QSettings):
        """Restore splitter positions from settings."""
        main_state = settings.value("quad_main_splitter")
        if main_state:
            self._main_splitter.restoreState(main_state)

        top_state = settings.value("quad_top_splitter")
        if top_state:
            self._top_splitter.restoreState(top_state)

        bottom_state = settings.value("quad_bottom_splitter")
        if bottom_state:
            self._bottom_splitter.restoreState(bottom_state)

    # --- Cleanup ---

    def cleanup(self):
        """Clean up OpenGL resources and widget references."""
        self._top_view.cleanup()
        self._front_view.cleanup()
        self._side_view.cleanup()
        self._perspective_view.cleanup()
        if self._grid_canvas:
            self._grid_canvas.setScene(None)
            self._grid_canvas = None
        if self._front_flow_view:
            self._front_flow_view.set_layout(None)
            self._front_flow_view = None
        if self._side_flow_view:
            self._side_flow_view.set_layout(None)
            self._side_flow_view = None
