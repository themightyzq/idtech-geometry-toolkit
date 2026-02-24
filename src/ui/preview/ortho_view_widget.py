"""
Orthographic view widget for single axis-aligned viewport.

Provides a QOpenGLWidget for rendering geometry from a fixed orthographic
viewpoint (Top, Front, or Side). Used in quad-view layout.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, List

from PyQt5.QtWidgets import QOpenGLWidget, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent

from .orthographic_camera import OrthographicCamera, ViewAxis
from .mesh_builder import RenderMesh, SurfaceMeshes
from quake_levelgenerator.src.ui import style_constants as sc

try:
    from OpenGL.GL import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


class OrthoViewWidget(QOpenGLWidget):
    """OpenGL widget for orthographic axis-aligned view.

    Displays geometry from a fixed viewpoint (Top, Front, or Side).
    Supports pan and zoom but no rotation.

    Signals:
        zoom_changed: Emitted when zoom level changes (float)
        pan_changed: Emitted when pan changes on an axis (str axis, float value)
        view_updated: Emitted when view needs refresh
    """

    zoom_changed = pyqtSignal(float)
    pan_changed = pyqtSignal(str, float)  # axis name, value
    view_updated = pyqtSignal()

    def __init__(self, view_axis: ViewAxis, parent=None):
        # Set up OpenGL format for macOS compatibility
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)
        self.setFormat(fmt)

        self.camera = OrthographicCamera(view_axis)
        self._view_axis = view_axis
        self._initialized = False
        self._use_legacy = False

        # Mesh data
        self._solid_vbo: Optional[int] = None
        self._solid_ebo: Optional[int] = None
        self._solid_triangle_count = 0
        self._wire_vbo: Optional[int] = None
        self._wire_ebo: Optional[int] = None
        self._wire_line_count = 0

        # Pending data for upload after GL init
        self._pending_mesh: Optional[RenderMesh] = None
        self._pending_wireframe: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Bounds for fitting
        self._bounds_min: Optional[Tuple[float, float, float]] = None
        self._bounds_max: Optional[Tuple[float, float, float]] = None

        # Mouse state
        self._last_mouse_pos = None
        self._mouse_button = None

        # Grid settings
        self._grid_spacing = 64.0  # 64-unit grid (idTech standard)
        self._show_grid = True
        self._show_axes = True

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # macOS compatibility
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        # Refresh timer
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self.update)

        # Create label overlay (avoids QPainter/OpenGL mixing crash on macOS)
        self._view_label = QLabel(self)
        self._view_label.setText(self.camera.get_view_label())
        self._view_label.setStyleSheet("""
            QLabel {
                color: #3C3C3C;
                background: transparent;
                font-size: 11pt;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self._view_label.move(5, 5)
        self._view_label.raise_()

    def initializeGL(self):
        """Initialize OpenGL resources."""
        if not OPENGL_AVAILABLE:
            return

        if self._initialized:
            return

        # Check OpenGL version
        try:
            version = glGetString(GL_VERSION)
            if version:
                version_str = version.decode('utf-8')
                major = int(version_str.split('.')[0])
                self._use_legacy = (major < 3)
        except Exception:
            self._use_legacy = True

        # Create buffers
        self._solid_vbo = glGenBuffers(1)
        self._solid_ebo = glGenBuffers(1)
        self._wire_vbo = glGenBuffers(1)
        self._wire_ebo = glGenBuffers(1)

        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)

        self._initialized = True

        # Upload pending data
        if self._pending_mesh:
            self._upload_mesh(self._pending_mesh)
        if self._pending_wireframe:
            self._upload_wireframe(*self._pending_wireframe)

        # Start refresh timer
        self._refresh_timer.start(16)  # ~60fps

    def resizeGL(self, w: int, h: int):
        """Handle widget resize."""
        if not OPENGL_AVAILABLE:
            return
        glViewport(0, 0, w, h)
        self.camera.set_viewport(w, h)

    def paintGL(self):
        """Render the scene."""
        if not self._initialized or not OPENGL_AVAILABLE:
            return

        # Clear with light gray background
        glClearColor(0.85, 0.85, 0.85, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()

        # Draw grid first (behind geometry)
        if self._show_grid:
            self._draw_grid(view, proj)

        # Draw solid geometry
        self._render_solid(view, proj)

        # Draw wireframe
        self._render_wireframe(view, proj)

        # Draw axes last (on top)
        if self._show_axes:
            self._draw_axes(view, proj)

    def _draw_grid(self, view: np.ndarray, proj: np.ndarray):
        """Draw grid lines on the view plane."""
        glUseProgram(0)

        # Set up matrices (fixed-function)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.ascontiguousarray(proj.T, dtype=np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(np.ascontiguousarray(view.T, dtype=np.float32))

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # Grid visible through geometry

        # Calculate grid extent based on zoom
        extent = self.camera.zoom * 3
        spacing = self._grid_spacing

        # Adjust spacing based on zoom (coarser when zoomed out)
        while extent / spacing > 50:
            spacing *= 2
        while extent / spacing < 10 and spacing > 16:
            spacing /= 2

        # Minor grid (lighter)
        glColor4f(0.75, 0.75, 0.75, 0.5)
        glLineWidth(1.0)
        self._draw_grid_lines(extent, spacing)

        # Major grid (darker, every 4 minor lines)
        glColor4f(0.5, 0.5, 0.5, 0.8)
        glLineWidth(1.5)
        self._draw_grid_lines(extent, spacing * 4)

        glEnable(GL_DEPTH_TEST)

    def _draw_grid_lines(self, extent: float, spacing: float):
        """Draw grid lines with given spacing."""
        center = self.camera.center
        num_lines = int(extent / spacing) + 1

        glBegin(GL_LINES)

        if self._view_axis == ViewAxis.TOP:
            # XY plane
            for i in range(-num_lines, num_lines + 1):
                x = center[0] + i * spacing
                x = round(x / spacing) * spacing
                glVertex3f(x, center[1] - extent, center[2])
                glVertex3f(x, center[1] + extent, center[2])
            for i in range(-num_lines, num_lines + 1):
                y = center[1] + i * spacing
                y = round(y / spacing) * spacing
                glVertex3f(center[0] - extent, y, center[2])
                glVertex3f(center[0] + extent, y, center[2])

        elif self._view_axis == ViewAxis.FRONT:
            # XZ plane
            for i in range(-num_lines, num_lines + 1):
                x = center[0] + i * spacing
                x = round(x / spacing) * spacing
                glVertex3f(x, center[1], center[2] - extent)
                glVertex3f(x, center[1], center[2] + extent)
            for i in range(-num_lines, num_lines + 1):
                z = center[2] + i * spacing
                z = round(z / spacing) * spacing
                glVertex3f(center[0] - extent, center[1], z)
                glVertex3f(center[0] + extent, center[1], z)

        else:  # SIDE
            # YZ plane
            for i in range(-num_lines, num_lines + 1):
                y = center[1] + i * spacing
                y = round(y / spacing) * spacing
                glVertex3f(center[0], y, center[2] - extent)
                glVertex3f(center[0], y, center[2] + extent)
            for i in range(-num_lines, num_lines + 1):
                z = center[2] + i * spacing
                z = round(z / spacing) * spacing
                glVertex3f(center[0], center[1] - extent, z)
                glVertex3f(center[0], center[1] + extent, z)

        glEnd()

    def _draw_axes(self, view: np.ndarray, proj: np.ndarray):
        """Draw axis indicators in corner of view."""
        glUseProgram(0)

        # Set up orthographic projection for screen-space drawing
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), 0, self.height(), -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)

        # Draw in bottom-left corner
        origin_x = 30
        origin_y = 30
        axis_length = 20

        # Determine which axes to show based on view
        if self._view_axis == ViewAxis.TOP:
            # X horizontal (right), Y vertical (up)
            glBegin(GL_LINES)
            glColor3f(*sc.AXIS_X_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x + axis_length, origin_y)
            glColor3f(*sc.AXIS_Y_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x, origin_y + axis_length)
            glEnd()

        elif self._view_axis == ViewAxis.FRONT:
            # X horizontal (right), Z vertical (up)
            glBegin(GL_LINES)
            glColor3f(*sc.AXIS_X_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x + axis_length, origin_y)
            glColor3f(*sc.AXIS_Z_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x, origin_y + axis_length)
            glEnd()

        else:  # SIDE
            # Y horizontal (right), Z vertical (up)
            glBegin(GL_LINES)
            glColor3f(*sc.AXIS_Y_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x + axis_length, origin_y)
            glColor3f(*sc.AXIS_Z_COLOR_TUPLE)
            glVertex2f(origin_x, origin_y)
            glVertex2f(origin_x, origin_y + axis_length)
            glEnd()

        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _render_solid(self, view: np.ndarray, proj: np.ndarray):
        """Render solid geometry using fixed-function pipeline."""
        if self._solid_triangle_count == 0:
            return

        glUseProgram(0)

        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.ascontiguousarray(proj.T, dtype=np.float32))

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(np.ascontiguousarray(view.T, dtype=np.float32))

        # Simple lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Light from view direction
        if self._view_axis == ViewAxis.TOP:
            light_pos = [0.0, 0.0, 1000.0, 0.0]
        elif self._view_axis == ViewAxis.FRONT:
            light_pos = [0.0, 1000.0, 0.0, 0.0]
        else:
            light_pos = [1000.0, 0.0, 0.0, 0.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])

        # Bind and draw
        stride = 11 * 4  # 11 floats per vertex

        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._solid_ebo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))

        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(32))

        glDrawElements(GL_TRIANGLES, self._solid_triangle_count * 3, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)

    def _render_wireframe(self, view: np.ndarray, proj: np.ndarray):
        """Render wireframe overlay."""
        if self._wire_line_count == 0:
            return

        glUseProgram(0)

        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.ascontiguousarray(proj.T, dtype=np.float32))

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(np.ascontiguousarray(view.T, dtype=np.float32))

        glDisable(GL_LIGHTING)

        # Slight depth offset to prevent z-fighting
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

        glColor3f(0.0, 0.0, 0.0)  # Black wireframe
        glLineWidth(1.0)

        glBindBuffer(GL_ARRAY_BUFFER, self._wire_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wire_ebo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 12, ctypes.c_void_p(0))

        glDrawElements(GL_LINES, self._wire_line_count * 2, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_POLYGON_OFFSET_LINE)

    def set_mesh(self, mesh: RenderMesh, wireframe_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Set the mesh to render.

        Args:
            mesh: RenderMesh with vertices and indices
            wireframe_data: Optional (vertices, indices) for wireframe
        """
        # Always store bounds so fit_to_bounds() works even before GL init
        self._bounds_min = mesh.bounds_min
        self._bounds_max = mesh.bounds_max

        if not self._initialized:
            self._pending_mesh = mesh
            self._pending_wireframe = wireframe_data
            return

        self._upload_mesh(mesh)
        if wireframe_data:
            self._upload_wireframe(*wireframe_data)

        self.update()

    def _upload_mesh(self, mesh: RenderMesh):
        """Upload mesh data to GPU."""
        if mesh.is_empty:
            self._solid_triangle_count = 0
            return

        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._solid_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

        self._solid_triangle_count = mesh.triangle_count

    def _upload_wireframe(self, vertices: np.ndarray, indices: np.ndarray):
        """Upload wireframe data to GPU."""
        if len(vertices) == 0:
            self._wire_line_count = 0
            return

        glBindBuffer(GL_ARRAY_BUFFER, self._wire_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wire_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self._wire_line_count = len(indices)

    def fit_to_bounds(self):
        """Fit camera to current mesh bounds."""
        if self._bounds_min and self._bounds_max:
            self.camera.fit_to_bounds(self._bounds_min, self._bounds_max)
            self.update()

    def set_zoom(self, zoom: float):
        """Set zoom level (for synchronization)."""
        self.camera.zoom = zoom
        self.update()

    def set_pan_axis(self, axis: str, value: float):
        """Set pan on a specific axis (for synchronization)."""
        self.camera.set_pan_axis_value(axis, value)
        self.update()

    # --- Mouse handling ---

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.pos()
        # Alt+Left = Middle button (for trackpad users)
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier:
            self._mouse_button = Qt.MiddleButton
        else:
            self._mouse_button = event.button()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_mouse_pos = None
        self._mouse_button = None
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse_pos is None:
            return

        dx = event.x() - self._last_mouse_pos.x()
        dy = event.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.pos()

        if self._mouse_button in (Qt.MiddleButton, Qt.RightButton):
            # Pan
            self.camera.pan(dx, dy)
            self._emit_pan_changed()
            self.update()

        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        # Ignore wheel while panning
        if self._mouse_button == Qt.MiddleButton:
            event.accept()
            return

        delta = event.angleDelta().y()
        if abs(delta) < 10:
            event.accept()
            return

        # Zoom at cursor position
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.camera.zoom_at(factor, event.x(), event.y())

        self.zoom_changed.emit(self.camera.zoom)
        self.update()
        event.accept()

    def _emit_pan_changed(self):
        """Emit pan_changed signals for affected axes."""
        if self._view_axis == ViewAxis.TOP:
            self.pan_changed.emit('x', self.camera.center[0])
            self.pan_changed.emit('y', self.camera.center[1])
        elif self._view_axis == ViewAxis.FRONT:
            self.pan_changed.emit('x', self.camera.center[0])
            self.pan_changed.emit('z', self.camera.center[2])
        else:  # SIDE
            self.pan_changed.emit('y', self.camera.center[1])
            self.pan_changed.emit('z', self.camera.center[2])

    # --- Keyboard handling ---

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.fit_to_bounds()
        else:
            super().keyPressEvent(event)

    def cleanup(self):
        """Clean up OpenGL resources."""
        if not self._initialized:
            return

        self._refresh_timer.stop()

        self.makeCurrent()
        if self._solid_vbo:
            glDeleteBuffers(1, [self._solid_vbo])
        if self._solid_ebo:
            glDeleteBuffers(1, [self._solid_ebo])
        if self._wire_vbo:
            glDeleteBuffers(1, [self._wire_vbo])
        if self._wire_ebo:
            glDeleteBuffers(1, [self._wire_ebo])
        self.doneCurrent()
