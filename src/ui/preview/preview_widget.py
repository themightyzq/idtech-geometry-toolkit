"""
QOpenGLWidget-based preview for real-time 3D visualization.

Integrates orbit camera, mesh building, and OpenGL rendering.
"""

import os
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import QOpenGLWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent, QKeyEvent
import numpy as np

from .camera import OrbitCamera
from .renderer import PreviewRenderer, RenderMode
from .mesh_builder import MeshBuilder, RenderMesh, build_wireframe_mesh, build_surface_meshes, SurfaceMeshes

from quake_levelgenerator.src.conversion.map_writer import Brush
from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS
from quake_levelgenerator.src.ui import style_constants as sc


import sys as _sys
DEBUG_PREVIEW = False  # Set to True to enable debug output

def _debug(msg):
    if DEBUG_PREVIEW:
        print(f"[PREVIEW] {msg}")
        _sys.stdout.flush()


class GLWidget(QOpenGLWidget):
    """Core OpenGL widget for 3D rendering."""

    def __init__(self, parent=None):
        _debug("GLWidget.__init__ starting")
        # Set up OpenGL format - use minimal settings for macOS compatibility
        fmt = QSurfaceFormat()
        # Don't request specific version - let the system decide
        # macOS with Metal may only provide GL 2.1 compatibility
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        # No multisampling - keep it simple
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        # Also set format on the widget itself
        self.setFormat(fmt)

        self.camera = OrbitCamera()
        self.renderer = PreviewRenderer()
        self._mesh: Optional[RenderMesh] = None
        self._pending_wireframe = None  # Store wireframe data for deferred upload
        self._has_mesh = False
        self._initialized = False

        # Mouse tracking
        self._last_mouse_pos = None
        self._mouse_button = None

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # macOS/Metal compatibility attributes
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        # Create timers but DON'T start them yet - wait for GL init
        # Starting timers before initializeGL() causes race condition crashes
        self._debug_timer = QTimer()
        self._debug_timer.timeout.connect(self.update)
        # Timer will be started in initializeGL()

        # WASD continuous movement (TrenchBroom-style)
        self._held_keys = set()
        self._movement_timer = QTimer()
        self._movement_timer.timeout.connect(self._process_movement)
        # Timer will be started in initializeGL()

        _debug("GLWidget.__init__ complete")

    def initializeGL(self):
        """Initialize OpenGL resources."""
        _debug("initializeGL called")
        try:
            if self.renderer.initialize():
                self._initialized = True
                _debug(f"Renderer initialized successfully")
                # Upload any pending mesh data that arrived before GL was ready
                if self._mesh and not self._mesh.is_empty:
                    _debug(f"Uploading pending mesh: {self._mesh.triangle_count} triangles")
                    self.renderer.upload_solid_mesh(self._mesh)
                    if self._pending_wireframe:
                        verts, indices = self._pending_wireframe
                        _debug(f"Uploading pending wireframe: {len(indices)} segments")
                        self.renderer.upload_wireframe_mesh(verts, indices)
                else:
                    _debug(f"No pending mesh to upload (mesh={self._mesh})")

                # NOW it's safe to start the timers - GL context is ready
                _debug("Starting update timers")
                self._debug_timer.start(16)  # ~60fps refresh
                self._movement_timer.start(16)  # ~60fps movement updates
            else:
                _debug("WARNING: Renderer initialization returned False")
                print("Warning: Renderer initialization returned False")
        except Exception as e:
            _debug(f"ERROR: Renderer initialization failed: {e}")
            print(f"Warning: Renderer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False

    def resizeGL(self, w: int, h: int):
        """Handle widget resize."""
        _debug(f"resizeGL: {w}x{h}")
        from OpenGL.GL import glViewport
        glViewport(0, 0, w, h)
        self.camera.set_aspect(w, h)

    def paintGL(self):
        """Render the scene."""
        if not self._initialized:
            _debug("paintGL: renderer not initialized, skipping")
            return
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        cam_pos = self.camera.get_position()
        self.renderer.render(view, proj, cam_pos)

    def set_mesh(self, mesh: RenderMesh, wireframe_data=None, surface_meshes: Optional[SurfaceMeshes] = None):
        """Update the mesh to render.

        Args:
            mesh: Combined mesh for bounds/camera fitting
            wireframe_data: Optional (vertices, indices) for wireframe
            surface_meshes: Optional SurfaceMeshes for per-surface texturing
        """
        _debug(f"set_mesh called: {mesh.triangle_count} triangles, empty={mesh.is_empty}")
        self._mesh = mesh
        self._pending_wireframe = wireframe_data  # Store for deferred upload
        self._pending_surface_meshes = surface_meshes  # Store for deferred upload
        self._has_mesh = not mesh.is_empty

        if not self._initialized:
            _debug("set_mesh: renderer not initialized, deferring upload")
            # Renderer not ready yet - mesh will be uploaded in initializeGL
            return

        try:
            _debug("set_mesh: uploading mesh to GPU")
            self.makeCurrent()

            # Upload surface meshes if available (for per-surface texturing)
            if surface_meshes is not None:
                _debug("set_mesh: uploading surface meshes")
                self.renderer.upload_surface_meshes(surface_meshes)
            else:
                self.renderer.upload_solid_mesh(mesh)

            if wireframe_data:
                verts, indices = wireframe_data
                _debug(f"set_mesh: uploading wireframe {len(indices)} segments")
                self.renderer.upload_wireframe_mesh(verts, indices)

            self.doneCurrent()
            _debug("set_mesh: upload complete")
        except Exception as e:
            _debug(f"ERROR in set_mesh: {e}")
            print(f"Warning: Failed to upload mesh: {e}")
            import traceback
            traceback.print_exc()

        self.update()

    def fit_to_bounds(self):
        """Fit camera to current mesh bounds."""
        if self._mesh and not self._mesh.is_empty:
            _debug(f"fit_to_bounds: mesh bounds min={self._mesh.bounds_min} max={self._mesh.bounds_max}")
            self.camera.fit_to_bounds(self._mesh.bounds_min, self._mesh.bounds_max)
            _debug(f"fit_to_bounds: camera position={self.camera.position} yaw={self.camera.yaw} pitch={self.camera.pitch}")
            self.update()

    def set_render_mode(self, mode: RenderMode):
        """Set the rendering mode."""
        self.renderer.render_mode = mode
        self.update()

    def set_preset_view(self, preset: str):
        """Set camera to a preset view."""
        self.camera.set_preset_view(preset)
        self.update()

    def set_texture(self, texture_path: Optional[str]):
        """Set the texture to use for textured rendering.

        Args:
            texture_path: Path to texture file, or None to disable
        """
        if not self._initialized:
            return

        self.makeCurrent()
        self.renderer.set_texture(texture_path)
        self.doneCurrent()
        self.update()

    def set_surface_textures(self, textures: Dict[str, Optional[str]]):
        """Set textures for each surface type (floor, ceiling, wall).

        Args:
            textures: Dict mapping surface type to texture file path
        """
        if not self._initialized:
            return

        self.makeCurrent()
        self.renderer.set_surface_textures(textures)
        self.doneCurrent()
        self.update()

    # --- Mouse input ---

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.pos()
        # Treat Alt/Option+Left as middle-button (for trackpad users)
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

        if self._mouse_button == Qt.RightButton:
            # Mouselook (FPS-style) - right click drag to look around
            self.camera.rotate(-dx, dy)
        elif self._mouse_button == Qt.MiddleButton:
            # Pan - move perpendicular to view
            self.camera.pan(dx, dy)
        # Left button reserved for future selection

        self.update()
        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        # Ignore wheel events while panning (fixes trackpad two-finger click zoom spike)
        if self._mouse_button == Qt.MiddleButton:
            event.accept()
            return

        delta = event.angleDelta().y()

        # Ignore tiny deltas (trackpad noise)
        if abs(delta) < 10:
            event.accept()
            return

        # Reduced sensitivity for smoother trackpad experience
        # Scale factor based on delta magnitude instead of fixed multiplier
        self.camera.zoom(-delta * 0.0005)
        self.update()
        event.accept()

    # --- Keyboard input ---

    def _process_movement(self):
        """Process continuous WASD movement based on held keys."""
        if not self._held_keys:
            return

        forward = 0.0
        right = 0.0
        up = 0.0

        if Qt.Key_W in self._held_keys:
            forward += 1.0
        if Qt.Key_S in self._held_keys:
            forward -= 1.0
        if Qt.Key_A in self._held_keys:
            right -= 1.0
        if Qt.Key_D in self._held_keys:
            right += 1.0
        if Qt.Key_Q in self._held_keys:
            up -= 1.0
        if Qt.Key_E in self._held_keys:
            up += 1.0

        if forward != 0.0 or right != 0.0 or up != 0.0:
            self.camera.move_continuous(forward, right, up)
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        # Track WASD + Q/E for continuous movement
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._held_keys.add(key)
            event.accept()
            return

        # View presets (single press)
        if key == Qt.Key_1:
            self.set_preset_view('front')
        elif key == Qt.Key_2:
            self.set_preset_view('back')
        elif key == Qt.Key_3:
            self.set_preset_view('left')
        elif key == Qt.Key_4:
            self.set_preset_view('right')
        elif key == Qt.Key_5:
            self.set_preset_view('top')
        elif key == Qt.Key_6:
            self.set_preset_view('bottom')
        elif key == Qt.Key_F:
            self.fit_to_bounds()
        else:
            super().keyPressEvent(event)
            return

        self.update()
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        # Stop movement when key released
        if key in self._held_keys:
            self._held_keys.discard(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def cleanup(self):
        """Clean up OpenGL resources."""
        self.makeCurrent()
        self.renderer.cleanup()
        self.doneCurrent()


class PreviewWidget(QWidget):
    """Complete preview widget with toolbar and GL canvas."""

    # Signal emitted when preview needs to be regenerated
    preview_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_primitive_name: Optional[str] = None
        self._current_params: Dict[str, Any] = {}
        self._last_primitive_name: Optional[str] = None  # Track for auto-fit decision
        self._should_auto_fit: bool = True  # Only fit when primitive type changes
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._regenerate_preview)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # GL canvas
        self._gl_widget = GLWidget()
        layout.addWidget(self._gl_widget, stretch=1)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)

        # Render mode buttons - NO setCheckable to avoid macOS accessibility crash
        self._current_render_mode = RenderMode.SOLID  # Manual state tracking

        self._solid_btn = QPushButton("Solid")
        self._solid_btn.clicked.connect(lambda: self._set_mode(RenderMode.SOLID))
        self._solid_btn.setToolTip("Solid shaded view")
        toolbar.addWidget(self._solid_btn)

        self._wire_btn = QPushButton("Wire")
        self._wire_btn.clicked.connect(lambda: self._set_mode(RenderMode.WIREFRAME))
        self._wire_btn.setToolTip("Wireframe view")
        toolbar.addWidget(self._wire_btn)

        self._both_btn = QPushButton("Both")
        self._both_btn.clicked.connect(lambda: self._set_mode(RenderMode.SOLID_WIREFRAME))
        self._both_btn.setToolTip("Solid with wireframe overlay")
        toolbar.addWidget(self._both_btn)

        self._textured_btn = QPushButton("Textured")
        self._textured_btn.clicked.connect(lambda: self._set_mode(RenderMode.TEXTURED))
        self._textured_btn.setToolTip("Textured view (uses Wall texture from Edit > Texture Settings)")
        toolbar.addWidget(self._textured_btn)

        toolbar.addStretch()

        # Fit button
        fit_btn = QPushButton("Fit (F)")
        fit_btn.clicked.connect(self._gl_widget.fit_to_bounds)
        fit_btn.setToolTip("Fit geometry in view (F)")
        toolbar.addWidget(fit_btn)

        # Stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        toolbar.addWidget(self._stats_label)

        layout.addLayout(toolbar)

        # Style toolbar buttons - apply initial styles
        fit_btn.setStyleSheet(self._get_toolbar_btn_style(False))
        self._update_render_mode_styles()

    def _get_toolbar_btn_style(self, selected: bool) -> str:
        """Get toolbar button style based on selection state."""
        if selected:
            # In high contrast mode, use black text on bright cyan for readability
            text_color = "#000000" if sc.HIGH_CONTRAST_MODE else "white"
            return f"""
                QPushButton {{
                    padding: 4px 10px;
                    border: 1px solid {sc.SELECTED_STATE};
                    border-radius: 4px;
                    background: {sc.SELECTED_STATE};
                    color: {text_color};
                    font-size: 11pt;
                }}
                QPushButton:hover {{
                    background: {sc.SELECTED_STATE_HOVER};
                }}
                QPushButton:focus {{
                    outline: 2px solid {sc.FOCUS_COLOR};
                    outline-offset: 2px;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    padding: 4px 10px;
                    border: 1px solid {sc.BORDER_MEDIUM};
                    border-radius: 4px;
                    background: {sc.BG_MEDIUM};
                    color: {sc.TEXT_PRIMARY};
                    font-size: 11pt;
                }}
                QPushButton:hover {{
                    background: {sc.BG_LIGHTER};
                }}
                QPushButton:focus {{
                    outline: 2px solid {sc.FOCUS_COLOR};
                    outline-offset: 2px;
                }}
            """

    def _update_render_mode_styles(self):
        """Update render mode button styles based on current mode."""
        self._solid_btn.setStyleSheet(self._get_toolbar_btn_style(self._current_render_mode == RenderMode.SOLID))
        self._wire_btn.setStyleSheet(self._get_toolbar_btn_style(self._current_render_mode == RenderMode.WIREFRAME))
        self._both_btn.setStyleSheet(self._get_toolbar_btn_style(self._current_render_mode == RenderMode.SOLID_WIREFRAME))
        self._textured_btn.setStyleSheet(self._get_toolbar_btn_style(
            self._current_render_mode in (RenderMode.TEXTURED, RenderMode.TEXTURED_WIREFRAME)))

    def _get_texture_from_settings(self) -> Optional[str]:
        """Get a valid texture file path from Texture Settings.

        Checks the wall texture first, then falls back to other surfaces.
        Only returns paths that point to actual existing image files.

        Returns:
            Path to a valid texture file, or None if none found
        """
        from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS, SURFACE_TYPES

        # Check surfaces in priority order (wall first, as it's most common)
        priority_order = ["wall", "floor", "ceiling", "structural", "trim"]

        for surface in priority_order:
            raw_value = TEXTURE_SETTINGS.get_raw_value(surface)
            if raw_value and os.path.isfile(raw_value):
                return raw_value

        return None

    def _get_all_surface_textures(self) -> Dict[str, Optional[str]]:
        """Get texture file paths for all surface types.

        Returns:
            Dict mapping surface type to texture file path (or None if not set)
        """
        from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS, SURFACE_TYPES

        textures = {}
        for surface in SURFACE_TYPES:  # floor, ceiling, wall, structural, trim
            raw_value = TEXTURE_SETTINGS.get_raw_value(surface)
            if raw_value and os.path.isfile(raw_value):
                textures[surface] = raw_value
            else:
                textures[surface] = None

        return textures

    def _set_mode(self, mode: RenderMode):
        """Set render mode and update button states."""
        _debug(f"_set_mode({mode})")
        # If switching to textured mode, get textures from Texture Settings
        if mode in (RenderMode.TEXTURED, RenderMode.TEXTURED_WIREFRAME):
            # Get per-surface textures
            surface_textures = self._get_all_surface_textures()
            _debug(f"Got surface textures: {surface_textures}")

            # Check if at least one texture is set
            has_any_texture = any(t is not None for t in surface_textures.values())
            if has_any_texture:
                self._gl_widget.set_surface_textures(surface_textures)
                _debug("Surface textures set on GL widget")
            else:
                # No valid texture file set - show message and stay on current mode
                QMessageBox.information(
                    self,
                    "No Texture File Set",
                    "To use textured preview, set image file paths for textures "
                    "in Edit > Texture Settings.\n\n"
                    "Use the '...' button to browse for TGA, PNG, or JPG files."
                )
                return

        self._current_render_mode = mode
        self._update_render_mode_styles()
        self._gl_widget.set_render_mode(mode)

    def update_primitive(self, primitive_name: str, params: Dict[str, Any]):
        """Update preview with new primitive parameters.

        Uses debouncing to avoid regenerating on every slider tick.
        Camera position is preserved when only parameters change.
        """
        _debug(f"update_primitive: {primitive_name}")

        # Only auto-fit when primitive type changes, not on parameter changes
        if primitive_name != self._last_primitive_name:
            self._should_auto_fit = True
            self._last_primitive_name = primitive_name

        self._current_primitive_name = primitive_name
        self._current_params = params.copy()

        # Debounce: wait 100ms after last change before regenerating
        self._debounce_timer.start(100)

    def _regenerate_preview(self):
        """Regenerate the preview mesh from current primitive."""
        _debug(f"_regenerate_preview: primitive={self._current_primitive_name}")
        if not self._current_primitive_name:
            _debug("_regenerate_preview: no primitive name, aborting")
            return

        cls = PRIMITIVE_CATALOG.get_primitive(self._current_primitive_name)
        if cls is None:
            _debug(f"_regenerate_preview: primitive '{self._current_primitive_name}' not found in catalog")
            return

        try:
            # Create primitive instance
            instance = cls()
            instance.apply_params(self._current_params)

            # Set surface textures from TEXTURE_SETTINGS
            # This ensures primitives use the configured textures for each surface type
            instance._texture_wall = TEXTURE_SETTINGS.get_texture("wall")
            instance._texture_floor = TEXTURE_SETTINGS.get_texture("floor")
            instance._texture_ceiling = TEXTURE_SETTINGS.get_texture("ceiling")
            instance._texture_trim = TEXTURE_SETTINGS.get_texture("trim")
            instance._texture_structural = TEXTURE_SETTINGS.get_texture("structural")

            # Generate brushes
            brushes = instance.generate()
            _debug(f"_regenerate_preview: generated {len(brushes)} brushes")

            # Build mesh
            self._update_from_brushes(brushes)

        except Exception as e:
            _debug(f"ERROR in _regenerate_preview: {e}")
            print(f"Preview generation failed: {e}")
            import traceback
            traceback.print_exc()
            self._stats_label.setText(f"Error: {e}")

    def _update_from_brushes(self, brushes: List[Brush]):
        """Update preview from a list of brushes."""
        _debug(f"_update_from_brushes: {len(brushes)} brushes")

        # Build surface meshes for per-surface texturing
        surface_meshes = build_surface_meshes(brushes)
        _debug(f"_update_from_brushes: built surface meshes - floor={surface_meshes.floor.triangle_count}, "
               f"ceiling={surface_meshes.ceiling.triangle_count}, wall={surface_meshes.wall.triangle_count}")

        # Also build combined mesh for bounds/stats
        builder = MeshBuilder()
        builder.add_brushes(brushes)
        mesh = builder.build()
        _debug(f"_update_from_brushes: built combined mesh with {mesh.triangle_count} tris, {mesh.vertex_count} verts")

        # Build wireframe
        wire_verts, wire_indices = build_wireframe_mesh(brushes)
        _debug(f"_update_from_brushes: built wireframe with {len(wire_indices)} segments")

        # Pass both surface meshes and combined mesh
        self._gl_widget.set_mesh(mesh, (wire_verts, wire_indices), surface_meshes)

        # Only auto-fit when primitive type changes (not on parameter changes)
        if mesh.vertex_count > 0 and self._should_auto_fit:
            _debug("_update_from_brushes: fitting to bounds (primitive type changed)")
            self._gl_widget.fit_to_bounds()
            self._should_auto_fit = False

        # Update stats
        self._stats_label.setText(
            f"{len(brushes)} brushes | {mesh.triangle_count} tris | {mesh.vertex_count} verts"
        )

    def set_brushes(self, brushes: List[Brush]):
        """Directly set brushes to preview (for layout mode)."""
        self._update_from_brushes(brushes)

    def fit_to_bounds(self):
        """Fit the camera to the geometry bounds."""
        self._gl_widget.fit_to_bounds()

    def set_texture(self, texture_path: Optional[str]):
        """Set the texture to use for textured rendering.

        Args:
            texture_path: Path to texture file, or None to disable
        """
        self._gl_widget.set_texture(texture_path)

    def cleanup(self):
        """Clean up resources."""
        self._gl_widget.cleanup()
