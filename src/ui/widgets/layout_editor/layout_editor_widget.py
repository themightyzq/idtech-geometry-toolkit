"""
Main layout editor widget combining grid canvas, palette, and properties.

This widget provides the complete node-based layout editing experience.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QTabWidget, QGroupBox,
    QFrame, QAction, QMessageBox, QFileDialog,
    QStackedWidget, QScrollArea, QTextEdit,
)
from quake_levelgenerator.src.ui.safe_checkbox import SafeCheckBox
from quake_levelgenerator.src.ui.safe_spinbox import SafeSpinBox, SafeDoubleSpinBox
from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox

# Alias for compatibility - use safe versions to avoid macOS crashes
QSpinBox = SafeSpinBox
QDoubleSpinBox = SafeDoubleSpinBox
QComboBox = SafeComboBox  # Click to cycle, no dropdown
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
import json
from pathlib import Path

from typing import Optional, List, Dict, Any

from .data_model import DungeonLayout, PlacedPrimitive, PrimitiveFootprint, CellCoord, PortalOverride
from .grid_canvas import GridCanvas
from .palette_widget import PaletteWidget, get_footprint, get_category, PRIMITIVE_FOOTPRINTS
# ValidationPanel moved to main_window.py tabs
from .layout_generator import LayoutGenerator
from .portal_editor import PortalEditorWidget
from .commands import (
    CommandManager, Command, PlacePrimitiveCommand, DeletePrimitiveCommand,
    MovePrimitiveCommand, RotatePrimitiveCommand, SetZOffsetCommand,
    DuplicatePrimitiveCommand, SetPrimitiveParameterCommand,
)
from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.ui.style_constants import set_accessible
from quake_levelgenerator.src.ui.preview.preview_widget import PreviewWidget
from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG


class PropertyEditor(QWidget):
    """Editor for selected primitive properties.

    Provides full parameter parity with Module Mode by dynamically generating
    parameter controls from the primitive's get_parameter_schema().
    """

    properties_changed = pyqtSignal(str, dict)  # primitive_id, new_properties
    rotation_requested = pyqtSignal(str, int)  # primitive_id, new_rotation
    z_offset_requested = pyqtSignal(str, float)  # primitive_id, new_z_offset
    parameter_changed = pyqtSignal(str, str, object)  # primitive_id, param_key, new_value

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_prim: Optional[PlacedPrimitive] = None
        self._updating = False  # Prevent signal loops during programmatic updates
        self._param_widgets: Dict[str, QWidget] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Info section
        self._info_label = QLabel("No selection")
        self._info_label.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        layout.addWidget(self._info_label)

        # Position section
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout(pos_group)

        cell_row = QHBoxLayout()
        cell_row.addWidget(QLabel("Cell:"))
        self._cell_x_spin = QSpinBox()
        self._cell_x_spin.setRange(-100, 100)
        self._cell_x_spin.setEnabled(False)
        set_accessible(self._cell_x_spin, "Cell X", "X position of the primitive on the grid")
        cell_row.addWidget(self._cell_x_spin)
        cell_row.addWidget(QLabel(","))
        self._cell_y_spin = QSpinBox()
        self._cell_y_spin.setRange(-100, 100)
        self._cell_y_spin.setEnabled(False)
        set_accessible(self._cell_y_spin, "Cell Y", "Y position of the primitive on the grid")
        cell_row.addWidget(self._cell_y_spin)
        pos_layout.addLayout(cell_row)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        self._rotation_combo = QComboBox()
        self._rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self._rotation_combo.setEnabled(False)
        self._rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)
        set_accessible(self._rotation_combo, "Rotation",
                      "Rotation angle of the selected primitive")
        rot_row.addWidget(self._rotation_combo)
        pos_layout.addLayout(rot_row)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z Offset:"))
        self._z_spin = QSpinBox()
        self._z_spin.setRange(-512, 512)
        self._z_spin.setSingleStep(16)
        self._z_spin.setSuffix(" units")
        self._z_spin.setEnabled(False)
        self._z_spin.valueChanged.connect(self._on_z_offset_changed)
        set_accessible(self._z_spin, "Z Offset",
                      "Vertical offset of the primitive in units")
        z_row.addWidget(self._z_spin)
        pos_layout.addLayout(z_row)

        layout.addWidget(pos_group)

        # Dynamic parameter area - scrollable for many parameters
        self._params_group = QGroupBox("Module Parameters")
        self._params_layout = QVBoxLayout(self._params_group)
        self._params_scroll = QScrollArea()
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(self._params_group)
        # Expand to fill available space - this is the main content area
        self._params_scroll.setMinimumHeight(200)
        layout.addWidget(self._params_scroll, stretch=1)  # Stretch to fill space

        # Validation warnings display
        self._warning_group = QGroupBox("Validation")
        warning_layout = QVBoxLayout(self._warning_group)
        self._warning_display = QTextEdit()
        self._warning_display.setReadOnly(True)
        self._warning_display.setMinimumHeight(60)
        self._warning_display.setMaximumHeight(120)
        set_accessible(self._warning_display, "Validation Messages",
                      "Shows validation warnings and errors for current configuration")
        self._warning_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2a2a2a;
                color: #90EE90;
                font-family: Menlo, Monaco, 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #444;
                border-radius: 4px;
            }}
            QTextEdit:focus {{
                border-color: {sc.FOCUS_COLOR};
            }}
        """)
        warning_layout.addWidget(self._warning_display)
        self._warning_group.setVisible(False)  # Hidden until warnings exist
        layout.addWidget(self._warning_group)

    def set_primitive(self, prim: Optional[PlacedPrimitive]):
        """Set the primitive to edit."""
        self._current_prim = prim

        if prim is None:
            self._info_label.setText("No selection")
            self._cell_x_spin.setEnabled(False)
            self._cell_y_spin.setEnabled(False)
            self._rotation_combo.setEnabled(False)
            self._z_spin.setEnabled(False)
            self._clear_param_ui()
            self._warning_group.setVisible(False)
            return

        self._updating = True  # Prevent signal emission during update
        self._info_label.setText(f"{prim.primitive_type}")
        self._cell_x_spin.setValue(prim.origin_cell.x)
        self._cell_y_spin.setValue(prim.origin_cell.y)
        self._rotation_combo.setCurrentIndex(prim.rotation // 90)
        self._z_spin.setValue(int(prim.z_offset))
        self._updating = False

        # Enable controls
        self._cell_x_spin.setEnabled(False)  # Position change via drag-to-move
        self._cell_y_spin.setEnabled(False)
        self._rotation_combo.setEnabled(True)
        self._z_spin.setEnabled(True)

        # Build dynamic parameter UI from primitive's schema
        self._rebuild_param_ui(prim)

        # Validate and display
        self._validate_and_display()

    def _clear_param_ui(self):
        """Clear all dynamic parameter widgets."""
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._param_widgets.clear()

        # Add placeholder
        placeholder = QLabel("Select a primitive to edit")
        placeholder.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        self._params_layout.addWidget(placeholder)

    def _rebuild_param_ui(self, prim: PlacedPrimitive):
        """Rebuild dynamic parameter widgets from primitive's schema."""
        self._clear_param_ui()

        # Remove placeholder (clear_param_ui adds one)
        if self._params_layout.count() > 0:
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # Get schema from primitive class
        cls = PRIMITIVE_CATALOG.get_primitive(prim.primitive_type)
        if cls is None:
            return

        schema = cls.get_parameter_schema()
        if not schema:
            return

        # Common focus styles
        spin_focus = f"QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {sc.FOCUS_COLOR}; }}"
        combo_focus = f"QComboBox:focus {{ border: 2px solid {sc.FOCUS_COLOR}; }}"

        for key, spec in schema.items():
            row = QHBoxLayout()
            label = QLabel(spec.get("label", key))
            label.setMinimumWidth(60)
            label.setStyleSheet("font-size: 10pt;")
            row.addWidget(label)

            # Build tooltip from schema
            tooltip_parts = []
            if "description" in spec:
                tooltip_parts.append(spec["description"])
            ptype = spec.get("type", "float")
            if ptype in ("float", "int") and ("min" in spec or "max" in spec):
                min_val = spec.get("min", 0)
                max_val = spec.get("max", 9999 if ptype == "float" else 999)
                tooltip_parts.append(f"Range: {min_val}-{max_val}")
            tooltip = "\n".join(tooltip_parts) if tooltip_parts else None

            # Get current value (user override or default)
            current_val = prim.parameters.get(key, spec.get("default"))

            if ptype == "float":
                w = QDoubleSpinBox()
                w.setRange(spec.get("min", 0), spec.get("max", 9999))
                w.setValue(float(current_val) if current_val is not None else spec.get("default", 0))
                w.setSingleStep(0.1)
                w.setStyleSheet(spin_focus)
                w.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            elif ptype == "int":
                w = QSpinBox()
                w.setRange(int(spec.get("min", 0)), int(spec.get("max", 999)))
                w.setValue(int(current_val) if current_val is not None else int(spec.get("default", 0)))
                w.setStyleSheet(spin_focus)
                w.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            elif ptype == "bool":
                w = SafeCheckBox()
                w.setChecked(bool(current_val) if current_val is not None else bool(spec.get("default", False)))
                w.toggled.connect(lambda val, k=key: self._on_param_changed(k, val))
            elif ptype == "choice":
                w = QComboBox()
                w.addItems(spec.get("choices", []))
                default = current_val if current_val is not None else spec.get("default", "")
                idx = w.findText(str(default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                w.setStyleSheet(combo_focus)
                w.currentTextChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            else:
                w = QLabel(f"[{ptype}]")

            # Apply tooltip if we have one
            if tooltip and hasattr(w, 'setToolTip'):
                w.setToolTip(tooltip)

            # Apply accessibility label
            param_label = spec.get("label", key)
            param_desc = spec.get("description", "")
            set_accessible(w, param_label, param_desc)

            row.addWidget(w)
            container = QWidget()
            container.setLayout(row)
            self._params_layout.addWidget(container)
            self._param_widgets[key] = w

        # Add stretch at end
        self._params_layout.addStretch()

    def _on_param_changed(self, key: str, value):
        """Handle parameter value change from widget."""
        if self._updating or self._current_prim is None:
            return

        # Emit signal for command creation
        self.parameter_changed.emit(self._current_prim.id, key, value)

        # Revalidate
        self._validate_and_display()

    def _validate_and_display(self):
        """Validate current primitive and display warnings if any."""
        if self._current_prim is None:
            self._warning_group.setVisible(False)
            return

        cls = PRIMITIVE_CATALOG.get_primitive(self._current_prim.primitive_type)
        if cls is None:
            self._warning_group.setVisible(False)
            return

        # Create instance and apply current parameters
        instance = cls()
        params = self._get_current_params()
        instance.apply_params(params)

        # Check if primitive has validate() method
        if not hasattr(instance, 'validate'):
            self._warning_group.setVisible(False)
            return

        try:
            result = instance.validate()
        except Exception:
            self._warning_group.setVisible(False)
            return

        if not result.has_warnings():
            self._warning_display.setText("✓ Configuration valid")
            self._warning_display.setStyleSheet("""
                QTextEdit {
                    background-color: #1a3a1a;
                    color: #88ff88;
                    font-family: Menlo, Monaco, 'Courier New', monospace;
                    font-size: 10pt;
                }
            """)
            self._warning_group.setVisible(True)
            return

        # Format and display warnings
        lines = []
        has_error = False
        for w in result.warnings:
            if w.severity == "error":
                has_error = True
                lines.append(f"❌ {w.arm_name}: {w.message}")
            else:
                lines.append(f"⚠️ {w.arm_name}: {w.message}")

        self._warning_display.setText("\n".join(lines))

        # Style based on severity
        if has_error:
            self._warning_display.setStyleSheet("""
                QTextEdit {
                    background-color: #3a1a1a;
                    color: #ff8888;
                    font-family: Menlo, Monaco, 'Courier New', monospace;
                    font-size: 10pt;
                }
            """)
        else:
            self._warning_display.setStyleSheet("""
                QTextEdit {
                    background-color: #3a3a1a;
                    color: #ffff88;
                    font-family: Menlo, Monaco, 'Courier New', monospace;
                    font-size: 10pt;
                }
            """)
        self._warning_group.setVisible(True)

    def _get_current_params(self) -> Dict[str, Any]:
        """Get current parameter values from widgets."""
        params: Dict[str, Any] = {}
        for key, w in self._param_widgets.items():
            if isinstance(w, SafeDoubleSpinBox):
                params[key] = w.value()
            elif isinstance(w, SafeSpinBox):
                params[key] = w.value()
            elif isinstance(w, SafeCheckBox):
                params[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                params[key] = w.currentText()
        return params

    def _on_rotation_changed(self, index: int):
        """Handle rotation combo change."""
        if self._updating or self._current_prim is None:
            return

        new_rotation = index * 90
        if new_rotation != self._current_prim.rotation:
            self.rotation_requested.emit(self._current_prim.id, new_rotation)

    def _on_z_offset_changed(self, value: int):
        """Handle Z offset spinbox change."""
        if self._updating or self._current_prim is None:
            return

        if float(value) != self._current_prim.z_offset:
            self.z_offset_requested.emit(self._current_prim.id, float(value))

    def refresh_from_primitive(self):
        """Refresh widget values from current primitive (call after command execution)."""
        if self._current_prim is not None:
            self.set_primitive(self._current_prim)


class LayoutEditorWidget(QWidget):
    """Complete layout editor widget."""

    # Signals
    layout_changed = pyqtSignal()  # Emitted when layout is modified
    preview_requested = pyqtSignal(list)  # Emitted with brushes for 3D preview
    file_saved = pyqtSignal(str)   # Emitted with file path after save
    file_loaded = pyqtSignal(str)  # Emitted with file path after load
    # Emitted when undo/redo state changes: (can_undo, can_redo, undo_desc, redo_desc)
    undo_state_changed = pyqtSignal(bool, bool, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = DungeonLayout()
        self._command_manager = CommandManager()
        self._is_3d_view = False
        self._show_flow = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Build the UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter with visible handles for resizing
        self._editor_splitter = QSplitter(Qt.Horizontal)
        self._editor_splitter.setHandleWidth(6)
        self._editor_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #444;
            }
            QSplitter::handle:hover {
                background: #666;
            }
            QSplitter::handle:pressed {
                background: #888;
            }
        """)
        main_layout.addWidget(self._editor_splitter)

        # Left panel: Palette - responsive sizing
        self._palette = PaletteWidget()
        self._palette.setMinimumWidth(180)  # Allow shrinking at low res
        # No maximum - let user expand as needed
        self._editor_splitter.addWidget(self._palette)

        # Center: Grid canvas with toolbar
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Grid canvas (must be created BEFORE toolbar since toolbar connects to it)
        self._canvas = GridCanvas()
        # IMPORTANT: Sync the canvas to use the widget's layout reference
        self._canvas.set_layout(self._layout)

        # 3D Preview widget for visualization
        self._preview_3d = PreviewWidget()

        # Stacked widget to switch between 2D and 3D views
        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._canvas)      # Index 0: 2D grid view
        self._view_stack.addWidget(self._preview_3d)  # Index 1: 3D preview

        # Mode banner - shows current mode with helpful hints
        self._mode_banner = QLabel()
        self._mode_banner.setAlignment(Qt.AlignCenter)
        self._update_mode_banner(False, "")
        center_layout.addWidget(self._mode_banner)

        # Add stacked view (2D canvas or 3D preview)
        center_layout.addWidget(self._view_stack)

        # Status bar
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar.setStyleSheet("background: #2d2d2d; padding: 4px;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(8, 2, 8, 2)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #c0c0c0; font-size: 11pt;")
        set_accessible(self._status_label, "Status", "Current editor status")
        status_layout.addWidget(self._status_label)

        status_layout.addStretch()

        # Flow metrics label
        self._metrics_label = QLabel("")
        self._metrics_label.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        set_accessible(self._metrics_label, "Flow Metrics",
                      "Path length, room count, and dead ends statistics")
        status_layout.addWidget(self._metrics_label)

        # Separator
        sep = QLabel(" | ")
        sep.setStyleSheet("color: #555;")
        status_layout.addWidget(sep)

        self._cell_label = QLabel("Cell: (0, 0)")
        self._cell_label.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        set_accessible(self._cell_label, "Cell Position",
                      "Current mouse position on the grid")
        status_layout.addWidget(self._cell_label)

        center_layout.addWidget(status_bar)
        self._editor_splitter.addWidget(center_widget)

        # Right panel: Properties + Validation - scrollable for many parameters
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # Properties section
        props_label = QLabel("Properties")
        props_label.setStyleSheet("font-weight: bold; color: #ccc; font-size: 12pt;")
        right_layout.addWidget(props_label)

        self._property_editor = PropertyEditor()
        right_layout.addWidget(self._property_editor, stretch=1)

        # Portal Overrides section
        portal_label = QLabel("Portal Overrides")
        portal_label.setStyleSheet("font-weight: bold; color: #ccc; font-size: 12pt; margin-top: 12px;")
        right_layout.addWidget(portal_label)

        self._portal_editor = PortalEditorWidget()
        right_layout.addWidget(self._portal_editor)

        # No stretch at end - let PropertyEditor expand to fill available space
        right_scroll.setWidget(right_widget)

        # Right panel - responsive sizing
        right_scroll.setMinimumWidth(200)  # Allow shrinking at low res
        # No maximum - let user expand as needed
        self._editor_splitter.addWidget(right_scroll)

        # Set splitter sizes - give more room to side panels
        self._editor_splitter.setSizes([280, 600, 320])
        self._editor_splitter.setStretchFactor(0, 0)  # Palette: prefer fixed
        self._editor_splitter.setStretchFactor(1, 1)  # Canvas: stretches
        self._editor_splitter.setStretchFactor(2, 0)  # Properties: prefer fixed

    def _connect_signals(self):
        """Connect widget signals."""
        # Palette -> Canvas
        self._palette.primitive_selected.connect(self._on_palette_select)
        self._palette.selection_cleared.connect(self._on_palette_clear)
        self._palette.floor_level_changed.connect(self._on_floor_level_changed)

        # Canvas signals
        self._canvas.primitive_selected.connect(self._on_primitive_selected)
        self._canvas.selection_cleared.connect(self._on_selection_cleared)
        self._canvas.cell_hovered.connect(self._on_cell_hover)
        self._canvas.command_requested.connect(self._on_command_requested)
        self._canvas.status_message.connect(self._on_status_message)
        self._canvas.mode_changed.connect(self._update_mode_banner)

        # Property editor signals
        self._property_editor.rotation_requested.connect(self._on_rotation_requested)
        self._property_editor.z_offset_requested.connect(self._on_z_offset_requested)
        self._property_editor.parameter_changed.connect(self._on_parameter_changed)

        # Portal editor signals
        self._portal_editor.portal_override_changed.connect(self._on_portal_override_changed)

        # Note: Validation panel moved to main window's tab bar

    # ---------------------------------------------------------------
    # Signal handlers
    # ---------------------------------------------------------------

    def _on_palette_select(self, prim_type: str, footprint: PrimitiveFootprint,
                           category: str):
        """Handle palette primitive selection."""
        # All new primitives placed at Z=0, user adjusts via property editor
        self._canvas.start_placement(prim_type, footprint, category, z_offset=0.0)
        self._status_label.setText(f"Placing: {prim_type} (R to rotate, Esc to cancel)")

    def _on_palette_clear(self):
        """Handle palette selection cleared."""
        self._canvas.cancel_placement()
        self._status_label.setText("Ready")

    def _on_floor_level_changed(self, z_offset: int):
        """Handle floor level change (legacy - floor UI removed)."""
        # Floor level UI removed, this is kept for signal compatibility
        pass

    def _on_command_requested(self, command: Command):
        """Handle a command request from canvas or property editor."""
        if self._command_manager.execute(command, self._layout):
            self._on_layout_modified()

            # Handle command-specific post-execution logic
            if isinstance(command, PlacePrimitiveCommand) and command.placed_id:
                # Select the newly placed primitive
                self._canvas.refresh_from_layout(command.placed_id)
                self._canvas.select_primitive(command.placed_id)
            elif isinstance(command, DeletePrimitiveCommand):
                # Clear property editor on successful delete
                self._canvas.refresh_from_layout()
                self._property_editor.set_primitive(None)
            else:
                self._canvas.refresh_from_layout()

            # Update status
            self._status_label.setText(f"Executed: {command.description}")
        else:
            self._status_label.setText(f"Failed: {command.description}")

    def _on_primitive_selected(self, prim_id: str):
        """Handle primitive selection on canvas."""
        prim = self._layout.primitives.get(prim_id)
        self._property_editor.set_primitive(prim)
        self._portal_editor.set_primitive(prim)
        if prim:
            self._status_label.setText(f"Selected: {prim.primitive_type}")

    def _on_selection_cleared(self):
        """Handle selection cleared."""
        self._property_editor.set_primitive(None)
        self._portal_editor.set_primitive(None)
        self._status_label.setText("Ready")

    def _on_rotation_requested(self, prim_id: str, new_rotation: int):
        """Handle rotation change request from property editor."""
        cmd = RotatePrimitiveCommand(primitive_id=prim_id, new_rotation=new_rotation)
        self._on_command_requested(cmd)

    def _on_z_offset_requested(self, prim_id: str, new_z_offset: float):
        """Handle Z offset change request from property editor."""
        cmd = SetZOffsetCommand(primitive_id=prim_id, new_z_offset=new_z_offset)
        self._on_command_requested(cmd)

    def _on_parameter_changed(self, prim_id: str, param_key: str, new_value):
        """Handle parameter change request from property editor."""
        cmd = SetPrimitiveParameterCommand(
            primitive_id=prim_id,
            param_key=param_key,
            new_value=new_value
        )
        if self._command_manager.execute(cmd, self._layout):
            self._on_layout_modified()
            # Refresh property editor to show updated validation
            self._property_editor.refresh_from_primitive()
            self._status_label.setText(f"Set {param_key} = {new_value}")
        else:
            self._status_label.setText(f"Failed to set {param_key}")

    def _on_portal_override_changed(self, prim_id: str, portal_id: str, override: PortalOverride):
        """Handle portal override change from portal editor.

        Portal overrides are stored directly on the PlacedPrimitive and don't
        use the undo/redo system (they're considered layout metadata).
        """
        prim = self._layout.primitives.get(prim_id)
        if prim is None:
            return

        # Update the primitive's portal_overrides (already done by portal editor)
        # Now update connections based on new portal state
        self._layout.connections = [
            c for c in self._layout.connections
            if not (c.primitive_a_id == prim_id and c.portal_a_id == portal_id) and
               not (c.primitive_b_id == prim_id and c.portal_b_id == portal_id)
        ]

        # Re-run auto-connect if portal is still enabled
        if override.enabled:
            self._layout.auto_connect(prim_id)

        # Update UI
        self._canvas.refresh_from_layout()
        self._on_layout_modified()

        # Status message
        if override.enabled:
            z_msg = f" (z={override.z_level_override})" if override.z_level_override is not None else ""
            self._status_label.setText(f"Portal {portal_id} enabled{z_msg}")
        else:
            self._status_label.setText(f"Portal {portal_id} disabled")

    def _on_cell_hover(self, x: int, y: int):
        """Handle cell hover."""
        self._cell_label.setText(f"Cell: ({x}, {y})")

    def _on_status_message(self, message: str):
        """Handle status message from canvas."""
        self._status_label.setText(message)

    def _update_mode_banner(self, is_placing: bool, primitive_type: str):
        """Update the mode banner based on current editor mode."""
        if is_placing:
            self._mode_banner.setText(
                f"  PLACEMENT MODE: {primitive_type}  |  "
                f"R = Rotate  |  Click = Place  |  Shift+Click = Place Multiple  |  Esc = Cancel"
            )
            self._mode_banner.setStyleSheet("""
                QLabel {
                    background: #4CAF50;
                    color: white;
                    padding: 6px 8px;
                    font-weight: bold;
                    font-size: 11pt;
                }
            """)
        else:
            self._mode_banner.setText(
                "  SELECTION MODE  |  "
                "Click = Select  |  Drag = Move  |  Delete = Remove  |  "
                "Choose primitive from palette to place"
            )
            self._mode_banner.setStyleSheet("""
                QLabel {
                    background: #607D8B;
                    color: white;
                    padding: 6px 8px;
                    font-weight: bold;
                    font-size: 11pt;
                }
            """)

    def _on_layout_modified(self):
        """Called after layout is modified to update UI state."""
        # Note: Validation panel is updated via layout_changed signal in main window
        self._update_undo_redo_state()
        self._update_flow_metrics()
        self.layout_changed.emit()
        self._update_preview()

    def _update_undo_redo_state(self):
        """Update enabled state and emit signal for menu bar updates."""
        self.undo_state_changed.emit(
            self._command_manager.can_undo,
            self._command_manager.can_redo,
            self._command_manager.undo_description or "",
            self._command_manager.redo_description or ""
        )

    def _on_undo(self):
        """Handle undo action."""
        desc = self._command_manager.undo(self._layout)
        if desc:
            self._canvas.refresh_from_layout()
            self._on_layout_modified()
            self._status_label.setText(f"Undone: {desc}")

            # Update property editor if selection changed
            selected = self._canvas._selected_id
            if selected and selected in self._layout.primitives:
                self._property_editor.set_primitive(self._layout.primitives[selected])
            else:
                self._property_editor.set_primitive(None)

    def _on_redo(self):
        """Handle redo action."""
        desc = self._command_manager.redo(self._layout)
        if desc:
            self._canvas.refresh_from_layout()
            self._on_layout_modified()
            self._status_label.setText(f"Redone: {desc}")

            # Update property editor if selection changed
            selected = self._canvas._selected_id
            if selected and selected in self._layout.primitives:
                self._property_editor.set_primitive(self._layout.primitives[selected])
            else:
                self._property_editor.set_primitive(None)

    def _on_duplicate(self):
        """Handle duplicate action."""
        selected_id = self._canvas._selected_id
        if not selected_id or selected_id not in self._layout.primitives:
            self._status_label.setText("No primitive selected to duplicate")
            return

        # Calculate offset: try to place adjacent to original
        prim = self._layout.primitives[selected_id]
        footprint = prim.footprint
        offset_x = footprint.width_cells if footprint else 1
        offset_y = 0

        cmd = DuplicatePrimitiveCommand(
            source_primitive_id=selected_id,
            offset_x=offset_x,
            offset_y=offset_y
        )
        self._on_command_requested(cmd)

        # Select the new primitive if created
        if cmd.duplicated_id:
            self._canvas.select_primitive(cmd.duplicated_id)

    def _on_flow_toggled(self, checked: bool):
        """Handle flow visualization toggle."""
        self._show_flow = checked
        self._canvas.set_show_flow(checked)
        self._update_flow_metrics()

    def _on_3d_preview_toggled(self, checked: bool):
        """Handle 3D preview toggle."""
        self._is_3d_view = checked

        if checked:
            # Switch to 3D view and generate geometry
            self._view_stack.setCurrentIndex(1)
            self._refresh_3d_preview()
            self._status_label.setText("3D Preview - Use mouse to navigate, Ctrl+3 to return to 2D")
            self._mode_banner.setText(
                "  3D PREVIEW MODE  |  "
                "Right-drag = Look  |  WASD = Move  |  Q/E = Up/Down  |  "
                "Ctrl+3 = Return to 2D"
            )
            self._mode_banner.setStyleSheet("""
                QLabel {
                    background: #7B1FA2;
                    color: white;
                    padding: 6px 8px;
                    font-weight: bold;
                    font-size: 11pt;
                }
            """)
        else:
            # Switch back to 2D view
            self._view_stack.setCurrentIndex(0)
            self._status_label.setText("Ready")
            self._update_mode_banner(self._canvas.is_placing,
                                     self._canvas.placement_type or "")

    def _refresh_3d_preview(self):
        """Generate brushes from layout and update 3D preview."""
        if not self._is_3d_view:
            return

        if not self._layout.primitives:
            self._preview_3d.set_brushes([])
            return

        try:
            # Generate brushes using LayoutGenerator
            generator = LayoutGenerator(self._layout)
            result = generator.generate()
            self._preview_3d.set_brushes(result.brushes)
        except Exception as e:
            self._status_label.setText(f"3D preview error: {e}")
            self._preview_3d.set_brushes([])

    def _update_flow_metrics(self):
        """Update the flow metrics display in the status bar."""
        if not self._show_flow:
            self._metrics_label.setText("")
            return

        metrics = self._canvas.get_flow_metrics()
        text = f"Path: {metrics['path_length']} | "
        text += f"R:H {metrics['room_count']}:{metrics['hall_count']} | "
        text += f"Dead Ends: {metrics['dead_ends']}"
        self._metrics_label.setText(text)

    def _on_new(self):
        """Create a new layout."""
        if self._layout.primitives:
            reply = QMessageBox.question(
                self, "New Layout",
                "Clear current layout? Unsaved changes will be lost.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self._canvas.clear_layout()
        self._layout = self._canvas.get_layout()
        self._command_manager.clear()  # Clear undo/redo history
        self._property_editor.set_primitive(None)
        self._update_undo_redo_state()
        self._status_label.setText("New layout created")
        self.layout_changed.emit()

    def _on_save(self):
        """Save layout to JSON file."""
        if not self._layout.primitives:
            QMessageBox.warning(self, "Empty Layout",
                               "Nothing to save. Add some primitives first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", "layout.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            data = self._layout.to_dict()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            self._status_label.setText(f"Saved: {Path(file_path).name}")
            self.file_saved.emit(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_load(self):
        """Load layout from JSON file."""
        if self._layout.primitives:
            reply = QMessageBox.question(
                self, "Load Layout",
                "Replace current layout? Unsaved changes will be lost.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Layout", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            layout = DungeonLayout.from_dict(data)

            # Restore footprints for each primitive
            from .palette_widget import get_footprint
            for prim in layout.primitives.values():
                footprint = get_footprint(prim.primitive_type)
                if footprint:
                    prim.set_footprint(footprint)

            self.set_layout(layout)
            self._command_manager.clear()  # Clear undo/redo history on load
            self._update_undo_redo_state()
            self._status_label.setText(f"Loaded: {Path(file_path).name}")
            self.file_loaded.emit(file_path)

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _update_preview(self):
        """Request 3D preview update."""
        # Refresh 3D preview if in 3D mode
        if self._is_3d_view:
            self._refresh_3d_preview()
        # Emit signal for external listeners
        self.preview_requested.emit([])

    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """
<h3>Keyboard Shortcuts</h3>

<table style="border-collapse: collapse; width: 100%;">
<tr><th align="left" style="padding: 4px; border-bottom: 1px solid #555;">Shortcut</th>
    <th align="left" style="padding: 4px; border-bottom: 1px solid #555;">Action</th></tr>

<tr><td style="padding: 4px;"><b>Ctrl+N</b></td><td style="padding: 4px;">New layout</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+S</b></td><td style="padding: 4px;">Save layout</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+O</b></td><td style="padding: 4px;">Open/load layout</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+Z</b></td><td style="padding: 4px;">Undo</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+Shift+Z</b></td><td style="padding: 4px;">Redo</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+D</b></td><td style="padding: 4px;">Duplicate selected</td></tr>
<tr><td style="padding: 4px;"><b>Delete</b></td><td style="padding: 4px;">Delete selected</td></tr>

<tr><td colspan="2" style="padding: 8px 4px 4px 4px; border-top: 1px solid #555;"><b>Placement Mode</b></td></tr>
<tr><td style="padding: 4px;"><b>R</b></td><td style="padding: 4px;">Rotate placement 90°</td></tr>
<tr><td style="padding: 4px;"><b>Click</b></td><td style="padding: 4px;">Place primitive</td></tr>
<tr><td style="padding: 4px;"><b>Shift+Click</b></td><td style="padding: 4px;">Place and continue placing</td></tr>
<tr><td style="padding: 4px;"><b>Escape</b></td><td style="padding: 4px;">Cancel placement</td></tr>

<tr><td colspan="2" style="padding: 8px 4px 4px 4px; border-top: 1px solid #555;"><b>Selection Mode</b></td></tr>
<tr><td style="padding: 4px;"><b>Click</b></td><td style="padding: 4px;">Select primitive</td></tr>
<tr><td style="padding: 4px;"><b>Drag</b></td><td style="padding: 4px;">Move primitive</td></tr>
<tr><td style="padding: 4px;"><b>Escape</b></td><td style="padding: 4px;">Clear selection</td></tr>

<tr><td colspan="2" style="padding: 8px 4px 4px 4px; border-top: 1px solid #555;"><b>Navigation (TrenchBroom-style)</b></td></tr>
<tr><td style="padding: 4px;"><b>Right-drag</b></td><td style="padding: 4px;">Pan view (trackpad: two-finger drag)</td></tr>
<tr><td style="padding: 4px;"><b>Middle-drag</b></td><td style="padding: 4px;">Pan view (trackpad: Option+drag)</td></tr>
<tr><td style="padding: 4px;"><b>Scroll</b></td><td style="padding: 4px;">Zoom (trackpad: two-finger scroll)</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl++ / Ctrl+-</b></td><td style="padding: 4px;">Zoom in/out</td></tr>
<tr><td style="padding: 4px;"><b>F</b></td><td style="padding: 4px;">Fit view to content</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+F</b></td><td style="padding: 4px;">Toggle flow visualization</td></tr>
<tr><td style="padding: 4px;"><b>Ctrl+3</b></td><td style="padding: 4px;">Toggle 3D preview</td></tr>
<tr><td style="padding: 4px;"><b>F1</b></td><td style="padding: 4px;">Show this help</td></tr>
</table>
"""
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def get_layout(self) -> DungeonLayout:
        """Get the current layout."""
        return self._canvas.get_layout()

    def select_primitive(self, primitive_id: str):
        """Select a primitive by its ID.

        This is used by external components (like validation panel)
        to navigate to a specific primitive.
        """
        if primitive_id and primitive_id in self._layout.primitives:
            self._canvas.select_primitive(primitive_id)

    def set_layout(self, layout: DungeonLayout):
        """Set the layout to edit."""
        self._layout = layout
        self._canvas.set_layout(layout)
        self._property_editor.set_primitive(None)
        self.layout_changed.emit()  # Notify main window to update validation

    def clear(self):
        """Clear the layout."""
        self._canvas.clear_layout()
        self._layout = self._canvas.get_layout()
        self._property_editor.set_primitive(None)
        self.layout_changed.emit()  # Notify main window to update validation

    def load_file(self, file_path: str) -> bool:
        """Load a layout from a specific file path.

        Args:
            file_path: Path to the JSON layout file.

        Returns:
            True if load succeeded, False otherwise.
        """
        # Confirm if unsaved changes
        if self._layout.primitives:
            reply = QMessageBox.question(
                self, "Load Layout",
                "Replace current layout? Unsaved changes will be lost.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return False

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            layout = DungeonLayout.from_dict(data)

            # Restore footprints for each primitive
            from .palette_widget import get_footprint
            for prim in layout.primitives.values():
                footprint = get_footprint(prim.primitive_type)
                if footprint:
                    prim.set_footprint(footprint)

            self.set_layout(layout)
            self._command_manager.clear()  # Clear undo/redo history on load
            self._update_undo_redo_state()
            self._status_label.setText(f"Loaded: {Path(file_path).name}")
            self.file_loaded.emit(file_path)
            return True

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return False

    def new_layout(self):
        """Create a new empty layout (public API for menu)."""
        self._on_new()

    def save_layout(self):
        """Save the current layout (public API for menu)."""
        self._on_save()

    def open_layout(self):
        """Open a layout file via dialog (public API for menu)."""
        self._on_load()

    def undo(self):
        """Undo the last action (public API for menu)."""
        self._on_undo()

    def redo(self):
        """Redo the last undone action (public API for menu)."""
        self._on_redo()

    def duplicate_selected(self):
        """Duplicate the selected primitive (public API for menu)."""
        self._on_duplicate()

    def delete_selected(self):
        """Delete the selected primitive (public API for menu)."""
        self._canvas.delete_selected()

    def set_show_flow(self, checked: bool):
        """Set flow visualization state (public API for menu)."""
        self._on_flow_toggled(checked)

    def set_3d_preview(self, checked: bool):
        """Set 3D preview state (public API for menu)."""
        self._on_3d_preview_toggled(checked)

    def zoom_in(self):
        """Zoom in on the canvas (public API for menu)."""
        self._canvas.zoom_in()

    def zoom_out(self):
        """Zoom out on the canvas (public API for menu)."""
        self._canvas.zoom_out()

    def fit_to_content(self):
        """Fit view to content (public API for menu)."""
        self._canvas.fit_to_content()

    def reset_view(self):
        """Reset view to origin (public API for menu)."""
        self._canvas.reset_view()
