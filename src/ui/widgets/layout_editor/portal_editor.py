"""
Portal Editor Widget for per-instance portal customization.

Allows users to override portal settings (enabled/disabled, z_level) on a
per-instance basis without modifying the base footprint definition.
"""

from typing import Optional, Dict

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFrame, QScrollArea
)
from quake_levelgenerator.src.ui.safe_checkbox import SafeCheckBox
from quake_levelgenerator.src.ui.safe_spinbox import SafeSpinBox

# Alias for compatibility
QSpinBox = SafeSpinBox

from .data_model import PlacedPrimitive, PortalOverride, Portal, PrimitiveFootprint


class PortalEditorWidget(QWidget):
    """Widget for editing portal overrides on a placed primitive.

    Displays a list of portals for the selected primitive with controls
    to enable/disable each portal and optionally override its z_level.

    Signals:
        portal_override_changed(str, str, PortalOverride): Emitted when a portal
            override changes. Args: primitive_id, portal_id, new_override
    """

    portal_override_changed = pyqtSignal(str, str, object)  # prim_id, portal_id, override

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._primitive: Optional[PlacedPrimitive] = None
        self._portal_widgets: Dict[str, '_PortalRow'] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        self._header = QLabel("Portal Overrides")
        self._header.setStyleSheet("font-weight: bold; color: #9C27B0;")
        layout.addWidget(self._header)

        # Scroll area for portal list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(200)

        self._portal_container = QWidget()
        self._portal_layout = QVBoxLayout(self._portal_container)
        self._portal_layout.setContentsMargins(0, 0, 0, 0)
        self._portal_layout.setSpacing(2)
        self._portal_layout.addStretch()

        scroll.setWidget(self._portal_container)
        layout.addWidget(scroll)

        # Info label when no primitive selected
        self._no_selection_label = QLabel("Select a module to edit portals")
        self._no_selection_label.setStyleSheet("color: #888; font-style: italic;")
        self._no_selection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._no_selection_label)

        self._update_visibility()

    def set_primitive(self, primitive: Optional[PlacedPrimitive]):
        """Set the primitive to edit.

        Args:
            primitive: The placed primitive, or None to clear
        """
        self._primitive = primitive
        self._rebuild_portal_list()
        self._update_visibility()

    def _rebuild_portal_list(self):
        """Rebuild the portal list for the current primitive."""
        # Clear existing widgets
        for widget in self._portal_widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self._portal_widgets.clear()

        if self._primitive is None or self._primitive.footprint is None:
            return

        # Create a row for each portal
        for portal in self._primitive.footprint.portals:
            row = _PortalRow(portal, self._primitive, self)
            row.override_changed.connect(self._on_portal_changed)

            # Insert before the stretch
            self._portal_layout.insertWidget(
                self._portal_layout.count() - 1, row
            )
            self._portal_widgets[portal.id] = row

    def _update_visibility(self):
        """Update widget visibility based on selection state."""
        has_primitive = self._primitive is not None
        has_portals = bool(self._portal_widgets)

        self._no_selection_label.setVisible(not has_primitive or not has_portals)
        self._header.setVisible(has_portals)
        self._portal_container.setVisible(has_portals)

    def _on_portal_changed(self, portal_id: str, override: PortalOverride):
        """Handle portal override change from a row widget."""
        if self._primitive is None:
            return

        # Update the primitive's portal_overrides
        self._primitive.portal_overrides[portal_id] = override

        # Emit signal for external handling
        self.portal_override_changed.emit(
            self._primitive.id, portal_id, override
        )

    def refresh(self):
        """Refresh the display from the current primitive's state."""
        if self._primitive is None:
            return

        for portal_id, widget in self._portal_widgets.items():
            widget.refresh_from_primitive(self._primitive)


class _PortalRow(QFrame):
    """A single row for editing one portal's overrides."""

    override_changed = pyqtSignal(str, object)  # portal_id, PortalOverride

    def __init__(self, portal: Portal, primitive: PlacedPrimitive,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._portal = portal
        self._primitive = primitive
        self._updating = False  # Prevent signal loops
        self._setup_ui()

    def _setup_ui(self):
        """Set up the row UI."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            _PortalRow {
                background-color: #2a2a2a;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Portal name and direction
        portal_label = QLabel(f"{self._portal.id}")
        portal_label.setFixedWidth(80)
        portal_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(portal_label)

        # Direction indicator
        dir_label = QLabel(f"({self._portal.direction.value})")
        dir_label.setStyleSheet("color: #888;")
        dir_label.setFixedWidth(60)
        layout.addWidget(dir_label)

        # Enabled checkbox
        self._enabled_cb = SafeCheckBox("Enabled")
        self._enabled_cb.setChecked(self._primitive.is_portal_enabled(self._portal.id))
        self._enabled_cb.stateChanged.connect(self._on_enabled_changed)
        layout.addWidget(self._enabled_cb)

        layout.addSpacing(16)

        # Z-level override
        z_label = QLabel("Z:")
        layout.addWidget(z_label)

        self._z_spin = QSpinBox()
        self._z_spin.setRange(-512, 512)
        self._z_spin.setSingleStep(16)
        self._z_spin.setSpecialValueText("default")
        self._z_spin.setMinimum(-999)  # Use -999 as "default" indicator

        # Get current z_level (override or default)
        current_z = self._primitive.get_portal_z_level(self._portal.id)
        has_override = (
            self._portal.id in self._primitive.portal_overrides and
            self._primitive.portal_overrides[self._portal.id].z_level_override is not None
        )

        if has_override:
            self._z_spin.setValue(current_z)
        else:
            self._z_spin.setValue(-999)  # Show "default"

        self._z_spin.valueChanged.connect(self._on_z_changed)
        layout.addWidget(self._z_spin)

        # Default z info
        default_z_label = QLabel(f"(def: {self._portal.z_level})")
        default_z_label.setStyleSheet("color: #666;")
        layout.addWidget(default_z_label)

        layout.addStretch()

    def _on_enabled_changed(self, state: int):
        """Handle enabled checkbox change."""
        if self._updating:
            return

        enabled = state == Qt.Checked
        self._emit_override(enabled=enabled)

    def _on_z_changed(self, value: int):
        """Handle z-level spinbox change."""
        if self._updating:
            return

        # -999 means "use default"
        z_override = None if value == -999 else value
        self._emit_override(z_level_override=z_override)

    def _emit_override(self, enabled: Optional[bool] = None,
                       z_level_override: Optional[int] = -1):
        """Emit an override change signal.

        Args:
            enabled: New enabled state, or None to keep current
            z_level_override: New z_level, -1 to keep current, None for "use default"
        """
        # Get current override or create new one
        if self._portal.id in self._primitive.portal_overrides:
            current = self._primitive.portal_overrides[self._portal.id]
            new_enabled = enabled if enabled is not None else current.enabled
            new_z = (z_level_override if z_level_override != -1
                     else current.z_level_override)
        else:
            new_enabled = enabled if enabled is not None else True
            new_z = z_level_override if z_level_override != -1 else None

        override = PortalOverride(enabled=new_enabled, z_level_override=new_z)
        self.override_changed.emit(self._portal.id, override)

    def refresh_from_primitive(self, primitive: PlacedPrimitive):
        """Refresh display from primitive state."""
        self._primitive = primitive
        self._updating = True

        self._enabled_cb.setChecked(primitive.is_portal_enabled(self._portal.id))

        has_override = (
            self._portal.id in primitive.portal_overrides and
            primitive.portal_overrides[self._portal.id].z_level_override is not None
        )
        if has_override:
            self._z_spin.setValue(
                primitive.portal_overrides[self._portal.id].z_level_override
            )
        else:
            self._z_spin.setValue(-999)

        self._updating = False
