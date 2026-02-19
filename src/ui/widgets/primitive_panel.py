"""
Module selection and parameter panel.

Dynamically generates controls from a module's ``get_parameter_schema()``.
"""

import math
import random
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QScrollArea, QFrame, QPushButton,
)
from quake_levelgenerator.src.ui.safe_checkbox import SafeCheckBox
from quake_levelgenerator.src.ui.safe_spinbox import SafeSpinBox, SafeDoubleSpinBox
from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox

# Alias for compatibility - use safe versions to avoid macOS crashes
QSpinBox = SafeSpinBox
QDoubleSpinBox = SafeDoubleSpinBox
QComboBox = SafeComboBox  # Click to cycle, no dropdown
from PyQt5.QtCore import pyqtSignal, Qt

from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.ui.style_constants import set_accessible


class PrimitivePanel(QWidget):
    """Panel for selecting a module and editing its parameters."""

    parameters_changed = pyqtSignal(dict)
    validation_changed = pyqtSignal(str, str)  # (message, severity: "ok"|"warning"|"error")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._param_widgets: Dict[str, QWidget] = {}
        self._current_prim_name: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # Common focus style for combo boxes
        combo_focus_style = f"QComboBox:focus {{ border: 2px solid {sc.FOCUS_COLOR}; }}"

        # Compact module selector - Category and Module on single row each
        selector_group = QGroupBox("Module Selection")
        selector_layout = QVBoxLayout(selector_group)
        selector_layout.setContentsMargins(6, 8, 6, 6)
        selector_layout.setSpacing(4)

        # Category row
        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        cat_label = QLabel("Category:")
        cat_label.setFixedWidth(60)
        cat_row.addWidget(cat_label)
        self._category_combo = QComboBox()
        categories = PRIMITIVE_CATALOG.list_categories()
        self._category_combo.addItems(categories)
        self._category_combo.setToolTip("Select a category of geometric modules")
        self._category_combo.setStyleSheet(combo_focus_style)
        set_accessible(self._category_combo, "Category",
                      "Select a category of geometric modules")
        cat_row.addWidget(self._category_combo)
        selector_layout.addLayout(cat_row)

        # Module row
        mod_row = QHBoxLayout()
        mod_row.setSpacing(4)
        mod_label = QLabel("Module:")
        mod_label.setFixedWidth(60)
        mod_row.addWidget(mod_label)
        self._primitive_combo = QComboBox()
        self._primitive_combo.setToolTip("Select a specific module to generate")
        self._primitive_combo.setStyleSheet(combo_focus_style)
        set_accessible(self._primitive_combo, "Module",
                      "Select a specific module to generate")
        mod_row.addWidget(self._primitive_combo)
        selector_layout.addLayout(mod_row)

        # Randomize button
        randomize_btn = QPushButton("Randomize")
        randomize_btn.setToolTip("Randomize all parameters (snap sizes to 8-unit grid)")
        randomize_btn.clicked.connect(self._on_randomize)
        selector_layout.addWidget(randomize_btn)

        main.addWidget(selector_group)

        # Dynamic parameter area - takes all remaining space
        self._param_group = QGroupBox("Parameters")
        param_group_layout = QVBoxLayout(self._param_group)
        param_group_layout.setContentsMargins(6, 8, 6, 6)

        # Inner widget to hold parameters
        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        self._param_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._param_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        param_group_layout.addWidget(scroll)

        # Give parameters all the stretch
        main.addWidget(self._param_group, stretch=1)

        # Signals
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        self._primitive_combo.currentTextChanged.connect(self._on_primitive_changed)

        # Initial population
        if categories:
            self._on_category_changed(categories[0])

    def _on_category_changed(self, category: str):
        self._primitive_combo.blockSignals(True)
        self._primitive_combo.clear()
        names = PRIMITIVE_CATALOG.list_primitives(category)
        self._primitive_combo.addItems(names)
        self._primitive_combo.blockSignals(False)
        if names:
            self._on_primitive_changed(names[0])

    def _on_primitive_changed(self, name: str):
        self._current_prim_name = name
        cls = PRIMITIVE_CATALOG.get_primitive(name)
        if cls is None:
            return
        schema = cls.get_parameter_schema()
        self._rebuild_param_ui(schema)
        self._emit_params()

    # Portal parameter names to keep enabled (not randomized to False)
    _PORTAL_PARAMS = {
        'has_entrance', 'has_exit', 'has_side',
        'portal_front', 'portal_back', 'portal_north', 'portal_south',
        'portal_east', 'portal_west'
    }

    def _on_randomize(self):
        """Randomize all parameters, snapping numeric values to 8-unit intervals.

        Portal boolean params are kept True to ensure usable geometry.
        """
        if not self._current_prim_name:
            return

        cls = PRIMITIVE_CATALOG.get_primitive(self._current_prim_name)
        if cls is None:
            return

        schema = cls.get_parameter_schema()

        for key, spec in schema.items():
            widget = self._param_widgets.get(key)
            if widget is None:
                continue

            ptype = spec.get("type", "float")

            if ptype == "float":
                min_val = spec.get("min", 0)
                max_val = spec.get("max", 9999)
                # Snap to 8-unit intervals
                min_snapped = math.ceil(min_val / 8) * 8
                max_snapped = math.floor(max_val / 8) * 8
                steps = (max_snapped - min_snapped) // 8
                if steps > 0:
                    value = min_snapped + random.randint(0, int(steps)) * 8
                else:
                    value = min_snapped
                widget.setValue(float(value))

            elif ptype == "int":
                min_val = int(spec.get("min", 0))
                max_val = int(spec.get("max", 999))
                # For counts (segments, sides), no snapping needed
                widget.setValue(random.randint(min_val, max_val))

            elif ptype == "bool":
                # Keep portal params enabled for usable geometry
                if key in self._PORTAL_PARAMS:
                    widget.setChecked(True)
                else:
                    widget.setChecked(random.choice([True, False]))

            elif ptype == "choice":
                choices = spec.get("choices", [])
                if choices:
                    idx = random.randint(0, len(choices) - 1)
                    widget.setCurrentIndex(idx)

        # Emit params changed to update preview
        self._emit_params()

    def _rebuild_param_ui(self, schema: Dict[str, Dict[str, Any]]):
        # Clear existing
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._param_widgets.clear()

        # Common focus styles
        spin_focus = f"QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {sc.FOCUS_COLOR}; }}"
        combo_focus = f"QComboBox:focus {{ border: 2px solid {sc.FOCUS_COLOR}; }}"

        for key, spec in schema.items():
            row = QHBoxLayout()
            label = QLabel(spec.get("label", key))
            label.setMinimumWidth(70)
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

            if ptype == "float":
                w = QDoubleSpinBox()
                w.setRange(spec.get("min", 0), spec.get("max", 9999))
                w.setValue(spec.get("default", 0))
                w.setSingleStep(0.1)
                w.setStyleSheet(spin_focus)
                w.valueChanged.connect(self._emit_params)
            elif ptype == "int":
                w = QSpinBox()
                w.setRange(int(spec.get("min", 0)), int(spec.get("max", 999)))
                w.setValue(int(spec.get("default", 0)))
                w.setStyleSheet(spin_focus)
                w.valueChanged.connect(self._emit_params)
            elif ptype == "bool":
                w = SafeCheckBox()
                w.setChecked(bool(spec.get("default", False)))
                w.toggled.connect(self._emit_params)
            elif ptype == "choice":
                w = QComboBox()
                w.addItems(spec.get("choices", []))
                default = spec.get("default", "")
                idx = w.findText(default)
                if idx >= 0:
                    w.setCurrentIndex(idx)
                w.setStyleSheet(combo_focus)
                w.currentTextChanged.connect(lambda _: self._emit_params())
            else:
                w = QLabel(f"[{ptype}]")

            # Apply tooltip if we have one
            if tooltip and hasattr(w, 'setToolTip'):
                w.setToolTip(tooltip)

            # Apply accessibility label using the parameter label
            param_label = spec.get("label", key)
            param_desc = spec.get("description", "")
            set_accessible(w, param_label, param_desc)

            row.addWidget(w)
            container = QWidget()
            container.setLayout(row)
            self._param_layout.addWidget(container)
            self._param_widgets[key] = w

    def _emit_params(self, *_args):
        params = self.get_parameters()
        # Validate and emit status
        self._validate_params(params)
        self.parameters_changed.emit(params)

    def _validate_params(self, params: Dict[str, Any]):
        """Validate current parameters and emit validation status signal.

        Args:
            params: Current parameter values to validate.
        """
        if not self._current_prim_name:
            return

        cls = PRIMITIVE_CATALOG.get_primitive(self._current_prim_name)
        if cls is None:
            return

        # Create instance and apply parameters
        instance = cls()
        instance.apply_params(params)

        # Check if primitive has validate() method
        if not hasattr(instance, 'validate'):
            return

        try:
            result = instance.validate()
        except Exception:
            self.validation_changed.emit("Validation error", "error")
            return

        if not result.has_warnings():
            self.validation_changed.emit("Configuration valid", "ok")
            return

        # Format warnings
        has_error = False
        messages = []
        for w in result.warnings:
            if w.severity == "error":
                has_error = True
                messages.append(f"{w.arm_name}: {w.message}")
            else:
                messages.append(f"{w.arm_name}: {w.message}")

        severity = "error" if has_error else "warning"
        self.validation_changed.emit("; ".join(messages), severity)

    def get_parameters(self) -> Dict[str, Any]:
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

    def get_primitive_name(self) -> Optional[str]:
        return self._current_prim_name
