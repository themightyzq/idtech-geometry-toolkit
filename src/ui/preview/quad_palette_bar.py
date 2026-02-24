"""
Compact palette bar for module selection in the Quad View interactive pane.

Provides Category and Module dropdowns in a single ~30px row above the GridCanvas.
Reuses palette data from the layout editor's palette_widget.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, Qt

from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox
from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import (
    PRIMITIVE_CATEGORIES, PRIMITIVE_FOOTPRINTS, get_footprint,
)
from quake_levelgenerator.src.ui.widgets.layout_editor.data_model import PrimitiveFootprint
from quake_levelgenerator.src.ui import style_constants as sc


class QuadPaletteBar(QWidget):
    """Compact module selector with Category and Module dropdowns.

    Signals:
        primitive_selected(str, PrimitiveFootprint, str):
            Emitted when a module is selected (type_name, footprint, category).
        selection_cleared():
            Emitted when selection is cleared (e.g. category changed).
    """

    primitive_selected = pyqtSignal(str, object, str)  # type, footprint, category
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Category dropdown
        cat_label = QLabel("Cat:")
        cat_label.setStyleSheet(f"color: {sc.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(cat_label)

        self._category_combo = SafeComboBox()
        self._category_combo.setFixedHeight(22)
        self._category_combo.setMinimumWidth(90)
        for cat_name in PRIMITIVE_CATEGORIES:
            self._category_combo.addItem(cat_name)
        layout.addWidget(self._category_combo)

        # Module dropdown
        mod_label = QLabel("Mod:")
        mod_label.setStyleSheet(f"color: {sc.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(mod_label)

        self._module_combo = SafeComboBox()
        self._module_combo.setFixedHeight(22)
        self._module_combo.setMinimumWidth(120)
        layout.addWidget(self._module_combo, 1)  # stretch

        # Populate initial modules
        self._populate_modules()

        # Connections
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)

        self.setStyleSheet(f"""
            QWidget {{
                background: {sc.BG_DARK};
            }}
        """)

    def _populate_modules(self):
        """Repopulate module dropdown based on current category."""
        self._module_combo.blockSignals(True)
        self._module_combo.clear()

        cat = self._category_combo.currentText()
        primitives = PRIMITIVE_CATEGORIES.get(cat, [])
        # Add a blank/none entry first
        self._module_combo.addItem("(none)")
        for prim_name in primitives:
            self._module_combo.addItem(prim_name)

        self._module_combo.blockSignals(False)

    def _on_category_changed(self, index: int):
        """Handle category selection change."""
        self._populate_modules()
        self.selection_cleared.emit()

    def _on_module_changed(self, index: int):
        """Handle module selection change."""
        mod_name = self._module_combo.currentText()
        if not mod_name or mod_name == "(none)":
            self.selection_cleared.emit()
            return

        footprint = get_footprint(mod_name)
        if footprint is None:
            self.selection_cleared.emit()
            return

        category = self._category_combo.currentText()
        self.primitive_selected.emit(mod_name, footprint, category)

    def clear_selection(self):
        """Reset module dropdown to (none)."""
        self._module_combo.blockSignals(True)
        self._module_combo.setCurrentIndex(0)
        self._module_combo.blockSignals(False)

    def current_category(self) -> str:
        """Get the currently selected category."""
        return self._category_combo.currentText()

    def current_module(self) -> str:
        """Get the currently selected module name, or empty string."""
        text = self._module_combo.currentText()
        return "" if text == "(none)" else text
