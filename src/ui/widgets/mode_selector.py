"""
Mode selector widget — toggles between Layout and Module generation modes.

Uses manual state management to avoid macOS Qt accessibility crash.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal

from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.ui.style_constants import set_accessible


class ModeSelector(QWidget):
    """Two exclusive toggle buttons: Layout / Module.

    IMPORTANT: Does NOT use setCheckable(True) to avoid macOS accessibility crash.
    Instead, manages selection state manually.
    """

    mode_changed = pyqtSignal(str)  # "layout" or "primitive"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._current_mode = "layout"  # Manual state tracking

        self._layout_btn = QPushButton("Layout")
        # NO setCheckable! We manage state manually
        self._layout_btn.setToolTip("Generate BSP dungeon layouts with rooms and corridors")
        set_accessible(self._layout_btn, "Layout Mode",
                      "Switch to layout mode for designing dungeons with rooms and corridors")

        self._primitive_btn = QPushButton("Module")
        # NO setCheckable! We manage state manually
        self._primitive_btn.setToolTip("Generate individual geometric modules (arches, stairs, halls)")
        set_accessible(self._primitive_btn, "Module Mode",
                      "Switch to module mode for generating individual geometric pieces")

        layout.addWidget(self._layout_btn)
        layout.addWidget(self._primitive_btn)

        # Connect clicks
        self._layout_btn.clicked.connect(lambda: self._select_mode("layout"))
        self._primitive_btn.clicked.connect(lambda: self._select_mode("primitive"))

        # Apply initial styles
        self._update_styles()

    def _get_btn_style(self, selected: bool) -> str:
        """Get button style based on selection state."""
        if selected:
            # In high contrast mode, use black text on bright cyan for readability
            text_color = "#000000" if sc.HIGH_CONTRAST_MODE else "white"
            return f"""
                QPushButton {{
                    padding: 8px 16px;
                    font-weight: bold;
                    border: 1px solid {sc.SELECTED_STATE};
                    border-radius: 4px;
                    background-color: {sc.SELECTED_STATE};
                    color: {text_color};
                }}
                QPushButton:hover {{
                    background-color: {sc.SELECTED_STATE_HOVER};
                }}
                QPushButton:focus {{
                    outline: 2px solid {sc.FOCUS_COLOR};
                    outline-offset: 2px;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    padding: 8px 16px;
                    font-weight: bold;
                    border: 1px solid {sc.BORDER_MEDIUM};
                    border-radius: 4px;
                    background-color: {sc.BG_LIGHT};
                    color: {sc.TEXT_SECONDARY};
                }}
                QPushButton:hover {{
                    background-color: {sc.BG_HIGHLIGHT};
                }}
                QPushButton:focus {{
                    outline: 2px solid {sc.FOCUS_COLOR};
                    outline-offset: 2px;
                }}
            """

    def _update_styles(self):
        """Update button styles based on current mode."""
        self._layout_btn.setStyleSheet(self._get_btn_style(self._current_mode == "layout"))
        self._primitive_btn.setStyleSheet(self._get_btn_style(self._current_mode == "primitive"))

    def _select_mode(self, mode: str):
        """Select a mode and emit signal if changed."""
        if self._current_mode != mode:
            self._current_mode = mode
            self._update_styles()
            self.mode_changed.emit(mode)

    def current_mode(self) -> str:
        return self._current_mode

    def set_mode(self, mode: str):
        """Programmatically set the mode."""
        if mode in ("layout", "primitive"):
            self._current_mode = mode
            self._update_styles()
