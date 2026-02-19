"""
Safe checkbox implementation to avoid macOS Qt accessibility crash.

On macOS with PyQt5, checkable buttons (QCheckBox, QPushButton with setCheckable)
crash in QAccessible::queryAccessibleInterface when the button state changes.

This module provides a SafeCheckBox that manually manages checked state WITHOUT
using Qt's setCheckable(True), which avoids the accessibility code path entirely.

The checkbox uses a styled QLabel indicator instead of text prefix for professional look.

Usage:
    from quake_levelgenerator.src.ui.safe_checkbox import SafeCheckBox
    checkbox = SafeCheckBox("Enable feature")
    checkbox.setChecked(True)
    checkbox.stateChanged.connect(on_state_changed)
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, Qt


class SafeCheckBox(QWidget):
    """
    A checkbox replacement that manually manages state to avoid macOS accessibility crash.

    IMPORTANT: This does NOT use setCheckable(True) because that triggers
    Qt's accessibility system which crashes on macOS. Instead, we manually
    track checked state and handle clicks ourselves.

    Provides the same interface as QCheckBox for common operations:
    - setChecked(bool)
    - isChecked() -> bool
    - stateChanged signal (emits Qt.Checked or Qt.Unchecked)
    - toggled signal (emits bool)
    """

    # Emits Qt.Checked (2) or Qt.Unchecked (0) to match QCheckBox.stateChanged
    stateChanged = pyqtSignal(int)
    # Emits bool to match QPushButton.toggled
    toggled = pyqtSignal(bool)

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._label_text = text
        self._checked = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Visual checkbox indicator using a styled QLabel (22x22 for better visibility)
        self._indicator = QLabel()
        self._indicator.setFixedSize(22, 22)
        self._indicator.setAlignment(Qt.AlignCenter)
        self._indicator.setCursor(Qt.PointingHandCursor)
        self._update_indicator()
        layout.addWidget(self._indicator)

        # Label with proper font size
        self._label = QLabel(text)
        self._label.setStyleSheet("color: #e0e0e0; font-size: 11pt;")
        self._label.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self._label)
        layout.addStretch()

        # Ensure minimum height for comfortable clicking
        self.setMinimumHeight(28)
        self.setFocusPolicy(Qt.StrongFocus)

    def _update_indicator(self):
        """Update the indicator appearance based on checked state."""
        if self._checked:
            self._indicator.setStyleSheet("""
                QLabel {
                    background: #4CAF50;
                    border: 2px solid #69c46d;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                    font-size: 11pt;
                }
            """)
            self._indicator.setText("\u2713")  # Unicode checkmark
        else:
            self._indicator.setStyleSheet("""
                QLabel {
                    background: #404040;
                    border: 2px solid #666666;
                    border-radius: 3px;
                }
            """)
            self._indicator.setText("")

    def _on_clicked(self):
        """Handle click by toggling state manually."""
        self._checked = not self._checked
        self._update_indicator()
        # Emit signals - use try/except in case widget is deleted by a slot
        state = Qt.Checked if self._checked else Qt.Unchecked
        try:
            self.toggled.emit(self._checked)
            self.stateChanged.emit(state)
        except RuntimeError:
            # Widget was deleted by a connected slot - this is fine
            pass

    def mousePressEvent(self, event):
        """Handle click on the entire widget."""
        if event.button() == Qt.LeftButton:
            self._on_clicked()

    def keyPressEvent(self, event):
        """Handle space/enter to toggle."""
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._on_clicked()
        else:
            super().keyPressEvent(event)

    def setText(self, text: str):
        """Set the label text."""
        self._label_text = text
        self._label.setText(text)

    def text(self) -> str:
        """Get the label text."""
        return self._label_text

    def isChecked(self) -> bool:
        """Return current checked state."""
        return self._checked

    def setChecked(self, checked: bool):
        """Set checked state and update display."""
        if self._checked != checked:
            self._checked = checked
            self._update_indicator()
            # Emit signals - use try/except in case widget is deleted by a slot
            state = Qt.Checked if self._checked else Qt.Unchecked
            try:
                self.toggled.emit(self._checked)
                self.stateChanged.emit(state)
            except RuntimeError:
                # Widget was deleted by a connected slot - this is fine
                pass

    def checkState(self) -> int:
        """Return Qt.Checked or Qt.Unchecked to match QCheckBox interface."""
        return Qt.Checked if self._checked else Qt.Unchecked

    def setCheckState(self, state: int):
        """Set state using Qt.Checked/Qt.Unchecked constants."""
        self.setChecked(state == Qt.Checked)

    def isCheckable(self) -> bool:
        """Always returns False - we don't use Qt's checkable mechanism."""
        return False

    def setEnabled(self, enabled: bool):
        """Set enabled state."""
        super().setEnabled(enabled)
        if enabled:
            self._label.setStyleSheet("color: #e0e0e0;")
            self._update_indicator()
        else:
            self._label.setStyleSheet("color: #666666;")
            self._indicator.setStyleSheet("""
                QLabel {
                    background: #333;
                    border: 2px solid #444;
                    border-radius: 3px;
                }
            """)
