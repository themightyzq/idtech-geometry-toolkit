"""
Safe menu action implementation to avoid macOS Qt accessibility crash.

On macOS with PyQt5, checkable actions (QAction with setCheckable(True))
crash in QAccessible::queryAccessibleInterface when the action state changes.

This module provides a SafeMenuAction that manually manages checked state WITHOUT
using Qt's setCheckable(True), which avoids the accessibility code path entirely.

The action uses a colored circle indicator instead of bracket prefix for professional look.

Usage:
    from quake_levelgenerator.src.ui.safe_menu_action import SafeMenuAction
    action = SafeMenuAction("Enable feature", parent)
    action.setChecked(True)
    action.toggled.connect(on_toggled)
"""

from PyQt5.QtWidgets import QAction, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen


def _create_indicator_icon(checked: bool, size: int = 16) -> QIcon:
    """Create a colored circle indicator icon.

    Args:
        checked: Whether to show the checked (green) or unchecked (gray) state.
        size: Icon size in pixels.

    Returns:
        QIcon with the appropriate indicator.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Draw the circle
    margin = 2
    circle_size = size - (margin * 2)

    if checked:
        # Green filled circle with checkmark
        painter.setBrush(QBrush(QColor("#4CAF50")))
        painter.setPen(QPen(QColor("#69c46d"), 1))
        painter.drawEllipse(margin, margin, circle_size, circle_size)

        # Draw checkmark
        painter.setPen(QPen(QColor("white"), 2))
        # Checkmark coordinates (scaled to icon size)
        x1, y1 = size * 0.25, size * 0.5
        x2, y2 = size * 0.45, size * 0.7
        x3, y3 = size * 0.75, size * 0.3
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.drawLine(int(x2), int(y2), int(x3), int(y3))
    else:
        # Gray outlined circle
        painter.setBrush(QBrush(QColor("#404040")))
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawEllipse(margin, margin, circle_size, circle_size)

    painter.end()
    return QIcon(pixmap)


class SafeMenuAction(QAction):
    """
    A menu action replacement that manually manages state to avoid macOS accessibility crash.

    IMPORTANT: This does NOT use setCheckable(True) because that triggers
    Qt's accessibility system which crashes on macOS. Instead, we manually
    track checked state and handle triggers ourselves.

    Provides a similar interface to checkable QAction:
    - setChecked(bool)
    - isChecked() -> bool
    - toggled signal (emits bool)

    The visual indicator is a colored circle icon instead of the bracket prefix.
    """

    # Emits bool when state changes
    toggled = pyqtSignal(bool)

    def __init__(self, text: str, parent: QWidget = None):
        super().__init__(text, parent)
        self._checked = False
        self._update_icon()

        # Connect our trigger to toggle behavior
        self.triggered.connect(self._on_triggered)

    def _update_icon(self):
        """Update the icon based on checked state."""
        icon = _create_indicator_icon(self._checked)
        self.setIcon(icon)

    def _on_triggered(self):
        """Handle action trigger by toggling state."""
        self._checked = not self._checked
        self._update_icon()
        try:
            self.toggled.emit(self._checked)
        except RuntimeError:
            # Widget was deleted by a connected slot - this is fine
            pass

    def isChecked(self) -> bool:
        """Return current checked state."""
        return self._checked

    def setChecked(self, checked: bool):
        """Set checked state and update display."""
        if self._checked != checked:
            self._checked = checked
            self._update_icon()
            try:
                self.toggled.emit(self._checked)
            except RuntimeError:
                # Widget was deleted by a connected slot - this is fine
                pass

    def isCheckable(self) -> bool:
        """Always returns False - we don't use Qt's checkable mechanism."""
        return False
