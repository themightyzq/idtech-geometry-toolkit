#!/usr/bin/env python3
"""
idTech Geometry Toolkit - Main Application Entry Point

This is the main entry point for the idTech Geometry Toolkit application.
It initializes the Qt application and launches the main UI.
"""

import sys
import os

# Set Qt environment variables before importing Qt modules.
# These help with macOS rendering compatibility.
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# CRITICAL: Disable Qt accessibility to prevent macOS crashes.
# PyQt5's accessibility system has deep bugs that cause SIGSEGV when
# interacting with QCheckBox, QSpinBox, QComboBox, etc.
# This disables accessibility features but makes the app usable.
os.environ['QT_ACCESSIBILITY'] = '0'

print("[MAIN] Starting application...")
sys.stdout.flush()

from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets

# Safety net: Monkey-patch QCheckBox to use manually-managed state.
# This prevents crashes if any code accidentally imports QCheckBox directly.
# IMPORTANT: Do NOT use setCheckable(True) - that triggers the accessibility crash.
class _SafeCheckBox(QtWidgets.QPushButton):
    """A QCheckBox replacement that manually manages state to avoid macOS crashes."""
    from PyQt5.QtCore import pyqtSignal, Qt
    stateChanged = pyqtSignal(int)
    toggled = pyqtSignal(bool)

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._label_text = text
        self._checked = False  # Manual state tracking - NO setCheckable!
        self._update_display()
        self.clicked.connect(self._on_clicked)

    def _update_display(self):
        prefix = "[x] " if self._checked else "[ ] "
        super().setText(prefix + self._label_text)

    def _on_clicked(self):
        from PyQt5.QtCore import Qt
        self._checked = not self._checked
        self._update_display()
        self.toggled.emit(self._checked)
        self.stateChanged.emit(Qt.Checked if self._checked else Qt.Unchecked)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        from PyQt5.QtCore import Qt
        if self._checked != checked:
            self._checked = checked
            self._update_display()
            self.toggled.emit(self._checked)
            self.stateChanged.emit(Qt.Checked if self._checked else Qt.Unchecked)

    def checkState(self):
        from PyQt5.QtCore import Qt
        return Qt.Checked if self._checked else Qt.Unchecked

    def setCheckState(self, state):
        from PyQt5.QtCore import Qt
        self.setChecked(state == Qt.Checked)

# Replace QCheckBox globally as safety net
QtWidgets.QCheckBox = _SafeCheckBox
import PyQt5.QtWidgets
PyQt5.QtWidgets.QCheckBox = _SafeCheckBox

print("[MAIN] PyQt5 imported")
sys.stdout.flush()

"""
Note: Legacy runtime patch modules have been removed. The pipeline now
integrates conversion, persistent output, and sane defaults natively.
"""

def main():
    """Main application entry point."""
    # Ensure package imports work when executed as a script
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Initialize Qt application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("idTech Geometry Toolkit")
    app.setApplicationDisplayName("idTech Geometry Toolkit")
    app.setOrganizationName("idTechGeometry")

    # Set application style
    app.setStyle("Fusion")

    # Create dark palette
    from PyQt5.QtGui import QPalette, QColor
    from PyQt5.QtCore import Qt

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(76, 175, 80))
    dark_palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

    # Import and create main window
    print("[MAIN] Importing MainWindow...")
    sys.stdout.flush()
    from quake_levelgenerator.src.ui.main_window import MainWindow
    print("[MAIN] Creating MainWindow...")
    sys.stdout.flush()
    window = MainWindow()
    print("[MAIN] MainWindow created")
    sys.stdout.flush()

    # Show the main window
    print("[MAIN] About to show window...")
    sys.stdout.flush()
    window.show()
    print("[MAIN] Window shown")
    sys.stdout.flush()

    # Process initial events before entering main loop
    app.processEvents()
    print("[MAIN] Initial events processed")
    sys.stdout.flush()

    print("idTech Geometry Toolkit - Started")

    # Run the application
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
