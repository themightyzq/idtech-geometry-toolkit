"""
Validation panel widget for real-time layout feedback.

Displays validation issues with severity indicators and clickable
references to problematic primitives.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QFrame, QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap

from .data_model import DungeonLayout
from .validation import (
    ValidationResult, ValidationIssue, ValidationSeverity,
    validate_layout,
)
from quake_levelgenerator.src.ui import style_constants as sc


def _severity_icon(severity: ValidationSeverity) -> QIcon:
    """Create an icon for a severity level."""
    size = 16
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    if severity == ValidationSeverity.ERROR:
        color = QColor(244, 67, 54)  # Red
    elif severity == ValidationSeverity.WARNING:
        color = QColor(255, 152, 0)  # Orange
    else:
        color = QColor(33, 150, 243)  # Blue

    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, size - 4, size - 4)
    painter.end()

    return QIcon(pixmap)


class ValidationPanel(QWidget):
    """Panel displaying layout validation results."""

    # Signal emitted when user clicks an issue (with primitive ID)
    issue_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._result: ValidationResult = None
        self._layout: DungeonLayout = None

        self._setup_ui()

    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with summary
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header.setStyleSheet("background: #2d2d2d; border-radius: 4px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        self._status_icon = QLabel()
        self._status_icon.setFixedSize(20, 20)
        header_layout.addWidget(self._status_icon)

        self._status_label = QLabel("No layout")
        self._status_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._status_label)

        header_layout.addStretch()

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #a0a0a0; font-size: 11pt;")
        header_layout.addWidget(self._stats_label)

        layout.addWidget(header)

        # Issues list
        self._issues_list = QListWidget()
        self._issues_list.setStyleSheet(f"""
            QListWidget {{
                background: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 11pt;
            }}
            QListWidget::item {{
                padding: 6px 4px;
                border-bottom: 1px solid #333;
                color: #e0e0e0;
            }}
            QListWidget::item:hover {{
                background: #2d2d2d;
            }}
            QListWidget::item:selected {{
                background: #3d3d3d;
                outline: 2px solid {sc.SELECTED_STATE};
            }}
            QListWidget:focus {{
                border-color: {sc.FOCUS_COLOR};
            }}
        """)
        self._issues_list.itemClicked.connect(self._on_issue_clicked)
        layout.addWidget(self._issues_list)

        # Refresh button
        refresh_btn = QPushButton("Refresh Validation")
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 6px 12px;
                background: #444;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background: #555;
            }}
            QPushButton:focus {{
                outline: 2px solid {sc.FOCUS_COLOR};
                outline-offset: 2px;
            }}
        """)
        refresh_btn.clicked.connect(self._refresh)
        layout.addWidget(refresh_btn)

    def set_layout(self, layout: DungeonLayout):
        """Set the layout to validate."""
        self._layout = layout
        self._refresh()

    def _refresh(self):
        """Re-run validation and update display."""
        self._issues_list.clear()

        if self._layout is None:
            self._status_label.setText("No layout")
            self._status_icon.clear()
            self._stats_label.setText("")
            return

        # Run validation
        self._result = validate_layout(self._layout)

        # Update header
        prim_count = len(self._layout.primitives)
        conn_count = len(self._layout.connections)

        if self._result.error_count > 0:
            self._status_label.setText("Errors Found")
            self._status_label.setStyleSheet("font-weight: bold; color: #f44336;")
            self._status_icon.setPixmap(_severity_icon(ValidationSeverity.ERROR).pixmap(16, 16))
        elif self._result.warning_count > 0:
            self._status_label.setText("Warnings")
            self._status_label.setStyleSheet("font-weight: bold; color: #ff9800;")
            self._status_icon.setPixmap(_severity_icon(ValidationSeverity.WARNING).pixmap(16, 16))
        elif prim_count > 0:
            self._status_label.setText("Valid")
            self._status_label.setStyleSheet("font-weight: bold; color: #4caf50;")
            self._status_icon.setPixmap(_severity_icon(ValidationSeverity.INFO).pixmap(16, 16))
        else:
            self._status_label.setText("Empty Layout")
            self._status_label.setStyleSheet("font-weight: bold; color: #888;")
            self._status_icon.clear()

        self._stats_label.setText(f"{prim_count} pieces, {conn_count} connections")

        # Add issues to list with text prefixes for accessibility
        for issue in self._result.issues:
            item = QListWidgetItem()
            item.setIcon(_severity_icon(issue.severity))

            # Add text prefix for accessibility (not relying on color alone)
            if issue.severity == ValidationSeverity.ERROR:
                prefix = "ERROR: "
                item.setForeground(QColor(244, 67, 54))
            elif issue.severity == ValidationSeverity.WARNING:
                prefix = "WARN: "
                item.setForeground(QColor(255, 152, 0))
            else:
                prefix = "INFO: "
                item.setForeground(QColor(158, 158, 158))

            item.setText(f"{prefix}{issue.message}")
            item.setData(Qt.UserRole, issue.primitive_id)
            self._issues_list.addItem(item)

        # Show success message if no issues
        if not self._result.issues and prim_count > 0:
            item = QListWidgetItem()
            item.setIcon(_severity_icon(ValidationSeverity.INFO))
            item.setText(f"OK: All {prim_count} primitives connected")
            item.setForeground(QColor(76, 175, 80))
            self._issues_list.addItem(item)

    def _on_issue_clicked(self, item: QListWidgetItem):
        """Handle issue item click."""
        prim_id = item.data(Qt.UserRole)
        if prim_id:
            self.issue_clicked.emit(prim_id)

    @property
    def result(self) -> ValidationResult:
        """Get the last validation result."""
        return self._result
