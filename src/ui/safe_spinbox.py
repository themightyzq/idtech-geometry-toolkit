"""
Safe SpinBox widgets that prevent macOS PyQt5 crashes.

PyQt5's QSpinBox is completely broken on macOS - crashes on wheel events,
button clicks, AND keyboard input. The sipQSpinBox wrapper has deep issues.

This module provides complete replacements using QLineEdit + QPushButtons,
which avoids all the problematic QSpinBox/QAbstractSpinBox code paths.

The widgets are styled to look like native spinboxes with +/- buttons.
"""

from PyQt5.QtWidgets import QLineEdit, QWidget, QHBoxLayout, QPushButton, QSizePolicy, QGraphicsOpacityEffect
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation


class SafeSpinBox(QWidget):
    """Integer input widget that replaces QSpinBox to avoid macOS crashes.

    Uses QLineEdit with QIntValidator and +/- buttons instead of QSpinBox,
    completely avoiding the buggy sipQSpinBox wrapper.

    Features:
    - +/- buttons for increment/decrement
    - Live value updates as you type (with debouncing)
    - Looks and behaves like a native spinbox
    """

    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 99
        self._value = 0
        self._single_step = 1
        self._suffix = ""
        self._prefix = ""
        self._special_value_text = None

        # Debounce timer for live typing feedback
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)  # 150ms debounce
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Decrement button
        self._dec_btn = QPushButton("-")
        self._dec_btn.setFixedWidth(32)
        self._dec_btn.setMinimumHeight(28)
        self._dec_btn.setFocusPolicy(Qt.NoFocus)
        self._dec_btn.clicked.connect(self._on_decrement)
        self._dec_btn.setStyleSheet("""
            QPushButton {
                background: #3d3d3d;
                border: 1px solid #555;
                border-right: none;
                border-radius: 0;
                font-weight: bold;
                font-size: 13pt;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:pressed { background: #333; }
        """)
        layout.addWidget(self._dec_btn)

        # Text input
        self._line_edit = QLineEdit()
        self._line_edit.setAlignment(Qt.AlignCenter)
        self._validator = QIntValidator(self._minimum, self._maximum)
        self._line_edit.setValidator(self._validator)
        self._line_edit.setText(str(self._value))
        self._line_edit.setMinimumWidth(70)
        self._line_edit.editingFinished.connect(self._on_editing_finished)
        self._line_edit.textChanged.connect(self._on_text_changed)
        self._line_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #555;
                border-radius: 0;
                background: #404040;
                color: #e0e0e0;
                padding: 2px 4px;
            }
            QLineEdit:focus { border-color: #64B5F6; }
        """)
        layout.addWidget(self._line_edit)

        # Increment button
        self._inc_btn = QPushButton("+")
        self._inc_btn.setFixedWidth(32)
        self._inc_btn.setMinimumHeight(28)
        self._inc_btn.setFocusPolicy(Qt.NoFocus)
        self._inc_btn.clicked.connect(self._on_increment)
        self._inc_btn.setStyleSheet("""
            QPushButton {
                background: #3d3d3d;
                border: 1px solid #555;
                border-left: none;
                border-radius: 0;
                font-weight: bold;
                font-size: 13pt;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:pressed { background: #333; }
        """)
        layout.addWidget(self._inc_btn)

    def _on_increment(self):
        """Handle increment button click."""
        new_value = min(self._value + self._single_step, self._maximum)
        if new_value != self._value:
            self._value = new_value
            self._update_display()
            try:
                self.valueChanged.emit(self._value)
            except RuntimeError:
                pass

    def _on_decrement(self):
        """Handle decrement button click."""
        new_value = max(self._value - self._single_step, self._minimum)
        if new_value != self._value:
            self._value = new_value
            self._update_display()
            try:
                self.valueChanged.emit(self._value)
            except RuntimeError:
                pass

    def _on_text_changed(self, text: str):
        """Handle text changes for live feedback (debounced)."""
        self._debounce_timer.start()

    def _on_debounce_timeout(self):
        """Process debounced text change."""
        try:
            text = self._line_edit.text()
            # Remove prefix/suffix for parsing
            if self._prefix and text.startswith(self._prefix):
                text = text[len(self._prefix):]
            if self._suffix and text.endswith(self._suffix):
                text = text[:-len(self._suffix)]

            try:
                new_value = int(text)
                new_value = max(self._minimum, min(self._maximum, new_value))
                if new_value != self._value:
                    self._value = new_value
                    self.valueChanged.emit(self._value)
            except ValueError:
                pass
        except RuntimeError:
            pass

    def _on_editing_finished(self):
        """Handle when user finishes editing."""
        try:
            text = self._line_edit.text()
            # Remove prefix/suffix for parsing
            if self._prefix and text.startswith(self._prefix):
                text = text[len(self._prefix):]
            if self._suffix and text.endswith(self._suffix):
                text = text[:-len(self._suffix)]

            try:
                new_value = int(text)
                new_value = max(self._minimum, min(self._maximum, new_value))
                if new_value != self._value:
                    self._value = new_value
                    self.valueChanged.emit(self._value)
            except ValueError:
                pass
            self._update_display()
        except RuntimeError:
            # Widget was deleted
            pass

    def value(self) -> int:
        return self._value

    def setValue(self, value: int):
        value = max(self._minimum, min(self._maximum, value))
        if value != self._value:
            self._value = value
            self._update_display()
            self.valueChanged.emit(self._value)

    def minimum(self) -> int:
        return self._minimum

    def setMinimum(self, minimum: int):
        self._minimum = minimum
        self._validator.setBottom(minimum)
        if self._value < minimum:
            self.setValue(minimum)

    def maximum(self) -> int:
        return self._maximum

    def setMaximum(self, maximum: int):
        self._maximum = maximum
        self._validator.setTop(maximum)
        if self._value > maximum:
            self.setValue(maximum)

    def setRange(self, minimum: int, maximum: int):
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def singleStep(self) -> int:
        return self._single_step

    def setSingleStep(self, step: int):
        self._single_step = step

    def setSuffix(self, suffix: str):
        self._suffix = suffix
        self._update_display()

    def setPrefix(self, prefix: str):
        self._prefix = prefix
        self._update_display()

    def setToolTip(self, tip: str):
        self._line_edit.setToolTip(tip)

    def setStyleSheet(self, style: str):
        # Apply style to line edit, converting QSpinBox selectors
        style = style.replace("QSpinBox", "QLineEdit")
        self._line_edit.setStyleSheet(style)

    def setSpecialValueText(self, text: str):
        """Set text to display when value equals minimum."""
        self._special_value_text = text
        self._update_display()

    def _update_display(self):
        """Update the displayed text."""
        try:
            if self._special_value_text is not None and self._value == self._minimum:
                self._line_edit.setText(self._special_value_text)
            else:
                self._line_edit.setText(f"{self._prefix}{self._value}{self._suffix}")
        except RuntimeError:
            # Widget was deleted
            pass

    def setReadOnly(self, readonly: bool):
        """Set read-only state."""
        self._line_edit.setReadOnly(readonly)

    def setEnabled(self, enabled: bool):
        """Set enabled state."""
        super().setEnabled(enabled)
        self._line_edit.setEnabled(enabled)

    def wheelEvent(self, event):
        """Ignore wheel events."""
        event.ignore()


class SafeDoubleSpinBox(QWidget):
    """Double input widget that replaces QDoubleSpinBox to avoid crashes.

    Features:
    - +/- buttons for increment/decrement
    - Live value updates as you type (with debouncing)
    - Looks and behaves like a native spinbox
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 99.99
        self._value = 0.0
        self._single_step = 1.0
        self._decimals = 2
        self._suffix = ""
        self._prefix = ""

        # Debounce timer for live typing feedback
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Decrement button
        self._dec_btn = QPushButton("-")
        self._dec_btn.setFixedWidth(32)
        self._dec_btn.setMinimumHeight(28)
        self._dec_btn.setFocusPolicy(Qt.NoFocus)
        self._dec_btn.clicked.connect(self._on_decrement)
        self._dec_btn.setStyleSheet("""
            QPushButton {
                background: #3d3d3d;
                border: 1px solid #555;
                border-right: none;
                border-radius: 0;
                font-weight: bold;
                font-size: 13pt;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:pressed { background: #333; }
        """)
        layout.addWidget(self._dec_btn)

        # Text input
        self._line_edit = QLineEdit()
        self._line_edit.setAlignment(Qt.AlignCenter)
        self._validator = QDoubleValidator(self._minimum, self._maximum, self._decimals)
        self._line_edit.setValidator(self._validator)
        self._line_edit.setText(f"{self._value:.{self._decimals}f}")
        self._line_edit.setMinimumWidth(80)
        self._line_edit.editingFinished.connect(self._on_editing_finished)
        self._line_edit.textChanged.connect(self._on_text_changed)
        self._line_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #555;
                border-radius: 0;
                background: #404040;
                color: #e0e0e0;
                padding: 2px 4px;
            }
            QLineEdit:focus { border-color: #64B5F6; }
        """)
        layout.addWidget(self._line_edit)

        # Increment button
        self._inc_btn = QPushButton("+")
        self._inc_btn.setFixedWidth(32)
        self._inc_btn.setMinimumHeight(28)
        self._inc_btn.setFocusPolicy(Qt.NoFocus)
        self._inc_btn.clicked.connect(self._on_increment)
        self._inc_btn.setStyleSheet("""
            QPushButton {
                background: #3d3d3d;
                border: 1px solid #555;
                border-left: none;
                border-radius: 0;
                font-weight: bold;
                font-size: 13pt;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:pressed { background: #333; }
        """)
        layout.addWidget(self._inc_btn)

    def _on_increment(self):
        """Handle increment button click."""
        new_value = min(self._value + self._single_step, self._maximum)
        if new_value != self._value:
            self._value = new_value
            self._update_display()
            try:
                self.valueChanged.emit(self._value)
            except RuntimeError:
                pass

    def _on_decrement(self):
        """Handle decrement button click."""
        new_value = max(self._value - self._single_step, self._minimum)
        if new_value != self._value:
            self._value = new_value
            self._update_display()
            try:
                self.valueChanged.emit(self._value)
            except RuntimeError:
                pass

    def _on_text_changed(self, text: str):
        """Handle text changes for live feedback (debounced)."""
        self._debounce_timer.start()

    def _on_debounce_timeout(self):
        """Process debounced text change."""
        try:
            text = self._line_edit.text()
            if self._prefix and text.startswith(self._prefix):
                text = text[len(self._prefix):]
            if self._suffix and text.endswith(self._suffix):
                text = text[:-len(self._suffix)]

            try:
                new_value = float(text)
                new_value = max(self._minimum, min(self._maximum, new_value))
                if new_value != self._value:
                    self._value = new_value
                    self.valueChanged.emit(self._value)
            except ValueError:
                pass
        except RuntimeError:
            pass

    def _on_editing_finished(self):
        """Handle when user finishes editing."""
        try:
            text = self._line_edit.text()
            if self._prefix and text.startswith(self._prefix):
                text = text[len(self._prefix):]
            if self._suffix and text.endswith(self._suffix):
                text = text[:-len(self._suffix)]

            try:
                new_value = float(text)
                new_value = max(self._minimum, min(self._maximum, new_value))
                if new_value != self._value:
                    self._value = new_value
                    self.valueChanged.emit(self._value)
            except ValueError:
                pass
            self._update_display()
        except RuntimeError:
            # Widget was deleted
            pass

    def _update_display(self):
        try:
            self._line_edit.setText(f"{self._prefix}{self._value:.{self._decimals}f}{self._suffix}")
        except RuntimeError:
            # Widget was deleted
            pass

    def value(self) -> float:
        return self._value

    def setValue(self, value: float):
        value = max(self._minimum, min(self._maximum, value))
        if value != self._value:
            self._value = value
            self._update_display()
            self.valueChanged.emit(self._value)

    def minimum(self) -> float:
        return self._minimum

    def setMinimum(self, minimum: float):
        self._minimum = minimum
        self._validator.setBottom(minimum)
        if self._value < minimum:
            self.setValue(minimum)

    def maximum(self) -> float:
        return self._maximum

    def setMaximum(self, maximum: float):
        self._maximum = maximum
        self._validator.setTop(maximum)
        if self._value > maximum:
            self.setValue(maximum)

    def setRange(self, minimum: float, maximum: float):
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def singleStep(self) -> float:
        return self._single_step

    def setSingleStep(self, step: float):
        self._single_step = step

    def setDecimals(self, decimals: int):
        self._decimals = decimals
        self._validator.setDecimals(decimals)
        self._update_display()

    def setSuffix(self, suffix: str):
        self._suffix = suffix
        self._update_display()

    def setPrefix(self, prefix: str):
        self._prefix = prefix
        self._update_display()

    def setToolTip(self, tip: str):
        self._line_edit.setToolTip(tip)

    def setStyleSheet(self, style: str):
        style = style.replace("QDoubleSpinBox", "QLineEdit")
        self._line_edit.setStyleSheet(style)

    def wheelEvent(self, event):
        """Ignore wheel events."""
        event.ignore()
