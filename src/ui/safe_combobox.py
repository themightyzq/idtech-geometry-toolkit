"""
Safe ComboBox widget that prevents macOS PyQt5 crashes.

PyQt5's QComboBox triggers accessibility crashes on macOS when the dropdown
opens. This module provides a replacement using a QMenu popup which doesn't
trigger the same accessibility issues.

Implementation Notes:
    - Uses QMenu popup instead of native QComboBox dropdown
    - Click anywhere on widget to show all options in a menu
    - Selected option shown with checkmark in menu
    - Signal emission is deferred via QTimer.singleShot(0) to allow menu
      cleanup before handlers run (prevents crashes with complex handlers)

API Compatibility:
    - Supports all common QComboBox methods: addItem, addItems, currentIndex,
      currentText, setCurrentIndex, setCurrentText, findText, findData, etc.
    - Emits same signals: currentIndexChanged, currentTextChanged, activated
    - No-op stubs for unsupported methods (setEditable, etc.)
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QMenu, QAction
from PyQt5.QtCore import pyqtSignal, Qt, QPoint, QTimer


class SafeComboBox(QWidget):
    """ComboBox replacement using QMenu popup for option selection.

    Uses a QMenu popup instead of QComboBox dropdown to avoid macOS
    accessibility crashes. Click anywhere on the widget to show all options.

    Visual Layout:
        +----------------------------------+
        | [Current Value]           [▼]   |  <- Click to open menu
        +----------------------------------+
                      ↓
                +------------------+
                | Option 1         |
                | Option 2    ✓    |  <- Checkmark on selected
                | Option 3         |
                +------------------+

    Thread Safety:
        Signal emission is deferred to next event loop iteration using
        QTimer.singleShot(0). This allows the QMenu to fully close before
        signal handlers execute, preventing crashes when handlers do complex
        operations (e.g., validation, UI updates in Layout Mode).
    """

    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)
    activated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []  # List of (text, data) tuples
        self._current_index = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Value display (clickable)
        self._value_label = QLabel()
        self._value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._value_label.setMinimumWidth(120)
        self._value_label.setCursor(Qt.PointingHandCursor)
        self._value_label.setStyleSheet("""
            QLabel {
                background: #404040;
                border: 1px solid #555;
                border-right: none;
                color: #e0e0e0;
                padding: 4px 8px;
            }
        """)
        layout.addWidget(self._value_label, 1)

        # Dropdown arrow indicator
        self._arrow_label = QLabel("▼")
        self._arrow_label.setAlignment(Qt.AlignCenter)
        self._arrow_label.setFixedWidth(24)
        self._arrow_label.setCursor(Qt.PointingHandCursor)
        self._arrow_label.setStyleSheet("""
            QLabel {
                background: #3d3d3d;
                border: 1px solid #555;
                color: #e0e0e0;
                padding: 4px 2px;
                font-size: 10px;
            }
            QLabel:hover {
                background: #4a4a4a;
            }
        """)
        layout.addWidget(self._arrow_label)

        # Menu for options
        self._menu = QMenu(self)
        self._menu.setStyleSheet("""
            QMenu {
                background: #404040;
                border: 1px solid #555;
                padding: 4px 0;
            }
            QMenu::item {
                background: transparent;
                color: #e0e0e0;
                padding: 6px 24px 6px 12px;
            }
            QMenu::item:selected {
                background: #505050;
            }
            QMenu::indicator {
                width: 16px;
                height: 16px;
                margin-left: 4px;
            }
            QMenu::indicator:checked {
                image: none;
            }
        """)

        self._update_display()

    def mousePressEvent(self, event):
        """Show menu on click."""
        if event.button() == Qt.LeftButton:
            self._show_menu()

    def _show_menu(self):
        """Show the options menu below the widget."""
        if len(self._items) == 0:
            return

        # Clear and rebuild menu
        self._menu.clear()

        for i, (text, _) in enumerate(self._items):
            action = QAction(text, self._menu)
            action.setCheckable(True)
            action.setChecked(i == self._current_index)
            # Use lambda with default argument to capture current i
            action.triggered.connect(lambda checked, idx=i: self._on_item_selected(idx))
            self._menu.addAction(action)

        # Position menu below the widget
        global_pos = self.mapToGlobal(QPoint(0, self.height()))
        self._menu.setMinimumWidth(self.width())
        self._menu.popup(global_pos)

    def _on_item_selected(self, index: int):
        """Handle menu item selection.

        Defers the actual selection to next event loop iteration to allow
        the QMenu to fully close before signal handlers run. This prevents
        crashes on macOS where complex signal handlers conflict with menu cleanup.
        """
        # Use QTimer.singleShot to defer until after menu closes
        QTimer.singleShot(0, lambda: self._apply_selection(index))

    def _apply_selection(self, index: int):
        """Apply the deferred selection after menu has closed."""
        self.setCurrentIndex(index)

    def addItem(self, text: str, userData=None):
        """Add an item to the combobox."""
        self._items.append((text, userData))
        # Auto-select first item
        if self._current_index == -1:
            self._current_index = 0
            self._update_display()

    def addItems(self, texts):
        """Add multiple items to the combobox."""
        for text in texts:
            self.addItem(text)

    def insertItem(self, index: int, text: str, userData=None):
        """Insert an item at the specified index."""
        self._items.insert(index, (text, userData))
        if self._current_index >= index:
            self._current_index += 1
        elif self._current_index == -1:
            self._current_index = 0
        self._update_display()

    def removeItem(self, index: int):
        """Remove the item at the specified index."""
        if 0 <= index < len(self._items):
            del self._items[index]
            if self._current_index == index:
                self._current_index = min(index, len(self._items) - 1)
            elif self._current_index > index:
                self._current_index -= 1
            self._update_display()

    def clear(self):
        """Remove all items."""
        self._items.clear()
        self._current_index = -1
        self._update_display()

    def count(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def currentIndex(self) -> int:
        """Return the current index."""
        return self._current_index

    def setCurrentIndex(self, index: int):
        """Set the current index."""
        if index < -1 or index >= len(self._items):
            return
        if index != self._current_index:
            self._current_index = index
            self._update_display()
            if index >= 0:
                try:
                    self.currentIndexChanged.emit(index)
                    self.currentTextChanged.emit(self._items[index][0])
                    self.activated.emit(index)
                except RuntimeError:
                    pass  # Widget deleted

    def currentText(self) -> str:
        """Return the current item's text."""
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return ""

    def setCurrentText(self, text: str):
        """Set current item by text."""
        for i, (item_text, _) in enumerate(self._items):
            if item_text == text:
                self.setCurrentIndex(i)
                return

    def currentData(self):
        """Return the current item's user data."""
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def itemText(self, index: int) -> str:
        """Return the text of item at index."""
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return ""

    def itemData(self, index: int):
        """Return the user data of item at index."""
        if 0 <= index < len(self._items):
            return self._items[index][1]
        return None

    def setItemText(self, index: int, text: str):
        """Set the text of item at index."""
        if 0 <= index < len(self._items):
            self._items[index] = (text, self._items[index][1])
            if index == self._current_index:
                self._update_display()

    def setItemData(self, index: int, data):
        """Set the user data of item at index."""
        if 0 <= index < len(self._items):
            self._items[index] = (self._items[index][0], data)

    def findText(self, text: str) -> int:
        """Find the index of item with given text."""
        for i, (item_text, _) in enumerate(self._items):
            if item_text == text:
                return i
        return -1

    def findData(self, data) -> int:
        """Find the index of item with given data."""
        for i, (_, item_data) in enumerate(self._items):
            if item_data == data:
                return i
        return -1

    def _update_display(self):
        """Update the label text to show current selection."""
        try:
            if 0 <= self._current_index < len(self._items):
                text = self._items[self._current_index][0]
                self._value_label.setText(text)
            else:
                self._value_label.setText("(none)")
        except RuntimeError:
            pass  # Widget deleted

    # QComboBox compatibility methods (no-ops)
    def setEditable(self, editable: bool):
        pass

    def isEditable(self) -> bool:
        return False

    def setMaxVisibleItems(self, count: int):
        pass

    def setMinimumContentsLength(self, characters: int):
        self._value_label.setMinimumWidth(characters * 8 + 20)

    def setSizeAdjustPolicy(self, policy):
        pass

    def setInsertPolicy(self, policy):
        pass

    def setPlaceholderText(self, text: str):
        pass
