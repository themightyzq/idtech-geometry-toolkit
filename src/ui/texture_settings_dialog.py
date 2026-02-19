"""
Texture Settings dialog for configuring placeholder texture names.

Provides a simple UI for users to set custom texture paths per surface type.
Textures are stored globally in QSettings.
"""

from typing import Dict

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFileDialog,
)
from PyQt5.QtCore import Qt

from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.generators.textures import (
    TEXTURE_SETTINGS, SURFACE_TYPES, DEFAULT_TEXTURES
)


class TextureSettingsDialog(QDialog):
    """Dialog for configuring texture settings."""

    def __init__(self, parent=None):
        """Initialize the texture settings dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._texture_edits: Dict[str, QLineEdit] = {}

        self._setup_ui()
        self._apply_styles()
        self._load_current_settings()

    def _setup_ui(self):
        """Create the dialog UI."""
        self.setWindowTitle("Texture Settings")
        self.setMinimumSize(500, 300)
        self.resize(550, 350)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === Info Label ===
        info_label = QLabel(
            "Set custom texture names for each surface type.\n"
            "Leave blank to use the default placeholder name.\n"
            "These are just placeholders - retexture in your level editor."
        )
        info_label.setStyleSheet(f"color: {sc.TEXT_SECONDARY}; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # === Surface Textures ===
        textures_group = QGroupBox("Surface Textures")
        textures_layout = QVBoxLayout(textures_group)

        # Create form layout for surfaces
        form_layout = QFormLayout()
        form_layout.setSpacing(8)
        form_layout.setContentsMargins(12, 12, 12, 12)

        for surface in SURFACE_TYPES:
            # Create row with text field and browse button
            row_widget = self._create_texture_row(surface)
            label = surface.replace("_", " ").title() + ":"
            form_layout.addRow(label, row_widget)

        textures_layout.addLayout(form_layout)
        layout.addWidget(textures_group)

        # === Buttons ===
        button_layout = QHBoxLayout()

        # Reset button on the left
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Clear all custom textures and use defaults")
        reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        # Cancel and OK on the right
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background: {sc.PRIMARY_ACTION};
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: {sc.BORDER_RADIUS_MD};
            }}
            QPushButton:hover {{
                background: {sc.PRIMARY_ACTION_HOVER};
            }}
        """)
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _create_texture_row(self, surface: str):
        """Create a row widget with text field and browse button.

        Args:
            surface: Surface type name

        Returns:
            QWidget containing the row
        """
        from PyQt5.QtWidgets import QWidget

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        # Text field
        edit = QLineEdit()
        edit.setPlaceholderText(f"default: {DEFAULT_TEXTURES.get(surface, surface)}")
        self._texture_edits[surface] = edit
        row_layout.addWidget(edit, stretch=1)

        # Browse button
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setToolTip(f"Browse for {surface} texture")
        browse_btn.clicked.connect(lambda checked, s=surface: self._on_browse(s))
        row_layout.addWidget(browse_btn)

        return row

    def _on_browse(self, surface: str):
        """Handle browse button click for a surface.

        Opens the system file explorer to let the user browse for a texture file.

        Args:
            surface: Surface type to browse for
        """
        # Get current value to use as starting directory
        current = self._texture_edits[surface].text()
        start_dir = ""
        if current:
            from pathlib import Path
            parent = Path(current).parent
            if parent.exists():
                start_dir = str(parent)

        # Open file dialog for texture selection
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {surface.title()} Texture",
            start_dir,
            "Image Files (*.tga *.png *.jpg *.jpeg *.bmp *.dds);;All Files (*)"
        )

        if file_path:
            self._texture_edits[surface].setText(file_path)

    def _apply_styles(self):
        """Apply consistent styling."""
        self.setStyleSheet(f"""
            QDialog {{
                background: {sc.BG_DARK};
                color: {sc.TEXT_PRIMARY};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {sc.BORDER_DARK};
                border-radius: {sc.BORDER_RADIUS_LG};
                margin-top: {sc.SPACING_SM}px;
                padding-top: {sc.SPACING_SM}px;
                background: {sc.BG_MEDIUM};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {sc.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {sc.TEXT_PRIMARY};
            }}
            QLineEdit {{
                background: {sc.BG_LIGHT};
                border: 1px solid {sc.BORDER_MEDIUM};
                border-radius: {sc.BORDER_RADIUS_MD};
                padding: {sc.SPACING_XS}px {sc.SPACING_SM}px;
                color: {sc.TEXT_PRIMARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {sc.FOCUS_COLOR};
            }}
            QPushButton {{
                background: {sc.BG_LIGHT};
                border: 1px solid {sc.BORDER_MEDIUM};
                border-radius: {sc.BORDER_RADIUS_MD};
                padding: 6px 12px;
                color: {sc.TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                background: {sc.BG_HIGHLIGHT};
            }}
        """)

    def _load_current_settings(self):
        """Load current texture settings into the form."""
        for surface in SURFACE_TYPES:
            raw_value = TEXTURE_SETTINGS.get_raw_value(surface)
            self._texture_edits[surface].setText(raw_value)

    def _on_reset(self):
        """Handle Reset to Defaults button."""
        for surface in SURFACE_TYPES:
            self._texture_edits[surface].clear()

    def _on_ok(self):
        """Handle OK button - save settings and close."""
        for surface in SURFACE_TYPES:
            value = self._texture_edits[surface].text().strip()
            TEXTURE_SETTINGS.set_texture(surface, value)
        self.accept()


def show_texture_settings(parent=None) -> bool:
    """Show the texture settings dialog.

    Args:
        parent: Parent widget

    Returns:
        True if user clicked OK, False if cancelled
    """
    dialog = TextureSettingsDialog(parent)
    return dialog.exec_() == QDialog.Accepted
