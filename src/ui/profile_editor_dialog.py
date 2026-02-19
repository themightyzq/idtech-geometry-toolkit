"""
Profile editor dialog for creating and editing custom game profiles.

Allows users to define custom entity and worldspawn settings for their own games/mods,
stored in ~/.config/quake_levelgenerator/profiles/

Note: Textures are now managed globally by TextureSettings, not per-profile.
"""

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox,
)

from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox
from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.generators.profiles import (
    GameProfile, PROFILE_CATALOG, save_profile, is_builtin_profile,
    reload_custom_profiles
)


class ProfileEditorDialog(QDialog):
    """Dialog for creating/editing custom game profiles."""

    def __init__(self, parent=None, profile: Optional[GameProfile] = None):
        """
        Initialize the profile editor.

        Args:
            parent: Parent widget
            profile: Existing profile to edit, or None for new profile
        """
        super().__init__(parent)
        self._editing_profile = profile

        self._setup_ui()
        self._apply_styles()

        if profile:
            self._load_profile(profile)
            # Disable name editing for built-in profiles
            if is_builtin_profile(profile.name):
                self._name_edit.setEnabled(False)
                self._name_edit.setToolTip("Built-in profile names cannot be changed")

    def _setup_ui(self):
        """Create the dialog UI."""
        self.setWindowTitle("Profile Editor")
        self.setMinimumSize(400, 250)
        self.resize(450, 280)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === Profile Identity ===
        identity_group = QGroupBox("Profile Settings")
        identity_layout = QFormLayout(identity_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("My Custom Game")
        identity_layout.addRow("Name:", self._name_edit)

        self._description_edit = QLineEdit()
        self._description_edit.setPlaceholderText("Custom settings for my game")
        identity_layout.addRow("Description:", self._description_edit)

        self._engine_combo = SafeComboBox()
        self._engine_combo.addItem("idTech 1 (Quake)", "idtech1")
        self._engine_combo.addItem("idTech 4 (Doom 3)", "idtech4")
        identity_layout.addRow("Engine:", self._engine_combo)

        layout.addWidget(identity_group)

        # === Info Label ===
        info_label = QLabel(
            "Profiles define entity classnames and worldspawn properties.\n"
            "Textures are configured globally via Edit > Texture Settings."
        )
        info_label.setStyleSheet(f"color: {sc.TEXT_SECONDARY}; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        # === Buttons ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Profile")
        save_btn.setStyleSheet(f"""
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
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

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
            QLineEdit:disabled {{
                background: {sc.BG_MEDIUM};
                color: {sc.TEXT_DISABLED};
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

    def _load_profile(self, profile: GameProfile):
        """Load an existing profile into the editor."""
        self._name_edit.setText(profile.name)
        self._description_edit.setText(profile.description)

        # Set engine
        engine_index = self._engine_combo.findData(profile.engine)
        if engine_index >= 0:
            self._engine_combo.setCurrentIndex(engine_index)

    def _on_save(self):
        """Handle save button click."""
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name",
                               "Please enter a profile name.")
            self._name_edit.setFocus()
            return

        # Check for name collision (unless editing same profile)
        existing = PROFILE_CATALOG.get_profile(name)
        if existing:
            if self._editing_profile is None or \
               self._editing_profile.name.lower() != name.lower():
                if is_builtin_profile(name):
                    QMessageBox.warning(
                        self, "Reserved Name",
                        f"'{name}' is a built-in profile and cannot be overwritten."
                    )
                    return
                reply = QMessageBox.question(
                    self, "Overwrite Profile",
                    f"A profile named '{name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        # Create profile (no textures - those are managed globally now)
        profile = GameProfile(
            name=name,
            engine=self._engine_combo.currentData() or "idtech4",
            description=self._description_edit.text().strip(),
            entities={"player_start": "info_player_start"},
            worldspawn={},
        )

        # Save to disk
        try:
            save_profile(profile)
            reload_custom_profiles()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error",
                                f"Failed to save profile: {e}")

    def get_profile(self) -> Optional[GameProfile]:
        """Get the created/edited profile (call after accept())."""
        name = self._name_edit.text().strip()
        if name:
            return PROFILE_CATALOG.get_profile(name)
        return None


def show_profile_editor(parent=None, profile: Optional[GameProfile] = None) -> Optional[GameProfile]:
    """
    Show the profile editor dialog.

    Args:
        parent: Parent widget
        profile: Existing profile to edit, or None for new profile

    Returns:
        The saved GameProfile if accepted, None if cancelled
    """
    dialog = ProfileEditorDialog(parent, profile)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_profile()
    return None
