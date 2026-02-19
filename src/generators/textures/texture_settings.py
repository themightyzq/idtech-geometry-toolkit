"""
Global texture settings stored in QSettings.

This module provides placeholder texture names for preview and export.
Per CLAUDE.md's "geometry-only" philosophy:
- This tool generates brush geometry only
- Textures are just placeholder strings for preview
- Users will retexture in their level editor (TrenchBroom, Radiant)
- No need for engine-specific texture mappings

Usage:
    from quake_levelgenerator.src.generators.textures.texture_settings import TEXTURE_SETTINGS

    # Get texture for a surface
    wall_tex = TEXTURE_SETTINGS.get_texture("wall")

    # Set custom texture path
    TEXTURE_SETTINGS.set_texture("wall", "textures/mymod/brick")

    # Reset to defaults
    TEXTURE_SETTINGS.reset_to_defaults()
"""

from PyQt5.QtCore import QSettings
from typing import Dict


# Surface types that can have custom textures
SURFACE_TYPES = ["wall", "floor", "ceiling", "trim", "structural"]

# Default placeholder texture names (simple names that work in any engine)
DEFAULT_TEXTURES = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "trim": "trim",
    "structural": "structural",
}


class TextureSettings:
    """Global texture settings stored in QSettings.

    Provides persistent storage for custom texture paths per surface type.
    Empty values fall back to default placeholder names.

    Note: QSettings is accessed lazily to avoid issues with object lifetime
    in test environments where QApplication may not persist.
    """

    def __init__(self):
        """Initialize texture settings."""
        self._settings = None

    def _get_settings(self) -> QSettings:
        """Get or create the QSettings instance.

        Creates a fresh QSettings on each call to handle cases where the
        underlying C++ object may have been deleted (e.g., in test environments).
        """
        try:
            # Test if the existing settings object is still valid
            if self._settings is not None:
                # Try to access it - will throw if deleted
                self._settings.organizationName()
                return self._settings
        except RuntimeError:
            # Object was deleted, need to recreate
            pass

        self._settings = QSettings("QuakeLevelGenerator", "Textures")
        return self._settings

    def get_texture(self, surface: str) -> str:
        """Get texture for a surface type.

        Args:
            surface: Surface type (wall, floor, ceiling, trim, structural)

        Returns:
            User's custom texture path if set, otherwise the default placeholder.
        """
        try:
            settings = self._get_settings()
            value = settings.value(f"textures/{surface}", "")
            if value:
                return value
        except RuntimeError:
            # QSettings not available (no QApplication), use defaults
            pass
        return DEFAULT_TEXTURES.get(surface, surface)

    def set_texture(self, surface: str, path: str):
        """Set custom texture path for a surface.

        Args:
            surface: Surface type (wall, floor, ceiling, trim, structural)
            path: Custom texture path, or empty string to use default
        """
        if surface in SURFACE_TYPES:
            try:
                settings = self._get_settings()
                settings.setValue(f"textures/{surface}", path)
            except RuntimeError:
                # QSettings not available (no QApplication), ignore
                pass

    def get_all_textures(self) -> Dict[str, str]:
        """Get all texture mappings.

        Returns:
            Dict mapping surface types to their texture names (custom or default)
        """
        return {s: self.get_texture(s) for s in SURFACE_TYPES}

    def get_raw_value(self, surface: str) -> str:
        """Get the raw stored value (may be empty).

        Args:
            surface: Surface type

        Returns:
            The stored value or empty string if not set
        """
        try:
            settings = self._get_settings()
            return settings.value(f"textures/{surface}", "")
        except RuntimeError:
            return ""

    def reset_to_defaults(self):
        """Clear all custom textures, reverting to defaults."""
        try:
            settings = self._get_settings()
            for surface in SURFACE_TYPES:
                settings.remove(f"textures/{surface}")
        except RuntimeError:
            # QSettings not available (no QApplication), ignore
            pass


# Global singleton instance
TEXTURE_SETTINGS = TextureSettings()


__all__ = [
    'TextureSettings',
    'TEXTURE_SETTINGS',
    'SURFACE_TYPES',
    'DEFAULT_TEXTURES',
]
