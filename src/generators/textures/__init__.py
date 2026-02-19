"""
Texture settings module for placeholder texture management.

This module provides simple placeholder texture names with optional user customization.
Per CLAUDE.md's "geometry-only" philosophy:
- This tool generates brush geometry only
- Textures are just placeholder strings for preview
- Users will retexture in their level editor (TrenchBroom, Radiant)

Usage:
    from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS

    # Get texture for a surface
    wall_tex = TEXTURE_SETTINGS.get_texture("wall")

    # Set custom texture path
    TEXTURE_SETTINGS.set_texture("wall", "textures/mymod/brick")
"""

from .texture_settings import (
    TextureSettings,
    TEXTURE_SETTINGS,
    SURFACE_TYPES,
    DEFAULT_TEXTURES,
)

__all__ = [
    'TextureSettings',
    'TEXTURE_SETTINGS',
    'SURFACE_TYPES',
    'DEFAULT_TEXTURES',
]
