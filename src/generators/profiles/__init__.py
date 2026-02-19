"""
Game profile system for multi-engine support.

This module provides the GameProfile system that allows the toolkit to target
different idTech engines (Quake 1, Doom 3, custom mods) with appropriate
texture/material names and entity classnames.

Usage:
    from quake_levelgenerator.src.generators.profiles import PROFILE_CATALOG

    # Get a profile
    profile = PROFILE_CATALOG.get_profile("Quake 1")

    # Use profile for texture lookup
    wall_texture = profile.get_texture("Medieval", "wall")

    # Use profile for entity mapping
    classname = profile.get_entity_classname("monster_grunt")

    # Create and save a custom profile
    from quake_levelgenerator.src.generators.profiles import (
        save_profile, load_all_saved_profiles, reload_custom_profiles
    )
    custom = GameProfile(name="My Game", engine="idtech4", description="...")
    save_profile(custom)
    reload_custom_profiles()  # Refresh catalog with new profile
"""

from .game_profile import GameProfile, ProfileCatalog
from .profile_storage import (
    get_profiles_dir,
    save_profile,
    load_profile,
    load_all_saved_profiles,
    list_saved_profiles,
    delete_profile,
    profile_exists,
)

# Global catalog singleton
PROFILE_CATALOG = ProfileCatalog()

# Built-in profile names (protected from deletion)
BUILTIN_PROFILE_NAMES = {"Quake 1", "Doom 3"}

# Register built-in profiles
from .builtin import QUAKE1_PROFILE, DOOM3_PROFILE

PROFILE_CATALOG.register(QUAKE1_PROFILE)
PROFILE_CATALOG.register(DOOM3_PROFILE)


def reload_custom_profiles() -> int:
    """
    Reload custom profiles from disk into the catalog.

    This function clears all non-builtin profiles and reloads
    all saved profiles from the profiles directory.

    Returns:
        Number of custom profiles loaded
    """
    # Remove all non-builtin profiles
    for name in list(PROFILE_CATALOG.list_profiles()):
        if name not in BUILTIN_PROFILE_NAMES:
            PROFILE_CATALOG.unregister(name)

    # Load all saved profiles
    loaded = 0
    for profile in load_all_saved_profiles():
        # Skip if name conflicts with builtin
        if profile.name in BUILTIN_PROFILE_NAMES:
            continue
        PROFILE_CATALOG.register(profile)
        loaded += 1

    return loaded


def is_builtin_profile(name: str) -> bool:
    """
    Check if a profile name is a built-in profile.

    Args:
        name: Profile name to check

    Returns:
        True if this is a built-in profile that cannot be deleted
    """
    return name in BUILTIN_PROFILE_NAMES


# Auto-load saved custom profiles on module import
reload_custom_profiles()


__all__ = [
    # Core classes
    'GameProfile',
    'ProfileCatalog',
    'PROFILE_CATALOG',
    # Built-in profiles
    'QUAKE1_PROFILE',
    'DOOM3_PROFILE',
    'BUILTIN_PROFILE_NAMES',
    # Storage functions
    'get_profiles_dir',
    'save_profile',
    'load_profile',
    'load_all_saved_profiles',
    'list_saved_profiles',
    'delete_profile',
    'profile_exists',
    # Utility functions
    'reload_custom_profiles',
    'is_builtin_profile',
]
