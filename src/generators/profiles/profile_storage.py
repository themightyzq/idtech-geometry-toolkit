"""
Profile persistence layer for custom game profiles.

Handles save/load of custom profiles to ~/.config/quake_levelgenerator/profiles/
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .game_profile import GameProfile

logger = logging.getLogger(__name__)


def get_profiles_dir() -> Path:
    """
    Get the directory for storing custom profiles.

    Returns:
        Path to ~/.config/quake_levelgenerator/profiles/
        Creates the directory if it doesn't exist.
    """
    config_dir = Path.home() / ".config" / "quake_levelgenerator" / "profiles"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _profile_to_dict(profile: GameProfile) -> Dict[str, Any]:
    """Convert a GameProfile to a JSON-serializable dictionary."""
    return {
        "name": profile.name,
        "engine": profile.engine,
        "description": profile.description,
        "entities": profile.entities,
        "worldspawn": profile.worldspawn,
    }


def _dict_to_profile(data: Dict[str, Any]) -> GameProfile:
    """Create a GameProfile from a dictionary."""
    return GameProfile(
        name=data.get("name", "Unknown"),
        engine=data.get("engine", "idtech1"),
        description=data.get("description", ""),
        entities=data.get("entities", {}),
        worldspawn=data.get("worldspawn", {}),
    )


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a profile name for use as a filename.

    Args:
        name: The profile name

    Returns:
        A safe filename (lowercase, spaces replaced with underscores, special chars removed)
    """
    # Replace spaces with underscores, convert to lowercase
    safe = name.lower().replace(" ", "_")
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")
    return safe or "profile"


def save_profile(profile: GameProfile) -> Path:
    """
    Save a profile to the profiles directory.

    Args:
        profile: The GameProfile to save

    Returns:
        Path to the saved file

    Raises:
        IOError: If the file cannot be written
    """
    profiles_dir = get_profiles_dir()
    filename = _sanitize_filename(profile.name) + ".json"
    file_path = profiles_dir / filename

    data = _profile_to_dict(profile)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return file_path


def load_profile(name: str) -> Optional[GameProfile]:
    """
    Load a profile by name from the profiles directory.

    Args:
        name: The profile name (will be converted to filename)

    Returns:
        GameProfile if found and valid, None otherwise
    """
    profiles_dir = get_profiles_dir()
    filename = _sanitize_filename(name) + ".json"
    file_path = profiles_dir / filename

    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _dict_to_profile(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_profile_from_path(file_path: Path) -> Optional[GameProfile]:
    """
    Load a profile from a specific file path.

    Args:
        file_path: Path to the JSON profile file

    Returns:
        GameProfile if valid, None otherwise
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _dict_to_profile(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def list_saved_profiles() -> List[str]:
    """
    List all saved custom profile names.

    Returns:
        Sorted list of profile names (without .json extension)
    """
    profiles_dir = get_profiles_dir()
    profile_names = []

    for file_path in profiles_dir.glob("*.json"):
        profile = load_profile_from_path(file_path)
        if profile:
            profile_names.append(profile.name)

    return sorted(profile_names)


def load_all_saved_profiles() -> List[GameProfile]:
    """
    Load all saved custom profiles.

    Returns:
        List of GameProfile instances from saved files
    """
    profiles_dir = get_profiles_dir()
    profiles = []

    for file_path in profiles_dir.glob("*.json"):
        profile = load_profile_from_path(file_path)
        if profile:
            profiles.append(profile)

    return profiles


def delete_profile(name: str) -> bool:
    """
    Delete a saved profile by name.

    Args:
        name: The profile name to delete

    Returns:
        True if deleted, False if not found
    """
    profiles_dir = get_profiles_dir()
    filename = _sanitize_filename(name) + ".json"
    file_path = profiles_dir / filename

    if file_path.exists():
        file_path.unlink()
        return True

    # Also try to find by iterating (in case filename doesn't match)
    for fp in profiles_dir.glob("*.json"):
        profile = load_profile_from_path(fp)
        if profile and profile.name == name:
            fp.unlink()
            return True

    return False


def profile_exists(name: str) -> bool:
    """
    Check if a profile with the given name exists.

    Args:
        name: The profile name to check

    Returns:
        True if the profile exists, False otherwise
    """
    profiles_dir = get_profiles_dir()
    filename = _sanitize_filename(name) + ".json"
    file_path = profiles_dir / filename

    if file_path.exists():
        return True

    # Also check by iterating (in case filename doesn't match)
    for fp in profiles_dir.glob("*.json"):
        profile = load_profile_from_path(fp)
        if profile and profile.name == name:
            return True

    return False
