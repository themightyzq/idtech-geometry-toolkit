"""
GameProfile dataclass and ProfileCatalog — defines entity and worldspawn mappings for target game engines.

Each profile maps generic entity names to game-specific classnames and provides
default worldspawn properties. This allows the toolkit to generate maps for different idTech engines.

Note: Textures are now managed separately by TextureSettings (geometry-only philosophy).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GameProfile:
    """
    Defines entity and worldspawn mappings for a target game engine.

    Profiles provide engine-specific translations for:
    - Entity classnames (generic → game-specific)
    - Worldspawn properties

    Note: Textures are now managed globally by TextureSettings, not per-profile.
    This aligns with the "geometry-only" philosophy where textures are just
    placeholders that users will replace in their level editor.

    Attributes:
        name: Display name (e.g., "Quake 1", "Doom 3")
        engine: Engine type ("idtech1" or "idtech4")
        description: Human-readable description
        entities: Dict mapping generic names to game classnames
        worldspawn: Dict of default worldspawn properties
    """

    # Identity
    name: str
    engine: str  # "idtech1" or "idtech4"
    description: str

    # Entity mappings: {generic_name: game_classname}
    entities: Dict[str, str] = field(default_factory=dict)

    # Worldspawn properties for this engine
    worldspawn: Dict[str, str] = field(default_factory=dict)

    def get_entity_classname(self, generic_name: str) -> str:
        """
        Map generic entity name to game-specific classname.

        Args:
            generic_name: Generic entity name (e.g., "monster_grunt", "player_start")

        Returns:
            Game-specific classname, or the generic name if no mapping exists.
        """
        return self.entities.get(generic_name, generic_name)

    def get_worldspawn_properties(self) -> Dict[str, str]:
        """
        Get default worldspawn properties for this engine.

        Returns:
            Dict of key-value pairs for worldspawn entity.
        """
        return dict(self.worldspawn)


class ProfileCatalog:
    """
    Registry of game profiles.

    Manages registration and lookup of GameProfile instances.
    Provides case-insensitive lookup and default profile selection.
    """

    def __init__(self):
        self._profiles: Dict[str, GameProfile] = {}

    def register(self, profile: GameProfile) -> None:
        """
        Register a profile in the catalog.

        Args:
            profile: GameProfile instance to register
        """
        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[GameProfile]:
        """
        Get a profile by name (case-insensitive).

        Args:
            name: Profile name to look up

        Returns:
            GameProfile if found, None otherwise
        """
        # Try exact match first
        if name in self._profiles:
            return self._profiles[name]

        # Try case-insensitive match
        name_lower = name.lower()
        for pname, profile in self._profiles.items():
            if pname.lower() == name_lower:
                return profile

        return None

    def list_profiles(self) -> List[str]:
        """
        List all profile names.

        Returns:
            Sorted list of profile names
        """
        return sorted(self._profiles.keys())

    def get_default_profile(self) -> GameProfile:
        """
        Get the default profile (Doom 3 / idTech 4).

        Returns:
            The Doom 3 profile, or first available profile
        """
        return self.get_profile("Doom 3") or next(iter(self._profiles.values()))

    def get_profiles_for_engine(self, engine: str) -> List[GameProfile]:
        """
        Get all profiles for a specific engine type.

        Args:
            engine: Engine type ("idtech1" or "idtech4")

        Returns:
            List of GameProfile instances matching the engine
        """
        return [p for p in self._profiles.values() if p.engine == engine]

    def unregister(self, name: str) -> bool:
        """
        Unregister a profile from the catalog.

        Args:
            name: Profile name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._profiles:
            del self._profiles[name]
            return True

        # Try case-insensitive match
        name_lower = name.lower()
        for pname in list(self._profiles.keys()):
            if pname.lower() == name_lower:
                del self._profiles[pname]
                return True

        return False

    def is_registered(self, name: str) -> bool:
        """
        Check if a profile is registered.

        Args:
            name: Profile name to check

        Returns:
            True if registered, False otherwise
        """
        return self.get_profile(name) is not None
