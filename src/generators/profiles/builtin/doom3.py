"""
Doom 3 game profile — entity classnames and worldspawn settings.

This profile defines entity classnames and worldspawn properties
compatible with Doom 3 and similar idTech 4 engines.

Note: Textures are managed globally by TextureSettings, not per-profile.
"""

from quake_levelgenerator.src.generators.profiles.game_profile import GameProfile


DOOM3_PROFILE = GameProfile(
    name="Doom 3",
    engine="idtech4",
    description="Doom 3 / idTech 4 format",
    entities={
        # Player start (required for valid maps)
        "player_start": "info_player_start",
    },
    worldspawn={
        # idTech 4 has minimal worldspawn requirements
        # Most properties are set via map settings
    },
)
