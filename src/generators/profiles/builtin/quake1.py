"""
Quake 1 game profile — entity classnames and worldspawn settings.

This profile defines entity classnames and worldspawn properties
compatible with the original Quake engine (idTech 1).

Note: Textures are managed globally by TextureSettings, not per-profile.
"""

from quake_levelgenerator.src.generators.profiles.game_profile import GameProfile


QUAKE1_PROFILE = GameProfile(
    name="Quake 1",
    engine="idtech1",
    description="Original Quake (1996) - idTech 1 format",
    entities={
        # Player start (required for valid maps)
        "player_start": "info_player_start",
    },
    worldspawn={
        "wad": "id1.wad",
        "light": "50",
        "_sunlight": "150",
        "_sunlight_pitch": "-45",
        "_sunlight_angle": "315",
    },
)
