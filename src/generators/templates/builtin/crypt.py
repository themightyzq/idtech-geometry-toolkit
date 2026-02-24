"""
Crypt template — underground burial complex.
"""

from ..base import GenerationTemplate


CRYPT_TEMPLATE = GenerationTemplate(
    name="Crypt",
    description="Underground burial complex with tight passages and tomb chambers. Oppressive and claustrophobic.",
    category="Atmospheric",

    # Layout parameters
    map_width=35,
    map_height=35,
    room_count=10,
    complexity=3,
    corridor_width=64,

    # Generation hints
    preferred_room_types=['Tomb', 'Ossuary', 'Shrine', 'Pit', 'Storage', 'Chamber'],
    preferred_hall_types=['StraightHall', 'SquareCorner', 'SquareCorner'],
    room_probability=0.4,
    min_hall_between_rooms=1,
    allow_dead_ends=True,
)
