"""
Fortress template — military stronghold with chokepoints.
"""

from ..base import GenerationTemplate


FORTRESS_TEMPLATE = GenerationTemplate(
    name="Fortress",
    description="Military stronghold with chokepoint corridors and tactical rooms. Structured defensive progression.",
    category="Combat",

    # Layout parameters
    map_width=40,
    map_height=40,
    room_count=10,
    complexity=4,
    corridor_width=96,

    # Generation hints
    preferred_room_types=['Stronghold', 'Gatehouse', 'Armory', 'Barracks', 'Prison', 'Vault'],
    preferred_hall_types=['StraightHall', 'StraightHall', 'TJunction', 'SquareCorner'],
    room_probability=0.4,
    min_hall_between_rooms=2,
    allow_dead_ends=False,
)
