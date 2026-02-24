"""
Arena template — hub-and-spoke combat complex.
"""

from ..base import GenerationTemplate


ARENA_TEMPLATE = GenerationTemplate(
    name="Arena",
    description="Hub-and-spoke combat complex. Large central rooms with multiple approach angles and wide corridors.",
    category="Combat",

    # Layout parameters
    map_width=35,
    map_height=35,
    room_count=6,
    complexity=5,
    corridor_width=128,

    # Generation hints
    preferred_room_types=['Colosseum', 'Arena', 'GreatHall', 'Stronghold', 'Pit', 'Chamber'],
    preferred_hall_types=['Crossroads', 'Crossroads', 'TJunction', 'StraightHall'],
    room_probability=0.5,
    min_hall_between_rooms=1,
    allow_dead_ends=False,
)
