"""
Arena template — large open combat spaces.
"""

from ..base import GenerationTemplate


ARENA_TEMPLATE = GenerationTemplate(
    name="Arena",
    description="Large open areas for intense combat encounters. Favors big rooms with wide corridors.",
    category="Combat",

    # Layout parameters
    map_width=40,
    map_height=40,
    room_count=8,
    complexity=4,
    corridor_width=128,

    # Generation hints
    preferred_room_types=['GreatHall', 'Arena', 'Stronghold', 'Chamber'],
    preferred_hall_types=['Crossroads', 'TJunction', 'StraightHall'],
    room_probability=0.6,
    min_hall_between_rooms=1,
    allow_dead_ends=False,
)
