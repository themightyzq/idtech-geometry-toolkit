"""
Cathedral template — grand atmospheric chambers.
"""

from ..base import GenerationTemplate


CATHEDRAL_TEMPLATE = GenerationTemplate(
    name="Cathedral",
    description="Grand chambers with tall spaces for atmospheric exploration. Favors vertical architecture.",
    category="Atmospheric",

    # Layout parameters
    map_width=35,
    map_height=50,
    room_count=10,
    complexity=5,
    corridor_width=96,

    # Generation hints
    preferred_room_types=['Sanctuary', 'GreatHall', 'Tomb', 'Cistern'],
    preferred_hall_types=['StraightHall', 'TJunction', 'Crossroads'],
    room_probability=0.5,
    min_hall_between_rooms=1,
    allow_dead_ends=True,
)
