"""
Cathedral template — grand processional halls.
"""

from ..base import GenerationTemplate


CATHEDRAL_TEMPLATE = GenerationTemplate(
    name="Cathedral",
    description="Grand processional halls leading to a central sanctuary. Side chapels and crypts branch from the main axis.",
    category="Atmospheric",

    # Layout parameters
    map_width=30,
    map_height=45,
    room_count=8,
    complexity=4,
    corridor_width=128,

    # Generation hints
    preferred_room_types=['Sanctuary', 'Cloister', 'GreatHall', 'Tomb', 'Shrine', 'Cistern', 'Hub', 'Processional', 'ThroneRoom'],
    preferred_hall_types=['StraightHall', 'StraightHall', 'WideCorridor', 'WideCorner', 'TJunction'],
    room_probability=0.45,
    min_hall_between_rooms=1,
    allow_dead_ends=True,
    max_shortcuts=1,
)
