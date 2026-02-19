"""
Fortress template — defensive chokepoints.
"""

from ..base import GenerationTemplate


FORTRESS_TEMPLATE = GenerationTemplate(
    name="Fortress",
    description="Defensive layout with chokepoints and strategic positions. Good for tactical combat.",
    category="Combat",

    # Layout parameters
    map_width=45,
    map_height=45,
    room_count=12,
    complexity=6,
    corridor_width=96,

    # Generation hints
    preferred_room_types=['Stronghold', 'Armory', 'Prison', 'Chamber', 'Tomb'],
    preferred_hall_types=['TJunction', 'StraightHall', 'SquareCorner'],
    room_probability=0.45,
    min_hall_between_rooms=2,
    allow_dead_ends=False,
)
