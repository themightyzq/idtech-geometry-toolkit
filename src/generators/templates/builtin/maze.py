"""
Maze template — dense winding corridors.
"""

from ..base import GenerationTemplate


MAZE_TEMPLATE = GenerationTemplate(
    name="Maze",
    description="Dense winding corridors with many turns and dead ends. Small rooms as rewards for exploration.",
    category="Exploration",

    # Layout parameters
    map_width=45,
    map_height=45,
    room_count=12,
    complexity=3,
    corridor_width=64,

    # Generation hints
    preferred_room_types=['Storage', 'Ossuary', 'Tower', 'Shrine', 'Sewer', 'Pit', 'Vestibule'],
    preferred_hall_types=['SquareCorner', 'SquareCorner', 'StraightHall', 'NarrowPassage'],
    room_probability=0.25,
    min_hall_between_rooms=3,
    allow_dead_ends=True,
)
