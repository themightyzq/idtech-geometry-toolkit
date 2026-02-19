"""
Maze template — complex winding corridors.
"""

from ..base import GenerationTemplate


MAZE_TEMPLATE = GenerationTemplate(
    name="Maze",
    description="Complex winding corridors for exploration. Many turns and dead ends.",
    category="Exploration",

    # Layout parameters
    map_width=50,
    map_height=50,
    room_count=20,
    complexity=8,
    corridor_width=64,

    # Generation hints
    preferred_room_types=['Tower', 'Storage', 'Cistern', 'Armory'],
    preferred_hall_types=['SquareCorner', 'StraightHall'],
    room_probability=0.3,
    min_hall_between_rooms=3,
    allow_dead_ends=True,
)
