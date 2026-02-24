"""
Dungeon template — classic dungeon crawl.
"""

from ..base import GenerationTemplate


DUNGEON_TEMPLATE = GenerationTemplate(
    name="Dungeon",
    description="Classic dungeon crawl with balanced variety. A mix of combat, exploration, and atmosphere.",
    category="Exploration",

    # Layout parameters
    map_width=35,
    map_height=35,
    room_count=10,
    complexity=4,
    corridor_width=96,

    # Generation hints
    preferred_room_types=['Chamber', 'Prison', 'Sewer', 'Armory', 'Storage', 'Tomb', 'Laboratory'],
    preferred_hall_types=['StraightHall', 'SquareCorner', 'TJunction'],
    room_probability=0.4,
    min_hall_between_rooms=1,
    allow_dead_ends=True,
)
