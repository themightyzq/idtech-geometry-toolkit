"""
Built-in generation templates.
"""

from .arena import ARENA_TEMPLATE
from .maze import MAZE_TEMPLATE
from .fortress import FORTRESS_TEMPLATE
from .cathedral import CATHEDRAL_TEMPLATE
from .crypt import CRYPT_TEMPLATE
from .dungeon import DUNGEON_TEMPLATE


def register_builtin_templates(catalog):
    """Register all built-in templates with the catalog."""
    catalog.register(ARENA_TEMPLATE)
    catalog.register(MAZE_TEMPLATE)
    catalog.register(FORTRESS_TEMPLATE)
    catalog.register(CATHEDRAL_TEMPLATE)
    catalog.register(CRYPT_TEMPLATE)
    catalog.register(DUNGEON_TEMPLATE)


__all__ = [
    'ARENA_TEMPLATE',
    'MAZE_TEMPLATE',
    'FORTRESS_TEMPLATE',
    'CATHEDRAL_TEMPLATE',
    'CRYPT_TEMPLATE',
    'DUNGEON_TEMPLATE',
    'register_builtin_templates',
]
