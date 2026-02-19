"""
Layout Types Module for 2D Dungeon Generation

This module provides data structures for representing 2D dungeon layouts
that can be converted to 3D idTech geometry.
"""

from .layout_types import (
    Layout2D,
    LayoutRoom,
    LayoutConnection,
    MultiLevelLayout,
    VerticalConnection,
    LayoutConverter,
    TileType,
    ConnectionType,
    TILE_SIZE
)

__all__ = [
    'Layout2D',
    'LayoutRoom',
    'LayoutConnection',
    'MultiLevelLayout',
    'VerticalConnection',
    'LayoutConverter',
    'TileType',
    'ConnectionType',
    'TILE_SIZE'
]

__version__ = '1.0.0'