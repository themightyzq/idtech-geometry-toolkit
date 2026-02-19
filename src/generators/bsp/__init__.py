"""
BSP (Binary Space Partitioning) Generator Module

This module provides the core BSP algorithm for procedural dungeon generation
in the idTech Map Generator project.
"""

from .bsp_generator import (
    BSPGenerator,
    BSPNode,
    Room,
    Corridor,
    Rectangle,
    RoomType,
    SplitDirection,
    # Constants
    IDTECH_PLAYER_WIDTH,
    IDTECH_PLAYER_HEIGHT,
    IDTECH_MIN_CLEARANCE,
    IDTECH_MIN_ROOM_SIZE,
    IDTECH_MIN_CORRIDOR_WIDTH,
    IDTECH_MAX_DEADEND_LENGTH,
    IDTECH_STANDARD_CEILING_HEIGHT,
    IDTECH_TALL_CEILING_HEIGHT
)

__all__ = [
    'BSPGenerator',
    'BSPNode',
    'Room',
    'Corridor',
    'Rectangle',
    'RoomType',
    'SplitDirection',
    'IDTECH_PLAYER_WIDTH',
    'IDTECH_PLAYER_HEIGHT',
    'IDTECH_MIN_CLEARANCE',
    'IDTECH_MIN_ROOM_SIZE',
    'IDTECH_MIN_CORRIDOR_WIDTH',
    'IDTECH_MAX_DEADEND_LENGTH',
    'IDTECH_STANDARD_CEILING_HEIGHT',
    'IDTECH_TALL_CEILING_HEIGHT'
]

__version__ = '1.0.0'