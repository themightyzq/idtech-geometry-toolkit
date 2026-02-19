"""
Converter from BSP export format to Layout2D format.

This module bridges the gap between the BSP generator's export format
and the brush generator's expected Layout2D input.
"""

from typing import Dict, Any
import numpy as np
import logging
from dataclasses import dataclass, field

from quake_levelgenerator.src.generators.layout.layout_types import (
    Layout2D, LayoutRoom, LayoutConnection, ConnectionType, TileType
)

logger = logging.getLogger(__name__)


def convert_bsp_to_layout2d(bsp_export: Dict[str, Any]) -> Layout2D:
    """
    Convert BSP generator export to Layout2D format.
    
    Args:
        bsp_export: Dictionary from BSPGenerator.export_layout()
        
    Returns:
        Layout2D object compatible with brush generator
        
    Raises:
        ValueError: If bsp_export is None or invalid
    """
    logger.info("Starting BSP to Layout2D conversion")
    
    # Validate input
    if bsp_export is None:
        raise ValueError("BSP export data is None")
    
    if not isinstance(bsp_export, dict):
        raise ValueError(f"BSP export must be dict, got {type(bsp_export)}")
    
    logger.debug(f"BSP export keys: {list(bsp_export.keys()) if bsp_export else 'None'}")
    
    # Check for required keys
    if 'rooms' not in bsp_export:
        raise ValueError("BSP export missing 'rooms' key")
    
    # Get map dimensions from config
    config = bsp_export.get('config', {})
    map_width = config.get('map_width', 2048)
    map_height = config.get('map_height', 2048)
    
    logger.debug(f"Map dimensions: {map_width}x{map_height}")
    
    # Convert to tile dimensions (divide by 64 for standard tile size)
    tile_width = map_width // 64
    tile_height = map_height // 64
    
    logger.debug(f"Tile dimensions: {tile_width}x{tile_height}")
    
    # Create Layout2D object
    layout = Layout2D(width=tile_width, height=tile_height)
    
    # Convert rooms
    rooms_data = bsp_export.get('rooms', [])
    if not rooms_data:
        logger.warning("No rooms found in BSP export data")
        raise ValueError("BSP export contains no rooms")
    
    logger.info(f"Converting {len(rooms_data)} rooms")
    
    for i, room_data in enumerate(rooms_data):
        try:
            # Validate room data
            if 'id' not in room_data:
                logger.error(f"Room {i} missing 'id' field")
                continue
            if 'bounds' not in room_data:
                logger.error(f"Room {i} missing 'bounds' field")
                continue
            
            bounds = room_data['bounds']
            required_bounds = ['x', 'y', 'width', 'height']
            for bound_key in required_bounds:
                if bound_key not in bounds:
                    logger.error(f"Room {i} bounds missing '{bound_key}' field")
                    continue
            
            # Create LayoutRoom from BSP room data
            room = LayoutRoom(
                room_id=room_data['id'],
                room_type=room_data.get('type', 'standard').lower(),
                x=bounds['x'] // 64,  # Convert to tiles
                y=bounds['y'] // 64,
                width=bounds['width'] // 64,
                height=bounds['height'] // 64,
                z_offset=room_data.get('z_offset', 0),
            )
            layout.rooms.append(room)
            logger.debug(f"Added room {room.room_id} at ({room.x}, {room.y}) size {room.width}x{room.height}")
            
        except Exception as e:
            logger.error(f"Failed to convert room {i}: {e}")
            continue
        
        # Fill the grid with floor tiles (rooms have floors)
        for tx in range(room.x, room.x + room.width):
            for ty in range(room.y, room.y + room.height):
                layout.set_tile(tx, ty, TileType.FLOOR)
    
    # Convert corridors/connections
    corridors_data = bsp_export.get('corridors', [])
    logger.info(f"Converting {len(corridors_data)} corridors")
    
    for i, corridor_data in enumerate(corridors_data):
        # Create LayoutConnection from BSP corridor data
        # Find the rooms this corridor connects
        start_room = None
        end_room = None
        
        for room in layout.rooms:
            if room.room_id == corridor_data.get('start_room_id'):
                start_room = room
            if room.room_id == corridor_data.get('end_room_id'):
                end_room = room
        
        if start_room and end_room:
            connection = LayoutConnection(
                start_room=start_room,
                end_room=end_room,
                connection_type=ConnectionType.CORRIDOR,
                width=corridor_data.get('width', 2)
            )
            
            # Convert path rectangles if available (world units) and store metadata
            if 'path' in corridor_data and corridor_data['path']:
                # Store world rectangles for precise doorway alignment later
                rects_world = [
                    {
                        'x': p.get('x', 0),
                        'y': p.get('y', 0),
                        'width': p.get('width', 0),
                        'height': p.get('height', 0),
                        'z': p.get('z', 0)
                    }
                    for p in corridor_data['path']
                ]
                connection.metadata['path_rects_world'] = rects_world
                # Build a coarse tile path from rect centers for the 2D grid
                connection.path = [
                    (p['x'] // 64, p['y'] // 64)
                    for p in corridor_data['path']
                ]
                logger.debug(f"Corridor {i} has {len(connection.path)} path points and {len(rects_world)} world rects")
            else:
                # Create a simple direct path between room centers
                start_center = (start_room.x + start_room.width // 2, start_room.y + start_room.height // 2)
                end_center = (end_room.x + end_room.width // 2, end_room.y + end_room.height // 2)
                connection.path = [start_center, end_center]
                logger.debug(f"Corridor {i} using direct path: {start_center} -> {end_center}")
            
            layout.connections.append(connection)
            
            # Add connection to both rooms' connection lists
            start_room.connections.append(connection)
            end_room.connections.append(connection)
            
            # Fill grid with floor tiles (corridors are walkable)
            if connection.path:
                for point in connection.path:
                    tx, ty = point
                    # Ensure we stay within bounds
                    for dx in range(max(1, connection.width)):
                        for dy in range(max(1, connection.width)):
                            new_tx, new_ty = tx + dx, ty + dy
                            if 0 <= new_tx < layout.width and 0 <= new_ty < layout.height:
                                layout.set_tile(new_tx, new_ty, TileType.FLOOR)
            
            logger.debug(f"Added corridor connection between rooms {start_room.room_id} and {end_room.room_id}")
        else:
            logger.warning(f"Corridor {i} missing start or end room (start_id: {corridor_data.get('start_room_id')}, end_id: {corridor_data.get('end_room_id')})")
    
    # Store metadata
    layout.metadata = {
        'original_config': config,
        'stats': bsp_export.get('stats', {})
    }
    
    # Final validation
    if not layout.rooms:
        raise ValueError("No rooms were successfully converted")
    
    logger.info(f"Conversion complete: {len(layout.rooms)} rooms, {len(layout.connections)} connections")
    
    # Log final layout structure for debugging
    for room in layout.rooms:
        logger.debug(f"Final room {room.room_id}: ({room.x}, {room.y}) {room.width}x{room.height}, connections: {len(room.connections)}")
    
    for connection in layout.connections:
        logger.debug(f"Final connection: {connection.start_room.room_id} -> {connection.end_room.room_id}, path points: {len(connection.path) if connection.path else 0}")
    
    return layout


def convert_bsp_rooms_directly(bsp_generator) -> Layout2D:
    """
    Convert directly from BSPGenerator instance to Layout2D.
    
    This is more efficient than going through export/import.
    
    Args:
        bsp_generator: BSPGenerator instance with generated rooms
        
    Returns:
        Layout2D object
    """
    # Get dimensions
    tile_width = bsp_generator.map_width // 64
    tile_height = bsp_generator.map_height // 64
    
    # Create Layout2D
    layout = Layout2D(width=tile_width, height=tile_height)
    
    # Convert rooms directly
    for room in bsp_generator.rooms:
        layout_room = LayoutRoom(
            room_id=room.id,
            room_type=room.room_type.name.lower(),
            x=room.bounds.x // 64,
            y=room.bounds.y // 64,
            width=room.bounds.width // 64,
            height=room.bounds.height // 64
        )
        layout.rooms.append(layout_room)
        
        # Fill grid with floor tiles
        for tx in range(layout_room.x, layout_room.x + layout_room.width):
            for ty in range(layout_room.y, layout_room.y + layout_room.height):
                layout.set_tile(tx, ty, TileType.FLOOR)
    
    # Convert corridors
    for corridor in bsp_generator.corridors:
        # Find connected rooms
        start_room = None
        end_room = None
        
        for lr in layout.rooms:
            for bsp_room in bsp_generator.rooms:
                if bsp_room.id == lr.room_id:
                    if bsp_room == corridor.room1:
                        start_room = lr
                    elif bsp_room == corridor.room2:
                        end_room = lr
                    break
        
        if start_room and end_room:
            connection = LayoutConnection(
                start_room=start_room,
                end_room=end_room,
                connection_type=ConnectionType.CORRIDOR,
                width=corridor.width // 64
            )
            
            # Simple path - just connect centers for now
            # In a real implementation, we'd trace the actual corridor path
            connection.path = [
                (start_room.center[0], start_room.center[1]),
                (end_room.center[0], end_room.center[1])
            ]
            
            layout.connections.append(connection)
            start_room.connections.append(connection)
            end_room.connections.append(connection)
    
    return layout
