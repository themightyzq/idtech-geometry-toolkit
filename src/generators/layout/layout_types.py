#!/usr/bin/env python3
"""
Layout Types for 2D Dungeon Generation

This module defines the core data structures for representing 2D dungeon layouts
that will be converted to 3D idTech geometry. These types bridge the gap between
abstract dungeon generation algorithms and concrete idTech map structures.

The layout system uses a tile-based approach internally but exports continuous
coordinates for idTech's brush-based geometry system.

Author: idTech Map Generator
License: MIT
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


# Tile size in idTech units (power of 2 for optimal BSP compilation)
TILE_SIZE = 64  # Each tile is 64x64 idTech units


class TileType(Enum):
    """Types of tiles in the 2D layout grid"""
    EMPTY = 0       # Void/solid space
    FLOOR = 1       # Walkable floor
    WALL = 2        # Solid wall
    DOOR = 3        # Door connection
    STAIRS_UP = 4   # Stairs going up
    STAIRS_DOWN = 5 # Stairs going down
    LIFT = 6        # Elevator/lift
    WATER = 7       # Water hazard
    LAVA = 8        # Lava hazard
    PILLAR = 9      # Decorative pillar


class ConnectionType(Enum):
    """Types of connections between layout elements"""
    CORRIDOR = auto()     # Standard corridor
    DOOR_SINGLE = auto()  # Single door
    DOOR_DOUBLE = auto()  # Double door
    ARCH = auto()         # Open archway
    STAIRS = auto()       # Stairway connection
    LIFT = auto()         # Elevator shaft
    TELEPORTER = auto()   # Teleporter (for non-contiguous connections)


@dataclass
class Layout2D:
    """
    Represents a 2D dungeon layout as a grid of tiles.
    
    This is the primary data structure for 2D layout generation,
    which will be converted to 3D idTech geometry.
    """
    
    width: int  # Width in tiles
    height: int  # Height in tiles
    grid: np.ndarray = field(init=False)  # 2D array of TileType values
    rooms: List['LayoutRoom'] = field(default_factory=list)
    connections: List['LayoutConnection'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the grid after dataclass initialization"""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
    
    def set_tile(self, x: int, y: int, tile_type: TileType):
        """Set a tile in the grid"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = tile_type.value
    
    def get_tile(self, x: int, y: int) -> TileType:
        """Get tile type at position"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return TileType(self.grid[y, x])
        return TileType.EMPTY
    
    def fill_rect(self, x: int, y: int, width: int, height: int, tile_type: TileType):
        """Fill a rectangular area with a tile type"""
        for dy in range(height):
            for dx in range(width):
                self.set_tile(x + dx, y + dy, tile_type)
    
    def draw_room(self, room: 'LayoutRoom'):
        """Draw a room on the grid"""
        # Fill floor
        self.fill_rect(
            room.x + 1, room.y + 1,
            room.width - 2, room.height - 2,
            TileType.FLOOR
        )
        
        # Draw walls
        # Top and bottom walls
        for x in range(room.x, room.x + room.width):
            self.set_tile(x, room.y, TileType.WALL)
            self.set_tile(x, room.y + room.height - 1, TileType.WALL)
        
        # Left and right walls
        for y in range(room.y, room.y + room.height):
            self.set_tile(room.x, y, TileType.WALL)
            self.set_tile(room.x + room.width - 1, y, TileType.WALL)
    
    def draw_corridor(self, corridor: 'LayoutConnection'):
        """Draw a corridor on the grid"""
        for point in corridor.path:
            x, y = point
            # Draw floor with walls on sides
            if corridor.connection_type == ConnectionType.CORRIDOR:
                self.set_tile(x, y, TileType.FLOOR)
                
                # Add walls around corridor if they're empty
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self.get_tile(nx, ny) == TileType.EMPTY:
                        self.set_tile(nx, ny, TileType.WALL)
    
    def to_world_coords(self, tile_x: int, tile_y: int) -> Tuple[int, int]:
        """Convert tile coordinates to idTech world coordinates"""
        return (tile_x * TILE_SIZE, tile_y * TILE_SIZE)
    
    def from_world_coords(self, world_x: int, world_y: int) -> Tuple[int, int]:
        """Convert idTech world coordinates to tile coordinates"""
        return (world_x // TILE_SIZE, world_y // TILE_SIZE)
    
    def get_connected_regions(self) -> List[Set[Tuple[int, int]]]:
        """
        Find all connected floor regions using flood fill.
        Returns a list of sets, each containing tile coordinates of a connected region.
        """
        visited = set()
        regions = []
        
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in visited and self.get_tile(x, y) == TileType.FLOOR:
                    # Start flood fill from this point
                    region = self._flood_fill(x, y, visited)
                    if region:
                        regions.append(region)
        
        return regions
    
    def _flood_fill(self, start_x: int, start_y: int, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Flood fill to find connected floor tiles"""
        region = set()
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x, y) in visited:
                continue
            
            if self.get_tile(x, y) != TileType.FLOOR:
                continue
            
            visited.add((x, y))
            region.add((x, y))
            
            # Check adjacent tiles
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    stack.append((nx, ny))
        
        return region
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the layout for common issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for disconnected regions
        regions = self.get_connected_regions()
        if len(regions) > 1:
            issues.append(f"Layout has {len(regions)} disconnected regions")
        elif len(regions) == 0:
            issues.append("Layout has no walkable floor")
        
        # Check room connectivity
        for room in self.rooms:
            if not room.has_connections():
                issues.append(f"Room at ({room.x}, {room.y}) has no connections")
        
        # Check for minimum playable area
        total_floor = np.sum(self.grid == TileType.FLOOR.value)
        if total_floor < 10:  # Minimum 10 tiles of floor
            issues.append(f"Insufficient floor area: {total_floor} tiles")
        
        return len(issues) == 0, issues


@dataclass
class LayoutRoom:
    """
    Represents a room in the 2D layout.
    
    Coordinates are in tile space, not idTech units.
    """
    x: int  # Left position in tiles
    y: int  # Top position in tiles
    width: int  # Width in tiles
    height: int  # Height in tiles
    room_id: int = 0
    room_type: str = "standard"
    z_offset: int = 0  # Floor height offset in idTech units
    connections: List['LayoutConnection'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of room in tile coordinates"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get room bounds as (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this room"""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height
    
    def intersects(self, other: 'LayoutRoom') -> bool:
        """Check if this room intersects with another"""
        return not (self.x + self.width <= other.x or 
                   other.x + other.width <= self.x or
                   self.y + self.height <= other.y or
                   other.y + other.height <= self.y)
    
    def distance_to(self, other: 'LayoutRoom') -> float:
        """Calculate Manhattan distance to another room"""
        c1 = self.center
        c2 = other.center
        return abs(c2[0] - c1[0]) + abs(c2[1] - c1[1])
    
    def has_connections(self) -> bool:
        """Check if room has any connections"""
        return len(self.connections) > 0
    
    def to_world_bounds(self) -> Tuple[int, int, int, int]:
        """Convert room bounds to idTech world coordinates"""
        return (
            self.x * TILE_SIZE,
            self.y * TILE_SIZE,
            (self.x + self.width) * TILE_SIZE,
            (self.y + self.height) * TILE_SIZE
        )


@dataclass
class LayoutConnection:
    """
    Represents a connection between rooms (corridor, door, etc).
    
    Path is a list of tile coordinates forming the connection.
    """
    start_room: LayoutRoom
    end_room: LayoutRoom
    connection_type: ConnectionType = ConnectionType.CORRIDOR
    path: List[Tuple[int, int]] = field(default_factory=list)
    width: int = 1  # Width in tiles
    metadata: Dict = field(default_factory=dict)
    
    def calculate_length(self) -> int:
        """Calculate the length of the connection path"""
        if len(self.path) < 2:
            return 0
        
        length = 0
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            length += abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
        
        return length
    
    def is_straight(self) -> bool:
        """Check if the connection is a straight line"""
        if len(self.path) < 2:
            return True
        
        # Check if all points are on same X or Y axis
        x_coords = [p[0] for p in self.path]
        y_coords = [p[1] for p in self.path]
        
        return len(set(x_coords)) == 1 or len(set(y_coords)) == 1


@dataclass
class MultiLevelLayout:
    """
    Container for multiple 2D layouts representing different floors/levels.
    
    This allows for true 3D dungeons with multiple floors connected by
    stairs, lifts, and other vertical transitions.
    """
    levels: List[Layout2D] = field(default_factory=list)
    vertical_connections: List['VerticalConnection'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_level(self, layout: Layout2D) -> int:
        """Add a new level and return its index"""
        self.levels.append(layout)
        return len(self.levels) - 1
    
    def connect_levels(self, level1_idx: int, level2_idx: int, 
                      x: int, y: int, connection_type: ConnectionType):
        """Create a vertical connection between levels"""
        if 0 <= level1_idx < len(self.levels) and 0 <= level2_idx < len(self.levels):
            connection = VerticalConnection(
                lower_level=level1_idx,
                upper_level=level2_idx,
                x=x,
                y=y,
                connection_type=connection_type
            )
            self.vertical_connections.append(connection)
            
            # Mark tiles on both levels
            if connection_type == ConnectionType.STAIRS:
                self.levels[level1_idx].set_tile(x, y, TileType.STAIRS_UP)
                self.levels[level2_idx].set_tile(x, y, TileType.STAIRS_DOWN)
            elif connection_type == ConnectionType.LIFT:
                self.levels[level1_idx].set_tile(x, y, TileType.LIFT)
                self.levels[level2_idx].set_tile(x, y, TileType.LIFT)
    
    def get_total_rooms(self) -> int:
        """Get total number of rooms across all levels"""
        return sum(len(level.rooms) for level in self.levels)
    
    def validate_all_levels(self) -> Tuple[bool, List[str]]:
        """Validate all levels and vertical connections"""
        all_issues = []
        
        for i, level in enumerate(self.levels):
            valid, issues = level.validate()
            for issue in issues:
                all_issues.append(f"Level {i}: {issue}")
        
        # Check vertical connections
        for conn in self.vertical_connections:
            if conn.lower_level >= len(self.levels):
                all_issues.append(f"Invalid lower level index: {conn.lower_level}")
            if conn.upper_level >= len(self.levels):
                all_issues.append(f"Invalid upper level index: {conn.upper_level}")
        
        return len(all_issues) == 0, all_issues


@dataclass
class VerticalConnection:
    """Represents a vertical connection between levels"""
    lower_level: int  # Index of lower level
    upper_level: int  # Index of upper level
    x: int  # X position in tiles
    y: int  # Y position in tiles
    connection_type: ConnectionType
    metadata: Dict = field(default_factory=dict)


class LayoutConverter:
    """
    Converts between different layout representations.
    
    This class bridges the gap between the BSP generator's room/corridor
    representation and the tile-based Layout2D representation.
    """
    
    @staticmethod
    def from_bsp_layout(bsp_rooms: List, bsp_corridors: List, 
                        map_width: int, map_height: int) -> Layout2D:
        """
        Convert BSP generator output to Layout2D.
        
        Args:
            bsp_rooms: List of rooms from BSP generator
            bsp_corridors: List of corridors from BSP generator
            map_width: Map width in idTech units
            map_height: Map height in idTech units
            
        Returns:
            Layout2D representation
        """
        # Calculate grid size in tiles
        grid_width = map_width // TILE_SIZE
        grid_height = map_height // TILE_SIZE
        
        # Create layout
        layout = Layout2D(width=grid_width, height=grid_height)
        
        # Convert and add rooms
        for bsp_room in bsp_rooms:
            # Convert idTech coordinates to tile coordinates
            tile_x = bsp_room.bounds.x // TILE_SIZE
            tile_y = bsp_room.bounds.y // TILE_SIZE
            tile_width = bsp_room.bounds.width // TILE_SIZE
            tile_height = bsp_room.bounds.height // TILE_SIZE
            
            # Create LayoutRoom
            layout_room = LayoutRoom(
                x=tile_x,
                y=tile_y,
                width=tile_width,
                height=tile_height,
                room_id=bsp_room.id,
                room_type=bsp_room.room_type.name
            )
            
            layout.rooms.append(layout_room)
            layout.draw_room(layout_room)
        
        # Convert and add corridors
        for bsp_corridor in bsp_corridors:
            path = []
            
            # Convert corridor rectangles to tile path
            for rect in bsp_corridor.path:
                # Get center of corridor segment
                cx = (rect.x + rect.width // 2) // TILE_SIZE
                cy = (rect.y + rect.height // 2) // TILE_SIZE
                
                # Add tiles along the corridor
                tile_width = max(1, rect.width // TILE_SIZE)
                tile_height = max(1, rect.height // TILE_SIZE)
                
                for dy in range(tile_height):
                    for dx in range(tile_width):
                        path.append((cx - tile_width // 2 + dx, cy - tile_height // 2 + dy))
            
            # Find corresponding layout rooms
            start_room = layout.rooms[bsp_corridor.start_room.id]
            end_room = layout.rooms[bsp_corridor.end_room.id]
            
            # Create LayoutConnection
            connection = LayoutConnection(
                start_room=start_room,
                end_room=end_room,
                connection_type=ConnectionType.CORRIDOR,
                path=path,
                width=max(1, bsp_corridor.width // TILE_SIZE)
            )
            
            layout.connections.append(connection)
            layout.draw_corridor(connection)
            
            # Update room connections
            start_room.connections.append(connection)
            end_room.connections.append(connection)
        
        return layout
    
    @staticmethod
    def to_ascii(layout: Layout2D) -> str:
        """
        Convert Layout2D to ASCII representation for debugging.
        
        Returns:
            ASCII string representation of the layout
        """
        # Tile type to ASCII character mapping
        char_map = {
            TileType.EMPTY.value: ' ',
            TileType.FLOOR.value: '.',
            TileType.WALL.value: '#',
            TileType.DOOR.value: '+',
            TileType.STAIRS_UP.value: '<',
            TileType.STAIRS_DOWN.value: '>',
            TileType.LIFT.value: 'E',
            TileType.WATER.value: '~',
            TileType.LAVA.value: '!',
            TileType.PILLAR.value: 'O'
        }
        
        lines = []
        for y in range(layout.height):
            line = ''
            for x in range(layout.width):
                tile_value = layout.grid[y, x]
                line += char_map.get(tile_value, '?')
            lines.append(line)
        
        return '\n'.join(lines)


def test_layout_system():
    """Test the layout system"""
    print("=== Testing Layout System ===\n")
    
    # Create a simple test layout
    layout = Layout2D(width=20, height=20)
    
    # Add some rooms
    room1 = LayoutRoom(x=2, y=2, width=5, height=5, room_id=0, room_type="entrance")
    room2 = LayoutRoom(x=10, y=10, width=6, height=4, room_id=1, room_type="standard")
    
    layout.rooms.append(room1)
    layout.rooms.append(room2)
    
    # Draw rooms
    layout.draw_room(room1)
    layout.draw_room(room2)
    
    # Create a corridor
    path = []
    # Horizontal segment
    for x in range(room1.center[0], room2.center[0]):
        path.append((x, room1.center[1]))
    # Vertical segment
    for y in range(room1.center[1], room2.center[1] + 1):
        path.append((room2.center[0], y))
    
    corridor = LayoutConnection(
        start_room=room1,
        end_room=room2,
        connection_type=ConnectionType.CORRIDOR,
        path=path
    )
    
    layout.connections.append(corridor)
    layout.draw_corridor(corridor)
    
    # Print ASCII representation
    print("Layout ASCII representation:")
    print(LayoutConverter.to_ascii(layout))
    
    # Validate
    valid, issues = layout.validate()
    print(f"\nValidation: {'PASSED' if valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Test multi-level
    print("\n=== Testing Multi-Level Layout ===\n")
    
    multi_layout = MultiLevelLayout()
    multi_layout.add_level(layout)
    
    # Create second level
    layout2 = Layout2D(width=20, height=20)
    room3 = LayoutRoom(x=8, y=8, width=7, height=7, room_id=2, room_type="arena")
    layout2.rooms.append(room3)
    layout2.draw_room(room3)
    multi_layout.add_level(layout2)
    
    # Connect levels
    multi_layout.connect_levels(0, 1, 12, 12, ConnectionType.STAIRS)
    
    print(f"Total rooms across {len(multi_layout.levels)} levels: {multi_layout.get_total_rooms()}")
    
    valid, issues = multi_layout.validate_all_levels()
    print(f"Multi-level validation: {'PASSED' if valid else 'FAILED'}")
    

if __name__ == "__main__":
    test_layout_system()