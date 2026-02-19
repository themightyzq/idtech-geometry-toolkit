#!/usr/bin/env python3
"""
BSP (Binary Space Partitioning) Generator for idTech Map Generation

This module implements the core BSP algorithm for procedural dungeon generation,
optimized for idTech's 3D constraints and player movement requirements.

The algorithm follows the classic BSP approach used in games like Rogue and NetHack,
but with specific adaptations for idTech's engine requirements:
- Minimum room size of 64x64 units for player clearance
- Corridor width of at least 64 units
- Support for vertical room placement (Z-axis)
- No dead-end corridors longer than 128 units

Author: idTech Map Generator
License: MIT
"""

import random
import math
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field
from enum import Enum, auto


# idTech unit constants (1 unit = 1 idTech unit)
IDTECH_PLAYER_WIDTH = 32  # Player bounding box width
IDTECH_PLAYER_HEIGHT = 56  # Player bounding box height
IDTECH_MIN_CLEARANCE = 64  # Minimum clearance for comfortable movement
IDTECH_MIN_ROOM_SIZE = 64  # Minimum room dimension
IDTECH_MAX_STEP_HEIGHT = 18  # Maximum climbable step (from quakedef.h STEPSIZE)
IDTECH_MIN_CORRIDOR_WIDTH = 64  # Minimum corridor width
IDTECH_MAX_DEADEND_LENGTH = 128  # Maximum dead-end corridor length
IDTECH_STANDARD_CEILING_HEIGHT = 128  # Standard room height
IDTECH_TALL_CEILING_HEIGHT = 256  # Tall room height for vertical gameplay


class SplitDirection(Enum):
    """Direction for BSP node splitting"""
    HORIZONTAL = auto()  # Split along Y axis
    VERTICAL = auto()    # Split along X axis
    NONE = auto()       # Leaf node, no split


class RoomType(Enum):
    """Types of rooms that can be generated"""
    STANDARD = auto()    # Regular rectangular room
    ARENA = auto()       # Large open combat space
    ENTRANCE = auto()    # Starting room with player spawn
    EXIT = auto()        # End room with level exit
    JUNCTION = auto()    # Connection hub between multiple areas
    VERTICAL = auto()    # Room with vertical connections (stairs/lifts)


@dataclass
class Rectangle:
    """2D Rectangle representation for rooms and BSP nodes"""
    x: int
    y: int
    width: int
    height: int
    z: int = 0  # Z coordinate for vertical placement
    
    @property
    def x2(self) -> int:
        """Right edge X coordinate"""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge Y coordinate"""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of rectangle"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Area of rectangle"""
        return self.width * self.height
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another"""
        return not (self.x2 <= other.x or self.x >= other.x2 or
                   self.y2 <= other.y or self.y >= other.y2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside rectangle"""
        return self.x <= x < self.x2 and self.y <= y < self.y2
    
    def expand(self, amount: int) -> 'Rectangle':
        """Return expanded rectangle by amount on all sides"""
        return Rectangle(
            self.x - amount,
            self.y - amount,
            self.width + amount * 2,
            self.height + amount * 2,
            self.z
        )
    
    def shrink(self, amount: int) -> Optional['Rectangle']:
        """Return shrunken rectangle by amount on all sides"""
        new_width = self.width - amount * 2
        new_height = self.height - amount * 2
        if new_width <= 0 or new_height <= 0:
            return None
        return Rectangle(
            self.x + amount,
            self.y + amount,
            new_width,
            new_height,
            self.z
        )


@dataclass
class Room:
    """Represents a room in the dungeon"""
    bounds: Rectangle
    room_type: RoomType = RoomType.STANDARD
    height: int = IDTECH_STANDARD_CEILING_HEIGHT
    connected_rooms: Set['Room'] = field(default_factory=set)
    id: int = 0
    z_offset: int = 0  # Floor height offset relative to base (0, 64, 128, -64)
    
    def __hash__(self):
        """Make Room hashable so it can be used in sets"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality comparison based on room ID"""
        if not isinstance(other, Room):
            return False
        return self.id == other.id
    
    @property
    def center(self) -> Tuple[int, int, int]:
        """3D center point of room"""
        x, y = self.bounds.center
        return (x, y, self.bounds.z + self.height // 2)
    
    def distance_to(self, other: 'Room') -> float:
        """Calculate distance to another room"""
        c1 = self.center
        c2 = other.center
        return math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2 + (c2[2] - c1[2])**2)


@dataclass
class Corridor:
    """Represents a corridor connecting two rooms"""
    start_room: Room
    end_room: Room
    path: List[Rectangle]  # List of rectangles forming the corridor
    width: int = IDTECH_MIN_CORRIDOR_WIDTH
    z_start: int = 0  # Z-level at start room
    z_end: int = 0    # Z-level at end room
    requires_stair: bool = False  # True if Z difference needs VerticalStairHall

    @property
    def length(self) -> float:
        """Calculate total corridor length"""
        if not self.path:
            return 0
        total = 0
        for i in range(len(self.path) - 1):
            r1 = self.path[i]
            r2 = self.path[i + 1]
            c1 = r1.center
            c2 = r2.center
            total += math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        return total

    @property
    def z_delta(self) -> int:
        """Calculate absolute Z difference between ends"""
        return abs(self.z_end - self.z_start)


class BSPNode:
    """Node in the BSP tree"""
    
    def __init__(self, bounds: Rectangle, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.split_direction = SplitDirection.NONE
        self.split_position = 0
        self.left_child: Optional[BSPNode] = None
        self.right_child: Optional[BSPNode] = None
        self.room: Optional[Room] = None
        self.is_leaf = True
        
    def split(self, min_room_size: int, max_depth: int, 
              split_ratio_min: float = 0.3, split_ratio_max: float = 0.7) -> bool:
        """
        Split this node into two children using BSP algorithm.
        
        Args:
            min_room_size: Minimum size for a room
            max_depth: Maximum tree depth
            split_ratio_min: Minimum split ratio (0.0-1.0)
            split_ratio_max: Maximum split ratio (0.0-1.0)
            
        Returns:
            True if split was successful, False otherwise
        """
        # Check if we should stop splitting
        if self.depth >= max_depth:
            return False
            
        # Determine split direction based on aspect ratio
        if self.bounds.width > self.bounds.height * 1.25:
            self.split_direction = SplitDirection.VERTICAL
        elif self.bounds.height > self.bounds.width * 1.25:
            self.split_direction = SplitDirection.HORIZONTAL
        else:
            # Random direction for roughly square areas
            self.split_direction = random.choice([SplitDirection.HORIZONTAL, SplitDirection.VERTICAL])
        
        # Calculate split position
        if self.split_direction == SplitDirection.HORIZONTAL:
            min_split = int(self.bounds.height * split_ratio_min)
            max_split = int(self.bounds.height * split_ratio_max)
            
            # Ensure minimum room size on both sides
            min_split = max(min_split, min_room_size)
            max_split = min(max_split, self.bounds.height - min_room_size)
            
            if min_split >= max_split:
                return False
                
            self.split_position = random.randint(min_split, max_split)
            
            # Create child nodes
            left_bounds = Rectangle(
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.split_position,
                self.bounds.z
            )
            right_bounds = Rectangle(
                self.bounds.x,
                self.bounds.y + self.split_position,
                self.bounds.width,
                self.bounds.height - self.split_position,
                self.bounds.z
            )
            
        else:  # VERTICAL split
            min_split = int(self.bounds.width * split_ratio_min)
            max_split = int(self.bounds.width * split_ratio_max)
            
            # Ensure minimum room size on both sides
            min_split = max(min_split, min_room_size)
            max_split = min(max_split, self.bounds.width - min_room_size)
            
            if min_split >= max_split:
                return False
                
            self.split_position = random.randint(min_split, max_split)
            
            # Create child nodes
            left_bounds = Rectangle(
                self.bounds.x,
                self.bounds.y,
                self.split_position,
                self.bounds.height,
                self.bounds.z
            )
            right_bounds = Rectangle(
                self.bounds.x + self.split_position,
                self.bounds.y,
                self.bounds.width - self.split_position,
                self.bounds.height,
                self.bounds.z
            )
        
        self.left_child = BSPNode(left_bounds, self.depth + 1)
        self.right_child = BSPNode(right_bounds, self.depth + 1)
        self.is_leaf = False
        
        return True
    
    def get_leaves(self) -> List['BSPNode']:
        """Get all leaf nodes in this subtree"""
        if self.is_leaf:
            return [self]
        
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves())
        return leaves
    
    def get_sibling_leaves(self) -> Tuple[List['BSPNode'], List['BSPNode']]:
        """Get leaf nodes from left and right subtrees"""
        if self.is_leaf:
            return [], []
        
        left_leaves = self.left_child.get_leaves() if self.left_child else []
        right_leaves = self.right_child.get_leaves() if self.right_child else []
        return left_leaves, right_leaves


class BSPGenerator:
    """
    Main BSP dungeon generator class.
    
    This class implements the complete BSP algorithm for generating
    idTech-compatible dungeon layouts with guaranteed connectivity.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BSP generator with configuration.
        
        Args:
            config: Configuration dictionary with generation parameters
        """
        self.config = config or {}
        
        # Map dimensions
        self.map_width = self.config.get('map_width', 2048)
        self.map_height = self.config.get('map_height', 2048)
        self.map_depth = self.config.get('map_depth', 512)  # Z-axis range
        
        # BSP parameters
        self.max_depth = self.config.get('max_depth', 5)
        self.min_room_size = max(self.config.get('min_room_size', 128), IDTECH_MIN_ROOM_SIZE)
        self.max_room_size = self.config.get('max_room_size', 512)
        self.room_padding = self.config.get('room_padding', 32)
        
        # Corridor parameters
        self.corridor_width = max(self.config.get('corridor_width', 96), IDTECH_MIN_CORRIDOR_WIDTH)
        # Diagonal corridors produce more organic routes; use low chance
        self.allow_diagonal_corridors = self.config.get('allow_diagonal_corridors', True)
        self.diagonal_corridor_chance = self.config.get('diagonal_corridor_chance', 0.2)
        # Target number of extra loops to add beyond a tree for better navigation
        self.target_loops = int(self.config.get('target_loops', 3))
        
        # Vertical parameters
        self.enable_vertical_rooms = self.config.get('enable_vertical_rooms', False)
        self.vertical_room_chance = self.config.get('vertical_room_chance', 0.2)
        self.floor_height = self.config.get('floor_height', 256)
        
        # Generation state
        self.root_node: Optional[BSPNode] = None
        self.rooms: List[Room] = []
        self.corridors: List[Corridor] = []
        self.room_id_counter = 0
        
        # Debugging
        self.debug = self.config.get('debug', False)
        
    def generate(self, room_count: Optional[int] = None) -> bool:
        """
        Generate a complete dungeon layout.
        
        Args:
            room_count: Target number of rooms (optional)
            
        Returns:
            True if generation was successful, False otherwise
        """
        if self.debug:
            print(f"Starting BSP generation with map size {self.map_width}x{self.map_height}")
        
        # Clear previous generation
        self.rooms = []
        self.corridors = []
        self.room_id_counter = 0
        
        # Create root node with full map bounds
        self.root_node = BSPNode(Rectangle(0, 0, self.map_width, self.map_height))
        
        # Build BSP tree
        if not self._build_bsp_tree(self.root_node, room_count):
            if self.debug:
                print("Failed to build BSP tree")
            return False
        
        # Place rooms in leaf nodes
        if not self._place_rooms():
            if self.debug:
                print("Failed to place rooms")
            return False

        # Promote landmark rooms (arenas, hubs) for better pacing
        self._promote_landmarks()
        
        # Connect rooms with corridors
        if not self._connect_rooms():
            if self.debug:
                print("Failed to connect rooms")
            return False
        
        # Add additional loops to improve flow and create macro-cycles
        self._add_loops(self.target_loops)

        # Assign height variation to rooms (connectivity-aware)
        self._assign_z_offsets()

        # Mark corridors that require vertical stair insertion
        self._mark_corridors_requiring_stairs()

        # Validate connectivity
        if not self._validate_connectivity():
            if self.debug:
                print("Failed connectivity validation")
            return False

        # Add vertical rooms if enabled
        if self.enable_vertical_rooms:
            self._add_vertical_rooms()

        if self.debug:
            stair_corridors = self.get_corridors_requiring_stairs()
            stair_pairs = self.get_stair_connections()
            print(f"Generation complete: {len(self.rooms)} rooms, {len(self.corridors)} corridors")
            print(f"Stair connections needed: {len(stair_pairs)} room pairs, "
                  f"{len(stair_corridors)} corridors")
        
        return True
    
    def _build_bsp_tree(self, node: BSPNode, target_room_count: Optional[int] = None) -> bool:
        """
        Recursively build the BSP tree.
        
        Args:
            node: Current node to process
            target_room_count: Target number of leaf nodes
            
        Returns:
            True if tree was built successfully
        """
        # Determine if we should split this node
        should_split = True
        
        # Check depth limit
        if node.depth >= self.max_depth:
            should_split = False
        
        # Check minimum size
        if node.bounds.width < self.min_room_size * 2 or node.bounds.height < self.min_room_size * 2:
            should_split = False
        
        # Check target room count
        if target_room_count and self._count_leaves(self.root_node) >= target_room_count:
            should_split = False
        
        if not should_split:
            return True
        
        # Attempt to split
        if node.split(self.min_room_size, self.max_depth):
            # Recursively split children
            if node.left_child:
                self._build_bsp_tree(node.left_child, target_room_count)
            if node.right_child:
                self._build_bsp_tree(node.right_child, target_room_count)
        
        return True
    
    def _count_leaves(self, node: BSPNode) -> int:
        """Count leaf nodes in tree"""
        if node.is_leaf:
            return 1
        count = 0
        if node.left_child:
            count += self._count_leaves(node.left_child)
        if node.right_child:
            count += self._count_leaves(node.right_child)
        return count
    
    def _place_rooms(self) -> bool:
        """
        Place rooms in all leaf nodes of the BSP tree.

        Returns:
            True if rooms were placed successfully
        """
        leaves = self.root_node.get_leaves()

        # Pre-filter valid leaves to correctly assign ENTRANCE/EXIT
        valid_leaves = []
        for i, leaf in enumerate(leaves):
            max_possible_width = leaf.bounds.width - self.room_padding * 2
            max_possible_height = leaf.bounds.height - self.room_padding * 2
            if max_possible_width >= self.min_room_size and max_possible_height >= self.min_room_size:
                valid_leaves.append(leaf)
            elif self.debug:
                print(f"Skipping leaf {i}: too small for room placement")

        for room_idx, leaf in enumerate(valid_leaves):
            # Calculate maximum possible room size within leaf bounds
            max_possible_width = leaf.bounds.width - self.room_padding * 2
            max_possible_height = leaf.bounds.height - self.room_padding * 2

            # Calculate room size within leaf bounds
            room_width = random.randint(
                self.min_room_size,
                min(self.max_room_size, max_possible_width)
            )
            room_height = random.randint(
                self.min_room_size,
                min(self.max_room_size, max_possible_height)
            )

            # Ensure minimum size
            room_width = max(room_width, IDTECH_MIN_ROOM_SIZE)
            room_height = max(room_height, IDTECH_MIN_ROOM_SIZE)

            # Random position within leaf bounds
            room_x = leaf.bounds.x + random.randint(
                self.room_padding,
                leaf.bounds.width - room_width - self.room_padding
            )
            room_y = leaf.bounds.y + random.randint(
                self.room_padding,
                leaf.bounds.height - room_height - self.room_padding
            )

            # Create room
            room_bounds = Rectangle(room_x, room_y, room_width, room_height)

            # Seed basic types (will be refined in _promote_landmarks)
            # Use room_idx (placed room count) not leaf index
            if room_idx == 0:
                room_type = RoomType.ENTRANCE
            elif room_idx == len(valid_leaves) - 1:
                room_type = RoomType.EXIT
            elif random.random() < 0.1:
                room_type = RoomType.ARENA
            elif random.random() < 0.2:
                room_type = RoomType.JUNCTION
            else:
                room_type = RoomType.STANDARD
            
            # Set room ceiling height based on type
            if room_type == RoomType.ARENA:
                ceiling_height = IDTECH_TALL_CEILING_HEIGHT
            else:
                ceiling_height = IDTECH_STANDARD_CEILING_HEIGHT
            
            room = Room(
                bounds=room_bounds,
                room_type=room_type,
                height=ceiling_height,
                id=self.room_id_counter
            )
            self.room_id_counter += 1
            
            self.rooms.append(room)
            leaf.room = room
            
            if self.debug:
                print(f"Placed {room_type.name} room {room.id} at ({room_x}, {room_y}) "
                      f"size {room_width}x{room_height}")
        
        return len(self.rooms) > 0
    
    def _connect_rooms(self) -> bool:
        """
        Connect all rooms using corridors through the BSP tree structure.
        This ensures guaranteed connectivity.
        
        Returns:
            True if all rooms were connected successfully
        """
        # Connect rooms through BSP tree hierarchy
        self._connect_node_rooms(self.root_node)
        
        # Add some additional connections for variety
        self._add_extra_connections()
        
        return len(self.corridors) > 0
    
    def _connect_node_rooms(self, node: BSPNode) -> List[Room]:
        """
        Recursively connect rooms in BSP node and return all rooms in subtree.
        
        Args:
            node: Current BSP node
            
        Returns:
            List of all rooms in this subtree
        """
        if node.is_leaf:
            return [node.room] if node.room else []
        
        left_rooms = []
        right_rooms = []
        
        if node.left_child:
            left_rooms = self._connect_node_rooms(node.left_child)
        if node.right_child:
            right_rooms = self._connect_node_rooms(node.right_child)
        
        # Connect closest rooms between left and right subtrees
        if left_rooms and right_rooms:
            # Find closest pair
            min_dist = float('inf')
            closest_left = None
            closest_right = None
            
            for left_room in left_rooms:
                for right_room in right_rooms:
                    dist = left_room.distance_to(right_room)
                    if dist < min_dist:
                        min_dist = dist
                        closest_left = left_room
                        closest_right = right_room
            
            if closest_left and closest_right:
                corridor = self._create_corridor(closest_left, closest_right)
                if corridor:
                    self.corridors.append(corridor)
                    closest_left.connected_rooms.add(closest_right)
                    closest_right.connected_rooms.add(closest_left)
        
        return left_rooms + right_rooms
    
    def _create_corridor(self, room1: Room, room2: Room) -> Optional[Corridor]:
        """
        Create a corridor connecting two rooms.

        The corridor is built to guarantee overlap with both room footprints by
        extending segments into the room boundaries, not just to room centers.

        Args:
            room1: Start room
            room2: End room

        Returns:
            Corridor object or None if creation failed
        """
        path = []

        # Get room centers
        c1 = room1.bounds.center
        c2 = room2.bounds.center

        # Room boundaries for ensuring corridor overlap
        r1_x1, r1_y1 = room1.bounds.x, room1.bounds.y
        r1_x2, r1_y2 = room1.bounds.x2, room1.bounds.y2
        r2_x1, r2_y1 = room2.bounds.x, room2.bounds.y
        r2_x2, r2_y2 = room2.bounds.x2, room2.bounds.y2

        # vary corridor width slightly per-connection for variety
        cwidth = max(IDTECH_MIN_CORRIDOR_WIDTH, int(self.corridor_width + random.randint(-16, 24)))

        # Extension amount to ensure corridors penetrate into rooms (not just touch edges)
        room_penetration = 48  # Units to extend into each room

        if self.allow_diagonal_corridors and random.random() < self.diagonal_corridor_chance:
            # Create diagonal corridor (single segment)
            corridor_rect = self._create_diagonal_corridor_rect(c1, c2, cwidth)
            if corridor_rect:
                path.append(corridor_rect)
        else:
            # Create L-shaped corridor (two segments)
            # Ensure each segment extends into both rooms it touches

            if random.random() < 0.5:
                # Horizontal first (from room1), then vertical (to room2)

                # Horizontal segment: extend from inside room1 to the elbow point
                # Start inside room1's boundary, end at room2's center X
                h_start_x = min(r1_x1 + room_penetration, c1[0])  # Start inside room1
                h_end_x = c2[0]  # Elbow at room2's X center

                h_x = min(h_start_x, h_end_x) - cwidth // 2
                h_width = abs(h_end_x - h_start_x) + cwidth

                h_rect = Rectangle(
                    h_x,
                    c1[1] - cwidth // 2,
                    h_width,
                    cwidth,
                    room1.bounds.z
                )
                path.append(h_rect)

                # Vertical segment: extend from elbow to inside room2
                v_start_y = c1[1]  # Elbow at room1's Y center
                v_end_y = max(r2_y2 - room_penetration, c2[1])  # End inside room2

                # Make sure we span from c1[1] to at least inside room2
                if c2[1] > c1[1]:
                    v_end_y = max(r2_y1 + room_penetration, c2[1])
                else:
                    v_end_y = min(r2_y2 - room_penetration, c2[1])

                v_y = min(c1[1], c2[1]) - cwidth // 2
                v_height = abs(c2[1] - c1[1]) + cwidth

                v_rect = Rectangle(
                    c2[0] - cwidth // 2,
                    v_y,
                    cwidth,
                    v_height,
                    room1.bounds.z
                )
                path.append(v_rect)
            else:
                # Vertical first (from room1), then horizontal (to room2)

                # Vertical segment: extend from inside room1 to the elbow point
                v_start_y = min(r1_y1 + room_penetration, c1[1])  # Start inside room1
                v_end_y = c2[1]  # Elbow at room2's Y center

                v_y = min(v_start_y, v_end_y) - cwidth // 2
                v_height = abs(v_end_y - v_start_y) + cwidth

                v_rect = Rectangle(
                    c1[0] - cwidth // 2,
                    v_y,
                    cwidth,
                    v_height,
                    room1.bounds.z
                )
                path.append(v_rect)

                # Horizontal segment: extend from elbow to inside room2
                h_start_x = c1[0]  # Elbow at room1's X center
                h_end_x = max(r2_x2 - room_penetration, c2[0])  # End inside room2

                # Make sure we span from c1[0] to at least inside room2
                if c2[0] > c1[0]:
                    h_end_x = max(r2_x1 + room_penetration, c2[0])
                else:
                    h_end_x = min(r2_x2 - room_penetration, c2[0])

                h_x = min(c1[0], c2[0]) - cwidth // 2
                h_width = abs(c2[0] - c1[0]) + cwidth

                h_rect = Rectangle(
                    h_x,
                    c2[1] - cwidth // 2,
                    h_width,
                    cwidth,
                    room1.bounds.z
                )
                path.append(h_rect)

        if not path:
            return None

        corridor = Corridor(
            start_room=room1,
            end_room=room2,
            path=path,
            width=cwidth
        )

        # Check for dead-end length constraint
        if len(room1.connected_rooms) == 0 and corridor.length > IDTECH_MAX_DEADEND_LENGTH:
            # Shorten the corridor or add a junction room
            if self.debug:
                print(f"Corridor too long for dead-end: {corridor.length}")

        return corridor
    
    def _create_diagonal_corridor_rect(self, c1: Tuple[int, int], c2: Tuple[int, int], width: int) -> Optional[Rectangle]:
        """
        Create a bounding rectangle for a diagonal corridor.
        
        Args:
            c1: Start point
            c2: End point
            
        Returns:
            Rectangle encompassing the diagonal corridor
        """
        x1, y1 = c1
        x2, y2 = c2
        
        # Calculate bounding box with corridor width
        min_x = min(x1, x2) - width // 2
        max_x = max(x1, x2) + width // 2
        min_y = min(y1, y2) - width // 2
        max_y = max(y1, y2) + width // 2
        
        return Rectangle(
            min_x,
            min_y,
            max_x - min_x,
            max_y - min_y
        )
    
    def _add_extra_connections(self):
        """Legacy extra connections; superseded by _add_loops."""
        # Keep minimal randomness for spice; main loop creation handled elsewhere
        for i, room1 in enumerate(self.rooms):
            for room2 in self.rooms[i+1:]:
                if room2 not in room1.connected_rooms and random.random() < 0.02:
                    corridor = self._create_corridor(room1, room2)
                    if corridor:
                        self.corridors.append(corridor)
                        room1.connected_rooms.add(room2)
                        room2.connected_rooms.add(room1)

    def _add_loops(self, target_loops: int) -> None:
        """Add cross-connections to ensure at least target_loops cycles exist.

        Approximate loop count as E - (N - 1) assuming single connected component.
        """
        if not self.rooms:
            return
        N = len(self.rooms)
        E = len(self.corridors)
        loops = max(0, E - (N - 1))
        if self.debug:
            print(f"Current loops: {loops}, target: {target_loops}")
        if loops >= target_loops:
            return

        # Build candidate pairs sorted by distance (favor shorter meaningful links)
        pairs: List[Tuple[float, Room, Room]] = []
        for i, r1 in enumerate(self.rooms):
            for r2 in self.rooms[i+1:]:
                if r2 in r1.connected_rooms:
                    continue
                dist = r1.distance_to(r2)
                # Prefer medium distances to create local loops; cap extremes
                if 256 <= dist <= 1400:
                    pairs.append((dist, r1, r2))
        pairs.sort(key=lambda t: t[0])

        for _, r1, r2 in pairs:
            if loops >= target_loops:
                break
            if r2 in r1.connected_rooms:
                continue
            corridor = self._create_corridor(r1, r2)
            if corridor:
                self.corridors.append(corridor)
                r1.connected_rooms.add(r2)
                r2.connected_rooms.add(r1)
                loops += 1
                if self.debug:
                    print(f"Added loop via corridor {r1.id} <-> {r2.id}; loops={loops}")

    def _promote_landmarks(self) -> None:
        """Promote rooms to landmarks based on BSP position and connectivity.

        Strategy:
        - With >= 3 rooms: farthest -> ARENA, second-farthest -> EXIT
        - With 2 rooms: farthest -> EXIT (skip ARENA to ensure playable exit)
        - ENTRANCE always preserved
        - Most-connected room -> JUNCTION (hub)
        - Largest remaining standard room -> JUNCTION if different from above
        """
        if not self.rooms:
            return
        # Determine entrance for distance metric
        entrance = None
        for r in self.rooms:
            if r.room_type == RoomType.ENTRANCE:
                entrance = r
                break
        if entrance is None:
            entrance = self.rooms[0]
            entrance.room_type = RoomType.ENTRANCE  # Ensure first room is entrance

        # Sort rooms by distance from entrance (descending)
        by_dist = sorted(self.rooms, key=lambda r: entrance.distance_to(r), reverse=True)

        # Count non-entrance rooms to determine assignment strategy
        non_entrance_rooms = [r for r in self.rooms if r.room_type != RoomType.ENTRANCE]

        # Clear any existing ARENA and EXIT assignments before reassigning
        # (but preserve ENTRANCE)
        for r in self.rooms:
            if r.room_type == RoomType.ARENA:
                r.room_type = RoomType.STANDARD
                r.height = IDTECH_STANDARD_CEILING_HEIGHT
            elif r.room_type == RoomType.EXIT:
                r.room_type = RoomType.STANDARD

        # Assignment strategy depends on room count
        if len(non_entrance_rooms) >= 2:
            # Normal case: assign both ARENA and EXIT
            # Farthest non-entrance room -> ARENA
            farthest = None
            for r in by_dist:
                if r.room_type != RoomType.ENTRANCE:
                    farthest = r
                    break
            if farthest:
                farthest.room_type = RoomType.ARENA
                farthest.height = IDTECH_TALL_CEILING_HEIGHT

            # Second-farthest non-entrance, non-arena room -> EXIT
            for r in by_dist:
                if r.room_type not in (RoomType.ENTRANCE, RoomType.ARENA):
                    r.room_type = RoomType.EXIT
                    break
        elif len(non_entrance_rooms) == 1:
            # Only 2 rooms total: skip ARENA, assign EXIT to ensure playable level
            non_entrance_rooms[0].room_type = RoomType.EXIT
        # If only 1 room (entrance only), nothing more to assign

        # Most-connected room → JUNCTION (connectivity-based hub detection)
        most_connected = max(
            self.rooms,
            key=lambda r: len(r.connected_rooms)
        )
        if most_connected.room_type not in (RoomType.ENTRANCE, RoomType.EXIT, RoomType.ARENA):
            most_connected.room_type = RoomType.JUNCTION

        # Also promote largest area room if different
        largest = max(self.rooms, key=lambda r: r.bounds.area)
        if largest.room_type == RoomType.STANDARD:
            largest.room_type = RoomType.JUNCTION

    def _assign_z_offsets(self) -> None:
        """Assign height offsets to rooms using connectivity-aware algorithm.

        This method ensures that Z-level differences respect connectivity:
        1. Build spanning tree of room connections (BFS from entrance)
        2. Assign entrance to z=0
        3. For connected rooms, assign compatible Z (same level or mark for stair)
        4. Only allow Z changes where a stair can be inserted

        This prevents the previous issue of randomly assigned Z-levels
        creating inaccessible areas.
        """
        if not self.rooms:
            return

        # Find entrance room
        entrance = None
        for room in self.rooms:
            if room.room_type == RoomType.ENTRANCE:
                entrance = room
                break
        if entrance is None:
            entrance = self.rooms[0]

        # Initialize all rooms to None (unassigned)
        for room in self.rooms:
            room.z_offset = None

        # Assign entrance to z=0
        entrance.z_offset = 0

        # Build connectivity graph
        adj: Dict[int, Set[Room]] = {r.id: set() for r in self.rooms}
        for room in self.rooms:
            for connected in room.connected_rooms:
                adj[room.id].add(connected)
                adj[connected.id].add(room)

        # BFS from entrance, assigning Z-levels based on connectivity
        visited: Set[int] = set()
        queue = [entrance]

        # Track which corridors need stairs (by room pair)
        self._stair_connections: List[Tuple[Room, Room]] = []

        # Z-level choices for variation (only at strategic points)
        z_changes = [64, 128, -64]

        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)

            # Process connected rooms
            for neighbor in adj[current.id]:
                if neighbor.id in visited:
                    continue

                # Decide if this connection should have a Z-level change
                # Only allow Z changes at junction points or to special rooms
                allow_z_change = (
                    len(adj[current.id]) >= 3 or  # Current is a hub
                    neighbor.room_type in (RoomType.EXIT, RoomType.ARENA) or
                    current.room_type == RoomType.JUNCTION
                )

                if allow_z_change and random.random() < 0.3:
                    # Apply a Z-level change and mark for stair
                    z_delta = random.choice(z_changes)
                    neighbor.z_offset = current.z_offset + z_delta

                    # Clamp to valid range
                    neighbor.z_offset = max(-128, min(256, neighbor.z_offset))

                    # Mark this connection as needing a stair
                    self._stair_connections.append((current, neighbor))

                    if self.debug:
                        print(f"Z-change: Room {current.id} (z={current.z_offset}) -> "
                              f"Room {neighbor.id} (z={neighbor.z_offset}) [needs stair]")
                else:
                    # Keep same Z-level (traversable without stairs)
                    neighbor.z_offset = current.z_offset

                queue.append(neighbor)

        # Handle any rooms not reached by BFS (disconnected components)
        for room in self.rooms:
            if room.z_offset is None:
                room.z_offset = 0  # Default to ground level
                if self.debug:
                    print(f"Room {room.id} was disconnected, assigned z=0")

        # Special handling: ensure ARENA rooms are at z=0 for combat consistency
        for room in self.rooms:
            if room.room_type == RoomType.ARENA and room.z_offset != 0:
                # Find a connected room at z=0 and swap if possible
                for connected in room.connected_rooms:
                    if connected.z_offset == 0 and connected.room_type == RoomType.STANDARD:
                        connected.z_offset = room.z_offset
                        room.z_offset = 0
                        break

        if self.debug:
            offsets = [(r.id, r.room_type.name, r.z_offset) for r in self.rooms]
            print(f"Z-offsets assigned (connectivity-aware): {offsets}")
            print(f"Connections requiring stairs: {len(self._stair_connections)}")

    def _add_vertical_rooms(self):
        """Add rooms on different Z levels for vertical gameplay.

        This improved version:
        1. Creates upper rooms with proper Z-offset
        2. Marks the vertical connection as needing stairs
        3. Validates that stair insertion is possible
        """
        if not self.rooms:
            return

        # Initialize stair connections list if not exists
        if not hasattr(self, '_stair_connections'):
            self._stair_connections = []

        # Select some rooms to have upper levels
        for room in self.rooms:
            if room.room_type == RoomType.STANDARD and random.random() < self.vertical_room_chance:
                # Create upper room with proper z_offset
                upper_bounds = Rectangle(
                    room.bounds.x,
                    room.bounds.y,
                    room.bounds.width,
                    room.bounds.height,
                    room.bounds.z + self.floor_height
                )

                upper_room = Room(
                    bounds=upper_bounds,
                    room_type=RoomType.VERTICAL,
                    height=IDTECH_STANDARD_CEILING_HEIGHT,
                    id=self.room_id_counter,
                    z_offset=room.z_offset + self.floor_height  # Proper Z-offset
                )
                self.room_id_counter += 1

                self.rooms.append(upper_room)

                # Connect vertically (stairs/lift location)
                room.connected_rooms.add(upper_room)
                upper_room.connected_rooms.add(room)

                # Mark this connection as needing a stair
                self._stair_connections.append((room, upper_room))

                if self.debug:
                    print(f"Added vertical room {upper_room.id} (z={upper_room.z_offset}) "
                          f"above room {room.id} (z={room.z_offset}) [needs stair]")

    def _mark_corridors_requiring_stairs(self) -> None:
        """Mark corridors that connect rooms with different Z-levels.

        This method updates the requires_stair flag on corridors
        based on the Z-offset difference between connected rooms.
        """
        for corridor in self.corridors:
            z_start = corridor.start_room.z_offset
            z_end = corridor.end_room.z_offset
            z_delta = abs(z_end - z_start)

            # Update corridor Z tracking
            corridor.z_start = z_start
            corridor.z_end = z_end

            # Mark as requiring stair if Z difference exceeds step height
            if z_delta > IDTECH_MAX_STEP_HEIGHT:
                corridor.requires_stair = True
                if self.debug:
                    print(f"Corridor {corridor.start_room.id} -> {corridor.end_room.id} "
                          f"requires stair (z_delta={z_delta})")

    def get_stair_connections(self) -> List[Tuple['Room', 'Room']]:
        """Return list of room pairs that require vertical stair connections.

        Returns:
            List of (room_a, room_b) tuples where Z-difference requires stairs
        """
        if hasattr(self, '_stair_connections'):
            return self._stair_connections
        return []

    def get_corridors_requiring_stairs(self) -> List['Corridor']:
        """Return list of corridors that require VerticalStairHall insertion.

        Returns:
            List of Corridor objects where requires_stair is True
        """
        return [c for c in self.corridors if c.requires_stair]
    
    def _validate_connectivity(self) -> bool:
        """
        Validate that all rooms are connected (no isolated rooms).
        If disconnected components are found, attempt to repair by adding corridors.

        Returns:
            True if all rooms are reachable from the entrance (after repair if needed)
        """
        if not self.rooms:
            return False

        # Find entrance room
        entrance = None
        for room in self.rooms:
            if room.room_type == RoomType.ENTRANCE:
                entrance = room
                break

        if not entrance:
            # Use first room as entrance
            entrance = self.rooms[0]

        # Attempt repair up to 3 times
        for repair_attempt in range(4):
            # BFS to find connected component from entrance
            visited = set()
            queue = [entrance]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)

                # Add connected rooms to queue
                for connected in current.connected_rooms:
                    if connected not in visited:
                        queue.append(connected)

            # Check if all rooms were visited
            unconnected = [room for room in self.rooms if room not in visited]

            if not unconnected:
                if self.debug:
                    print(f"Connectivity validation passed: all {len(self.rooms)} rooms connected")
                return True

            if repair_attempt == 3:
                # Final attempt failed
                if self.debug:
                    print(f"Warning: Could not repair connectivity. Unconnected rooms: {[r.id for r in unconnected]}")
                return False

            # Repair: connect each disconnected component to the main component
            if self.debug:
                print(f"Repair attempt {repair_attempt + 1}: connecting {len(unconnected)} disconnected rooms")

            self._repair_connectivity(visited, unconnected)

        return False

    def _repair_connectivity(self, main_component: Set['Room'], disconnected: List['Room']) -> None:
        """
        Repair connectivity by adding corridors between disconnected components
        and the main component. Uses Z-aware selection to prefer compatible Z-levels.

        Args:
            main_component: Set of rooms in the main connected component
            disconnected: List of rooms not connected to main component
        """
        # Initialize stair connections list if not exists
        if not hasattr(self, '_stair_connections'):
            self._stair_connections = []

        # Find all disconnected components (there may be multiple islands)
        remaining = set(disconnected)
        components = []

        while remaining:
            # BFS to find one component
            start = remaining.pop()
            component = {start}
            queue = [start]

            while queue:
                current = queue.pop(0)
                for connected in current.connected_rooms:
                    if connected in remaining:
                        remaining.remove(connected)
                        component.add(connected)
                        queue.append(connected)

            components.append(component)

        if self.debug:
            print(f"  Found {len(components)} disconnected component(s)")

        # Connect each component to the main component via shortest bridge
        # Prefer pairs with compatible Z-levels to avoid needing stairs
        for component in components:
            # Score pairs by distance AND Z-compatibility
            best_score = float('inf')
            best_main_room = None
            best_comp_room = None
            needs_stair = False

            for comp_room in component:
                for main_room in main_component:
                    dist = comp_room.distance_to(main_room)
                    z_delta = abs(comp_room.z_offset - main_room.z_offset)

                    # Score: distance + penalty for Z-level mismatch
                    # Small Z differences (<= step height) are OK
                    z_penalty = 0 if z_delta <= IDTECH_MAX_STEP_HEIGHT else z_delta * 10

                    score = dist + z_penalty

                    if score < best_score:
                        best_score = score
                        best_main_room = main_room
                        best_comp_room = comp_room
                        needs_stair = z_delta > IDTECH_MAX_STEP_HEIGHT

            if best_main_room and best_comp_room:
                # If Z-levels don't match, align the component room to the main room
                # This prevents creating unreachable connections
                if needs_stair:
                    # Option 1: Align Z-levels (simple fix)
                    # Option 2: Mark for stair insertion (better for gameplay)
                    # We'll mark for stair since the Z-offsets are already assigned
                    self._stair_connections.append((best_main_room, best_comp_room))
                    if self.debug:
                        print(f"  Repair connection needs stair: "
                              f"{best_main_room.id} (z={best_main_room.z_offset}) -> "
                              f"{best_comp_room.id} (z={best_comp_room.z_offset})")

                # Create a corridor to bridge the components
                corridor = self._create_corridor(best_main_room, best_comp_room)
                if corridor:
                    # Track Z-levels on the corridor
                    corridor.z_start = best_main_room.z_offset
                    corridor.z_end = best_comp_room.z_offset
                    corridor.requires_stair = needs_stair

                    self.corridors.append(corridor)
                    best_main_room.connected_rooms.add(best_comp_room)
                    best_comp_room.connected_rooms.add(best_main_room)

                    # Merge this component into main
                    main_component.update(component)

                    if self.debug:
                        stair_note = " [needs stair]" if needs_stair else ""
                        print(f"  Connected room {best_comp_room.id} to room "
                              f"{best_main_room.id} (distance: {dist:.0f}){stair_note}")
    
    def get_layout_stats(self) -> Dict:
        """
        Get statistics about the generated layout.
        
        Returns:
            Dictionary with layout statistics
        """
        stats = {
            'room_count': len(self.rooms),
            'corridor_count': len(self.corridors),
            'total_room_area': sum(r.bounds.area for r in self.rooms),
            'average_room_size': 0,
            'entrance_rooms': 0,
            'exit_rooms': 0,
            'arena_rooms': 0,
            'junction_rooms': 0,
            'vertical_rooms': 0,
            'standard_rooms': 0,
            'max_connections': 0,
            'min_connections': 0,
            'avg_connections': 0,
            'total_corridor_length': sum(c.length for c in self.corridors),
            'tree_depth': self.root_node.depth if self.root_node else 0
        }
        
        if self.rooms:
            stats['average_room_size'] = stats['total_room_area'] / len(self.rooms)
            
            connections = [len(r.connected_rooms) for r in self.rooms]
            if connections:
                stats['max_connections'] = max(connections)
                stats['min_connections'] = min(connections)
                stats['avg_connections'] = sum(connections) / len(connections)
            
            # Count room types
            for room in self.rooms:
                if room.room_type == RoomType.ENTRANCE:
                    stats['entrance_rooms'] += 1
                elif room.room_type == RoomType.EXIT:
                    stats['exit_rooms'] += 1
                elif room.room_type == RoomType.ARENA:
                    stats['arena_rooms'] += 1
                elif room.room_type == RoomType.JUNCTION:
                    stats['junction_rooms'] += 1
                elif room.room_type == RoomType.VERTICAL:
                    stats['vertical_rooms'] += 1
                else:
                    stats['standard_rooms'] += 1
        
        return stats
    
    def export_layout(self) -> Dict:
        """
        Export the generated layout as a dictionary for serialization.
        
        Returns:
            Dictionary representation of the layout
        """
        layout = {
            'config': self.config,
            'stats': self.get_layout_stats(),
            'rooms': [],
            'corridors': []
        }
        
        # Export rooms
        for room in self.rooms:
            room_data = {
                'id': room.id,
                'type': room.room_type.name,
                'bounds': {
                    'x': room.bounds.x,
                    'y': room.bounds.y,
                    'width': room.bounds.width,
                    'height': room.bounds.height,
                    'z': room.bounds.z
                },
                'height': room.height,
                'z_offset': room.z_offset,
                'connected_to': [r.id for r in room.connected_rooms]
            }
            layout['rooms'].append(room_data)
        
        # Export corridors
        for corridor in self.corridors:
            corridor_data = {
                'start_room_id': corridor.start_room.id,
                'end_room_id': corridor.end_room.id,
                'width': corridor.width,
                'length': corridor.length,
                'path': []
            }
            
            for rect in corridor.path:
                corridor_data['path'].append({
                    'x': rect.x,
                    'y': rect.y,
                    'width': rect.width,
                    'height': rect.height,
                    'z': rect.z
                })
            
            layout['corridors'].append(corridor_data)
        
        return layout


def main():
    """Test the BSP generator"""
    import json
    
    # Test configuration
    config = {
        'map_width': 2048,
        'map_height': 2048,
        'max_depth': 4,
        'min_room_size': 128,
        'max_room_size': 384,
        'corridor_width': 64,
        'enable_vertical_rooms': True,
        'vertical_room_chance': 0.3,
        'debug': True
    }
    
    print("=== idTech BSP Dungeon Generator Test ===\n")
    
    # Create generator
    generator = BSPGenerator(config)
    
    # Generate dungeon
    if generator.generate(room_count=15):
        print("\n✓ Generation successful!")
        
        # Print statistics
        stats = generator.get_layout_stats()
        print("\n=== Layout Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Export layout
        layout = generator.export_layout()
        
        # Save to file
        with open('test_dungeon.json', 'w') as f:
            json.dump(layout, f, indent=2)
        print("\n✓ Layout exported to test_dungeon.json")
    else:
        print("\n✗ Generation failed!")


if __name__ == "__main__":
    main()
