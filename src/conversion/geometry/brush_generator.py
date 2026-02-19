"""
3D Brush Generation System for idTech Map Generator

This module converts 2D BSP layouts into proper 3D idTech geometry using the
"single-brush hollow rooms" approach. Each room becomes 6 separate convex brushes
(floor, ceiling, 4 walls) to ensure valid idTech map compilation.

Architecture Pattern: Builder Pattern with Strategy variations for room types
Dependencies: layout_types (2D data), map_writer (Brush/Plane structures)
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

# Import layout structures - using the modern layout_types module
from quake_levelgenerator.src.generators.layout.layout_types import (
    Layout2D, LayoutRoom, LayoutConnection, ConnectionType,
    TileType, TILE_SIZE, MultiLevelLayout
)
from quake_levelgenerator.src.conversion.map_writer import Brush, Plane
from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Named Constants - Replace magic numbers with descriptive identifiers
# =============================================================================

# Geometry thresholds
MIN_WALL_SEGMENT_SIZE = 16          # Minimum wall segment before a door opening
WALL_CONNECTION_TOLERANCE = 20      # Tolerance for detecting wall-connection alignment
CORRIDOR_FLOOR_PADDING = 32.0       # Padding for corridor floor overlap at seams
THRESHOLD_FLOOR_PAD_FACTOR = 3      # Multiplier for threshold floor padding

# Boundary wall defaults
BOUNDARY_WALL_THICKNESS = 32        # Thickness of perimeter boundary walls
BOUNDARY_WALL_MARGIN = 64           # Extra margin above room height for boundary
BOUNDARY_FLOOR_DEPTH = 96           # Depth of boundary floor below level
BOUNDARY_FLOOR_OFFSET = 32          # Offset between boundary floor and room floor

# Platform dimensions
LIFT_PLATFORM_THICKNESS = 8         # Thin lift platform height

# Detail geometry sizes
DETAIL_DOOR_HALF_WIDTH = 48         # Half-width for door frame trim
DETAIL_WALKWAY_HEIGHT = 64          # Arena raised walkway height
DETAIL_WALKWAY_DEPTH = 96           # Arena walkway depth from wall
DETAIL_RAMP_WIDTH = 64              # Arena ramp width
DETAIL_PILLAR_SIZE = 32             # Arena pillar size

# Room feature thresholds
MIN_ROOM_SIZE_FOR_DETAILS = 192     # Minimum room dimension for detail features
LARGE_ROOM_THRESHOLD = 256          # Room size threshold for buttresses/beams


class RoomHeightType(Enum):
    """Height profiles for different room types"""
    STANDARD = 128      # Normal rooms
    ARENA = 256         # Tall combat spaces
    CORRIDOR = 128      # Hallway height
    ENTRANCE = 128      # Starting areas
    EXIT = 128          # Exit areas
    VERTICAL = 384      # Multi-level connections


class TextureSet:
    """Texture assignments for different surface types.

    Reads from TEXTURE_SETTINGS for user-configurable textures (wall, floor, ceiling,
    structural, trim). Special textures remain as constants.
    """

    @property
    def WALL_DEFAULT(self) -> str:
        """Wall texture from user settings."""
        return TEXTURE_SETTINGS.get_texture("wall")

    @property
    def FLOOR_DEFAULT(self) -> str:
        """Floor texture from user settings."""
        return TEXTURE_SETTINGS.get_texture("floor")

    @property
    def CEILING_DEFAULT(self) -> str:
        """Ceiling texture from user settings."""
        return TEXTURE_SETTINGS.get_texture("ceiling")

    @property
    def STRUCTURAL_DEFAULT(self) -> str:
        """Structural texture from user settings."""
        return TEXTURE_SETTINGS.get_texture("structural")

    @property
    def TRIM_DEFAULT(self) -> str:
        """Trim texture from user settings."""
        return TEXTURE_SETTINGS.get_texture("trim")

    # Room type specific textures - use properties for dynamic lookup
    @property
    def ARENA_WALL(self) -> str:
        return self.WALL_DEFAULT

    @property
    def ARENA_FLOOR(self) -> str:
        return self.FLOOR_DEFAULT

    @property
    def ENTRANCE_WALL(self) -> str:
        return self.WALL_DEFAULT

    @property
    def ENTRANCE_FLOOR(self) -> str:
        return self.FLOOR_DEFAULT

    @property
    def CORRIDOR_WALL(self) -> str:
        return self.WALL_DEFAULT

    @property
    def CORRIDOR_FLOOR(self) -> str:
        return self.FLOOR_DEFAULT

    # Special textures - these remain as constants (not user-configurable)
    DOOR_FRAME = "DOOR02_1"
    STAIRS = "STEP1"
    LIFT_PLATFORM = "PLAT_TOP1"
    WATER = "*WATER1"
    LAVA = "*LAVA1"
    SLIME = "*SLIME0"


@dataclass
class GeometrySettings:
    """Configuration for 3D geometry generation"""
    # Dimensions
    wall_thickness: float = 8.0         # Brush thickness for walls (idTech standard)
    floor_thickness: float = 8.0        # Brush thickness for floors (idTech standard)
    ceiling_thickness: float = 8.0      # Brush thickness for ceilings (idTech standard)
    
    # Heights (Z-axis)
    base_floor_z: float = 0.0          # Base Z level for ground floor
    standard_height: float = 128.0     # Standard room height
    arena_height: float = 256.0        # Arena room height
    
    # Grid alignment
    grid_size: float = TILE_SIZE       # Units per tile (64)
    snap_to_grid: bool = True          # Force grid alignment
    
    # Corridor settings
    corridor_width: float = 128.0      # Minimum corridor width
    corridor_height: float = 128.0     # Corridor height
    
    # Door settings
    door_width: float = 64.0           # Standard door width
    door_height: float = 96.0          # Standard door height
    door_frame_thickness: float = 8.0  # Door frame depth
    
    # Stair settings
    stair_step_height: float = 16.0    # Height per step
    stair_step_depth: float = 32.0     # Depth per step
    
    # Multi-level settings
    level_separation: float = 512.0    # Z distance between levels
    
    # Optimization
    merge_adjacent_brushes: bool = True
    remove_hidden_faces: bool = False  # Not recommended for idTech
    
    # Textures
    texture_set: TextureSet = field(default_factory=TextureSet)


class BrushGenerator:
    """
    Core 3D brush generation system that converts 2D layouts to idTech geometry.
    
    Implements the Builder pattern to construct complex 3D structures from
    simple 2D room/corridor definitions. Each room type uses a strategy for
    appropriate height and texture selection.
    """
    
    def __init__(self, settings: Optional[GeometrySettings] = None):
        """Initialize the brush generator with configuration"""
        self.settings = settings or GeometrySettings()
        # Corridor-room junction overlap (used consistently for corridors and wall margins)
        self._junction_overlap: float = 96.0
        # Ensure openings are at least as wide as corridors for player clearance
        if self.settings.door_width < self.settings.corridor_width:
            self.settings.door_width = float(self.settings.corridor_width)
        self.generated_brushes: List[Brush] = []
        self.brush_id_counter = 0
        
        # Track room connections for door placement
        self.room_connections: Dict[int, Set[int]] = {}
        
        # Cache for optimization
        self.room_brush_cache: Dict[int, List[Brush]] = {}
        
        logger.info(
            "BrushGenerator initialized | grid=%d, corridor_width=%.0f, door_width=%.0f",
            self.settings.grid_size,
            self.settings.corridor_width,
            self.settings.door_width,
        )
    
    def generate_from_layout(self, layout: Layout2D, z_level: float = 0.0) -> List[Brush]:
        """
        Generate all brushes from a 2D layout.
        
        This is the main entry point that orchestrates the conversion of
        2D layout data into 3D idTech brushes.
        
        Args:
            layout: 2D layout containing rooms and corridors
            z_level: Base Z coordinate for this level
            
        Returns:
            List of all generated brushes ready for MAP file writing
        """
        logger.info("Generating brushes from layout: %dx%d tiles", layout.width, layout.height)
        
        # Clear previous generation
        self.generated_brushes.clear()
        self.room_connections.clear()
        self.room_brush_cache.clear()
        
        # Analyze room connections
        self._analyze_connections(layout)
        
        # Generate room brushes
        for room in layout.rooms:
            room_brushes = self.generate_room_brushes(room, z_level)
            self.generated_brushes.extend(room_brushes)
            self.room_brush_cache[room.room_id] = room_brushes
        
        # Generate corridor brushes
        for connection in layout.connections:
            corridor_brushes = self.generate_corridor_brushes(connection, z_level)
            self.generated_brushes.extend(corridor_brushes)
        
        # Generate boundary walls around the entire level
        boundary_brushes = self._generate_boundary_walls(layout, z_level)
        self.generated_brushes.extend(boundary_brushes)
        
        # Optimize if enabled
        if self.settings.merge_adjacent_brushes:
            self.generated_brushes = self.optimize_brushes(self.generated_brushes)
        
        # Validate all brushes (filter None from detail generators too)
        valid_brushes = []
        for brush in self.generated_brushes:
            if brush is None:
                continue
            if self.validate_brush_geometry(brush):
                valid_brushes.append(brush)
            else:
                logger.warning("Invalid brush geometry detected and removed")
        
        logger.info("Generated %d valid brushes", len(valid_brushes))
        return valid_brushes
    
    def generate_room_brushes(self, room: LayoutRoom, z_level: float = 0.0) -> List[Brush]:
        """
        Generate brushes for a single room using the hollow box approach.
        
        Creates 6 separate brushes: floor, ceiling, and 4 walls.
        Handles door openings by splitting walls where connections exist.
        
        Args:
            room: Room definition with bounds and type
            z_level: Base Z coordinate for this room
            
        Returns:
            List of brushes forming the hollow room
        """
        brushes = []
        
        # Convert tile coordinates to idTech units
        x1, y1, x2, y2 = room.to_quake_bounds()
        
        # Determine room height based on type, applying per-room z_offset
        room_height = self._get_room_height(room.room_type)
        room_z_offset = getattr(room, 'z_offset', 0)
        floor_z = z_level + self.settings.base_floor_z + room_z_offset
        ceiling_z = floor_z + room_height
        
        # Get textures for this room type
        textures = self._get_room_textures(room.room_type)
        
        # Create floor brush
        floor_brush = self._create_floor_brush(
            x1, y1, x2, y2, floor_z, textures['floor']
        )
        brushes.append(floor_brush)
        
        # Create ceiling brush
        ceiling_brush = self._create_ceiling_brush(
            x1, y1, x2, y2, ceiling_z, textures['ceiling']
        )
        brushes.append(ceiling_brush)
        
        # Create wall brushes with door openings
        wall_brushes = self._create_room_walls(
            room, x1, y1, x2, y2, floor_z, ceiling_z, textures['wall']
        )
        brushes.extend(wall_brushes)
        
        # Add special features based on room type
        if room.room_type == "vertical":
            vertical_brushes = self._create_vertical_connection(
                x1, y1, x2, y2, floor_z, ceiling_z
            )
            brushes.extend(vertical_brushes)

        # Phase 3: Architectural detail geometry
        detail_brushes = self._generate_room_details(
            room, x1, y1, x2, y2, floor_z, ceiling_z
        )
        brushes.extend(detail_brushes)

        # Phase 4: Arena room geometry
        if room.room_type == "arena":
            arena_brushes = self._generate_arena_geometry(
                x1, y1, x2, y2, floor_z, ceiling_z
            )
            brushes.extend(arena_brushes)

        # Phase 5: Primitive-based room geometry for typed rooms
        if room.room_type in ("entrance", "exit"):
            prim_brushes = self._generate_primitive_geometry(
                room, x1, y1, x2, y2, floor_z, ceiling_z
            )
            brushes.extend(prim_brushes)

        return brushes
    
    def generate_corridor_brushes(self, connection: LayoutConnection, 
                                 z_level: float = 0.0) -> List[Brush]:
        """
        Generate brushes for a corridor connecting two rooms.
        
        Creates a simple corridor between room centers with proper overlap
        to ensure no leaks between rooms.
        
        Args:
            connection: Corridor connection with room references
            z_level: Base Z coordinate
            
        Returns:
            List of brushes forming the corridor
        """
        brushes = []
        
        if not connection.start_room or not connection.end_room:
            logger.warning("Invalid corridor connection - missing rooms")
            return brushes
        
        # Compute per-room z offsets for height variation
        start_z_off = getattr(connection.start_room, 'z_offset', 0)
        end_z_off = getattr(connection.end_room, 'z_offset', 0)
        base_floor_z = z_level + self.settings.base_floor_z
        # Use the lower of the two rooms as the corridor base
        lo_z = base_floor_z + min(start_z_off, end_z_off)
        hi_z = base_floor_z + max(start_z_off, end_z_off)
        height_diff = hi_z - lo_z
        # Corridor ceiling must accommodate the higher room
        floor_z = lo_z
        ceiling_z = hi_z + self.settings.corridor_height

        textures = self._get_corridor_textures()

        # Prefer corridor geometry from world rectangles if available
        try:
            rects_world = getattr(connection, 'metadata', {}).get('path_rects_world') if hasattr(connection, 'metadata') else None
        except Exception:
            rects_world = None
        if rects_world:
            for rw in rects_world:
                rx1 = float(rw.get('x', 0))
                ry1 = float(rw.get('y', 0))
                rx2 = rx1 + float(rw.get('width', 0))
                ry2 = ry1 + float(rw.get('height', 0))
                # Pad generously to guarantee overlap at seams and through door frames
                rx1 -= CORRIDOR_FLOOR_PADDING
                ry1 -= CORRIDOR_FLOOR_PADDING
                rx2 += CORRIDOR_FLOOR_PADDING
                ry2 += CORRIDOR_FLOOR_PADDING

                # Floor and ceiling for this segment
                floor = self._create_floor_brush(rx1, ry1, rx2, ry2, floor_z, textures['floor'])
                brushes.append(floor)
                ceiling = self._create_ceiling_brush(rx1, ry1, rx2, ry2, ceiling_z, textures['ceiling'])
                brushes.append(ceiling)
                # Side walls trimmed near junctions
                walls = self._create_corridor_walls(rx1, ry1, rx2, ry2, floor_z, ceiling_z, textures['wall'])
                brushes.extend(walls)

            # Generate stairs if height difference exists
            if height_diff > 0 and rects_world:
                stair_brushes = self._generate_corridor_stairs(
                    connection, rects_world, lo_z, hi_z, textures
                )
                brushes.extend(stair_brushes)
            return brushes
        
        # Get room bounds
        start_bounds = connection.start_room.to_quake_bounds()
        end_bounds = connection.end_room.to_quake_bounds()
        
        # Calculate centers
        start_center_x = (start_bounds[0] + start_bounds[2]) / 2
        start_center_y = (start_bounds[1] + start_bounds[3]) / 2
        end_center_x = (end_bounds[0] + end_bounds[2]) / 2
        end_center_y = (end_bounds[1] + end_bounds[3]) / 2
        
        # Create a corridor that directly connects room centers
        corridor_width = self.settings.corridor_width
        
        # Determine if corridor should be horizontal or vertical based on distance
        dx = abs(end_center_x - start_center_x)
        dy = abs(end_center_y - start_center_y)
        
        if dx > dy:
            # Horizontal corridor
            min_x = min(start_center_x, end_center_x)
            max_x = max(start_center_x, end_center_x)
            center_y = (start_center_y + end_center_y) / 2
            min_y = center_y - corridor_width / 2
            max_y = center_y + corridor_width / 2
        else:
            # Vertical corridor
            min_y = min(start_center_y, end_center_y)
            max_y = max(start_center_y, end_center_y)
            center_x = (start_center_x + end_center_x) / 2
            min_x = center_x - corridor_width / 2
            max_x = center_x + corridor_width / 2
        
        # Extend corridors to overlap with room boundaries
        # Use a generous, unified overlap to guarantee seams close at doorways
        overlap = int(self._junction_overlap)
        min_x -= overlap
        max_x += overlap
        min_y -= overlap
        max_y += overlap
        
        # Create corridor floor (this will overlap room floors, which is fine)
        floor = self._create_floor_brush(
            min_x, min_y, max_x, max_y, floor_z, textures['floor']
        )
        brushes.append(floor)
        
        # Create corridor ceiling
        ceiling = self._create_ceiling_brush(
            min_x, min_y, max_x, max_y, ceiling_z, textures['ceiling']
        )
        brushes.append(ceiling)
        
        # Add corridor side walls with small margins to avoid interfering at room junctions
        walls = self._create_corridor_walls(
            min_x, min_y, max_x, max_y, floor_z, ceiling_z, textures['wall']
        )
        brushes.extend(walls)

        # Generate stairs for height differences in fallback path
        if height_diff > 0:
            stair_brushes = self._generate_corridor_stairs_fallback(
                min_x, min_y, max_x, max_y, lo_z, hi_z,
                start_z_off <= end_z_off, dx > dy, textures
            )
            brushes.extend(stair_brushes)

        return brushes
    
    def _create_floor_brush(self, x1: float, y1: float, x2: float, y2: float,
                           z: float, texture: str) -> Brush:
        """Create a floor brush with proper thickness.

        The floor surface is at z, with thickness extending downward.
        This is a thin wrapper around _create_box_brush for semantic clarity.
        """
        z_bottom = z - self.settings.floor_thickness
        z_top = z  # Walking surface
        return self._create_box_brush(x1, y1, x2, y2, z_bottom, z_top, texture)

    def _create_ceiling_brush(self, x1: float, y1: float, x2: float, y2: float,
                            z: float, texture: str) -> Brush:
        """Create a ceiling brush with proper thickness.

        The visible ceiling surface is at z, with thickness extending upward.
        This is a thin wrapper around _create_box_brush for semantic clarity.
        """
        z_bottom = z  # Visible ceiling surface
        z_top = z + self.settings.ceiling_thickness  # Extends up
        return self._create_box_brush(x1, y1, x2, y2, z_bottom, z_top, texture)
    
    def _create_room_walls(self, room: LayoutRoom, x1: float, y1: float, 
                          x2: float, y2: float, floor_z: float, 
                          ceiling_z: float, texture: str) -> List[Brush]:
        """
        Create wall brushes for a room with door openings where corridors connect.
        
        For now, use a conservative approach: create solid walls first, then
        punch holes using negative brushes (detail brushes) for doors.
        This prevents leaks while allowing movement.
        """
        walls = []
        
        # Get connection points for this room
        connection_points = self._get_room_connection_points(room)
        
        if not connection_points:
            # Fallback: if the room has declared connections but we couldn't align points,
            # open a central doorway on the wall facing the nearest connected room.
            conns = getattr(room, 'connections', []) or []
            if conns:
                # Pick target direction by vector to the first connected room's center
                other = conns[0].end_room if conns[0].start_room == room else conns[0].start_room
                ox1, oy1, ox2, oy2 = other.to_quake_bounds()
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                rcx = (x1 + x2) / 2
                rcy = (y1 + y2) / 2
                dx = ocx - rcx
                dy = ocy - rcy
                door_min_clear = int(self.settings.corridor_width + 32)
                door_width = max(self.settings.door_width, door_min_clear, 112)
                if abs(dx) > abs(dy):
                    # Face east or west
                    if dx > 0:
                        # door on east wall
                        walls.append(self._create_wall_brush(x1, y1, x1, y2, floor_z, ceiling_z, texture, 'west'))
                        walls.extend(self._create_simple_door_wall(x1, y1, x2, y2, floor_z, ceiling_z, texture, 'east', door_width))
                        walls.append(self._create_wall_brush(x1, y1, x2, y1, floor_z, ceiling_z, texture, 'north'))
                        walls.append(self._create_wall_brush(x1, y2, x2, y2, floor_z, ceiling_z, texture, 'south'))
                    else:
                        walls.append(self._create_wall_brush(x2, y1, x2, y2, floor_z, ceiling_z, texture, 'east'))
                        walls.extend(self._create_simple_door_wall(x1, y1, x2, y2, floor_z, ceiling_z, texture, 'west', door_width))
                        walls.append(self._create_wall_brush(x1, y1, x2, y1, floor_z, ceiling_z, texture, 'north'))
                        walls.append(self._create_wall_brush(x1, y2, x2, y2, floor_z, ceiling_z, texture, 'south'))
                else:
                    # Face north or south
                    if dy > 0:
                        walls.append(self._create_wall_brush(x1, y1, x1, y2, floor_z, ceiling_z, texture, 'west'))
                        walls.append(self._create_wall_brush(x2, y1, x2, y2, floor_z, ceiling_z, texture, 'east'))
                        walls.extend(self._create_simple_door_wall(x1, y1, x2, y2, floor_z, ceiling_z, texture, 'south', door_width))
                        walls.append(self._create_wall_brush(x1, y1, x2, y1, floor_z, ceiling_z, texture, 'north'))
                    else:
                        walls.append(self._create_wall_brush(x1, y1, x1, y2, floor_z, ceiling_z, texture, 'west'))
                        walls.append(self._create_wall_brush(x2, y1, x2, y2, floor_z, ceiling_z, texture, 'east'))
                        walls.extend(self._create_simple_door_wall(x1, y1, x2, y2, floor_z, ceiling_z, texture, 'north', door_width))
                        walls.append(self._create_wall_brush(x1, y2, x2, y2, floor_z, ceiling_z, texture, 'south'))
                return walls
            # No connections declared: create solid walls
            walls.append(self._create_wall_brush(x1, y1, x2, y1, floor_z, ceiling_z, texture, 'north'))
            walls.append(self._create_wall_brush(x1, y2, x2, y2, floor_z, ceiling_z, texture, 'south'))
            walls.append(self._create_wall_brush(x1, y1, x1, y2, floor_z, ceiling_z, texture, 'west'))
            walls.append(self._create_wall_brush(x2, y1, x2, y2, floor_z, ceiling_z, texture, 'east'))
            return walls
        
        # For connected rooms, create walls with door openings aligned to corridor points
        # Ensure doorway clear span strictly exceeds corridor width to account for
        # diagonal clearances against jambs; add generous safety.
        door_min_clear = int(self.settings.corridor_width + CORRIDOR_FLOOR_PADDING)
        door_width = max(self.settings.door_width, door_min_clear, 112)

        tol = WALL_CONNECTION_TOLERANCE
        north_pts = [cp for cp in connection_points if abs(cp[1] - y1) < tol]
        south_pts = [cp for cp in connection_points if abs(cp[1] - y2) < tol]
        west_pts  = [cp for cp in connection_points if abs(cp[0] - x1) < tol]
        east_pts  = [cp for cp in connection_points if abs(cp[0] - x2) < tol]

        def door_segments_h(x_start: float, x_end: float, pts: List[Tuple[float, float]]):
            if not pts:
                return None
            # Span door to cover min..max of connection x positions, padded
            xs = [p[0] for p in pts]
            dmin = max(x_start + 16, min(xs) - door_min_clear // 2)
            dmax = min(x_end - 16, max(xs) + door_min_clear // 2)
            if dmax <= dmin:
                # Fallback to centered door if degenerate
                dc = (x_start + x_end) / 2
                dmin = max(x_start + 16, dc - door_width / 2)
                dmax = min(x_end - 16, dc + door_width / 2)
            return dmin, dmax

        def door_segments_v(y_start: float, y_end: float, pts: List[Tuple[float, float]]):
            if not pts:
                return None
            ys = [p[1] for p in pts]
            dmin = max(y_start + 16, min(ys) - door_min_clear // 2)
            dmax = min(y_end - 16, max(ys) + door_min_clear // 2)
            if dmax <= dmin:
                dc = (y_start + y_end) / 2
                dmin = max(y_start + 16, dc - door_width / 2)
                dmax = min(y_end - 16, dc + door_width / 2)
            return dmin, dmax

        def add_threshold_floor_h(x_start: float, x_end: float, wall_y: float):
            # Add a stitched floor across the doorway threshold to guarantee continuity
            pad = max(48.0, self.settings.wall_thickness * THRESHOLD_FLOOR_PAD_FACTOR)
            edge_margin = self.settings.floor_thickness  # Small extension past door edges
            fx1 = x_start - edge_margin
            fx2 = x_end + edge_margin
            fy1 = wall_y - pad
            fy2 = wall_y + pad
            return self._create_floor_brush(fx1, fy1, fx2, fy2, floor_z, self._get_corridor_textures()['floor'])

        def add_threshold_floor_v(y_start: float, y_end: float, wall_x: float):
            pad = max(48.0, self.settings.wall_thickness * THRESHOLD_FLOOR_PAD_FACTOR)
            edge_margin = self.settings.floor_thickness
            fy1 = y_start - edge_margin
            fy2 = y_end + edge_margin
            fx1 = wall_x - pad
            fx2 = wall_x + pad
            return self._create_floor_brush(fx1, fy1, fx2, fy2, floor_z, self._get_corridor_textures()['floor'])

        # Helper to create wall with door opening for a given side
        def _create_wall_with_door(wall_start: float, wall_end: float,
                                   wall_pos: float, door_seg: Optional[Tuple[float, float]],
                                   orientation: str, is_horizontal: bool) -> List[Brush]:
            """Create wall segments around a door opening."""
            result = []
            lintel_margin = self.settings.floor_thickness  # Space below ceiling for lintel

            if door_seg is None:
                # No door - create solid wall
                if is_horizontal:
                    result.append(self._create_wall_brush(
                        wall_start, wall_pos, wall_end, wall_pos,
                        floor_z, ceiling_z, texture, orientation))
                else:
                    result.append(self._create_wall_brush(
                        wall_pos, wall_start, wall_pos, wall_end,
                        floor_z, ceiling_z, texture, orientation))
                return result

            dmin, dmax = door_seg

            # Wall segment before door
            if dmin > wall_start + MIN_WALL_SEGMENT_SIZE:
                if is_horizontal:
                    result.append(self._create_wall_brush(
                        wall_start, wall_pos, dmin, wall_pos,
                        floor_z, ceiling_z, texture, orientation))
                else:
                    result.append(self._create_wall_brush(
                        wall_pos, wall_start, wall_pos, dmin,
                        floor_z, ceiling_z, texture, orientation))

            # Wall segment after door
            if dmax < wall_end - MIN_WALL_SEGMENT_SIZE:
                if is_horizontal:
                    result.append(self._create_wall_brush(
                        dmax, wall_pos, wall_end, wall_pos,
                        floor_z, ceiling_z, texture, orientation))
                else:
                    result.append(self._create_wall_brush(
                        wall_pos, dmax, wall_pos, wall_end,
                        floor_z, ceiling_z, texture, orientation))

            # Door lintel (above opening)
            door_lintel_z = floor_z + min(self.settings.door_height, ceiling_z - floor_z - lintel_margin)
            if door_lintel_z < ceiling_z - lintel_margin:
                if is_horizontal:
                    result.append(self._create_wall_brush(
                        dmin, wall_pos, dmax, wall_pos,
                        door_lintel_z, ceiling_z, texture, orientation))
                else:
                    result.append(self._create_wall_brush(
                        wall_pos, dmin, wall_pos, dmax,
                        door_lintel_z, ceiling_z, texture, orientation))

            # Threshold floor
            if is_horizontal:
                result.append(add_threshold_floor_h(dmin, dmax, wall_pos))
            else:
                result.append(add_threshold_floor_v(dmin, dmax, wall_pos))

            return result

        # North wall (y1)
        north_seg = door_segments_h(x1, x2, north_pts) if north_pts else None
        walls.extend(_create_wall_with_door(x1, x2, y1, north_seg, 'north', is_horizontal=True))

        # South wall (y2)
        south_seg = door_segments_h(x1, x2, south_pts) if south_pts else None
        walls.extend(_create_wall_with_door(x1, x2, y2, south_seg, 'south', is_horizontal=True))

        # West wall (x1)
        west_seg = door_segments_v(y1, y2, west_pts) if west_pts else None
        walls.extend(_create_wall_with_door(y1, y2, x1, west_seg, 'west', is_horizontal=False))

        # East wall (x2)
        east_seg = door_segments_v(y1, y2, east_pts) if east_pts else None
        walls.extend(_create_wall_with_door(y1, y2, x2, east_seg, 'east', is_horizontal=False))
        
        return walls
    
    def _create_wall_brush(self, x1: float, y1: float, x2: float, y2: float,
                          floor_z: float, ceiling_z: float, texture: str,
                          orientation: str) -> Brush:
        """Create a single wall brush with proper thickness"""
        thickness = self.settings.wall_thickness
        
        # Use the box brush method for all walls  
        if orientation == 'north':
            # Wall along y1
            return self._create_box_brush(x1, y1 - thickness, x2, y1, floor_z, ceiling_z, texture)
        elif orientation == 'south':
            # Wall along y2
            return self._create_box_brush(x1, y2, x2, y2 + thickness, floor_z, ceiling_z, texture)
        elif orientation == 'west':
            # Wall along x1
            return self._create_box_brush(x1 - thickness, y1, x1, y2, floor_z, ceiling_z, texture)
        else:  # east
            # Wall along x2
            return self._create_box_brush(x2, y1, x2 + thickness, y2, floor_z, ceiling_z, texture)
    
    def _create_corridor_walls(self, x1: float, y1: float, x2: float, y2: float,
                              floor_z: float, ceiling_z: float, texture: str) -> List[Brush]:
        """Create corridor walls only where they don't interfere with room connections"""
        walls = []
        thickness = self.settings.wall_thickness
        
        # Determine if corridor is horizontal or vertical
        width = x2 - x1
        height = y2 - y1
        
        # Only create walls in the middle sections of corridors
        # to avoid interfering with room connections
        # Avoid corridor walls intruding into rooms by leaving a generous margin
        # derived from the same corridor overlap into rooms used elsewhere.
        overlap = int(self._junction_overlap)  # keep in sync with generate_corridor_brushes
        # Keep ends at least ~half a corridor width away from room junctions
        margin = max(
            overlap + 32,
            int(self.settings.wall_thickness + 32),
            int(self.settings.corridor_width * 0.5)
        )
        
        if width > height:
            # Horizontal corridor - create north and south walls with margins
            if width > margin * 2:
                wall_x1 = x1 + margin
                wall_x2 = x2 - margin
                walls.append(self._create_wall_brush(
                    wall_x1, y1, wall_x2, y1, floor_z, ceiling_z, texture, 'north'
                ))
                walls.append(self._create_wall_brush(
                    wall_x1, y2, wall_x2, y2, floor_z, ceiling_z, texture, 'south'
                ))
        else:
            # Vertical corridor - create west and east walls with margins
            if height > margin * 2:
                wall_y1 = y1 + margin
                wall_y2 = y2 - margin
                walls.append(self._create_wall_brush(
                    x1, wall_y1, x1, wall_y2, floor_z, ceiling_z, texture, 'west'
                ))
                walls.append(self._create_wall_brush(
                    x2, wall_y1, x2, wall_y2, floor_z, ceiling_z, texture, 'east'
                ))
        
        return walls
    
    def _create_vertical_connection(self, x1: float, y1: float, x2: float, y2: float,
                                   floor_z: float, ceiling_z: float) -> List[Brush]:
        """Create stairs or lift shaft for vertical connections"""
        brushes = []
        
        # Calculate center for vertical shaft
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        shaft_size = min(x2 - x1, y2 - y1) * 0.5
        
        # Create lift platform or stairs
        num_steps = int((ceiling_z - floor_z) / self.settings.stair_step_height)
        
        if num_steps <= 16:  # Use stairs for short distances
            brushes.extend(self._create_stairs(
                cx - shaft_size/2, cy - shaft_size/2,
                cx + shaft_size/2, cy + shaft_size/2,
                floor_z, ceiling_z, num_steps
            ))
        else:  # Use lift for tall connections
            brushes.append(self._create_lift_platform(
                cx - shaft_size/2, cy - shaft_size/2,
                cx + shaft_size/2, cy + shaft_size/2,
                floor_z
            ))
        
        return brushes
    
    def _create_stairs(self, x1: float, y1: float, x2: float, y2: float,
                      floor_z: float, ceiling_z: float, num_steps: int) -> List[Brush]:
        """Generate stair geometry"""
        stairs = []
        step_height = (ceiling_z - floor_z) / num_steps
        step_depth = (y2 - y1) / num_steps
        
        for i in range(num_steps):
            step_z = floor_z + i * step_height
            step_y1 = y1 + i * step_depth
            step_y2 = y1 + (i + 1) * step_depth
            
            # Create step brush
            step = self._create_box_brush(
                x1, step_y1, x2, step_y2,
                step_z, step_z + step_height,
                self.settings.texture_set.STAIRS
            )
            stairs.append(step)
        
        return stairs
    
    def _create_lift_platform(self, x1: float, y1: float, x2: float, y2: float,
                             z: float) -> Brush:
        """Create a lift platform brush"""
        return self._create_box_brush(
            x1, y1, x2, y2,
            z, z + LIFT_PLATFORM_THICKNESS,
            self.settings.texture_set.LIFT_PLATFORM
        )
    
    @staticmethod
    def _snap(v: float) -> float:
        """Snap a coordinate to the nearest integer for clean idTech geometry."""
        return float(round(v))

    def _create_box_brush(self, x1: float, y1: float, x2: float, y2: float,
                         z1: float, z2: float, texture: str) -> Brush:
        """Helper to create a simple box brush with proper idTech plane format"""
        # Snap all coordinates to integers
        x1, y1, x2, y2 = self._snap(x1), self._snap(y1), self._snap(x2), self._snap(y2)
        z1, z2 = self._snap(z1), self._snap(z2)
        # Use the actual box boundary coordinates for non-colinear points
        planes = [
            # Left face (X = x1)
            Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2), texture),
            # Right face (X = x2)
            Plane((x2, y1, z1), (x2, y1, z2), (x2, y2, z1), texture),
            # Front face (Y = y1)
            Plane((x1, y1, z1), (x1, y1, z2), (x2, y1, z1), texture),
            # Back face (Y = y2)
            Plane((x1, y2, z1), (x2, y2, z1), (x1, y2, z2), texture),
            # Bottom face (Z = z1)
            Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1), texture),
            # Top face (Z = z2)
            Plane((x1, y1, z2), (x1, y2, z2), (x2, y1, z2), texture)
        ]
        
        return Brush(planes=planes, brush_id=self._next_brush_id())
    
    def _generate_boundary_walls(self, layout: Layout2D, z_level: float) -> List[Brush]:
        """Generate solid enclosure around the entire level perimeter.

        Adds four perimeter walls plus a boundary floor and ceiling to ensure
        the world is sealed (prevents leaks to the void during qbsp).
        """
        brushes = []

        # Convert layout bounds to idTech units
        width = layout.width * self.settings.grid_size
        height = layout.height * self.settings.grid_size

        thickness = BOUNDARY_WALL_THICKNESS
        wall_height = self.settings.standard_height + BOUNDARY_WALL_MARGIN
        floor_offset = BOUNDARY_FLOOR_OFFSET

        boundary_texture = self.settings.texture_set.ARENA_WALL  # METAL1_1
        floor_texture = self.settings.texture_set.FLOOR_DEFAULT  # GROUND1_6
        ceiling_texture = self.settings.texture_set.CEILING_DEFAULT  # CEIL1_1

        # North boundary
        brushes.append(self._create_box_brush(
            -thickness, -thickness, width + thickness, 0,
            z_level - floor_offset, z_level + wall_height,
            boundary_texture
        ))

        # South boundary
        brushes.append(self._create_box_brush(
            -thickness, height, width + thickness, height + thickness,
            z_level - floor_offset, z_level + wall_height,
            boundary_texture
        ))

        # West boundary
        brushes.append(self._create_box_brush(
            -thickness, 0, 0, height,
            z_level - floor_offset, z_level + wall_height,
            boundary_texture
        ))

        # East boundary
        brushes.append(self._create_box_brush(
            width, 0, width + thickness, height,
            z_level - floor_offset, z_level + wall_height,
            boundary_texture
        ))

        # Boundary floor slab (under everything)
        brushes.append(self._create_box_brush(
            -thickness, -thickness, width + thickness, height + thickness,
            z_level - BOUNDARY_FLOOR_DEPTH, z_level - floor_offset,
            floor_texture
        ))

        # Boundary ceiling slab (above everything)
        brushes.append(self._create_box_brush(
            -thickness, -thickness, width + thickness, height + thickness,
            z_level + wall_height, z_level + wall_height + BOUNDARY_FLOOR_DEPTH,
            ceiling_texture
        ))

        return brushes

    # ------------------------------------------------------------------
    # Phase 3: Architectural detail geometry
    # ------------------------------------------------------------------

    def _generate_room_details(self, room, x1, y1, x2, y2, floor_z, ceiling_z):
        """Add architectural detail brushes to a room."""
        from quake_levelgenerator.src.conversion.geometry.detail_brushes import (
            floor_trim_strips, ceiling_beams, wall_buttresses, pillars_for_room,
            door_frame_trim,
        )
        brushes = []
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        ifloor = int(floor_z)
        iceil = int(ceiling_z)
        room_w = ix2 - ix1
        room_h = iy2 - iy1

        # Floor trim strips for all rooms
        brushes.extend(floor_trim_strips(ix1, iy1, ix2, iy2, ifloor))

        # Pillars for large rooms (> 256 units wide)
        brushes.extend(pillars_for_room(ix1, iy1, ix2, iy2, ifloor, iceil))

        # Wall buttresses for long walls
        if room_w > LARGE_ROOM_THRESHOLD or room_h > LARGE_ROOM_THRESHOLD:
            brushes.extend(wall_buttresses(ix1, iy1, ix2, iy2, ifloor, iceil))

        # Ceiling beams for arena rooms
        if room.room_type == "arena":
            brushes.extend(ceiling_beams(ix1, iy1, ix2, iy2, iceil))

        # Door frame trim on connections
        connection_points = self._get_room_connection_points(room)
        tol = WALL_CONNECTION_TOLERANCE
        lintel_clearance = MIN_WALL_SEGMENT_SIZE  # Space below ceiling for trim
        for cp in connection_points:
            cpx, cpy = cp
            # Determine which wall this connection is on
            if abs(cpy - y1) < tol or abs(cpy - y2) < tol:
                # North or south wall - door spans X
                dx_min = max(ix1, int(cpx - DETAIL_DOOR_HALF_WIDTH))
                dx_max = min(ix2, int(cpx + DETAIL_DOOR_HALF_WIDTH))
                brushes.extend(door_frame_trim(
                    dx_min, iy1 if abs(cpy - y1) < tol else iy2,
                    dx_max, iy1 if abs(cpy - y1) < tol else iy2,
                    ifloor, iceil - lintel_clearance, axis="x",
                ))
            elif abs(cpx - x1) < tol or abs(cpx - x2) < tol:
                dy_min = max(iy1, int(cpy - DETAIL_DOOR_HALF_WIDTH))
                dy_max = min(iy2, int(cpy + DETAIL_DOOR_HALF_WIDTH))
                brushes.extend(door_frame_trim(
                    ix1 if abs(cpx - x1) < tol else ix2, dy_min,
                    ix1 if abs(cpx - x1) < tol else ix2, dy_max,
                    ifloor, iceil - lintel_clearance, axis="y",
                ))

        return brushes

    # ------------------------------------------------------------------
    # Phase 4: Arena room geometry
    # ------------------------------------------------------------------

    def _generate_arena_geometry(self, x1, y1, x2, y2, floor_z, ceiling_z):
        """Generate raised walkways, ramps, and pillars for arena rooms."""
        from quake_levelgenerator.src.conversion.geometry.detail_brushes import _box, pillar
        brushes = []
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        ifloor = int(floor_z)
        iceil = int(ceiling_z)

        walkway_h = DETAIL_WALKWAY_HEIGHT
        walkway_d = DETAIL_WALKWAY_DEPTH
        walk_top = ifloor + walkway_h
        ramp_w = DETAIL_RAMP_WIDTH

        arena_texture = self.settings.texture_set.ARENA_WALL  # METAL1_1
        stair_texture = self.settings.texture_set.STAIRS  # STEP1

        # North walkway
        brushes.append(_box(ix1, iy2 - walkway_d, ifloor, ix2, iy2, walk_top, arena_texture))
        # South walkway
        brushes.append(_box(ix1, iy1, ifloor, ix2, iy1 + walkway_d, walk_top, arena_texture))
        # West walkway
        brushes.append(_box(ix1, iy1 + walkway_d, ifloor, ix1 + walkway_d, iy2 - walkway_d, walk_top, arena_texture))

        # Ramp from south walkway to ground floor (center of south wall)
        ramp_cx = (ix1 + ix2) // 2
        ramp_x1 = ramp_cx - ramp_w // 2
        ramp_x2 = ramp_cx + ramp_w // 2
        ramp_y1 = iy1 + walkway_d
        ramp_y2 = ramp_y1 + walkway_h * 2  # Ramp extends into arena

        # Approximate ramp as stepped blocks
        ramp_step_height = int(self.settings.stair_step_height)
        n_ramp_steps = max(1, walkway_h // ramp_step_height)
        for i in range(n_ramp_steps):
            step_z = walk_top - (i + 1) * (walkway_h / n_ramp_steps)
            sy1 = ramp_y1 + i * (ramp_y2 - ramp_y1) / n_ramp_steps
            sy2 = ramp_y1 + (i + 1) * (ramp_y2 - ramp_y1) / n_ramp_steps
            brushes.append(_box(
                ramp_x1, int(sy1), ifloor, ramp_x2, int(sy2), int(step_z), stair_texture
            ))

        # Cover pillars on ground floor
        cx = (ix1 + ix2) // 2
        cy = (iy1 + iy2) // 2
        offset = min(ix2 - ix1, iy2 - iy1) // 4
        for px, py in [(cx - offset, cy - offset), (cx + offset, cy - offset),
                       (cx - offset, cy + offset), (cx + offset, cy + offset)]:
            brushes.extend(pillar(px, py, ifloor, iceil, DETAIL_PILLAR_SIZE, arena_texture))

        return brushes

    def _generate_primitive_geometry(self, room, x1, y1, x2, y2, floor_z, ceiling_z):
        """Generate interior-only architectural detail for typed rooms.

        Instead of calling Chapel/Crypt generate() (which emit full enclosed
        room shells that overlap BSP room geometry), we place only interior
        detail brushes: pillar rows, raised platforms, alcoves.

        ENTRANCE rooms: nave pillar rows + raised apse platform.
        EXIT rooms: central sarcophagus platform + wall alcoves.
        """
        from quake_levelgenerator.src.conversion.geometry.detail_brushes import _box, pillar
        brushes: List[Brush] = []
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        ifloor = int(floor_z)
        iceil = int(ceiling_z)
        room_w = ix2 - ix1
        room_l = iy2 - iy1

        # Configuration for entrance/exit room details
        PILLAR_INSET_RATIO = 4          # Pillar distance from wall = room_width / this
        MIN_PILLAR_SPACING = 128        # Minimum spacing between pillar rows
        NAVE_PILLAR_SIZE = 24           # Pillar size for entrance nave
        APSE_PLATFORM_HEIGHT = 32       # Raised platform height in entrance
        APSE_MAX_DEPTH = 96             # Maximum depth of apse platform

        SARCOPHAGUS_HEIGHT = 24         # Central platform height in exit room
        SARCOPHAGUS_MAX_WIDTH = 96      # Maximum sarcophagus platform width
        SARCOPHAGUS_MAX_LENGTH = 128    # Maximum sarcophagus platform length

        ALCOVE_DEPTH = 24               # Wall alcove depth
        ALCOVE_WIDTH = 48               # Wall alcove width
        ALCOVE_HEIGHT = 64              # Wall alcove height
        ALCOVE_BASE_OFFSET = 32         # Height above floor for alcove
        ALCOVE_SPACING = 128            # Spacing between alcoves

        wall_texture = self.settings.texture_set.WALL_DEFAULT  # WBRICK1_5
        metal_texture = "METAL1_2"  # Exit room accent texture

        try:
            if room.room_type == "entrance":
                # Nave pillar rows along the length of the room
                if room_w >= MIN_ROOM_SIZE_FOR_DETAILS and room_l >= MIN_ROOM_SIZE_FOR_DETAILS:
                    pillar_inset_x = room_w // PILLAR_INSET_RATIO
                    spacing = max(MIN_PILLAR_SPACING, room_l // PILLAR_INSET_RATIO)
                    py = iy1 + spacing
                    while py < iy2 - spacing // 2:
                        brushes.extend(pillar(ix1 + pillar_inset_x, py, ifloor, iceil, NAVE_PILLAR_SIZE, wall_texture))
                        brushes.extend(pillar(ix2 - pillar_inset_x, py, ifloor, iceil, NAVE_PILLAR_SIZE, wall_texture))
                        py += spacing

                # Raised apse platform at far end if room is long enough
                if room_l > LARGE_ROOM_THRESHOLD:
                    apse_depth = min(APSE_MAX_DEPTH, room_l // 5)
                    edge_margin = MIN_WALL_SEGMENT_SIZE
                    brushes.append(_box(
                        ix1 + edge_margin, iy2 - apse_depth, ifloor,
                        ix2 - edge_margin, iy2 - edge_margin, ifloor + APSE_PLATFORM_HEIGHT,
                        wall_texture
                    ))

            elif room.room_type == "exit":
                # Central sarcophagus platform
                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2
                plat_w = min(SARCOPHAGUS_MAX_WIDTH, room_w // 3)
                plat_l = min(SARCOPHAGUS_MAX_LENGTH, room_l // 3)
                brushes.append(_box(
                    cx - plat_w // 2, cy - plat_l // 2, ifloor,
                    cx + plat_w // 2, cy + plat_l // 2, ifloor + SARCOPHAGUS_HEIGHT,
                    metal_texture
                ))

                # Wall alcoves (recessed niches) on long walls
                if room_l >= LARGE_ROOM_THRESHOLD:
                    alcove_margin = DETAIL_WALKWAY_HEIGHT  # Reuse as safe margin from ends
                    alcove_base = ifloor + ALCOVE_BASE_OFFSET
                    for ay in range(iy1 + alcove_margin, iy2 - alcove_margin, ALCOVE_SPACING):
                        # West wall alcove shelf
                        brushes.append(_box(
                            ix1, ay, alcove_base,
                            ix1 + ALCOVE_DEPTH, ay + ALCOVE_WIDTH, alcove_base + ALCOVE_HEIGHT,
                            metal_texture
                        ))
                        # East wall alcove shelf
                        brushes.append(_box(
                            ix2 - ALCOVE_DEPTH, ay, alcove_base,
                            ix2, ay + ALCOVE_WIDTH, alcove_base + ALCOVE_HEIGHT,
                            metal_texture
                        ))

        except Exception as e:
            logger.warning(f"Failed to generate primitive geometry for {room.room_type} room: {e}")

        return brushes

    # ------------------------------------------------------------------
    # Staircase generation for height-varying corridors
    # ------------------------------------------------------------------

    def _generate_corridor_stairs(self, connection, rects_world, lo_z, hi_z, textures):
        """Generate step brushes along corridor world rects to bridge a height gap."""
        brushes = []
        height_diff = hi_z - lo_z
        step_h = int(self.settings.stair_step_height)  # 16
        step_d = int(self.settings.stair_step_depth)    # 32
        n_steps = max(1, int(height_diff / step_h))
        step_h = height_diff / n_steps  # exact division

        # Use the first rect as stair region
        rw = rects_world[0]
        rx1 = float(rw.get('x', 0))
        ry1 = float(rw.get('y', 0))
        rw_w = float(rw.get('width', 0))
        rw_h = float(rw.get('height', 0))

        # Determine corridor main axis
        start_z_off = getattr(connection.start_room, 'z_offset', 0)
        end_z_off = getattr(connection.end_room, 'z_offset', 0)
        ascending = start_z_off < end_z_off  # stairs go up from start to end

        if rw_w >= rw_h:
            # Horizontal corridor — steps span X
            for i in range(n_steps):
                sz = lo_z + i * step_h
                sx = rx1 + (i * rw_w / n_steps)
                sx2 = rx1 + ((i + 1) * rw_w / n_steps)
                if not ascending:
                    sx, sx2 = rx1 + rw_w - (i + 1) * rw_w / n_steps, rx1 + rw_w - i * rw_w / n_steps
                brushes.append(self._create_floor_brush(
                    sx, ry1, sx2, ry1 + rw_h,
                    sz + step_h, TextureSet.STAIRS
                ))
        else:
            # Vertical corridor — steps span Y
            for i in range(n_steps):
                sz = lo_z + i * step_h
                sy = ry1 + (i * rw_h / n_steps)
                sy2 = ry1 + ((i + 1) * rw_h / n_steps)
                if not ascending:
                    sy, sy2 = ry1 + rw_h - (i + 1) * rw_h / n_steps, ry1 + rw_h - i * rw_h / n_steps
                brushes.append(self._create_floor_brush(
                    rx1, sy, rx1 + rw_w, sy2,
                    sz + step_h, TextureSet.STAIRS
                ))
        return brushes

    def _generate_corridor_stairs_fallback(self, min_x, min_y, max_x, max_y,
                                            lo_z, hi_z, ascending, horizontal, textures):
        """Fallback stair generation for corridors without world rects."""
        brushes = []
        height_diff = hi_z - lo_z
        step_h = int(self.settings.stair_step_height)
        n_steps = max(1, int(height_diff / step_h))
        step_h = height_diff / n_steps

        if horizontal:
            span = max_x - min_x
            for i in range(n_steps):
                sz = lo_z + i * step_h
                if ascending:
                    sx = min_x + i * span / n_steps
                    sx2 = min_x + (i + 1) * span / n_steps
                else:
                    sx = max_x - (i + 1) * span / n_steps
                    sx2 = max_x - i * span / n_steps
                brushes.append(self._create_floor_brush(
                    sx, min_y, sx2, max_y,
                    sz + step_h, TextureSet.STAIRS
                ))
        else:
            span = max_y - min_y
            for i in range(n_steps):
                sz = lo_z + i * step_h
                if ascending:
                    sy = min_y + i * span / n_steps
                    sy2 = min_y + (i + 1) * span / n_steps
                else:
                    sy = max_y - (i + 1) * span / n_steps
                    sy2 = max_y - i * span / n_steps
                brushes.append(self._create_floor_brush(
                    min_x, sy, max_x, sy2,
                    sz + step_h, TextureSet.STAIRS
                ))
        return brushes

    def optimize_brushes(self, brushes: List[Brush]) -> List[Brush]:
        """
        Optimize brush geometry by merging adjacent coplanar brushes.
        
        This reduces the total brush count for better map compilation performance.
        """
        # For now, return brushes as-is
        # Full optimization would require complex geometry analysis
        return brushes
    
    def validate_brush_geometry(self, brush: Brush) -> bool:
        """
        Validate that a brush has valid convex geometry.
        
        idTech requires:
        - Minimum 4 planes (tetrahedron)
        - All planes must form a convex hull
        - No degenerate (zero-area) faces
        """
        if len(brush.planes) < 4:
            return False
        
        # Check for degenerate planes
        for plane in brush.planes:
            if plane.p1 == plane.p2 or plane.p2 == plane.p3 or plane.p1 == plane.p3:
                return False
        
        # TODO: Implement full convexity check
        # For now, assume our generation creates valid brushes
        
        return True
    
    def generate_multi_level(self, multi_layout: MultiLevelLayout) -> List[Brush]:
        """
        Generate brushes for a multi-level dungeon.
        
        Each level is generated at a different Z height with vertical
        connections between them.
        """
        all_brushes = []
        
        for level_idx, layout in enumerate(multi_layout.levels):
            z_level = level_idx * self.settings.level_separation
            level_brushes = self.generate_from_layout(layout, z_level)
            all_brushes.extend(level_brushes)
        
        # TODO: Add vertical connection brushes between levels
        
        return all_brushes
    
    # Helper methods
    
    def _next_brush_id(self) -> int:
        """Get next unique brush ID"""
        self.brush_id_counter += 1
        return self.brush_id_counter
    
    def _get_room_height(self, room_type: str) -> float:
        """Get appropriate height for room type"""
        heights = {
            "standard": self.settings.standard_height,
            "arena": self.settings.arena_height,
            "entrance": self.settings.standard_height,
            "exit": self.settings.standard_height,
            "vertical": self.settings.standard_height * 3,
            "corridor": self.settings.corridor_height
        }
        return heights.get(room_type.lower(), self.settings.standard_height)
    
    def _get_room_textures(self, room_type: str) -> Dict[str, str]:
        """Get texture set for room type"""
        tex = self.settings.texture_set
        
        texture_sets = {
            "arena": {
                "wall": tex.ARENA_WALL,
                "floor": tex.ARENA_FLOOR,
                "ceiling": tex.CEILING_DEFAULT
            },
            "entrance": {
                "wall": tex.ENTRANCE_WALL,
                "floor": tex.ENTRANCE_FLOOR,
                "ceiling": tex.CEILING_DEFAULT
            },
            "corridor": {
                "wall": tex.CORRIDOR_WALL,
                "floor": tex.CORRIDOR_FLOOR,
                "ceiling": tex.CEILING_DEFAULT
            }
        }
        
        return texture_sets.get(room_type.lower(), {
            "wall": tex.WALL_DEFAULT,
            "floor": tex.FLOOR_DEFAULT,
            "ceiling": tex.CEILING_DEFAULT
        })
    
    def _get_corridor_textures(self) -> Dict[str, str]:
        """Get texture set for corridors"""
        tex = self.settings.texture_set
        return {
            "wall": tex.CORRIDOR_WALL,
            "floor": tex.CORRIDOR_FLOOR,
            "ceiling": tex.CEILING_DEFAULT
        }
    
    def _analyze_connections(self, layout: Layout2D):
        """Analyze and cache room connections for door placement"""
        self.room_connections.clear()
        
        # Build room ID to room object mapping
        room_map = {room.room_id: room for room in layout.rooms}
        
        for connection in layout.connections:
            room1_id = connection.start_room.room_id
            room2_id = connection.end_room.room_id
            
            if room1_id not in self.room_connections:
                self.room_connections[room1_id] = set()
            if room2_id not in self.room_connections:
                self.room_connections[room2_id] = set()
            
            self.room_connections[room1_id].add(room2_id)
            self.room_connections[room2_id].add(room1_id)
            
            # Ensure rooms have connection references
            if connection not in connection.start_room.connections:
                connection.start_room.connections.append(connection)
            if connection not in connection.end_room.connections:
                connection.end_room.connections.append(connection)
    
    def _get_room_connection_points(self, room: LayoutRoom) -> List[Tuple[float, float]]:
        """Get connection points where corridors attach to this room.

        Prefer precise corridor rectangles from connection metadata when available
        (stored in world units by the BSP->layout converter). Fall back to
        room-center vector alignment otherwise.
        """
        points: List[Tuple[float, float]] = []
        room_x1, room_y1, room_x2, room_y2 = room.to_quake_bounds()
        room_cx = (room_x1 + room_x2) / 2
        room_cy = (room_y1 + room_y2) / 2

        for conn in getattr(room, 'connections', []) or []:
            rects = conn.metadata.get('path_rects_world') if hasattr(conn, 'metadata') else None
            if rects:
                # For each rect, if it touches or overlaps a room side, record midpoint on that side
                tol = max(8, int(self.settings.corridor_width // 2))
                for r in rects:
                    cx1 = r.get('x', 0)
                    cy1 = r.get('y', 0)
                    cx2 = cx1 + r.get('width', 0)
                    cy2 = cy1 + r.get('height', 0)
                    cxc = (cx1 + cx2) / 2
                    cyc = (cy1 + cy2) / 2
                    # Helper: interval overlap
                    def overlap(a1, a2, b1, b2):
                        return not (a2 < b1 or b2 < a1)
                    # Consider a rect as connecting if its band crosses or tightly approaches the wall line
                    # North wall (y = room_y1)
                    if (cy1 - tol) <= room_y1 <= (cy2 + tol) and overlap(cx1, cx2, room_x1, room_x2):
                        points.append((max(room_x1, min(cxc, room_x2)), room_y1))
                    # South wall (y = room_y2)
                    if (cy1 - tol) <= room_y2 <= (cy2 + tol) and overlap(cx1, cx2, room_x1, room_x2):
                        points.append((max(room_x1, min(cxc, room_x2)), room_y2))
                    # West wall (x = room_x1)
                    if (cx1 - tol) <= room_x1 <= (cx2 + tol) and overlap(cy1, cy2, room_y1, room_y2):
                        points.append((room_x1, max(room_y1, min(cyc, room_y2))))
                    # East wall (x = room_x2)
                    if (cx1 - tol) <= room_x2 <= (cx2 + tol) and overlap(cy1, cy2, room_y1, room_y2):
                        points.append((room_x2, max(room_y1, min(cyc, room_y2))))
            else:
                # Fallback to center-to-center direction
                other = conn.end_room if conn.start_room == room else conn.start_room
                ox1, oy1, ox2, oy2 = other.to_quake_bounds()
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                dx = ocx - room_cx
                dy = ocy - room_cy
                if abs(dx) > abs(dy):
                    points.append((room_x2 if dx > 0 else room_x1, room_cy))
                else:
                    points.append((room_cx, room_y2 if dy > 0 else room_y1))
        # Deduplicate points (avoid duplicate door requests)
        unique = []
        seen = set()
        for p in points:
            key = (int(round(p[0] / 4.0)), int(round(p[1] / 4.0)))  # coarse bucket
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
    
    def _create_simple_door_wall(self, x1: float, y1: float, x2: float, y2: float,
                                floor_z: float, ceiling_z: float, texture: str,
                                orientation: str, door_width: float) -> List[Brush]:
        """Create a wall with a simple central door opening - safer approach"""
        wall_brushes = []
        
        if orientation in ['north', 'south']:
            # Horizontal wall
            wall_length = x2 - x1
            if wall_length <= door_width + 32:  # Too small for door
                return []  # Leave completely open
            
            # Calculate door position in center
            door_center = x1 + wall_length / 2
            door_start = door_center - door_width / 2
            door_end = door_center + door_width / 2
            
            # Left wall segment
            if door_start > x1 + 16:  # Minimum wall segment
                wall_brushes.append(self._create_wall_brush(
                    x1, y1, door_start, y2, floor_z, ceiling_z, texture, orientation
                ))
            
            # Right wall segment
            if door_end < x2 - 16:  # Minimum wall segment
                wall_brushes.append(self._create_wall_brush(
                    door_end, y1, x2, y2, floor_z, ceiling_z, texture, orientation
                ))
            
            # Door lintel (top part)
            door_lintel_z = floor_z + min(self.settings.door_height, ceiling_z - floor_z - 8)
            if door_lintel_z < ceiling_z - 8:
                wall_brushes.append(self._create_wall_brush(
                    door_start, y1, door_end, y2, door_lintel_z, ceiling_z, texture, orientation
                ))
        
        else:
            # Vertical wall
            wall_length = y2 - y1
            if wall_length <= door_width + 32:  # Too small for door
                return []  # Leave completely open
            
            # Calculate door position in center
            door_center = y1 + wall_length / 2
            door_start = door_center - door_width / 2
            door_end = door_center + door_width / 2
            
            # Bottom wall segment
            if door_start > y1 + 16:  # Minimum wall segment
                wall_brushes.append(self._create_wall_brush(
                    x1, y1, x2, door_start, floor_z, ceiling_z, texture, orientation
                ))
            
            # Top wall segment
            if door_end < y2 - 16:  # Minimum wall segment
                wall_brushes.append(self._create_wall_brush(
                    x1, door_end, x2, y2, floor_z, ceiling_z, texture, orientation
                ))
            
            # Door lintel (top part)
            door_lintel_z = floor_z + min(self.settings.door_height, ceiling_z - floor_z - 8)
            if door_lintel_z < ceiling_z - 8:
                wall_brushes.append(self._create_wall_brush(
                    x1, door_start, x2, door_end, door_lintel_z, ceiling_z, texture, orientation
                ))
        
        return wall_brushes
    
    def _create_wall_segment(self, segment: Tuple[float, float, float, float],
                           floor_z: float, ceiling_z: float, texture: str,
                           room_center: Optional[Tuple[float, float]] = None) -> Brush:
        """Create a wall brush for a segment.

        Args:
            segment: (x1, y1, x2, y2) bounds of the wall segment
            floor_z: Floor Z coordinate
            ceiling_z: Ceiling Z coordinate
            texture: Wall texture name
            room_center: Optional (cx, cy) to determine wall orientation relative to room.
                        If not provided, uses geometric midpoint of segment bounds.

        Returns:
            A wall brush with proper orientation.
        """
        x1, y1, x2, y2 = segment
        seg_cx = (x1 + x2) / 2
        seg_cy = (y1 + y2) / 2

        # Use room center if provided, otherwise infer from segment geometry
        if room_center is not None:
            ref_cx, ref_cy = room_center
        else:
            # Fallback: assume segment is on the boundary of some region
            # and determine orientation from the segment's own geometry
            ref_cx, ref_cy = seg_cx, seg_cy

        # Determine orientation based on segment shape and position relative to reference
        if abs(x2 - x1) > abs(y2 - y1):
            # Horizontal wall - determine if north (facing south) or south (facing north)
            # Wall is "north" if it's above the reference point
            if seg_cy < ref_cy:
                return self._create_wall_brush(x1, y1, x2, y1, floor_z, ceiling_z, texture, 'north')
            else:
                return self._create_wall_brush(x1, y2, x2, y2, floor_z, ceiling_z, texture, 'south')
        else:
            # Vertical wall - determine if west (facing east) or east (facing west)
            # Wall is "west" if it's to the left of the reference point
            if seg_cx < ref_cx:
                return self._create_wall_brush(x1, y1, x1, y2, floor_z, ceiling_z, texture, 'west')
            else:
                return self._create_wall_brush(x2, y1, x2, y2, floor_z, ceiling_z, texture, 'east')
    
    def _path_to_segments(self, path: List[Tuple[int, int]], 
                         width: int) -> List[Tuple[float, float, float, float]]:
        """Convert a tile path to rectangular segments in idTech coordinates"""
        segments = []
        
        if not path:
            return segments
        
        # Group consecutive points into segments
        current_segment = [path[0]]
        
        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            
            # Check if points are adjacent
            if abs(curr[0] - prev[0]) <= 1 and abs(curr[1] - prev[1]) <= 1:
                current_segment.append(curr)
            else:
                # Start new segment
                if current_segment:
                    seg_bounds = self._segment_to_bounds(current_segment, width)
                    if seg_bounds:
                        segments.append(seg_bounds)
                current_segment = [curr]
        
        # Add final segment
        if current_segment:
            seg_bounds = self._segment_to_bounds(current_segment, width)
            if seg_bounds:
                segments.append(seg_bounds)
        
        return segments
    
    def _segment_to_bounds(self, points: List[Tuple[int, int]], 
                          width: int) -> Optional[Tuple[float, float, float, float]]:
        """Convert a list of tile points to a bounding box in idTech coordinates"""
        if not points:
            return None
        
        # Find min/max tile coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        min_x = min(x_coords) * self.settings.grid_size
        max_x = (max(x_coords) + width) * self.settings.grid_size
        min_y = min(y_coords) * self.settings.grid_size
        max_y = (max(y_coords) + width) * self.settings.grid_size
        
        return (min_x, min_y, max_x, max_y)
