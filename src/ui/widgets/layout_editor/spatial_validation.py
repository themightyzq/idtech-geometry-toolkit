"""
Spatial validation system for 3D interpenetration detection in multi-floor layouts.

Provides axis-aligned bounding box (AABB) based collision detection to prevent
primitives at different Z-levels from overlapping in 3D space.

Problem solved:
- Room at z=0, height=128, wall_thickness=16 -> geometry Z in [-16, 144]
- Room at z=128, height=128 -> geometry Z in [112, 272]
- This creates a 32-unit overlap in Z range [112, 144] - detected by this system!

Per CLAUDE.md Section 5 (Sealed Geometry Rules):
- Floor/ceiling extend by wall_thickness (t) in ALL directions
- This means geometry Z range is [z_offset - t, z_offset + height + t]
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set

from .data_model import DungeonLayout, PlacedPrimitive, PrimitiveFootprint, Connection
from .palette_widget import PRIMITIVE_FOOTPRINTS


# =============================================================================
# PRIMITIVE HEIGHT LOOKUP TABLE
# =============================================================================
# Heights for all 25 primitives from CLAUDE.md
# Format: 'PrimitiveType': (default_height, height_param_name, wall_thickness)
# If height_param_name is None, height is fixed at default_height

PRIMITIVE_HEIGHTS: Dict[str, Tuple[float, Optional[str], float]] = {
    # === HALLS (6 total) ===
    # Standard hall height is 128, wall_thickness=16
    'StraightHall': (128.0, 'hall_height', 16.0),
    'SecretHall': (128.0, 'hall_height', 16.0),
    'TJunction': (128.0, 'hall_height', 16.0),
    'Crossroads': (128.0, 'hall_height', 16.0),
    'SquareCorner': (128.0, 'hall_height', 16.0),
    'VerticalStairHall': (160.0, None, 16.0),  # Connects adjacent floors (height_change=160)

    # === ROOMS (19 total) ===
    # Most rooms use height=128 and t=16
    'Sanctuary': (192.0, 'nave_height', 16.0),  # Nave is taller
    'Tomb': (128.0, 'height', 16.0),
    'Tower': (256.0, 'height', 16.0),  # Towers are tall
    'Chamber': (128.0, 'height', 16.0),
    'Storage': (128.0, 'height', 16.0),
    'GreatHall': (192.0, 'height', 16.0),  # Great halls are taller
    'Prison': (128.0, 'height', 16.0),
    'Armory': (128.0, 'height', 16.0),
    'Cistern': (128.0, 'height', 16.0),
    'Stronghold': (192.0, 'height', 32.0),  # Thick walls
    'Courtyard': (128.0, 'height', 16.0),
    'Arena': (160.0, 'height', 16.0),  # Arenas are taller
    'Laboratory': (128.0, 'height', 16.0),
    'Vault': (128.0, 'height', 24.0),  # Thick walls
    'Barracks': (128.0, 'height', 16.0),
    'Shrine': (128.0, 'height', 16.0),
    'Pit': (128.0, 'height', 16.0),
    'Antechamber': (128.0, 'height', 16.0),
    'SecretChamber': (128.0, 'height', 16.0),

    # === STRUCTURAL (5 total) - placed inside rooms ===
    'StraightStaircase': (128.0, 'height_change', 8.0),
    'Arch': (96.0, 'arch_height', 8.0),
    'Pillar': (128.0, 'height', 8.0),
    'Buttress': (128.0, 'height', 8.0),
    'Battlement': (48.0, 'height', 8.0),

    # === CONNECTIVE (4 total) - placed inside rooms ===
    'Bridge': (128.0, 'height', 8.0),
    'Platform': (64.0, 'platform_height', 8.0),
    'Rampart': (96.0, 'height', 8.0),
    'Gallery': (128.0, 'height', 8.0),
}

# Overlap tolerance per CLAUDE.md Section 5: "Brush overlaps: >= 8 units at corners/junctions"
OVERLAP_TOLERANCE = 8.0


class CollisionType(Enum):
    """Type of spatial collision detected."""
    FLOOR_CEILING = "floor_ceiling"  # Most common in multi-floor layouts
    WALL_WALL = "wall_wall"          # XY overlaps at same or nearby Z
    FULL_OVERLAP = "full_overlap"    # Complete 3D overlap
    INTENTIONAL = "intentional"      # Connected primitives (OK)


@dataclass
class AABB:
    """Axis-Aligned Bounding Box for primitive geometry."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def intersects(self, other: 'AABB') -> bool:
        """Check if this AABB intersects another AABB."""
        # Two AABBs intersect if they overlap on all three axes
        return (
            self.min_x < other.max_x and self.max_x > other.min_x and
            self.min_y < other.max_y and self.max_y > other.min_y and
            self.min_z < other.max_z and self.max_z > other.min_z
        )

    def intersection_volume(self, other: 'AABB') -> float:
        """Calculate the intersection volume with another AABB."""
        if not self.intersects(other):
            return 0.0

        # Calculate overlap on each axis
        overlap_x = min(self.max_x, other.max_x) - max(self.min_x, other.min_x)
        overlap_y = min(self.max_y, other.max_y) - max(self.min_y, other.min_y)
        overlap_z = min(self.max_z, other.max_z) - max(self.min_z, other.min_z)

        return max(0.0, overlap_x) * max(0.0, overlap_y) * max(0.0, overlap_z)

    def z_overlap_range(self, other: 'AABB') -> Optional[Tuple[float, float]]:
        """Get the Z range of overlap with another AABB, if any."""
        if not self.intersects(other):
            return None

        overlap_min_z = max(self.min_z, other.min_z)
        overlap_max_z = min(self.max_z, other.max_z)

        if overlap_min_z < overlap_max_z:
            return (overlap_min_z, overlap_max_z)
        return None

    @property
    def volume(self) -> float:
        """Calculate the volume of this AABB."""
        return (
            (self.max_x - self.min_x) *
            (self.max_y - self.min_y) *
            (self.max_z - self.min_z)
        )


@dataclass
class SpatialCollision:
    """A detected spatial collision between two primitives."""
    primitive_a_id: str
    primitive_b_id: str
    collision_type: CollisionType
    overlap_volume: float
    overlap_z_range: Tuple[float, float]
    is_critical: bool
    message: str
    suggestion: str

    def __str__(self) -> str:
        return (
            f"{self.collision_type.value}: {self.message} "
            f"(Z range: {self.overlap_z_range[0]:.0f}-{self.overlap_z_range[1]:.0f})"
        )


class SpatialValidator:
    """
    Validates 3D spatial relationships between primitives in a layout.

    Detects interpenetration between primitives at different Z-levels
    that would cause brush collision in the generated map.
    """

    def __init__(self, layout: DungeonLayout):
        """Initialize with a dungeon layout to validate."""
        self._layout = layout
        self._primitive_bounds: Dict[str, AABB] = {}
        self._connected_pairs: Set[Tuple[str, str]] = set()

        # Pre-compute data
        self._compute_all_bounds()
        self._build_connection_set()

    def _compute_all_bounds(self) -> None:
        """Compute AABBs for all primitives in the layout."""
        for prim_id, prim in self._layout.primitives.items():
            footprint = prim.footprint or PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
            if footprint:
                self._primitive_bounds[prim_id] = self.compute_primitive_aabb(
                    prim, footprint
                )

    def _build_connection_set(self) -> None:
        """Build set of connected primitive pairs for intentional overlap detection."""
        for conn in self._layout.connections:
            # Store both orderings for easy lookup
            pair1 = (conn.primitive_a_id, conn.primitive_b_id)
            pair2 = (conn.primitive_b_id, conn.primitive_a_id)
            self._connected_pairs.add(pair1)
            self._connected_pairs.add(pair2)

    def compute_primitive_aabb(
        self,
        prim: PlacedPrimitive,
        footprint: PrimitiveFootprint
    ) -> AABB:
        """
        Compute the axis-aligned bounding box for a primitive's geometry.

        Per CLAUDE.md Section 5 (Sealed Geometry Rules):
        - Floor extends DOWN by wall_thickness (t)
        - Ceiling extends UP by wall_thickness (t)
        - Walls span the full floor/ceiling extent

        Args:
            prim: The placed primitive
            footprint: The primitive's footprint definition

        Returns:
            AABB representing the primitive's 3D bounds
        """
        grid_size = self._layout.grid_size

        # Get height info from lookup table
        height_info = PRIMITIVE_HEIGHTS.get(prim.primitive_type)
        if height_info:
            default_height, height_param, wall_thickness = height_info
            # Check if height is overridden in parameters
            if height_param and height_param in prim.parameters:
                height = float(prim.parameters[height_param])
            else:
                height = default_height
            t = wall_thickness
        else:
            # Default values if primitive not in table
            height = 128.0
            t = 16.0

        # Get rotated footprint size
        fp_width, fp_depth = footprint.rotated_size(prim.rotation)

        # XY bounds from footprint cells
        # NOTE: Per CLAUDE.md Section 5, floor/ceiling extend by t in XY,
        # BUT walls stay within the cell boundary. For collision detection,
        # we use cell-based bounds to avoid false positives between adjacent
        # but non-connected primitives.
        min_x = prim.origin_cell.x * grid_size
        max_x = (prim.origin_cell.x + fp_width) * grid_size
        min_y = prim.origin_cell.y * grid_size
        max_y = (prim.origin_cell.y + fp_depth) * grid_size

        # Z bounds from z_offset and sealed geometry extensions
        # Floor extends DOWN by t, ceiling extends UP by t
        # This is the critical dimension for cross-floor collision detection
        min_z = prim.z_offset - t
        max_z = prim.z_offset + height + t

        return AABB(min_x, min_y, min_z, max_x, max_y, max_z)

    def check_pair_collision(
        self,
        prim_a_id: str,
        prim_b_id: str
    ) -> Optional[SpatialCollision]:
        """
        Check if two primitives have a spatial collision.

        Args:
            prim_a_id: First primitive ID
            prim_b_id: Second primitive ID

        Returns:
            SpatialCollision if collision detected, None otherwise
        """
        bounds_a = self._primitive_bounds.get(prim_a_id)
        bounds_b = self._primitive_bounds.get(prim_b_id)

        if not bounds_a or not bounds_b:
            return None

        # Check for intersection
        if not bounds_a.intersects(bounds_b):
            return None

        # Get primitives for context
        prim_a = self._layout.primitives.get(prim_a_id)
        prim_b = self._layout.primitives.get(prim_b_id)

        if not prim_a or not prim_b:
            return None

        # Calculate overlap details
        overlap_volume = bounds_a.intersection_volume(bounds_b)
        z_range = bounds_a.z_overlap_range(bounds_b)

        if z_range is None:
            return None

        # Determine collision type
        z_a = prim_a.z_offset
        z_b = prim_b.z_offset

        # Check if this is a Z-level difference issue
        z_diff = abs(z_a - z_b)

        if z_diff < 2:
            # Same Z level - likely XY overlap (should be caught by 2D overlap check)
            collision_type = CollisionType.WALL_WALL
        elif z_diff < 200:
            # Different Z levels - floor/ceiling collision
            collision_type = CollisionType.FLOOR_CEILING
        else:
            # Large Z difference with overlap - full overlap
            collision_type = CollisionType.FULL_OVERLAP

        # Check if critical (exceeds tolerance)
        is_critical = overlap_volume > (OVERLAP_TOLERANCE ** 3)

        # Generate message and suggestion
        type_a = prim_a.primitive_type
        type_b = prim_b.primitive_type
        message = (
            f"{type_a} (z={z_a:.0f}) collides with "
            f"{type_b} (z={z_b:.0f})"
        )

        # Calculate minimum safe Z separation
        height_a_info = PRIMITIVE_HEIGHTS.get(type_a, (128.0, None, 16.0))
        height_b_info = PRIMITIVE_HEIGHTS.get(type_b, (128.0, None, 16.0))
        t_a = height_a_info[2]
        t_b = height_b_info[2]
        height_a = prim_a.parameters.get(height_a_info[1], height_a_info[0]) if height_a_info[1] else height_a_info[0]

        min_separation = height_a + t_a + t_b
        suggestion = f"Increase Z separation to {min_separation:.0f}+ units"

        return SpatialCollision(
            primitive_a_id=prim_a_id,
            primitive_b_id=prim_b_id,
            collision_type=collision_type,
            overlap_volume=overlap_volume,
            overlap_z_range=z_range,
            is_critical=is_critical,
            message=message,
            suggestion=suggestion,
        )

    def is_intentional_overlap(
        self,
        prim_a_id: str,
        prim_b_id: str,
        collision: SpatialCollision
    ) -> bool:
        """
        Check if a collision is actually an intentional overlap.

        Intentional overlaps include:
        1. Connected primitives at same Z-level (sealed geometry overlap at junction)
        2. VerticalStairHall connections (designed to span floors)
        3. Adjacent primitives with same Z (wall extensions naturally overlap)

        Args:
            prim_a_id: First primitive ID
            prim_b_id: Second primitive ID
            collision: The detected collision

        Returns:
            True if the overlap is intentional and should not be reported
        """
        # Check if primitives are connected
        if (prim_a_id, prim_b_id) not in self._connected_pairs:
            return False

        prim_a = self._layout.primitives.get(prim_a_id)
        prim_b = self._layout.primitives.get(prim_b_id)

        if not prim_a or not prim_b:
            return False

        # VerticalStairHall is always OK - designed to span floors
        if prim_a.primitive_type == 'VerticalStairHall':
            return True
        if prim_b.primitive_type == 'VerticalStairHall':
            return True

        # Same Z-level connected primitives - sealed geometry overlap is expected
        # Per CLAUDE.md Section 5: Floor/ceiling extend by wall_thickness (t) in ALL directions
        # This means adjacent connected primitives at the same Z will have overlapping
        # sealed geometry extensions, which is intentional and correct.
        z_diff = abs(prim_a.z_offset - prim_b.z_offset)
        if z_diff < 2:
            # Connected at same Z level - overlap is intentional
            # The sealed geometry extensions (wall thickness) naturally create
            # overlapping volumes at junctions. This is by design.
            return True

        return False

    def validate_all(self) -> List[SpatialCollision]:
        """
        Validate all primitive pairs for spatial collisions.

        Returns:
            List of SpatialCollision objects for detected collisions
        """
        collisions: List[SpatialCollision] = []
        prim_ids = list(self._layout.primitives.keys())

        # Check all pairs (O(n^2), but layouts are typically small)
        for i, prim_a_id in enumerate(prim_ids):
            for prim_b_id in prim_ids[i + 1:]:
                collision = self.check_pair_collision(prim_a_id, prim_b_id)
                if collision:
                    # Filter out intentional overlaps
                    if not self.is_intentional_overlap(
                        prim_a_id, prim_b_id, collision
                    ):
                        collisions.append(collision)

        return collisions

    def get_primitive_bounds(self, prim_id: str) -> Optional[AABB]:
        """Get the cached AABB for a primitive."""
        return self._primitive_bounds.get(prim_id)


def compute_minimum_z_separation(
    prim_type_a: str,
    prim_type_b: str,
    params_a: Optional[Dict] = None,
    params_b: Optional[Dict] = None,
) -> float:
    """
    Calculate the minimum Z-offset separation to avoid collision.

    For two primitives at same XY to not collide:
    z_separation >= height_a + wall_thickness_a + wall_thickness_b

    Args:
        prim_type_a: Type of first primitive
        prim_type_b: Type of second primitive
        params_a: Optional parameters for first primitive
        params_b: Optional parameters for second primitive

    Returns:
        Minimum Z separation in units
    """
    # Get height info for primitive A
    info_a = PRIMITIVE_HEIGHTS.get(prim_type_a, (128.0, None, 16.0))
    default_height_a, height_param_a, t_a = info_a

    # Get actual height from params or use default
    if params_a and height_param_a and height_param_a in params_a:
        height_a = float(params_a[height_param_a])
    else:
        height_a = default_height_a

    # Get wall thickness for primitive B
    info_b = PRIMITIVE_HEIGHTS.get(prim_type_b, (128.0, None, 16.0))
    t_b = info_b[2]

    # Minimum separation = height_a + t_a (ceiling extension) + t_b (floor extension)
    return height_a + t_a + t_b


def check_placement_collision(
    layout: DungeonLayout,
    new_prim: PlacedPrimitive,
    footprint: PrimitiveFootprint,
) -> Optional[SpatialCollision]:
    """
    Check if placing a new primitive would cause a spatial collision.

    Used during random layout generation to validate placements before
    adding them to the layout.

    Args:
        layout: The current dungeon layout
        new_prim: The primitive being placed (not yet in layout)
        footprint: The footprint for the new primitive

    Returns:
        SpatialCollision if collision detected, None if placement is valid
    """
    # Create temporary validator without the new primitive
    validator = SpatialValidator(layout)

    # Compute bounds for the new primitive
    new_bounds = validator.compute_primitive_aabb(new_prim, footprint)

    # Check against all existing primitives
    for prim_id, prim in layout.primitives.items():
        existing_bounds = validator.get_primitive_bounds(prim_id)
        if not existing_bounds:
            continue

        # Check for intersection
        if new_bounds.intersects(existing_bounds):
            # Only flag if Z levels are different (same-Z handled by 2D overlap)
            z_diff = abs(new_prim.z_offset - prim.z_offset)
            if z_diff > 2:
                overlap_volume = new_bounds.intersection_volume(existing_bounds)
                z_range = new_bounds.z_overlap_range(existing_bounds)

                if z_range and overlap_volume > OVERLAP_TOLERANCE ** 3:
                    return SpatialCollision(
                        primitive_a_id=new_prim.id,
                        primitive_b_id=prim_id,
                        collision_type=CollisionType.FLOOR_CEILING,
                        overlap_volume=overlap_volume,
                        overlap_z_range=z_range,
                        is_critical=True,
                        message=f"Placement would collide with {prim.primitive_type}",
                        suggestion="Choose different cell or Z-level",
                    )

    return None
