"""
Unified Portal System for idTech Map Generator.

Provides standardized portal generation across ALL primitives (halls AND rooms).
This ensures consistent portal dimensions for proper connections between geometry.

Portal sizes:
- STANDARD: 96x88 (width x height) - Default for halls and rooms
- Player bounding box: 32x32x56 - Minimum portal: 48x64

CLAUDE.md BINDING: This module implements Section 14 (Hall Portal Alignment)
and Section 15 (Room Portal Handling) requirements for unified portal handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Callable, Optional, Dict, ClassVar

from quake_levelgenerator.src.conversion.map_writer import Brush


# ==============================================================================
# PORTAL DIMENSION CONSTANTS
# ==============================================================================

# Standard portal dimensions (width x height)
# These are the canonical values that MUST be used for all portal generation
PORTAL_WIDTH = 96
PORTAL_HEIGHT = 88

# Future portal sizes (not yet implemented)
PORTAL_NARROW_WIDTH = 64
PORTAL_NARROW_HEIGHT = 80
PORTAL_GRAND_WIDTH = 128
PORTAL_GRAND_HEIGHT = 112

# idTech player constraints
PLAYER_WIDTH = 32
PLAYER_DEPTH = 32
PLAYER_HEIGHT = 56
MIN_PORTAL_WIDTH = 48   # Minimum for player passage
MIN_PORTAL_HEIGHT = 64  # Minimum for player passage


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class PortalDirection(Enum):
    """Cardinal direction for portal facing."""
    NORTH = "north"  # +Y direction
    SOUTH = "south"  # -Y direction
    EAST = "east"    # +X direction
    WEST = "west"    # -X direction

    def opposite(self) -> 'PortalDirection':
        """Return the opposite direction."""
        opposites = {
            PortalDirection.NORTH: PortalDirection.SOUTH,
            PortalDirection.SOUTH: PortalDirection.NORTH,
            PortalDirection.EAST: PortalDirection.WEST,
            PortalDirection.WEST: PortalDirection.EAST,
        }
        return opposites[self]


# ==============================================================================
# PORTAL TAG SYSTEM
# ==============================================================================
# Tags mark exact portal positions during generation.
# Modules create tags at their actual portal centers, then the layout generator
# validates that connected portals have matching tags.

@dataclass
class PortalTag:
    """Tag marking exact position of a portal or connection point.

    Created by modules during generate() at actual portal positions.
    Used by TagRegistry to validate connected portals interlock correctly.

    The key insight is that modules know exactly where their portals are,
    so they register tags at those positions. The layout generator then
    transforms tags to world space and validates alignment.
    """
    primitive_id: str              # ID of primitive that created this tag (set by layout gen)
    portal_id: str                 # ID of portal (entrance, exit, side, front, back, etc.)
    center_x: float                # X coordinate (local space, transformed to world by layout gen)
    center_y: float                # Y coordinate (local space, transformed to world by layout gen)
    center_z: float                # Z coordinate (floor level of portal opening)
    direction: PortalDirection     # Which way portal faces
    width: float = PORTAL_WIDTH    # Portal opening width
    height: float = PORTAL_HEIGHT  # Portal opening height
    tag_type: str = "portal"       # "portal" or "stair_connection"

    # Class-level tolerance for position matching
    POSITION_TOLERANCE: ClassVar[float] = 2.0
    DIMENSION_TOLERANCE: ClassVar[float] = 0.5

    def matches(self, other: 'PortalTag') -> bool:
        """Check if tags interlock (opposite directions, aligned positions).

        Two tags match if:
        1. They face opposite directions
        2. Their positions are within POSITION_TOLERANCE
        3. Their dimensions match within DIMENSION_TOLERANCE

        Args:
            other: Another PortalTag to compare against

        Returns:
            True if tags interlock correctly
        """
        # Check opposite directions
        if self.direction.opposite() != other.direction:
            return False

        # Check position alignment
        if abs(self.center_x - other.center_x) > self.POSITION_TOLERANCE:
            return False
        if abs(self.center_y - other.center_y) > self.POSITION_TOLERANCE:
            return False
        if abs(self.center_z - other.center_z) > self.POSITION_TOLERANCE:
            return False

        # Check dimension match
        if abs(self.width - other.width) > self.DIMENSION_TOLERANCE:
            return False
        if abs(self.height - other.height) > self.DIMENSION_TOLERANCE:
            return False

        return True

    def get_mismatch_details(self, other: 'PortalTag') -> str:
        """Get detailed description of why tags don't match."""
        issues = []

        if self.direction.opposite() != other.direction:
            issues.append(f"Direction: {self.direction.value} vs {other.direction.value} "
                         f"(expected opposite)")

        pos_diff_x = abs(self.center_x - other.center_x)
        pos_diff_y = abs(self.center_y - other.center_y)
        pos_diff_z = abs(self.center_z - other.center_z)

        if pos_diff_x > self.POSITION_TOLERANCE:
            issues.append(f"X offset: {pos_diff_x:.1f} units (self: {self.center_x:.1f}, "
                         f"other: {other.center_x:.1f})")
        if pos_diff_y > self.POSITION_TOLERANCE:
            issues.append(f"Y offset: {pos_diff_y:.1f} units (self: {self.center_y:.1f}, "
                         f"other: {other.center_y:.1f})")
        if pos_diff_z > self.POSITION_TOLERANCE:
            issues.append(f"Z offset: {pos_diff_z:.1f} units (self: {self.center_z:.1f}, "
                         f"other: {other.center_z:.1f})")

        if abs(self.width - other.width) > self.DIMENSION_TOLERANCE:
            issues.append(f"Width: self {self.width:.0f}, other {other.width:.0f}")
        if abs(self.height - other.height) > self.DIMENSION_TOLERANCE:
            issues.append(f"Height: self {self.height:.0f}, other {other.height:.0f}")

        return "; ".join(issues) if issues else "No issues"

    def transformed(self, offset_x: float, offset_y: float, offset_z: float,
                   rotation: int = 0) -> 'PortalTag':
        """Return a new tag with coordinates transformed to world space.

        Args:
            offset_x, offset_y, offset_z: World position offset
            rotation: Rotation in degrees (0, 90, 180, 270)

        Returns:
            New PortalTag with transformed coordinates and direction
        """
        # Rotate coordinates
        x, y = self.center_x, self.center_y
        if rotation == 90:
            x, y = y, -x
        elif rotation == 180:
            x, y = -x, -y
        elif rotation == 270:
            x, y = -y, x

        # Rotate direction
        dir_rotations = {
            0: self.direction,
            90: {
                PortalDirection.NORTH: PortalDirection.EAST,
                PortalDirection.EAST: PortalDirection.SOUTH,
                PortalDirection.SOUTH: PortalDirection.WEST,
                PortalDirection.WEST: PortalDirection.NORTH,
            }[self.direction],
            180: self.direction.opposite(),
            270: {
                PortalDirection.NORTH: PortalDirection.WEST,
                PortalDirection.WEST: PortalDirection.SOUTH,
                PortalDirection.SOUTH: PortalDirection.EAST,
                PortalDirection.EAST: PortalDirection.NORTH,
            }[self.direction],
        }
        new_direction = dir_rotations.get(rotation, self.direction)

        return PortalTag(
            primitive_id=self.primitive_id,
            portal_id=self.portal_id,
            center_x=x + offset_x,
            center_y=y + offset_y,
            center_z=self.center_z + offset_z,
            direction=new_direction,
            width=self.width,
            height=self.height,
            tag_type=self.tag_type,
        )


@dataclass
class PortalSpec:
    """Specification for a portal opening.

    Used to configure portal generation with unified dimensions.

    When is_secret is True, the portal generates a solid wall with CLIP texture
    instead of an open portal. This creates a walk-through secret passage.
    """
    enabled: bool = True
    width: float = PORTAL_WIDTH
    height: float = PORTAL_HEIGHT
    is_secret: bool = False  # When True, generates CLIP wall instead of open portal

    # CLIP texture for walk-through walls
    CLIP_TEXTURE: str = "CLIP"

    @classmethod
    def standard(cls, enabled: bool = True) -> 'PortalSpec':
        """Create a standard-sized portal spec."""
        return cls(enabled=enabled, width=PORTAL_WIDTH, height=PORTAL_HEIGHT)

    @classmethod
    def disabled(cls) -> 'PortalSpec':
        """Create a disabled portal spec (solid wall)."""
        return cls(enabled=False)

    @classmethod
    def secret(cls) -> 'PortalSpec':
        """Create a secret portal spec (CLIP wall for walk-through)."""
        return cls(enabled=True, is_secret=True)


@dataclass
class PortalWorldPosition:
    """World position and dimensions of a generated portal.

    Used for validating that connected portals align correctly.
    """
    primitive_id: str              # ID of the primitive containing this portal
    portal_id: str                 # ID of the portal within the primitive
    center_x: float                # X coordinate of portal center
    center_y: float                # Y coordinate of portal center
    center_z: float                # Z coordinate of portal center (bottom of opening)
    width: float                   # Portal opening width
    height: float                  # Portal opening height
    direction: PortalDirection     # Which way the portal faces

    # Tolerance for position matching (in idTech units)
    POSITION_TOLERANCE = 2.0
    DIMENSION_TOLERANCE = 0.5

    def matches(self, other: 'PortalWorldPosition') -> bool:
        """Check if this portal aligns with another portal.

        Portals match if:
        1. Their centers are within POSITION_TOLERANCE
        2. Their dimensions match within DIMENSION_TOLERANCE
        3. They face opposite directions
        """
        # Check opposite directions
        if self.direction.opposite() != other.direction:
            return False

        # Check position alignment
        if abs(self.center_x - other.center_x) > self.POSITION_TOLERANCE:
            return False
        if abs(self.center_y - other.center_y) > self.POSITION_TOLERANCE:
            return False
        if abs(self.center_z - other.center_z) > self.POSITION_TOLERANCE:
            return False

        # Check dimension match
        if abs(self.width - other.width) > self.DIMENSION_TOLERANCE:
            return False
        if abs(self.height - other.height) > self.DIMENSION_TOLERANCE:
            return False

        return True

    def get_mismatch_details(self, other: 'PortalWorldPosition') -> str:
        """Get detailed description of why portals don't match."""
        issues = []

        if self.direction.opposite() != other.direction:
            issues.append(f"Direction: {self.direction.value} vs {other.direction.value} "
                         f"(expected opposite)")

        pos_diff_x = abs(self.center_x - other.center_x)
        pos_diff_y = abs(self.center_y - other.center_y)
        pos_diff_z = abs(self.center_z - other.center_z)

        if pos_diff_x > self.POSITION_TOLERANCE:
            issues.append(f"X offset: {pos_diff_x:.1f} units (expected: {self.center_x:.1f}, "
                         f"actual: {other.center_x:.1f})")
        if pos_diff_y > self.POSITION_TOLERANCE:
            issues.append(f"Y offset: {pos_diff_y:.1f} units (expected: {self.center_y:.1f}, "
                         f"actual: {other.center_y:.1f})")
        if pos_diff_z > self.POSITION_TOLERANCE:
            issues.append(f"Z offset: {pos_diff_z:.1f} units (expected: {self.center_z:.1f}, "
                         f"actual: {other.center_z:.1f})")

        if abs(self.width - other.width) > self.DIMENSION_TOLERANCE:
            issues.append(f"Width: expected {self.width:.0f}, got {other.width:.0f}")
        if abs(self.height - other.height) > self.DIMENSION_TOLERANCE:
            issues.append(f"Height: expected {self.height:.0f}, got {other.height:.0f}")

        return "; ".join(issues) if issues else "No issues"


@dataclass
class PortalMismatch:
    """Details about a portal alignment mismatch."""
    primitive_a_id: str
    portal_a_id: str
    primitive_b_id: str
    portal_b_id: str
    position_a: PortalWorldPosition
    position_b: PortalWorldPosition

    def __str__(self) -> str:
        details = self.position_a.get_mismatch_details(self.position_b)
        return (f"Portal Alignment Error: {self.primitive_a_id}.{self.portal_a_id} "
                f"does not match {self.primitive_b_id}.{self.portal_b_id} - {details}")


# ==============================================================================
# PORTAL WALL GENERATION
# ==============================================================================

# Type alias for box creation function (matches GeometricPrimitive._box signature)
BoxFunc = Callable[[float, float, float, float, float, float], Brush]
# Type alias for box with texture function
BoxWithTextureFunc = Callable[[float, float, float, float, float, float, str], Brush]


def generate_portal_wall(
    box_func: BoxFunc,
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    portal_spec: PortalSpec,
    portal_axis: str = "x",
    portal_center: Optional[float] = None,
    box_with_texture_func: Optional[BoxWithTextureFunc] = None,
) -> Tuple[List[Brush], Optional[PortalWorldPosition]]:
    """Generate a wall with an optional portal opening.

    Creates unified portal geometry following CLAUDE.md sealed geometry rules:
    - When portal is enabled: creates 3 brushes (left, right, lintel)
    - When portal is disabled: creates 1 solid wall brush
    - When portal is_secret: creates solid CLIP-textured wall for walk-through

    Args:
        box_func: Function to create box brushes (typically self._box from primitive)
        x1, y1, z1: Minimum corner of wall bounds
        x2, y2, z2: Maximum corner of wall bounds
        portal_spec: Portal specification (enabled, width, height, is_secret)
        portal_axis: "x" if portal spans X direction, "y" if spans Y direction
        portal_center: Center of portal along the axis (default: wall center)
        box_with_texture_func: Optional function for creating CLIP-textured boxes

    Returns:
        Tuple of (brush_list, portal_world_position)
        portal_world_position is None if portal is disabled

    BINDING: Per CLAUDE.md Section 5, portals create 3-piece walls when enabled.
    """
    brushes: List[Brush] = []
    portal_position: Optional[PortalWorldPosition] = None

    if not portal_spec.enabled:
        # Solid wall - no portal
        brushes.append(box_func(x1, y1, z1, x2, y2, z2))
        return brushes, None

    # Handle secret portal: create CLIP wall instead of open portal
    # Still return PortalWorldPosition for alignment validation
    if portal_spec.is_secret and box_with_texture_func:
        brushes.append(box_with_texture_func(x1, y1, z1, x2, y2, z2, PortalSpec.CLIP_TEXTURE))
        # Calculate portal position for validation even for secret portals
        pw2 = portal_spec.width / 2
        ph = portal_spec.height
        if portal_axis == "x":
            center = portal_center if portal_center is not None else (x1 + x2) / 2
            portal_position = PortalWorldPosition(
                primitive_id="",
                portal_id="",
                center_x=center,
                center_y=(y1 + y2) / 2,
                center_z=z1 + ph / 2,
                width=portal_spec.width,
                height=portal_spec.height,
                direction=PortalDirection.SOUTH if y1 < y2 else PortalDirection.NORTH,
            )
        else:
            center = portal_center if portal_center is not None else (y1 + y2) / 2
            portal_position = PortalWorldPosition(
                primitive_id="",
                portal_id="",
                center_x=(x1 + x2) / 2,
                center_y=center,
                center_z=z1 + ph / 2,
                width=portal_spec.width,
                height=portal_spec.height,
                direction=PortalDirection.WEST if x1 < x2 else PortalDirection.EAST,
            )
        return brushes, portal_position

    # Calculate portal bounds
    pw2 = portal_spec.width / 2
    ph = portal_spec.height
    wall_height = z2 - z1

    # Validate portal fits within wall
    wall_span = (x2 - x1) if portal_axis == "x" else (y2 - y1)
    if portal_spec.width >= wall_span:
        # Portal too wide for wall - create solid wall to prevent geometry leak
        brushes.append(box_func(x1, y1, z1, x2, y2, z2))
        return brushes, None

    if portal_axis == "x":
        # Portal spans X direction (for front/back walls along Y)
        center = portal_center if portal_center is not None else (x1 + x2) / 2
        portal_x1 = center - pw2
        portal_x2 = center + pw2

        # Left piece (x1 to portal left edge)
        if portal_x1 > x1:
            brushes.append(box_func(x1, y1, z1, portal_x1, y2, z2))

        # Right piece (portal right edge to x2)
        if portal_x2 < x2:
            brushes.append(box_func(portal_x2, y1, z1, x2, y2, z2))

        # Lintel above portal (if portal doesn't reach ceiling)
        if ph < wall_height:
            brushes.append(box_func(portal_x1, y1, z1 + ph, portal_x2, y2, z2))

        # Create portal position for validation
        portal_position = PortalWorldPosition(
            primitive_id="",  # Set by caller
            portal_id="",     # Set by caller
            center_x=center,
            center_y=(y1 + y2) / 2,
            center_z=z1 + ph / 2,  # Center of portal opening
            width=portal_spec.width,
            height=portal_spec.height,
            direction=PortalDirection.SOUTH if y1 < y2 else PortalDirection.NORTH,
        )

    else:
        # Portal spans Y direction (for left/right walls along X)
        center = portal_center if portal_center is not None else (y1 + y2) / 2
        portal_y1 = center - pw2
        portal_y2 = center + pw2

        # Front piece (y1 to portal front edge)
        if portal_y1 > y1:
            brushes.append(box_func(x1, y1, z1, x2, portal_y1, z2))

        # Back piece (portal back edge to y2)
        if portal_y2 < y2:
            brushes.append(box_func(x1, portal_y2, z1, x2, y2, z2))

        # Lintel above portal (if portal doesn't reach ceiling)
        if ph < wall_height:
            brushes.append(box_func(x1, portal_y1, z1 + ph, x2, portal_y2, z2))

        # Create portal position for validation
        portal_position = PortalWorldPosition(
            primitive_id="",  # Set by caller
            portal_id="",     # Set by caller
            center_x=(x1 + x2) / 2,
            center_y=center,
            center_z=z1 + ph / 2,  # Center of portal opening
            width=portal_spec.width,
            height=portal_spec.height,
            direction=PortalDirection.WEST if x1 < x2 else PortalDirection.EAST,
        )

    return brushes, portal_position


def validate_portal_alignment(
    connections: List[Tuple[PortalWorldPosition, PortalWorldPosition]]
) -> List[PortalMismatch]:
    """Validate that all connected portals align correctly.

    Args:
        connections: List of (portal_a_position, portal_b_position) tuples

    Returns:
        List of PortalMismatch objects for any misaligned portals
    """
    mismatches: List[PortalMismatch] = []

    for pos_a, pos_b in connections:
        if not pos_a.matches(pos_b):
            mismatches.append(PortalMismatch(
                primitive_a_id=pos_a.primitive_id,
                portal_a_id=pos_a.portal_id,
                primitive_b_id=pos_b.primitive_id,
                portal_b_id=pos_b.portal_id,
                position_a=pos_a,
                position_b=pos_b,
            ))

    return mismatches
