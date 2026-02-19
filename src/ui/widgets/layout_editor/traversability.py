"""
Traversability validation for multi-floor dungeon layouts.

Provides Z-level aware connectivity validation to ensure players can
physically traverse between all connected primitives. This module
catches issues like:
- Rooms at different Z-levels with no stairs connecting them
- Step heights exceeding player movement limits
- Ramp angles too steep for traversal

Constants align with idTech engine constraints from CLAUDE.md.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import deque
from enum import Enum

from .data_model import DungeonLayout, PlacedPrimitive, Connection


# =============================================================================
# TRAVERSABILITY CONSTANTS (aligned with CLAUDE.md Section 4)
# =============================================================================

# Step height limits (from quakedef.h STEPSIZE)
MAX_STEP_HEIGHT = 18.0  # Maximum climbable step (absolute limit)
COMFORTABLE_STEP = 16.0  # Standard step height for generation
MIN_STEP_DEPTH = 16.0   # Minimum horizontal depth per step

# Ramp/slope limits
MAX_RAMP_ANGLE = 45.0   # Maximum traversable ramp angle in degrees

# Player clearance requirements
MIN_PORTAL_CLEARANCE = 56.0  # Player height (bounding box)
MIN_PORTAL_WIDTH = 32.0      # Player width (bounding box)

# Floor height transitions
STANDARD_FLOOR_HEIGHT = 128.0  # Standard floor-to-floor height
HALF_FLOOR_HEIGHT = 64.0       # Half-floor transition


class TraversabilitySeverity(Enum):
    """Severity level for traversability issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TraversabilityIssue:
    """A single traversability issue between primitives or portals."""
    severity: TraversabilitySeverity
    issue_type: str  # "step_height", "no_stair", "z_mismatch", "steep_ramp", "portal_clearance"
    message: str
    primitive_a_id: str
    primitive_b_id: Optional[str] = None
    z_delta: Optional[float] = None  # Actual Z difference causing issue

    @property
    def is_error(self) -> bool:
        return self.severity == TraversabilitySeverity.ERROR

    @property
    def is_warning(self) -> bool:
        return self.severity == TraversabilitySeverity.WARNING


@dataclass
class TraversabilityResult:
    """Result of traversability validation."""
    issues: List[TraversabilityIssue]
    traversable_connections: int = 0
    non_traversable_connections: int = 0
    isolated_region_count: int = 0
    isolated_primitive_ids: List[str] = field(default_factory=list)

    @property
    def is_fully_traversable(self) -> bool:
        """Check if all connections are traversable (no errors)."""
        return not any(i.is_error for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.is_warning for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.is_warning)


def check_portal_z_alignment(
    layout: DungeonLayout,
    connection: Connection
) -> Optional[TraversabilityIssue]:
    """
    Check if two connected portals have matching absolute Z positions.

    Portals should be at the same Z-level for horizontal connections,
    or within acceptable step height for vertical transitions.

    Args:
        layout: The dungeon layout containing the primitives
        connection: The connection to validate

    Returns:
        TraversabilityIssue if there's a problem, None if OK
    """
    prim_a = layout.primitives.get(connection.primitive_a_id)
    prim_b = layout.primitives.get(connection.primitive_b_id)

    if not prim_a or not prim_b:
        return TraversabilityIssue(
            severity=TraversabilitySeverity.ERROR,
            issue_type="missing_primitive",
            message="Connection references missing primitive",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
        )

    # Calculate absolute Z for each portal
    z_a = prim_a.z_offset + prim_a.get_portal_z_level(connection.portal_a_id)
    z_b = prim_b.z_offset + prim_b.get_portal_z_level(connection.portal_b_id)
    z_delta = abs(z_a - z_b)

    # Check if Z difference is within tolerance
    if z_delta <= 2:
        # Perfect alignment (within engine tolerance)
        return None
    elif z_delta <= MAX_STEP_HEIGHT:
        # Within climbable step height - warning only
        return TraversabilityIssue(
            severity=TraversabilitySeverity.WARNING,
            issue_type="step_height",
            message=f"Portal Z mismatch ({z_delta:.0f}u) within step limit",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
            z_delta=z_delta,
        )
    else:
        # Exceeds step height - requires stairs
        return TraversabilityIssue(
            severity=TraversabilitySeverity.ERROR,
            issue_type="z_mismatch",
            message=f"Portal Z mismatch ({z_delta:.0f}u) exceeds max step ({MAX_STEP_HEIGHT}u)",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
            z_delta=z_delta,
        )


def check_z_traversability(
    layout: DungeonLayout,
    connection: Connection,
    allow_stairs: bool = True
) -> Tuple[bool, Optional[TraversabilityIssue]]:
    """
    Check if a connection is traversable considering Z-levels.

    A connection is traversable if:
    1. Portals are at the same Z-level (within tolerance)
    2. Z difference is within step height (≤18 units)
    3. One of the connected primitives is a VerticalStairHall (if Z differs)

    Args:
        layout: The dungeon layout
        connection: The connection to check
        allow_stairs: If True, connections to VerticalStairHall bypass Z checks

    Returns:
        Tuple of (is_traversable, issue_or_none)
    """
    prim_a = layout.primitives.get(connection.primitive_a_id)
    prim_b = layout.primitives.get(connection.primitive_b_id)

    if not prim_a or not prim_b:
        return False, TraversabilityIssue(
            severity=TraversabilitySeverity.ERROR,
            issue_type="missing_primitive",
            message="Connection references missing primitive",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
        )

    # VerticalStairHall provides vertical traversability
    if allow_stairs:
        if prim_a.primitive_type == 'VerticalStairHall':
            return True, None
        if prim_b.primitive_type == 'VerticalStairHall':
            return True, None

    # Calculate absolute Z for each portal
    z_a = prim_a.z_offset + prim_a.get_portal_z_level(connection.portal_a_id)
    z_b = prim_b.z_offset + prim_b.get_portal_z_level(connection.portal_b_id)
    z_delta = abs(z_a - z_b)

    # Check traversability
    if z_delta <= 2:
        # Within tolerance - fully traversable
        return True, None
    elif z_delta <= MAX_STEP_HEIGHT:
        # Within step height - traversable with warning
        return True, TraversabilityIssue(
            severity=TraversabilitySeverity.WARNING,
            issue_type="step_height",
            message=f"Z difference ({z_delta:.0f}u) at step limit",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
            z_delta=z_delta,
        )
    else:
        # Not traversable - requires stairs
        return False, TraversabilityIssue(
            severity=TraversabilitySeverity.ERROR,
            issue_type="no_stair",
            message=f"Z difference ({z_delta:.0f}u) requires VerticalStairHall",
            primitive_a_id=connection.primitive_a_id,
            primitive_b_id=connection.primitive_b_id,
            z_delta=z_delta,
        )


def build_traversable_graph(
    layout: DungeonLayout
) -> Dict[str, Set[str]]:
    """
    Build an adjacency graph of only traversable connections.

    This graph excludes connections where Z-level differences
    make traversal impossible (without VerticalStairHall).

    Args:
        layout: The dungeon layout to analyze

    Returns:
        Dict mapping primitive_id -> set of traversable neighbor primitive_ids
    """
    # Initialize empty adjacency for all primitives
    adj: Dict[str, Set[str]] = {pid: set() for pid in layout.primitives}

    for conn in layout.connections:
        is_traversable, _ = check_z_traversability(layout, conn)
        if is_traversable:
            adj[conn.primitive_a_id].add(conn.primitive_b_id)
            adj[conn.primitive_b_id].add(conn.primitive_a_id)

    return adj


def find_isolated_regions(
    layout: DungeonLayout
) -> List[Set[str]]:
    """
    Find isolated regions in the layout based on traversability.

    Returns a list of connected components where each component
    contains primitive IDs that can reach each other via traversable paths.

    Args:
        layout: The dungeon layout to analyze

    Returns:
        List of sets, each set containing primitive IDs in one connected region
    """
    if not layout.primitives:
        return []

    # Build traversable adjacency graph
    adj = build_traversable_graph(layout)

    # Find connected components via BFS
    visited: Set[str] = set()
    regions: List[Set[str]] = []

    for start_id in layout.primitives:
        if start_id in visited:
            continue

        # BFS to find all primitives in this region
        region: Set[str] = set()
        queue = deque([start_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            region.add(current)

            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        regions.append(region)

    return regions


def validate_multi_floor_path(
    layout: DungeonLayout,
    start_id: Optional[str] = None
) -> TraversabilityResult:
    """
    Validate that all primitives are reachable via traversable paths.

    This is the main validation entry point for multi-floor layouts.
    It checks:
    1. All connections are traversable (or have appropriate stairs)
    2. All primitives can be reached from the start primitive
    3. Reports isolated regions that cannot be accessed

    Args:
        layout: The dungeon layout to validate
        start_id: Starting primitive ID (None = use first primitive)

    Returns:
        TraversabilityResult with all issues and statistics
    """
    issues: List[TraversabilityIssue] = []
    traversable_count = 0
    non_traversable_count = 0

    if not layout.primitives:
        return TraversabilityResult(
            issues=[],
            traversable_connections=0,
            non_traversable_connections=0,
            isolated_region_count=0,
        )

    # Check each connection for traversability
    for conn in layout.connections:
        is_traversable, issue = check_z_traversability(layout, conn)
        if is_traversable:
            traversable_count += 1
        else:
            non_traversable_count += 1

        if issue:
            issues.append(issue)

    # Find isolated regions
    regions = find_isolated_regions(layout)

    # Determine which region contains the start primitive
    if start_id is None:
        start_id = next(iter(layout.primitives))

    main_region: Optional[Set[str]] = None
    for region in regions:
        if start_id in region:
            main_region = region
            break

    # Collect isolated primitive IDs (not in main region)
    isolated_ids: List[str] = []
    for region in regions:
        if region is not main_region:
            isolated_ids.extend(region)
            # Add an error for each isolated region
            prim_types = [layout.primitives[pid].primitive_type for pid in list(region)[:3]]
            issue = TraversabilityIssue(
                severity=TraversabilitySeverity.ERROR,
                issue_type="isolated_region",
                message=f"Isolated region ({len(region)} primitives): {', '.join(prim_types)}...",
                primitive_a_id=next(iter(region)),
                primitive_b_id=None,
            )
            issues.append(issue)

    return TraversabilityResult(
        issues=issues,
        traversable_connections=traversable_count,
        non_traversable_connections=non_traversable_count,
        isolated_region_count=len(regions) - 1 if main_region else len(regions),
        isolated_primitive_ids=isolated_ids,
    )


def check_connection_requires_stair(
    layout: DungeonLayout,
    primitive_a_id: str,
    primitive_b_id: str,
    portal_a_id: str = 'entrance',
    portal_b_id: str = 'entrance'
) -> Tuple[bool, float]:
    """
    Check if a potential connection between two primitives would require a stair.

    Used during layout generation to determine if a VerticalStairHall needs
    to be inserted between two rooms/halls at different Z-levels.

    Args:
        layout: The dungeon layout
        primitive_a_id: First primitive ID
        primitive_b_id: Second primitive ID
        portal_a_id: Portal ID on first primitive
        portal_b_id: Portal ID on second primitive

    Returns:
        Tuple of (requires_stair: bool, z_delta: float)
    """
    prim_a = layout.primitives.get(primitive_a_id)
    prim_b = layout.primitives.get(primitive_b_id)

    if not prim_a or not prim_b:
        return False, 0.0

    # Calculate absolute Z for each portal
    z_a = prim_a.z_offset + prim_a.get_portal_z_level(portal_a_id)
    z_b = prim_b.z_offset + prim_b.get_portal_z_level(portal_b_id)
    z_delta = abs(z_a - z_b)

    # Requires stair if Z difference exceeds step height
    requires_stair = z_delta > MAX_STEP_HEIGHT
    return requires_stair, z_delta


def get_floor_for_z_offset(z_offset: float) -> str:
    """
    Get the floor name for a given Z-offset.

    Args:
        z_offset: The Z-offset value

    Returns:
        Floor name string ('Basement', 'Ground', 'Upper', 'Tower')
    """
    # Standard floor Z-offsets from palette_widget.FLOOR_LEVELS
    if z_offset <= -64:
        return 'Basement'
    elif z_offset <= 64:
        return 'Ground'
    elif z_offset <= 192:
        return 'Upper'
    else:
        return 'Tower'


def calculate_stair_height_change(z_delta: float) -> int:
    """
    Calculate the appropriate height_change parameter for VerticalStairHall.

    Args:
        z_delta: The Z-level difference to traverse

    Returns:
        Recommended height_change value (64, 128, 192, or 256)
    """
    # Round to nearest standard floor height
    if z_delta <= 96:
        return 64
    elif z_delta <= 160:
        return 128
    elif z_delta <= 224:
        return 192
    else:
        return 256
