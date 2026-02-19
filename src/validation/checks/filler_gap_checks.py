"""
Filler brush and sealing gap detection checks.

Validates brush geometry for:
- GEOM-020: Filler brush detection (axis-aligned boxes that are removable)
- SEAL-010: Hull boundary gaps in sealing-critical modules
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..core import Severity, ValidationIssue, ValidationResult
from ..rules import GEOM_020, SEAL_010


# Configuration constants
FILLER_MIN_VOL = 8 * 8 * 8  # Minimum volume for filler candidate
HULL_SAMPLE_STEP = 32       # Grid step for hull boundary sampling
MIN_WALL_THICKNESS = 16     # Per best practices doc


Point3D = Tuple[float, float, float]


@dataclass
class BrushAABB:
    """Axis-aligned bounding box for a brush."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    brush_index: int

    @property
    def size_x(self) -> float:
        return self.max_x - self.min_x

    @property
    def size_y(self) -> float:
        return self.max_y - self.min_y

    @property
    def size_z(self) -> float:
        return self.max_z - self.min_z

    @property
    def volume(self) -> float:
        return self.size_x * self.size_y * self.size_z

    def contains_point(self, x: float, y: float, z: float, tolerance: float = 0.1) -> bool:
        """Check if point is inside this AABB."""
        return (self.min_x - tolerance <= x <= self.max_x + tolerance and
                self.min_y - tolerance <= y <= self.max_y + tolerance and
                self.min_z - tolerance <= z <= self.max_z + tolerance)

    def overlaps(self, other: 'BrushAABB', tolerance: float = 0.1) -> bool:
        """Check if this AABB overlaps another."""
        return (self.min_x - tolerance < other.max_x + tolerance and
                self.max_x + tolerance > other.min_x - tolerance and
                self.min_y - tolerance < other.max_y + tolerance and
                self.max_y + tolerance > other.min_y - tolerance and
                self.min_z - tolerance < other.max_z + tolerance and
                self.max_z + tolerance > other.min_z - tolerance)


@dataclass
class PortalBounds:
    """Bounds for a portal opening."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    portal_id: str

    def contains_point(self, x: float, y: float, z: float, tolerance: float = 2.0) -> bool:
        """Check if point is within portal opening."""
        return (self.min_x - tolerance <= x <= self.max_x + tolerance and
                self.min_y - tolerance <= y <= self.max_y + tolerance and
                self.min_z - tolerance <= z <= self.max_z + tolerance)


def compute_brush_aabb(brush, brush_index: int) -> BrushAABB:
    """Compute axis-aligned bounding box for a brush.

    Args:
        brush: Brush object with planes attribute
        brush_index: Index of brush in brush list

    Returns:
        BrushAABB with min/max coordinates
    """
    points = []
    for plane in brush.planes:
        points.extend([plane.p1, plane.p2, plane.p3])

    if not points:
        return BrushAABB(0, 0, 0, 0, 0, 0, brush_index)

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)

    return BrushAABB(min_x, max_x, min_y, max_y, min_z, max_z, brush_index)


def is_axis_aligned_box(brush) -> bool:
    """Check if brush is an axis-aligned box (AABB prism).

    An axis-aligned box has exactly 6 planes, each aligned to X, Y, or Z axis.

    Args:
        brush: Brush object with planes attribute

    Returns:
        True if brush is an axis-aligned box
    """
    if len(brush.planes) != 6:
        return False

    axis_counts = {'x': 0, 'y': 0, 'z': 0}

    for plane in brush.planes:
        p1, p2, p3 = plane.p1, plane.p2, plane.p3

        # Compute normal from cross product
        v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        )

        # Check if normal is axis-aligned (one component dominates)
        abs_normal = (abs(normal[0]), abs(normal[1]), abs(normal[2]))
        max_comp = max(abs_normal)

        if max_comp < 1e-6:
            return False  # Degenerate plane

        # Check dominance (one axis should be > 99% of total)
        total = abs_normal[0] + abs_normal[1] + abs_normal[2]
        if abs_normal[0] / total > 0.99:
            axis_counts['x'] += 1
        elif abs_normal[1] / total > 0.99:
            axis_counts['y'] += 1
        elif abs_normal[2] / total > 0.99:
            axis_counts['z'] += 1
        else:
            return False  # Not axis-aligned

    # Should have exactly 2 planes for each axis
    return axis_counts['x'] == 2 and axis_counts['y'] == 2 and axis_counts['z'] == 2


def extract_portal_bounds(
    portal_tags: List,
    module_aabb: BrushAABB
) -> List[PortalBounds]:
    """Extract portal bounds from portal tags.

    Args:
        portal_tags: List of PortalTag objects
        module_aabb: Module's overall bounding box

    Returns:
        List of PortalBounds
    """
    bounds = []
    for tag in portal_tags:
        # Portal tag has center_x, center_y, center_z, width, height
        hw = tag.width / 2
        hh = tag.height / 2

        # Determine portal orientation from direction
        direction = str(tag.direction).lower()

        if 'north' in direction or 'south' in direction:
            # Portal faces N/S, opening is in X-Z plane
            bounds.append(PortalBounds(
                min_x=tag.center_x - hw,
                max_x=tag.center_x + hw,
                min_y=tag.center_y - 1,  # Thin Y extent
                max_y=tag.center_y + 1,
                min_z=tag.center_z,
                max_z=tag.center_z + tag.height,
                portal_id=tag.portal_id
            ))
        else:
            # Portal faces E/W, opening is in Y-Z plane
            bounds.append(PortalBounds(
                min_x=tag.center_x - 1,
                max_x=tag.center_x + 1,
                min_y=tag.center_y - hw,
                max_y=tag.center_y + hw,
                min_z=tag.center_z,
                max_z=tag.center_z + tag.height,
                portal_id=tag.portal_id
            ))

    return bounds


def is_connector_adjacent(
    brush_aabb: BrushAABB,
    portal_bounds: List[PortalBounds]
) -> bool:
    """Check if brush is adjacent to any portal/connector.

    Args:
        brush_aabb: Brush bounding box
        portal_bounds: List of portal bounds

    Returns:
        True if brush touches or is near any portal
    """
    for portal in portal_bounds:
        # Create a slightly expanded portal AABB for adjacency check
        portal_aabb = BrushAABB(
            portal.min_x - 8, portal.max_x + 8,
            portal.min_y - 8, portal.max_y + 8,
            portal.min_z - 8, portal.max_z + 8,
            -1
        )
        if brush_aabb.overlaps(portal_aabb):
            return True
    return False


def is_hull_participating(
    brush_aabb: BrushAABB,
    module_aabb: BrushAABB,
    tolerance: float = 8.0
) -> bool:
    """Check if brush participates in module's outer hull boundary.

    A brush participates in hull if any of its faces are at or near
    the module's outer boundary.

    Args:
        brush_aabb: Brush bounding box
        module_aabb: Module's overall bounding box
        tolerance: Distance from hull to count as participating

    Returns:
        True if brush is on the hull boundary
    """
    # Check if any brush face is at module boundary
    at_min_x = abs(brush_aabb.min_x - module_aabb.min_x) < tolerance
    at_max_x = abs(brush_aabb.max_x - module_aabb.max_x) < tolerance
    at_min_y = abs(brush_aabb.min_y - module_aabb.min_y) < tolerance
    at_max_y = abs(brush_aabb.max_y - module_aabb.max_y) < tolerance
    at_min_z = abs(brush_aabb.min_z - module_aabb.min_z) < tolerance
    at_max_z = abs(brush_aabb.max_z - module_aabb.max_z) < tolerance

    return at_min_x or at_max_x or at_min_y or at_max_y or at_min_z or at_max_z


def detect_filler_brushes(
    brushes: List,
    portal_tags: List = None,
    module_name: str = None,
    min_volume: float = FILLER_MIN_VOL
) -> List[ValidationIssue]:
    """Detect filler brushes that are candidates for removal.

    A filler brush is:
    1. An axis-aligned box
    2. Volume >= FILLER_MIN_VOL
    3. NOT connector-adjacent
    4. NOT hull-participating (for sealed modules)

    Args:
        brushes: List of Brush objects
        portal_tags: Optional list of PortalTag objects
        module_name: Module name for error messages
        min_volume: Minimum volume to consider as filler

    Returns:
        List of ValidationIssue for detected filler brushes
    """
    issues = []

    if not brushes:
        return issues

    # Compute AABBs for all brushes
    brush_aabbs = [compute_brush_aabb(b, i) for i, b in enumerate(brushes)]

    # Compute module AABB
    module_aabb = BrushAABB(
        min(b.min_x for b in brush_aabbs),
        max(b.max_x for b in brush_aabbs),
        min(b.min_y for b in brush_aabbs),
        max(b.max_y for b in brush_aabbs),
        min(b.min_z for b in brush_aabbs),
        max(b.max_z for b in brush_aabbs),
        -1
    )

    # Extract portal bounds
    portal_bounds = []
    if portal_tags:
        portal_bounds = extract_portal_bounds(portal_tags, module_aabb)

    # Check each brush
    for i, brush in enumerate(brushes):
        aabb = brush_aabbs[i]

        # Skip if not axis-aligned box
        if not is_axis_aligned_box(brush):
            continue

        # Skip if volume too small
        if aabb.volume < min_volume:
            continue

        # Skip if connector-adjacent
        if is_connector_adjacent(aabb, portal_bounds):
            continue

        # Skip if hull-participating
        if is_hull_participating(aabb, module_aabb):
            continue

        # This is a filler candidate
        issues.append(ValidationIssue(
            severity=GEOM_020.severity,
            code=GEOM_020.code,
            message=GEOM_020.format_message(
                details=f"AABB at ({aabb.min_x:.0f},{aabb.min_y:.0f},{aabb.min_z:.0f})-"
                       f"({aabb.max_x:.0f},{aabb.max_y:.0f},{aabb.max_z:.0f})"
            ),
            rule_reference=GEOM_020.rule_reference,
            remediation=GEOM_020.format_remediation(brush_index=i, volume=aabb.volume),
            location=f"Brush #{i}",
            module=module_name,
        ))

    return issues


def detect_hull_gaps(
    brushes: List,
    portal_tags: List = None,
    module_name: str = None,
    sample_step: float = HULL_SAMPLE_STEP
) -> List[ValidationIssue]:
    """Detect gaps in module hull boundary.

    Samples points on the module's outer hull and checks if each point
    is either:
    1. Inside a portal opening
    2. Covered by a brush

    If neither, it's a gap.

    Args:
        brushes: List of Brush objects
        portal_tags: Optional list of PortalTag objects
        module_name: Module name for error messages
        sample_step: Grid step for sampling hull boundary

    Returns:
        List of ValidationIssue for detected gaps
    """
    issues = []

    if not brushes:
        return issues

    # Compute AABBs
    brush_aabbs = [compute_brush_aabb(b, i) for i, b in enumerate(brushes)]

    # Compute module AABB
    module_aabb = BrushAABB(
        min(b.min_x for b in brush_aabbs),
        max(b.max_x for b in brush_aabbs),
        min(b.min_y for b in brush_aabbs),
        max(b.max_y for b in brush_aabbs),
        min(b.min_z for b in brush_aabbs),
        max(b.max_z for b in brush_aabbs),
        -1
    )

    # Extract portal bounds
    portal_bounds = []
    if portal_tags:
        portal_bounds = extract_portal_bounds(portal_tags, module_aabb)

    gaps = []

    # Sample each face of the module hull
    def sample_face(x_range, y_range, z_range, face_name):
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point = (x, y, z)

                    # Check if in portal
                    in_portal = any(p.contains_point(x, y, z) for p in portal_bounds)
                    if in_portal:
                        continue

                    # Check if covered by brush
                    covered = any(b.contains_point(x, y, z) for b in brush_aabbs)
                    if covered:
                        continue

                    # Gap detected
                    gaps.append((point, face_name))

    # Generate sample ranges
    def frange(start, stop, step):
        vals = []
        v = start
        while v <= stop:
            vals.append(v)
            v += step
        return vals

    x_samples = frange(module_aabb.min_x, module_aabb.max_x, sample_step)
    y_samples = frange(module_aabb.min_y, module_aabb.max_y, sample_step)
    z_samples = frange(module_aabb.min_z, module_aabb.max_z, sample_step)

    # Sample hull faces (only for sealed modules, expensive check)
    # For now, only check floor and ceiling (Z faces) as most common gap locations
    # Full hull check would sample all 6 faces

    # Floor face (min Z)
    sample_face(x_samples, y_samples, [module_aabb.min_z], "floor")

    # Ceiling face (max Z)
    sample_face(x_samples, y_samples, [module_aabb.max_z], "ceiling")

    # Report unique gaps (deduplicate nearby points)
    reported_positions = set()
    for point, face in gaps:
        # Round to grid for deduplication
        grid_pos = (round(point[0] / 16) * 16,
                   round(point[1] / 16) * 16,
                   round(point[2] / 16) * 16)

        if grid_pos in reported_positions:
            continue
        reported_positions.add(grid_pos)

        issues.append(ValidationIssue(
            severity=SEAL_010.severity,
            code=SEAL_010.code,
            message=SEAL_010.format_message(
                position=f"{face} at ({point[0]:.0f},{point[1]:.0f},{point[2]:.0f})"
            ),
            rule_reference=SEAL_010.rule_reference,
            remediation=SEAL_010.format_remediation(min_thickness=MIN_WALL_THICKNESS),
            location=f"{face} @ {grid_pos}",
            module=module_name,
        ))

    return issues


def scan_module_filler_gaps(
    brushes: List,
    portal_tags: List = None,
    module_name: str = None,
    is_sealing_critical: bool = True,
    check_filler: bool = True,
    check_gaps: bool = True
) -> ValidationResult:
    """Run filler and gap detection on a module's brushes.

    Args:
        brushes: List of Brush objects
        portal_tags: Optional list of PortalTag objects
        module_name: Module name for error messages
        is_sealing_critical: If True, check for hull gaps (SEAL-010)
        check_filler: If True, check for filler brushes (GEOM-020)
        check_gaps: If True, check for hull gaps

    Returns:
        ValidationResult with all detected issues
    """
    result = ValidationResult()

    if check_filler:
        filler_issues = detect_filler_brushes(
            brushes, portal_tags, module_name
        )
        for issue in filler_issues:
            result.add_issue(issue)

    if check_gaps and is_sealing_critical:
        gap_issues = detect_hull_gaps(
            brushes, portal_tags, module_name
        )
        for issue in gap_issues:
            result.add_issue(issue)

    return result
