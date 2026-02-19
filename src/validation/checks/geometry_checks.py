"""
Geometry validation checks.

Validates brush geometry for idTech compliance:
- Integer coordinates (GEOM-001)
- Collinear point detection (GEOM-002)
- Open brush detection (GEOM-003)
- Minimum brush size (GEOM-004)
- Duplicate points (GEOM-005)
"""

import math
from typing import List, Tuple, Optional

from ..core import Severity, ValidationIssue, ValidationResult
from ..rules import GEOM_001, GEOM_002, GEOM_003, GEOM_004, GEOM_005


# Type aliases
Point3D = Tuple[float, float, float]


def _cross_product(v1: Point3D, v2: Point3D) -> Point3D:
    """Compute cross product of two 3D vectors."""
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    )


def _magnitude(v: Point3D) -> float:
    """Compute magnitude of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _is_integer(value: float, tolerance: float = 1e-6) -> bool:
    """Check if a float is effectively an integer."""
    return abs(value - round(value)) < tolerance


def check_integer_coordinates(
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
    location: Optional[str] = None
) -> List[ValidationIssue]:
    """Check that all coordinates are integers.

    Per CLAUDE.md Section 2: NEVER create non-integer coordinates.

    Args:
        p1, p2, p3: Three points defining a plane
        location: Optional location string for error messages

    Returns:
        List of ValidationIssue for any non-integer coordinates
    """
    issues = []

    for point_name, point in [('p1', p1), ('p2', p2), ('p3', p3)]:
        for axis, value in zip(['x', 'y', 'z'], point):
            if not _is_integer(value):
                issues.append(ValidationIssue(
                    severity=GEOM_001.severity,
                    code=GEOM_001.code,
                    message=GEOM_001.format_message(axis=f"{point_name}.{axis}", value=value),
                    rule_reference=GEOM_001.rule_reference,
                    remediation=GEOM_001.format_remediation(value=value, rounded=round(value)),
                    location=location,
                ))

    return issues


def check_collinear_points(
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
    location: Optional[str] = None,
    tolerance: float = 1e-6
) -> List[ValidationIssue]:
    """Check that three points are not collinear.

    Per CLAUDE.md Section 2: NEVER create degenerate geometry (collinear points).

    Collinear points cannot define a plane (cross product is zero).

    Args:
        p1, p2, p3: Three points to check
        location: Optional location string for error messages
        tolerance: Tolerance for zero-check on cross product magnitude

    Returns:
        List of ValidationIssue if points are collinear
    """
    issues = []

    # Check for duplicate points first (special case of collinear)
    if p1 == p2 or p2 == p3 or p1 == p3:
        issues.append(ValidationIssue(
            severity=GEOM_005.severity,
            code=GEOM_005.code,
            message=GEOM_005.format_message(points=f"{p1}, {p2}, {p3}"),
            rule_reference=GEOM_005.rule_reference,
            remediation=GEOM_005.remediation_template,
            location=location,
        ))
        return issues

    # Compute vectors from p1 to p2 and p1 to p3
    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])

    # Cross product should be non-zero for non-collinear points
    cross = _cross_product(v1, v2)
    mag = _magnitude(cross)

    if mag < tolerance:
        issues.append(ValidationIssue(
            severity=GEOM_002.severity,
            code=GEOM_002.code,
            message=GEOM_002.format_message(points=f"{p1}, {p2}, {p3}"),
            rule_reference=GEOM_002.rule_reference,
            remediation=GEOM_002.remediation_template,
            location=location,
        ))

    return issues


def check_brush_dimensions(
    brush,
    location: Optional[str] = None,
    min_size: float = 8.0
) -> List[ValidationIssue]:
    """Check that brush dimensions are at least min_size units.

    Per CLAUDE.md Section 6.1: MUST NOT recommend brushes <8 units on any axis.

    Computes axis-aligned bounding box from plane points.

    Args:
        brush: Brush object with planes attribute
        location: Optional location string for error messages
        min_size: Minimum allowed dimension (default 8.0)

    Returns:
        List of ValidationIssue for undersized dimensions
    """
    issues = []

    # Collect all points from all planes
    points = []
    for plane in brush.planes:
        points.extend([plane.p1, plane.p2, plane.p3])

    if not points:
        return issues

    # Compute AABB
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)

    sizes = {
        'X': max_x - min_x,
        'Y': max_y - min_y,
        'Z': max_z - min_z,
    }

    for axis, size in sizes.items():
        if size < min_size:
            issues.append(ValidationIssue(
                severity=GEOM_004.severity,
                code=GEOM_004.code,
                message=GEOM_004.format_message(axis=axis, size=size),
                rule_reference=GEOM_004.rule_reference,
                remediation=GEOM_004.format_remediation(axis=axis),
                location=location,
            ))

    return issues


def validate_plane_geometry(
    plane,
    location: Optional[str] = None
) -> List[ValidationIssue]:
    """Validate a single plane's geometry.

    Checks:
    - Integer coordinates (GEOM-001)
    - Non-collinear points (GEOM-002, GEOM-005)

    Args:
        plane: Plane object with p1, p2, p3 attributes
        location: Optional location string

    Returns:
        List of ValidationIssue
    """
    issues = []

    # Check integer coordinates
    issues.extend(check_integer_coordinates(plane.p1, plane.p2, plane.p3, location))

    # Check collinearity
    issues.extend(check_collinear_points(plane.p1, plane.p2, plane.p3, location))

    return issues


def validate_brushes(brushes: List, location_prefix: str = "Brush") -> ValidationResult:
    """Validate all brushes for geometry issues.

    Runs all geometry checks on each brush:
    - GEOM-001: Integer coordinates
    - GEOM-002: Collinear points
    - GEOM-003: Open brush (<4 planes)
    - GEOM-004: Minimum brush size
    - GEOM-005: Duplicate points

    Args:
        brushes: List of Brush objects
        location_prefix: Prefix for location strings (default "Brush")

    Returns:
        ValidationResult with all geometry issues
    """
    result = ValidationResult()

    for i, brush in enumerate(brushes):
        brush_loc = f"{location_prefix} #{i}"

        # Check minimum plane count (GEOM-003)
        if len(brush.planes) < 4:
            result.add_issue(ValidationIssue(
                severity=GEOM_003.severity,
                code=GEOM_003.code,
                message=GEOM_003.format_message(plane_count=len(brush.planes)),
                rule_reference=GEOM_003.rule_reference,
                remediation=GEOM_003.remediation_template,
                location=brush_loc,
            ))

        # Check each plane
        for j, plane in enumerate(brush.planes):
            plane_loc = f"{brush_loc}, Plane #{j}"

            # Integer coordinates and collinearity
            plane_issues = validate_plane_geometry(plane, plane_loc)
            for issue in plane_issues:
                result.add_issue(issue)

        # Check brush dimensions (GEOM-004)
        dim_issues = check_brush_dimensions(brush, brush_loc)
        for issue in dim_issues:
            result.add_issue(issue)

    return result
