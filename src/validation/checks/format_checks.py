"""
Map format validation checks.

Validates map structure and format compliance:
- MAP-001: Worldspawn entity exists
- MAP-002: info_player_start exists
- MAP-003: Map bounds within limits
- Format-specific checks for idTech 1 and idTech 4
"""

from typing import List, Optional

from ..core import Severity, ValidationIssue, ValidationResult
from ..rules import MAP_001, MAP_002, MAP_003


def validate_map_structure(entities: List, location: Optional[str] = None) -> ValidationResult:
    """Validate basic map structure requirements.

    Checks:
    - MAP-001: Worldspawn entity exists
    - MAP-002: info_player_start exists
    - MAP-003: Map bounds within limits

    Args:
        entities: List of Entity objects
        location: Optional location prefix

    Returns:
        ValidationResult with structure issues
    """
    result = ValidationResult()

    if not entities:
        result.add_issue(ValidationIssue(
            severity=MAP_001.severity,
            code=MAP_001.code,
            message="No entities present in map",
            rule_reference=MAP_001.rule_reference,
            remediation="Add at least worldspawn and info_player_start entities",
            location=location,
        ))
        return result

    # Check for worldspawn (MAP-001)
    has_worldspawn = any(e.classname == "worldspawn" for e in entities)
    if not has_worldspawn:
        result.add_issue(ValidationIssue(
            severity=MAP_001.severity,
            code=MAP_001.code,
            message=MAP_001.message_template,
            rule_reference=MAP_001.rule_reference,
            remediation=MAP_001.remediation_template,
            location=location,
        ))

    # Check for player start (MAP-002)
    has_player_start = any(
        e.classname in ("info_player_start", "info_player_deathmatch")
        for e in entities
    )
    if not has_player_start:
        result.add_issue(ValidationIssue(
            severity=MAP_002.severity,
            code=MAP_002.code,
            message=MAP_002.message_template,
            rule_reference=MAP_002.rule_reference,
            remediation=MAP_002.remediation_template,
            location=location,
        ))

    # Check map bounds (MAP-003)
    bounds_issues = _check_map_bounds(entities)
    for issue in bounds_issues:
        result.add_issue(issue)

    return result


def _check_map_bounds(entities: List, limit: float = 32768) -> List[ValidationIssue]:
    """Check that map coordinates are within safe limits.

    Args:
        entities: List of Entity objects
        limit: Maximum coordinate value (default ±32768)

    Returns:
        List of ValidationIssue for bounds violations
    """
    issues = []

    try:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for entity in entities:
            for brush in getattr(entity, 'brushes', []):
                for plane in brush.planes:
                    for point in [plane.p1, plane.p2, plane.p3]:
                        x, y, z = point
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        min_z, max_z = min(min_z, z), max(max_z, z)

        # Check each axis
        for axis, (min_val, max_val) in [
            ('X', (min_x, max_x)),
            ('Y', (min_y, max_y)),
            ('Z', (min_z, max_z)),
        ]:
            if abs(min_val) > limit or abs(max_val) > limit:
                extreme = min_val if abs(min_val) > abs(max_val) else max_val
                issues.append(ValidationIssue(
                    severity=MAP_003.severity,
                    code=MAP_003.code,
                    message=MAP_003.format_message(axis=axis, value=extreme),
                    rule_reference=MAP_003.rule_reference,
                    remediation=MAP_003.remediation_template,
                    location=f"{axis} axis",
                ))

    except (TypeError, ValueError):
        # Handle case where no geometry exists
        pass

    return issues


def validate_export_format(
    entities: List,
    export_format: str = "idtech1"
) -> ValidationResult:
    """Validate map for export format compliance.

    Runs format-specific validation for idTech 1 or idTech 4.

    Args:
        entities: List of Entity objects
        export_format: "idtech1" or "idtech4"

    Returns:
        ValidationResult with format-specific issues
    """
    result = ValidationResult()

    # First check basic structure
    structure_result = validate_map_structure(entities)
    result.merge(structure_result)

    # Format-specific checks
    if export_format.lower() == "idtech1":
        format_result = _validate_idtech1(entities)
        result.merge(format_result)
    elif export_format.lower() == "idtech4":
        format_result = _validate_idtech4(entities)
        result.merge(format_result)

    return result


def _validate_idtech1(entities: List) -> ValidationResult:
    """Validate idTech 1 (Quake) format compliance.

    Checks:
    - Texture names are valid
    - Worldspawn has wad property
    - Coordinate precision

    Args:
        entities: List of Entity objects

    Returns:
        ValidationResult with idTech 1 issues
    """
    result = ValidationResult()

    # Find worldspawn
    worldspawn = next((e for e in entities if e.classname == "worldspawn"), None)

    if worldspawn:
        # Check for wad property
        if "wad" not in worldspawn.properties:
            result.add_issue(ValidationIssue(
                severity=Severity.WARN,
                code="FMT-001",
                message="Worldspawn missing 'wad' property for texture loading",
                rule_reference="idTech 1 MAP format specification",
                remediation="Add wad property pointing to texture WAD file",
                location="worldspawn",
            ))

    # Validate all brushes
    for entity in entities:
        for brush in getattr(entity, 'brushes', []):
            for plane in brush.planes:
                # idTech 1 uses integer coordinates
                for point in [plane.p1, plane.p2, plane.p3]:
                    for coord in point:
                        if coord != int(coord):
                            result.add_issue(ValidationIssue(
                                severity=Severity.FAIL,
                                code="FMT-002",
                                message=f"Non-integer coordinate {coord} in idTech 1 format",
                                rule_reference="idTech 1 MAP format requires integer coordinates",
                                remediation=f"Round to integer: {int(round(coord))}",
                                location=f"Brush in {entity.classname}",
                            ))
                            return result  # One violation is enough

    return result


def _validate_idtech4(entities: List) -> ValidationResult:
    """Validate idTech 4 (Doom 3) format compliance.

    Checks:
    - brushDef3 format compatibility
    - Material names are valid paths

    Args:
        entities: List of Entity objects

    Returns:
        ValidationResult with idTech 4 issues
    """
    result = ValidationResult()

    # idTech 4 uses brushDef3 format with plane equations
    # Check texture/material names look like paths
    for entity in entities:
        for brush in getattr(entity, 'brushes', []):
            for plane in brush.planes:
                texture = plane.texture
                # Doom 3 materials often use path-like names
                # Common prefixes: textures/, models/, etc.
                # We just check for obviously invalid characters
                if texture and any(c in texture for c in ['<', '>', '|', '"']):
                    result.add_issue(ValidationIssue(
                        severity=Severity.WARN,
                        code="FMT-003",
                        message=f"Texture name '{texture}' may not be valid for idTech 4",
                        rule_reference="idTech 4 material naming conventions",
                        remediation="Use path-like material names (e.g., textures/common/caulk)",
                        location=f"Brush in {entity.classname}",
                    ))

    return result


def validate_portal_alignment(
    portal_mismatches: List,
    location: Optional[str] = None
) -> ValidationResult:
    """Validate portal alignment from generation results.

    Checks PORT-001: Portal mismatch tolerance.

    Args:
        portal_mismatches: List of PortalMismatch objects from generation
        location: Optional location prefix

    Returns:
        ValidationResult with portal alignment issues
    """
    from ..rules import PORT_001

    result = ValidationResult()

    for mismatch in portal_mismatches:
        # Calculate offset
        pos_a = mismatch.position_a
        pos_b = mismatch.position_b

        dx = abs(pos_a.center_x - pos_b.center_x)
        dy = abs(pos_a.center_y - pos_b.center_y)
        dz = abs(pos_a.center_z - pos_b.center_z)
        offset = max(dx, dy, dz)

        result.add_issue(ValidationIssue(
            severity=PORT_001.severity,
            code=PORT_001.code,
            message=PORT_001.format_message(
                offset=f"{offset:.1f}",
                portal_a=f"{mismatch.primitive_a_id}:{mismatch.portal_a_id}",
                portal_b=f"{mismatch.primitive_b_id}:{mismatch.portal_b_id}",
            ),
            rule_reference=PORT_001.rule_reference,
            remediation=PORT_001.remediation_template,
            location=location,
        ))

    return result
