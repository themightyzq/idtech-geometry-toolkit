"""
Module compliance validation checks.

Validates modules against contract requirements:
- Footprint definition exists (MOD-001)
- generate() returns non-empty brush list (MOD-002)
- Parameter schema exists (MOD-003)
- Catalog registration (MOD-004)
- Rotation correctness (MOD-005)
"""

from typing import List, Type, Optional, Set

from ..core import Severity, ValidationIssue, ValidationResult
from ..rules import MOD_001, MOD_002, MOD_003, MOD_004, MOD_005
from .geometry_checks import validate_brushes


def validate_module_registration(
    module_cls: Type,
    check_footprint: bool = True,
    check_catalog: bool = True
) -> ValidationResult:
    """Validate a module class at registration time.

    Checks:
    - MOD-001: Footprint definition exists
    - MOD-003: Parameter schema exists
    - MOD-004: Catalog registration (optional)

    Args:
        module_cls: The GeometricPrimitive subclass to validate
        check_footprint: Whether to check footprint exists
        check_catalog: Whether to check catalog registration

    Returns:
        ValidationResult with any registration issues
    """
    result = ValidationResult()
    module_name = module_cls.get_display_name()

    # Check parameter schema exists (MOD-003)
    try:
        schema = module_cls.get_parameter_schema()
        if not schema:
            result.add_issue(ValidationIssue(
                severity=MOD_003.severity,
                code=MOD_003.code,
                message=MOD_003.format_message(module=module_name),
                rule_reference=MOD_003.rule_reference,
                remediation=MOD_003.remediation_template,
                location=module_name,
            ))
    except AttributeError:
        result.add_issue(ValidationIssue(
            severity=MOD_003.severity,
            code=MOD_003.code,
            message=MOD_003.format_message(module=module_name),
            rule_reference=MOD_003.rule_reference,
            remediation=MOD_003.remediation_template,
            location=module_name,
        ))

    # Check footprint exists (MOD-001)
    if check_footprint:
        try:
            from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import PRIMITIVE_FOOTPRINTS
            # Try both with and without spaces (display name vs key name)
            key_name = module_name.replace(' ', '')
            if module_name not in PRIMITIVE_FOOTPRINTS and key_name not in PRIMITIVE_FOOTPRINTS:
                result.add_issue(ValidationIssue(
                    severity=MOD_001.severity,
                    code=MOD_001.code,
                    message=MOD_001.format_message(module=module_name),
                    rule_reference=MOD_001.rule_reference,
                    remediation=MOD_001.remediation_template,
                    location=module_name,
                ))
        except ImportError:
            # Can't check footprints if module not importable
            pass

    # Check catalog registration (MOD-004)
    if check_catalog:
        try:
            from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
            if PRIMITIVE_CATALOG.get_primitive(module_name) is None:
                result.add_issue(ValidationIssue(
                    severity=MOD_004.severity,
                    code=MOD_004.code,
                    message=MOD_004.format_message(module=module_name),
                    rule_reference=MOD_004.rule_reference,
                    remediation=MOD_004.remediation_template,
                    location=module_name,
                ))
        except ImportError:
            pass

    return result


def validate_module_contract(
    module_cls: Type,
    check_rotations: bool = True,
    rotations: List[int] = None
) -> ValidationResult:
    """Validate a module's full contract compliance.

    Runs comprehensive validation:
    - MOD-002: generate() returns non-empty brush list
    - MOD-005: Geometry fits footprint at all rotations
    - All geometry checks on generated brushes

    Args:
        module_cls: The GeometricPrimitive subclass to validate
        check_rotations: Whether to check at multiple rotations
        rotations: Rotations to test (default [0, 90, 180, 270])

    Returns:
        ValidationResult with all contract issues
    """
    result = ValidationResult()
    module_name = module_cls.get_display_name()
    test_rotations = rotations or [0, 90, 180, 270]

    # First check registration
    reg_result = validate_module_registration(module_cls)
    result.merge(reg_result)

    # Test generation at each rotation
    for rotation in (test_rotations if check_rotations else [0]):
        try:
            instance = module_cls()
            instance.params.rotation = rotation

            # Generate brushes
            brushes = instance.generate()

            # Check non-empty (MOD-002)
            if not brushes:
                result.add_issue(ValidationIssue(
                    severity=MOD_002.severity,
                    code=MOD_002.code,
                    message=MOD_002.format_message(module=module_name),
                    rule_reference=MOD_002.rule_reference,
                    remediation=MOD_002.remediation_template,
                    location=f"{module_name} @ {rotation}°",
                ))
                continue

            # Run geometry validation on generated brushes
            geom_result = validate_brushes(
                brushes,
                location_prefix=f"{module_name}@{rotation}° Brush"
            )
            result.merge(geom_result)

            # Check footprint bounds (MOD-005)
            if check_rotations:
                footprint_issues = _check_footprint_bounds(
                    module_cls, brushes, rotation
                )
                for issue in footprint_issues:
                    result.add_issue(issue)

        except Exception as e:
            # Generation failed - add error
            result.add_issue(ValidationIssue(
                severity=Severity.FAIL,
                code="MOD-002",
                message=f"Module {module_name} generate() raised exception: {e}",
                rule_reference=MOD_002.rule_reference,
                remediation="Fix the exception in generate() method",
                location=f"{module_name} @ {rotation}°",
            ))

    return result


def _check_footprint_bounds(
    module_cls: Type,
    brushes: List,
    rotation: int
) -> List[ValidationIssue]:
    """Check that generated brushes fit within footprint bounds.

    Args:
        module_cls: The module class
        brushes: Generated brushes
        rotation: Current rotation

    Returns:
        List of ValidationIssue for any bounds violations
    """
    issues = []
    module_name = module_cls.get_display_name()

    try:
        from quake_levelgenerator.src.ui.widgets.layout_editor.palette_widget import PRIMITIVE_FOOTPRINTS

        # Try both with and without spaces
        footprint = PRIMITIVE_FOOTPRINTS.get(module_name)
        if not footprint:
            key_name = module_name.replace(' ', '')
            footprint = PRIMITIVE_FOOTPRINTS.get(key_name)
        if not footprint:
            return issues  # No footprint to check against

        # Get rotated footprint size
        fp_width, fp_depth = footprint.rotated_size(rotation)

        # Default grid size
        grid_size = 128

        # Expected bounds (geometry should be normalized to origin)
        max_x = fp_width * grid_size
        max_y = fp_depth * grid_size

        # Collect all points
        for brush in brushes:
            for plane in brush.planes:
                for point in [plane.p1, plane.p2, plane.p3]:
                    x, y, z = point
                    # Check X bounds (allow small tolerance for wall thickness)
                    if x < -32 or x > max_x + 32:
                        issues.append(ValidationIssue(
                            severity=MOD_005.severity,
                            code=MOD_005.code,
                            message=MOD_005.format_message(
                                module=module_name, rotation=rotation
                            ),
                            rule_reference=MOD_005.rule_reference,
                            remediation=MOD_005.remediation_template,
                            location=f"{module_name} @ {rotation}°: X={x} exceeds bounds [0, {max_x}]",
                        ))
                        return issues  # One violation is enough
                    # Check Y bounds
                    if y < -32 or y > max_y + 32:
                        issues.append(ValidationIssue(
                            severity=MOD_005.severity,
                            code=MOD_005.code,
                            message=MOD_005.format_message(
                                module=module_name, rotation=rotation
                            ),
                            rule_reference=MOD_005.rule_reference,
                            remediation=MOD_005.remediation_template,
                            location=f"{module_name} @ {rotation}°: Y={y} exceeds bounds [0, {max_y}]",
                        ))
                        return issues

    except ImportError:
        pass

    return issues


def get_all_module_classes() -> List[Type]:
    """Get all registered module classes from the catalog.

    Returns:
        List of GeometricPrimitive subclasses
    """
    try:
        from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
        classes = []
        for name in PRIMITIVE_CATALOG.list_primitives():
            cls = PRIMITIVE_CATALOG.get_primitive(name)
            if cls:
                classes.append(cls)
        return classes
    except ImportError:
        return []


def validate_all_modules(
    check_rotations: bool = True,
    strict: bool = False
) -> ValidationResult:
    """Validate all registered modules.

    Args:
        check_rotations: Whether to check at multiple rotations
        strict: If True, treat WARN as FAIL

    Returns:
        ValidationResult with all module issues
    """
    result = ValidationResult()

    for module_cls in get_all_module_classes():
        module_result = validate_module_contract(module_cls, check_rotations)

        # In strict mode, promote warnings to failures
        if strict:
            for issue in module_result.warnings:
                issue.severity = Severity.FAIL

        result.merge(module_result)

    return result
