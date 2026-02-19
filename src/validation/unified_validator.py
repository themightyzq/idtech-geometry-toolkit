"""
Unified validator orchestrator.

Central class that coordinates all validation checks across pipeline stages.
Provides the main API for validation throughout the toolkit.
"""

import logging
from typing import List, Optional, Type

from .core import ValidationResult, ValidationStage, Severity
from .checks.geometry_checks import validate_brushes
from .checks.module_checks import (
    validate_module_registration,
    validate_module_contract,
    validate_all_modules,
)
from .checks.format_checks import (
    validate_map_structure,
    validate_export_format,
    validate_portal_alignment,
)

logger = logging.getLogger(__name__)


class UnifiedValidator:
    """Central orchestrator for all validation checks.

    Coordinates validation at each pipeline stage:
    - MODULE_REGISTRATION: When modules are registered in catalog
    - PLACEMENT: When primitives are placed in a layout
    - GENERATION: When brush geometry is generated
    - EXPORT: When map is written to file

    Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
    - Tests: All must pass
    - Format: map-format-validator passes for idTech 1 and idTech 4
    - Geometry: No degenerate geometry, all integer coordinates
    - Sealing: Sealed geometry rules verified
    - Portals: Connected portals match within ±2 units

    Attributes:
        strict_mode: If True, treat WARN as FAIL
        enabled: If False, skip all validation (for performance testing)
    """

    def __init__(self, strict_mode: bool = False, enabled: bool = True):
        """Initialize the unified validator.

        Args:
            strict_mode: If True, promote WARN to FAIL
            enabled: If False, skip validation (returns empty results)
        """
        self.strict_mode = strict_mode
        self.enabled = enabled
        self._validation_history: List[ValidationResult] = []

    def validate_module_registration(
        self,
        module_cls: Type,
        check_footprint: bool = True,
        check_catalog: bool = False  # Don't check catalog during registration
    ) -> ValidationResult:
        """Validate a module class at registration time.

        Stage: MODULE_REGISTRATION

        Checks:
        - MOD-001: Footprint definition exists
        - MOD-003: Parameter schema exists

        Args:
            module_cls: GeometricPrimitive subclass
            check_footprint: Whether to verify footprint exists
            check_catalog: Whether to check catalog (usually False during registration)

        Returns:
            ValidationResult with any registration issues
        """
        if not self.enabled:
            return ValidationResult(stage=ValidationStage.MODULE_REGISTRATION)

        result = validate_module_registration(
            module_cls,
            check_footprint=check_footprint,
            check_catalog=check_catalog,
        )
        result.stage = ValidationStage.MODULE_REGISTRATION

        self._apply_strict_mode(result)
        self._record_result(result)

        return result

    def validate_module_contract(
        self,
        module_cls: Type,
        check_rotations: bool = True
    ) -> ValidationResult:
        """Validate a module's full contract compliance.

        Comprehensive validation including:
        - All registration checks
        - MOD-002: generate() returns non-empty brush list
        - MOD-005: Geometry fits footprint at all rotations
        - All geometry checks on generated brushes

        Args:
            module_cls: GeometricPrimitive subclass
            check_rotations: Whether to test at 0°, 90°, 180°, 270°

        Returns:
            ValidationResult with all contract issues
        """
        if not self.enabled:
            return ValidationResult(stage=ValidationStage.MODULE_REGISTRATION)

        result = validate_module_contract(
            module_cls,
            check_rotations=check_rotations,
        )
        result.stage = ValidationStage.MODULE_REGISTRATION

        self._apply_strict_mode(result)
        self._record_result(result)

        return result

    def validate_placement(
        self,
        layout,
        check_overlaps: bool = True,
        check_connectivity: bool = True
    ) -> ValidationResult:
        """Validate layout placement before generation.

        Stage: PLACEMENT

        Checks:
        - Primitive overlap detection
        - Connection validity
        - Footprint bounds

        Args:
            layout: DungeonLayout object
            check_overlaps: Whether to check for primitive overlaps
            check_connectivity: Whether to validate connections

        Returns:
            ValidationResult with placement issues
        """
        if not self.enabled:
            return ValidationResult(stage=ValidationStage.PLACEMENT)

        result = ValidationResult(stage=ValidationStage.PLACEMENT)

        # Validate using existing SpatialValidator
        try:
            from quake_levelgenerator.src.ui.widgets.layout_editor.spatial_validation import (
                SpatialValidator
            )

            spatial_validator = SpatialValidator(layout)

            if check_overlaps:
                for prim in layout.primitives:
                    overlaps = spatial_validator.check_overlap(prim, exclude_id=prim.id)
                    if overlaps:
                        for overlap_id in overlaps:
                            result.add_issue(result.add_issue(
                                Severity.WARN,
                                "SPAT-001",
                                f"Primitive {prim.id} overlaps with {overlap_id}",
                                "Check primitive placement",
                                None,
                                f"{prim.primitive_type} @ {prim.cell}",
                            ))

            if check_connectivity:
                # Validate all connections reference valid primitives and portals
                for conn in layout.connections:
                    prim_a = layout.get_primitive(conn.primitive_a_id)
                    prim_b = layout.get_primitive(conn.primitive_b_id)

                    if not prim_a or not prim_b:
                        from .core import ValidationIssue
                        result.add_issue(ValidationIssue(
                            severity=Severity.FAIL,
                            code="CONN-001",
                            message=f"Connection references invalid primitive",
                            rule_reference="CLAUDE.md Section 8 - Portal Alignment System",
                            remediation="Ensure both primitives exist in layout",
                            location=f"{conn.primitive_a_id} -> {conn.primitive_b_id}",
                        ))

        except ImportError:
            logger.warning("SpatialValidator not available for placement validation")

        self._apply_strict_mode(result)
        self._record_result(result)

        return result

    def validate_generation(
        self,
        brushes: List,
        portal_mismatches: List = None,
        check_geometry: bool = True,
        check_portals: bool = True
    ) -> ValidationResult:
        """Validate generated brush geometry.

        Stage: GENERATION

        Checks:
        - GEOM-*: All geometry checks
        - PORT-001: Portal alignment

        Args:
            brushes: List of Brush objects
            portal_mismatches: List of PortalMismatch objects (optional)
            check_geometry: Whether to run geometry checks
            check_portals: Whether to validate portal alignment

        Returns:
            ValidationResult with generation issues
        """
        if not self.enabled:
            return ValidationResult(stage=ValidationStage.GENERATION)

        result = ValidationResult(stage=ValidationStage.GENERATION)

        # Geometry validation
        if check_geometry and brushes:
            geom_result = validate_brushes(brushes)
            result.merge(geom_result)

        # Portal validation
        if check_portals and portal_mismatches:
            portal_result = validate_portal_alignment(portal_mismatches)
            result.merge(portal_result)

        self._apply_strict_mode(result)
        self._record_result(result)

        return result

    def validate_export(
        self,
        entities: List,
        export_format: str = "idtech1",
        check_structure: bool = True,
        check_format: bool = True
    ) -> ValidationResult:
        """Validate map for export.

        Stage: EXPORT

        Checks:
        - MAP-*: Map structure (worldspawn, player_start)
        - FMT-*: Format-specific requirements

        Args:
            entities: List of Entity objects
            export_format: "idtech1" or "idtech4"
            check_structure: Whether to validate map structure
            check_format: Whether to validate format compliance

        Returns:
            ValidationResult with export issues
        """
        if not self.enabled:
            return ValidationResult(stage=ValidationStage.EXPORT)

        result = ValidationResult(stage=ValidationStage.EXPORT)

        if check_structure:
            structure_result = validate_map_structure(entities)
            result.merge(structure_result)

        if check_format:
            format_result = validate_export_format(entities, export_format)
            result.merge(format_result)

        self._apply_strict_mode(result)
        self._record_result(result)

        return result

    def validate_all_modules(
        self,
        check_rotations: bool = True
    ) -> ValidationResult:
        """Validate all registered modules.

        Comprehensive audit of all modules in the catalog.

        Args:
            check_rotations: Whether to test at all rotations

        Returns:
            ValidationResult with all module issues
        """
        if not self.enabled:
            return ValidationResult()

        result = validate_all_modules(
            check_rotations=check_rotations,
            strict=self.strict_mode,
        )

        self._record_result(result)

        return result

    def get_history(self) -> List[ValidationResult]:
        """Get validation history.

        Returns:
            List of all ValidationResult objects from this session
        """
        return self._validation_history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self._validation_history.clear()

    def _apply_strict_mode(self, result: ValidationResult) -> None:
        """Apply strict mode to a result (promote WARN to FAIL)."""
        if self.strict_mode:
            for issue in result.issues:
                if issue.severity == Severity.WARN:
                    issue.severity = Severity.FAIL

    def _record_result(self, result: ValidationResult) -> None:
        """Record a result in history."""
        self._validation_history.append(result)


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_default_validator: Optional[UnifiedValidator] = None


def get_validator(
    strict_mode: bool = None,
    enabled: bool = None
) -> UnifiedValidator:
    """Get the default validator instance.

    Creates a singleton instance on first call. Subsequent calls
    return the same instance unless reset_validator() is called.

    Args:
        strict_mode: Override strict mode (None = use existing/default)
        enabled: Override enabled state (None = use existing/default)

    Returns:
        UnifiedValidator instance
    """
    global _default_validator

    if _default_validator is None:
        _default_validator = UnifiedValidator(
            strict_mode=strict_mode or False,
            enabled=enabled if enabled is not None else True,
        )
    else:
        # Update settings if specified
        if strict_mode is not None:
            _default_validator.strict_mode = strict_mode
        if enabled is not None:
            _default_validator.enabled = enabled

    return _default_validator


def reset_validator() -> None:
    """Reset the default validator instance.

    Use this to force creation of a new validator with fresh settings.
    """
    global _default_validator
    _default_validator = None
