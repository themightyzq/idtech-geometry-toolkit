"""
Validation rule definitions with CLAUDE.md references.

Each rule has:
- Code: Unique identifier (e.g., "GEOM-001")
- Severity: FAIL, WARN, or INFO
- Rule reference: Section in CLAUDE.md
- Message template: Human-readable description
- Remediation: Suggested fix

Rules are organized by category:
- GEOM: Geometry validation
- SEAL: Sealed geometry
- PORT: Portal alignment
- TRAV: Traversability
- MAP: Map structure
- MOD: Module compliance
"""

from dataclasses import dataclass
from typing import Optional

from .core import Severity


@dataclass(frozen=True)
class ValidationRule:
    """Definition of a validation rule.

    Attributes:
        code: Unique rule code (e.g., "GEOM-001")
        severity: Default severity for this rule
        rule_reference: CLAUDE.md section reference
        message_template: Template for error message (use {placeholders})
        remediation_template: Template for suggested fix
        description: Full description of the rule
    """
    code: str
    severity: Severity
    rule_reference: str
    message_template: str
    remediation_template: Optional[str] = None
    description: Optional[str] = None

    def format_message(self, **kwargs) -> str:
        """Format the message template with provided values."""
        return self.message_template.format(**kwargs)

    def format_remediation(self, **kwargs) -> Optional[str]:
        """Format the remediation template with provided values."""
        if self.remediation_template:
            return self.remediation_template.format(**kwargs)
        return None


# =============================================================================
# GEOMETRY RULES (GEOM)
# Per CLAUDE.md Section 2 (Absolute Rules [BINDING])
# =============================================================================

GEOM_001 = ValidationRule(
    code="GEOM-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 2 - NEVER: create non-integer coordinates",
    message_template="Non-integer coordinate detected: {axis}={value}",
    remediation_template="Round coordinate to nearest integer: {value} -> {rounded}",
    description="All brush coordinates must be integers per idTech engine requirements"
)

GEOM_002 = ValidationRule(
    code="GEOM-002",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 2 - NEVER: create degenerate geometry (collinear points)",
    message_template="Collinear points in plane definition: {points}",
    remediation_template="Ensure three points define a valid plane (non-zero cross product)",
    description="Three points defining a plane must not be collinear"
)

GEOM_003 = ValidationRule(
    code="GEOM-003",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 2 - NEVER: create degenerate geometry (open brushes)",
    message_template="Open brush with only {plane_count} planes (minimum 4 required)",
    remediation_template="Add more planes to close the brush volume",
    description="Brushes must have at least 4 planes to form a closed volume"
)

GEOM_004 = ValidationRule(
    code="GEOM-004",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 6.1 - MUST NOT recommend brushes <8 units on any axis",
    message_template="Brush dimension {axis}={size} is less than 8 units",
    remediation_template="Increase {axis} dimension to at least 8 units",
    description="All brush dimensions must be at least 8 units"
)

GEOM_005 = ValidationRule(
    code="GEOM-005",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 2 - NEVER: create degenerate geometry",
    message_template="Duplicate points in plane definition: {points}",
    remediation_template="Ensure all three plane points are distinct",
    description="All three points defining a plane must be unique"
)

GEOM_006 = ValidationRule(
    code="GEOM-006",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 4 - Grid alignment: 64-unit",
    message_template="Coordinate {value} is not aligned to 64-unit grid",
    remediation_template="Snap to grid: {value} -> {snapped}",
    description="Coordinates should be aligned to 64-unit grid when possible"
)

GEOM_020 = ValidationRule(
    code="GEOM-020",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 6.1 - Remove unnecessary filler brushes",
    message_template="Filler brush detected: {details}",
    remediation_template="Remove filler brush at index {brush_index} (volume={volume})",
    description="Axis-aligned filler brushes that are not hull or connector adjacent can be removed"
)


# =============================================================================
# SEALED GEOMETRY RULES (SEAL)
# Per CLAUDE.md Section 5 (Sealed Geometry Rules [BINDING])
# =============================================================================

SEAL_001 = ValidationRule(
    code="SEAL-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 5 - Floor/ceiling: Extend by wall_thickness (t) in ALL directions",
    message_template="Floor/ceiling does not extend by wall_thickness: {details}",
    remediation_template="Extend floor/ceiling by t ({thickness} units) in all directions",
    description="Floor and ceiling must extend beyond walls by wall_thickness"
)

SEAL_002 = ValidationRule(
    code="SEAL-002",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 5 - Walls: Span full floor/ceiling extent",
    message_template="Wall does not span full floor/ceiling extent: {details}",
    remediation_template="Extend wall to span from (oy - t) to (oy + length + t)",
    description="Walls must span the full Y extent including floor/ceiling extensions"
)

SEAL_003 = ValidationRule(
    code="SEAL-003",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 5 - Brush overlaps: >=8 units at corners/junctions",
    message_template="Insufficient brush overlap at junction: {overlap} units (need >=8)",
    remediation_template="Increase overlap to at least 8 units",
    description="Brush overlaps at corners and junctions must be at least 8 units"
)

SEAL_010 = ValidationRule(
    code="SEAL-010",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 5 - Room/hall modules MUST be completely sealed",
    message_template="Gap detected on hull boundary at {position}",
    remediation_template="Add sealing brush of at least {min_thickness} units thickness",
    description="Hull boundary gap where no brush covers and no portal exists"
)


# =============================================================================
# PORTAL RULES (PORT)
# Per CLAUDE.md Section 3 (Quality Gates [BINDING])
# =============================================================================

PORT_001 = ValidationRule(
    code="PORT-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 3 - Portals: Connected portals match within +/-2 units",
    message_template="Portal mismatch: {offset} units between {portal_a} and {portal_b}",
    remediation_template="Adjust portal positions to align within 2 units",
    description="Connected portals must align within 2 units tolerance"
)

PORT_002 = ValidationRule(
    code="PORT-002",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 7 - Portal Alignment: Portal width/height fixed by shared parameters",
    message_template="Portal dimension mismatch: {details}",
    remediation_template="Ensure both portals use same width/height",
    description="Connected portals should have matching dimensions"
)

PORT_003 = ValidationRule(
    code="PORT-003",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 7 - Portal Alignment",
    message_template="Portal tag not registered for {primitive}:{portal}",
    remediation_template="Ensure module calls _register_portal_tag() during generate()",
    description="All portals should register tags for alignment validation"
)


# =============================================================================
# TRAVERSABILITY RULES (TRAV)
# Per CLAUDE.md Section 4 (idTech Engine Constraints [BINDING])
# =============================================================================

TRAV_001 = ValidationRule(
    code="TRAV-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 4 - Max step height: 18 units",
    message_template="Step height {height} exceeds maximum 18 units",
    remediation_template="Reduce step height to 18 units or less (comfortable: 16)",
    description="Individual step height must not exceed 18 units"
)

TRAV_002 = ValidationRule(
    code="TRAV-002",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 4 - Max step height: comfortable 16 units",
    message_template="Step height {height} exceeds comfortable 16 units",
    remediation_template="Consider reducing step height to 16 units for better gameplay",
    description="Step height over 16 units may feel uncomfortable"
)

TRAV_003 = ValidationRule(
    code="TRAV-003",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 4 - Max ramp angle: 45 degrees",
    message_template="Ramp angle {angle}° exceeds maximum 45°",
    remediation_template="Reduce ramp angle to 45° or less",
    description="Ramp angles must not exceed 45 degrees"
)

TRAV_004 = ValidationRule(
    code="TRAV-004",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 4 - Min step depth: 16 units",
    message_template="Step depth {depth} is less than minimum 16 units",
    remediation_template="Increase step depth to at least 16 units",
    description="Step depth must be at least 16 units"
)

TRAV_005 = ValidationRule(
    code="TRAV-005",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 4 - Player bbox: 32x32x56 units",
    message_template="Passage too narrow: {width} units (player needs 32)",
    remediation_template="Widen passage to at least 32 units",
    description="Passages must be at least 32 units wide for player movement"
)

TRAV_006 = ValidationRule(
    code="TRAV-006",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 4 - Player bbox: 32x32x56 units",
    message_template="Ceiling too low: {height} units (player needs 56)",
    remediation_template="Raise ceiling to at least 56 units",
    description="Ceilings must be at least 56 units high for player movement"
)


# =============================================================================
# MAP STRUCTURE RULES (MAP)
# Per CLAUDE.md Section 2.1 and Section 3
# =============================================================================

MAP_001 = ValidationRule(
    code="MAP-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 3 - Quality Gates",
    message_template="Missing worldspawn entity",
    remediation_template="Add worldspawn entity with required properties",
    description="Every map must have a worldspawn entity"
)

MAP_002 = ValidationRule(
    code="MAP-002",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 2.1 - The only entity this tool creates is info_player_start",
    message_template="Missing info_player_start entity",
    remediation_template="Add info_player_start entity at spawn location",
    description="Every map must have at least one info_player_start entity"
)

MAP_003 = ValidationRule(
    code="MAP-003",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 11 - idTech Geometry Toolkit",
    message_template="Map bounds exceed safe limits: {axis}={value} (limit: ±32768)",
    remediation_template="Move geometry closer to origin or reduce map size",
    description="Map coordinates should stay within ±32768 units"
)


# =============================================================================
# MODULE COMPLIANCE RULES (MOD)
# Per CLAUDE.md Section 10 (Module Checklist [BINDING])
# =============================================================================

MOD_001 = ValidationRule(
    code="MOD-001",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 10 - Add footprint to PRIMITIVE_FOOTPRINTS",
    message_template="Module {module} missing footprint definition",
    remediation_template="Add footprint to PRIMITIVE_FOOTPRINTS in palette_widget.py",
    description="Every module must have a footprint definition"
)

MOD_002 = ValidationRule(
    code="MOD-002",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 10 - Implement generate() returning List[Brush]",
    message_template="Module {module} generate() returned empty brush list",
    remediation_template="Ensure generate() returns at least one brush",
    description="Module generate() must return non-empty brush list"
)

MOD_003 = ValidationRule(
    code="MOD-003",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 10 - Add get_parameter_schema() for GUI",
    message_template="Module {module} missing parameter schema",
    remediation_template="Implement get_parameter_schema() class method",
    description="Every module must have a parameter schema"
)

MOD_004 = ValidationRule(
    code="MOD-004",
    severity=Severity.FAIL,
    rule_reference="CLAUDE.md Section 10 - Register in catalog.py",
    message_template="Module {module} not registered in catalog",
    remediation_template="Add module class to PRIMITIVE_CATALOG registration loop",
    description="Every module must be registered in the catalog"
)

MOD_005 = ValidationRule(
    code="MOD-005",
    severity=Severity.WARN,
    rule_reference="CLAUDE.md Section 9 - Rotation: Geometry fits footprint at all rotations",
    message_template="Module {module} geometry exceeds footprint at rotation {rotation}°",
    remediation_template="Adjust geometry or footprint to match at all rotations",
    description="Module geometry must fit within footprint at all rotations"
)


# =============================================================================
# RULE REGISTRY
# =============================================================================

ALL_RULES = {
    # Geometry
    'GEOM-001': GEOM_001,
    'GEOM-002': GEOM_002,
    'GEOM-003': GEOM_003,
    'GEOM-004': GEOM_004,
    'GEOM-005': GEOM_005,
    'GEOM-006': GEOM_006,
    'GEOM-020': GEOM_020,
    # Sealed
    'SEAL-001': SEAL_001,
    'SEAL-002': SEAL_002,
    'SEAL-003': SEAL_003,
    'SEAL-010': SEAL_010,
    # Portal
    'PORT-001': PORT_001,
    'PORT-002': PORT_002,
    'PORT-003': PORT_003,
    # Traversability
    'TRAV-001': TRAV_001,
    'TRAV-002': TRAV_002,
    'TRAV-003': TRAV_003,
    'TRAV-004': TRAV_004,
    'TRAV-005': TRAV_005,
    'TRAV-006': TRAV_006,
    # Map
    'MAP-001': MAP_001,
    'MAP-002': MAP_002,
    'MAP-003': MAP_003,
    # Module
    'MOD-001': MOD_001,
    'MOD-002': MOD_002,
    'MOD-003': MOD_003,
    'MOD-004': MOD_004,
    'MOD-005': MOD_005,
}


def get_rule(code: str) -> Optional[ValidationRule]:
    """Get a rule by its code.

    Args:
        code: Rule code (e.g., "GEOM-001")

    Returns:
        ValidationRule if found, None otherwise
    """
    return ALL_RULES.get(code)


def get_rules_by_category(prefix: str) -> list:
    """Get all rules with a given prefix.

    Args:
        prefix: Category prefix (e.g., "GEOM", "PORT")

    Returns:
        List of ValidationRule objects
    """
    return [rule for code, rule in ALL_RULES.items() if code.startswith(prefix)]
