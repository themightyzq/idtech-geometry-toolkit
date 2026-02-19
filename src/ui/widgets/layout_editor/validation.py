"""
Validation system for dungeon layouts.

Provides:
- Connectivity validation (BFS traversal)
- Sealed geometry checks
- Portal alignment validation
- Z-level traversability validation (multi-floor support)
- Real-time feedback panel
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .data_model import DungeonLayout, PlacedPrimitive, Portal, CellCoord
from .traversability import (
    validate_multi_floor_path,
    TraversabilityResult,
    TraversabilitySeverity,
    MAX_STEP_HEIGHT,
)
from .spatial_validation import (
    SpatialValidator,
    SpatialCollision,
    CollisionType,
)

# Debug flag for terminal output
VALIDATION_DEBUG = True

def _debug_print(message: str) -> None:
    """Print validation debug info to terminal."""
    if VALIDATION_DEBUG:
        print(f"[VALIDATION] {message}")


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    message: str
    primitive_id: Optional[str] = None
    portal_id: Optional[str] = None
    cell: Optional[CellCoord] = None

    @property
    def is_error(self) -> bool:
        return self.severity == ValidationSeverity.ERROR

    @property
    def is_warning(self) -> bool:
        return self.severity == ValidationSeverity.WARNING


@dataclass
class ValidationResult:
    """Result of layout validation."""
    issues: List[ValidationIssue]
    connected_count: int = 0
    disconnected_ids: List[str] = None

    def __post_init__(self):
        if self.disconnected_ids is None:
            self.disconnected_ids = []

    @property
    def is_valid(self) -> bool:
        """Check if layout passes validation (no errors)."""
        return not any(i.is_error for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(i.is_warning for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.is_warning)


class LayoutValidator:
    """Validator for dungeon layouts."""

    def validate(self, layout: DungeonLayout) -> ValidationResult:
        """Run all validation checks on the layout."""
        issues = []

        # Empty layout check
        if not layout.primitives:
            return ValidationResult(
                issues=[ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Layout is empty"
                )],
                connected_count=0
            )

        # Check for overlapping primitives (2D cell overlaps)
        issues.extend(self._check_overlaps(layout))

        # Check for spatial collisions (3D interpenetration)
        issues.extend(self._check_spatial_collisions(layout))

        # Check portal alignment
        issues.extend(self._check_portal_alignment(layout))

        # Check connectivity (basic BFS without Z-level consideration)
        connectivity_result = self._check_connectivity(layout)
        issues.extend(connectivity_result[0])

        # Check Z-level traversability (multi-floor aware)
        traversability_issues = self._check_z_level_traversability(layout)
        issues.extend(traversability_issues)

        # Check for missing vertical connectors between adjacent floors
        floor_connector_issues = self._check_floor_connectors(layout)
        issues.extend(floor_connector_issues)

        # Log all validation results to terminal
        result = ValidationResult(
            issues=issues,
            connected_count=connectivity_result[1],
            disconnected_ids=connectivity_result[2]
        )

        if issues:
            error_count = result.error_count
            warning_count = result.warning_count
            _debug_print(f"=== VALIDATION COMPLETE: {error_count} errors, {warning_count} warnings ===")
            for issue in issues:
                if issue.severity == ValidationSeverity.ERROR:
                    _debug_print(f"  [ERROR] {issue.message}")
                elif issue.severity == ValidationSeverity.WARNING:
                    _debug_print(f"  [WARN] {issue.message}")

        return result

    def _check_overlaps(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """Check for overlapping primitives (2D cell-based)."""
        issues = []
        cell_owners: Dict[Tuple[int, int], str] = {}

        for prim_id, prim in layout.primitives.items():
            for cell in prim.occupied_cells():
                key = (cell.x, cell.y)
                if key in cell_owners:
                    other_id = cell_owners[key]
                    # Only report if they're at the same Z-level
                    # (different Z with same XY is handled by spatial validation)
                    other_prim = layout.primitives.get(other_id)
                    if other_prim and abs(prim.z_offset - other_prim.z_offset) < 2:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Cell overlap at ({cell.x}, {cell.y})",
                            primitive_id=prim_id,
                            cell=cell
                        ))
                else:
                    cell_owners[key] = prim_id

        return issues

    def _check_spatial_collisions(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """
        Check for 3D spatial collisions (interpenetration between Z-levels).

        This catches cases where primitives at different Z-offsets have
        overlapping geometry due to sealed geometry extensions (floor/ceiling
        extend by wall_thickness in Z direction).

        Returns:
            List of ValidationIssues for detected spatial collisions
        """
        issues = []

        # Create spatial validator and run collision detection
        validator = SpatialValidator(layout)
        collisions = validator.validate_all()

        if collisions:
            _debug_print(f"=== SPATIAL COLLISIONS DETECTED: {len(collisions)} ===")

        for collision in collisions:
            # Log to terminal
            prim_a = layout.primitives.get(collision.primitive_a_id)
            prim_b = layout.primitives.get(collision.primitive_b_id)
            prim_a_name = prim_a.primitive_type if prim_a else "Unknown"
            prim_b_name = prim_b.primitive_type if prim_b else "Unknown"
            z_a = prim_a.z_offset if prim_a else "?"
            z_b = prim_b.z_offset if prim_b else "?"

            _debug_print(
                f"  {prim_a_name}(z={z_a}) <-> {prim_b_name}(z={z_b}) "
                f"| type={collision.collision_type.value} "
                f"| Z-range={collision.overlap_z_range}"
            )
            # Map collision type to severity
            if collision.is_critical:
                severity = ValidationSeverity.ERROR
            else:
                severity = ValidationSeverity.WARNING

            # Create detailed message
            z_min, z_max = collision.overlap_z_range
            message = (
                f"[Spatial] {collision.message} "
                f"(Z overlap: {z_min:.0f}-{z_max:.0f})"
            )

            issues.append(ValidationIssue(
                severity=severity,
                message=message,
                primitive_id=collision.primitive_a_id,
            ))

            # Add suggestion as info
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"  Suggestion: {collision.suggestion}",
                primitive_id=collision.primitive_a_id,
            ))

        return issues

    def _check_portal_alignment(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """Check that connected portals are properly aligned."""
        issues = []

        for conn in layout.connections:
            prim_a = layout.primitives.get(conn.primitive_a_id)
            prim_b = layout.primitives.get(conn.primitive_b_id)

            if not prim_a or not prim_b:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Connection references missing primitive",
                    primitive_id=conn.primitive_a_id if not prim_a else conn.primitive_b_id
                ))
                continue

            # Find portals
            portal_a = None
            portal_b = None
            for p in prim_a.get_portals():
                if p.id == conn.portal_a_id:
                    portal_a = p
                    break
            for p in prim_b.get_portals():
                if p.id == conn.portal_b_id:
                    portal_b = p
                    break

            if not portal_a or not portal_b:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Connection references missing portal",
                    primitive_id=conn.primitive_a_id,
                    portal_id=conn.portal_a_id if not portal_a else conn.portal_b_id
                ))
                continue

            # Check size compatibility
            if abs(portal_a.width - portal_b.width) > 16:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Portal width mismatch ({portal_a.width} vs {portal_b.width})",
                    primitive_id=conn.primitive_a_id,
                    portal_id=conn.portal_a_id
                ))

            if abs(portal_a.height - portal_b.height) > 16:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Portal height mismatch ({portal_a.height} vs {portal_b.height})",
                    primitive_id=conn.primitive_a_id,
                    portal_id=conn.portal_a_id
                ))

        return issues

    def _check_z_level_traversability(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """
        Check that all primitives are reachable considering Z-level differences.

        This validation ensures:
        1. Connected primitives have matching portal Z-levels (or are stairs)
        2. No isolated regions exist due to Z-level mismatches
        3. Step heights don't exceed player limits

        Returns:
            List of ValidationIssues for Z-level problems
        """
        issues = []

        # Run full traversability validation
        trav_result = validate_multi_floor_path(layout)

        # Convert traversability issues to validation issues
        for trav_issue in trav_result.issues:
            # Map traversability severity to validation severity
            if trav_issue.severity == TraversabilitySeverity.ERROR:
                severity = ValidationSeverity.ERROR
            elif trav_issue.severity == TraversabilitySeverity.WARNING:
                severity = ValidationSeverity.WARNING
            else:
                severity = ValidationSeverity.INFO

            # Create validation issue with Z-level context
            z_info = ""
            if trav_issue.z_delta is not None:
                z_info = f" (Z delta: {trav_issue.z_delta:.0f}u)"

            issues.append(ValidationIssue(
                severity=severity,
                message=f"[Traversability] {trav_issue.message}{z_info}",
                primitive_id=trav_issue.primitive_a_id,
            ))

        # Report summary of isolated regions if any
        if trav_result.isolated_region_count > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{trav_result.isolated_region_count} isolated region(s) due to Z-level mismatches",
            ))

        return issues

    def _check_floor_connectors(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """
        Check that adjacent floor levels are connected by VerticalStairHall primitives.

        This validation catches layouts where multiple z-levels exist but lack
        vertical connectors between them. Without stairs, upper/lower floors
        are unreachable even if they have valid connections at their own level.

        Enhanced to check CONNECTIVITY of stair portals, not just existence.
        A stair with a disconnected portal is not a valid floor connector.

        Returns:
            List of ValidationIssues for missing or disconnected floor connectors
        """
        issues = []

        if not layout.primitives:
            return issues

        # Collect all unique z-offsets
        z_offsets: Set[float] = set()
        for prim in layout.primitives.values():
            z_offsets.add(prim.z_offset)

        # Single floor - no vertical connectors needed
        if len(z_offsets) <= 1:
            return issues

        # Sort z-levels to find adjacent pairs
        sorted_z = sorted(z_offsets)

        # Find all VerticalStairHall primitives and check their connection status
        # Track: (bottom_z, top_z, bottom_connected, top_connected, prim_id)
        vertical_connectors: List[Tuple[float, float, bool, bool, str]] = []

        for prim_id, prim in layout.primitives.items():
            if prim.primitive_type == "VerticalStairHall":
                bottom_z = prim.z_offset
                # Get z_level for top portal (uses enhanced get_portal_z_level with fallback)
                top_z_level = prim.get_portal_z_level('top')
                top_z = bottom_z + top_z_level

                # Check if each portal is connected
                bottom_connected = False
                top_connected = False

                for conn in layout.connections:
                    # Check bottom portal
                    if (conn.primitive_a_id == prim_id and conn.portal_a_id == 'bottom') or \
                       (conn.primitive_b_id == prim_id and conn.portal_b_id == 'bottom'):
                        bottom_connected = True
                    # Check top portal
                    if (conn.primitive_a_id == prim_id and conn.portal_a_id == 'top') or \
                       (conn.primitive_b_id == prim_id and conn.portal_b_id == 'top'):
                        top_connected = True

                vertical_connectors.append((bottom_z, top_z, bottom_connected, top_connected, prim_id))

                # Report specific disconnection issues for this stair
                if not bottom_connected:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Disconnected: VerticalStairHall bottom portal at z={bottom_z:.0f}",
                        primitive_id=prim_id,
                        portal_id='bottom',
                    ))
                if not top_connected:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Disconnected: VerticalStairHall top portal at z={top_z:.0f}",
                        primitive_id=prim_id,
                        portal_id='top',
                    ))

        # Check each adjacent z-level pair
        for i in range(len(sorted_z) - 1):
            lower_z = sorted_z[i]
            upper_z = sorted_z[i + 1]

            # Check if any FULLY CONNECTED VerticalStairHall connects these levels
            has_full_connector = False
            has_partial_connector = False

            for bottom_z, top_z, bottom_conn, top_conn, _ in vertical_connectors:
                if abs(bottom_z - lower_z) < 1.0 and abs(top_z - upper_z) < 1.0:
                    if bottom_conn and top_conn:
                        has_full_connector = True
                        break
                    else:
                        # Stair exists but one or both portals disconnected
                        has_partial_connector = True

            if not has_full_connector:
                if has_partial_connector:
                    # Stair exists but is disconnected
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"[Floor] VerticalStairHall between z={lower_z:.0f} and z={upper_z:.0f} has disconnected portal(s)",
                    ))
                else:
                    # No stair at all
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"[Floor] No VerticalStairHall between z={lower_z:.0f} and z={upper_z:.0f}",
                    ))
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"  Add a VerticalStairHall at z={lower_z:.0f} with height_change={upper_z - lower_z:.0f}",
                    ))

        return issues

    def _check_connectivity(self, layout: DungeonLayout) -> Tuple[List[ValidationIssue], int, List[str]]:
        """
        Check that all primitives are connected via BFS traversal.

        Returns: (issues, connected_count, disconnected_ids)
        """
        issues = []

        if not layout.primitives:
            return issues, 0, []

        # Build adjacency graph from connections
        adj: Dict[str, Set[str]] = {pid: set() for pid in layout.primitives}

        for conn in layout.connections:
            if conn.primitive_a_id in adj and conn.primitive_b_id in adj:
                adj[conn.primitive_a_id].add(conn.primitive_b_id)
                adj[conn.primitive_b_id].add(conn.primitive_a_id)

        # BFS from first primitive
        start_id = next(iter(layout.primitives))
        visited: Set[str] = set()
        queue = deque([start_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Find disconnected primitives
        all_ids = set(layout.primitives.keys())
        disconnected = all_ids - visited

        if disconnected:
            for pid in disconnected:
                prim = layout.primitives[pid]
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Disconnected: {prim.primitive_type}",
                    primitive_id=pid
                ))

        return issues, len(visited), list(disconnected)

    def get_reachable_from(self, layout: DungeonLayout, start_id: str) -> Set[str]:
        """Get all primitives reachable from a starting primitive."""
        if start_id not in layout.primitives:
            return set()

        # Build adjacency graph
        adj: Dict[str, Set[str]] = {pid: set() for pid in layout.primitives}
        for conn in layout.connections:
            if conn.primitive_a_id in adj and conn.primitive_b_id in adj:
                adj[conn.primitive_a_id].add(conn.primitive_b_id)
                adj[conn.primitive_b_id].add(conn.primitive_a_id)

        # BFS
        visited: Set[str] = set()
        queue = deque([start_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited


# Singleton validator instance
_validator = LayoutValidator()


def validate_layout(layout: DungeonLayout) -> ValidationResult:
    """Validate a dungeon layout."""
    return _validator.validate(layout)
