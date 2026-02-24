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

from .data_model import DungeonLayout, PlacedPrimitive, Portal, CellCoord, PortalDirection
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

# Multi-floor room types - these have internal stairs connecting entrance (z=0) to upper (z=160)
MULTI_FLOOR_ROOM_TYPES = {
    'Amphitheater', 'CatwalkChamber', 'BalconyRoom', 'SunkenChamber',
    'LibraryArchive', 'Grotto', 'RadialShrine', 'Forge',
}

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
                message=f"{trav_result.isolated_region_count} isolated region(s) not connected to main layout - add halls to create paths",
            ))

        return issues

    def _check_floor_connectors(self, layout: DungeonLayout) -> List[ValidationIssue]:
        """
        Check that adjacent floor levels are connected by vertical connectors.

        Vertical connectors include:
        - VerticalStairHall (bottom/top portals at z_level=0/160)
        - Multi-floor rooms (entrance/upper portals at z_level=0/160)

        This validation catches layouts where multiple z-levels exist but lack
        vertical connectors between them. Without stairs or multi-floor rooms,
        upper/lower floors are unreachable even if they have valid connections
        at their own level.

        Enhanced to check CONNECTIVITY of connector portals, not just existence.
        A connector with a disconnected portal is not a valid floor connector.

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

        # Find all vertical connectors (VerticalStairHall and multi-floor rooms)
        # Track: (bottom_z, top_z, bottom_connected, top_connected, prim_id, connector_type)
        vertical_connectors: List[Tuple[float, float, bool, bool, str, str]] = []

        for prim_id, prim in layout.primitives.items():
            # Check VerticalStairHall
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

                vertical_connectors.append((bottom_z, top_z, bottom_connected, top_connected, prim_id, 'VerticalStairHall'))

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

            # Check multi-floor rooms (entrance at z=0, upper at z=160)
            elif prim.primitive_type in MULTI_FLOOR_ROOM_TYPES:
                entrance_z = prim.z_offset
                # Upper portal is at z_level=160 relative to room origin
                upper_z_level = prim.get_portal_z_level('upper')
                upper_z = entrance_z + upper_z_level

                # Check if each portal is connected
                entrance_connected = False
                upper_connected = False

                for conn in layout.connections:
                    # Check entrance portal
                    if (conn.primitive_a_id == prim_id and conn.portal_a_id == 'entrance') or \
                       (conn.primitive_b_id == prim_id and conn.portal_b_id == 'entrance'):
                        entrance_connected = True
                    # Check upper portal
                    if (conn.primitive_a_id == prim_id and conn.portal_a_id == 'upper') or \
                       (conn.primitive_b_id == prim_id and conn.portal_b_id == 'upper'):
                        upper_connected = True

                vertical_connectors.append((entrance_z, upper_z, entrance_connected, upper_connected, prim_id, prim.primitive_type))

                # Report specific disconnection issues for this multi-floor room
                if not entrance_connected:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Disconnected: {prim.primitive_type} entrance portal at z={entrance_z:.0f}",
                        primitive_id=prim_id,
                        portal_id='entrance',
                    ))
                if not upper_connected:
                    # Only report as error if the upper portal COULD connect to something
                    # (i.e., there's a floor at or above its z-level).
                    # Multi-floor rooms on the TOP floor have upper portals that extend
                    # ABOVE the highest floor - these are expected dead ends, not errors.
                    max_floor_z = sorted_z[-1]
                    if upper_z <= max_floor_z + 2:  # +2 for tolerance
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Disconnected: {prim.primitive_type} upper portal at z={upper_z:.0f}",
                            primitive_id=prim_id,
                            portal_id='upper',
                        ))

        # Check each adjacent z-level pair
        for i in range(len(sorted_z) - 1):
            lower_z = sorted_z[i]
            upper_z = sorted_z[i + 1]

            # Check if any FULLY CONNECTED vertical connector connects these levels
            has_full_connector = False
            has_partial_connector = False

            for bottom_z, top_z, bottom_conn, top_conn, _, connector_type in vertical_connectors:
                if abs(bottom_z - lower_z) < 1.0 and abs(top_z - upper_z) < 1.0:
                    if bottom_conn and top_conn:
                        has_full_connector = True
                        break
                    else:
                        # Connector exists but one or both portals disconnected
                        has_partial_connector = True

            if not has_full_connector:
                if has_partial_connector:
                    # Connector exists but is disconnected
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"[Floor] Floor connector between z={lower_z:.0f} and z={upper_z:.0f} has disconnected portal(s)",
                    ))
                else:
                    # No connector at all
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"[Floor] No floor connector between z={lower_z:.0f} and z={upper_z:.0f}",
                    ))
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"  Add VerticalStairHall or multi-floor room at z={lower_z:.0f} (height_change={upper_z - lower_z:.0f})",
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
            # Build reverse lookup: which primitives is each disconnected one connected to?
            disconnected_peers: Dict[str, Set[str]] = {pid: set() for pid in disconnected}
            for conn in layout.connections:
                if conn.primitive_a_id in disconnected_peers and conn.primitive_b_id in disconnected:
                    disconnected_peers[conn.primitive_a_id].add(conn.primitive_b_id)
                    disconnected_peers[conn.primitive_b_id].add(conn.primitive_a_id)

            # Find open (unconnected) portals on disconnected primitives
            for pid in disconnected:
                prim = layout.primitives[pid]
                peers = disconnected_peers[pid]

                # Find which portals are NOT connected
                connected_portals = set()
                for conn in layout.connections:
                    if conn.primitive_a_id == pid:
                        connected_portals.add(conn.portal_a_id)
                    elif conn.primitive_b_id == pid:
                        connected_portals.add(conn.portal_b_id)

                open_portals = []
                if prim.footprint:
                    for portal in prim.footprint.portals:
                        if portal.id not in connected_portals:
                            portal_cell = prim.get_portal_world_cell(portal)
                            portal_dir = portal.rotated_direction(prim.rotation)
                            open_portals.append((portal.id, portal_cell, portal_dir))

                if peers:
                    # Has connections but in an isolated cluster
                    peer_names = [layout.primitives[p].primitive_type for p in list(peers)[:2]]
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Isolated cluster: {prim.primitive_type} + {', '.join(peer_names)}",
                        primitive_id=pid
                    ))
                    # Show which portal needs connection to main layout
                    if open_portals:
                        portal_id, portal_cell, portal_dir = open_portals[0]
                        adjacent = portal_cell.neighbor(portal_dir)
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"  -> '{portal_id}' portal at ({portal_cell.x},{portal_cell.y}) faces {portal_dir.value}, needs hall at ({adjacent.x},{adjacent.y})",
                            primitive_id=pid
                        ))
                else:
                    # Truly disconnected - no connections at all
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"No connections: {prim.primitive_type} at ({prim.origin_cell.x},{prim.origin_cell.y})",
                        primitive_id=pid
                    ))
                    # Show where portals are
                    if open_portals:
                        portal_id, portal_cell, portal_dir = open_portals[0]
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"  -> '{portal_id}' portal at ({portal_cell.x},{portal_cell.y}) needs adjacent primitive",
                            primitive_id=pid
                        ))

        # If there are disconnected clusters, analyze the gap to main layout
        if disconnected and visited:
            gap_analysis = self._analyze_layout_gap(layout, visited, disconnected)
            issues.extend(gap_analysis)

        return issues, len(visited), list(disconnected)

    def _analyze_layout_gap(self, layout: DungeonLayout, main_ids: Set[str],
                              disconnected_ids: Set[str]) -> List[ValidationIssue]:
        """
        Analyze the gap between main layout and disconnected clusters.

        Finds open portals on main layout and nearest open portals on disconnected
        clusters, then calculates what's needed to bridge them.

        Returns:
            List of ValidationIssues with gap analysis
        """
        issues = []

        # Find open portals on main layout
        main_open_portals = []
        for pid in main_ids:
            prim = layout.primitives[pid]
            if not prim.footprint:
                continue

            connected_portals = set()
            for conn in layout.connections:
                if conn.primitive_a_id == pid:
                    connected_portals.add(conn.portal_a_id)
                elif conn.primitive_b_id == pid:
                    connected_portals.add(conn.portal_b_id)

            for portal in prim.footprint.portals:
                if portal.id not in connected_portals:
                    portal_cell = prim.get_portal_world_cell(portal)
                    portal_dir = portal.rotated_direction(prim.rotation)
                    portal_z = prim.z_offset + prim.get_portal_z_level(portal.id)
                    main_open_portals.append((pid, prim.primitive_type, portal.id,
                                              portal_cell, portal_dir, portal_z))

        # Find open portals on disconnected clusters
        disconnected_open_portals = []
        for pid in disconnected_ids:
            prim = layout.primitives[pid]
            if not prim.footprint:
                continue

            connected_portals = set()
            for conn in layout.connections:
                if conn.primitive_a_id == pid:
                    connected_portals.add(conn.portal_a_id)
                elif conn.primitive_b_id == pid:
                    connected_portals.add(conn.portal_b_id)

            for portal in prim.footprint.portals:
                if portal.id not in connected_portals:
                    portal_cell = prim.get_portal_world_cell(portal)
                    portal_dir = portal.rotated_direction(prim.rotation)
                    portal_z = prim.z_offset + prim.get_portal_z_level(portal.id)
                    disconnected_open_portals.append((pid, prim.primitive_type, portal.id,
                                                       portal_cell, portal_dir, portal_z))

        if not main_open_portals or not disconnected_open_portals:
            return issues

        # Find closest pair of compatible portals (same Z-level, could connect)
        best_gap = None
        best_main = None
        best_disconnected = None

        for main_info in main_open_portals:
            _, main_type, main_portal_id, main_cell, main_dir, main_z = main_info
            # Cell that main portal faces
            main_facing = main_cell.neighbor(main_dir)

            for disc_info in disconnected_open_portals:
                _, disc_type, disc_portal_id, disc_cell, disc_dir, disc_z = disc_info

                # Check Z compatibility
                if abs(main_z - disc_z) > 2:
                    continue

                # Check if portals could face each other
                disc_facing = disc_cell.neighbor(disc_dir)

                # Calculate gap - number of cells between facing cells
                # Portals connect when facing cells are adjacent
                dx = abs(main_facing.x - disc_facing.x)
                dy = abs(main_facing.y - disc_facing.y)

                # Check if they're on the same axis (could connect with halls)
                if main_dir in (PortalDirection.NORTH, PortalDirection.SOUTH):
                    # Vertical connection - need same X
                    if main_facing.x == disc_facing.x:
                        gap = abs(main_facing.y - disc_facing.y)
                        if best_gap is None or gap < best_gap:
                            best_gap = gap
                            best_main = main_info
                            best_disconnected = disc_info
                else:
                    # Horizontal connection - need same Y
                    if main_facing.y == disc_facing.y:
                        gap = abs(main_facing.x - disc_facing.x)
                        if best_gap is None or gap < best_gap:
                            best_gap = gap
                            best_main = main_info
                            best_disconnected = disc_info

        if best_gap is not None and best_main and best_disconnected:
            _, main_type, main_portal_id, main_cell, main_dir, _ = best_main
            _, disc_type, disc_portal_id, disc_cell, disc_dir, _ = best_disconnected
            main_facing = main_cell.neighbor(main_dir)
            disc_facing = disc_cell.neighbor(disc_dir)

            if best_gap == 0:
                # Adjacent but not connected - should have connected automatically
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Gap analysis: {main_type}.{main_portal_id} is adjacent to {disc_type}.{disc_portal_id} but not connected (check Z-levels)",
                ))
            elif best_gap <= 4:
                # Small gap - show exactly what's needed
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Gap analysis: Need {best_gap} cell(s) of halls between main layout and isolated cluster",
                ))
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"  Main: {main_type}.{main_portal_id} at ({main_cell.x},{main_cell.y}) facing {main_dir.value}",
                ))
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"  Isolated: {disc_type}.{disc_portal_id} at ({disc_cell.x},{disc_cell.y}) facing {disc_dir.value}",
                ))
                # Show suggested hall positions
                if main_dir in (PortalDirection.NORTH, PortalDirection.SOUTH):
                    # Vertical gap
                    start_y = min(main_facing.y, disc_facing.y)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"  -> Place StraightHall(s) at x={main_facing.x} from y={start_y} onward",
                    ))
                else:
                    # Horizontal gap
                    start_x = min(main_facing.x, disc_facing.x)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"  -> Place StraightHall(s) at y={main_facing.y} from x={start_x} onward",
                    ))
            else:
                # Large gap or misaligned
                dx = abs(main_facing.x - disc_facing.x)
                dy = abs(main_facing.y - disc_facing.y)
                if dx > 0 and dy > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Gap analysis: Main and isolated clusters are misaligned (X offset: {dx}, Y offset: {dy})",
                    ))
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"  Use SquareCorner or TJunction to change direction, plus halls to bridge",
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Gap analysis: {best_gap} cells between main layout and isolated cluster",
                    ))

        return issues

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
