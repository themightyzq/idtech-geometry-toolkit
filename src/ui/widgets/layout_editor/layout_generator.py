"""
Layout generator for converting DungeonLayout to idTech brushes.

Takes the node-based layout and generates actual brush geometry by:
1. Instantiating primitives from the catalog
2. Configuring portal parameters based on connections
3. Translating to world positions
4. Combining all brushes

Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
- Generation includes validation gates for geometry and portal alignment
- FAIL severity issues are reported in GenerationResult.warnings
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane, Entity
from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
from quake_levelgenerator.src.generators.primitives.portal_system import (
    PORTAL_WIDTH, PORTAL_HEIGHT, PortalWorldPosition, PortalMismatch,
    validate_portal_alignment, PortalDirection as PortalDir, PortalTag
)
from quake_levelgenerator.src.generators.profiles import PROFILE_CATALOG, GameProfile
from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS

from .data_model import (
    DungeonLayout, PlacedPrimitive, Portal, Connection, CellCoord, PortalDirection,
    PrimitiveFootprint
)
from .palette_widget import PRIMITIVE_FOOTPRINTS
from .spatial_validation import SpatialValidator


# =============================================================================
# TAG REGISTRY FOR PORTAL ALIGNMENT
# =============================================================================
# Central registry for collecting and validating portal tags during generation.
# Modules register tags at actual portal positions, then the registry validates
# that connected portals interlock correctly.

@dataclass
class TagRegistry:
    """Central registry for collecting portal tags during generation.

    Modules generate tags at their actual portal positions (in local coords).
    The layout generator transforms these to world space and registers them here.
    After all geometry is generated, validate_connections() checks alignment.
    """
    _tags: Dict[Tuple[str, str], PortalTag] = field(default_factory=dict)

    def register_tag(self, tag: PortalTag) -> None:
        """Register a portal tag.

        Args:
            tag: PortalTag with primitive_id already set
        """
        key = (tag.primitive_id, tag.portal_id)
        self._tags[key] = tag

    def get_tag(self, primitive_id: str, portal_id: str) -> Optional[PortalTag]:
        """Get a registered tag by ID.

        Args:
            primitive_id: ID of the primitive
            portal_id: ID of the portal

        Returns:
            PortalTag if found, None otherwise
        """
        return self._tags.get((primitive_id, portal_id))

    def validate_connections(
        self,
        connections: List['Connection']
    ) -> List[PortalMismatch]:
        """Validate all connected portals have matching tags.

        Args:
            connections: List of Connection objects from the layout

        Returns:
            List of PortalMismatch objects for any misaligned portals
        """
        mismatches: List[PortalMismatch] = []

        for conn in connections:
            tag_a = self._tags.get((conn.primitive_a_id, conn.portal_a_id))
            tag_b = self._tags.get((conn.primitive_b_id, conn.portal_b_id))

            if tag_a is None or tag_b is None:
                # Tags not registered - can't validate
                # (warning should be added by caller)
                continue

            if not tag_a.matches(tag_b):
                # Create PortalWorldPosition objects for mismatch reporting
                pos_a = PortalWorldPosition(
                    primitive_id=tag_a.primitive_id,
                    portal_id=tag_a.portal_id,
                    center_x=tag_a.center_x,
                    center_y=tag_a.center_y,
                    center_z=tag_a.center_z,
                    width=tag_a.width,
                    height=tag_a.height,
                    direction=tag_a.direction,
                )
                pos_b = PortalWorldPosition(
                    primitive_id=tag_b.primitive_id,
                    portal_id=tag_b.portal_id,
                    center_x=tag_b.center_x,
                    center_y=tag_b.center_y,
                    center_z=tag_b.center_z,
                    width=tag_b.width,
                    height=tag_b.height,
                    direction=tag_b.direction,
                )
                mismatches.append(PortalMismatch(
                    primitive_a_id=conn.primitive_a_id,
                    portal_a_id=conn.portal_a_id,
                    primitive_b_id=conn.primitive_b_id,
                    portal_b_id=conn.portal_b_id,
                    position_a=pos_a,
                    position_b=pos_b,
                ))

        return mismatches

    def clear(self) -> None:
        """Clear all registered tags."""
        self._tags.clear()


# =============================================================================
# ROOM PORTAL ALIGNMENT SYSTEM
# =============================================================================
# Room primitives that use the systemic post-rotation portal alignment correction.
# Adding a new room primitive = add to this set + follow the convention = automatic alignment.
#
# Convention: Room primitives MUST generate their entrance portal at room center (X=0 in local coords)
# The layout generator computes the correction purely from footprint definitions.

ROOM_PRIMITIVES_WITH_ENTRANCE = {
    'Tower', 'Storage', 'Prison',
    'Sanctuary', 'Tomb', 'Chamber', 'Armory', 'Cistern', 'Stronghold', 'Courtyard',
    'Arena', 'Laboratory', 'Vault', 'Barracks', 'Shrine', 'Pit',
    'Gatehouse', 'Sewer', 'Ossuary', 'Cloister', 'Colosseum',
    'ThroneRoom',
}

# Multi-portal rooms that handle alignment via offset parameters instead of systemic correction.
# These rooms have portals on different walls, so a single whole-room shift can't align all portals.
# Includes multi-floor rooms which have entrance (SOUTH) + upper (NORTH) portals.
MULTI_PORTAL_ROOMS = {
    'GreatHall', 'Antechamber', 'Hub',
    'Vestibule', 'Processional', 'DoglegRoom',
    'Amphitheater', 'CatwalkChamber', 'BalconyRoom', 'SunkenChamber',
    'LibraryArchive', 'Grotto', 'RadialShrine', 'Forge',
}

# Multi-floor rooms - these have entrance at z_level=0 and upper portal at z_level=160.
# They function as vertical connectors, allowing halls on different floors to connect through them.
MULTI_FLOOR_ROOMS = {
    'Amphitheater', 'CatwalkChamber', 'BalconyRoom', 'SunkenChamber',
    'LibraryArchive', 'Grotto', 'RadialShrine', 'Forge',
}

# All room primitives (enclosed spaces)
# Halls are excluded as they are traversal spaces
ROOM_PRIMITIVES: Set[str] = {
    'Sanctuary', 'Tomb', 'Tower', 'Chamber', 'Storage',
    'GreatHall', 'Prison', 'Armory', 'Cistern', 'Stronghold', 'Courtyard',
    'Arena', 'Laboratory', 'Vault', 'Barracks', 'Shrine', 'Pit', 'Antechamber',
    'Hub',
    'Gatehouse', 'Sewer', 'Ossuary', 'Cloister', 'Colosseum',
    'ThroneRoom', 'Vestibule', 'Processional', 'DoglegRoom',
    # Multi-Floor Rooms (rooms with internal upper portals and stairs)
    'Amphitheater', 'CatwalkChamber', 'BalconyRoom', 'SunkenChamber',
    'LibraryArchive', 'Grotto', 'RadialShrine', 'Forge',
}


def _rotate_point(x: float, y: float, rotation: int) -> Tuple[float, float]:
    """Rotate a 2D point around origin by rotation degrees clockwise.

    Args:
        x, y: Point coordinates
        rotation: Rotation in degrees (0, 90, 180, 270)

    Returns:
        Rotated (x, y) coordinates
    """
    if rotation == 90:
        return (y, -x)
    elif rotation == 180:
        return (-x, -y)
    elif rotation == 270:
        return (-y, x)
    return (x, y)


def _compute_room_portal_correction(
    prim: PlacedPrimitive,
    footprint: PrimitiveFootprint,
    grid_size: int
) -> Tuple[float, float]:
    """Compute correction to align room portal with footprint cell.

    SYSTEMIC: Works for ANY room primitive without primitive-specific code.
    Assumes convention: entrance portal at room center (local X=0).

    The correction is computed AFTER rotation and normalization, ensuring
    that the portal ends up at the correct footprint cell position regardless
    of rotation.

    Args:
        prim: The placed primitive
        footprint: The primitive's footprint definition
        grid_size: Size of grid cells in idTech units

    Returns:
        (dx, dy) correction to apply after normalization, before world translation.
    """
    # Find entrance portal in footprint (generic lookup)
    portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
    if portal is None:
        return (0.0, 0.0)

    # Get rotated footprint dimensions
    fp_width, fp_depth = footprint.rotated_size(prim.rotation)

    # Compute target position from rotated portal cell (using FOOTPRINT data only)
    rotated_cell = portal.world_cell(
        CellCoord(0, 0), prim.rotation,
        footprint.width_cells, footprint.depth_cells
    )
    rotated_dir = portal.rotated_direction(prim.rotation)

    # Target portal position: center of the portal's cell
    target_x = (rotated_cell.x + 0.5) * grid_size
    target_y = (rotated_cell.y + 0.5) * grid_size

    # CONVENTION: Room generates with entrance at center (local X=0)
    # After rotation and normalization, room center is at:
    norm_center_x = (fp_width * grid_size) / 2
    norm_center_y = (fp_depth * grid_size) / 2

    # Compute correction to move room so portal aligns with target cell
    # For brush correction: only shift perpendicular to portal wall (the wall stays at room edge)
    # For tag-based validation: tags need to match actual portal positions
    #
    # The portal direction determines which axis the portal wall lies along:
    # - SOUTH/NORTH portals: wall parallel to X axis, shift X to align center
    # - EAST/WEST portals: wall parallel to Y axis, shift Y to align center

    if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
        # Portal varies along X axis - shift X to align with target cell
        correction_x = target_x - norm_center_x
        correction_y = 0.0
    else:  # EAST or WEST
        # Portal varies along Y axis - shift Y to align with target cell
        correction_x = 0.0
        correction_y = target_y - norm_center_y

    return (correction_x, correction_y)


@dataclass
class GenerationResult:
    """Result of layout generation."""
    brushes: List[Brush]
    spawn_position: Tuple[float, float, float]
    primitive_count: int
    brush_count: int
    warnings: List[str]
    portal_mismatches: List[PortalMismatch] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)


def _translate_brush(brush: Brush, offset: Tuple[float, float, float]) -> Brush:
    """Translate a brush by an offset."""
    ox, oy, oz = offset
    new_planes = []
    for plane in brush.planes:
        new_p1 = (plane.p1[0] + ox, plane.p1[1] + oy, plane.p1[2] + oz)
        new_p2 = (plane.p2[0] + ox, plane.p2[1] + oy, plane.p2[2] + oz)
        new_p3 = (plane.p3[0] + ox, plane.p3[1] + oy, plane.p3[2] + oz)
        new_planes.append(Plane(
            p1=new_p1,
            p2=new_p2,
            p3=new_p3,
            texture=plane.texture,
            x_offset=plane.x_offset,
            y_offset=plane.y_offset,
            rotation=plane.rotation,
            x_scale=plane.x_scale,
            y_scale=plane.y_scale,
        ))
    return Brush(planes=new_planes)


def _rotate_brush_90(brush: Brush) -> Brush:
    """Rotate a brush 90 degrees around the Z axis (clockwise).

    Clockwise rotation (looking down at XY plane):
    (x, y, z) -> (y, -x, z)

    This matches the portal rotation convention in data_model.py.
    """
    new_planes = []
    for plane in brush.planes:
        # Rotate points: (x, y, z) -> (y, -x, z) [clockwise]
        new_p1 = (plane.p1[1], -plane.p1[0], plane.p1[2])
        new_p2 = (plane.p2[1], -plane.p2[0], plane.p2[2])
        new_p3 = (plane.p3[1], -plane.p3[0], plane.p3[2])
        new_planes.append(Plane(
            p1=new_p1,
            p2=new_p2,
            p3=new_p3,
            texture=plane.texture,
            x_offset=plane.x_offset,
            y_offset=plane.y_offset,
            rotation=plane.rotation,
            x_scale=plane.x_scale,
            y_scale=plane.y_scale,
        ))
    return Brush(planes=new_planes)


def _rotate_brush(brush: Brush, rotation: int) -> Brush:
    """Rotate a brush by 0, 90, 180, or 270 degrees around Z axis."""
    result = brush
    for _ in range(rotation // 90):
        result = _rotate_brush_90(result)
    return result


def _compute_brush_bounds_xy(brushes: List[Brush]) -> Tuple[float, float, float, float]:
    """Compute the XY bounding box of a list of brushes.

    Returns (min_x, min_y, max_x, max_y).
    """
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for brush in brushes:
        for plane in brush.planes:
            for p in [plane.p1, plane.p2, plane.p3]:
                min_x = min(min_x, p[0])
                min_y = min(min_y, p[1])
                max_x = max(max_x, p[0])
                max_y = max(max_y, p[1])

    return (min_x, min_y, max_x, max_y)


def _normalize_brushes_to_origin(brushes: List[Brush], footprint_width: int,
                                  footprint_depth: int, grid_size: int) -> List[Brush]:
    """Normalize brush geometry so it aligns with footprint cells starting at origin.

    After rotation, primitive geometry may have various offsets from the expected
    position. This function shifts geometry so it starts at (0, 0).

    The goal is that a primitive placed at cell (0, 0) has its geometry filling
    the footprint cells starting from (0, 0).
    """
    if not brushes:
        return brushes

    min_x, min_y, max_x, max_y = _compute_brush_bounds_xy(brushes)

    # Simply shift geometry so its bottom-left corner is at (0, 0)
    shift_x = -min_x
    shift_y = -min_y

    if abs(shift_x) > 0.1 or abs(shift_y) > 0.1:
        brushes = [_translate_brush(b, (shift_x, shift_y, 0)) for b in brushes]

    return brushes


class LayoutGenerator:
    """Generates idTech brushes from a DungeonLayout."""

    def __init__(
        self,
        layout: DungeonLayout,
        random_seed: Optional[int] = None,
        game_profile: Optional[GameProfile] = None,
    ):
        """
        Initialize the layout generator.

        Args:
            layout: The dungeon layout to generate brushes from
            random_seed: Optional seed for reproducible generation
            game_profile: Game profile for engine-specific texture mappings.
                         If None, uses the default profile (Doom 3).
        """
        self._layout = layout
        self._warnings: List[str] = []
        self._random_seed = random_seed
        # Track portal world positions for validation
        # Key: (primitive_id, portal_id), Value: PortalWorldPosition
        self._portal_positions: Dict[Tuple[str, str], PortalWorldPosition] = {}

        # Tag registry for tag-based portal validation (Phase 2)
        self._tag_registry = TagRegistry()

        # Resolve game profile
        self._profile: GameProfile = game_profile or PROFILE_CATALOG.get_default_profile()

    def generate(self) -> GenerationResult:
        """Generate all brushes from the layout."""
        all_brushes: List[Brush] = []
        spawn_position = (0.0, 0.0, 32.0)  # Default spawn
        self._portal_positions.clear()
        self._tag_registry.clear()

        if not self._layout.primitives:
            return GenerationResult(
                brushes=[],
                spawn_position=spawn_position,
                primitive_count=0,
                brush_count=0,
                warnings=["Layout is empty"]
            )

        # Pre-flight spatial validation (3D interpenetration check)
        self._check_spatial_collisions()

        # Build connection map for portal configuration
        portal_connections = self._build_portal_connections()

        # Generate brushes for each primitive
        for prim_id, prim in self._layout.primitives.items():
            brushes = self._generate_primitive(prim, portal_connections.get(prim_id, {}))
            all_brushes.extend(brushes)
            # Compute and store portal world positions for this primitive
            self._compute_portal_positions(prim)

        # Validate portal alignment using tag-based validation
        tag_mismatches = self._tag_registry.validate_connections(self._layout.connections)

        # Also run legacy validation for comparison/fallback
        portal_mismatches = self._validate_portal_alignment()

        # Prefer tag-based mismatches if available, otherwise use legacy
        if tag_mismatches:
            portal_mismatches = tag_mismatches

        # Add mismatch warnings
        for mismatch in portal_mismatches:
            self._warnings.append(str(mismatch))

        # Calculate spawn position (center of first primitive's full footprint)
        # Must be at footprint center so the entity is inside polygonal rooms
        first_prim = next(iter(self._layout.primitives.values()))
        cell = first_prim.origin_cell
        grid_size = self._layout.grid_size
        footprint = first_prim.footprint or PRIMITIVE_FOOTPRINTS.get(first_prim.primitive_type)
        if footprint:
            fp_w, fp_d = footprint.rotated_size(first_prim.rotation)
            spawn_x = (cell.x + fp_w / 2) * grid_size
            spawn_y = (cell.y + fp_d / 2) * grid_size
        else:
            spawn_x = cell.x * grid_size + grid_size / 2
            spawn_y = cell.y * grid_size + grid_size / 2
        spawn_z = first_prim.z_offset + 32  # 32 units above floor
        spawn_position = (spawn_x, spawn_y, spawn_z)

        entities: List[Entity] = []

        # Run validation gate per CLAUDE.md Section 3 (Quality Gates [BINDING])
        self._run_validation_gate(all_brushes, portal_mismatches)

        # Center the layout at world origin so geometry appears at (0,0,0)
        # in the 3D preview and exported maps
        offset_x, offset_y = self._compute_centering_offset()
        if offset_x != 0 or offset_y != 0:
            all_brushes = [_translate_brush(b, (offset_x, offset_y, 0))
                           for b in all_brushes]
            spawn_position = (
                spawn_position[0] + offset_x,
                spawn_position[1] + offset_y,
                spawn_position[2],
            )

        return GenerationResult(
            brushes=all_brushes,
            spawn_position=spawn_position,
            primitive_count=len(self._layout.primitives),
            brush_count=len(all_brushes),
            warnings=self._warnings,
            portal_mismatches=portal_mismatches,
            entities=entities,
        )

    def _run_validation_gate(
        self,
        brushes: List[Brush],
        portal_mismatches: List[PortalMismatch]
    ) -> None:
        """Run validation gate on generated geometry.

        Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
        - Geometry: No degenerate geometry, all integer coordinates
        - Portals: Connected portals match within ±2 units

        Adds issues to self._warnings rather than raising exceptions
        to allow users to see and fix issues.

        Args:
            brushes: Generated brush geometry
            portal_mismatches: Portal alignment mismatches
        """
        try:
            from quake_levelgenerator.src.validation import get_validator

            validator = get_validator()
            result = validator.validate_generation(
                brushes,
                portal_mismatches,
                check_geometry=True,
                check_portals=True,
            )

            # Add validation issues as warnings (don't fail generation)
            # Per CLAUDE.md Section 3: Export gate blocks; placement gate logs.
            if result.issues:
                for issue in result.errors:
                    self._warnings.append(
                        f"[FAIL] {issue.code} module={issue.location or '-'} "
                        f"connector=- transform=- file={issue.file_path or '-'} :: "
                        f"{issue.message} :: fix={issue.remediation or 'N/A'}"
                    )
                for issue in result.warnings:
                    logger.warning(f"Validation: {issue.code}: {issue.message}")

        except ImportError:
            # Validation package not available - skip
            pass
        except Exception as e:
            logger.warning(f"Validation gate error: {e}")

    def _compute_centering_offset(self) -> Tuple[float, float]:
        """Compute a world-space offset that centers the layout at the origin.

        Finds the bounding box of all placed primitives (using their footprints
        and rotations) and returns the negative of the center so that applying
        the offset moves the layout center to (0, 0).

        The result is always a multiple of 64 (grid-aligned per CLAUDE.md §4),
        ensuring all coordinates remain integers.
        """
        if not self._layout.primitives:
            return (0.0, 0.0)

        grid_size = self._layout.grid_size
        min_cx = float('inf')
        min_cy = float('inf')
        max_cx = float('-inf')
        max_cy = float('-inf')

        for prim in self._layout.primitives.values():
            footprint = prim.footprint or PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
            if footprint:
                fp_w, fp_d = footprint.rotated_size(prim.rotation)
            else:
                fp_w, fp_d = 1, 1

            cx = prim.origin_cell.x
            cy = prim.origin_cell.y
            min_cx = min(min_cx, cx)
            min_cy = min(min_cy, cy)
            max_cx = max(max_cx, cx + fp_w)
            max_cy = max(max_cy, cy + fp_d)

        # Center of bounding box in world units.
        # (min + max) * grid_size / 2 is always a multiple of 64 because
        # min/max are integers and grid_size is 128.
        center_x = (min_cx + max_cx) * grid_size / 2.0
        center_y = (min_cy + max_cy) * grid_size / 2.0

        # Snap to 64-unit grid for safety
        center_x = round(center_x / 64) * 64
        center_y = round(center_y / 64) * 64

        return (-center_x, -center_y)

    def _build_portal_connections(self) -> Dict[str, Dict[str, bool]]:
        """
        Build a map of primitive_id -> portal_id -> is_connected.

        Used to configure portal_* parameters on primitives.
        """
        result: Dict[str, Dict[str, bool]] = {}

        for prim_id in self._layout.primitives:
            result[prim_id] = {}

        for conn in self._layout.connections:
            if conn.primitive_a_id in result:
                result[conn.primitive_a_id][conn.portal_a_id] = True
            if conn.primitive_b_id in result:
                result[conn.primitive_b_id][conn.portal_b_id] = True

        return result

    def _check_spatial_collisions(self) -> None:
        """
        Pre-flight check for 3D spatial collisions in the layout.

        Detects interpenetration between primitives at different Z-levels
        before generating geometry. Adds warnings for any detected collisions.
        """
        # Ensure all primitives have footprints set
        for prim in self._layout.primitives.values():
            if prim.footprint is None:
                footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
                if footprint:
                    prim.set_footprint(footprint)

        # Run spatial validation
        validator = SpatialValidator(self._layout)
        collisions = validator.validate_all()

        # Add warnings for detected collisions
        for collision in collisions:
            if collision.is_critical:
                z_min, z_max = collision.overlap_z_range
                self._warnings.append(
                    f"SPATIAL COLLISION: {collision.message} "
                    f"(Z overlap: {z_min:.0f}-{z_max:.0f}). "
                    f"{collision.suggestion}"
                )

    def _transform_tag_like_brush(
        self, tag: 'PortalTag', prim: PlacedPrimitive,
        footprint, grid_size: int,
        world_x: float, world_y: float, world_z: float
    ) -> 'PortalTag':
        """Transform a portal tag using the same logic as brush transformations.

        For halls: use footprint portal data to compute exact world position at cell edges.
        This ensures tags match actual portal positions in the layout.
        """
        # For halls with footprints, compute position from footprint data
        if footprint and prim.primitive_type not in ROOM_PRIMITIVES_WITH_ENTRANCE:
            # Find matching portal in footprint
            fp_portal = next((p for p in footprint.portals if p.id == tag.portal_id), None)
            if fp_portal:
                # Get rotated cell position in world coordinates
                rotated_cell = fp_portal.world_cell(
                    prim.origin_cell, prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = fp_portal.rotated_direction(prim.rotation)

                # Tag position at cell edge based on direction
                if rotated_dir == PortalDirection.SOUTH:
                    tag_x = (rotated_cell.x + 0.5) * grid_size
                    tag_y = rotated_cell.y * grid_size  # South edge
                elif rotated_dir == PortalDirection.NORTH:
                    tag_x = (rotated_cell.x + 0.5) * grid_size
                    tag_y = (rotated_cell.y + 1) * grid_size  # North edge
                elif rotated_dir == PortalDirection.WEST:
                    tag_x = rotated_cell.x * grid_size  # West edge
                    tag_y = (rotated_cell.y + 0.5) * grid_size
                else:  # EAST
                    tag_x = (rotated_cell.x + 1) * grid_size  # East edge
                    tag_y = (rotated_cell.y + 0.5) * grid_size
                tag_z = prim.z_offset + tag.center_z

                return PortalTag(
                    primitive_id=prim.id,
                    portal_id=tag.portal_id,
                    center_x=tag_x,
                    center_y=tag_y,
                    center_z=tag_z,
                    direction=rotated_dir,
                    width=tag.width,
                    height=tag.height,
                    tag_type=tag.tag_type,
                )

        # Fallback: apply basic transformations
        # 1. Apply rotation
        world_tag = tag.transformed(0, 0, 0, prim.rotation)

        # 2. Apply world translation (no normalization for fallback)
        world_tag = world_tag.transformed(world_x, world_y, world_z, 0)

        return world_tag

    def _generate_primitive(self, prim: PlacedPrimitive,
                            connected_portals: Dict[str, Tuple[bool, bool]]) -> List[Brush]:
        """Generate brushes for a single primitive."""

        grid_size = self._layout.grid_size

        # Get the primitive class from catalog
        cls = PRIMITIVE_CATALOG.get_primitive(prim.primitive_type)
        if cls is None:
            self._warnings.append(f"Unknown primitive type: {prim.primitive_type}")
            return []

        # Get footprint - use primitive's footprint if set, otherwise look up from catalog
        footprint = prim.footprint
        if footprint is None:
            footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
            if footprint is None:
                self._warnings.append(f"No footprint for {prim.primitive_type}")

        try:
            # Create instance
            instance = cls()

            # Apply parameters
            params = dict(prim.parameters)

            # Apply surface-specific textures from global settings
            for surface in ['wall', 'floor', 'ceiling', 'trim', 'structural']:
                params[f'texture_{surface}'] = TEXTURE_SETTINGS.get_texture(surface)
            # Default 'texture' param for backward compatibility
            params['texture'] = TEXTURE_SETTINGS.get_texture('wall')

            # Configure geometry dimensions based on footprint
            self._configure_geometry_params(prim, params, footprint)

            # Configure portal parameters based on connections
            self._configure_portal_params(prim, params, connected_portals)

            instance.apply_params(params)

            # Set internal params directly (not in schema, so apply_params ignores them)
            # These are prefixed with underscore to indicate they're internal
            for key, value in params.items():
                if key.startswith('_') and hasattr(instance, key):
                    setattr(instance, key, value)

            # Override class-level constants that apply_params doesn't handle
            if 'WALL_THICKNESS' in params:
                instance.WALL_THICKNESS = params['WALL_THICKNESS']

            # Set surface textures from TEXTURE_SETTINGS (same as Module Mode)
            # These are internal attributes that control per-surface texturing
            instance._texture_wall = TEXTURE_SETTINGS.get_texture('wall')
            instance._texture_floor = TEXTURE_SETTINGS.get_texture('floor')
            instance._texture_ceiling = TEXTURE_SETTINGS.get_texture('ceiling')
            instance._texture_trim = TEXTURE_SETTINGS.get_texture('trim')
            instance._texture_structural = TEXTURE_SETTINGS.get_texture('structural')

            # Generate brushes (at origin)
            brushes = instance.generate()

            # Rotate if needed
            if prim.rotation != 0:
                brushes = [_rotate_brush(b, prim.rotation) for b in brushes]

            # Normalize geometry to align with footprint cells
            # This is critical for ensuring portals align between adjacent primitives
            if footprint:
                # Get rotated footprint dimensions
                fp_width, fp_depth = footprint.rotated_size(prim.rotation)
                brushes = _normalize_brushes_to_origin(
                    brushes, fp_width, fp_depth, grid_size
                )

            # Apply portal alignment correction for room primitives (SYSTEMIC)
            # Works for ANY room primitive in ROOM_PRIMITIVES_WITH_ENTRANCE.
            # Computes correction purely from footprint definitions.
            if footprint and prim.primitive_type in ROOM_PRIMITIVES_WITH_ENTRANCE:
                dx, dy = _compute_room_portal_correction(prim, footprint, grid_size)
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    brushes = [_translate_brush(b, (dx, dy, 0)) for b in brushes]

            # Translate to world position
            world_x = prim.origin_cell.x * grid_size
            world_y = prim.origin_cell.y * grid_size
            world_z = prim.z_offset

            brushes = [_translate_brush(b, (world_x, world_y, world_z)) for b in brushes]

            # Collect and transform portal tags to world space
            # For room primitives with entrance: compute target position from footprint
            # For halls: apply brush-like transformations
            for tag in instance.get_generated_tags():
                world_tag = tag.transformed(0, 0, 0, 0)  # Copy tag

                if footprint and prim.primitive_type in ROOM_PRIMITIVES_WITH_ENTRANCE:
                    # For rooms: place tag at exact portal position on cell edge
                    # Find matching portal in footprint
                    fp_portal = next((p for p in footprint.portals if p.id == tag.portal_id), None)
                    if fp_portal:
                        # Get rotated cell position in world coordinates
                        rotated_cell = fp_portal.world_cell(
                            prim.origin_cell, prim.rotation,
                            footprint.width_cells, footprint.depth_cells
                        )
                        rotated_dir = fp_portal.rotated_direction(prim.rotation)

                        # Tag position depends on portal direction:
                        # - SOUTH/NORTH portals: X at cell center, Y at cell edge
                        # - EAST/WEST portals: Y at cell center, X at cell edge
                        if rotated_dir == PortalDirection.SOUTH:
                            tag_x = (rotated_cell.x + 0.5) * grid_size
                            tag_y = rotated_cell.y * grid_size  # South edge
                        elif rotated_dir == PortalDirection.NORTH:
                            tag_x = (rotated_cell.x + 0.5) * grid_size
                            tag_y = (rotated_cell.y + 1) * grid_size  # North edge
                        elif rotated_dir == PortalDirection.WEST:
                            tag_x = rotated_cell.x * grid_size  # West edge
                            tag_y = (rotated_cell.y + 0.5) * grid_size
                        else:  # EAST
                            tag_x = (rotated_cell.x + 1) * grid_size  # East edge
                            tag_y = (rotated_cell.y + 0.5) * grid_size
                        tag_z = prim.z_offset + tag.center_z

                        world_tag = PortalTag(
                            primitive_id=prim.id,
                            portal_id=tag.portal_id,
                            center_x=tag_x,
                            center_y=tag_y,
                            center_z=tag_z,
                            direction=rotated_dir,
                            width=tag.width,
                            height=tag.height,
                            tag_type=tag.tag_type,
                        )
                    else:
                        # Fallback: apply transformations like brushes
                        world_tag = self._transform_tag_like_brush(
                            tag, prim, footprint, grid_size, world_x, world_y, world_z
                        )
                else:
                    # For halls: apply brush-like transformations
                    world_tag = self._transform_tag_like_brush(
                        tag, prim, footprint, grid_size, world_x, world_y, world_z
                    )

                # Set primitive ID and register
                world_tag.primitive_id = prim.id
                self._tag_registry.register_tag(world_tag)

            return brushes

        except Exception as e:
            self._warnings.append(f"Failed to generate {prim.primitive_type}: {e}")
            return []

    def _configure_geometry_params(self, prim: PlacedPrimitive, params: Dict,
                                      footprint) -> None:
        """Configure geometry dimensions based on footprint.

        Ensures primitive geometry fills its footprint cells so portals align
        correctly between adjacent primitives.

        HIERARCHICAL OVERRIDE: User parameters (in prim.parameters) take precedence
        over computed defaults. This allows full parameter editing in Layout Mode.

        Args:
            prim: The placed primitive being configured
            params: Parameters dict to modify (mutated in-place)
            footprint: PrimitiveFootprint for this primitive type (may come from
                      prim.footprint or from PRIMITIVE_FOOTPRINTS lookup)
        """
        if not footprint:
            return

        grid_size = self._layout.grid_size
        fp_width = footprint.width_cells
        fp_depth = footprint.depth_cells

        # User parameters from prim.parameters take precedence over computed values
        user_params = prim.parameters

        def set_if_not_user(key: str, value) -> None:
            """Set parameter only if user hasn't overridden it."""
            if key not in user_params:
                params[key] = value

        ptype = prim.primitive_type

        if ptype == 'Crossroads':
            # Crossroads is centered in its footprint
            # Arms extend from center to the edges of the footprint
            # For a 3x3 footprint, center is at (1.5, 1.5) cells
            #
            # Geometry extends: arm_length + t in each direction from center
            # Total extent = 2 * (arm_length + t)
            # For footprint to fit: 2 * (arm_length + t) = footprint_size
            # So: arm_length = footprint_size / 2 - t
            #
            # IMPORTANT: Use consistent hall_width with StraightHall for portal alignment
            t = params.get('wall_thickness', 16)
            hall_width = grid_size - 2*t  # Same as StraightHall for alignment
            set_if_not_user('hall_width', hall_width)
            hw = hall_width / 2

            # Distance from center to edge of footprint
            half_width = (fp_width * grid_size) / 2
            half_depth = (fp_depth * grid_size) / 2

            # Arm length accounts for wall thickness at end
            set_if_not_user('arm_north_length', half_depth - t)
            set_if_not_user('arm_south_length', half_depth - t)
            set_if_not_user('arm_east_length', half_width - t)
            set_if_not_user('arm_west_length', half_width - t)

        elif ptype == 'StraightHall':
            # StraightHall has 1xN footprint, must fit within 1 cell width
            # Geometry X extent: 2*hw + 2*t (hall_width + walls on both sides)
            # For 1-cell footprint to fit: 2*hw + 2*t = 128
            # So: hw = (128 - 2*t) / 2 = (128 - 32) / 2 = 48, hall_width = 96
            #
            # Geometry Y extent: (oy - t) to (oy + length + t) = length + 2*t
            # For footprint to fit: length + 2*t = footprint → length = footprint - 2*t
            t = params.get('wall_thickness', 16)
            set_if_not_user('hall_width', grid_size - 2*t)  # 96 for 128 grid, fits in 1 cell
            set_if_not_user('length', fp_depth * grid_size - 2*t)

        elif ptype == 'NarrowPassage':
            # NarrowPassage: same geometry as StraightHall (1x2 footprint)
            t = params.get('wall_thickness', 16)
            set_if_not_user('hall_width', grid_size - 2*t)
            set_if_not_user('length', fp_depth * grid_size - 2*t)

        elif ptype == 'WideCorridor':
            # WideCorridor: 3x2 footprint, wider corridor centered on middle column
            t = params.get('wall_thickness', 16)
            set_if_not_user('hall_width', grid_size - 2*t)
            params['_center_x_offset'] = 1.5 * grid_size  # Internal, always computed
            params['_center_y_offset'] = 1.0 * grid_size  # Internal, always computed

        elif ptype == 'TJunction':
            # TRUE T-Junction: crossbar WEST-EAST (row 0), stem NORTH (row 1)
            # Footprint 3x2:
            #   Row 1: [ ] [N] [ ]  <- NORTH portal at (1,1) - stem top
            #   Row 0: [W] [+] [E]  <- WEST at (0,0), EAST at (2,0) - crossbar
            #              ^--- center junction (no portal)
            #
            # IMPORTANT: Use consistent hall_width with StraightHall for portal alignment
            t = params.get('wall_thickness', 16)
            hall_width = grid_size - 2*t  # Same as StraightHall for alignment
            set_if_not_user('hall_width', hall_width)
            hw = hall_width / 2

            # Center of the T is at cell (1, 0) center
            # X: cell 1 center = 1.5 * grid_size
            # Y: row 0 center = 0.5 * grid_size
            params['_center_x_offset'] = 1.5 * grid_size  # Internal, always computed
            params['_center_y_offset'] = 0.5 * grid_size  # Internal, always computed

            # Crossbar spans from cell 0 to cell 2 (X direction)
            # Total X span = 3 * grid_size, minus wall thickness at ends
            # Crossbar length = distance from WEST portal to EAST portal
            set_if_not_user('crossbar_length', 3 * grid_size - 2*t)  # = 384 - 32 = 352

            # Stem spans from center (row 0 middle) to far edge (row 1 end)
            # Center is at Y = 0.5 * grid_size, stem must reach Y = fp_depth * grid_size
            # stem_length = (fp_depth - 0.5) * grid_size - t
            # For 3x2 footprint: (2 - 0.5) * 128 - 16 = 1.5 * 128 - 16 = 176
            set_if_not_user('stem_length', (fp_depth - 0.5) * grid_size - t)

        elif ptype == 'SquareCorner':
            # SquareCorner: 90-degree corner (2x2 footprint)
            # Junction should be at cell (0,1) center for proper portal alignment:
            # - Portal 'a' at cell (0,0) facing SOUTH: X center = 64
            # - Portal 'b' at cell (1,1) facing EAST: Y center = 192
            #
            # Corridor cells: (0,0), (0,1), (1,1) - L-shaped
            # Cell (1,0) is solid wall
            #
            # IMPORTANT: Use consistent hall_width with StraightHall for portal alignment
            t = params.get('wall_thickness', 16)
            hall_width = grid_size - 2*t  # Same as StraightHall for alignment
            set_if_not_user('hall_width', hall_width)
            hw = hall_width / 2

            # Junction center at cell (0,1) = local position (0.5*grid, 1.5*grid)
            params['_center_x_offset'] = 0.5 * grid_size  # 64 - center of column 0
            params['_center_y_offset'] = 1.5 * grid_size  # 192 - center of row 1

            # Arm lengths based on junction position to footprint edges
            arm_len = min(fp_width, fp_depth) * grid_size - hw - 2*t
            set_if_not_user('arm_length', arm_len)

        elif ptype == 'WideCorner':
            # WideCorner: sweeping 90-degree corner (2x2 footprint)
            # Same portal positions as SquareCorner, wider interior turn
            t = params.get('wall_thickness', 16)
            hall_width = grid_size - 2*t
            set_if_not_user('hall_width', hall_width)
            hw = hall_width / 2

            # Junction center at cell (0,1) = local position (0.5*grid, 1.5*grid)
            params['_center_x_offset'] = 0.5 * grid_size
            params['_center_y_offset'] = 1.5 * grid_size

            arm_len = min(fp_width, fp_depth) * grid_size - hw - 2*t
            set_if_not_user('arm_length', arm_len)

        elif ptype == 'VerticalStairHall':
            # VerticalStairHall: 2x4 footprint, stairwell connecting two floor levels
            # Geometry fills full footprint with walls INSIDE (not extending beyond)
            # - Portal openings at footprint edges for proper alignment
            # - Walls overlap with landing floors
            #
            # IMPORTANT: Use consistent hall_width with StraightHall for portal alignment
            t = params.get('wall_thickness', 16)
            hall_width = grid_size - 2*t  # Same as StraightHall for alignment
            set_if_not_user('hall_width', hall_width)
            hw = hall_width / 2

            # Stair run fills footprint minus landings at each end
            # Total Y = full footprint (walls are inside, not extending beyond)
            # Landing at each end = hw
            # stair_run = footprint - 2*landing
            total_y = fp_depth * grid_size
            set_if_not_user('stair_run', total_y - 2*hw)

            # Height change is one floor (160 units by default to match FLOOR_LEVELS)
            # This matches the z_level=160 on the top portal
            set_if_not_user('height_change', 160.0)

        # === ROOM PRIMITIVES ===
        # Room dimensions must match footprint so geometry aligns with grid

        elif ptype == 'Sanctuary':
            # Sanctuary (formerly Chapel): nave_width is full width, nave_length is depth
            # Footprint: 3x4 cells = 384 x 512 units
            # Room t=16: full width = footprint - 2*t
            #
            # FORCE single_nave and apse=False:
            # Other types (basilica, cruciform, hall_church) add aisles/transepts
            # that extend geometry beyond the footprint boundary, causing BSP leaks.
            # Apse extends geometry beyond the back wall. These MUST be forced
            # (not set_if_not_user) because the parameter randomizer may set them.
            params['sanctuary_type'] = 'single_nave'
            params['apse'] = False
            set_if_not_user('nave_width', fp_width * grid_size - 32)  # Subtract for walls (2*t)
            set_if_not_user('nave_length', fp_depth * grid_size - 32)
            # Sanctuary default nave_height (192) exceeds hall_height (128).
            # Force to 128 to prevent BSP leaks at portal boundaries.
            set_if_not_user('nave_height', 128)

        elif ptype == 'Tomb':
            # Tomb (formerly Crypt): width is FULL width (not half), length is depth
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: full width = footprint - 2*t
            #
            # FORCE alcove_count=0 and coffin_layout='rows':
            # The 'alcoves' coffin_layout skips solid side walls and extends
            # floor/ceiling by alcove_depth (48u) beyond the footprint boundary,
            # causing BSP leaks at module connections. These params MUST be
            # forced (not set_if_not_user) because the random layout generator
            # may set coffin_layout='alcoves' in the primitive's parameters.
            params['alcove_count'] = 0
            params['coffin_layout'] = 'rows'
            set_if_not_user('width', fp_width * grid_size - 32)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Tomb default height (96) is shorter than hall_height (128).
            # At portal boundaries, this creates a gap between the Tomb ceiling
            # and the hall ceiling, causing BSP leaks. Force tomb_height to 128
            # to match hall height in layout mode.
            set_if_not_user('tomb_height', 128)
            set_if_not_user('shell_sides', 4)  # Force rectangular — multi-portal room

        elif ptype == 'Tower':
            # Tower: uses tower_radius (octagonal approximation)
            # Footprint: 2x2 cells = 256 x 256 units
            # Room t=16: radius = footprint/2 - t
            # Radius should fit within the footprint, so use half the smaller dimension
            set_if_not_user('tower_radius', min(fp_width, fp_depth) * grid_size / 2 - 16)
            # Tower default tower_height (768) — use 256 in layout mode.
            set_if_not_user('tower_height', 256)
            # Portal alignment handled by systemic post-rotation correction

        elif ptype == 'Chamber':
            # Chamber: width is half-width, length is depth
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Storage':
            # Storage (merged Cellar + Storeroom): width is half-width, length is depth
            # Footprint: 2x2 cells = 256 x 256 units
            # Disable alcoves to keep geometry within footprint
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('alcove_count', 0)
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Portal alignment handled by systemic post-rotation correction

        elif ptype == 'GreatHall':
            # GreatHall: width is half-width, length is depth
            # Footprint: 4x6 cells = 512 x 768 units
            # Disable fireplace in layout mode: it creates an asymmetric Y extent
            # (48-unit alcove beyond back wall) that complicates portal offset math
            # at non-zero rotations, and produces non-integer Z coords (nh*0.6).
            t = 16  # wall thickness
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t  # Standard room formula (no fireplace)
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('fireplace', False)  # Disable in layout mode
            set_if_not_user('random_seed', 1)  # Fixed seed to disable random variance
            # GreatHall default height (192) exceeds hall_height (128).
            # Force to 128 to prevent BSP leaks at portal boundaries.
            set_if_not_user('height', 128)

            # MULTI-PORTAL ROOM: GreatHall has entrance (SOUTH) + side (EAST) portals.
            # We can't use systemic correction (whole-room shift) because the portals
            # are on different walls. Instead, we compute offsets for each portal.
            #
            # Portal offset derivation (see CLAUDE.md §10.1):
            # After generate(), the room is rotated and normalized. The normalization
            # shifts all coordinates so the bounding box starts at (0,0).
            #
            # For the entrance (SOUTH wall, local X axis, symmetric about 0):
            #   Local X range: [-(hw+t), hw+t]. After any rotation + normalize,
            #   the mapping is either DIRECT (world_pos = local_X + hw+t) or
            #   INVERSE (world_pos = -local_X + hw+t).
            #   DIRECT: rot=0 (SOUTH), rot=270 (EAST). INVERSE: rot=90 (WEST), rot=180 (NORTH).
            #
            # For the side portal (EAST wall, local Y axis):
            #   Local Y range: [-t, nl+t]. After rotation + normalize,
            #   DIRECT: world_pos = local_Y + t (rot=0, rot=90)
            #   INVERSE: world_pos = (nl+t) - local_Y (rot=180, rot=270)
            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH (front) wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    # Entrance varies along X axis
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    # INVERSE mapping at rot=180: negate offset
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    # After rotation, entrance is on EAST/WEST wall, varies along Y
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    # INVERSE mapping at rot=90 (entrance→WEST): negate offset
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # SIDE PORTAL: on EAST (right) wall, varies along Y
            side_portal = next((p for p in footprint.portals if p.id == 'side'), None)
            if side_portal:
                rotated_cell = side_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = side_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.EAST, PortalDirection.WEST):
                    target_y = (rotated_cell.y + 0.5) * grid_size
                else:
                    target_y = (rotated_cell.x + 0.5) * grid_size

                # DIRECT mapping (rot=0, rot=90): world_pos = (nl/2 + offset) + t
                # INVERSE mapping (rot=180, rot=270): world_pos = (nl + t) - (nl/2 + offset)
                if prim.rotation in (0, 90):
                    params['_side_y_offset'] = target_y - nl / 2 - t
                else:
                    params['_side_y_offset'] = nl / 2 + t - target_y

        elif ptype == 'Prison':
            # Prison (formerly Dungeon): width is half-width, length is depth
            # Footprint: 4x4 cells = 512 x 512 units
            # Disable cell alcoves to keep geometry within footprint
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('cell_count', 0)
            set_if_not_user('cell_depth', 0)
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Prison default height (96) is shorter than hall_height (128).
            # Force to 128 to prevent BSP leaks at portal boundaries.
            set_if_not_user('height', 128)
            # Portal alignment handled by systemic post-rotation correction

        elif ptype == 'Armory':
            # Armory (formerly Armoury): width is half-width, length is depth
            # Footprint: 3x2 cells = 384 x 256 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Cistern':
            # Cistern: uses width (half-width) and length
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Stronghold':
            # Stronghold (formerly Keep): larger fortified room
            # Footprint: 5x5 cells = 640 x 640 units
            # Force wall_thickness=16 in layout mode to match hall wall thickness.
            # Stronghold defaults to t=32 (extra thick), but in layout mode the
            # vestibule floor t-extension must match the connecting hall's t=16.
            # If t != 16, the vestibule pushes the portal wall inward after
            # normalization, creating a gap with the hall wall.
            set_if_not_user('wall_thickness', 16)
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)   # = 640/2 - 16 = 304
            set_if_not_user('length', fp_depth * grid_size - 32)         # = 640 - 32 = 608
            # Stronghold default (3 levels * 128 = 384) exceeds hall_height (128).
            # Force to 1 level to prevent BSP leaks at portal boundaries.
            set_if_not_user('levels', 1)
            # Portal is at center (cell 2,0 = footprint center), systemic correction works

        elif ptype == 'Courtyard':
            # Courtyard: open-air room with sky ceiling
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('corner_towers', False)
            # Courtyard default height (192) exceeds hall_height (128).
            # Force to 128 to prevent BSP leaks at portal boundaries.
            set_if_not_user('height', 128)

        # === NEW ROOM PRIMITIVES ===

        elif ptype == 'Arena':
            # Arena: combat space with sunken pit and gallery
            # Footprint: 4x4 cells = 512 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('arena_pillars', 0)  # Disable pillars for cleaner geometry

        elif ptype == 'Laboratory':
            # Laboratory: workshop with tables and alcoves
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('alcove_count', 0)  # Disable alcoves in layout mode

        elif ptype == 'Vault':
            # Vault: secure storage with thick walls
            # Footprint: 3x2 cells = 384 x 256 units
            # Force WALL_THICKNESS=16 in layout mode to match hall wall thickness.
            # Vault defaults to WALL_THICKNESS=24 (extra thick), but in layout mode
            # the vestibule floor t-extension must match the connecting hall's t=16.
            params['WALL_THICKNESS'] = 16  # Override class constant (applied after apply_params)
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Barracks':
            # Barracks: sleeping quarters with bed alcoves
            # Footprint: 3x4 cells = 384 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('bed_alcoves', 0)  # Disable alcoves in layout mode

        elif ptype == 'Shrine':
            # Shrine: small worship space
            # Footprint: 2x2 cells = 256 x 256 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Pit':
            # Pit: hazard room with deep central pit
            # Footprint: 2x2 cells = 256 x 256 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Pit default height (96) is shorter than hall_height (128).
            # Force to 128 to prevent BSP leaks at portal boundaries.
            set_if_not_user('height', 128)

        elif ptype == 'Gatehouse':
            # Gatehouse: fortified chokepoint room
            # Footprint: 3x3 cells
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Sewer':
            # Sewer: bifurcated passage with central channel
            # Footprint: 3x3 cells
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Ossuary':
            # Ossuary: narrow passage with wall niches
            # Footprint: 2x3 cells
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Cloister':
            # Cloister: perimeter arcade with pillar ring
            # Footprint: 3x3 cells
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Disable sunken center in layout mode — extends below floor
            set_if_not_user('sunken_depth', 0)

        elif ptype == 'Colosseum':
            # Colosseum: tiered arena room
            # Footprint: 4x4 cells
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)

        elif ptype == 'Antechamber':
            # Antechamber: transitional hub with multiple portals
            # Footprint: 2x2 cells = 256 x 256 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            t = 16
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('has_pillars', False)  # Disable pillars in layout mode
            set_if_not_user('central_well', False)
            set_if_not_user('shell_sides', 4)  # Force rectangular — multi-portal room

            # MULTI-PORTAL ROOM: Antechamber has entrance (SOUTH), exit (NORTH),
            # side_east (EAST), and side_west (WEST) portals.
            # Compute offsets for each portal axis (same approach as GreatHall).
            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # EXIT PORTAL: on NORTH wall, varies along X (same axis as entrance)
            exit_portal = next((p for p in footprint.portals if p.id == 'exit'), None)
            if exit_portal:
                rotated_cell = exit_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = exit_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_exit_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_exit_x_offset'] = offset

            # SIDE PORTALS: on EAST/WEST walls, vary along Y
            # Both side_east and side_west are at the same Y cell row,
            # so they share _side_y_offset.
            side_portal = next((p for p in footprint.portals if p.id == 'side_east'), None)
            if side_portal:
                rotated_cell = side_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = side_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.EAST, PortalDirection.WEST):
                    target = (rotated_cell.y + 0.5) * grid_size
                else:
                    target = (rotated_cell.x + 0.5) * grid_size
                # DIRECT mapping (rot=0, rot=90): world = nl/2 + offset + t
                # INVERSE mapping (rot=180, rot=270): world = nl/2 + 2t - offset
                if prim.rotation in (0, 90):
                    params['_side_y_offset'] = target - nl / 2 - t
                else:
                    params['_side_y_offset'] = nl / 2 + t - target

        elif ptype == 'Hub':
            # Hub: central room with 4 portals (SOUTH, NORTH, EAST, WEST)
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            t = 16
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('height', 128)

            # MULTI-PORTAL ROOM: Hub has entrance (SOUTH), exit (NORTH),
            # side_east (EAST), and side_west (WEST) portals.
            # Compute offsets for each portal axis (same approach as Antechamber).
            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # EXIT PORTAL: on NORTH wall, varies along X (same axis as entrance)
            exit_portal = next((p for p in footprint.portals if p.id == 'exit'), None)
            if exit_portal:
                rotated_cell = exit_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = exit_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_exit_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_exit_x_offset'] = offset

            # SIDE PORTALS: on EAST/WEST walls, vary along Y
            side_portal = next((p for p in footprint.portals if p.id == 'side_east'), None)
            if side_portal:
                rotated_cell = side_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = side_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.EAST, PortalDirection.WEST):
                    target = (rotated_cell.y + 0.5) * grid_size
                else:
                    target = (rotated_cell.x + 0.5) * grid_size
                # DIRECT mapping (rot=0, rot=90): world = nl/2 + offset + t
                # INVERSE mapping (rot=180, rot=270): world = nl/2 + 2t - offset
                if prim.rotation in (0, 90):
                    params['_side_y_offset'] = target - nl / 2 - t
                else:
                    params['_side_y_offset'] = nl / 2 + t - target

        # === NEW ARCHETYPE ROOMS ===

        elif ptype == 'ThroneRoom':
            # ThroneRoom: 3x3 room with raised dais at back
            # Single portal (entrance SOUTH) — uses systemic correction
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('height', 128)

        elif ptype == 'Vestibule':
            # Vestibule: 2x1 through-passage buffer room
            # Multi-portal (entrance SOUTH + exit NORTH)
            t = 16
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('height', 128)

            # MULTI-PORTAL ROOM: Vestibule has entrance (SOUTH) + exit (NORTH)
            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # EXIT PORTAL: on NORTH wall, varies along X
            exit_portal = next((p for p in footprint.portals if p.id == 'exit'), None)
            if exit_portal:
                rotated_cell = exit_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = exit_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_exit_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_exit_x_offset'] = offset

        elif ptype == 'Processional':
            # Processional: 2x4 through-passage with colonnade
            # Multi-portal (entrance SOUTH + exit NORTH) — same offset pattern as Vestibule
            t = 16
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('height', 128)

            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            exit_portal = next((p for p in footprint.portals if p.id == 'exit'), None)
            if exit_portal:
                rotated_cell = exit_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = exit_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_exit_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_exit_x_offset'] = offset

        elif ptype == 'DoglegRoom':
            # DoglegRoom: 3x3 L-shaped room
            # Multi-portal (entrance SOUTH + exit EAST)
            t = 16
            hw = (fp_width * grid_size) / 2 - t
            nl = fp_depth * grid_size - 2 * t
            set_if_not_user('width', hw)
            set_if_not_user('length', nl)
            set_if_not_user('height', 128)

            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # EXIT PORTAL: on EAST wall, varies along Y
            exit_portal = next((p for p in footprint.portals if p.id == 'exit'), None)
            if exit_portal:
                rotated_cell = exit_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = exit_portal.rotated_direction(prim.rotation)

                # DoglegRoom exit portal offset computation.
                # The exit is on the EAST wall in local coords. _exit_y_offset shifts
                # the portal along the wall's Y axis. After rotation + normalization,
                # the local Y maps to a world coordinate through the rotation transform:
                #   rot=0/90 (forward):  world_coord = local_Y + t
                #   rot=180/270 (reverse): world_coord = (nl + 2*t) - local_Y
                arm_split = round(nl * 2.0 / 3.0 / 16) * 16
                base_portal_y = arm_split + (nl - arm_split) / 2  # default local Y

                if rotated_dir in (PortalDirection.EAST, PortalDirection.WEST):
                    target = (rotated_cell.y + 0.5) * grid_size
                else:
                    target = (rotated_cell.x + 0.5) * grid_size

                if prim.rotation in (0, 90):
                    # Forward: world = local_Y + t → local_Y = target - t
                    params['_exit_y_offset'] = (target - t) - base_portal_y
                else:
                    # Reverse (180, 270): world = (nl + 2t) - local_Y → local_Y = (nl + 2t) - target
                    params['_exit_y_offset'] = (nl + 2 * t) - target - base_portal_y

        # === MULTI-FLOOR ROOM PRIMITIVES ===
        # These rooms have internal upper portals and stairs for vertical gameplay

        elif ptype == 'Amphitheater':
            # Amphitheater: tiered arena with concentric rings
            # Footprint: 4x4 cells = 512 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('tier_count', 3)
            set_if_not_user('shell_sides', 4)  # Square for layout mode
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        elif ptype == 'CatwalkChamber':
            # CatwalkChamber: pit with spanning catwalks
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        elif ptype == 'BalconyRoom':
            # BalconyRoom: room with elevated balcony
            # Footprint: 3x4 cells = 384 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        elif ptype == 'SunkenChamber':
            # SunkenChamber: room with lowered central basin
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        elif ptype == 'LibraryArchive':
            # LibraryArchive: tall room with shelf alcoves
            # Footprint: 3x4 cells = 384 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            # Disable alcove depth in layout mode: alcoves extend geometry
            # by alcove_depth (24) beyond walls on each side, overflowing the footprint.
            # Setting depth=0 keeps alcove_rows/cols for wall segment logic but
            # prevents geometry from extending outside the room envelope.
            set_if_not_user('alcove_depth', 0)
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('room_height', 128)

        elif ptype == 'Grotto':
            # Grotto: irregular cave with varied floor
            # Footprint: 3x3 cells = 384 x 384 units
            # IMPORTANT: Grotto.WALL_THICKNESS = 24 (class constant, NOT settable via params).
            # Dimension calculations MUST use t=24 to match what generate() actually uses,
            # otherwise floor/ceiling extend 8 units beyond footprint per side.
            t = 24  # Must match Grotto.WALL_THICKNESS
            set_if_not_user('width', (fp_width * grid_size) / 2 - t)
            set_if_not_user('length', fp_depth * grid_size - 2*t)
            set_if_not_user('shell_sides', 4)  # Square for layout mode
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)
            # Disable features that extend geometry beyond the expected Z envelope:
            # - floor_variation pushes floor below z=0 (walls extend to oz - floor_var)
            # - stalactites hang from ceiling into room space
            # - irregularity offsets walls inward, creating gaps between segments
            set_if_not_user('floor_variation', 0)
            set_if_not_user('stalactite_count', 0)
            set_if_not_user('irregularity', 0)

        elif ptype == 'RadialShrine':
            # RadialShrine: shrine with radiating alcoves
            # Footprint: 4x4 cells = 512 x 512 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('alcove_count', 0)  # Disable radiating alcoves in layout mode
            set_if_not_user('shell_sides', 4)  # Square for layout mode
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        elif ptype == 'Forge':
            # Forge: multi-tier industrial room
            # Footprint: 3x3 cells = 384 x 384 units
            # Room t=16: width = footprint/2 - t, length = footprint - 2*t
            set_if_not_user('width', (fp_width * grid_size) / 2 - 16)
            set_if_not_user('length', fp_depth * grid_size - 32)
            set_if_not_user('tier_count', 2)
            # Force height to 128 at entrance level to match hall_height
            set_if_not_user('height', 128)

        # =====================================================================
        # MULTI-FLOOR ROOM PORTAL OFFSET COMPUTATION
        # =====================================================================
        # Multi-floor rooms have entrance (SOUTH) + upper (NORTH) portals on
        # different walls. Like GreatHall/Antechamber, they need per-portal
        # offsets instead of systemic whole-room correction.
        if ptype in MULTI_FLOOR_ROOMS and footprint:
            fp_w, fp_d = footprint.rotated_size(prim.rotation)

            # ENTRANCE PORTAL: on SOUTH wall, varies along X
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                if rotated_dir in (PortalDirection.SOUTH, PortalDirection.NORTH):
                    target_x = (rotated_cell.x + 0.5) * grid_size
                    norm_center_x = (fp_w * grid_size) / 2
                    offset = target_x - norm_center_x
                    if prim.rotation == 180:
                        offset = -offset
                    params['_entrance_x_offset'] = offset
                else:
                    target_y = (rotated_cell.y + 0.5) * grid_size
                    norm_center_y = (fp_d * grid_size) / 2
                    offset = target_y - norm_center_y
                    if prim.rotation == 90:
                        offset = -offset
                    params['_entrance_x_offset'] = offset

            # UPPER PORTAL: on NORTH wall, varies along X
            # Upper portal uses the same _entrance_x_offset since both portals
            # are at local X=0 (room center). The layout generator's tag system
            # places tags from footprint data, so we don't need a separate offset.

        # =====================================================================
        # POLYGONAL ROOM PORTAL TARGET COMPUTATION
        # =====================================================================
        # When shell_sides != 4, rooms use polygonal wall generation which places
        # portals at segment midpoints. This causes misalignment with halls that
        # expect portals at grid cell positions.
        #
        # Solution: Compute the expected portal target position in the room's
        # local coordinate system and pass it to the generator.
        #
        # The room generates geometry centered at (cx, cy) where:
        # - cx = origin_x (usually 0 for normalized rooms)
        # - cy = origin_y + length/2 (center of the room)
        #
        # The portal target should be at the cell boundary position, converted
        # to the room's local coordinate system.
        #
        shell_sides = user_params.get('shell_sides', params.get('shell_sides', 4))
        if shell_sides != 4 and footprint and footprint.portals:
            # Get the entrance portal (all rooms use 'entrance' for SOUTH portal)
            entrance_portal = next((p for p in footprint.portals if p.id == 'entrance'), None)
            if entrance_portal:
                # Get portal cell position considering rotation
                rotated_cell = entrance_portal.world_cell(
                    CellCoord(0, 0), prim.rotation,
                    footprint.width_cells, footprint.depth_cells
                )
                rotated_dir = entrance_portal.rotated_direction(prim.rotation)

                # Calculate portal position in world space (relative to footprint origin)
                # The hall expects the portal at the cell boundary
                if rotated_dir == PortalDirection.SOUTH:
                    # SOUTH portal: X at cell center, Y at cell bottom edge (0)
                    portal_world_x = (rotated_cell.x + 0.5) * grid_size
                    portal_world_y = 0  # South edge of footprint
                elif rotated_dir == PortalDirection.NORTH:
                    # NORTH portal: X at cell center, Y at cell top edge
                    fp_w, fp_d = footprint.rotated_size(prim.rotation)
                    portal_world_x = (rotated_cell.x + 0.5) * grid_size
                    portal_world_y = fp_d * grid_size
                elif rotated_dir == PortalDirection.WEST:
                    # WEST portal: X at left edge (0), Y at cell center
                    portal_world_x = 0
                    portal_world_y = (rotated_cell.y + 0.5) * grid_size
                elif rotated_dir == PortalDirection.EAST:
                    # EAST portal: X at right edge, Y at cell center
                    fp_w, fp_d = footprint.rotated_size(prim.rotation)
                    portal_world_x = fp_w * grid_size
                    portal_world_y = (rotated_cell.y + 0.5) * grid_size
                else:
                    portal_world_x = None
                    portal_world_y = None

                if portal_world_x is not None:
                    # Convert to room's local coordinate system
                    # Room is centered at (fp_width/2 * grid, length/2)
                    # Polygonal rooms compute center as (ox, oy + length/2)
                    # where ox=0 and oy=0 for normalized geometry
                    #
                    # The target X should be relative to the room center (cx)
                    # The target Y should be relative to the room center (cy)
                    fp_w, fp_d = footprint.rotated_size(prim.rotation)
                    room_center_x = (fp_w * grid_size) / 2
                    length = params.get('length', fp_d * grid_size - 32)
                    room_center_y = length / 2

                    # Portal target in room's local coords (relative to room center)
                    # Note: room origin is at (0, 0), center is at (0, length/2)
                    # So target relative to center = world pos - center pos
                    # Use underscore prefix so these get copied to instance attributes
                    params['_portal_target_x'] = portal_world_x - room_center_x
                    params['_portal_target_y'] = portal_world_y - room_center_y

    def _configure_portal_params(self, prim: PlacedPrimitive,
                                  params: Dict, connected_portals: Dict[str, Tuple[bool, bool]]):
        """Configure portal_* parameters based on connections and overrides.

        Portal enabling logic:
        1. Check for portal override on the primitive
        2. If override.enabled is False, portal is always disabled
        3. Otherwise, enable based on connection status
        """

        # Map portal IDs to parameter names for different primitive types
        # NOTE: StraightHall has confusing naming conventions:
        # - Data model 'front' portal faces NORTH (top of footprint)
        # - Generator 'portal_front' controls the SOUTH wall (Y=0)
        # So we need to swap them:
        # - 'front' (NORTH-facing) → 'portal_back' (opens NORTH wall)
        # - 'back' (SOUTH-facing) → 'portal_front' (opens SOUTH wall)
        #
        # SquareCorner uses consistent naming:
        # - Data model 'a' faces SOUTH, generator 'portal_a' controls SOUTH wall
        # - Data model 'b' faces EAST, generator 'portal_b' controls EAST wall
        # Portal parameter mapping: data model portal IDs -> generator boolean params
        #
        # HALLS: Use _wall_with_portal() pattern, portal params default True
        # - StraightHall has swapped naming (front->portal_back, back->portal_front)
        # - TJunction uses cardinal directions (west, east, north)
        # - SquareCorner uses consistent naming (a->portal_a, b->portal_b)
        # - Crossroads uses cardinal directions
        #
        # ROOMS: Use has_entrance/has_exit/has_side booleans, default True
        # - All rooms have 'entrance' portal facing SOUTH
        # - Crypt also has 'exit' portal facing NORTH
        # - GreatHall also has 'side' portal facing EAST
        portal_param_map = {
            # === HALLS ===
            'StraightHall': {'front': 'portal_back', 'back': 'portal_front'},
            'TJunction': {
                'west': 'portal_west', 'east': 'portal_east', 'north': 'portal_north'
            },
            'Crossroads': {
                'north': 'portal_north', 'south': 'portal_south',
                'east': 'portal_east', 'west': 'portal_west'
            },
            'SquareCorner': {'a': 'portal_a', 'b': 'portal_b'},
        'WideCorner': {'a': 'portal_a', 'b': 'portal_b'},
            'VerticalStairHall': {'bottom': 'portal_bottom', 'top': 'portal_top'},
            'NarrowPassage': {'front': 'portal_back', 'back': 'portal_front'},
            'WideCorridor': {'south': 'portal_south', 'north': 'portal_north'},
            # === ROOMS (all use has_entrance, some have additional portals) ===
            'Sanctuary': {'entrance': 'has_entrance'},
            'Tomb': {'entrance': 'has_entrance', 'exit': 'has_exit'},
            'Tower': {'entrance': 'has_entrance'},
            'Chamber': {'entrance': 'has_entrance'},
            'Storage': {'entrance': 'has_entrance'},
            'GreatHall': {'entrance': 'has_entrance', 'side': 'has_side'},
            'Prison': {'entrance': 'has_entrance'},
            'Armory': {'entrance': 'has_entrance'},
            'Cistern': {'entrance': 'has_entrance'},
            'Stronghold': {'entrance': 'has_entrance'},
            'Courtyard': {'entrance': 'has_entrance'},
            # New rooms
            'Arena': {'entrance': 'has_entrance'},
            'Laboratory': {'entrance': 'has_entrance'},
            'Vault': {'entrance': 'has_entrance'},
            'Barracks': {'entrance': 'has_entrance'},
            'Shrine': {'entrance': 'has_entrance'},
            'Pit': {'entrance': 'has_entrance'},
            'Gatehouse': {'entrance': 'has_entrance'},
            'Sewer': {'entrance': 'has_entrance'},
            'Ossuary': {'entrance': 'has_entrance'},
            'Cloister': {'entrance': 'has_entrance'},
            'Colosseum': {'entrance': 'has_entrance'},
            'ThroneRoom': {'entrance': 'has_entrance'},
            'Vestibule': {'entrance': 'has_entrance', 'exit': 'has_exit'},
            'Processional': {'entrance': 'has_entrance', 'exit': 'has_exit'},
            'DoglegRoom': {'entrance': 'has_entrance', 'exit': 'has_exit'},
            'Antechamber': {
                'entrance': 'has_entrance', 'exit': 'has_exit',
                'side_east': 'has_side_east', 'side_west': 'has_side_west'
            },
            'Hub': {
                'entrance': 'has_entrance', 'exit': 'has_exit',
                'side_east': 'has_side_east', 'side_west': 'has_side_west'
            },
            # Multi-Floor rooms (entrance at z=0, upper portal at z=160 for floor connections)
            'Amphitheater': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'CatwalkChamber': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'BalconyRoom': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'SunkenChamber': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'LibraryArchive': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'Grotto': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'RadialShrine': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
            'Forge': {'entrance': 'has_entrance', 'upper': 'has_upper_portal'},
        }

        param_map = portal_param_map.get(prim.primitive_type, {})

        for portal_id, param_name in param_map.items():
            # Check for portal override first
            if not prim.is_portal_enabled(portal_id):
                # Portal is explicitly disabled via override
                params[param_name] = False
            else:
                is_connected = connected_portals.get(portal_id, False)
                params[param_name] = is_connected

    def _create_box_brush(self, x1: float, y1: float, z1: float,
                          x2: float, y2: float, z2: float) -> Brush:
        """Create a simple axis-aligned box brush.

        Uses the same inward-facing normal convention as base.py's _box() method
        for consistency with module-generated brushes.

        Args:
            x1, y1, z1: Minimum corner coordinates
            x2, y2, z2: Maximum corner coordinates

        Returns:
            A Brush object representing the box
        """
        # Ensure coordinates are integers for idTech compatibility
        x1, y1, z1 = int(x1), int(y1), int(z1)
        x2, y2, z2 = int(x2), int(y2), int(z2)

        # Normalize: ensure min <= max
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if z1 > z2:
            z1, z2 = z2, z1

        # idTech 1 convention: normals point inward (toward brush center)
        # Matches base.py _box() winding order exactly
        planes = [
            Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2)),  # Left (X=x1)
            Plane((x2, y1, z1), (x2, y1, z2), (x2, y2, z1)),  # Right (X=x2)
            Plane((x1, y1, z1), (x1, y1, z2), (x2, y1, z1)),  # Front (Y=y1)
            Plane((x1, y2, z1), (x2, y2, z1), (x1, y2, z2)),  # Back (Y=y2)
            Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1)),  # Bottom (Z=z1)
            Plane((x1, y1, z2), (x1, y2, z2), (x2, y1, z2)),  # Top (Z=z2)
        ]
        return Brush(planes)

    def _compute_portal_positions(self, prim: PlacedPrimitive) -> None:
        """Compute and store world positions for all portals of a primitive.

        Portal world positions are computed based on:
        - Primitive origin cell
        - Primitive rotation
        - Portal cell offset and direction
        - Grid size and standard portal dimensions

        These positions are used to validate that connected portals align correctly.
        """
        grid_size = self._layout.grid_size
        footprint = prim.footprint or PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if not footprint:
            return

        for portal in footprint.portals:
            if not portal.enabled:
                continue

            # Get world cell for this portal
            # Use footprint dimensions directly to ensure correct rotation calculation
            # (prim._footprint may be None when loaded from JSON, but we have
            # the footprint from PRIMITIVE_FOOTPRINTS lookup above)
            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )

            # Get rotated direction
            rotated_dir = portal.rotated_direction(prim.rotation)

            # Convert data model PortalDirection to portal_system PortalDirection
            dir_map = {
                PortalDirection.NORTH: PortalDir.NORTH,
                PortalDirection.SOUTH: PortalDir.SOUTH,
                PortalDirection.EAST: PortalDir.EAST,
                PortalDirection.WEST: PortalDir.WEST,
            }
            portal_dir = dir_map[rotated_dir]

            # Compute world position
            # Portal center is at the center of the cell, at the wall boundary
            cell_center_x = world_cell.x * grid_size + grid_size / 2
            cell_center_y = world_cell.y * grid_size + grid_size / 2

            # Adjust center based on portal direction (portal is at cell boundary)
            if rotated_dir == PortalDirection.NORTH:
                portal_center_x = cell_center_x
                portal_center_y = (world_cell.y + 1) * grid_size  # Top of cell
            elif rotated_dir == PortalDirection.SOUTH:
                portal_center_x = cell_center_x
                portal_center_y = world_cell.y * grid_size  # Bottom of cell
            elif rotated_dir == PortalDirection.EAST:
                portal_center_x = (world_cell.x + 1) * grid_size  # Right of cell
                portal_center_y = cell_center_y
            else:  # WEST
                portal_center_x = world_cell.x * grid_size  # Left of cell
                portal_center_y = cell_center_y

            # Z is at floor level plus portal z_level plus half portal height
            # (portal.z_level accounts for multi-floor portals like VerticalStairHall)
            portal_center_z = prim.z_offset + portal.z_level + PORTAL_HEIGHT / 2

            pos = PortalWorldPosition(
                primitive_id=prim.id,
                portal_id=portal.id,
                center_x=portal_center_x,
                center_y=portal_center_y,
                center_z=portal_center_z,
                width=PORTAL_WIDTH,
                height=PORTAL_HEIGHT,
                direction=portal_dir,
            )
            self._portal_positions[(prim.id, portal.id)] = pos

    def _validate_portal_alignment(self) -> List[PortalMismatch]:
        """Validate that all connected portals align correctly.

        Returns a list of PortalMismatch objects describing any misalignments.
        """
        connections_to_validate: List[Tuple[PortalWorldPosition, PortalWorldPosition]] = []

        for conn in self._layout.connections:
            key_a = (conn.primitive_a_id, conn.portal_a_id)
            key_b = (conn.primitive_b_id, conn.portal_b_id)

            pos_a = self._portal_positions.get(key_a)
            pos_b = self._portal_positions.get(key_b)

            if pos_a is None:
                self._warnings.append(
                    f"Portal position not found: {conn.primitive_a_id}.{conn.portal_a_id}"
                )
                continue
            if pos_b is None:
                self._warnings.append(
                    f"Portal position not found: {conn.primitive_b_id}.{conn.portal_b_id}"
                )
                continue

            connections_to_validate.append((pos_a, pos_b))

        return validate_portal_alignment(connections_to_validate)


def generate_from_layout(
    layout: DungeonLayout,
    random_seed: Optional[int] = None,
    game_profile: Optional[GameProfile] = None,
) -> GenerationResult:
    """Convenience function to generate brushes from a layout.

    Args:
        layout: The dungeon layout to generate brushes from
        random_seed: Optional seed for reproducible generation
        game_profile: Game profile for engine-specific texture mappings.
                     If None, uses the default profile.

    Returns:
        GenerationResult with brushes, entities, and metadata
    """
    generator = LayoutGenerator(
        layout,
        random_seed=random_seed,
        game_profile=game_profile,
    )
    return generator.generate()
