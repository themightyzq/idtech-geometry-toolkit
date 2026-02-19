"""
Data model for the node-based layout editor.

Defines the core data structures for dungeon layouts:
- CellCoord: Grid position (x, y integers)
- PortalDirection: Cardinal direction enum (NORTH, SOUTH, EAST, WEST)
- Portal: Connection point on a primitive with direction and cell offset
- PrimitiveFootprint: Size in cells and portal definitions for a primitive type
- PlacedPrimitive: A primitive instance on the grid with position and rotation
- Connection: Link between two portals on adjacent primitives
- DungeonLayout: Complete layout with primitives and connections

Rotation System:
- Primitives support 0°, 90°, 180°, 270° rotation (clockwise)
- Portal._rotate_offset() transforms cell offsets accounting for footprint dimensions
- Portal.rotated_direction() returns the direction after applying rotation
- Footprint.rotated_size() returns (width, depth) after rotation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

from quake_levelgenerator.src.generators.primitives.portal_system import (
    PORTAL_WIDTH, PORTAL_HEIGHT
)


# =============================================================================
# LEGACY TYPE MIGRATION
# =============================================================================
# Maps deprecated primitive types to their modern equivalents with parameters.
# LJunction is migrated to SquareCorner (same geometry).

PRIMITIVE_TYPE_ALIASES: Dict[str, Tuple[str, Dict[str, Any]]] = {
    'LJunction': ('SquareCorner', {}),
    'Corner': ('SquareCorner', {}),  # Old unified Corner -> SquareCorner
}


def _migrate_primitive_type(
    primitive_type: str,
    parameters: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Migrate deprecated primitive types to their modern equivalents.

    Args:
        primitive_type: Original primitive type name
        parameters: Original parameters dict

    Returns:
        Tuple of (new_type, merged_parameters)
    """
    if primitive_type in PRIMITIVE_TYPE_ALIASES:
        new_type, default_params = PRIMITIVE_TYPE_ALIASES[primitive_type]
        # Merge default migration params with user params (user params take precedence)
        merged_params = dict(default_params)
        merged_params.update(parameters)

        # Handle Corner/LJunction arm_a_length/arm_b_length -> arm_length mapping
        # SquareCorner uses single arm_length, pick the minimum of both arms
        if 'arm_a_length' in parameters and 'arm_b_length' in parameters:
            merged_params['arm_length'] = min(parameters['arm_a_length'],
                                              parameters['arm_b_length'])

        return new_type, merged_params

    return primitive_type, parameters


class PortalDirection(Enum):
    """Cardinal direction for portal facing."""
    NORTH = "north"  # +Y direction
    SOUTH = "south"  # -Y direction
    EAST = "east"    # +X direction
    WEST = "west"    # -X direction

    def opposite(self) -> 'PortalDirection':
        """Return the opposite direction."""
        opposites = {
            PortalDirection.NORTH: PortalDirection.SOUTH,
            PortalDirection.SOUTH: PortalDirection.NORTH,
            PortalDirection.EAST: PortalDirection.WEST,
            PortalDirection.WEST: PortalDirection.EAST,
        }
        return opposites[self]


@dataclass
class CellCoord:
    """Grid cell coordinate."""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, CellCoord):
            return False
        return self.x == other.x and self.y == other.y

    def __add__(self, other: 'CellCoord') -> 'CellCoord':
        return CellCoord(self.x + other.x, self.y + other.y)

    def neighbor(self, direction: PortalDirection) -> 'CellCoord':
        """Get the neighboring cell in the given direction."""
        offsets = {
            PortalDirection.NORTH: CellCoord(0, 1),
            PortalDirection.SOUTH: CellCoord(0, -1),
            PortalDirection.EAST: CellCoord(1, 0),
            PortalDirection.WEST: CellCoord(-1, 0),
        }
        return self + offsets[direction]

    def to_world(self, grid_size: int) -> Tuple[float, float]:
        """Convert to world coordinates (center of cell)."""
        return (self.x * grid_size + grid_size / 2,
                self.y * grid_size + grid_size / 2)

    @staticmethod
    def from_world(x: float, y: float, grid_size: int) -> 'CellCoord':
        """Convert from world coordinates to cell."""
        return CellCoord(int(x // grid_size), int(y // grid_size))


@dataclass
class Portal:
    """Connection point on a primitive."""
    id: str                           # Unique identifier for this portal
    direction: PortalDirection        # Which way the portal faces
    cell_offset: CellCoord            # Offset from primitive origin (in cells)
    width: int = PORTAL_WIDTH         # Portal opening width (unified constant)
    height: int = PORTAL_HEIGHT       # Portal opening height (unified constant)
    enabled: bool = True              # Whether this portal is open
    z_level: int = 0                  # Z-offset relative to primitive's z_offset (for multi-level)

    def world_cell(self, primitive_origin: CellCoord, rotation: int,
                   footprint_width: int = 1, footprint_depth: int = 1) -> CellCoord:
        """Get the world cell position of this portal given primitive origin and rotation.

        Args:
            primitive_origin: The origin cell of the primitive
            rotation: Rotation in degrees (0, 90, 180, 270)
            footprint_width: Original footprint width (before rotation)
            footprint_depth: Original footprint depth (before rotation)
        """
        # Rotate the offset based on primitive rotation
        rotated = self._rotate_offset(self.cell_offset, rotation, footprint_width, footprint_depth)
        return primitive_origin + rotated

    def rotated_direction(self, rotation: int) -> PortalDirection:
        """Get the portal direction after applying rotation."""
        directions = [PortalDirection.NORTH, PortalDirection.EAST,
                      PortalDirection.SOUTH, PortalDirection.WEST]
        idx = directions.index(self.direction)
        rotations = rotation // 90
        new_idx = (idx + rotations) % 4
        return directions[new_idx]

    @staticmethod
    def _rotate_offset(offset: CellCoord, rotation: int,
                       footprint_width: int = 1, footprint_depth: int = 1) -> CellCoord:
        """Rotate a cell offset by the given degrees (0, 90, 180, 270).

        Args:
            offset: The cell offset to rotate
            rotation: Rotation in degrees (0, 90, 180, 270)
            footprint_width: Original footprint width before rotation
            footprint_depth: Original footprint depth before rotation

        Returns:
            Rotated cell offset that stays within the rotated bounding box
        """
        x, y = offset.x, offset.y
        w, d = footprint_width, footprint_depth

        if rotation == 90:
            # 90° clockwise: (x, y) -> (y, w-1-x)
            # The top of the footprint becomes the right side
            return CellCoord(y, w - 1 - x)
        elif rotation == 180:
            # 180°: (x, y) -> (w-1-x, d-1-y)
            return CellCoord(w - 1 - x, d - 1 - y)
        elif rotation == 270:
            # 270° clockwise (90° counter-clockwise): (x, y) -> (d-1-y, x)
            # The top of the footprint becomes the left side
            return CellCoord(d - 1 - y, x)
        return offset  # 0 degrees


@dataclass
class PrimitiveFootprint:
    """Footprint definition for a primitive type."""
    width_cells: int                  # Width in grid cells
    depth_cells: int                  # Depth in grid cells
    portals: List[Portal] = field(default_factory=list)  # Portal definitions

    def rotated_size(self, rotation: int) -> Tuple[int, int]:
        """Get (width, depth) after rotation."""
        if rotation in (90, 270):
            return (self.depth_cells, self.width_cells)
        return (self.width_cells, self.depth_cells)

    def occupied_cells(self, origin: CellCoord, rotation: int) -> List[CellCoord]:
        """Get all cells occupied by this footprint at the given origin and rotation."""
        w, d = self.rotated_size(rotation)
        cells = []
        for dx in range(w):
            for dy in range(d):
                cells.append(CellCoord(origin.x + dx, origin.y + dy))
        return cells


@dataclass
class PortalOverride:
    """Override settings for a specific portal on a placed primitive.

    Allows per-instance customization of portal settings without modifying
    the base footprint definition.
    """
    enabled: bool = True                      # Enable/disable this portal
    z_level_override: Optional[int] = None    # Custom z_level (None = use default from footprint)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {'enabled': self.enabled}
        if self.z_level_override is not None:
            result['z_level_override'] = self.z_level_override
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PortalOverride':
        """Create from dictionary."""
        return PortalOverride(
            enabled=data.get('enabled', True),
            z_level_override=data.get('z_level_override'),
        )


@dataclass
class PlacedPrimitive:
    """A primitive instance placed on the grid."""
    id: str                                  # UUID
    primitive_type: str                      # "StraightHall", "Chapel", etc.
    origin_cell: CellCoord                   # Grid position (bottom-left corner)
    rotation: int = 0                        # 0, 90, 180, 270 degrees
    parameters: Dict[str, Any] = field(default_factory=dict)  # Primitive params
    z_offset: float = 0.0                    # Height for multi-level
    portal_overrides: Dict[str, PortalOverride] = field(default_factory=dict)  # Per-portal overrides

    # Cached footprint (set by editor)
    _footprint: Optional[PrimitiveFootprint] = field(default=None, repr=False)

    @staticmethod
    def create(primitive_type: str, origin: CellCoord, rotation: int = 0,
               parameters: Optional[Dict[str, Any]] = None,
               z_offset: float = 0.0,
               portal_overrides: Optional[Dict[str, PortalOverride]] = None) -> 'PlacedPrimitive':
        """Factory method to create a new placed primitive."""
        return PlacedPrimitive(
            id=str(uuid.uuid4()),
            primitive_type=primitive_type,
            origin_cell=origin,
            rotation=rotation,
            parameters=parameters or {},
            z_offset=z_offset,
            portal_overrides=portal_overrides or {},
        )

    def set_footprint(self, footprint: PrimitiveFootprint):
        """Set the footprint for this primitive."""
        self._footprint = footprint

    @property
    def footprint(self) -> Optional[PrimitiveFootprint]:
        return self._footprint

    def occupied_cells(self) -> List[CellCoord]:
        """Get all cells occupied by this primitive."""
        if self._footprint is None:
            # Default 1x1 footprint
            return [self.origin_cell]
        return self._footprint.occupied_cells(self.origin_cell, self.rotation)

    def get_portals(self) -> List[Portal]:
        """Get portals with world positions computed."""
        if self._footprint is None:
            return []
        return self._footprint.portals

    def get_portal_world_cell(self, portal: Portal) -> CellCoord:
        """Get the world cell position of a portal."""
        if self._footprint:
            return portal.world_cell(
                self.origin_cell, self.rotation,
                self._footprint.width_cells, self._footprint.depth_cells
            )
        return portal.world_cell(self.origin_cell, self.rotation)

    def get_portal_z_level(self, portal_id: str) -> int:
        """Get z_level for a portal, checking for override first.

        Args:
            portal_id: The portal ID to look up

        Returns:
            The z_level for the portal (override if set, otherwise footprint default)
        """
        # Check for override first
        if portal_id in self.portal_overrides:
            override = self.portal_overrides[portal_id]
            if override.z_level_override is not None:
                return override.z_level_override

        # Special handling for VerticalStairHall - top portal z_level depends on height_change
        # The top portal z_level is the height_change parameter, not a static footprint value
        if self.primitive_type == 'VerticalStairHall' and portal_id == 'top':
            height_change = self.parameters.get('height_change', 160.0)
            return int(height_change)

        # Try cached footprint
        if self._footprint:
            for portal in self._footprint.portals:
                if portal.id == portal_id:
                    return portal.z_level

        # Fallback: lookup from PRIMITIVE_FOOTPRINTS when _footprint is None
        # This handles cases like JSON reload where footprints aren't cached
        from .palette_widget import PRIMITIVE_FOOTPRINTS
        footprint = PRIMITIVE_FOOTPRINTS.get(self.primitive_type)
        if footprint:
            for portal in footprint.portals:
                if portal.id == portal_id:
                    return portal.z_level

        return 0

    def is_portal_enabled(self, portal_id: str) -> bool:
        """Check if a portal is enabled, checking for override first.

        Args:
            portal_id: The portal ID to look up

        Returns:
            True if the portal is enabled
        """
        # Check for override first
        if portal_id in self.portal_overrides:
            return self.portal_overrides[portal_id].enabled

        # Fall back to footprint default
        if self._footprint:
            for portal in self._footprint.portals:
                if portal.id == portal_id:
                    return portal.enabled

        return True

    def get_absolute_portal_z(self, portal_id: str) -> float:
        """Get the absolute world Z position for a portal.

        Combines the primitive's z_offset with the portal's z_level to get
        the actual world Z coordinate where the portal floor sits.

        Args:
            portal_id: The portal ID to look up

        Returns:
            Absolute Z position in world units
        """
        return self.z_offset + self.get_portal_z_level(portal_id)


@dataclass
class Connection:
    """Connection between two portals.

    A connection links two portals on different primitives.
    When is_secret is True, the connection generates a CLIP wall
    instead of an open portal (walk-through secret passage).
    """
    primitive_a_id: str
    portal_a_id: str
    primitive_b_id: str
    portal_b_id: str
    is_secret: bool = False  # When True, generates CLIP wall instead of open portal


@dataclass
class DungeonLayout:
    """Complete dungeon layout with primitives and connections."""
    grid_size: int = 128                              # Cell size in idTech units
    primitives: Dict[str, PlacedPrimitive] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)

    # Editor state (not serialized)
    _selected_id: Optional[str] = field(default=None, repr=False)

    def add_primitive(self, prim: PlacedPrimitive) -> bool:
        """Add a primitive to the layout. Returns False if placement invalid."""
        # Check for collisions - only on same z-level
        # Multi-floor layouts have primitives on different z-levels that can
        # share the same XY cell coordinates without collision
        new_cells = set(prim.occupied_cells())
        new_z = prim.z_offset

        for existing in self.primitives.values():
            # Skip collision check for primitives on different floors
            # Floor separation is 160 units - primitives 100+ units apart are on different floors
            if abs(existing.z_offset - new_z) > 100:
                continue

            existing_cells = set(existing.occupied_cells())
            if new_cells & existing_cells:
                return False  # Collision

        self.primitives[prim.id] = prim
        return True

    def remove_primitive(self, prim_id: str) -> bool:
        """Remove a primitive and its connections."""
        if prim_id not in self.primitives:
            return False

        # Remove associated connections
        self.connections = [
            c for c in self.connections
            if c.primitive_a_id != prim_id and c.primitive_b_id != prim_id
        ]

        del self.primitives[prim_id]
        return True

    def get_primitive_at_cell(self, cell: CellCoord) -> Optional[PlacedPrimitive]:
        """Get the primitive occupying a given cell, if any."""
        for prim in self.primitives.values():
            if cell in prim.occupied_cells():
                return prim
        return None

    def get_all_occupied_cells(self) -> Dict[CellCoord, str]:
        """Get a map of all occupied cells to their primitive IDs."""
        result = {}
        for prim in self.primitives.values():
            for cell in prim.occupied_cells():
                result[cell] = prim.id
        return result

    def can_place_at(self, footprint: PrimitiveFootprint, origin: CellCoord,
                     rotation: int = 0) -> bool:
        """Check if a footprint can be placed at the given position."""
        cells = footprint.occupied_cells(origin, rotation)
        occupied = self.get_all_occupied_cells()
        for cell in cells:
            if cell in occupied:
                return False
        return True

    def find_matching_portal(self, prim_id: str, portal: Portal) -> Optional[Tuple[str, Portal]]:
        """Find a portal on an adjacent primitive that could connect to this one.

        Portals connect when:
        1. They are in adjacent cells facing each other
        2. Both portals are enabled (checking overrides)
        3. Their absolute Z positions match (within 2-unit tolerance)

        Absolute Z = primitive.z_offset + get_portal_z_level(portal.id)
        """
        prim = self.primitives.get(prim_id)
        if not prim:
            return None

        # Check if this portal is enabled (respecting overrides)
        if not prim.is_portal_enabled(portal.id):
            return None

        portal_cell = prim.get_portal_world_cell(portal)
        portal_dir = portal.rotated_direction(prim.rotation)

        # Calculate this portal's absolute Z position (respecting overrides)
        portal_z = prim.z_offset + prim.get_portal_z_level(portal.id)

        # Look for primitive in adjacent cell
        adjacent_cell = portal_cell.neighbor(portal_dir)
        adjacent_prim = self.get_primitive_at_cell(adjacent_cell)

        if adjacent_prim is None or adjacent_prim.id == prim_id:
            return None

        # Check if adjacent primitive has a facing portal
        for other_portal in adjacent_prim.get_portals():
            # Check if other portal is enabled (respecting overrides)
            if not adjacent_prim.is_portal_enabled(other_portal.id):
                continue

            other_cell = adjacent_prim.get_portal_world_cell(other_portal)
            other_dir = other_portal.rotated_direction(adjacent_prim.rotation)

            # Portals connect if:
            # 1. The other portal is at the adjacent cell (where we're looking)
            # 2. The other portal faces back toward us (opposite direction)
            # 3. Their Z positions match (within 2-unit tolerance)
            if other_cell == adjacent_cell and other_dir == portal_dir.opposite():
                other_z = adjacent_prim.z_offset + adjacent_prim.get_portal_z_level(other_portal.id)
                if abs(portal_z - other_z) <= 2:
                    return (adjacent_prim.id, other_portal)

        return None

    def auto_connect(self, prim_id: str):
        """Automatically connect matching portals for a primitive."""
        prim = self.primitives.get(prim_id)
        if not prim:
            return

        for portal in prim.get_portals():
            match = self.find_matching_portal(prim_id, portal)
            if match:
                other_prim_id, other_portal = match
                # Check if connection already exists
                exists = any(
                    (c.primitive_a_id == prim_id and c.portal_a_id == portal.id and
                     c.primitive_b_id == other_prim_id and c.portal_b_id == other_portal.id) or
                    (c.primitive_a_id == other_prim_id and c.portal_a_id == other_portal.id and
                     c.primitive_b_id == prim_id and c.portal_b_id == portal.id)
                    for c in self.connections
                )
                if not exists:
                    self.connections.append(Connection(
                        primitive_a_id=prim_id,
                        portal_a_id=portal.id,
                        primitive_b_id=other_prim_id,
                        portal_b_id=other_portal.id,
                    ))

    def to_dict(self) -> dict:
        """Serialize layout to dictionary for JSON export."""
        primitives_dict = {}
        for pid, p in self.primitives.items():
            prim_data = {
                'id': p.id,
                'primitive_type': p.primitive_type,
                'origin_cell': {'x': p.origin_cell.x, 'y': p.origin_cell.y},
                'rotation': p.rotation,
                'parameters': p.parameters,
                'z_offset': p.z_offset,
            }
            # Only include portal_overrides if there are any
            if p.portal_overrides:
                prim_data['portal_overrides'] = {
                    portal_id: override.to_dict()
                    for portal_id, override in p.portal_overrides.items()
                }
            primitives_dict[pid] = prim_data

        return {
            'grid_size': self.grid_size,
            'primitives': primitives_dict,
            'connections': [
                {
                    'primitive_a_id': c.primitive_a_id,
                    'portal_a_id': c.portal_a_id,
                    'primitive_b_id': c.primitive_b_id,
                    'portal_b_id': c.portal_b_id,
                    'is_secret': c.is_secret,
                }
                for c in self.connections
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> 'DungeonLayout':
        """Deserialize layout from dictionary.

        Handles migration of deprecated primitive types (e.g., LJunction, Corner)
        to their modern equivalent (SquareCorner).
        """
        layout = DungeonLayout(grid_size=data.get('grid_size', 128))

        for pid, pdata in data.get('primitives', {}).items():
            # Deserialize portal_overrides if present
            portal_overrides = {}
            if 'portal_overrides' in pdata:
                for portal_id, override_data in pdata['portal_overrides'].items():
                    portal_overrides[portal_id] = PortalOverride.from_dict(override_data)

            # Migrate deprecated primitive types to modern equivalents
            original_type = pdata['primitive_type']
            original_params = pdata.get('parameters', {})
            migrated_type, migrated_params = _migrate_primitive_type(
                original_type, original_params
            )

            prim = PlacedPrimitive(
                id=pdata['id'],
                primitive_type=migrated_type,
                origin_cell=CellCoord(pdata['origin_cell']['x'], pdata['origin_cell']['y']),
                rotation=pdata.get('rotation', 0),
                parameters=migrated_params,
                z_offset=pdata.get('z_offset', 0.0),
                portal_overrides=portal_overrides,
            )
            layout.primitives[pid] = prim

        for cdata in data.get('connections', []):
            layout.connections.append(Connection(
                primitive_a_id=cdata['primitive_a_id'],
                portal_a_id=cdata['portal_a_id'],
                primitive_b_id=cdata['primitive_b_id'],
                portal_b_id=cdata['portal_b_id'],
                is_secret=cdata.get('is_secret', False),  # Default False for backward compat
            ))

        return layout

    @property
    def selected(self) -> Optional[PlacedPrimitive]:
        """Get currently selected primitive."""
        if self._selected_id and self._selected_id in self.primitives:
            return self.primitives[self._selected_id]
        return None

    def select(self, prim_id: Optional[str]):
        """Set selection."""
        self._selected_id = prim_id if prim_id in self.primitives else None

    def clear_selection(self):
        """Clear selection."""
        self._selected_id = None
