"""
Layout state data structures for the generation pipeline.

This module defines the core data structures for tracking:
- Named paths through the dungeon (main path, branches)
- Markers for content placement (spawn points, items, keys)
- Layout metadata and progression information

These structures implement Phase B of the Grid Flow adoption plan.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto


class MarkerType(Enum):
    """Categories of markers that can be placed in a layout."""
    SPAWN_POINT = auto()      # Player spawn location
    LEVEL_GOAL = auto()       # Level exit / objective
    KEY = auto()              # Key pickup
    LOCK = auto()             # Locked door/gate
    ITEM = auto()             # Generic item pickup
    ENEMY = auto()            # Enemy spawn point
    SECRET = auto()           # Secret area marker
    CHECKPOINT = auto()       # Progress checkpoint
    CUSTOM = auto()           # User-defined marker


@dataclass
class Marker:
    """
    Named marker for content placement.

    Markers are abstract anchor points that decouple layout from content.
    The layout generator places markers, and content/theme systems consume
    them to instantiate actual game entities.

    Attributes:
        name: Unique identifier (e.g., "SpawnPoint", "KeyRed", "Boss1")
        marker_type: Category of marker
        position: World coordinates (x, y, z)
        room_id: ID of the room containing this marker (if applicable)
        tags: Additional metadata for content systems
    """
    name: str
    marker_type: MarkerType
    position: Tuple[float, float, float]
    room_id: Optional[int] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def with_tag(self, key: str, value: Any) -> 'Marker':
        """Return a copy of this marker with an additional tag."""
        new_tags = dict(self.tags)
        new_tags[key] = value
        return Marker(
            name=self.name,
            marker_type=self.marker_type,
            position=self.position,
            room_id=self.room_id,
            tags=new_tags
        )


@dataclass
class PathNetwork:
    """
    Named path through the dungeon layout.

    Paths track sequences of connected rooms for gameplay purposes:
    - Main path: Critical route from entrance to exit
    - Branch paths: Optional side routes for exploration/secrets
    - Loop paths: Circular routes that reconnect to main path

    Attributes:
        name: Path identifier (e.g., "main", "treasure_branch", "shortcut")
        room_ids: Ordered list of room IDs in this path
        is_critical: True if this path is required for level completion
        start_anchor: Name of the starting marker/room type
        end_anchor: Name of the ending marker/room type
    """
    name: str
    room_ids: List[int]
    is_critical: bool = False
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None

    @property
    def length(self) -> int:
        """Number of rooms in this path."""
        return len(self.room_ids)

    def contains_room(self, room_id: int) -> bool:
        """Check if a room is part of this path."""
        return room_id in self.room_ids

    def room_index(self, room_id: int) -> Optional[int]:
        """Get the index of a room in this path, or None if not found."""
        try:
            return self.room_ids.index(room_id)
        except ValueError:
            return None

    def progress_at_room(self, room_id: int) -> Optional[float]:
        """
        Calculate progress scalar (0.0 to 1.0) for a room in this path.

        Returns None if the room is not in this path.
        """
        idx = self.room_index(room_id)
        if idx is None:
            return None
        if len(self.room_ids) <= 1:
            return 1.0
        return idx / (len(self.room_ids) - 1)


@dataclass
class RoomProgress:
    """
    Progress information for a single room.

    Attributes:
        room_id: The room's unique identifier
        progress_scalar: 0.0 at entrance, 1.0 at exit (along critical path)
        distance_from_entrance: Number of rooms from entrance
        distance_to_exit: Number of rooms to exit
        on_critical_path: True if room is on the main path
        branch_name: Name of branch path if on a branch, None otherwise
    """
    room_id: int
    progress_scalar: float = 0.0
    distance_from_entrance: int = 0
    distance_to_exit: int = 0
    on_critical_path: bool = False
    branch_name: Optional[str] = None


@dataclass
class GateConstraint:
    """
    Key-lock dependency that gates player progression.

    Gates implement locked door mechanics where the player must obtain
    a key before passing through a locked passage. This enforces
    non-linear exploration while maintaining required progression.

    The lock is placed on an edge (corridor) rather than a node (room)
    to prevent bypass via alternate routes.

    Attributes:
        key_marker: Name of the key marker (e.g., "KeyRed")
        lock_marker: Name of the lock marker (e.g., "LockRed")
        key_room_id: Room where the key is placed
        lock_edge: The corridor edge (room_a_id, room_b_id) where lock is placed
        bypass_prevention: Strategy to prevent circumventing the lock
            - "one_way_door": Add one-way doors around the locked passage
            - "blocked_edge": Simply block the edge until key obtained
    """
    key_marker: str
    lock_marker: str
    key_room_id: int
    lock_edge: Tuple[int, int]
    bypass_prevention: str = "one_way_door"

    def get_key_marker_name(self) -> str:
        """Get the key marker name."""
        return self.key_marker

    def get_lock_marker_name(self) -> str:
        """Get the lock marker name."""
        return self.lock_marker

    def blocks_edge(self, room_a: int, room_b: int) -> bool:
        """Check if this gate blocks a specific edge."""
        edge = (min(room_a, room_b), max(room_a, room_b))
        normalized_lock = (min(self.lock_edge[0], self.lock_edge[1]),
                          max(self.lock_edge[0], self.lock_edge[1]))
        return edge == normalized_lock


@dataclass
class LayoutState:
    """
    Complete state of a generated layout with paths and markers.

    This is the central data structure passed between pipeline passes.
    It contains all information needed to understand the layout's
    structure, progression, and content placement.

    Attributes:
        rooms: Raw room data from BSP generator
        corridors: Raw corridor data from BSP generator
        paths: Named paths through the layout
        markers: Content placement markers
        gates: Key-lock gate constraints for progression gating
        room_progress: Progress information per room
        metadata: Additional generation metadata
        seed: The random seed used for generation
    """
    rooms: List[Dict[str, Any]] = field(default_factory=list)
    corridors: List[Dict[str, Any]] = field(default_factory=list)
    paths: Dict[str, PathNetwork] = field(default_factory=dict)
    markers: List[Marker] = field(default_factory=list)
    gates: List[GateConstraint] = field(default_factory=list)
    room_progress: Dict[int, RoomProgress] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    @classmethod
    def from_bsp_export(cls, bsp_export: Dict[str, Any], seed: int = 0) -> 'LayoutState':
        """
        Create a LayoutState from BSP generator export data.

        Args:
            bsp_export: Result of BSPGenerator.export_layout()
            seed: The seed used for generation

        Returns:
            New LayoutState initialized with room and corridor data
        """
        return cls(
            rooms=bsp_export.get('rooms', []),
            corridors=bsp_export.get('corridors', []),
            seed=seed,
            metadata={
                'source': 'bsp_generator',
                'room_count': len(bsp_export.get('rooms', [])),
                'corridor_count': len(bsp_export.get('corridors', [])),
            }
        )

    def get_room_by_id(self, room_id: int) -> Optional[Dict[str, Any]]:
        """Find a room by its ID."""
        for room in self.rooms:
            if room.get('id') == room_id:
                return room
        return None

    def get_room_by_type(self, room_type: str) -> Optional[Dict[str, Any]]:
        """Find the first room of a given type."""
        for room in self.rooms:
            if room.get('type', '').lower() == room_type.lower():
                return room
        return None

    def get_rooms_by_type(self, room_type: str) -> List[Dict[str, Any]]:
        """Find all rooms of a given type."""
        return [r for r in self.rooms if r.get('type', '').lower() == room_type.lower()]

    def get_main_path(self) -> Optional[PathNetwork]:
        """Get the critical/main path if it exists."""
        return self.paths.get('main')

    def get_markers_by_type(self, marker_type: MarkerType) -> List[Marker]:
        """Get all markers of a specific type."""
        return [m for m in self.markers if m.marker_type == marker_type]

    def get_marker_by_name(self, name: str) -> Optional[Marker]:
        """Find a marker by its name."""
        for marker in self.markers:
            if marker.name == name:
                return marker
        return None

    def add_marker(self, marker: Marker) -> None:
        """Add a marker to the layout."""
        self.markers.append(marker)

    def add_path(self, path: PathNetwork) -> None:
        """Add a named path to the layout."""
        self.paths[path.name] = path

    def add_gate(self, gate: GateConstraint) -> None:
        """Add a gate constraint to the layout."""
        self.gates.append(gate)

    def get_gate_by_key(self, key_marker: str) -> Optional[GateConstraint]:
        """Find a gate by its key marker name."""
        for gate in self.gates:
            if gate.key_marker == key_marker:
                return gate
        return None

    def get_gates_blocking_edge(self, room_a: int, room_b: int) -> List[GateConstraint]:
        """Get all gates that block a specific edge."""
        return [g for g in self.gates if g.blocks_edge(room_a, room_b)]

    def is_edge_locked(self, room_a: int, room_b: int) -> bool:
        """Check if an edge is blocked by any gate."""
        return len(self.get_gates_blocking_edge(room_a, room_b)) > 0

    def get_entrance_room(self) -> Optional[Dict[str, Any]]:
        """Get the entrance room."""
        return self.get_room_by_type('entrance')

    def get_exit_room(self) -> Optional[Dict[str, Any]]:
        """Get the exit room."""
        return self.get_room_by_type('exit')

    def get_connected_rooms(self, room_id: int) -> List[int]:
        """Get IDs of all rooms directly connected to a room via corridors."""
        connected = []
        for corridor in self.corridors:
            start = corridor.get('start_room_id', corridor.get('room_a_id'))
            end = corridor.get('end_room_id', corridor.get('room_b_id'))
            if start == room_id:
                connected.append(end)
            elif end == room_id:
                connected.append(start)
        return connected

    def build_adjacency_map(self) -> Dict[int, List[int]]:
        """Build a map of room_id -> list of connected room_ids."""
        adj = {}
        for room in self.rooms:
            room_id = room.get('id')
            if room_id is not None:
                adj[room_id] = []

        for corridor in self.corridors:
            start = corridor.get('start_room_id', corridor.get('room_a_id'))
            end = corridor.get('end_room_id', corridor.get('room_b_id'))
            if start in adj:
                adj[start].append(end)
            if end in adj:
                adj[end].append(start)

        return adj
