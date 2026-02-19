"""
Gate generation passes for the layout pipeline.

These passes create key-lock gating mechanics that enforce progression:
- Keys placed on branch paths (requires exploration)
- Locks placed on main path edges (gates progression)
- Bypass validation ensures locks cannot be circumvented
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque

from .base import LayoutPass, PassConfig, PassResult
from ..layout_state import LayoutState, Marker, MarkerType, GateConstraint


class AddGatesPass(LayoutPass):
    """
    Add key-lock gates to enforce exploration and progression.

    This pass identifies branch paths and places keys on them,
    then places corresponding locks on the main path. The player
    must explore branch areas to find keys before progressing.

    Gate placement strategy:
    1. Find rooms on branch paths (not on critical path)
    2. For each gate to create, select a branch room for the key
    3. Select a main path edge for the lock (after the branch junction)
    4. Create GateConstraint and corresponding markers

    Configuration options:
    - gate_count: Number of key-lock pairs to create (default: 1)
    - key_colors: List of key color names (default: ['Red', 'Blue', 'Yellow'])
    - min_branch_depth: Minimum rooms from junction to place key (default: 1)
    """

    @property
    def name(self) -> str:
        return "AddGates"

    @property
    def description(self) -> str:
        return "Add key-lock gates for progression gating"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")

        if not state.get_main_path():
            errors.append("Main path not computed (run CreateMainPathPass first)")

        if not state.room_progress:
            errors.append("Room progress not computed (run ComputeProgressPass first)")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Generate key-lock gates."""
        result = PassResult(success=True, state=state)

        # Configuration
        gate_count = config.options.get('gate_count', 1)
        key_colors = config.options.get('key_colors', ['Red', 'Blue', 'Yellow'])
        min_branch_depth = config.options.get('min_branch_depth', 1)
        key_height = config.options.get('key_height', 24.0)
        lock_height = config.options.get('lock_height', 0.0)

        main_path = state.get_main_path()
        if not main_path:
            result.success = False
            result.add_error("No main path available")
            return result

        # Find branch rooms (not on critical path)
        branch_rooms = self._find_branch_rooms(state, main_path)

        if not branch_rooms:
            result.add_warning("No branch rooms available for key placement")
            return result

        # Find main path edges that can have locks placed
        main_path_edges = self._get_main_path_edges(state, main_path)

        if not main_path_edges:
            result.add_warning("No main path edges available for lock placement")
            return result

        gates_created = 0

        # Create gates up to the requested count or available resources
        max_gates = min(gate_count, len(branch_rooms), len(main_path_edges), len(key_colors))

        # Sort branch rooms by distance from junction (prefer deeper branches)
        branch_rooms_sorted = self._sort_branches_by_depth(state, main_path, branch_rooms)

        # Sort edges by progress (place locks at increasing progress points)
        edges_sorted = self._sort_edges_by_progress(state, main_path_edges)

        for i in range(max_gates):
            key_color = key_colors[i]
            key_marker_name = f"Key{key_color}"
            lock_marker_name = f"Lock{key_color}"

            # Select key placement room
            if i >= len(branch_rooms_sorted):
                break
            key_room_id = branch_rooms_sorted[i]
            key_room = state.get_room_by_id(key_room_id)

            if not key_room:
                continue

            # Select lock edge (after the branch junction point)
            lock_edge = self._select_lock_edge_for_key(
                state, main_path, key_room_id, edges_sorted, i
            )

            if not lock_edge:
                result.add_warning(f"Could not find suitable lock edge for {key_color} key")
                continue

            # Get key room center for marker placement
            bounds = key_room.get('bounds', {})
            key_x = bounds.get('x', 0) + bounds.get('width', 0) / 2
            key_y = bounds.get('y', 0) + bounds.get('height', 0) / 2
            key_z = key_room.get('z_offset', 0) + key_height

            # Create key marker
            key_marker = Marker(
                name=key_marker_name,
                marker_type=MarkerType.KEY,
                position=(key_x, key_y, key_z),
                room_id=key_room_id,
                tags={
                    'key_color': key_color,
                    'gate_index': i,
                }
            )
            state.add_marker(key_marker)

            # Calculate lock position (corridor between rooms)
            lock_x, lock_y, lock_z = self._get_lock_position(state, lock_edge, lock_height)

            # Create lock marker
            lock_marker = Marker(
                name=lock_marker_name,
                marker_type=MarkerType.LOCK,
                position=(lock_x, lock_y, lock_z),
                room_id=None,  # Lock is on edge, not in a room
                tags={
                    'lock_color': key_color,
                    'gate_index': i,
                    'edge': lock_edge,
                }
            )
            state.add_marker(lock_marker)

            # Create gate constraint
            gate = GateConstraint(
                key_marker=key_marker_name,
                lock_marker=lock_marker_name,
                key_room_id=key_room_id,
                lock_edge=lock_edge,
                bypass_prevention="one_way_door"
            )
            state.add_gate(gate)

            gates_created += 1

        result.metrics['gates_created'] = gates_created
        result.metrics['branch_rooms_available'] = len(branch_rooms)
        result.metrics['main_path_edges_available'] = len(main_path_edges)

        if gates_created < gate_count:
            result.add_warning(
                f"Created {gates_created} of {gate_count} requested gates "
                f"(limited by available branches/edges)"
            )

        return result

    def _find_branch_rooms(self, state: LayoutState, main_path) -> List[int]:
        """Find all rooms not on the critical path."""
        main_path_set = set(main_path.room_ids)
        branch_rooms = []

        for room in state.rooms:
            room_id = room.get('id')
            if room_id is not None and room_id not in main_path_set:
                branch_rooms.append(room_id)

        return branch_rooms

    def _get_main_path_edges(self, state: LayoutState, main_path) -> List[Tuple[int, int]]:
        """Get all edges along the main path."""
        edges = []
        room_ids = main_path.room_ids

        for i in range(len(room_ids) - 1):
            edge = (room_ids[i], room_ids[i + 1])
            edges.append(edge)

        return edges

    def _sort_branches_by_depth(
        self, state: LayoutState, main_path, branch_rooms: List[int]
    ) -> List[int]:
        """Sort branch rooms by their depth from the main path (deeper first)."""
        main_path_set = set(main_path.room_ids)
        adj = state.build_adjacency_map()

        # Calculate depth of each branch room from main path
        depths = {}
        for room_id in branch_rooms:
            depth = self._calculate_branch_depth(room_id, main_path_set, adj)
            depths[room_id] = depth

        # Sort by depth (descending - deeper first for better key placement)
        return sorted(branch_rooms, key=lambda r: depths.get(r, 0), reverse=True)

    def _calculate_branch_depth(
        self, room_id: int, main_path_set: Set[int], adj: Dict[int, List[int]]
    ) -> int:
        """Calculate how many rooms deep a branch room is from the main path."""
        if room_id in main_path_set:
            return 0

        visited = {room_id}
        queue = deque([(room_id, 0)])

        while queue:
            current, depth = queue.popleft()

            for neighbor in adj.get(current, []):
                if neighbor in main_path_set:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return 0  # Shouldn't happen in connected graph

    def _sort_edges_by_progress(
        self, state: LayoutState, edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Sort edges by the progress of their first room (ascending)."""
        def edge_progress(edge: Tuple[int, int]) -> float:
            room_id = edge[0]
            if room_id in state.room_progress:
                return state.room_progress[room_id].progress_scalar
            return 0.0

        return sorted(edges, key=edge_progress)

    def _select_lock_edge_for_key(
        self,
        state: LayoutState,
        main_path,
        key_room_id: int,
        sorted_edges: List[Tuple[int, int]],
        gate_index: int
    ) -> Optional[Tuple[int, int]]:
        """
        Select an appropriate edge for the lock.

        The lock should be placed after the branch point where the key is,
        ensuring the player must visit the branch before progressing.
        """
        # Find the junction point where this branch connects to main path
        main_path_set = set(main_path.room_ids)
        adj = state.build_adjacency_map()

        junction_room = self._find_junction_for_branch(key_room_id, main_path_set, adj)

        if junction_room is None:
            # Fallback: use edge based on gate index
            if gate_index < len(sorted_edges):
                return sorted_edges[gate_index]
            return None

        # Find the junction's position in the main path
        junction_index = None
        for i, room_id in enumerate(main_path.room_ids):
            if room_id == junction_room:
                junction_index = i
                break

        if junction_index is None:
            return None

        # Place lock on an edge AFTER the junction (so key is needed to progress)
        # Skip ahead by at least 1 edge to give player room to approach
        target_edge_index = junction_index

        if target_edge_index < len(sorted_edges):
            return sorted_edges[target_edge_index]

        return None

    def _find_junction_for_branch(
        self, branch_room_id: int, main_path_set: Set[int], adj: Dict[int, List[int]]
    ) -> Optional[int]:
        """Find the main path room that connects to this branch."""
        visited = {branch_room_id}
        queue = deque([branch_room_id])

        while queue:
            current = queue.popleft()

            for neighbor in adj.get(current, []):
                if neighbor in main_path_set:
                    return neighbor
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return None

    def _get_lock_position(
        self, state: LayoutState, edge: Tuple[int, int], height_offset: float
    ) -> Tuple[float, float, float]:
        """Calculate the position for a lock marker on an edge."""
        room_a = state.get_room_by_id(edge[0])
        room_b = state.get_room_by_id(edge[1])

        if not room_a or not room_b:
            return (0.0, 0.0, height_offset)

        # Get centers of both rooms
        bounds_a = room_a.get('bounds', {})
        bounds_b = room_b.get('bounds', {})

        center_a_x = bounds_a.get('x', 0) + bounds_a.get('width', 0) / 2
        center_a_y = bounds_a.get('y', 0) + bounds_a.get('height', 0) / 2

        center_b_x = bounds_b.get('x', 0) + bounds_b.get('width', 0) / 2
        center_b_y = bounds_b.get('y', 0) + bounds_b.get('height', 0) / 2

        # Lock goes at midpoint between rooms
        lock_x = (center_a_x + center_b_x) / 2
        lock_y = (center_a_y + center_b_y) / 2

        # Use average Z offset
        z_a = room_a.get('z_offset', 0)
        z_b = room_b.get('z_offset', 0)
        lock_z = (z_a + z_b) / 2 + height_offset

        return (lock_x, lock_y, lock_z)

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        errors = []

        # Each gate should have both key and lock markers
        for gate in state.gates:
            key_marker = state.get_marker_by_name(gate.key_marker)
            lock_marker = state.get_marker_by_name(gate.lock_marker)

            if not key_marker:
                errors.append(f"Missing key marker: {gate.key_marker}")
            if not lock_marker:
                errors.append(f"Missing lock marker: {gate.lock_marker}")

        return errors


def validate_no_bypass(state: LayoutState) -> Dict[str, Any]:
    """
    Validate that all gates cannot be bypassed.

    For each gate, this verifies that:
    1. The key is reachable from the entrance without crossing any lock
    2. The area beyond the lock is NOT reachable without the key
    3. With the key, the entire layout is reachable

    Returns a dict with:
    - valid: True if all gates are properly configured
    - gates: List of validation results per gate
    - errors: List of validation errors (empty if valid)
    """
    result = {
        'valid': True,
        'gates': [],
        'errors': []
    }

    if not state.gates:
        # No gates to validate
        return result

    entrance = state.get_entrance_room()
    exit_room = state.get_exit_room()

    if not entrance:
        result['valid'] = False
        result['errors'].append("No entrance room found")
        return result

    entrance_id = entrance.get('id')

    for gate in state.gates:
        gate_result = _validate_single_gate(state, gate, entrance_id)
        result['gates'].append(gate_result)

        if not gate_result['valid']:
            result['valid'] = False
            result['errors'].extend(gate_result['errors'])

    return result


def _validate_single_gate(
    state: LayoutState, gate: GateConstraint, entrance_id: int
) -> Dict[str, Any]:
    """Validate a single gate cannot be bypassed."""
    result = {
        'gate': gate.key_marker,
        'valid': True,
        'errors': [],
        'key_reachable': False,
        'lock_blocks_progress': False,
        'fully_reachable_with_key': False,
    }

    adj = state.build_adjacency_map()

    # Build adjacency map excluding the locked edge
    adj_without_lock = _build_adjacency_excluding_edge(state, gate.lock_edge)

    # 1. Check if key is reachable from entrance without crossing any locks
    reachable_without_lock = _find_reachable_rooms(entrance_id, adj_without_lock)

    if gate.key_room_id in reachable_without_lock:
        result['key_reachable'] = True
    else:
        result['valid'] = False
        result['errors'].append(
            f"Key {gate.key_marker} (room {gate.key_room_id}) is not reachable "
            f"from entrance without crossing the lock"
        )

    # 2. Check if exit is NOT reachable without crossing the lock
    exit_room = state.get_exit_room()
    if exit_room:
        exit_id = exit_room.get('id')
        if exit_id not in reachable_without_lock:
            result['lock_blocks_progress'] = True
        else:
            result['valid'] = False
            result['errors'].append(
                f"Lock {gate.lock_marker} can be bypassed - exit is reachable "
                f"without crossing the locked edge"
            )

    # 3. Check if entire layout is reachable with the key (using normal adjacency)
    reachable_with_key = _find_reachable_rooms(entrance_id, adj)
    all_room_ids = {r.get('id') for r in state.rooms if r.get('id') is not None}

    if reachable_with_key == all_room_ids:
        result['fully_reachable_with_key'] = True
    else:
        unreachable = all_room_ids - reachable_with_key
        result['errors'].append(
            f"Some rooms are unreachable even with key: {unreachable}"
        )

    return result


def _build_adjacency_excluding_edge(
    state: LayoutState, excluded_edge: Tuple[int, int]
) -> Dict[int, List[int]]:
    """Build adjacency map with one edge removed."""
    adj = {}

    for room in state.rooms:
        room_id = room.get('id')
        if room_id is not None:
            adj[room_id] = []

    excluded_normalized = (
        min(excluded_edge[0], excluded_edge[1]),
        max(excluded_edge[0], excluded_edge[1])
    )

    for corridor in state.corridors:
        start = corridor.get('start_room_id', corridor.get('room_a_id'))
        end = corridor.get('end_room_id', corridor.get('room_b_id'))

        # Skip the excluded edge
        edge_normalized = (min(start, end), max(start, end))
        if edge_normalized == excluded_normalized:
            continue

        if start in adj:
            adj[start].append(end)
        if end in adj:
            adj[end].append(start)

    return adj


def _find_reachable_rooms(start_id: int, adj: Dict[int, List[int]]) -> Set[int]:
    """Find all rooms reachable from start_id using BFS."""
    visited = {start_id}
    queue = deque([start_id])

    while queue:
        current = queue.popleft()

        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


class ValidateGatesPass(LayoutPass):
    """
    Validate that all gates are properly configured and cannot be bypassed.

    This pass runs bypass validation on all gates and reports any issues.
    It does not modify the state, only validates it.
    """

    @property
    def name(self) -> str:
        return "ValidateGates"

    @property
    def description(self) -> str:
        return "Validate that gates cannot be bypassed"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Validate all gates."""
        result = PassResult(success=True, state=state)

        validation = validate_no_bypass(state)

        result.metrics['gates_validated'] = len(state.gates)
        result.metrics['validation_passed'] = validation['valid']

        if not validation['valid']:
            for error in validation['errors']:
                result.add_error(error)
            result.success = False

        # Include detailed results in metrics for debugging
        result.metrics['gate_details'] = validation['gates']

        return result

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        # This is a read-only validation pass
        return []
