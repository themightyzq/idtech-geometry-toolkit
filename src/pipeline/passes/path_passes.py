"""
Path computation passes for the layout pipeline.

These passes analyze the dungeon layout to identify:
- Critical path from entrance to exit
- Branch paths for exploration
- Progress scalars for difficulty scaling
"""

from collections import deque
from typing import List, Dict, Optional, Set, Tuple

from .base import LayoutPass, PassConfig, PassResult
from ..layout_state import LayoutState, PathNetwork, RoomProgress


class CreateMainPathPass(LayoutPass):
    """
    Compute the critical path from entrance to exit.

    This pass uses BFS (Breadth-First Search) to find the shortest path
    from the entrance room to the exit room, then stores it as the
    "main" path in the layout state.

    The critical path is essential for:
    - Difficulty progression (items/enemies scale along path)
    - Key-lock placement (keys on branches, locks on main path)
    - Player guidance (main objectives follow this route)
    """

    @property
    def name(self) -> str:
        return "CreateMainPath"

    @property
    def description(self) -> str:
        return "Find the critical path from entrance to exit using BFS"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")
            return errors

        entrance = state.get_entrance_room()
        if entrance is None:
            errors.append("No entrance room found")

        exit_room = state.get_exit_room()
        if exit_room is None:
            errors.append("No exit room found")

        if not state.corridors:
            errors.append("No corridors connecting rooms")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Find the shortest path from entrance to exit."""
        result = PassResult(success=True, state=state)

        entrance = state.get_entrance_room()
        exit_room = state.get_exit_room()

        if entrance is None or exit_room is None:
            result.add_error("Cannot compute path: missing entrance or exit")
            return result

        entrance_id = entrance.get('id')
        exit_id = exit_room.get('id')

        # Build adjacency map
        adjacency = state.build_adjacency_map()

        # BFS to find shortest path
        path = self._bfs_shortest_path(adjacency, entrance_id, exit_id)

        if path is None:
            result.add_error(f"No path found from entrance (room {entrance_id}) to exit (room {exit_id})")
            return result

        # Create the main path network
        main_path = PathNetwork(
            name="main",
            room_ids=path,
            is_critical=True,
            start_anchor="entrance",
            end_anchor="exit"
        )

        state.add_path(main_path)

        result.metrics['main_path_length'] = len(path)
        result.metrics['entrance_room_id'] = entrance_id
        result.metrics['exit_room_id'] = exit_id

        return result

    def _bfs_shortest_path(
        self,
        adjacency: Dict[int, List[int]],
        start: int,
        goal: int
    ) -> Optional[List[int]]:
        """
        Find the shortest path between two rooms using BFS.

        Args:
            adjacency: Map of room_id -> list of connected room_ids
            start: Starting room ID
            goal: Target room ID

        Returns:
            List of room IDs forming the path, or None if no path exists
        """
        if start == goal:
            return [start]

        if start not in adjacency:
            return None

        # BFS with path reconstruction
        visited: Set[int] = {start}
        queue: deque = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            for neighbor in adjacency.get(current, []):
                if neighbor == goal:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        errors = []

        main_path = state.get_main_path()
        if main_path is None:
            errors.append("Main path was not created")
        elif main_path.length < 2:
            errors.append("Main path is too short (less than 2 rooms)")

        return errors


class ComputeProgressPass(LayoutPass):
    """
    Compute progress scalars for all rooms.

    This pass assigns a progress_scalar to each room based on its
    position along the critical path:
    - 0.0 for the entrance room
    - 1.0 for the exit room
    - Interpolated values for rooms in between

    Rooms not on the critical path get progress based on their
    closest connection to a critical path room.
    """

    @property
    def name(self) -> str:
        return "ComputeProgress"

    @property
    def description(self) -> str:
        return "Calculate progress scalars for difficulty scaling"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")

        if 'main' not in state.paths:
            errors.append("Main path must be computed first (run CreateMainPathPass)")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Compute progress for all rooms."""
        result = PassResult(success=True, state=state)

        main_path = state.get_main_path()
        if main_path is None:
            result.add_error("No main path found")
            return result

        # Build set of rooms on critical path
        critical_set = set(main_path.room_ids)
        path_length = len(main_path.room_ids)

        # Compute progress for each room
        for room in state.rooms:
            room_id = room.get('id')
            if room_id is None:
                continue

            if room_id in critical_set:
                # Room is on critical path - compute exact progress
                idx = main_path.room_ids.index(room_id)
                progress = idx / (path_length - 1) if path_length > 1 else 1.0

                room_progress = RoomProgress(
                    room_id=room_id,
                    progress_scalar=progress,
                    distance_from_entrance=idx,
                    distance_to_exit=path_length - 1 - idx,
                    on_critical_path=True,
                    branch_name=None
                )
            else:
                # Room is on a branch - find closest critical path room
                closest_progress = self._find_closest_critical_progress(
                    state, room_id, main_path
                )

                room_progress = RoomProgress(
                    room_id=room_id,
                    progress_scalar=closest_progress,
                    distance_from_entrance=-1,  # Not on critical path
                    distance_to_exit=-1,
                    on_critical_path=False,
                    branch_name=self._identify_branch(state, room_id, main_path)
                )

            state.room_progress[room_id] = room_progress

        result.metrics['rooms_on_critical_path'] = len(critical_set)
        result.metrics['rooms_on_branches'] = len(state.rooms) - len(critical_set)

        return result

    def _find_closest_critical_progress(
        self,
        state: LayoutState,
        room_id: int,
        main_path: PathNetwork
    ) -> float:
        """
        Find the progress scalar of the nearest critical path room.

        Uses BFS to find the closest room that's on the critical path.
        """
        critical_set = set(main_path.room_ids)
        adjacency = state.build_adjacency_map()

        if room_id in critical_set:
            return main_path.progress_at_room(room_id) or 0.5

        # BFS to find nearest critical path room
        visited: Set[int] = {room_id}
        queue: deque = deque([room_id])

        while queue:
            current = queue.popleft()

            for neighbor in adjacency.get(current, []):
                if neighbor in critical_set:
                    # Found a critical path room
                    return main_path.progress_at_room(neighbor) or 0.5

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Fallback if somehow disconnected
        return 0.5

    def _identify_branch(
        self,
        state: LayoutState,
        room_id: int,
        main_path: PathNetwork
    ) -> Optional[str]:
        """
        Identify which branch a room belongs to.

        For now, returns a generic branch name based on the
        critical path room it connects to.
        """
        critical_set = set(main_path.room_ids)
        adjacency = state.build_adjacency_map()

        # BFS to find the critical path room this branches from
        visited: Set[int] = {room_id}
        queue: deque = deque([room_id])

        while queue:
            current = queue.popleft()

            for neighbor in adjacency.get(current, []):
                if neighbor in critical_set:
                    idx = main_path.room_ids.index(neighbor)
                    return f"branch_from_room_{neighbor}"

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return "unknown_branch"

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.room_progress:
            errors.append("No room progress information computed")

        # Verify all rooms have progress
        for room in state.rooms:
            room_id = room.get('id')
            if room_id is not None and room_id not in state.room_progress:
                errors.append(f"Room {room_id} missing progress information")

        return errors
