"""
Marker generation passes for the layout pipeline.

These passes create markers at strategic locations:
- SpawnPoint at entrance
- LevelGoal at exit
- Item/enemy markers along paths based on progression
"""

from typing import List, Dict, Any, Optional, Tuple

from .base import LayoutPass, PassConfig, PassResult
from ..layout_state import LayoutState, Marker, MarkerType


class AddMarkersPass(LayoutPass):
    """
    Generate content placement markers from room types and positions.

    This pass creates the foundational markers that content systems
    use to place actual game entities:

    - SpawnPoint: Player start location (entrance room)
    - LevelGoal: Level exit/objective (exit room)
    - Junction markers: Connection points at junction rooms
    - Arena markers: Combat encounter areas

    Markers are placed at room centers by default, with Z offset
    for proper player/entity height.
    """

    @property
    def name(self) -> str:
        return "AddMarkers"

    @property
    def description(self) -> str:
        return "Generate content placement markers from room types"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Generate markers from room data."""
        result = PassResult(success=True, state=state)

        # Configuration options
        spawn_height = config.options.get('spawn_height', 32.0)
        goal_height = config.options.get('goal_height', 32.0)
        mark_junctions = config.options.get('mark_junctions', True)
        mark_arenas = config.options.get('mark_arenas', True)

        markers_created = 0

        for room in state.rooms:
            room_id = room.get('id')
            room_type = room.get('type', 'standard').lower()
            bounds = room.get('bounds', {})

            # Calculate room center
            center_x = bounds.get('x', 0) + bounds.get('width', 0) / 2
            center_y = bounds.get('y', 0) + bounds.get('height', 0) / 2
            floor_z = room.get('z_offset', 0)

            if room_type == 'entrance':
                # Create spawn point marker
                marker = Marker(
                    name="SpawnPoint",
                    marker_type=MarkerType.SPAWN_POINT,
                    position=(center_x, center_y, floor_z + spawn_height),
                    room_id=room_id,
                    tags={
                        'room_type': room_type,
                        'is_primary': True,
                    }
                )
                state.add_marker(marker)
                markers_created += 1

            elif room_type == 'exit':
                # Create level goal marker
                marker = Marker(
                    name="LevelGoal",
                    marker_type=MarkerType.LEVEL_GOAL,
                    position=(center_x, center_y, floor_z + goal_height),
                    room_id=room_id,
                    tags={
                        'room_type': room_type,
                        'is_primary': True,
                    }
                )
                state.add_marker(marker)
                markers_created += 1

            elif room_type == 'junction' and mark_junctions:
                # Create checkpoint marker at junctions
                marker = Marker(
                    name=f"Junction_{room_id}",
                    marker_type=MarkerType.CHECKPOINT,
                    position=(center_x, center_y, floor_z + spawn_height),
                    room_id=room_id,
                    tags={
                        'room_type': room_type,
                    }
                )
                state.add_marker(marker)
                markers_created += 1

            elif room_type == 'arena' and mark_arenas:
                # Create combat encounter marker
                marker = Marker(
                    name=f"Arena_{room_id}",
                    marker_type=MarkerType.ENEMY,
                    position=(center_x, center_y, floor_z + spawn_height),
                    room_id=room_id,
                    tags={
                        'room_type': room_type,
                        'encounter_type': 'arena',
                    }
                )
                state.add_marker(marker)
                markers_created += 1

        result.metrics['markers_created'] = markers_created

        # Validate we have essential markers
        if not state.get_marker_by_name("SpawnPoint"):
            result.add_warning("No SpawnPoint marker created (no entrance room?)")

        if not state.get_marker_by_name("LevelGoal"):
            result.add_warning("No LevelGoal marker created (no exit room?)")

        return result

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        errors = []

        spawn_markers = state.get_markers_by_type(MarkerType.SPAWN_POINT)
        if not spawn_markers:
            errors.append("No spawn point markers created")

        goal_markers = state.get_markers_by_type(MarkerType.LEVEL_GOAL)
        if not goal_markers:
            errors.append("No level goal markers created")

        return errors


class AddProgressionMarkersPass(LayoutPass):
    """
    Add markers along the critical path based on progression.

    This pass uses the progress_scalar to place markers at
    regular intervals along the main path:

    - Item markers at low-medium progression (health, ammo)
    - Enemy markers at medium-high progression
    - Secret markers on branch paths
    """

    @property
    def name(self) -> str:
        return "AddProgressionMarkers"

    @property
    def description(self) -> str:
        return "Place markers along paths based on difficulty progression"

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        errors = []

        if not state.rooms:
            errors.append("No rooms in layout")

        if not state.room_progress:
            errors.append("Room progress not computed (run ComputeProgressPass first)")

        return errors

    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """Generate progression-based markers."""
        result = PassResult(success=True, state=state)

        # Configuration
        item_interval = config.options.get('item_interval', 0.2)  # Every 20% progress
        enemy_start_progress = config.options.get('enemy_start', 0.3)  # Enemies after 30%
        mark_branches_as_secrets = config.options.get('mark_branch_secrets', True)

        markers_created = 0
        last_item_progress = 0.0
        last_enemy_progress = enemy_start_progress

        # Sort rooms by progress for ordered traversal
        sorted_rooms = []
        for room in state.rooms:
            room_id = room.get('id')
            if room_id is not None and room_id in state.room_progress:
                progress_info = state.room_progress[room_id]
                sorted_rooms.append((room, progress_info))

        sorted_rooms.sort(key=lambda x: x[1].progress_scalar)

        for room, progress_info in sorted_rooms:
            room_id = room.get('id')
            room_type = room.get('type', 'standard').lower()
            progress = progress_info.progress_scalar

            # Skip entrance and exit (already have markers)
            if room_type in ('entrance', 'exit'):
                continue

            bounds = room.get('bounds', {})
            center_x = bounds.get('x', 0) + bounds.get('width', 0) / 2
            center_y = bounds.get('y', 0) + bounds.get('height', 0) / 2
            floor_z = room.get('z_offset', 0)

            # Item markers at regular intervals on critical path
            if progress_info.on_critical_path:
                if progress - last_item_progress >= item_interval:
                    marker = Marker(
                        name=f"Item_{room_id}",
                        marker_type=MarkerType.ITEM,
                        position=(center_x, center_y, floor_z + 16),
                        room_id=room_id,
                        tags={
                            'progress': progress,
                            'suggested_items': self._suggest_items(progress),
                        }
                    )
                    state.add_marker(marker)
                    markers_created += 1
                    last_item_progress = progress

                # Enemy markers in later parts of the path
                if progress >= enemy_start_progress:
                    if progress - last_enemy_progress >= item_interval:
                        marker = Marker(
                            name=f"Enemy_{room_id}",
                            marker_type=MarkerType.ENEMY,
                            position=(center_x, center_y, floor_z + 24),
                            room_id=room_id,
                            tags={
                                'progress': progress,
                                'difficulty': self._difficulty_tier(progress),
                            }
                        )
                        state.add_marker(marker)
                        markers_created += 1
                        last_enemy_progress = progress

            # Mark branch rooms as potential secrets
            elif mark_branches_as_secrets and not progress_info.on_critical_path:
                marker = Marker(
                    name=f"Secret_{room_id}",
                    marker_type=MarkerType.SECRET,
                    position=(center_x, center_y, floor_z + 16),
                    room_id=room_id,
                    tags={
                        'branch': progress_info.branch_name,
                        'progress': progress,
                    }
                )
                state.add_marker(marker)
                markers_created += 1

        result.metrics['progression_markers_created'] = markers_created

        return result

    def _suggest_items(self, progress: float) -> List[str]:
        """Suggest appropriate items based on progress."""
        if progress < 0.3:
            return ['health_small', 'shells']
        elif progress < 0.6:
            return ['health_medium', 'shells', 'nails']
        elif progress < 0.8:
            return ['health_large', 'rockets', 'cells']
        else:
            return ['health_mega', 'armor', 'cells']

    def _difficulty_tier(self, progress: float) -> str:
        """Determine difficulty tier based on progress."""
        if progress < 0.4:
            return 'easy'
        elif progress < 0.7:
            return 'medium'
        else:
            return 'hard'
